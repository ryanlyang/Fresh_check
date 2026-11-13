"""Disposable two-A100 numerical, resume, memory and throughput evidence."""
import copy
import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F

from . import model as M, protocol as P
from part_ev_gpu_scaling import benchmark as B
from part_ev_layerwise_ddp.engine import root_call, validate
from part_ev_fullscale.checkpoint import (seed_everything, atomic_torch_save, trusted_load,
                                         capture_rank_state, restore_rank_state)
from part_ev_fullscale.optimizer import build_optimizer, optimizer_metadata
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_screen.precision import model_precision
from scripts.check_part_ev_screen_amp_recovery import finite_state


def fresh(device, world, *, trim=True):
    seed_everything(101)
    base = M.build_model(trim=trim).to(device)
    if world == 2:
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
        model = DDP(base, device_ids=[device.index], output_device=device.index,
                    find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        model = base
    return base, model


def active(base):
    with torch.no_grad():
        for state in M.states(base):
            state.output.normal_(std=.02)
            if state.memory_reader is not None: state.memory_reader.normal_(std=.02)
            if state.memory_gate is not None: state.memory_gate[:3].normal_(std=.1)


def delayed_activation(device,rank,world):
    """Fresh model, three successful AMP/Ranger updates; skipped steps do not count."""
    base,model=fresh(device,world,trim=False)
    optimizer=build_optimizer(base,P.recipes()['ranger_full_v1'])
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    values,truth=B.tensors(B.split_batch(B.fixed_batch(8,17),rank,world),device)
    initial=[p.detach().clone() for p in M.writers_and_gates(base)]
    successes=skips=0
    first_zero=first_output=False
    for attempt in range(P.POLICY['delayed_max_attempts']):
        result=B.step(base,model,optimizer,scaler,values,truth,P.recipes()['ranger_full_v1'],attempt,world)
        if len(set(B.gather((result['skipped'],scaler.get_scale()),world)))!=1:
            raise ValueError('AMP ranks disagree during delayed activation')
        skips+=result['skipped']
        if result['skipped']: continue
        if successes==0:
            first_zero=all(p.grad is not None and p.grad.isfinite().all() and not p.grad.count_nonzero()
                          and torch.equal(p.detach(),saved) for p,saved in zip(M.writers_and_gates(base),initial))
            first_output=all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                             for p in M.readers(base))
            if not first_zero or not first_output:
                raise ValueError('Unexpected first-update writer/reader gradient behavior')
        successes+=1
        if successes==P.POLICY['delayed_successful_updates']: break
    matrices_active=all(p.detach().isfinite().all() and not torch.equal(p.detach(),saved)
                       for p,saved in zip(M.writers_and_gates(base),initial))
    gradients_active=all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                         for p in [*M.writers_and_gates(base),*M.readers(base)])
    gradients_active=gradients_active and all(s.memory_gate.grad.count_nonzero(dim=1).min()>0
                                              for s in M.states(base) if s.memory_gate is not None)
    passed=bool(successes==3 and first_zero and first_output and matrices_active and gradients_active)
    if not B.agree(passed,device,world): raise ValueError('Memory did not activate within the fixed update budget')
    return dict(passed=True,attempts=attempt+1,successful_updates=successes,skips=skips,
                first_success_writer_gradients_zero=True,first_success_reader_gradients_nonzero=True,
                all_writers_and_gates_updated=True,all_memory_gradients_nonzero=True)


def attention_invariance(device):
    """Same-input attention/ordinary EV invariant; memory only changes messages."""
    seed_everything(101)
    model=M.build_model(trim=False).to(device)
    active(model)
    block=model.mod.blocks[2]
    attention=block.attn.attention
    old=M.reference_model('state_layerwise',trim=False).to(device).mod.blocks[2].attn.attention
    old.load_state_dict({k:v for k,v in attention.state_dict().items() if not M.memory_key('.'+k)})
    query=torch.randn(17,2,128,device=device)
    valid=torch.ones(2,17,dtype=torch.bool,device=device);valid[1,-4:]=False
    stem=torch.randn(2,64,17,17,device=device)
    mask=torch.randn(16,17,17,device=device)
    memory=torch.rand(2,17,17,32,device=device)*2-1
    native=[]
    def observed(_module,_inputs,result):
        native.append(tuple(t.detach().clone() for t in result))
    handles=[a.reference.register_forward_hook(observed) for a in (old,attention)]
    try:
        with torch.no_grad():
            old.bind(stem,valid);seed_everything(173)
            old(query,query,query,key_padding_mask=~valid,attn_mask=mask)
            attention.bind(stem,valid);block.attn._bias=mask;seed_everything(173)
            M.particle_attention(block,query,~valid,memory)
        if len(native)!=2:raise ValueError('Expected one native attention call per model')
        for x,y in zip(*native):torch.testing.assert_close(x,y,rtol=0,atol=0)
        ev=[torch.einsum('hod,bhid->bhio',a.edge_projection,
            torch.einsum('bhij,bdij->bhid',v[1],stem)) for a,v in zip((old,attention),native)]
        torch.testing.assert_close(*ev,rtol=0,atol=0)
        return dict(kind='unchanged_attention_and_ev',passed=True,native_calls_per_model=1)
    finally:
        for h in handles:h.remove()
        old.clear();model.mod.pair_embed.clear();model._state_pair_cache.clear()


def compare(a, b, *, gradient=False, distributed=False):
    if distributed:
        rtol = P.PARITY['ddp_gradient_rtol' if gradient else 'ddp_forward_rtol']
        atol = P.PARITY['ddp_atol']
    else:
        rtol = P.PARITY['gradient_rtol' if gradient else 'forward_rtol']
        atol = P.PARITY['gradient_atol' if gradient else 'forward_atol']
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)


def boundary_safety(device):
    """Run masking/lifetime/precision/overflow assertions on the actual device."""
    seed_everything(101)
    base=M.build_model(trim=False).to(device).eval();active(base)
    values,_=B.tensors(B.fixed_batch(2,9),device)
    values[-1][1,:,-3:]=False
    snapshots=[]
    def record(_module,_args,result):
        memory=result[2]
        if memory is not None:snapshots.append(memory.detach())
    handles=[s.register_forward_hook(record) for s in M.states(base)]
    try:
        with torch.no_grad(),model_precision(base,device,training=device.type=='cuda'):
            base(*values)
        if len(snapshots)!=7:raise ValueError('Expected seven memory boundaries')
        for memory in snapshots:
            if (memory.dtype!=torch.float32 or not memory.isfinite().all()
                    or memory.abs().max()>1.000001
                    or memory[1,-3:].count_nonzero() or memory[1,:,-3:].count_nonzero()):
                raise ValueError('Invalid carried-memory precision/bounds/mask')
        if torch.allclose(snapshots[0],snapshots[0].transpose(1,2)):
            raise ValueError('Pair memory unexpectedly symmetric')
        if not snapshots[0][:,0,0].count_nonzero():raise ValueError('Valid self memory disappeared')
    finally:
        for h in handles:h.remove()
        snapshots.clear()
    with torch.no_grad(),model_precision(base,device,training=False):
        expected=base(*values)
        perm=torch.tensor([8,2,4,0,1,3,5,7,6],device=device)
        compare(base(*(v[:,:,perm] for v in values)),expected)
        base(*(v[:1,:,:1] for v in values))
        compare(base(*values),expected)
        writer=M.states(base)[2].memory_writer
        saved=writer.bias.detach().clone();writer.bias.fill_(float('inf'))
        try:
            try:base(*values)
            except FloatingPointError:pass
            else:raise ValueError('Bounded tanh concealed writer overflow')
        finally:writer.bias.copy_(saved)
        compare(base(*values),expected)
    if (base._state_pair_cache.normalized is not None
            or any(b.attn._bias is not None or b.attn.attention._relation_stem is not None for b in base.mod.blocks)):
        raise ValueError('Forward context leaked after exception')
    return dict(kind='boundaries_masks_lifetime',passed=True,boundaries=7,
                memory_precision='fp32',overflow_rejected=True,padding_and_permutation=True)


def correctness(device, rank, world):
    """Identity, active chunk backward, and global-batch SyncBN/DDP oracle."""
    from part_ev_message_study.parity import initial_parity
    seed_everything(101)
    initial=M.build_model(trim=False).to(device)
    adapter=SimpleNamespace(seed_everything=seed_everything,capture_rank_state=capture_rank_state,
        restore_rank_state=restore_rank_state,tensors=B.tensors,model_precision=model_precision)
    initialization=initial_parity(adapter,initial,P.TASK,B.fixed_batch(8,17),device,
        lambda _architecture,**kw:M.reference_model('state_layerwise',seed=101,**kw))
    del initial
    gc.collect(); torch.cuda.empty_cache()
    from .oracle import verify as recurrence_oracle
    records = [attention_invariance(device),recurrence_oracle(device),boundary_safety(device)]
    delayed=delayed_activation(device,rank,world)
    gc.collect(); torch.cuda.empty_cache()
    for kind in ('zero_identity','active_chunk'):
        seed_everything(101)
        fast = M.build_model(trim=False).to(device)
        seed_everything(101)
        oracle = (M.reference_model('state_layerwise',trim=False) if kind=='zero_identity'
                  else M.build_model(trim=False,chunk=128,checkpointed=False)).to(device)
        active(fast)
        if kind == 'zero_identity':
            with torch.no_grad():
                for p in M.readers(fast): p.zero_()
        oracle.load_state_dict({k:v for k,v in fast.state_dict().items()
                                if kind!='zero_identity' or not M.memory_key(k)})
        values,y = B.tensors(B.fixed_batch(2,67),device)
        values[-1][1,:,-9:] = False
        outputs, gradients = [], []
        for target in (oracle,fast):
            seed_everything(918)
            with model_precision(target,device,training=False):
                output = target(*values)
                F.cross_entropy(output,y).backward()
            outputs.append(output.detach())
            gradients.append({n:p.grad.detach().clone() for n,p in target.named_parameters() if p.grad is not None})
        compare(*outputs)
        for name, gradient in gradients[0].items(): compare(gradient,gradients[1][name],gradient=True)
        for p in (M.readers(fast) if kind=='zero_identity' else [*M.readers(fast),*M.writers_and_gates(fast)]):
            if p.grad is None or not p.grad.isfinite().all() or not p.grad.count_nonzero():
                raise ValueError('New memory parameter has no finite learning signal')
        records.append(dict(kind=kind,passed=True,particles=67,
                            max_output_error=float((outputs[0]-outputs[1]).abs().max())))
        del fast,oracle,target,outputs,gradients,gradient,output,values,y,p
        gc.collect(); torch.cuda.empty_cache()

    base, wrapped = fresh(device,world,trim=False)
    active(base)
    # Set parameters identically on both ranks after disposable activation.
    for p in base.parameters():
        if world>1: dist.broadcast(p.data,src=0)
    B.disable_dropout(base)
    seed_everything(101)
    oracle = M.build_model(trim=False).to(device)
    oracle.load_state_dict(base.state_dict()); B.disable_dropout(oracle)
    batch = B.fixed_batch(8,17)
    values,y = B.tensors(batch,device)
    with model_precision(oracle,device,training=False):
        expected=oracle(*values); F.cross_entropy(expected,y).backward()
    values,y = B.tensors(B.split_batch(batch,rank,world),device)
    with model_precision(base,device,training=False):
        actual=wrapped(*values); F.cross_entropy(actual,y).backward()
    compare(actual,expected[rank*8//world:(rank+1)*8//world],distributed=True)
    reference_params=dict(oracle.named_parameters())
    for name,p in base.named_parameters():
        if p.grad is None or reference_params[name].grad is None:
            raise ValueError(f'Unused distributed parameter: {name}')
        compare(p.grad,reference_params[name].grad,gradient=True,distributed=True)
    records.append(dict(kind='active_SyncBN_DDP',passed=True))
    return B.gather(dict(passed=True,rank=rank,checks=records,tolerances=P.PARITY,
                         initialization=initialization,delayed_activation=delayed),world)


def save_probe(path, base, optimizer, scaler, spec, rank, world):
    states = B.gather(capture_rank_state(base),world)
    payload = dict(campaign_sha256=spec['content_hash'],model=base.state_dict(),
        optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rank_states=states)
    root_call(lambda: atomic_torch_save(payload,path),rank,world)


def restore_probe(path, base, optimizer, scaler, spec, rank, world):
    payload = trusted_load(path)
    if payload['campaign_sha256'] != spec['content_hash'] or len(payload['rank_states']) != world:
        raise ValueError('Restart probe identity differs')
    base.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scaler.load_state_dict(payload['scaler'])
    restore_rank_state(base,payload['rank_states'][rank])


def assert_nested(actual, expected):
    if isinstance(expected,torch.Tensor):
        compare(actual.cpu(),expected.cpu(),gradient=True)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise ValueError('Restart state keys differ')
        for key in expected: assert_nested(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected): raise ValueError('Restart state lengths differ')
        for a,b in zip(actual,expected): assert_nested(a,b)
    elif actual != expected:
        raise ValueError(f'Restart state differs: {actual!r} != {expected!r}')


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def assert_rng(actual,expected):
    if (actual['python']!=expected['python'] or actual['numpy'][0]!=expected['numpy'][0]
            or not np.array_equal(actual['numpy'][1],expected['numpy'][1])
            or actual['numpy'][2:]!=expected['numpy'][2:]):
        raise ValueError('Python/NumPy RNG diverged after restart')
    torch.testing.assert_close(actual['torch'],expected['torch'],rtol=0,atol=0)
    if len(actual['cuda'])!=len(expected['cuda']): raise ValueError('CUDA RNG device count changed')
    for a,b in zip(actual['cuda'],expected['cuda']):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert_nested(actual['trimmers'],expected['trimmers'])


def diagnostic_sample(base, values, device):
    """Fixed-sample eval diagnostics, no optimizer step or data-cursor advance."""
    rng = capture_rank_state(base)
    flags = [(m,m.training) for m in base.modules()]
    trimmers = [(m,m.enabled) for m in base.modules() if type(m).__name__=='SequenceTrimmer']
    buffers = {n:b.detach().clone() for n,b in base.named_buffers()}
    switches = [(s,s.collect_diagnostics,s.last_diagnostics) for s in M.states(base)]
    try:
        base.eval()
        for m,_ in trimmers: m.enabled=False
        for a,_,_ in switches: a.collect_diagnostics=True
        with torch.no_grad(),model_precision(base,device,training=False):
            base(*values)
        return dict(events=len(values[0]),particles=values[0].shape[-1],precision='fp32',
            dropout=False,optimizer_steps=0,
            layers=[copy.deepcopy(s.last_diagnostics) for s in M.states(base)])
    finally:
        with torch.no_grad():
            for name,b in base.named_buffers(): b.copy_(buffers[name])
        for m,enabled in trimmers: m.enabled=enabled
        restore_rank_state(base,rng)
        for m,training in flags: m.training=training
        for a,collect,last in switches: a.collect_diagnostics,a.last_diagnostics=collect,last


def restart_probe(output, base, model, optimizer, scaler, values, truth, spec, rank, world):
    path = Path(output)/'probes/restart.pt'
    B.synchronize(values[0].device,world)
    start=time.monotonic()
    save_probe(path,base,optimizer,scaler,spec,rank,world)
    seconds=max(B.gather(time.monotonic()-start,world))
    expected=[]
    for index in range(P.POLICY['restart_steps']):
        result=B.step(base,model,optimizer,scaler,values,truth,spec['recipe'],index,world)
        expected.append((result['loss'],result['skipped']))
    snapshot=cpu_copy((base.state_dict(),optimizer.state_dict(),scaler.state_dict()))
    expected_rng=capture_rank_state(base)
    restore_probe(path,base,optimizer,scaler,spec,rank,world)
    actual=[]
    for index in range(P.POLICY['restart_steps']):
        result=B.step(base,model,optimizer,scaler,values,truth,spec['recipe'],index,world)
        actual.append((result['loss'],result['skipped']))
    compare(torch.tensor(actual),torch.tensor(expected))
    assert_nested((base.state_dict(),optimizer.state_dict(),scaler.state_dict()),snapshot)
    assert_rng(capture_rank_state(base),expected_rng)
    if not B.agree(sum(r[1] for r in actual)==0,values[0].device,world):
        raise ValueError('AMP did not settle for restart verification')
    return dict(passed=True,updates=7,ranks=world,full_state=True),seconds


def run(output,spec,rank,world,device):
    deadline=time.monotonic()+P.POLICY['max_seconds']
    checks=correctness(device,rank,world)
    gc.collect(); torch.cuda.empty_cache()
    base,model=fresh(device,world)
    optimizer=build_optimizer(base,spec['recipe'])
    root_call(lambda:P.write_json(Path(output)/'probes/optimizer.json',optimizer_metadata(base,optimizer)),rank,world)
    scaler=torch.amp.GradScaler('cuda')
    loader,_=build_loader('train',spec['inventory'],256,rank=rank,world_size=world,workers=1,seed=101)
    val_loader=None
    try:
        iterator=iter(loader)
        before=time.monotonic(); first=next(iterator); cold=time.monotonic()-before
        seed_everything(101+rank)
        torch.cuda.reset_peak_memory_stats(device)
        stream=B.measure(base,model,optimizer,scaler,
            lambda i:B.tensors(first if i==0 else next(iterator),device),spec['recipe'],device,world,
            P.POLICY['warmup'],P.POLICY['measured'],deadline,label='pair_memory_real_data')
        if any(not p.detach().count_nonzero() for p in M.readers(base)):
            raise ValueError('Memory reader remained inactive after sustained updates')
        for module in base.modules():
            if type(module).__name__=='SequenceTrimmer': module.enabled=False
        values,truth=B.tensors(B.split_batch(B.fixed_batch(),rank,world),device)
        fixed=B.measure(base,model,optimizer,scaler,lambda i:(values,truth),spec['recipe'],device,world,
                        P.POLICY['fixed_warmup'],P.POLICY['fixed_steps'],deadline,label='pair_memory_fixed128')
        # Resume on stochastic native trimming, not a permanently fixed test graph.
        for module in base.modules():
            if type(module).__name__=='SequenceTrimmer': module.enabled=True
        restart,seconds=restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world)
        diagnostic_values,_=B.tensors(B.fixed_batch(8,17),device)
        diagnostics=B.gather(diagnostic_sample(base,diagnostic_values,device),world)
        if not B.agree(not B.STOP and time.monotonic()<deadline,device,world):
            raise TimeoutError('Benchmark budget exhausted before validation')
        if rank==0: val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
        metric=validate(base,iter(val_loader) if rank==0 else None,device,rank,world,
                        P.POLICY['validation_events'],audit=True)
        if metric is None: raise TimeoutError('Interrupted validation')
        def inference_probe():
            base.eval()
            for module in base.modules():
                if type(module).__name__=='SequenceTrimmer': module.enabled=False
            with torch.no_grad(),model_precision(base,device,training=False):
                for _ in range(5): base(*values)
                torch.cuda.synchronize(device); begin=time.monotonic()
                for _ in range(20):
                    if B.STOP or time.monotonic()>=deadline: raise TimeoutError('Inference interrupted')
                    prediction=base(*values)
                    if not torch.isfinite(prediction).all(): raise ValueError('Nonfinite FP32 inference')
                torch.cuda.synchronize(device)
            throughput_seconds=time.monotonic()-begin
            single=[x[:1] for x in values]
            with torch.no_grad(),model_precision(base,device,training=False):
                for _ in range(10):base(*single)
                torch.cuda.synchronize(device);begin=time.monotonic()
                for _ in range(100):
                    if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Latency probe interrupted')
                    prediction=base(*single)
                torch.cuda.synchronize(device)
                if not prediction.isfinite().all():raise ValueError('Nonfinite batch-one prediction')
            return dict(batch_size=256,particles=128,batches=20,seconds=throughput_seconds,precision='fp32',
                        batch1_iterations=100,batch1_seconds=time.monotonic()-begin)
        inference=root_call(inference_probe,rank,world)
        finite=B.agree(finite_state(torch,(base.state_dict(),optimizer.state_dict())),device,world)
        prop=torch.cuda.get_device_properties(device)
        gpus=B.gather(dict(rank=rank,name=prop.name,total_memory=prop.total_memory,
             peak_allocated=torch.cuda.max_memory_allocated(device),
             peak_reserved=torch.cuda.max_memory_reserved(device),cold_fetch_seconds=cold,
             first_input_sha256=B.batch_fingerprint(first)),world)
        healthy=(stream['healthy'] and fixed['healthy'] and finite
                 and all('A100' in g['name'] and g['peak_allocated']<.9*g['total_memory'] for g in gpus))
        result=dict(complete=True,passed=healthy,authorizes_training=False,campaign_sha256=spec['content_hash'],
            correctness=checks,stream=stream,fixed=fixed,restart=restart,checkpoint_seconds=seconds,
            validation=metric,inference=inference,diagnostics=diagnostics,gpus=gpus,finite_state=finite,
            runtime=dict(world_size=world,local_batch=256,global_batch=512,workers_per_rank=1,
                batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,
                nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}))
        return result
    finally:
        shutdown_loader(loader)
        if val_loader is not None: shutdown_loader(val_loader)
