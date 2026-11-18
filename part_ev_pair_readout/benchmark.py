"""Disposable two-A100 readout correctness, learning, restart and speed evidence."""
import copy
import gc
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from . import model as M, protocol as P
from .oracle import verify as pooling_oracle
from part_ev_gpu_scaling import benchmark as B
from part_ev_layerwise_ddp.engine import root_call, validate
from part_ev_fullscale.checkpoint import (seed_everything, capture_rank_state, restore_rank_state,
                                         atomic_torch_save, trusted_load)
from part_ev_fullscale.optimizer import build_optimizer, optimizer_metadata
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_screen.precision import model_precision
from scripts.check_part_ev_screen_amp_recovery import finite_state
# These helpers accept the model/spec explicitly. No old module globals patched.
from part_ev_pair_memory.benchmark import cpu_copy, assert_nested, assert_rng


def fresh(device, world, *, trim=True, baseline=False):
    seed_everything(101)
    base = (M.reference_model('state_layerwise',trim=trim) if baseline else M.build_model(trim=trim)).to(device)
    if world == 2:
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
        model = DDP(base,device_ids=[device.index],output_device=device.index,
                    find_unused_parameters=False,gradient_as_bucket_view=True)
    else: model = base
    return base,model


def active(base, *, readout=True):
    with torch.no_grad():
        for state in M.state_branches(base): state.output.normal_(std=.02)
        for head in base.mod.pair_embed.reference.embed.heads:
            for p in head.parameters(): p.add_(torch.randn_like(p)*.001)
        if readout:
            for p in M.output_parameters(base): p.normal_(std=.02)


def compare(a,b,*,gradient=False,distributed=False):
    rtol = P.PARITY[('ddp_' if distributed else '')+('gradient_rtol' if gradient else 'forward_rtol')]
    atol = P.PARITY['ddp_atol' if distributed else 'gradient_atol' if gradient else 'forward_atol']
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)


def delayed_activation(device,rank,world):
    base,model = fresh(device,world,trim=False)
    optimizer = build_optimizer(base,P.recipes()['ranger_full_v1'])
    scaler = torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    values,truth = B.tensors(B.split_batch(B.fixed_batch(8,17),rank,world),device)
    before = [p.detach().clone() for p in M.upstream_parameters(base)]
    successes=skips=0
    initial_zero=initial_signal=False
    for attempt in range(20):
        row = B.step(base,model,optimizer,scaler,values,truth,P.recipes()['ranger_full_v1'],attempt,world)
        if len(set(B.gather((row['skipped'],scaler.get_scale()),world))) != 1:
            raise ValueError('AMP ranks disagree during readout activation')
        skips += row['skipped']
        if row['skipped']: continue
        if successes == 0:
            initial_zero = all(p.grad is not None and p.grad.isfinite().all() and not p.grad.count_nonzero()
                and torch.equal(p.detach(),old) for p,old in zip(M.upstream_parameters(base),before))
            initial_signal = all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                                 for p in M.output_parameters(base))
        successes += 1
        if successes == 3: break
    updated = all(not torch.equal(p.detach(),old) for p,old in zip(M.upstream_parameters(base),before))
    gradients = all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                    for p in base.mod.pair_readout.parameters())
    passed = bool(successes==3 and initial_zero and initial_signal and updated and gradients)
    if not B.agree(passed,device,world): raise ValueError('Readout did not activate correctly')
    return dict(passed=True,attempts=attempt+1,successful_updates=successes,skips=skips,
        initial_upstream_zero=bool(initial_zero),initial_output_signal=bool(initial_signal),
        upstream_updated=bool(updated),all_readout_gradients=bool(gradients))


def boundaries(device):
    seed_everything(101)
    model = M.build_model(trim=False).to(device).eval(); active(model)
    values,_ = B.tensors(B.fixed_batch(3,9),device)
    values[-1][1,:,-3:] = False
    values[-1][2,:,1:] = False
    with torch.no_grad(),model_precision(model,device,training=False):
        expected = model(*values)
        perm = torch.tensor([4,8,1,3,6,0,7,2,5],device=device)
        compare(model(*(v[:,:,perm] for v in values)),expected)
        original = M.reference_model('state_layerwise',trim=False).to(device).eval()
        original.load_state_dict({k:v for k,v in model.state_dict().items() if not M.readout_key(k)})
        compare(model(*(v[2:] for v in values)),original(*(v[2:] for v in values)))
        saved = model.mod.pair_readout.key.weight.clone()
        model.mod.pair_readout.key.weight.fill_(float('inf'))
        if model(*values).isfinite().all(): raise ValueError('Nonfinite readout concealed by fusion')
        model.mod.pair_readout.key.weight.copy_(saved)
        compare(model(*values),expected)
    if model._state_pair_cache.normalized is not None or any(b.attn._bias is not None for b in model.mod.blocks):
        raise ValueError('Readout forward context leaked')
    return dict(kind='boundaries_masks_lifetime',passed=True,empty_pairs=True,overflow_rejected=True)


def compare_particle_paths(reference, candidate):
    """Only computed floating activations get the declared FP32 tolerance."""
    for trace in (reference, candidate):
        if (len(trace['attention']) != 8 or len(trace['padding_masks']) != 8
                or len(trace['trimmed']) != 1):
            raise ValueError('Repeated/missing native attention or trimming')
    for field in ('inputs', 'trimmed', 'padding_masks'):
        torch.testing.assert_close(reference[field], candidate[field], rtol=0, atol=0,
                                   msg=lambda msg: f'Exact particle-path {field}: {msg}')
    assert_rng(candidate['rng'], reference['rng'])
    errors = []
    for block, (a, b) in enumerate(zip(reference['attention'], candidate['attention'])):
        if len(a) != 2 or len(b) != 2:
            raise ValueError('Expected native attention output and weights')
        maximum = 0.
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=P.PARITY['forward_rtol'],
                atol=P.PARITY['forward_atol'], msg=lambda msg: f'Particle block {block}: {msg}')
            maximum = max(maximum, float((x-y).abs().max()))
        errors.append(maximum)
    return dict(kind='unchanged_particle_path', passed=True, attention_calls=8, trimmer_calls=1,
        exact_inputs_masks_trimming_rng=True, forward_rtol=P.PARITY['forward_rtol'],
        forward_atol=P.PARITY['forward_atol'], block_max_abs_difference=errors)


def particle_path(device):
    seed_everything(101)
    model=M.build_model(trim=True).to(device);active(model)
    old=M.reference_model('state_layerwise',trim=True).to(device)
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not M.readout_key(k)})
    def same_parameters():
        actual={n:p.detach() for n,p in model.named_parameters() if not M.readout_key(n)}
        expected={n:p.detach() for n,p in old.named_parameters()}
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    same_parameters()
    values,_=B.tensors(B.fixed_batch(4,17),device)
    values[-1][:,:,10:]=False
    original=cpu_copy(values)
    results=[]
    for target in (old,model):
        records=[];trims=[];masks=[]
        target.mod.trimmer._counter=10
        hooks=[b.attn.attention.reference.register_forward_hook(
            lambda _m,_a,out: records.append(tuple(t.detach().clone() for t in out))) for b in target.mod.blocks]
        hooks += [b.attn.attention.reference.register_forward_pre_hook(
            lambda _m,_a,kw: masks.append(cpu_copy(kw['key_padding_mask'])), with_kwargs=True)
            for b in target.mod.blocks]
        hooks.append(target.mod.trimmer.register_forward_hook(lambda _m,_a,out:trims.append(cpu_copy(out))))
        try:
            seed_everything(991)
            with torch.no_grad():target(*values)
            torch.testing.assert_close(cpu_copy(values),original,rtol=0,atol=0)
            results.append(dict(inputs=cpu_copy(values),trimmed=trims,padding_masks=masks,
                                attention=records,rng=capture_rank_state(target)))
        finally:
            for h in hooks:h.remove()
    same_parameters()
    return dict(compare_particle_paths(*results),exact_base_parameters=True)


def correctness(device,rank,world):
    records=[pooling_oracle(device),boundaries(device),particle_path(device)]
    delayed=delayed_activation(device,rank,world)
    for kind in ('zero_identity','active_chunk','active_amp_chunk'):
        seed_everything(101);fast=M.build_model(trim=False).to(device)
        seed_everything(101)
        oracle=(M.reference_model('state_layerwise',trim=False) if kind=='zero_identity'
                else M.build_model(trim=False,chunk=128,checkpointed=False)).to(device)
        active(fast,readout=kind!='zero_identity')
        oracle.load_state_dict({k:v for k,v in fast.state_dict().items()
                               if kind!='zero_identity' or not M.readout_key(k)})
        values,y=B.tensors(B.fixed_batch(2,67),device);values[-1][1,:,-9:]=False
        outputs=[];grads=[]
        for target in (oracle,fast):
            seed_everything(918)
            with model_precision(target,device,training=kind=='active_amp_chunk'):
                out=target(*values);F.cross_entropy(out,y).backward()
            outputs.append(out.detach())
            grads.append({n:p.grad.detach().clone() for n,p in target.named_parameters() if p.grad is not None})
        if kind=='active_amp_chunk':
            torch.testing.assert_close(*outputs,rtol=P.PARITY['amp_forward_rtol'],atol=P.PARITY['amp_forward_atol'])
            for name,g in grads[0].items():
                torch.testing.assert_close(g,grads[1][name],rtol=P.PARITY['amp_gradient_rtol'],atol=P.PARITY['amp_gradient_atol'])
        else:
            compare(*outputs)
            for name,g in grads[0].items():compare(g,grads[1][name],gradient=True)
        parameters=M.output_parameters(fast) if kind=='zero_identity' else list(fast.mod.pair_readout.parameters())
        if any(p.grad is None or not p.grad.isfinite().all() or not p.grad.count_nonzero() for p in parameters):
            raise ValueError('Readout parameter missing learning signal')
        records.append(dict(kind=kind,passed=True,particles=67))
        del target,oracle,fast,parameters,grads,outputs,out,g
        gc.collect();torch.cuda.empty_cache()
    for kind in ('active_SyncBN_DDP','no_pairs_DDP'):
        base,wrapped=fresh(device,world,trim=False);active(base);B.disable_dropout(base)
        for p in base.parameters():
            if world>1:dist.broadcast(p.data,src=0)
        oracle=M.build_model(trim=False).to(device)
        oracle.load_state_dict(base.state_dict());B.disable_dropout(oracle)
        batch=B.fixed_batch(8,17)
        if kind=='no_pairs_DDP':batch[0]['pf_mask'][:4,:,1:]=False
        values,y=B.tensors(batch,device)
        with model_precision(oracle,device,training=False):
            expected=oracle(*values);F.cross_entropy(expected,y).backward()
        values,y=B.tensors(B.split_batch(batch,rank,world),device)
        with model_precision(base,device,training=False):
            actual=wrapped(*values);F.cross_entropy(actual,y).backward()
        compare(actual,expected[rank*8//world:(rank+1)*8//world],distributed=True)
        reference_params=dict(oracle.named_parameters())
        for name,p in base.named_parameters():
            if p.grad is None or reference_params[name].grad is None:raise ValueError('Unused DDP parameter: '+name)
            compare(p.grad,reference_params[name].grad,gradient=True,distributed=True)
        records.append(dict(kind=kind,passed=True))
        del wrapped,base,oracle,reference_params,p,actual,expected
        gc.collect();torch.cuda.empty_cache()
    # SyncBN reduces uneven populations correctly; no equal-N assumption in
    # the readout, no unused-parameter workaround, second step exercises DDP's reducer.
    base,wrapped=fresh(device,world,trim=True);active(base)
    optimizer=build_optimizer(base,P.recipes()['ranger_full_v1'])
    values,y=B.tensors(B.split_batch(B.fixed_batch(8,67),rank,world),device)
    values[-1][:,:,33 if rank==0 else 67:]=False
    base.mod.trimmer._counter=10
    lengths=[]
    handle=base.mod.trimmer.register_forward_hook(lambda _m,_i,out:lengths.append(out[0].shape[-1]))
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with model_precision(base,device,training=False):
                logits=wrapped(*values);F.cross_entropy(logits,y).backward()
            if not B.agree(all(p.grad is not None and p.grad.isfinite().all() for p in base.parameters()),device,world):
                raise ValueError('Unequal-length DDP graph or gradient failure')
            optimizer.step()
    finally:handle.remove()
    rank_lengths=B.gather(lengths,world)
    if world==2 and rank_lengths!=[[33,33],[67,67]]:raise ValueError('Unequal native trimming was not exercised')
    if len(set(B.gather(B.parameter_digest(base),world)))!=1:raise ValueError('Unequal-length replicas diverged')
    records.append(dict(kind='unequal_lengths_DDP',passed=True,rank_lengths=rank_lengths))
    return B.gather(dict(passed=True,rank=rank,checks=records,tolerances=P.PARITY,delayed_activation=delayed),world)


def diagnostic_sample(base,values,device):
    rng=capture_rank_state(base)
    flags=[(m,m.training) for m in base.modules()]
    trims=[(m,m.enabled) for m in base.modules() if type(m).__name__=='SequenceTrimmer']
    try:
        base.eval()
        for m,_ in trims:m.enabled=False
        base._state_pair_cache.clear()
        with torch.no_grad(),model_precision(base,device,training=False):
            logits,(h,s,delta,present,healthy,projection)=base.mod.forward_details(
                values[1],values[2],values[3],diagnostic=True)
            # Detached scalar diagnostics only, never retained pair activations.
            similarity=F.cosine_similarity(s[:,0],s[:,1],dim=-1)
            disabled=base.mod.fc(h)
            swapped=base.mod.fc(base.mod.pair_readout.fuse(h,s.flip(1),present)[0])
            zeroed=base.mod.fc(base.mod.pair_readout.fuse(h,torch.zeros_like(s),present)[0])
            return dict(projection,events=len(values[0]),particles=values[0].shape[-1],healthy=bool(healthy),
                summary_rms=float(s.square().mean().sqrt()),summary_similarity=float(similarity.mean()),
                query0_norm=float(base.mod.pair_readout.queries[0].norm()),
                query1_norm=float(base.mod.pair_readout.queries[1].norm()),
                delta_to_class_norm=float(delta.norm()/h.norm().clamp_min(1e-12)),
                disabled_logit_rms_change=float((disabled-logits).square().mean().sqrt()),
                swapped_summary_logit_rms_change=float((swapped-logits).square().mean().sqrt()),
                zero_summary_logit_rms_change=float((zeroed-logits).square().mean().sqrt()),
                logits_finite=bool(logits.isfinite().all()),eligible_jets=int(present.sum()))
    finally:
        base._state_pair_cache.clear()
        for m,enabled in trims:m.enabled=enabled
        for m,training in flags:m.training=training
        restore_rank_state(base,rng)


def restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world):
    path=Path(output)/'probes/restart.pt'
    B.synchronize(values[0].device,world);start=time.monotonic()
    rank_states=B.gather(capture_rank_state(base),world)
    progress=dict(attempt=19997,checks=0,fixture_cursor=0)
    # Synthetic rolling-batch stream with an explicit replay cursor. It checks
    # restart plumbing, NOT a production ROOT-stream restart certification.
    root_call(lambda:atomic_torch_save(dict(campaign_sha256=spec['content_hash'],
        model=base.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        rank_states=rank_states,progress=progress),path),rank,world)
    seconds=max(B.gather(time.monotonic()-start,world))
    def sequence(cursor):
        rows=[]
        for _ in range(7):
            inputs=[v.roll(cursor['fixture_cursor'],dims=-1) for v in values]
            row=B.step(base,model,optimizer,scaler,inputs,truth,spec['recipe'],cursor['attempt'],world)
            rows.append((row['loss'],row['skipped']))
            cursor['attempt']+=1;cursor['fixture_cursor']+=1
            if cursor['attempt']%20000==0:
                def validation_boundary():
                    base.eval()
                    with torch.no_grad(),model_precision(base,values[0].device,training=False):
                        result=base(*(v[:8] for v in inputs))
                    if not result.isfinite().all():raise ValueError('Restart validation not finite')
                    return result.cpu()
                root_call(validation_boundary,rank,world);cursor['checks']+=1
        return rows,cursor
    expected,final_progress=sequence(copy.deepcopy(progress))
    state=cpu_copy((base.state_dict(),optimizer.state_dict(),scaler.state_dict()));rng=capture_rank_state(base)
    # Repeat twice from the same serialized state, distinguishing ordinary
    # repeatability from resume behavior, and crossing a Lookahead boundary.
    for _ in range(2):
        saved=trusted_load(path)
        if saved['campaign_sha256']!=spec['content_hash'] or len(saved['rank_states'])!=world:
            raise ValueError('Probe checkpoint identity changed')
        base.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']);restore_rank_state(base,saved['rank_states'][rank])
        actual,completed=sequence(saved['progress'])
        compare(torch.tensor(actual),torch.tensor(expected))
        assert_nested((base.state_dict(),optimizer.state_dict(),scaler.state_dict()),state)
        assert_rng(capture_rank_state(base),rng)
        if completed!=final_progress:raise ValueError('Restart update/validation/replay cursors differ')
    if not B.agree(sum(r[1] for r in actual)==0,values[0].device,world):raise ValueError('Restart AMP unsettled')
    return dict(P.RESTART,ranks=world),seconds


def inference(base,values,device,deadline):
    base.eval()
    for m in base.modules():
        if type(m).__name__=='SequenceTrimmer':m.enabled=False
    def timed(inputs,warmup,count):
        with torch.no_grad(),model_precision(base,device,training=False):
            for _ in range(warmup):base(*inputs)
            torch.cuda.synchronize(device);start=time.monotonic()
            for _ in range(count):
                if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Inference budget exhausted')
                prediction=base(*inputs)
                if not prediction.isfinite().all():raise ValueError('Nonfinite inference')
            torch.cuda.synchronize(device)
        return time.monotonic()-start
    return dict(batch_size=256,particles=128,batches=20,precision='fp32',
        seconds=timed(values,5,20),batch1_iterations=100,
        batch1_seconds=timed([v[:1] for v in values],10,100))


def measure_arm(output,spec,rank,world,device,deadline,*,baseline=False):
    base,model=fresh(device,world,baseline=baseline)
    optimizer=build_optimizer(base,spec['recipe']);scaler=torch.amp.GradScaler('cuda')
    if not baseline:root_call(lambda:P.write_json(Path(output)/'probes/optimizer.json',optimizer_metadata(base,optimizer)),rank,world)
    loader,_=build_loader('train',spec['inventory'],256,rank=rank,world_size=world,workers=1,seed=101)
    val_loader=None
    try:
        iterator=iter(loader);before=time.monotonic();first=next(iterator);cold=time.monotonic()-before
        seed_everything(101+rank);torch.cuda.reset_peak_memory_stats(device)
        label='baseline_state_ev' if baseline else 'pair_readout'
        stream=B.measure(base,model,optimizer,scaler,
            lambda i:B.tensors(first if i==0 else next(iterator),device),spec['recipe'],device,world,
            20,4200,deadline,label=label+'_stream')
        for m in base.modules():
            if type(m).__name__=='SequenceTrimmer':m.enabled=False
        values,truth=B.tensors(B.split_batch(B.fixed_batch(),rank,world),device)
        fixed=B.measure(base,model,optimizer,scaler,lambda i:(values,truth),spec['recipe'],device,world,
                        10,40,deadline,label=label+'_fixed128')
        result=dict(stream=stream,fixed=fixed,parameters=sum(p.numel() for p in base.parameters()))
        if not baseline:
            for m in base.modules():
                if type(m).__name__=='SequenceTrimmer':m.enabled=True
            restart,seconds=restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world)
            from .diagnostics import gradient_norms
            result['readout_gradient_norms']=B.gather(gradient_norms(base),world)
            diag_values,_=B.tensors(B.fixed_batch(8,17),device)
            diagnostics=B.gather(diagnostic_sample(base,diag_values,device),world)
            if not B.agree(not B.STOP and time.monotonic()<deadline,device,world):raise TimeoutError('No validation time')
            if rank==0:val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
            # Iteration guard makes inherited validation obey this diagnostic's deadline.
            def bounded_val():
                for batch in val_loader:
                    if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Validation deadline')
                    yield batch
            metric=validate(base,iter(bounded_val()) if rank==0 else None,device,rank,world,128000,audit=True)
            if metric is None:raise TimeoutError('Validation interrupted')
            # Explicit worst-length FP32 validation batch, not only typical lengths.
            def full_validation_memory():
                full,_=B.tensors(B.fixed_batch(512,128),device)
                base.eval()
                for m in base.modules():
                    if type(m).__name__=='SequenceTrimmer':m.enabled=False
                with torch.no_grad(),model_precision(base,device,training=False):
                    if not base(*full).isfinite().all():raise ValueError('Invalid full128 validation')
                return True
            root_call(full_validation_memory,rank,world)
            result.update(restart=restart,checkpoint_seconds=seconds,validation=metric,diagnostics=diagnostics,
                          full128_validation_passed=True)
        result['inference']=root_call(lambda:inference(base,values,device,deadline),rank,world)
        result['finite_state']=B.agree(finite_state(torch,(base.state_dict(),optimizer.state_dict())),device,world)
        prop=torch.cuda.get_device_properties(device)
        result['gpus']=B.gather(dict(rank=rank,name=prop.name,total_memory=prop.total_memory,
            peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),
            cold_fetch_seconds=cold,first_input_sha256=B.batch_fingerprint(first)),world)
        return result
    finally:
        shutdown_loader(loader)
        if val_loader is not None:shutdown_loader(val_loader)


def run(output,spec,rank,world,device):
    deadline=time.monotonic()+P.POLICY['max_seconds']
    gpu=torch.cuda.get_device_properties(device)
    identities=B.gather((gpu.name,gpu.total_memory),world)
    if world!=2 or len(set(identities))!=1 or 'A100' not in gpu.name:
        raise ValueError('Need two matching same-node A100s, no implicit hardware substitution')
    if rank==0:print(dict(stage='readout_correctness',parameters=M.PARAMETERS),flush=True)
    checks=correctness(device,rank,world)
    gc.collect();torch.cuda.empty_cache()
    baseline=measure_arm(output,spec,rank,world,device,deadline,baseline=True)
    gc.collect();torch.cuda.empty_cache()
    candidate=measure_arm(output,spec,rank,world,device,deadline)
    healthy=all(r['stream']['healthy'] and r['fixed']['healthy'] and r['finite_state'] and
        all('A100' in g['name'] and g['peak_allocated']<.9*g['total_memory'] for g in r['gpus'])
        for r in (baseline,candidate))
    result=dict(candidate,complete=True,passed=healthy,authorizes_training=False,campaign_sha256=spec['content_hash'],
        baseline_parameters=baseline['parameters'],baseline_stream=baseline['stream'],baseline_fixed=baseline['fixed'],
        baseline_inference=baseline['inference'],baseline_gpus=baseline['gpus'],correctness=checks,
        baseline_finite_state=baseline['finite_state'],
        measurement_order=['baseline_state_ev','pair_readout'],
        runtime=dict(world_size=world,local_batch=256,global_batch=512,workers_per_rank=1,
            batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,
            nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}))
    return result
