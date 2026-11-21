"""Disposable two-A100 support correctness, learning, restart and speed evidence."""
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
from .oracle import verify as aggregation_oracle
from .diagnostics import diagnostic_sample
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


def active(base, *, support=True):
    with torch.no_grad():
        for state in M.state_branches(base): state.output.normal_(std=.02)
        for head in base.mod.pair_embed.reference.embed.heads:
            for p in head.parameters(): p.add_(torch.randn_like(p)*.001)
        if support:
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
            raise ValueError('AMP ranks disagree during support activation')
        skips += row['skipped']
        if row['skipped']: continue
        if successes == 0:
            initial_zero = all(p.grad is not None and p.grad.isfinite().all() and not p.grad.count_nonzero()
                and torch.equal(p.detach(),old) for p,old in zip(M.upstream_parameters(base),before))
            initial_signal = all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                                 for p in M.output_parameters(base))
        successes += 1
        if successes == 8: break
    updated = all(not torch.equal(p.detach(),old) for p,old in zip(M.upstream_parameters(base),before))
    gradients = all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                    for n,p in base.named_parameters() if M.support_key(n))
    passed = bool(successes==8 and initial_zero and initial_signal and updated and gradients)
    if not B.agree(passed,device,world): raise ValueError('Support did not activate correctly')
    return dict(passed=True,attempts=attempt+1,successful_updates=successes,skips=skips,
        initial_upstream_zero=bool(initial_zero),initial_output_signal=bool(initial_signal),
        upstream_updated=bool(updated),all_support_gradients=bool(gradients))


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
        original.load_state_dict({k:v for k,v in model.state_dict().items() if not M.support_key(k)})
        compare(model(*(v[2:] for v in values)),original(*(v[2:] for v in values)))
        saved = M.branches(model)[0].gate.bias.clone()
        M.branches(model)[0].gate.bias.fill_(float('inf'))
        if model(*values).isfinite().all(): raise ValueError('Nonfinite support concealed by fusion')
        M.branches(model)[0].gate.bias.copy_(saved)
        compare(model(*values),expected)
    if model._state_pair_cache.normalized is not None or any(b.attn._bias is not None for b in model.mod.blocks):
        raise ValueError('Support forward context leaked')
    return dict(kind='boundaries_masks_lifetime',passed=True,empty_pairs=True,overflow_rejected=True)


def compare_particle_paths(reference, candidate):
    """Only computed attention activations get the existing FP32 tolerance."""
    for trace in (reference, candidate):
        if (len(trace['attention']) != 8 or len(trace['padding_masks']) != 8
                or len(trace['trimmed']) != 1):
            raise ValueError('Repeated/missing native attention or trimming')
    for field in ('inputs', 'trimmed', 'padding_masks'):
        torch.testing.assert_close(reference[field], candidate[field], rtol=0, atol=0,
                                   msg=lambda msg: f'Exact particle-path {field}: {msg}')
    assert_rng(candidate['rng'], reference['rng'])
    errors = []
    for block, (expected, actual) in enumerate(zip(reference['attention'], candidate['attention'])):
        if len(expected) != 2 or len(actual) != 2:
            raise ValueError('Expected native attention output and weights')
        maximum = 0.
        for x, y in zip(expected, actual):
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f'Nonfinite native attention at particle block {block}')
            torch.testing.assert_close(y, x, rtol=P.PARITY['forward_rtol'],
                atol=P.PARITY['forward_atol'], msg=lambda msg: f'Particle block {block}: {msg}')
            maximum = max(maximum, float((x-y).abs().max()))
        errors.append(maximum)
    return dict(kind='native_path_at_initialization', passed=True,
        version=P.NATIVE_PATH_VERSION, attention_calls=8, trimmer_calls=1,
        exact_inputs_masks_trimming_rng=True, forward_rtol=P.PARITY['forward_rtol'],
        forward_atol=P.PARITY['forward_atol'], block_max_abs_difference=errors)


def particle_path(device):
    seed_everything(101)
    model=M.build_model(trim=True).to(device);active(model,support=False)
    old=M.reference_model('state_layerwise',trim=True).to(device)
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not M.support_key(k)})
    expected_parameters=cpu_copy(dict(old.named_parameters()))
    def same_parameters(target):
        actual=cpu_copy({n:p for n,p in target.named_parameters() if not M.support_key(n)})
        torch.testing.assert_close(actual,expected_parameters,rtol=0,atol=0,
                                   msg=lambda msg: f'Exact base parameters: {msg}')
    def zero_support_outputs():
        if any(not torch.isfinite(p).all() or p.count_nonzero() for p in M.output_parameters(model)):
            raise ValueError('Native-path initialization requires exactly zero support outputs')
    same_parameters(model);zero_support_outputs()
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
            torch.testing.assert_close(cpu_copy(values),original,rtol=0,atol=0,
                                       msg=lambda msg: f'Particle-path input mutation: {msg}')
            same_parameters(target);zero_support_outputs()
            results.append(dict(inputs=cpu_copy(values),trimmed=trims,padding_masks=masks,
                                attention=records,rng=capture_rank_state(target)))
        finally:
            for h in hooks:h.remove()
    return dict(compare_particle_paths(*results),exact_base_parameters=True,zero_support_outputs=True)


def correctness(device,rank,world):
    records=[aggregation_oracle(device),boundaries(device),particle_path(device)]
    delayed=delayed_activation(device,rank,world)
    for kind in ('zero_identity','active_chunk','active_amp_chunk'):
        seed_everything(101);fast=M.build_model(trim=False).to(device)
        seed_everything(101)
        oracle=(M.reference_model('state_layerwise',trim=False) if kind=='zero_identity'
                else M.build_model(trim=False,chunk=128,checkpointed=False,oracle=True)).to(device)
        active(fast,support=kind!='zero_identity')
        oracle.load_state_dict({k:v for k,v in fast.state_dict().items()
                               if kind!='zero_identity' or not M.support_key(k)})
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
        parameters=M.output_parameters(fast) if kind=='zero_identity' else [p for n,p in fast.named_parameters() if M.support_key(n)]
        if any(p.grad is None or not p.grad.isfinite().all() or not p.grad.count_nonzero() for p in parameters):
            raise ValueError('Support parameter missing learning signal')
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
    # the support, no unused-parameter workaround, second step exercises DDP's reducer.
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


def restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world):
    path=Path(output)/'probes/restart.pt'
    B.synchronize(values[0].device,world);start=time.monotonic()
    rank_states=B.gather(capture_rank_state(base),world)
    progress=dict(attempt=19997,checks=0,fixture_cursor=0,successful_updates=0,skips=0,
                  best_accuracy=None,best_check=None,best_step=None)
    # Synthetic rolling-batch stream with an explicit replay cursor. It checks
    # restart plumbing, NOT a production ROOT-stream restart certification.
    root_call(lambda:atomic_torch_save(dict(campaign_sha256=spec['content_hash'],
        model=base.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        rank_states=rank_states,progress=progress),path),rank,world)
    seconds=max(B.gather(time.monotonic()-start,world))
    midpoint_path=Path(output)/'probes/selected_restart.pt';midpoint_times=[]
    def sequence(cursor,*,count=7,best_weights=None,save_midpoint=False):
        rows=[]
        for _ in range(count):
            inputs=[v.roll(cursor['fixture_cursor'],dims=-1) for v in values]
            row=B.step(base,model,optimizer,scaler,inputs,truth,spec['recipe'],cursor['attempt'],world)
            rows.append((row['loss'],row['skipped']))
            cursor['skips']+=int(row['skipped']);cursor['successful_updates']+=int(not row['skipped'])
            cursor['attempt']+=1;cursor['fixture_cursor']+=1
            if cursor['attempt']%20000==0:
                def validation_boundary():
                    base.eval()
                    with torch.no_grad(),model_precision(base,values[0].device,training=False):
                        result=base(*(v[:8] for v in inputs))
                    if not result.isfinite().all():raise ValueError('Restart validation not finite')
                    return float((result.argmax(-1)==truth[:8]).float().mean())
                accuracy=root_call(validation_boundary,rank,world);cursor['checks']+=1
                if cursor['best_accuracy'] is None or accuracy>cursor['best_accuracy']:
                    cursor.update(best_accuracy=accuracy,best_check=cursor['checks'],best_step=cursor['attempt'])
                    best_weights=cpu_copy(base.state_dict())
            if save_midpoint and cursor['attempt']==20001:
                # Resume AFTER selection as well as before it. Rank-local copies
                # also support the CPU/Gloo fixture's unsynchronized BN buffers;
                # the actual GPU benchmark uses SyncBatchNorm on both ranks.
                before=time.monotonic()
                states=B.gather(capture_rank_state(base),world)
                selected=B.gather(best_weights,world)
                root_call(lambda:atomic_torch_save(dict(campaign_sha256=spec['content_hash'],
                    model=base.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    rank_states=states,progress=copy.deepcopy(cursor),best_by_rank=selected),midpoint_path),rank,world)
                midpoint_times.extend(B.gather(time.monotonic()-before,world))
        return rows,cursor,best_weights
    expected,final_progress,expected_best=sequence(copy.deepcopy(progress),save_midpoint=True)
    state=cpu_copy((base.state_dict(),optimizer.state_dict(),scaler.state_dict()));rng=capture_rank_state(base)
    # Repeat twice from the same serialized state, distinguishing ordinary
    # repeatability from resume behavior, and crossing a Lookahead boundary.
    for _ in range(2):
        saved=trusted_load(path)
        if saved['campaign_sha256']!=spec['content_hash'] or len(saved['rank_states'])!=world:
            raise ValueError('Probe checkpoint identity changed')
        base.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']);restore_rank_state(base,saved['rank_states'][rank])
        actual,completed,actual_best=sequence(saved['progress'])
        assert_nested(actual_best,expected_best)
        compare(torch.tensor(actual),torch.tensor(expected))
        assert_nested((base.state_dict(),optimizer.state_dict(),scaler.state_dict()),state)
        assert_rng(capture_rank_state(base),rng)
        if completed!=final_progress:raise ValueError('Restart update/validation/replay cursors differ')
    saved=trusted_load(midpoint_path)
    if (saved['campaign_sha256']!=spec['content_hash'] or saved['progress']['best_step']!=20000
            or saved['progress']['best_check']!=1 or len(saved['best_by_rank'])!=world):
        raise ValueError('Selected checkpoint bookkeeping did not round-trip')
    base.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler']);restore_rank_state(base,saved['rank_states'][rank])
    tail,completed,selected=sequence(saved['progress'],count=3,best_weights=saved['best_by_rank'][rank])
    compare(torch.tensor(tail),torch.tensor(expected[-3:]))
    assert_nested(selected,expected_best)
    assert_nested((base.state_dict(),optimizer.state_dict(),scaler.state_dict()),state)
    assert_rng(capture_rank_state(base),rng)
    if completed!=final_progress:raise ValueError('Resume after validation lost progress/selection')
    if not B.agree(sum(r[1] for r in actual)==0,values[0].device,world):raise ValueError('Restart AMP unsettled')
    return dict(P.RESTART,ranks=world),max(seconds,*midpoint_times)


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


def validation_batches(loader,deadline,sample):
    """Capture only the first eight CPU records from this validation probe."""
    for batch in loader:
        if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Validation deadline')
        if not sample:
            sample.append(tuple({k:v[:8].detach().cpu().clone() for k,v in mapping.items()}
                                for mapping in batch[:2]))
        yield batch


def measure_arm(output,spec,rank,world,device,deadline,*,baseline=False):
    base,model=fresh(device,world,baseline=baseline)
    optimizer=build_optimizer(base,spec['recipe']);scaler=torch.amp.GradScaler('cuda')
    if not baseline:root_call(lambda:P.write_json(Path(output)/'probes/optimizer.json',optimizer_metadata(base,optimizer)),rank,world)
    loader,_=build_loader('train',spec['inventory'],256,rank=rank,world_size=world,workers=1,seed=101)
    val_loader=None
    try:
        iterator=iter(loader);before=time.monotonic();first=next(iterator);cold=time.monotonic()-before
        seed_everything(101+rank);torch.cuda.reset_peak_memory_stats(device)
        label='baseline_state_ev' if baseline else 'pair_support'
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
            result['support_gradient_norms']=B.gather(gradient_norms(base),world)
            validation_sample=[]
            if not B.agree(not B.STOP and time.monotonic()<deadline,device,world):raise TimeoutError('No validation time')
            if rank==0:val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
            # Iteration guard makes inherited validation obey this diagnostic's deadline.
            iterator=validation_batches(val_loader,deadline,validation_sample) if rank==0 else None
            metric=validate(base,iterator,device,rank,world,128000,audit=True)
            if metric is None:raise TimeoutError('Validation interrupted')
            sample=root_call(lambda:validation_sample[0],rank,world)
            diag_values,_=B.tensors(sample,device)
            diagnostics=B.gather(dict(diagnostic_sample(base,diag_values,device),rank=rank,
                population='first_8_validation_probe',sample_sha256=B.batch_fingerprint(sample)),world)
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
        import resource
        result['host_peak_rss_bytes']=B.gather(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),world)
        if not baseline:result['parameter_inventory_sha256']=P.digest(P.parameter_inventory(base))
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
    if rank==0:print(dict(stage='support_correctness',parameters=M.PARAMETERS),flush=True)
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
        measurement_order=['baseline_state_ev','pair_support'],
        runtime=dict(world_size=world,local_batch=256,global_batch=512,workers_per_rank=1,
            batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,
            nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}))
    return result
