"""Disposable two-A100 triangle correctness, learning, restart and speed evidence."""
import gc
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from . import model as M, protocol as P
from .oracle import verify as triangle_oracle
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


def active(base, *, triangle=True):
    with torch.no_grad():
        for state in M.state_branches(base): state.output.normal_(std=.02)
        for head in base.mod.pair_embed.reference.embed.heads:
            for p in head.parameters(): p.add_(torch.randn_like(p)*.001)
        if triangle:
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
            raise ValueError('AMP ranks disagree during triangle activation')
        skips += row['skipped']
        if row['skipped']: continue
        if successes == 0:
            initial_zero = all(p.grad is not None and p.grad.isfinite().all() and not p.grad.count_nonzero()
                and torch.equal(p.detach(),old) for p,old in zip(M.upstream_parameters(base),before))
            initial_signal = all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                                 for p in M.output_parameters(base))
        successes += 1
        # Early unrectified RAdam updates can lie below a float32 weight ULP.
        # Cross its rectification/first Lookahead boundary, without changing LR
        # or initialization, before requiring EVERY upstream matrix to move.
        if successes == P.POLICY['delayed_successful_updates']: break
    updated = all(not torch.equal(p.detach(),old) for p,old in zip(M.upstream_parameters(base),before))
    gradients = all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
                    for n,p in base.named_parameters() if M.triangle_key(n))
    passed = bool(successes==P.POLICY['delayed_successful_updates'] and initial_zero and initial_signal and updated and gradients)
    if not B.agree(passed,device,world): raise ValueError('Triangle did not activate correctly')
    return dict(passed=True,attempts=attempt+1,successful_updates=successes,skips=skips,
        initial_upstream_zero=bool(initial_zero),initial_output_signal=bool(initial_signal),
        upstream_updated=bool(updated),all_triangle_gradients=bool(gradients))


def boundaries(device):
    seed_everything(101);model=M.build_model(trim=False).to(device).eval();active(model)
    values,_=B.tensors(B.fixed_batch(3,11),device)
    values[-1][1,:,-3:]=False;values[-1][2,:,2:]=False
    values[2][0]=values[2][0,:,:1]  # One jet takes the full cutoff-tie fallback.
    with torch.no_grad(),model_precision(model,device,training=False):
        expected=model(*values)
        perm=torch.tensor([4,8,1,10,3,6,0,7,2,9,5],device=device)
        compare(model(*(v[:,:,perm] for v in values)),expected)
        original=M.reference_model('state_layerwise',trim=False).to(device).eval()
        original.load_state_dict({k:v for k,v in model.state_dict().items() if not M.triangle_key(k)})
        compare(model(*(v[2:] for v in values)),original(*(v[2:] for v in values)))
        saved=M.branches(model)[0].gate.bias.clone()
        M.branches(model)[0].gate.bias.fill_(float('inf'))
        if model(*values).isfinite().all():raise ValueError('Nonfinite triangle preactivation hidden')
        M.branches(model)[0].gate.bias.copy_(saved)
        compare(model(*values),expected)
    if model._state_pair_cache.normalized is not None or any(b.attn._bias is not None for b in model.mod.blocks):
        raise ValueError('Triangle forward context leaked')
    return dict(kind='boundaries_masks_lifetime',passed=True,no_third_particle=True,overflow_rejected=True)


def particle_path(device):
    seed_everything(101);model=M.build_model(trim=True).to(device);active(model,triangle=False)
    old=M.reference_model('state_layerwise',trim=True).to(device)
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not M.triangle_key(k)})
    values,truth=B.tensors(B.fixed_batch(4,17),device);values[-1][:,:,10:]=False
    records=[];geometries=[]
    for target in (old,model):
        native=[];trims=[];target.mod.trimmer._counter=10
        hooks=[b.attn.attention.reference.register_forward_hook(
            lambda _m,_a,out:native.append(tuple(t.detach().clone() for t in out))) for b in target.mod.blocks]
        hooks.append(target.mod.trimmer.register_forward_hook(lambda _m,_i,out:trims.append(out[0].shape[-1])))
        if target is model:
            hooks.append(target.mod.triangle_neighbors.register_forward_hook(lambda _m,args,_o:geometries.append(args[0].shape[-1])))
        try:
            seed_everything(991)
            prediction=target(*values);F.cross_entropy(prediction,truth).backward()
        finally:
            for h in hooks:h.remove()
        if len(native)!=8 or trims!=[10]:raise ValueError('Native trimming/attention repeated')
        records.append(native)
    if geometries!=[10]:raise ValueError('Geometry must be post-trim and never rebuilt on checkpoint backward')
    for a,b in zip(*records):
        for x,y in zip(a,b):compare(x,y)
    return dict(kind='native_path_and_geometry',passed=True,attention_calls=8,trimmer_calls=1,geometry_calls=1)


def correctness(device,rank,world):
    records=[triangle_oracle(device),boundaries(device),particle_path(device)]
    delayed=delayed_activation(device,rank,world)
    for kind in ('zero_identity','active_chunk','active_amp_chunk'):
        seed_everything(101);fast=M.build_model(trim=False).to(device)
        seed_everything(101)
        oracle=(M.reference_model('state_layerwise',trim=False) if kind=='zero_identity'
                else M.build_model(trim=False,chunk=128,checkpointed=False,backend='dense_oracle')).to(device)
        active(fast,triangle=kind!='zero_identity')
        oracle.load_state_dict({k:v for k,v in fast.state_dict().items()
                               if kind!='zero_identity' or not M.triangle_key(k)})
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
        parameters=M.output_parameters(fast) if kind=='zero_identity' else [p for n,p in fast.named_parameters() if M.triangle_key(n)]
        if any(p.grad is None or not p.grad.isfinite().all() or not p.grad.count_nonzero() for p in parameters):
            raise ValueError('Triangle parameter missing learning signal')
        records.append(dict(kind=kind,passed=True,particles=67))
        del target,oracle,fast,parameters,grads,outputs,out,g
        gc.collect();torch.cuda.empty_cache()
    for kind in ('active_SyncBN_DDP','no_pairs_DDP','tie_only_rank_DDP'):
        base,wrapped=fresh(device,world,trim=False);active(base);B.disable_dropout(base)
        for p in base.parameters():
            if world>1:dist.broadcast(p.data,src=0)
        oracle=M.build_model(trim=False).to(device)
        oracle.load_state_dict(base.state_dict());B.disable_dropout(oracle)
        batch=B.fixed_batch(8,17)
        if kind=='no_pairs_DDP':batch[0]['pf_mask'][:4,:,2:]=False
        if kind=='tie_only_rank_DDP':batch[0]['pf_vectors'][:4]=batch[0]['pf_vectors'][:4,:,:1]
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
    # the triangle, no unused-parameter workaround, second step exercises DDP's reducer.
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


from .diagnostics import diagnostic_sample,gradient_norms,NeighborStats,stage_profile,tied_batch

from part_ev_pair_readout.benchmark import restart_probe as legacy_restart_probe,inference


def restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world,
                  *,deadline=float('inf')):
    # The GH200 path and other models retain their existing admission contract.
    if world == 1:
        return legacy_restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world)
    from .restart import probe
    return probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world,deadline=deadline)

def measure_arm(output,spec,rank,world,device,deadline,*,baseline=False):
    import resource
    base,model=fresh(device,world,baseline=baseline)
    optimizer=build_optimizer(base,spec['recipe']);scaler=torch.amp.GradScaler('cuda')
    if not baseline:root_call(lambda:P.write_json(Path(output)/'probes/optimizer.json',optimizer_metadata(base,optimizer)),rank,world)
    loader,_=build_loader('train',spec['inventory'],256,rank=rank,world_size=world,workers=1,seed=101)
    val_loader=None
    stats=None if baseline else NeighborStats(base)
    try:
        iterator=iter(loader);before=time.monotonic();first=next(iterator);cold=time.monotonic()-before
        seed_everything(101+rank);torch.cuda.reset_peak_memory_stats(device)
        label='baseline_state_ev' if baseline else 'pair_triangle'
        if stats:stats.phase='train'
        stream=B.measure(base,model,optimizer,scaler,
            lambda i:B.tensors(first if i==0 else next(iterator),device),spec['recipe'],device,world,
            20,4200,deadline,label=label+'_stream')
        # All real-data training is finished. Release prefetched host buffers
        # and workers before synthetic stress tests and diagnostic profiling.
        shutdown_loader(loader)
        del iterator
        if stats:stats.phase=None
        for m in base.modules():
            if type(m).__name__=='SequenceTrimmer':m.enabled=False
        values,truth=B.tensors(B.split_batch(B.fixed_batch(),rank,world),device)
        fixed=B.measure(base,model,optimizer,scaler,lambda i:(values,truth),spec['recipe'],device,world,
                        10,40,deadline,label=label+'_fixed128')
        result=dict(stream=stream,fixed=fixed,parameters=sum(p.numel() for p in base.parameters()))
        if not baseline:
            stats.phase='adversarial'
            tied_values,tied_truth=B.tensors(B.split_batch(tied_batch(512,128),rank,world),device)
            result['adversarial_ties']=B.measure(base,model,optimizer,scaler,lambda i:(tied_values,tied_truth),
                spec['recipe'],device,world,2,3,deadline,label='triangle_all_ties128')
            stats.phase=None
            # Stage profiling is separate from the sustained performance measurement.
            real_values,real_truth=B.tensors(first,device)
            result['profiles']=B.gather(dict(rank=rank,
                real_train=stage_profile(base,model,real_values,real_truth,device,deadline,label='real_train'),
                adversarial_train=stage_profile(base,model,tied_values,tied_truth,device,deadline,label='adversarial_train')),world)
            del tied_values,tied_truth,real_values,real_truth
            for m in base.modules():
                if type(m).__name__=='SequenceTrimmer':m.enabled=True
            restart,seconds=restart_probe(output,base,model,optimizer,scaler,values,truth,spec,rank,world,
                                          deadline=deadline)
            result['triangle_gradient_norms']=B.gather(gradient_norms(base),world)
            diag_values,_=B.tensors(B.fixed_batch(8,17),device)
            diagnostics=B.gather(dict(diagnostic_sample(base,diag_values,device),rank=rank),world)
            if not B.agree(not B.STOP and time.monotonic()<deadline,device,world):raise TimeoutError('No validation time')
            if rank==0:val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
            # Iteration guard makes inherited validation obey this diagnostic's deadline.
            val_first=[]
            def bounded_val():
                for batch in val_loader:
                    if B.STOP or time.monotonic()>=deadline:raise TimeoutError('Validation deadline')
                    if not val_first:val_first.append(batch)
                    yield batch
            stats.phase='validation'
            metric=validate(base,iter(bounded_val()) if rank==0 else None,device,rank,world,128000,audit=True)
            stats.phase=None
            if metric is None:raise TimeoutError('Validation interrupted')
            def validation_profile():
                v,y=B.tensors(val_first[0],device)
                return stage_profile(base,base,v,y,device,deadline,training=False,label='real_validation')
            profile=root_call(validation_profile,rank,world)
            result['profiles'][0]['real_validation']=profile
            val_first.clear()
            # Explicit worst-length FP32 validation batch, not only typical lengths.
            def full_validation_memory(ties=False):
                full,_=B.tensors(tied_batch(512,128) if ties else B.fixed_batch(512,128),device)
                base.eval()
                for m in base.modules():
                    if type(m).__name__=='SequenceTrimmer':m.enabled=False
                with torch.no_grad(),model_precision(base,device,training=False):
                    if not base(*full).isfinite().all():raise ValueError('Invalid full128 validation')
                return True
            root_call(full_validation_memory,rank,world)
            root_call(lambda:full_validation_memory(True),rank,world)
            result['neighbor_statistics']=B.gather(dict(rank=rank,**stats.report()),world)
            result.update(restart=restart,checkpoint_seconds=seconds,validation=metric,diagnostics=diagnostics,
                          full128_validation_passed=True,full128_tied_validation_passed=True)
        result['inference']=root_call(lambda:inference(base,values,device,deadline),rank,world)
        result['host_peak_rss_bytes']=B.gather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,world)
        result['finite_state']=B.agree(finite_state(torch,(base.state_dict(),optimizer.state_dict())),device,world)
        prop=torch.cuda.get_device_properties(device)
        result['gpus']=B.gather(dict(rank=rank,name=prop.name,total_memory=prop.total_memory,
            peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),
            cold_fetch_seconds=cold,first_input_sha256=B.batch_fingerprint(first)),world)
        return result
    finally:
        if stats:stats.close()
        shutdown_loader(loader)
        if val_loader is not None:shutdown_loader(val_loader)


def run(output,spec,rank,world,device):
    deadline=time.monotonic()+P.POLICY['max_seconds']
    gpu=torch.cuda.get_device_properties(device)
    identities=B.gather((gpu.name,gpu.total_memory),world)
    if world!=2 or len(set(identities))!=1 or 'A100' not in gpu.name:
        raise ValueError('Need two matching same-node A100s, no implicit hardware substitution')
    if rank==0:print(dict(stage='triangle_correctness',parameters=M.PARAMETERS),flush=True)
    checks=correctness(device,rank,world)
    gc.collect();torch.cuda.empty_cache()
    baseline=measure_arm(output,spec,rank,world,device,deadline,baseline=True)
    gc.collect();torch.cuda.empty_cache()
    candidate=measure_arm(output,spec,rank,world,device,deadline)
    healthy=all(r['stream']['healthy'] and r['fixed']['healthy'] and r['finite_state'] and
        all('A100' in g['name'] and g['peak_allocated']<.9*g['total_memory'] for g in r['gpus'])
        for r in (baseline,candidate))
    result=dict(candidate,backend=M.CONFIG['backend'],complete=True,passed=healthy,authorizes_training=False,campaign_sha256=spec['content_hash'],
        baseline_parameters=baseline['parameters'],baseline_stream=baseline['stream'],baseline_fixed=baseline['fixed'],
        baseline_inference=baseline['inference'],baseline_gpus=baseline['gpus'],correctness=checks,
        baseline_finite_state=baseline['finite_state'],
        measurement_order=['baseline_state_ev','pair_triangle'],
        runtime=dict(world_size=world,local_batch=256,global_batch=512,workers_per_rank=1,
            batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,
            nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}))
    return result
