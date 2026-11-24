"""Paired disposable GPU admission; never produces production checkpoints."""
import gc
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import protocol as P, runtime as R, verification as V, restart as S, diagnostics as D
from .config import POLICY, COUNTS, PARITY
from .engine import validate
from part_ev_fullscale.data import build_loader, shutdown_loader
from part_ev_fullscale.checkpoint import atomic_torch_save, capture_rank_state, restore_rank_state
from part_ev_screen.data import batch_fingerprint
from part_ev_screen.precision import model_precision, audit_precision
from scripts.run_part_ev_screen import classification_metrics


def stream_replay(spec,launch,rank,world,device):
    resources=launch['resources'];loaders=[]
    # Iterator construction/replay must not advance the model's global RNG.
    # Real loaders have their own deterministic generators; still restore the
    # complete rank RNG explicitly, as the production recovery path does.
    from torch import nn
    rng_owner=nn.Identity()
    rng=capture_rank_state(rng_owner)
    def create(split):
        root=split=='val'
        loader,_=build_loader(split,spec['inventory'],512 if root else resources['local_batch'],
            rank=0 if root else rank,world_size=1 if root else world,
            workers=2 if root else resources['workers'],seed=101)
        loaders.append(loader);return iter(loader)
    records={}
    try:
        for split in ('train','val'):
            def check():
                a=create(split)
                fingerprints=[batch_fingerprint(next(a)) for _ in range(3)]
                shutdown_loader(loaders.pop())
                b=create(split)
                for _ in range(2):next(b)
                actual=batch_fingerprint(next(b));shutdown_loader(loaders.pop())
                if actual!=fingerprints[2]:raise ValueError('Real ROOT cursor replay differs')
                return dict(consumed_batches=2,next_sha256=actual,passed=True)
            records[split]=R.root_call(check,rank,world) if split=='val' else R.all_call(check,world)
        restore_rank_state(rng_owner,rng)
        S.compare(capture_rank_state(rng_owner),rng,exact=True)
        return dict(passed=True,rank=rank,streams=records,real_stream=True,model_updates=0,rng_restored=True)
    finally:
        for loader in loaders:shutdown_loader(loader)
        restore_rank_state(rng_owner,rng)


def measure(base,model,opt,scaler,supplier,device,rank,world,*,warmup,steps,window=1000):
    rows=[];skips=0;precision=None
    for i in range(warmup+steps):
        if R.stopping(device,world):raise TimeoutError('Benchmark deadline/stop before complete evidence')
        values,y=supplier(i)
        result=R.step(base,model,opt,scaler,values,y,i,world,audit=i==0)
        if i==0:precision=result['precision']
        if i+1==warmup:
            R.synchronize(device,world);start=window_start=time.monotonic()
        if i>=warmup:
            skips+=result['skipped']
            completed=i+1-warmup
            if completed%window==0 or completed==steps:
                R.synchronize(device,world);now=time.monotonic()
                rows.append(dict(steps=min(window,steps-(len(rows)*window)),seconds=now-window_start));window_start=now
    R.synchronize(device,world);seconds=time.monotonic()-start
    record=dict(rank=rank,windows=rows,seconds=seconds,skips=skips,precision=precision,
        parameters_sha256=R.parameter_digest(base),scale=scaler.get_scale(),lookahead=opt.step_counter)
    ranks=R.gather(record,world)
    return dict(measured_steps=steps,ranks=ranks,seconds=max(r['seconds'] for r in ranks),
                skips=skips,global_batch=512,jets_per_second=512*steps/max(r['seconds'] for r in ranks))


def inference(base,device):
    results={};base.eval()
    with torch.no_grad(),model_precision(base,device,training=False):
        for name,batch,warm,steps in [('throughput',256,10,20),('latency',1,10,100),('full_validation',512,1,3)]:
            values,_=R.tensors(R.fixed_batch(batch,128),device)
            for _ in range(warm):base(*values)
            R.synchronize(device);start=time.monotonic()
            with audit_precision(base,expected_dtype=torch.float32) as precision:
                for _ in range(steps):
                    output=base(*values)
                    if not torch.isfinite(output).all():raise ValueError('Nonfinite inference')
            R.synchronize(device)
            results[name]=dict(batch=batch,particles=128,steps=steps,seconds=time.monotonic()-start,precision=precision)
    return results


def checkpoints(base,opt,scaler,output,arm,device,rank,world):
    directory=Path(output)/'probes'/arm
    state=S.snapshot(base,opt,scaler)
    ranks=R.gather(state,world)
    def measure_io():
        start=time.monotonic();atomic_torch_save(ranks,directory/'io_restart.pt');restart=time.monotonic()-start
        start=time.monotonic();saved=S.trusted_load(directory/'io_restart.pt');load=time.monotonic()-start
        S.compare(saved,ranks,exact=True)
        start=time.monotonic();atomic_torch_save(dict(model=base.state_dict()),directory/'io_weights.pt');weights=time.monotonic()-start
        return dict(restart_seconds=restart,load_seconds=load,weights_seconds=weights)
    return R.root_call(measure_io,rank,world)


def run_arm(output,spec,launch,arm,device,rank,world):
    checks={}
    checks['initial_identity']=R.all_call(lambda:V.initial_identity(device,rank,world),world)
    checks['active_fp32']=R.all_call(lambda:V.active_parity(arm,device,rank,world),world)
    checks['active_amp']=R.all_call(lambda:V.active_parity(arm,device,rank,world,amp=True),world)
    checks['distributed']=R.all_call(lambda:V.distributed(arm,device,rank,world),world)
    checks['rejection']=R.all_call(lambda:V.rejection(arm,device,rank,world),world)
    if arm=='workspace':checks['attention_oracle']=R.all_call(lambda:V.oracle(device),world)
    base,model,opt,scaler,learning=V.learning(arm,device,rank,world)
    checks['learning']=learning
    restart=S.probe(base,model,opt,scaler,device,rank,world,Path(output)/'probes'/arm)
    # Disposable probes do not contaminate measured initialization/data/RNG.
    del base,model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    base,model,opt,scaler=R.fresh(arm,device,world)
    resources=launch['resources'];loader=None;val_loader=None
    try:
        loader,_=R.all_call(lambda:build_loader('train',spec['inventory'],resources['local_batch'],
            rank=rank,world_size=world,workers=resources['workers'],seed=101),world)
        iterator=iter(loader);start=time.monotonic()
        first=R.all_call(lambda:next(iterator),world);cold=time.monotonic()-start
        first_hash=batch_fingerprint(first)
        R.seed_everything(101+rank);torch.cuda.reset_peak_memory_stats(device)
        def supplier(i):return R.tensors(first if i==0 else R.all_call(lambda:next(iterator),world),device)
        stream=measure(base,model,opt,scaler,supplier,device,rank,world,
            warmup=POLICY['warmup'],steps=POLICY['measured'])
        shutdown_loader(loader);loader=None
        for module in base.modules():
            if type(module).__name__=='SequenceTrimmer':module.enabled=False
        values,y=R.tensors(R.split_batch(R.fixed_batch(),rank,world),device)
        fixed=measure(base,model,opt,scaler,lambda _:(values,y),device,rank,world,warmup=10,steps=40)
        def setup_validation():
            nonlocal val_loader
            val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
        R.root_call(setup_validation,rank,world)
        validation=validate(base,iter(val_loader) if rank==0 else None,device,rank,world,128000,audit=True)
        if validation is None:raise TimeoutError('Incomplete validation evidence')
        R.root_call(lambda:shutdown_loader(val_loader),rank,world);val_loader=None
        inf=R.root_call(lambda:inference(base,device),rank,world)
        io=checkpoints(base,opt,scaler,output,arm,device,rank,world)
        def metric_cost():
            # Synthetic cost only, no accuracy result or persistent event cache.
            rng=np.random.default_rng(711)
            logits=rng.standard_normal((5_000_000,10),dtype=np.float32)
            labels=np.arange(5_000_000,dtype=np.int64)%10
            start=time.monotonic();classification_metrics(logits,labels,rejection=True)
            return time.monotonic()-start
        metric_seconds=R.root_call(metric_cost,rank,world)
        finite=R.agree(R.finite_state((base.state_dict(),opt.state_dict(),scaler.state_dict())),device,world)
        diagnostic=R.root_call(lambda:D.sample(base,device),rank,world)
        gpu=dict(rank=rank,peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),host=D.memory())
        gpus=R.gather(gpu,world)
        check_rows=R.gather(dict(rank=rank,checks=checks,restart=restart),world)
        # Full-sized timing is complete before bounded profiler work.
        del base,model,opt,scaler;gc.collect();torch.cuda.empty_cache()
        profiles=R.all_call(lambda:D.profile(arm,device),world)
        return dict(arm=arm,parameters=COUNTS[arm],stream=stream,fixed=fixed,
            first_batch_sha256=R.gather(first_hash,world),cold_fetch_seconds=max(R.gather(cold,world)),
            validation=validation,inference=inf,checkpoint=io,metric_5m_seconds=metric_seconds,finite_state=finite,gpus=gpus,
            correctness=check_rows,diagnostics=diagnostic,profiles=R.gather(dict(rank=rank,profiles=profiles),world))
    finally:
        if loader is not None:shutdown_loader(loader)
        if val_loader is not None:shutdown_loader(val_loader)


def run(output,spec,launch,device,rank,world,gpu):
    output=Path(output)
    if (output/'benchmark.json').exists():raise FileExistsError('Evidence already exists; use a fresh campaign')
    R.DEADLINE=min(R.DEADLINE,time.monotonic()+POLICY['max_seconds'])
    error=None
    try:
        replay=R.gather(stream_replay(spec,launch,rank,world,device),world)
        models=[run_arm(output,spec,launch,arm,device,rank,world) for arm in COUNTS]
        if models[0]['first_batch_sha256']!=models[1]['first_batch_sha256']:
            raise ValueError('Paired streams do not start from identical rank batches')
        runtime=dict(platform=launch['platform'],resources=launch['resources'],gpu=gpu,
            global_batch=512,tf32=False,cudnn_benchmark=False,
            nccl_environment={k:v for k,v in __import__('os').environ.items() if k.startswith(('NCCL_','TORCH_NCCL_'))})
        row=P.hashed(dict(schema=P.SCHEMA,complete=True,campaign_sha256=spec['content_hash'],
            policy=POLICY,parity=PARITY,runtime=runtime,models=models,stream_replay=replay,
            authorizes_training=False))
        # Validate raw evidence before atomically publishing success.
        from .evidence import validate as admission
        admission(spec,row,launch)
        R.root_call(lambda:P.write(output/'benchmark.json',row),rank,world)
        return row
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        R.root_call(lambda:P.write(output/'benchmark_failure.json',P.hashed(dict(complete=False,error=error))),rank,world)
        raise
