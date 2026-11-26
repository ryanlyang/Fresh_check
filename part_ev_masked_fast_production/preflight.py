"""One-candidate production qualification, not another batch-size sweep."""
import gc
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from part_ev_masked_predictive import benchmark as B, runtime as R, protocol as O
from part_ev_fullscale.data import build_loader, shutdown_loader
from scripts.run_part_ev_screen import classification_metrics
from . import protocol as P, verification as V
from .config import POLICY, RESOURCES
from .execution import attempt, engine, private_module


def measured_attempt(s,*args,**kwargs):
    return attempt(s,*args,inspect_latents=(s.attempts+1)%100==0,**kwargs)


def run(output,launch,device,gpu):
    output=Path(output)
    if (output/'preflight.json').exists() or (output/'preflight_failure.json').exists():
        raise FileExistsError('Existing preflight evidence; use review, never overwrite')
    R.DEADLINE=min(R.DEADLINE,time.monotonic()+POLICY['max_seconds'])
    row=dict(schema=P.SCHEMA,launch_sha256=launch['content_hash'],gpu=gpu,policy=POLICY,
             complete=False,authorizes_training=False)
    train_loader=val_loader=None
    try:
        raw=P.checked(Path(launch['speed'])/'results.json')
        legacy=next(r for r in raw['variants'] if r['name']=='fast32')
        row['legacy_fast32']=V.legacy_diagnostic(device,legacy.get('error'))
        row['parity']=[]
        for repeat in range(POLICY['parity_repeats']):
            for amp in (False,True):
                row['parity'].append(dict(V.parity(device,amp=amp),repeat=repeat))
        row['restart']=V.restart_probe(V.fresh(device),output/'probes/restart')
        row['transitions']=V.restart_probe(V.fresh(device),output/'probes/transitions',transitions=True)
        spec=O.load(launch['reference'])
        row['real_stream_resume']=V.real_stream_resume(output,spec,RESOURCES,device)
        runtime=dict(gpu=gpu,resources=RESOURCES,execution_sha256=launch['execution_sha256'])
        diagnostic=P.hashed(dict(runtime=runtime,diagnostic=True,launch_sha256=launch['content_hash']))
        row['production_checkpoint']=V.production_checkpoint(output,diagnostic,device)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        s=R.fresh('masked',device,1,trim=True)
        train_loader,_=build_loader('train',spec['inventory'],512,workers=RESOURCES['workers'],seed=101)
        iterator=iter(train_loader);start=time.monotonic();first=next(iterator)
        row['cold_fetch_seconds']=time.monotonic()-start;R.seed_everything(101)
        measured=private_module(B,attempt=measured_attempt)
        def measure(supplier,*,phase,steps,warmup,window):
            record=measured.measure(s,supplier,warmup=warmup,steps=steps,phase=phase,window=window)
            record['microbatch']=256
            return record
        row['stream']=measure(lambda i:R.tensors(first if i==0 else next(iterator),device),
                              phase='active',steps=4200,warmup=20,window=1000)
        row['clean']=measure(lambda _:R.tensors(next(iterator),device),phase='clean',steps=200,warmup=20,window=200)
        shutdown_loader(train_loader);train_loader=None;s.base.student.mod.trimmer.enabled=False
        x,y=R.tensors(R.fixed_batch(512,128),device)
        row['fixed']=measure(lambda _:(x,y),phase='active',steps=40,warmup=10,window=40)
        val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
        row['validation']=engine().validate(s.base.student,iter(val_loader),device,0,1,128000,audit=True)
        if row['validation'] is None:raise TimeoutError('Incomplete production validation timing')
        shutdown_loader(val_loader);val_loader=None
        student=R.fresh_student('masked',device,trim=False)
        student.load_state_dict(s.base.student.state_dict(),strict=True)
        row['inference']=B.inference(student,device);del student
        row['checkpoint']=B.io(s,output)
        rng=np.random.default_rng(711);scores=rng.standard_normal((5000000,10),dtype=np.float32)
        truth=np.arange(5000000,dtype=np.int64)%10;start=time.monotonic()
        classification_metrics(scores,truth,rejection=True)
        row['metric_5m_seconds']=time.monotonic()-start;del scores,truth
        row['finite_state']=R.finite_state(s.state_dict())
        row['peak_allocated']=torch.cuda.max_memory_allocated(device)
        row['peak_reserved']=torch.cuda.max_memory_reserved(device)
        row['complete']=True
        P.validate_evidence(launch,row)
        P.write(output/'preflight.json',P.hashed(row))
        summary=P.review(output);P.write(output/'summary.json',P.hashed(summary))
        return summary
    except BaseException as exc:
        P.write(output/'preflight_failure.json',P.hashed(dict(complete=False,authorizes_training=False,
            error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),partial_evidence=row)))
        raise
    finally:
        if train_loader is not None:shutdown_loader(train_loader)
        if val_loader is not None:shutdown_loader(val_loader)
