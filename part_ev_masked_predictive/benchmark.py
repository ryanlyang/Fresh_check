"""Paired phase-aware target-device admission; disposable weights only."""
import gc
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import runtime as R,protocol as P,verification as V,restart as S,diagnostics as D
from .step import attempt
from .views import Generators,create,digest,slot
from .teacher import Teacher
from .config import COUNTS,TRAINABLE,TEACHER,POLICY,PARITY
from .engine import validate
from part_ev_fullscale.data import build_loader,shutdown_loader
from part_ev_fullscale.checkpoint import atomic_torch_save,trusted_load,capture_rank_state,restore_rank_state
from part_ev_screen.data import batch_fingerprint
from part_ev_screen.precision import model_precision,audit_precision
from scripts.run_part_ev_screen import classification_metrics,atomic_npz


def replay(spec,launch,rank,world,device):
    resources=launch['resources'];owner=torch.nn.Identity();rng=capture_rank_state(owner);opened=[]
    auxiliary=Generators(rank);before=auxiliary.state_dict()
    teacher=Teacher(R.build_model('masked',trim=False).to(device))
    teacher_state=S.cpu_copy(teacher.state_dict())
    def next_target(view):
        indices=view.eligible.nonzero().flatten()
        if not len(indices):raise ValueError('No eligible ROOT jet for target continuation probe')
        index=int(indices[0]);item=slot(view,index,index+1)
        return teacher.targets(item,amp=device.type=='cuda')
    def make(split):
        loader,_=build_loader(split,spec['inventory'],512 if split=='val' else resources['local_batch'],
            rank=0 if split=='val' else rank,world_size=1 if split=='val' else world,
            workers=2 if split=='val' else resources['workers'],seed=101)
        opened.append(loader);return iter(loader)
    result={}
    try:
        for split in ('train','val'):
            def check():
                a=make(split);batches=[next(a) for _ in range(3)];expected=batch_fingerprint(batches[-1])
                x,_=R.tensors(batches[-1],device)
                g=Generators(rank);v=create((x[1],x[2],x[3]),g);mask=v.fingerprint();target=next_target(v)
                shutdown_loader(opened.pop());a=make(split)
                for _ in range(2):next(a)
                continued=next(a);actual=batch_fingerprint(continued);y,_=R.tensors(continued,device)
                continued_view=create((y[1],y[2],y[3]),Generators(rank));repeated=continued_view.fingerprint()
                actual_target=next_target(continued_view);S.compare(actual_target,target)
                shutdown_loader(opened.pop())
                if actual!=expected or repeated!=mask:raise ValueError('ROOT replay/mask continuation changed')
                return dict(passed=True,consumed_batches=2,next_sha256=actual,next_mask_sha256=mask,
                    next_target_verified=True,next_target_events=1,next_target_sha256=digest(actual_target),
                    target_scope='fresh_frozen_teacher_next_batch')
            result[split]=R.root_call(check,rank,world) if split=='val' else R.all_call(check,world)
        S.compare(auxiliary.state_dict(),before,exact=True);S.compare(teacher.state_dict(),teacher_state,exact=True)
        restore_rank_state(owner,rng)
        return dict(passed=True,rank=rank,real_stream=True,streams=result,no_aux_draws_during_replay=True,rng_restored=True)
    finally:
        for loader in opened:shutdown_loader(loader)
        restore_rank_state(owner,rng)


def measure(s,supplier,*,warmup,steps,phase,window=1000):
    timings=[];skips=0;first=None;eligible=targets=positive=0;clean_total=aux_total=0.
    for i in range(warmup+steps):
        if R.stopping(s.device,s.world):raise TimeoutError('Incomplete phase benchmark')
        x,y=supplier(i)
        label=(100000 if phase=='active' else 900000)+i
        row=attempt(s,x,y,schedule_label=label,diagnostic=True,audit=i==0)
        if i==0:first=row
        if i+1==warmup:R.synchronize(s.device,s.world);start=window_start=time.monotonic()
        if i>=warmup:
            skips+=row['skipped'];eligible+=row['eligible'];targets+=row['targets'];positive+=int(row['auxiliary']>0)
            clean_total+=row['ce'];aux_total+=row['auxiliary'];done=i+1-warmup
            if done%window==0 or done==steps:
                R.synchronize(s.device,s.world);now=time.monotonic()
                timings.append(dict(steps=min(window,steps-len(timings)*window),seconds=now-window_start));window_start=now
    R.synchronize(s.device,s.world);seconds=time.monotonic()-start
    record=dict(rank=s.rank,windows=timings,seconds=seconds,skips=skips,parameters_sha256=R.parameter_digest(s.base),
        teacher_sha256=None if s.teacher is None else R.parameter_digest(s.teacher.model),scale=s.scaler.get_scale(),
        lookahead=s.opt.step_counter,precision=first['precision'],auxiliary_precision=first['auxiliary_precision'],
        teacher_precision=first['teacher_precision'],visible_precision=first['visible_precision'],
        rng_isolated=first['rng_isolated'],eligible=eligible,targets=targets,positive_aux_attempts=positive,
        mean_ce=clean_total/steps,mean_auxiliary=aux_total/steps,ema_successes=s.teacher.successes if s.teacher else 0)
    ranks=R.gather(record,s.world);seconds=max(r['seconds'] for r in ranks)
    return dict(phase=phase,diagnostic_schedule=POLICY['diagnostic_schedule'],label_start=100000 if phase=='active' else 900000,
        measured_steps=steps,warmup=warmup,ranks=ranks,seconds=seconds,global_batch=512,skips=skips,
        jets_per_second=512*steps/seconds,auxiliary_selected=256 if s.arm=='masked' and phase=='active' else 0,
        microbatch=32,observed_local_batch=len(x[0]),session_attempts=s.attempts)


def inference(student,device):
    results={};student.eval()
    with torch.no_grad(),model_precision(student,device,training=False):
        for name,batch,warm,steps in [('throughput',256,10,20),('latency',1,10,100),('full_validation',512,1,3)]:
            x,_=R.tensors(R.fixed_batch(batch,128),device)
            for _ in range(warm):student(*x)
            R.synchronize(device);start=time.monotonic()
            with audit_precision(student,expected_dtype=torch.float32) as precision:
                for _ in range(steps):
                    result=student(*x)
                    if not torch.isfinite(result).all():raise ValueError('Nonfinite exported inference')
            R.synchronize(device);results[name]=dict(batch=batch,particles=128,steps=steps,seconds=time.monotonic()-start,precision=precision)
    return results


def io(s,output):
    directory=Path(output)/'probes'/s.arm;R.synchronize(s.device,s.world);start=time.monotonic()
    saved=R.gather(S.capture(s),s.world);R.synchronize(s.device,s.world)
    transfer=max(R.gather(time.monotonic()-start,s.world))
    def measure():
        start=time.monotonic();atomic_torch_save(dict(phase='disposable_io',sessions=saved),directory/'io_restart.pt');write=time.monotonic()-start
        start=time.monotonic();restored=trusted_load(directory/'io_restart.pt');load=time.monotonic()-start
        S.compare(restored['sessions'],saved,exact=True)
        start=time.monotonic();atomic_torch_save(dict(model=s.base.student.state_dict()),directory/'io_student.pt');weights=time.monotonic()-start
        scores=np.random.default_rng(811).standard_normal((100000,10),dtype=np.float32);labels=np.arange(100000,dtype=np.int64)%10
        start=time.monotonic();atomic_npz(directory/'io_predictions.npz',logits=scores,labels=labels);predwrite=time.monotonic()-start
        start=time.monotonic()
        with np.load(directory/'io_predictions.npz',allow_pickle=False) as row:a,b=row['logits'],row['labels']
        predload=time.monotonic()-start
        if not np.array_equal(a,scores) or not np.array_equal(b,labels):raise ValueError('Prediction I/O changed data')
        return dict(restart_seconds=transfer+write,transfer_seconds=transfer,full_state_includes_transfer=True,
            load_seconds=load,weights_seconds=weights,prediction_events=100000,
            prediction_write_seconds=predwrite,prediction_load_seconds=predload,student_only=True)
    return R.root_call(measure,s.rank,s.world)


def run_arm(output,spec,launch,arm,device,rank,world):
    gates={}
    gates['native']=R.all_call(lambda:V.native_identity(device,rank,world),world)
    gates['failure']=R.all_call(lambda:V.rejection(device,rank,world,arm),world)
    if arm=='masked':
        for name,action in [('predictor',lambda:V.predictor_oracle(device)),('views',lambda:V.view_oracle(device)),('leakage',lambda:V.leakage(device)),
            ('sequential_fp32',lambda:V.sequential(device)),('sequential_amp',lambda:V.sequential(device,amp=True)),
            ('checkpoint_fp32',lambda:V.checkpoint_parity(device)),('checkpoint_amp',lambda:V.checkpoint_parity(device,amp=True)),
            ('auxiliary_gradient',lambda:V.auxiliary_gradient(device)),('taps',lambda:V.taps(device)),
            ('distributed',lambda:V.distributed(device,rank,world)),('skip',lambda:V.controlled_skip(device,rank,world))]:
            gates[name]=R.all_call(action,world)
    s,gates['learning']=V.learning(device,rank,world,arm)
    restart=S.probe(s,Path(output)/'probes'/arm)
    transitions=S.probe(s,Path(output)/'probes'/arm,transitions=True)
    del s;gc.collect();torch.cuda.empty_cache();s=R.fresh(arm,device,world,rank=rank)
    train_loader=val_loader=None
    try:
        r=launch['resources']
        train_loader,_=R.all_call(lambda:build_loader('train',spec['inventory'],r['local_batch'],rank=rank,
            world_size=world,workers=r['workers'],seed=101),world)
        iterator=iter(train_loader);start=time.monotonic();first=R.all_call(lambda:next(iterator),world);cold=time.monotonic()-start
        first_hash=batch_fingerprint(first);R.seed_everything(101+rank);torch.cuda.reset_peak_memory_stats(device)
        supplier=lambda i:R.tensors(first if i==0 else R.all_call(lambda:next(iterator),world),device)
        stream=measure(s,supplier,warmup=20,steps=4200,phase='active')
        clean=measure(s,lambda _:R.tensors(R.all_call(lambda:next(iterator),world),device),warmup=20,steps=200,
            phase='clean',window=200) if arm=='masked' else None
        shutdown_loader(train_loader);train_loader=None;s.base.student.mod.trimmer.enabled=False
        x,y=R.tensors(R.split_batch(R.fixed_batch(512,128),rank,world),device)
        fixed=measure(s,lambda _:(x,y),warmup=10,steps=40,phase='active')
        def setup_val():
            nonlocal val_loader
            val_loader,_=build_loader('val',spec['inventory'],512,workers=2,seed=101)
        R.root_call(setup_val,rank,world)
        validation=validate(s.base.student,iter(val_loader) if rank==0 else None,device,rank,world,128000,audit=True)
        if validation is None:raise TimeoutError('Incomplete validation')
        R.root_call(lambda:shutdown_loader(val_loader),rank,world);val_loader=None
        def exported():
            student=R.fresh_student(arm,device,trim=False);student.load_state_dict(s.base.student.state_dict(),strict=True)
            return inference(student,device)
        inf=R.root_call(exported,rank,world);checkpoint=io(s,output)
        def metric_cost():
            rng=np.random.default_rng(711);scores=rng.standard_normal((5000000,10),dtype=np.float32)
            y=np.arange(5000000,dtype=np.int64)%10;parts=np.array_split(scores,50)
            start=time.monotonic();classification_metrics(np.concatenate(parts),y,rejection=True);return time.monotonic()-start
        metric=R.root_call(metric_cost,rank,world)
        finite=R.agree(R.finite_state(s.state_dict()),device,world)
        sample=R.root_call(lambda:D.sample(s),rank,world)
        peaks=R.gather(dict(rank=rank,peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),host=D.memory()),world)
        correctness=R.gather(dict(rank=rank,checks=gates,restart=restart,transitions=transitions),world)
        del s;gc.collect();torch.cuda.empty_cache()
        profiles=R.all_call(lambda:D.profile(arm,device),world)
        return dict(arm=arm,parameters=COUNTS[arm],trainable_parameters=TRAINABLE[arm],teacher_parameters=TEACHER if arm=='masked' else 0,
            stream=stream,clean=clean,fixed=fixed,first_batch_sha256=R.gather(first_hash,world),cold_fetch_seconds=max(R.gather(cold,world)),
            validation=validation,inference=inf,checkpoint=checkpoint,metric_5m_seconds=metric,finite_state=finite,
            gpus=peaks,correctness=correctness,diagnostics=sample,profiles=R.gather(dict(rank=rank,profiles=profiles),world))
    finally:
        if train_loader is not None:shutdown_loader(train_loader)
        if val_loader is not None:shutdown_loader(val_loader)


def run(output,spec,launch,device,rank,world,gpu):
    output=Path(output)
    if (output/'benchmark.json').exists():raise FileExistsError('Use a fresh campaign')
    R.DEADLINE=min(R.DEADLINE,time.monotonic()+POLICY['max_seconds'])
    try:
        replays=R.gather(replay(spec,launch,rank,world,device),world)
        models=[run_arm(output,spec,launch,a,device,rank,world) for a in COUNTS]
        runtime=dict(platform=launch['platform'],resources=launch['resources'],gpu=gpu,global_batch=512,
            tf32=False,cudnn_benchmark=False,broadcast_buffers=False,
            nccl_environment={k:v for k,v in __import__('os').environ.items() if k.startswith(('NCCL_','TORCH_NCCL_'))})
        row=P.hashed(dict(schema=P.SCHEMA,complete=True,campaign_sha256=spec['content_hash'],policy=POLICY,parity=PARITY,
            runtime=runtime,models=models,stream_replay=replays,authorizes_training=False))
        from .evidence import validate as admission
        admission(spec,row,launch);R.root_call(lambda:P.write(output/'benchmark.json',row),rank,world);return row
    except BaseException as exc:
        errors=R.gather(f'{type(exc).__name__}: {exc}',world)
        R.root_call(lambda:P.write(output/'benchmark_failure.json',P.hashed(dict(complete=False,errors=errors,authorizes_training=False))),rank,world)
        raise
