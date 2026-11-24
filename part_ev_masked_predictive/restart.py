"""Exact restoration and bounded seven-update trajectory probes, not training."""
import copy
from pathlib import Path
import numpy as np
import torch
from . import runtime as R
from .config import PARITY
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load, capture_rank_state, restore_rank_state
from part_ev_screen.data import batch_fingerprint


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare(actual,expected,*,exact=False,path='state',report=None):
    worst=0.
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape or actual.dtype!=expected.dtype:
            raise ValueError('Tensor identity changed: '+path)
        a,b=actual.detach().cpu(),expected.detach().cpu()
        if exact or not a.is_floating_point():
            if not torch.equal(a,b): raise ValueError('Exact tensor changed: '+path)
        else:
            error=(a-b).abs();limit=torch.minimum(torch.full_like(b,PARITY['restart_absolute']),
                PARITY['restart_atol']+PARITY['restart_rtol']*b.abs())
            if not torch.isfinite(a).all() or not (error<=limit).all():
                raise ValueError(f'Bounded state error at {path}: {error.max().item()}')
            worst=float(error.max()) if error.numel() else 0.
            if report is not None and worst>report['maximum']:report.update(maximum=worst,path=path)
    elif isinstance(expected,np.ndarray):
        if not np.array_equal(actual,expected): raise ValueError('RNG array changed: '+path)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise ValueError('State keys changed: '+path)
        for k,v in expected.items(): worst=max(worst,compare(actual[k],v,exact=exact,path=path+'.'+str(k),report=report))
    elif isinstance(expected,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected): raise ValueError('State length/type changed: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):
            worst=max(worst,compare(a,b,exact=exact,path=path+f'.{i}',report=report))
    elif actual!=expected: raise ValueError('Scalar metadata changed: '+path)
    return worst


def capture(session):return cpu_copy(session.state_dict())


def restore(session,saved):
    session.load_state_dict(saved);compare(capture(session),saved,exact=True)


def probe(session,directory,*,transitions=False):
    from .step import attempt
    s=session;device=s.device;rank=s.rank;world=s.world;directory=Path(directory)
    s.base.student.mod.trimmer.enabled=True;s.base.student.mod.trimmer._counter=5
    initial=capture(s);all_initial=R.gather(initial,world)
    path=directory/('transition_initial.pt' if transitions else 'initial.pt')
    R.root_call(lambda:atomic_torch_save(all_initial,path),rank,world)
    labels=[19999,20000,20001,39999,40000,899999,900000] if transitions else list(range(100000,100007))
    references=[];evidence=[]
    for trial in ('reference','repeat','disk','resumed'):
        saved=R.all_call(lambda:trusted_load(path)[rank],world) if trial in ('disk','resumed') else initial
        restore(s,saved);rows=[];worst=dict(maximum=0.,path=None)
        for i,u in enumerate(labels):
            if trial=='resumed' and i==3:
                middle=R.gather(capture(s),world);midpath=directory/('transition_middle.pt' if transitions else 'middle.pt')
                R.root_call(lambda:atomic_torch_save(middle,midpath),rank,world)
                restore(s,R.all_call(lambda:trusted_load(midpath)[rank],world))
            batch=R.split_batch(R.fixed_batch(8,17+i%3),rank,world);x,y=R.tensors(batch,device)
            row=attempt(s,x,y,schedule_label=u,diagnostic=True,audit=True)
            if row['skipped']:raise ValueError('Settled restart probe skipped an update')
            if i==3:
                rng=capture_rank_state(s.base);flags=[(m,m.training) for m in s.base.modules()]
                s.base.student.eval()
                with torch.no_grad():
                    from part_ev_screen.precision import model_precision
                    with model_precision(s.base.student,device,training=False):out=s.base.student(*x)
                    if not torch.isfinite(out).all():raise ValueError('Probe validation nonfinite')
                for m,flag in flags:m.training=flag
                restore_rank_state(s.base,rng)
            state=capture(s)
            record={k:row[k] for k in ('loss','ce','auxiliary','coefficient','momentum','eligible','targets',
                'skipped','scale','ema_successes','attempt','schedule_label','mask_sha256','bn_isolated')}
            record.update(input_sha256=batch_fingerprint(batch),generator_sha256=None if s.generators is None else
                __import__('part_ev_masked_predictive.views',fromlist=['digest']).digest(list(s.generators.state_dict()['generators'].values())),
                update=i+1,lookahead=s.opt.step_counter,validation_batches=int(i>=3))
            if trial=='reference':references.append((record,state))
            else:
                expected,old=references[i]
                for name in record:
                    if name in ('loss','ce','auxiliary'):
                        limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(expected[name]))
                        if abs(record[name]-expected[name])>limit:raise ValueError('Restart loss bound: '+name)
                    elif record[name]!=expected[name]:raise ValueError('Exact restart schedule changed: '+name)
                compare(state['rng'],old['rng'],exact=True)
                compare(state['generators'],old['generators'],exact=True)
                compare(state['scaler'],old['scaler'],exact=True)
                for key in ('model','optimizer','teacher'):compare(state[key],old[key],path=key,report=worst)
            rows.append(record)
        evidence.append(dict(trial=trial,updates=rows,exact_initial_restore=True,
            max_state_error=worst['maximum'],worst_state_path=worst['path']))
    return dict(passed=True,rank=rank,updates=7,trials=evidence,full_state=True,validation_boundaries=1,
                policy=PARITY,schedule='transitions' if transitions else 'active',labels=labels)
