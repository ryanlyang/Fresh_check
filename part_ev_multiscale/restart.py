"""Exact restoration and bounded seven-update trajectory probes, not training."""
import copy
from pathlib import Path
import numpy as np
import torch
from . import runtime as R
from .config import PARITY
from .geometry import observe
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


def snapshot(base,opt,scaler):
    return cpu_copy(dict(model=base.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=capture_rank_state(base)))


def load(saved,base,opt,scaler):
    base.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler']);restore_rank_state(base,saved['rng'])
    compare(snapshot(base,opt,scaler),saved,exact=True)


def probe(base,model,opt,scaler,device,rank,world,directory):
    directory=Path(directory)
    base.mod.trimmer.enabled=True
    base.mod.trimmer._counter=5
    initial=snapshot(base,opt,scaler)
    all_initial=R.gather(initial,world)
    path=directory/'initial.pt'
    R.root_call(lambda:atomic_torch_save(all_initial,path),rank,world)
    reference=[];evidence=[]
    for trial in ('reference','repeat','disk','resumed'):
        saved=R.all_call(lambda:trusted_load(path)[rank],world) if trial in ('disk','resumed') else initial
        load(saved,base,opt,scaler)
        rows=[];maximum=0.;worst_path=dict(maximum=0.,path=None)
        for i in range(7):
            if trial=='resumed' and i==3:
                middle=R.gather(snapshot(base,opt,scaler),world)
                midpath=directory/'middle.pt'
                R.root_call(lambda:atomic_torch_save(middle,midpath),rank,world)
                saved=R.all_call(lambda:trusted_load(midpath)[rank],world)
                load(saved,base,opt,scaler)
            batch=R.fixed_batch(8,19+i%3)
            local_batch=R.split_batch(batch,rank,world);input_hash=batch_fingerprint(local_batch)
            values,y=R.tensors(local_batch,device)
            with observe() as geometry:
                row=R.step(base,model,opt,scaler,values,y,100+i,world)
            if row['skipped']: raise ValueError('Settled restart probe skipped an update')
            if i==3:
                # Diagnostic validation boundary: FP32, mode/RNG/trimmer restored.
                rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
                base.eval()
                with torch.no_grad():
                    from part_ev_screen.precision import model_precision
                    with model_precision(base,device,training=False):
                        pred=base(*values)
                    if not torch.isfinite(pred).all(): raise ValueError('Probe validation nonfinite')
                for m,flag in flags:m.training=flag
                restore_rank_state(base,rng)
            state=snapshot(base,opt,scaler)
            if trial=='reference': reference.append((row['loss'],state))
            else:
                loss,expected,_=reference[i]
                limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(loss))
                if abs(row['loss']-loss)>limit: raise ValueError('Restart loss bound exceeded')
                compare(state['rng'],expected['rng'],exact=True)
                compare(state['scaler'],expected['scaler'],exact=True)
                compare(geometry,reference[i][2],exact=True,path='geometry')
                maximum=max(maximum,compare(state['model'],expected['model'],path='model',report=worst_path),
                            compare(state['optimizer'],expected['optimizer'],path='optimizer',report=worst_path))
            if trial=='reference':reference[-1]=(*reference[-1],geometry)
            rows.append(dict(update=i+1,loss=row['loss'],skipped=row['skipped'],scale=scaler.get_scale(),
                             lookahead=opt.step_counter,geometry=geometry,input_sha256=input_hash,
                             learning_rates=[g['lr'] for g in opt.param_groups],train_cursor=i+1,
                             probe_validation_batches=int(i>=3)))
        evidence.append(dict(trial=trial,updates=rows,max_state_error=maximum,worst_state_path=worst_path['path'],
                             exact_initial_restore=True))
    return dict(passed=True,rank=rank,updates=7,trials=evidence,full_state=True,
                validation_boundaries=1,policy=PARITY)

