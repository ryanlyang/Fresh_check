"""Exact restoration and bounded seven-update trajectory probes, not training."""
import copy
from pathlib import Path
import numpy as np
import torch
from . import runtime as R
from .config import PARITY
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load, capture_rank_state, restore_rank_state


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare(actual,expected,*,exact=False,path='state',worst_path=None):
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
            if worst_path is not None and worst>worst_path.get('error',0.):worst_path.update(error=worst,path=path)
    elif isinstance(expected,np.ndarray):
        if not np.array_equal(actual,expected): raise ValueError('RNG array changed: '+path)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise ValueError('State keys changed: '+path)
        for k,v in expected.items(): worst=max(worst,compare(actual[k],v,exact=exact,path=path+'.'+str(k),worst_path=worst_path))
    elif isinstance(expected,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected): raise ValueError('State length/type changed: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):
            worst=max(worst,compare(a,b,exact=exact,path=path+f'.{i}',worst_path=worst_path))
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
    # Probe-only, declared nondegenerate fixture. Preserve the learned expert
    # outputs and populated optimizer; separate router biases protect discrete
    # route identities from accidental tiny second/third-margin ties. This is
    # never used in production, learning admission or real-data timing.
    from .model import banks
    with torch.no_grad():
        for bank in banks(base):bank.router[2].bias.copy_(torch.tensor([8.,4.,-4.,-8.],device=device))
    initial=snapshot(base,opt,scaler)
    all_initial=R.gather(initial,world)
    path=directory/'initial.pt'
    R.root_call(lambda:atomic_torch_save(all_initial,path),rank,world)
    reference=[];evidence=[]
    for trial in ('reference','repeat','disk','resumed'):
        saved=R.all_call(lambda:trusted_load(path)[rank],world) if trial in ('disk','resumed') else initial
        load(saved,base,opt,scaler)
        rows=[];maximum=0.;worst_path={}
        for i in range(7):
            if trial=='resumed' and i==3:
                middle=R.gather(snapshot(base,opt,scaler),world)
                midpath=directory/'middle.pt'
                R.root_call(lambda:atomic_torch_save(middle,midpath),rank,world)
                saved=R.all_call(lambda:trusted_load(midpath)[rank],world)
                load(saved,base,opt,scaler)
            batch=R.fixed_batch(8,7+i%3)
            values,y=R.tensors(R.split_batch(batch,rank,world),device)
            row=R.step(base,model,opt,scaler,values,y,100+i,world,route_details=True)
            from part_ev_screen.data import batch_fingerprint
            row['input_sha256']=batch_fingerprint(R.split_batch(batch,rank,world))
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
            if trial=='reference': reference.append((row,state))
            else:
                expected_row,expected=reference[i]
                for key in ('loss','ce','balance'):
                    a,b=row[key],expected_row[key]
                    if b is None:
                        if a is not None:raise ValueError('Control gained auxiliary loss')
                    else:
                        limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(b))
                        if abs(a-b)>limit:raise ValueError('Restart computed loss bound exceeded: '+key)
                for key in ('input_sha256','lr','skipped'):
                    if row[key]!=expected_row[key]:raise ValueError('Exact restart metadata changed: '+key)
                if row['routes']['sha256']!=expected_row['routes']['sha256']:raise ValueError('Restart discrete routes changed')
                if row['routing'] is not None:
                    for key in ('requested','executed','tokens','primary','secondary'):
                        if row['routing'][key]!=expected_row['routing'][key]:raise ValueError('Restart route counts changed')
                compare(state['rng'],expected['rng'],exact=True)
                compare(state['scaler'],expected['scaler'],exact=True)
                maximum=max(maximum,compare(state['model'],expected['model'],worst_path=worst_path),
                            compare(state['optimizer'],expected['optimizer'],worst_path=worst_path))
            rows.append(dict(update=i+1,loss=row['loss'],skipped=row['skipped'],scale=scaler.get_scale(),
                ce=row['ce'],balance=row['balance'],lr=row['lr'],input_sha256=row['input_sha256'],
                routes=row['routes'],counts=None if row['routing'] is None else
                    {k:row['routing'][k] for k in ('requested','executed','tokens','primary','secondary')},
                lookahead=opt.step_counter))
        evidence.append(dict(trial=trial,updates=rows,max_state_error=maximum,
            worst_tensor_path=worst_path.get('path'),exact_initial_restore=True))
    return dict(passed=True,rank=rank,updates=7,trials=evidence,full_state=True,
                validation_boundaries=1,policy=PARITY,
                fixture='trained_outputs_separated_router_biases_rolling_native_trim_dropout_v1')
