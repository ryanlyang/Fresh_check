"""Actual composed-model checks, including nonzero state and full-recipe restart."""
import copy
import io
import torch
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_screen.precision import model_precision
from part_selected_screen.memory import compare, PARITY
from part_selected_fast.parity import execution_parity, compare_states
from scripts.run_part_ev_screen import tensors
from .steps import training_step
from .model import state_branches
from .protocol import recipes


def small_batch(batch, count=8):
    return ({k:v[:count] for k,v in batch[0].items()}, {'_label_':batch[1]['_label_'][:count]}, {})


def full_recipe_restart(model, batch, device):
    """Matched uninterrupted/reloaded pilots; disposable models, full LR boundary."""
    recipe = recipes()['ranger_full_v1']
    state, rng = copy.deepcopy(model.state_dict()), capture_rank_state(model)
    modes = [(m,m.training) for m in model.modules()]
    small = small_batch(batch)
    observations = []
    steps = [0,1,2,3,4,5,699999,700000,979999,980000]
    try:
        for resume in (False, True):
            model.load_state_dict(state); restore_rank_state(model, rng)
            optimizer = build_optimizer(model, recipe)
            scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
            updates = []
            for position, step in enumerate(steps):
                row = training_step(model, optimizer, scaler, small, device, recipe, step)
                updates.append((row['lr'], row['amp_skipped']))
                if resume and position == 2:
                    stream = io.BytesIO()
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scaler=scaler.state_dict(), rng=capture_rank_state(model)), stream)
                    stream.seek(0)
                    saved = torch.load(stream, map_location=device, weights_only=False)
                    model.load_state_dict(saved['model'])
                    optimizer = build_optimizer(model, recipe); optimizer.load_state_dict(saved['optimizer'])
                    scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
                    scaler.load_state_dict(saved['scaler']); restore_rank_state(model, saved['rng'])
            observations.append(dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                                     scaler=scaler.state_dict(), rng=capture_rank_state(model), updates=updates))
        def equal(a,b):
            if isinstance(a,torch.Tensor):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                if a.keys()!=b.keys(): raise ValueError('Restart keys differ')
                for key in a: equal(a[key],b[key])
            elif isinstance(a,(tuple,list)):
                if len(a)!=len(b): raise ValueError('Restart length differs')
                for x,y in zip(a,b): equal(x,y)
            else:
                import numpy as np
                if isinstance(a,np.ndarray):
                    if not np.array_equal(a,b): raise ValueError('Restart NumPy RNG differs')
                elif a!=b: raise ValueError('Restart scalar differs')
        equal(*observations)
        return dict(passed=True, steps=steps, serialized_restart_after=3, optimizer_and_rng_exact=True)
    finally:
        model.load_state_dict(state); model.zero_grad(set_to_none=True); restore_rank_state(model,rng)
        for module, training in modes: module.training=training


def state_integration(candidate, ordinary, batch, device):
    """Zero-state identity and nonzero gradients in every selected state/family/head."""
    state, rng = copy.deepcopy(candidate.state_dict()), capture_rank_state(candidate)
    modes = [(m,m.training) for m in candidate.modules()]
    values, truth = tensors(small_batch(batch),device)
    try:
        ordinary.load_state_dict({k:v for k,v in state.items() if '.state.' not in k},strict=True)
        candidate.eval(); ordinary.eval()
        with torch.no_grad(), model_precision(candidate,device,training=False), model_precision(ordinary,device,training=False):
            restore_rank_state(candidate,rng); actual=candidate(*values)
            restore_rank_state(ordinary,rng); expected=ordinary(*values)
        zero = compare(actual,expected,PARITY['fp32'])
        with torch.no_grad():
            for branch in state_branches(candidate): branch.output.fill_(.01)
        candidate.train(); restore_rank_state(candidate,rng)
        with model_precision(candidate,device,training=False):
            logits=candidate(*values)
            torch.nn.functional.cross_entropy(logits,truth).backward()
        groups={}
        for name,p in candidate.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise ValueError('Missing/nonfinite integration gradient: '+name)
            if '.state.' in name:
                if not p.grad.count_nonzero(): raise ValueError('Inactive state parameter: '+name)
                groups[name]=True
        if len(groups)!=32: raise ValueError('Expected four trained state tensors in each of eight blocks')
        for prefix in ('pt','track_node','track_pair','region'):
            module=getattr(candidate.mod.pair_embed.relations,prefix)
            if not any(p.grad is not None and p.grad.count_nonzero() for p in module.parameters()):
                raise ValueError('Selected family disconnected: '+prefix)
        for head in candidate.mod.pair_embed.heads:
            if not any(p.grad is not None and p.grad.count_nonzero() for p in head.parameters()):
                raise ValueError('Layerwise head disconnected')
        if candidate._state_pair_cache.stem is not None or candidate._state_pair_cache.normalized is not None:
            raise ValueError('State cache retained after forward')
        return dict(passed=True,zero_state=zero,state_gradient_tensors=len(groups),all_families_and_heads=True)
    finally:
        candidate.load_state_dict(state); candidate.zero_grad(set_to_none=True); restore_rank_state(candidate,rng)
        for module,training in modes: module.training=training


def selected_parity(candidate, reference, batch, device, ordinary=None):
    result=dict(passed=False)
    if ordinary is not None:
        result['integration']=state_integration(candidate,ordinary,batch,device)
    result['execution']=execution_parity(candidate,reference,batch,device)
    if ordinary is not None:
        saved=copy.deepcopy(candidate.state_dict()); rng=capture_rank_state(candidate)
        try:
            with torch.no_grad():
                for branch in state_branches(candidate): branch.output.fill_(.01)
            result['nonzero_execution']=execution_parity(candidate,reference,batch,device)
        finally:
            candidate.load_state_dict(saved); restore_rank_state(candidate,rng)
    result['full_recipe_restart']=full_recipe_restart(candidate,batch,device)
    result['passed']=True
    return result
