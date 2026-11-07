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


def _cpu_state(value):
    """Independent CPU observations also establish completion of CUDA writes."""
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_state(v) for v in value)
    return copy.deepcopy(value)


def _exact(a, b, path='state'):
    import numpy as np
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError('Nonfinite restart state: '+path)
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda text: path+': '+text)
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys(): raise ValueError('Restart keys differ: '+path)
        for key in a: _exact(a[key], b[key], path+'.'+str(key))
    elif type(a) is type(b) and isinstance(a, (tuple, list)):
        if len(a) != len(b): raise ValueError('Restart length differs: '+path)
        for index, (x, y) in enumerate(zip(a, b)): _exact(x, y, f'{path}[{index}]')
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError('Restart NumPy state differs: '+path)
    elif type(a) is not type(b) or a != b:
        raise ValueError('Restart value differs: '+path)


def full_recipe_restart(model, batch, device):
    """Synchronized correctness pilot only; no extra sync in measured training.

Warm native kernels once on disposable updates, then restore the initial state
and RNG before EVERY trial. Check exact serialization and both repeat/resume
trajectories, without treating instrumented parity as proof of unsynchronized
bitwise reproducibility. No changed precision, tolerance, or deterministic flags.
"""
    recipe = recipes()['ranger_full_v1']
    state, rng = _cpu_state(model.state_dict()), _cpu_state(capture_rank_state(model))
    modes = [(m,m.training) for m in model.modules()]
    small = small_batch(batch)
    steps = [0,1,2,3,4,5,699999,700000,979999,980000]

    def synchronize():
        if device.type == 'cuda': torch.cuda.synchronize(device)

    def reset():
        synchronize()
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        for module, training in modes: module.training = training
        optimizer = build_optimizer(model, recipe)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
        restore_rank_state(model, rng)
        synchronize()
        return optimizer, scaler

    def snapshot(optimizer, scaler):
        synchronize()
        return _cpu_state(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               scaler=scaler.state_dict(), rng=capture_rank_state(model)))

    observations = []
    sequence = []
    try:
        optimizer, scaler = reset()
        for step in steps:
            training_step(model, optimizer, scaler, small, device, recipe, step)
        del optimizer, scaler
        for trial in ('uninterrupted', 'repeat', 'resumed'):
            optimizer, scaler = reset()
            initial = snapshot(optimizer, scaler)
            if trial == 'uninterrupted': initial_reference = initial
            else: _exact(initial_reference, initial, trial+'.initial')
            updates = []
            for position, step in enumerate(steps):
                row = training_step(model, optimizer, scaler, small, device, recipe, step)
                updates.append((row['lr'], row['amp_skipped'], scaler.get_scale()))
                if trial == 'resumed' and position == 2:
                    before = snapshot(optimizer, scaler)
                    stream = io.BytesIO()
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scaler=scaler.state_dict(), rng=capture_rank_state(model)), stream)
                    stream.seek(0)
                    saved = torch.load(stream, map_location=device, weights_only=False)
                    _exact(before, _cpu_state(saved), 'deserialized')
                    model.load_state_dict(saved['model'])
                    optimizer = build_optimizer(model, recipe); optimizer.load_state_dict(saved['optimizer'])
                    scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
                    scaler.load_state_dict(saved['scaler']); restore_rank_state(model, saved['rng'])
                    _exact(before, snapshot(optimizer, scaler), 'restored')
                    del before, saved, stream
                observed = snapshot(optimizer, scaler)
                if trial == 'uninterrupted': observations.append(observed)
                else: _exact(observations[position], observed, f'{trial}.update_{position+1}')
            if trial == 'uninterrupted': sequence = updates
            else: _exact(sequence, updates, trial+'.lr_skip_scale')
            del optimizer, scaler
        return dict(passed=True, contract='synchronized_exact_restart_v2', steps=steps,
                    warmup_steps=len(steps), trials=['uninterrupted','repeat','resumed'],
                    serialized_restart_after=3, optimizer_and_rng_exact=True,
                    exact_roundtrip=True, exact_each_update=True, rtol=0, atol=0,
                    scope='synchronized_disposable_pilot_not_unsynchronized_reproducibility')
    finally:
        synchronize()
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
