"""Real full-step equivalence, not just zero-initialized forward equality."""
import torch
from part_ev_masked_predictive import runtime as R, restart as K, verification as V
from part_ev_masked_predictive.step import attempt as reference
from part_ev_masked_predictive.config import PARITY
from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load
from .step import attempt as optimized


def execute(s, x, y, variant, **kwargs):
    if variant['optimized']:
        return optimized(s, x, y, microbatch=variant['microbatch'], **kwargs)
    return reference(s, x, y, **kwargs)


def parity(variant, device, *, amp, batch=512, particles=97):
    s = R.fresh('masked', device, 1, trim=False)
    V.activate(s.base.student)
    s.scaler = torch.amp.GradScaler(device.type, init_scale=128, enabled=amp and device.type=='cuda')
    fixture = R.fixed_batch(batch, particles)
    lengths = 3 + torch.arange(batch)*17 % (particles-2)
    lengths[0] = particles
    fixture[0]['pf_mask'] &= torch.arange(particles)[None,None,:] < lengths[:,None,None]
    x, y = R.tensors(fixture, device)
    initial = K.capture(s); states = []; rows = []; gradients = []
    hook = s.opt.register_step_pre_hook(lambda opt, args, kwargs: gradients.append(
        {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in s.base.named_parameters()}))
    try:
        for mode in (dict(optimized=False), variant):
            K.restore(s, initial)
            rows.append(execute(s, x, y, mode, schedule_label=100000, diagnostic=True, amp=amp, audit=True))
            if rows[-1]['skipped']: raise ValueError('Parity update skipped; no parity claim')
            states.append(K.capture(s))
    finally:
        hook.remove()
    if len(gradients)!=2 or gradients[0].keys()!=gradients[1].keys():
        raise ValueError('Full gradient coverage changed')
    worst = 0.
    for name, old in gradients[0].items():
        new = gradients[1][name]
        if old is None:
            if new is not None: raise ValueError('Unused gradient changed: '+name)
        else:
            V.close(new, old, gradient=True, amp=amp)
            worst = max(worst, float((old-new).abs().max()))
    for name in ('loss', 'ce', 'auxiliary'):
        V.close(torch.tensor(rows[1][name]), torch.tensor(rows[0][name]), amp=amp)
    for name in ('mask_sha256', 'eligible', 'targets', 'modes', 'coefficient', 'momentum',
                 'skipped', 'scale', 'ema_successes', 'bn_isolated', 'rng_isolated'):
        if rows[0][name]!=rows[1][name]: raise ValueError('Exact transaction changed: '+name)
    for name in ('rng', 'generators', 'scaler'):
        K.compare(states[1][name], states[0][name], exact=True)
    state_error = max(K.compare(states[1][name], states[0][name]) for name in ('model','optimizer','teacher'))
    for name, _ in s.base.named_buffers():
        K.compare(states[1]['model'][name], states[0]['model'][name], exact=True)
    return dict(passed=True, amp=amp, batch=batch, particles=particles,
        gradients=len(gradients[0]), max_gradient_absolute=worst, max_state_absolute=state_error,
        exact_masks_rng_scaler=True, exact_bn_buffers=True, varied_lengths=True, full_optimizer_teacher=True)


def restart(variant, device, path):
    s = R.fresh('masked', device, 1, trim=False)
    s.scaler = torch.amp.GradScaler(device.type, init_scale=128, enabled=device.type=='cuda')
    initial = K.capture(s); records = []
    labels = [19999, 20000, 20001, 39999, 40000, 899999, 900000]
    for resumed in (False, True):
        K.restore(s, initial)
        for index, label in enumerate(labels):
            if resumed and index==3:
                middle = K.capture(s); atomic_torch_save(middle, path)
                s = R.fresh('masked', device, 1, trim=False)
                s.scaler = torch.amp.GradScaler(device.type, init_scale=128, enabled=device.type=='cuda')
                K.restore(s, trusted_load(path))  # Exact entire state, including masks/EMA.
            x, y = R.tensors(R.fixed_batch(8, 17+index%3), device)
            row = execute(s, x, y, variant, schedule_label=label, diagnostic=True, audit=True)
            if row['skipped']: raise ValueError('Restart probe skipped an update')
            state = K.capture(s)
            if not resumed: records.append((row,state))
            else:
                old, expected = records[index]
                for name in ('mask_sha256','eligible','targets','modes','skipped','scale','attempt','ema_successes'):
                    if row[name]!=old[name]: raise ValueError('Restart transaction changed: '+name)
                for name in ('loss','ce','auxiliary'):
                    limit = min(PARITY['loss_absolute'], PARITY['loss_atol']+PARITY['loss_rtol']*abs(old[name]))
                    if not abs(row[name]-old[name])<=limit: raise ValueError('Restart loss bound: '+name)
                for name in ('rng','generators','scaler'): K.compare(state[name], expected[name], exact=True)
                K.compare(state, expected)
    return dict(passed=True, updates=7, exact_roundtrip=True, full_state=True,
                schedule_boundaries=labels, lookahead_crossed=True)
