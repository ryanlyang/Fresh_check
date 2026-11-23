"""Bounded SPORC trajectory checks, with exact full-state serialization/restoration.

Only post-update model floating tensors and Ranger moments/slow weights receive
tolerances. RNG, scaler, optimizer options/counters, metadata and fixture progress
never do. This is a synthetic probe, not real ROOT-stream restart certification.
"""
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from .restart_policy import POLICY, METADATA, validate_evidence
from part_ev_fullscale.checkpoint import (atomic_torch_save, trusted_load,
                                         capture_rank_state, restore_rank_state)
from part_ev_pair_memory.benchmark import cpu_copy
from part_ev_gpu_scaling import benchmark as B
from part_ev_layerwise_ddp.engine import root_call
from part_ev_screen.precision import model_precision


def compare_state(expected, actual, *, exact=False):
    """Check every leaf; numerical budgets are elementwise AND absolutely capped."""
    result = dict(passed=True, finite=True, exact_discrete=True, exact=True,
                  leaves=0, floating_leaves=0, max_abs_error=0., max_budget_fraction=0., worst=[])

    def floating_path(path):
        return (bool(path) and path[0] == 'model'
                or (len(path) == 3 and path[:2] == ('optimizer', 'slow'))
                or (len(path) == 5 and path[:3] == ('optimizer', 'inner', 'state')
                    and path[-1] in ('exp_avg', 'exp_avg_sq')))

    def walk(a, b, path):
        label = '.'.join(map(str, path))
        if type(a) is not type(b) and not (isinstance(a, dict) and isinstance(b, dict)):
            raise ValueError('Restart type differs: '+label)
        if isinstance(a, dict):
            if a.keys() != b.keys(): raise ValueError('Restart keys differ: '+label)
            for key in a: walk(a[key], b[key], (*path, key))
            return
        if isinstance(a, (list, tuple)):
            if len(a) != len(b): raise ValueError('Restart length differs: '+label)
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, (*path, i))
            return
        result['leaves'] += 1
        if isinstance(a, (torch.Tensor, np.ndarray)):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Restart shape/dtype differs: '+label)
            a, b = torch.as_tensor(a).detach().cpu(), torch.as_tensor(b).detach().cpu()
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError('Nonfinite restart state: '+label)
            floating = a.is_floating_point() and floating_path(path)
            result['floating_leaves'] += int(floating)
            if torch.equal(a, b): return
            if exact or not floating:
                raise ValueError('Exact restart state differs: '+label)
            error = (a.double()-b.double()).abs()
            limit = (POLICY['state']['atol']+POLICY['state']['rtol']*a.double().abs()).clamp_max(
                POLICY['state']['absolute_cap'])
            maximum, fraction = float(error.max()), float((error/limit).max())
            if fraction > 1:
                raise ValueError(f'Restart floating state exceeds bounds: {label}, max_abs={maximum}, budget_fraction={fraction}')
            result['exact'] = False
            result['max_abs_error'] = max(result['max_abs_error'], maximum)
            result['max_budget_fraction'] = max(result['max_budget_fraction'], fraction)
            result['worst'].append(dict(path=label, max_abs_error=maximum, max_budget_fraction=fraction))
            result['worst'].sort(key=lambda row: row['max_budget_fraction'], reverse=True)
            del result['worst'][5:]
        else:
            if isinstance(a, (float, np.floating)) and not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError('Nonfinite restart scalar: '+label)
            if a != b: raise ValueError('Exact restart scalar differs: '+label)

    walk(expected, actual, ())
    return result


def trials(initial, *, restore, snapshot, step, boundary, disk_initial, roundtrip,
           checked=lambda action: action(), stop=lambda: None):
    """Callback-based engine also exercised with real Ranger on CPU in tests."""
    initial = cpu_copy(initial)
    decoded = checked(disk_initial)
    report = dict(passed=True, checkpoint_roundtrip=checked(
        lambda: compare_state(initial, decoded, exact=True)), trials={})
    references = []
    for name in POLICY['trials']:
        payload = checked(disk_initial) if name == 'disk_start' else initial
        progress = copy.deepcopy(payload['progress'])

        def load():
            restore(cpu_copy(payload))
            return compare_state(payload, snapshot(progress), exact=True)

        trial = dict(initial_restore=checked(load), updates=[], comparisons=[])
        for position in range(POLICY['updates']):
            stop()
            update = step(progress)
            progress['attempt'] += 1
            progress['fixture_cursor'] += 1
            if progress['attempt'] % 20000 == 0:
                boundary(progress)
                progress['checks'] += 1
            if name == 'resume_after_3' and position+1 == POLICY['resume_after']:
                before = checked(lambda: cpu_copy(snapshot(progress)))
                decoded = checked(lambda: roundtrip(cpu_copy(before)))
                trial['mid_deserialization'] = checked(lambda: compare_state(before, decoded, exact=True))

                def reload_mid():
                    restore(cpu_copy(decoded))
                    return compare_state(before, snapshot(decoded['progress']), exact=True)

                trial['mid_restore'] = checked(reload_mid)
                progress = copy.deepcopy(decoded['progress'])
            state = checked(lambda: cpu_copy(snapshot(progress)))
            trial['updates'].append(update)
            if name == 'reference':
                references.append(state)
            else:
                trial['comparisons'].append(checked(lambda: compare_state(references[position], state)))
        trial['progress'] = copy.deepcopy(progress)
        report['trials'][name] = trial
    return report


def probe(output, base, model, optimizer, scaler, values, truth, spec, rank, world,
          *, deadline=float('inf')):
    """Fresh two-rank fixture. Every local audit failure is shared before proceeding."""
    if world != 2:
        raise ValueError('Bounded Triangle restart probe is SPORC/two-rank only')
    if spec.get('restart_policy') != POLICY:
        raise ValueError('Fresh campaign with versioned restart policy required')
    output = Path(output)

    def checked(action):
        result, error = None, None
        try:
            result = action()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = B.gather(error, world)
        if any(errors): raise ValueError('Triangle restart audit failed on ranks: '+str(errors))
        return result

    def stop():
        if not B.agree(not B.STOP and time.monotonic() < deadline, values[0].device, world):
            raise TimeoutError('Triangle restart probe interrupted; no admission')

    path = output/'probes/restart.pt'
    mid = output/f'probes/restart_mid_rank_{rank}.pt'
    def ensure_fresh():
        if path.exists() or mid.exists():
            raise FileExistsError('Restart evidence exists; use a fresh benchmark campaign')
    checked(ensure_fresh)

    def snapshot(progress):
        return dict(model=base.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    rng=capture_rank_state(base), progress=copy.deepcopy(progress))

    progress = dict(attempt=POLICY['initial_attempt'], checks=0, fixture_cursor=0)
    initial = checked(lambda: cpu_copy(snapshot(progress)))
    rank_states = B.gather(initial['rng'], world)
    B.synchronize(values[0].device, world)
    start = time.monotonic()
    # Preserve the old timing scope: serialize live device tensors so measured
    # checkpoint time still includes device-to-host transfer, not just CPU I/O.
    root_call(lambda: atomic_torch_save(dict(campaign_sha256=spec['content_hash'],
        model=base.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        rank_states=rank_states, progress=progress), path), rank, world)
    seconds = max(B.gather(time.monotonic()-start, world))

    def disk_initial():
        saved = trusted_load(path)
        if saved['campaign_sha256'] != spec['content_hash'] or len(saved['rank_states']) != world:
            raise ValueError('Restart checkpoint identity changed')
        return dict(model=saved['model'], optimizer=saved['optimizer'], scaler=saved['scaler'],
                    rng=saved['rank_states'][rank], progress=saved['progress'])

    def restore(saved):
        base.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        base.zero_grad(set_to_none=True)
        base.train()
        base._state_pair_cache.clear()
        restore_rank_state(base, saved['rng'])

    def step(cursor):
        inputs = [v.roll(cursor['fixture_cursor'], dims=-1) for v in values]
        scale, counter = scaler.get_scale(), optimizer.step_counter
        row = B.step(base, model, optimizer, scaler, inputs, truth, spec['recipe'], cursor['attempt'], world)
        return dict(attempt=cursor['attempt'], fixture_cursor=cursor['fixture_cursor'], loss=row['loss'],
                    skipped=row['skipped'], scale_before=scale, scale_after=scaler.get_scale(),
                    lookahead_before=counter, lookahead_after=optimizer.step_counter,
                    lr=optimizer.param_groups[0]['lr'])

    def boundary(cursor):
        def validate():
            inputs = [v.roll(cursor['fixture_cursor']-1, dims=-1) for v in values]
            base.eval()
            with torch.no_grad(), model_precision(base, values[0].device, training=False):
                prediction = base(*(v[:8] for v in inputs))
            if not prediction.isfinite().all(): raise ValueError('Nonfinite restart validation')
            return True
        root_call(validate, rank, world)

    def roundtrip(state):
        atomic_torch_save(state, mid)
        return trusted_load(mid)

    report = trials(initial, restore=restore, snapshot=snapshot, step=step, boundary=boundary,
                    disk_initial=disk_initial, roundtrip=roundtrip, checked=checked, stop=stop)
    report['rank'] = rank
    ranks = B.gather(report, world)
    schedules = [[{k: v for k, v in u.items() if k != 'loss'}
                  for u in r['trials']['reference']['updates']] for r in ranks]
    evidence = dict(METADATA, policy=copy.deepcopy(POLICY), rank_evidence=ranks,
                    ranks_agree_on_update_schedule=schedules[0] == schedules[1])
    # All ranks have the same evidence. Recompute admission rather than trusting flags.
    checked(lambda: validate_evidence(evidence))
    return evidence, seconds
