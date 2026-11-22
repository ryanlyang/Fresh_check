"""Diagnostic comparisons only; never relax or authorize a benchmark gate."""
import copy
import math
import torch
import numpy as np


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare_tree(expected, actual):
    """Scan every leaf exactly, retaining bounded diagnostics, including nonfinites."""
    result = dict(exact=True, finite=True, leaves=0, differing_leaves=0,
                  maximum_absolute_error=0., first_mismatches=[], largest_errors=[], components={})

    def record(path, exact, error=0., finite=True, detail=None):
        group = result['components'].setdefault(path.split('.', 1)[0],
            dict(leaves=0, differing_leaves=0, maximum_absolute_error=0.))
        result['leaves'] += 1
        group['leaves'] += 1
        result['finite'] &= finite
        if exact and finite:
            return
        result['exact'] = False
        result['differing_leaves'] += 1
        group['differing_leaves'] += 1
        result['maximum_absolute_error'] = max(result['maximum_absolute_error'], error)
        group['maximum_absolute_error'] = max(group['maximum_absolute_error'], error)
        row = dict(path=path, maximum_absolute_error=error, finite=finite, detail=detail)
        if len(result['first_mismatches']) < 12:
            result['first_mismatches'].append(row)
        result['largest_errors'].append(row)
        result['largest_errors'].sort(key=lambda r: r['maximum_absolute_error'], reverse=True)
        del result['largest_errors'][12:]

    def walk(x, y, path):
        if type(x) is not type(y) and not (isinstance(x, dict) and isinstance(y, dict)):
            record(path, False, detail=f'types: {type(x).__name__}/{type(y).__name__}')
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            if x.shape != y.shape or x.dtype != y.dtype:
                record(path, False, detail='shape/dtype mismatch')
                return
            a, b = torch.as_tensor(x).detach().cpu(), torch.as_tensor(y).detach().cpu()
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            error = float((a.double()-b.double()).abs().max()) if finite and a.numel() else 0.
            record(path, bool(torch.equal(a, b)), error, finite)
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                record(path, False, detail='keys differ')
            for key in x:
                if key in y:
                    walk(x[key], y[key], f'{path}.{key}' if path else str(key))
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                record(path, False, detail='length differs')
            for i, (a, b) in enumerate(zip(x, y)):
                walk(a, b, f'{path}[{i}]')
        else:
            finite = not isinstance(x, float) or (math.isfinite(x) and math.isfinite(y))
            error = abs(x-y) if finite and isinstance(x, (int, float)) else 0.
            record(path, x == y, error, finite)
    walk(expected, actual, '')
    return result


def trials(initial, *, restore, snapshot, step, boundary, disk_initial, roundtrip,
           stop=lambda: None, emit=lambda name, row: None, steps=7, resume_after=3):
    """Callbacks isolate frozen science; CPU tests can exercise genuine Ranger.

    Repeated common-start uninterrupted sequences establish a repeatability
    control. Disk-start and mid-sequence restore trials add restart comparisons.
    Exact differences are evidence, not failures of diagnostic collection.
    """
    initial = cpu_copy(initial)
    report = dict(complete=False, authorizes_training=False, authorizes_benchmark_retry=False,
                  low_instrumentation={}, trials={})

    def start(payload):
        progress = copy.deepcopy(payload['progress'])
        restore(cpu_copy(payload))
        return progress, compare_tree(payload, snapshot(progress))

    def advance(progress):
        stop()
        row = step(progress)
        progress['attempt'] += 1
        progress['fixture_cursor'] += 1
        if progress['attempt'] % 20000 == 0:
            boundary(progress)
            progress['checks'] += 1
        return row

    # Preserve a low-instrumentation loss comparison before full CPU snapshots
    # introduce synchronization. No per-update tensor copies or extra collectives.
    for name in ('memory', 'disk'):
        progress, restored = start(initial if name == 'memory' else disk_initial())
        updates = [advance(progress) for _ in range(steps)]
        final = cpu_copy(snapshot(progress))
        row = dict(restore=restored, updates=updates)
        if name == 'memory':
            low_reference = final
        else:
            row['final_vs_memory'] = compare_tree(low_reference, final)
        report['low_instrumentation'][name] = row
        emit('low_'+name, row)
    del low_reference, final

    references = []
    for name in ('memory_a', 'memory_b', 'disk_start', 'resume_after_3'):
        payload = disk_initial() if name == 'disk_start' else initial
        progress, restored = start(payload)
        row = dict(initial_restore=restored, updates=[], comparisons=[], roundtrips=[])
        for position in range(steps):
            update = advance(progress)
            row['updates'].append(update)
            state = cpu_copy(snapshot(progress))
            if name == 'resume_after_3' and position+1 == resume_after:
                # Save rank-local buffers/RNG exactly as observed. This is an
                # isolated roundtrip, not a production checkpoint or stream replay.
                saved = roundtrip(cpu_copy(state))
                decoded = compare_tree(state, saved)
                restore(cpu_copy(saved))
                progress = copy.deepcopy(saved['progress'])
                after = cpu_copy(snapshot(progress))
                row['roundtrips'].append(dict(after_updates=position+1,
                    deserialization=decoded, restoration=compare_tree(state, after)))
                state = after
            if name == 'memory_a':
                references.append(state)
            else:
                row['comparisons'].append(dict(after_updates=position+1,
                    **compare_tree(references[position], state)))
            emit(name+f'_update_{position+1}', dict(update=update,
                 comparison=row['comparisons'][-1] if row['comparisons'] else None))
        row['first_divergence'] = next((r['after_updates'] for r in row['comparisons']
                                        if not r['exact']), None)
        report['trials'][name] = row
        emit(name, row)
    report['complete'] = True
    report['all_initial_restores_exact'] = all(r['initial_restore']['exact']
                                               for r in report['trials'].values())
    report['mid_roundtrip_exact'] = all(r[k]['exact']
        for r in report['trials']['resume_after_3']['roundtrips']
        for k in ('deserialization', 'restoration'))
    schedules = [[{k: v for k, v in update.items() if k != 'loss'} for update in row['updates']]
                 for row in report['trials'].values()]
    report['trial_update_schedules_equal'] = all(compare_tree(schedules[0], s)['exact'] for s in schedules[1:])
    report['total_skipped_updates'] = sum(u['skipped'] for row in report['trials'].values() for u in row['updates'])
    return report
