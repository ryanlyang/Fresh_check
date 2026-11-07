#!/usr/bin/env python3
"""Read frozen seven-model code; write only a separate restart diagnostic.

No tolerance changes, benchmark admission, data caches, or production weights.
Each architecture runs in a fresh process so lazy optimizer imports cannot
contaminate the next architecture's checked preparation environment.
"""
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCHEMA = 'selected_restart_diagnostic_v1'
INDICES = (0, 1, 3, 4)
STEPS = (0, 1, 2, 3, 4, 5, 699999, 700000, 979999, 980000)
POLICY = dict(indices=list(INDICES), steps=list(STEPS), events=8,
              trials=['uninterrupted_a', 'uninterrupted_b', 'resumed_after_3'],
              original_gate='run_unchanged_before_instrumented_trials',
              walltime_seconds=7200, shutdown_margin_seconds=300,
              comparison='exact_all_leaves_no_relaxed_tolerance',
              authorizes_training=False, authorizes_benchmark_retry=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def plan(output):
    output = Path(output).resolve(strict=True)
    p = read(output / 'plan.json')
    if p['schema'] != SCHEMA or p['policy'] != POLICY:
        raise ValueError('Diagnostic plan/policy changed')
    if output.parent != Path(p['campaign']) / 'diagnostics':
        raise ValueError('Diagnostic must be a direct child of campaign/diagnostics')
    for name, expected in p['diagnostic_source'].items():
        if Path(name).name != name or sha(output / name) != expected:
            raise ValueError('Diagnostic source changed: ' + name)
    for name, expected in p['artifacts'].items():
        if sha(Path(p['campaign']) / name) != expected:
            raise ValueError('Campaign artifact changed: ' + name)
    return p


def cpu_copy(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare_tree(a, b):
    """Scan every leaf, even after failure; retain bounded path-level evidence."""
    import numpy as np
    import torch
    result = dict(exact=True, finite=True, leaves=0, differing_leaves=0,
                  maximum_absolute_error=0., first_mismatches=[], largest_errors=[], components={})

    def record(path, exact, error=0., finite=True, detail=None):
        component = path.split('.', 1)[0].split('[', 1)[0]
        group = result['components'].setdefault(component, dict(leaves=0, differing_leaves=0,
                                                                maximum_absolute_error=0.))
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
        if len(result['first_mismatches']) < 20:
            result['first_mismatches'].append(row)
        result['largest_errors'].append(row)
        result['largest_errors'].sort(key=lambda r: r['maximum_absolute_error'], reverse=True)
        del result['largest_errors'][20:]

    def walk(x, y, path):
        if type(x) is not type(y) and not (isinstance(x, dict) and isinstance(y, dict)):
            record(path, False, detail=f'types: {type(x).__name__} / {type(y).__name__}')
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            if x.shape != y.shape or x.dtype != y.dtype:
                record(path, False, detail=f'shape/dtype: {x.shape}/{x.dtype} / {y.shape}/{y.dtype}')
                return
            x, y = torch.as_tensor(x), torch.as_tensor(y)
            finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            error = float((x.double()-y.double()).abs().max()) if finite and x.numel() else 0.
            record(path, bool(torch.equal(x, y)), error, finite, dict(shape=list(x.shape), dtype=str(x.dtype)))
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                record(path, False, detail='dictionary keys differ')
            for key in x:
                if key in y:
                    walk(x[key], y[key], f'{path}.{key}' if path else str(key))
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                record(path, False, detail='sequence lengths differ')
            for i, (left, right) in enumerate(zip(x, y)):
                walk(left, right, f'{path}[{i}]')
        else:
            finite = not isinstance(x, float) or (math.isfinite(x) and math.isfinite(y))
            record(path, x == y, finite=finite, detail=None if x == y else f'{x!r} / {y!r}')
    walk(a, b, '')
    return result


def trials(model, batch, device, recipe, *, build_optimizer, training_step,
           capture_rank_state, restore_rank_state, check_stop=lambda: None):
    """Use the frozen test's ten updates and restart point, plus repeat control."""
    import torch
    initial, rng = cpu_copy(model.state_dict()), cpu_copy(capture_rank_state(model))
    modes = [(m, m.training) for m in model.modules()]
    batch = ({k: v[:8] for k, v in batch[0].items()}, {'_label_': batch[1]['_label_'][:8]}, {})
    reference = []
    comparisons, restore_checks, update_records, initial_checks = {}, [], {}, {}

    def snapshot(optimizer, scaler):
        return cpu_copy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), rng=capture_rank_state(model)))

    def scaler_new():
        return torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    try:
        for name in POLICY['trials']:
            check_stop()
            model.load_state_dict(initial)
            model.zero_grad(set_to_none=True)
            restore_rank_state(model, rng)
            for module, mode in modes:
                module.training = mode
            optimizer, scaler = build_optimizer(model, recipe), scaler_new()
            start = snapshot(optimizer, scaler)
            if name == 'uninterrupted_a':
                initial_reference = start
            else:
                initial_checks[name] = compare_tree(initial_reference, start)
            comparisons[name], update_records[name] = [], []
            for position, step in enumerate(STEPS):
                check_stop()
                row = training_step(model, optimizer, scaler, batch, device, recipe, step)
                update_records[name].append(dict(step=step, lr=row['lr'], amp_skipped=row['amp_skipped'],
                                                  scale=float(scaler.get_scale())))
                if name == 'resumed_after_3' and position == 2:
                    before = snapshot(optimizer, scaler)
                    stream = io.BytesIO()
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scaler=scaler.state_dict(), rng=capture_rank_state(model)), stream)
                    stream.seek(0)
                    saved = torch.load(stream, map_location=device, weights_only=False)
                    deserialize = compare_tree(before, cpu_copy(saved))
                    model.load_state_dict(saved['model'])
                    optimizer = build_optimizer(model, recipe)
                    optimizer.load_state_dict(saved['optimizer'])
                    scaler = scaler_new()
                    scaler.load_state_dict(saved['scaler'])
                    restore_rank_state(model, saved['rng'])
                    restore_checks.append(dict(after_updates=3, deserialization=deserialize,
                        restored_state=compare_tree(before, snapshot(optimizer, scaler))))
                    del saved, stream, before
                observed = snapshot(optimizer, scaler)
                if name == 'uninterrupted_a':
                    reference.append(observed)
                else:
                    comparisons[name].append(dict(after_updates=position+1, step=step,
                                                    **compare_tree(reference[position], observed)))
            del optimizer, scaler
        repeat = comparisons['uninterrupted_b']
        resumed = comparisons['resumed_after_3']
        roundtrip_exact = all(r[k]['exact'] for r in restore_checks for k in ('deserialization', 'restored_state'))
        return dict(complete=True, exact_roundtrip=roundtrip_exact,
                    identical_initial_states=all(r['exact'] for r in initial_checks.values()),
                    uninterrupted_repeat_exact=all(r['exact'] for r in repeat),
                    resumed_exact=all(r['exact'] for r in resumed), initial_checks=initial_checks,
                    roundtrip_checks=restore_checks, comparisons=comparisons, updates=update_records,
                    update_sequences_equal=len({json.dumps(v, sort_keys=True) for v in update_records.values()}) == 1,
                    first_repeat_divergence=next((r['after_updates'] for r in repeat if not r['exact']), None),
                    first_resume_divergence=next((r['after_updates'] for r in resumed if not r['exact']), None),
                    authorizes_training=False, authorizes_benchmark_retry=False)
    finally:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        restore_rank_state(model, rng)
        for module, mode in modes:
            module.training = mode


def worker(output, index, stop):
    p = plan(output)
    if index not in INDICES:
        raise ValueError('Only the four failed standard arms are diagnosed')
    source = Path(p['campaign']) / 'source'
    sys.path.insert(0, str(source))
    from scripts.submit_part_selected_fast import verify_snapshot
    verify_snapshot(Path(p['campaign']))
    from part_selected_fullscale import benchmark as B
    from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
    bench = B.Bench(Path(p['campaign']), index, stop)
    if bench.spec['content_hash'] != p['campaign_sha256']:
        raise ValueError('Prepared campaign hash differs from diagnostic plan')
    loader = bench.loader('train')
    try:
        model = bench.model()
        # Match benchmark's cold path before taking the pilot RNG snapshot.
        warm_optimizer = B.build_optimizer(model, bench.recipe)
        import torch
        warm_scaler = torch.amp.GradScaler('cuda')
        seed_everything(101)
        first = next(bench.batches(loader))
        fingerprint = B.selected_data.batch_fingerprint(first)
        if fingerprint != bench.spec['input_reference']['fingerprints']['0']:
            raise ValueError('First input batch differs from frozen campaign')
        # Retain the uninstrumented symptom: CPU state snapshots below introduce
        # synchronization and can change the manifestation of GPU nondeterminism.
        try:
            original_gate = dict(passed=True, result=B.full_recipe_restart(model, first, bench.device))
        except (AssertionError, ValueError) as error:
            original_gate = dict(passed=False, error=f'{type(error).__name__}: {error}',
                                 traceback=traceback.format_exc())
        result = trials(model, first, bench.device, bench.recipe, build_optimizer=B.build_optimizer,
                        training_step=B.training_step, capture_rank_state=capture_rank_state,
                        restore_rank_state=restore_rank_state, check_stop=stop)
        result.update(task=bench.task, runtime=bench.runtime, input_fingerprint=fingerprint,
                      original_gate=original_gate,
                      campaign_sha256=bench.spec['content_hash'], policy=POLICY,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      cudnn_deterministic=torch.backends.cudnn.deterministic,
                      cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                      parameters=sum(p.numel() for p in model.parameters()))
        write_new(output / f'task_{index}.json', result)
        del warm_optimizer, warm_scaler
    finally:
        B.selected_data.shutdown_loader(loader)


def review(output, save=False):
    p = plan(output)
    rows = []
    for index in INDICES:
        path = output / f'task_{index}.json'
        r = read(path) if path.exists() else {}
        rows.append(dict(index=index, model=r.get('task', {}).get('run_id', str(index)),
                         original_gate_passed=r.get('original_gate', {}).get('passed'),
                         **{k: r.get(k) for k in ('complete', 'exact_roundtrip', 'identical_initial_states',
                           'uninterrupted_repeat_exact', 'resumed_exact', 'update_sequences_equal',
                           'first_repeat_divergence', 'first_resume_divergence')},
                         error=r.get('error')))
    result = dict(campaign=p['campaign'], complete=all(r['complete'] for r in rows), models=rows,
                  authorizes_training=False, authorizes_benchmark_retry=False,
                  note='Completion means evidence collected, not parity passed. Inspect task JSON for every component and worst tensor paths.')
    if save:
        write_new(output / 'report.json', result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'worker', 'review'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args(argv)
    output = args.output.resolve(strict=True)
    p = plan(output)
    if args.phase == 'review':
        review(output)
        return 0
    start = time.monotonic()
    def stop(*_):
        if _ or time.monotonic()-start > POLICY['walltime_seconds']-POLICY['shutdown_margin_seconds']:
            raise TimeoutError('Two-hour diagnostic incomplete; no production authorization')
    for name in ('SIGTERM', 'SIGUSR1'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    if args.phase == 'worker':
        try:
            worker(output, args.task_index, stop)
        except Exception as error:
            target = output / f'task_{args.task_index}.json'
            if not target.exists():
                write_new(target, dict(complete=False, error=f'{type(error).__name__}: {error}',
                                       traceback=traceback.format_exc(), authorizes_training=False))
            raise
        return 0
    for index in INDICES:
        stop()
        if (output / f'task_{index}.json').exists():
            raise FileExistsError('Use a fresh diagnostic; never overwrite earlier evidence')
        print(f'Diagnosing task {index}', flush=True)
        seconds = POLICY['walltime_seconds']-POLICY['shutdown_margin_seconds']-(time.monotonic()-start)
        try:
            result = subprocess.run([p['python'], '-B', '-u', str(output / Path(__file__).name),
                                     '--phase', 'worker', '--output', str(output), '--task-index', str(index)],
                                    timeout=seconds, check=False)
            if result.returncode:
                print(f'Task {index} diagnostic exited {result.returncode}; retained failure evidence.', flush=True)
        except subprocess.TimeoutExpired:
            break
    return 0 if review(output, save=True)['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
