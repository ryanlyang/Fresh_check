"""Read-only, stdlib-only segmented State-EV progress and validation viewer.

Reads small JSON files/logs and optionally queries squeue. Never imports a trainer,
opens checkpoints/predictions/data, acquires locks, or changes/submits any jobs.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


BASE = Path('/home/ryreu/atlas/Fresh_check/checkpoints')
DEFAULT_CAMPAIGN = BASE / 'part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z'
DEFAULT_CONTROLS = BASE / 'part_ev_state_fullscale/state_full_20261024T185049Z'
STAGES = {'training', 'validation', 'training_complete', 'resume_data_replay',
          'segment_handoff'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def optional_json(path, warnings):
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        warnings.append(f'Could not read {path}: {exc}')
        return {}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validation_rows(run, interval, warnings):
    rows = {}
    paths = sorted((run / 'validation').glob('check_*.json'))
    paths += sorted((run / 'validation').glob('epoch_*.json'))
    for path in paths:
        row = optional_json(path, warnings)
        if not row:
            continue
        check = int(path.stem.rsplit('_', 1)[1])
        if (row.get('check', check) != check or check < 1
                or row.get('step') != check * interval
                or not finite(row.get('accuracy')) or not 0 <= row['accuracy'] <= 1):
            raise ValueError(f'Invalid validation check: {path}')
        row = dict(row, check=check)
        if check in rows and row != rows[check]:
            raise ValueError(f'Conflicting validation check {check}: {run}')
        rows[check] = row
    return rows


def queue_snapshot():
    try:
        result = subprocess.run(['squeue', '--me', '-h', '-o', '%i|%T|%j'],
                                capture_output=True, text=True, timeout=10, check=True)
        return {parts[0]: parts[1] for line in result.stdout.splitlines()
                if len(parts := line.strip().split('|')) == 3}, None
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f'Scheduler unavailable ({exc}); showing latest log, not assuming it is running.'


def choose_log(campaign, queue):
    paths = set(campaign.glob('segment_chains/*/logs/evseg_segment_*.out'))
    paths.update(campaign.glob('logs/evddp_train_*.out'))
    # An allocated job may not have created its log yet. Do not fall back to an
    # old segment and describe its last training update as current activity.
    for ledger in campaign.glob('segment_chains/*/submitted_job_ids.tsv'):
        for line in ledger.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == 'segment':
                job = fields[1]
                if queue and queue.get(job) == 'RUNNING':
                    paths.add(ledger.parent / 'logs' / f'evseg_segment_{job}.out')
    def key(path):
        job = path.stem.rsplit('_', 1)[1]
        return (bool(queue and queue.get(job) == 'RUNNING'),
                path.stat().st_mtime if path.exists() else 0, str(path))
    return max(paths, key=key) if paths else None


def log_records(path):
    latest, by_stage = {}, {}
    if path and path.is_file():
        with path.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # Includes an unfinished final line in a live log.
                if isinstance(row, dict) and row.get('stage') in STAGES:
                    latest = row
                    by_stage[row['stage']] = row
    return latest, by_stage


def progress_lines(campaign, run, recipe, vals, queue, now, stale_minutes, warnings):
    path = choose_log(campaign, queue)
    latest, by_stage = log_records(path)
    lines = ['=== Running State-EV progress ===']
    step = max((row['step'] for row in vals.values()), default=0)
    if path:
        job = path.stem.rsplit('_', 1)[1]
        status = queue.get(job, 'not in active queue') if queue is not None else 'unknown'
        lines += [f'Job: {job} | Scheduler: {status}', f'Log: {path}']
        age = max(0, now - path.stat().st_mtime) / 60 if path.exists() else math.inf
        lines.append(f'Log last modified {age:.1f} minutes ago.' if path.exists()
                     else 'Active segment has not created its stdout log yet.')
        if path.parent.parent.parent.name == 'segment_chains':
            chain = path.parent.parent
            plan = optional_json(chain / 'plan.json', warnings)
            step = max(step, plan.get('initial', {}).get('step', 0))
            for receipt in (chain / 'receipts').glob('segment_*.json'):
                row = optional_json(receipt, warnings)
                step = max(step, row.get('after', {}).get('step', 0))
        for row in by_stage.values():
            step = max(step, row.get('step', 0), row.get('after', {}).get('step', 0))
    else:
        status, age = 'unknown', math.inf
        lines.append('No segment training logs found yet.')

    total, interval = recipe['total_steps'], recipe['steps_per_epoch']
    checks = total // interval
    contiguous = 0
    while contiguous + 1 in vals:
        contiguous += 1
    if contiguous != len(vals):
        warnings.append('Validation history has a gap; next-check calculation uses the contiguous prefix.')
    done = (run / 'training_complete.json').is_file()
    if done:
        completion = optional_json(run / 'training_complete.json', warnings)
        step = max(step, completion.get('step', 0))
    lines += [f'Latest observed update: {step:,}/{total:,} ({100 * step / total:.2f}%)',
              f'Completed validation checks: {len(vals)}/{checks}']
    if done:
        return lines + ['Training completion marker present.'], path
    if contiguous == checks:
        return lines + ['All validation checks recorded; waiting for training finalization.'], path

    target = (contiguous + 1) * interval
    remaining = max(0, target - step)
    within = max(0, min(interval, step - contiguous * interval))
    lines += [f'Next validation: check {contiguous + 1}, after update {target:,}',
              f'Training toward that check: {within:,}/{interval:,} '
              f'({100 * within / interval:.1f}%); {remaining:,} updates left.']
    durations = [vals[i]['seconds'] for i in sorted(vals)[-5:]
                 if finite(vals[i].get('seconds')) and vals[i]['seconds'] > 0]
    duration = statistics.median(durations) if durations else None
    if duration is not None:
        lines.append(f'Typical full validation duration: {duration / 60:.1f} minutes '
                     '(median of recent checks).')

    stage = latest.get('stage')
    if stage == 'resume_data_replay':
        replay_done, replay_total = latest['batches'], latest['total']
        percent = 100 * replay_done / replay_total if replay_total else 100
        lines += [f'Latest phase: restart data replay ({latest.get("stream", "unknown")} stream): '
                  f'{replay_done:,}/{replay_total:,} batches ({percent:.1f}%).',
                  'Replay restores the data position; it does not advance training updates.',
                  'ETA withheld during replay: this log has no per-record timestamps to estimate its remaining time.']
    elif stage == 'segment_handoff':
        lines += ['Latest phase: segment handoff. Waiting for the next segment; no live training ETA.']
    elif status != 'RUNNING' or age > stale_minutes or not latest:
        lines += ['Live ETA unavailable: job is not confirmed running, log is stale, or no progress is logged yet.']
    elif remaining == 0:
        lines += ['Validation is due or underway (possibly checkpointing); its live percentage is not logged.',
                  'No precise remaining validation ETA is available; use the typical full duration above.']
    else:
        rate = by_stage.get('training', {}).get('global_jets_per_second')
        if finite(rate) and rate > 0:
            seconds = remaining * recipe['global_batch_size'] / rate
            lines.append(f'Approx. training time until validation starts: {seconds / 60:.1f} minutes.')
            if duration is not None:
                lines.append(f'Approx. time until the result: {(seconds + duration) / 60:.1f} minutes.')
        else:
            lines.append('ETA unavailable until this segment logs a training throughput.')
    if age > stale_minutes:
        lines.append('WARNING: stale log; the phase above is only the last observed activity, not proof of a stall.')
    return lines, path


def best(rows, field='accuracy', cutoff=None):
    candidates = [rows[i] for i in sorted(rows) if (cutoff is None or i <= cutoff)
                  and finite(rows[i].get(field))]
    return max(candidates, key=lambda row: row[field]) if candidates else None


def metric_cell(row):
    if row is None:
        return '--'
    auc = f'{row["macro_auc_ovo"]:.6f}' if finite(row.get('macro_auc_ovo')) else '--'
    return f'{100 * row["accuracy"]:.4f}% / {auc}'


def history_lines(models, interval):
    lines = ['=== Every completed validation check: accuracy / macro OVO AUC ===',
             f'{"Check":>5} {"Step":>10}' + ''.join(f'{name:>27}' for name, _, _ in models)]
    for i in sorted(set().union(*(set(rows) for _, _, rows in models))):
        lines.append(f'{i:5d} {i * interval:10,d}' +
                     ''.join(f'{metric_cell(rows.get(i)):>27}' for _, _, rows in models))
    lines += ['', '=== Best so far (training progress may differ) ===']
    for name, _, rows in models:
        acc, auc = best(rows), best(rows, 'macro_auc_ovo')
        if acc:
            lines.append(f'{name}: best accuracy {100 * acc["accuracy"]:.4f}% at check {acc["check"]}; '
                         f'accuracy / AUC there: {metric_cell(acc)}')
        if auc:
            lines.append(f'  Best AUC independently: {auc["macro_auc_ovo"]:.6f} at check {auc["check"]}')

    cutoff = 0
    while models and all(cutoff + 1 in rows for _, _, rows in models):
        cutoff += 1
    if cutoff and len(models) > 1:
        lines += ['', f'=== Best through common check {cutoff}: equal update cutoff ===']
        baselines = [(name, best(rows, cutoff=cutoff)) for name, _, rows in models[:-1]]
        for name, _, rows in models:
            row = best(rows, cutoff=cutoff)
            lines.append(f'{name:16} {100 * row["accuracy"]:.4f}% at check {row["check"]:2d} '
                         + ''.join(f'| delta {label}: '
                                   f'{100 * (row["accuracy"] - baseline["accuracy"]):+.4f} pp '
                                   for label, baseline in baselines))
    elif len(models) > 1:
        lines += ['', 'No common completed validation prefix yet.']
    return lines


def report(campaign, controls, queue, *, now=None, stale_minutes=30):
    warnings = []
    spec = read_json(campaign / 'campaign.json')
    task = spec['task']
    recipe = spec['recipes'][task['recipe']]
    interval = recipe['steps_per_epoch']
    run = campaign / 'runs' / task['run_id'] / f'seed_{task["seed"]}'
    vals = validation_rows(run, interval, warnings)
    models = []
    if controls is not None:
        if (controls / 'campaign.json').is_file():
            control_spec = read_json(controls / 'campaign.json')
            for label, architecture in [('ParT', 'part'), ('ParT-EV', 'part_ev')]:
                matches = [t for t in control_spec['tasks']
                           if t['architecture'] == architecture and t['seed'] == task['seed']]
                if len(matches) != 1:
                    raise ValueError(f'Need one {label} seed {task["seed"]} in {controls}')
                t = matches[0]
                control_recipe = (control_spec['recipe'] if 'recipe' in control_spec
                                  else control_spec['recipes'][t['recipe']])
                for field in ('total_steps', 'steps_per_epoch', 'global_batch_size',
                              'validation_events_per_epoch'):
                    if recipe[field] != control_recipe[field]:
                        raise ValueError(f'Control {label} differs in {field}; do not mix short/full schedules')
                directory = controls / 'runs' / t.get('run_id', architecture) / f'seed_{t["seed"]}'
                models.append((label, directory, validation_rows(directory, interval, warnings)))
        else:
            warnings.append(f'Control campaign missing: {controls}; displaying State-EV only.')
    models.append(('State-EV', run, vals))
    lines = [f'Campaign: {campaign}', f'Seed: {task["seed"]}']
    lines += [f'{label} validation source: {directory}' for label, directory, _ in models]
    lines += [f'{label}: {len(rows)}/{recipe["total_steps"] // interval} checks recorded'
              for label, _, rows in models]
    progress, _ = progress_lines(campaign, run, recipe, vals, queue,
                                 time.time() if now is None else now, stale_minutes, warnings)
    lines += ['', *progress, '', *history_lines(models, interval), '',
              f'Checks are {interval:,} updates, not full dataset epochs; -- means not recorded.',
              'Checkpoint selection uses accuracy; independently best AUC is descriptive only.',
              'ETAs start at the latest logged update, using cumulative training throughput; '
              'they exclude checkpoint overhead, stalls, replay and queue gaps.',
              'Single-seed training-time validation, not full-5M evaluation or independent test results.',
              'State-EV: two GPUs/SyncBatchNorm. Historical controls: one GPU/native BatchNorm; '
              'these contrasts do not isolate architecture alone.',
              'Read-only display, not a checkpoint/hash audit. No checkpoint/data files opened or jobs changed.']
    lines += [f'WARNING: {message}' for message in warnings]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--controls', type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument('--no-controls', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Do not query Slurm; suppress live ETAs.')
    parser.add_argument('--stale-minutes', type=float, default=30)
    args = parser.parse_args(argv)
    if not math.isfinite(args.stale_minutes) or args.stale_minutes <= 0:
        parser.error('--stale-minutes must be finite and positive')
    queue, warning = (None, None) if args.offline else queue_snapshot()
    try:
        print(report(args.campaign, None if args.no_controls else args.controls, queue,
                     stale_minutes=args.stale_minutes))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Status could not be read: {exc}\nNo files or jobs changed.\n')
    if warning:
        print(f'WARNING: {warning}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
