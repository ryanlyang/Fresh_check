#!/usr/bin/env python3
"""Resource-only admission overlay for an unchanged, already-benchmarked campaign.

The original spec, source, inputs, benchmarks and checkpoint identities stay intact.
Only the *in-memory argument to the original admission function* receives the new
walltimes. Every other original admission check is still executed. The overlay is
explicit, hash-bound, and installed only in this worker process, never on disk in
the original source tree or in an environment.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import runpy
import sys

SCHEMA = 'part_selected_tigris_walltime_recovery_v1'
APPROVAL = 'part_selected_tigris_walltime_approval_v1'
RUNS = ('SELECTED_PART_SHARED', 'SELECTED_EV_SHARED',
        'SELECTED_PART_LAYERWISE', 'SELECTED_EV_LAYERWISE')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def hashed(row):
    payload = {k: v for k, v in row.items() if k != 'content_hash'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    return dict(payload, content_hash=digest)


def checked(path, schema):
    row = read(path)
    if not isinstance(row, dict) or row.get('schema') != schema or row != hashed(row):
        raise ValueError(f'Invalid recovery/campaign metadata: {path}')
    return row


def positive_hours(value):
    if isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0:
        raise ValueError('Walltimes must be positive whole hours')
    return int(value)


def partition_hours(value):
    if value.upper() in ('UNLIMITED', 'INFINITE'):
        return math.inf
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    fields = clock.split(':')
    if len(fields) != 3 or not all(v.isdigit() for v in [days, *fields]):
        raise ValueError(f'Unrecognized partition MaxTime: {value}')
    hours, minutes, seconds = map(int, fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f'Invalid partition MaxTime: {value}')
    return int(days)*24 + hours + minutes/60 + seconds/3600


def original_files(spec):
    return ['experiment_spec.json', 'launch.json', 'launch.env', 'source.sha256'] + [
        f"benchmarks/{task['run_id']}.json" for task in spec['tasks']]


def estimates(output, spec):
    rows = []
    for task in spec['tasks']:
        row = read(output/'benchmarks'/f"{task['run_id']}.json")
        if (row.get('passed') is not True or row.get('task') != task
                or row.get('campaign_sha256') != spec['content_hash']):
            raise ValueError('Recovery requires all four successful original benchmark records')
        train = float(row['projected_training_hours'])*1.3
        validation = row['validation']
        if validation['events'] <= 0:
            raise ValueError('Missing validation timing population')
        evaluate = float(validation['seconds'])*5000000/validation['events']/3600*1.3
        if not all(math.isfinite(v) and v > 0 for v in (train, evaluate)):
            raise ValueError('Invalid benchmark timing projection')
        rows.append(dict(run_id=task['run_id'], train_hours_with_margin=train,
                         evaluate_hours_with_margin=evaluate))
    return rows


def configure(output, recovery, train_hours, evaluate_hours, max_time, *, dry_run=False):
    output, recovery = Path(output).resolve(strict=True), Path(recovery).resolve()
    # A separate small sibling of the original source/inputs/runs, never a rewrite.
    if recovery.parent != output/'recoveries':
        raise ValueError('Recovery must be an immediate child of CAMPAIGN/recoveries')
    spec = checked(output/'experiment_spec.json', 'part_selected_tigris_v1')
    if [t['run_id'] for t in spec['tasks']] != list(RUNS):
        raise ValueError('Expected the four selected-relation tasks in original order')
    if read(output/'launch.json') != spec['launch']:
        raise ValueError('Original launch metadata changed')
    original = {k: positive_hours(spec['launch'][k]) for k in ('train_hours', 'evaluate_hours')}
    limits = dict(train_hours=positive_hours(train_hours),
                  evaluate_hours=positive_hours(evaluate_hours))
    if any(limits[k] < original[k] for k in original) or limits == original:
        raise ValueError('Recovery must only increase walltime(s), not reduce or keep both unchanged')
    if max(2, *limits.values()) > partition_hours(max_time):
        raise ValueError(f'Requested walltime exceeds partition MaxTime={max_time}')
    rows = estimates(output, spec)
    for row in rows:
        if (row['train_hours_with_margin'] > limits['train_hours']
                or row['evaluate_hours_with_margin'] > limits['evaluate_hours']):
            raise ValueError(f"{row['run_id']} needs at least {row['train_hours_with_margin']:.2f} h training "
                             f"and {row['evaluate_hours_with_margin']:.2f} h evaluation (30% margin included)")
    manifest = hashed(dict(schema=SCHEMA, campaign=str(output),
        campaign_sha256=spec['content_hash'], original_walltimes=original, walltimes=limits,
        original_files={name: sha(output/name) for name in original_files(spec)},
        estimates=rows, scope='walltimes_only_no_scientific_or_environment_changes'))
    path = recovery/'request.json'
    if path.exists():
        if checked(path, SCHEMA) != manifest:
            raise ValueError('Recovery request is frozen; use its unchanged limits when resuming')
    elif not dry_run:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    for row in rows:
        print(f"{row['run_id']}: required train {row['train_hours_with_margin']:.2f} h; "
              f"evaluate {row['evaluate_hours_with_margin']:.2f} h", flush=True)
    print(f"Requested train {limits['train_hours']} h; evaluate {limits['evaluate_hours']} h. "
          'Original campaign and benchmarks unchanged.', flush=True)
    return manifest


def load_request(output, recovery):
    output, recovery = Path(output).resolve(strict=True), Path(recovery).resolve(strict=True)
    if recovery.parent != output/'recoveries':
        raise ValueError('Recovery is outside this campaign')
    record = checked(recovery/'request.json', SCHEMA)
    spec = checked(output/'experiment_spec.json', 'part_selected_tigris_v1')
    if record['campaign'] != str(output) or record['campaign_sha256'] != spec['content_hash']:
        raise ValueError('Recovery belongs to another campaign')
    if set(record['original_files']) != set(original_files(spec)):
        raise ValueError('Recovery is missing original evidence hashes')
    if any(sha(output/name) != digest for name, digest in record['original_files'].items()):
        raise ValueError('Original campaign or benchmark evidence changed after recovery was frozen')
    limits, original = record['walltimes'], record['original_walltimes']
    if set(limits) != {'train_hours', 'evaluate_hours'} or set(original) != set(limits):
        raise ValueError('Only training/evaluation walltimes may be overridden')
    if original != {k: positive_hours(spec['launch'][k]) for k in limits}:
        raise ValueError('Original walltimes differ')
    if any(positive_hours(limits[k]) < original[k] for k in limits) or limits == original:
        raise ValueError('Invalid walltime extension')
    return record


@contextlib.contextmanager
def admission_overlay(protocol, output, spec, request):
    """Run the original validator unchanged except its explicitly approved budgets."""
    original = protocol.require_approval

    def require_approval(candidate_output, candidate_spec, runtime=None):
        if Path(candidate_output).resolve() != Path(output).resolve() or candidate_spec != spec:
            raise ValueError('Recovery cannot change the campaign or its scientific protocol')
        admission_spec = copy.deepcopy(candidate_spec)
        admission_spec['launch'].update({k: str(v) for k, v in request['walltimes'].items()})
        return original(candidate_output, admission_spec, runtime)

    protocol.require_approval = require_approval
    try:
        yield require_approval
    finally:
        protocol.require_approval = original


def execute(output, recovery, phase, task_index=None):
    if phase not in ('gate', 'train', 'lock', 'evaluate', 'report'):
        raise ValueError('Recovery cannot rerun preparation or benchmarks')
    if phase in ('train', 'evaluate') and (type(task_index) is not int or task_index not in range(4)):
        raise ValueError('Expected task index 0..3')
    output, recovery = Path(output).resolve(strict=True), Path(recovery).resolve(strict=True)
    request = load_request(output, recovery)
    source = output/'source'
    sys.path.insert(0, str(source))
    protocol = importlib.import_module('part_selected_tigris.protocol')
    if Path(protocol.__file__).resolve() != (source/'part_selected_tigris/protocol.py').resolve():
        raise ValueError('Must import the original frozen protocol, not the current checkout')
    spec = protocol.load_campaign(output)  # Original source/input/environment validation.
    with admission_overlay(protocol, output, spec, request) as require_approval:
        rows = require_approval(output, spec)  # Full original parity/runtime/input/AMP/memory gate.
        approval = protocol.hashed(dict(schema=APPROVAL, recovery_sha256=request['content_hash'],
            campaign_sha256=spec['content_hash'], walltimes=request['walltimes'],
            benchmark_files=request['original_files'], passed=True))
        path = recovery/'approval.json'
        with protocol.file_lock(recovery/'.approval.lock'):
            if path.exists():
                if read(path) != approval:
                    raise ValueError('Recovery approval changed')
            elif phase == 'gate':
                protocol.write_json(path, approval)
            else:
                raise ValueError('Recovery gate must approve before production')
        print(json.dumps(dict(stage='walltime_recovery', phase=phase,
            recovery_sha256=request['content_hash'], campaign_sha256=spec['content_hash'],
            walltimes=request['walltimes'], original_benchmarks=len(rows))), flush=True)
        if phase == 'gate':
            return
        previous = sys.argv
        try:
            driver = source/'scripts/run_part_selected_tigris.py'
            sys.argv = [str(driver), '--phase', phase, '--output', str(output)]
            if phase in ('train', 'evaluate'):
                sys.argv += ['--task-index', str(task_index)]
            runpy.run_path(str(driver), run_name='__main__')
        finally:
            sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('configure', 'gate', 'train', 'lock', 'evaluate', 'report'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--train-hours', type=int)
    parser.add_argument('--evaluate-hours', type=int)
    parser.add_argument('--partition-max-time')
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.phase == 'configure':
        configure(args.output, args.recovery, args.train_hours, args.evaluate_hours,
                  args.partition_max_time, dry_run=args.dry_run)
    else:
        if args.dry_run or args.train_hours is not None or args.evaluate_hours is not None:
            parser.error('Worker phases use only the frozen recovery request')
        execute(args.output, args.recovery, args.phase, args.task_index)


if __name__ == '__main__':
    main()
