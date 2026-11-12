"""Small, stdlib-only contracts for independent full-validation rejection jobs."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

BASE = Path('/home/ryreu/atlas/Fresh_check/checkpoints')
PART = BASE / 'part_ev_state_fullscale/state_full_20261024T185049Z'
STATE = BASE / 'part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z'
IDS = {'part': 'PART_RANGER', 'state': 'EV_STATE_LAYERWISE_RANGER'}
SOURCES = ('scripts/part_ev_rejection_eval_support.py',
           'scripts/run_part_ev_rejection_eval.py',
           'scripts/submit_part_ev_rejection_eval.py',
           'sbatch/run_part_ev_rejection_eval.sh')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def seal(row):
    return dict(row, content_hash=digest(row))


def checked(row):
    if row.get('content_hash') != digest({k: v for k, v in row.items() if k != 'content_hash'}):
        raise ValueError('JSON content hash mismatch')
    return row


def write(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(row, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def python_from_config(path, variable):
    values = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(variable + '='):
            values.append(shlex.split(line.split('=', 1)[1]))
    if len(values) != 1 or len(values[0]) != 1:
        raise ValueError(f'Cannot identify frozen Python in {path}')
    return values[0][0]


def describe(root, key):
    root = Path(root).resolve(strict=True)
    spec = checked(read(root / 'campaign.json'))
    tasks = [t for t in spec['tasks'] if t['run_id'] == IDS[key] and t['seed'] == 101]
    if len(tasks) != 1:
        raise ValueError(f'Expected exactly one {key} seed-101 task')
    task = tasks[0]
    run = root / 'runs' / task['run_id'] / 'seed_101'
    done = read(run / 'training_complete.json')
    recipe = spec['recipes'][task['recipe']]
    if (recipe['total_steps'] != 1_000_000 or recipe['epochs'] != 50
            or recipe['steps_per_epoch'] != 20_000 or recipe['global_batch_size'] != 512
            or recipe['evaluation_population'] != 5_000_000
            or done['step'] != 1_000_000 or done['validated_checks'] != 50
            or done['campaign_sha256'] != spec['content_hash'] or done['task'] != task):
        raise ValueError(f'{key}: full training is incomplete or belongs to another recipe')
    paths = [run / 'validation' / f'check_{i:02d}.json' for i in range(1, 51)]
    if len(list((run / 'validation').glob('check_*.json'))) != 50:
        raise ValueError('Expected exactly 50 validation checks')
    checks = [read(p) for p in paths]
    for i, r in enumerate(checks, 1):
        if (r['check'] != i or r['step'] != 20_000 * i
                or r['events'] != recipe['validation_events_per_epoch']
                or not math.isfinite(r['accuracy']) or not 0 <= r['accuracy'] <= 1):
            raise ValueError('Invalid validation sequence')
    best = max(checks, key=lambda r: r['accuracy'])
    if (done['best_check'], done['best_accuracy']) != (best['check'], best['accuracy']):
        raise ValueError('Checkpoint selection must be first highest validation accuracy')
    checkpoint = run / 'best_model_val.pt'
    if sha(checkpoint) != done['best_checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash differs')
    files = sorted(spec['inventory']['splits']['val']['files'], key=lambda r: r['name'])
    if (len(files) != 50 or len({r['name'] for r in files}) != 50
            or any(r['num_entries'] != 100_000 for r in files)):
        raise ValueError('Expected 50 complete official validation files')
    if key == 'part':
        python = read(root / 'launch.json')['python']
    else:
        python = python_from_config(root / 'config.sh', 'EVDDP_PYTHON')
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError('Frozen donor Python is unavailable')
    paths += [root / 'campaign.json', root / 'source.sha256',
              run / 'training_complete.json', run / 'optimizer.json', checkpoint]
    return dict(key=key, root=str(root), run=str(run), python=python, task=task,
                best_check=best['check'], checkpoint_sha256=done['best_checkpoint_sha256'],
                campaign_sha256=spec['content_hash'], files=files,
                training_runtime=done['runtime'],
                pins={str(p.relative_to(root)): sha(p) for p in paths})


def verify_pins(donor):
    root = Path(donor['root'])
    for relative, checksum in donor['pins'].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != checksum:
            raise ValueError(f'Donor artifact changed: {path}')


def verify_frozen_source(donor):
    root = Path(donor['root'])
    source = (root / 'source').resolve(strict=True)
    for line in (root / 'source.sha256').read_text().splitlines():
        checksum, relative = line.split(maxsplit=1)
        path = (source / relative.lstrip('*')).resolve(strict=True)
        if not path.is_relative_to(source) or sha(path) != checksum:
            raise ValueError(f'Donor frozen source changed: {path}')


def verify_snapshot(output):
    output = Path(output)
    plan = checked(read(output / 'plan.json'))
    for relative, checksum in plan['source'].items():
        if relative not in SOURCES or sha(output / 'source' / relative) != checksum:
            raise ValueError('Evaluation source snapshot changed')
    if set(plan['source']) != set(SOURCES):
        raise ValueError('Incomplete evaluation source snapshot')
    return plan


def active_ids():
    # Expanded arrays, never infer completion from an unavailable scheduler.
    return set(command(['squeue', '--me', '-r', '-h', '-o', '%i']).split())


def ledger_jobs(path):
    if not Path(path).is_file():
        return []
    jobs = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit():
            raise ValueError(f'Malformed job ledger: {path}')
        jobs.append(fields[1])
    return jobs


def refuse_active(donors, extra_ledgers=(), live=None):
    live = active_ids() if live is None else live
    ledgers = list(extra_ledgers)
    for donor in donors.values():
        root = Path(donor['root'])
        ledgers += [root / 'submitted_job_ids.tsv']
        ledgers += list(root.glob('segment_chains/*/submitted_job_ids.tsv'))
    parents = {j.split('_')[0] for j in live}
    for ledger in ledgers:
        overlaps = parents.intersection(ledger_jobs(ledger))
        if overlaps:
            raise ValueError(f'Active old/duplicate jobs {sorted(overlaps)} in {ledger}; '
                             'inspect them first. Nothing cancelled automatically.')


def validate_arrays(logits, truth, events):
    import numpy as np
    if (logits.shape != (events, 10) or truth.shape != (events,)
            or not np.issubdtype(truth.dtype, np.integer)
            or not np.isfinite(logits).all() or np.any((truth < 0) | (truth >= 10))):
        raise ValueError('Invalid or incomplete validation predictions')


def rejection_metrics(logits, truth, names):
    """Same log-odds/tie convention as the frozen metric, without AUC/softmax."""
    import numpy as np
    validate_arrays(logits, truth, len(truth))
    support = np.bincount(truth, minlength=10)
    if len(names) != 10 or np.any(support == 0):
        raise ValueError('Need all ten classes')
    result = dict(events=len(truth), accuracy=float((logits.argmax(1) == truth).mean()),
                  class_support=dict(zip(names, support.tolist())), qcd_rejection={})
    for i, name in enumerate(names[1:], 1):
        score = logits[:, i].astype(np.float64) - logits[:, 0].astype(np.float64)
        signal = np.sort(score[truth == i])
        qcd = score[truth == 0]
        rows = {}
        for efficiency in (0.3, 0.5):
            threshold = float(signal[-math.ceil(efficiency * len(signal))])
            accepted = int(np.count_nonzero(qcd >= threshold))
            signal_accepted = int(np.count_nonzero(signal >= threshold))
            rows[str(efficiency)] = dict(target_signal_efficiency=efficiency,
                achieved_signal_efficiency=signal_accepted / len(signal),
                threshold_log_odds=threshold, signal_accepted=signal_accepted,
                signal_total=len(signal), qcd_accepted=accepted, qcd_total=len(qcd),
                rejection=len(qcd) / accepted if accepted else None)
        result['qcd_rejection'][name] = rows
    return result


def comparison_text(rows):
    lines = ['# Full-5M validation: ParT versus layerwise State-EV', '',
             'Seed 101; accuracy-selected full-schedule checkpoints. Not independent test results.',
             'Training differs: ParT one GPU/native BatchNorm; State-EV two GPUs/SyncBatchNorm.',
             'Positive change means higher QCD rejection. ZB means zero QCD accepted, not a measured infinity.', '']
    for key in ('part', 'state'):
        if key in rows:
            r = rows[key]
            lines.append(f"{key}: accuracy {100*r['metrics']['accuracy']:.4f}%, selected check {r['best_check']}")
        else:
            lines.append(f'{key}: evaluation pending or incomplete')
    if len(rows) != 2:
        for key, row in rows.items():
            lines += ['', f'## {key}: available rejection results', '',
                      '| Signal | Efficiency | Rejection | QCD accepted |',
                      '|---|---:|---:|---:|']
            for name, targets in row['metrics']['qcd_rejection'].items():
                for eff, r in targets.items():
                    value = 'ZB' if r['rejection'] is None else f"{r['rejection']:,.2f}"
                    lines.append(f"| {name} | {100*float(eff):.0f}% | {value} | {r['qcd_accepted']} |")
        return '\n'.join(lines)
    lines += ['', '| Signal | Efficiency | ParT rejection | State-EV rejection | Change (%) | QCD accepted P/S |',
              '|---|---:|---:|---:|---:|---:|']
    for name, targets in rows['part']['metrics']['qcd_rejection'].items():
        for eff, p in targets.items():
            s = rows['state']['metrics']['qcd_rejection'][name][eff]
            rp, rs = p['rejection'], s['rejection']
            delta = '--' if rp is None or rs is None else f'{100*(rs/rp-1):+.2f}'
            fmt = lambda v: 'ZB' if v is None else f'{v:,.2f}'
            lines.append(f"| {name} | {100*float(eff):.0f}% | {fmt(rp)} | {fmt(rs)} | {delta} | "
                         f"{p['qcd_accepted']}/{s['qcd_accepted']} |")
    lines += ['', 'Hqql/Tbl are also shown at 30%/50% here, not their usual 99%/99.5% working points.',
              'Validation overlaps checkpoint selection; single-seed differences do not isolate architecture.']
    return '\n'.join(lines)
