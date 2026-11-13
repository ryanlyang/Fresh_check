#!/usr/bin/env python3
"""Read-only CPU post-processing of saved evrej predictions; no model/data loading."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part_ev_rejection_eval_support as S
from run_part_ev_rejection_eval import chunk_contract, read_chunk

# Fixed ten-class ordering used by these full-schedule JetClass campaigns.
CLASS_NAMES = ('QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl')
TARGETS = {'Hqql': 0.99, 'Tbl': 0.995}
FILES = 50
ENTRIES_PER_FILE = 100_000


def working_point(signal, background, efficiency):
    import numpy as np
    if not len(signal) or not len(background):
        raise ValueError('Signal and QCD populations must both be present')
    # Identical threshold and inclusive tie convention to calculate_metrics.
    signal = np.sort(signal)
    threshold = float(signal[-math.ceil(efficiency * len(signal))])
    accepted = int(np.count_nonzero(background >= threshold))
    signal_accepted = int(np.count_nonzero(signal >= threshold))
    return dict(target_signal_efficiency=efficiency,
                achieved_signal_efficiency=signal_accepted / len(signal),
                threshold_log_odds=threshold, signal_accepted=signal_accepted,
                signal_total=len(signal), qcd_accepted=accepted, qcd_total=len(background),
                rejection=len(background) / accepted if accepted else None)


def inspect_model(output, plan, key):
    import numpy as np
    donor = plan['donors'][key]
    result = S.checked(S.read(output / key / 'metrics.json'))
    population = FILES * ENTRIES_PER_FILE
    if (result['plan_sha256'] != plan['content_hash'] or result['model'] != key
            or result['task'] != donor['task']
            or result['campaign_sha256'] != donor['campaign_sha256']
            or result['checkpoint_sha256'] != donor['checkpoint_sha256']
            or result['best_check'] != donor['best_check']
            or result['population'] != 'official_validation_5M'
            or result['independent_test'] is not False
            or result['metrics']['events'] != population
            or len(result['identity_manifest']) != FILES or len(donor['files']) != FILES):
        raise ValueError(f'{key}: saved evaluation identity/coverage mismatch')
    support = np.zeros(10, dtype=np.int64)
    parts = {name: {'signal': [], 'qcd': []} for name in TARGETS}
    label_hashes = []
    for i, record in enumerate(donor['files']):
        if record['num_entries'] != ENTRIES_PER_FILE:
            raise ValueError('Unexpected validation file size')
        path = output / key / f'file_{i:03d}.npz'
        logits, labels, receipt = read_chunk(path, chunk_contract(plan, donor, record))
        if receipt != result['identity_manifest'][i]:
            raise ValueError('Prediction sidecar differs from completed evaluation manifest')
        support += np.bincount(labels, minlength=10)
        label_hashes.append(receipt['labels_sha256'])
        for name in TARGETS:
            index = CLASS_NAMES.index(name)
            score = logits[:, index].astype(np.float64) - logits[:, 0].astype(np.float64)
            parts[name]['signal'].append(score[labels == index])
            parts[name]['qcd'].append(score[labels == 0])
        if (i + 1) % 10 == 0:
            print(f'{key}: verified {i+1}/{FILES} prediction files', file=sys.stderr, flush=True)
    if (not np.array_equal(support, np.full(10, population // 10))
            or result['metrics']['class_support'] != dict(zip(CLASS_NAMES, support.tolist()))):
        raise ValueError('Expected complete, balanced validation population')
    rows = {name: working_point(np.concatenate(parts[name]['signal']),
                                np.concatenate(parts[name]['qcd']), efficiency)
            for name, efficiency in TARGETS.items()}
    return rows, label_hashes


def comparison(output):
    output = Path(output).resolve(strict=True)
    plan = S.verify_snapshot(output)
    if plan['schema'] != 'part_ev_rejection_eval_v1' or set(plan['donors']) != {'part', 'state'}:
        raise ValueError('Expected the independent ParT/State-EV validation pair')
    files = plan['donors']['part']['files']
    if (files != plan['donors']['state']['files']
            or len({r['name'] for r in files}) != FILES):
        raise ValueError('Validation populations differ or contain duplicate files')
    rows, identities = {}, {}
    for key in ('part', 'state'):
        rows[key], identities[key] = inspect_model(output, plan, key)
    if identities['part'] != identities['state']:
        raise ValueError('Event labels/order differ between models')
    lines = ['# Leptonic-channel QCD rejection', '',
             '| Signal | Signal efficiency | ParT rejection | State-EV rejection | Change (%) | QCD accepted P/S |',
             '|---|---:|---:|---:|---:|---:|']
    for name, eff in TARGETS.items():
        p, s = rows['part'][name], rows['state'][name]
        rp, rs = p['rejection'], s['rejection']
        change = '--' if rp is None or rs is None else f'{100 * (rs/rp-1):+.2f}'
        fmt = lambda r: 'ZB' if r is None else f'{r:,.2f}'
        label = 'Hlvqq (Hqql)' if name == 'Hqql' else 'Tblv (Tbl)'
        lines.append(f"| {label} | {100*eff:g}% | {fmt(rp)} | {fmt(rs)} | {change} | "
                     f"{p['qcd_accepted']}/{s['qcd_accepted']} |")
    lines += ['', 'Achieved signal efficiencies (ties included):']
    for name in TARGETS:
        p, s = rows['part'][name], rows['state'][name]
        lines.append(f"{name}: ParT {100*p['achieved_signal_efficiency']:.5f}%; "
                     f"State-EV {100*s['achieved_signal_efficiency']:.5f}%.")
    lines += ['', 'Positive change = higher rejection. ZB = zero QCD accepted, not measured infinity.',
              'Same saved, accuracy-selected checkpoints; single-seed full-5M validation, not independent test.',
              'Training runtimes differ: one GPU/native BatchNorm vs two GPUs/SyncBatchNorm.',
              'Read-only CPU post-processing. No jobs, models, ROOT data, checkpoints or reports changed.']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Existing evrej evaluation directory')
    args = parser.parse_args(argv)
    print(comparison(args.output))


if __name__ == '__main__':
    main()
