"""Paired capacity comparisons; ordinary EV remains seed-101-only."""
from pathlib import Path
import statistics
import numpy as np
from part_ev_message_study.controls import validate_evaluation
from . import protocol as P
from .engine import audited_training, lock_finalists


def summarize(models):
    by_key = {(r['task']['architecture'], r['task']['seed']): r['metrics'] for r in models.values()}
    result = {}
    for metric in ('accuracy', 'macro_auc_ovo', 'macro_auc_ovr', 'loss'):
        effects = {}
        for label, left, right in (
            ('state_minus_part', 'state_fast', 'part'),
            ('matched_minus_part', 'part_state_matched', 'part'),
            ('state_minus_matched', 'state_fast', 'part_state_matched'),
        ):
            values = [by_key[left, s][metric] - by_key[right, s][metric] for s in P.SEEDS]
            effects[label] = dict(seeds=list(P.SEEDS), values=values,
                                 mean=statistics.mean(values), sample_std=statistics.stdev(values))
        effects['state_minus_ev_seed101'] = by_key['state_fast', 101][metric] - by_key['part_ev', 101][metric]
        effects['ev_minus_part_seed101'] = by_key['part_ev', 101][metric] - by_key['part', 101][metric]
        result[metric] = effects
    return result


def report(output, spec, engine):
    output = Path(output)
    with P.file_lock(output / '.report.lock', blocking=False):
        P.require_approval(output, spec)
        lock_finalists(output, spec)
        models, common = {}, None
        for index, task in enumerate(spec['tasks']):
            root, source, trained = audited_training(output, spec, index)
            row, labels = validate_evaluation(output, spec, task, trained,
                                             recompute=engine.classification_metrics)
            if common is not None and not np.array_equal(common, labels):
                raise ValueError('Evaluation identities/labels differ')
            common = labels
            row['training_source'] = dict(root=str(root), campaign_sha256=source['content_hash'],
                                         reused=index not in P.TRAIN_INDICES)
            models[P.task_key(task)] = row
        effects = summarize(models)
        payload = dict(schema=P.SCHEMA, campaign_sha256=spec['content_hash'], models=models,
            contrasts=effects, population='official_validation_5M', independent_test=False,
            capacity_match=P.CONFIG, ordinary_ev_seeds=[101])
        payload['content_hash'] = P.digest(payload)
        directory = output / 'reports'
        P.write_json(directory / 'state_capacity_report.json', payload)
        lines = ['# Full-schedule all-eight State-EV and capacity-matched ParT', '',
            '1M updates, 50 checkpoint-validation checks. Full 5M validation, not independent testing.',
            'Baseline and ordinary-EV controls retain their original training provenance.',
            'Wide ParT: 2,323,750 parameters; State-EV: 2,323,710 (+40 for the control).',
            'Three matched seeds for State-EV, wide ParT and ParT; ordinary EV has seed 101 only.', '',
            '| Model | Seed | Reused training | Parameters | Accuracy (%) | OVO AUC | OVR AUC | Check |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
        for row in models.values():
            t, m = row['task'], row['metrics']
            lines.append(f"| {t['run_id']} | {t['seed']} | {row['training_source']['reused']} | "
                f"{P.COUNTS[t['architecture']]:,} | {100*m['accuracy']:.4f} | {m['macro_auc_ovo']:.6f} | "
                f"{m['macro_auc_ovr']:.6f} | {row['best_check']} |")
        lines += ['', '| Model | Accuracy (%), mean +/- sample SD | OVO AUC, mean +/- sample SD |',
                  '|---|---:|---:|']
        for arch in ('part', 'state_fast', 'part_state_matched'):
            metrics = [r['metrics'] for r in models.values() if r['task']['architecture'] == arch]
            acc, auc = ([m[k] for m in metrics] for k in ('accuracy', 'macro_auc_ovo'))
            lines.append(f"| {arch} | {100*statistics.mean(acc):.4f} +/- {100*statistics.stdev(acc):.4f} | "
                         f"{statistics.mean(auc):.6f} +/- {statistics.stdev(auc):.6f} |")
        lines += ['', '## Paired three-seed contrasts', '',
            'Sample SD describes seed variation, not a confidence interval.', '',
            '| Contrast | Accuracy delta (pp), mean +/- SD | OVO AUC delta, mean +/- SD |', '|---|---:|---:|']
        for key in ('state_minus_part', 'matched_minus_part', 'state_minus_matched'):
            a, b = effects['accuracy'][key], effects['macro_auc_ovo'][key]
            lines.append(f"| {key} | {100*a['mean']:+.4f} +/- {100*a['sample_std']:.4f} | "
                         f"{b['mean']:+.6f} +/- {b['sample_std']:.6f} |")
        for key in ('state_minus_ev_seed101', 'ev_minus_part_seed101'):
            lines.append(f"\nSeed 101 only: {key} = {100*effects['accuracy'][key]:+.4f} pp; "
                         f"OVO AUC {effects['macro_auc_ovo'][key]:+.6f}.")
        lines += ['', '## Learning curves (accuracy %)', '', '| Check | ' + ' | '.join(models) + ' |',
                  '|---:|' + '---:|'*len(models)]
        for i in range(50):
            lines.append(f'| {i+1} | ' + ' | '.join(
                f"{100*r['validation_checks'][i]['accuracy']:.4f}" for r in models.values()) + ' |')
        lines += ['', '## QCD rejection', '', 'ZB means zero QCD acceptances; counts are retained in JSON.',
                  '', '| Signal / efficiency | ' + ' | '.join(models) + ' |', '|---|' + '---:|'*len(models)]
        for signal, targets in next(iter(models.values()))['metrics']['qcd_rejection'].items():
            for target in targets:
                values = [r['metrics']['qcd_rejection'][signal][target]['rejection'] for r in models.values()]
                lines.append(f'| {signal} / {100*float(target):g}% | ' +
                             ' | '.join('ZB' if v is None else f'{v:.2f}' for v in values) + ' |')
        lines += ['', '## Recorded cost', '',
                  '| Model / seed | Training h | Checkpoint-validation h | AMP skips |',
                  '|---|---:|---:|---:|']
        for key, row in models.items():
            t = row['training']
            lines.append(f"| {key} | {t['train_seconds']/3600:.2f} | "
                         f"{t['validation_seconds']/3600:.2f} | {t['amp_skipped_steps']} |")
        lines += ['', 'Timers exclude queue time, startup/replay and some checkpoint/diagnostic overhead.']
        (directory / 'state_capacity_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print(directory / 'state_capacity_report.md', flush=True)
        return payload
