"""Five-model full-validation report; no invented seed uncertainty or independent test claim."""
from pathlib import Path
import numpy as np
from part_ev_message_study.controls import validate_evaluation
from . import protocol as P
from .engine import audited_training, lock_finalists

CONTRASTS = (
    ('layerwise_minus_shared_state', 'state_layerwise', 'state_fast'),
    ('layerwise_state_minus_wide_part', 'state_layerwise', 'part_state_matched'),
    ('layerwise_state_minus_part', 'state_layerwise', 'part'),
    ('shared_state_minus_part', 'state_fast', 'part'),
    ('shared_state_minus_ev', 'state_fast', 'part_ev'),
    ('wide_part_minus_part', 'part_state_matched', 'part'),
    ('ev_minus_part', 'part_ev', 'part'),
)


def summarize(models):
    if (len(models) != 5 or any(r['task']['seed'] != 101 for r in models.values())
            or {r['task']['architecture'] for r in models.values()} != set(P.COUNTS)):
        raise ValueError('Expected exactly five different seed-101 architectures')
    by_arch = {r['task']['architecture']: r['metrics'] for r in models.values()}
    return {metric: {label: by_arch[left][metric]-by_arch[right][metric]
                    for label, left, right in CONTRASTS}
            for metric in ('accuracy', 'macro_auc_ovo', 'macro_auc_ovr', 'loss')}


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
            capacity_match=P.CONFIG, seeds=[101])
        payload['content_hash'] = P.digest(payload)
        directory = output / 'reports'
        P.write_json(directory / 'state_compare_report.json', payload)
        lines = ['# Full-schedule State-EV: bias and baseline capacity', '',
            'Seed 101 only. 1M updates, 50 checkpoint-validation checks.',
            'Full 5M validation, not independent testing; this population overlaps checkpoint selection.',
            'Single-seed differences do not establish reproducibility or statistical significance.',
            'Baseline ParT and ordinary EV retain their original training provenance.',
            'Wide ParT uses SHARED bias, no EV, no state branch, and width-601 particle FFNs.',
            'Its 2,327,894 parameters exceed layerwise State-EV (2,327,462) by 432 (0.0186%).',
            'Layerwise State-EV versus wide ParT changes bias design and message mechanism together;',
            'this is not an isolated state-conditioning contrast or a compute-matched comparison.', '',
            '| Model | Reused training | Parameters | Accuracy (%) | OVO AUC | OVR AUC | Check |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in models.values():
            t, m = row['task'], row['metrics']
            lines.append(f"| {t['run_id']} | {row['training_source']['reused']} | "
                f"{P.COUNTS[t['architecture']]:,} | {100*m['accuracy']:.4f} | {m['macro_auc_ovo']:.6f} | "
                f"{m['macro_auc_ovr']:.6f} | {row['best_check']} |")
        lines += ['', '## Seed-101 contrasts', '',
                  '| Contrast | Accuracy delta (pp) | OVO AUC delta |', '|---|---:|---:|']
        for key, _, _ in CONTRASTS:
            lines.append(f"| {key} | {100*effects['accuracy'][key]:+.4f} | "
                         f"{effects['macro_auc_ovo'][key]:+.6f} |")
        lines += ['', '## Learning curves (accuracy %)', '', '| Check | ' + ' | '.join(models) + ' |',
                  '|---:|' + '---:|'*len(models)]
        for i in range(50):
            lines.append(f'| {i+1} | ' + ' | '.join(
                f"{100*r['validation_checks'][i]['accuracy']:.4f}" for r in models.values()) + ' |')
        lines += ['', '## QCD rejection', '', 'ZB means zero QCD acceptances; counts remain in JSON.',
                  '', '| Signal / efficiency | ' + ' | '.join(models) + ' |', '|---|' + '---:|'*len(models)]
        for signal, targets in next(iter(models.values()))['metrics']['qcd_rejection'].items():
            for target in targets:
                values = [r['metrics']['qcd_rejection'][signal][target]['rejection'] for r in models.values()]
                lines.append(f'| {signal} / {100*float(target):g}% | ' +
                             ' | '.join('ZB' if v is None else f'{v:.2f}' for v in values) + ' |')
        lines += ['', '## Recorded cost', '',
                  '| Model | Training h | Checkpoint-validation h | AMP skips |', '|---|---:|---:|---:|']
        for key, row in models.items():
            t = row['training']
            lines.append(f"| {key} | {t['train_seconds']/3600:.2f} | "
                         f"{t['validation_seconds']/3600:.2f} | {t['amp_skipped_steps']} |")
        lines += ['', 'Timers exclude queue time, startup/replay and some checkpoint/diagnostic overhead.']
        (directory / 'state_compare_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print(directory / 'state_compare_report.md', flush=True)
        return payload
