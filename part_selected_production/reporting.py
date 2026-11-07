"""Validated five-million-jet results; no unverified external control contrasts."""
from pathlib import Path
from . import protocol as P
from .engine import audit


def report(output, plan):
    import numpy as np
    from part_screen_common.controls import validate_predictions
    from part_ev_fullscale.protocol import file_lock
    from scripts.run_part_ev_screen import classification_metrics
    output = Path(output)
    with file_lock(output/'.report.lock', blocking=False):
        rows = []
        population = plan['science']['inventory']['splits']['val']['files']
        for i, task in enumerate(plan['science']['tasks']):
            trained = audit(output, plan, i)
            run = P.run_directory(output, task)
            evaluated = validate_predictions(run, task, plan['content_hash'], trained['checkpoint_sha256'], population)
            logits, labels = [], []
            for record in evaluated['predictions']:
                with np.load(record['path'], allow_pickle=False) as arrays:
                    logits.append(arrays['logits']); labels.append(arrays['labels'])
            recomputed = classification_metrics(np.concatenate(logits), np.concatenate(labels), rejection=True)
            if recomputed != evaluated['metrics']:
                raise ValueError('Full-validation metrics differ from verified predictions')
            del logits, labels
            rows.append(dict(task=task, parameters=plan['science']['parameters'][task['architecture']],
                best_check=trained['best_check'], metrics=evaluated['metrics'],
                metrics_sha256=evaluated['metrics_sha256'], checkpoint_sha256=trained['checkpoint_sha256'],
                checks=trained['checks'], train_seconds=trained['complete']['train_seconds'],
                validation_seconds=trained['complete']['validation_seconds']))
        contrasts = []
        for name, a, b in (('State EV shared vs shared capacity control', 3, 0),
                           ('Selected State EV layerwise vs selected capacity control', 6, 2),
                           ('Selected State EV layerwise vs selected baseline', 6, 5)):
            contrasts.append(dict(name=name, accuracy_pp=100*(rows[a]['metrics']['accuracy']-rows[b]['metrics']['accuracy']),
                                  auc_ovo=rows[a]['metrics']['macro_auc_ovo']-rows[b]['metrics']['macro_auc_ovo']))
        controls = P.read(Path(plan['benchmark'])/'controls/registry.json')
        result = P.hashed(dict(schema=P.SCHEMA, campaign_sha256=plan['content_hash'], complete=True,
            population='official_validation_5M', independent_test=False, models=rows, contrasts=contrasts,
            external_controls=dict(status='pending_separate_compatibility_audit', registry=controls),
            note='Single seed; 5M validation overlaps checkpoint selection. No test-data access or automatic control reruns.'))
        P.write(output/'reports/production_report.json', result)
        lines = ['# Seven-run full-schedule Tigris comparison', '', result['note'], '',
                 '| Model | Parameters | Accuracy (%) | Macro OVO AUC | Selected check |',
                 '|---|---:|---:|---:|---:|']
        for row in rows:
            lines.append(f"| {row['task']['run_id']} | {row['parameters']:,} | {100*row['metrics']['accuracy']:.4f} | "
                         f"{row['metrics']['macro_auc_ovo']:.6f} | {row['best_check']} |")
        lines += ['', '## Internal contrasts', '', '| Contrast | Accuracy delta (pp) | OVO AUC delta |', '|---|---:|---:|']
        for row in contrasts:
            lines.append(f"| {row['name']} | {row['accuracy_pp']:+.4f} | {row['auc_ovo']:+.6f} |")
        lines += ['', 'Baseline ParT, ordinary EV, and existing standard layerwise State-EV remain external controls. '
                  'Their metadata registry is preserved in JSON; matched metric contrasts require a separate '
                  'checkpoint/population/recipe compatibility audit. No numerical results are silently reused.', '',
                  '## QCD rejection', '', 'ZB = zero accepted QCD events; JSON retains passing counts.', '',
                  '| Signal / efficiency | ' + ' | '.join(r['task']['run_id'] for r in rows) + ' |',
                  '|---|' + '---:|'*len(rows)]
        for signal, targets in rows[0]['metrics']['qcd_rejection'].items():
            for efficiency in targets:
                cells = []
                for row in rows:
                    rejection = row['metrics']['qcd_rejection'][signal][efficiency]['rejection']
                    cells.append('ZB' if rejection is None else f'{rejection:.2f}')
                lines.append(f'| {signal} / {100*float(efficiency):g}% | ' + ' | '.join(cells) + ' |')
        lines += ['', '## Training-time validation curves', '',
                  'Each check is 20,000 updates, not a dataset epoch. These use 1.28M events per check.', '',
                  '| Check | ' + ' | '.join(r['task']['run_id'] for r in rows) + ' |', '|---:|'+'---:|'*len(rows)]
        for i in range(50):
            lines.append(f'| {i+1} | '+' | '.join(f"{100*r['checks'][i]['accuracy']:.4f}%" for r in rows)+' |')
        (output/'reports/production_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        return result


def status(output, plan):
    rows = []
    for task in plan['science']['tasks']:
        run = P.run_directory(output, task)
        checks = [P.read(p) for p in sorted((run/'validation').glob('check_*.json'))]
        best = max(checks, key=lambda r:r['accuracy']) if checks else None
        rows.append(dict(model=task['run_id'], recorded_checks=len(checks),
            best_accuracy=None if best is None else best['accuracy'],
            best_check=None if best is None else best['check'],
            training_marker=(run/'training_complete.json').is_file(),
            evaluation_marker=(run/'full_validation/metrics.json').is_file()))
    return dict(models=rows, note='Read-only progress, not final integrity approval. A crash may leave an uncommitted validation record.')
