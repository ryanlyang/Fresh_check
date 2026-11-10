"""Audited validation-only reporting and read-only progress."""
from pathlib import Path
from . import protocol as P
from .engine import audit


def ready(output,plan,indices):
    """A completion marker alone is never sufficient to skip a stage."""
    from part_screen_common.controls import validate_predictions
    jobs=[]
    for i in indices:
        task=plan['science']['tasks'][i]; run=P.run_directory(output,task)
        training=not (run/'training_complete.json').exists()
        if not training:
            trained=audit(output,plan,i)
        evaluated=(run/'full_validation/metrics.json').exists()
        if evaluated:
            if training: raise ValueError('Evaluation exists without completed training')
            validate_predictions(run,task,plan['content_hash'],trained['checkpoint_sha256'],
                plan['science']['inventory']['splits']['val']['files'])
        if training or not evaluated: jobs.append([i,training,True])
    return jobs


def report(output,plan):
    import numpy as np
    from part_screen_common.controls import validate_predictions
    from part_ev_fullscale.protocol import file_lock
    from scripts.run_part_ev_screen import classification_metrics
    output=Path(output)
    with file_lock(output/'.report.lock',blocking=False):
        rows=[]; identities=[]
        for i,task in enumerate(plan['science']['tasks']):
            trained=audit(output,plan,i)
            evaluated=validate_predictions(P.run_directory(output,task),task,plan['content_hash'],
                trained['checkpoint_sha256'],plan['science']['inventory']['splits']['val']['files'])
            logits=[]; labels=[]
            for record in evaluated['predictions']:
                with np.load(record['path'],allow_pickle=False) as arrays:
                    logits.append(arrays['logits']); labels.append(arrays['labels'])
            recomputed=classification_metrics(np.concatenate(logits),np.concatenate(labels),rejection=True)
            del logits,labels
            if recomputed != evaluated['metrics']: raise ValueError('Metrics differ from verified predictions')
            identities.append(evaluated['labels_by_file'])
            rows.append(dict(task=task,parameters=plan['science']['parameters'][task['architecture']],
                best_check=trained['best_check'],checks=trained['checks'],metrics=evaluated['metrics'],
                checkpoint_sha256=trained['checkpoint_sha256'],metrics_sha256=evaluated['metrics_sha256'],
                compute=trained['complete']['compute'],train_seconds=trained['complete']['train_seconds'],
                validation_seconds=trained['complete']['validation_seconds']))
        if identities[0] != identities[1]: raise ValueError('Arms evaluated different validation labels/entry ordering')
        actual=[r['compute']['dense_training_flops'] for r in rows]
        result=P.hashed(dict(schema=P.SCHEMA,campaign_sha256=plan['content_hash'],complete=True,
            population='official_validation_5M',independent_test=False,models=rows,
            contrast=dict(name='Compact State-EV minus matched ParT',
                accuracy_pp=100*(rows[0]['metrics']['accuracy']-rows[1]['metrics']['accuracy']),
                macro_auc_ovo=rows[0]['metrics']['macro_auc_ovo']-rows[1]['metrics']['macro_auc_ovo']),
            planned_compute=plan['review']['budget'],observed_matched_to_compact_dense_flop_ratio=actual[1]/actual[0],
            note='Single seed. Full 5M validation overlaps checkpoint selection; not independent test evidence. '
                 'Parameters and estimated dense training FLOPs are matched, not GPU hours or all arithmetic. '
                 'The control has more exposure; 50 selection opportunities each. No external control results imported.'))
        P.write(output/'reports/production_report.json',result)
        lines=['# Compact State-EV versus matched ParT','',result['note'],'',
            '| Model | Parameters | Updates | Accuracy (%) | Macro OVO AUC | Best check |',
            '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['task']['run_id']} | {r['parameters']:,} | {r['compute']['attempted_updates']:,} | "
                f"{100*r['metrics']['accuracy']:.4f} | {r['metrics']['macro_auc_ovo']:.6f} | {r['best_check']} |")
        c=result['contrast']
        lines += ['',f"Compact minus control: {c['accuracy_pp']:+.4f} percentage points; OVO AUC {c['macro_auc_ovo']:+.6f}.",
            '',f"Observed control / compact dense training FLOPs: {actual[1]/actual[0]:.6f}. "
            'Frozen update budgets are not adjusted after observing outcomes.','',
            '## QCD rejection','', 'ZB means zero accepted QCD events; JSON retains counts.','',
            '| Signal / efficiency | Compact State-EV | Matched ParT |','|---|---:|---:|']
        for signal,targets in rows[0]['metrics']['qcd_rejection'].items():
            for efficiency in targets:
                cells=[]
                for r in rows:
                    value=r['metrics']['qcd_rejection'][signal][efficiency]['rejection']
                    cells.append('ZB' if value is None else f'{value:.2f}')
                lines.append(f'| {signal} / {100*float(efficiency):g}% | '+ ' | '.join(cells)+' |')
        lines += ['', '## Training-time validation','',
            'Checks are equal fractions of each schedule, NOT equal updates or full dataset epochs.','',
            '| Check | Compact step | Compact accuracy | Control step | Control accuracy |',
            '|---:|---:|---:|---:|---:|']
        for a,b in zip(rows[0]['checks'],rows[1]['checks']):
            lines.append(f"| {a['check']} | {a['step']:,} | {100*a['accuracy']:.4f}% | {b['step']:,} | {100*b['accuracy']:.4f}% |")
        (output/'reports/production_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return result


def status(output,plan):
    rows=[]
    for task in plan['science']['tasks']:
        run=P.run_directory(output,task)
        checks=[P.read(p) for p in sorted((run/'validation').glob('check_*.json'))]
        best=max(checks,key=lambda r:r['accuracy']) if checks else None
        rows.append(dict(model=task['run_id'],recorded_checks=len(checks),
            best_accuracy=None if best is None else best['accuracy'],best_check=None if best is None else best['check'],
            training_marker=(run/'training_complete.json').is_file(),
            evaluation_marker=(run/'full_validation/metrics.json').is_file(),checks=checks))
    return dict(models=rows,note='Read-only progress, not completion approval. A crash can leave an uncommitted check.')
