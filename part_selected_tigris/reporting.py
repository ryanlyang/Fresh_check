"""Eight configurations, four direct selected-minus-standard contrasts."""
from pathlib import Path
import math
from . import protocol as P
from part_selected_screen.reporting import controls, contrasts
from part_screen_common.contracts import hashed
from part_screen_common.controls import selected_training, validate_predictions
from part_layerwise_screen.reporting import paired_accuracy


def report(output):
    output = Path(output)
    spec = P.load_campaign(output)
    with P.file_lock(output/'.report.lock', blocking=False):
        target_runtime = P.training_runtime(output, spec)
        registry = controls(output, spec)
        models = {name: dict(reused=True, parameters=row['training']['optimizer']['parameters'],
            training=row['training'], evaluation=row['evaluation']) for name, row in registry.items()}
        lock = P.read_json(output/'selection/locked_checkpoints.json')
        if lock.get('campaign_sha256') != spec['content_hash'] or len(lock['checkpoints']) != 4:
            raise ValueError('Selected checkpoint lock differs')
        for index, task in enumerate(spec['tasks']):
            run = P.task_directory(output, index)
            training = selected_training(run, task, spec['content_hash'],
                spec['recipes'][task['recipe']], target_runtime)
            if (lock['checkpoints'][index] != dict(task=task, checkpoint=training['checkpoint'], sha256=training['checkpoint_sha256'])
                    or training['optimizer']['parameters'] != P.COUNTS[task['architecture']]):
                raise ValueError('Checkpoint/optimizer identity changed')
            evaluation = validate_predictions(run, task, spec['content_hash'], training['checkpoint_sha256'],
                spec['inventory']['splits']['val']['files'])
            models[task['run_id']] = dict(reused=False, parameters=P.COUNTS[task['architecture']],
                                        training=training, evaluation=evaluation)
        labels = models['PART_RANGER']['evaluation']['labels_by_file']
        if any(r['evaluation']['labels_by_file'] != labels for r in models.values()):
            raise ValueError('Comparison event identities/labels differ')
        effects = {metric: contrasts({k: r['evaluation']['metrics'][metric] for k, r in models.items()})
                   for metric in ('accuracy', 'macro_auc_ovo', 'macro_auc_ovr', 'loss')}
        paired = {new: paired_accuracy(models[new]['evaluation'], models[old]['evaluation'])
                  for new, old in P.MATCHED.items()}
        for new, old in P.MATCHED.items():
            if not math.isclose(paired[new]['accuracy_difference'], effects['accuracy'][f'{new}_minus_{old}'], abs_tol=1e-12):
                raise ValueError('Paired accuracy disagrees with report metrics')
        result = hashed(dict(schema=P.REPORT_SCHEMA, campaign_sha256=spec['content_hash'],
            complete=True, population='official_validation_5M', single_seed_exploratory=True,
            hardware_transfer=spec['hardware_transfer'], target_runtime=target_runtime,
            donor_runtime=spec['donor']['runtime'], environment=spec['environment'],
            environment_differences_from_donor=spec['environment_differences_from_donor'],
            input_reference=spec['input_reference'],
            models=models, contrasts=effects, paired_accuracy=paired))
        destination = output/'reports/selected_tigris_report.json'
        if destination.exists() and P.read_json(destination) != result:
            raise ValueError('Completed comparison changed')
        P.write_json(destination, result)
        lines = ['# Selected relations: PT + TRACK + REGION', '',
            'Seed 101, same 200k-update Ranger recipe. Official 5M validation, not independent test results.',
            'Selected models: Tigris GH200 / ARM. Reused controls: SPORC A100 / x86.',
            'Selected-minus-standard differences are cross-hardware exploratory comparisons, not hardware-matched effects.',
            'The four selected models share one GH200 runtime; their internal EV/bias contrasts are hardware-matched.',
            'Four controls are reused read-only. Added encoders are not parameter-matched; selected pairs are directional.', '',
            '| Model | Parameters | Accuracy (%) | Macro OVO AUC | Best check |', '|---|---:|---:|---:|---:|']
        for name, row in models.items():
            m = row['evaluation']['metrics']
            lines.append(f"| {name} | {row['parameters']:,} | {100*m['accuracy']:.4f} | {m['macro_auc_ovo']:.6f} | {row['training']['best_check']} |")
        lines += ['', '## Contrasts (selected-minus-standard comparisons are cross-hardware)', '', '| Contrast | Accuracy delta (pp) | OVO AUC delta |', '|---|---:|---:|']
        lines += [f"| {k} | {100*v:+.4f} | {effects['macro_auc_ovo'][k]:+.6f} |" for k, v in effects['accuracy'].items()]
        lines += ['', '## QCD rejection', '', '| Signal / efficiency | '+' | '.join(models)+' |', '|---|'+'---:|'*8]
        for signal, targets in models['PART_RANGER']['evaluation']['metrics']['qcd_rejection'].items():
            for target in targets:
                values = [r['evaluation']['metrics']['qcd_rejection'][signal][target]['rejection'] for r in models.values()]
                lines.append(f'| {signal} / {100*float(target):g}% | '+' | '.join('ZB' if v is None else f'{v:.2f}' for v in values)+' |')
        lines += ['', 'ZB: zero QCD acceptances; counts retained in JSON. One-seed differences do not establish reproducibility.']
        # Atomic publication, so status readers never see a partial final report.
        import os, tempfile
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent, delete=False) as handle:
            handle.write('\n'.join(lines)+'\n')
            temporary = handle.name
        os.replace(temporary, destination.with_suffix('.md'))
        print(destination.with_suffix('.md'))
