"""One evaluated candidate; no fabricated/reused-control contrast."""
from pathlib import Path
from part_ev_message_study.controls import validate_evaluation
from scripts.run_part_ev_screen import classification_metrics
from . import protocol as P
from .execution import engine


def report(output,spec):
    trained=engine().audit_training(output,spec,'masked')
    row,_=validate_evaluation(Path(output)/'production',spec,P.task('masked'),trained,recompute=classification_metrics)
    result=P.hashed(dict(complete=True,campaign_sha256=spec['content_hash'],launch_sha256=spec['launch_sha256'],
        model=row,parameters=spec['parameters']['masked'],trainable_parameters=spec['trainable_parameters']['masked'],
        preflight_sha256=spec['benchmark_sha256'],control_submitted=False,independent_test=False,
        note='Single seed 101; full 5M validation overlaps checkpoint selection. '
             'Execution-optimized masked-predictive State-EV; no accuracy-retention or matched-control claim.'))
    directory=Path(output)/'production/reports';P.write(directory/'fast256_report.json',result)
    metrics=row['metrics']
    lines=['# Masked-predictive fast256','',result['note'],'',
        f"Accuracy: {100*metrics['accuracy']:.4f}%",
        f"Macro OVO AUC: {metrics['macro_auc_ovo']:.6f}",
        f"Selected validation check: {row['best_check']}",'',
        'Complete rejection metrics, counts and provenance are retained in fast256_report.json.','']
    (directory/'fast256_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
