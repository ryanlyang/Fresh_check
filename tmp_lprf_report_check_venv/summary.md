# Local Particle Residual Field Report

- Contract: `local_particle_residual_field_report_v1`
- OK: `True`
- Problems: `0`

## Main Signals
- Best stack-val tagger: `F3` accuracy=0.7895666666666666 CE=0.5978446557156245
- Best final-test tagger: `D5_seed3` accuracy=0.7528333333333334 CE=0.6975854826016473
- Best stack-val reconstructor: `C6` MAE=0.045216287125349044 zero_MAE=0.07866805612246196
- Best final-test fusion: `G1/scalar_weighted_logit_mean` accuracy=0.75644

## Outputs
- `tagger_metrics_csv`: `tmp_lprf_report_check_venv\tagger_metrics.csv`
- `reconstructor_metrics_csv`: `tmp_lprf_report_check_venv\reconstructor_metrics.csv`
- `oracle_gap_csv`: `tmp_lprf_report_check_venv\oracle_gap.csv`
- `control_gap_csv`: `tmp_lprf_report_check_venv\control_gap.csv`
- `field_importance_csv`: `tmp_lprf_report_check_venv\field_importance.csv`
- `fusion_metrics_csv`: `tmp_lprf_report_check_venv\fusion_metrics.csv`
- `provenance_audit_json`: `tmp_lprf_report_check_venv\provenance_audit.json`
- `summary_md`: `tmp_lprf_report_check_venv\summary.md`
- `run_report_json`: `tmp_lprf_report_check_venv\run_report.json`
