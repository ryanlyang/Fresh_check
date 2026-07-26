# Local Residual Field P7b High-Data Seed Study

This campaign tests whether the validated P7b advantage over A0 persists when
the deployable models train on three million jets.

## Locked design

- Training: 3,000,000 jets
- Checkpoint-selection `model_val`: 250,000 jets
- Locked comparison `stack_val`: 500,000 jets
- Sealed final test: 1,000,000 jets
- Matched seeds: `20421`, `20522`, `20623`
- HLT profile: `fixed_hlt_v2_realistic`, degradation strength `2.5`
- A0: from scratch, HLT-only
- P7b: the already selected `Orobust_light`, alpha endpoint `1.0` recipe
- One 3M C0 predictor and one 3M Orobust consumer are frozen and shared by
  the three P7b student seeds.
- P7b checkpoints are deployable and require only HLT particles at inference.

No `stack_train` allocation is created because this is a model comparison,
not a learned-fusion campaign. Offline and residual-field caches are built
only for `model_train`, `model_val`, and `stack_val`. No privileged
`final_test` cache is allowed.

## Submit validation campaign on Tigris

Choose a stable campaign ID and use the same ID for every resume:

```bash
cd /home/ryreu/atlas/Fresh_check
bash sbatch/submit_lprf_p7b_high_data_seed_study_tigris.sh \
  full_validation p7b_highdata3m_20260726
```

The submitter is completion-aware and can be rerun with the same command. It
queues or reuses:

1. Split manifest
2. HLT cache, including the sealed HLT-only final-test representation
3. Offline cache without final test
4. Residual-field targets without final test
5. Shared C0, A0 seed `20421`, and Orobust source training
6. Orobust registration and immutable study preflight
7. A0 seeds `20522` and `20623`
8. P7b seeds `20421`, `20522`, and `20623`
9. Validation-only matched-seed report

The primary result is:

```text
checkpoints/local_particle_residual_field_high_data_seed_study/
  p7b_highdata3m_20260726/validation_report/run_report.json
```

`full_validation` never queues final-test predictions.

## Confirm final test

Review and freeze the validation report first. Only then run:

```bash
cd /home/ryreu/atlas/Fresh_check
CONFIRM_FINAL_TEST=1 \
  bash sbatch/submit_lprf_p7b_high_data_seed_study_tigris.sh \
  final_test p7b_highdata3m_20260726
```

This evaluates all six locked deployable checkpoints on the same one-million
jet final test. The prediction audit requires:

- `deployable=true`
- HLT-only allowed inputs
- no target fields
- no teacher logits
- exactly 1,000,000 jets
- checkpoint hashes matching the frozen validation runs

The final artifact is:

```text
checkpoints/local_particle_residual_field_high_data_seed_study/
  p7b_highdata3m_20260726/final_report/run_report.json
```

## Useful resume stages

```bash
bash sbatch/submit_lprf_p7b_high_data_seed_study_tigris.sh prepare CAMPAIGN_ID
bash sbatch/submit_lprf_p7b_high_data_seed_study_tigris.sh train CAMPAIGN_ID
bash sbatch/submit_lprf_p7b_high_data_seed_study_tigris.sh validation_report CAMPAIGN_ID
```

The default source-data root is
`/home/ryreu/atlas/PracticeTagging/data`. The submitter requires at least
50 GiB free before a fresh cache build; override
`LOCAL_RESIDUAL_FIELD_HIGH_DATA_MIN_FREE_GIB` only after checking the projected
cache footprint.
