from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH / name).read_text(encoding="utf-8")


def test_step8_submitter_queues_inputs_reconstructors_taggers_predictions_and_fusion():
    text = _read("submit_local_particle_residual_field_experiment.sh")

    assert "LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE" in text
    assert "fixed_hlt_v2_realistic" in text
    assert "LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH:=2.5" in text
    assert "LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE:=5000000" in text
    assert "LOCAL_RESIDUAL_FIELD_STACK_TRAIN_SIZE:=2000000" in text
    assert "run_build_fresh_splits.sh" in text
    assert "run_build_fresh_hlt_cache.sh" in text
    assert "run_cache_architecture_view_offline_inputs.sh" in text
    assert "run_cache_local_particle_residual_fields.sh" in text
    assert "run_train_local_residual_reconstructor.sh" in text
    assert "run_train_local_residual_field_tagger.sh" in text
    assert "run_predict_local_residual_field_tagger.sh" in text
    assert "run_local_residual_field_fusion.sh" in text
    assert "run_pd10_train_teacher.sh" in text
    assert "run_pd10_cache_teacher_logits.sh" in text
    assert "G0:A0,D5" in text
    assert "G1:D5,D5_seed1,D5_seed2,D5_seed3" in text
    assert "G2:D5,D6" in text
    assert "G3:E6,E5,E3" in text
    assert "LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS:=G0 G1 G2 G3" in text
    assert "LOCAL_RESIDUAL_FIELD_CACHE_SPLITS:=model_train model_val stack_train stack_val final_test" in text
    assert "LOCAL_RESIDUAL_FIELD_TARGET_SPLITS:=model_train model_val stack_val" in text
    assert "LOCAL_RESIDUAL_FIELD_OFFLINE_SPLITS:=${LOCAL_RESIDUAL_FIELD_TARGET_SPLITS}" in text
    assert "LOCAL_RESIDUAL_FIELD_SUBMIT_TEACHER_LOGITS:=1" in text
    assert "LOCAL_RESIDUAL_FIELD_TEACHER_LOGIT_SPLITS:=model_train model_val stack_val" in text


def test_step8_runner_scripts_call_expected_python_entrypoints():
    split_builder = _read("run_build_fresh_splits.sh")
    tagger_runner = _read("run_train_local_residual_field_tagger.sh")
    assert "scripts/cache_local_particle_residual_fields.py" in _read("run_cache_local_particle_residual_fields.sh")
    assert "scripts/train_local_residual_reconstructor.py" in _read("run_train_local_residual_reconstructor.sh")
    assert "scripts/train_local_residual_field_tagger.py" in tagger_runner
    assert "scripts/predict_local_residual_field_tagger.py" in _read("run_predict_local_residual_field_tagger.sh")
    assert "scripts/run_local_residual_field_fusion.py" in _read("run_local_residual_field_fusion.sh")
    assert "scripts/write_local_residual_field_report.py" in _read("run_write_local_residual_field_report.sh")
    assert "SKIP_UNREADABLE_ROOT_FILES" in split_builder
    assert "--skip-unreadable-files" in split_builder
    assert "LOCAL_RESIDUAL_FIELD_TAGGER_MIN_SELECTION_VALID_FRACTION:=0.99" in tagger_runner
    assert "--min-selection-valid-fraction" in tagger_runner
    assert "LOCAL_RESIDUAL_FIELD_TAGGER_RESIDUAL_CLIP_VALUE:=8.0" in tagger_runner
    assert "--residual-field-clip-value" in tagger_runner
    assert "LOCAL_RESIDUAL_FIELD_TARGET_DTYPE:=float16" in _read("run_cache_local_particle_residual_fields.sh")
    assert "LOCAL_RESIDUAL_FIELD_INCLUDE_FINAL_TEST_TARGETS:=0" in _read("run_cache_local_particle_residual_fields.sh")


def test_step8_predictions_and_fusion_use_stack_train_for_fitting():
    submitter = _read("submit_local_particle_residual_field_experiment.sh")
    predictor = _read("run_predict_local_residual_field_tagger.sh")
    fusion = _read("run_local_residual_field_fusion.sh")

    assert "LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS:=stack_train stack_val final_test" in submitter
    assert "LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS:=stack_train stack_val final_test" in predictor
    assert "LOCAL_RESIDUAL_FIELD_FUSION_SPLITS:=stack_train stack_val final_test" in fusion
    assert "LOCAL_RESIDUAL_FIELD_FUSION_FIT_SPLIT:=stack_train" in fusion


def test_step8_d5_recipe_requires_warm_start_and_teacher_logits():
    text = _read("run_train_local_residual_field_tagger.sh")

    assert "require_baseline_checkpoint" in text
    assert "require_kd_logits" in text
    assert 'A1)\n    field_source="hlt_only"\n    require_baseline_checkpoint' in text
    assert 'D5|D5_seed*)' in text
    assert 'echo "${RUN_ID} requires LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR' in text


def test_step8_tier_a_is_clean_hlt_controls_and_b0_is_one_radius_oracle():
    text = _read("run_train_local_residual_field_tagger.sh")

    assert 'A0)\n    field_source="hlt_only"' in text
    assert 'A1)\n    field_source="hlt_only"' in text
    assert 'A2)\n    field_source="hlt_only"' in text
    assert 'B0)\n    field_source="oracle"\n    field_subset=("r0p02.*")' in text
    assert 'F5) field_source="zero"; use_baseline_if_available ;;' in text
    assert 'if [[ "${field_source}" != "hlt_only" ]]; then' in text


def test_step8_submitter_preflights_baseline_kd_and_required_fusion_groups():
    text = _read("submit_local_particle_residual_field_experiment.sh")
    report = _read("run_write_local_residual_field_report.sh")
    runner = _read("run_train_local_residual_field_tagger.sh")

    assert "preflight_campaign_requirements" in text
    assert "LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=}" in text
    assert 'sbatch_args+=(--account="${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT}")' in text
    assert "tagger_requires_baseline" in text
    assert "tagger_requires_kd" in text
    assert "tagger_reconstructor_run_id" in text
    assert "report_required_recon_ids" in text
    assert 'fresh_split_words tagger_ids "${LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS}"' in text
    assert 'if [[ "${run_id}" == "D6" ]]; then' in text
    assert 'if [[ "${needs_d6}" -eq 1 && -n "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}" ]]; then' in text
    assert 'output+=("${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}")' in text
    assert "IFS=' '" in text
    assert 'printf \'%s\\n\' "${output[*]}"' in text
    assert 'LOCAL_RESIDUAL_FIELD_REQUIRED_RECON_RUN_IDS="$(report_required_recon_ids)"' in text
    assert "baseline_checkpoint_path_for_campaign" in text
    assert '${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/best_model_val.pt' in text
    assert 'fresh_require_file "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}"' in text
    assert "teacher_logits_complete" in text
    assert "internal_teacher_logits_will_be_built" in text
    assert 'PD10_TEACHER_SKIP_FINAL_TEST=1' in text
    assert 'PD10_TEACHER_LOGIT_SPLITS="${LOCAL_RESIDUAL_FIELD_TEACHER_LOGIT_SPLITS}"' in text
    assert '${split}_predictions.npz' in text
    assert 'offline_part_teacher_10class/${split}_predictions.npz' in text
    assert 'local_residual_offline_kd_teacher' in text
    assert 'local_residual_offline_kd_logits' in text
    assert 'LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID:=C5' in text
    assert '${reco_jobs[${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}]:-}' in text
    assert 'use_d6_checkpoint() { use_recon_checkpoint "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}"; }' in runner
    assert "--required-fusion-groups" in report


def test_step8_cluster_wrappers_keep_sporcsubmit_and_tigris_settings_explicit():
    sporc = _read("submit_local_particle_residual_field_sporcsubmit_pilot_and_highdata.sh")
    tigris = _read("submit_local_particle_residual_field_tigris_pilot_and_highdata.sh")

    assert "LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION:=tier3" in sporc
    assert "LOCAL_RESIDUAL_FIELD_GPU_GRES:=gpu:1" in sporc
    assert "LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE=pilot" in sporc
    assert "LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE=highdata" in sporc
    assert "LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION:=tigris" in tigris
    assert "LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=}" in tigris
    assert "export LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT" in tigris
    assert "LOCAL_RESIDUAL_FIELD_GPU_GRES:=gpu:gh200:1" in tigris
    assert "PD10_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data" in tigris
    assert "LOCAL_RESIDUAL_FIELD_DATA_DIR:=${PD10_DATA_DIR}" in tigris
    assert "OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints" in tigris
    assert "DEVICE:=cuda" in tigris
    assert "PYTHONNOUSERSITE:=1" in tigris
    assert "export CONDA_ENV CONDA_BASE PYTHONNOUSERSITE DEVICE" in tigris
    assert "SKIP_EXISTING:=0" in tigris
    assert "unset LOCAL_RESIDUAL_FIELD_ROOT" in tigris
    assert "CONDA_ENV:=atlas_kd_tigris" in tigris
    assert "CONDA_BASE:=/home/ryreu/miniforge3-aarch64" in tigris


def test_step8_hlt_cache_runner_passes_hlt_profile_to_builder():
    text = _read("run_build_fresh_hlt_cache.sh")
    assert ": \"${HLT_PROFILE:=fixed_hlt_v1}\"" in text
    assert "--hlt-profile" in text


def test_step8_pd10_teacher_logit_cache_only_requires_requested_offline_splits():
    text = _read("run_pd10_cache_teacher_logits.sh")
    assert 'fresh_split_words split_args "${PD10_TEACHER_LOGIT_SPLITS}"' in text
    assert 'for split in "${split_args[@]}"; do' in text
    assert '${PD10_OFFLINE_CACHE_DIR}/${split}_offline_metadata.json' in text
    assert 'final_test_offline_metadata.json' not in text
