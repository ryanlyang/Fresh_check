from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "sbatch" / "reset_adaptive_binary_models_recovery_tigris.sh"
).read_text(encoding="utf-8")
PROBE_SUBMITTER = (
    ROOT / "sbatch" / "submit_adaptive_binary_runtime_batch_probes_tigris.sh"
).read_text(encoding="utf-8")
PROBE_WORKER = (
    ROOT / "scripts" / "probe_adaptive_binary_runtime_batch.py"
).read_text(encoding="utf-8")
RESUME_SCRIPT = (
    ROOT / "sbatch" / "resume_adaptive_binary_models_from_contracts_tigris.sh"
).read_text(encoding="utf-8")
CONTRACT_REPAIR_SCRIPT = (
    ROOT
    / "sbatch"
    / "repair_adaptive_binary_missing_contracts_and_resume_tigris.sh"
).read_text(encoding="utf-8")


def test_clean_model_recovery_requires_explicit_confirmation() -> None:
    assert "ABPH_CONFIRM_CLEAN_MODEL_RECOVERY" in SCRIPT
    assert "Set ABPH_CONFIRM_CLEAN_MODEL_RECOVERY=1" in SCRIPT


def test_clean_model_recovery_cancels_only_named_abph_jobs() -> None:
    assert "squeue --me" in SCRIPT
    assert "$2 ~ /^abph_/" in SCRIPT
    assert 'scancel "${stale_jobs[@]}"' in SCRIPT


def test_clean_model_recovery_preserves_preparation_and_rebuilds_contracts() -> None:
    assert "runtime_batch_measurements runtime_batch_contracts" in SCRIPT
    assert "submit_adaptive_binary_runtime_batch_probes_tigris.sh" in SCRIPT
    assert "ABPH_STAGE_MODE=models" in SCRIPT
    assert "retained: inputs, baselines, storage acceptance" in SCRIPT


def test_clean_model_recovery_gates_models_on_every_contract() -> None:
    assert """awk -F $'\\t' '$2 == "compile" {print $4}'""" in SCRIPT
    assert 'dependency="afterok:$(IFS=:; echo "${contract_jobs[*]}")"' in SCRIPT
    assert "--dependency=" in SCRIPT


def test_clean_model_recovery_rebuilds_missing_shared_targets_first() -> None:
    assert 'target_mode}" == "rank_local_build"' in SCRIPT
    assert "no shared target cache is required" in SCRIPT
    assert "model_train_exclusive_kt_adaptive_binary_targets_metadata.json" in SCRIPT
    assert "run_adaptive_binary_targets.sh\" cache" in SCRIPT
    assert "run_adaptive_binary_targets.sh\" preflight" in SCRIPT
    assert 'ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY="afterok:${target_preflight_job}"' in SCRIPT


def test_runtime_probes_accept_the_target_rebuild_dependency() -> None:
    assert "ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY" in PROBE_SUBMITTER
    assert '"--dependency=${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}"' in PROBE_SUBMITTER


def test_runtime_probes_consume_the_immutable_target_mode() -> None:
    assert "ABPH_TARGET_MODE_REPORT" in PROBE_SUBMITTER
    assert "export ABPH_TARGET_MODE_REPORT" in PROBE_SUBMITTER
    assert 'fresh_require_file "${ABPH_TARGET_MODE_REPORT}"' in PROBE_SUBMITTER
    assert 'root / "audits" / "target_mode_selection.json"' in PROBE_WORKER
    assert "target_mode_report=" in PROBE_WORKER
    assert 'offline_cache_dir=root / "inputs" / "offline_cache"' in PROBE_WORKER


def test_runtime_probe_default_excludes_single_gpu_oracle_references() -> None:
    assert "ABPH_TRAINED_RECONSTRUCTOR_VARIANTS" in PROBE_SUBMITTER
    assert "ABPH_RECONSTRUCTOR_VARIANTS" not in PROBE_SUBMITTER
    assert "ABPH_RENDERER_VARIANTS" not in PROBE_SUBMITTER


def test_full_probe_matrix_is_gated_by_one_rank_local_canary() -> None:
    assert "abph_clean_ranklocal_canary" in SCRIPT
    assert "B1_semantic_query_root root_hierarchy 64" in SCRIPT
    assert 'ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY="afterok:${canary_job}"' in SCRIPT
    canary_index = SCRIPT.index("abph_clean_ranklocal_canary")
    matrix_index = SCRIPT.index(
        'bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_runtime_batch_probes_tigris.sh"'
    )
    assert canary_index < matrix_index


def test_models_resume_preflights_before_cancelling_or_submitting() -> None:
    assert "ABPH_CONFIRM_MODELS_RESUME" in RESUME_SCRIPT
    assert "ABPH_STAGE_MODE=models" in RESUME_SCRIPT
    assert "ABPH_RECONSTRUCTOR_PARALLELISM=ddp8" in RESUME_SCRIPT
    preflight = RESUME_SCRIPT.index("DRY_RUN=1 bash")
    cancellation = RESUME_SCRIPT.index('scancel "${stale_jobs[@]}"')
    submission = RESUME_SCRIPT.index("DRY_RUN=0 bash")
    assert preflight < cancellation < submission


def test_models_resume_waits_for_inflight_required_contracts() -> None:
    assert "ABPH_TRAINED_RECONSTRUCTOR_VARIANTS" in RESUME_SCRIPT
    assert "missing_contract_variants" in RESUME_SCRIPT
    assert "abph_clean_model_recovery_contracts_*.tsv" in RESUME_SCRIPT
    assert "fresh_abph_models_resume" in RESUME_SCRIPT
    assert '--dependency="${dependency}"' in RESUME_SCRIPT
    assert "--export=ALL,ABPH_CONFIRM_MODELS_RESUME=1" in RESUME_SCRIPT
    assert "Models resume will continue automatically" in RESUME_SCRIPT


def test_models_resume_reuses_prepared_campaign_evidence() -> None:
    assert "abph_full_submission.json" in RESUME_SCRIPT
    assert "runtime_acceptance" in RESUME_SCRIPT
    assert "storage_projection" in RESUME_SCRIPT
    assert "tagger_acceptance" in RESUME_SCRIPT
    assert "unset ABPH_TAGGER_DDP_ACCEPTANCE_PATH" in RESUME_SCRIPT
    assert "retained: inputs, baselines, targets" in RESUME_SCRIPT


def test_contract_repair_replaces_stale_jobs_and_resumes_automatically() -> None:
    assert "ABPH_CONFIRM_CONTRACT_REPAIR" in CONTRACT_REPAIR_SCRIPT
    assert "ABPH_TRAINED_RECONSTRUCTOR_VARIANTS" in CONTRACT_REPAIR_SCRIPT
    assert "submit_adaptive_binary_runtime_batch_probes_tigris.sh" in (
        CONTRACT_REPAIR_SCRIPT
    )
    assert "fresh_abph_models_resume" in CONTRACT_REPAIR_SCRIPT
    assert "resume_adaptive_binary_models_from_contracts_tigris.sh" in (
        CONTRACT_REPAIR_SCRIPT
    )
    cancellation = CONTRACT_REPAIR_SCRIPT.index('scancel "${stale_jobs[@]}"')
    probes = CONTRACT_REPAIR_SCRIPT.index(
        "submit_adaptive_binary_runtime_batch_probes_tigris.sh"
    )
    continuation = CONTRACT_REPAIR_SCRIPT.index(
        "--job-name=fresh_abph_models_resume"
    )
    assert cancellation < probes < continuation


def test_contract_repair_archives_selected_immutable_evidence_before_probes() -> None:
    assert 'archive_root="${campaign_root}/archives/runtime_contract_repair_${stamp}"' in (
        CONTRACT_REPAIR_SCRIPT
    )
    assert "for variant in \"$@\"; do" in CONTRACT_REPAIR_SCRIPT
    assert (
        "for artifact_kind in runtime_batch_measurements runtime_batch_contracts"
        in CONTRACT_REPAIR_SCRIPT
    )
    assert 'source_path="${campaign_root}/${artifact_kind}/${variant}"' in (
        CONTRACT_REPAIR_SCRIPT
    )
    assert '[[ -d "${source_path}" && ! -L "${source_path}" ]]' in (
        CONTRACT_REPAIR_SCRIPT
    )
    assert 'mv -- "${source_path}" "${destination_path}"' in CONTRACT_REPAIR_SCRIPT

    archive = CONTRACT_REPAIR_SCRIPT.index('archive_root="${campaign_root}')
    probes = CONTRACT_REPAIR_SCRIPT.index(
        "submit_adaptive_binary_runtime_batch_probes_tigris.sh"
    )
    assert archive < probes
