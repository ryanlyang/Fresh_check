from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_step12_submitters_cover_all_stages_and_dependency_gates() -> None:
    generic = (ROOT / "sbatch" / "submit_lprf_p7b_fusion_campaign.sh").read_text(encoding="utf-8")
    tigris = (ROOT / "sbatch" / "submit_lprf_p7b_fusion_campaign_tigris.sh").read_text(encoding="utf-8")
    common = (ROOT / "sbatch" / "common.sh").read_text(encoding="utf-8")
    for stage in ("preflight", "train_seed_control", "cache_stack", "fit_candidates", "select", "evaluate_final", "report", "full_campaign"):
        assert stage in generic
    assert "--dependency=afterok:" in generic
    assert "PRINT_ONLY" in generic or "fresh_is_dry_run" in generic
    assert "selected_fusion.json" in generic
    assert 'submit_final "${JOBS[selector]}"' in generic
    assert 'submit_report "${JOBS[final]}" "${JOBS[replay]}"' in generic
    assert "stability_plan.json" in generic
    assert "flock -n 9" in generic
    assert "submission_jobs.tsv" in generic
    assert "active_submission_job" in generic and "squeue -h -j" in generic
    assert "'%T|%r'" in generic
    assert "DependencyNeverSatisfied" in generic
    assert "action=cancel_and_resubmit" in generic and 'scancel "${job_id}"' in generic
    assert "normalize_dependency" in generic
    assert "dependency_chain_changed" in generic
    assert 'active_submission_job "${label}" "${completion}" "${dependency}"' in generic
    assert '"$(normalize_dependency "${dependency}")" >>"${SUBMISSION_MANIFEST}"' in generic
    assert "reu-aisocial" in generic and "reu-aisocial" in tigris
    assert "PYTHONNOUSERSITE=1" in tigris
    assert 'LPRF_FUSION_CONDA_BASE:=/home/ryreu/miniforge3-aarch64' in tigris
    assert 'LPRF_FUSION_CONDA_ENV:=atlas_kd_tigris' in tigris
    assert 'CONDA_BASE="${LPRF_FUSION_CONDA_BASE}"' in tigris
    assert 'CONDA_ENV="${LPRF_FUSION_CONDA_ENV}"' in tigris
    assert "export PROJECT_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_BASE CONDA_ENV DEVICE" in tigris
    assert 'source "${explicit_conda_sh}"' in common


def test_step12_every_new_tigris_job_uses_common_setup_and_full_account() -> None:
    names = (
        "run_fit_local_residual_field_fusion_candidate.sh",
        "run_plan_local_residual_field_fusion_stability.sh",
        "run_select_local_residual_field_fusion.sh",
        "run_replay_selected_local_residual_field_fusion_recipe.sh",
        "run_evaluate_selected_local_residual_field_fusion.sh",
        "run_benchmark_selected_local_residual_field_fusion.sh",
        "run_audit_selected_local_residual_field_fusion_bootstraps.sh",
        "run_write_local_residual_field_fusion_campaign_report.sh",
    )
    for name in names:
        text = (ROOT / "sbatch" / name).read_text(encoding="utf-8")
        assert "#SBATCH --account=reu-aisocial" in text
        assert "export PYTHONNOUSERSITE=1" in text
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "fresh_setup" in text
        assert "source /home/ryreu/miniconda3/condabin/conda" not in text


def test_step12_final_jobs_have_no_candidate_override() -> None:
    final = (ROOT / "sbatch" / "run_evaluate_selected_local_residual_field_fusion.sh").read_text(encoding="utf-8")
    runtime = (ROOT / "sbatch" / "run_benchmark_selected_local_residual_field_fusion.sh").read_text(encoding="utf-8")
    bootstrap = (ROOT / "sbatch" / "run_audit_selected_local_residual_field_fusion_bootstraps.sh").read_text(encoding="utf-8")
    assert "--selected-fusion" in final and "--candidate-id" not in final and "--group-id" not in final
    assert "--selected-fusion" in runtime and "--candidate-id" not in runtime and "--group-id" not in runtime
    assert "--selected-fusion" in bootstrap and "--candidate-id" not in bootstrap and "--group-id" not in bootstrap
