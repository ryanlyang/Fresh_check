from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"
SCRIPTS_DIR = REPO_ROOT / "scripts"

EXPECTED_RUN_IDS = (
    "A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 C4 C5 C6 "
    "D0 D1 D2 D3 D4 D5 E0 E1 E2 E3 E4 E5 E6 "
    "F0 F1 F2 F3 F4 Fseed Fshuffle G0 G1 G2 G3"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_step10_required_submitter_scripts_exist_and_use_common_runner_contracts() -> None:
    runner_names = (
        "run_canonical_state_audit_inputs.sh",
        "run_canonical_state_cache_phi.sh",
        "run_canonical_state_variant.sh",
        "run_write_canonical_state_report.sh",
    )
    submitter_names = (
        "submit_canonical_state_experiment.sh",
        "submit_canonical_state_pilot_and_highdata.sh",
    )
    for name in runner_names + submitter_names:
        assert (SBATCH_DIR / name).exists(), name
        text = _read(SBATCH_DIR / name)
        assert "#!/usr/bin/env bash" in text
        assert "set -euo pipefail" in text
        assert "fresh_prepare_submitter" in text or "fresh_setup" in text
    for name in runner_names:
        text = _read(SBATCH_DIR / name)
        assert "#SBATCH --job-name=" in text
        assert "#SBATCH --output=fresh_check_logs/%x_%j.out" in text
        assert "#SBATCH --error=fresh_check_logs/%x_%j.err" in text
        assert "fresh_write_run_config" in text
        assert "fresh_run" in text


def test_step10_report_cli_exists_and_requires_final_test_confirmation() -> None:
    writer = _read(SCRIPTS_DIR / "write_canonical_state_report.py")
    assert "CanonicalStateReportConfig" in writer
    assert "build_canonical_state_report" in writer
    assert "--confirm-final-test" in writer
    assert "--allow-missing-runs" in writer


def test_step10_variant_cli_has_real_backend_and_explicit_planning_stub() -> None:
    runner = _read(SCRIPTS_DIR / "train_canonical_state_variant.py")
    assert "--emit-planning-stub" in runner
    assert "not_scientific_output" in runner
    assert "CanonicalStateVariantRunConfig" in runner
    assert "run_canonical_state_variant" in runner
    assert "canonical_state_required_dependencies" in runner


def test_step10_submitter_has_pilot_highdata_modes_and_hlt_v2_strength_2p5() -> None:
    submitter = _read(SBATCH_DIR / "submit_canonical_state_experiment.sh")
    pair = _read(SBATCH_DIR / "submit_canonical_state_pilot_and_highdata.sh")

    assert "CANONICAL_STATE_CAMPAIGN_MODE:=highdata" in submitter
    assert "pilot)" in submitter
    assert "highdata)" in submitter
    assert "CANONICAL_STATE_MODEL_TRAIN_SIZE:=5000000" in submitter
    assert "CANONICAL_STATE_MODEL_VAL_SIZE:=1000000" in submitter
    assert "CANONICAL_STATE_STACK_TRAIN_SIZE:=3000000" in submitter
    assert "CANONICAL_STATE_STACK_VAL_SIZE:=1000000" in submitter
    assert "CANONICAL_STATE_FINAL_TEST_SIZE:=1000000" in submitter
    assert "CANONICAL_STATE_MODEL_TRAIN_SIZE=\"${CANONICAL_STATE_MODEL_TRAIN_SIZE}\"" in submitter
    assert "--max-train-jets" in _read(SBATCH_DIR / "run_canonical_state_variant.sh")
    assert "CANONICAL_STATE_HLT_PROFILE:=fixed_hlt_v2_realistic" in submitter
    assert "CANONICAL_STATE_HLT_DEGRADATION_STRENGTH:=2.5" in submitter
    assert "CANONICAL_STATE_DATA_DIR:=${PD10_DATA_DIR}" in submitter
    assert "PD10_HLT_PROFILE=\"${CANONICAL_STATE_HLT_PROFILE}\"" in submitter
    assert "PD10_HLT_DEGRADATION_STRENGTH=\"${CANONICAL_STATE_HLT_DEGRADATION_STRENGTH}\"" in submitter
    assert "PD10_DATA_DIR=\"${CANONICAL_STATE_DATA_DIR}\"" in submitter
    assert "CANONICAL_STATE_CAMPAIGN_MODE=pilot" in pair
    assert "CANONICAL_STATE_CAMPAIGN_MODE=highdata" in pair


def test_step10_submitter_enumerates_all_expected_jobs_and_dependencies() -> None:
    submitter = _read(SBATCH_DIR / "submit_canonical_state_experiment.sh")

    assert f"CANONICAL_STATE_RUN_IDS:={EXPECTED_RUN_IDS}" in submitter
    for run_id in EXPECTED_RUN_IDS.split():
        assert run_id in submitter
    assert "CANONICAL_STATE_SINGLE_RUN_IDS:=A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 C4 C5 C6 D0 D1 D2 D3 D4 D5 E0 E1 E2 E3 E4 E5 E6" in submitter
    assert "CANONICAL_STATE_FUSION_RUN_IDS:=F0 F1 F2 F3 F4 Fseed Fshuffle" in submitter
    assert "CANONICAL_STATE_ORACLE_RUN_IDS:=G0 G1 G2 G3" in submitter
    assert "afterok_args" in submitter
    assert "canonical_state_splits" in submitter
    assert "canonical_state_hlt_cache" in submitter
    assert "canonical_state_offline_cache" in submitter
    assert "canonical_state_phi_hlt" in submitter
    assert "canonical_state_phi_offline" in submitter
    assert "run_cache_architecture_view_offline_inputs.sh" in submitter
    assert "ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR=\"${CANONICAL_STATE_OFFLINE_CACHE_DIR}\"" in submitter
    assert "ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS=\"${CANONICAL_STATE_OFFLINE_SPLITS}\"" in submitter
    assert "canonical_state_report" in submitter
    assert "expected_jobs:" in submitter


def test_step10_reuse_and_skip_existing_paths_are_strict() -> None:
    submitter = _read(SBATCH_DIR / "submit_canonical_state_experiment.sh")

    assert "prequeue_validate_inputs" in submitter
    assert "scripts/audit_canonical_state_step1_inputs.py" in submitter
    assert "CANONICAL_STATE_PREQUEUE_VALIDATE_INPUTS:=1" in submitter
    assert "canonical_variant_output_complete" in submitter
    assert "archive_incomplete_variant_output" in submitter
    assert "SKIP_EXISTING" in submitter
    assert "found incomplete canonical-state output" in submitter


def test_step10_runners_wire_phi_cache_and_report_outputs() -> None:
    phi = _read(SBATCH_DIR / "run_canonical_state_cache_phi.sh")
    report = _read(SBATCH_DIR / "run_write_canonical_state_report.sh")

    assert "scripts/cache_canonical_state_phi.py" in phi
    assert "--source-view" in phi
    assert "CANONICAL_STATE_ALLOW_FINAL_TEST_OFFLINE_ORACLE" in phi
    assert "--expected-model-train" in phi
    assert "--expected-final-test" in phi
    assert "scripts/write_canonical_state_report.py" in report
    assert "CANONICAL_STATE_REPORT_RUN_IDS" in report
    assert "canonical_state_report.json" in report
    assert "tagging_metrics.csv" in report
    assert "fusion_comparison.csv" in report
