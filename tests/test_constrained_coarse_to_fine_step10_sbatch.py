from __future__ import annotations

from pathlib import Path

from teacher_logit_reco.constrained_coarse_to_fine import (
    D4_UNCERTAINTY_GATED,
    EndToEndScheduleConfig,
    end_to_end_phase,
)


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH / name).read_text(encoding="utf-8")


RUNNERS = (
    "run_cache_constrained_coarse_to_fine_targets.sh",
    "run_train_constrained_coarse_to_fine_reconstructor.sh",
    "run_train_constrained_coarse_to_fine_tagger.sh",
    "run_cache_constrained_coarse_to_fine_predictions.sh",
    "run_alias_constrained_coarse_to_fine_predictions.sh",
    "run_alias_constrained_coarse_to_fine_tagger.sh",
    "run_constrained_coarse_to_fine_fusion.sh",
    "run_write_constrained_coarse_to_fine_report.sh",
)


def test_step10_workers_are_strict_slurm_jobs() -> None:
    for name in RUNNERS:
        text = _read(name)
        assert "#!/usr/bin/env bash" in text
        assert "#SBATCH --job-name=" in text
        assert "#SBATCH --output=fresh_check_logs/%x_%j.out" in text
        assert "#SBATCH --error=fresh_check_logs/%x_%j.err" in text
        assert "set -euo pipefail" in text
        assert "fresh_setup" in text
        assert "fresh_write_run_config" in text
        assert "fresh_run" in text


def test_step10_all_requested_queue_entry_points_exist() -> None:
    names = (
        "submit_constrained_coarse_to_fine_experiment.sh",
        "submit_constrained_coarse_to_fine_pilot.sh",
        "submit_constrained_coarse_to_fine_highdata.sh",
        "submit_constrained_coarse_to_fine_pilot_and_highdata.sh",
        "submit_constrained_coarse_to_fine_targets_only.sh",
        "submit_constrained_coarse_to_fine_reconstructors_only.sh",
        "submit_constrained_coarse_to_fine_taggers_only.sh",
        "submit_constrained_coarse_to_fine_depth_d5.sh",
        "submit_constrained_coarse_to_fine_d8_only.sh",
        "submit_constrained_coarse_to_fine_report_only.sh",
        "submit_constrained_coarse_to_fine_tigris_pilot_and_highdata.sh",
        "submit_constrained_coarse_to_fine_sporcsubmit_pilot_and_highdata.sh",
    )
    for name in names:
        assert (SBATCH / name).is_file(), name


def test_step10_graph_uses_hlt_v2_s2p5_and_protocol_sizes() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert "CONSTRAINED_C2F_HLT_PROFILE:=fixed_hlt_v2_realistic" in text
    assert "CONSTRAINED_C2F_HLT_DEGRADATION_STRENGTH:=2.5" in text
    assert "CONSTRAINED_C2F_MODEL_TRAIN_SIZE:=500000" in text
    assert "CONSTRAINED_C2F_MODEL_TRAIN_SIZE:=5000000" in text
    assert "CONSTRAINED_C2F_STACK_TRAIN_SIZE:=2000000" in text
    assert "CONSTRAINED_C2F_FINAL_TEST_SIZE:=1000000" in text


def test_step10_graph_queues_complete_reconstructor_and_depth_families() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert "B0 B1 B2 B3 B4 B5 B6 B7" in text
    assert "C0 C1 C2 C3 C4 C5 C6 C5-B1 C5-B2 C5-B3 C5-no-slot" in text
    assert "D0 D1 D2 D3 D4 D5 D5-B1 D5-B2 D5-B3 D6 D8" in text
    assert "E0 E1 E2 E3 E5" in text
    tagger = _read("run_train_constrained_coarse_to_fine_tagger.sh")
    assert "best_c=c5_b3" in tagger
    assert "stochastic_${index}@${index}" in tagger
    assert "c2f_tagger_D5-B3_alias" in text
    assert "run_alias_constrained_coarse_to_fine_tagger.sh" in text
    assert "run_alias_constrained_coarse_to_fine_predictions.sh" in text


def test_step10_submitter_is_dependency_aware_and_records_job_ids() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert '--dependency="afterok:${dependency}"' in text
    assert 'cache_dep="$(join_dependencies "${hlt_jid}" "${offline_jid}")"' in text
    assert "declare -A recon_jids" in text
    assert "declare -A tagger_jids" in text
    assert "job_ids.tsv" in text
    assert "source_status_hash" in text


def test_step10_reuse_and_posthoc_paths_fail_closed() -> None:
    text = _read("submit_constrained_coarse_to_fine_experiment.sh")
    assert "Required HLT cache is incomplete" in text
    assert "Required hierarchy target cache is incomplete" in text
    assert "requires incomplete reconstructor" in text
    assert "CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT" in text
    assert "F0-F5 fusion requires external ${external} predictions" in text
    assert "Report-only rerun requires complete predictions" in text
    assert "CONSTRAINED_C2F_SUBMIT_FUSION:=0" in text
    assert "CONSTRAINED_C2F_SUBMIT_REPORT:=0" in text


def test_step10_pair_is_concurrent_and_split_specific() -> None:
    text = _read("submit_constrained_coarse_to_fine_pilot_and_highdata.sh")
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=pilot" in text
    assert "CONSTRAINED_C2F_CAMPAIGN_MODE=highdata" in text
    assert "--dependency" not in text
    assert "CONSTRAINED_C2F_PILOT_HLT_WARM_START_CHECKPOINT" in text
    assert "CONSTRAINED_C2F_HIGHDATA_HLT_WARM_START_CHECKPOINT" in text


def test_step10_cluster_wrappers_set_correct_environments_and_resources() -> None:
    tigris = _read("submit_constrained_coarse_to_fine_tigris_pilot_and_highdata.sh")
    assert "atlas_kd_tigris" in tigris
    assert "miniforge3-aarch64" in tigris
    assert "gpu:gh200:1" in tigris
    assert "220G" in tigris
    sporc = _read("submit_constrained_coarse_to_fine_sporcsubmit_pilot_and_highdata.sh")
    assert "atlas_kd" in sporc
    assert "tier3" in sporc
    assert "gpu:1" in sporc
    assert "300G" in sporc


def test_step10_saves_only_selected_checkpoints_by_default() -> None:
    recon = _read("run_train_constrained_coarse_to_fine_reconstructor.sh")
    tagger = _read("run_train_constrained_coarse_to_fine_tagger.sh")
    assert "CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT:=0" in recon
    assert "CONSTRAINED_C2F_TAGGER_SAVE_LAST_CHECKPOINT:=0" in tagger
    assert "--no-save-last-checkpoint" in recon
    assert "--no-save-last-checkpoint" in tagger


def test_frozen_tagger_sweep_never_unfreezes_reconstructor() -> None:
    schedule = EndToEndScheduleConfig(
        frozen_reconstructor_epochs=1,
        terminal_decoder_epochs=1,
        upper_hierarchy_epochs=1,
    )
    for epoch in (0, 1, 2, 20):
        phase = end_to_end_phase(epoch, D4_UNCERTAINTY_GATED, schedule)
        assert phase.name == "frozen_reconstructor_tagger"
        assert not phase.terminal_decoder_trainable
        assert not phase.upper_hierarchy_trainable
