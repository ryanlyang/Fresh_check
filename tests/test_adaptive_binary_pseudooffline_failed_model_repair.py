from __future__ import annotations

from pathlib import Path

from scripts.repair_adaptive_binary_failed_model_wave import (
    _drop_completed_afterok_dependencies,
    _reconcile_dependency_job,
    _replacement_jobs,
    repaired_variant_command,
)
from scripts.repair_adaptive_binary_runtime_batch_contract import _job_replacements
from scripts.repair_adaptive_binary_runtime_batch_contract import (
    replace_one_dependency_job,
)


def _row(variant: str) -> dict:
    return {
        "key": f"variant:{variant}",
        "command": [
            "sbatch",
            "--parsable",
            "--job-name=abph_reconstructor",
            "--nodes=8",
            "--ntasks=8",
            "--ntasks-per-node=1",
            "--gres=gpu:gh200:1",
            "--output=/dev/null",
            "--error=/dev/null",
            "--dependency=afterok:18842:18812",
            "/repo/sbatch/run_adaptive_binary_variant.sh",
            variant,
        ],
    }


def test_learned_variant_repair_preserves_distributed_topology() -> None:
    command = repaired_variant_command(
        _row("B2_semantic_query_probabilistic"),
        variant="B2_semantic_query_probabilistic",
        dependencies=("18832",),
        log_dir=Path("/logs"),
    )
    assert "--nodes=8" in command
    assert "--ntasks=8" in command
    assert "--dependency=afterok:18832" in command
    assert "--output=/logs/abph_repair_B2_semantic_query_probabilistic_%j.out" in command


def test_oracle_variant_repair_is_single_rank_and_drops_old_topology() -> None:
    command = repaired_variant_command(
        _row("B4_oracle_root_diagnostic"),
        variant="B4_oracle_root_diagnostic",
        dependencies=(),
        log_dir=Path("/logs"),
    )
    assert command.count("--nodes=1") == 1
    assert command.count("--ntasks=1") == 1
    assert "--nodes=8" not in command
    assert "--ntasks=8" not in command
    assert "--dependency=afterok:18842:18812" not in command


def test_interrupted_repair_can_resume_an_existing_replacement() -> None:
    assert _replacement_jobs(
        ["B2_semantic_query_probabilistic=19619"]
    ) == {"B2_semantic_query_probabilistic": "19619"}


def test_interrupted_repair_is_idempotent_for_already_rewired_consumers() -> None:
    assert _reconcile_dependency_job(
        "afterok:19416,afterok:19619",
        old_job_id="18932",
        new_job_id="19619",
    ) == ("afterok:19416,afterok:19619", False)
    assert _reconcile_dependency_job(
        "afterok:19416,afterok:18932",
        old_job_id="18932",
        new_job_id="19619",
    ) == ("afterok:19416,afterok:19619", True)


def test_completed_prerequisites_are_not_readded_to_pending_jobs() -> None:
    states = {
        "19416": "COMPLETED",
        "19619": "PENDING",
        "77": "RUNNING",
    }
    assert _drop_completed_afterok_dependencies(
        "afterok:19416,afterok:19619:77",
        state_for_job=states.__getitem__,
    ) == "afterok:19619:77"


def test_failed_model_wave_accepts_numeric_prerequisite_replacements() -> None:
    assert _job_replacements(["18932=19619"]) == {"18932": "19619"}


def test_failed_model_wave_can_supersede_a_failed_replacement() -> None:
    assert replace_one_dependency_job(
        "afterok:19618",
        old_job_ids=("18928", "19618"),
        new_job_id="19720",
    ) == "afterok:19720"
