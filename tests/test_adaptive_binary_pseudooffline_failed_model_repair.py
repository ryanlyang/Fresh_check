from __future__ import annotations

from pathlib import Path

from scripts.repair_adaptive_binary_failed_model_wave import (
    repaired_variant_command,
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
