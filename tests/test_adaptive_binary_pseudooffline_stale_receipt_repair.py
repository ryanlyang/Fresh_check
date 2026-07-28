from __future__ import annotations

from pathlib import Path

from scripts.repair_adaptive_binary_stale_receipts import (
    repaired_receipt_command,
)


def test_stale_receipt_repair_preserves_worker_and_sets_live_dependency() -> None:
    command = repaired_receipt_command(
        {
            "key": "receipt:B0_pooled_mlp_root",
            "command": [
                "sbatch",
                "--parsable",
                "--job-name=abph_consumer_receipt",
                "--output=/dev/null",
                "--error=/dev/null",
                "--dependency=afterok:18928",
                "/repo/sbatch/run_adaptive_binary_consumer_receipt.sh",
                "B0_pooled_mlp_root",
            ],
        },
        old_job_id="18929",
        dependency="afterok:19618",
        log_dir=Path("/logs"),
    )
    assert "--dependency=afterok:19618" in command
    assert "--job-name=abph_repair_receipt_18929" in command
    assert "/repo/sbatch/run_adaptive_binary_consumer_receipt.sh" in command
    assert command[-1] == "B0_pooled_mlp_root"
