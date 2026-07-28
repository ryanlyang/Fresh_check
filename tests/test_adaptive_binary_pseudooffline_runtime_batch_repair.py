from __future__ import annotations

from scripts.repair_adaptive_binary_runtime_batch_contract import (
    _live_dependency,
    repaired_command,
    replace_dependency_job,
)


def test_runtime_batch_repair_replaces_only_failed_contract_dependency() -> None:
    assert (
        replace_dependency_job(
            "afterok:19091:18847:19092",
            old_job_id="18847",
            new_job_id="19123",
        )
        == "afterok:19091:19123:19092"
    )


def test_runtime_batch_repair_rejects_missing_or_duplicate_contract_reference() -> None:
    import pytest

    with pytest.raises(ValueError, match="found 0"):
        replace_dependency_job(
            "afterok:19091",
            old_job_id="18847",
            new_job_id="19123",
        )
    with pytest.raises(ValueError, match="found 2"):
        replace_dependency_job(
            "afterok:18847:18847",
            old_job_id="18847",
            new_job_id="19123",
        )


def test_runtime_batch_repair_preserves_probe_topology_and_removes_old_gate() -> None:
    command = repaired_command(
        {
            "key": "runtime_batch_probe:C0_direct_8_group_set:root_hierarchy:b64",
            "command": [
                "sbatch",
                "--parsable",
                "--nodes=8",
                "--ntasks=8",
                "--job-name=abph_runtime_batch_probe",
                "--output=/dev/null",
                "--error=/dev/null",
                "--dependency=afterok:18817",
                "/repo/sbatch/run_adaptive_binary_runtime_batch_probe.sh",
                "C0_direct_8_group_set",
                "root_hierarchy",
                "64",
            ],
        },
        label="C0_probe",
        dependencies=(),
        log_dir=None,
    )
    assert "--nodes=8" in command
    assert "--ntasks=8" in command
    assert "--dependency=afterok:18817" not in command
    assert "--job-name=abph_repair_C0_probe" in command
    assert command[-3:] == [
        "C0_direct_8_group_set",
        "root_hierarchy",
        "64",
    ]


def test_runtime_batch_repair_can_retain_small_failure_logs(tmp_path) -> None:
    command = repaired_command(
        {
            "key": "runtime_batch_probe:D5_oracle_groups_particles:renderer_distribution:b32",
            "command": [
                "sbatch",
                "--parsable",
                "--output=/dev/null",
                "--error=/dev/null",
                "/repo/sbatch/run_adaptive_binary_runtime_batch_probe.sh",
                "D5_oracle_groups_particles",
                "renderer_distribution",
                "32",
            ],
        },
        label="D5_probe",
        dependencies=(),
        log_dir=tmp_path,
    )
    assert f"--output={tmp_path}/abph_repair_D5_probe_%j.out" in command
    assert f"--error={tmp_path}/abph_repair_D5_probe_%j.err" in command


def test_live_dependency_strips_all_slurm_state_annotations(monkeypatch) -> None:
    class Completed:
        stdout = (
            "JobId=18938 "
            "Dependency=afterok:19416(failed),"
            "afterok:19619(unfulfilled),afterok:77(fulfilled)"
        )

    monkeypatch.setattr(
        "scripts.repair_adaptive_binary_runtime_batch_contract.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    assert _live_dependency("18938") == "afterok:19416,afterok:19619,afterok:77"
