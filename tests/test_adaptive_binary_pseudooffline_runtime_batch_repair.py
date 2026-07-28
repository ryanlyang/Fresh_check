from __future__ import annotations

from scripts.repair_adaptive_binary_runtime_batch_contract import (
    _consumer_dependencies,
    _drop_completed_afterok_dependencies,
    _job_replacements,
    _live_dependency,
    _replace_dependency_if_present,
    repaired_command,
    replace_dependency_job,
    replace_one_dependency_job,
    update_pending_dependency,
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


def test_contract_repair_can_replace_a_companion_failed_model_dependency() -> None:
    assert _job_replacements(["18932=19619"]) == {"18932": "19619"}
    dependency = _replace_dependency_if_present(
        "afterok:19700,afterok:18932",
        old_job_id="18932",
        new_job_id="19619",
    )
    assert dependency == "afterok:19700,afterok:19619"


def test_contract_repair_replaces_a_failed_prior_repair() -> None:
    assert replace_one_dependency_job(
        "afterok:19617:18948",
        old_job_ids=("18922", "19617"),
        new_job_id="19700",
    ) == "afterok:19700:18948"


def test_failed_contract_consumer_is_rebuilt_against_new_contract(
    monkeypatch,
) -> None:
    states = {
        "18801": "COMPLETED",
        "18812": "COMPLETED",
        "19700": "PENDING",
    }
    monkeypatch.setattr(
        "scripts.repair_adaptive_binary_storage_acceptance_graph._slurm_job_state",
        states.__getitem__,
    )
    assert _consumer_dependencies(
        {
            "dependencies": ["18827", "18812", "18801"],
        },
        old_contract="18827",
        new_contract="19700",
        dry_run=False,
    ) == ["19700"]


def test_dependency_update_thaws_dependency_never_satisfied(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Completed:
        def __init__(self, returncode=0, stderr="") -> None:
            self.returncode = returncode
            self.stderr = stderr
            self.stdout = ""

    def run(command, **kwargs):
        command = tuple(command)
        calls.append(command)
        if command[:2] == ("scontrol", "update") and len(calls) == 1:
            return Completed(1, "Job dependency problem for job 18968")
        return Completed()

    monkeypatch.setattr(
        "scripts.repair_adaptive_binary_runtime_batch_contract.subprocess.run",
        run,
    )
    assert update_pending_dependency("18968", "afterok:19700") is True
    assert calls == [
        (
            "scontrol",
            "update",
            "JobId=18968",
            "Dependency=afterok:19700",
        ),
        ("scontrol", "requeuehold", "18968"),
        (
            "scontrol",
            "update",
            "JobId=18968",
            "Dependency=afterok:19700",
        ),
        ("scontrol", "release", "18968"),
    ]


def test_contract_repair_drops_completed_companion_dependencies() -> None:
    states = {"19416": "COMPLETED", "19700": "PENDING"}
    assert _drop_completed_afterok_dependencies(
        "afterok:19416,afterok:19700",
        state_for_job=states.__getitem__,
    ) == "afterok:19700"
