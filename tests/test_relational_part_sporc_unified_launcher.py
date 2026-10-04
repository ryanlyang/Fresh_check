"""Exercise the real unified launcher with fake scheduler/Conda executables."""
import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("shared_sporc_launcher_tests", Path(__file__).with_name("test_relational_part_sporc_launcher.py"))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
launcher, fresh_args, shell_path = _shared.launcher, _shared.fresh_args, _shared.shell_path

pytestmark = pytest.mark.parametrize("launcher", ["unified"], indirect=True)


def test_tier3_unified_graph_is_one_validation_then_24_metric_tasks(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output))
    assert result.returncode == 0, result.stderr
    records = calls()
    assert [r["phase"] for r in records] == ["build", "validate", "infer", "aggregate", "report"]
    assert all("--partition=tier3" in r["arguments"] for r in records)
    assert "--array=0-39%4" in records[2]["arguments"]
    assert "--array=0-23%2" in records[3]["arguments"]
    for i, row in enumerate(records[1:], 1):
        assert f"--dependency=afterok:{700000+i}" in row["arguments"]
        assert "--kill-on-invalid-dep=yes" in row["arguments"]
    assert (output / "unified_campaign.json").is_file()
    assert not (output / "sporc_campaign.json").exists()


def test_dry_run_does_not_create_or_submit_and_partition_is_explicit(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), "--partition", "sporc", "--jobs", "20", "--concurrency", "3", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--partition=sporc" in result.stderr
    assert "--array=0-19%3" in result.stderr and "--array=0-23%2" in result.stderr
    assert not output.exists() and not calls()


def test_source_job_discovery_keeps_original_tigris_graph_untouched(launcher):
    run, source, output, calls = launcher
    (source / "submitted_job_ids.txt").write_text("infer 207556\naggregate 207557\nreport 207558\n")
    result = run("--source-job", "207556", "--search-root", shell_path(source.parent), "--output", shell_path(output))
    assert result.returncode == 0, result.stderr
    assert len(calls()) == 5
    assert (source / "submitted_job_ids.txt").read_text().startswith("infer 207556\n")


def test_resume_rejects_live_jobs_and_legacy_migration_manifest(launcher):
    run, source, output, calls = launcher
    assert run(*fresh_args(source, output)).returncode == 0
    live = run("--resume", shell_path(output), extra_env={"RPT_TEST_SQUEUE": "700003_0|unified"})
    assert live.returncode != 0 and len(calls()) == 5
    (output / "unified_campaign.json").rename(output / "sporc_campaign.json")
    old = run("--resume", shell_path(output))
    assert old.returncode != 0 and len(calls()) == 5


def test_aggregate_limit_updated_to_24_and_phase_cannot_be_inherited(launcher):
    run, source, output, calls = launcher
    result = run(*fresh_args(source, output), extra_env={"RPT_UNIFIED_PHASE": "wrong",
                  "RPT_UNIFIED_AGGREGATE_CONCURRENCY": "24", "SBATCH_DEPENDENCY": "afterok:123"})
    assert result.returncode == 0, result.stderr
    assert calls()[0]["phase"] == "build"
    assert not any(a.startswith("--dependency") for a in calls()[0]["arguments"])
    assert "--array=0-23%24" in calls()[3]["arguments"]
