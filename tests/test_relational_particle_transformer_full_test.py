"""Focused tests for the supplemental inference-only official test workflow."""
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rpt_full_test_driver", ROOT / "scripts/evaluate_relational_part_offline_full_test.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_archive_exact_official_coverage_and_no_links(tmp_path):
    path = tmp_path / "test.tar"
    with tarfile.open(path, "w") as archive:
        for name in sorted(driver.NAMES):
            item = tarfile.TarInfo("test_20M/" + name)
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
    assert driver.inspect_archive(path)["files"] == 200
    with tarfile.open(path, "a") as archive:
        item = tarfile.TarInfo("test_20M/escape")
        item.type = tarfile.SYMTYPE
        item.linkname = "../../sensitive"
        archive.addfile(item)
    with pytest.raises(ValueError, match="Unsafe"):
        driver.inspect_archive(path)


@pytest.mark.parametrize("jobs", [1, 7, 20, 200])
def test_gpu_assignments_cover_each_file_once(jobs):
    assignments = driver.file_assignment(list(range(200)), jobs)
    flattened = sum(assignments, [])
    assert sorted(flattened) == list(range(200))
    assert max(map(len, assignments)) - min(map(len, assignments)) <= 1
    if jobs == 20:
        assert {len(a) for a in assignments} == {10}


@pytest.mark.parametrize("jobs", [0, -1, 201])
def test_invalid_gpu_count_rejected(jobs):
    with pytest.raises(ValueError):
        driver.file_assignment(list(range(200)), jobs)


def split_file(path, test_overlap=False):
    payload = {"max_constits": 128, "splits": {
        name: [{"file": "renamed/" + ("HToBB_100.root" if test_overlap else "HToBB_010.root"), "entry": i, "label": 1}]
        for i, name in enumerate(("model_train", "model_val", "stack_val", "final_test"))}}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_overlap_audit_survives_directory_relocation(tmp_path):
    path = tmp_path / "split.json.gz"
    split_file(path)
    assert driver.overlap_audit(path, driver.NAMES)["all_original_splits_file_disjoint"]
    split_file(path, test_overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        driver.overlap_audit(path, driver.NAMES)


def test_json_hash_compatible_and_immutable(tmp_path):
    from teacher_logit_reco.relational_part.contracts import with_content_hash
    value = {"contract": "test", "text": "unicode α", "count": 12}
    assert driver.hashed(value) == with_content_hash(value)
    path = tmp_path / "one.json"
    driver.write_json(path, driver.hashed(value))
    driver.write_json(path, driver.hashed(value))
    with pytest.raises(FileExistsError):
        driver.write_json(path, driver.hashed({"count": 13}))
    path.write_text('{"content_hash":"broken"}')
    with pytest.raises(ValueError, match="authentication"):
        driver.read_json(path)


def test_reusable_chunk_authenticates_bytes_lineage_and_all_models(tmp_path):
    path = tmp_path / "chunk.npz"
    metadata = {"entry_start": 0, "entry_stop": 3, "plan_sha256": "a" * 64}
    arrays = {f"logits_{i:02d}": np.ones((3, 10), dtype=np.float32) * i for i in range(12)}
    np.savez_compressed(path, **arrays)
    sidecar = driver.hashed({"metadata": metadata, "npz_sha256": driver.digest(path)})
    driver.write_json(path.with_suffix(".json"), sidecar)
    actual, _ = driver.read_chunk(path, metadata, keys={"logits_03"})
    np.testing.assert_array_equal(actual["logits_03"], arrays["logits_03"])
    with pytest.raises(ValueError, match="lineage"):
        driver.read_chunk(path, {**metadata, "plan_sha256": "b" * 64})
    with path.open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="bytes"):
        driver.read_chunk(path, metadata)


@pytest.mark.parametrize("values", [[1, 2, 3], [3, 1, 1, 9, 3], [0, 0, 0], []])
def test_vectorized_ranking_exactly_matches_original(values):
    from teacher_logit_reco.relational_part.evaluation import _average_ranks
    values = np.asarray(values, dtype=np.float64)
    np.testing.assert_array_equal(driver.average_ranks(values), _average_ranks(values))


def test_metrics_match_original_with_added_global_operating_points():
    from teacher_logit_reco.relational_part.evaluation import evaluate_logits, _average_ranks
    from teacher_logit_reco.relational_part import evaluation
    labels = np.tile(np.arange(10), 10)
    logits = np.random.default_rng(501).normal(size=(100, 10)).astype(np.float32)
    reference = evaluate_logits(logits, labels, split="official_test_20M")
    result = driver.calculate_metrics(logits, labels)
    for key in ("accuracy", "cross_entropy", "macro_per_class_accuracy", "ece_15_bin_top_label", "brier_score", "confusion_matrix", "one_vs_rest_auc"):
        assert result[key] == reference[key]
    for name in driver.CLASS_NAMES[1:]:
        for target in ("0.3", "0.5"):
            for key, value in reference["qcd_signal_rejection"][name][target].items():
                assert result["qcd_signal_rejection"][name][target][key] == value
        assert set(result["qcd_signal_rejection"][name]) == {str(t) for t in driver.TARGETS}
    assert evaluation._average_ranks is _average_ranks


def test_zero_background_reports_lower_bound_not_zero_or_finite_point():
    labels = np.tile(np.arange(10), 2)
    logits = np.eye(10, dtype=np.float32)[labels] * 20
    result = driver.calculate_metrics(logits, labels)
    row = result["qcd_signal_rejection"]["Hbb"]["0.3"]
    assert row["background_rejection"] is None
    assert row["background_rejection_is_infinite"]
    assert row["qcd_false_positive_count"] == 0
    assert row["conditional_rejection_wilson_95"][0] > 1
    assert row["conditional_rejection_wilson_95"][1] is None


def test_count_bootstrap_is_paired_balanced_and_deterministic():
    labels = np.tile(np.arange(10), 30)
    baseline = labels.copy()
    candidate = labels.copy()
    candidate[:30] = (candidate[:30] + 1) % 10
    baseline[30:45] = (baseline[30:45] + 1) % 10
    result = driver.paired_accuracy(candidate, baseline, labels)
    assert result == driver.paired_accuracy(candidate, baseline, labels)
    assert result["mean_accuracy_difference"] == pytest.approx(-0.05)
    same = driver.paired_accuracy(baseline, baseline, labels)
    assert same["confidence_interval_95"] == [0.0, 0.0]
    opposite = driver.paired_accuracy(baseline, candidate, labels)
    assert opposite["mean_accuracy_difference"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="balanced"):
        driver.paired_accuracy(candidate[:-1], baseline[:-1], labels[:-1])


def test_inference_reuses_preprocessing_and_isolates_model_input_mutation(monkeypatch):
    import awkward as ak
    import torch
    from teacher_logit_reco.relational_part import ca_tree
    from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, LABEL_BRANCHES
    n = 65  # Exercise full batch plus partial batch.
    arrays = {branch: ak.Array([[1.0, 2.0]] * n) for branch in PARTICLE_READ_BRANCHES}
    for i, branch in enumerate(LABEL_BRANCHES):
        arrays[branch] = np.full(n, int(i == 1))
    record = ak.Array(arrays)
    calls = []
    monkeypatch.setattr(ca_tree, "build_compiled_tree", lambda *args: calls.append(1) or {"dummy": True})
    class Model(torch.nn.Module):
        def forward(self, features, region_trees):
            result = features.sum((1, 2)).unsqueeze(1).expand(-1, 10).clone()
            features.zero_()  # Must not affect the next checkpoint.
            return result
    output = driver.infer_arrays(record, 1, [Model() for _ in range(12)], object(), torch.device("cpu"))
    assert len(calls) == n  # Not twelve times n.
    for values in output.values():
        assert values.shape == (65, 10)
        np.testing.assert_array_equal(values, output["logits_00"])
    wrong = ak.with_field(record, np.zeros(n), "label_Hbb")
    with pytest.raises(ValueError, match="Label branches"):
        driver.infer_arrays(wrong, 1, [Model()], object(), torch.device("cpu"))


@pytest.mark.parametrize("tensor_counter", [False, True])
def test_transient_trimmer_restored_at_each_chunk_without_changing_flags(tensor_counter):
    import torch
    model = torch.nn.Module()
    model.trimmer = torch.nn.Module()
    model.trimmer.enabled = True
    model.trimmer._counter = torch.tensor(0) if tensor_counter else 0
    driver.reset_transient_trimmers([model])
    model.trimmer._counter += 13
    driver.reset_transient_trimmers([model])
    assert int(model.trimmer._counter) == 0
    assert model.trimmer.enabled is True
    assert not model.state_dict()  # No persistent weights/buffers were added.


def test_submission_is_inference_only_and_pins_driver():
    text = (ROOT / "sbatch/submit_relational_part_offline_full_test.sh").read_text()
    assert '${RPT_FULL_TEST_JOBS:=20}' in text
    assert 'submit run' in text and 'submit aggregate' in text and 'submit report' in text
    assert 'submit train' not in text
    assert '--resume' in text
    assert '/driver.py' in text and 'worktree add --detach' in text
    assert '--kill-on-invalid-dep=yes' in text


@pytest.fixture
def shell_workspace():
    scratch = ROOT / ".tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rpt-full-test-shell-", dir=scratch) as name:
        yield Path(name)


def test_mock_slurm_submission_resume_and_live_graph_guard(shell_workspace):
    """Execute the real Bash submitter without SSH, Slurm, conda or GPUs."""
    if os.name == "nt":
        pytest.skip("POSIX shell/filesystem integration; Windows MSYS mkdir is incompatible with the managed sandbox")
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is not installed")
    tmp_path = shell_workspace
    def shell_path(path):
        value = path.as_posix()
        return "/" + value[0].lower() + value[2:] if os.name == "nt" else value
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "campaign_spec.json").write_text(json.dumps({"source": {"commit": "a" * 40}}))
    data, source = tmp_path / "data", tmp_path / "source"
    data.mkdir()
    source.mkdir()
    conda_dir = tmp_path / "conda/etc/profile.d"
    conda_dir.mkdir(parents=True)
    (conda_dir / "conda.sh").write_text("conda() { :; }\nexport CONDA_PREFIX='" + shell_path(tmp_path) + "'\n", newline="\n")
    output = tmp_path / "output"
    log = tmp_path / "sbatch.log"
    prefix = r'''
function sbatch() { printf '%s\n' "$*" >> "$MOCK_LOG"; echo 12345; }
function squeue() { if [[ "${MOCK_LIVE:-0}" == 1 ]]; then echo '12345'; fi; }
function flock() { :; }
function python() { "$MOCK_PYTHON" "$@"; }
export -f sbatch squeue flock python
'''
    env = dict(os.environ, RPT_FULL_TEST_PARENT=shell_path(parent),
               RPT_FULL_TEST_DATA=shell_path(data), RPT_FULL_TEST_SOURCE=shell_path(source),
               RPT_FULL_TEST_OUTPUT=shell_path(output), CONDA_BASE=shell_path(tmp_path / "conda"),
               MOCK_LOG=shell_path(log), MOCK_PYTHON=shell_path(Path(sys.executable)))
    command = prefix + '\nsource "' + shell_path(ROOT / "sbatch/submit_relational_part_offline_full_test.sh") + '"'
    first = subprocess.run([bash, "-c", command], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stdout + first.stderr
    submitted = log.read_text().splitlines()
    assert len(submitted) == 4
    assert "--array=0-19%8" in submitted[1]
    assert "--dependency=afterok:12345" in submitted[1]
    assert "--array=0-11%3" in submitted[2]
    assert (output / "driver.py").read_bytes() == (ROOT / "scripts/evaluate_relational_part_offline_full_test.py").read_bytes()
    # Retry works even when the first preparation job died before creating a plan.
    resume = command + ' --resume "' + shell_path(output) + '"'
    retry = subprocess.run([bash, "-c", resume], env=env, text=True, capture_output=True)
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert len(log.read_text().splitlines()) == 8
    # A live old graph is never duplicated.
    live = subprocess.run([bash, "-c", resume], env={**env, "MOCK_LIVE": "1"}, text=True, capture_output=True)
    assert live.returncode != 0
    assert "still queued/running" in live.stderr
    assert len(log.read_text().splitlines()) == 8
