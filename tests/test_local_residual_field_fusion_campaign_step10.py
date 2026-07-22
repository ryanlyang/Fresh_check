from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from jetclass_fresh.fusion import PredictionBlock
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.local_particle_residual_field import evaluate_selected_fusion_final
from teacher_logit_reco.local_particle_residual_field import fusion_final as final_module
from scripts.evaluate_selected_local_residual_field_fusion import parse_args
from tests.test_local_residual_field_fusion_campaign_step9 import _selection_fixture


def _final_block(member: str) -> PredictionBlock:
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    logits = np.zeros((len(labels), 10), dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 2.0
    identities = [JetIdentity(file="final.root", entry=index, label=int(label)) for index, label in enumerate(labels)]
    return PredictionBlock(
        model_name=member, split="final_test", logits=logits, probs=np.zeros_like(logits), labels=labels,
        jet_ids=identities,
        metadata={
            "dataset_metadata": {"alignment_report": {"source_manifest_hash": "m", "hlt_content_hash": "h"}},
            "runtime_inputs": "HLT_only", "deployable": True, "selection_allowed": False,
            "uses_true_fields": False, "uses_offline_particles": False, "uses_teacher_logits_at_runtime": False,
        },
    )


def test_step10_final_api_and_cli_accept_no_method_or_hyperparameter_override() -> None:
    assert tuple(inspect.signature(evaluate_selected_fusion_final).parameters) == ("selected_fusion_json",)
    parsed = parse_args(["--selected-fusion", "selected.json"])
    assert vars(parsed) == {"selected_fusion": "selected.json"}


def test_step10_locked_final_evaluation_uses_selection_and_writes_immutable_result(
    tmp_path: Path, monkeypatch,
) -> None:
    selected_path = _selection_fixture(tmp_path, monkeypatch)
    blocks = {member: _final_block(member) for member in ("A0", "A0_seed1", "P7b")}
    monkeypatch.setattr(
        final_module, "_validate_selection_dependencies",
        lambda _path, _selection: (
            {"split_manifest_hash": "m", "hlt_content_hashes": {"final_test": "h"}},
            {"manifest_hash": "d" * 64},
        ),
    )
    monkeypatch.setattr(final_module, "_development_feature_config", lambda _root, member: {"checkpoint": member})
    monkeypatch.setattr(
        final_module, "_load_or_cache_final_prediction",
        lambda *, member, **_kwargs: blocks[member],
    )
    monkeypatch.setattr(
        final_module, "paired_multiclass_bootstrap",
        lambda *_args, **_kwargs: {"replicates": 1000, "sampled_index_hash": "a" * 64},
    )
    monkeypatch.setattr(
        final_module, "paired_binary_projection_bootstrap",
        lambda *_args, signal_name, **_kwargs: {
            "replicates": 1000, "signal_name": signal_name, "sampled_index_hash": "b" * 64,
        },
    )
    report = evaluate_selected_fusion_final(selected_path)

    assert report["ok"] is True
    assert report["final_test_status"] == "exploratory_previously_partially_opened"
    assert len(report["selected_results"]) == 2
    assert all(row["candidate_id"] == "L0_mean_logits" for row in report["selected_results"])
    assert all(row["runtime_inputs"] == "HLT_only" and row["deployable"] for row in report["selected_results"])
    result_path = selected_path.parent.parent / "final_evaluation" / report["selected_fusion_artifact_hash"][:16] / "final_evaluation.json"
    assert result_path.is_file()


def test_step10_failed_final_evaluation_removes_current_and_abandoned_staging(
    tmp_path: Path, monkeypatch,
) -> None:
    selected_path = _selection_fixture(tmp_path, monkeypatch)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    final_parent = selected_path.parent.parent / "final_evaluation"
    final_parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{selected['artifact_hash'][:16]}.staging_"
    abandoned = final_parent / f"{prefix}abandoned"
    abandoned.mkdir()
    (abandoned / "large_partial_cache.bin").write_bytes(b"partial")
    monkeypatch.setattr(
        final_module, "_validate_selection_dependencies",
        lambda _path, _selection: ({"split_manifest_hash": "m", "hlt_content_hashes": {}}, {}),
    )
    monkeypatch.setattr(final_module, "_development_feature_config", lambda _root, member: {"checkpoint": member})
    monkeypatch.setattr(
        final_module, "_load_or_cache_final_prediction",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic final failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic final failure"):
        evaluate_selected_fusion_final(selected_path)
    assert list(final_parent.glob(f"{prefix}*")) == []


def test_step10_live_lock_preserves_concurrent_staging_directory(tmp_path: Path, monkeypatch) -> None:
    selected_path = _selection_fixture(tmp_path, monkeypatch)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    final_parent = selected_path.parent.parent / "final_evaluation"
    final_parent.mkdir(parents=True, exist_ok=True)
    artifact_prefix = selected["artifact_hash"][:16]
    live_staging = final_parent / f".{artifact_prefix}.staging_live_job"
    live_staging.mkdir()
    (live_staging / "active.bin").write_bytes(b"active")
    published_root = final_parent / artifact_prefix
    with final_module._final_evaluation_lock(published_root):
        with pytest.raises(RuntimeError, match="lock is already held"):
            evaluate_selected_fusion_final(selected_path)
    assert live_staging.is_dir()
    assert (live_staging / "active.bin").is_file()


def test_step10_advisory_lock_is_released_without_deleting_lock_file(tmp_path: Path) -> None:
    published_root = tmp_path / "final_evaluation" / "selection"
    lock_path = published_root.with_name(f".{published_root.name}.lock")

    with final_module._final_evaluation_lock(published_root):
        assert lock_path.is_file()
    with final_module._final_evaluation_lock(published_root):
        assert lock_path.is_file()


def test_step10_final_evaluation_uses_one_selection_snapshot(tmp_path: Path, monkeypatch) -> None:
    selected_path = _selection_fixture(tmp_path, monkeypatch)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    loads = []
    observed = {}

    def load_once(path):
        loads.append(path)
        return selected

    def evaluate_snapshot(path, *, selection, selection_sha256, staging_paths):
        observed.update(path=path, selection=selection, sha256=selection_sha256, staging=staging_paths)
        return {"ok": True}

    monkeypatch.setattr(final_module, "load_selected_fusion_set", load_once)
    monkeypatch.setattr(final_module, "_evaluate_selected_fusion_final_impl", evaluate_snapshot)

    assert evaluate_selected_fusion_final(selected_path) == {"ok": True}
    assert len(loads) == 1
    assert observed["selection"] is selected
    assert observed["sha256"] == final_module.sha256_file(selected_path)
