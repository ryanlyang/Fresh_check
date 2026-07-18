from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_RANK_LOCAL_TARGET_MODE,
    ABPH_SHARED_TRANSIENT_TARGET_MODE,
    ABPH_TARGET_MODE_SELECTION_CONTRACT,
    AdaptiveBinaryTargetBatchSource,
    RamCapacityProbe,
    RankLocalWorkspace,
    build_adaptive_binary_targets,
    rank_local_target_metadata,
    select_target_mode,
    write_target_mode_selection,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.config import canonical_hash
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_STORAGE_PROJECTION_CONTRACT,
)


def _probe(root: Path, *, free: int = 10_000_000) -> RamCapacityProbe:
    return RamCapacityProbe(
        root=root,
        filesystem_type="tmpfs",
        filesystem_free_bytes=free,
        cgroup_limit_bytes=free,
        cgroup_current_bytes=0,
        slurm_allocation_bytes=free,
    )


def _measurement() -> dict:
    payload = {
        "contract": "adaptive_binary_real_target_sample_v1",
        "manifest_hash": "manifest",
        "hlt_content_hash": "hlt-train",
        "offline_content_hash": "offline-train",
        "sample_jet_identity_hash": "sample-identities",
        "data_dir": "/dataset",
        "groupings": {
            grouping: {
                "n_jets": 100,
                "stored_bytes_per_jet": 100.0,
                "logical_bytes_per_jet": 1000.0,
            }
            for grouping in ("exclusive_kt", "cambridge_aachen")
        },
    }
    payload["content_hash"] = canonical_hash(payload)
    return payload


def _projection(non_target_bytes: int) -> dict:
    return {
        "contract": ABPH_STORAGE_PROJECTION_CONTRACT,
        "content_hash": "projection",
        "rows": [
            {
                "artifact_family": "selected_models",
                "expected_bytes": non_target_bytes,
                "atomic_write_overhead_bytes": 0,
                "active_from_wave": 0,
                "active_through_wave": 1,
            }
        ],
    }


def _split_provenance() -> dict[str, dict[str, str]]:
    return {
        "model_train": {
            "source_manifest_hash": "manifest",
            "hlt_content_hash": "hlt-train",
            "offline_content_hash": "offline-train",
            "jet_identity_hash": "ids-train",
        },
        "model_val": {
            "source_manifest_hash": "manifest",
            "hlt_content_hash": "hlt-val",
            "offline_content_hash": "offline-val",
            "jet_identity_hash": "ids-val",
        },
    }


def test_rank_workspace_reservations_preserve_headroom_and_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "abph-123-2"
    workspace = RankLocalWorkspace(
        root, job_id="123", rank=2, probe=_probe(tmp_path)
    )
    assert workspace.reservation_limit_bytes == 8_000_000
    if os.name != "nt":
        assert (root.stat().st_mode & 0o777) == 0o700
    for name in ("inputs", "targets", "pseudo", "checkpoints", "codec_scratch"):
        assert (root / name).is_dir()

    reservation = workspace.reserve(owner="rank2", role="target_slice", expected_bytes=3_000_000)
    stage = root / "targets" / "stage"
    stage.mkdir()
    (stage / "payload.bin").write_bytes(b"x" * 2_000_000)
    assert workspace.commit_tree(reservation, stage) == 2_000_000
    assert workspace.reserved_bytes == 2_000_000
    with pytest.raises(ValueError, match="escapes owned workspace"):
        workspace.commit_tree(reservation, tmp_path / "unrelated")
    with pytest.raises(MemoryError, match="20% headroom"):
        workspace.reserve(owner="rank2", role="too_large", expected_bytes=6_000_001)
    workspace.release(reservation)
    workspace.cleanup(require_empty=True)
    assert not root.exists()


def test_workspace_headroom_is_relative_to_the_full_cgroup_limit(tmp_path: Path) -> None:
    probe = RamCapacityProbe(
        root=tmp_path,
        filesystem_type="tmpfs",
        filesystem_free_bytes=1_000,
        cgroup_limit_bytes=1_000,
        cgroup_current_bytes=300,
        slurm_allocation_bytes=1_000,
    )
    # 20% of the original allocation remains free, not merely 20% of the
    # post-allocation remainder.
    assert probe.reservation_limit_bytes == 500


def test_target_mode_is_measured_and_immutable(tmp_path: Path) -> None:
    shared = select_target_mode(
        campaign_root=tmp_path,
        campaign_mode="pilot",
        split_sizes={"model_train": 1000, "model_val": 200},
        measurement=_measurement(),
        storage_projection=_projection(2_000_000),
        workspace_capacity={"reservation_limit_bytes": 8_000_000},
        hlt_cache_bytes=1_000_000,
        current_campaign_bytes=1_500_000,
        target_chunk_size=512,
        source_provenance_by_split=_split_provenance(),
    )
    assert shared["selected_mode"] == ABPH_SHARED_TRANSIENT_TARGET_MODE

    rank_local = select_target_mode(
        campaign_root=tmp_path,
        campaign_mode="highdata",
        split_sizes={"model_train": 5_000_000, "model_val": 1_000_000},
        measurement=_measurement(),
        storage_projection=_projection(23_500_000_000),
        workspace_capacity={"reservation_limit_bytes": 8_000_000},
        hlt_cache_bytes=1_000_000,
        current_campaign_bytes=1_500_000,
        target_chunk_size=512,
        source_provenance_by_split=_split_provenance(),
    )
    assert rank_local["selected_mode"] == ABPH_RANK_LOCAL_TARGET_MODE
    path = tmp_path / "selection.json"
    write_target_mode_selection(path, rank_local)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_mode"] = ABPH_SHARED_TRANSIENT_TARGET_MODE
    path.write_text(json.dumps(payload), encoding="utf-8")
    from teacher_logit_reco.adaptive_binary_pseudooffline import load_target_mode_selection

    with pytest.raises(ValueError, match="content hash"):
        load_target_mode_selection(path, campaign_root=tmp_path)


def test_rank_local_metadata_is_split_bound() -> None:
    selection = _measurement()
    selection.update(
        {
            "contract": ABPH_TARGET_MODE_SELECTION_CONTRACT,
            "selected_mode": ABPH_RANK_LOCAL_TARGET_MODE,
            "target_chunk_size": 2,
            "source_manifest_hash": "manifest",
            "content_hash": "selection",
            "source_provenance_by_split": {
                "model_val": {
                    "source_manifest_hash": "manifest",
                    "hlt_content_hash": "hlt-val",
                    "offline_content_hash": "offline-val",
                    "jet_identity_hash": "ids-val",
                }
            },
        }
    )
    metadata = rank_local_target_metadata(
        selection=selection,
        split="model_val",
        grouping="exclusive_kt",
        n_jets=5,
        jet_identity_hash_value="ids-val",
    )
    assert metadata["hlt_content_hash"] == "hlt-val"
    assert metadata["offline_content_hash"] == "offline-val"
    assert [(row["start"], row["stop"]) for row in metadata["shards"]] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]


def test_rank_local_source_builds_only_planned_slices(tmp_path: Path, monkeypatch) -> None:
    import teacher_logit_reco.adaptive_binary_pseudooffline.production as production

    root = tmp_path / "campaign"
    (root / "targets").mkdir(parents=True)
    hlt_dir = root / "inputs" / "hlt_cache"
    hlt_dir.mkdir(parents=True)
    offline_dir = root / "inputs" / "offline_cache"
    offline_dir.mkdir(parents=True)
    (root / "inputs" / "split_manifest").mkdir(parents=True)
    manifest_path = root / "inputs" / "split_manifest" / "split_manifest.json.gz"
    manifest_path.write_bytes(b"placeholder")
    identities = [
        JetIdentity(file="HToBB_010.root", entry=index, label=0) for index in range(4)
    ]
    tokens = np.zeros((4, 128, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((4, 128), dtype=bool)
    mask[:, :3] = True
    tokens[:, :3, 0] = np.asarray((30.0, 20.0, 10.0))
    tokens[:, :3, 3] = np.asarray((31.0, 21.0, 11.0))
    hlt = JetView(
        tokens=tokens,
        mask=mask,
        labels=np.zeros(4, dtype=np.int64),
        jet_ids=identities,
        split="model_train",
        metadata={"source_manifest_hash": "manifest", "hlt_content_hash": "hlt"},
    )
    identity_hash = production.jet_identity_hash(identities)
    (hlt_dir / "model_train_fixed_hlt_metadata.json").write_text(
        json.dumps({"n_jets": 4}), encoding="utf-8"
    )
    (offline_dir / "model_train_offline_metadata.json").write_text(
        json.dumps({"offline_content_hash": "offline"}), encoding="utf-8"
    )
    selection = {
        "contract": ABPH_TARGET_MODE_SELECTION_CONTRACT,
        "campaign_root": str(root.resolve()),
        "campaign_mode": "pilot",
        "selected_mode": ABPH_RANK_LOCAL_TARGET_MODE,
        "selection_is_worker_overrideable": False,
        "measurement_content_hash": "measurement",
        "storage_projection_content_hash": "projection",
        "source_manifest_hash": "manifest",
        "hlt_content_hash": "hlt",
        "offline_content_hash": "offline",
        "data_dir": str(tmp_path / "raw"),
        "target_chunk_size": 2,
        "grouping_measurements": {
            "exclusive_kt": {"logical_bytes_per_jet": 1_000_000.0}
        },
        "source_provenance_by_split": {
            "model_train": {
                "source_manifest_hash": "manifest",
                "hlt_content_hash": "hlt",
                "offline_content_hash": "offline",
                "jet_identity_hash": identity_hash,
            },
            "model_val": {
                "source_manifest_hash": "manifest",
                "hlt_content_hash": "hlt-val",
                "offline_content_hash": "offline-val",
                "jet_identity_hash": "ids-val",
            },
        },
    }
    selection["content_hash"] = canonical_hash(selection)
    selection_path = root / "audits" / "target_mode_selection.json"
    write_target_mode_selection(selection_path, selection)
    monkeypatch.setattr(production, "load_cached_hlt_view", lambda *args, **kwargs: hlt)
    observed: list[tuple[JetIdentity, ...]] = []

    def load_slice(**kwargs):
        rows = tuple(kwargs["identities"])
        observed.append(rows)
        indices = [identities.index(row) for row in rows]
        return JetView(
            tokens=tokens[indices].copy(),
            mask=mask[indices].copy(),
            labels=np.zeros(len(rows), dtype=np.int64),
            jet_ids=list(rows),
            split="model_train",
        )

    monkeypatch.setattr(production, "build_rank_local_offline_view", load_slice)
    workspace = RankLocalWorkspace(
        tmp_path / "ram" / "abph-job-0",
        job_id="job",
        rank=0,
        probe=_probe(tmp_path, free=100_000_000),
    )
    source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        target_cache_dir=root / "targets",
        split="model_train",
        grouping="exclusive_kt",
        batch_size=2,
        shuffle_shards=False,
        seed=1,
        target_mode_report=selection_path,
        offline_cache_dir=offline_dir,
        manifest_path=manifest_path,
        workspace=workspace,
    )
    plan = source.derive_next_plan(global_update=0, accumulation_index=0)
    batch = source.prepare_planned_batch(plan)
    assert tuple(observed) == (tuple(identities[:2]),)
    assert batch["targets"].n_jets == 2
    assert batch["global_batch_plan_hash"] == plan.plan_hash
    assert workspace.reserved_bytes > 0


def test_slurm_workers_use_cleanup_trapped_rank_workspaces() -> None:
    repo = Path(__file__).resolve().parents[1]
    helper = (repo / "sbatch" / "adaptive_binary_ram_workspace.sh").read_text(
        encoding="utf-8"
    )
    wrapper = (
        repo / "sbatch" / "run_with_adaptive_binary_ram_workspace.sh"
    ).read_text(encoding="utf-8")
    variant = (repo / "sbatch" / "run_adaptive_binary_variant.sh").read_text(
        encoding="utf-8"
    )
    assert "trap 'status=$?" in helper
    assert "HUP" in helper and "INT" in helper and "TERM" in helper
    assert "abph_setup_ram_workspace" in wrapper
    assert "run_with_adaptive_binary_ram_workspace.sh" in variant
