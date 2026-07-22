from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (
    build_prediction_anchored_execution_spec,
    validate_prediction_anchored_execution_spec,
    write_prediction_anchored_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_numerical import (
    run_streamed_r0_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_r0 import (
    StreamedR0TrainConfig,
)
from teacher_logit_reco.local_particle_residual_field.bridge_splits import (
    ChildSplitSpec,
    ParentPartitionSpec,
    PredictionAnchoredSplitConfig,
    build_child_split_manifest,
)


def _parent() -> SplitManifest:
    splits = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        splits[split] = [
            JetIdentity(
                file=f"{split}/class-{label}.root",
                entry=split_index * 1000 + label * 10 + local,
                label=label,
            )
            for label in range(len(LABEL_NAMES))
            for local in range(2)
        ]
    return SplitManifest(
        data_dir="/synthetic",
        max_constits=4,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: 20 for split in SPLIT_ORDER},
        split_seeds={split: 100 + i for i, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits=splits,
        metadata={"file_level_separation_claimed": False},
    )


def _config() -> PredictionAnchoredSplitConfig:
    return PredictionAnchoredSplitConfig(
        contract="prediction_anchored_split_config_v1_execution_test",
        parent_split_counts=tuple((split, 20) for split in SPLIT_ORDER),
        partitions=(
            ParentPartitionSpec(
                "stack_train",
                810_101,
                (
                    ChildSplitSpec("stack_train_consumer", 10, "consumer_training"),
                    ChildSplitSpec("stack_train_distill", 10, "reconstructor_training"),
                ),
            ),
            ParentPartitionSpec(
                "model_val",
                810_202,
                (
                    ChildSplitSpec("model_val_stop", 10, "checkpoint_selection"),
                    ChildSplitSpec("model_val_select", 10, "configuration_selection"),
                ),
            ),
            ParentPartitionSpec(
                "stack_val",
                810_303,
                (
                    ChildSplitSpec(
                        "stack_val_consumer", 10, "consumer_confirmation", "consumer_preconfirmation"
                    ),
                    ChildSplitSpec(
                        "stack_val_deploy", 10, "deployable_confirmation", "deployable_preconfirmation"
                    ),
                ),
            ),
        ),
    )


def _write_sources(root: Path, parent: SplitManifest) -> tuple[Path, Path]:
    from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

    hlt_root = root / "hlt"
    offline_root = root / "offline"
    hlt_root.mkdir()
    offline_root.mkdir()
    parent_hash = manifest_hash(parent)
    for split in ("model_train", "model_val", "stack_train", "stack_val"):
        identities = parent.splits[split]
        n = len(identities)
        tokens = np.zeros((n, 4, RAW_TOKEN_DIM), dtype=np.float32)
        mask = np.ones((n, 4), dtype=bool)
        labels = np.asarray([item.label for item in identities], dtype=np.int64)
        files = list(dict.fromkeys(item.file for item in identities))
        file_lookup = {name: index for index, name in enumerate(files)}
        file_indices = np.asarray([file_lookup[item.file] for item in identities], dtype=np.int32)
        entries = np.asarray([item.entry for item in identities], dtype=np.int64)
        for event in range(n):
            for particle in range(4):
                pt = 1.0 + 0.02 * event + 0.1 * particle
                eta = -0.3 + 0.12 * particle
                phi = -0.5 + 0.03 * event + 0.04 * particle
                tokens[event, particle, 0] = pt
                tokens[event, particle, 1] = eta
                tokens[event, particle, 2] = phi
                tokens[event, particle, 3] = pt * np.cosh(eta)
                tokens[event, particle, 5 + particle] = 1.0
        for name, target_root, offset in (
            ("hlt", hlt_root, 0.0),
            ("offline", offline_root, 0.08),
        ):
            source_tokens = tokens.copy()
            source_tokens[..., 0] += offset
            source_tokens[..., 3] = source_tokens[..., 0] * np.cosh(source_tokens[..., 1])
            arrays = {
                "tokens": source_tokens,
                "mask": mask,
                "labels": labels,
                "jet_file_indices": file_indices,
                "jet_entries": entries,
            }
            if name == "hlt":
                npz = target_root / f"{split}_fixed_hlt.npz"
                metadata_path = target_root / f"{split}_fixed_hlt_metadata.json"
                content_key = "hlt_content_hash"
            else:
                npz = target_root / f"{split}_offline.npz"
                metadata_path = target_root / f"{split}_offline_metadata.json"
                content_key = "offline_content_hash"
            np.savez_compressed(npz, **arrays)
            metadata_path.write_text(
                json.dumps(
                    {
                        "n_jets": n,
                        "jet_files": files,
                        "jet_identity_hash": jet_identity_hash(identities),
                        "source_manifest_hash": parent_hash,
                        content_key: hash_arrays(arrays),
                    }
                ),
                encoding="utf-8",
            )
    return hlt_root, offline_root


def _fixture(tmp_path: Path):
    parent = _parent()
    parent_path = tmp_path / "parent.json"
    save_split_manifest(parent, parent_path, pretty=True)
    child = build_child_split_manifest(parent, config=_config())
    child_path = tmp_path / "child.json"
    write_immutable_json(child_path, child)
    hlt_root, offline_root = _write_sources(tmp_path, parent)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline-checkpoint-placeholder")
    spec = build_prediction_anchored_execution_spec(
        parent_manifest_path=parent_path,
        child_manifest_path=child_path,
        hlt_cache_dir=hlt_root,
        offline_cache_dir=offline_root,
        baseline_checkpoint_path=baseline,
        r0_config=StreamedR0TrainConfig(
            output_dir="unused",
            epochs=1,
            seed=7,
            early_stop_patience=-1,
            device="cpu",
            d_model=20,
            num_heads=5,
            num_layers=1,
            context_layers=1,
            dropout=0.0,
            attention_dropout=0.0,
        ),
    )
    spec_path = tmp_path / "execution.json"
    write_prediction_anchored_execution_spec(spec_path, spec)
    return spec, spec_path


def test_execution_spec_binds_all_development_sources_and_forbids_final_oracle(tmp_path):
    spec, spec_path = _fixture(tmp_path)
    audit = validate_prediction_anchored_execution_spec(spec, verify_file_hashes=True)
    assert audit["source_splits"] == ["model_train", "model_val", "stack_train", "stack_val"]
    assert audit["final_test_hlt_only"] is True
    assert "final_test" not in spec["sources"]
    assert spec["final_test_policy"]["offline_source_bound"] is False
    assert spec_path.is_file()


def test_bound_streamed_r0_executor_trains_real_weights_without_dense_artifacts(tmp_path):
    spec, spec_path = _fixture(tmp_path)
    output = tmp_path / "r0"
    result = run_streamed_r0_from_execution_spec(
        spec_path,
        output_dir=output,
        ram_root=tmp_path / "ram",
        allocation_id="mini",
        batch_size=10,
        shard_size=8,
        device="cpu",
        capacity_bytes=64 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    assert result["execution_spec_sha256"] == spec["content_hash"]
    assert result["one_open_per_compressed_source"] is True
    assert result["persistent_dense_fields_written"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "r0_metrics.json",
        "r0_registration.json",
        "r0_weights.pt",
    ]
    assert not list(output.glob("*.npz")) and not list(output.glob("*.npy"))
    registration = json.loads((output / "r0_registration.json").read_text())
    assert registration["split_manifest"] == spec["child_manifest"]["content_hash"]
    assert registration["input_availability"] == "hlt_only"
    assert not (tmp_path / "ram" / "prediction_anchored_bridge_mini").exists()

