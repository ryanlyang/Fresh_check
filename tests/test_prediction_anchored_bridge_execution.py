from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (
    build_campaign_registry,
    record_registry_measurements,
    require_production_ready,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign_policy import (
    build_campaign_reservations,
)
from teacher_logit_reco.local_particle_residual_field.bridge_semantic_evidence import (
    require_post_teacher_release,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (
    build_prediction_anchored_execution_spec,
    validate_prediction_anchored_execution_spec,
    write_prediction_anchored_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_consumer import (
    A0_C250,
    A0_C250_LONG,
    A0_S500,
    ConsumerCampaignConfig,
    T10_ALL50_CLEAN,
    T10_CLEAN,
    initialize_step3_root_from_reference,
    publish_evaluated_teacher_replica,
)
from teacher_logit_reco.local_particle_residual_field.bridge_consumer_execution import (
    confirm_selected_consumer_from_execution_spec,
    run_consumer_campaign_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_numerical import (
    prepare_bridge_inputs_from_execution_spec,
    run_streamed_r0_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_evaluation import (
    ALL50_TEACHER_NAMESPACE,
    ALTERNATE_TEACHER_NAMESPACE,
    N3_F0_TEACHER_NAMESPACE,
    PRIMARY_TEACHER_NAMESPACE,
)
from teacher_logit_reco.local_particle_residual_field.bridge_logits import (
    build_live_teacher_config,
    load_teacher_logit_cache,
)
from teacher_logit_reco.local_particle_residual_field.bridge_teacher_execution import (
    bind_teacher_set_from_execution_spec,
    cache_teacher_logits_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_r0 import (
    StreamedR0TrainConfig,
)
from teacher_logit_reco.local_particle_residual_field.bridge_production import (
    build_prediction_anchored_tigris_graph,
)
from teacher_logit_reco.local_particle_residual_field.bridge_reconstruction_execution import (
    ReconstructionCampaignConfig,
    ReconstructionReplicaResult,
    publish_l0_early_replay_manifest,
    publish_reconstruction_paired_replicas,
    run_reconstruction_pack_from_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_splits import (
    ChildSplitSpec,
    ParentPartitionSpec,
    PredictionAnchoredSplitConfig,
    build_child_split_manifest,
)


class _ToyConfig:
    def __init__(self, run_id: str, field_dim: int) -> None:
        self.run_id = run_id
        self.field_dim = field_dim

    def to_dict(self):
        return {"run_id": self.run_id, "field_dim": self.field_dim, "toy": True}


class _ToyPart(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 12)
        self.head = nn.Linear(12, len(LABEL_NAMES))

    def forward(self, values, mask):
        hidden = torch.tanh(self.projection(values))
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.head(pooled)


class _ToyConsumer(nn.Module):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        field_dim = 0 if run_id in {A0_C250, A0_C250_LONG, A0_S500} else 50
        self.config = _ToyConfig(run_id, field_dim)
        self.part_model = _ToyPart(1 + field_dim)

    def forward(
        self,
        points,
        features,
        lorentz_vectors,
        mask,
        *,
        tokens,
        raw_mask,
        indices=None,
        oracle_fields=None,
    ):
        del points, features, lorentz_vectors, mask, indices
        values = tokens[..., :1]
        if self.config.field_dim:
            if oracle_fields is None:
                raise ValueError("toy field consumer requires fields")
            values = torch.cat([values, oracle_fields], dim=-1)
        return self.part_model(values, raw_mask)


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
            for local in range(4)
        ]
    return SplitManifest(
        data_dir="/synthetic",
        max_constits=4,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: 40 for split in SPLIT_ORDER},
        split_seeds={split: 100 + i for i, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits=splits,
        metadata={"file_level_separation_claimed": False},
    )


def _config() -> PredictionAnchoredSplitConfig:
    return PredictionAnchoredSplitConfig(
        contract="prediction_anchored_split_config_v1_execution_test",
        parent_split_counts=tuple((split, 40) for split in SPLIT_ORDER),
        partitions=(
            ParentPartitionSpec(
                "stack_train",
                810_101,
                (
                    ChildSplitSpec("stack_train_consumer", 20, "consumer_training"),
                    ChildSplitSpec("stack_train_distill", 20, "reconstructor_training"),
                ),
            ),
            ParentPartitionSpec(
                "model_val",
                810_202,
                (
                    ChildSplitSpec("model_val_stop", 20, "checkpoint_selection"),
                    ChildSplitSpec("model_val_select", 20, "configuration_selection"),
                ),
            ),
            ParentPartitionSpec(
                "stack_val",
                810_303,
                (
                    ChildSplitSpec(
                        "stack_val_consumer", 20, "consumer_confirmation", "consumer_preconfirmation"
                    ),
                    ChildSplitSpec(
                        "stack_val_deploy", 20, "deployable_confirmation", "deployable_preconfirmation"
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


def _fixture(tmp_path: Path, *, r0_d_model: int = 20):
    parent = _parent()
    parent_path = tmp_path / "parent.json"
    save_split_manifest(parent, parent_path, pretty=True)
    child = build_child_split_manifest(parent, config=_config())
    child_path = tmp_path / "child.json"
    write_immutable_json(child_path, child)
    hlt_root, offline_root = _write_sources(tmp_path, parent)
    baseline = tmp_path / "baseline.pt"
    torch.save({"model_state_dict": _ToyConsumer(A0_C250).state_dict()}, baseline)
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
            d_model=int(r0_d_model),
            num_heads=5,
            num_layers=1,
            context_layers=1,
            dropout=0.0,
            attention_dropout=0.0,
        ),
        consumer_config=ConsumerCampaignConfig(
            baseline_steps=1,
            bridge_finetune_steps=1,
            batch_size=10,
            evaluation_interval_steps=1,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            model_size="tiny",
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


def test_consumer_executor_trains_all_paired_rows_and_selection_evidence_in_ram(tmp_path):
    spec, spec_path = _fixture(tmp_path, r0_d_model=160)
    r0_output = tmp_path / "r0_consumer"
    r0_result = run_streamed_r0_from_execution_spec(
        spec_path,
        output_dir=r0_output,
        ram_root=tmp_path / "ram_r0_consumer",
        allocation_id="mini_r0_consumer",
        batch_size=20,
        shard_size=16,
        device="cpu",
        capacity_bytes=96 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    bridge_inputs = tmp_path / "bridge_inputs"
    prepared = prepare_bridge_inputs_from_execution_spec(
        spec_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        output_dir=bridge_inputs,
        ram_root=tmp_path / "ram_bridge_inputs",
        allocation_id="mini_bridge_inputs",
        batch_size=20,
        shard_size=16,
        device="cpu",
        capacity_bytes=96 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    assert prepared["fit_child"] == "stack_train_distill"
    recipe_paths = {
        "physical45": bridge_inputs / "bridge_recipe_physical45.json",
        "all50": bridge_inputs / "bridge_recipe_all50.json",
    }
    replicas = tmp_path / "replicas"
    evidence = tmp_path / "consumer_evidence"
    result = run_consumer_campaign_from_execution_spec(
        spec_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        physical45_recipe_path=recipe_paths["physical45"],
        all50_recipe_path=recipe_paths["all50"],
        replica_output_dir=replicas,
        evaluation_output_dir=evidence,
        ram_root=tmp_path / "ram_consumers",
        allocation_id="mini_consumers",
        device="cpu",
        shard_size=16,
        generation_batch_size=20,
        evaluation_batch_size=20,
        bootstrap_resamples=8,
        capacity_bytes=128 * 1024 * 1024,
        allow_unverified_test_root=True,
        model_factory=lambda run_id, config: _ToyConsumer(run_id),
    )
    assert result["replica_count"] == 24
    assert result["one_open_per_compressed_source"] is True
    assert result["persistent_dense_fields_written"] is False
    assert len(list(replicas.glob("*.pt"))) == 24
    assert len(list(replicas.glob("*.metrics.json"))) == 24
    selection_hashes = set()
    for metrics_path in replicas.glob("*.metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        assert "model_val_select" in metrics
        selection_hashes.add(metrics["model_val_select"]["split_sha256"])
    child_manifest = json.loads(Path(spec["child_manifest"]["path"]).read_text())
    assert selection_hashes == {
        child_manifest["children"]["model_val_select"]["content_hash"]
    }
    for run_id in ("T10_clean", "T10_robust", "T10_all50_clean"):
        aggregate = json.loads(
            (evidence / run_id / "selection_aggregate.json").read_text()
        )
        assert aggregate["run_id"] == run_id
        assert aggregate["paired_seed_ids"] == [101, 202, 303]
        assert len(aggregate["mean_response_curve"]) == 5
        assert len(aggregate["mean_negative_control_gains"]) == 5
    assert not list(evidence.rglob("*.npz")) and not list(evidence.rglob("*.npy"))
    assert not (tmp_path / "ram_consumers" / "prediction_anchored_bridge_mini_consumers").exists()

    # A field-independent selected checkpoint has exact zero bridge gain.  It
    # exercises the sealed source/access path deterministically and must pass.
    confirmation_model = _ToyConsumer(T10_CLEAN)
    initialize_step3_root_from_reference(
        confirmation_model,
        spec["baseline_checkpoint"]["path"],
        run_id="Tpred",
        map_location="cpu",
    )
    selected_checkpoint = tmp_path / "selected_t10_source.pt"
    torch.save(
        {
            "model_state_dict": confirmation_model.state_dict(),
            "model_config": confirmation_model.config.to_dict(),
        },
        selected_checkpoint,
    )
    physical_recipe = json.loads(recipe_paths["physical45"].read_text())
    clean_aggregate_raw = json.loads(
        (evidence / T10_CLEAN / "selection_aggregate.json").read_text()
    )
    clean_aggregate = with_content_hash(
        {
            key: value
            for key, value in clean_aggregate_raw.items()
            if key != "content_hash"
        }
        | {
            "median_seed_id": 101,
            "median_checkpoint_sha256": sha256_file(selected_checkpoint),
            "eligible": True,
        }
    )
    consumer_publications = tmp_path / "consumers"
    publish_evaluated_teacher_replica(
        run_id=T10_CLEAN,
        selection_aggregate=clean_aggregate,
        replica_checkpoint_paths={
            101: selected_checkpoint,
            202: selected_checkpoint,
            303: selected_checkpoint,
        },
        output_dir=consumer_publications / T10_CLEAN,
    )
    for teacher_run in (T10_ALL50_CLEAN, "T10_robust"):
        aggregate = json.loads(
            (evidence / teacher_run / "selection_aggregate.json").read_text()
        )
        publish_evaluated_teacher_replica(
            run_id=teacher_run,
            selection_aggregate=aggregate,
            replica_checkpoint_paths={
                seed: replicas / f"{teacher_run}__seed{seed}.pt"
                for seed in (101, 202, 303)
            },
            output_dir=consumer_publications / teacher_run,
        )
    selected_checkpoint = consumer_publications / T10_CLEAN / "median_weights.pt"
    preconfirmation = with_content_hash(
        {
            "contract": "prediction_anchored_consumer_preconfirmation_v1",
            "status": "LOCKED_AWAITING_STACK_VAL_CONSUMER",
            "selected_consumer_id": "T10_clean_seed_101",
            "selected_consumer_recipe": "T10_clean",
            "checkpoint_path": str(selected_checkpoint),
            "checkpoint_sha256": sha256_file(selected_checkpoint),
            "selected_median_seed_id": 101,
            "selected_rho_endpoint": 0.10,
            "recipe_aggregate_metrics": clean_aggregate,
            "bridge_channel_policy": "physical45",
            "bridge_recipe_sha256": physical_recipe["content_hash"],
            "f0_checkpoint_sha256": r0_result["checkpoint_sha256"],
            "stack_val_consumer_opened": False,
            "deployable": False,
        }
    )
    preconfirmation_path = tmp_path / "preconfirmation.json"
    write_immutable_json(preconfirmation_path, preconfirmation)

    def toy_loader(path, selected_device):
        payload = torch.load(path, map_location=selected_device, weights_only=False)
        run_id = str((payload.get("model_config") or {}).get("run_id", T10_CLEAN))
        model = _ToyConsumer(run_id).to(selected_device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model, payload

    confirmation = confirm_selected_consumer_from_execution_spec(
        spec_path,
        preconfirmation_path=preconfirmation_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        physical45_recipe_path=recipe_paths["physical45"],
        output_dir=tmp_path / "selection",
        ram_root=tmp_path / "ram_confirmation",
        allocation_id="mini_confirmation",
        device="cpu",
        batch_size=20,
        shard_size=16,
        capacity_bytes=96 * 1024 * 1024,
        allow_unverified_test_root=True,
        model_loader=toy_loader,
    )
    assert confirmation["status"] == "CONFIRMED_LOCKED"
    assert confirmation["f0_accuracy"] == confirmation["bridge_0p10_accuracy"]
    assert (tmp_path / "selection" / "selected_bridge_consumer.json").is_file()

    bindings = bind_teacher_set_from_execution_spec(
        spec_path,
        selected_consumer_path=tmp_path / "selection" / "selected_bridge_consumer.json",
        physical45_recipe_path=recipe_paths["physical45"],
        all50_recipe_path=recipe_paths["all50"],
        all50_scaler_path=bridge_inputs / "bridge_scalers_all50.json",
        consumer_evaluation_root=evidence,
        consumer_publication_root=consumer_publications,
        output_dir=tmp_path / "bindings",
    )
    assert set(bindings["binding_paths"]) >= {"primary", "all50"}
    child_payload = json.loads(Path(spec["child_manifest"]["path"]).read_text())
    distill_sha256 = child_payload["children"]["stack_train_distill"]["content_hash"]
    selected_path = tmp_path / "selection" / "selected_bridge_consumer.json"
    cache_root = tmp_path / "teacher_logits"
    cache_specs = [
        (
            PRIMARY_TEACHER_NAMESPACE,
            tmp_path / "bindings" / "primary.json",
            recipe_paths["physical45"],
            selected_path,
        ),
        (
            N3_F0_TEACHER_NAMESPACE,
            tmp_path / "bindings" / "primary.json",
            recipe_paths["physical45"],
            selected_path,
        ),
        (
            ALL50_TEACHER_NAMESPACE,
            tmp_path / "bindings" / "all50.json",
            recipe_paths["all50"],
            None,
        ),
    ]
    if "alternate" in bindings["binding_paths"]:
        cache_specs.append(
            (
                ALTERNATE_TEACHER_NAMESPACE,
                tmp_path / "bindings" / "alternate.json",
                recipe_paths["physical45"],
                None,
            )
        )
    manifests = {}
    for namespace, binding_path, recipe_path, selected_consumer_path in cache_specs:
        cache_result = cache_teacher_logits_from_execution_spec(
            spec_path,
            binding_path=binding_path,
            namespace=namespace,
            selected_consumer_path=selected_consumer_path,
            r0_checkpoint_path=r0_result["checkpoint"],
            r0_registration_path=r0_result["registration"],
            bridge_recipe_path=recipe_path,
            output_root=cache_root,
            ram_root=tmp_path / f"ram_cache_{namespace}",
            allocation_id=f"mini_cache_{namespace}",
            device="cpu",
            batch_size=20,
            shard_size=16,
            capacity_bytes=96 * 1024 * 1024,
            allow_unverified_test_root=True,
            model_loader=toy_loader,
        )
        assert cache_result["event_count"] == 20
        assert cache_result["one_open_per_compressed_source"] is True
        assert cache_result["persistent_dense_fields_written"] is False
        binding = json.loads(binding_path.read_text())
        selected_payload = (
            json.loads(selected_path.read_text()) if selected_consumer_path else None
        )
        live = build_live_teacher_config(binding, primary_selection=selected_payload)
        manifest, arrays = load_teacher_logit_cache(
            cache_root / namespace,
            binding=binding,
            live_teacher_config=live,
            stack_train_distill_manifest_sha256=distill_sha256,
            primary_selection=selected_payload,
        )
        manifests[namespace] = (manifest, arrays)
        assert sorted(path.name for path in (cache_root / namespace).iterdir()) == [
            "teacher_logits.npz",
            "teacher_logits_manifest.json",
        ]
    assert np.array_equal(
        manifests[PRIMARY_TEACHER_NAMESPACE][1]["event_identity_hashes"],
        manifests[N3_F0_TEACHER_NAMESPACE][1]["event_identity_hashes"],
    )
    assert np.allclose(
        manifests[PRIMARY_TEACHER_NAMESPACE][1]["logits"],
        manifests[N3_F0_TEACHER_NAMESPACE][1]["logits"],
    )
    assert not list(cache_root.rglob("*field*.npy"))

    registry = build_campaign_registry(alternate_teacher_valid="alternate" in bindings["binding_paths"])
    registry = record_registry_measurements(
        registry, {row["canonical_run_id"]: 1024 for row in registry["runs"]}
    )
    release = require_post_teacher_release(
        registry,
        selected_consumer=json.loads(selected_path.read_text()),
        primary_binding=json.loads((tmp_path / "bindings" / "primary.json").read_text()),
    )
    write_immutable_json(tmp_path / "selection" / "post_teacher_release.json", release)
    readiness = require_production_ready(
        registry, fixed_persistent_bytes=4096, selected_budget_bytes=5 * 1024**3
    )
    reservations = build_campaign_reservations(
        registry,
        execution_spec=spec,
        production_readiness=readiness,
        fixed_parent_artifacts={
            "r0": {"sha256": "1" * 64, "size_bytes": 512, "path": tmp_path / "r0.pt"},
            "consumer": {"sha256": "2" * 64, "size_bytes": 512, "path": tmp_path / "t10.pt"},
            "metadata": {"sha256": "3" * 64, "size_bytes": 512, "path": tmp_path / "metadata"},
        },
        final_deployable_bundle_bytes=512,
    )
    graph = build_prediction_anchored_tigris_graph(
        registry, reservations=reservations, execution_spec=spec,
        artifact_root=str(tmp_path), pack_size=1
    )
    graph_path = tmp_path / "postteacher_graph.json"
    write_immutable_json(graph_path, graph)
    mini_reconstruction_config = ReconstructionCampaignConfig(
        field_warmup_steps=1,
        phase2_epochs=1,
        batch_size=20,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        early_stop_patience=-1,
        c0_model_width=20,
        dropout=0.0,
    )
    l1_node = next(
        row["node_id"]
        for row in graph["nodes"]
        if row["configuration_run_ids"] == ["D10_L1_ce_only"]
    )
    l1_replicas = tmp_path / "l1_replicas"
    l1_report = run_reconstruction_pack_from_execution_spec(
        spec_path,
        graph_path=graph_path,
        node_id=l1_node,
        artifact_root=tmp_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        physical45_scaler_path=bridge_inputs / "bridge_scalers_physical45.json",
        all50_scaler_path=bridge_inputs / "bridge_scalers_all50.json",
        absolute_scaler_path=None,
        deployed_reference_path=None,
        replica_output_dir=l1_replicas,
        ram_root=tmp_path / "ram_l1_executor",
        allocation_id="mini_l1_executor",
        device="cpu",
        shard_size=16,
        config=mini_reconstruction_config,
        capacity_bytes=192 * 1024 * 1024,
        allow_unverified_test_root=True,
        model_loader=toy_loader,
    )
    assert l1_report["replica_count"] == 3
    assert l1_report["results"][0]["target_cache_loaded"] is False
    assert l1_report["results"][0]["teacher_checkpoint_sha256"] == sha256_file(
        selected_checkpoint
    )
    assert len(list(l1_replicas.glob("*.pt"))) == 3
    l1_metrics = json.loads(
        (l1_replicas / "D10_L1_ce_only__seed101.metrics.json").read_text()
    )
    assert l1_metrics["teacher_lineage"]["mode"] == "locked_primary_teacher"
    assert l1_metrics["model_val_select"]["reliability_channels_exact_pass_through"] is True
    semantic = l1_metrics["model_val_select"]["semantic_evidence"]
    assert semantic["perturbation_audit_seeds"] == [9101, 9102, 9103, 9104]
    assert semantic["distribution_distance"]["validation_split"] == "model_val_select"
    assert semantic["final_test_accessed"] is False
    assert l1_metrics["deployable_checkpoint_requires_teacher_at_inference"] is False

    def load_l0_replicas(root: Path) -> list[ReconstructionReplicaResult]:
        loaded = []
        for seed in (101, 202, 303):
            checkpoint = root / f"D10_L0_bridge_only__seed{seed}.pt"
            metrics_path = root / f"D10_L0_bridge_only__seed{seed}.metrics.json"
            loaded.append(
                ReconstructionReplicaResult(
                    run_id="D10_L0_bridge_only",
                    seed_id=seed,
                    metrics=json.loads(metrics_path.read_text()),
                    weights_payload=torch.load(
                        checkpoint, map_location="cpu", weights_only=False
                    ),
                    source_checkpoint_sha256=sha256_file(checkpoint),
                )
            )
        return loaded

    # B3's teacher-free L0 run keeps only deterministic replay evidence.  Once
    # the selected consumer is confirmed and bound, B6 replays the exact same
    # three fits and adds the common model_val_select evaluation.
    early_l0_replicas = tmp_path / "early_l0_replicas"
    early_l0_report = run_reconstruction_pack_from_execution_spec(
        spec_path,
        graph_path=graph_path,
        node_id="b3_l0_paired3",
        artifact_root=tmp_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        physical45_scaler_path=bridge_inputs / "bridge_scalers_physical45.json",
        all50_scaler_path=bridge_inputs / "bridge_scalers_all50.json",
        absolute_scaler_path=None,
        deployed_reference_path=None,
        replica_output_dir=early_l0_replicas,
        ram_root=tmp_path / "ram_l0_early_executor",
        allocation_id="mini_l0_early_executor",
        device="cpu",
        shard_size=16,
        config=mini_reconstruction_config,
        capacity_bytes=192 * 1024 * 1024,
        allow_unverified_test_root=True,
        model_loader=toy_loader,
    )
    assert early_l0_report["l0_postteacher_deterministic_replay"] is False
    assert early_l0_report["deterministic_l0_training_enforced"] is True
    assert early_l0_report["cublas_workspace_config"] == ":4096:8"
    early_publication = publish_l0_early_replay_manifest(
        load_l0_replicas(early_l0_replicas),
        output_dir=tmp_path / "l0_early" / "D10_L0_bridge_only",
    )
    assert early_publication["persistent_artifacts"] == ["replay_manifest.json"]

    replay_l0_replicas = tmp_path / "replay_l0_replicas"
    replay_l0_report = run_reconstruction_pack_from_execution_spec(
        spec_path,
        graph_path=graph_path,
        node_id="b6_l0_postteacher_eval_paired3",
        artifact_root=tmp_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        physical45_scaler_path=bridge_inputs / "bridge_scalers_physical45.json",
        all50_scaler_path=bridge_inputs / "bridge_scalers_all50.json",
        absolute_scaler_path=None,
        deployed_reference_path=None,
        replica_output_dir=replay_l0_replicas,
        ram_root=tmp_path / "ram_l0_replay_executor",
        allocation_id="mini_l0_replay_executor",
        device="cpu",
        shard_size=16,
        config=mini_reconstruction_config,
        capacity_bytes=192 * 1024 * 1024,
        allow_unverified_test_root=True,
        model_loader=toy_loader,
    )
    assert replay_l0_report["configuration_run_ids"] == ["D10_L0_bridge_only"]
    assert replay_l0_report["l0_postteacher_deterministic_replay"] is True
    assert replay_l0_report["deterministic_l0_training_enforced"] is True
    assert replay_l0_report["l0_early_replay_sha256"] == json.loads(
        (tmp_path / "l0_early" / "D10_L0_bridge_only" / "replay_manifest.json").read_text()
    )["content_hash"]
    for seed in (101, 202, 303):
        metrics = json.loads(
            (
                replay_l0_replicas
                / f"D10_L0_bridge_only__seed{seed}.metrics.json"
            ).read_text()
        )
        assert metrics["model_val_select"]["accuracy"] >= 0.0
        assert metrics["postteacher_evaluation_lineage"][
            "teacher_used_only_for_model_val_select_evaluation"
        ] is True
        assert metrics["teacher_lineage"]["mode"] == "preteacher_l0_exception"
        assert metrics["teacher_lineage"]["teacher_checkpoint_sha256"] is None

    final_l0 = publish_reconstruction_paired_replicas(
        load_l0_replicas(replay_l0_replicas),
        output_dir=tmp_path / "reconstructors" / "D10_L0_bridge_only",
        l0_postteacher=True,
    )
    assert final_l0["persistent_artifacts"] == [
        "aggregate_metrics.json",
        "median_weights.pt",
        "publication.json",
    ]
    l0_aggregate = json.loads(Path(final_l0["aggregate"]).read_text())
    assert l0_aggregate["aggregation_phase"] == "postteacher_common_model_val_select"


def test_b3_l0_repository_executor_trains_three_real_replicas_from_one_ram_stage(tmp_path):
    spec, spec_path = _fixture(tmp_path, r0_d_model=160)
    r0_output = tmp_path / "r0_l0"
    r0_result = run_streamed_r0_from_execution_spec(
        spec_path,
        output_dir=r0_output,
        ram_root=tmp_path / "ram_r0_l0",
        allocation_id="mini_r0_l0",
        batch_size=20,
        shard_size=16,
        device="cpu",
        capacity_bytes=96 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    bridge_inputs = tmp_path / "bridge_inputs_l0"
    prepare_bridge_inputs_from_execution_spec(
        spec_path,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        output_dir=bridge_inputs,
        ram_root=tmp_path / "ram_bridge_l0",
        allocation_id="mini_bridge_l0",
        batch_size=20,
        shard_size=16,
        device="cpu",
        capacity_bytes=96 * 1024 * 1024,
        allow_unverified_test_root=True,
    )

    registry = build_campaign_registry(alternate_teacher_valid=False)
    registry = record_registry_measurements(
        registry, {row["canonical_run_id"]: 1024 for row in registry["runs"]}
    )
    readiness = require_production_ready(
        registry, fixed_persistent_bytes=4096, selected_budget_bytes=5 * 1024**3
    )
    reservations = build_campaign_reservations(
        registry,
        execution_spec=spec,
        production_readiness=readiness,
        fixed_parent_artifacts={
            "r0": {"sha256": "1" * 64, "size_bytes": 512, "path": tmp_path / "r0.pt"},
            "consumer": {"sha256": "2" * 64, "size_bytes": 512, "path": tmp_path / "t10.pt"},
            "metadata": {"sha256": "3" * 64, "size_bytes": 512, "path": tmp_path / "metadata"},
        },
        final_deployable_bundle_bytes=512,
    )
    artifact_root = tmp_path / "campaign_l0"
    graph = build_prediction_anchored_tigris_graph(
        registry, reservations=reservations, execution_spec=spec,
        artifact_root=str(artifact_root)
    )
    graph_path = tmp_path / "graph_l0.json"
    write_immutable_json(graph_path, graph)
    replicas = tmp_path / "l0_replicas"
    report = run_reconstruction_pack_from_execution_spec(
        spec_path,
        graph_path=graph_path,
        node_id="b3_l0_paired3",
        artifact_root=artifact_root,
        r0_checkpoint_path=r0_result["checkpoint"],
        r0_registration_path=r0_result["registration"],
        physical45_scaler_path=bridge_inputs / "bridge_scalers_physical45.json",
        all50_scaler_path=bridge_inputs / "bridge_scalers_all50.json",
        absolute_scaler_path=None,
        deployed_reference_path=None,
        replica_output_dir=replicas,
        ram_root=tmp_path / "ram_l0_executor",
        allocation_id="mini_l0_executor",
        device="cpu",
        shard_size=16,
        config=ReconstructionCampaignConfig(
            field_warmup_steps=1,
            phase2_epochs=1,
            batch_size=20,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            early_stop_patience=-1,
            c0_model_width=20,
            dropout=0.0,
        ),
        capacity_bytes=192 * 1024 * 1024,
        allow_unverified_test_root=True,
    )
    assert report["configuration_run_ids"] == ["D10_L0_bridge_only"]
    assert report["replica_count"] == 3
    assert report["one_open_per_compressed_source"] is True
    assert report["persistent_dense_fields_written"] is False
    assert len(list(replicas.glob("*.pt"))) == 3
    assert len(list(replicas.glob("*.metrics.json"))) == 3
    assert not list(replicas.glob("*.npz")) and not list(replicas.glob("*.npy"))
    assert not (tmp_path / "ram_l0_executor" / "prediction_anchored_bridge_mini_l0_executor").exists()
