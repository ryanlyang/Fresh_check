#!/usr/bin/env python3
"""Generate ABPH pseudo views, teacher logits, or teacher-free tagger logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.dual_view import build_part_inputs_torch  # noqa: E402
from jetclass_fresh.hlt_baseline import (  # noqa: E402
    build_particle_transformer_classifier,
    require_torch,
    resolve_device,
)
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.architecture_view_part import load_cached_offline_view  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    AdaptiveBinaryReconstructorModel,
    ABPH_CONSUMER_PSEUDO_CONTRACT,
    ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION,
    DeployablePseudoViewCacheConfig,
    FrozenPseudoBatchSource,
    LogitPredictionBlock,
    PseudoViewInputs,
    SelectedReconstructorPseudoGenerator,
    build_frozen_reconstructor_ram_source,
    build_shared_root_dual_reconstructor,
    build_variant_hierarchy_aware_tagger,
    canonical_hash,
    consumer_pseudo_array_names,
    describe_reconstructor_model,
    generate_deployable_pseudo_view_cache,
    load_hlt_prediction_source,
    load_selected_reconstructor,
    package_deployable_pseudo_views,
    require_deployable_pseudo_view_cache,
    resolve_variant_config,
    variant_spec,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.checkpoints import (  # noqa: E402
    ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT,
    build_compact_selected_checkpoint,
    load_torch_checkpoint,
    selected_model_state,
    streaming_storage_enabled,
    write_selected_checkpoint,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.bundled_scoring import (  # noqa: E402
    ABPH_BUNDLED_SCORING_CONTRACT,
    encode_logit_only_npz,
    group_scoring_members,
    scoring_source_family,
    source_generation_hash,
    validate_logit_only_npz,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (  # noqa: E402
    StorageArtifactClass,
    write_quota_managed_bytes,
    write_quota_managed_json,
)


ABPH_PREDICTION_EXECUTOR_CONTRACT = "adaptive_binary_pseudooffline_prediction_executor_v1"


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positive, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _detailed_classifier_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    classes = int(values.shape[1])
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    per_class = []
    aucs = []
    for class_index in range(classes):
        support = int(confusion[class_index].sum())
        auc = _binary_auc(probabilities[:, class_index], truth == class_index)
        if auc is not None:
            aucs.append(auc)
        per_class.append(
            {
                "class_index": class_index,
                "class_name": LABEL_NAMES[class_index] if class_index < len(LABEL_NAMES) else str(class_index),
                "support": support,
                "accuracy": (
                    float(confusion[class_index, class_index] / support)
                    if support
                    else None
                ),
                "ovr_auc": auc,
            }
        )
    picked = probabilities[np.arange(truth.shape[0]), truth].clip(1.0e-12, 1.0)
    return {
        "available": True,
        "accuracy": float((predictions == truth).mean()),
        "loss": float(-np.log(picked).mean()),
        "cross_entropy": float(-np.log(picked).mean()),
        "macro_ovr_auc": float(np.mean(aucs)) if aucs else None,
        "macro_per_class_accuracy": float(
            np.mean([row["accuracy"] for row in per_class if row["accuracy"] is not None])
        ),
        "per_class": per_class,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
        "n_jets": int(truth.shape[0]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    member_group = parser.add_mutually_exclusive_group(required=True)
    member_group.add_argument("--variant")
    member_group.add_argument("--members", nargs="+")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ABPH_PREDICTION_BATCH_SIZE", "128")))
    parser.add_argument("--shard-size", type=int, default=int(os.environ.get("ABPH_PREDICTION_SHARD_SIZE", "1024")))
    parser.add_argument("--teacher-logits", action="store_true")
    parser.add_argument("--smoke", action="store_true", default=os.environ.get("ABPH_SMOKE", "0") == "1")
    parser.add_argument("--overwrite", action="store_true", default=os.environ.get("OVERWRITE", "0") == "1")
    return parser


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: str | Path, *, device: Any = "cpu") -> Mapping[str, Any]:
    return load_torch_checkpoint(path, device=device)


def _state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("checkpoint_contract") == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        return selected_model_state(payload)
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    raise KeyError("checkpoint has no model state")


def _component_state(
    payload: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    components = payload.get("component_state_dicts")
    if isinstance(components, Mapping) and isinstance(components.get(name), Mapping):
        return components[name]
    legacy = payload.get(f"{name}_state_dict")
    return legacy if isinstance(legacy, Mapping) else None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _predictor(model: Any, batch: Any, evaluation_seed: int, device: str):
    torch = require_torch()
    output = model.deploy(
        torch.as_tensor(batch.tokens, device=device),
        torch.as_tensor(batch.mask, device=device).bool(),
        evaluation_seed=int(evaluation_seed),
    )
    return package_deployable_pseudo_views(
        output.hierarchy_output, output.rendered_views
    )


def _write_composed_reconstructor_checkpoint(
    *,
    model: AdaptiveBinaryReconstructorModel,
    template_path: Path,
    output_path: Path,
    provenance_extra: Mapping[str, Any],
) -> Path:
    torch = require_torch()
    template = dict(_torch_load(template_path))
    if streaming_storage_enabled():
        config = dict(
            template.get("resolved_variant_config")
            or template.get("config")
            or {}
        )
        config_hash = str(
            template.get("resolved_variant_config_hash")
            or config.get("config_hash")
            or canonical_hash(config)
        )
        compact = build_compact_selected_checkpoint(
            model_state_dict=model.state_dict(),
            checkpoint_role="best_model_val",
            model_metadata=describe_reconstructor_model(model, model.module_groups()),
            resolved_variant_config=config,
            resolved_variant_config_hash=config_hash,
            validation=template.get("validation"),
            provenance={
                **dict(template.get("provenance") or {}),
                **dict(provenance_extra),
                "source_git_commit": os.environ.get(
                    "FRESH_SOURCE_COMMIT", "recorded_by_slurm_run_config"
                ),
                "source_status_hash": os.environ.get(
                    "FRESH_SOURCE_STATUS_HASH", "recorded_by_slurm_run_config"
                ),
            },
            runtime_contracts=dict(template.get("runtime_contracts") or {}),
            schedule_contracts=dict(template.get("schedule_contracts") or {}),
            extra_metadata={"composed_prediction_artifact": dict(provenance_extra)},
        )
        campaign_root = output_path.parents[2]
        write_selected_checkpoint(
            output_path,
            compact,
            campaign_root=campaign_root,
            artifact_role="composed_selected_reconstructor_checkpoint",
            run_id=output_path.parent.name,
        )
        return output_path
    template["checkpoint_role"] = "best_model_val"
    template["model_state_dict"] = model.state_dict()
    template["online_model_state_dict"] = model.state_dict()
    template["model_metadata"] = describe_reconstructor_model(
        model, model.module_groups()
    )
    template["config"] = {
        **dict(template.get("config") or {}),
        "composed_prediction_artifact": dict(provenance_extra),
    }
    template["provenance"] = {
        **dict(template.get("provenance") or {}),
        **dict(provenance_extra),
        "source_git_commit": os.environ.get("FRESH_SOURCE_COMMIT", "recorded_by_slurm_run_config"),
        "source_status_hash": os.environ.get("FRESH_SOURCE_STATUS_HASH", "recorded_by_slurm_run_config"),
    }
    template["final_test_loaded"] = False
    template["teacher_logits_loaded"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output_path)
    return output_path


def _reconstructor_for_source(
    root: Path,
    source_name: str,
    *,
    device: Any,
    smoke: bool,
) -> tuple[AdaptiveBinaryReconstructorModel, Path, str]:
    if source_name == "E7_shared_root_dual":
        kt = root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt"
        ca = root / "runs" / "D2_ca32_mh4_particles" / "best_model_val.pt"
        model = build_shared_root_dual_reconstructor(
            kt, ca, device=device, smoke=smoke
        )
        checkpoint = _write_composed_reconstructor_checkpoint(
            model=model,
            template_path=kt,
            output_path=root / "runs" / "_E7_shared_root_dual_reconstructor" / "best_model_val.pt",
            provenance_extra={
                "composition": "one_kt_compiled_root_with_kt_and_ca_below_root_branches",
                "kt_checkpoint_sha256": _sha256(kt),
                "ca_checkpoint_sha256": _sha256(ca),
            },
        )
        config_hash = canonical_hash(
            {"source": source_name, "kt": _sha256(kt), "ca": _sha256(ca)}
        )
        return model, checkpoint, config_hash
    base_name = source_name.split("__seed", 1)[0]
    if base_name.startswith("F"):
        tagger_checkpoint = root / "runs" / source_name / "best_model_val.pt"
        payload = _torch_load(tagger_checkpoint, device=device)
        reconstructor_state = _component_state(payload, "reconstructor")
        if isinstance(reconstructor_state, Mapping):
            resolved = resolve_variant_config(base_name)
            dual = bool(resolved["model"]["fusion"].get("dual_hierarchy"))
            if dual:
                model = build_shared_root_dual_reconstructor(
                    root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt",
                    root / "runs" / "D2_ca32_mh4_particles" / "best_model_val.pt",
                    device=device,
                    smoke=smoke,
                )
                template = (
                    root
                    / "runs"
                    / "_E7_shared_root_dual_reconstructor"
                    / "best_model_val.pt"
                )
                if not template.is_file():
                    _, template, _ = _reconstructor_for_source(
                        root,
                        "E7_shared_root_dual",
                        device=device,
                        smoke=smoke,
                    )
            else:
                model = AdaptiveBinaryReconstructorModel(
                    hierarchy_names=("exclusive_kt",),
                    variant_name="D1_kt32_mh4_particles",
                    smoke=smoke,
                ).to(device)
                template = root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt"
            model.load_state_dict(reconstructor_state, strict=True)
            checkpoint = _write_composed_reconstructor_checkpoint(
                model=model,
                template_path=template,
                output_path=root / "runs" / f"_{source_name}_joint_reconstructor" / "best_model_val.pt",
                provenance_extra={
                    "composition": (
                        "joint_tagger_selected_shared_root_dual_reconstructor"
                        if dual
                        else "joint_tagger_selected_reconstructor"
                    ),
                    "hierarchy_names": list(model.hierarchy_names),
                    "tagger_checkpoint_sha256": _sha256(tagger_checkpoint),
                },
            )
            return model, checkpoint, canonical_hash(
                {"source": source_name, "tagger": _sha256(tagger_checkpoint)}
            )
    if source_name == "D2_ca32_mh4_particles":
        variant = source_name
    else:
        variant = "D1_kt32_mh4_particles"
    checkpoint = root / "runs" / variant / "best_model_val.pt"
    model = load_selected_reconstructor(
        checkpoint,
        variant_name=variant,
        device=device,
        smoke=smoke,
    )
    return model, checkpoint, resolve_variant_config(variant)["resolved_config_hash"]


def _ensure_pseudo_cache(
    root: Path,
    source_name: str,
    split: str,
    *,
    device: Any,
    batch_size: int,
    shard_size: int,
    smoke: bool,
    overwrite: bool,
) -> Path:
    destination = root / "pseudo_predictions" / source_name / split
    if destination.is_dir() and not overwrite:
        require_deployable_pseudo_view_cache(destination)
        return destination
    model, checkpoint, config_hash = _reconstructor_for_source(
        root, source_name, device=device, smoke=smoke
    )
    source = load_hlt_prediction_source(
        root / "inputs" / "split_manifest" / "split_manifest.json.gz",
        root / "inputs" / "hlt_cache",
        split,
        campaign_mode=os.environ.get("ABPH_CAMPAIGN_MODE", "pilot"),
        max_jets=(int(os.environ.get("ABPH_SMOKE_JETS", "0")) or None) if smoke else None,
        batch_size=batch_size,
    )
    config = DeployablePseudoViewCacheConfig(
        checkpoint_path=checkpoint,
        output_cache_dir=destination,
        split=split,
        resolved_variant_config_hash=config_hash,
        campaign_mode=os.environ.get("ABPH_CAMPAIGN_MODE", "pilot"),
        device=str(device),
        shard_size=shard_size,
        overwrite=overwrite,
        reuse_existing=not overwrite,
        source_git_commit=os.environ.get("FRESH_SOURCE_COMMIT", "recorded_by_slurm_run_config"),
        source_status_hash=os.environ.get("FRESH_SOURCE_STATUS_HASH", "recorded_by_slurm_run_config"),
    )
    generate_deployable_pseudo_view_cache(
        model, model.module_groups(), source, config, predictor=_predictor
    )
    return destination


def _offline_teacher_logits(
    root: Path,
    splits: list[str],
    *,
    device: Any,
    batch_size: int,
    smoke: bool,
) -> None:
    if any(split == "final_test" for split in splits):
        raise ValueError("teacher logits are forbidden for final_test")
    checkpoint = root / "runs" / "A4_offline_part_ceiling" / "best_model_val.pt"
    resolved = resolve_variant_config("A4_offline_part_ceiling")
    overrides = dict(resolved["model"]["hlt_part"])
    overrides.pop("enabled", None)
    model = build_particle_transformer_classifier(
        num_classes=10, model_size="base", overrides=overrides
    ).to(device)
    model.load_state_dict(_state(_torch_load(checkpoint, device=device)), strict=True)
    model.eval()
    torch = require_torch()
    for split in splits:
        view = load_cached_offline_view(root / "inputs" / "offline_cache", split, verify_hash=True)
        maximum = min(len(view.labels), int(os.environ.get("ABPH_SMOKE_JETS", "128"))) if smoke else len(view.labels)
        logits = []
        with torch.no_grad():
            for start in range(0, maximum, batch_size):
                stop = min(start + batch_size, maximum)
                tokens = torch.as_tensor(view.tokens[start:stop], device=device)
                mask = torch.as_tensor(view.mask[start:stop], device=device).bool()
                part = build_part_inputs_torch(tokens, mask, max_constits=128)
                logits.append(
                    model(
                        part["points"],
                        part["features"],
                        part["lorentz_vectors"],
                        part["mask"],
                    ).detach().cpu().numpy()
                )
        values = np.concatenate(logits, axis=0)
        output_dir = root / "teacher_logits" / "A4_offline_part_ceiling"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{split}.npz",
            logits=values.astype(np.float32),
            labels=np.asarray(view.labels[:maximum], dtype=np.int64),
            source_indices=np.arange(maximum, dtype=np.int64),
        )
        _atomic_json(
            output_dir / f"{split}_metadata.json",
            {
                "contract": ABPH_PREDICTION_EXECUTOR_CONTRACT,
                "ok": True,
                "kind": "training_only_offline_teacher_logits",
                "split": split,
                "n_jets": maximum,
                "checkpoint_sha256": _sha256(checkpoint),
                "prediction_sha256": _sha256(output_dir / f"{split}.npz"),
                "provenance": {
                    **_split_provenance(view),
                    "offline_content_hash": view.metadata.get(
                        "offline_content_hash"
                    )
                    or view.metadata.get("content_hash"),
                    "offline_inputs_loaded": True,
                    "teacher_logits_loaded": False,
                },
                "final_test_loaded": False,
                "selection_eligible": False,
            },
        )


def _split_provenance(view: Any) -> dict[str, Any]:
    metadata = view.metadata
    labels = np.asarray(view.labels, dtype=np.int64)
    return {
        "source_manifest_hash": metadata.get("source_manifest_hash"),
        "jet_identity_hash": jet_identity_hash(view.jet_ids),
        "label_hash": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
        "class_mapping_hash": canonical_hash({"label_names": list(LABEL_NAMES)}),
        "hlt_content_hash": metadata.get("hlt_content_hash"),
        "hlt_profile": metadata.get("hlt_profile"),
        "hlt_profile_version": metadata.get("hlt_profile_version"),
        "hlt_degradation_strength": metadata.get("hlt_degradation_strength"),
        "hlt_params_hash": metadata.get("hlt_params_hash"),
        "source_git_commit": os.environ.get(
            "FRESH_SOURCE_COMMIT", "recorded_by_slurm_run_config"
        ),
        "source_status_hash": os.environ.get(
            "FRESH_SOURCE_STATUS_HASH", "recorded_by_slurm_run_config"
        ),
        "offline_inputs_loaded": False,
        "teacher_logits_loaded": False,
        "hypothesis_selection_used_offline_target": False,
        "fusion_fitted_on_final_test": False,
    }


def _consumer_pseudo_provenance(source: Any) -> dict[str, Any]:
    members: dict[str, str] = {}
    for cache_dir, metadata in zip(source.cache_dirs, source.metadata):
        hierarchy_names = tuple(metadata.get("hierarchy_names") or ())
        frontier_depths = dict(metadata.get("frontier_depths") or {})
        expected = consumer_pseudo_array_names(hierarchy_names, frontier_depths)
        schema = dict(metadata.get("prediction_schema") or {})
        missing = [name for name in expected if name not in schema]
        if missing:
            raise ValueError(f"pseudo cache {cache_dir} lacks consumer arrays: {missing}")
        schema_hash = canonical_hash(
            {
                "contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
                "schema_version": ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION,
                "arrays": {name: schema[name] for name in sorted(expected)},
            }
        )
        declared = metadata.get("consumer_pseudo_schema_hash")
        if declared not in (None, schema_hash):
            raise ValueError(f"pseudo cache {cache_dir} consumer schema hash mismatch")
        members[cache_dir.name] = schema_hash
    return {
        "consumer_pseudo_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
        "consumer_pseudo_schema_hash": canonical_hash({"members": members}),
        "consumer_pseudo_member_schema_hashes": members,
        "consumer_only_pseudo_at_tagger_boundary": True,
    }


def _jet_identity_strings(view: Any, indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            f"{view.jet_ids[int(index)].file}:{view.jet_ids[int(index)].entry}:"
            f"{view.jet_ids[int(index)].label}"
            for index in np.asarray(indices, dtype=np.int64)
        ],
        dtype=np.str_,
    )


def _update_run_report(
    run_dir: Path,
    *,
    split: str,
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    teacher_free: bool,
) -> None:
    path = run_dir / "run_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("metrics", {})[split] = dict(metrics)
    provenance_rows = payload.setdefault("provenance", {})
    existing_provenance = (
        provenance_rows.get(split)
        if isinstance(provenance_rows.get(split), Mapping)
        else {}
    )
    provenance_rows[split] = {**dict(existing_provenance), **dict(provenance)}
    diagnostics = payload.setdefault("diagnostics", {})
    existing = diagnostics.get(split) if isinstance(diagnostics.get(split), Mapping) else {}
    diagnostics[split] = {
        **dict(existing),
        "offline_inputs_loaded": False,
        "teacher_logits_loaded": not teacher_free,
        "final_test_loaded": split == "final_test",
        "final_test_attestation": (
            {
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": False,
                "offline_targets_loaded": False,
                "hypothesis_selection_used_offline_target": False,
                "fusion_fitted_on_final_test": False,
            }
            if split == "final_test"
            else None
        ),
    }
    _atomic_json(path, payload)


def _baseline_logits(
    root: Path,
    member: str,
    split: str,
    *,
    device: Any,
    batch_size: int,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, Any]:
    base_name = member.split("__seed", 1)[0]
    resolved = resolve_variant_config(base_name)
    overrides = dict(resolved["model"]["hlt_part"])
    overrides.pop("enabled", None)
    overrides.pop("capacity_match_target", None)
    overrides.pop("capacity_match_policy", None)
    overrides.pop("capacity_control_kind", None)
    model = build_particle_transformer_classifier(
        num_classes=10, model_size="base", overrides=overrides
    ).to(device)
    checkpoint = root / "runs" / member / "best_model_val.pt"
    model.load_state_dict(_state(_torch_load(checkpoint, device=device)), strict=True)
    model.eval()
    view = load_cached_hlt_view(root / "inputs" / "hlt_cache", split, verify_hash=True)
    maximum = min(len(view.labels), int(os.environ.get("ABPH_SMOKE_JETS", "128"))) if smoke else len(view.labels)
    torch = require_torch()
    rows = []
    with torch.no_grad():
        for start in range(0, maximum, batch_size):
            stop = min(start + batch_size, maximum)
            tokens = torch.as_tensor(view.tokens[start:stop], device=device)
            mask = torch.as_tensor(view.mask[start:stop], device=device).bool()
            part = build_part_inputs_torch(tokens, mask, max_constits=128)
            rows.append(
                model(
                    part["points"],
                    part["features"],
                    part["lorentz_vectors"],
                    part["mask"],
                ).detach().cpu().numpy()
            )
    return np.concatenate(rows), np.asarray(view.labels[:maximum]), view


def _tagger_logits(
    root: Path,
    member: str,
    split: str,
    *,
    device: Any,
    batch_size: int,
    shard_size: int,
    smoke: bool,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    base_name = member.split("__seed", 1)[0]
    resolved = resolve_variant_config(base_name)
    model = build_variant_hierarchy_aware_tagger(base_name, smoke=smoke).to(device)
    checkpoint = root / "runs" / member / "best_model_val.pt"
    checkpoint_payload = _torch_load(checkpoint, device=device)
    model.load_state_dict(_state(checkpoint_payload), strict=True)
    model.eval()
    run_id = str(resolved["variant"]["run_id"])
    if base_name.startswith("F") and isinstance(
        _component_state(checkpoint_payload, "reconstructor"), Mapping
    ):
        sources = (member,)
    elif bool(resolved["model"]["fusion"].get("dual_hierarchy")) and run_id != "E11":
        sources = ("E7_shared_root_dual",)
    elif run_id == "E11":
        sources = ("D1_kt32_mh4_particles", "D2_ca32_mh4_particles")
    elif str(resolved["model"]["hierarchy"].get("grouping")) == "cambridge_aachen":
        sources = ("D2_ca32_mh4_particles",)
    else:
        sources = ("D1_kt32_mh4_particles",)
    if streaming_storage_enabled():
        generators = None
        if sources == (member,):
            reconstructor_state = _component_state(
                checkpoint_payload, "reconstructor"
            )
            if not isinstance(reconstructor_state, Mapping):
                raise ValueError(
                    f"{member} lacks the selected joint reconstructor state"
                )
            dual = bool(resolved["model"]["fusion"].get("dual_hierarchy"))
            if dual:
                reconstructor = build_shared_root_dual_reconstructor(
                    root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt",
                    root / "runs" / "D2_ca32_mh4_particles" / "best_model_val.pt",
                    device=device,
                    smoke=smoke,
                )
            else:
                reconstructor = AdaptiveBinaryReconstructorModel(
                    hierarchy_names=("exclusive_kt",),
                    variant_name="D1_kt32_mh4_particles",
                    smoke=smoke,
                ).to(device)
            reconstructor.load_state_dict(reconstructor_state, strict=True)
            generators = (
                SelectedReconstructorPseudoGenerator(
                    reconstructor,
                    source_name=member,
                    checkpoint_hashes={member: _sha256(checkpoint)},
                    device=device,
                ),
            )
        source = build_frozen_reconstructor_ram_source(
            root,
            sources,
            split=split,
            batch_size=batch_size,
            device=device,
            smoke=smoke,
            independent_roots=(run_id == "E11"),
            maximum_batches=(1 if smoke else None),
            generators=generators,
        )
    else:
        cache_dirs = tuple(
            _ensure_pseudo_cache(
                root,
                source_name,
                split,
                device=device,
                batch_size=batch_size,
                shard_size=shard_size,
                smoke=smoke,
                overwrite=overwrite,
            )
            for source_name in sources
        )
        source = FrozenPseudoBatchSource(
            hlt_cache_dir=root / "inputs" / "hlt_cache",
            cache_dirs=cache_dirs,
            split=split,
            batch_size=batch_size,
            independent_roots=(run_id == "E11"),
            maximum_batches=(1 if smoke else None),
        )
    torch = require_torch()
    logits = []
    labels = []
    indices = []
    with torch.no_grad():
        for batch in source.iter_batches(shuffle=False, seed=24731):
            tokens = torch.as_tensor(batch.hlt_tokens, device=device)
            mask = torch.as_tensor(batch.hlt_mask, device=device).bool()
            pseudo = PseudoViewInputs.from_deployable_batch(batch.pseudo, device=device, dtype=tokens.dtype)
            roots = None if batch.independent_roots is None else {
                name: torch.as_tensor(value, device=device, dtype=tokens.dtype)
                for name, value in batch.independent_roots.items()
            }
            output = model(tokens, mask, pseudo, independent_root_ledgers=roots)
            logits.append(output.logits.detach().cpu().numpy())
            labels.append(np.asarray(batch.labels, dtype=np.int64))
            indices.append(np.asarray(batch.indices, dtype=np.int64))
    return (
        np.concatenate(logits),
        np.concatenate(labels),
        np.concatenate(indices),
        source.hlt,
        source,
    )


def _load_tagger_for_scoring(
    root: Path,
    member: str,
    *,
    device: Any,
    smoke: bool,
) -> tuple[Any, Mapping[str, Any], str, str]:
    base_name = member.split("__seed", 1)[0]
    checkpoint = root / "runs" / member / "best_model_val.pt"
    payload = _torch_load(checkpoint, device="cpu")
    model = build_variant_hierarchy_aware_tagger(base_name, smoke=smoke)
    model.load_state_dict(_state(payload), strict=True)
    model.to(device).eval()
    return (
        model,
        payload,
        _sha256(checkpoint),
        str(resolve_variant_config(base_name)["resolved_config_hash"]),
    )


def _build_bundled_ram_source(
    root: Path,
    family: Any,
    split: str,
    *,
    device: Any,
    batch_size: int,
    smoke: bool,
) -> Any:
    generators = None
    if family.kind == "joint_checkpoint":
        member = str(family.joint_checkpoint_member)
        base_name = member.split("__seed", 1)[0]
        resolved = resolve_variant_config(base_name)
        checkpoint = root / "runs" / member / "best_model_val.pt"
        payload = _torch_load(checkpoint, device="cpu")
        reconstructor_state = _component_state(payload, "reconstructor")
        if not isinstance(reconstructor_state, Mapping):
            raise ValueError(f"{member} lacks its selected joint reconstructor")
        if bool(resolved["model"]["fusion"].get("dual_hierarchy")):
            reconstructor = build_shared_root_dual_reconstructor(
                root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt",
                root / "runs" / "D2_ca32_mh4_particles" / "best_model_val.pt",
                device=device,
                smoke=smoke,
            )
        else:
            reconstructor = AdaptiveBinaryReconstructorModel(
                hierarchy_names=("exclusive_kt",),
                variant_name="D1_kt32_mh4_particles",
                smoke=smoke,
            ).to(device)
        reconstructor.load_state_dict(reconstructor_state, strict=True)
        generators = (
            SelectedReconstructorPseudoGenerator(
                reconstructor,
                source_name=member,
                checkpoint_hashes={member: _sha256(checkpoint)},
                device=device,
            ),
        )
    source = build_frozen_reconstructor_ram_source(
        root,
        family.source_names,
        split=split,
        batch_size=batch_size,
        device=device,
        smoke=smoke,
        independent_roots=bool(family.independent_roots),
        maximum_batches=(1 if smoke else None),
        generators=generators,
        execution_mode="full_rank_cache",
    )
    source.release_generator_gpu_models()
    return source


def _calibrated_model_bundle_size(
    model: Any,
    *,
    device: Any,
    requested: int,
) -> tuple[int, dict[str, Any]]:
    torch = require_torch()
    parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in model.parameters()
    )
    configured = max(1, int(requested))
    if str(getattr(device, "type", device)) == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        memory_limit = max(1, int(0.45 * free_bytes) // max(parameter_bytes, 1))
    else:
        free_bytes = None
        memory_limit = configured
    selected = max(1, min(configured, memory_limit))
    return selected, {
        "requested_models_per_bundle": configured,
        "selected_models_per_bundle": selected,
        "probe_model_parameter_bytes": parameter_bytes,
        "cuda_free_bytes_at_calibration": free_bytes,
        "gpu_fraction_reserved_for_models": 0.45,
    }


def _score_tagger_bundle(
    root: Path,
    members: list[str],
    source: Any,
    *,
    device: Any,
    smoke: bool,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, str],
    dict[str, str],
    dict[str, Any],
]:
    torch = require_torch()
    probe, _payload, _checkpoint_hash, _config_hash = _load_tagger_for_scoring(
        root, members[0], device=device, smoke=smoke
    )
    bundle_size, calibration = _calibrated_model_bundle_size(
        probe,
        device=device,
        requested=int(os.environ.get("ABPH_SCORING_MODEL_BUNDLE_SIZE", "4")),
    )
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    outputs: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    reference_labels: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    for bundle_start in range(0, len(members), bundle_size):
        selected_members = members[bundle_start : bundle_start + bundle_size]
        models: dict[str, Any] = {}
        for member in selected_members:
            model, _payload, checkpoint_hash, config_hash = _load_tagger_for_scoring(
                root, member, device=device, smoke=smoke
            )
            models[member] = model
            checkpoint_hashes[member] = checkpoint_hash
            config_hashes[member] = config_hash
        rows = {member: [] for member in selected_members}
        labels = []
        indices = []
        with torch.no_grad():
            for batch in source.iter_batches(shuffle=False, seed=24731):
                tokens = torch.as_tensor(batch.hlt_tokens, device=device)
                mask = torch.as_tensor(batch.hlt_mask, device=device).bool()
                pseudo = PseudoViewInputs.from_deployable_batch(
                    batch.pseudo, device=device, dtype=tokens.dtype
                )
                roots = (
                    None
                    if batch.independent_roots is None
                    else {
                        name: torch.as_tensor(value, device=device, dtype=tokens.dtype)
                        for name, value in batch.independent_roots.items()
                    }
                )
                for member, model in models.items():
                    rows[member].append(
                        model(
                            tokens,
                            mask,
                            pseudo,
                            independent_root_ledgers=roots,
                        ).logits.detach().cpu().numpy()
                    )
                labels.append(np.asarray(batch.labels, dtype=np.int64))
                indices.append(np.asarray(batch.indices, dtype=np.int64))
        current_labels = np.concatenate(labels)
        current_indices = np.concatenate(indices)
        if reference_labels is None:
            reference_labels = current_labels
            reference_indices = current_indices
        elif not np.array_equal(reference_labels, current_labels) or not np.array_equal(
            reference_indices, current_indices
        ):
            raise RuntimeError("scoring model bundles changed source identity order")
        outputs.update(
            {member: np.concatenate(values) for member, values in rows.items()}
        )
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert reference_labels is not None and reference_indices is not None
    return (
        outputs,
        reference_labels,
        reference_indices,
        checkpoint_hashes,
        config_hashes,
        calibration,
    )


def _persist_logit_only_artifact(
    root: Path,
    destination: Path,
    data: bytes,
    *,
    provenance_hash: str,
    run_id: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if streaming_storage_enabled():
        write_quota_managed_bytes(
            root,
            destination,
            data,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role="logit_only_fusion_input",
            source_provenance_hash=provenance_hash,
            run_id=run_id,
        )
    else:
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)


def _persist_scoring_json(
    root: Path,
    destination: Path,
    payload: Mapping[str, Any],
    *,
    provenance_hash: str,
    run_id: str,
) -> None:
    if streaming_storage_enabled():
        write_quota_managed_json(
            root,
            destination,
            payload,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role="bundled_scoring_metadata",
            source_provenance_hash=provenance_hash,
            run_id=run_id,
        )
    else:
        _atomic_json(destination, payload)


def _write_bundled_classifier_predictions(
    root: Path,
    members: list[str],
    splits: list[str],
    *,
    device: Any,
    batch_size: int,
    smoke: bool,
    overwrite: bool,
) -> None:
    grouped = group_scoring_members(members)
    if len(grouped) != 1:
        raise ValueError(
            "one bundled scoring allocation must contain exactly one pseudo source family"
        )
    family = scoring_source_family(members[0])
    if tuple(grouped[family.key]) != tuple(members):
        raise ValueError("bundled scoring member order changed during family resolution")
    if "final_test" in splits:
        raise PermissionError(
            "bundled scoring is model-validation/stack-only; final_test uses frozen claims"
        )
    for split in splits:
        if family.kind == "hlt_only":
            if len(members) != 1:
                raise ValueError("HLT baseline scoring currently requires one member")
            member = members[0]
            logits, labels, view = _baseline_logits(
                root,
                member,
                split,
                device=device,
                batch_size=batch_size,
                smoke=smoke,
            )
            indices = np.arange(len(labels), dtype=np.int64)
            member_logits = {member: logits}
            checkpoint_hashes = {
                member: _sha256(root / "runs" / member / "best_model_val.pt")
            }
            config_hashes = {
                member: str(
                    resolve_variant_config(member.split("__seed", 1)[0])[
                        "resolved_config_hash"
                    ]
                )
            }
            source_hash = source_generation_hash(
                family,
                split=split,
                hlt_content_hash=str(view.metadata.get("hlt_content_hash")),
                jet_identity_hash=jet_identity_hash(view.jet_ids[: len(labels)]),
                generator_source_hash="hlt_only",
                checkpoint_hashes={},
                consumer_schema_hashes=(),
            )
            source_provenance = {}
            source_telemetry = {
                "execution_mode": "hlt_only",
                "pseudo_generation_count": 0,
            }
            calibration = {"selected_models_per_bundle": 1}
        else:
            source = _build_bundled_ram_source(
                root,
                family,
                split,
                device=device,
                batch_size=batch_size,
                smoke=smoke,
            )
            try:
                (
                    member_logits,
                    labels,
                    indices,
                    checkpoint_hashes,
                    config_hashes,
                    calibration,
                ) = _score_tagger_bundle(
                    root,
                    members,
                    source,
                    device=device,
                    smoke=smoke,
                )
                view = source.hlt
                source_provenance = _consumer_pseudo_provenance(source)
                source_checkpoint_hashes: dict[str, str] = {}
                for row in source.metadata:
                    source_checkpoint_hashes.update(
                        {str(name): str(value) for name, value in row["checkpoint_hashes"].items()}
                    )
                source_hash = source_generation_hash(
                    family,
                    split=split,
                    hlt_content_hash=source.hlt_content_hash,
                    jet_identity_hash=source.rank_identity_hash,
                    generator_source_hash=source.source_hash,
                    checkpoint_hashes=source_checkpoint_hashes,
                    consumer_schema_hashes=tuple(
                        str(row["consumer_pseudo_schema_hash"])
                        for row in source.metadata
                    ),
                )
                source_telemetry = source.telemetry()
            finally:
                source.close()
        jet_ids = _jet_identity_strings(view, indices)
        common_provenance = {
            **_split_provenance(view),
            **source_provenance,
            "source_family": family.to_dict(),
            "source_generation_hash": source_hash,
            "ordered_scoring_identity_hash": jet_identity_hash(
                tuple(view.jet_ids[int(index)] for index in indices)
            ),
            "pseudo_representations_written_persistently": False,
        }
        artifacts: dict[str, Any] = {}
        for member in members:
            output_dir = root / "logit_predictions" / member
            prediction_path = output_dir / f"{split}.npz"
            metadata_path = output_dir / f"{split}_metadata.json"
            if (prediction_path.exists() or metadata_path.exists()) and not overwrite:
                raise FileExistsError(
                    f"bundled scoring output already exists for {member}/{split}"
                )
            logits = np.asarray(member_logits[member], dtype=np.float32)
            encoded = encode_logit_only_npz(
                logits=logits,
                labels=labels,
                jet_ids=jet_ids,
                source_indices=indices,
            )
            block = LogitPredictionBlock(
                member=member,
                split=split,
                logits=logits,
                labels=np.asarray(labels, dtype=np.int64),
                jet_ids=jet_ids,
                checkpoint_hash=checkpoint_hashes[member],
                resolved_config_hash=config_hashes[member],
                provenance=common_provenance,
            )
            _persist_logit_only_artifact(
                root,
                prediction_path,
                encoded,
                provenance_hash=source_hash,
                run_id=f"bundle:{family.key}:{split}:{member}",
            )
            artifact_report = validate_logit_only_npz(prediction_path)
            metadata = {
                "contract": ABPH_PREDICTION_EXECUTOR_CONTRACT,
                "bundled_scoring_contract": ABPH_BUNDLED_SCORING_CONTRACT,
                "ok": True,
                "variant_name": member,
                "split": split,
                "n_jets": len(labels),
                "prediction_sha256": artifact_report["sha256"],
                "checkpoint_hash": checkpoint_hashes[member],
                "checkpoint_sha256": checkpoint_hashes[member],
                "resolved_config_hash": config_hashes[member],
                "prediction_hash": block.prediction_hash,
                "source_generation_hash": source_hash,
                "source_family": family.to_dict(),
                "logit_only_artifact": artifact_report,
                "provenance": common_provenance,
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": False,
                "final_test_attestation": None,
            }
            _persist_scoring_json(
                root,
                metadata_path,
                metadata,
                provenance_hash=source_hash,
                run_id=f"bundle_metadata:{family.key}:{split}:{member}",
            )
            _update_run_report(
                root / "runs" / member,
                split=split,
                metrics=_detailed_classifier_metrics(logits, labels),
                provenance=common_provenance,
                teacher_free=True,
            )
            artifacts[member] = {
                "prediction_path": str(prediction_path),
                "prediction_sha256": artifact_report["sha256"],
                "metadata_path": str(metadata_path),
                "checkpoint_hash": checkpoint_hashes[member],
                "prediction_hash": block.prediction_hash,
            }
        report = {
            "contract": ABPH_BUNDLED_SCORING_CONTRACT,
            "ok": True,
            "split": split,
            "source_family": family.to_dict(),
            "source_generation_hash": source_hash,
            "members": list(members),
            "n_jets": int(len(labels)),
            "ordered_scoring_identity_hash": common_provenance[
                "ordered_scoring_identity_hash"
            ],
            "source_telemetry": source_telemetry,
            "model_bundle_calibration": calibration,
            "artifacts": artifacts,
            "persisted_array_contract": [
                "float32 logits",
                "int64 labels",
                "ordered string identities",
                "int64 source indices",
            ],
            "pseudo_representations_written_persistently": False,
        }
        report["content_hash"] = canonical_hash(report)
        report_path = (
            root
            / "logit_predictions"
            / "_bundles"
            / family.key
            / f"{split}_bundle_report.json"
        )
        _persist_scoring_json(
            root,
            report_path,
            report,
            provenance_hash=source_hash,
            run_id=f"bundle_report:{family.key}:{split}",
        )


def _write_classifier_predictions(
    root: Path,
    member: str,
    splits: list[str],
    *,
    device: Any,
    batch_size: int,
    shard_size: int,
    smoke: bool,
    overwrite: bool,
) -> None:
    base_name = member.split("__seed", 1)[0]
    tier = variant_spec(base_name).tier
    for split in splits:
        pseudo_source = None
        if tier == "A":
            logits, labels, view = _baseline_logits(
                root, member, split, device=device, batch_size=batch_size, smoke=smoke
            )
            indices = np.arange(len(labels), dtype=np.int64)
        else:
            logits, labels, indices, view, pseudo_source = _tagger_logits(
                root,
                member,
                split,
                device=device,
                batch_size=batch_size,
                shard_size=shard_size,
                smoke=smoke,
                overwrite=overwrite,
            )
        output_dir = root / "logit_predictions" / member
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dir / f"{split}.npz"
        jet_ids = _jet_identity_strings(view, indices)
        np.savez_compressed(
            prediction_path,
            logits=logits.astype(np.float32),
            labels=labels.astype(np.int64),
            jet_ids=jet_ids,
            source_indices=indices.astype(np.int64),
        )
        metrics = _detailed_classifier_metrics(logits, labels)
        checkpoint_hash = _sha256(root / "runs" / member / "best_model_val.pt")
        resolved_hash = resolve_variant_config(base_name)["resolved_config_hash"]
        provenance = _split_provenance(view)
        if pseudo_source is not None:
            provenance.update(_consumer_pseudo_provenance(pseudo_source))
        block = LogitPredictionBlock(
            member=member,
            split=split,
            logits=logits.astype(np.float32),
            labels=labels.astype(np.int64),
            jet_ids=jet_ids,
            checkpoint_hash=checkpoint_hash,
            resolved_config_hash=resolved_hash,
            provenance=provenance,
        )
        metadata = {
            "contract": ABPH_PREDICTION_EXECUTOR_CONTRACT,
            "ok": True,
            "variant_name": member,
            "split": split,
            "n_jets": len(labels),
            "prediction_sha256": _sha256(prediction_path),
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_sha256": checkpoint_hash,
            "resolved_config_hash": resolved_hash,
            "prediction_hash": block.prediction_hash,
            "provenance": provenance,
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "final_test_attestation": (
                {
                    "offline_inputs_loaded": False,
                    "teacher_logits_loaded": False,
                    "offline_targets_loaded": False,
                    "hypothesis_selection_used_offline_target": False,
                    "fusion_fitted_on_final_test": False,
                }
                if split == "final_test"
                else None
            ),
        }
        _atomic_json(output_dir / f"{split}_metadata.json", metadata)
        _update_run_report(
            root / "runs" / member,
            split=split,
            metrics=metrics,
            provenance=metadata["provenance"],
            teacher_free=True,
        )
        if pseudo_source is not None and hasattr(pseudo_source, "close"):
            pseudo_source.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.campaign_root)
    device = resolve_device(args.device)
    splits = [str(value) for value in args.splits]
    if args.members:
        if args.teacher_logits:
            raise ValueError("bundled scoring cannot load teacher logits")
        if not streaming_storage_enabled():
            raise ValueError("--members is restricted to streaming_30gb_v1")
        _write_bundled_classifier_predictions(
            root,
            [str(value) for value in args.members],
            splits,
            device=device,
            batch_size=args.batch_size,
            smoke=args.smoke,
            overwrite=args.overwrite,
        )
        return 0
    assert args.variant is not None
    if args.teacher_logits:
        if args.variant != "A4_offline_part_ceiling":
            raise ValueError("--teacher-logits is restricted to A4")
        _offline_teacher_logits(
            root,
            splits,
            device=device,
            batch_size=args.batch_size,
            smoke=args.smoke,
        )
    elif args.variant in {
        "D1_kt32_mh4_particles",
        "D2_ca32_mh4_particles",
        "E7_shared_root_dual",
    }:
        if streaming_storage_enabled():
            raise ValueError(
                "streaming_30gb_v1 forbids persistent pseudo prediction caches; "
                "frozen taggers and diagnostics generate consumer pseudo views in RAM"
            )
        for split in splits:
            _ensure_pseudo_cache(
                root,
                args.variant,
                split,
                device=device,
                batch_size=args.batch_size,
                shard_size=args.shard_size,
                smoke=args.smoke,
                overwrite=args.overwrite,
            )
    else:
        _write_classifier_predictions(
            root,
            args.variant,
            splits,
            device=device,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            smoke=args.smoke,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
