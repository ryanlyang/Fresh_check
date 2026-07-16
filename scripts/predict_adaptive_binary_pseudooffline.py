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
    DeployablePseudoViewCacheConfig,
    FrozenPseudoBatchSource,
    LogitPredictionBlock,
    PseudoViewInputs,
    build_shared_root_dual_reconstructor,
    build_variant_hierarchy_aware_tagger,
    canonical_hash,
    describe_reconstructor_model,
    generate_deployable_pseudo_view_cache,
    load_hlt_prediction_source,
    load_selected_reconstructor,
    package_deployable_pseudo_views,
    require_deployable_pseudo_view_cache,
    resolve_variant_config,
    variant_spec,
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
    parser.add_argument("--variant", required=True)
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
    torch = require_torch()
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path} is not a mapping")
    return payload


def _state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    raise KeyError("checkpoint has no model state")


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
        reconstructor_state = payload.get("reconstructor_state_dict")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    base_name = member.split("__seed", 1)[0]
    resolved = resolve_variant_config(base_name)
    model = build_variant_hierarchy_aware_tagger(base_name, smoke=smoke).to(device)
    checkpoint = root / "runs" / member / "best_model_val.pt"
    checkpoint_payload = _torch_load(checkpoint, device=device)
    model.load_state_dict(_state(checkpoint_payload), strict=True)
    model.eval()
    run_id = str(resolved["variant"]["run_id"])
    if base_name.startswith("F") and isinstance(
        checkpoint_payload.get("reconstructor_state_dict"), Mapping
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
    cache_dirs = tuple(
        _ensure_pseudo_cache(
            root,
            source,
            split,
            device=device,
            batch_size=batch_size,
            shard_size=shard_size,
            smoke=smoke,
            overwrite=overwrite,
        )
        for source in sources
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
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(indices), source.hlt


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
        if tier == "A":
            logits, labels, view = _baseline_logits(
                root, member, split, device=device, batch_size=batch_size, smoke=smoke
            )
            indices = np.arange(len(labels), dtype=np.int64)
        else:
            logits, labels, indices, view = _tagger_logits(
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.campaign_root)
    device = resolve_device(args.device)
    splits = [str(value) for value in args.splits]
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
