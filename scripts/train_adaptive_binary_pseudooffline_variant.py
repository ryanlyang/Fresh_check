#!/usr/bin/env python3
"""Train one concrete A-F/G1 adaptive-binary campaign variant."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import (  # noqa: E402
    HLTBaselineTrainConfig,
    build_particle_transformer_classifier,
    require_torch,
    resolve_device,
    train_hlt_baseline,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from jetclass_fresh.dual_view import build_part_inputs_torch  # noqa: E402
from jetclass_fresh.offline_teacher import (  # noqa: E402
    OfflineTeacherTrainConfig,
    train_offline_teacher,
)
from teacher_logit_reco.architecture_view_part import load_cached_offline_view  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_runtime import (  # noqa: E402
    _detailed_eval_metrics,
    _initialize_tagger,
)
from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_LEVEL_CAPACITIES,
    ABPH_RECONSTRUCTOR_VARIANTS,
    AdaptiveBinaryReconstructorModel,
    AdaptiveBinaryTargetBatchSource,
    PseudoViewInputs,
    ReconstructorCurriculumConfig,
    ReconstructorTrainerConfig,
    canonical_hash,
    build_variant_hierarchy_aware_tagger,
    load_selected_reconstructor,
    package_trainable_pseudo_views,
    reconstructor_runtime_provenance,
    reconstructor_step,
    resolve_variant_config,
    train_reconstructor_curriculum,
    variant_spec,
)


def _offline_particle_oracle_pseudo(
    pseudo: PseudoViewInputs,
    targets,
    *,
    device,
) -> PseudoViewInputs:
    """Replace rendered particles with the exact offline target particle view."""

    torch = require_torch()
    arrays = dict(pseudo.arrays)
    canonical = torch.as_tensor(
        targets.particle_targets, device=device, dtype=torch.float32
    )
    particle_mask = torch.as_tensor(
        targets.particle_mask, device=device
    ).bool()
    views = int(arrays["hypothesis_latent"].shape[1])
    canonical = canonical[:, None].expand(-1, views, -1, -1).contiguous()
    expanded_mask = particle_mask[:, None].expand(-1, views, -1).contiguous()
    for hierarchy_name in pseudo.hierarchy_names:
        prefix = f"particle__{hierarchy_name}__"
        side = torch.zeros(
            (*expanded_mask.shape, 6), device=device, dtype=canonical.dtype
        )
        side[..., 2] = expanded_mask.to(canonical.dtype)
        arrays[prefix + "canonical_features"] = canonical
        arrays[prefix + "mask"] = expanded_mask
        arrays[prefix + "side_channels"] = side
        arrays[prefix + "uncertainty"] = torch.zeros_like(
            expanded_mask, dtype=canonical.dtype
        )
        arrays[prefix + "group_indices"] = torch.zeros_like(
            expanded_mask, dtype=torch.long
        )
        local = torch.arange(
            canonical.shape[2], device=device, dtype=torch.long
        )[None, None].expand_as(expanded_mask)
        arrays[prefix + "local_slot_indices"] = local
        arrays[prefix + "slot_hidden"] = torch.zeros(
            (*expanded_mask.shape, arrays[prefix + "slot_hidden"].shape[-1]),
            device=device,
            dtype=canonical.dtype,
        )
    result = PseudoViewInputs(
        arrays=arrays,
        view_names=pseudo.view_names,
        hierarchy_names=pseudo.hierarchy_names,
        frontier_depths=pseudo.frontier_depths,
        diagnostics={
            **dict(pseudo.diagnostics),
            "model_val_oracle_offline_particles_substituted": True,
        },
    )
    result.validate()
    return result


def _evaluate_true_offline_particle_branch(
    args: argparse.Namespace,
    output_dir: Path,
    source: AdaptiveBinaryTargetBatchSource,
    *,
    resolved_config_hash: str,
) -> tuple[dict, dict]:
    """Run D6 through the real E0 pseudo branch with oracle particles."""

    torch = require_torch()
    root = Path(args.campaign_root)
    device = resolve_device(str(args.device))
    reconstructor = load_selected_reconstructor(
        root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt",
        variant_name="D1_kt32_mh4_particles",
        device=device,
        smoke=bool(args.smoke),
    )
    tagger = build_variant_hierarchy_aware_tagger(
        "E0_pseudo_only", num_classes=10, smoke=bool(args.smoke)
    ).to(device)
    initialization = _initialize_tagger(
        tagger,
        root=root,
        variant_name="D6_true_offline_particles",
        initialization="warm_start_hlt_and_offline_pseudo",
    )
    reconstructor.eval()
    tagger.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in source.iter_epoch():
            tokens = batch["hlt_tokens"].to(device)
            mask = batch["hlt_mask"].to(device).bool()
            deployed = reconstructor.deploy(tokens, mask, evaluation_seed=24731)
            pseudo = _offline_particle_oracle_pseudo(
                package_trainable_pseudo_views(deployed),
                batch["targets"],
                device=device,
            )
            output = tagger(tokens, mask, pseudo)
            logits.append(output.logits.detach().cpu().numpy())
            labels.append(batch["labels"].numpy())
    if not logits:
        raise RuntimeError("D6 oracle pseudo-particle evaluation produced no batches")
    values = np.concatenate(logits, axis=0)
    truth = np.concatenate(labels, axis=0)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    picked = probabilities[np.arange(len(truth)), truth].clip(1.0e-12, 1.0)
    metrics = {
        "available": True,
        "accuracy": float((values.argmax(axis=1) == truth).mean()),
        "loss": float(-np.log(picked).mean()),
        "cross_entropy": float(-np.log(picked).mean()),
        "n_jets": int(len(truth)),
        **_detailed_eval_metrics(values, truth),
    }
    checkpoint_payload = {
        "checkpoint_contract": "adaptive_binary_pseudooffline_oracle_diagnostic_v2",
        "checkpoint_role": "best_model_val",
        "variant_name": "D6_true_offline_particles",
        "resolved_variant_config_hash": str(resolved_config_hash),
        "model_state_dict": tagger.state_dict(),
        "source_checkpoint_hashes": {
            "A0_hlt_part": _sha256(
                root / "runs" / "A0_hlt_part" / "best_model_val.pt"
            ),
            "A4_offline_part_ceiling": _sha256(
                root / "runs" / "A4_offline_part_ceiling" / "best_model_val.pt"
            ),
            "D1_kt32_mh4_particles": _sha256(
                root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt"
            ),
        },
        "oracle_model_val_only": True,
        "oracle_particles_actually_routed_through_pseudo_branch": True,
        "final_test_loaded": False,
        "teacher_logits_loaded": False,
    }
    torch.save(checkpoint_payload, output_dir / "best_model_val.pt")
    return metrics, initialization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed-index", type=int, default=1)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ABPH_BATCH_SIZE", "64")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("ABPH_NUM_WORKERS", "0")))
    parser.add_argument("--smoke", action="store_true", default=os.environ.get("ABPH_SMOKE", "0") == "1")
    parser.add_argument("--maximum-updates", type=int, default=None)
    return parser


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_provenance(view, *, target=None) -> dict:
    metadata = view.metadata
    labels = view.labels
    import hashlib
    import numpy as np
    from jetclass_fresh.hlt_cache import jet_identity_hash
    from jetclass_fresh.jetclass_data import LABEL_NAMES

    result = {
        "source_manifest_hash": metadata.get("source_manifest_hash"),
        "jet_identity_hash": jet_identity_hash(view.jet_ids),
        "label_hash": hashlib.sha256(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).hexdigest(),
        "class_mapping_hash": canonical_hash({"label_names": list(LABEL_NAMES)}),
        "hlt_content_hash": metadata.get("hlt_content_hash"),
        "hlt_profile": metadata.get("hlt_profile"),
        "hlt_profile_version": metadata.get("hlt_profile_version"),
        "hlt_degradation_strength": metadata.get("hlt_degradation_strength"),
        "hlt_params_hash": metadata.get("hlt_params_hash"),
    }
    if target is not None:
        result.update(
            {
                "offline_cache_content_hash": target.get("offline_content_hash"),
                "hierarchy_target_content_hash": target.get("target_content_hash"),
                "hierarchy_target_schema_hash": canonical_hash(
                    {
                        "root": target.get("root_feature_names"),
                        "group": target.get("group_feature_names"),
                        "particle": target.get("particle_target_names"),
                    }
                ),
                "grouping_algorithm_hash": canonical_hash(target.get("layout", {})),
                "root_ledger_schema_hash": canonical_hash(
                    {"root_feature_names": target.get("root_feature_names")}
                ),
                "normalization_hash": "identity_physical_units_v1",
            }
        )
    return result


def _selected_classifier_metrics(
    model,
    checkpoint: Path,
    view,
    *,
    device: str,
    batch_size: int,
    smoke: bool,
) -> dict:
    torch = require_torch()
    resolved_device = resolve_device(device)
    try:
        payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    except TypeError:  # pragma: no cover - older research PyTorch
        payload = torch.load(checkpoint, map_location=resolved_device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"selected classifier checkpoint {checkpoint} lacks model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(resolved_device).eval()
    maximum = (
        min(len(view.labels), int(os.environ.get("ABPH_SMOKE_JETS", "128")))
        if smoke
        else len(view.labels)
    )
    logits = []
    with torch.no_grad():
        for start in range(0, maximum, int(batch_size)):
            stop = min(start + int(batch_size), maximum)
            tokens = torch.as_tensor(view.tokens[start:stop], device=resolved_device)
            mask = torch.as_tensor(view.mask[start:stop], device=resolved_device).bool()
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
    labels = np.asarray(view.labels[:maximum], dtype=np.int64)
    details = _detailed_eval_metrics(values, labels)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    picked = probabilities[np.arange(labels.shape[0]), labels].clip(1.0e-12, 1.0)
    return {
        "available": True,
        "accuracy": float((values.argmax(axis=1) == labels).mean()),
        "loss": float(-np.log(picked).mean()),
        "cross_entropy": float(-np.log(picked).mean()),
        "n_jets": int(labels.shape[0]),
        **details,
    }


def _train_baseline(args: argparse.Namespace, resolved: dict, output_dir: Path) -> dict:
    root = Path(args.campaign_root)
    variant = str(args.variant)
    hlt_config = dict(resolved["model"]["hlt_part"])
    enabled = hlt_config.pop("enabled", None)
    del enabled
    hlt_config.pop("capacity_match_target", None)
    hlt_config.pop("capacity_match_policy", None)
    hlt_config.pop("capacity_control_kind", None)
    model = build_particle_transformer_classifier(
        num_classes=10,
        model_size="base",
        overrides=hlt_config,
    )
    seed = 24731 + 1009 * (int(args.seed_index) - 1)
    initialization_checkpoint_hash = None
    if variant == "A1_hlt_schedule_control":
        checkpoint = root / "runs" / "A0_hlt_part" / "best_model_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("A1 requires the selected A0 checkpoint")
        import torch

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older research PyTorch
            payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        initialization_checkpoint_hash = _sha256(checkpoint)
    epochs = int(os.environ.get("ABPH_TAGGER_EPOCHS", "20"))
    if args.smoke:
        epochs = 1
    if variant == "A4_offline_part_ceiling":
        report = train_offline_teacher(
            OfflineTeacherTrainConfig(
                output_dir=str(output_dir),
                manifest_path=str(root / "inputs" / "split_manifest" / "split_manifest.json.gz"),
                data_dir=os.environ.get("ABPH_DATA_DIR", "/home/ryreu/atlas/PracticeTagging/data/jetclass_part1"),
                seed=seed,
                batch_size=int(args.batch_size),
                epochs=epochs,
                num_workers=int(args.num_workers),
                device=args.device,
                model_size="base",
                max_train_batches=(1 if args.smoke else None),
                max_val_batches=(1 if args.smoke else None),
            ),
            model=model,
        )
        # Offline upper ceiling is intentionally not a deployable final claim.
        hlt_view = load_cached_hlt_view(root / "inputs" / "hlt_cache", "model_val", verify_hash=True)
    else:
        report = train_hlt_baseline(
            HLTBaselineTrainConfig(
                output_dir=str(output_dir),
                cache_dir=str(root / "inputs" / "hlt_cache"),
                seed=seed,
                batch_size=int(args.batch_size),
                epochs=epochs,
                lr=(2.0e-4 if variant == "A1_hlt_schedule_control" else 1.0e-3),
                weight_decay=(0.01 if variant == "A1_hlt_schedule_control" else 1.0e-4),
                adam_beta1=0.9,
                adam_beta2=(0.95 if variant == "A1_hlt_schedule_control" else 0.999),
                lr_schedule=(
                    "cosine_per_update"
                    if variant == "A1_hlt_schedule_control"
                    else "constant"
                ),
                cosine_eta_min=(
                    1.0e-5 if variant == "A1_hlt_schedule_control" else 0.0
                ),
                num_workers=int(args.num_workers),
                device=args.device,
                model_size="base",
                max_train_batches=(1 if args.smoke else None),
                max_val_batches=(1 if args.smoke else None),
                early_stop_patience=(
                    epochs + 1 if variant == "A1_hlt_schedule_control" else 5
                ),
            ),
            model=model,
        )
        hlt_view = load_cached_hlt_view(root / "inputs" / "hlt_cache", "model_val", verify_hash=True)
    checkpoint = output_dir / "best_model_val.pt"
    evaluation_view = (
        load_cached_offline_view(
            root / "inputs" / "offline_cache", "model_val", verify_hash=True
        )
        if variant == "A4_offline_part_ceiling"
        else hlt_view
    )
    metrics = _selected_classifier_metrics(
        model,
        checkpoint,
        evaluation_view,
        device=args.device,
        batch_size=int(args.batch_size),
        smoke=bool(args.smoke),
    )
    return {
        "contract": "adaptive_binary_pseudooffline_baseline_run_v1",
        "ok": True,
        "variant_name": variant,
        "variant": resolved["variant"],
        "resolved_variant_config_hash": resolved["resolved_config_hash"],
        "selected_checkpoint_hash": _sha256(checkpoint),
        "best_model_val_checkpoint_sha256": _sha256(checkpoint),
        "source_git_commit": "recorded_by_slurm_run_config",
        "source_status_hash": "recorded_by_slurm_run_config",
        "metrics": {"model_val": metrics},
        "provenance": {
            "model_val": _split_provenance(hlt_view),
            "artifact": {
                "resolved_variant_config_hash": resolved["resolved_config_hash"],
                "selected_checkpoint_hash": _sha256(checkpoint),
                "source_git_commit": "recorded_by_slurm_run_config",
                "source_status_hash": "recorded_by_slurm_run_config",
            },
        },
        "diagnostics": {
            "model_val": {
                "offline_inputs_loaded": variant == "A4_offline_part_ceiling",
                "teacher_logits_loaded": False,
            },
            "capacity": {
                "trainable_parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                ),
                "control_kind": resolved["model"]["hlt_part"].get(
                    "capacity_control_kind"
                ),
                "exact_parameter_match_claimed": False,
            },
            "schedule_contract": (
                {
                    "matched_to": "F0_primary_dual_hierarchy_joint",
                    "initialization": "selected_A0_warm_start",
                    "initialization_checkpoint_hash": initialization_checkpoint_hash,
                    "hlt_branch_trainable_from_epoch": 0,
                    "hlt_branch_trainable_through_final_epoch": True,
                    "epoch_budget": epochs,
                    "optimizer": "AdamW",
                    "learning_rate": 2.0e-4,
                    "adam_betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                    "lr_schedule": "cosine_per_update",
                    "cosine_eta_min": 1.0e-5,
                    "gradient_clip_norm": 1.0,
                    "pseudo_branch_present": False,
                    "reconstructor_present": False,
                }
                if variant == "A1_hlt_schedule_control"
                else None
            ),
        },
        "training_report": report,
    }


def _maximum_capacity(resolved: dict) -> int:
    hierarchy = resolved["model"]["hierarchy"]
    if not bool(hierarchy.get("enabled")):
        return 1
    capacities = tuple(int(value) for value in hierarchy.get("capacities", ()))
    if capacities == (8,):
        return 8
    return max(capacities)


def _load_selected_hlt_encoder(
    model: AdaptiveBinaryReconstructorModel,
    checkpoint: Path,
) -> dict:
    """Load the selected A0 ParT into a production reconstructor encoder."""

    torch = require_torch()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older research PyTorch
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"HLT warm-start checkpoint {checkpoint} lacks model_state_dict")
    reference = getattr(model.hlt_encoder, "reference_model", None)
    if reference is None:
        if model.smoke:
            return {
                "loaded": False,
                "smoke_native_encoder": True,
                "source_checkpoint_sha256": _sha256(checkpoint),
            }
        raise TypeError("production reconstructor lacks a Weaver reference HLT encoder")
    result = reference.load_state_dict(state, strict=True)
    return {
        "loaded": True,
        "source_variant": "A0_hlt_part",
        "source_checkpoint_sha256": _sha256(checkpoint),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "strict": True,
    }


def _write_oracle_reference_report(
    args: argparse.Namespace,
    resolved: dict,
    output_dir: Path,
) -> dict:
    """Write model-val-only B4/D6 ceilings without pretending they deploy."""

    import torch

    root = Path(args.campaign_root)
    variant = str(args.variant)
    grouping = str(resolved["model"]["hierarchy"].get("grouping", "exclusive_kt"))
    source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        target_cache_dir=root / "targets",
        split="model_val",
        grouping=grouping,
        batch_size=max(1, int(args.batch_size)),
        shuffle_shards=False,
        seed=24731,
        maximum_batches=(1 if args.smoke else None),
    )
    checkpoint = output_dir / "best_model_val.pt"
    if variant == "D6_true_offline_particles":
        metrics, initialization = _evaluate_true_offline_particle_branch(
            args,
            output_dir,
            source,
            resolved_config_hash=resolved["resolved_config_hash"],
        )
        metrics["diagnostics"] = {
            "oracle_model_val_only": True,
            "oracle_quantity": "true_offline_particles_routed_through_pseudo_branch",
            "copied_A4_metrics": False,
            "actual_pseudo_branch_forward_pass": True,
            "final_test_loaded": False,
        }
    else:
        initialization = None
        checkpoint_payload = {
            "checkpoint_contract": "adaptive_binary_pseudooffline_oracle_diagnostic_v1",
            "checkpoint_role": "best_model_val",
            "variant_name": variant,
            "resolved_variant_config_hash": resolved["resolved_config_hash"],
            "model_state_dict": {},
            "oracle_model_val_only": True,
            "final_test_loaded": False,
            "teacher_logits_loaded": False,
        }
        torch.save(checkpoint_payload, checkpoint)
        metrics = {
            "available": True,
            "loss": 0.0,
            "accuracy": None,
            "n_jets": len(source.hlt_view.labels),
            "diagnostics": {
                "oracle_model_val_only": True,
                "oracle_quantity": "true_offline_root_ledger",
                "final_test_loaded": False,
            },
        }
    checkpoint_hash = _sha256(checkpoint)
    provenance = _split_provenance(source.hlt_view, target=source.metadata)
    return {
        "contract": "adaptive_binary_pseudooffline_oracle_diagnostic_v1",
        "ok": True,
        "variant_name": variant,
        "variant": resolved["variant"],
        "resolved_variant_config_hash": resolved["resolved_config_hash"],
        "selected_checkpoint_hash": checkpoint_hash,
        "source_git_commit": "recorded_by_slurm_run_config",
        "source_status_hash": "recorded_by_slurm_run_config",
        "metrics": {"model_val": metrics},
        "provenance": {
            "model_val": provenance,
            "artifact": {
                "resolved_variant_config_hash": resolved["resolved_config_hash"],
                "selected_checkpoint_hash": checkpoint_hash,
                "source_git_commit": "recorded_by_slurm_run_config",
                "source_status_hash": "recorded_by_slurm_run_config",
            },
        },
        "diagnostics": {
            "model_val": metrics["diagnostics"],
            "initialization": initialization,
        },
    }


def _train_reconstructor(args: argparse.Namespace, resolved: dict, output_dir: Path) -> dict:
    root = Path(args.campaign_root)
    grouping = str(resolved["model"]["hierarchy"].get("grouping", "exclusive_kt"))
    renderer_enabled = bool(resolved["model"]["renderer"].get("enabled"))
    distribution_enabled = bool(resolved["model"]["distribution"].get("enabled"))
    if distribution_enabled and not renderer_enabled:
        raise ValueError("distribution variant lacks its required renderer")
    model = AdaptiveBinaryReconstructorModel(
        hierarchy_names=(grouping,), variant_name=args.variant, smoke=bool(args.smoke)
    )
    tier = str(resolved["variant"]["tier"])
    warm_start = {"loaded": False, "source_variant": None, "parameter_tensors": 0}
    dependencies = tuple(str(value) for value in resolved["variant"].get("dependencies", ()))
    source_variant = next(
        (
            value
            for value in dependencies
            if variant_spec(value).tier in {"B", "C"}
        ),
        None,
    )
    if source_variant is None and tier == "B":
        warm_start = _load_selected_hlt_encoder(
            model,
            root / "runs" / "A0_hlt_part" / "best_model_val.pt",
        )
    elif source_variant is not None:
        import torch

        source_path = root / "runs" / source_variant / "best_model_val.pt"
        try:
            payload = torch.load(source_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older research PyTorch
            payload = torch.load(source_path, map_location="cpu")
        source_state = payload.get("model_state_dict")
        if not isinstance(source_state, dict):
            raise ValueError(f"warm-start checkpoint {source_path} lacks model_state_dict")
        current = model.state_dict()
        copied = 0
        for name, value in source_state.items():
            if name not in current or current[name].shape != value.shape:
                continue
            if tier == "C" and not name.startswith(("hlt_encoder.", "root_predictor.")):
                continue
            if tier == "B" and not name.startswith(("hlt_encoder.", "root_predictor.")):
                continue
            current[name] = value
            copied += 1
        if copied == 0:
            raise RuntimeError(f"{args.variant} copied no tensors from {source_variant}")
        model.load_state_dict(current, strict=True)
        warm_start = {
            "loaded": True,
            "source_variant": source_variant,
            "source_checkpoint_sha256": _sha256(source_path),
            "parameter_tensors": copied,
        }
    train_source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        target_cache_dir=root / "targets",
        split="model_train",
        grouping=grouping,
        batch_size=int(args.batch_size),
        shuffle_shards=True,
        seed=24731 + 1009 * (int(args.seed_index) - 1),
        maximum_batches=(1 if args.smoke else None),
    )
    val_source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        target_cache_dir=root / "targets",
        split="model_val",
        grouping=grouping,
        batch_size=int(args.batch_size),
        shuffle_shards=False,
        seed=24732,
        maximum_batches=(1 if args.smoke else int(os.environ.get("ABPH_MAX_VAL_BATCHES", "0")) or None),
    )
    target_metadata = train_source.metadata
    provenance = reconstructor_runtime_provenance(
        variant_name=args.variant,
        target_metadata=target_metadata,
        hlt_metadata=train_source.hlt_view.metadata,
    )
    smoke_updates = 1 if args.smoke else None
    inherited_updates = 1 if tier in {"C", "D"} else None
    inherited_hierarchy_updates = 1 if tier == "D" else None
    maximum_capacity = _maximum_capacity(resolved)
    curriculum = ReconstructorCurriculumConfig(
        root_updates=smoke_updates or inherited_updates or int(os.environ.get("ABPH_ROOT_UPDATES", "150000")),
        hierarchy_updates_per_depth=smoke_updates or inherited_hierarchy_updates or int(os.environ.get("ABPH_HIERARCHY_UPDATES", "80000")),
        renderer_updates=smoke_updates or int(os.environ.get("ABPH_RENDERER_UPDATES", "200000")),
        distribution_updates=smoke_updates or int(os.environ.get("ABPH_DISTRIBUTION_UPDATES", "200000")),
        evaluation_interval=1 if args.smoke else int(os.environ.get("ABPH_EVAL_INTERVAL", "2000")),
        maximum_capacity=maximum_capacity,
        hierarchy_capacities=(
            ()
            if maximum_capacity == 1
            else tuple(
                int(value)
                for value in resolved["model"]["hierarchy"].get("capacities", ())
            )
        ),
        renderer_enabled=renderer_enabled,
        distribution_enabled=distribution_enabled,
    )
    batch_size = int(args.batch_size)
    if 1024 % batch_size or 512 % batch_size:
        raise ValueError("batch size must divide both locked effective batch sizes 1024 and 512")
    trainer = ReconstructorTrainerConfig(
        output_dir=str(output_dir),
        seed=24731 + 1009 * (int(args.seed_index) - 1),
        device=args.device,
        amp=not args.smoke,
        gradient_accumulation_steps=1,
        root_hierarchy_gradient_accumulation_steps=(1 if args.smoke else 1024 // batch_size),
        renderer_distribution_gradient_accumulation_steps=(1 if args.smoke else 512 // batch_size),
        root_hierarchy_effective_batch_size=(batch_size if args.smoke else 1024),
        renderer_distribution_effective_batch_size=(batch_size if args.smoke else 512),
        curriculum=curriculum,
    )
    report = train_reconstructor_curriculum(
        model,
        model.module_groups(),
        train_source,
        val_source.iter_epoch,
        reconstructor_step,
        trainer,
        provenance=provenance,
        maximum_optimizer_updates=args.maximum_updates,
    )
    if not report["ok"]:
        return {**report, "variant_name": args.variant, "variant": resolved["variant"]}
    checkpoint = output_dir / "best_model_val.pt"
    curves = json.loads((output_dir / "training_curves.json").read_text(encoding="utf-8"))
    last_validation = curves["evaluations"][-1]["model_val_rollout"]
    val_view = val_source.hlt_view
    report.update(
        {
            "variant_name": args.variant,
            "variant": resolved["variant"],
            "resolved_variant_config_hash": resolved["resolved_config_hash"],
            "selected_checkpoint_hash": _sha256(checkpoint),
            "source_git_commit": provenance["source_git_commit"],
            "source_status_hash": provenance["source_status_hash"],
            "metrics": {
                "model_val": {
                    "available": True,
                    "loss": last_validation["selection_score"],
                    "n_jets": last_validation["n_jets"],
                    "diagnostics": last_validation.get("metrics", {}),
                }
            },
            "provenance": {
                "model_val": _split_provenance(val_view, target=val_source.metadata),
                "artifact": {
                    "resolved_variant_config_hash": resolved["resolved_config_hash"],
                    "selected_checkpoint_hash": _sha256(checkpoint),
                    "source_git_commit": provenance["source_git_commit"],
                    "source_status_hash": provenance["source_status_hash"],
                },
            },
            "initialization": warm_start,
        }
    )
    if str(args.variant) == "C8_unconstrained_split_control":
        report.setdefault("diagnostics", {})["unconstrained_control_scope"] = {
            "unconstrained_auxiliary_child_heads_trained": True,
            "deployment_rollout_remains_compiler_constrained": True,
            "complete_unconstrained_rollout_claimed": False,
            "selection_eligible": False,
        }
    if model.sample_root:
        import torch

        diagnostic_batch = next(iter(val_source.iter_epoch()))
        with torch.no_grad():
            roots = model.sample_compiled_roots(
                diagnostic_batch["hlt_tokens"],
                diagnostic_batch["hlt_mask"],
                count=4,
                seed=24731,
            )
        ledgers = torch.stack([item.root_ledger for item in roots], dim=1)
        report.setdefault("diagnostics", {})["sampled_root_model_val"] = {
            "sample_count": 4,
            "shared_root_enforced": False,
            "root_sampling_source": "calibrated_semantic_root_heads",
            "downstream_hierarchy_rollout_performed": False,
            "tagging_claim": False,
            "mean_feature_variance": float(
                ledgers.to(torch.float64).var(dim=1, unbiased=False).mean().cpu()
            ),
            "maximum_feature_variance": float(
                ledgers.to(torch.float64).var(dim=1, unbiased=False).max().cpu()
            ),
            "offline_inputs_loaded": False,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolved = resolve_variant_config(args.variant)
    spec = variant_spec(args.variant)
    output_name = args.variant if int(args.seed_index) == 1 else f"{args.variant}__seed{args.seed_index}"
    output_dir = Path(args.campaign_root) / "runs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.tier == "A":
        report = _train_baseline(args, resolved, output_dir)
    elif args.variant in {"B4_oracle_root_diagnostic", "D6_true_offline_particles"}:
        report = _write_oracle_reference_report(args, resolved, output_dir)
    elif args.variant in ABPH_RECONSTRUCTOR_VARIANTS or spec.tier == "D":
        report = _train_reconstructor(args, resolved, output_dir)
    else:
        from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_runtime import train_tagger_variant

        report = train_tagger_variant(args, resolved, output_dir)
    _atomic_json(output_dir / "run_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
