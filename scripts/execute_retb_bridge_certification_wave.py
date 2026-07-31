#!/usr/bin/env python3
"""Certify complete Stage-E targets and materialize the joint score table."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_certification import (  # noqa: E402
    build_bridge_candidate_eligibility,
    certify_bridge_content,
    certify_offline_noninferiority,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_selection import (  # noqa: E402
    select_joint_bridge_coordinates,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (  # noqa: E402
    evaluate_fusion,
    make_fusion_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    validate_stage_e_template_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


INDEX_CONTRACT = "retb_bridge_certification_index_v3"
SCORE_TABLE_CONTRACT = "retb_joint_bridge_coordinate_score_table_v3"
READOUT_SCORE_CONTRACT = "retb_bridge_coordinate_readout_score_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _slug(template: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(template))[:16]


def _candidate_root(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    template: Mapping[str, Any],
) -> Path:
    return (
        root
        / "runs"
        / "stage_e"
        / "targets"
        / shape
        / expert
        / f"seed_{seed}"
        / _slug(template)
    )


def _parent(
    root: Path, *, shape: str, expert: str, seed: int
) -> Path:
    return (
        root
        / "selection"
        / "stage_e_parents"
        / shape
        / expert
        / f"seed_{seed}"
    )


def _pilot_output(
    root: Path, *, shape: str, expert: str, seed: int
) -> Path:
    identity = f"pilot_t0:{shape}:{expert}:seed_{seed}"
    return (
        root
        / "runs"
        / "stage_e"
        / "pilots"
        / canonical_sha256(identity)[:20]
    )


def _normalize(values: np.ndarray, artifact: Mapping[str, Any]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    mean = np.asarray(artifact["mean"], dtype=np.float32)
    scale = np.maximum(
        np.asarray(artifact["standard_deviation"], dtype=np.float32), 1.0e-4
    )
    if data.shape[1:] != mean.shape:
        raise ValueError("bridge certification normalizer shape differs")
    result = (data - mean[None]) / scale[None]
    if not np.isfinite(result).all():
        raise FloatingPointError("bridge certification input is nonfinite")
    return result


def _metric_row(
    *, logits: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any]:
    metrics = evaluate_classification(logits, labels, split="val_design")
    return {
        "seed": int(seed),
        "accuracy": metrics["accuracy"],
        "cross_entropy": metrics["cross_entropy"],
        "per_class_efficiency": metrics["per_class_efficiency"],
    }


def _select_template(
    root: Path,
    *,
    shape: str,
    expert: str,
    mode: str,
    templates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    candidates = []
    for template in templates:
        if template["target_mode"] != mode:
            continue
        metrics = [
            load_hashed_json(
                _candidate_root(
                    root,
                    shape=shape,
                    expert=expert,
                    seed=seed,
                    template=template,
                )
                / "val_stop_metrics.json"
            )["metrics"]
            for seed in (101, 202, 303)
        ]
        candidates.append(
            {
                "template": dict(template),
                "mean_accuracy": float(
                    np.mean([row["accuracy"] for row in metrics])
                ),
                "mean_cross_entropy": float(
                    np.mean([row["cross_entropy"] for row in metrics])
                ),
            }
        )
    if not candidates:
        raise ValueError(f"no candidate templates for {mode}")
    maximum = max(row["mean_accuracy"] for row in candidates)
    return min(
        (
            row
            for row in candidates
            if maximum - row["mean_accuracy"] <= 0.0001
        ),
        key=lambda row: (
            row["mean_cross_entropy"],
            canonical_sha256(row["template"]),
        ),
    )["template"]


def _t0_arrays(
    root: Path, *, shape: str, expert: str, seed: int, split: str
) -> dict[str, np.ndarray]:
    return _npz(
        _pilot_output(root, shape=shape, expert=expert, seed=seed)
        / f"{split}_coordinate_arrays.npz"
    )


def _coordinate_arrays(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    mode: str,
    selected_templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    split: str,
) -> dict[str, np.ndarray]:
    return _npz(
        _coordinate_array_path(
            root,
            shape=shape,
            expert=expert,
            seed=seed,
            mode=mode,
            selected_templates=selected_templates,
            split=split,
        )
    )


def _coordinate_array_path(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    mode: str,
    selected_templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    split: str,
) -> Path:
    if mode == "T0_PURE":
        return (
            _pilot_output(root, shape=shape, expert=expert, seed=seed)
            / f"{split}_coordinate_arrays.npz"
        )
    return (
        _candidate_root(
            root,
            shape=shape,
            expert=expert,
            seed=seed,
            template=selected_templates[(shape, expert, mode)],
        )
        / f"{split}_coordinate_arrays.npz"
    )


def _coordinate_normalizer(
    root: Path,
    *,
    shape: str,
    expert: str,
    mode: str,
    pipeline_seed: int = 101,
    selected_templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if mode == "T2_PROJECT":
        return load_hashed_json(
            _candidate_root(
                root,
                shape=shape,
                expert=expert,
                seed=pipeline_seed,
                template=selected_templates[(shape, expert, mode)],
            )
            / "bridge_normalizer.json"
        )
    return load_hashed_json(
        _parent(
            root,
            shape=shape,
            expert=expert,
            seed=pipeline_seed,
        )
        / "target_normalizer.json"
    )


def _fusion_flops(split_arrays: Mapping[str, Any]) -> int:
    banks = split_arrays["token_banks"]
    sequence = 1 + sum(int(values.shape[1]) for values in banks.values())
    projections = sum(
        2 * int(values.shape[1]) * int(values.shape[2]) * 128
        for values in banks.values()
    )
    width = 128
    # QKV/output, QK+AV, and expansion-four feed-forward multiply-add FLOPs.
    block = (
        8 * sequence * width * width
        + 4 * sequence * sequence * width
        + 16 * sequence * width * width
    )
    return int(projections + 3 * block + 2 * width * 10)


def _predictor_profile(
    root: Path,
    *,
    shape: str,
    expert: str,
    mode: str,
    pipeline_seed: int = 101,
    selected_templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, int]:
    parent_arrays = _npz(
        _parent(
            root,
            shape=shape,
            expert=expert,
            seed=pipeline_seed,
        )
        / "val_design_pilot_dataset.npz"
    )
    if mode == "T0_PURE":
        checkpoint = (
            _pilot_output(
                root,
                shape=shape,
                expert=expert,
                seed=pipeline_seed,
            )
            / "best_model_val.pt"
        )
        state_key = "model_state_dict"
        output_dimension = int(parent_arrays["target_tokens"].shape[-1])
    else:
        template = selected_templates[(shape, expert, mode)]
        checkpoint = (
            _candidate_root(
                root,
                shape=shape,
                expert=expert,
                seed=pipeline_seed,
                template=template,
            )
            / "best_model_val.pt"
        )
        state_key = "predictor_state_dict"
        output_dimension = int(
            template["bridge_dimension"]
            or parent_arrays["target_tokens"].shape[-1]
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get(state_key)
    if not isinstance(state, Mapping):
        raise ValueError("bridge predictor checkpoint state differs")
    parameter_count = sum(
        int(value.numel())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    k = int(parent_arrays["target_tokens"].shape[1])
    d = int(parent_arrays["target_tokens"].shape[2])
    target_hlt = parent_arrays[f"hlt_tokens_{expert}"]
    target_hlt = target_hlt[0] if target_hlt.ndim == 4 else target_hlt
    evidence_projection = 2 * d * (
        int(target_hlt.shape[1]) * int(target_hlt.shape[2])
        + sum(
            int(
                (
                    parent_arrays[f"hlt_tokens_{name}"][0]
                    if parent_arrays[f"hlt_tokens_{name}"].ndim == 4
                    else parent_arrays[f"hlt_tokens_{name}"]
                ).shape[1]
            )
            * int(
                (
                    parent_arrays[f"hlt_tokens_{name}"][0]
                    if parent_arrays[f"hlt_tokens_{name}"].ndim == 4
                    else parent_arrays[f"hlt_tokens_{name}"]
                ).shape[2]
            )
            for name in EXPERT_ORDER
        )
        + int(
            (
                parent_arrays["unbiased_particle_states"][0]
                if parent_arrays["unbiased_particle_states"].ndim == 4
                else parent_arrays["unbiased_particle_states"]
            ).shape[1]
        )
        * int(
            (
                parent_arrays["unbiased_particle_states"][0]
                if parent_arrays["unbiased_particle_states"].ndim == 4
                else parent_arrays["unbiased_particle_states"]
            ).shape[2]
        )
    )
    memory_length = int(target_hlt.shape[1]) + sum(
        int(
            (
                parent_arrays[f"hlt_tokens_{name}"][0]
                if parent_arrays[f"hlt_tokens_{name}"].ndim == 4
                else parent_arrays[f"hlt_tokens_{name}"]
            ).shape[1]
        )
        for name in EXPERT_ORDER
    ) + int(
        (
            parent_arrays["unbiased_particle_states"][0]
            if parent_arrays["unbiased_particle_states"].ndim == 4
            else parent_arrays["unbiased_particle_states"]
        ).shape[1]
    )
    decoder_layer = (
        8 * k * d * d
        + 4 * k * k * d
        + 4 * k * d * d
        + 4 * memory_length * d * d
        + 4 * k * memory_length * d
        + 16 * k * d * d
    )
    output_projection = (
        0 if output_dimension == d else 2 * k * d * output_dimension
    )
    return {
        "parameter_count": int(parameter_count),
        "analytical_inference_flops": int(
            evidence_projection
            + 3 * decoder_layer
            + output_projection
            + 2 * k * d
        ),
    }


def _fit_readout(
    *,
    root: Path,
    shape: str,
    target_tuple: tuple[str, ...],
    variant: str,
    pipeline_seed: int = 101,
    selected_templates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    miniature: bool,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    identity = {
        "shape_id": shape,
        "target_tuple": list(target_tuple),
        "variant": variant,
        "readout_seed": 41703,
        "pipeline_seed": int(pipeline_seed),
    }
    output = (
        root
        / "runs"
        / "stage_e"
        / "coordinate_readouts"
        / canonical_sha256(identity)[:24]
    )
    score_path = output / "score.json"
    if score_path.is_file():
        return load_hashed_json(
            score_path, expected_contract=READOUT_SCORE_CONTRACT
        )
    predictor_profiles = {
        expert: _predictor_profile(
            root,
            shape=shape,
            expert=expert,
            mode=mode,
            pipeline_seed=pipeline_seed,
            selected_templates=selected_templates,
        )
        for expert, mode in zip(EXPERT_ORDER, target_tuple, strict=True)
    }
    split_arrays = {}
    coordinate_array_hashes: dict[str, dict[str, str]] = {}
    for split in ("model_train", "val_stop", "val_design"):
        banks, logits = {}, {}
        coordinate_array_hashes[split] = {}
        token_error_sum = 0.0
        token_error_count = 0
        identities = labels = None
        for expert, mode in zip(EXPERT_ORDER, target_tuple, strict=True):
            arrays = _coordinate_arrays(
                root,
                shape=shape,
                expert=expert,
                seed=pipeline_seed,
                mode=mode,
                selected_templates=selected_templates,
                split=split,
            )
            coordinate_array_hashes[split][expert] = _sha256(
                _coordinate_array_path(
                    root,
                    shape=shape,
                    expert=expert,
                    seed=pipeline_seed,
                    mode=mode,
                    selected_templates=selected_templates,
                    split=split,
                )
            )
            if identities is None:
                identities, labels = arrays["identities"], arrays["labels"]
            elif not np.array_equal(
                identities, arrays["identities"]
            ) or not np.array_equal(labels, arrays["labels"]):
                raise ValueError("bridge coordinate readout identities differ")
            required = {
                "predicted_hlt_tokens",
                "predicted_expert_logits",
                "predicted_fusion_logits",
            }
            if not required.issubset(arrays):
                raise ValueError(
                    "bridge coordinate arrays lack deployable predictions"
                )
            banks[expert] = arrays["predicted_hlt_tokens"]
            logits[expert] = arrays["predicted_expert_logits"]
            normalizer = _coordinate_normalizer(
                root,
                shape=shape,
                expert=expert,
                mode=mode,
                pipeline_seed=pipeline_seed,
                selected_templates=selected_templates,
            )
            moving_normal = _normalize(
                arrays["moving_tokens"], normalizer
            )
            predicted_normal = _normalize(
                arrays["predicted_hlt_tokens"], normalizer
            )
            difference = np.abs(predicted_normal - moving_normal)
            huber = np.where(
                difference <= 1.0,
                0.5 * difference * difference,
                difference - 0.5,
            )
            token_error_sum += float(huber.sum(dtype=np.float64))
            token_error_count += int(huber.size)
        split_arrays[split] = {
            "identities": identities,
            "labels": labels,
            "token_banks": banks,
            "expert_logits": logits,
            "normalized_token_error": (
                token_error_sum / token_error_count
            ),
        }
    torch.manual_seed(41703)
    model = build_fusion_model(
        variant,
        bank_dimensions={
            expert: int(
                split_arrays["model_train"]["token_banks"][expert].shape[-1]
            )
            for expert in EXPERT_ORDER
        },
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-4, weight_decay=1.0e-4
    )
    epochs = 2 if miniature else 40
    train_loader = make_fusion_loader(
        split_arrays["model_train"],
        batch_size=batch_size,
        seed=41703,
        training=True,
    )
    stop_loader = make_fusion_loader(
        split_arrays["val_stop"],
        batch_size=batch_size,
        seed=0,
        training=False,
    )
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output_logits = model(
                token_banks={
                    name: value.to(device)
                    for name, value in batch["token_banks"].items()
                },
                expert_logits={
                    name: value.to(device)
                    for name, value in batch["expert_logits"].items()
                },
            )
            loss = torch.nn.functional.cross_entropy(
                output_logits, batch["labels"].to(device)
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(
                    "bridge coordinate readout gradient is nonfinite"
                )
            optimizer.step()
        metrics, _ = evaluate_fusion(
            model, stop_loader, device=device, split="val_stop"
        )
        key = (-metrics["accuracy"], metrics["cross_entropy"], epoch)
        if best is None or key < best[0]:
            best = (
                key,
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            )
    model.load_state_dict(best[1], strict=True)
    metrics, _ = evaluate_fusion(
        model,
        make_fusion_loader(
            split_arrays["val_design"],
            batch_size=batch_size,
            seed=0,
            training=False,
        ),
        device=device,
        split="val_design",
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model_val.pt"
    torch.save({**identity, "model_state_dict": best[1]}, checkpoint)
    normalizers = {}
    for expert, mode in zip(EXPERT_ORDER, target_tuple, strict=True):
        if mode == "T0_PURE":
            path = (
                _parent(
                    root,
                    shape=shape,
                    expert=expert,
                    seed=pipeline_seed,
                )
                / "target_normalizer.json"
            )
        elif mode == "T2_PROJECT":
            path = (
                _candidate_root(
                    root,
                    shape=shape,
                    expert=expert,
                    seed=pipeline_seed,
                    template=selected_templates[(shape, expert, mode)],
                )
                / "bridge_normalizer.json"
            )
        else:
            path = (
                _parent(
                    root,
                    shape=shape,
                    expert=expert,
                    seed=pipeline_seed,
                )
                / "target_normalizer.json"
            )
        normalizers[expert] = load_hashed_json(path)["content_hash"]
    normalizer_set = bind_source(
        with_content_hash(
            {
                "contract": "retb_bridge_coordinate_normalizer_set_v1",
                "schema_version": 1,
                **identity,
                "normalizer_hashes": normalizers,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "normalizer_set.json", normalizer_set)
    score = bind_source(
        with_content_hash(
            {
                "contract": READOUT_SCORE_CONTRACT,
                "schema_version": 3,
                **identity,
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "normalized_token_error": float(
                    split_arrays["val_design"][
                        "normalized_token_error"
                    ]
                ),
                "measured_flops": float(
                    _fusion_flops(split_arrays["val_design"])
                    + sum(
                        profile["analytical_inference_flops"]
                        for profile in predictor_profiles.values()
                    )
                ),
                "parameter_count": (
                    sum(parameter.numel() for parameter in model.parameters())
                    + sum(
                        profile["parameter_count"]
                        for profile in predictor_profiles.values()
                    )
                ),
                "predictor_profiles": predictor_profiles,
                "readout_sha256": _sha256(checkpoint),
                "fusion_sha256": _sha256(checkpoint),
                "normalizer_set_sha256": normalizer_set["content_hash"],
                "coordinate_array_sha256": coordinate_array_hashes,
                "utility_input": (
                    "deployable_predicted_HLT_tokens_and_predicted_expert_logits"
                ),
                "target_cache_namespace": (
                    f"stage_e/{shape}/{canonical_sha256(identity)[:24]}"
                ),
                "checkpoint_path": str(checkpoint),
                "fixed_epoch_budget_completed": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(score_path, score)
    return score


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_e_templates.json"
    )
    validate_stage_e_template_registry(registry)
    output = args.output_dir or args.campaign_root / "selection" / "stage_e"
    output.mkdir(parents=True, exist_ok=True)
    selected_templates: dict[
        tuple[str, str, str], Mapping[str, Any]
    ] = {}
    for shape in registry["shapes"]:
        for expert in EXPERT_ORDER:
            for mode in (
                "T1_ANCHORED_BRIDGE",
                "T1_TASK_BRIDGE",
                "T2_PROJECT",
                "T3_LOGIT",
            ):
                selected_templates[(shape, expert, mode)] = _select_template(
                    args.campaign_root,
                    shape=shape,
                    expert=expert,
                    mode=mode,
                    templates=registry["candidate_templates"],
                )
    eligibility: dict[tuple[str, str, str], dict[str, Any]] = {}
    records = []
    for shape in registry["shapes"]:
        for expert in EXPERT_ORDER:
            t0_registrations = {
                seed: load_hashed_json(
                    _parent(
                        args.campaign_root,
                        shape=shape,
                        expert=expert,
                        seed=seed,
                    )
                    / "t0_registration.json"
                )
                for seed in (101, 202, 303)
            }
            t0 = bind_source(
                build_bridge_candidate_eligibility(
                    target_mode="T0_PURE",
                    expert_id=expert,
                    shape_id=shape,
                    checkpoint_hashes_by_seed={
                        seed: artifact["checkpoint_sha256"]
                        for seed, artifact in t0_registrations.items()
                    },
                ),
                source_snapshot=source_snapshot(REPO_ROOT),
            )
            t0_path = output / shape / expert / "T0_PURE" / "eligibility.json"
            write_immutable_json(t0_path, t0)
            eligibility[(shape, expert, "T0_PURE")] = t0
            t0_rows = []
            for seed in (101, 202, 303):
                arrays = _t0_arrays(
                    args.campaign_root,
                    shape=shape,
                    expert=expert,
                    seed=seed,
                    split="val_design",
                )
                t0_rows.append(
                    _metric_row(
                        logits=arrays["moving_fusion_logits"],
                        labels=arrays["labels"],
                        seed=seed,
                    )
                )
            for mode in (
                "T1_ANCHORED_BRIDGE",
                "T1_TASK_BRIDGE",
                "T2_PROJECT",
            ):
                template = selected_templates[(shape, expert, mode)]
                candidate_rows, registrations, certifications = [], {}, []
                for seed in (101, 202, 303):
                    candidate = _candidate_root(
                        args.campaign_root,
                        shape=shape,
                        expert=expert,
                        seed=seed,
                        template=template,
                    )
                    registration = load_hashed_json(
                        candidate / "checkpoint_registration.json"
                    )
                    registrations[seed] = registration
                    arrays = _npz(candidate / "certification_arrays.npz")
                    candidate_rows.append(
                        _metric_row(
                            logits=arrays["moving_fusion_logits"],
                            labels=arrays["labels"],
                            seed=seed,
                        )
                    )
                    parent = _parent(
                        args.campaign_root,
                        shape=shape,
                        expert=expert,
                        seed=seed,
                    )
                    t0_normalizer = load_hashed_json(
                        parent / "target_normalizer.json"
                    )
                    coordinate_normalizer = (
                        load_hashed_json(candidate / "bridge_normalizer.json")
                        if mode == "T2_PROJECT"
                        else t0_normalizer
                    )
                    certification = bind_source(
                        certify_bridge_content(
                            target_mode=mode,
                            expert_id=expert,
                            shape_id=shape,
                            pipeline_seed=seed,
                            moving_tokens=_normalize(
                                arrays["moving_tokens"],
                                coordinate_normalizer,
                            ),
                            t0_tokens=_normalize(
                                arrays["t0_tokens"], t0_normalizer
                            ),
                            predicted_hlt_tokens=_normalize(
                                arrays["predicted_hlt_tokens"],
                                coordinate_normalizer,
                            ),
                            frozen_moving_logits={
                                "expert": arrays["moving_expert_logits"],
                                "fusion": arrays["moving_fusion_logits"],
                            },
                            frozen_t0_logits={
                                "expert": arrays["t0_expert_logits"],
                                "fusion": arrays["t0_fusion_logits"],
                            },
                            identities=[
                                str(value)
                                for value in arrays["identities"].tolist()
                            ],
                            labels=arrays["labels"],
                            candidate_checkpoint_sha256=registration[
                                "checkpoint_sha256"
                            ],
                            t0_checkpoint_sha256=t0_registrations[seed][
                                "checkpoint_sha256"
                            ],
                            identity_manifest_sha256=load_hashed_json(
                                parent / "parent_bundle.json"
                            )["dataset_evidence"]["val_design"][
                                "identity_manifest_sha256"
                            ],
                            coordinate_normalizer_sha256=(
                                coordinate_normalizer["content_hash"]
                            ),
                            t0_normalizer_sha256=t0_normalizer[
                                "content_hash"
                            ],
                            decoded_tokens=(
                                _normalize(
                                    arrays["decoded_tokens"], t0_normalizer
                                )
                                if mode == "T2_PROJECT"
                                else None
                            ),
                        ),
                        source_snapshot=source_snapshot(REPO_ROOT),
                    )
                    cert_path = (
                        output
                        / shape
                        / expert
                        / mode
                        / f"seed_{seed}_content.json"
                    )
                    write_immutable_json(cert_path, certification)
                    certifications.append(certification)
                candidate_bundle = bind_source(
                    with_content_hash(
                        {
                            "contract": "retb_bridge_candidate_metric_bundle_v1",
                            "schema_version": 1,
                            "shape_id": shape,
                            "expert_id": expert,
                            "target_mode": mode,
                            "selected_template": dict(template),
                            "rows": candidate_rows,
                        }
                    ),
                    source_snapshot=source_snapshot(REPO_ROOT),
                )
                t0_bundle = bind_source(
                    with_content_hash(
                        {
                            "contract": "retb_bridge_t0_metric_bundle_v1",
                            "schema_version": 1,
                            "shape_id": shape,
                            "expert_id": expert,
                            "rows": t0_rows,
                        }
                    ),
                    source_snapshot=source_snapshot(REPO_ROOT),
                )
                noninferiority = bind_source(
                    certify_offline_noninferiority(
                        target_mode=mode,
                        candidate_rows=candidate_rows,
                        t0_rows=t0_rows,
                        candidate_bundle_sha256=candidate_bundle[
                            "content_hash"
                        ],
                        t0_bundle_sha256=t0_bundle["content_hash"],
                    ),
                    source_snapshot=source_snapshot(REPO_ROOT),
                )
                eligible = bind_source(
                    build_bridge_candidate_eligibility(
                        target_mode=mode,
                        expert_id=expert,
                        shape_id=shape,
                        checkpoint_hashes_by_seed={
                            seed: registrations[seed]["checkpoint_sha256"]
                            for seed in (101, 202, 303)
                        },
                        noninferiority=noninferiority,
                        content_certifications=certifications,
                    ),
                    source_snapshot=source_snapshot(REPO_ROOT),
                )
                mode_root = output / shape / expert / mode
                for name, artifact in (
                    ("candidate_metrics.json", candidate_bundle),
                    ("t0_metrics.json", t0_bundle),
                    ("noninferiority.json", noninferiority),
                    ("eligibility.json", eligible),
                ):
                    write_immutable_json(mode_root / name, artifact)
                eligibility[(shape, expert, mode)] = eligible
                records.append(
                    {
                        "shape_id": shape,
                        "expert_id": expert,
                        "target_mode": mode,
                        "selected_template": dict(template),
                        "eligibility_sha256": eligible["content_hash"],
                        "offline_noninferior": eligible[
                            "offline_noninferior"
                        ],
                        "bridge_content_certified": eligible[
                            "bridge_content_certified"
                        ],
                    }
                )
            t3_template = selected_templates[(shape, expert, "T3_LOGIT")]
            t3_rows = []
            for seed in (101, 202, 303):
                t3_root = _candidate_root(
                    args.campaign_root,
                    shape=shape,
                    expert=expert,
                    seed=seed,
                    template=t3_template,
                )
                t3_metric = load_hashed_json(
                    t3_root / "val_design_metrics.json"
                )
                t3_registration = load_hashed_json(
                    t3_root / "checkpoint_registration.json"
                )
                t3_rows.append(
                    {
                        "seed": seed,
                        "checkpoint_sha256": t3_registration[
                            "checkpoint_sha256"
                        ],
                        "metrics_sha256": t3_metric["content_hash"],
                        "metrics": t3_metric["metrics"],
                    }
                )
            records.append(
                {
                    "shape_id": shape,
                    "expert_id": expert,
                    "target_mode": "T3_LOGIT",
                    "selected_template": dict(t3_template),
                    "three_seed_rows": t3_rows,
                    "maximum_performance_eligible": False,
                    "token_fidelity_claim": False,
                }
            )
    # The locked coordinate tuple is shape-independent downstream. Use the
    # already locked high-accuracy shape for its complete joint beam search.
    shape = "SHAPE_HIGH"
    eligible_modes, individual_metrics, eligibility_hashes = {}, {}, {}
    for expert in EXPERT_ORDER:
        modes = [
            mode
            for mode in (
                "T0_PURE",
                "T1_ANCHORED_BRIDGE",
                "T1_TASK_BRIDGE",
                "T2_PROJECT",
            )
            if eligibility[(shape, expert, mode)][
                "maximum_performance_eligible"
            ]
        ]
        if "T0_PURE" not in modes:
            raise RuntimeError("T0 bridge fallback was lost")
        eligible_modes[expert] = modes
        eligibility_hashes[expert] = {
            mode: eligibility[(shape, expert, mode)]["content_hash"]
            for mode in modes
        }
        individual_metrics[expert] = {}
        for mode in modes:
            arrays = _coordinate_arrays(
                args.campaign_root,
                shape=shape,
                expert=expert,
                seed=101,
                mode=mode,
                selected_templates=selected_templates,
                split="val_design",
            )
            metrics = evaluate_classification(
                arrays["predicted_fusion_logits"],
                arrays["labels"],
                split="val_design",
            )
            profile = _predictor_profile(
                args.campaign_root,
                shape=shape,
                expert=expert,
                mode=mode,
                selected_templates=selected_templates,
            )
            individual_metrics[expert][mode] = {
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "measured_flops": float(
                    profile["analytical_inference_flops"]
                ),
                "parameter_count": profile["parameter_count"],
                "coordinate_array_sha256": _sha256(
                    _coordinate_array_path(
                        args.campaign_root,
                        shape=shape,
                        expert=expert,
                        seed=101,
                        mode=mode,
                        selected_templates=selected_templates,
                        split="val_design",
                    )
                ),
            }
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    scores: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def score(
        variant: str, target_tuple: tuple[str, ...], seed: int
    ) -> dict[str, Any]:
        if seed != 41703:
            raise ValueError("bridge readout seed differs")
        key = (variant, tuple(target_tuple))
        if key not in scores:
            scores[key] = _fit_readout(
                root=args.campaign_root,
                shape=shape,
                target_tuple=tuple(target_tuple),
                variant=variant,
                selected_templates=selected_templates,
                miniature=miniature,
                batch_size=args.batch_size,
                device=device,
            )
        return scores[key]

    # Populate exactly the deterministic rows the selector will consume.
    select_joint_bridge_coordinates(
        eligible_modes=eligible_modes,
        default_metrics=individual_metrics,
        pooled_scorer=lambda values, seed: score(
            "F_POOLED_MLP", values, seed
        ),
        transformer_scorer=lambda values, seed: score(
            "F_TOKEN_TRANSFORMER", values, seed
        ),
        shape_id=shape,
        eligibility_hashes=eligibility_hashes,
    )
    score_table = bind_source(
        with_content_hash(
            {
                "contract": SCORE_TABLE_CONTRACT,
                "schema_version": 3,
                "shape_id": shape,
                "eligible_modes": eligible_modes,
                "individual_metrics": individual_metrics,
                "rows": [
                    {
                        "readout": variant,
                        "target_tuple": list(target_tuple),
                        "score_sha256": artifact["content_hash"],
                        **{
                            name: value
                            for name, value in artifact.items()
                            if name
                            in {
                                "accuracy",
                                "cross_entropy",
                                "normalized_token_error",
                                "measured_flops",
                                "parameter_count",
                                "readout_sha256",
                                "fusion_sha256",
                                "normalizer_set_sha256",
                                "target_cache_namespace",
                            }
                        },
                    }
                    for (variant, target_tuple), artifact in sorted(
                        scores.items(), key=lambda value: value[0]
                    )
                ],
                "utility_input": (
                    "deployable_predicted_HLT_tokens_and_predicted_expert_logits"
                ),
                "all_readouts_fresh_per_tuple": True,
                "all_scientific_failures_retained": True,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    score_path = output / "bridge_coordinate_score_table.json"
    write_immutable_json(score_path, score_table)
    index = bind_source(
        with_content_hash(
            {
                "contract": INDEX_CONTRACT,
                "schema_version": 3,
                "stage_e_template_registry_sha256": registry["content_hash"],
                "complete_candidate_row_count": registry[
                    "candidate_membership_count"
                ],
                "selected_candidate_group_count": len(records),
                "records": records,
                "eligibility_paths": {
                    f"{expert}:{mode}": str(
                        output / shape / expert / mode / "eligibility.json"
                    )
                    for expert in EXPERT_ORDER
                    for mode in eligible_modes[expert]
                },
                "bridge_coordinate_score_table_sha256": score_table[
                    "content_hash"
                ],
                "T0_fallback_present_for_every_expert": True,
                "pilot_all_predicted_utility": True,
                "scientific_failure_stops_workflow": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "bridge_certification_index.json", index)
    eligibility_index = bind_source(
        with_content_hash(
            {
                "contract": "retb_bridge_eligibility_index_v1",
                "schema_version": 1,
                "bridge_certification_index_sha256": index["content_hash"],
                "eligible_modes": eligible_modes,
                "eligibility_hashes": eligibility_hashes,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "bridge_eligibility_index.json", eligibility_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
