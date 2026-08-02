#!/usr/bin/env python3
"""Execute one frozen HOSD probe row from identity-bound NumPy inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_probe_result,
    build_tap_probe,
    continuous_probe_metrics,
    latent_probe_metrics,
    load_and_validate_campaign,
    statistical_references,
    teacher_probe_metrics,
    pair_probe_metrics,
    deterministic_pair_indices,
    HBaseParticleTransformer,
    validate_frozen_encoder,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PROBE_COMPLETION_CONTRACT,
    PROBE_ENCODER_LOCK_CONTRACT,
    STAGE_C_PLAN_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.replicas import (  # noqa: E402
    replica_for,
)
from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (  # noqa: E402
    try_finalize_row_wave,
)


def _finalize_probe_wave(campaign_root, plan, campaign):
    return try_finalize_row_wave(
        wave_id="stage_c_probes",
        expected_paths={
            item["row_id"]: campaign_root
            / "probes"
            / item["row_id"]
            / "probe_completion.json"
            for item in plan["probe_rows"]
        },
        expected_rows={
            item["row_id"]: {
                "row_id": item["row_id"],
                "target_id": item["target_id"],
                "probe_kind": item["probe_kind"],
                "tap": item["tap"],
            }
            for item in plan["probe_rows"]
        },
        expected_contract=PROBE_COMPLETION_CONTRACT,
        parent_hashes={"stage_c_plan": plan["content_hash"]},
        source=campaign["source"],
        output=campaign_root / "probes" / "probe_completion.json",
    )


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {"identities", "labels", "target", "target_mask"}
    if not required.issubset(arrays):
        raise ValueError(f"{path} lacks {sorted(required - set(arrays))}")
    identities = tuple(str(value) for value in arrays["identities"].tolist())
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("probe identities are empty or duplicated")
    if arrays["labels"].shape != (len(identities),):
        raise ValueError("probe label population differs")
    if arrays["target"].shape != arrays["target_mask"].shape:
        raise ValueError("probe target/mask shapes differ")
    if not (
        arrays["target"].shape[0] == len(identities)
        or arrays["target"].shape[:2] == (4, len(identities))
    ):
        raise ValueError("probe target population differs")
    arrays["_identity_strings"] = np.asarray(identities)
    return arrays


def _identity_hash(arrays: dict[str, np.ndarray]) -> str:
    return hashlib.sha256(
        json.dumps(
            arrays["_identity_strings"].tolist(),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _tap_probe_head_type(target: np.ndarray) -> str:
    """Classify canonical/evaluation or replica-stacked pair target layouts."""
    dimensions = int(np.asarray(target).ndim)
    if dimensions in {4, 5}:
        return "pair"
    if dimensions == 2 or dimensions == 3:
        return "global"
    raise ValueError("probe target rank is not registered")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_tap_populations(args, row, lock, *, device):
    """Encode one exact frozen tap into bounded worker RAM, never disk."""
    import importlib
    import torch
    from scripts.build_hosd_probe_taps import (
        _dataset,
        _mapping,
        capture_tap_in_memory,
    )

    required = (
        args.baseline_checkpoint,
        args.train_labels,
        args.val_stop_labels,
        args.design_select_labels,
    )
    if any(value is None for value in required):
        raise ValueError("streamed tap execution lacks checkpoint/label inputs")
    if _sha256_file(args.baseline_checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("streamed tap checkpoint differs from encoder lock")
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    encoder = HBaseParticleTransformer(weaver_module=module)
    checkpoint = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.to(device).eval()
    validate_frozen_encoder(encoder, lock)
    datasets = {
        "model_train": _dataset(
            _mapping(
                args.train_cache,
                "--train-cache",
                required_replicas={0, 1, 2, 3},
            ),
            args.train_labels,
            "model_train",
            realization_policy="R_MULTI",
        ),
        "val_stop": _dataset(
            _mapping(
                args.val_stop_cache,
                "--val-stop-cache",
                required_replicas={0},
            ),
            args.val_stop_labels,
            "val_stop",
            realization_policy="R_FIXED",
        ),
        "design_select": _dataset(
            _mapping(
                args.design_select_cache,
                "--design-select-cache",
                required_replicas={0},
            ),
            args.design_select_labels,
            "design_select",
            realization_policy="R_FIXED",
        ),
    }
    populations = {}
    for role, dataset in datasets.items():
        replicas = range(4) if role == "model_train" else (0,)
        states = None
        masks = None
        for offset, replica in enumerate(replicas):
            replica_states, replica_masks = capture_tap_in_memory(
                encoder, dataset, replica, tap=row["tap"], device=device
            )
            if role == "model_train":
                if states is None:
                    states = np.empty(
                        (4, *replica_states.shape), dtype=replica_states.dtype
                    )
                    masks = np.empty(
                        (4, *replica_masks.shape), dtype=replica_masks.dtype
                    )
                states[offset] = replica_states
                masks[offset] = replica_masks
            else:
                states, masks = replica_states, replica_masks
            del replica_states, replica_masks
        if states is None or masks is None:
            raise RuntimeError("streamed tap population is empty")
        populations[role] = {
            "identities": tuple(str(value) for value in dataset.identities),
            "states": states,
            "particle_mask": masks,
        }
    del datasets, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return populations


def _target_metrics(row: dict, prediction: np.ndarray, data: dict) -> dict:
    target_id = row["target_id"]
    def calculate(predicted, target, mask):
        if target_id.startswith("T_OFFLINE_LOGITS_"):
            return teacher_probe_metrics(predicted, target)
        if target_id == "T_OFFLINE_POOLED_LATENT":
            return latent_probe_metrics(predicted, target, mask)
        if target_id.startswith("T_HLT_TRACK_PAIR_"):
            return pair_probe_metrics(
                predicted, target, mask, binary_channels=0
            )
        if target_id.startswith("T_HLT_REGION_PAIR_"):
            return pair_probe_metrics(
                predicted, target, mask, binary_channels=3
            )
        return continuous_probe_metrics(predicted, target, mask)
    overall = calculate(prediction, data["target"], data["target_mask"])
    per_class = {}
    for class_index in range(10):
        selected = data["labels"] == class_index
        per_class[str(class_index)] = (
            None
            if not selected.any()
            else calculate(
                prediction[selected],
                data["target"][selected],
                data["target_mask"][selected],
            )
        )
    return {**overall, "per_class": per_class}


def _validation_target_loss(prediction, target, mask, target_id):
    import torch
    predicted = torch.from_numpy(np.asarray(prediction)).float()
    truth = torch.from_numpy(np.asarray(target)).float()
    valid = torch.from_numpy(np.asarray(mask)).bool()
    if target_id.startswith("T_OFFLINE_LOGITS_"):
        probability = torch.softmax(truth / 2.0, -1)
        return float(4.0 * torch.nn.functional.kl_div(
            torch.log_softmax(predicted / 2.0, -1),
            probability,
            reduction="batchmean",
        ))
    if target_id == "T_HLT_REGION_PAIR_8":
        pieces = []
        for channel in range(3):
            selected = valid[..., channel]
            if bool(selected.any()):
                pieces.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted[..., channel].masked_select(selected),
                    truth[..., channel].masked_select(selected),
                ))
        for channel in range(3, 8):
            selected = valid[..., channel]
            if bool(selected.any()):
                pieces.append(torch.nn.functional.huber_loss(
                    predicted[..., channel].masked_select(selected),
                    truth[..., channel].masked_select(selected),
                    delta=1.0,
                ))
        return float(sum(pieces) / len(pieces))
    return float(torch.nn.functional.huber_loss(
        predicted.masked_select(valid),
        truth.masked_select(valid),
        delta=1.0,
    ))


def _training_batch(data: dict, selected: np.ndarray, *, epoch: int) -> dict:
    population = len(data["labels"])
    identities = data["_identity_strings"]
    replicas = np.asarray([
        replica_for(
            policy="R_MULTI",
            logical_role="model_train",
            epoch=int(epoch) - 1,
            canonical_identity=str(identities[index]),
        )
        for index in selected
    ], dtype=np.int64)
    output = {}
    for key, value in data.items():
        if key.startswith("_") or value.ndim == 0:
            continue
        if value.shape[:2] == (4, population):
            output[key] = np.stack([
                value[replica, index]
                for replica, index in zip(replicas, selected)
            ])
        elif value.shape[0] == population:
            output[key] = value[selected]
    output["replica_ids"] = replicas
    return output


def _predict(model, data: dict, kind: str, *, device, batch_size: int = 256):
    import torch
    if data["target"].ndim == 4:
        batch_size = 1
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["labels"]), batch_size):
            stop = min(start + batch_size, len(data["labels"]))
            if kind in {"P_LINEAR", "P_SHALLOW"}:
                states = torch.from_numpy(data["states"][start:stop]).float().to(device)
                mask = torch.from_numpy(data["particle_mask"][start:stop]).bool().to(device)
                output = model(states, mask)
                value = output[0] if isinstance(output, tuple) else output["value"]
            elif kind == "P_RAW_MLP":
                summary = torch.from_numpy(data["raw_summary"][start:stop]).float().to(device)
                context = torch.from_numpy(data["jet_context"][start:stop]).float().to(device)
                value = model(summary, context)["value"]
            else:
                target_array = data["target"][start:stop]
                availability_array = data.get(
                    "availability", data["target_mask"]
                )[start:stop]
                if target_array.ndim == 4:
                    weight = availability_array.astype(bool)
                    denominator = weight.sum(axis=(1, 2)).clip(min=1)
                    target_array = (
                        np.where(weight, target_array, 0).sum(axis=(1, 2))
                        / denominator
                    )
                    availability_array = weight.any(axis=(1, 2))
                target = torch.from_numpy(target_array).float().to(device)
                availability = torch.from_numpy(availability_array).float().to(device)
                value = model(target, availability)
            values.append(value.float().cpu().numpy())
    return np.concatenate(values)


def _sampled_pair_loss(model, batch: dict, identities, *, epoch: int, target_id: str, device):
    import torch
    states = torch.from_numpy(batch["states"]).float().to(device)
    target = torch.from_numpy(batch["target"]).float().to(device)
    target_mask = torch.from_numpy(batch["target_mask"]).bool().to(device)
    event_rows, left_rows, right_rows, local_rows = [], [], [], []
    strata = []
    for event in range(len(identities)):
        applicable = batch["target_mask"][event].any(axis=-1)
        left, right = np.nonzero(applicable)
        keep = left != right
        left, right = left[keep], right[keep]
        pair_ids = [f"{int(i)}:{int(j)}" for i, j in zip(left, right)]
        positive = (
            [
                bool(batch["target"][event, i, j, :3].any())
                for i, j in zip(left, right)
            ]
            if target_id == "T_HLT_REGION_PAIR_8"
            else None
        )
        chosen = deterministic_pair_indices(
            epoch=epoch,
            identity=str(identities[event]),
            target_id=target_id,
            pair_ids=pair_ids,
            positive=positive,
        )
        for local in chosen:
            event_rows.append(event)
            left_rows.append(int(left[local]))
            right_rows.append(int(right[local]))
            local_rows.append((event, int(left[local]), int(right[local])))
            strata.append(None if positive is None else bool(positive[local]))
    if not event_rows:
        raise ValueError("pair probe batch has no applicable off-diagonal pairs")
    event_tensor = torch.tensor(event_rows, device=device)
    left_tensor = torch.tensor(left_rows, device=device)
    right_tensor = torch.tensor(right_rows, device=device)
    prediction = model.forward_pairs(
        states, event_tensor, left_tensor, right_tensor
    )
    pair_target = target[event_tensor, left_tensor, right_tensor]
    pair_mask = target_mask[event_tensor, left_tensor, right_tensor]
    event_losses = []
    for event in range(len(identities)):
        selected_event = event_tensor == event
        if not bool(selected_event.any()):
            continue
        if target_id == "T_HLT_REGION_PAIR_8":
            stratum_losses = []
            for positive_value in (True, False):
                selected = selected_event & torch.tensor(
                    [value is positive_value for value in strata],
                    device=device,
                )
                if not bool(selected.any()):
                    continue
                pieces = []
                for channel in range(3):
                    channel_mask = pair_mask[selected, channel]
                    if bool(channel_mask.any()):
                        pieces.append(
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                prediction[selected, channel].masked_select(channel_mask),
                                pair_target[selected, channel].masked_select(channel_mask),
                            )
                        )
                for channel in range(3, 8):
                    channel_mask = pair_mask[selected, channel]
                    if bool(channel_mask.any()):
                        pieces.append(torch.nn.functional.huber_loss(
                            prediction[selected, channel].masked_select(channel_mask),
                            pair_target[selected, channel].masked_select(channel_mask),
                            delta=1.0,
                        ))
                if pieces:
                    stratum_losses.append(sum(pieces) / len(pieces))
            event_losses.append(sum(stratum_losses) / len(stratum_losses))
        else:
            selected_mask = pair_mask[selected_event]
            event_losses.append(torch.nn.functional.huber_loss(
                prediction[selected_event].masked_select(selected_mask),
                pair_target[selected_event].masked_select(selected_mask),
                delta=1.0,
            ))
    return torch.stack(event_losses).mean()


def _learned_probe_loss(
    model,
    batch,
    *,
    kind,
    row,
    identities,
    epoch,
    device,
):
    import torch
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        availability_logits = None
        if kind in {"P_LINEAR", "P_SHALLOW"}:
            if batch["target"].ndim == 4:
                prediction = None
            else:
                output = model(
                    torch.from_numpy(batch["states"]).float().to(device),
                    torch.from_numpy(batch["particle_mask"]).bool().to(device),
                )
                prediction = output["value"]
                availability_logits = output.get("availability_logits")
        elif kind == "P_RAW_MLP":
            output = model(
                torch.from_numpy(batch["raw_summary"]).float().to(device),
                torch.from_numpy(batch["jet_context"]).float().to(device),
            )
            prediction = output["value"]
            availability_logits = output["availability_logits"]
        else:
            oracle_target = batch["target"]
            oracle_availability = batch.get("availability", batch["target_mask"])
            if oracle_target.ndim == 4:
                weight = oracle_availability.astype(bool)
                denominator = weight.sum(axis=(1, 2)).clip(min=1)
                oracle_target = (
                    np.where(weight, oracle_target, 0).sum(axis=(1, 2))
                    / denominator
                )
                oracle_availability = weight.any(axis=(1, 2))
            prediction = model(
                torch.from_numpy(oracle_target).float().to(device),
                torch.from_numpy(oracle_availability).float().to(device),
            )
        if kind == "P_TARGET_TO_CLASS_ORACLE":
            return torch.nn.functional.cross_entropy(
                prediction, torch.from_numpy(batch["labels"]).long().to(device)
            )
        if batch["target"].ndim == 4:
            loss = _sampled_pair_loss(
                model,
                batch,
                identities,
                epoch=epoch,
                target_id=row["target_id"],
                device=device,
            )
        else:
            target = torch.from_numpy(batch["target"]).float().to(device)
            mask = torch.from_numpy(batch["target_mask"]).bool().to(device)
            if row["target_id"].startswith("T_OFFLINE_LOGITS_"):
                teacher_probability = torch.softmax(target.detach() / 2.0, -1)
                loss = 4.0 * torch.nn.functional.kl_div(
                    torch.log_softmax(prediction / 2.0, -1),
                    teacher_probability,
                    reduction="batchmean",
                )
            else:
                loss = torch.nn.functional.huber_loss(
                    prediction.masked_select(mask),
                    target.masked_select(mask),
                    delta=1.0,
                )
        if availability_logits is not None:
            if "availability" not in batch:
                raise ValueError("global learned probe lacks availability groups")
            availability_target = torch.from_numpy(
                batch["availability"]
            ).float().to(device)
            if tuple(availability_target.shape) != tuple(
                availability_logits.shape
            ):
                raise ValueError("probe availability target/logit shapes differ")
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                availability_logits, availability_target
            )
        return loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id")
    parser.add_argument("--train-npz", type=Path)
    parser.add_argument("--val-stop-npz", type=Path)
    parser.add_argument("--design-select-npz", type=Path)
    parser.add_argument("--probe-encoder-lock", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--design-select-cache", action="append", default=[])
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--val-stop-labels", type=Path)
    parser.add_argument("--design-select-labels", type=Path)
    parser.add_argument("--input-lineage", type=Path)
    parser.add_argument("--target-cache-manifest-sha256", required=False)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    for resource in (
        "model_train_hlt",
        "model_train_labels",
        "model_train_targets",
        "val_stop_hlt",
        "val_stop_labels",
        "val_stop_targets",
        "design_select_hlt",
        "design_select_labels",
        "design_select_targets",
    ):
        authorize_access(worker_role="probe_worker", requested_resource=resource)
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_c_execution_plan.json",
        expected_contract=STAGE_C_PLAN_CONTRACT,
    )
    if plan.get("source") != campaign["source"]:
        raise ValueError("Stage-C plan source differs")
    if args.row_id is None or args.dry_run:
        print(json.dumps({
            "executed": False,
            "stage_c_plan_sha256": plan["content_hash"],
            "probe_row_count": plan["probe_row_count"],
        }, indent=2, sort_keys=True))
        return 0
    row = next((item for item in plan["probe_rows"] if item["row_id"] == args.row_id), None)
    if row is None:
        raise ValueError("probe row is absent from the frozen Stage-C plan")
    if args.input_lineage is None:
        raise ValueError("probe execution requires --input-lineage")
    input_lineage = load_hashed_json(
        args.input_lineage, expected_contract="hosd_probe_input_lineage_v2"
    )
    if (
        input_lineage.get("source") != campaign["source"]
        or input_lineage.get("campaign_spec_sha256") != campaign["content_hash"]
        or input_lineage.get("stage_c_plan_sha256") != plan["content_hash"]
        or input_lineage.get("row_id") != args.row_id
        or input_lineage.get("target_id") != row["target_id"]
    ):
        raise ValueError("probe input lineage differs")
    if (
        args.target_cache_manifest_sha256 is not None
        and args.target_cache_manifest_sha256 != input_lineage["content_hash"]
    ):
        raise ValueError("declared probe input-lineage hash differs")
    required = (args.train_npz, args.val_stop_npz, args.design_select_npz)
    if any(value is None for value in required):
        raise ValueError("probe execution requires train/val-stop/design-select NPZs")
    train, val, design = (_load(path) for path in required)
    kind = row["probe_kind"]
    output = args.output_dir or args.campaign_root / "probes" / args.row_id
    output.mkdir(parents=True, exist_ok=True)
    reusable_completion = output / "probe_completion.json"
    if reusable_completion.is_file():
        completion = load_hashed_json(
            reusable_completion, expected_contract=PROBE_COMPLETION_CONTRACT
        )
        if (
            completion.get("source") != campaign["source"]
            or completion["row_id"] != args.row_id
            or completion["stage_c_plan_sha256"] != plan["content_hash"]
        ):
            raise ValueError("reusable probe completion lineage differs")
        result = load_hashed_json(
            output / "probe_result.json", expected_contract="hosd_probe_result_v1"
        )
        if result["content_hash"] != completion["probe_result_sha256"]:
            raise ValueError("reusable probe result hash differs")
        wave = _finalize_probe_wave(args.campaign_root, plan, campaign)
        print(json.dumps({
            "row_id": args.row_id,
            "probe_result_sha256": result["content_hash"],
            "completion_sha256": completion["content_hash"],
            "reused": True,
            "wave_completion_sha256": (
                None if wave is None else wave["content_hash"]
            ),
        }, indent=2, sort_keys=True))
        return 0
    cache_sha = input_lineage["content_hash"]
    lock_sha = None
    if kind in {"P_LINEAR", "P_SHALLOW"}:
        if args.probe_encoder_lock is None:
            raise ValueError("tap probes require the immutable seed-101 encoder lock")
        lock = load_hashed_json(
            args.probe_encoder_lock, expected_contract=PROBE_ENCODER_LOCK_CONTRACT
        )
        if lock.get("source") != campaign["source"]:
            raise ValueError("probe encoder lock source differs")
        lock_sha = lock["content_hash"]
        for name, data in (("train", train), ("val_stop", val), ("design_select", design)):
            if "states" in data or "particle_mask" in data:
                raise ValueError(f"{name} unexpectedly persists frozen tap states")

    if kind == "P_STATISTICAL_REFERENCES":
        statistical_target = train["target"]
        statistical_mask = train["target_mask"]
        statistical_labels = train["labels"]
        if statistical_target.shape[:2] == (4, len(train["labels"])):
            statistical_target = statistical_target.reshape(
                4 * len(train["labels"]), *statistical_target.shape[2:]
            )
            statistical_mask = statistical_mask.reshape(
                4 * len(train["labels"]), *statistical_mask.shape[2:]
            )
            statistical_labels = np.tile(train["labels"], 4)
        stats = statistical_references(
            statistical_target, statistical_mask, statistical_labels,
            target_kind=(
                "teacher_logits"
                if row["target_id"].startswith("T_OFFLINE_LOGITS_")
                else "mixed_region_pair"
                if row["target_id"] == "T_HLT_REGION_PAIR_8"
                else "continuous"
            ),
        )
        # Evaluate both named controls explicitly on design_select.
        prior_base = np.asarray(stats["P_PRIOR"], dtype=np.float32)
        if row["target_id"].startswith("T_OFFLINE_LOGITS_"):
            prior_base = np.log(np.clip(prior_base, 1e-12, None))
        prior = np.broadcast_to(
            prior_base,
            design["target"].shape,
        )
        prior_metrics = _target_metrics(row, prior, design)
        conditional_base = np.asarray([
            stats["P_CLASS_CONDITIONAL_ORACLE"][str(int(label))]
            for label in design["labels"]
        ], dtype=np.float32)
        if row["target_id"].startswith("T_OFFLINE_LOGITS_"):
            conditional_base = np.log(np.clip(conditional_base, 1e-12, None))
        conditional_rows = np.broadcast_to(
            conditional_base.reshape(
                len(conditional_base),
                *([1] * (design["target"].ndim - 2)),
                conditional_base.shape[-1],
            ),
            design["target"].shape,
        )
        conditional_metrics = _target_metrics(row, conditional_rows, design)
        metrics = {
            "P_PRIOR": prior_metrics,
            "P_CLASS_CONDITIONAL_ORACLE": conditional_metrics,
            "statistics": stats,
        }
        checkpoint_sha = None
    else:
        import torch
        from teacher_logit_reco.hlt_offline_structure_distillation.probes import (
            RawSummaryProbe,
            TargetToClassOracle,
        )
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device
        )
        component_seed = int(row["component_seed"])
        torch.manual_seed(component_seed)
        if kind in {"P_LINEAR", "P_SHALLOW"}:
            streamed = _stream_tap_populations(
                args, row, lock, device=device
            )
            for role, data in (
                ("model_train", train),
                ("val_stop", val),
                ("design_select", design),
            ):
                if streamed[role]["identities"] != tuple(
                    str(value) for value in data["_identity_strings"].tolist()
                ):
                    raise ValueError(f"{role} streamed tap identities differ")
                data["states"] = streamed[role]["states"]
                data["particle_mask"] = streamed[role]["particle_mask"]
        output_dim = int(train["target"].shape[-1])
        availability_groups = int(
            train["availability"].shape[-1]
            if "availability" in train
            else 1
        )
        if kind in {"P_LINEAR", "P_SHALLOW"}:
            if "states" not in train or "particle_mask" not in train:
                raise ValueError("streamed tap populations are absent")
            head_type = _tap_probe_head_type(train["target"])
            model = build_tap_probe(
                probe_kind=kind,
                tap=row["tap"],
                input_dimension=int(train["states"].shape[-1]),
                target_dimension=output_dim,
                head_type=head_type,
                symmetric=row["target_id"] == "T_HLT_REGION_PAIR_8",
                availability_groups=availability_groups,
            )
        elif kind == "P_RAW_MLP":
            if "raw_summary" not in train or "jet_context" not in train:
                raise ValueError("raw probe inputs lack summary/context")
            model = RawSummaryProbe(
                int(train["raw_summary"].shape[-1]),
                output_dim,
                availability_groups,
            )
        elif kind == "P_TARGET_TO_CLASS_ORACLE":
            availability_dim = int(
                train.get("availability", train["target_mask"]).shape[-1]
            )
            model = TargetToClassOracle(output_dim, availability_dim)
        else:
            raise ValueError("unknown probe kind")
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4
        )
        epochs = 2 if campaign["campaign_profile"] == "miniature_test" else 40
        best = None
        microbatches_per_epoch = math.ceil(len(train["labels"]) / 64)
        updates_per_epoch = math.ceil(microbatches_per_epoch / 2)
        total_updates = epochs * updates_per_epoch
        warmup_updates = min(total_updates, max(1, math.ceil(0.05 * total_updates)))
        update_ordinal = 0
        start_epoch = 1
        last_checkpoint = output / "last.pt"
        if last_checkpoint.is_file():
            resume = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
            if (
                resume.get("contract") != "hosd_probe_checkpoint_v1"
                or resume.get("kind") != "resumable_last"
                or resume.get("row_id") != args.row_id
                or resume.get("pipeline_seed") != int(row["pipeline_seed"])
                or resume.get("component_seed") != int(row["component_seed"])
                or resume.get("stage_c_plan_sha256") != plan["content_hash"]
                or resume.get("probe_encoder_lock_sha256") != lock_sha
            ):
                raise ValueError("probe resume lineage differs")
            model.load_state_dict(resume["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            best = resume["best"]
            update_ordinal = int(resume["optimizer_update_ordinal"])
            start_epoch = int(resume["epoch_completed"]) + 1
        for epoch in range(start_epoch, epochs + 1):
            order = np.random.default_rng(
                int(row["pipeline_seed"]) * 1_000_003 + epoch
            ).permutation(len(train["labels"]))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation_events = 0
            for batch_index, start in enumerate(range(0, len(order), 64), start=1):
                selected = order[start:start + 64]
                batch = _training_batch(train, selected, epoch=epoch)
                loss = _learned_probe_loss(
                    model,
                    batch,
                    kind=kind,
                    row=row,
                    identities=train["_identity_strings"][selected],
                    epoch=epoch,
                    device=device,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("probe loss is nonfinite")
                events = len(selected)
                (loss * events).backward()
                accumulation_events += events
                step_now = batch_index % 2 == 0 or batch_index == microbatches_per_epoch
                if not step_now:
                    continue
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulation_events)
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(gradient)):
                    raise FloatingPointError("probe gradient is nonfinite")
                update_ordinal += 1
                if update_ordinal <= warmup_updates:
                    learning_rate = 1e-3 * update_ordinal / warmup_updates
                elif total_updates == warmup_updates:
                    learning_rate = 1e-3
                else:
                    progress = (
                        (update_ordinal - warmup_updates)
                        / (total_updates - warmup_updates)
                    )
                    learning_rate = 1e-5 + 0.5 * (1e-3 - 1e-5) * (
                        1 + math.cos(math.pi * progress)
                    )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_events = 0
            val_prediction = _predict(model, val, kind, device=device)
            if kind == "P_TARGET_TO_CLASS_ORACLE":
                val_loss = float(
                    torch.nn.functional.cross_entropy(
                        torch.from_numpy(val_prediction),
                        torch.from_numpy(val["labels"]).long(),
                    )
                )
            else:
                val_loss = _validation_target_loss(
                    val_prediction,
                    val["target"],
                    val["target_mask"],
                    row["target_id"],
                )
            candidate = (val_loss, epoch, {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            })
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            torch.save({
                "contract": "hosd_probe_checkpoint_v1",
                "schema_version": 1,
                "kind": "resumable_last",
                "row_id": args.row_id,
                "pipeline_seed": int(row["pipeline_seed"]),
                "component_seed": int(row["component_seed"]),
                "epoch_completed": epoch,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update_ordinal,
                "best": best,
                "stage_c_plan_sha256": plan["content_hash"],
                "probe_encoder_lock_sha256": lock_sha,
            }, last_checkpoint)
        model.load_state_dict(best[2], strict=True)
        checkpoint = output / "best_probe_val.pt"
        torch.save({
            "contract": "hosd_probe_checkpoint_v1",
            "schema_version": 1,
            "row_id": args.row_id,
            "pipeline_seed": int(row["pipeline_seed"]),
            "component_seed": int(row["component_seed"]),
            "selected_epoch": best[1],
            "model_state_dict": best[2],
            "stage_c_plan_sha256": plan["content_hash"],
            "probe_encoder_lock_sha256": lock_sha,
        }, checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if last_checkpoint.exists():
            last_checkpoint.unlink()
        prediction = _predict(model, design, kind, device=device)
        if kind == "P_TARGET_TO_CLASS_ORACLE":
            predicted_class = prediction.argmax(1)
            metrics = {
                "accuracy": float((predicted_class == design["labels"]).mean()),
                "intrinsic_class_information_oracle": True,
            }
        else:
            metrics = _target_metrics(row, prediction, design)
    result = build_probe_result(
        row=row,
        split="design_select",
        metrics=metrics,
        identity_order_sha256=_identity_hash(design),
        target_cache_manifest_sha256=cache_sha,
        probe_encoder_lock_sha256=lock_sha,
        checkpoint_sha256=checkpoint_sha,
        input_artifact_hashes={
            "model_train": _sha256_file(args.train_npz),
            "val_stop": _sha256_file(args.val_stop_npz),
            "design_select": _sha256_file(args.design_select_npz),
        },
        source=campaign["source"],
    )
    write_immutable_json(output / "probe_result.json", result)
    completion = with_content_hash({
        "contract": PROBE_COMPLETION_CONTRACT,
        "schema_version": 1,
        "source": campaign["source"],
        "row_id": args.row_id,
        "stage_c_plan_sha256": plan["content_hash"],
        "probe_result_sha256": result["content_hash"],
        "performance_based_termination": False,
    })
    write_immutable_json(output / "probe_completion.json", completion)
    wave = _finalize_probe_wave(args.campaign_root, plan, campaign)
    print(json.dumps({
        "row_id": args.row_id,
        "probe_result_sha256": result["content_hash"],
        "completion_sha256": completion["content_hash"],
        "wave_completion_sha256": None if wave is None else wave["content_hash"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
