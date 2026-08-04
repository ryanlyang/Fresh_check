#!/usr/bin/env python3
"""Train one streamed CE/KD/mixed frozen-expert supplemental fusion bank."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_retb_frozen_token_cache import _infer_expert  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.determinism import (  # noqa: E402
    optimizer_update_counts,
    scheduled_learning_rate,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    BankProjection,
    FusionTransformerBlock,
    RMSNorm,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import (  # noqa: E402
    FUSION_VARIANTS,
    SUPPLEMENTAL_BANK_RESULT_CONTRACT,
    SUPPLEMENTAL_PLAN_CONTRACT,
    file_sha256,
    select_fixed_budget_checkpoint,
    set_deterministic_seed,
    validate_supplemental_plan,
)


class _BankDataset(torch.utils.data.Dataset):
    def __init__(self, arrays: Mapping[str, Any], order: Sequence[str]) -> None:
        self.order = tuple(order)
        self.labels = torch.as_tensor(arrays["labels"], dtype=torch.long)
        self.tokens = {
            name: torch.as_tensor(arrays["tokens"][name], dtype=torch.float32)
            for name in self.order
        }
        self.logits = {
            name: torch.as_tensor(arrays["logits"][name], dtype=torch.float32)
            for name in self.order
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "labels": self.labels[index],
            "tokens": {name: self.tokens[name][index] for name in self.order},
            "logits": {name: self.logits[name][index] for name in self.order},
        }


class _LogitLinear(torch.nn.Module):
    def __init__(self, order: Sequence[str]) -> None:
        super().__init__()
        self.order = tuple(order)
        self.classifier = torch.nn.Linear(10 * len(self.order), 10)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.classifier(
            torch.cat([batch["logits"][name] for name in self.order], dim=-1)
        )


class _PooledMLP(torch.nn.Module):
    def __init__(self, order: Sequence[str], dimensions: Mapping[str, int]) -> None:
        super().__init__()
        self.order = tuple(order)
        self.projections = torch.nn.ModuleDict(
            {name: BankProjection(int(dimensions[name])) for name in self.order}
        )
        self.classifier = torch.nn.Sequential(
            RMSNorm(128 * len(self.order)),
            torch.nn.Linear(128 * len(self.order), 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 10),
        )

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        values = [
            self.projections[name](batch["tokens"][name]).mean(dim=1)
            for name in self.order
        ]
        return self.classifier(torch.cat(values, dim=-1))


class _TokenTransformer(torch.nn.Module):
    def __init__(self, order: Sequence[str], dimensions: Mapping[str, int]) -> None:
        super().__init__()
        self.order = tuple(order)
        self.projections = torch.nn.ModuleDict(
            {name: BankProjection(int(dimensions[name])) for name in self.order}
        )
        self.expert_embedding = torch.nn.Embedding(len(self.order), 128)
        self.slot_embedding = torch.nn.Embedding(16, 128)
        self.source_embedding = torch.nn.Parameter(torch.zeros(128))
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, 128))
        torch.nn.init.normal_(self.class_token, std=0.02)
        self.blocks = torch.nn.ModuleList(
            [FusionTransformerBlock() for _ in range(3)]
        )
        self.norm = RMSNorm(128)
        self.classifier = torch.nn.Linear(128, 10)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        rows = []
        for index, name in enumerate(self.order):
            values = self.projections[name](batch["tokens"][name])
            slots = int(values.shape[1])
            if slots > 16:
                raise ValueError("supplemental bank exceeds 16 summary tokens")
            slot_ids = torch.arange(slots, device=values.device)
            rows.append(
                values
                + self.expert_embedding.weight[index].view(1, 1, -1)
                + self.slot_embedding(slot_ids).view(1, slots, -1)
                + self.source_embedding.view(1, 1, -1)
            )
        sequence = torch.cat(rows, dim=1)
        sequence = torch.cat(
            (self.class_token.expand(sequence.shape[0], -1, -1), sequence),
            dim=1,
        )
        for block in self.blocks:
            sequence = block(sequence)
        return self.classifier(self.norm(sequence[:, 0]))


def _model(
    variant: str, order: Sequence[str], dimensions: Mapping[str, int]
) -> torch.nn.Module:
    if variant == "TRAINED_LOGIT_LINEAR":
        return _LogitLinear(order)
    if variant == "POOLED_MLP":
        return _PooledMLP(order, dimensions)
    if variant == "TOKEN_TRANSFORMER":
        return _TokenTransformer(order, dimensions)
    raise ValueError("supplemental learned fusion variant differs")


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    return value


def _loader(
    arrays: Mapping[str, Any],
    order: Sequence[str],
    *,
    training: bool,
    seed: int,
    batch_size: int,
) -> Any:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        _BankDataset(arrays, order),
        batch_size=int(batch_size),
        shuffle=bool(training),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _mean_logits(batch: Mapping[str, Any], order: Sequence[str]) -> torch.Tensor:
    return torch.stack([batch["logits"][name] for name in order], dim=0).mean(0)


def _evaluate(
    model: torch.nn.Module | None,
    loader: Any,
    *,
    order: Sequence[str],
    split: str,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if model is not None:
        model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            values = _mean_logits(batch, order) if model is None else model(batch)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("supplemental fusion logits are nonfinite")
            logits.append(values.float().cpu())
            labels.append(batch["labels"].long().cpu())
    scores = torch.cat(logits).numpy()
    truth = torch.cat(labels).numpy()
    return evaluate_classification(scores, truth, split=split), scores, truth


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _train_variant(
    *,
    variant: str,
    bank_id: str,
    order: Sequence[str],
    dimensions: Mapping[str, int],
    arrays: Mapping[str, Mapping[str, Any]],
    output: Path,
    plan: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    variant_root = output / variant
    result_path = variant_root / "result.json"
    if result_path.is_file():
        existing = load_hashed_json(result_path)
        if (
            existing.get("plan_sha256") != plan["content_hash"]
            or existing.get("bank_id") != bank_id
            or existing.get("variant") != variant
        ):
            raise ValueError("reusable supplemental fusion result differs")
        return existing
    variant_root.mkdir(parents=True, exist_ok=True)
    val_stop_loader = _loader(
        arrays["val_stop"], order, training=False, seed=101, batch_size=batch_size
    )
    val_design_loader = _loader(
        arrays["val_design"], order, training=False, seed=101, batch_size=batch_size
    )
    if variant == "MEAN_LOGITS":
        stop_metrics, _, _ = _evaluate(
            None, val_stop_loader, order=order, split="val_stop", device=device
        )
        design_metrics, scores, truth = _evaluate(
            None, val_design_loader, order=order, split="val_design", device=device
        )
        checkpoint_sha = None
        rows: list[dict[str, Any]] = []
        selected_epoch = None
    else:
        seed = 101 + list(FUSION_VARIANTS).index(variant)
        set_deterministic_seed(seed)
        model = _model(variant, order, dimensions).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0e-3, betas=(0.9, 0.999), weight_decay=1.0e-4
        )
        train_loader = _loader(
            arrays["model_train"],
            order,
            training=True,
            seed=seed,
            batch_size=batch_size,
        )
        rows = []
        candidates: dict[int, Mapping[str, Any]] = {}
        resume_path = variant_root / "resume_state.pt"
        start_epoch = 1
        update = 0
        counts = optimizer_update_counts(
            training_event_count=len(train_loader.dataset),
            maximum_epochs=40,
            microbatch_size=batch_size,
            gradient_accumulation_steps=1,
        )
        if resume_path.is_file():
            resume = torch.load(resume_path, map_location=device, weights_only=False)
            if (
                resume.get("plan_sha256") != plan["content_hash"]
                or resume.get("bank_id") != bank_id
                or resume.get("variant") != variant
            ):
                raise ValueError("supplemental fusion resume lineage differs")
            model.load_state_dict(resume["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            rows = list(resume["rows"])
            candidates = dict(resume["candidates"])
            start_epoch = int(resume["next_epoch"])
            update = int(resume["optimizer_updates_completed"])
        for epoch in range(start_epoch, 41):
            model.train()
            objective, count = 0.0, 0
            for raw in train_loader:
                batch = _move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                values = model(batch)
                loss = torch.nn.functional.cross_entropy(values, batch["labels"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("supplemental fusion loss is nonfinite")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("supplemental fusion gradient is nonfinite")
                update += 1
                learning_rate = scheduled_learning_rate(
                    update_ordinal=update,
                    total_optimizer_updates=counts["total_optimizer_updates"],
                    warmup_updates=counts["warmup_updates"],
                    base_learning_rate=1.0e-3,
                    minimum_learning_rate=1.0e-5,
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                size = int(batch["labels"].numel())
                objective += float(loss.detach().cpu()) * size
                count += size
            metrics, _, _ = _evaluate(
                model,
                val_stop_loader,
                order=order,
                split="val_stop",
                device=device,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "optimizer_updates_completed": update,
                    "learning_rate": learning_rate,
                    "train_cross_entropy": objective / count,
                    "val_stop": {
                        "accuracy": metrics["accuracy"],
                        "cross_entropy": metrics["cross_entropy"],
                    },
                }
            )
            candidates[epoch] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best = max(float(row["val_stop"]["accuracy"]) for row in rows)
            keep = {
                int(row["epoch"])
                for row in rows
                if best - float(row["val_stop"]["accuracy"]) <= 0.0001
            }
            candidates = {key: value for key, value in candidates.items() if key in keep}
            _atomic_torch_save(
                {
                    "plan_sha256": plan["content_hash"],
                    "bank_id": bank_id,
                    "variant": variant,
                    "next_epoch": epoch + 1,
                    "optimizer_updates_completed": update,
                    "rows": rows,
                    "candidates": candidates,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                resume_path,
            )
        if update != counts["total_optimizer_updates"]:
            raise RuntimeError(
                "supplemental fusion fixed optimizer-update budget was incomplete"
            )
        selected_epoch = select_fixed_budget_checkpoint(rows)
        model.load_state_dict(candidates[selected_epoch], strict=True)
        stop_metrics, _, _ = _evaluate(
            model, val_stop_loader, order=order, split="val_stop", device=device
        )
        design_metrics, scores, truth = _evaluate(
            model, val_design_loader, order=order, split="val_design", device=device
        )
        checkpoint = variant_root / "best_model_val.pt"
        _atomic_torch_save(
            {
                "contract": "retb_supplemental_offline_fusion_checkpoint_v1",
                "schema_version": 1,
                "plan_sha256": plan["content_hash"],
                "bank_id": bank_id,
                "variant": variant,
                "selected_epoch": selected_epoch,
                "model_state_dict": candidates[selected_epoch],
            },
            checkpoint,
        )
        checkpoint_sha = file_sha256(checkpoint)
        resume_path.unlink(missing_ok=True)
    predictions = variant_root / "val_design_predictions.npz"
    if predictions.is_file():
        with np.load(predictions, allow_pickle=False) as prior:
            if not (
                set(prior.files) == {"logits", "labels"}
                and np.array_equal(prior["logits"], scores)
                and np.array_equal(prior["labels"], truth)
            ):
                raise FileExistsError(
                    "supplemental fusion predictions differ on restart"
                )
    else:
        with predictions.open("xb") as handle:
            np.savez_compressed(handle, logits=scores, labels=truth)
    result = bind_source(
        with_content_hash(
            {
                "contract": SUPPLEMENTAL_BANK_RESULT_CONTRACT,
                "schema_version": 1,
                "plan_sha256": plan["content_hash"],
                "bank_id": bank_id,
                "variant": variant,
                "expert_order": list(order),
                "selected_epoch": selected_epoch,
                "epochs_completed": len(rows),
                "optimizer_update_counts": (
                    None if variant == "MEAN_LOGITS" else counts
                ),
                "fixed_budget_completed": variant == "MEAN_LOGITS" or len(rows) == 40,
                "performance_based_termination": False,
                "checkpoint_sha256": checkpoint_sha,
                "val_stop_metrics": stop_metrics,
                "val_design_metrics": design_metrics,
                "val_design_predictions_sha256": file_sha256(predictions),
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    plan = load_hashed_json(
        args.plan, expected_contract=SUPPLEMENTAL_PLAN_CONTRACT
    )
    validate_supplemental_plan(plan)
    if args.batch_size != int(plan["fusion_training_protocol"]["batch_size"]):
        raise ValueError("supplemental fusion batch size differs from its plan")
    if args.bank_id not in plan["banks"]:
        raise ValueError("unknown supplemental fusion bank")
    bank = plan["banks"][args.bank_id]
    parent = Path(plan["parent_campaign_root"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type != "cuda" or "GH200" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("production supplemental fusion requires a GH200")
    relation = load_hashed_json(
        parent / "inputs/normalization/offline_500k/relation.json"
    )
    region = load_hashed_json(
        parent / "inputs/normalization/offline_500k/region.json"
    )
    arrays: dict[str, dict[str, Any]] = {}
    for split in ("model_train", "val_stop", "val_design"):
        input_path = parent / "inputs" / "offline" / split / "offline_inputs.npz"
        with np.load(input_path, allow_pickle=False) as payload:
            raw = {name: np.asarray(payload[name]) for name in payload.files}
        tokens, logits = {}, {}
        for member in bank["members"]:
            registration = load_hashed_json(member["registration_path"])
            values, scores, _ = _infer_expert(
                expert=member["expert_id"],
                run={
                    "run_id": member["run_id"],
                    "seed": 101,
                    "configuration": member["configuration"],
                },
                registration=registration,
                checkpoint_path=Path(member["checkpoint_path"]),
                arrays=raw,
                relation_normalization=relation,
                region_normalization=region,
                region_tree_root=parent / "inputs/region_tree/offline",
                split=split,
                batch_size=128,
                device=device,
                collect_relation_sensitivity=False,
            )
            tokens[member["expert_id"]] = values
            logits[member["expert_id"]] = scores
        arrays[split] = {"labels": raw["labels"], "tokens": tokens, "logits": logits}
    dimensions = {
        name: int(arrays["model_train"]["tokens"][name].shape[-1])
        for name in bank["expert_order"]
    }
    results = [
        _train_variant(
            variant=variant,
            bank_id=args.bank_id,
            order=bank["expert_order"],
            dimensions=dimensions,
            arrays=arrays,
            output=args.output_dir,
            plan=plan,
            batch_size=args.batch_size,
            device=device,
        )
        for variant in FUSION_VARIANTS
    ]
    print(json.dumps({"bank_id": args.bank_id, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
