"""Fixed-budget frozen-token fusion training and design-split inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .contracts import (
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_classification
from .expert_training import preferred_expert_epoch
from .fusion_cache import load_frozen_token_cache
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FUSION_TRAINING_CONTRACT = "retb_offline_fusion_training_v1"
FUSION_CHECKPOINT_CONTRACT = "retb_offline_fusion_checkpoint_v1"
FUSION_REGISTRATION_CONTRACT = "retb_offline_fusion_registration_v1"
FUSION_CURVES_CONTRACT = "retb_offline_fusion_curves_v1"
FUSION_DESIGN_INFERENCE_CONTRACT = "retb_offline_fusion_val_design_v1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB fusion training")
    return torch


@dataclass(frozen=True)
class OfflineFusionTrainingConfig:
    seed: int
    variant: str = "F_TOKEN_TRANSFORMER"
    maximum_epochs: int = 40
    learning_rate: float = 5.0e-4
    minimum_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    batch_size: int = 512
    accuracy_window: float = 0.0001
    num_workers: int = 0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("fusion campaign profile is unknown")
        if self.seed not in {101, 202, 303, 41701, 41702, 41703}:
            raise ValueError("fusion seed is not registered")
        if self.variant not in {
            "F_TRAINED_LOGIT_LINEAR",
            "F_POOLED_MLP",
            "F_TOKEN_TRANSFORMER",
            "SUBSET_LOGIT_LINEAR",
            "SUBSET_POOLED_MLP",
        }:
            raise ValueError("cached fusion trainer received unsupported variant")
        if min(self.maximum_epochs, self.batch_size) <= 0:
            raise ValueError("fusion schedule is invalid")
        if self.num_workers != 0:
            raise ValueError("fusion num_workers must remain zero")
        if self.campaign_profile == "production" and (
            self.maximum_epochs != 40
            or self.learning_rate != 5.0e-4
            or self.batch_size != 512
            or self.weight_decay != 1.0e-4
            or self.gradient_clip != 1.0
        ):
            raise ValueError("production frozen-fusion protocol drifted")

    def artifact(
        self,
        *,
        global_determinism_sha256: str,
        fusion_architecture_sha256: str,
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": FUSION_TRAINING_CONTRACT,
                "schema_version": 1,
                "config": asdict(self),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "fusion_architecture_sha256": require_sha256(
                    fusion_architecture_sha256,
                    name="fusion_architecture_sha256",
                ),
                "optimizer": "AdamW",
                "betas": [0.9, 0.999],
                "schedule": "exact_integer_warmup_then_cosine",
                "epoch_selector": (
                    "val_stop_max_accuracy_0p0001_window_min_CE_earliest"
                ),
                "architecture_selection_split": "val_design_after_checkpoint_lock",
                "experts_frozen": True,
                "whole_bank_dropout": 0.0,
                "early_stopping": False,
                "performance_based_termination": False,
            }
        )


class FrozenFusionDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    def __init__(self, arrays: Mapping[str, Any]) -> None:
        _require_torch()
        self.identities = tuple(str(value) for value in arrays["identities"])
        self.labels = np.asarray(arrays["labels"], dtype=np.int64)
        self.token_banks = {
            name: np.asarray(arrays["token_banks"][name], dtype=np.float32)
            for name in EXPERT_ORDER
        }
        self.expert_logits = {
            name: np.asarray(arrays["expert_logits"][name], dtype=np.float32)
            for name in EXPERT_ORDER
        }
        if len(self.labels) != len(self.identities):
            raise ValueError("fusion dataset identity count differs")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "identity": self.identities[index],
            "label": self.labels[index],
            "token_banks": {
                name: self.token_banks[name][index] for name in EXPERT_ORDER
            },
            "expert_logits": {
                name: self.expert_logits[name][index] for name in EXPERT_ORDER
            },
        }


def _collate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    module = _require_torch()
    return {
        "identities": [row["identity"] for row in rows],
        "labels": module.as_tensor(
            [row["label"] for row in rows], dtype=module.long
        ),
        "token_banks": {
            name: module.from_numpy(
                np.stack([row["token_banks"][name] for row in rows])
            ).float()
            for name in EXPERT_ORDER
        },
        "expert_logits": {
            name: module.from_numpy(
                np.stack([row["expert_logits"][name] for row in rows])
            ).float()
            for name in EXPERT_ORDER
        },
    }


def make_fusion_loader(
    arrays: Mapping[str, Any],
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    module = _require_torch()
    dataset = FrozenFusionDataset(arrays)
    generator = module.Generator().manual_seed(int(seed))
    return module.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(training),
        generator=generator if training else None,
        num_workers=0,
        drop_last=False,
        collate_fn=_collate,
    )


def _move(value: Any, device: Any) -> Any:
    module = _require_torch()
    if isinstance(value, module.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    return value


def _forward(model: Any, batch: Mapping[str, Any]) -> Any:
    return model(
        token_banks=batch["token_banks"],
        expert_logits=batch["expert_logits"],
    )


def evaluate_fusion(
    model: Any,
    loader: Any,
    *,
    device: Any,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _require_torch()
    prior = bool(model.training)
    model.eval()
    logits = []
    labels = []
    identities = []
    try:
        with module.no_grad():
            for raw in loader:
                batch = _move(raw, device)
                output = _forward(model, batch)
                if not bool(module.isfinite(output).all()):
                    raise FloatingPointError("fusion logits are nonfinite")
                logits.append(output.float().cpu().numpy())
                labels.append(batch["labels"].cpu().numpy())
                identities.extend(raw["identities"])
    finally:
        model.train(prior)
    values = np.concatenate(logits)
    truth = np.concatenate(labels)
    if len(identities) != len(set(identities)):
        raise ValueError("fusion evaluation identities are duplicated")
    metrics = evaluate_classification(values, truth, split=split)
    return metrics, {
        "logits": values.astype(np.float32),
        "labels": truth.astype(np.int64),
        "identities": np.asarray(identities),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    module = _require_torch()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        module.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_frozen_fusion(
    *,
    model: Any,
    model_train_manifest: str | Path,
    val_stop_manifest: str | Path,
    output_dir: str | Path,
    run_id: str,
    run_registry_sha256: str,
    global_determinism_sha256: str,
    fusion_architecture_sha256: str,
    config: OfflineFusionTrainingConfig,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    train_meta, train_arrays = load_frozen_token_cache(model_train_manifest)
    val_meta, val_arrays = load_frozen_token_cache(val_stop_manifest)
    if train_meta["split"] != "model_train" or val_meta["split"] != "val_stop":
        raise ValueError("fusion trainer received unauthorized cache splits")
    lineage_keys = (
        "pipeline_seed",
        "shape_id",
        "allocation",
        "expert_checkpoint_hashes",
        "expert_registration_hashes",
    )
    if any(train_meta[key] != val_meta[key] for key in lineage_keys):
        raise ValueError("fusion train/val cache lineage differs")
    if train_meta["pipeline_seed"] != config.seed:
        raise ValueError("fusion cache and training seed differ")
    contract = config.artifact(
        global_determinism_sha256=global_determinism_sha256,
        fusion_architecture_sha256=fusion_architecture_sha256,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "fusion_registration.json"
    if registration_path.exists():
        registration = load_hashed_json(
            registration_path, expected_contract=FUSION_REGISTRATION_CONTRACT
        )
        expected = {
            "run_id": run_id,
            "training_contract_sha256": contract["content_hash"],
            "model_train_cache_sha256": train_meta["content_hash"],
            "val_stop_cache_sha256": val_meta["content_hash"],
        }
        if any(registration.get(k) != v for k, v in expected.items()):
            raise ValueError("reusable fusion registration lineage differs")
        if _sha256(root / "best_model_val.pt") != registration["checkpoint_sha256"]:
            raise ValueError("reusable fusion checkpoint bytes differ")
        return registration
    resolved = module.device(device)
    if config.campaign_profile == "production":
        if (
            resolved.type != "cuda"
            or not module.cuda.is_bf16_supported()
            or "GH200" not in module.cuda.get_device_name(resolved).upper()
        ):
            raise RuntimeError("production frozen fusion requires GH200 BF16")
    model.to(resolved)
    train_loader = make_fusion_loader(
        train_arrays,
        batch_size=config.batch_size,
        seed=config.seed,
        training=True,
    )
    val_loader = make_fusion_loader(
        val_arrays,
        batch_size=config.batch_size,
        seed=config.seed,
        training=False,
    )
    optimizer = module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_arrays["labels"]),
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.batch_size,
        gradient_accumulation_steps=1,
    )
    rows = []
    candidates: dict[int, dict[str, Any]] = {}
    update = 0
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        objective_sum = 0.0
        event_count = 0
        for raw in train_loader:
            batch = _move(raw, resolved)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = module.nn.functional.cross_entropy(
                logits, batch["labels"], reduction="mean"
            )
            if not bool(module.isfinite(loss)):
                raise FloatingPointError("fusion objective is nonfinite")
            loss.backward()
            norm = module.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not bool(module.isfinite(norm)):
                raise FloatingPointError("fusion gradient is nonfinite")
            update += 1
            lr = scheduled_learning_rate(
                update_ordinal=update,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=config.learning_rate,
                minimum_learning_rate=config.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            events = int(batch["labels"].numel())
            objective_sum += float(loss.detach().cpu()) * events
            event_count += events
        metrics, _ = evaluate_fusion(
            model, val_loader, device=resolved, split="val_stop"
        )
        row = {
            "epoch": epoch,
            "optimizer_update_ordinal": update,
            "learning_rate": lr,
            "train_cross_entropy": objective_sum / event_count,
            "val_stop": {
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
            },
        }
        rows.append(row)
        candidates[epoch] = {
            name: value.detach().cpu().clone()
            if isinstance(value, module.Tensor)
            else value
            for name, value in model.state_dict().items()
        }
        maximum = max(r["val_stop"]["accuracy"] for r in rows)
        keep = {
            r["epoch"]
            for r in rows
            if maximum - r["val_stop"]["accuracy"] <= config.accuracy_window
        }
        candidates = {key: value for key, value in candidates.items() if key in keep}
    selected = preferred_expert_epoch(
        rows, accuracy_window=config.accuracy_window
    )
    model.load_state_dict(candidates[int(selected["epoch"])], strict=True)
    final_metrics, _ = evaluate_fusion(
        model, val_loader, device=resolved, split="val_stop"
    )
    checkpoint = root / "best_model_val.pt"
    _atomic_save(
        {
            "contract": FUSION_CHECKPOINT_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "epoch": int(selected["epoch"]),
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": require_sha256(
                run_registry_sha256, name="run_registry_sha256"
            ),
            "model_train_cache_sha256": train_meta["content_hash"],
            "val_stop_cache_sha256": val_meta["content_hash"],
            "shape_id": train_meta["shape_id"],
            "allocation": train_meta["allocation"],
            "expert_checkpoint_hashes": train_meta[
                "expert_checkpoint_hashes"
            ],
            "model_state_dict": candidates[int(selected["epoch"])],
        },
        checkpoint,
    )
    curves = with_content_hash(
        {
            "contract": FUSION_CURVES_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "rows": rows,
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_budget_completed": len(rows) == config.maximum_epochs,
            "stopped_early": False,
            "performance_based_termination": False,
            "optimizer_update_counts": counts,
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    write_immutable_json(root / "val_stop_metrics.json", final_metrics)
    registration = with_content_hash(
        {
            "contract": FUSION_REGISTRATION_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "seed": config.seed,
            "variant": config.variant,
            "shape_id": train_meta["shape_id"],
            "allocation": train_meta["allocation"],
            "training_contract_sha256": contract["content_hash"],
            "model_train_cache_sha256": train_meta["content_hash"],
            "val_stop_cache_sha256": val_meta["content_hash"],
            "checkpoint_sha256": _sha256(checkpoint),
            "selected_epoch": int(selected["epoch"]),
            "selected_val_stop_metrics_sha256": final_metrics["content_hash"],
            "training_curves_sha256": curves["content_hash"],
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": True,
            "stopped_early": False,
            "performance_based_termination": False,
            "expert_checkpoint_hashes": train_meta["expert_checkpoint_hashes"],
            "expert_parameters_updated": False,
            "whole_bank_dropout": 0.0,
            "retained_checkpoints": ["best_model_val.pt"],
        }
    )
    write_immutable_json(registration_path, registration)
    return registration


def infer_fusion_val_design(
    *,
    model: Any,
    checkpoint_path: str | Path,
    val_design_manifest: str | Path,
    output_path: str | Path,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    module = _require_torch()
    checkpoint = module.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("contract") != FUSION_CHECKPOINT_CONTRACT:
        raise ValueError("fusion design inference checkpoint contract differs")
    cache_meta, arrays = load_frozen_token_cache(val_design_manifest)
    if cache_meta["split"] != "val_design":
        raise ValueError("fusion design inference requires val_design")
    if (
        cache_meta["shape_id"] != checkpoint.get("shape_id")
        or cache_meta["allocation"] != checkpoint.get("allocation")
        or cache_meta["expert_checkpoint_hashes"]
        != checkpoint.get("expert_checkpoint_hashes")
    ):
        raise ValueError("fusion design cache shape differs")
    resolved = module.device(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved)
    loader = make_fusion_loader(
        arrays, batch_size=512, seed=0, training=False
    )
    metrics, prediction = evaluate_fusion(
        model, loader, device=resolved, split="val_design"
    )
    result = with_content_hash(
        {
            "contract": FUSION_DESIGN_INFERENCE_CONTRACT,
            "schema_version": 1,
            "split": "val_design",
            "checkpoint_sha256": _sha256(Path(checkpoint_path)),
            "cache_manifest_sha256": cache_meta["content_hash"],
            "metrics": metrics,
            "event_count": len(prediction["labels"]),
            "identity_order_sha256": hashlib.sha256(
                "\n".join(map(str, prediction["identities"])).encode()
            ).hexdigest(),
            "checkpoint_selection_affected": False,
        }
    )
    write_immutable_json(output_path, result)
    return result


__all__ = [
    "FUSION_CHECKPOINT_CONTRACT",
    "FUSION_DESIGN_INFERENCE_CONTRACT",
    "FUSION_REGISTRATION_CONTRACT",
    "FUSION_TRAINING_CONTRACT",
    "OfflineFusionTrainingConfig",
    "evaluate_fusion",
    "infer_fusion_val_design",
    "make_fusion_loader",
    "train_frozen_fusion",
]
