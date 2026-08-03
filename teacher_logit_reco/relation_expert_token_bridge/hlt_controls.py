"""Executable matched native-HLT single-model controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from teacher_logit_reco.relational_part.model import (
    RelationalParticleTransformer,
    WideBaseParticleTransformer,
)

from .contracts import (
    load_hashed_json,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_classification
from .expert_training import preferred_expert_epoch
from .hlt_experts import HLT_EVALUATION_REALIZATION_POLICY

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


HLT_CONTROL_TRAINING_CONTRACT = "retb_native_hlt_control_training_v2"
HLT_CONTROL_REGISTRATION_CONTRACT = "retb_native_hlt_control_registration_v2"
HLT_CONTROL_CURVES_CONTRACT = "retb_native_hlt_control_curves_v2"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for native HLT controls")
    return torch


class NativeHLTClassifierAdapter(
    torch.nn.Module if torch is not None else object
):
    def __init__(self, classifier: Any):
        super().__init__()
        self.classifier = classifier

    def forward(self, *, features: Any, vectors: Any, mask: Any, **_kwargs: Any):
        return self.classifier(
            points=features[:, -2:, :],
            features=features,
            lorentz_vectors=vectors,
            mask=mask,
        )


def build_hlt_matched_control_model(
    control_id: str,
    *,
    weaver_module: Any,
    wide_capacity_artifact: Mapping[str, Any] | None = None,
) -> Any:
    if control_id == "H_BASE":
        model = RelationalParticleTransformer(weaver_module=weaver_module)
    elif control_id == "H_WIDE":
        if wide_capacity_artifact is None:
            raise ValueError("H_WIDE requires the locked capacity artifact")
        model = WideBaseParticleTransformer(
            capacity_artifact=wide_capacity_artifact,
            weaver_module=weaver_module,
        )
    else:
        raise ValueError("aggregate unbiased controls route through native fusion")
    return NativeHLTClassifierAdapter(model)


@dataclass(frozen=True)
class NativeHLTControlTrainingConfig:
    seed: int
    control_id: str
    maximum_epochs: int = 40
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 128
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.control_id not in {"H_BASE", "H_WIDE"}:
            raise ValueError("native HLT single-model control is unknown")
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("native HLT control profile is unknown")
        if (
            self.microbatch_size * self.gradient_accumulation_steps
            != self.effective_batch_size
        ):
            raise ValueError("native HLT control effective batch differs")
        if self.campaign_profile == "production" and (
            self.seed not in {101, 202, 303}
            or self.maximum_epochs != 40
            or self.effective_batch_size != 128
        ):
            raise ValueError("native HLT control production protocol drifted")

    def artifact(self, *, global_determinism_sha256: str) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": HLT_CONTROL_TRAINING_CONTRACT,
                "schema_version": 2,
                "config": asdict(self),
                "training_realization_policy": "R_MULTI",
                "evaluation_realization_policy": (
                    HLT_EVALUATION_REALIZATION_POLICY
                ),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "objective": "unweighted_HLT_cross_entropy",
                "offline_targets_permitted": False,
                "fixed_epoch_budget": True,
                "performance_based_termination": False,
            }
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    module = _require_torch()
    return {
        name: value.to(device) if isinstance(value, module.Tensor) else value
        for name, value in batch.items()
    }


def _inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: batch[name]
        for name in ("features", "vectors", "mask", "raw_tokens", "region_trees")
        if name in batch
    }


def _precision(device: Any) -> tuple[bool, Any | None, str]:
    module = _require_torch()
    resolved = module.device(device)
    enabled = resolved.type == "cuda"
    if enabled and not module.cuda.is_bf16_supported():
        raise RuntimeError("production native HLT control requires BF16")
    return enabled, module.bfloat16 if enabled else None, (
        "bf16" if enabled else "fp32"
    )


def _evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    module = _require_torch()
    enabled, dtype, _ = _precision(device)
    model.eval()
    logits, labels = [], []
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with module.autocast(
                device_type=module.device(device).type,
                dtype=dtype,
                enabled=enabled,
            ):
                output = model(**_inputs(batch))
            if not bool(module.isfinite(output).all()):
                raise FloatingPointError("native HLT control logits are nonfinite")
            logits.append(output.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return evaluate_classification(
        np.concatenate(logits), np.concatenate(labels), split="val_stop"
    )


def train_native_hlt_control(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    output_dir: str | Path,
    run_id: str,
    run_registry_sha256: str,
    lineage_hashes: Mapping[str, str],
    global_determinism_sha256: str,
    config: NativeHLTControlTrainingConfig,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    if (
        getattr(train_loader.dataset, "logical_role", None) != "model_train"
        or getattr(val_stop_loader.dataset, "logical_role", None) != "val_stop"
        or getattr(train_loader.dataset, "realization_policy", None) != "R_MULTI"
        or getattr(val_stop_loader.dataset, "realization_policy", None)
        != HLT_EVALUATION_REALIZATION_POLICY
        or getattr(train_loader.dataset, "offline_target_tokens", None) is not None
        or getattr(val_stop_loader.dataset, "offline_target_tokens", None) is not None
    ):
        raise ValueError("native HLT control loader contract differs")
    contract = config.artifact(
        global_determinism_sha256=global_determinism_sha256
    )
    registry_sha = require_sha256(run_registry_sha256, name="run_registry_sha256")
    parents = {
        name: require_sha256(value, name=f"lineage_hashes.{name}")
        for name, value in sorted(lineage_hashes.items())
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "checkpoint_registration.json"
    if registration_path.exists():
        registration = load_hashed_json(
            registration_path,
            expected_contract=HLT_CONTROL_REGISTRATION_CONTRACT,
        )
        checkpoint = root / "best_model_val.pt"
        if (
            registration["run_id"] != run_id
            or registration["training_contract_sha256"] != contract["content_hash"]
            or registration["run_registry_sha256"] != registry_sha
            or registration["lineage_hashes"] != parents
            or not checkpoint.is_file()
            or _sha256(checkpoint) != registration["checkpoint_sha256"]
        ):
            raise ValueError("reusable native HLT control differs")
        return registration
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError("production native HLT control requires GH200")
    enabled, dtype, precision_mode = _precision(resolved)
    model.to(resolved)
    optimizer = module.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=1.0e-4
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    rows, best_state, update = [], None, 0
    for epoch in range(1, config.maximum_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved)
            current = int(batch["labels"].numel())
            with module.autocast(
                device_type=resolved.type, dtype=dtype, enabled=enabled
            ):
                logits = model(**_inputs(batch))
                loss = module.nn.functional.cross_entropy(
                    logits, batch["labels"]
                )
            if not bool(module.isfinite(loss)):
                raise FloatingPointError("native HLT control loss is nonfinite")
            (loss * current).backward()
            accumulated += current
            if (
                batch_index % config.gradient_accumulation_steps
                and batch_index != len(train_loader)
            ):
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
            norm = module.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(module.isfinite(norm)):
                raise FloatingPointError("native HLT control gradient is nonfinite")
            update += 1
            lr = scheduled_learning_rate(
                update_ordinal=update,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=1.0e-3,
                minimum_learning_rate=1.0e-5,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        metrics = _evaluate(model, val_stop_loader, resolved)
        rows.append(
            {
                "epoch": epoch,
                "val_stop": {
                    "accuracy": metrics["accuracy"],
                    "cross_entropy": metrics["cross_entropy"],
                },
            }
        )
        if int(preferred_expert_epoch(rows)["epoch"]) == epoch:
            best_state = {
                name: value.detach().cpu().clone()
                if isinstance(value, module.Tensor)
                else value
                for name, value in model.state_dict().items()
            }
    if best_state is None or update != counts["total_optimizer_updates"]:
        raise RuntimeError("native HLT control fixed budget drifted")
    selected = preferred_expert_epoch(rows)
    model.load_state_dict(best_state, strict=True)
    metrics = _evaluate(model, val_stop_loader, resolved)
    checkpoint = root / "best_model_val.pt"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".best_model_val.", suffix=".tmp", dir=root
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        module.save(
            {
                "run_id": run_id,
                "model_state_dict": best_state,
                "selected_epoch": int(selected["epoch"]),
                "training_contract_sha256": contract["content_hash"],
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    curves = with_content_hash(
        {
            "contract": HLT_CONTROL_CURVES_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "rows": rows,
            "optimizer_update_counts": counts,
            "fixed_budget_completed": True,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    registration = with_content_hash(
        {
            "contract": HLT_CONTROL_REGISTRATION_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "control_id": config.control_id,
            "seed": config.seed,
            "training_realization_policy": "R_MULTI",
            "evaluation_realization_policy": (
                HLT_EVALUATION_REALIZATION_POLICY
            ),
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": registry_sha,
            "lineage_hashes": parents,
            "checkpoint_sha256": _sha256(checkpoint),
            "training_curves_sha256": curves["content_hash"],
            "val_stop_metrics": metrics,
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": True,
            "offline_targets_consumed": False,
            "performance_based_termination": False,
            "precision_mode": precision_mode,
        }
    )
    write_immutable_json(registration_path, registration)
    return registration


__all__ = [
    "NativeHLTClassifierAdapter",
    "NativeHLTControlTrainingConfig",
    "build_hlt_matched_control_model",
    "train_native_hlt_control",
]
