"""Training for Architecture-View Residual ParT variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part.train import (
    _manifest_metadata,
    _metrics_without_prediction_arrays,
    _selection_score,
    _verify_hlt_params,
    local_compression_label_filter_names_to_indices,
)
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .checkpoint import (
    compute_architecture_view_init_logit_diff_vs_baseline,
    sha256_file,
    warm_start_architecture_view_part_model,
)
from .config import (
    ARCHITECTURE_VIEW_BINARY_LABEL_FILTER,
    ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH,
    ARCHITECTURE_VIEW_LABEL_NAMES,
    ARCHITECTURE_VIEW_PART_CONTRACT,
    ARCHITECTURE_VIEW_PRIMARY_METRIC,
    ARCHITECTURE_VIEW_PRIMARY_METRIC_DIRECTION,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANTS,
    ArchitectureViewConfig,
    normalize_architecture_view_variant,
)
from .model import (
    ARCHITECTURE_VIEW_MODEL_CONTRACT,
    ARCHITECTURE_VIEW_MODEL_STEP,
    ArchitectureViewResidualParT,
    build_architecture_view_residual_part,
)


ARCHITECTURE_VIEW_TRAIN_STEP = "architecture_view_part_step3_train"
ARCHITECTURE_VIEW_TRAIN_CONTRACT = f"{ARCHITECTURE_VIEW_PART_CONTRACT}_train_v1"
ARCHITECTURE_VIEW_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "macro_per_class_accuracy",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
ARCHITECTURE_VIEW_LOWER_IS_BETTER_SELECTION_METRICS = {
    "loss",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
}


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if np.isfinite(output) else float(default)


def architecture_view_label_filter_names_to_indices(
    values: Sequence[str],
    *,
    manifest_path: str | Path | None,
    label_names: Sequence[str] = ARCHITECTURE_VIEW_LABEL_NAMES,
) -> tuple[int, ...]:
    return local_compression_label_filter_names_to_indices(
        values,
        manifest_path=manifest_path,
        label_names=label_names,
    )


@dataclass
class ArchitectureViewTaggerTrainConfig:
    """Training config for one architecture-view residual ParT run."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    baseline_checkpoint: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = True
    seed: int = 6207
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    adapter_lr: float = 3.0e-4
    part_lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_stack_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    selection_metric: str = ARCHITECTURE_VIEW_PRIMARY_METRIC
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    require_baseline_split_manifest_hash: bool = True
    expected_hlt_degradation_strength: float = ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH
    label_names: tuple[str, ...] = ARCHITECTURE_VIEW_LABEL_NAMES
    label_filter: tuple[int, ...] = ARCHITECTURE_VIEW_BINARY_LABEL_FILTER
    variant: str = "av_all_views"
    view_dim: int = 32
    hidden_dim: int = 64
    pn_k: int = 16
    pn_layers: int = 2
    pfn_hidden_dim: int = 64
    pcnn_channels: int = 64
    pcnn_layers: int = 2
    fusion_hidden_dim: int = 96
    part_embed_dim: int = 128
    dropout: float = 0.05
    attention_dropout: float = 0.05
    gate_bias_init: float = -5.0
    random_control_seed: int = 2907
    delta_l2_weight: float = 1.0e-4
    freeze_part_epochs: int = 2

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.manifest_path = str(self.manifest_path)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.baseline_checkpoint = str(self.baseline_checkpoint)
        if not self.baseline_checkpoint:
            raise ValueError("baseline_checkpoint is required for architecture-view training")
        for split_field, expected in (
            ("train_split", "model_train"),
            ("val_split", "model_val"),
            ("stack_val_split", "stack_val"),
            ("final_test_split", "final_test"),
        ):
            value = str(getattr(self, split_field))
            if value != expected:
                raise ValueError(f"{split_field} must be {expected!r}")
            setattr(self, split_field, value)
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge model_train/model_val selection")
        if not bool(self.confirm_final_test):
            raise ValueError("Architecture-view runner requires --confirm-final-test for guarded final evaluation")
        if str(self.selection_metric) != ARCHITECTURE_VIEW_PRIMARY_METRIC:
            raise ValueError("Architecture-view checkpoint selection must use fpr_at_signal_eff_0p50")
        self.variant = normalize_architecture_view_variant(self.variant)
        if self.variant not in ARCHITECTURE_VIEW_VARIANTS:
            raise ValueError(f"unknown architecture-view variant {self.variant!r}")
        for field_name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "view_dim",
            "hidden_dim",
            "pn_k",
            "pn_layers",
            "pfn_hidden_dim",
            "pcnn_channels",
            "pcnn_layers",
            "fusion_hidden_dim",
            "part_embed_dim",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        for field_name in ("adapter_lr", "part_lr"):
            value = float(getattr(self, field_name))
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        for field_name in ("weight_decay", "grad_clip_norm", "delta_l2_weight"):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            setattr(self, field_name, value)
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        if int(self.freeze_part_epochs) < 0:
            raise ValueError("freeze_part_epochs cannot be negative")
        self.freeze_part_epochs = int(self.freeze_part_epochs)
        for field_name in ("dropout", "attention_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            setattr(self, field_name, value)
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_stack_val_batches = _optional_nonnegative_int(
            self.max_stack_val_batches,
            field_name="max_stack_val_batches",
        )
        self.max_final_test_batches = _optional_nonnegative_int(
            self.max_final_test_batches,
            field_name="max_final_test_batches",
        )
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_stack_val_jets = _optional_positive_int(self.max_stack_val_jets, field_name="max_stack_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        self.label_names = tuple(str(name) for name in self.label_names)
        self.label_filter = tuple(int(value) for value in self.label_filter)
        if self.label_names != ARCHITECTURE_VIEW_LABEL_NAMES:
            raise ValueError(f"architecture-view uses active binary labels {ARCHITECTURE_VIEW_LABEL_NAMES}")
        if self.label_filter != ARCHITECTURE_VIEW_BINARY_LABEL_FILTER:
            raise ValueError(f"architecture-view expects binary-cache labels {ARCHITECTURE_VIEW_BINARY_LABEL_FILTER}")
        self.random_control_seed = int(self.random_control_seed)
        self.gate_bias_init = float(self.gate_bias_init)
        self.expected_hlt_degradation_strength = float(self.expected_hlt_degradation_strength)

    @property
    def resolved_num_classes(self) -> int:
        return 2

    def model_config(self) -> ArchitectureViewConfig:
        return ArchitectureViewConfig(
            view_dim=int(self.view_dim),
            hidden_dim=int(self.hidden_dim),
            pn_k=int(self.pn_k),
            pn_layers=int(self.pn_layers),
            pfn_hidden_dim=int(self.pfn_hidden_dim),
            pcnn_channels=int(self.pcnn_channels),
            pcnn_layers=int(self.pcnn_layers),
            fusion_hidden_dim=int(self.fusion_hidden_dim),
            part_embed_dim=int(self.part_embed_dim),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            gate_bias_init=float(self.gate_bias_init),
            random_control_seed=int(self.random_control_seed),
        )


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _load_dataset(config: ArchitectureViewTaggerTrainConfig, split: str, *, max_jets: int | None) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    audit = _verify_hlt_params(
        view.metadata,
        split=split,
        expected_strength=float(config.expected_hlt_degradation_strength),
        required=bool(config.verify_hlt_params),
    )
    dataset = SubtokenHLTJetDataset(
        view,
        label_filter=tuple(config.label_filter),
        label_names=tuple(config.label_names),
        max_jets=max_jets,
    )
    dataset.metadata["hlt_protocol_audit"] = audit
    return dataset


def _flatten_scalar_diagnostics(diagnostics: Mapping[str, Any], *, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in diagnostics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            output.update(_flatten_scalar_diagnostics(value, prefix=f"{name}."))
            continue
        if hasattr(value, "detach"):
            try:
                if int(value.numel()) == 1:
                    output[name] = float(value.detach().cpu().item())
            except Exception:
                continue
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                output[name] = numeric
    return output


def _delta_l2_mean_from_output(output: Any) -> Any:
    torch = require_torch()
    delta = output.view_output.delta_h
    mask = output.view_output.mask.bool()
    if not bool(mask.any()):
        return delta.new_zeros(())
    return delta.square().sum(dim=-1)[mask].mean()


def _write_epoch_metrics_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in curves:
        train = row.get("train", {}) if isinstance(row.get("train"), Mapping) else {}
        val = row.get("model_val", {}) if isinstance(row.get("model_val"), Mapping) else {}
        val_binary = val.get("binary_metrics") if isinstance(val.get("binary_metrics"), Mapping) else {}
        payload = {
            "epoch": row.get("epoch"),
            "train_loss": train.get("loss"),
            "train_ce_loss": train.get("ce_loss"),
            "train_delta_l2_loss": train.get("delta_l2_loss"),
            "train_delta_l2_mean": train.get("delta_l2_mean"),
            "train_accuracy": train.get("accuracy"),
            "model_val_loss": val.get("loss"),
            "model_val_ce_loss": val.get("ce_loss"),
            "model_val_delta_l2_loss": val.get("delta_l2_loss"),
            "model_val_delta_l2_mean": val.get("delta_l2_mean"),
            "model_val_accuracy": val.get("accuracy"),
            "model_val_auc": val_binary.get("auc"),
            "model_val_fpr_at_signal_eff_0p50": val_binary.get("fpr_at_signal_eff_0p50"),
        }
        for prefix, metrics in (("train", train), ("model_val", val)):
            diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), Mapping) else {}
            for key, value in sorted(diagnostics.items()):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    safe_key = str(key).replace(".", "_").replace("/", "_")
                    payload[f"{prefix}_diag_{safe_key}"] = numeric
        rows.append(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames) if fieldnames else ("epoch",), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _set_part_model_trainable(model: ArchitectureViewResidualParT, trainable: bool) -> None:
    for parameter in model.part_model.parameters():
        parameter.requires_grad_(bool(trainable))
    model.part_model.train(bool(trainable))


def _optimizer_for_model(model: ArchitectureViewResidualParT, config: ArchitectureViewTaggerTrainConfig):
    torch = require_torch()
    adapter_params: list[Any] = []
    for module in (model.view_module, model.context_control, model.context_control_gate):
        adapter_params.extend([param for param in module.parameters() if param.requires_grad])
    part_params = [param for param in model.part_model.parameters() if param.requires_grad]
    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(config.adapter_lr)})
    if part_params:
        groups.append({"params": part_params, "lr": float(config.part_lr)})
    if not groups:
        return None
    return torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))


def run_architecture_view_tagger_epoch(
    model: ArchitectureViewResidualParT,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float = 1.0,
    max_batches: int | None = None,
    delta_l2_weight: float = 0.0,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    label_names: Sequence[str] = ARCHITECTURE_VIEW_LABEL_NAMES,
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_delta_l2_loss = 0.0
    total_delta_l2_mean = 0.0
    total_correct = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    diagnostic_totals: dict[str, float] = {}
    diagnostic_weight_sum = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            labels = batch["labels"]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
                need_output = bool(training or collect_diagnostics or float(delta_l2_weight) > 0.0)
                output = model(
                    batch["tokens"],
                    batch["mask"],
                    return_outputs=need_output,
                    max_constits=int(batch["tokens"].shape[1]),
                )
                logits = output.logits if need_output else output
                ce_loss = criterion(logits, labels)
                delta_l2_mean = _delta_l2_mean_from_output(output) if need_output else logits.new_zeros(())
                delta_l2_loss = delta_l2_mean * float(delta_l2_weight)
                loss = ce_loss + delta_l2_loss
            if training:
                if scaler is not None and bool(scaler.is_enabled()):
                    scaler.scale(loss).backward()
                    if float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()
            preds = logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_ce_loss += float(ce_loss.detach().item()) * batch_size
            total_delta_l2_loss += float(delta_l2_loss.detach().item()) * batch_size
            total_delta_l2_mean += float(delta_l2_mean.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float32))
            if collect_diagnostics:
                full_output = output if need_output else model(
                    batch["tokens"],
                    batch["mask"],
                    return_outputs=True,
                    max_constits=int(batch["tokens"].shape[1]),
                )
                flat = _flatten_scalar_diagnostics(full_output.diagnostics())
                for key, value in flat.items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * batch_size
                diagnostic_weight_sum += float(batch_size)
    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "ce_loss": total_ce_loss / float(total_seen),
        "delta_l2_loss": total_delta_l2_loss / float(total_seen),
        "delta_l2_mean": total_delta_l2_mean / float(total_seen),
        "delta_l2_weight": float(delta_l2_weight),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else None
        if logits_np is not None:
            metrics["_prediction_arrays"] = {
                "preds": preds_np,
                "labels": labels_np,
                "logits": logits_np,
            }
        metrics.update(
            classification_metrics_from_predictions(
                preds=preds_np,
                labels=labels_np,
                loss_sum=total_loss,
                logits=logits_np,
                label_names=tuple(label_names),
            )
        )
    if diagnostic_weight_sum > 0.0:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_sum
            for key, value in sorted(diagnostic_totals.items())
        }
    return metrics


def architecture_view_tagger_checkpoint_payload(
    model: ArchitectureViewResidualParT,
    optimizer,
    *,
    epoch: int,
    config: ArchitectureViewTaggerTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    baseline_load_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "variant": str(config.variant),
        "variant_behavior": model.variant_behavior(),
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "metrics": dict(metrics),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": dict(source),
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": dict(baseline_load_report),
        "experiment_step": ARCHITECTURE_VIEW_TRAIN_STEP,
        "model_step": ARCHITECTURE_VIEW_MODEL_STEP,
        "output_contract": model.output_contract,
    }


def load_architecture_view_tagger_checkpoint(path: str | Path, *, device=None) -> tuple[ArchitectureViewResidualParT, dict[str, Any]]:
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain train config")
    config = ArchitectureViewTaggerTrainConfig(**dict(config_payload))
    model = build_architecture_view_residual_part(config.model_config(), variant=config.variant)
    model.load_state_dict(payload["model_state_dict"])
    model_config = payload.get("model_config") if isinstance(payload.get("model_config"), Mapping) else {}
    baseline_report = model_config.get("baseline_checkpoint") if isinstance(model_config, Mapping) else None
    if not baseline_report and isinstance(payload.get("baseline_load_report"), Mapping):
        baseline_report = payload["baseline_load_report"]
    if isinstance(baseline_report, Mapping):
        model.baseline_checkpoint_report = dict(baseline_report)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def _evaluate_dataset(
    model: ArchitectureViewResidualParT,
    dataset: SubtokenHLTJetDataset,
    config: ArchitectureViewTaggerTrainConfig,
    *,
    device,
    criterion,
    seed_offset: int,
    max_batches: int | None,
) -> dict[str, Any]:
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + int(seed_offset),
    )
    return run_architecture_view_tagger_epoch(
        model,
        loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=max_batches,
        delta_l2_weight=float(config.delta_l2_weight),
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
    )


def train_architecture_view_tagger(
    config: ArchitectureViewTaggerTrainConfig,
    *,
    model: ArchitectureViewResidualParT | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train one architecture-view residual ParT variant."""

    run_start_time = time.perf_counter()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    manifest_info = _manifest_metadata(config.manifest_path)
    manifest_sha = manifest_info.get("manifest_hash") if manifest_info.get("load_ok") else None
    if bool(config.require_baseline_split_manifest_hash):
        if not bool(manifest_info.get("load_ok")):
            raise ValueError(
                "require_baseline_split_manifest_hash=True requires a readable split manifest; "
                f"failed to load {config.manifest_path!r}: {manifest_info.get('load_error', 'unknown error')}"
            )
        if not manifest_sha:
            raise ValueError("require_baseline_split_manifest_hash=True requires a non-empty manifest_hash")

    train_dataset = train_dataset or _load_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"architecture-view tagger expects raw token dim {RAW_TOKEN_DIM}")

    train_loader = make_subtoken_hlt_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_subtoken_hlt_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )

    checkpoint_model = model or build_architecture_view_residual_part(
        config.model_config(),
        variant=str(config.variant),
    )
    checkpoint_model = checkpoint_model.to(device)
    baseline_report = warm_start_architecture_view_part_model(
        checkpoint_model,
        config.baseline_checkpoint,
        map_location=device,
        expected_split_manifest_hash=manifest_sha if bool(config.require_baseline_split_manifest_hash) else None,
        expected_label_names=tuple(config.label_names),
        expected_label_filter=tuple(config.label_filter),
        expected_num_classes=int(config.resolved_num_classes),
        expected_hlt_degradation_strength=float(config.expected_hlt_degradation_strength),
        require_metadata=True,
        require_all_part_keys=True,
    ).to_dict()
    save_json(diagnostics_dir / "baseline_load_report.json", baseline_report)
    init_logit_sample_count = min(4, int(len(train_dataset)))
    init_logit_diff: dict[str, Any] = {}
    if init_logit_sample_count > 0:
        sample_tokens = torch.as_tensor(train_dataset.tokens[:init_logit_sample_count], device=device).float()
        sample_mask = torch.as_tensor(train_dataset.mask[:init_logit_sample_count], device=device).bool()
        init_logit_diff = compute_architecture_view_init_logit_diff_vs_baseline(
            checkpoint_model,
            sample_tokens,
            sample_mask,
            max_constits=int(sample_tokens.shape[1]),
            attach=True,
        )
        save_json(diagnostics_dir / "init_logit_diff_vs_baseline.json", init_logit_diff)

    is_baseline_recheck = config.variant == ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
    if is_baseline_recheck or int(config.freeze_part_epochs) > 0:
        _set_part_model_trainable(checkpoint_model, False)
    optimizer = None if is_baseline_recheck else _optimizer_for_model(checkpoint_model, config)
    train_model = checkpoint_model
    if config.compile_model and not is_baseline_recheck and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": ARCHITECTURE_VIEW_TRAIN_STEP,
        "model_step": ARCHITECTURE_VIEW_MODEL_STEP,
        "output_contract": checkpoint_model.output_contract,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "source": source,
        "manifest": manifest_info,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.resolved_num_classes),
        "selection_metric": str(config.selection_metric),
        "variant": str(config.variant),
        "variant_behavior": checkpoint_model.variant_behavior(),
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_report,
        "init_logit_diff_vs_baseline": init_logit_diff,
        "leakage_rule": (
            "Step 3 consumes cached fixed-HLT tokens only. Training uses model_train, "
            "checkpoint selection uses model_val, and final_test is loaded only after model_val selection."
        ),
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    if is_baseline_recheck:
        checkpoint_model.eval()
        best_val_metrics = _evaluate_dataset(
            checkpoint_model,
            val_dataset,
            config,
            device=device,
            criterion=criterion,
            seed_offset=1,
            max_batches=config.max_val_batches,
        )
        val_score, selection_value = _selection_score(best_val_metrics, str(config.selection_metric))
        best_val_score = float(val_score)
        curves = [
            {
                "epoch": 0,
                "train": {"loss": None, "accuracy": None, "n_jets": 0, "mode": "evaluation_only_baseline_recheck"},
                "model_val": _metrics_without_prediction_arrays(best_val_metrics),
            }
        ]
        payload = architecture_view_tagger_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=0,
            config=config,
            metrics=curves[0],
            source=source,
            baseline_load_report=baseline_report,
        )
        torch.save(payload, output_dir / "best_model_val.pt")
        torch.save(payload, output_dir / "last.pt")
        best_epoch = 0
        best_val_accuracy = _finite_float(best_val_metrics.get("accuracy"), default=-1.0)
        best_val_loss = _finite_float(best_val_metrics.get("loss"), default=float("inf"))
    else:
        curves: list[dict[str, Any]] = []
        best_val_score = float("-inf")
        selection_value = float("nan")
        best_val_accuracy = -1.0
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        for epoch in range(1, int(config.epochs) + 1):
            if epoch == int(config.freeze_part_epochs) + 1 and int(config.freeze_part_epochs) > 0:
                _set_part_model_trainable(checkpoint_model, True)
                optimizer = _optimizer_for_model(checkpoint_model, config)
            train_metrics = run_architecture_view_tagger_epoch(
                train_model,
                train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                amp=bool(config.amp),
                grad_clip_norm=float(config.grad_clip_norm),
                max_batches=config.max_train_batches,
                delta_l2_weight=float(config.delta_l2_weight),
                collect_predictions=False,
                collect_diagnostics=True,
                label_names=tuple(config.label_names),
            )
            val_metrics = run_architecture_view_tagger_epoch(
                checkpoint_model,
                val_loader,
                device=device,
                criterion=criterion,
                amp=False,
                max_batches=config.max_val_batches,
                delta_l2_weight=float(config.delta_l2_weight),
                collect_predictions=True,
                collect_diagnostics=True,
                label_names=tuple(config.label_names),
            )
            val_score, current_selection_value = _selection_score(val_metrics, str(config.selection_metric))
            row = {
                "epoch": int(epoch),
                "train": _metrics_without_prediction_arrays(train_metrics),
                "model_val": _metrics_without_prediction_arrays(val_metrics),
                "selection_metric": str(config.selection_metric),
                "selection_metric_value": float(current_selection_value),
                "selection_score": float(val_score),
                "part_model_trainable": any(param.requires_grad for param in checkpoint_model.part_model.parameters()),
            }
            curves.append(row)
            improved = val_score > best_val_score
            if improved:
                best_val_score = float(val_score)
                selection_value = float(current_selection_value)
                best_val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
                best_val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
                best_epoch = int(epoch)
                epochs_without_improvement = 0
                payload = architecture_view_tagger_checkpoint_payload(
                    checkpoint_model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    source=source,
                    baseline_load_report=baseline_report,
                )
                torch.save(payload, output_dir / "best_model_val.pt")
                save_json(output_dir / "model_val_report.json", _metrics_without_prediction_arrays(val_metrics))
            else:
                epochs_without_improvement += 1
            payload = architecture_view_tagger_checkpoint_payload(
                checkpoint_model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                source=source,
                baseline_load_report=baseline_report,
            )
            torch.save(payload, output_dir / "last.pt")
            save_json(output_dir / "training_curves.json", {"epochs": curves})
            _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)
            if int(config.early_stop_patience) >= 0 and epochs_without_improvement > int(config.early_stop_patience):
                break
        if best_epoch < 0:
            raise RuntimeError("training completed without selecting a best checkpoint")
        best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
        checkpoint_model.load_state_dict(best_payload["model_state_dict"])
        best_val_metrics = _evaluate_dataset(
            checkpoint_model,
            val_dataset,
            config,
            device=device,
            criterion=criterion,
            seed_offset=1,
            max_batches=config.max_val_batches,
        )
        val_score, selection_value = _selection_score(best_val_metrics, str(config.selection_metric))
        best_val_score = float(val_score)
        best_val_accuracy = _finite_float(best_val_metrics.get("accuracy"), default=-1.0)
        best_val_loss = _finite_float(best_val_metrics.get("loss"), default=float("inf"))
        save_json(output_dir / "model_val_report.json", _metrics_without_prediction_arrays(best_val_metrics))

    save_json(output_dir / "training_curves.json", {"epochs": curves})
    _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

    stack_val_dataset = stack_val_dataset or _load_dataset(config, config.stack_val_split, max_jets=config.max_stack_val_jets)
    stack_val_metrics = _evaluate_dataset(
        checkpoint_model,
        stack_val_dataset,
        config,
        device=device,
        criterion=criterion,
        seed_offset=2,
        max_batches=config.max_stack_val_batches,
    )
    final_test_metadata = None
    final_test_metrics = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_test_metrics = _evaluate_dataset(
            checkpoint_model,
            final_test_dataset,
            config,
            device=device,
            criterion=criterion,
            seed_offset=3,
            max_batches=config.max_final_test_batches,
        )
        final_test_metadata = dict(final_test_dataset.metadata)
    elapsed_seconds = float(time.perf_counter() - run_start_time)
    epochs_completed = len(curves) if not is_baseline_recheck else 0
    report = {
        "experiment_step": ARCHITECTURE_VIEW_TRAIN_STEP,
        "train_contract": ARCHITECTURE_VIEW_TRAIN_CONTRACT,
        "model_step": ARCHITECTURE_VIEW_MODEL_STEP,
        "output_contract": checkpoint_model.output_contract,
        "variant": str(config.variant),
        "variant_behavior": checkpoint_model.variant_behavior(),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": ARCHITECTURE_VIEW_PRIMARY_METRIC_DIRECTION,
        "best_model_selection_metric_value": float(selection_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": _metrics_without_prediction_arrays(best_val_metrics, keep_prediction_arrays=True),
        "stack_val_metrics": _metrics_without_prediction_arrays(stack_val_metrics),
        "final_test_metrics": None
        if final_test_metrics is None
        else _metrics_without_prediction_arrays(final_test_metrics, keep_prediction_arrays=True),
        "epochs_completed": int(epochs_completed),
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": source,
        "manifest": manifest_info,
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_report,
        "baseline_checkpoint_path": baseline_report.get("baseline_checkpoint_path"),
        "baseline_checkpoint_hash": baseline_report.get("baseline_checkpoint_hash"),
        "baseline_checkpoint_selection_metric": baseline_report.get("baseline_checkpoint_selection_metric"),
        "baseline_checkpoint_hlt_degradation_strength": baseline_report.get(
            "baseline_checkpoint_hlt_degradation_strength"
        ),
        "baseline_checkpoint_split_manifest_hash": baseline_report.get("baseline_checkpoint_split_manifest_hash"),
        "baseline_checkpoint_label_names": baseline_report.get("baseline_checkpoint_label_names"),
        "baseline_checkpoint_label_filter": baseline_report.get("baseline_checkpoint_label_filter"),
        "baseline_checkpoint_num_classes": baseline_report.get("baseline_checkpoint_num_classes"),
        "part_config": baseline_report.get("part_config"),
        "init_logit_diff_vs_baseline": init_logit_diff,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "stack_val_dataset": dict(stack_val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": int(epochs_completed),
            "seconds_per_completed_epoch": None
            if epochs_completed <= 0
            else elapsed_seconds / float(epochs_completed),
        },
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "model_val_report.json", _metrics_without_prediction_arrays(best_val_metrics))
    save_json(output_dir / "stack_val_report.json", _metrics_without_prediction_arrays(stack_val_metrics))
    if final_test_metrics is not None:
        save_json(output_dir / "final_test_report.json", _metrics_without_prediction_arrays(final_test_metrics))
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "ARCHITECTURE_VIEW_LOWER_IS_BETTER_SELECTION_METRICS",
    "ARCHITECTURE_VIEW_SELECTION_METRICS",
    "ARCHITECTURE_VIEW_TRAIN_CONTRACT",
    "ARCHITECTURE_VIEW_TRAIN_STEP",
    "ArchitectureViewTaggerTrainConfig",
    "architecture_view_label_filter_names_to_indices",
    "architecture_view_tagger_checkpoint_payload",
    "load_architecture_view_tagger_checkpoint",
    "run_architecture_view_tagger_epoch",
    "train_architecture_view_tagger",
]
