"""Training entry points for local-compression feature-adapter ParT.

Step 12 adds the first real runner for ``LocalCompressionFeatureAdapterParT``.
It trains on cached fixed-HLT tokens, selects checkpoints on ``model_val`` with
FPR@50, and only evaluates final_test after the best model-val checkpoint has
been chosen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength, load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM, load_split_manifest, manifest_hash

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.label_filters import label_names_to_manifest_indices
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .checkpoint import (
    compute_init_logit_diff_vs_baseline,
    sha256_file,
    warm_start_local_compression_part_model,
)
from .config import (
    LOCAL_COMPRESSION_BINARY_LABEL_FILTER,
    LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID,
    LOCAL_COMPRESSION_GATE_NONE,
    LOCAL_COMPRESSION_PART_CONTRACT,
    LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_COMPRESSION_PRIMARY_METRIC,
    LOCAL_COMPRESSION_PRIMARY_METRIC_DIRECTION,
    LOCAL_COMPRESSION_SIGNAL_LABEL,
    LOCAL_COMPRESSION_BACKGROUND_LABEL,
    LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
    LOCAL_COMPRESSION_VARIANTS,
    LocalCompressionPartConfig,
    normalize_local_compression_gate_mode,
    normalize_local_compression_pool_mode,
    normalize_local_compression_split_name,
    normalize_local_compression_variant,
)
from .model import (
    LOCAL_COMPRESSION_MODEL_CONTRACT,
    LOCAL_COMPRESSION_MODEL_STEP,
    LocalCompressionFeatureAdapterParT,
    build_local_compression_feature_adapter_part,
)


LOCAL_COMPRESSION_TRAIN_STEP = "local_compression_part_step12_train"
LOCAL_COMPRESSION_TRAIN_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_train_v1"
LOCAL_COMPRESSION_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "macro_per_class_accuracy",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
LOCAL_COMPRESSION_LOWER_IS_BETTER_SELECTION_METRICS = {"loss", "fpr_at_signal_eff_0p30", "fpr_at_signal_eff_0p50"}
LOCAL_COMPRESSION_HLT_PARAM_TOLERANCE = 1.0e-9


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


def _label_names_from_filter(label_filter: Sequence[int]) -> tuple[str, ...]:
    return tuple(str(LABEL_NAMES[int(index)]) for index in label_filter)


def _default_qcd_hgg_filter() -> tuple[int, int]:
    return tuple(int(value) for value in LOCAL_COMPRESSION_BINARY_LABEL_FILTER)


def _resolve_label_filter(label_filter: Sequence[int] | None) -> tuple[int, ...]:
    if label_filter:
        output = tuple(int(value) for value in label_filter)
    else:
        output = _default_qcd_hgg_filter()
    if len(output) != 2:
        raise ValueError("local-compression Step 12 is binary QCD/Hgg and requires exactly two labels")
    if min(output) < 0 or max(output) >= len(LABEL_NAMES):
        raise ValueError(f"label_filter contains values outside JetClass label range: {output}")
    return output


def _resolve_label_names(label_names: Sequence[str] | None, label_filter: Sequence[int]) -> tuple[str, ...]:
    if label_names:
        output = tuple(str(name) for name in label_names)
    else:
        if tuple(label_filter) == tuple(LOCAL_COMPRESSION_BINARY_LABEL_FILTER):
            output = tuple(LOCAL_COMPRESSION_SOURCE_LABEL_NAMES)
        else:
            output = _label_names_from_filter(label_filter)
    if len(output) != len(tuple(label_filter)):
        raise ValueError("label_names length must match label_filter length")
    return output


def local_compression_label_filter_names_to_indices(
    values: Sequence[str],
    *,
    manifest_path: str | Path | None,
    label_names: Sequence[str] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
) -> tuple[int, ...]:
    """Resolve CLI label-filter names in the active binary-cache label space."""

    if not values:
        return tuple(LOCAL_COMPRESSION_BINARY_LABEL_FILTER)
    try:
        return label_names_to_manifest_indices(values, manifest_path=manifest_path)
    except Exception:
        by_active_name = {str(name): index for index, name in enumerate(tuple(label_names))}
        output: list[int] = []
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            if text.isdigit():
                output.append(int(text))
            elif text in by_active_name:
                output.append(int(by_active_name[text]))
            else:
                raise
        return tuple(output)


def _manifest_metadata(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {resolved}")
    payload: dict[str, Any] = {
        "path": str(resolved),
        "file_sha256": sha256_file(resolved),
    }
    try:
        manifest = load_split_manifest(resolved)
    except Exception as exc:
        payload.update(
            {
                "load_ok": False,
                "load_error": f"{type(exc).__name__}: {exc}",
                "manifest_hash": None,
                "class_names": None,
                "split_counts": None,
            }
        )
    else:
        payload.update(
            {
                "load_ok": True,
                "manifest_hash": manifest_hash(manifest),
                "class_names": list(manifest.class_names),
                "split_counts": {split: len(rows) for split, rows in manifest.splits.items()},
            }
        )
    return payload


def _lookup_selection_metric(metrics: Mapping[str, Any], metric_name: str) -> float:
    if metric_name in metrics:
        try:
            return float(metrics[metric_name])
        except (TypeError, ValueError):
            return float("nan")
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping) and metric_name in binary:
        try:
            return float(binary[metric_name])
        except (TypeError, ValueError):
            return float("nan")
    raise KeyError(f"validation metrics do not contain selection metric {metric_name!r}")


def _selection_metric_requires_predictions(metric_name: str) -> bool:
    return metric_name not in {"accuracy", "loss"}


def _selection_score(metrics: Mapping[str, Any], metric_name: str) -> tuple[float, float]:
    value = _lookup_selection_metric(metrics, metric_name)
    if np.isnan(value):
        return float("-inf"), value
    if metric_name in LOCAL_COMPRESSION_LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


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
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if np.isfinite(numeric):
                output[name] = numeric
    return output


def _expected_hlt_params_for_strength(strength: float) -> dict[str, float]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(float(strength)))


def _verify_hlt_params(metadata: Mapping[str, Any], *, split: str, expected_strength: float, required: bool) -> dict[str, Any]:
    expected = _expected_hlt_params_for_strength(float(expected_strength))
    actual = metadata.get("hlt_params") if isinstance(metadata.get("hlt_params"), Mapping) else {}
    problems: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            problems.append(f"missing hlt param {key}")
            continue
        try:
            actual_value = float(actual[key])
        except (TypeError, ValueError):
            problems.append(f"hlt param {key} is not numeric: {actual[key]!r}")
            continue
        if abs(actual_value - float(expected_value)) > LOCAL_COMPRESSION_HLT_PARAM_TOLERANCE:
            problems.append(f"hlt param {key}={actual_value} expected {float(expected_value)}")
    report = {
        "split": str(split),
        "required": bool(required),
        "ok": len(problems) == 0,
        "expected_hlt_degradation_strength": float(expected_strength),
        "expected_hlt_params": expected,
        "actual_hlt_params": dict(actual),
        "problems": problems,
    }
    if bool(required) and problems:
        raise ValueError(f"HLT cache for split {split!r} is not HLT0.6: {'; '.join(problems)}")
    return report


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _metrics_without_prediction_arrays(metrics: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(metrics)
    clean.pop("_prediction_arrays", None)
    return clean


def _write_epoch_metrics_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in curves:
        train = row.get("train", {}) if isinstance(row.get("train"), Mapping) else {}
        val = row.get("model_val", {}) if isinstance(row.get("model_val"), Mapping) else {}
        val_binary = val.get("binary_metrics") if isinstance(val.get("binary_metrics"), Mapping) else {}
        rows.append(
            {
                "epoch": row.get("epoch"),
                "train_loss": train.get("loss"),
                "train_accuracy": train.get("accuracy"),
                "model_val_loss": val.get("loss"),
                "model_val_accuracy": val.get("accuracy"),
                "model_val_auc": val_binary.get("auc"),
                "model_val_fpr_at_signal_eff_0p50": val_binary.get("fpr_at_signal_eff_0p50"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()) if rows else ("epoch",))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class LocalCompressionTaggerTrainConfig:
    """Training config for one Step 12 local-compression tagger run."""

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
    seed: int = 3207
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    adapter_lr: float = 3.0e-4
    part_lr: float = 3.0e-5
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
    selection_metric: str = LOCAL_COMPRESSION_PRIMARY_METRIC
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH
    label_names: tuple[str, ...] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = ()
    variant: str = "lc_context_gated"
    embed_dim: int = 96
    local_layers: int = 2
    local_heads: int = 4
    context_layers: int = 1
    context_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    pool_mode: str = "learned_query"
    gate_mode: str | None = None
    delta_scale: float = 1.0
    freeze_pid_deltas: bool = False
    freeze_geometry_deltas: bool = False
    freeze_part_epochs: int = 0

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.manifest_path = str(self.manifest_path)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.baseline_checkpoint = str(self.baseline_checkpoint)
        if not self.baseline_checkpoint:
            raise ValueError("baseline_checkpoint is required for Step 12")
        self.train_split = normalize_local_compression_split_name(self.train_split)
        self.val_split = normalize_local_compression_split_name(self.val_split)
        self.stack_val_split = normalize_local_compression_split_name(self.stack_val_split)
        self.final_test_split = normalize_local_compression_split_name(self.final_test_split)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 12 must train on model_train and select on model_val")
        if self.final_test_split != "final_test":
            raise ValueError("Step 12 final_test_split must be final_test")
        if not bool(self.confirm_final_test):
            raise ValueError("Step 12 requires --confirm-final-test for guarded final evaluation")
        if self.selection_metric != LOCAL_COMPRESSION_PRIMARY_METRIC:
            raise ValueError("Step 12 checkpoint selection must use fpr_at_signal_eff_0p50")
        self.variant = normalize_local_compression_variant(self.variant)
        if self.variant not in LOCAL_COMPRESSION_VARIANTS:
            raise ValueError(f"unknown local-compression variant {self.variant!r}")
        self.pool_mode = normalize_local_compression_pool_mode(self.pool_mode)
        gate_mode = self.gate_mode
        if gate_mode is None:
            gate_mode = LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID if self.variant == "lc_context_gated" else LOCAL_COMPRESSION_GATE_NONE
        self.gate_mode = normalize_local_compression_gate_mode(gate_mode)
        self.label_filter = _resolve_label_filter(self.label_filter)
        self.label_names = _resolve_label_names(self.label_names, self.label_filter)
        if tuple(self.label_names) != tuple(LOCAL_COMPRESSION_SOURCE_LABEL_NAMES):
            raise ValueError("Step 12 local-compression training is frozen to label names QCD/Hgg")
        if tuple(self.label_filter) != tuple(LOCAL_COMPRESSION_BINARY_LABEL_FILTER):
            raise ValueError("Step 12 expects binary-cache labels QCD=0, Hgg=1")
        for name in ("seed", "batch_size", "eval_batch_size", "epochs", "num_workers", "early_stop_patience"):
            value = int(getattr(self, name))
            if value < 0 or (name != "num_workers" and value == 0):
                raise ValueError(f"{name} must be positive" if name != "num_workers" else "num_workers must be non-negative")
            setattr(self, name, value)
        self.freeze_part_epochs = _optional_nonnegative_int(self.freeze_part_epochs, field_name="freeze_part_epochs") or 0
        for name in ("max_train_batches", "max_val_batches", "max_stack_val_batches", "max_final_test_batches"):
            setattr(self, name, _optional_positive_int(getattr(self, name), field_name=name))
        for name in ("max_train_jets", "max_val_jets", "max_stack_val_jets", "max_final_test_jets"):
            setattr(self, name, _optional_positive_int(getattr(self, name), field_name=name))
        if float(self.adapter_lr) <= 0.0 or float(self.part_lr) <= 0.0:
            raise ValueError("adapter_lr and part_lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.grad_clip_norm) <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if float(self.delta_scale) < 0.0:
            raise ValueError("delta_scale must be non-negative")

    @property
    def resolved_num_classes(self) -> int:
        return len(tuple(self.label_names))

    def model_config(self) -> LocalCompressionPartConfig:
        return LocalCompressionPartConfig(
            num_classes=int(self.resolved_num_classes),
            variant=self.variant,
            embed_dim=int(self.embed_dim),
            local_layers=int(self.local_layers),
            local_heads=int(self.local_heads),
            context_layers=int(self.context_layers),
            context_heads=int(self.context_heads),
            mlp_ratio=float(self.mlp_ratio),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            pool_mode=str(self.pool_mode),
            gate_mode=str(self.gate_mode),
            delta_scale=float(self.delta_scale),
            freeze_pid_deltas=bool(self.freeze_pid_deltas),
            freeze_geometry_deltas=bool(self.freeze_geometry_deltas),
        )


def _load_dataset(config: LocalCompressionTaggerTrainConfig, split: str, *, max_jets: int | None) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    audit = _verify_hlt_params(
        view.metadata,
        split=split,
        expected_strength=float(config.expected_hlt_degradation_strength),
        required=bool(config.verify_hlt_params),
    )
    dataset = SubtokenHLTJetDataset(
        view,
        label_filter=config.label_filter,
        label_names=config.label_names,
        max_jets=max_jets,
    )
    dataset.metadata["hlt_protocol_audit"] = audit
    return dataset


def load_baseline_checkpoint_into_part_model(
    model: LocalCompressionFeatureAdapterParT,
    checkpoint_path: str | Path,
    *,
    device=None,
    require: bool = True,
    expected_split_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Load a frozen HLT ParT checkpoint into ``model.part_model`` only."""

    report = warm_start_local_compression_part_model(
        model,
        checkpoint_path,
        map_location=device or "cpu",
        expected_selection_metric=LOCAL_COMPRESSION_PRIMARY_METRIC,
        expected_hlt_degradation_strength=LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
        expected_split_manifest_hash=expected_split_manifest_hash,
        require_metadata=False,
        require_all_part_keys=bool(require),
    )
    return report.to_dict()


def _set_part_model_trainable(model: LocalCompressionFeatureAdapterParT, trainable: bool) -> dict[str, Any]:
    for parameter in model.part_model.parameters():
        parameter.requires_grad = bool(trainable)
    return {
        "part_model_trainable": bool(trainable),
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }


def _optimizer_for_model(model: LocalCompressionFeatureAdapterParT, config: LocalCompressionTaggerTrainConfig):
    torch = require_torch()
    part_param_ids = {id(param) for param in model.part_model.parameters()}
    adapter_params = [param for param in model.parameters() if id(param) not in part_param_ids and param.requires_grad]
    part_params = [param for param in model.part_model.parameters() if param.requires_grad]
    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(config.adapter_lr), "name": "adapter"})
    if part_params:
        groups.append({"params": part_params, "lr": float(config.part_lr), "name": "part_model"})
    if not groups:
        raise ValueError("no trainable parameters available for optimizer")
    return torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))


def run_local_compression_tagger_epoch(
    model,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    label_names: Sequence[str] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
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
                output = model(
                    batch["tokens"],
                    batch["mask"],
                    return_outputs=bool(collect_diagnostics),
                    max_constits=int(batch["tokens"].shape[1]),
                )
                logits = output.logits if collect_diagnostics else output
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and bool(scaler.is_enabled()):
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()
            preds = logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float32))
            if collect_diagnostics:
                diagnostics = _flatten_scalar_diagnostics(output.diagnostics())
                for key, value in diagnostics.items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * batch_size
                diagnostic_weight_sum += float(batch_size)
    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else None
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


def local_compression_tagger_checkpoint_payload(
    model: LocalCompressionFeatureAdapterParT,
    optimizer,
    *,
    epoch: int,
    config: LocalCompressionTaggerTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    baseline_load_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "metrics": dict(metrics),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": dict(source),
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": dict(baseline_load_report),
        "experiment_step": LOCAL_COMPRESSION_TRAIN_STEP,
        "model_step": LOCAL_COMPRESSION_MODEL_STEP,
        "output_contract": model.output_contract,
    }


def load_local_compression_tagger_checkpoint(path: str | Path, *, device=None) -> tuple[LocalCompressionFeatureAdapterParT, dict[str, Any]]:
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain train config")
    config = LocalCompressionTaggerTrainConfig(**dict(config_payload))
    model = build_local_compression_feature_adapter_part(config.model_config())
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


def train_local_compression_tagger(
    config: LocalCompressionTaggerTrainConfig,
    *,
    model: LocalCompressionFeatureAdapterParT | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train the Step 12 local-compression HLT-only tagger."""

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

    train_dataset = train_dataset or _load_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"local-compression tagger expects raw token dim {RAW_TOKEN_DIM}")
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

    checkpoint_model = model or build_local_compression_feature_adapter_part(config.model_config())
    checkpoint_model = checkpoint_model.to(device)
    baseline_load_report = load_baseline_checkpoint_into_part_model(
        checkpoint_model,
        config.baseline_checkpoint,
        device=device,
        require=True,
        expected_split_manifest_hash=manifest_sha,
    )
    save_json(diagnostics_dir / "baseline_load_report.json", baseline_load_report)
    init_logit_sample_count = min(4, int(len(train_dataset)))
    init_logit_diff = {}
    if init_logit_sample_count > 0:
        sample_tokens = torch.as_tensor(train_dataset.tokens[:init_logit_sample_count], device=device).float()
        sample_mask = torch.as_tensor(train_dataset.mask[:init_logit_sample_count], device=device).bool()
        init_logit_diff = compute_init_logit_diff_vs_baseline(
            checkpoint_model,
            sample_tokens,
            sample_mask,
            max_constits=int(sample_tokens.shape[1]),
            attach=True,
        )
        save_json(diagnostics_dir / "init_logit_diff_vs_baseline.json", init_logit_diff)
    if config.freeze_part_epochs > 0:
        _set_part_model_trainable(checkpoint_model, False)
    optimizer = _optimizer_for_model(checkpoint_model, config)
    train_model = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": LOCAL_COMPRESSION_TRAIN_STEP,
        "model_step": LOCAL_COMPRESSION_MODEL_STEP,
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
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_load_report,
        "init_logit_diff_vs_baseline": init_logit_diff,
        "leakage_rule": (
            "Step 12 consumes cached fixed-HLT tokens only. Training uses model_train, "
            "checkpoint selection uses model_val, and final_test is loaded only after model_val selection."
        ),
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        if int(config.freeze_part_epochs) > 0 and epoch == int(config.freeze_part_epochs) + 1:
            _set_part_model_trainable(checkpoint_model, True)
            optimizer = _optimizer_for_model(checkpoint_model, config)
        train_metrics = run_local_compression_tagger_epoch(
            train_model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
            collect_predictions=False,
            collect_diagnostics=False,
            label_names=tuple(config.label_names),
        )
        val_metrics = run_local_compression_tagger_epoch(
            train_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=_selection_metric_requires_predictions(str(config.selection_metric)),
            collect_diagnostics=False,
            label_names=tuple(config.label_names),
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = local_compression_tagger_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            metrics=row,
            source=source,
            baseline_load_report=baseline_load_report,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_score = float(val_score)
            best_selection_metric_value = float(selection_value)
            best_val_accuracy = float(val_accuracy)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError("local-compression tagger did not produce a valid model_val checkpoint")

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    checkpoint_model.load_state_dict(best_payload["model_state_dict"])
    checkpoint_model.eval()
    best_model = checkpoint_model
    best_val_metrics = run_local_compression_tagger_epoch(
        best_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
    )
    metrics_by_split: dict[str, Mapping[str, Any]] = {"model_val": best_val_metrics}

    stack_val_dataset = stack_val_dataset or _load_dataset(config, config.stack_val_split, max_jets=config.max_stack_val_jets)
    stack_val_loader = make_subtoken_hlt_loader(
        stack_val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    stack_val_metrics = run_local_compression_tagger_epoch(
        best_model,
        stack_val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_stack_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
    )
    metrics_by_split["stack_val"] = stack_val_metrics

    final_test_metadata = None
    final_test_metrics = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_test_loader = make_subtoken_hlt_loader(
            final_test_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 3,
        )
        final_test_metrics = run_local_compression_tagger_epoch(
            best_model,
            final_test_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=tuple(config.label_names),
        )
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)

    elapsed_seconds = float(time.perf_counter() - run_start_time)
    report = {
        "experiment_step": LOCAL_COMPRESSION_TRAIN_STEP,
        "train_contract": LOCAL_COMPRESSION_TRAIN_CONTRACT,
        "model_step": LOCAL_COMPRESSION_MODEL_STEP,
        "output_contract": best_model.output_contract,
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": LOCAL_COMPRESSION_PRIMARY_METRIC_DIRECTION,
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": _metrics_without_prediction_arrays(best_val_metrics),
        "stack_val_metrics": _metrics_without_prediction_arrays(stack_val_metrics),
        "final_test_metrics": None if final_test_metrics is None else _metrics_without_prediction_arrays(final_test_metrics),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "config": asdict(config),
        "model_config": best_model.to_config_dict(),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": source,
        "manifest": manifest_info,
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_load_report,
        "baseline_checkpoint_path": baseline_load_report.get("baseline_checkpoint_path"),
        "baseline_checkpoint_hash": baseline_load_report.get("baseline_checkpoint_hash"),
        "baseline_checkpoint_selection_metric": baseline_load_report.get("baseline_checkpoint_selection_metric"),
        "baseline_checkpoint_hlt_degradation_strength": baseline_load_report.get(
            "baseline_checkpoint_hlt_degradation_strength"
        ),
        "baseline_checkpoint_split_manifest_hash": baseline_load_report.get(
            "baseline_checkpoint_split_manifest_hash"
        ),
        "part_config": baseline_load_report.get("part_config"),
        "adapter_config": best_model.to_config_dict().get("adapter_config"),
        "init_logit_diff_vs_baseline": init_logit_diff,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "stack_val_dataset": dict(stack_val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": len(curves),
            "seconds_per_completed_epoch": elapsed_seconds / float(len(curves)) if curves else None,
        },
        "walltime_seconds": elapsed_seconds,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "LOCAL_COMPRESSION_LOWER_IS_BETTER_SELECTION_METRICS",
    "LOCAL_COMPRESSION_SELECTION_METRICS",
    "LOCAL_COMPRESSION_TRAIN_CONTRACT",
    "LOCAL_COMPRESSION_TRAIN_STEP",
    "LocalCompressionTaggerTrainConfig",
    "local_compression_label_filter_names_to_indices",
    "load_baseline_checkpoint_into_part_model",
    "load_local_compression_tagger_checkpoint",
    "local_compression_tagger_checkpoint_payload",
    "run_local_compression_tagger_epoch",
    "train_local_compression_tagger",
]
