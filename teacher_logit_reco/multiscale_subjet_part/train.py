"""Step 10 training loop for multi-scale subjet HLT ParT comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength, load_cached_hlt_view
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .model import (
    MULTISCALE_SUBJET_CLASSIFIER_VARIANTS,
    MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
    MultiScaleSubjetPartClassifier,
    build_multiscale_subjet_comparison_classifier,
    normalize_multiscale_subjet_variant,
)
from .protocol import (
    MULTISCALE_SUBJET_BINARY_LABEL_FILTER,
    MULTISCALE_SUBJET_CONTRACT,
    MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
    MULTISCALE_SUBJET_PRIMARY_METRIC,
    MULTISCALE_SUBJET_PROTOCOL_STEP,
    MULTISCALE_SUBJET_SOURCE_LABEL_NAMES,
    default_multiscale_subjet_part_protocol,
    multiscale_subjet_part_protocol_manifest,
)


MULTISCALE_SUBJET_TRAIN_STEP = "multiscale_subjet_part_step10_train"
MULTISCALE_SUBJET_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "macro_per_class_accuracy",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
MULTISCALE_SUBJET_LOWER_IS_BETTER_SELECTION_METRICS = {"loss", "fpr_at_signal_eff_0p30", "fpr_at_signal_eff_0p50"}
MULTISCALE_SUBJET_HLT_PARAM_TOLERANCE = 1.0e-9


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if np.isfinite(output) else float(default)


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
    if metric_name in MULTISCALE_SUBJET_LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1.0e-300, None)


def _threshold_at_signal_efficiency(labels: np.ndarray, scores: np.ndarray, target_efficiency: float) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_pos = int(np.sum(positives))
    if n_pos == 0:
        return float("nan")
    positive_scores = np.sort(scores[positives])[::-1]
    threshold_index = min(max(int(np.ceil(float(target_efficiency) * n_pos)) - 1, 0), n_pos - 1)
    return float(positive_scores[threshold_index])


def _metrics_at_fixed_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0 or not np.isfinite(float(threshold)):
        return {
            "signal_efficiency": float("nan"),
            "false_positive_rate": float("nan"),
            "background_rejection": float("nan"),
        }
    signal_efficiency = float(np.mean(scores[positives] >= float(threshold)))
    false_positive_rate = float(np.mean(scores[negatives] >= float(threshold)))
    return {
        "signal_efficiency": signal_efficiency,
        "false_positive_rate": false_positive_rate,
        "background_rejection": float("inf") if false_positive_rate == 0.0 else float(1.0 / false_positive_rate),
    }


def _prediction_arrays(metrics: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    arrays = metrics.get("_prediction_arrays") if isinstance(metrics, Mapping) else None
    return arrays if isinstance(arrays, Mapping) else None


def _validation_threshold_final_test_metrics(
    model_val_metrics: Mapping[str, Any],
    final_test_metrics: Mapping[str, Any],
    *,
    target_signal_efficiency: float = 0.50,
) -> dict[str, float]:
    """Choose threshold on model_val and apply it once to final_test."""

    val_arrays = _prediction_arrays(model_val_metrics)
    final_arrays = _prediction_arrays(final_test_metrics)
    if not isinstance(val_arrays, Mapping) or not isinstance(final_arrays, Mapping):
        return {}
    val_logits = val_arrays.get("logits")
    val_labels = val_arrays.get("labels")
    final_logits = final_arrays.get("logits")
    final_labels = final_arrays.get("labels")
    if val_logits is None or val_labels is None or final_logits is None or final_labels is None:
        return {}
    val_logits = np.asarray(val_logits, dtype=np.float64)
    final_logits = np.asarray(final_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    final_labels = np.asarray(final_labels, dtype=np.int64)
    if val_logits.ndim != 2 or final_logits.ndim != 2 or val_logits.shape[1] != 2 or final_logits.shape[1] != 2:
        return {}
    val_scores = _softmax_numpy(val_logits)[:, 1]
    final_scores = _softmax_numpy(final_logits)[:, 1]
    threshold = _threshold_at_signal_efficiency(val_labels, val_scores, float(target_signal_efficiency))
    val_at_threshold = _metrics_at_fixed_threshold(val_labels, val_scores, threshold)
    final_at_threshold = _metrics_at_fixed_threshold(final_labels, final_scores, threshold)
    return {
        "validation_threshold_target_signal_efficiency": float(target_signal_efficiency),
        "validation_threshold": float(threshold),
        "validation_threshold_model_val_signal_efficiency": val_at_threshold["signal_efficiency"],
        "validation_threshold_model_val_fpr": val_at_threshold["false_positive_rate"],
        "validation_threshold_model_val_background_rejection": val_at_threshold["background_rejection"],
        "validation_threshold_final_test_signal_efficiency": final_at_threshold["signal_efficiency"],
        "validation_threshold_final_test_fpr": final_at_threshold["false_positive_rate"],
        "validation_threshold_final_test_background_rejection": final_at_threshold["background_rejection"],
    }


def _expected_hlt_params_for_strength(strength: float) -> dict[str, float]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(float(strength)))


def compare_hlt_params_for_multiscale_subjet(
    actual: Mapping[str, Any] | None,
    *,
    expected_strength: float = MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
) -> dict[str, Any]:
    """Compare cache HLT params against the frozen HLT0.6 protocol."""

    expected = _expected_hlt_params_for_strength(float(expected_strength))
    problems: list[str] = []
    actual_payload = actual if isinstance(actual, Mapping) else {}
    if not isinstance(actual, Mapping):
        problems.append("metadata does not contain an hlt_params mapping")
    for key, expected_value in expected.items():
        if key not in actual_payload:
            problems.append(f"missing hlt param {key}")
            continue
        try:
            actual_value = float(actual_payload[key])
        except (TypeError, ValueError):
            problems.append(f"hlt param {key} is not numeric: {actual_payload[key]!r}")
            continue
        if abs(actual_value - float(expected_value)) > MULTISCALE_SUBJET_HLT_PARAM_TOLERANCE:
            problems.append(f"hlt param {key}={actual_value} expected {float(expected_value)}")
    return {
        "ok": len(problems) == 0,
        "expected_hlt_degradation_strength": float(expected_strength),
        "expected_hlt_params": expected,
        "actual_hlt_params": dict(actual_payload),
        "problems": problems,
    }


def _verify_hlt_cache_protocol(
    metadata: Mapping[str, Any],
    *,
    split: str,
    expected_strength: float,
    required: bool,
) -> dict[str, Any]:
    audit = compare_hlt_params_for_multiscale_subjet(
        metadata.get("hlt_params") if isinstance(metadata, Mapping) else None,
        expected_strength=float(expected_strength),
    )
    audit["split"] = str(split)
    audit["required"] = bool(required)
    if bool(required) and not bool(audit["ok"]):
        raise ValueError(
            "HLT cache does not match the multi-scale subjet HLT0.6 protocol for "
            f"split {split}: {audit['problems']}"
        )
    return audit


@dataclass
class MultiScaleSubjetTrainConfig:
    """Configuration for one Step 10 baseline/variant training run."""

    output_dir: str
    hlt_cache_dir: str
    variant: str = "multiscale_subjet_residual_part_adapter"
    ablation_profile: str = ""
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = False
    seed: int = 4107
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    lr: float = 3.0e-4
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
    selection_metric: str = MULTISCALE_SUBJET_PRIMARY_METRIC
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH

    num_classes: int = 2
    label_names: tuple[str, ...] = MULTISCALE_SUBJET_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = MULTISCALE_SUBJET_BINARY_LABEL_FILTER

    model_size: str = "base"
    max_constits: int = 128
    token_dim: int = 128
    token_hidden_dim: int = 256
    assignment_embed_dim: int = 64
    assignment_hidden_dim: int = 128
    assignment_temperature: float = 1.0
    assignment_geometry_bias_strength: float = 2.0
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 256
    transformer_pair_bias_hidden_dim: int = 64
    readback_hidden_dim: int = 128
    readback_heads: int = 4
    readback_delta_hidden_dim: int = 256
    branch_hidden_dim: int = 256
    residual_gamma_init: float = 0.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    scale_profile: str = "default"
    use_assignment_scale_embedding: bool = True
    use_token_scale_embedding: bool = True
    use_subjet_pair_bias: bool = True
    use_scale_pair_embedding: bool = True
    random_control_seed: int = 2027
    weight_threshold: float = 0.0

    def __post_init__(self) -> None:
        self.variant = normalize_multiscale_subjet_variant(self.variant)
        self.ablation_profile = str(self.ablation_profile or self.variant).strip()
        if self.variant not in MULTISCALE_SUBJET_CLASSIFIER_VARIANTS:
            raise ValueError(f"variant must be one of {MULTISCALE_SUBJET_CLASSIFIER_VARIANTS}")
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 10 trains only on model_train and selects only on model_val")
        if self.stack_val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 10 evaluates only stack_val/final_test after model_val selection")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set confirm_split_settings=True to acknowledge model_train/model_val-only selection")
        protocol = default_multiscale_subjet_part_protocol()
        if str(self.selection_metric) not in MULTISCALE_SUBJET_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {MULTISCALE_SUBJET_SELECTION_METRICS}")
        if str(self.selection_metric) != str(protocol.selection_metric):
            raise ValueError(f"Step 10 protocol selects checkpoints with {protocol.selection_metric}")
        self.expected_hlt_degradation_strength = float(self.expected_hlt_degradation_strength)
        if abs(float(self.expected_hlt_degradation_strength) - float(protocol.hlt_degradation_strength)) > 1.0e-12:
            raise ValueError(f"Step 10 protocol requires HLT degradation strength {protocol.hlt_degradation_strength}")
        for field_name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "num_classes",
            "max_constits",
            "token_dim",
            "token_hidden_dim",
            "assignment_embed_dim",
            "assignment_hidden_dim",
            "transformer_heads",
            "transformer_ffn_dim",
            "transformer_pair_bias_hidden_dim",
            "readback_hidden_dim",
            "readback_heads",
            "readback_delta_hidden_dim",
            "branch_hidden_dim",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        self.transformer_layers = int(self.transformer_layers)
        if self.transformer_layers < 0:
            raise ValueError("transformer_layers must be non-negative")
        if int(self.token_dim) % int(self.transformer_heads) != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if int(self.readback_hidden_dim) % int(self.readback_heads) != 0:
            raise ValueError("readback_hidden_dim must be divisible by readback_heads")
        if str(self.model_size) not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        for field_name in ("assignment_temperature", "assignment_geometry_bias_strength"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be non-negative and finite")
            if field_name == "assignment_temperature" and value <= 0.0:
                raise ValueError("assignment_temperature must be positive")
            setattr(self, field_name, value)
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if float(self.grad_clip_norm) < 0.0:
            raise ValueError("grad_clip_norm cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        for field_name in ("dropout", "attention_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            setattr(self, field_name, value)
        self.residual_gamma_init = float(self.residual_gamma_init)
        if not np.isfinite(self.residual_gamma_init):
            raise ValueError("residual_gamma_init must be finite")
        self.weight_threshold = float(self.weight_threshold)
        if self.weight_threshold < 0.0:
            raise ValueError("weight_threshold must be non-negative")
        self.scale_profile = str(self.scale_profile).strip().lower().replace("-", "_")
        # Validated in the model config too; importing here keeps bad Slurm args
        # from running long enough to allocate a GPU.
        from .features import multiscale_subjet_scale_specs_for_profile

        multiscale_subjet_scale_specs_for_profile(self.scale_profile)
        self.use_assignment_scale_embedding = bool(self.use_assignment_scale_embedding)
        self.use_token_scale_embedding = bool(self.use_token_scale_embedding)
        self.use_subjet_pair_bias = bool(self.use_subjet_pair_bias)
        self.use_scale_pair_embedding = bool(self.use_scale_pair_embedding)
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_stack_val_batches = _optional_nonnegative_int(self.max_stack_val_batches, field_name="max_stack_val_batches")
        self.max_final_test_batches = _optional_nonnegative_int(self.max_final_test_batches, field_name="max_final_test_batches")
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_stack_val_jets = _optional_positive_int(self.max_stack_val_jets, field_name="max_stack_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        self.label_names = tuple(str(name) for name in self.label_names)
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if int(self.num_classes) != len(self.label_names) or int(self.num_classes) != len(self.label_filter):
            raise ValueError("num_classes, label_names, and label_filter must have matching lengths")
        if int(self.num_classes) != 2:
            raise ValueError("Step 10 multi-scale subjet protocol is binary QCD/Hgg only")
        if tuple(self.label_names) != tuple(protocol.binary_label_names):
            raise ValueError("Step 10 multi-scale subjet protocol is frozen to QCD/Hgg labels")
        if tuple(self.label_filter) != tuple(protocol.binary_label_filter):
            raise ValueError("Step 10 expects binary-cache labels QCD=0, Hgg=1")


def _load_multiscale_subjet_dataset(
    config: MultiScaleSubjetTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    view.metadata["hlt_protocol_audit"] = _verify_hlt_cache_protocol(
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
    dataset.metadata["hlt_protocol_audit"] = view.metadata["hlt_protocol_audit"]
    return dataset


def build_multiscale_subjet_tagger_for_config(
    config: MultiScaleSubjetTrainConfig,
    *,
    part_model: Any | None = None,
) -> MultiScaleSubjetPartClassifier:
    """Build the Step 10 baseline/variant model for a train config."""

    return build_multiscale_subjet_comparison_classifier(
        config.variant,
        num_classes=int(config.num_classes),
        model_size=str(config.model_size),
        max_constits=int(config.max_constits),
        token_dim=int(config.token_dim),
        token_hidden_dim=int(config.token_hidden_dim),
        assignment_embed_dim=int(config.assignment_embed_dim),
        assignment_hidden_dim=int(config.assignment_hidden_dim),
        assignment_temperature=float(config.assignment_temperature),
        assignment_geometry_bias_strength=float(config.assignment_geometry_bias_strength),
        transformer_layers=int(config.transformer_layers),
        transformer_heads=int(config.transformer_heads),
        transformer_ffn_dim=int(config.transformer_ffn_dim),
        transformer_pair_bias_hidden_dim=int(config.transformer_pair_bias_hidden_dim),
        readback_hidden_dim=int(config.readback_hidden_dim),
        readback_heads=int(config.readback_heads),
        readback_delta_hidden_dim=int(config.readback_delta_hidden_dim),
        branch_hidden_dim=int(config.branch_hidden_dim),
        residual_gamma_init=float(config.residual_gamma_init),
        dropout=float(config.dropout),
        attention_dropout=float(config.attention_dropout),
        scale_profile=str(config.scale_profile),
        use_assignment_scale_embedding=bool(config.use_assignment_scale_embedding),
        use_token_scale_embedding=bool(config.use_token_scale_embedding),
        use_subjet_pair_bias=bool(config.use_subjet_pair_bias),
        use_scale_pair_embedding=bool(config.use_scale_pair_embedding),
        random_control_seed=int(config.random_control_seed),
        weight_threshold=float(config.weight_threshold),
        part_model=part_model,
    )


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _flatten_scalar_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in diagnostics.items():
        if hasattr(value, "detach"):
            try:
                if int(value.numel()) == 1:
                    output[key] = float(value.detach().cpu().item())
            except Exception:
                continue
        elif isinstance(value, (int, float, bool)):
            numeric = float(value)
            if np.isfinite(numeric):
                output[key] = numeric
    return output


def _json_safe_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "_prediction_arrays"}


def run_multiscale_subjet_tagger_epoch(
    model,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one train/eval epoch for a Step 10 multiscale-subjet model."""

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
    autocast_enabled = bool(amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = None
                if collect_diagnostics:
                    try:
                        output = model(batch["tokens"], batch["mask"], return_outputs=True)
                        logits = output.logits
                    except TypeError:
                        logits = model(batch["tokens"], batch["mask"])
                else:
                    logits = model(batch["tokens"], batch["mask"])
                loss = criterion(logits, batch["labels"])

            if training:
                if not bool(getattr(loss, "requires_grad", False)):
                    raise RuntimeError("multiscale subjet training loss does not require grad")
                if scaler is not None and scaler.is_enabled():
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

            labels = batch["labels"]
            preds = logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float32))
            if collect_diagnostics and output is not None and hasattr(output, "diagnostics"):
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
                label_names=label_names,
            )
        )
        metrics["_prediction_arrays"] = {
            "preds": preds_np,
            "labels": labels_np,
            "logits": logits_np,
        }
    if diagnostic_weight_sum > 0.0:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_sum
            for key, value in sorted(diagnostic_totals.items())
        }
    return metrics


def _write_epoch_metrics_csv(path: Path, curves: list[dict[str, Any]]) -> None:
    keys: set[str] = {"epoch"}
    rows: list[dict[str, Any]] = []
    for row in curves:
        flat = {"epoch": row["epoch"]}
        for split in ("train", "model_val"):
            for key, value in row.get(split, {}).items():
                if isinstance(value, (int, float)):
                    flat[f"{split}_{key}"] = value
                    keys.add(f"{split}_{key}")
                if key == "binary_metrics" and isinstance(value, Mapping):
                    for b_key, b_value in value.items():
                        if isinstance(b_value, (int, float)):
                            flat[f"{split}_{b_key}"] = b_value
                            keys.add(f"{split}_{b_key}")
        rows.append(flat)
    ordered = ["epoch"] + sorted(key for key in keys if key != "epoch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, metrics_by_split: Mapping[str, Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for split, metrics in metrics_by_split.items():
        binary = metrics.get("binary_metrics", {})
        rows.append(
            {
                "split": split,
                "loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
                "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                "auc": binary.get("auc") if isinstance(binary, Mapping) else None,
                "fpr_at_signal_eff_0p30": binary.get("fpr_at_signal_eff_0p30") if isinstance(binary, Mapping) else None,
                "fpr_at_signal_eff_0p50": binary.get("fpr_at_signal_eff_0p50") if isinstance(binary, Mapping) else None,
                "background_rejection_at_signal_eff_0p50": (
                    binary.get("background_rejection_at_signal_eff_0p50") if isinstance(binary, Mapping) else None
                ),
                "validation_threshold_final_test_fpr": (
                    binary.get("validation_threshold_final_test_fpr") if isinstance(binary, Mapping) else None
                ),
                "validation_threshold_final_test_signal_efficiency": (
                    binary.get("validation_threshold_final_test_signal_efficiency") if isinstance(binary, Mapping) else None
                ),
                "n_jets": metrics.get("n_jets"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "loss",
                "accuracy",
                "macro_per_class_accuracy",
                "auc",
                "fpr_at_signal_eff_0p30",
                "fpr_at_signal_eff_0p50",
                "background_rejection_at_signal_eff_0p50",
                "validation_threshold_final_test_fpr",
                "validation_threshold_final_test_signal_efficiency",
                "n_jets",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_best_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    row = {
        "variant": report.get("variant"),
        "best_epoch": report.get("best_epoch"),
        "selection_metric": report.get("selection_metric"),
        "best_model_selection_metric_value": report.get("best_model_selection_metric_value"),
        "best_model_selection_score": report.get("best_model_selection_score"),
        "best_model_val_accuracy": report.get("best_model_val_accuracy"),
        "best_model_val_loss": report.get("best_model_val_loss"),
    }
    for split_key, metrics_key in (
        ("model_val", "best_model_val_metrics"),
        ("stack_val", "stack_val_metrics"),
        ("final_test", "final_test_metrics"),
    ):
        metrics = report.get(metrics_key)
        if not isinstance(metrics, Mapping):
            continue
        row[f"{split_key}_accuracy"] = metrics.get("accuracy")
        row[f"{split_key}_loss"] = metrics.get("loss")
        binary = metrics.get("binary_metrics")
        if isinstance(binary, Mapping):
            row[f"{split_key}_auc"] = binary.get("auc")
            row[f"{split_key}_fpr_at_signal_eff_0p30"] = binary.get("fpr_at_signal_eff_0p30")
            row[f"{split_key}_fpr_at_signal_eff_0p50"] = binary.get("fpr_at_signal_eff_0p50")
            row[f"{split_key}_validation_threshold_final_test_fpr"] = binary.get("validation_threshold_final_test_fpr")
            row[f"{split_key}_validation_threshold_final_test_signal_efficiency"] = binary.get(
                "validation_threshold_final_test_signal_efficiency"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def multiscale_subjet_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: MultiScaleSubjetTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_config": model.to_config_dict() if hasattr(model, "to_config_dict") else {},
        "metrics": _json_safe_metrics(metrics),
        "variant": str(config.variant),
        "profile": str(config.ablation_profile),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.num_classes),
        "source": dict(source),
        "experiment_step": MULTISCALE_SUBJET_TRAIN_STEP,
        "protocol_step": MULTISCALE_SUBJET_PROTOCOL_STEP,
        "protocol_contract": MULTISCALE_SUBJET_CONTRACT,
        "output_contract": getattr(model, "output_contract", None),
    }


def load_multiscale_subjet_checkpoint(path: str | Path, *, device=None):
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain a train config")
    config = MultiScaleSubjetTrainConfig(**dict(config_payload))
    model = build_multiscale_subjet_tagger_for_config(config)
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def train_multiscale_subjet_tagger(
    config: MultiScaleSubjetTrainConfig,
    *,
    model: Any | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train/evaluate one Step 10 baseline or multi-scale subjet variant."""

    run_start_time = time.perf_counter()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_multiscale_subjet_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_multiscale_subjet_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"multiscale subjet tagger expects raw token dim {RAW_TOKEN_DIM}")
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

    checkpoint_model = model or build_multiscale_subjet_tagger_for_config(config)
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if bool(config.compile_model) and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": MULTISCALE_SUBJET_TRAIN_STEP,
        "protocol": multiscale_subjet_part_protocol_manifest(),
        "variant": str(config.variant),
        "profile": str(config.ablation_profile),
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict() if hasattr(checkpoint_model, "to_config_dict") else {},
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.num_classes),
        "selection_metric": str(config.selection_metric),
        "leakage_rule": (
            "Step 10 consumes cached fixed-HLT tokens and labels only. Training uses model_train, "
            "checkpoint selection uses model_val, and stack_val/final_test are loaded only after "
            "model_val checkpoint selection."
        ),
        "final_test_loaded_during_training": False,
        "stack_val_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
        "diagnostics_dir": str(diagnostics_dir),
    }
    save_json(output_dir / "config.json", run_metadata)
    save_json(diagnostics_dir / "run_config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_multiscale_subjet_tagger_epoch(
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
        val_metrics = run_multiscale_subjet_tagger_epoch(
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
        row = {"epoch": int(epoch), "train": _json_safe_metrics(train_metrics), "model_val": _json_safe_metrics(val_metrics)}
        curves.append(row)
        save_json(diagnostics_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = multiscale_subjet_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            metrics=row,
            source=source,
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
        raise FloatingPointError("multiscale subjet tagger did not produce a valid model_val checkpoint")

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    if model is None:
        best_model, _ = load_multiscale_subjet_checkpoint(output_dir / "best_model_val.pt", device=device)
    else:
        checkpoint_model.load_state_dict(best_payload["model_state_dict"])
        checkpoint_model.eval()
        best_model = checkpoint_model

    best_val_metrics = run_multiscale_subjet_tagger_epoch(
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
    metrics_by_split: dict[str, Mapping[str, Any]] = {"model_val": _json_safe_metrics(best_val_metrics)}

    stack_val_dataset = stack_val_dataset or _load_multiscale_subjet_dataset(
        config,
        config.stack_val_split,
        max_jets=config.max_stack_val_jets,
    )
    stack_val_loader = make_subtoken_hlt_loader(
        stack_val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    stack_val_metrics = run_multiscale_subjet_tagger_epoch(
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
    metrics_by_split["stack_val"] = _json_safe_metrics(stack_val_metrics)

    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_multiscale_subjet_dataset(
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
        final_test_metrics = run_multiscale_subjet_tagger_epoch(
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
        threshold_metrics = _validation_threshold_final_test_metrics(best_val_metrics, final_test_metrics)
        if threshold_metrics:
            final_test_metrics = dict(final_test_metrics)
            final_binary = dict(final_test_metrics.get("binary_metrics") or {})
            final_binary.update(threshold_metrics)
            final_test_metrics["binary_metrics"] = final_binary
            final_test_metrics.update(threshold_metrics)
        metrics_by_split["final_test"] = _json_safe_metrics(final_test_metrics)
        final_test_metadata = dict(final_test_dataset.metadata)

    elapsed_seconds = float(time.perf_counter() - run_start_time)
    _write_summary_csv(diagnostics_dir / "summary_metrics.csv", metrics_by_split)

    report = {
        "experiment_step": MULTISCALE_SUBJET_TRAIN_STEP,
        "protocol_step": MULTISCALE_SUBJET_PROTOCOL_STEP,
        "protocol_contract": MULTISCALE_SUBJET_CONTRACT,
        "protocol": multiscale_subjet_part_protocol_manifest(),
        "output_contract": getattr(best_model, "output_contract", None),
        "variant": str(config.variant),
        "profile": str(config.ablation_profile),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": (
            "minimize" if str(config.selection_metric) in MULTISCALE_SUBJET_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"
        ),
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": _json_safe_metrics(best_val_metrics),
        "stack_val_metrics": _json_safe_metrics(stack_val_metrics),
        "final_test_metrics": None if final_test_metrics is None else _json_safe_metrics(final_test_metrics),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(diagnostics_dir / "training_curves.json"),
        "summary_metrics_csv": str(diagnostics_dir / "summary_metrics.csv"),
        "config": asdict(config),
        "model_config": best_model.to_config_dict() if hasattr(best_model, "to_config_dict") else {},
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.num_classes),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "stack_val_dataset": dict(stack_val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "no_final_test_evaluation": not bool(config.confirm_final_test),
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": len(curves),
            "seconds_per_completed_epoch": elapsed_seconds / float(len(curves)) if curves else None,
        },
        "walltime_seconds": elapsed_seconds,
        "inference_consumes_hlt_only": True,
        "final_test_loaded_during_training": False,
        "stack_val_loaded_during_training": False,
        "diagnostics_dir": str(diagnostics_dir),
        "large_artifacts_dir": str(output_dir),
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    save_json(diagnostics_dir / "run_report.json", report)
    _write_best_metrics_csv(diagnostics_dir / "best_metrics.csv", report)
    return report


__all__ = [
    "MULTISCALE_SUBJET_LOWER_IS_BETTER_SELECTION_METRICS",
    "MULTISCALE_SUBJET_SELECTION_METRICS",
    "MULTISCALE_SUBJET_TRAIN_STEP",
    "MultiScaleSubjetTrainConfig",
    "build_multiscale_subjet_tagger_for_config",
    "compare_hlt_params_for_multiscale_subjet",
    "load_multiscale_subjet_checkpoint",
    "multiscale_subjet_checkpoint_payload",
    "run_multiscale_subjet_tagger_epoch",
    "train_multiscale_subjet_tagger",
]
