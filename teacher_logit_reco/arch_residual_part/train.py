"""Training for Version A architecture residual experts."""

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

from teacher_logit_reco.local_compression_part.checkpoint import load_hlt_part_baseline_checkpoint
from teacher_logit_reco.local_compression_part.config import (
    LOCAL_COMPRESSION_BINARY_LABEL_FILTER,
    LOCAL_COMPRESSION_PART_CONTRACT,
    LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_COMPRESSION_PRIMARY_METRIC,
    LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
)
from teacher_logit_reco.local_compression_part.train import (
    _manifest_metadata,
    _selection_score,
    _verify_hlt_params,
    local_compression_label_filter_names_to_indices,
)
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .model import (
    ARCH_RESIDUAL_ARCHITECTURES,
    ARCH_RESIDUAL_MODEL_CONTRACT,
    ARCH_RESIDUAL_MODEL_STEP,
    ArchResidualExpertConfig,
    ArchResidualPartModel,
    build_arch_residual_part_model,
    normalize_architecture,
)


ARCH_RESIDUAL_TRAIN_STEP = "arch_residual_part_version_a_train"
ARCH_RESIDUAL_TRAIN_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_arch_residual_train_v1"
ARCH_RESIDUAL_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
ARCH_RESIDUAL_LOWER_IS_BETTER = {"loss", "fpr_at_signal_eff_0p30", "fpr_at_signal_eff_0p50"}


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


@dataclass
class ArchResidualTrainConfig:
    """Config for one Version A residual-expert run."""

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
    seed: int = 7307
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
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
    selection_metric: str = LOCAL_COMPRESSION_PRIMARY_METRIC
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    require_baseline_split_manifest_hash: bool = True
    expected_hlt_degradation_strength: float = LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH
    label_names: tuple[str, ...] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = LOCAL_COMPRESSION_BINARY_LABEL_FILTER
    architecture: str = "pfn"
    hidden_dim: int = 128
    particle_layers: int = 3
    global_layers: int = 2
    edge_k: int = 16
    dropout: float = 0.05
    condition_on_baseline: bool = True
    gamma_init: float = 1.0
    residual_scale: float = 1.0
    residual_l2_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.manifest_path = str(self.manifest_path)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.baseline_checkpoint = str(self.baseline_checkpoint)
        if not self.baseline_checkpoint:
            raise ValueError("baseline_checkpoint is required")
        for split_field, expected in (
            ("train_split", "model_train"),
            ("val_split", "model_val"),
            ("stack_val_split", "stack_val"),
            ("final_test_split", "final_test"),
        ):
            value = str(getattr(self, split_field))
            if value != expected:
                raise ValueError(f"{split_field} must be {expected!r}")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge model_train/model_val-only selection")
        if not bool(self.confirm_final_test):
            raise ValueError("Version A runner requires --confirm-final-test for guarded final evaluation")
        if str(self.selection_metric) != LOCAL_COMPRESSION_PRIMARY_METRIC:
            raise ValueError("Version A checkpoint selection must use fpr_at_signal_eff_0p50")
        self.architecture = normalize_architecture(self.architecture)
        for field_name in ("batch_size", "eval_batch_size", "epochs", "hidden_dim", "particle_layers", "global_layers", "edge_k"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        for field_name in ("lr", "residual_scale"):
            value = float(getattr(self, field_name))
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        for field_name in ("weight_decay", "grad_clip_norm", "residual_l2_weight"):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            setattr(self, field_name, value)
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        self.dropout = float(self.dropout)
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
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
        if self.label_names != LOCAL_COMPRESSION_SOURCE_LABEL_NAMES:
            raise ValueError(f"Version A uses active binary labels {LOCAL_COMPRESSION_SOURCE_LABEL_NAMES}")
        if self.label_filter != LOCAL_COMPRESSION_BINARY_LABEL_FILTER:
            raise ValueError(f"Version A expects binary-cache labels {LOCAL_COMPRESSION_BINARY_LABEL_FILTER}")

    def model_config(self) -> ArchResidualExpertConfig:
        return ArchResidualExpertConfig(
            architecture=str(self.architecture),
            canonical_feature_dim=19,
            hidden_dim=int(self.hidden_dim),
            particle_layers=int(self.particle_layers),
            global_layers=int(self.global_layers),
            edge_k=int(self.edge_k),
            dropout=float(self.dropout),
            condition_on_baseline=bool(self.condition_on_baseline),
            gamma_init=float(self.gamma_init),
            residual_scale=float(self.residual_scale),
            label_names=tuple(self.label_names),
        )


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _load_dataset(config: ArchResidualTrainConfig, split: str, *, max_jets: int | None) -> SubtokenHLTJetDataset:
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


def _metrics_without_prediction_arrays(metrics: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(metrics)
    clean.pop("_prediction_arrays", None)
    return clean


def _score_arrays_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1.0e-12, None))[:, 1].astype(np.float32)


def _json_prediction_arrays(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    arrays = metrics.get("_prediction_arrays")
    if not isinstance(arrays, Mapping):
        return None
    labels = np.asarray(arrays.get("labels"), dtype=np.int64)
    logits = np.asarray(arrays.get("logits"), dtype=np.float32)
    if labels.size == 0 or logits.ndim != 2:
        return None
    return {
        "labels": labels.tolist(),
        "scores": _score_arrays_from_logits(logits).tolist(),
        "logits": logits.tolist(),
        "score_name": "p_Hgg_from_fused_logits",
        "n_jets": int(labels.shape[0]),
    }


def _flatten_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in diagnostics.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten_diagnostics(value).items():
                output[f"{key}.{sub_key}"] = sub_value
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            output[str(key)] = numeric
    return output


def run_arch_residual_epoch(
    model: ArchResidualPartModel,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float = 1.0,
    max_batches: int | None = None,
    residual_l2_weight: float = 0.0,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    label_names: Sequence[str] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    model.baseline.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_residual_l2_loss = 0.0
    total_residual_l2_mean = 0.0
    total_correct = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    collected_baseline_logits: list[np.ndarray] = []
    collected_residual_logits: list[np.ndarray] = []
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
                    return_outputs=True,
                    max_constits=int(batch["tokens"].shape[1]),
                )
                logits = output.logits
                ce_loss = criterion(logits, labels)
                residual_l2_mean = output.correction.square().mean()
                residual_l2_loss = residual_l2_mean * float(residual_l2_weight)
                loss = ce_loss + residual_l2_loss
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
            total_residual_l2_loss += float(residual_l2_loss.detach().item()) * batch_size
            total_residual_l2_mean += float(residual_l2_mean.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float32))
                collected_baseline_logits.append(output.baseline_logits.detach().cpu().numpy().astype(np.float32))
                collected_residual_logits.append(output.residual_only_logits.detach().cpu().numpy().astype(np.float32))
            if collect_diagnostics:
                flat = _flatten_diagnostics(output.diagnostics())
                for key, value in flat.items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * batch_size
                diagnostic_weight_sum += float(batch_size)
    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "ce_loss": total_ce_loss / float(total_seen),
        "residual_l2_loss": total_residual_l2_loss / float(total_seen),
        "residual_l2_mean": total_residual_l2_mean / float(total_seen),
        "residual_l2_weight": float(residual_l2_weight),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else np.zeros((0, 2), dtype=np.float32)
        baseline_logits_np = (
            np.concatenate(collected_baseline_logits, axis=0)
            if collected_baseline_logits
            else np.zeros((0, 2), dtype=np.float32)
        )
        residual_logits_np = (
            np.concatenate(collected_residual_logits, axis=0)
            if collected_residual_logits
            else np.zeros((0, 2), dtype=np.float32)
        )
        metrics["_prediction_arrays"] = {"preds": preds_np, "labels": labels_np, "logits": logits_np}
        metrics.update(
            classification_metrics_from_predictions(
                preds=preds_np,
                labels=labels_np,
                loss_sum=total_loss,
                logits=logits_np,
                label_names=tuple(label_names),
            )
        )
        metrics["baseline_metrics"] = classification_metrics_from_predictions(
            preds=baseline_logits_np.argmax(axis=1),
            labels=labels_np,
            logits=baseline_logits_np,
            label_names=tuple(label_names),
        )
        metrics["residual_only_metrics"] = classification_metrics_from_predictions(
            preds=residual_logits_np.argmax(axis=1),
            labels=labels_np,
            logits=residual_logits_np,
            label_names=tuple(label_names),
        )
    if diagnostic_weight_sum > 0.0:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_sum
            for key, value in sorted(diagnostic_totals.items())
        }
    return metrics


def _write_epoch_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        train = row.get("train", {}) if isinstance(row.get("train"), Mapping) else {}
        val = row.get("model_val", {}) if isinstance(row.get("model_val"), Mapping) else {}
        val_binary = val.get("binary_metrics") if isinstance(val.get("binary_metrics"), Mapping) else {}
        payload = {
            "epoch": row.get("epoch"),
            "train_loss": train.get("loss"),
            "train_ce_loss": train.get("ce_loss"),
            "train_residual_l2_mean": train.get("residual_l2_mean"),
            "train_accuracy": train.get("accuracy"),
            "model_val_loss": val.get("loss"),
            "model_val_ce_loss": val.get("ce_loss"),
            "model_val_residual_l2_mean": val.get("residual_l2_mean"),
            "model_val_accuracy": val.get("accuracy"),
            "model_val_auc": val_binary.get("auc"),
            "model_val_fpr_at_signal_eff_0p50": val_binary.get("fpr_at_signal_eff_0p50"),
        }
        for prefix, metrics in (("train", train), ("model_val", val)):
            diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), Mapping) else {}
            for key, value in diagnostics.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    payload[f"{prefix}_diag_{str(key).replace('.', '_')}"] = numeric
        flat_rows.append(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in flat_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["epoch"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


def _checkpoint_payload(
    model: ArchResidualPartModel,
    optimizer,
    *,
    epoch: int,
    config: ArchResidualTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    baseline_load_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "metrics": dict(metrics),
        "source": dict(source),
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": dict(baseline_load_report),
        "experiment_step": ARCH_RESIDUAL_TRAIN_STEP,
        "model_step": ARCH_RESIDUAL_MODEL_STEP,
        "output_contract": model.output_contract,
    }


def _evaluate_dataset(
    model: ArchResidualPartModel,
    dataset: SubtokenHLTJetDataset,
    config: ArchResidualTrainConfig,
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
    return run_arch_residual_epoch(
        model,
        loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=max_batches,
        residual_l2_weight=float(config.residual_l2_weight),
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
    )


def train_arch_residual_tagger(
    config: ArchResidualTrainConfig,
    *,
    model: ArchResidualPartModel | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train one frozen-ParT architecture residual expert."""

    run_start = time.perf_counter()
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
    if int(train_dataset.tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"arch residual expects raw token dim {RAW_TOKEN_DIM}")

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

    model = model or build_arch_residual_part_model(config.model_config())
    model = model.to(device)
    baseline_report = load_hlt_part_baseline_checkpoint(
        config.baseline_checkpoint,
        model.baseline.part_model,
        map_location=device,
        expected_selection_metric=LOCAL_COMPRESSION_PRIMARY_METRIC,
        expected_hlt_degradation_strength=float(config.expected_hlt_degradation_strength),
        expected_split_manifest_hash=manifest_sha if bool(config.require_baseline_split_manifest_hash) else None,
        expected_label_names=tuple(config.label_names),
        expected_label_filter=tuple(config.label_filter),
        expected_num_classes=2,
        require_metadata=True,
        require_all_part_keys=True,
    ).to_dict()
    model.baseline.baseline_checkpoint_report = dict(baseline_report)
    model.baseline.eval()
    for parameter in model.baseline.parameters():
        parameter.requires_grad = False
    save_json(diagnostics_dir / "baseline_load_report.json", baseline_report)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": ARCH_RESIDUAL_TRAIN_STEP,
        "train_contract": ARCH_RESIDUAL_TRAIN_CONTRACT,
        "model_step": ARCH_RESIDUAL_MODEL_STEP,
        "output_contract": model.output_contract,
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "source": source,
        "manifest": manifest_info,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "selection_metric": str(config.selection_metric),
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_report,
        "baseline_is_frozen": True,
        "fused_score_rule": "fused_logits = z_hlt_part + [-0.5*delta, +0.5*delta]",
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_value = float("nan")
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_arch_residual_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
            residual_l2_weight=float(config.residual_l2_weight),
            collect_predictions=False,
            collect_diagnostics=True,
            label_names=tuple(config.label_names),
        )
        val_metrics = run_arch_residual_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            residual_l2_weight=float(config.residual_l2_weight),
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=tuple(config.label_names),
        )
        row = {
            "epoch": int(epoch),
            "train": _metrics_without_prediction_arrays(train_metrics),
            "model_val": _metrics_without_prediction_arrays(val_metrics),
        }
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_csv(diagnostics_dir / "epoch_metrics.csv", curves)
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            config=config,
            metrics=row,
            source=source,
            baseline_load_report=baseline_report,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_score = float(val_score)
            best_selection_value = float(selection_value)
            best_val_accuracy = float(val_accuracy)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0:
        raise FloatingPointError("arch residual training did not produce a model_val checkpoint")

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    model.eval()
    model.baseline.eval()

    best_val_metrics = _evaluate_dataset(
        model,
        val_dataset,
        config,
        device=device,
        criterion=criterion,
        seed_offset=1,
        max_batches=config.max_val_batches,
    )
    stack_val_dataset = stack_val_dataset or _load_dataset(
        config,
        config.stack_val_split,
        max_jets=config.max_stack_val_jets,
    )
    stack_val_metrics = _evaluate_dataset(
        model,
        stack_val_dataset,
        config,
        device=device,
        criterion=criterion,
        seed_offset=2,
        max_batches=config.max_stack_val_batches,
    )
    final_test_dataset = final_test_dataset or _load_dataset(
        config,
        config.final_test_split,
        max_jets=config.max_final_test_jets,
    )
    final_test_metrics = _evaluate_dataset(
        model,
        final_test_dataset,
        config,
        device=device,
        criterion=criterion,
        seed_offset=3,
        max_batches=config.max_final_test_batches,
    )

    elapsed = float(time.perf_counter() - run_start)
    report = {
        "experiment_step": ARCH_RESIDUAL_TRAIN_STEP,
        "train_contract": ARCH_RESIDUAL_TRAIN_CONTRACT,
        "model_step": ARCH_RESIDUAL_MODEL_STEP,
        "output_contract": model.output_contract,
        "architecture": str(config.architecture),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": "minimize",
        "best_model_selection_metric_value": float(best_selection_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": {
            **_metrics_without_prediction_arrays(best_val_metrics),
            "prediction_arrays": _json_prediction_arrays(best_val_metrics),
        },
        "stack_val_metrics": _metrics_without_prediction_arrays(stack_val_metrics),
        "final_test_metrics": {
            **_metrics_without_prediction_arrays(final_test_metrics),
            "prediction_arrays": _json_prediction_arrays(final_test_metrics),
        },
        "baseline_model_val_metrics": best_val_metrics.get("baseline_metrics"),
        "baseline_stack_val_metrics": stack_val_metrics.get("baseline_metrics"),
        "baseline_final_test_metrics": final_test_metrics.get("baseline_metrics"),
        "residual_only_model_val_metrics": best_val_metrics.get("residual_only_metrics"),
        "residual_only_stack_val_metrics": stack_val_metrics.get("residual_only_metrics"),
        "residual_only_final_test_metrics": final_test_metrics.get("residual_only_metrics"),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "source": source,
        "manifest": manifest_info,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "stack_val_dataset": dict(stack_val_dataset.metadata),
        "final_test_dataset": dict(final_test_dataset.metadata),
        "final_test_evaluated": True,
        "baseline_checkpoint": str(config.baseline_checkpoint),
        "baseline_load_report": baseline_report,
        "baseline_checkpoint_hash": baseline_report.get("baseline_checkpoint_hash"),
        "baseline_checkpoint_split_manifest_hash": baseline_report.get("baseline_checkpoint_split_manifest_hash"),
        "baseline_checkpoint_selection_metric": baseline_report.get("baseline_checkpoint_selection_metric"),
        "baseline_checkpoint_hlt_degradation_strength": baseline_report.get(
            "baseline_checkpoint_hlt_degradation_strength"
        ),
        "runtime": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
            "epochs_completed": len(curves),
            "seconds_per_completed_epoch": elapsed / float(len(curves)) if curves else None,
        },
        "walltime_seconds": elapsed,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "ARCH_RESIDUAL_ARCHITECTURES",
    "ARCH_RESIDUAL_LOWER_IS_BETTER",
    "ARCH_RESIDUAL_SELECTION_METRICS",
    "ARCH_RESIDUAL_TRAIN_CONTRACT",
    "ARCH_RESIDUAL_TRAIN_STEP",
    "ArchResidualTrainConfig",
    "local_compression_label_filter_names_to_indices",
    "run_arch_residual_epoch",
    "train_arch_residual_tagger",
]
