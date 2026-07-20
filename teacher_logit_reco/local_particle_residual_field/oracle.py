"""Frozen oracle-consumer wrapper for local residual-field curriculum training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import require_torch, resolve_device

from .tagger import (
    LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
    ORACLE_RESIDUAL_FIELD_SOURCES,
    LocalResidualFieldAugmentedParT,
    normalize_residual_field_source,
)
from .train import _torch_load_checkpoint


LOCAL_RESIDUAL_FIELD_FROZEN_ORACLE_CONSUMER_CONTRACT = "local_residual_field_frozen_oracle_consumer_v1"


@dataclass(frozen=True)
class FrozenOracleConsumerConfig:
    """Configuration for a frozen local residual-field oracle consumer."""

    checkpoint: str
    consumer_id: str | None = None
    alpha: float = 1.0
    teacher_config_path: str | None = None
    run_report_path: str | None = None
    oracle_logit_only_fallback: bool = False
    oracle_forward_microbatch_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", str(self.checkpoint))
        object.__setattr__(self, "consumer_id", None if not self.consumer_id else str(self.consumer_id))
        object.__setattr__(
            self,
            "teacher_config_path",
            None if not self.teacher_config_path else str(self.teacher_config_path),
        )
        object.__setattr__(
            self,
            "run_report_path",
            None if not self.run_report_path else str(self.run_report_path),
        )
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "oracle_logit_only_fallback", bool(self.oracle_logit_only_fallback))
        microbatch_size = self.oracle_forward_microbatch_size
        if microbatch_size is not None:
            microbatch_size = int(microbatch_size)
            if microbatch_size <= 0:
                raise ValueError("oracle_forward_microbatch_size must be positive when provided")
        object.__setattr__(self, "oracle_forward_microbatch_size", microbatch_size)


@dataclass(frozen=True)
class FrozenOracleConsumerOutput:
    """Logits from a frozen oracle consumer for true and predicted fields."""

    teacher_logits_true: Any | None
    teacher_logits_pred: Any | None
    diagnostics: Mapping[str, Any]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"oracle metadata must contain a JSON object: {json_path}")
    return dict(payload)


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _metric_accuracy(payload: Mapping[str, Any]) -> float | None:
    for key in ("accuracy", "best_model_val_accuracy"):
        value = payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping) and metrics.get("accuracy") is not None:
        try:
            return float(metrics["accuracy"])
        except (TypeError, ValueError):
            return None
    return None


def _tuple_ints(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


class FrozenLocalResidualFieldOracleConsumer:
    """Frozen differentiable wrapper around an oracle residual-field tagger.

    Teacher parameters are always frozen.  The true-field branch is detached by
    default, while the predicted-field branch preserves gradients with respect
    to the supplied predicted residual fields.
    """

    def __init__(
        self,
        config: FrozenOracleConsumerConfig | Mapping[str, Any],
        *,
        device: Any = "cpu",
        map_location: Any | None = None,
    ) -> None:
        torch = require_torch()
        self.config = config if isinstance(config, FrozenOracleConsumerConfig) else FrozenOracleConsumerConfig(**dict(config))
        self.device = resolve_device(str(device))
        checkpoint_path = Path(self.config.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"oracle consumer checkpoint does not exist: {checkpoint_path}")
        load_location = "cpu" if self.config.oracle_logit_only_fallback else (map_location or self.device)
        payload = _torch_load_checkpoint(checkpoint_path, map_location=load_location)
        if not isinstance(payload, Mapping):
            raise ValueError("oracle consumer checkpoint payload must be a mapping")
        if not isinstance(payload.get("model_state_dict"), Mapping):
            raise ValueError("oracle consumer checkpoint is missing model_state_dict")
        if not isinstance(payload.get("model_config"), Mapping):
            raise ValueError("oracle consumer checkpoint is missing model_config")
        model_config = dict(payload.get("model_config") or {})
        model_contract = model_config.get("contract")
        if model_contract != LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT:
            raise ValueError(
                f"oracle consumer model contract {model_contract!r} != {LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT!r}"
            )
        model_config.pop("contract", None)
        model_config.pop("augmented_feature_dim", None)
        field_source = normalize_residual_field_source(model_config.get("field_source"))
        if field_source not in ORACLE_RESIDUAL_FIELD_SOURCES:
            raise ValueError(
                f"frozen oracle consumer requires an oracle field source, got {field_source!r}"
            )
        self.model_config = dict(model_config)
        self.model = None
        if not self.config.oracle_logit_only_fallback:
            self.model = LocalResidualFieldAugmentedParT(model_config).to(self.device)
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
            self._enforce_frozen_eval()

        default_teacher_config = checkpoint_path.with_name("teacher_config.json")
        default_run_report = checkpoint_path.with_name("run_report.json")
        self.teacher_config = _load_optional_json(self.config.teacher_config_path or default_teacher_config)
        self.run_report = _load_optional_json(self.config.run_report_path or default_run_report)
        teacher_source = self.teacher_config.get("field_source")
        if teacher_source and normalize_residual_field_source(teacher_source) != field_source:
            raise ValueError(
                f"teacher_config field_source {teacher_source!r} does not match checkpoint field_source {field_source!r}"
            )
        self.checkpoint_hash = _sha256_file(checkpoint_path)
        self.payload_metadata = {
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_metrics": dict(_nested_mapping(payload, "metrics")),
            "checkpoint_train_config": dict(_nested_mapping(payload, "config")),
            "checkpoint_model_config": dict(model_config),
            "selected_field_indices": list(payload.get("selected_field_indices") or ()),
            "selected_field_names": list(payload.get("selected_field_names") or ()),
        }
        self.selected_field_indices = _tuple_ints(
            self.teacher_config.get("selected_field_indices")
            or payload.get("selected_field_indices")
            or ()
        )
        self.selected_field_names = tuple(
            str(name)
            for name in (
                self.teacher_config.get("selected_field_names")
                or payload.get("selected_field_names")
                or model_config.get("field_names")
                or ()
            )
        )
        self.field_subset = tuple(
            str(value)
            for value in (
                self.teacher_config.get("field_subset")
                or _nested_mapping(payload, "config").get("field_subset")
                or ()
            )
        )
        self.consumer_id = (
            self.config.consumer_id
            or str(self.teacher_config.get("teacher_id") or self.run_report.get("teacher_id") or checkpoint_path.parent.name)
        )
        self.torch = torch

    def _enforce_frozen_eval(self) -> None:
        if self.model is None:
            return
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def to(self, device: Any) -> "FrozenLocalResidualFieldOracleConsumer":
        """Move the training-only consumer without making it a registered submodule."""

        self.device = resolve_device(str(device))
        if self.model is not None:
            self.model.to(self.device)
            self._enforce_frozen_eval()
        return self

    @property
    def diagnostics(self) -> dict[str, Any]:
        train_config = self.payload_metadata.get("checkpoint_train_config")
        train_split = None
        if isinstance(train_config, Mapping):
            train_split = train_config.get("train_split")
        if train_split is None:
            train_split = _nested_mapping(self.teacher_config, "train_config").get("train_split")
        if train_split is None:
            train_split = _nested_mapping(self.run_report, "config").get("train_split")
        best_val = _metric_accuracy(self.payload_metadata.get("checkpoint_metrics", {}))
        if best_val is None:
            best_val = _metric_accuracy(
                self.teacher_config.get("best_model_val", {})
                if isinstance(self.teacher_config.get("best_model_val"), Mapping)
                else {}
            )
        if best_val is None:
            best_val = _metric_accuracy(
                self.run_report.get("best_model_val", {})
                if isinstance(self.run_report.get("best_model_val"), Mapping)
                else {}
            )
        return {
            "contract": LOCAL_RESIDUAL_FIELD_FROZEN_ORACLE_CONSUMER_CONTRACT,
            "model_contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
            "consumer_id": str(self.consumer_id),
            "alpha": float(self.config.alpha),
            "teacher_field_subset": list(self.field_subset),
            "teacher_selected_field_indices": [int(index) for index in self.selected_field_indices],
            "teacher_selected_field_names": list(self.selected_field_names),
            "teacher_checkpoint": str(self.config.checkpoint),
            "teacher_checkpoint_hash": str(self.checkpoint_hash),
            "teacher_train_split": None if train_split is None else str(train_split),
            "teacher_model_val_accuracy": best_val,
            "oracle_logit_only_fallback": bool(self.config.oracle_logit_only_fallback),
            "oracle_input_gradient_distillation_enabled": not bool(self.config.oracle_logit_only_fallback),
            "oracle_forward_microbatch_size": self.config.oracle_forward_microbatch_size,
            "field_source": str(self.model_config.get("field_source") or ""),
        }

    def parameters_frozen(self) -> bool:
        return self.model is None or all(not bool(parameter.requires_grad) for parameter in self.model.parameters())

    def _prepare_fields(self, fields: Any, *, alpha: float) -> Any:
        torch = self.torch
        output = fields.to(device=self.device)
        if output.ndim != 3:
            raise ValueError(f"oracle fields must have shape [B, P, F], got {tuple(output.shape)}")
        model_field_dim = int(self.model_config.get("field_dim") or int(output.shape[-1]))
        model_source_indices = tuple(int(index) for index in self.model_config.get("source_field_indices") or ())
        if int(output.shape[-1]) == model_field_dim:
            selected = output
        elif model_source_indices:
            selected = output
        elif self.selected_field_indices:
            index_tensor = torch.as_tensor(self.selected_field_indices, device=output.device, dtype=torch.long)
            if int(index_tensor.max().detach().cpu().item()) >= int(output.shape[-1]):
                raise ValueError(
                    f"selected field index {int(index_tensor.max().detach().cpu().item())} "
                    f"is outside supplied field dim {int(output.shape[-1])}"
                )
            selected = output.index_select(dim=-1, index=index_tensor)
        else:
            raise ValueError(
                f"supplied field dim {int(output.shape[-1])} does not match teacher field_dim={model_field_dim}, "
                "and no selected_field_indices are available"
            )
        return selected * float(alpha)

    @staticmethod
    def _slice_batch(value: Any | None, start: int, stop: int) -> Any | None:
        return None if value is None else value[start:stop]

    def _consumer_logits(
        self,
        *,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        tokens: Any | None,
        raw_mask: Any | None,
        indices: Any | None,
        fields: Any,
        alpha: float,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("oracle consumer model is unavailable in logit-only fallback mode")
        self._enforce_frozen_eval()
        prepared = self._prepare_fields(fields, alpha=float(alpha))
        batch_size = int(prepared.shape[0])
        microbatch_size = self.config.oracle_forward_microbatch_size or batch_size
        logits: list[Any] = []
        for start in range(0, batch_size, int(microbatch_size)):
            stop = min(start + int(microbatch_size), batch_size)
            logits.append(
                self.model(
                    self._slice_batch(points, start, stop).to(device=self.device),
                    self._slice_batch(features, start, stop).to(device=self.device),
                    self._slice_batch(lorentz_vectors, start, stop).to(device=self.device),
                    self._slice_batch(mask, start, stop).to(device=self.device),
                    tokens=(
                        None
                        if tokens is None
                        else self._slice_batch(tokens, start, stop).to(device=self.device)
                    ),
                    raw_mask=(
                        None
                        if raw_mask is None
                        else self._slice_batch(raw_mask, start, stop).to(device=self.device)
                    ),
                    indices=(
                        None
                        if indices is None
                        else self._slice_batch(indices, start, stop).to(device=self.device)
                    ),
                    residual_fields=prepared[start:stop],
                    return_outputs=False,
                )
            )
        if not logits:
            raise ValueError("oracle consumer received an empty batch")
        return logits[0] if len(logits) == 1 else self.torch.cat(logits, dim=0)

    def _cached_true_logits(self, logits: Any, *, expected_batch_size: int) -> Any:
        torch = self.torch
        output = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
        if output.ndim != 2:
            raise ValueError(f"cached teacher logits must have shape [B, C], got {tuple(output.shape)}")
        if int(output.shape[0]) != int(expected_batch_size):
            raise ValueError(
                f"cached teacher logits batch size {int(output.shape[0])} != HLT batch size {int(expected_batch_size)}"
            )
        return output.to(device=self.device).detach()

    def _validate_cached_logits_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        selected_alpha: float,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "oracle_logit_only_fallback requires cached_true_logits_metadata with checkpoint and alpha provenance"
            )
        checkpoint_hash = metadata.get("checkpoint_hash")
        if str(checkpoint_hash or "") != str(self.checkpoint_hash):
            raise ValueError("cached teacher logits checkpoint_hash does not match the frozen consumer")
        cached_consumer_id = metadata.get("teacher_id") or metadata.get("model_name")
        if cached_consumer_id and str(cached_consumer_id) != str(self.consumer_id):
            raise ValueError(
                f"cached teacher logits consumer {cached_consumer_id!r} != selected consumer {self.consumer_id!r}"
            )
        cached_model_config = _nested_mapping(metadata, "model_config")
        cached_alpha = metadata.get("oracle_field_alpha")
        if cached_alpha is None:
            cached_alpha = cached_model_config.get("oracle_field_alpha")
        try:
            cached_alpha_value = float(cached_alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("cached teacher logits metadata is missing oracle_field_alpha") from exc
        if not math.isclose(cached_alpha_value, float(selected_alpha), rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError(
                f"cached teacher logits alpha {cached_alpha_value} != selected alpha {float(selected_alpha)}"
            )

    def __call__(
        self,
        *,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        tokens: Any | None = None,
        raw_mask: Any | None = None,
        indices: Any | None = None,
        true_fields: Any | None = None,
        predicted_fields: Any | None = None,
        cached_true_logits: Any | None = None,
        cached_true_logits_metadata: Mapping[str, Any] | None = None,
        alpha: float | None = None,
        detach_true: bool = True,
    ) -> FrozenOracleConsumerOutput:
        torch = self.torch
        selected_alpha = float(self.config.alpha if alpha is None else alpha)
        if not math.isfinite(selected_alpha) or selected_alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not bool(detach_true):
            raise ValueError("frozen oracle true-field logits must always be detached")
        true_logits = None
        pred_logits = None
        if self.config.oracle_logit_only_fallback:
            if cached_true_logits is None:
                raise ValueError(
                    "oracle_logit_only_fallback requires cached_true_logits; differentiable predicted-field "
                    "oracle logits are disabled"
                )
            self._validate_cached_logits_metadata(
                cached_true_logits_metadata,
                selected_alpha=selected_alpha,
            )
            true_logits = self._cached_true_logits(
                cached_true_logits,
                expected_batch_size=int(features.shape[0]),
            )
        else:
            if cached_true_logits is not None or cached_true_logits_metadata is not None:
                raise ValueError(
                    "cached_true_logits and its metadata may only be used with oracle_logit_only_fallback=true"
                )
            if true_fields is not None:
                with torch.no_grad():
                    true_logits = self._consumer_logits(
                        points=points,
                        features=features,
                        lorentz_vectors=lorentz_vectors,
                        mask=mask,
                        tokens=tokens,
                        raw_mask=raw_mask,
                        indices=indices,
                        fields=true_fields,
                        alpha=selected_alpha,
                    ).detach()
            if predicted_fields is not None:
                pred_logits = self._consumer_logits(
                    points=points,
                    features=features,
                    lorentz_vectors=lorentz_vectors,
                    mask=mask,
                    tokens=tokens,
                    raw_mask=raw_mask,
                    indices=indices,
                    fields=predicted_fields,
                    alpha=selected_alpha,
                )
        diagnostics = dict(self.diagnostics)
        diagnostics["alpha"] = selected_alpha
        diagnostics["teacher_logits_true_present"] = bool(true_logits is not None)
        diagnostics["teacher_logits_pred_present"] = bool(pred_logits is not None)
        diagnostics["parameters_frozen"] = bool(self.parameters_frozen())
        diagnostics["teacher_logits_true_source"] = (
            "cached" if self.config.oracle_logit_only_fallback else ("live_true_fields" if true_logits is not None else None)
        )
        diagnostics["oracle_pred_logits_disabled_by_fallback"] = bool(
            self.config.oracle_logit_only_fallback
        )
        diagnostics["cached_true_logits_metadata_validated"] = bool(
            self.config.oracle_logit_only_fallback and cached_true_logits_metadata is not None
        )
        return FrozenOracleConsumerOutput(
            teacher_logits_true=true_logits,
            teacher_logits_pred=pred_logits,
            diagnostics=diagnostics,
        )


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FROZEN_ORACLE_CONSUMER_CONTRACT",
    "FrozenOracleConsumerConfig",
    "FrozenOracleConsumerOutput",
    "FrozenLocalResidualFieldOracleConsumer",
]
