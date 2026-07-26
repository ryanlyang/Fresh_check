"""Correlated error sampling and robust particle-view consumer training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .consumer import ParticleViewConsumer
from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .predictor import HierarchicalParticleViewPredictor
from .train_predictor import (
    PARTICLE_VIEW_PVIEW0_CHECKPOINT_CONTRACT,
    PARTICLE_VIEW_PVIEW0_REGISTRATION_CONTRACT,
    validate_pview0_registration,
)


PARTICLE_VIEW_RESIDUAL_SAMPLER_CONTRACT = (
    "particle_view_correlated_residual_sampler_v1"
)
PARTICLE_VIEW_ROBUST_MIXTURE_CONTRACT = "particle_view_robust_mixture_v1"
PARTICLE_VIEW_ROBUST_CONSUMER_REGISTRATION_CONTRACT = (
    "particle_view_robust_consumer_registration_v1"
)
PARTICLE_VIEW_ROBUST_CHECKPOINT_CONTRACT = (
    "particle_view_robust_consumer_checkpoint_v1"
)
PARTICLE_VIEW_PAIRED_CONSUMER_METRICS_CONTRACT = (
    "particle_view_paired_consumer_metrics_v1"
)
PARTICLE_VIEW_PREDICTOR_OVERFIT_WARNING_CONTRACT = (
    "particle_view_predictor_overfit_warning_v1"
)

ROBUST_VIEW_PROBABILITIES = {
    "true": 0.30,
    "predicted": 0.45,
    "perturbed": 0.15,
    "zero": 0.10,
}
PREDICTION_VARIANT_WEIGHTS = {
    "snapshot_epoch_2": 0.25,
    "snapshot_epoch_3": 0.25,
    "snapshot_epoch_4": 0.25,
    "mc_dropout_epoch_4": 0.25,
}
ROBUST_CONSUMER_LINEAGE_FIELDS = (
    "source_manifest_sha256",
    "train_identity_sha256",
    "model_val_stop_split_sha256",
    "target_selection_sha256",
    "coordinate_binding_sha256",
    "selected_view_publication_sha256",
    "train_view_cache_manifest_sha256",
    "model_val_stop_view_cache_manifest_sha256",
    "clean_consumer_registration_sha256",
    "clean_consumer_checkpoint_sha256",
    "pview0_registration_sha256",
    "pview0_checkpoint_sha256",
    "residual_sampler_registration_sha256",
)


def _array_hash(values: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (values, mask):
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _as_numpy(value: np.ndarray | torch.Tensor, *, dtype=None) -> np.ndarray:
    result = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.ascontiguousarray(result)


def _residual_diagnostics(
    residual: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    selected = residual[mask]
    if not selected.size:
        raise ValueError("residual diagnostics require valid particles")
    counts = mask.sum(axis=1) * residual.shape[2]
    event_norm = np.sqrt(
        np.square(residual).sum(axis=(1, 2))
        / np.maximum(counts, 1)
    )
    eligible = counts > 0
    covariance = np.cov(selected.astype(np.float64), rowvar=False, ddof=0)
    covariance = np.atleast_2d(covariance)
    spectrum = np.linalg.eigvalsh(covariance)
    return {
        "event_residual_norm_quantiles": {
            str(percentile): float(
                np.quantile(event_norm[eligible], percentile / 100.0)
            )
            for percentile in (50, 90, 95, 99)
        },
        "per_dimension_rms": [
            float(value)
            for value in np.sqrt(
                np.mean(np.square(selected.astype(np.float64)), axis=0)
            )
        ],
        "covariance_matrix": covariance.tolist(),
        "covariance_spectrum": spectrum.tolist(),
        "events": int(residual.shape[0]),
        "valid_particles": int(mask.sum()),
    }


@dataclass(frozen=True)
class CorrelatedResidualSamplerConfig:
    residual_clip: float = 12.0
    inflation_minimum: float = 1.0
    inflation_maximum: float = 2.0
    inflation_quantiles: tuple[int, ...] = (90, 95, 99)
    residual_sampling_seed: int = 610_271
    mc_dropout_seed: int = 610_337
    residual_mask_schema: str = "sampled_event_then_current_mask_v1"
    inflation_version: str = "max_q90_q95_q99_heldout_train_clip_1_2_v1"
    contract: str = PARTICLE_VIEW_RESIDUAL_SAMPLER_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.residual_clip != 12.0
            or self.inflation_minimum != 1.0
            or self.inflation_maximum != 2.0
            or self.inflation_quantiles != (90, 95, 99)
            or self.residual_sampling_seed != 610_271
            or self.mc_dropout_seed != 610_337
            or self.residual_mask_schema
            != "sampled_event_then_current_mask_v1"
            or self.inflation_version
            != "max_q90_q95_q99_heldout_train_clip_1_2_v1"
        ):
            raise ValueError("correlated residual sampler contract changed")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            **self.__dict__,
            "inflation_quantiles": list(self.inflation_quantiles),
            "sampling_unit": "whole_masked_train_event",
            "preserves_particle_dimension_correlation": True,
            "independent_coordinate_gaussian_sampling": False,
        }
        return payload


class CorrelatedResidualSampler:
    def __init__(
        self,
        residuals: np.ndarray,
        masks: np.ndarray,
        *,
        inflation_factor: float,
        registration: Mapping[str, Any],
    ) -> None:
        self.residuals = np.ascontiguousarray(residuals, dtype="<f4")
        self.masks = np.ascontiguousarray(masks, dtype=np.bool_)
        self.inflation_factor = float(inflation_factor)
        self.registration = dict(registration)
        validate_content_hash(
            self.registration,
            expected_contract=PARTICLE_VIEW_RESIDUAL_SAMPLER_CONTRACT,
        )
        if self.registration.get("config") != (
            CorrelatedResidualSamplerConfig().to_payload()
        ):
            raise ValueError("correlated residual sampler config mismatch")
        if (
            self.residuals.ndim != 3
            or self.masks.shape != self.residuals.shape[:2]
            or not 1.0 <= self.inflation_factor <= 2.0
        ):
            raise ValueError("correlated residual sampler arrays are invalid")
        if _array_hash(self.residuals, self.masks) != self.registration[
            "residual_logical_content_sha256"
        ]:
            raise ValueError("correlated residual sampler content hash mismatch")
        self._rng = np.random.default_rng(
            int(self.registration["config"]["residual_sampling_seed"])
        )

    def reset(self) -> None:
        self._rng = np.random.default_rng(
            int(self.registration["config"]["residual_sampling_seed"])
        )

    def clone(self) -> "CorrelatedResidualSampler":
        return CorrelatedResidualSampler(
            self.residuals,
            self.masks,
            inflation_factor=self.inflation_factor,
            registration=self.registration,
        )

    def sample(
        self,
        current_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        if current_mask.ndim != 2 or current_mask.dtype != torch.bool:
            raise ValueError("current residual-sampling mask must be boolean [B,P]")
        if current_mask.shape[1] != self.residuals.shape[1]:
            raise ValueError("residual/current particle dimensions differ")
        indices = self._rng.integers(
            0, self.residuals.shape[0], size=current_mask.shape[0]
        )
        sampled = torch.as_tensor(
            self.residuals[indices],
            device=current_mask.device,
            dtype=torch.float32,
        )
        sampled_mask = torch.as_tensor(
            self.masks[indices], device=current_mask.device
        )
        valid = current_mask & sampled_mask
        sampled = sampled * self.inflation_factor
        sampled = torch.where(
            valid[:, :, None], sampled, torch.zeros_like(sampled)
        )
        return sampled, indices


def fit_correlated_residual_sampler(
    *,
    train_true_view: np.ndarray | torch.Tensor,
    train_prediction: np.ndarray | torch.Tensor,
    train_mask: np.ndarray | torch.Tensor,
    model_val_stop_true_view: np.ndarray | torch.Tensor,
    model_val_stop_prediction: np.ndarray | torch.Tensor,
    model_val_stop_mask: np.ndarray | torch.Tensor,
    train_identity_sha256: str,
    model_val_stop_split_sha256: str,
    coordinate_binding_sha256: str,
    pview0_checkpoint_sha256: str,
    snapshot_sha256: Sequence[str],
    config: CorrelatedResidualSamplerConfig | None = None,
) -> tuple[CorrelatedResidualSampler, dict[str, Any] | None]:
    config = config or CorrelatedResidualSamplerConfig()
    for name, value in (
        ("train_identity_sha256", train_identity_sha256),
        ("model_val_stop_split_sha256", model_val_stop_split_sha256),
        ("coordinate_binding_sha256", coordinate_binding_sha256),
        ("pview0_checkpoint_sha256", pview0_checkpoint_sha256),
    ):
        require_sha256(name, value)
    if len(snapshot_sha256) != 3:
        raise ValueError("residual sampler requires final three snapshots")
    for index, value in enumerate(snapshot_sha256):
        require_sha256(f"snapshot_sha256[{index}]", value)
    train_true = _as_numpy(train_true_view, dtype=np.float32)
    train_pred = _as_numpy(train_prediction, dtype=np.float32)
    train_valid = _as_numpy(train_mask, dtype=bool)
    stop_true = _as_numpy(model_val_stop_true_view, dtype=np.float32)
    stop_pred = _as_numpy(model_val_stop_prediction, dtype=np.float32)
    stop_valid = _as_numpy(model_val_stop_mask, dtype=bool)
    if (
        train_true.shape != train_pred.shape
        or train_valid.shape != train_true.shape[:2]
        or stop_true.shape != stop_pred.shape
        or stop_valid.shape != stop_true.shape[:2]
        or train_true.shape[1:] != stop_true.shape[1:]
    ):
        raise ValueError("residual sampler source shapes differ")
    train_residual = np.clip(
        train_true - train_pred, -config.residual_clip, config.residual_clip
    )
    stop_residual = np.clip(
        stop_true - stop_pred, -config.residual_clip, config.residual_clip
    )
    train_residual = np.where(train_valid[:, :, None], train_residual, 0.0)
    stop_residual = np.where(stop_valid[:, :, None], stop_residual, 0.0)
    if not np.isfinite(train_residual).all() or not np.isfinite(stop_residual).all():
        raise ValueError("residual sampler sources contain non-finite values")
    train_diagnostics = _residual_diagnostics(train_residual, train_valid)
    stop_diagnostics = _residual_diagnostics(stop_residual, stop_valid)
    ratio_values = {}
    for quantile in config.inflation_quantiles:
        train_value = train_diagnostics["event_residual_norm_quantiles"][
            str(quantile)
        ]
        stop_value = stop_diagnostics["event_residual_norm_quantiles"][
            str(quantile)
        ]
        ratio = (
            stop_value / train_value
            if train_value > 0
            else (1.0 if stop_value == 0 else float("inf"))
        )
        ratio_values[str(quantile)] = ratio
    raw_factor = max(ratio_values.values())
    ratios = {
        name: value if np.isfinite(value) else "infinity"
        for name, value in ratio_values.items()
    }
    inflation = float(np.clip(raw_factor, 1.0, 2.0))
    residuals = np.ascontiguousarray(train_residual, dtype="<f4")
    masks = np.ascontiguousarray(train_valid, dtype=np.bool_)
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_RESIDUAL_SAMPLER_CONTRACT,
            "config": config.to_payload(),
            "config_sha256": canonical_sha256(config.to_payload()),
            "train_identity_sha256": train_identity_sha256,
            "model_val_stop_split_sha256": model_val_stop_split_sha256,
            "coordinate_binding_sha256": coordinate_binding_sha256,
            "pview0_checkpoint_sha256": pview0_checkpoint_sha256,
            "snapshot_sha256": list(snapshot_sha256),
            "residual_logical_content_sha256": _array_hash(
                residuals, masks
            ),
            "residual_shape": list(residuals.shape),
            "residual_dtype": "float32_little_endian",
            "residual_mask_schema": config.residual_mask_schema,
            "train_diagnostics": train_diagnostics,
            "model_val_stop_diagnostics": stop_diagnostics,
            "heldout_train_tail_ratios": ratios,
            "raw_inflation_factor": (
                raw_factor if np.isfinite(raw_factor) else "infinity"
            ),
            "inflation_factor": inflation,
            "inflation_saturated_at_two": raw_factor > 2.0,
            "residual_events_persisted_to_disk": False,
            "ram_resident_training_resource": True,
        }
    )
    warning = None
    if raw_factor > 2.0:
        warning = with_content_hash(
            {
                "contract": PARTICLE_VIEW_PREDICTOR_OVERFIT_WARNING_CONTRACT,
                "warning_code": "WARN_PVIEW_HELDOUT_TAIL_RATIO_ABOVE_2",
                "observed_raw_inflation_factor": (
                    raw_factor if np.isfinite(raw_factor) else "infinity"
                ),
                "applied_inflation_factor": 2.0,
                "supporting_sampler_sha256": registration["content_hash"],
                "interpretation": (
                    "model_val_stop residual tails exceed train tails by "
                    "more than the permitted robust inflation."
                ),
                "stops_execution": False,
                "affects_selectability": False,
            }
        )
    return (
        CorrelatedResidualSampler(
            residuals,
            masks,
            inflation_factor=inflation,
            registration=registration,
        ),
        warning,
    )


def publish_correlated_residual_sampler(
    output_dir: str | Path,
    *,
    sampler: CorrelatedResidualSampler,
    warning: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Publish metadata only; the large residual bank remains RAM-resident."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(
        root / "correlated_residual_sampler.json", sampler.registration
    )
    if warning is not None:
        validate_content_hash(
            warning,
            expected_contract=PARTICLE_VIEW_PREDICTOR_OVERFIT_WARNING_CONTRACT,
        )
        if warning["supporting_sampler_sha256"] != sampler.registration[
            "content_hash"
        ]:
            raise ValueError("residual sampler warning lineage mismatch")
        write_immutable_json(
            root / "predictor_tail_warning.json", warning
        )
    report = with_content_hash(
        {
            "contract": "particle_view_residual_sampler_publication_v1",
            "sampler_registration_sha256": sampler.registration[
                "content_hash"
            ],
            "warning_sha256": (
                warning["content_hash"] if warning is not None else None
            ),
            "residual_events_persisted_to_disk": False,
            "ram_resident_training_resource": True,
        }
    )
    write_immutable_json(root / "residual_sampler_publication.json", report)
    return report


class SnapshotDropoutPredictionBank:
    """Frozen final-three snapshots plus deterministic MC-dropout prediction."""

    variant_names = tuple(PREDICTION_VARIANT_WEIGHTS)

    def __init__(
        self,
        snapshot_models: Sequence[HierarchicalParticleViewPredictor],
        *,
        snapshot_sha256: Sequence[str],
        mc_dropout_seed: int = 610_337,
        pview0_registration_sha256: str | None = None,
        pview0_lineage: Mapping[str, str] | None = None,
    ) -> None:
        if len(snapshot_models) != 3 or len(snapshot_sha256) != 3:
            raise ValueError("prediction bank requires final three snapshots")
        self.snapshot_models = tuple(snapshot_models)
        self.snapshot_sha256 = tuple(snapshot_sha256)
        for index, (model, digest) in enumerate(
            zip(self.snapshot_models, self.snapshot_sha256)
        ):
            require_sha256(f"snapshot_sha256[{index}]", digest)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("prediction-bank snapshots must be frozen")
            model.eval()
        if len(
            {
                model.config.content_hash
                for model in self.snapshot_models
            }
        ) != 1:
            raise ValueError("prediction-bank architecture configs differ")
        if self.snapshot_models[-1].config.dropout != 0.05:
            raise ValueError("MC-dropout bank requires canonical dropout 0.05")
        dropout_modules = [
            module
            for module in self.snapshot_models[-1].modules()
            if isinstance(module, nn.Dropout)
        ]
        if not dropout_modules or any(
            float(module.p) != 0.05 for module in dropout_modules
        ):
            raise ValueError(
                "MC-dropout bank requires active predictor dropout modules at 0.05"
            )
        self.mc_dropout_seed = int(mc_dropout_seed)
        if self.mc_dropout_seed != 610_337:
            raise ValueError("MC-dropout seed changed")
        self._dropout_calls = 0
        if pview0_registration_sha256 is not None:
            require_sha256(
                "pview0_registration_sha256",
                pview0_registration_sha256,
            )
        self.pview0_registration_sha256 = pview0_registration_sha256
        self.pview0_lineage = (
            dict(pview0_lineage) if pview0_lineage is not None else None
        )

    @property
    def architecture_config_sha256(self) -> str:
        return self.snapshot_models[-1].config.content_hash

    @property
    def dropout_probability(self) -> float:
        return float(self.snapshot_models[-1].config.dropout)

    def reset(self) -> None:
        self._dropout_calls = 0

    def to(self, device: str | torch.device) -> "SnapshotDropoutPredictionBank":
        for model in self.snapshot_models:
            model.to(device)
            model.eval()
        return self

    def clone(self) -> "SnapshotDropoutPredictionBank":
        return SnapshotDropoutPredictionBank(
            self.snapshot_models,
            snapshot_sha256=self.snapshot_sha256,
            mc_dropout_seed=self.mc_dropout_seed,
            pview0_registration_sha256=self.pview0_registration_sha256,
            pview0_lineage=self.pview0_lineage,
        )

    def predict(
        self,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rows = {}
        with torch.no_grad():
            for index, model in enumerate(self.snapshot_models):
                model.eval()
                rows[f"snapshot_epoch_{index + 2}"] = model(
                    features, lorentz_vectors, mask
                ).mean
            final = self.snapshot_models[-1]
            final.eval()
            dropout_modules = [
                module for module in final.modules() if isinstance(module, nn.Dropout)
            ]
            devices = (
                [features.device.index or 0]
                if features.is_cuda
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(
                    self.mc_dropout_seed + self._dropout_calls
                )
                if features.is_cuda:
                    torch.cuda.manual_seed_all(
                        self.mc_dropout_seed + self._dropout_calls
                    )
                for module in dropout_modules:
                    module.train()
                rows["mc_dropout_epoch_4"] = final(
                    features, lorentz_vectors, mask
                ).mean
            final.eval()
        self._dropout_calls += 1
        if tuple(rows) != self.variant_names:
            raise RuntimeError("prediction-bank variant order changed")
        valid = mask[:, 0] if mask.ndim == 3 else mask
        for name, values in rows.items():
            if not torch.isfinite(values).all():
                raise FloatingPointError(
                    f"prediction-bank variant {name} is non-finite"
                )
            if (~valid).any() and values[~valid].abs().max().item() != 0:
                raise ValueError(
                    f"prediction-bank variant {name} is nonzero on padding"
                )
        return rows


def load_snapshot_dropout_prediction_bank(
    template_model: HierarchicalParticleViewPredictor,
    *,
    pview0_registration_path: str | Path,
    expected_lineage: Mapping[str, str] | None = None,
    device: str | torch.device = "cpu",
) -> SnapshotDropoutPredictionBank:
    path = Path(pview0_registration_path)
    registration = json.loads(path.read_text(encoding="utf-8"))
    validate_content_hash(
        registration,
        expected_contract=PARTICLE_VIEW_PVIEW0_REGISTRATION_CONTRACT,
    )
    validated = validate_pview0_registration(
        registration,
        root=path.parent,
        expected_lineage=expected_lineage,
        expected_architecture_config_sha256=template_model.config.content_hash,
    )
    models = []
    for expected_epoch, snapshot_path in zip(
        (2, 3, 4), validated["snapshot_paths"]
    ):
        checkpoint = torch.load(
            snapshot_path, map_location=device, weights_only=False
        )
        if (
            checkpoint.get("contract")
            != PARTICLE_VIEW_PVIEW0_CHECKPOINT_CONTRACT
            or checkpoint.get("architecture_config_sha256")
            != template_model.config.content_hash
            or checkpoint.get("lineage") != registration["lineage"]
            or checkpoint.get("epoch") != expected_epoch
            or checkpoint.get("warmup_config_sha256")
            != registration["warmup_config_sha256"]
        ):
            raise ValueError("Pview_0 snapshot payload mismatch")
        model = deepcopy(template_model).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return SnapshotDropoutPredictionBank(
        models,
        snapshot_sha256=registration["snapshot_sha256"],
        pview0_registration_sha256=registration["content_hash"],
        pview0_lineage=registration["lineage"],
    )


@dataclass(frozen=True)
class RobustViewMixtureConfig:
    mixture_seed: int = 610_411
    mixture_cycle_length: int = 20
    prediction_variant_cycle_length: int = 4
    view_clip: float = 6.0
    contract: str = PARTICLE_VIEW_ROBUST_MIXTURE_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.mixture_seed != 610_411
            or self.mixture_cycle_length != 20
            or self.prediction_variant_cycle_length != 4
            or self.view_clip != 6.0
        ):
            raise ValueError("robust view mixture contract changed")

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "source_probabilities": ROBUST_VIEW_PROBABILITIES,
            "prediction_variant_weights": PREDICTION_VARIANT_WEIGHTS,
            "selection_unit": "whole_batch",
            "source_schedule_counts_per_cycle": {
                "true": 6,
                "predicted": 9,
                "perturbed": 3,
                "zero": 2,
            },
            "prediction_variants_equal_weight": True,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass
class RobustViewBatch:
    view: torch.Tensor
    source: str
    prediction_variant: str | None
    residual_indices: tuple[int, ...]


class DeterministicRobustViewMixer:
    def __init__(
        self,
        sampler: CorrelatedResidualSampler,
        config: RobustViewMixtureConfig | None = None,
    ) -> None:
        self.sampler = sampler
        self.config = config or RobustViewMixtureConfig()
        rng = np.random.default_rng(self.config.mixture_seed)
        sources = (
            ["true"] * 6
            + ["predicted"] * 9
            + ["perturbed"] * 3
            + ["zero"] * 2
        )
        self._sources = tuple(np.asarray(sources)[rng.permutation(20)].tolist())
        variants = np.asarray(tuple(PREDICTION_VARIANT_WEIGHTS))
        self._variants = tuple(variants[rng.permutation(4)].tolist())
        self.reset()

    def reset(self) -> None:
        self._source_index = 0
        self._variant_index = 0
        self.sampler.reset()

    def next(
        self,
        *,
        true_view: torch.Tensor,
        predictions: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> RobustViewBatch:
        if tuple(predictions) != tuple(PREDICTION_VARIANT_WEIGHTS):
            raise ValueError("robust prediction variant inventory changed")
        if mask.ndim == 3:
            mask = mask[:, 0]
        if mask.dtype != torch.bool or mask.shape != true_view.shape[:2]:
            raise ValueError("robust mixture view/mask shapes differ")
        source = self._sources[self._source_index % len(self._sources)]
        self._source_index += 1
        variant = None
        indices: tuple[int, ...] = ()
        if source == "true":
            mixed = true_view
        elif source == "zero":
            mixed = torch.zeros_like(true_view)
        else:
            variant = self._variants[
                self._variant_index % len(self._variants)
            ]
            self._variant_index += 1
            mixed = predictions[variant]
            if source == "perturbed":
                residual, sampled = self.sampler.sample(mask)
                mixed = mixed + residual.to(dtype=mixed.dtype)
                indices = tuple(int(index) for index in sampled)
        mixed = mixed.clamp(-self.config.view_clip, self.config.view_clip)
        mixed = torch.where(
            mask[:, :, None], mixed, torch.zeros_like(mixed)
        )
        return RobustViewBatch(
            view=mixed,
            source=source,
            prediction_variant=variant,
            residual_indices=indices,
        )


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(str(tuple(values.shape)).encode())
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(model: nn.Module) -> str:
    return _state_dict_sha256(model.state_dict())


@dataclass(frozen=True)
class RobustConsumerTrainConfig:
    maximum_epochs: int = 40
    early_stop_patience: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 128
    gradient_clip: float = 1.0
    seed: int = 101

    def __post_init__(self) -> None:
        if (
            self.maximum_epochs != 40
            or self.early_stop_patience != 8
            or self.learning_rate != 3.0e-4
            or self.weight_decay != 1.0e-4
            or self.batch_size != 128
            or self.gradient_clip != 1.0
            or self.seed not in {101, 202, 303}
        ):
            raise ValueError("robust consumer training recipe changed")

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "optimizer": "AdamW",
            "objective": {"cross_entropy": 1.0, "trust": 0.01},
            "checkpoint_selection": [
                "highest_weighted_endpoint_accuracy",
                "lowest_weighted_endpoint_cross_entropy",
                "earliest_epoch",
            ],
            "checkpoint_selection_split": "model_val_stop",
        }


def _consumer_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required = {
        "points",
        "features",
        "lorentz_vectors",
        "mask",
        "labels",
        "true_view",
    }
    if not required.issubset(raw):
        raise ValueError("robust-consumer batch field inventory is incomplete")
    forbidden = {"offline_tokens", "oracle_logits", "teacher_logits"}
    if forbidden.intersection(raw):
        raise ValueError("robust-consumer batch exposed offline descendants")
    batch = {
        name: raw[name].to(device=device, non_blocking=True)
        for name in required
    }
    valid = batch["mask"][:, 0] if batch["mask"].ndim == 3 else batch["mask"]
    if (
        valid.dtype != torch.bool
        or valid.shape != batch["true_view"].shape[:2]
        or (~valid.any(dim=1)).any()
        or not torch.isfinite(batch["true_view"]).all()
        or batch["true_view"][valid].abs().max().item() > 6.0
    ):
        raise ValueError("robust-consumer canonical true-view contract failed")
    if (~valid).any() and batch["true_view"][~valid].abs().max().item() != 0:
        raise ValueError("robust-consumer true view is nonzero on padding")
    return batch


def _consumer_logits(
    model: ParticleViewConsumer,
    batch: Mapping[str, torch.Tensor],
    view: torch.Tensor,
):
    return model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        view,
        augment_clean_view=False,
    )


def evaluate_robust_consumer(
    model: ParticleViewConsumer,
    loader,
    *,
    prediction_bank: SnapshotDropoutPredictionBank,
    sampler: CorrelatedResidualSampler,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    prediction_bank.reset()
    sampler.reset()
    names = ("true", "predicted", "perturbed", "zero")
    correct = {name: 0.0 for name in names}
    ce = {name: 0.0 for name in names}
    total = 0
    with torch.no_grad():
        for raw in loader:
            batch = _consumer_batch(raw, device)
            valid = batch["mask"][:, 0]
            variants = prediction_bank.predict(
                batch["features"],
                batch["lorentz_vectors"],
                batch["mask"],
            )
            residual, _ = sampler.sample(valid)
            views = {
                "true": (batch["true_view"],),
                "predicted": tuple(variants.values()),
                "perturbed": tuple(
                    value + residual.to(value.dtype)
                    for value in variants.values()
                ),
                "zero": (torch.zeros_like(batch["true_view"]),),
            }
            count = int(batch["labels"].numel())
            for name, candidates in views.items():
                candidate_correct = 0.0
                candidate_ce = 0.0
                for view in candidates:
                    view = torch.where(
                        valid[:, :, None],
                        view.clamp(-6.0, 6.0),
                        torch.zeros_like(view),
                    )
                    logits = _consumer_logits(model, batch, view).logits
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            "robust-consumer validation logits are non-finite"
                        )
                    candidate_correct += float(
                        logits.argmax(dim=1).eq(batch["labels"]).sum().item()
                    )
                    candidate_ce += float(
                        torch.nn.functional.cross_entropy(
                            logits.float(),
                            batch["labels"],
                            reduction="sum",
                        ).item()
                    )
                correct[name] += candidate_correct / len(candidates)
                ce[name] += candidate_ce / len(candidates)
            total += count
    if total == 0:
        raise ValueError("robust-consumer validation loader is empty")
    endpoints = {
        name: {
            "accuracy": correct[name] / total,
            "cross_entropy": ce[name] / total,
        }
        for name in names
    }
    weighted_accuracy = sum(
        ROBUST_VIEW_PROBABILITIES[name] * endpoints[name]["accuracy"]
        for name in names
    )
    weighted_ce = sum(
        ROBUST_VIEW_PROBABILITIES[name] * endpoints[name]["cross_entropy"]
        for name in names
    )
    return {
        "weighted_accuracy": weighted_accuracy,
        "weighted_cross_entropy": weighted_ce,
        "endpoints": endpoints,
        "examples": total,
        "snapshot_dropout_variants_equal_weight": True,
    }


def train_robust_consumer(
    *,
    clean_consumer: ParticleViewConsumer,
    train_loader,
    model_val_stop_loader,
    prediction_bank: SnapshotDropoutPredictionBank,
    sampler: CorrelatedResidualSampler,
    output_dir: str | Path,
    lineage: Mapping[str, str],
    clean_consumer_checkpoint_path: str | Path,
    mixture_config: RobustViewMixtureConfig | None = None,
    train_config: RobustConsumerTrainConfig | None = None,
    device: str | torch.device = "cpu",
) -> tuple[ParticleViewConsumer, dict[str, Any]]:
    if set(lineage) != set(ROBUST_CONSUMER_LINEAGE_FIELDS):
        raise ValueError("robust-consumer lineage field inventory mismatch")
    lineage = dict(lineage)
    for name, value in lineage.items():
        require_sha256(name, value)
    if (
        lineage["residual_sampler_registration_sha256"]
        != sampler.registration["content_hash"]
    ):
        raise ValueError("robust-consumer sampler lineage mismatch")
    if (
        lineage["pview0_checkpoint_sha256"]
        != prediction_bank.snapshot_sha256[-1]
    ):
        raise ValueError("robust-consumer final Pview_0 snapshot mismatch")
    if (
        prediction_bank.pview0_registration_sha256
        != lineage["pview0_registration_sha256"]
        or prediction_bank.pview0_lineage is None
    ):
        raise ValueError("robust-consumer Pview_0 registration mismatch")
    for name in (
        "clean_consumer_registration_sha256",
        "clean_consumer_checkpoint_sha256",
        "coordinate_binding_sha256",
        "selected_view_publication_sha256",
        "train_identity_sha256",
    ):
        if prediction_bank.pview0_lineage.get(name) != lineage[name]:
            raise ValueError(
                f"robust-consumer Pview_0 ancestor {name} mismatch"
            )
    clean_checkpoint_path = Path(clean_consumer_checkpoint_path)
    if (
        sha256_file(clean_checkpoint_path)
        != lineage["clean_consumer_checkpoint_sha256"]
    ):
        raise ValueError("robust-consumer clean checkpoint file hash mismatch")
    mixture_config = mixture_config or RobustViewMixtureConfig()
    train_config = train_config or RobustConsumerTrainConfig()
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if (
        loader_batch_size is not None
        and int(loader_batch_size) != train_config.batch_size
    ):
        raise ValueError("robust-consumer train loader batch size differs")
    torch.manual_seed(train_config.seed)
    device = torch.device(device)
    clean_consumer = clean_consumer.to(device)
    clean_consumer.eval()
    initialization_sha = _state_sha256(clean_consumer)
    clean_checkpoint = torch.load(
        clean_checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(clean_checkpoint, Mapping)
        or "model_state_dict" not in clean_checkpoint
        or _state_dict_sha256(clean_checkpoint["model_state_dict"])
        != initialization_sha
    ):
        raise ValueError(
            "robust-consumer clean model state differs from its checkpoint"
        )
    robust = deepcopy(clean_consumer).to(device)
    if _state_sha256(robust) != initialization_sha:
        raise RuntimeError("robust consumer did not exactly copy Cview_clean")
    for parameter in robust.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        robust.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    training_bank = prediction_bank.clone().to(device)
    training_sampler = sampler.clone()
    mixer = DeterministicRobustViewMixer(training_sampler, mixture_config)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "best_model_val_stop.pt"
    rows = []
    best_key = None
    stale = 0
    updates = 0
    source_counts = {name: 0 for name in ROBUST_VIEW_PROBABILITIES}
    variant_counts = {name: 0 for name in PREDICTION_VARIANT_WEIGHTS}
    for epoch in range(1, train_config.maximum_epochs + 1):
        robust.train()
        for raw in train_loader:
            batch = _consumer_batch(raw, device)
            with torch.no_grad():
                predictions = training_bank.predict(
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                )
                mixed = mixer.next(
                    true_view=batch["true_view"],
                    predictions=predictions,
                    mask=batch["mask"],
                )
            source_counts[mixed.source] += 1
            if mixed.prediction_variant is not None:
                variant_counts[mixed.prediction_variant] += 1
            optimizer.zero_grad(set_to_none=True)
            output = _consumer_logits(robust, batch, mixed.view)
            loss = torch.nn.functional.cross_entropy(
                output.logits, batch["labels"]
            ) + 0.01 * output.trust_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("robust-consumer loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                robust.parameters(), train_config.gradient_clip
            )
            optimizer.step()
            updates += 1
        validation = evaluate_robust_consumer(
            robust,
            model_val_stop_loader,
            prediction_bank=prediction_bank.clone().to(device),
            sampler=sampler.clone(),
            device=device,
        )
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "model_val_stop": validation,
        }
        rows.append(row)
        key = (
            -float(validation["weighted_accuracy"]),
            float(validation["weighted_cross_entropy"]),
            epoch,
        )
        if best_key is None or key < best_key:
            best_key = key
            stale = 0
            torch.save(
                {
                    "contract": PARTICLE_VIEW_ROBUST_CHECKPOINT_CONTRACT,
                    "role": "Cview_robust",
                    "model_state_dict": robust.state_dict(),
                    "consumer_config": robust.config.to_payload(),
                    "consumer_config_sha256": robust.config.content_hash,
                    "lineage": lineage,
                    "mixture_config": mixture_config.to_payload(),
                    "mixture_config_sha256": mixture_config.content_hash,
                    "train_config": train_config.to_payload(),
                    "train_config_sha256": canonical_sha256(
                        train_config.to_payload()
                    ),
                    "clean_consumer_initialization_sha256": initialization_sha,
                    "epoch": epoch,
                    "optimizer_updates": updates,
                    "model_val_stop": validation,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= train_config.early_stop_patience:
            break
    selected = min(
        rows,
        key=lambda row: (
            -row["model_val_stop"]["weighted_accuracy"],
            row["model_val_stop"]["weighted_cross_entropy"],
            row["epoch"],
        ),
    )
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_ROBUST_CONSUMER_REGISTRATION_CONTRACT,
            "role": "Cview_robust",
            "consumer_config": robust.config.to_payload(),
            "consumer_config_sha256": robust.config.content_hash,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "lineage": lineage,
            "clean_consumer_initialization_sha256": initialization_sha,
            "pview0_architecture_config_sha256": (
                prediction_bank.architecture_config_sha256
            ),
            "snapshot_sha256": list(prediction_bank.snapshot_sha256),
            "mc_dropout_mode": "dropout_modules_train_all_other_modules_eval",
            "mc_dropout_probability": prediction_bank.dropout_probability,
            "mc_dropout_seed": prediction_bank.mc_dropout_seed,
            "prediction_variant_weights": PREDICTION_VARIANT_WEIGHTS,
            "residual_sampler_registration": sampler.registration,
            "residual_sampler_registration_sha256": sampler.registration[
                "content_hash"
            ],
            "mixture_config": mixture_config.to_payload(),
            "mixture_config_sha256": mixture_config.content_hash,
            "train_config": train_config.to_payload(),
            "train_config_sha256": canonical_sha256(
                train_config.to_payload()
            ),
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "optimizer_updates": updates,
            "model_val_stop": selected["model_val_stop"],
            "source_counts": source_counts,
            "prediction_variant_counts": variant_counts,
            "pview0_frozen": True,
            "gview_frozen": True,
            "offline_teacher_frozen": True,
            "a0_frozen_as_external_ancestor": True,
            "frozen_after_registration": True,
            "model_val_select_loaded": False,
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(
        root / "robust_consumer_training_curves.json",
        with_content_hash(
            {
                "contract": "particle_view_robust_consumer_curves_v1",
                "lineage": lineage,
                "mixture_config_sha256": mixture_config.content_hash,
                "train_config_sha256": canonical_sha256(
                    train_config.to_payload()
                ),
                "epochs": rows,
            }
        ),
    )
    write_immutable_json(
        root / "robust_consumer_registration.json", registration
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    robust.load_state_dict(checkpoint["model_state_dict"], strict=True)
    robust.eval()
    for parameter in robust.parameters():
        parameter.requires_grad_(False)
    return robust, registration


def load_registered_robust_consumer(
    model: ParticleViewConsumer,
    *,
    registration_path: str | Path,
    expected_lineage: Mapping[str, str],
    expected_snapshot_sha256: Sequence[str],
    expected_sampler_sha256: str,
    expected_pview_architecture_config_sha256: str,
) -> ParticleViewConsumer:
    path = Path(registration_path)
    registration = json.loads(path.read_text(encoding="utf-8"))
    validate_content_hash(
        registration,
        expected_contract=PARTICLE_VIEW_ROBUST_CONSUMER_REGISTRATION_CONTRACT,
    )
    if set(expected_lineage) != set(ROBUST_CONSUMER_LINEAGE_FIELDS):
        raise ValueError("expected robust-consumer lineage inventory differs")
    for name, value in expected_lineage.items():
        require_sha256(name, value)
    if registration["lineage"] != dict(expected_lineage):
        raise ValueError("robust-consumer lineage mismatch")
    if list(expected_snapshot_sha256) != registration["snapshot_sha256"]:
        raise ValueError("robust-consumer snapshot lineage mismatch")
    if (
        registration["residual_sampler_registration_sha256"]
        != expected_sampler_sha256
    ):
        raise ValueError("robust-consumer sampler lineage mismatch")
    if registration["consumer_config_sha256"] != model.config.content_hash:
        raise ValueError("robust-consumer architecture config mismatch")
    require_sha256(
        "expected_pview_architecture_config_sha256",
        expected_pview_architecture_config_sha256,
    )
    if (
        registration["pview0_architecture_config_sha256"]
        != expected_pview_architecture_config_sha256
    ):
        raise ValueError("robust-consumer Pview_0 architecture mismatch")
    if registration["mixture_config"] != RobustViewMixtureConfig().to_payload():
        raise ValueError("robust-consumer mixture contract mismatch")
    if registration["train_config"] != RobustConsumerTrainConfig().to_payload():
        raise ValueError("robust-consumer optimizer/schedule contract mismatch")
    if registration["prediction_variant_weights"] != PREDICTION_VARIANT_WEIGHTS:
        raise ValueError("robust-consumer prediction variant weights mismatch")
    validate_content_hash(
        registration["residual_sampler_registration"],
        expected_contract=PARTICLE_VIEW_RESIDUAL_SAMPLER_CONTRACT,
    )
    if (
        registration["residual_sampler_registration"]["config"]
        != CorrelatedResidualSamplerConfig().to_payload()
    ):
        raise ValueError("robust-consumer residual sampler config mismatch")
    if (
        registration["residual_sampler_registration"]["content_hash"]
        != expected_sampler_sha256
    ):
        raise ValueError("nested robust-consumer sampler mismatch")
    if (
        registration["mc_dropout_mode"]
        != "dropout_modules_train_all_other_modules_eval"
        or registration["mc_dropout_probability"] != 0.05
        or registration["mc_dropout_seed"] != 610_337
    ):
        raise ValueError("robust-consumer MC-dropout contract mismatch")
    checkpoint_path = path.parent / registration["checkpoint_file"]
    if sha256_file(checkpoint_path) != registration["checkpoint_sha256"]:
        raise ValueError("robust-consumer checkpoint hash mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected_checkpoint = {
        "lineage": registration["lineage"],
        "mixture_config_sha256": registration["mixture_config_sha256"],
        "train_config_sha256": registration["train_config_sha256"],
        "clean_consumer_initialization_sha256": registration[
            "clean_consumer_initialization_sha256"
        ],
        "consumer_config_sha256": registration["consumer_config_sha256"],
    }
    for name, value in expected_checkpoint.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"robust-consumer checkpoint {name} mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_paired_consumer_metrics(
    *,
    clean_metrics: Mapping[str, Any],
    robust_metrics: Mapping[str, Any],
    clean_consumer_checkpoint_sha256: str,
    robust_consumer_checkpoint_sha256: str,
    pview0_checkpoint_sha256: str,
    coordinate_binding_sha256: str,
    split_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("clean_consumer_checkpoint_sha256", clean_consumer_checkpoint_sha256),
        ("robust_consumer_checkpoint_sha256", robust_consumer_checkpoint_sha256),
        ("pview0_checkpoint_sha256", pview0_checkpoint_sha256),
        ("coordinate_binding_sha256", coordinate_binding_sha256),
        ("split_sha256", split_sha256),
    ):
        require_sha256(name, value)
    for name, metrics in (("clean", clean_metrics), ("robust", robust_metrics)):
        validate_content_hash(
            metrics,
            expected_contract="particle_view_counterfactual_metrics_v1",
        )
        if metrics.get("split") != "model_val_select":
            raise ValueError(f"{name} paired metrics use the wrong split")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_PAIRED_CONSUMER_METRICS_CONTRACT,
            "split": "model_val_select",
            "split_sha256": split_sha256,
            "clean_consumer_checkpoint_sha256": clean_consumer_checkpoint_sha256,
            "robust_consumer_checkpoint_sha256": robust_consumer_checkpoint_sha256,
            "pview0_checkpoint_sha256": pview0_checkpoint_sha256,
            "coordinate_binding_sha256": coordinate_binding_sha256,
            "clean": dict(clean_metrics),
            "robust": dict(robust_metrics),
            "robust_minus_clean": {
                endpoint: (
                    robust_metrics[endpoint]["accuracy"]
                    - clean_metrics[endpoint]["accuracy"]
                )
                for endpoint in ("zero_view", "true_view", "predicted_view")
            },
            "paired_consumer_requirement_satisfied": True,
        }
    )


__all__ = [
    "PARTICLE_VIEW_PAIRED_CONSUMER_METRICS_CONTRACT",
    "PARTICLE_VIEW_PREDICTOR_OVERFIT_WARNING_CONTRACT",
    "PARTICLE_VIEW_RESIDUAL_SAMPLER_CONTRACT",
    "PARTICLE_VIEW_ROBUST_CHECKPOINT_CONTRACT",
    "PARTICLE_VIEW_ROBUST_CONSUMER_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_ROBUST_MIXTURE_CONTRACT",
    "PREDICTION_VARIANT_WEIGHTS",
    "ROBUST_CONSUMER_LINEAGE_FIELDS",
    "ROBUST_VIEW_PROBABILITIES",
    "CorrelatedResidualSampler",
    "CorrelatedResidualSamplerConfig",
    "DeterministicRobustViewMixer",
    "RobustConsumerTrainConfig",
    "RobustViewBatch",
    "RobustViewMixtureConfig",
    "SnapshotDropoutPredictionBank",
    "build_paired_consumer_metrics",
    "evaluate_robust_consumer",
    "fit_correlated_residual_sampler",
    "load_registered_robust_consumer",
    "load_snapshot_dropout_prediction_bank",
    "publish_correlated_residual_sampler",
    "train_robust_consumer",
]
