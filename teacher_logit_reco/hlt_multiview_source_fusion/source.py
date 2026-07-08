"""Source-model training helpers for deployable HLT multiview fusion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from teacher_logit_reco.hlt_self_dualview.tri_view import (
    HLTTriViewSourceConfig,
    cache_hlt_triview_source_predictions,
    train_hlt_triview_source,
)
from teacher_logit_reco.privileged_distill_v3.config import PDV3_MODEL_SPLIT_SIZES

from .config import (
    HLTMVExperimentConfig,
    HLTMVExperimentLayout,
    HLT_MV_ALLOWED_INPUTS,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_DEPLOYMENT_INPUTS,
    HLT_MV_EXPERIMENT_NAME,
    build_hlt_mv_random_hlt_source_names,
    default_hlt_mv_experiment_config,
    default_hlt_mv_experiment_layout,
    normalize_hlt_mv_source_name,
    normalize_hlt_mv_strengths,
)
from teacher_logit_reco.hlt_self_dualview.config import normalize_hlt_sdv_strength


HLT_MV_SOURCE_EXPERIMENT_STEP = "hlt_mv_step2_source_training"
HLT_MV_SOURCE_CONTRACT = "hlt_multiview_source_part_v1"
HLT_MV_SOURCE_PREDICTION_EXPERIMENT_STEP = "hlt_mv_step3_source_prediction_caching"
HLT_MV_SOURCE_PREDICTION_CONTRACT = "hlt_multiview_source_prediction_cache_v1"
HLT_MV_SOURCE_PREDICTION_REPORT = "prediction_cache_report.json"
HLT_MV_SOURCE_PREDICTION_SPLITS: tuple[str, ...] = ("model_val", "final_test")

HLT_MV_SOURCE_DEFAULT_BATCH_SIZE = 128
HLT_MV_SOURCE_DEFAULT_EVAL_BATCH_SIZE = 128
HLT_MV_SOURCE_DEFAULT_EPOCHS = 10
HLT_MV_SOURCE_DEFAULT_LR = 3.0e-4
HLT_MV_SOURCE_DEFAULT_WEIGHT_DECAY = 1.0e-4
HLT_MV_SOURCE_DEFAULT_EARLY_STOP_PATIENCE = 3
HLT_MV_SOURCE_DEFAULT_GRAD_CLIP_NORM = 1.0
HLT_MV_SOURCE_DEFAULT_MODEL_SIZE = "base"
HLT_MV_SOURCE_DEFAULT_AMP = False
HLT_MV_SOURCE_DEFAULT_MAX_TRAIN_JETS = int(PDV3_MODEL_SPLIT_SIZES["model_train"])
HLT_MV_SOURCE_DEFAULT_MAX_VAL_JETS = int(PDV3_MODEL_SPLIT_SIZES["model_val"])
HLT_MV_SOURCE_DEFAULT_MAX_FINAL_TEST_JETS = int(PDV3_MODEL_SPLIT_SIZES["final_test"])

HLT_MV_SOURCE_VIEW_HLT = "fixed_hlt"
HLT_MV_SOURCE_VIEW_HLT2 = "hlt2"

_HLT_SOURCE_RE = re.compile(r"^hlt_part_seed(?P<seed>\d+)$")
_HLT2_SOURCE_RE = re.compile(r"^hlt2_part_s(?P<major>\d+)p(?P<minor>\d+)_seed(?P<seed>\d+)$")


def _config_for(config: HLTMVExperimentConfig | None) -> HLTMVExperimentConfig:
    return config or default_hlt_mv_experiment_config()


def _layout_for(
    *,
    layout: HLTMVExperimentLayout | None = None,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
) -> HLTMVExperimentLayout:
    return layout or default_hlt_mv_experiment_layout(
        output_root=output_root,
        pdv3_experiment_name=pdv3_experiment_name,
    )


def _require_known_source_name(name: str, config: HLTMVExperimentConfig | None = None) -> str:
    source_name = normalize_hlt_mv_source_name(name)
    cfg = _config_for(config)
    valid_names = set(cfg.source_model_names) | set(cfg.random_hlt_source_names)
    if source_name not in valid_names:
        hlt_match = _HLT_SOURCE_RE.match(source_name)
        hlt2_match = _HLT2_SOURCE_RE.match(source_name)
        if hlt_match:
            return source_name
        if hlt2_match:
            strength = hlt_mv_strength_from_source_name(source_name)
            if strength in cfg.hlt2_source_seeds:
                return source_name
        known = ", ".join(sorted(valid_names))
        raise ValueError(f"Unknown HLT-MV source model {source_name!r}; expected one of: {known}")
    return source_name


def hlt_mv_source_seed_from_name(name: str) -> int:
    """Return the integer seed encoded in an HLT-MV source model name."""

    source_name = normalize_hlt_mv_source_name(name)
    hlt_match = _HLT_SOURCE_RE.match(source_name)
    if hlt_match:
        return int(hlt_match.group("seed"))
    hlt2_match = _HLT2_SOURCE_RE.match(source_name)
    if hlt2_match:
        return int(hlt2_match.group("seed"))
    raise ValueError(f"Cannot parse seed from HLT-MV source name {source_name!r}.")


def hlt_mv_strength_from_source_name(name: str) -> float:
    """Return the HLT2 strength encoded in an HLT2 source name."""

    source_name = normalize_hlt_mv_source_name(name)
    match = _HLT2_SOURCE_RE.match(source_name)
    if not match:
        raise ValueError(f"HLT-MV source {source_name!r} is not an HLT2 source model.")
    strength = float(f"{int(match.group('major'))}.{match.group('minor')}")
    return normalize_hlt_sdv_strength(strength)


def hlt_mv_source_view_from_name(
    name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> str:
    """Map a source model name to the cache view consumed by its ParT branch."""

    source_name = _require_known_source_name(name, config=config)
    if _HLT_SOURCE_RE.match(source_name):
        return HLT_MV_SOURCE_VIEW_HLT
    if _HLT2_SOURCE_RE.match(source_name):
        return HLT_MV_SOURCE_VIEW_HLT2
    raise ValueError(f"Cannot infer source view from {source_name!r}.")


def hlt_mv_source_cache_dir(
    layout: HLTMVExperimentLayout,
    source_name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> Path:
    """Return the existing HLT/HLT2 cache directory for a source model."""

    view = hlt_mv_source_view_from_name(source_name, config=config)
    if view == HLT_MV_SOURCE_VIEW_HLT:
        return layout.hlt_cache_dir
    return layout.hlt2_cache_dir(hlt_mv_strength_from_source_name(source_name))


def hlt_mv_source_output_dir(
    layout: HLTMVExperimentLayout,
    source_name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> Path:
    """Return the canonical output directory for a source model."""

    source_name = _require_known_source_name(source_name, config=config)
    cfg = _config_for(config)
    if source_name in set(cfg.random_hlt_source_names):
        return layout.random_hlt_source_dir(source_name)
    return layout.source_model_dir(source_name)


def normalize_hlt_mv_source_prediction_splits(splits: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(str(split).strip() for split in splits)
    allowed = set(HLT_MV_SOURCE_PREDICTION_SPLITS)
    if not normalized:
        raise ValueError("HLT-MV source prediction split list cannot be empty.")
    if any(not split for split in normalized):
        raise ValueError(f"HLT-MV source prediction split names cannot be empty: {splits!r}")
    unknown = [split for split in normalized if split not in allowed]
    if unknown:
        raise ValueError(f"HLT-MV source prediction splits must be {HLT_MV_SOURCE_PREDICTION_SPLITS}, got {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"HLT-MV source prediction splits must be unique, got {normalized!r}.")
    return normalized


def hlt_mv_source_prediction_paths(
    layout: HLTMVExperimentLayout,
    source_name: str,
    split: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> tuple[Path, Path]:
    normalized_split = normalize_hlt_mv_source_prediction_splits((split,))[0]
    source_name = _require_known_source_name(source_name, config=config)
    root = hlt_mv_source_output_dir(layout, source_name, config=config) / "predictions" / source_name
    return root / f"{normalized_split}_predictions.npz", root / f"{normalized_split}_predictions_metadata.json"


def build_hlt_mv_source_config(
    *,
    source_name: str,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str | None = None,
    layout: HLTMVExperimentLayout | None = None,
    run_config: HLTMVExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    source_view: str | None = None,
    seed: int | None = None,
    batch_size: int = HLT_MV_SOURCE_DEFAULT_BATCH_SIZE,
    eval_batch_size: int = HLT_MV_SOURCE_DEFAULT_EVAL_BATCH_SIZE,
    epochs: int = HLT_MV_SOURCE_DEFAULT_EPOCHS,
    lr: float = HLT_MV_SOURCE_DEFAULT_LR,
    weight_decay: float = HLT_MV_SOURCE_DEFAULT_WEIGHT_DECAY,
    early_stop_patience: int = HLT_MV_SOURCE_DEFAULT_EARLY_STOP_PATIENCE,
    num_workers: int = 0,
    device: str = "auto",
    amp: bool = HLT_MV_SOURCE_DEFAULT_AMP,
    grad_clip_norm: float = HLT_MV_SOURCE_DEFAULT_GRAD_CLIP_NORM,
    model_size: str = HLT_MV_SOURCE_DEFAULT_MODEL_SIZE,
    compile_model: bool = False,
    evaluate_model_val_predictions: bool = True,
    evaluate_final_test: bool = True,
    confirm_final_test: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_final_test_batches: int | None = None,
    max_train_jets: int | None = HLT_MV_SOURCE_DEFAULT_MAX_TRAIN_JETS,
    max_val_jets: int | None = HLT_MV_SOURCE_DEFAULT_MAX_VAL_JETS,
    max_final_test_jets: int | None = HLT_MV_SOURCE_DEFAULT_MAX_FINAL_TEST_JETS,
    overwrite: bool = False,
) -> HLTTriViewSourceConfig:
    """Build a scratch single-view source config for the HLT-MV run grid."""

    cfg = _config_for(run_config)
    source_name = _require_known_source_name(source_name, config=cfg)
    lay = _layout_for(
        layout=layout,
        output_root=output_root,
        pdv3_experiment_name=cfg.pdv3_experiment_name if pdv3_experiment_name is None else pdv3_experiment_name,
    )
    inferred_view = hlt_mv_source_view_from_name(source_name, config=cfg)
    if source_view is not None and str(source_view) != inferred_view:
        raise ValueError(f"source_view for {source_name} must be {inferred_view!r}, got {source_view!r}.")
    expected_strengths = normalize_hlt_mv_strengths(cfg.strengths)
    if (
        inferred_view == HLT_MV_SOURCE_VIEW_HLT2
        and hlt_mv_strength_from_source_name(source_name) not in expected_strengths
        and cache_dir is None
    ):
        raise ValueError(f"HLT2 source {source_name!r} is not in configured strength grid {expected_strengths!r}.")
    return HLTTriViewSourceConfig(
        output_dir=str(Path(output_dir) if output_dir is not None else hlt_mv_source_output_dir(lay, source_name, config=cfg)),
        cache_dir=str(Path(cache_dir) if cache_dir is not None else hlt_mv_source_cache_dir(lay, source_name, config=cfg)),
        source_name=source_name,
        source_view=inferred_view,
        warm_start_checkpoint=None,
        seed=hlt_mv_source_seed_from_name(source_name) if seed is None else int(seed),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        early_stop_patience=int(early_stop_patience),
        num_workers=int(num_workers),
        device=str(device),
        amp=bool(amp),
        grad_clip_norm=float(grad_clip_norm),
        compile_model=bool(compile_model),
        evaluate_model_val_predictions=bool(evaluate_model_val_predictions),
        evaluate_final_test=bool(evaluate_final_test),
        confirm_final_test=bool(confirm_final_test),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_final_test_batches=max_final_test_batches,
        max_train_jets=max_train_jets,
        max_val_jets=max_val_jets,
        max_final_test_jets=max_final_test_jets,
        model_size=str(model_size),
        overwrite=bool(overwrite),
        experiment_name=HLT_MV_EXPERIMENT_NAME,
        experiment_step=HLT_MV_SOURCE_EXPERIMENT_STEP,
        source_contract=HLT_MV_SOURCE_CONTRACT,
        prediction_contract=HLT_MV_SOURCE_PREDICTION_CONTRACT,
        allowed_inputs=HLT_MV_ALLOWED_INPUTS,
        deployment_inputs=HLT_MV_DEPLOYMENT_INPUTS,
    )


def train_hlt_mv_source(config: HLTTriViewSourceConfig) -> dict[str, Any]:
    """Train one HLT-MV source model using the reusable single-view trainer."""

    if config.warm_start_checkpoint is not None:
        raise ValueError("HLT-MV source models must train from scratch; warm_start_checkpoint must be None.")
    return train_hlt_triview_source(config)


def cache_hlt_mv_source_predictions(
    config: HLTTriViewSourceConfig,
    *,
    splits: tuple[str, ...] | list[str] = HLT_MV_SOURCE_PREDICTION_SPLITS,
) -> dict[str, Any]:
    """Cache model-val/final-test predictions for one trained HLT-MV source model."""

    if config.source_contract != HLT_MV_SOURCE_CONTRACT:
        raise ValueError(f"Expected HLT-MV source contract {HLT_MV_SOURCE_CONTRACT!r}, got {config.source_contract!r}.")
    normalized_splits = normalize_hlt_mv_source_prediction_splits(splits)
    prediction_config = HLTTriViewSourceConfig(
        **{
            **config.__dict__,
            "experiment_step": HLT_MV_SOURCE_PREDICTION_EXPERIMENT_STEP,
            "evaluate_model_val_predictions": "model_val" in normalized_splits,
            "evaluate_final_test": "final_test" in normalized_splits,
        }
    )
    return cache_hlt_triview_source_predictions(prediction_config, splits=normalized_splits)


def hlt_mv_source_groups(config: HLTMVExperimentConfig | None = None) -> dict[str, tuple[str, ...]]:
    cfg = _config_for(config)
    return {
        "source_models": tuple(cfg.source_model_names),
        "random_hlt_controls": build_hlt_mv_random_hlt_source_names(cfg.random_hlt_seeds),
    }


__all__ = [
    "HLT_MV_SOURCE_CONTRACT",
    "HLT_MV_SOURCE_DEFAULT_AMP",
    "HLT_MV_SOURCE_DEFAULT_BATCH_SIZE",
    "HLT_MV_SOURCE_DEFAULT_EARLY_STOP_PATIENCE",
    "HLT_MV_SOURCE_DEFAULT_EPOCHS",
    "HLT_MV_SOURCE_DEFAULT_EVAL_BATCH_SIZE",
    "HLT_MV_SOURCE_DEFAULT_GRAD_CLIP_NORM",
    "HLT_MV_SOURCE_DEFAULT_LR",
    "HLT_MV_SOURCE_DEFAULT_MAX_FINAL_TEST_JETS",
    "HLT_MV_SOURCE_DEFAULT_MAX_TRAIN_JETS",
    "HLT_MV_SOURCE_DEFAULT_MAX_VAL_JETS",
    "HLT_MV_SOURCE_DEFAULT_MODEL_SIZE",
    "HLT_MV_SOURCE_DEFAULT_WEIGHT_DECAY",
    "HLT_MV_SOURCE_EXPERIMENT_STEP",
    "HLT_MV_SOURCE_PREDICTION_EXPERIMENT_STEP",
    "HLT_MV_SOURCE_PREDICTION_CONTRACT",
    "HLT_MV_SOURCE_PREDICTION_REPORT",
    "HLT_MV_SOURCE_PREDICTION_SPLITS",
    "HLT_MV_SOURCE_VIEW_HLT",
    "HLT_MV_SOURCE_VIEW_HLT2",
    "build_hlt_mv_source_config",
    "cache_hlt_mv_source_predictions",
    "hlt_mv_source_cache_dir",
    "hlt_mv_source_groups",
    "hlt_mv_source_output_dir",
    "hlt_mv_source_prediction_paths",
    "hlt_mv_source_seed_from_name",
    "hlt_mv_source_view_from_name",
    "hlt_mv_strength_from_source_name",
    "normalize_hlt_mv_source_prediction_splits",
    "train_hlt_mv_source",
]
