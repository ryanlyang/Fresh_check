"""Particle dual-view helpers for HLT multiview fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.hlt_self_dualview import (
    HLTSDVTrainConfig,
    HLT_SDV_BRANCH2_HLT2,
    HLT_SDV_DEFAULT_BATCH_SIZE,
    HLT_SDV_DEFAULT_BRANCH_LR,
    HLT_SDV_DEFAULT_EARLY_STOP_PATIENCE,
    HLT_SDV_DEFAULT_EPOCHS,
    HLT_SDV_DEFAULT_EVAL_BATCH_SIZE,
    HLT_SDV_DEFAULT_HEAD_LR,
    HLT_SDV_DEFAULT_HEAD_WARMUP_EPOCHS,
    HLT_SDV_DEFAULT_HEAD_WARMUP_LR,
    HLT_SDV_DEFAULT_SEED,
    HLT_SDV_DEFAULT_WEIGHT_DECAY,
    hlt_sdv_strength_from_variant,
    normalize_hlt_sdv_strength,
    train_hlt_sdv_model,
)
from teacher_logit_reco.privileged_distill_v3.config import PDV3_MODEL_SPLIT_SIZES

from .config import (
    HLTMVExperimentConfig,
    HLTMVExperimentLayout,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    default_hlt_mv_experiment_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_hlt2_source_name,
    hlt_mv_pretrained_dualview_name,
    hlt_mv_scratch_dualview_name,
    normalize_hlt_mv_source_name,
    normalize_hlt_mv_strengths,
)


HLT_MV_PRETRAINED_DUALVIEW_EXPERIMENT_STEP = "hlt_mv_step5_pretrained_particle_dualview"
HLT_MV_PRETRAINED_DUALVIEW_CONTRACT = "hlt_multiview_pretrained_particle_dualview_v1"
HLT_MV_PRETRAINED_DUALVIEW_REPORT = "hlt_mv_pretrained_dualview_report.json"
HLT_MV_SCRATCH_DUALVIEW_EXPERIMENT_STEP = "hlt_mv_step6_scratch_particle_dualview"
HLT_MV_SCRATCH_DUALVIEW_CONTRACT = "hlt_multiview_scratch_particle_dualview_v1"
HLT_MV_SCRATCH_DUALVIEW_REPORT = "hlt_mv_scratch_dualview_report.json"
HLT_MV_SCRATCH_DUALVIEW_UNUSED_CHECKPOINT = "unused_scratch_no_branch_init"

HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BATCH_SIZE = HLT_SDV_DEFAULT_BATCH_SIZE
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE = HLT_SDV_DEFAULT_EVAL_BATCH_SIZE
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EPOCHS = HLT_SDV_DEFAULT_EPOCHS
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS = HLT_SDV_DEFAULT_HEAD_WARMUP_EPOCHS
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_LR = HLT_SDV_DEFAULT_HEAD_WARMUP_LR
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BRANCH_LR = HLT_SDV_DEFAULT_BRANCH_LR
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_LR = HLT_SDV_DEFAULT_HEAD_LR
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_WEIGHT_DECAY = HLT_SDV_DEFAULT_WEIGHT_DECAY
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE = HLT_SDV_DEFAULT_EARLY_STOP_PATIENCE
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_SEED = HLT_SDV_DEFAULT_SEED
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MODEL_SIZE = "base"
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM = 512
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_REPRESENTATION_DIM = 256
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_DROPOUT = 0.05
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_GRAD_CLIP_NORM = 1.0
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_AMP = False
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_TRAIN_JETS = int(PDV3_MODEL_SPLIT_SIZES["model_train"])
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_VAL_JETS = int(PDV3_MODEL_SPLIT_SIZES["model_val"])
HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS = int(PDV3_MODEL_SPLIT_SIZES["final_test"])
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BATCH_SIZE = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BATCH_SIZE
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EPOCHS = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EPOCHS
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS = 0
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_LR = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_LR
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BRANCH_LR = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BRANCH_LR
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_LR = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_LR
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_WEIGHT_DECAY = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_WEIGHT_DECAY
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_SEED = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_SEED
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MODEL_SIZE = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MODEL_SIZE
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_REPRESENTATION_DIM = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_REPRESENTATION_DIM
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_DROPOUT = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_DROPOUT
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_GRAD_CLIP_NORM = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_GRAD_CLIP_NORM
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_AMP = False
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_TRAIN_JETS = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_TRAIN_JETS
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_VAL_JETS = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_VAL_JETS
HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS


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


def hlt_mv_pretrained_dualview_strength_from_name(
    model_name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> float:
    """Return the HLT2 strength encoded in a pretrained dual-view model name."""

    cfg = _config_for(config)
    name = normalize_hlt_mv_source_name(model_name)
    try:
        strength = hlt_sdv_strength_from_variant(name)
    except ValueError as exc:
        raise ValueError(f"HLT-MV pretrained dual-view name is not strength-encoded: {model_name!r}") from exc
    if strength is None:
        raise ValueError(f"HLT-MV pretrained dual-view name must be sdv_hlt_hlt2_sXpYY, got {model_name!r}")
    normalized = normalize_hlt_sdv_strength(strength)
    if hlt_mv_pretrained_dualview_name(normalized) != name:
        raise ValueError(f"HLT-MV pretrained dual-view name cannot include scratch/control suffixes: {model_name!r}")
    if normalized not in normalize_hlt_mv_strengths(cfg.strengths):
        raise ValueError(f"HLT-MV strength {normalized:.2f} is not in configured grid {cfg.strengths!r}.")
    return normalized


def hlt_mv_pretrained_dualview_source_names(
    model_name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> tuple[str, str]:
    """Return the canonical HLT and matching HLT2 source names for a dual-view run."""

    cfg = _config_for(config)
    strength = hlt_mv_pretrained_dualview_strength_from_name(model_name, config=cfg)
    hlt2_source = hlt_mv_hlt2_source_name(strength, seeds_by_strength=cfg.hlt2_source_seeds)
    return cfg.canonical_hlt_source_name, hlt2_source


def hlt_mv_pretrained_dualview_checkpoint_paths(
    layout: HLTMVExperimentLayout,
    model_name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> tuple[Path, Path]:
    """Return branch A/B initializer checkpoint paths for a pretrained dual-view run."""

    hlt_source, hlt2_source = hlt_mv_pretrained_dualview_source_names(model_name, config=config)
    return (
        layout.source_model_dir(hlt_source) / "best_model_val.pt",
        layout.source_model_dir(hlt2_source) / "best_model_val.pt",
    )


def hlt_mv_pretrained_dualview_output_dir(layout: HLTMVExperimentLayout, model_name: str) -> Path:
    """Return the canonical output directory for one pretrained dual-view model."""

    name = normalize_hlt_mv_source_name(model_name)
    return layout.pretrained_dualview_model_dir(name)


def hlt_mv_pretrained_dualview_prediction_paths(
    layout: HLTMVExperimentLayout,
    model_name: str,
    split: str,
) -> tuple[Path, Path]:
    """Return cached prediction paths written by the reused HLT-SDV trainer."""

    name = normalize_hlt_mv_source_name(model_name)
    if split not in {"model_val", "final_test"}:
        raise ValueError(f"HLT-MV pretrained dual-view predictions are only cached for model_val/final_test, got {split!r}")
    root = layout.pretrained_dualview_model_dir(name) / "predictions" / name
    return root / f"{split}_predictions.npz", root / f"{split}_predictions_metadata.json"


def hlt_mv_scratch_dualview_strength_from_name(
    model_name: str,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> float:
    """Return the HLT2 strength encoded in a scratch dual-view model name."""

    cfg = _config_for(config)
    name = normalize_hlt_mv_source_name(model_name)
    if not name.endswith("_scratch"):
        raise ValueError(f"HLT-MV scratch dual-view name must end with _scratch, got {model_name!r}")
    base_name = name.removesuffix("_scratch")
    strength = hlt_mv_pretrained_dualview_strength_from_name(base_name, config=cfg)
    if hlt_mv_scratch_dualview_name(strength) != name:
        raise ValueError(f"HLT-MV scratch dual-view name is not canonical: {model_name!r}")
    return strength


def hlt_mv_scratch_dualview_output_dir(layout: HLTMVExperimentLayout, model_name: str) -> Path:
    """Return the canonical output directory for one scratch dual-view model."""

    name = normalize_hlt_mv_source_name(model_name)
    return layout.scratch_dualview_model_dir(name)


def hlt_mv_scratch_dualview_prediction_paths(
    layout: HLTMVExperimentLayout,
    model_name: str,
    split: str,
) -> tuple[Path, Path]:
    """Return cached prediction paths written by the reused HLT-SDV trainer."""

    name = normalize_hlt_mv_source_name(model_name)
    if split not in {"model_val", "final_test"}:
        raise ValueError(f"HLT-MV scratch dual-view predictions are only cached for model_val/final_test, got {split!r}")
    root = layout.scratch_dualview_model_dir(name) / "predictions" / name
    return root / f"{split}_predictions.npz", root / f"{split}_predictions_metadata.json"


def build_hlt_mv_pretrained_dualview_config(
    *,
    model_name: str,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str | None = None,
    layout: HLTMVExperimentLayout | None = None,
    run_config: HLTMVExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    hlt_cache_dir: str | Path | None = None,
    hlt2_cache_dir: str | Path | None = None,
    hlt_checkpoint: str | Path | None = None,
    hlt2_checkpoint: str | Path | None = None,
    seed: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_SEED,
    batch_size: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BATCH_SIZE,
    eval_batch_size: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE,
    epochs: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EPOCHS,
    head_warmup_epochs: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS,
    head_warmup_lr: float = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_LR,
    branch_lr: float = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BRANCH_LR,
    head_lr: float = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_LR,
    weight_decay: float = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_WEIGHT_DECAY,
    dropout: float = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_DROPOUT,
    fusion_hidden_dim: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM,
    representation_dim: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_REPRESENTATION_DIM,
    early_stop_patience: int = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE,
    num_workers: int = 0,
    device: str = "auto",
    amp: bool = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_AMP,
    grad_clip_norm: float = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_GRAD_CLIP_NORM,
    model_size: str = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MODEL_SIZE,
    compile_model: bool = False,
    evaluate_model_val_predictions: bool = True,
    evaluate_final_test: bool = True,
    confirm_final_test: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_final_test_batches: int | None = None,
    max_train_jets: int | None = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_TRAIN_JETS,
    max_val_jets: int | None = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_VAL_JETS,
    max_final_test_jets: int | None = HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS,
    overwrite: bool = False,
) -> HLTSDVTrainConfig:
    """Build the HLT-SDV config for one HLT-MV pretrained dual-view model."""

    cfg = _config_for(run_config)
    name = normalize_hlt_mv_source_name(model_name)
    strength = hlt_mv_pretrained_dualview_strength_from_name(name, config=cfg)
    lay = _layout_for(
        layout=layout,
        output_root=output_root,
        pdv3_experiment_name=cfg.pdv3_experiment_name if pdv3_experiment_name is None else pdv3_experiment_name,
    )
    default_hlt_checkpoint, default_hlt2_checkpoint = hlt_mv_pretrained_dualview_checkpoint_paths(
        lay,
        name,
        config=cfg,
    )
    if int(head_warmup_epochs) <= 0:
        raise ValueError("Pretrained HLT-MV dual-view runs must keep a positive head_warmup_epochs.")
    return HLTSDVTrainConfig(
        output_dir=str(Path(output_dir) if output_dir is not None else hlt_mv_pretrained_dualview_output_dir(lay, name)),
        hlt_cache_dir=str(Path(hlt_cache_dir) if hlt_cache_dir is not None else lay.hlt_cache_dir),
        hlt2_cache_dir=str(Path(hlt2_cache_dir) if hlt2_cache_dir is not None else lay.hlt2_cache_dir(strength)),
        hlt_teacher_checkpoint=str(Path(hlt_checkpoint) if hlt_checkpoint is not None else default_hlt_checkpoint),
        hlt2_branch_checkpoint=str(Path(hlt2_checkpoint) if hlt2_checkpoint is not None else default_hlt2_checkpoint),
        variant_name=name,
        branch2_mode=HLT_SDV_BRANCH2_HLT2,
        seed=int(seed),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        epochs=int(epochs),
        head_warmup_epochs=int(head_warmup_epochs),
        head_warmup_lr=float(head_warmup_lr),
        branch_lr=float(branch_lr),
        head_lr=float(head_lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        fusion_hidden_dim=int(fusion_hidden_dim),
        representation_dim=int(representation_dim),
        early_stop_patience=int(early_stop_patience),
        num_workers=int(num_workers),
        device=str(device),
        amp=bool(amp),
        grad_clip_norm=float(grad_clip_norm),
        model_size=str(model_size),
        compile_model=bool(compile_model),
        initialize_branches=True,
        evaluate_model_val_predictions=bool(evaluate_model_val_predictions),
        evaluate_final_test=bool(evaluate_final_test),
        confirm_final_test=bool(confirm_final_test),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_final_test_batches=max_final_test_batches,
        max_train_jets=max_train_jets,
        max_val_jets=max_val_jets,
        max_final_test_jets=max_final_test_jets,
        overwrite=bool(overwrite),
    )


def build_hlt_mv_scratch_dualview_config(
    *,
    model_name: str,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str | None = None,
    layout: HLTMVExperimentLayout | None = None,
    run_config: HLTMVExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    hlt_cache_dir: str | Path | None = None,
    hlt2_cache_dir: str | Path | None = None,
    unused_hlt_checkpoint: str | Path | None = None,
    seed: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_SEED,
    batch_size: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BATCH_SIZE,
    eval_batch_size: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE,
    epochs: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EPOCHS,
    head_warmup_epochs: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS,
    head_warmup_lr: float = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_LR,
    branch_lr: float = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BRANCH_LR,
    head_lr: float = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_LR,
    weight_decay: float = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_WEIGHT_DECAY,
    dropout: float = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_DROPOUT,
    fusion_hidden_dim: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM,
    representation_dim: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_REPRESENTATION_DIM,
    early_stop_patience: int = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE,
    num_workers: int = 0,
    device: str = "auto",
    amp: bool = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_AMP,
    grad_clip_norm: float = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_GRAD_CLIP_NORM,
    model_size: str = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MODEL_SIZE,
    compile_model: bool = False,
    evaluate_model_val_predictions: bool = True,
    evaluate_final_test: bool = True,
    confirm_final_test: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_final_test_batches: int | None = None,
    max_train_jets: int | None = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_TRAIN_JETS,
    max_val_jets: int | None = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_VAL_JETS,
    max_final_test_jets: int | None = HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS,
    overwrite: bool = False,
) -> HLTSDVTrainConfig:
    """Build the HLT-SDV config for one scratch HLT-MV dual-view model."""

    cfg = _config_for(run_config)
    name = normalize_hlt_mv_source_name(model_name)
    strength = hlt_mv_scratch_dualview_strength_from_name(name, config=cfg)
    lay = _layout_for(
        layout=layout,
        output_root=output_root,
        pdv3_experiment_name=cfg.pdv3_experiment_name if pdv3_experiment_name is None else pdv3_experiment_name,
    )
    if int(head_warmup_epochs) != 0:
        raise ValueError("Scratch HLT-MV dual-view runs must use head_warmup_epochs=0.")
    return HLTSDVTrainConfig(
        output_dir=str(Path(output_dir) if output_dir is not None else hlt_mv_scratch_dualview_output_dir(lay, name)),
        hlt_cache_dir=str(Path(hlt_cache_dir) if hlt_cache_dir is not None else lay.hlt_cache_dir),
        hlt2_cache_dir=str(Path(hlt2_cache_dir) if hlt2_cache_dir is not None else lay.hlt2_cache_dir(strength)),
        hlt_teacher_checkpoint=str(
            Path(unused_hlt_checkpoint) if unused_hlt_checkpoint is not None else HLT_MV_SCRATCH_DUALVIEW_UNUSED_CHECKPOINT
        ),
        hlt2_branch_checkpoint=None,
        variant_name=name,
        branch2_mode=HLT_SDV_BRANCH2_HLT2,
        seed=int(seed),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        epochs=int(epochs),
        head_warmup_epochs=0,
        head_warmup_lr=float(head_warmup_lr),
        branch_lr=float(branch_lr),
        head_lr=float(head_lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        fusion_hidden_dim=int(fusion_hidden_dim),
        representation_dim=int(representation_dim),
        early_stop_patience=int(early_stop_patience),
        num_workers=int(num_workers),
        device=str(device),
        amp=bool(amp),
        grad_clip_norm=float(grad_clip_norm),
        model_size=str(model_size),
        compile_model=bool(compile_model),
        initialize_branches=False,
        evaluate_model_val_predictions=bool(evaluate_model_val_predictions),
        evaluate_final_test=bool(evaluate_final_test),
        confirm_final_test=bool(confirm_final_test),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_final_test_batches=max_final_test_batches,
        max_train_jets=max_train_jets,
        max_val_jets=max_val_jets,
        max_final_test_jets=max_final_test_jets,
        overwrite=bool(overwrite),
    )


def train_hlt_mv_pretrained_dualview(config: HLTSDVTrainConfig) -> dict[str, Any]:
    """Train one pretrained-branch HLT-MV particle dual-view model."""

    if not bool(config.initialize_branches):
        raise ValueError("HLT-MV pretrained dual-view runs must initialize both branches.")
    if not config.hlt2_branch_checkpoint:
        raise ValueError("HLT-MV pretrained dual-view runs require a matching HLT2 branch checkpoint.")
    if int(config.head_warmup_epochs) <= 0:
        raise ValueError("HLT-MV pretrained dual-view runs require head_warmup_epochs >= 1.")
    report = train_hlt_sdv_model(config)
    hlt_mv_report = {
        **report,
        "hlt_mv_contract": HLT_MV_PRETRAINED_DUALVIEW_CONTRACT,
        "hlt_mv_experiment_step": HLT_MV_PRETRAINED_DUALVIEW_EXPERIMENT_STEP,
        "hlt_mv_model_family": "pretrained_particle_dualview",
        "hlt_mv_branch_initializers": {
            "branch_a": {
                "view": "fixed_hlt",
                "checkpoint": str(config.hlt_teacher_checkpoint),
            },
            "branch_b": {
                "view": "hlt2",
                "checkpoint": str(config.hlt2_branch_checkpoint),
            },
        },
        "head_warmup_required": True,
    }
    save_json(Path(config.output_dir) / HLT_MV_PRETRAINED_DUALVIEW_REPORT, hlt_mv_report)
    return hlt_mv_report


def train_hlt_mv_scratch_dualview(config: HLTSDVTrainConfig) -> dict[str, Any]:
    """Train one from-scratch HLT-MV particle dual-view model."""

    if bool(config.initialize_branches):
        raise ValueError("HLT-MV scratch dual-view runs must not initialize branches.")
    if config.hlt2_branch_checkpoint:
        raise ValueError("HLT-MV scratch dual-view runs must not provide an HLT2 branch checkpoint.")
    if int(config.head_warmup_epochs) != 0:
        raise ValueError("HLT-MV scratch dual-view runs require head_warmup_epochs=0.")
    report = train_hlt_sdv_model(config)
    hlt_mv_report = {
        **report,
        "hlt_mv_contract": HLT_MV_SCRATCH_DUALVIEW_CONTRACT,
        "hlt_mv_experiment_step": HLT_MV_SCRATCH_DUALVIEW_EXPERIMENT_STEP,
        "hlt_mv_model_family": "scratch_particle_dualview",
        "hlt_mv_branch_initializers": {
            "branch_a": None,
            "branch_b": None,
        },
        "head_warmup_required": False,
    }
    save_json(Path(config.output_dir) / HLT_MV_SCRATCH_DUALVIEW_REPORT, hlt_mv_report)
    return hlt_mv_report


__all__ = [
    "HLT_MV_PRETRAINED_DUALVIEW_CONTRACT",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_AMP",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BATCH_SIZE",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_BRANCH_LR",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_DROPOUT",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EPOCHS",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_GRAD_CLIP_NORM",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_LR",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_HEAD_WARMUP_LR",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_TRAIN_JETS",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MAX_VAL_JETS",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_MODEL_SIZE",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_REPRESENTATION_DIM",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_SEED",
    "HLT_MV_PRETRAINED_DUALVIEW_DEFAULT_WEIGHT_DECAY",
    "HLT_MV_PRETRAINED_DUALVIEW_EXPERIMENT_STEP",
    "HLT_MV_PRETRAINED_DUALVIEW_REPORT",
    "HLT_MV_SCRATCH_DUALVIEW_CONTRACT",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_AMP",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BATCH_SIZE",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_BRANCH_LR",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_DROPOUT",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EARLY_STOP_PATIENCE",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EPOCHS",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_EVAL_BATCH_SIZE",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_FUSION_HIDDEN_DIM",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_GRAD_CLIP_NORM",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_LR",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_EPOCHS",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_HEAD_WARMUP_LR",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_FINAL_TEST_JETS",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_TRAIN_JETS",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MAX_VAL_JETS",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_MODEL_SIZE",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_REPRESENTATION_DIM",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_SEED",
    "HLT_MV_SCRATCH_DUALVIEW_DEFAULT_WEIGHT_DECAY",
    "HLT_MV_SCRATCH_DUALVIEW_EXPERIMENT_STEP",
    "HLT_MV_SCRATCH_DUALVIEW_REPORT",
    "HLT_MV_SCRATCH_DUALVIEW_UNUSED_CHECKPOINT",
    "build_hlt_mv_pretrained_dualview_config",
    "build_hlt_mv_scratch_dualview_config",
    "hlt_mv_pretrained_dualview_checkpoint_paths",
    "hlt_mv_pretrained_dualview_output_dir",
    "hlt_mv_pretrained_dualview_prediction_paths",
    "hlt_mv_pretrained_dualview_source_names",
    "hlt_mv_pretrained_dualview_strength_from_name",
    "hlt_mv_scratch_dualview_output_dir",
    "hlt_mv_scratch_dualview_prediction_paths",
    "hlt_mv_scratch_dualview_strength_from_name",
    "train_hlt_mv_pretrained_dualview",
    "train_hlt_mv_scratch_dualview",
]
