"""Tri-view helpers for HLT multiview source/fusion experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.hlt_self_dualview import (
    HLT_TRIVIEW_MODEL_NAME,
    HLTTriViewTrainConfig,
    train_hlt_triview_model,
)
from teacher_logit_reco.privileged_distill_v3.config import PDV3_MODEL_SPLIT_SIZES

from .config import (
    HLTMVExperimentConfig,
    HLTMVExperimentLayout,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_TRIVIEW_MODEL_NAME,
    default_hlt_mv_experiment_config,
    default_hlt_mv_experiment_layout,
    hlt_mv_hlt2_source_name,
    normalize_hlt_mv_source_name,
)


HLT_MV_TRIVIEW_EXPERIMENT_STEP = "hlt_mv_step7_triview_particle_fusion"
HLT_MV_TRIVIEW_CONTRACT = "hlt_multiview_triview_particle_fusion_v1"
HLT_MV_TRIVIEW_REPORT = "hlt_mv_triview_report.json"

HLT_MV_TRIVIEW_S0P35_STRENGTH = 0.35
HLT_MV_TRIVIEW_S1P00_STRENGTH = 1.00

HLT_MV_TRIVIEW_DEFAULT_BATCH_SIZE = 64
HLT_MV_TRIVIEW_DEFAULT_EVAL_BATCH_SIZE = 96
HLT_MV_TRIVIEW_DEFAULT_EPOCHS = 10
HLT_MV_TRIVIEW_DEFAULT_HEAD_WARMUP_EPOCHS = 1
HLT_MV_TRIVIEW_DEFAULT_HEAD_WARMUP_LR = 1.0e-3
HLT_MV_TRIVIEW_DEFAULT_BRANCH_LR = 2.0e-5
HLT_MV_TRIVIEW_DEFAULT_HEAD_LR = 3.0e-4
HLT_MV_TRIVIEW_DEFAULT_WEIGHT_DECAY = 1.0e-4
HLT_MV_TRIVIEW_DEFAULT_DROPOUT = 0.05
HLT_MV_TRIVIEW_DEFAULT_FUSION_HIDDEN_DIM = 512
HLT_MV_TRIVIEW_DEFAULT_REPRESENTATION_DIM = 256
HLT_MV_TRIVIEW_DEFAULT_EARLY_STOP_PATIENCE = 3
HLT_MV_TRIVIEW_DEFAULT_GRAD_CLIP_NORM = 1.0
HLT_MV_TRIVIEW_DEFAULT_SEED = 9201
HLT_MV_TRIVIEW_DEFAULT_MODEL_SIZE = "base"
HLT_MV_TRIVIEW_DEFAULT_AMP = False
HLT_MV_TRIVIEW_DEFAULT_MAX_TRAIN_JETS = int(PDV3_MODEL_SPLIT_SIZES["model_train"])
HLT_MV_TRIVIEW_DEFAULT_MAX_VAL_JETS = int(PDV3_MODEL_SPLIT_SIZES["model_val"])
HLT_MV_TRIVIEW_DEFAULT_MAX_FINAL_TEST_JETS = int(PDV3_MODEL_SPLIT_SIZES["final_test"])


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


def normalize_hlt_mv_triview_name(model_name: str = HLT_MV_TRIVIEW_MODEL_NAME) -> str:
    """Validate the single HLT-MV tri-view model name."""

    name = normalize_hlt_mv_source_name(model_name)
    if name != HLT_MV_TRIVIEW_MODEL_NAME or name != HLT_TRIVIEW_MODEL_NAME:
        raise ValueError(f"HLT-MV tri-view model must be {HLT_MV_TRIVIEW_MODEL_NAME!r}, got {model_name!r}.")
    return name


def hlt_mv_triview_source_names(
    *,
    config: HLTMVExperimentConfig | None = None,
) -> tuple[str, str, str]:
    """Return the canonical HLT, HLT2 s0p35, and HLT2 s1p00 source names."""

    cfg = _config_for(config)
    return (
        cfg.canonical_hlt_source_name,
        hlt_mv_hlt2_source_name(HLT_MV_TRIVIEW_S0P35_STRENGTH, seeds_by_strength=cfg.hlt2_source_seeds),
        hlt_mv_hlt2_source_name(HLT_MV_TRIVIEW_S1P00_STRENGTH, seeds_by_strength=cfg.hlt2_source_seeds),
    )


def hlt_mv_triview_checkpoint_paths(
    layout: HLTMVExperimentLayout,
    *,
    config: HLTMVExperimentConfig | None = None,
) -> tuple[Path, Path, Path]:
    """Return branch initializer checkpoints for the canonical tri-view model."""

    hlt_source, hlt2_s0p35_source, hlt2_s1p00_source = hlt_mv_triview_source_names(config=config)
    return (
        layout.source_model_dir(hlt_source) / "best_model_val.pt",
        layout.source_model_dir(hlt2_s0p35_source) / "best_model_val.pt",
        layout.source_model_dir(hlt2_s1p00_source) / "best_model_val.pt",
    )


def hlt_mv_triview_output_dir(
    layout: HLTMVExperimentLayout,
    model_name: str = HLT_MV_TRIVIEW_MODEL_NAME,
) -> Path:
    """Return the canonical output directory for the tri-view model."""

    return layout.triview_model_dir(normalize_hlt_mv_triview_name(model_name))


def hlt_mv_triview_prediction_paths(
    layout: HLTMVExperimentLayout,
    split: str,
    *,
    model_name: str = HLT_MV_TRIVIEW_MODEL_NAME,
) -> tuple[Path, Path]:
    """Return cached prediction paths written by the reused tri-view trainer."""

    name = normalize_hlt_mv_triview_name(model_name)
    if split not in {"model_val", "final_test"}:
        raise ValueError(f"HLT-MV tri-view predictions are only cached for model_val/final_test, got {split!r}")
    root = layout.triview_model_dir(name) / "predictions" / name
    return root / f"{split}_predictions.npz", root / f"{split}_predictions_metadata.json"


def build_hlt_mv_triview_config(
    *,
    model_name: str = HLT_MV_TRIVIEW_MODEL_NAME,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str | None = None,
    layout: HLTMVExperimentLayout | None = None,
    run_config: HLTMVExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    hlt_cache_dir: str | Path | None = None,
    hlt2_s0p35_cache_dir: str | Path | None = None,
    hlt2_s1p00_cache_dir: str | Path | None = None,
    hlt_checkpoint: str | Path | None = None,
    hlt2_s0p35_checkpoint: str | Path | None = None,
    hlt2_s1p00_checkpoint: str | Path | None = None,
    seed: int = HLT_MV_TRIVIEW_DEFAULT_SEED,
    batch_size: int = HLT_MV_TRIVIEW_DEFAULT_BATCH_SIZE,
    eval_batch_size: int = HLT_MV_TRIVIEW_DEFAULT_EVAL_BATCH_SIZE,
    epochs: int = HLT_MV_TRIVIEW_DEFAULT_EPOCHS,
    head_warmup_epochs: int = HLT_MV_TRIVIEW_DEFAULT_HEAD_WARMUP_EPOCHS,
    head_warmup_lr: float = HLT_MV_TRIVIEW_DEFAULT_HEAD_WARMUP_LR,
    branch_lr: float = HLT_MV_TRIVIEW_DEFAULT_BRANCH_LR,
    head_lr: float = HLT_MV_TRIVIEW_DEFAULT_HEAD_LR,
    weight_decay: float = HLT_MV_TRIVIEW_DEFAULT_WEIGHT_DECAY,
    dropout: float = HLT_MV_TRIVIEW_DEFAULT_DROPOUT,
    fusion_hidden_dim: int = HLT_MV_TRIVIEW_DEFAULT_FUSION_HIDDEN_DIM,
    representation_dim: int = HLT_MV_TRIVIEW_DEFAULT_REPRESENTATION_DIM,
    early_stop_patience: int = HLT_MV_TRIVIEW_DEFAULT_EARLY_STOP_PATIENCE,
    num_workers: int = 0,
    device: str = "auto",
    amp: bool = HLT_MV_TRIVIEW_DEFAULT_AMP,
    grad_clip_norm: float = HLT_MV_TRIVIEW_DEFAULT_GRAD_CLIP_NORM,
    model_size: str = HLT_MV_TRIVIEW_DEFAULT_MODEL_SIZE,
    compile_model: bool = False,
    evaluate_model_val_predictions: bool = True,
    evaluate_final_test: bool = True,
    confirm_final_test: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_final_test_batches: int | None = None,
    max_train_jets: int | None = HLT_MV_TRIVIEW_DEFAULT_MAX_TRAIN_JETS,
    max_val_jets: int | None = HLT_MV_TRIVIEW_DEFAULT_MAX_VAL_JETS,
    max_final_test_jets: int | None = HLT_MV_TRIVIEW_DEFAULT_MAX_FINAL_TEST_JETS,
    overwrite: bool = False,
) -> HLTTriViewTrainConfig:
    """Build the reusable tri-view trainer config for HLT-MV Step 7."""

    cfg = _config_for(run_config)
    name = normalize_hlt_mv_triview_name(model_name)
    lay = _layout_for(
        layout=layout,
        output_root=output_root,
        pdv3_experiment_name=cfg.pdv3_experiment_name if pdv3_experiment_name is None else pdv3_experiment_name,
    )
    default_hlt_checkpoint, default_s0p35_checkpoint, default_s1p00_checkpoint = hlt_mv_triview_checkpoint_paths(
        lay,
        config=cfg,
    )
    if int(head_warmup_epochs) <= 0:
        raise ValueError("HLT-MV tri-view runs require head_warmup_epochs >= 1.")
    return HLTTriViewTrainConfig(
        output_dir=str(Path(output_dir) if output_dir is not None else hlt_mv_triview_output_dir(lay, name)),
        hlt_cache_dir=str(Path(hlt_cache_dir) if hlt_cache_dir is not None else lay.hlt_cache_dir),
        hlt2_s0p35_cache_dir=str(
            Path(hlt2_s0p35_cache_dir)
            if hlt2_s0p35_cache_dir is not None
            else lay.hlt2_cache_dir(HLT_MV_TRIVIEW_S0P35_STRENGTH)
        ),
        hlt2_s1p00_cache_dir=str(
            Path(hlt2_s1p00_cache_dir)
            if hlt2_s1p00_cache_dir is not None
            else lay.hlt2_cache_dir(HLT_MV_TRIVIEW_S1P00_STRENGTH)
        ),
        hlt_source_checkpoint=str(Path(hlt_checkpoint) if hlt_checkpoint is not None else default_hlt_checkpoint),
        hlt2_s0p35_source_checkpoint=str(
            Path(hlt2_s0p35_checkpoint) if hlt2_s0p35_checkpoint is not None else default_s0p35_checkpoint
        ),
        hlt2_s1p00_source_checkpoint=str(
            Path(hlt2_s1p00_checkpoint) if hlt2_s1p00_checkpoint is not None else default_s1p00_checkpoint
        ),
        model_name=name,
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


def train_hlt_mv_triview(config: HLTTriViewTrainConfig) -> dict[str, Any]:
    """Train the canonical HLT-MV tri-view particle fusion model."""

    normalize_hlt_mv_triview_name(config.model_name)
    if int(config.head_warmup_epochs) <= 0:
        raise ValueError("HLT-MV tri-view runs require head_warmup_epochs >= 1.")
    report = train_hlt_triview_model(config)
    hlt_mv_report = {
        **report,
        "hlt_mv_contract": HLT_MV_TRIVIEW_CONTRACT,
        "hlt_mv_experiment_step": HLT_MV_TRIVIEW_EXPERIMENT_STEP,
        "hlt_mv_model_family": "triview_particle_fusion",
        "hlt_mv_branch_initializers": {
            "hlt": str(config.hlt_source_checkpoint),
            "hlt2_s0p35": str(config.hlt2_s0p35_source_checkpoint),
            "hlt2_s1p00": str(config.hlt2_s1p00_source_checkpoint),
        },
        "head_warmup_required": True,
    }
    save_json(Path(config.output_dir) / HLT_MV_TRIVIEW_REPORT, hlt_mv_report)
    return hlt_mv_report


__all__ = [
    "HLT_MV_TRIVIEW_CONTRACT",
    "HLT_MV_TRIVIEW_DEFAULT_AMP",
    "HLT_MV_TRIVIEW_DEFAULT_BATCH_SIZE",
    "HLT_MV_TRIVIEW_DEFAULT_BRANCH_LR",
    "HLT_MV_TRIVIEW_DEFAULT_DROPOUT",
    "HLT_MV_TRIVIEW_DEFAULT_EARLY_STOP_PATIENCE",
    "HLT_MV_TRIVIEW_DEFAULT_EPOCHS",
    "HLT_MV_TRIVIEW_DEFAULT_EVAL_BATCH_SIZE",
    "HLT_MV_TRIVIEW_DEFAULT_FUSION_HIDDEN_DIM",
    "HLT_MV_TRIVIEW_DEFAULT_GRAD_CLIP_NORM",
    "HLT_MV_TRIVIEW_DEFAULT_HEAD_LR",
    "HLT_MV_TRIVIEW_DEFAULT_HEAD_WARMUP_EPOCHS",
    "HLT_MV_TRIVIEW_DEFAULT_HEAD_WARMUP_LR",
    "HLT_MV_TRIVIEW_DEFAULT_MAX_FINAL_TEST_JETS",
    "HLT_MV_TRIVIEW_DEFAULT_MAX_TRAIN_JETS",
    "HLT_MV_TRIVIEW_DEFAULT_MAX_VAL_JETS",
    "HLT_MV_TRIVIEW_DEFAULT_MODEL_SIZE",
    "HLT_MV_TRIVIEW_DEFAULT_REPRESENTATION_DIM",
    "HLT_MV_TRIVIEW_DEFAULT_SEED",
    "HLT_MV_TRIVIEW_DEFAULT_WEIGHT_DECAY",
    "HLT_MV_TRIVIEW_EXPERIMENT_STEP",
    "HLT_MV_TRIVIEW_REPORT",
    "HLT_MV_TRIVIEW_S0P35_STRENGTH",
    "HLT_MV_TRIVIEW_S1P00_STRENGTH",
    "build_hlt_mv_triview_config",
    "hlt_mv_triview_checkpoint_paths",
    "hlt_mv_triview_output_dir",
    "hlt_mv_triview_prediction_paths",
    "hlt_mv_triview_source_names",
    "normalize_hlt_mv_triview_name",
    "train_hlt_mv_triview",
]
