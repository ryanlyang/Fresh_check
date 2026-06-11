"""First P-CNN teacher-logit reconstruction experiment harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from jetclass_fresh.fusion import prediction_paths
from jetclass_fresh.hlt_baseline import save_json

from .particle_cnn_reconstructor import PARTICLE_CNN_ORDERING_ASSUMPTION
from .predict_particle_cnn import (
    TeacherLogitParticleCnnPredictionConfig,
    collect_teacher_logit_particle_cnn_predictions,
    default_model_name_for_teacher_architecture,
)
from .train_particle_cnn import TeacherLogitParticleCnnTrainConfig, train_teacher_logit_particle_cnn_reco


EXPERIMENT_STEP = "teacher_logit_reco_step7_particle_cnn_first_experiment"
DEFAULT_FIRST_EXPERIMENT_SPLITS = ["stack_val"]


@dataclass(frozen=True)
class PredictionComparisonSpec:
    """Saved prediction block to compare with the P-CNN reconstructor outputs."""

    name: str
    prediction_dir: str
    model_name: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": str(self.name),
            "prediction_dir": str(self.prediction_dir),
            "model_name": str(self.model_name),
        }


@dataclass
class TeacherLogitParticleCnnFirstExperimentConfig:
    """Configuration for the first P-CNN -> frozen teacher-logit experiment."""

    output_dir: str
    teacher_checkpoint: str
    manifest_path: str = "checkpoints/jetclass_fresh_splits/split_manifest.json.gz"
    hlt_cache_dir: str = "checkpoints/jetclass_fresh_hlt_cache"
    data_dir: str | None = None
    teacher_architecture: str = "part"
    model_name: str | None = None
    splits: list[str] = field(default_factory=lambda: list(DEFAULT_FIRST_EXPERIMENT_SPLITS))
    seed: int = 1205
    batch_size: int = 64
    predict_batch_size: int = 128
    epochs: int = 20
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    predict_num_workers: int = 0
    device: str = "auto"
    predict_device: str | None = None
    amp: bool = True
    predict_amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = 50_000
    max_val_jets: int | None = 10_000
    max_prediction_jets_per_split: int | None = 50_000
    verify_hlt_hash: bool = True
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    compile_model: bool = False
    max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    hidden_channels: int = 128
    num_blocks: int = 6
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)
    decoder_dims: tuple[int, ...] = (256, 128)
    slot_dim: int | None = None
    num_extra_candidates: int = 32
    dropout: float = 0.05
    max_delta_logpt: float = 1.0
    max_delta_eta: float = 0.35
    max_delta_phi: float = 0.35
    max_delta_loge: float = 1.0
    parent_weight_bias: float = 2.0
    extra_weight_bias: float = -3.0
    max_total_extra_pt_fraction: float = 0.20
    max_extra_delta_eta: float = 1.25
    max_extra_delta_phi: float = 1.25
    teacher_kl_weight: float = 1.0
    ce_weight: float = 0.25
    correction_budget_weight: float = 0.05
    jet_summary_weight: float = 0.05
    temperature: float = 2.0
    confirm_final_test: bool = False
    overwrite_output: bool = False
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    fit_final_stacker: bool = False
    comparison_specs: list[PredictionComparisonSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kernel_sizes = tuple(int(value) for value in self.kernel_sizes)
        self.dilations = tuple(int(value) for value in self.dilations)
        self.context_mlp_dims = tuple(int(dim) for dim in self.context_mlp_dims)
        self.decoder_dims = tuple(int(dim) for dim in self.decoder_dims)
        if self.slot_dim is not None:
            self.slot_dim = int(self.slot_dim)
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.predict_batch_size) <= 0:
            raise ValueError("predict_batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if int(self.hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if int(self.num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if len(self.kernel_sizes) != int(self.num_blocks):
            raise ValueError("kernel_sizes length must match num_blocks")
        if len(self.dilations) != int(self.num_blocks):
            raise ValueError("dilations length must match num_blocks")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd values")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("dilations must all be positive")
        if not self.context_mlp_dims:
            raise ValueError("context_mlp_dims must contain at least one dimension")
        if not self.decoder_dims:
            raise ValueError("decoder_dims must contain at least one dimension")
        if "final_test" in list(self.splits) and not bool(self.confirm_final_test):
            raise ValueError("Refusing to include final_test without confirm_final_test=True")

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or default_model_name_for_teacher_architecture(self.teacher_architecture)

    @property
    def train_output_dir(self) -> Path:
        return Path(self.output_dir) / "train" / self.resolved_model_name

    @property
    def prediction_output_dir(self) -> Path:
        return Path(self.output_dir) / "prediction_collection" / self.resolved_model_name

    @property
    def prediction_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"


def _read_prediction_metadata(prediction_dir: str | Path, model_name: str, split: str) -> Dict[str, Any]:
    _, meta_path = prediction_paths(prediction_dir, model_name, split)
    if not meta_path.exists():
        return {
            "split": str(split),
            "model_name": str(model_name),
            "metadata_path": str(meta_path),
            "missing": True,
        }
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "split": str(split),
        "model_name": str(model_name),
        "metadata_path": str(meta_path),
        "missing": False,
        "n_jets": metadata.get("n_jets"),
        "metrics": dict(metadata.get("metrics") or {}),
        "model_kind": metadata.get("model_kind"),
        "allowed_inputs": metadata.get("allowed_inputs"),
        "reconstructor_architecture": metadata.get("reconstructor_architecture"),
        "reconstructor_ordering_assumption": metadata.get("reconstructor_ordering_assumption"),
        "experiment_step": metadata.get("experiment_step"),
    }


def prediction_metric_summary(
    *,
    name: str,
    prediction_dir: str | Path,
    model_name: str,
    splits: Sequence[str],
) -> Dict[str, Any]:
    """Summarize saved prediction-block metadata without loading logits."""

    return {
        "name": str(name),
        "prediction_dir": str(prediction_dir),
        "model_name": str(model_name),
        "splits": {
            str(split): _read_prediction_metadata(prediction_dir, model_name, str(split))
            for split in splits
        },
    }


def comparison_delta_summary(primary: Dict[str, Any], comparisons: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute P-CNN-minus-comparison metric deltas where both metrics exist."""

    output: Dict[str, Any] = {}
    for split, primary_row in primary.get("splits", {}).items():
        primary_metrics = dict(primary_row.get("metrics") or {})
        split_rows = {}
        for comparison in comparisons:
            comp_row = dict(comparison.get("splits", {}).get(split) or {})
            comp_metrics = dict(comp_row.get("metrics") or {})
            metric_deltas = {}
            for metric_name, primary_value in primary_metrics.items():
                comp_value = comp_metrics.get(metric_name)
                if isinstance(primary_value, (int, float)) and isinstance(comp_value, (int, float)):
                    metric_deltas[metric_name] = float(primary_value) - float(comp_value)
            split_rows[comparison["name"]] = {
                "comparison_model_name": comparison.get("model_name"),
                "missing": bool(comp_row.get("missing", True)),
                "metric_deltas": metric_deltas,
            }
        output[split] = split_rows
    return output


def build_first_experiment_report(
    *,
    config: TeacherLogitParticleCnnFirstExperimentConfig,
    train_report: Dict[str, Any],
    prediction_report: Dict[str, Any],
) -> Dict[str, Any]:
    pcnn_summary = prediction_metric_summary(
        name="particle_cnn_reco",
        prediction_dir=config.prediction_dir,
        model_name=config.resolved_model_name,
        splits=config.splits,
    )
    comparison_summaries = [
        prediction_metric_summary(
            name=spec.name,
            prediction_dir=spec.prediction_dir,
            model_name=spec.model_name,
            splits=config.splits,
        )
        for spec in config.comparison_specs
    ]
    return {
        "experiment_step": EXPERIMENT_STEP,
        "reconstructor_architecture": "particle_cnn",
        "ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
        "model_name": config.resolved_model_name,
        "output_dir": str(config.output_dir),
        "train_output_dir": str(config.train_output_dir),
        "prediction_output_dir": str(config.prediction_output_dir),
        "prediction_dir": str(config.prediction_dir),
        "config": {
            **asdict(config),
            "comparison_specs": [spec.to_dict() for spec in config.comparison_specs],
        },
        "train_report": train_report,
        "prediction_report": prediction_report,
        "prediction_metrics": pcnn_summary,
        "comparison_metrics": comparison_summaries,
        "comparison_deltas": comparison_delta_summary(pcnn_summary, comparison_summaries),
        "fits_final_stacker": False,
        "requested_final_stacker": bool(config.fit_final_stacker),
        "research_questions": {
            "standalone_signal": "Compare particle_cnn_reco stack_val metrics against raw-HLT, GT-reco, PN-reco, and PFN-reco rows.",
            "fusion_diversity": (
                "Use the saved PCNN prediction blocks with independent fusion after Step 8 runners are added, "
                "or pass comparison prediction directories here to inspect metric deltas now."
            ),
            "rank_bias": (
                "Inspect whether the order-sensitive PCNN residual adds signal that is not present in "
                "permutation-invariant PFN, graph PN, or attention GT reconstructors."
            ),
        },
        "leakage_rules": {
            "training": "P-CNN reconstructor trains on model_train and selects on model_val only.",
            "prediction": "Prediction consumes cached fixed-HLT views only; offline constituents are not loaded.",
            "final_test": "final_test is included only when confirm_final_test=True.",
        },
    }


def run_teacher_logit_particle_cnn_first_experiment(
    config: TeacherLogitParticleCnnFirstExperimentConfig,
) -> Dict[str, Any]:
    """Train P-CNN -> frozen teacher, collect prediction blocks, and write a report."""

    if bool(config.fit_final_stacker):
        raise NotImplementedError(
            "Step 7 intentionally does not fit a final stacker. Use the saved prediction blocks "
            "with the independent fusion path once Step 8 runners are in place."
        )

    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(config.overwrite_output):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TeacherLogitParticleCnnTrainConfig(
        output_dir=str(config.train_output_dir),
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        data_dir=config.data_dir,
        teacher_checkpoint=config.teacher_checkpoint,
        teacher_architecture=config.teacher_architecture,
        seed=config.seed,
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        num_workers=config.num_workers,
        device=config.device,
        amp=config.amp,
        grad_clip_norm=config.grad_clip_norm,
        early_stop_patience=config.early_stop_patience,
        max_train_batches=config.max_train_batches,
        max_val_batches=config.max_val_batches,
        max_train_jets=config.max_train_jets,
        max_val_jets=config.max_val_jets,
        verify_hlt_hash=config.verify_hlt_hash,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
        compile_model=config.compile_model,
        max_constits=config.max_constits,
        teacher_weight_threshold=config.teacher_weight_threshold,
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        kernel_sizes=config.kernel_sizes,
        dilations=config.dilations,
        context_dim=config.context_dim,
        context_mlp_dims=config.context_mlp_dims,
        decoder_dims=config.decoder_dims,
        slot_dim=config.slot_dim,
        num_extra_candidates=config.num_extra_candidates,
        dropout=config.dropout,
        max_delta_logpt=config.max_delta_logpt,
        max_delta_eta=config.max_delta_eta,
        max_delta_phi=config.max_delta_phi,
        max_delta_loge=config.max_delta_loge,
        parent_weight_bias=config.parent_weight_bias,
        extra_weight_bias=config.extra_weight_bias,
        max_total_extra_pt_fraction=config.max_total_extra_pt_fraction,
        max_extra_delta_eta=config.max_extra_delta_eta,
        max_extra_delta_phi=config.max_extra_delta_phi,
        teacher_kl_weight=config.teacher_kl_weight,
        ce_weight=config.ce_weight,
        correction_budget_weight=config.correction_budget_weight,
        jet_summary_weight=config.jet_summary_weight,
        temperature=config.temperature,
    )
    train_report = train_teacher_logit_particle_cnn_reco(train_config)

    predict_config = TeacherLogitParticleCnnPredictionConfig(
        output_dir=str(config.prediction_output_dir),
        prediction_dir=str(config.prediction_dir),
        hlt_cache_dir=config.hlt_cache_dir,
        reconstructor_checkpoint=train_report["checkpoint"],
        teacher_checkpoint=config.teacher_checkpoint,
        teacher_architecture=config.teacher_architecture,
        model_name=config.resolved_model_name,
        splits=list(config.splits),
        batch_size=config.predict_batch_size,
        num_workers=config.predict_num_workers,
        device=config.predict_device or config.device,
        amp=config.predict_amp,
        max_jets_per_split=config.max_prediction_jets_per_split,
        overwrite_predictions=config.overwrite_predictions,
        skip_existing_predictions=config.skip_existing_predictions,
        confirm_final_test=config.confirm_final_test,
        max_constits=config.max_constits,
        teacher_weight_threshold=config.teacher_weight_threshold,
    )
    prediction_report = collect_teacher_logit_particle_cnn_predictions(predict_config)
    report = build_first_experiment_report(
        config=config,
        train_report=train_report,
        prediction_report=prediction_report,
    )
    save_json(output_dir / "first_experiment_report.json", report)
    return report


__all__ = [
    "DEFAULT_FIRST_EXPERIMENT_SPLITS",
    "EXPERIMENT_STEP",
    "PredictionComparisonSpec",
    "TeacherLogitParticleCnnFirstExperimentConfig",
    "build_first_experiment_report",
    "comparison_delta_summary",
    "prediction_metric_summary",
    "run_teacher_logit_particle_cnn_first_experiment",
]
