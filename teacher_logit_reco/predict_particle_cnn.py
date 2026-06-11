"""Prediction collection for teacher-logit P-CNN-style reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
from pathlib import Path
from typing import Any, Dict, Mapping

from jetclass_fresh.fusion import STACK_SPLITS, load_prediction_block, prediction_paths, save_prediction_block
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetView

from .particle_cnn_reconstructor import PARTICLE_CNN_ORDERING_ASSUMPTION, ParticleCnnReconstructor
from .predict_global_transformer import (
    evaluate_teacher_logit_reco_model,
    teacher_architecture_from_payload,
    teacher_checkpoint_from_payload,
)
from .reconstructor_builders import (
    infer_reconstructor_architecture_from_payload,
    load_teacher_logit_reconstructor_checkpoint,
)
from .teachers import assert_teacher_frozen, load_frozen_teacher, normalize_teacher_architecture
from .train_global_transformer import source_metadata
from .train_particle_cnn import EXPERIMENT_STEP as TRAIN_EXPERIMENT_STEP


PREDICT_EXPERIMENT_STEP = "teacher_logit_reco_step6_particle_cnn_predictions"
RECONSTRUCTOR_ARCHITECTURE = "particle_cnn"


@dataclass
class TeacherLogitParticleCnnPredictionConfig:
    """Configuration for P-CNN Step 6 prediction block generation."""

    output_dir: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    prediction_dir: str | None = None
    teacher_checkpoint: str | None = None
    teacher_architecture: str | None = None
    model_name: str | None = None
    splits: list[str] = field(default_factory=lambda: list(STACK_SPLITS))
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    max_jets_per_split: int | None = None
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    strict_checkpoint: bool = True

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_jets_per_split is not None and int(self.max_jets_per_split) < 0:
            raise ValueError("max_jets_per_split must be non-negative when provided")
        if "final_test" in list(self.splits) and not bool(self.confirm_final_test):
            raise ValueError("Refusing to generate final_test predictions without confirm_final_test=True")

    @property
    def resolved_prediction_dir(self) -> Path:
        if self.prediction_dir is not None:
            return Path(self.prediction_dir)
        return Path(self.output_dir) / "predictions"


def load_particle_cnn_reconstructor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device,
    strict: bool = True,
) -> tuple[ParticleCnnReconstructor, Dict[str, Any]]:
    """Load a Step 5 P-CNN-style reconstructor checkpoint."""

    model, payload = load_teacher_logit_reconstructor_checkpoint(
        checkpoint_path,
        device=device,
        strict=bool(strict),
        expected_architecture=RECONSTRUCTOR_ARCHITECTURE,
    )
    return model, payload


def default_model_name_for_teacher_architecture(architecture: str | None) -> str:
    arch = normalize_teacher_architecture(architecture)
    return f"pcnn_reco_to_{arch}_teacher"


def collect_teacher_logit_particle_cnn_predictions(
    config: TeacherLogitParticleCnnPredictionConfig,
    *,
    reconstructor=None,
    teacher=None,
    hlt_views: Mapping[str, JetView] | None = None,
) -> Dict[str, Any]:
    """Generate fusion-compatible prediction blocks for requested splits."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.resolved_prediction_dir
    reports: Dict[str, Any] = {}

    payload: Dict[str, Any] = {}
    if reconstructor is None:
        reconstructor, payload = load_particle_cnn_reconstructor_checkpoint(
            config.reconstructor_checkpoint,
            device=device,
            strict=bool(config.strict_checkpoint),
        )
    else:
        reconstructor = reconstructor.to(device).eval()

    if teacher is None:
        teacher_checkpoint = teacher_checkpoint_from_payload(payload, override_checkpoint=config.teacher_checkpoint)
        teacher_architecture = teacher_architecture_from_payload(
            payload,
            override_architecture=config.teacher_architecture,
        )
        teacher = load_frozen_teacher(
            teacher_checkpoint,
            architecture=teacher_architecture,
            device=str(device),
            max_constits=int(config.max_constits),
            weight_threshold=float(config.teacher_weight_threshold),
        )
    else:
        teacher.model = teacher.model.to(device).eval()
        teacher.device = device
        assert_teacher_frozen(teacher)

    model_name = config.model_name or default_model_name_for_teacher_architecture(teacher.metadata.get("architecture"))
    checkpoint_metadata = {
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "model_kind": "teacher_logit_particle_cnn_reco",
        "training_step": TRAIN_EXPERIMENT_STEP,
        "reconstructor_checkpoint": str(config.reconstructor_checkpoint),
        "reconstructor_architecture": infer_reconstructor_architecture_from_payload(
            payload,
            architecture=RECONSTRUCTOR_ARCHITECTURE,
        ),
        "reconstructor_ordering_assumption": payload.get("ordering_assumption", PARTICLE_CNN_ORDERING_ASSUMPTION),
        "reconstructor_checkpoint_epoch": payload.get("epoch"),
        "reconstructor_experiment_step": payload.get("experiment_step"),
        "reconstructor_model_config": dict(payload.get("model_config") or {}),
        "reconstructor_loss_config": dict(payload.get("loss_config") or {}),
        "teacher_checkpoint": teacher.metadata.get("checkpoint_path"),
        "allowed_inputs": "cached_fixed_hlt_only_then_reconstructed_soft_view_to_frozen_teacher",
        "source": source_metadata(),
    }

    reports[model_name] = {}
    for split in list(config.splits):
        npz_path, _ = prediction_paths(prediction_dir, model_name, split)
        if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
            reports[model_name][split] = load_prediction_block(prediction_dir, model_name, split).metadata
            continue
        view = hlt_views[split] if hlt_views is not None and split in hlt_views else load_cached_hlt_view(
            config.hlt_cache_dir,
            split,
        )
        block = evaluate_teacher_logit_reco_model(
            model_name,
            reconstructor,
            teacher,
            view,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            amp=config.amp,
            max_jets=config.max_jets_per_split,
            checkpoint_metadata=checkpoint_metadata,
        )
        reports[model_name][split] = save_prediction_block(
            block,
            prediction_dir,
            overwrite=bool(config.overwrite_predictions),
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    output = {
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "reconstructor_architecture": RECONSTRUCTOR_ARCHITECTURE,
        "ordering_assumption": PARTICLE_CNN_ORDERING_ASSUMPTION,
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "splits": list(config.splits),
        "config": asdict(config),
        "teacher": dict(teacher.metadata),
        "reports": reports,
        "leakage_rule": (
            "Prediction generation loads cached fixed-HLT views only. Offline constituents are not loaded; "
            "the frozen teacher sees only the reconstructed soft view produced from HLT tokens. "
            f"The P-CNN rank axis assumes {PARTICLE_CNN_ORDERING_ASSUMPTION}."
        ),
    }
    save_json(output_dir / "prediction_collection_report.json", output)
    return output


__all__ = [
    "PREDICT_EXPERIMENT_STEP",
    "RECONSTRUCTOR_ARCHITECTURE",
    "TeacherLogitParticleCnnPredictionConfig",
    "collect_teacher_logit_particle_cnn_predictions",
    "default_model_name_for_teacher_architecture",
    "load_particle_cnn_reconstructor_checkpoint",
]
