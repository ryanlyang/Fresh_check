"""Prediction collection for aggressive teacher-logit reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
from pathlib import Path
from typing import Any, Dict, Mapping

from jetclass_fresh.fusion import STACK_SPLITS, load_prediction_block, prediction_paths, save_prediction_block
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetView

from .reconstructor_builders import (
    infer_reconstructor_architecture_from_payload,
    load_teacher_logit_reconstructor_checkpoint,
    normalize_reconstructor_architecture,
)
from .teachers import assert_teacher_frozen, load_frozen_teacher, normalize_teacher_architecture
from .train_aggressive_reconstructors import (
    AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES,
    EXPERIMENT_STEP as TRAIN_EXPERIMENT_STEP,
)
from .train_global_transformer import source_metadata
from .predict_global_transformer import (
    evaluate_teacher_logit_reco_model,
    teacher_architecture_from_payload,
    teacher_checkpoint_from_payload,
)


PREDICT_EXPERIMENT_STEP = "teacher_logit_reco_step9_aggressive_predictions"

AGGRESSIVE_MODEL_NAME_PREFIXES = {
    "aggressive_global_transformer": "aggt",
    "aggressive_particle_net": "agpn",
    "aggressive_particle_flow": "agpfn",
    "aggressive_particle_cnn": "agpcnn",
}


@dataclass
class TeacherLogitAggressivePredictionConfig:
    """Configuration for Step 9 aggressive prediction block generation."""

    output_dir: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    prediction_dir: str | None = None
    teacher_checkpoint: str | None = None
    teacher_architecture: str | None = None
    reco_architecture: str | None = None
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
        if self.reco_architecture is not None:
            arch = normalize_reconstructor_architecture(self.reco_architecture)
            if arch not in AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES:
                expected = ", ".join(AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES)
                raise ValueError(
                    f"Step 9 aggressive prediction requires an aggressive architecture; "
                    f"got {self.reco_architecture!r}, expected one of: {expected}"
                )

    @property
    def resolved_prediction_dir(self) -> Path:
        if self.prediction_dir is not None:
            return Path(self.prediction_dir)
        return Path(self.output_dir) / "predictions"

    @property
    def expected_reco_architecture(self) -> str | None:
        if self.reco_architecture is None:
            return None
        return normalize_reconstructor_architecture(self.reco_architecture)


def validate_aggressive_reconstructor_architecture(architecture: str | None) -> str:
    arch = normalize_reconstructor_architecture(architecture)
    if arch not in AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES:
        expected = ", ".join(AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES)
        raise ValueError(f"Expected aggressive reconstructor architecture, found {arch!r}; expected one of: {expected}")
    return arch


def load_aggressive_reconstructor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device,
    strict: bool = True,
    expected_architecture: str | None = None,
):
    """Load any Step 8 aggressive reconstructor checkpoint."""

    expected = None
    if expected_architecture is not None:
        expected = validate_aggressive_reconstructor_architecture(expected_architecture)
    model, payload = load_teacher_logit_reconstructor_checkpoint(
        checkpoint_path,
        device=device,
        strict=bool(strict),
        expected_architecture=expected,
    )
    arch = validate_aggressive_reconstructor_architecture(
        infer_reconstructor_architecture_from_payload(payload, architecture=expected)
    )
    return model, payload, arch


def default_model_name_for_reco_teacher_architecture(
    reco_architecture: str,
    teacher_architecture: str | None,
) -> str:
    arch = validate_aggressive_reconstructor_architecture(reco_architecture)
    teacher_arch = normalize_teacher_architecture(teacher_architecture)
    return f"{AGGRESSIVE_MODEL_NAME_PREFIXES[arch]}_reco_to_{teacher_arch}_teacher"


def collect_teacher_logit_aggressive_predictions(
    config: TeacherLogitAggressivePredictionConfig,
    *,
    reconstructor=None,
    teacher=None,
    hlt_views: Mapping[str, JetView] | None = None,
) -> Dict[str, Any]:
    """Generate fusion-compatible prediction blocks from aggressive reconstructors."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.resolved_prediction_dir
    reports: Dict[str, Any] = {}

    payload: Dict[str, Any] = {}
    if reconstructor is None:
        reconstructor, payload, reco_architecture = load_aggressive_reconstructor_checkpoint(
            config.reconstructor_checkpoint,
            device=device,
            strict=bool(config.strict_checkpoint),
            expected_architecture=config.expected_reco_architecture,
        )
    else:
        reconstructor = reconstructor.to(device).eval()
        reco_architecture = validate_aggressive_reconstructor_architecture(config.expected_reco_architecture)

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

    model_name = config.model_name or default_model_name_for_reco_teacher_architecture(
        reco_architecture,
        teacher.metadata.get("architecture"),
    )
    checkpoint_metadata = {
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "model_kind": "teacher_logit_aggressive_reco",
        "training_step": TRAIN_EXPERIMENT_STEP,
        "aggression_level": payload.get("aggression_level"),
        "reconstructor_checkpoint": str(config.reconstructor_checkpoint),
        "reconstructor_architecture": reco_architecture,
        "reconstructor_checkpoint_epoch": payload.get("epoch"),
        "reconstructor_experiment_step": payload.get("experiment_step"),
        "reconstructor_model_config": dict(payload.get("model_config") or {}),
        "reconstructor_loss_config": dict(payload.get("loss_config") or {}),
        "teacher_checkpoint": teacher.metadata.get("checkpoint_path"),
        "allowed_inputs": "cached_fixed_hlt_only_then_aggressive_reconstructed_soft_view_to_frozen_teacher",
        "source": source_metadata(),
    }

    reports[model_name] = {}
    for split in list(config.splits):
        npz_path, _ = prediction_paths(prediction_dir, model_name, split)
        if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
            reports[model_name][split] = load_prediction_block(prediction_dir, model_name, split).metadata
            continue
        view = hlt_views[split] if hlt_views is not None and split in hlt_views else load_cached_hlt_view(config.hlt_cache_dir, split)
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
        "reconstructor_architecture": reco_architecture,
        "aggression_level": payload.get("aggression_level"),
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "splits": list(config.splits),
        "config": asdict(config),
        "teacher": dict(teacher.metadata),
        "reports": reports,
        "leakage_rule": (
            "Prediction generation loads cached fixed-HLT views only. Offline constituents are not loaded; "
            "the frozen teacher sees only the aggressive reconstructed soft view produced from HLT tokens."
        ),
    }
    save_json(output_dir / "prediction_collection_report.json", output)
    return output


__all__ = [
    "AGGRESSIVE_MODEL_NAME_PREFIXES",
    "PREDICT_EXPERIMENT_STEP",
    "TeacherLogitAggressivePredictionConfig",
    "collect_teacher_logit_aggressive_predictions",
    "default_model_name_for_reco_teacher_architecture",
    "load_aggressive_reconstructor_checkpoint",
    "validate_aggressive_reconstructor_architecture",
]
