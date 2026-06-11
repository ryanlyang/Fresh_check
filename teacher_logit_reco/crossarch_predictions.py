"""Step 6 prediction-block helpers for the cross-architecture experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .crossarch_experiment import (
    DIRECT_HLT_ARCHITECTURES,
    EXPERIMENT_NAME,
    RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
    CrossArchExperimentLayout,
    hlt_model_name,
    normalize_reconstructor_architecture,
    normalize_teacher_architecture,
    reco_model_name,
)
EXPERIMENT_STEP = "crossarch_step6_prediction_blocks"
RECO_PREDICT_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:teacher_logit_reco"
HLT_PREDICT_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:direct_hlt"

RECONSTRUCTOR_PREDICT_SCRIPTS: dict[str, str] = {
    "gt": "scripts/predict_teacher_logit_global_transformer_reco.py",
    "pn": "scripts/predict_teacher_logit_particle_net_reco.py",
    "pfn": "scripts/predict_teacher_logit_particle_flow_reco.py",
    "pcnn": "scripts/predict_teacher_logit_particle_cnn_reco.py",
}
HLT_PREDICT_SCRIPT = "scripts/predict_crossarch_hlt_baseline.py"


@dataclass(frozen=True)
class CrossArchPredictionSpec:
    """One source that should write stack/final-test prediction blocks."""

    model_name: str
    source_kind: str
    checkpoint: Path
    prediction_source_dir: Path
    run_output_dir: Path
    predict_script: str
    reco_architecture: str | None = None
    teacher_architecture: str | None = None
    hlt_architecture: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_step": EXPERIMENT_STEP,
            "model_name": self.model_name,
            "source_kind": self.source_kind,
            "checkpoint": str(self.checkpoint),
            "prediction_source_dir": str(self.prediction_source_dir),
            "run_output_dir": str(self.run_output_dir),
            "predict_script": self.predict_script,
            "reco_architecture": self.reco_architecture,
            "teacher_architecture": self.teacher_architecture,
            "hlt_architecture": self.hlt_architecture,
        }


def predict_script_for_reconstructor(reco_architecture: str) -> str:
    reco = normalize_reconstructor_architecture(reco_architecture)
    return RECONSTRUCTOR_PREDICT_SCRIPTS[reco]


def crossarch_reco_prediction_spec(
    reco_architecture: str,
    teacher_architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> CrossArchPredictionSpec:
    reco = normalize_reconstructor_architecture(reco_architecture)
    teacher = normalize_teacher_architecture(teacher_architecture)
    layout = CrossArchExperimentLayout(output_root=output_root)
    model_name = reco_model_name(reco, teacher)
    return CrossArchPredictionSpec(
        model_name=model_name,
        source_kind="teacher_logit_reco",
        checkpoint=layout.reco_model_checkpoint(reco, teacher),
        prediction_source_dir=layout.prediction_source_dir(model_name),
        run_output_dir=layout.prediction_runs_dir / "reco" / model_name,
        predict_script=predict_script_for_reconstructor(reco),
        reco_architecture=reco,
        teacher_architecture=teacher,
    )


def crossarch_hlt_prediction_spec(
    architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> CrossArchPredictionSpec:
    arch = normalize_teacher_architecture(architecture)
    layout = CrossArchExperimentLayout(output_root=output_root)
    model_name = hlt_model_name(arch)
    return CrossArchPredictionSpec(
        model_name=model_name,
        source_kind="direct_hlt",
        checkpoint=layout.hlt_baseline_checkpoint(arch),
        prediction_source_dir=layout.prediction_source_dir(model_name),
        run_output_dir=layout.prediction_runs_dir / "hlt" / model_name,
        predict_script=HLT_PREDICT_SCRIPT,
        hlt_architecture=arch,
    )


def build_crossarch_reco_prediction_specs(
    reconstructors: Iterable[str] = RECONSTRUCTOR_ARCHITECTURES,
    teachers: Iterable[str] = TEACHER_ARCHITECTURES,
    *,
    output_root: str | Path = "checkpoints",
) -> tuple[CrossArchPredictionSpec, ...]:
    return tuple(
        crossarch_reco_prediction_spec(reco, teacher, output_root=output_root)
        for reco in reconstructors
        for teacher in teachers
    )


def build_crossarch_hlt_prediction_specs(
    architectures: Iterable[str] = DIRECT_HLT_ARCHITECTURES,
    *,
    output_root: str | Path = "checkpoints",
) -> tuple[CrossArchPredictionSpec, ...]:
    return tuple(crossarch_hlt_prediction_spec(architecture, output_root=output_root) for architecture in architectures)


def build_crossarch_prediction_specs(
    *,
    include_hlt: bool = True,
    output_root: str | Path = "checkpoints",
) -> tuple[CrossArchPredictionSpec, ...]:
    reco_specs = build_crossarch_reco_prediction_specs(output_root=output_root)
    if not include_hlt:
        return reco_specs
    return reco_specs + build_crossarch_hlt_prediction_specs(output_root=output_root)


__all__ = [
    "EXPERIMENT_STEP",
    "HLT_PREDICT_EXPERIMENT_STEP",
    "HLT_PREDICT_SCRIPT",
    "RECONSTRUCTOR_PREDICT_SCRIPTS",
    "RECO_PREDICT_EXPERIMENT_STEP",
    "CrossArchPredictionSpec",
    "build_crossarch_hlt_prediction_specs",
    "build_crossarch_prediction_specs",
    "build_crossarch_reco_prediction_specs",
    "crossarch_hlt_prediction_spec",
    "crossarch_reco_prediction_spec",
    "predict_script_for_reconstructor",
]
