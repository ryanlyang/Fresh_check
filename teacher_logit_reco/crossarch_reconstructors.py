"""Step 5 helpers for cross-architecture teacher-logit reconstructors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .crossarch_experiment import (
    EXPERIMENT_NAME,
    RECONSTRUCTOR_ARCHITECTURES,
    RECONSTRUCTOR_IMPLEMENTATIONS,
    TEACHER_ARCHITECTURES,
    CrossArchExperimentLayout,
    normalize_reconstructor_architecture,
    normalize_teacher_architecture,
    reco_model_name,
)


EXPERIMENT_STEP = "crossarch_step5_teacher_logit_reconstructors"
TRAIN_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:train"

RECONSTRUCTOR_TRAIN_SCRIPTS: dict[str, str] = {
    "gt": "scripts/train_teacher_logit_global_transformer_reco.py",
    "pn": "scripts/train_teacher_logit_particle_net_reco.py",
    "pfn": "scripts/train_teacher_logit_particle_flow_reco.py",
    "pcnn": "scripts/train_teacher_logit_particle_cnn_reco.py",
}


@dataclass(frozen=True)
class CrossArchReconstructorSpec:
    """One Step 5 training target in the 4x4 reco/teacher grid."""

    reco_architecture: str
    teacher_architecture: str
    model_name: str
    output_dir: Path
    teacher_checkpoint: Path
    train_script: str
    reconstructor_implementation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_step": TRAIN_EXPERIMENT_STEP,
            "reco_architecture": self.reco_architecture,
            "teacher_architecture": self.teacher_architecture,
            "model_name": self.model_name,
            "output_dir": str(self.output_dir),
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "train_script": self.train_script,
            "reconstructor_implementation": self.reconstructor_implementation,
        }


def train_script_for_reconstructor(reco_architecture: str) -> str:
    """Return the CLI script that trains one reconstructor architecture."""

    reco = normalize_reconstructor_architecture(reco_architecture)
    return RECONSTRUCTOR_TRAIN_SCRIPTS[reco]


def crossarch_reconstructor_model_name(reco_architecture: str, teacher_architecture: str) -> str:
    return reco_model_name(reco_architecture, teacher_architecture)


def crossarch_reconstructor_dir(
    reco_architecture: str,
    teacher_architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    return CrossArchExperimentLayout(output_root=output_root).reco_model_dir(
        reco_architecture,
        teacher_architecture,
    )


def crossarch_reconstructor_checkpoint(
    reco_architecture: str,
    teacher_architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    return crossarch_reconstructor_dir(
        reco_architecture,
        teacher_architecture,
        output_root=output_root,
    ) / "best_model_val.pt"


def crossarch_teacher_checkpoint_path(
    teacher_architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    return CrossArchExperimentLayout(output_root=output_root).offline_teacher_checkpoint(teacher_architecture)


def build_crossarch_reconstructor_spec(
    reco_architecture: str,
    teacher_architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> CrossArchReconstructorSpec:
    reco = normalize_reconstructor_architecture(reco_architecture)
    teacher = normalize_teacher_architecture(teacher_architecture)
    layout = CrossArchExperimentLayout(output_root=output_root)
    return CrossArchReconstructorSpec(
        reco_architecture=reco,
        teacher_architecture=teacher,
        model_name=reco_model_name(reco, teacher),
        output_dir=layout.reco_model_dir(reco, teacher),
        teacher_checkpoint=layout.offline_teacher_checkpoint(teacher),
        train_script=train_script_for_reconstructor(reco),
        reconstructor_implementation=RECONSTRUCTOR_IMPLEMENTATIONS[reco],
    )


def build_crossarch_reconstructor_specs(
    reconstructors: Iterable[str] = RECONSTRUCTOR_ARCHITECTURES,
    teachers: Iterable[str] = TEACHER_ARCHITECTURES,
    *,
    output_root: str | Path = "checkpoints",
) -> tuple[CrossArchReconstructorSpec, ...]:
    specs: list[CrossArchReconstructorSpec] = []
    for reco_architecture in reconstructors:
        reco = normalize_reconstructor_architecture(reco_architecture)
        for teacher_architecture in teachers:
            teacher = normalize_teacher_architecture(teacher_architecture)
            specs.append(
                build_crossarch_reconstructor_spec(
                    reco,
                    teacher,
                    output_root=output_root,
                )
            )
    return tuple(specs)


__all__ = [
    "EXPERIMENT_STEP",
    "RECONSTRUCTOR_TRAIN_SCRIPTS",
    "TRAIN_EXPERIMENT_STEP",
    "CrossArchReconstructorSpec",
    "build_crossarch_reconstructor_spec",
    "build_crossarch_reconstructor_specs",
    "crossarch_reconstructor_checkpoint",
    "crossarch_reconstructor_dir",
    "crossarch_reconstructor_model_name",
    "crossarch_teacher_checkpoint_path",
    "train_script_for_reconstructor",
]
