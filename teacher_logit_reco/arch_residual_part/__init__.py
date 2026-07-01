"""Architecture-residual experts on top of a frozen HLT ParT baseline."""

from .model import (
    ARCH_RESIDUAL_ARCHITECTURES,
    ARCH_RESIDUAL_MODEL_CONTRACT,
    ARCH_RESIDUAL_MODEL_STEP,
    ArchResidualExpertConfig,
    ArchResidualPartModel,
    ArchResidualPartOutput,
    build_arch_residual_part_model,
)
from .train import (
    ARCH_RESIDUAL_TRAIN_CONTRACT,
    ARCH_RESIDUAL_TRAIN_STEP,
    ArchResidualTrainConfig,
    train_arch_residual_tagger,
)

__all__ = [
    "ARCH_RESIDUAL_ARCHITECTURES",
    "ARCH_RESIDUAL_MODEL_CONTRACT",
    "ARCH_RESIDUAL_MODEL_STEP",
    "ARCH_RESIDUAL_TRAIN_CONTRACT",
    "ARCH_RESIDUAL_TRAIN_STEP",
    "ArchResidualExpertConfig",
    "ArchResidualPartModel",
    "ArchResidualPartOutput",
    "ArchResidualTrainConfig",
    "build_arch_residual_part_model",
    "train_arch_residual_tagger",
]
