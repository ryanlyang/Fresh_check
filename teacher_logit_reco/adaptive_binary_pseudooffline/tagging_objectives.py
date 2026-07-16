"""Fail-closed Step-11 tagging objectives for the ABPH campaign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .config import canonical_hash
from .tagger import HierarchyAwareTaggerOutput
from .variants import resolve_variant_config


ABPH_TAGGING_OBJECTIVE_CONTRACT = "adaptive_binary_pseudooffline_tagging_objective_v1"
ABPH_TEACHER_LOGIT_SPLITS: tuple[str, ...] = ("model_train", "model_val")


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised on compute nodes
        raise RuntimeError("PyTorch is required for ABPH tagging objectives") from exc
    return torch, functional


@dataclass(frozen=True)
class TaggingObjectiveConfig:
    """Resolved, hashable loss contract for one E/F-tier tagging run."""

    variant_name: str
    resolved_config_hash: str
    label_ce: float = 1.0
    label_smoothing: float = 0.02
    hlt_anchor_ce: float = 0.20
    joint_reconstruction: float = 0.10
    pseudo_aux_ce: float = 0.0
    hierarchy_aux_ce: float = 0.0
    cross_view_agreement: float = 0.0
    offline_logit_kd: float = 0.0
    kd_temperature: float = 2.0

    def __post_init__(self) -> None:
        nonnegative = (
            "label_ce",
            "label_smoothing",
            "hlt_anchor_ce",
            "joint_reconstruction",
            "pseudo_aux_ce",
            "hierarchy_aux_ce",
            "cross_view_agreement",
            "offline_logit_kd",
        )
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0,1)")
        if float(self.kd_temperature) <= 0.0:
            raise ValueError("kd_temperature must be positive")
        if float(self.label_ce) <= 0.0:
            raise ValueError("label CE must remain active")

    @property
    def requires_teacher_logits(self) -> bool:
        return float(self.offline_logit_kd) > 0.0

    @property
    def required_auxiliary_logits(self) -> tuple[str, ...]:
        required: list[str] = []
        if self.pseudo_aux_ce > 0.0:
            required.append("pseudo")
        if self.hierarchy_aux_ce > 0.0:
            required.append("hierarchy")
        return tuple(required)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": ABPH_TAGGING_OBJECTIVE_CONTRACT,
            "variant_name": self.variant_name,
            "resolved_config_hash": self.resolved_config_hash,
            "label_ce": float(self.label_ce),
            "label_smoothing": float(self.label_smoothing),
            "hlt_anchor_ce": float(self.hlt_anchor_ce),
            "joint_reconstruction": float(self.joint_reconstruction),
            "pseudo_aux_ce": float(self.pseudo_aux_ce),
            "hierarchy_aux_ce": float(self.hierarchy_aux_ce),
            "cross_view_agreement": float(self.cross_view_agreement),
            "offline_logit_kd": float(self.offline_logit_kd),
            "kd_temperature": float(self.kd_temperature),
            "requires_teacher_logits": self.requires_teacher_logits,
            "teacher_logit_splits": list(ABPH_TEACHER_LOGIT_SPLITS),
            "required_auxiliary_logits": list(self.required_auxiliary_logits),
        }
        payload["objective_hash"] = canonical_hash(payload)
        return payload


def tagging_objective_config(variant: str) -> TaggingObjectiveConfig:
    """Resolve registry objective weights and verify data/loss declarations agree."""

    resolved = resolve_variant_config(variant)
    objective = dict(resolved["training"]["objective"])
    config = TaggingObjectiveConfig(
        variant_name=str(resolved["variant"]["name"]),
        resolved_config_hash=str(resolved["resolved_config_hash"]),
        label_ce=float(objective.get("label_ce", 1.0)),
        label_smoothing=float(objective.get("label_smoothing", 0.0)),
        hlt_anchor_ce=float(objective.get("hlt_anchor_ce", 0.0)),
        joint_reconstruction=float(objective.get("joint_reconstruction", 0.0)),
        pseudo_aux_ce=float(objective.get("pseudo_aux_ce", 0.0)),
        hierarchy_aux_ce=float(objective.get("hierarchy_aux_ce", 0.0)),
        cross_view_agreement=float(objective.get("cross_view_agreement", 0.0)),
        offline_logit_kd=float(objective.get("offline_logit_kd", 0.0)),
        kd_temperature=float(objective.get("kd_temperature", 2.0)),
    )
    declared = bool(resolved["data"].get("requires_teacher_logits", False))
    if declared != config.requires_teacher_logits:
        raise ValueError(
            f"{config.variant_name} teacher-logit data declaration disagrees with its KD weight"
        )
    return config


def teacher_logits_for_objective(
    config: TaggingObjectiveConfig,
    split: str,
    loader: Callable[[str], Any] | None,
) -> Any | None:
    """Load privileged logits only for an explicit KD training/validation recipe."""

    split_name = str(split)
    if not config.requires_teacher_logits:
        return None
    if split_name not in ABPH_TEACHER_LOGIT_SPLITS:
        raise ValueError(
            f"teacher logits are forbidden for {split_name}; KD checkpoints evaluate teacher-free"
        )
    if loader is None:
        raise FileNotFoundError(
            f"{config.variant_name} requires offline teacher logits on {split_name}"
        )
    logits = loader(split_name)
    if logits is None:
        raise FileNotFoundError(
            f"{config.variant_name} teacher-logit loader returned no {split_name} data"
        )
    return logits


@dataclass(frozen=True)
class TaggingObjectiveOutput:
    total: Any
    raw_terms: Mapping[str, Any]
    weighted_terms: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def _scalar(value: Any, name: str, *, reference: Any) -> Any:
    torch, _ = _require_torch()
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not bool(torch.isfinite(tensor)):
        raise FloatingPointError(f"{name} is nonfinite")
    return tensor


def compute_tagging_objective(
    output: HierarchyAwareTaggerOutput,
    labels: Any,
    config: TaggingObjectiveConfig,
    *,
    split: str,
    reconstruction_loss: Any | None = None,
    auxiliary_logits: Mapping[str, Any] | None = None,
    teacher_logits: Any | None = None,
) -> TaggingObjectiveOutput:
    """Compose the paired F-tier objectives without optional-loss fallthrough."""

    torch, functional = _require_torch()
    split_name = str(split)
    if split_name == "final_test" and teacher_logits is not None:
        raise ValueError("final-test tagging is unconditionally teacher-logit free")
    if config.requires_teacher_logits:
        if split_name not in ABPH_TEACHER_LOGIT_SPLITS:
            if teacher_logits is not None:
                raise ValueError("KD supervision is forbidden outside model_train/model_val")
        elif teacher_logits is None:
            raise FileNotFoundError(
                f"{config.variant_name} requires teacher logits on {split_name}"
            )
    elif teacher_logits is not None:
        raise ValueError(
            f"non-KD variant {config.variant_name} must not load teacher logits"
        )

    target = torch.as_tensor(labels, device=output.logits.device, dtype=torch.long)
    if target.ndim != 1 or output.logits.ndim != 2 or output.logits.shape[0] != target.shape[0]:
        raise ValueError("tagging logits and labels have incompatible shapes")
    if output.baseline_logits.shape != output.logits.shape:
        raise ValueError("HLT anchor logits differ from fused logits")

    raw: dict[str, Any] = {
        "label_ce": functional.cross_entropy(
            output.logits, target, label_smoothing=float(config.label_smoothing)
        )
    }
    if config.hlt_anchor_ce > 0.0:
        raw["hlt_anchor_ce"] = functional.cross_entropy(
            output.baseline_logits,
            target,
            label_smoothing=float(config.label_smoothing),
        )
    if config.joint_reconstruction > 0.0:
        if reconstruction_loss is None:
            raise KeyError("active joint reconstruction loss is missing")
        raw["joint_reconstruction"] = _scalar(
            reconstruction_loss, "joint_reconstruction", reference=output.logits
        )

    branches = dict(auxiliary_logits or {})
    for branch in config.required_auxiliary_logits:
        if branch not in branches:
            raise KeyError(f"active {branch} auxiliary logits are missing")
        if branches[branch].shape != output.logits.shape:
            raise ValueError(f"{branch} auxiliary logits have the wrong shape")
        raw[f"{branch}_aux_ce"] = functional.cross_entropy(
            branches[branch], target, label_smoothing=float(config.label_smoothing)
        )
    if config.cross_view_agreement > 0.0:
        if "pseudo" not in branches:
            raise KeyError("cross-view agreement requires pseudo branch logits")
        fused_log_prob = functional.log_softmax(output.logits, dim=-1)
        pseudo_prob = functional.softmax(branches["pseudo"].detach(), dim=-1)
        raw["cross_view_agreement"] = functional.kl_div(
            fused_log_prob, pseudo_prob, reduction="batchmean"
        )
    if config.offline_logit_kd > 0.0 and split_name in ABPH_TEACHER_LOGIT_SPLITS:
        if teacher_logits is None:  # guarded above; keeps type narrowing explicit
            raise FileNotFoundError("teacher logits disappeared before KD composition")
        teacher = torch.as_tensor(
            teacher_logits, device=output.logits.device, dtype=output.logits.dtype
        )
        if teacher.shape != output.logits.shape:
            raise ValueError("teacher and student logits have incompatible shapes")
        temperature = float(config.kd_temperature)
        raw["offline_logit_kd"] = functional.kl_div(
            functional.log_softmax(output.logits / temperature, dim=-1),
            functional.softmax(teacher.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature * temperature

    weights = {
        "label_ce": config.label_ce,
        "hlt_anchor_ce": config.hlt_anchor_ce,
        "joint_reconstruction": config.joint_reconstruction,
        "pseudo_aux_ce": config.pseudo_aux_ce,
        "hierarchy_aux_ce": config.hierarchy_aux_ce,
        "cross_view_agreement": config.cross_view_agreement,
        "offline_logit_kd": config.offline_logit_kd,
    }
    weighted = {name: value * float(weights[name]) for name, value in raw.items()}
    total = sum(weighted.values(), output.logits.new_zeros(()))
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("composed tagging objective is nonfinite")
    teacher_loaded = teacher_logits is not None
    return TaggingObjectiveOutput(
        total=total,
        raw_terms=raw,
        weighted_terms=weighted,
        diagnostics={
            "contract": ABPH_TAGGING_OBJECTIVE_CONTRACT,
            "variant_name": config.variant_name,
            "objective_hash": config.to_dict()["objective_hash"],
            "split": split_name,
            "teacher_logits_required": config.requires_teacher_logits,
            "teacher_logits_loaded": teacher_loaded,
            "representation_kd_enabled": False,
            "required_terms": list(raw),
        },
    )


__all__ = [
    "ABPH_TAGGING_OBJECTIVE_CONTRACT",
    "ABPH_TEACHER_LOGIT_SPLITS",
    "TaggingObjectiveConfig",
    "TaggingObjectiveOutput",
    "compute_tagging_objective",
    "tagging_objective_config",
    "teacher_logits_for_objective",
]
