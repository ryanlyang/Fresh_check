"""Required direct-target and auxiliary objectives for binary accounting heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .binary_accounting import (
    ABPH_BINARY_COUNT_SUPPORT,
    AccountingState,
    BinarySplitPrediction,
    CompiledBinarySplit,
)
from .root_transforms import ROOT_FEATURE_INDEX, ROOT_SHAPE_FEATURE_NAMES
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import ROOT_FEATURE_NAMES, TOPOLOGY_ACTIVE_SPLIT


ABPH_BINARY_OBJECTIVE_CONTRACT = "adaptive_binary_pseudooffline_binary_objective_v1"
ABPH_MIN_REQUIRED_BINARY_LOSS_WEIGHT = 1.0e-4


@dataclass(frozen=True)
class BinaryAccountingLossWeights:
    topology_nll: float = 1.0
    count_nll: float = 0.60
    type_count_huber: float = 0.60
    charge_nll: float = 0.40
    four_vector_huber: float = 1.0
    scalar_pt_huber: float = 0.20
    type_energy_huber: float = 0.25
    type_pt_huber: float = 0.25
    shape_huber: float = 0.35
    auxiliary_consistency: float = 0.15

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < ABPH_MIN_REQUIRED_BINARY_LOSS_WEIGHT:
                raise ValueError(
                    f"required binary loss {name} must be finite and at least "
                    f"{ABPH_MIN_REQUIRED_BINARY_LOSS_WEIGHT}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"contract": ABPH_BINARY_OBJECTIVE_CONTRACT, **asdict(self)}


@dataclass(frozen=True)
class BinaryAccountingLossOutput:
    total: Any
    components: Mapping[str, Any]
    weights: BinaryAccountingLossWeights

    def detached_components(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu()) for name, value in self.components.items()
        }


def _target_field(target: Any, name: str) -> Any:
    return target[:, :, ROOT_FEATURE_INDEX[name]]


def _scaled_huber(predicted: Any, target: Any) -> Any:
    torch = require_torch()
    scale = target.detach().abs().clamp_min(1.0)
    return torch.nn.functional.smooth_l1_loss(predicted / scale, target / scale)


def compute_binary_accounting_losses(
    prediction: BinarySplitPrediction,
    compiled: CompiledBinarySplit,
    parent: AccountingState,
    target_child_ledger: Any,
    target_child_mask: Any,
    target_topology: Any,
    *,
    weights: BinaryAccountingLossWeights | None = None,
) -> BinaryAccountingLossOutput:
    """Supervise hard decisions and soft observables without conflating them."""

    torch = require_torch()
    resolved = weights or BinaryAccountingLossWeights()
    target = torch.as_tensor(
        target_child_ledger,
        device=compiled.child_ledger.device,
        dtype=compiled.child_ledger.dtype,
    )
    child_mask = torch.as_tensor(target_child_mask, device=target.device).bool()
    topology = torch.as_tensor(target_topology, device=target.device).to(torch.long)
    batch = parent.batch_size
    if target.shape != (batch, 2, len(ROOT_FEATURE_NAMES)):
        raise ValueError("target child ledger must have shape [B, 2, root_feature_dim]")
    if child_mask.shape != (batch, 2) or topology.shape != (batch,):
        raise ValueError("target child masks/topology do not match the parent batch")
    if not bool(((topology == 1) | (topology == 2)).all()):
        raise ValueError("target topology must contain terminal/split states")
    target_class = topology - 1
    topology_nll = torch.nn.functional.cross_entropy(prediction.topology_logits, target_class)
    split = topology == int(TOPOLOGY_ACTIVE_SPLIT)
    zero = prediction.count_logits.sum() * 0.0
    if bool(split.any()):
        if not bool(child_mask[split].all()):
            raise ValueError("every split target must expose exactly two active children")
        parent_count = parent.constituent_count[split]
        support = torch.arange(1, ABPH_BINARY_COUNT_SUPPORT + 1, device=target.device)
        valid_count = support[None, :] < parent_count[:, None]
        count_logits = prediction.count_logits[split].masked_fill(~valid_count, float("-inf"))
        target_count_one = _target_field(target, "constituent_count")[split, 0].round().to(torch.long)
        count_nll = torch.nn.functional.cross_entropy(count_logits, target_count_one - 1)

        target_types = torch.stack(
            tuple(_target_field(target, f"count_{name}") for name in ABPH_PID_CATEGORIES),
            dim=-1,
        )[split]
        type_count_huber = torch.nn.functional.smooth_l1_loss(
            compiled.relaxed_child_type_counts[split], target_types
        )
        target_charge_one = _target_field(target, "integer_charge")[split, 0].round().to(torch.long)
        charge_nll = torch.nn.functional.cross_entropy(
            prediction.charge_logits[split], target_charge_one + ABPH_MAX_PARTICLES
        )
        target_p4 = torch.stack(
            tuple(_target_field(target, name) for name in ("energy", "px", "py", "pz")),
            dim=-1,
        )[split]
        four_vector_huber = _scaled_huber(compiled.child_four_vector[split], target_p4)
        scalar_pt_huber = _scaled_huber(
            compiled.child_scalar_sum_pt[split],
            _target_field(target, "scalar_sum_pt")[split],
        )
        target_type_energy = torch.stack(
            tuple(_target_field(target, f"energy_{name}") for name in ABPH_PID_CATEGORIES),
            dim=-1,
        )[split]
        target_type_pt = torch.stack(
            tuple(_target_field(target, f"scalar_pt_{name}") for name in ABPH_PID_CATEGORIES),
            dim=-1,
        )[split]
        type_energy_huber = _scaled_huber(
            compiled.child_type_energy[split], target_type_energy
        )
        type_pt_huber = _scaled_huber(
            compiled.child_type_scalar_pt[split], target_type_pt
        )
        target_shape = torch.stack(
            tuple(_target_field(target, name) for name in ROOT_SHAPE_FEATURE_NAMES),
            dim=-1,
        )[split]
        shape_huber = _scaled_huber(compiled.child_shape_features[split], target_shape)
        scalar_consistency = _scaled_huber(
            compiled.child_scalar_sum_pt[split].sum(dim=1), parent.scalar_sum_pt[split]
        )
        energy_consistency = _scaled_huber(
            compiled.child_type_energy[split].sum(dim=1), parent.type_energy[split]
        )
        pt_consistency = _scaled_huber(
            compiled.child_type_scalar_pt[split].sum(dim=1), parent.type_scalar_pt[split]
        )
        auxiliary_consistency = (
            scalar_consistency + energy_consistency + pt_consistency
        ) / 3.0
    else:
        count_nll = zero
        type_count_huber = zero
        charge_nll = zero
        four_vector_huber = zero
        scalar_pt_huber = zero
        type_energy_huber = zero
        type_pt_huber = zero
        shape_huber = zero
        auxiliary_consistency = zero
    components = {
        "topology_nll": topology_nll,
        "count_nll": count_nll,
        "type_count_huber": type_count_huber,
        "charge_nll": charge_nll,
        "four_vector_huber": four_vector_huber,
        "scalar_pt_huber": scalar_pt_huber,
        "type_energy_huber": type_energy_huber,
        "type_pt_huber": type_pt_huber,
        "shape_huber": shape_huber,
        "auxiliary_consistency": auxiliary_consistency,
    }
    total = sum(
        components[name] * float(getattr(resolved, name)) for name in components
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("binary accounting objective produced a nonfinite loss")
    return BinaryAccountingLossOutput(total=total, components=components, weights=resolved)


__all__ = [
    "ABPH_BINARY_OBJECTIVE_CONTRACT",
    "ABPH_MIN_REQUIRED_BINARY_LOSS_WEIGHT",
    "BinaryAccountingLossOutput",
    "BinaryAccountingLossWeights",
    "compute_binary_accounting_losses",
]
