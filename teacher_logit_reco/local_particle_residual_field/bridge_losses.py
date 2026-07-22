"""Exact C0 loss mappings, target-aligned smoothness, and reachability metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bridge import physical_loss_groups
from .bridge_contracts import with_content_hash
from .bridge_reconstructor import C0CorrectionOutput, TorchBridgeScalers


PREDICTION_ANCHORED_C0_LOSS_RECIPE_CONTRACT = (
    "prediction_anchored_c0_loss_recipe_v1"
)
PREDICTION_ANCHORED_NEIGHBOR_GRAPH_CONTRACT = (
    "prediction_anchored_directed_neighbor_graph_v1"
)
PREDICTION_ANCHORED_REACHABILITY_METRICS_CONTRACT = (
    "prediction_anchored_bridge_reachability_metrics_v1"
)

C0_CANONICAL_RUN_IDS = tuple(f"D10_L{index}_{suffix}" for index, suffix in (
    (0, "bridge_only"),
    (1, "ce_only"),
    (2, "kd_only"),
    (3, "kd_ce"),
    (4, "kd_bridge"),
    (5, "ce_bridge"),
    (6, "kd_ce_bridge"),
    (7, "plus_anchor"),
    (8, "full_c0"),
    (9, "full_true_target"),
    (10, "no_trust"),
))
C0_ALIAS_TO_CANONICAL = {"D10_A0_c0_delta": "D10_L8_full_c0"}

RADIUS_VALUES = (0.02, 0.05, 0.10)
SEMANTIC_OFFSETS = {
    "pt_density": tuple(range(0, 5)),
    "centroid": tuple(range(5, 8)),
    "multiplicity": tuple(range(8, 10)),
    "composition": tuple(range(10, 15)),
}


@dataclass(frozen=True)
class C0LossRecipe:
    run_id: str
    kd: float
    ce: float
    bridge: float
    true: float
    anchor: float
    smooth: float
    gate: float = 0.0
    field_warmup: bool = False
    requires_selected_teacher: bool = True
    requires_target_logit_cache: bool = False
    trust_bound_enabled: bool = True
    selectable_for_primary_deployment: bool = True
    preteacher_l0_exception: bool = False

    def __post_init__(self) -> None:
        if self.run_id not in C0_CANONICAL_RUN_IDS:
            raise ValueError(f"unknown C0 loss run {self.run_id!r}")
        for name in ("kd", "ce", "bridge", "true", "anchor", "smooth", "gate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss coefficient {name} must be finite and nonnegative")
        if self.gate != 0:
            raise ValueError("Step 5 C0 has no gate loss")
        if bool(self.requires_target_logit_cache) != (self.kd > 0):
            raise ValueError("target-logit cache requirement must exactly follow nonzero KD")
        if self.preteacher_l0_exception != (self.run_id == "D10_L0_bridge_only"):
            raise ValueError("only L0 may use the preteacher checkpoint exception")
        if self.trust_bound_enabled == (self.run_id == "D10_L10_no_trust"):
            raise ValueError("only L10 may disable the trust bound")

    def phase_coefficients(self, phase: str) -> dict[str, float]:
        if phase not in {"field_warmup", "distillation"}:
            raise ValueError("training phase must be field_warmup or distillation")
        if phase == "field_warmup":
            if not self.field_warmup:
                raise ValueError(f"{self.run_id} does not declare field warm-up")
            return {
                "kd": 0.0,
                "ce": 0.0,
                "bridge": float(self.bridge),
                "true": 0.0,
                "anchor": float(self.anchor),
                "smooth": float(self.smooth),
                "gate": 0.0,
            }
        return {
            name: float(getattr(self, name))
            for name in ("kd", "ce", "bridge", "true", "anchor", "smooth", "gate")
        }

    def to_artifact(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = PREDICTION_ANCHORED_C0_LOSS_RECIPE_CONTRACT
        payload["distillation_coefficients"] = self.phase_coefficients("distillation")
        payload["warmup_coefficients"] = (
            self.phase_coefficients("field_warmup") if self.field_warmup else None
        )
        payload["architecture"] = "D10_A0_c0_delta"
        payload["gate_present"] = False
        return with_content_hash(payload)


def c0_loss_recipes() -> dict[str, C0LossRecipe]:
    recipes = (
        C0LossRecipe(
            "D10_L0_bridge_only", 0, 0, 1.0, 0, 0.02, 0.01,
            field_warmup=True, requires_selected_teacher=False,
            preteacher_l0_exception=True,
        ),
        C0LossRecipe(
            "D10_L1_ce_only", 0, 1.0, 0, 0, 0, 0,
            field_warmup=False, requires_target_logit_cache=False,
        ),
        C0LossRecipe(
            "D10_L2_kd_only", 1.0, 0, 0, 0, 0, 0,
            field_warmup=False, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L3_kd_ce", 1.0, 0.50, 0, 0, 0, 0,
            field_warmup=False, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L4_kd_bridge", 1.0, 0, 0.20, 0, 0.02, 0.01,
            field_warmup=True, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L5_ce_bridge", 0, 0.50, 0.20, 0, 0.02, 0.01,
            field_warmup=True, requires_target_logit_cache=False,
        ),
        C0LossRecipe(
            "D10_L6_kd_ce_bridge", 1.0, 0.50, 0.20, 0, 0, 0,
            field_warmup=True, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L7_plus_anchor", 1.0, 0.50, 0.20, 0, 0.02, 0,
            field_warmup=True, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L8_full_c0", 1.0, 0.50, 0.20, 0, 0.02, 0.01,
            field_warmup=True, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L9_full_true_target", 1.0, 0.50, 0.20, 0.05, 0.02, 0.01,
            field_warmup=True, requires_target_logit_cache=True,
        ),
        C0LossRecipe(
            "D10_L10_no_trust", 1.0, 0.50, 0.20, 0, 0.02, 0.01,
            field_warmup=True, requires_target_logit_cache=True,
            trust_bound_enabled=False, selectable_for_primary_deployment=False,
        ),
    )
    return {recipe.run_id: recipe for recipe in recipes}


def resolve_c0_loss_recipe(run_id: str) -> C0LossRecipe:
    canonical = C0_ALIAS_TO_CANONICAL.get(str(run_id), str(run_id))
    try:
        return c0_loss_recipes()[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown Step 5 C0 run ID {run_id!r}") from exc


@dataclass(frozen=True)
class DirectedNeighborGraph:
    neighbor_indices: torch.Tensor
    edge_valid: torch.Tensor
    delta_r: torch.Tensor
    cap: int
    support: float


def build_directed_neighbor_graph(
    hlt_tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    cap: int = 32,
    support: float = 0.30,
) -> DirectedNeighborGraph:
    """32 nearest valid non-self neighbors, ties by source-particle index."""

    tokens = hlt_tokens.to(dtype=torch.float32)
    valid = mask.to(device=tokens.device, dtype=torch.bool)
    if tokens.ndim != 3 or tokens.shape[-1] < 3 or valid.shape != tokens.shape[:2]:
        raise ValueError("neighbor graph tokens/mask do not align")
    if int(cap) <= 0 or float(support) <= 0:
        raise ValueError("neighbor graph cap/support must be positive")
    batch, particles = valid.shape
    eta = tokens[..., 1]
    phi = tokens[..., 2]
    d_eta = eta[:, :, None] - eta[:, None, :]
    d_phi = torch.remainder(phi[:, :, None] - phi[:, None, :] + math.pi, 2 * math.pi) - math.pi
    distance = torch.sqrt(torch.clamp(d_eta.square() + d_phi.square(), min=0.0))
    source_index = torch.arange(particles, device=tokens.device)
    self_edge = source_index[None, :, None] == source_index[None, None, :]
    allowed = (
        valid[:, :, None]
        & valid[:, None, :]
        & ~self_edge
        & (distance <= float(support))
    )
    ranked_distance = distance.masked_fill(~allowed, float("inf"))
    # stable=True makes exact distance ties retain ascending source index.
    order = torch.argsort(ranked_distance, dim=-1, stable=True)
    keep = min(int(cap), int(particles))
    indices = order[..., :keep]
    chosen_distance = torch.gather(distance, 2, indices)
    chosen_valid = torch.gather(allowed, 2, indices)
    indices = torch.where(chosen_valid, indices, torch.zeros_like(indices))
    chosen_distance = torch.where(chosen_valid, chosen_distance, torch.zeros_like(chosen_distance))
    return DirectedNeighborGraph(
        neighbor_indices=indices,
        edge_valid=chosen_valid,
        delta_r=chosen_distance,
        cap=int(cap),
        support=float(support),
    )


def masked_group_balanced_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sigma_delta: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Entry→channel→semantic-group means in standardized physical45 space."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] < 45:
        raise ValueError("group-balanced Huber fields do not align")
    valid = mask.to(device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape[:2]:
        raise ValueError("group-balanced Huber mask does not align")
    scale = sigma_delta.to(device=prediction.device, dtype=prediction.dtype)[:45]
    standardized = (prediction[..., :45] - target[..., :45]) / scale
    element = F.smooth_l1_loss(
        standardized, torch.zeros_like(standardized), reduction="none", beta=float(beta)
    )
    weights = valid.to(dtype=element.dtype).unsqueeze(-1)
    channel_mean = (element * weights).sum(dim=(0, 1)) / torch.clamp(
        weights.sum(dim=(0, 1)), min=1.0
    )
    group_values: dict[str, torch.Tensor] = {}
    for name, indices in physical_loss_groups().items():
        group_values[name] = channel_mean[indices].mean()
    if not group_values:
        raise ValueError("no nonempty physical loss groups")
    return torch.stack(list(group_values.values())).mean(), group_values


def local_smoothness_loss(
    physical_correction: torch.Tensor,
    hlt_tokens: torch.Tensor,
    mask: torch.Tensor,
    sigma_delta: torch.Tensor,
    *,
    graph: DirectedNeighborGraph | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact Gaussian-weighted directed-edge smoothness over twelve groups."""

    if physical_correction.ndim != 3 or physical_correction.shape[-1] != 45:
        raise ValueError("smoothness correction must have shape [B,P,45]")
    valid = mask.to(device=physical_correction.device, dtype=torch.bool)
    graph = graph or build_directed_neighbor_graph(hlt_tokens, valid)
    standardized = physical_correction / sigma_delta[:45].to(
        device=physical_correction.device, dtype=physical_correction.dtype
    )
    batch, particles, _ = standardized.shape
    neighbors = graph.neighbor_indices
    batch_index = torch.arange(batch, device=standardized.device)[:, None, None]
    source = standardized[batch_index, neighbors]
    target = standardized[:, :, None, :]
    difference = target - source
    groups: dict[str, torch.Tensor] = {}
    for radius_index, radius in enumerate(RADIUS_VALUES):
        edge_weight = torch.exp(-0.5 * graph.delta_r.square() / float(radius) ** 2)
        edge_weight = edge_weight * graph.edge_valid.to(dtype=edge_weight.dtype)
        weight_sum = edge_weight.sum()
        if float(weight_sum.detach().cpu().item()) <= 0:
            continue
        for semantic, offsets in SEMANTIC_OFFSETS.items():
            indices = [15 * radius_index + offset for offset in offsets]
            squared = difference[..., indices].square().sum(dim=-1)
            numerator = (edge_weight * squared).sum()
            denominator = len(indices) * weight_sum + 1.0e-6
            groups[f"r{radius:.2f}.{semantic}"] = numerator / denominator
    if not groups:
        return physical_correction.sum() * 0.0, {}
    return torch.stack(list(groups.values())).mean(), groups


def distillation_kl_loss(
    live_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    if live_logits.shape != target_logits.shape or live_logits.ndim != 2:
        raise ValueError("KD live/target logits must align as [B,C]")
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("KD temperature must be positive")
    target_probability = torch.softmax(target_logits.detach() / tau, dim=-1)
    live_log_probability = torch.log_softmax(live_logits / tau, dim=-1)
    return F.kl_div(live_log_probability, target_probability, reduction="batchmean") * tau**2


def anchor_regularization(
    f_hat: torch.Tensor,
    f0: torch.Tensor,
    mask: torch.Tensor,
    trust_scale: torch.Tensor,
) -> torch.Tensor:
    squared = (f_hat[..., :45] - f0.detach()[..., :45]).square()
    squared = squared / (trust_scale[:45].to(squared) ** 2 + 1.0e-12)
    valid = mask.to(device=squared.device, dtype=squared.dtype).unsqueeze(-1)
    return (squared * valid).sum() / torch.clamp(valid.sum() * 45, min=1.0)


def compute_c0_objective(
    output: C0CorrectionOutput,
    batch: Mapping[str, Any],
    recipe: C0LossRecipe,
    scalers: TorchBridgeScalers,
    *,
    phase: str,
    live_logits: torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    coefficients = recipe.phase_coefficients(phase)
    zero = output.f_hat.sum() * 0.0
    components: dict[str, torch.Tensor] = {name: zero for name in coefficients}
    bridge_groups: dict[str, torch.Tensor] = {}
    smooth_groups: dict[str, torch.Tensor] = {}
    mask = output.mask
    if coefficients["kd"] > 0:
        if live_logits is None or target_logits is None:
            raise ValueError(f"{recipe.run_id} requires bound live and cached target logits")
        components["kd"] = distillation_kl_loss(
            live_logits, target_logits, temperature=temperature
        )
    elif target_logits is not None:
        raise ValueError(f"{recipe.run_id} has zero KD and must not load target logits")
    if coefficients["ce"] > 0:
        if live_logits is None or "labels" not in batch:
            raise ValueError(f"{recipe.run_id} requires live logits and labels for CE")
        components["ce"] = F.cross_entropy(
            live_logits, torch.as_tensor(batch["labels"], device=live_logits.device, dtype=torch.long)
        )
    if coefficients["bridge"] > 0:
        if "bridge_fields" not in batch:
            raise ValueError(f"{recipe.run_id} requires virtual bridge fields")
        components["bridge"], bridge_groups = masked_group_balanced_huber(
            output.f_hat,
            torch.as_tensor(batch["bridge_fields"], device=output.f_hat.device, dtype=output.f_hat.dtype),
            mask,
            scalers.sigma_delta,
        )
    if coefficients["true"] > 0:
        if "true_fields" not in batch:
            raise ValueError(f"{recipe.run_id} requires true fields for its declared weak-truth term")
        components["true"], _ = masked_group_balanced_huber(
            output.f_hat,
            torch.as_tensor(batch["true_fields"], device=output.f_hat.device, dtype=output.f_hat.dtype),
            mask,
            scalers.sigma_delta,
        )
    if coefficients["anchor"] > 0:
        components["anchor"] = anchor_regularization(
            output.f_hat,
            torch.as_tensor(batch["f0"], device=output.f_hat.device, dtype=output.f_hat.dtype),
            mask,
            scalers.trust_scale,
        )
    if coefficients["smooth"] > 0:
        components["smooth"], smooth_groups = local_smoothness_loss(
            output.physical_correction,
            torch.as_tensor(batch["hlt_tokens"], device=output.f_hat.device, dtype=output.f_hat.dtype),
            mask,
            scalers.sigma_delta,
        )
    if coefficients["gate"] != 0:
        raise AssertionError("Step 5 C0 unexpectedly activated a gate loss")
    weighted = {
        name: components[name] * float(coefficients[name]) for name in coefficients
    }
    total = torch.stack(list(weighted.values())).sum()
    diagnostics = {
        "run_id": recipe.run_id,
        "phase": phase,
        "coefficients": coefficients,
        "raw_components": {
            name: float(value.detach().cpu().item()) for name, value in components.items()
        },
        "weighted_components": {
            name: float(value.detach().cpu().item()) for name, value in weighted.items()
        },
        "total": float(total.detach().cpu().item()),
        "bridge_groups": {
            name: float(value.detach().cpu().item()) for name, value in bridge_groups.items()
        },
        "smooth_groups": {
            name: float(value.detach().cpu().item()) for name, value in smooth_groups.items()
        },
        "gate_present": False,
        "temperature": float(temperature),
    }
    return total, diagnostics


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return ranks


def _vector_reachability(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    prediction = prediction.astype(np.float64, copy=False).reshape(-1)
    target = target.astype(np.float64, copy=False).reshape(-1)
    error = prediction - target
    target_variance = float(np.var(target))
    target_energy = float(np.mean(np.square(target)))
    mse = float(np.mean(np.square(error)))
    explained = None if target_variance <= 0 else float(1.0 - np.var(error) / target_variance)
    nmse = None if target_energy <= 0 else float(mse / target_energy)
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = float(np.linalg.norm(centered_prediction) * np.linalg.norm(centered_target))
    pearson = None if denominator <= 0 else float(np.dot(centered_prediction, centered_target) / denominator)
    ranked_prediction = _rankdata(prediction)
    ranked_target = _rankdata(target)
    rank_denominator = float(
        np.linalg.norm(ranked_prediction - ranked_prediction.mean())
        * np.linalg.norm(ranked_target - ranked_target.mean())
    )
    spearman = (
        None
        if rank_denominator <= 0
        else float(
            np.dot(
                ranked_prediction - ranked_prediction.mean(),
                ranked_target - ranked_target.mean(),
            )
            / rank_denominator
        )
    )
    cosine_denominator = float(np.linalg.norm(prediction) * np.linalg.norm(target))
    cosine = None if cosine_denominator <= 0 else float(np.dot(prediction, target) / cosine_denominator)
    return {
        "explained_variance": explained,
        "normalized_mse": nmse,
        "pearson": pearson,
        "spearman": spearman,
        "cosine": cosine,
        "mse": mse,
    }


def bridge_reachability_metrics(
    predicted_correction: torch.Tensor | np.ndarray,
    target_correction: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(
        predicted_correction.detach().cpu().numpy()
        if isinstance(predicted_correction, torch.Tensor)
        else predicted_correction,
        dtype=np.float64,
    )
    target = np.asarray(
        target_correction.detach().cpu().numpy()
        if isinstance(target_correction, torch.Tensor)
        else target_correction,
        dtype=np.float64,
    )
    valid = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask, dtype=bool)
    if predicted.shape != target.shape or predicted.shape[-1] != 45 or valid.shape != predicted.shape[:2]:
        raise ValueError("reachability correction/mask shapes do not align")
    groups: dict[str, Any] = {}
    radii: dict[str, Any] = {}
    for radius_index, radius in enumerate(RADIUS_VALUES):
        radius_indices = list(range(15 * radius_index, 15 * (radius_index + 1)))
        radii[f"r{radius:.2f}"] = _vector_reachability(
            predicted[..., radius_indices][valid], target[..., radius_indices][valid]
        )
        for semantic, offsets in SEMANTIC_OFFSETS.items():
            indices = [15 * radius_index + offset for offset in offsets]
            groups[f"r{radius:.2f}.{semantic}"] = _vector_reachability(
                predicted[..., indices][valid], target[..., indices][valid]
            )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_REACHABILITY_METRICS_CONTRACT,
            "target": "0.10_times_true_delta_physical45",
            "valid_particle_count": int(valid.sum()),
            "overall": _vector_reachability(predicted[valid], target[valid]),
            "by_radius": radii,
            "by_radius_semantic_group": groups,
            "selection_threshold_applied": False,
        }
    )


__all__ = [
    "PREDICTION_ANCHORED_C0_LOSS_RECIPE_CONTRACT",
    "PREDICTION_ANCHORED_NEIGHBOR_GRAPH_CONTRACT",
    "PREDICTION_ANCHORED_REACHABILITY_METRICS_CONTRACT",
    "C0_CANONICAL_RUN_IDS",
    "C0_ALIAS_TO_CANONICAL",
    "RADIUS_VALUES",
    "SEMANTIC_OFFSETS",
    "C0LossRecipe",
    "DirectedNeighborGraph",
    "c0_loss_recipes",
    "resolve_c0_loss_recipe",
    "build_directed_neighbor_graph",
    "masked_group_balanced_huber",
    "local_smoothness_loss",
    "distillation_kl_loss",
    "anchor_regularization",
    "compute_c0_objective",
    "bridge_reachability_metrics",
]
