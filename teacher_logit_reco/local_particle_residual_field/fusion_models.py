"""Small deployable R0--R4 heads over frozen member representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from .fusion_campaign import FUSION_FAMILY_REPRESENTATION, default_fusion_candidate_specs


LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_MODEL_CONTRACT = (
    "local_residual_field_representation_fusion_model_v1"
)
REPRESENTATION_FUSION_PARAMETER_CAP = 1_000_000


@dataclass(frozen=True)
class RepresentationFusionOutput:
    logits: torch.Tensor
    gate: torch.Tensor | None = None
    correction: torch.Tensor | None = None


def _candidate_ids() -> set[str]:
    return {
        spec.candidate_id
        for spec in default_fusion_candidate_specs()
        if spec.family == FUSION_FAMILY_REPRESENTATION
    }


class FrozenRepresentationFusionHead(nn.Module):
    """A fusion-only module; frozen backbones and their states are never attached."""

    def __init__(
        self,
        candidate_id: str,
        embedding_dim_a: int,
        embedding_dim_b: int,
        *,
        num_classes: int = 10,
        hidden_width: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate_id not in _candidate_ids():
            raise ValueError(f"unsupported representation fusion candidate {candidate_id!r}")
        if int(embedding_dim_a) <= 0 or int(embedding_dim_b) <= 0:
            raise ValueError("embedding dimensions must be positive")
        if int(hidden_width) not in {64, 128}:
            raise ValueError("hidden_width must be one of the locked values 64 or 128")
        if float(dropout) not in {0.0, 0.1}:
            raise ValueError("dropout must be one of the locked values 0.0 or 0.1")
        self.candidate_id = candidate_id
        self.embedding_dim_a = int(embedding_dim_a)
        self.embedding_dim_b = int(embedding_dim_b)
        self.common_embedding_dim = min(self.embedding_dim_a, self.embedding_dim_b)
        self.num_classes = int(num_classes)
        self.hidden_width = int(hidden_width)
        self.dropout = float(dropout)
        dimensions_differ = self.embedding_dim_a != self.embedding_dim_b
        self.projection_a: nn.Module = (
            nn.Linear(self.embedding_dim_a, self.common_embedding_dim, bias=False)
            if dimensions_differ else nn.Identity()
        )
        self.projection_b: nn.Module = (
            nn.Linear(self.embedding_dim_b, self.common_embedding_dim, bias=False)
            if dimensions_differ else nn.Identity()
        )
        feature_dim = 4 * self.common_embedding_dim + 3 * self.num_classes
        output_dim = 1 if candidate_id == "R2_scalar_event_gate" else self.num_classes
        if candidate_id == "R0_linear_embeddings":
            self.head = nn.Linear(feature_dim, output_dim)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, self.hidden_width),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_width, output_dim),
            )
        if candidate_id == "R4_A0_anchored_residual":
            final = self.head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        if self.trainable_parameter_count > REPRESENTATION_FUSION_PARAMETER_CAP:
            raise ValueError(
                f"fusion module has {self.trainable_parameter_count} parameters, exceeding "
                f"the cap {REPRESENTATION_FUSION_PARAMETER_CAP}"
            )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def backbone_parameter_count(self) -> int:
        return 0

    def interaction_features(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> torch.Tensor:
        if embedding_a.ndim != 2 or embedding_b.ndim != 2:
            raise ValueError("member embeddings must have shape [batch, dimension]")
        if logits_a.shape != logits_b.shape or logits_a.ndim != 2 or logits_a.shape[1] != self.num_classes:
            raise ValueError("member logits must be aligned [batch, num_classes] tensors")
        if embedding_a.shape[0] != logits_a.shape[0] or embedding_b.shape[0] != logits_a.shape[0]:
            raise ValueError("embedding and logit batch sizes do not align")
        projected_a = functional.normalize(self.projection_a(embedding_a), p=2.0, dim=1, eps=1.0e-8)
        projected_b = functional.normalize(self.projection_b(embedding_b), p=2.0, dim=1, eps=1.0e-8)
        return torch.cat(
            [
                projected_a,
                projected_b,
                torch.abs(projected_a - projected_b),
                projected_a * projected_b,
                logits_a,
                logits_b,
                logits_b - logits_a,
            ],
            dim=1,
        )

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> RepresentationFusionOutput:
        features = self.interaction_features(embedding_a, embedding_b, logits_a, logits_b)
        raw = self.head(features)
        if self.candidate_id == "R2_scalar_event_gate":
            gate = torch.sigmoid(raw)
            return RepresentationFusionOutput(logits=gate * logits_a + (1.0 - gate) * logits_b, gate=gate)
        if self.candidate_id == "R3_classwise_event_gate":
            gate = torch.sigmoid(raw)
            return RepresentationFusionOutput(logits=gate * logits_a + (1.0 - gate) * logits_b, gate=gate)
        if self.candidate_id == "R4_A0_anchored_residual":
            return RepresentationFusionOutput(logits=logits_a + raw, correction=raw)
        return RepresentationFusionOutput(logits=raw)


def representation_fusion_diagnostics(
    output: RepresentationFusionOutput,
    *,
    logits_a: torch.Tensor,
) -> dict[str, Any]:
    """Serializable gate/correction diagnostics required by the campaign plan."""

    diagnostics: dict[str, Any] = {}
    if output.gate is not None:
        gate = output.gate.detach().float().cpu().reshape(-1)
        clipped = torch.clamp(gate, 1.0e-8, 1.0 - 1.0e-8)
        entropy = -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
        diagnostics["gate"] = {
            "shape": list(output.gate.shape),
            "minimum": float(gate.min()),
            "maximum": float(gate.max()),
            "mean": float(gate.mean()),
            "standard_deviation": float(gate.std(unbiased=False)),
            "quantiles": [float(value) for value in torch.quantile(gate, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))],
            "mean_binary_entropy": float(entropy.mean()),
        }
    if output.correction is not None:
        correction = output.correction.detach().float().cpu()
        correction_norm = torch.linalg.vector_norm(correction, dim=1)
        changed = torch.argmax(output.logits.detach(), dim=1) != torch.argmax(logits_a.detach(), dim=1)
        diagnostics["correction"] = {
            "mean_l2_norm": float(correction_norm.mean()),
            "maximum_l2_norm": float(correction_norm.max()),
            "fraction_a0_predictions_changed": float(changed.float().mean()),
        }
    return diagnostics


__all__ = [
    "LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_MODEL_CONTRACT",
    "REPRESENTATION_FUSION_PARAMETER_CAP",
    "RepresentationFusionOutput",
    "FrozenRepresentationFusionHead",
    "representation_fusion_diagnostics",
]
