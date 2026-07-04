"""View fusion and baseline-safe embedding residual gate modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .branches import ArchitectureViewBranchBank, ArchitectureViewBranchOutput
from .config import (
    ARCHITECTURE_VIEW_BRANCHES,
    ARCHITECTURE_VIEW_PART_CONTRACT,
    ArchitectureViewConfig,
    normalize_architecture_view_branch,
)
from .inputs import sanitize_architecture_view_tokens

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ARCHITECTURE_VIEW_FUSION_CONTRACT = "architecture_view_fusion_gate_v1"


def align_particle_mask_to_length(mask: Any, target_particles: int) -> Any:
    """Return a particle mask with the same sequence length as an adapter tensor."""

    target_particles = int(target_particles)
    if int(mask.shape[1]) == target_particles:
        return mask.bool()
    if int(mask.shape[1]) > target_particles:
        return mask[:, :target_particles].bool()
    pad = mask.new_zeros((int(mask.shape[0]), target_particles - int(mask.shape[1])), dtype=mask.dtype)
    return _torch.cat((mask, pad), dim=1).bool()


@dataclass(frozen=True)
class ArchitectureViewFusionOutput:
    """Fused view tensor and the eventual ParT embedding-space residual."""

    view_embeddings: dict[str, Any]
    combined_view: Any
    delta_h: Any
    gate: Any
    mask: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        delta_norm = self.delta_h.norm(dim=-1)
        delta_mask = align_particle_mask_to_length(self.mask, int(delta_norm.shape[1]))
        delta_valid = delta_mask.to(dtype=self.delta_h.dtype)
        delta_denom = delta_valid.sum().clamp_min(1.0)
        flat_delta_norm = delta_norm[delta_mask]
        gate_values = self.gate.squeeze(-1)
        gate_mask = align_particle_mask_to_length(self.mask, int(gate_values.shape[1]))
        gate_valid = gate_mask.to(dtype=gate_values.dtype)
        gate_denom = gate_valid.sum().clamp_min(1.0)
        flat_gate = gate_values[gate_mask]
        combined_norm = self.combined_view.norm(dim=-1) if int(self.combined_view.shape[-1]) else delta_norm.new_zeros(delta_norm.shape)
        combined_mask = align_particle_mask_to_length(self.mask, int(combined_norm.shape[1]))
        flat_combined_norm = combined_norm[combined_mask]
        if int(flat_delta_norm.numel()) == 0:
            delta_p90 = delta_norm.new_zeros(())
            delta_max = delta_norm.new_zeros(())
        else:
            delta_p90 = _torch.quantile(flat_delta_norm.float(), 0.90)
            delta_max = flat_delta_norm.max()
        if int(flat_gate.numel()) == 0:
            gate_p10 = gate_values.new_zeros(())
            gate_p90 = gate_values.new_zeros(())
        else:
            gate_p10 = _torch.quantile(flat_gate.float(), 0.10)
            gate_p90 = _torch.quantile(flat_gate.float(), 0.90)
        if int(flat_combined_norm.numel()) == 0:
            combined_mean = combined_norm.new_zeros(())
            combined_p90 = combined_norm.new_zeros(())
        else:
            combined_mean = flat_combined_norm.mean()
            combined_p90 = _torch.quantile(flat_combined_norm.float(), 0.90)
        return {
            "contract": ARCHITECTURE_VIEW_FUSION_CONTRACT,
            "view_names": list(self.view_embeddings),
            "combined_view_shape": list(self.combined_view.shape),
            "delta_h_shape": list(self.delta_h.shape),
            "gate_shape": list(self.gate.shape),
            "mean_delta_h_norm": float(((delta_norm * delta_valid).sum() / delta_denom).detach().cpu().item()),
            "delta_h_norm_p90": float(delta_p90.detach().cpu().item()),
            "delta_h_norm_max": float(delta_max.detach().cpu().item()),
            "mean_gate": float(((gate_values * gate_valid).sum() / gate_denom).detach().cpu().item()),
            "gate_p10": float(gate_p10.detach().cpu().item()),
            "gate_p90": float(gate_p90.detach().cpu().item()),
            "adapter_output_norm_mean": float(combined_mean.detach().cpu().item()),
            "adapter_output_norm_p90": float(combined_p90.detach().cpu().item()),
            **self.diagnostics,
        }


def _masked_zero(value: Any, mask: Any) -> Any:
    return value * mask[:, :, None].to(dtype=value.dtype)


class ArchitectureViewFusion(_ModuleBase):
    """Fuse branch outputs and produce a gated ``delta_h``.

    The final projection is zero-initialized so Step 2 can recover the exact HLT
    ParT baseline before training.
    """

    def __init__(
        self,
        config: ArchitectureViewConfig | None = None,
        *,
        enabled_views: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config or ArchitectureViewConfig()
        views = self.config.enabled_views if enabled_views is None else enabled_views
        self.enabled_views = tuple(normalize_architecture_view_branch(v) for v in views)
        input_dim = int(self.config.view_dim) * len(self.enabled_views)
        self.input_dim = input_dim
        if input_dim > 0:
            self.fuse = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, int(self.config.fusion_hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
                torch.nn.Linear(int(self.config.fusion_hidden_dim), int(self.config.fusion_hidden_dim)),
                torch.nn.GELU(),
            )
            self.delta_projection = torch.nn.Linear(int(self.config.fusion_hidden_dim), int(self.config.part_embed_dim))
            self.gate = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, int(self.config.fusion_hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(self.config.fusion_hidden_dim), 1),
            )
            torch.nn.init.zeros_(self.delta_projection.weight)
            torch.nn.init.zeros_(self.delta_projection.bias)
            final_gate = self.gate[-1]
            torch.nn.init.zeros_(final_gate.weight)
            torch.nn.init.constant_(final_gate.bias, float(self.config.gate_bias_init))
        else:
            self.fuse = torch.nn.Identity()
            self.delta_projection = torch.nn.Identity()
            self.gate = torch.nn.Identity()

    def forward(
        self,
        branch_outputs: Mapping[str, ArchitectureViewBranchOutput | Any],
        mask: Any,
    ) -> ArchitectureViewFusionOutput:
        torch = require_torch()
        mask = mask.to(dtype=torch.bool)
        if not self.enabled_views:
            batch_size, num_particles = mask.shape
            combined = torch.zeros(
                batch_size,
                num_particles,
                0,
                dtype=torch.float32,
                device=mask.device,
            )
            delta_h = torch.zeros(
                batch_size,
                num_particles,
                int(self.config.part_embed_dim),
                dtype=torch.float32,
                device=mask.device,
            )
            gate = torch.zeros(batch_size, num_particles, 1, dtype=torch.float32, device=mask.device)
            return ArchitectureViewFusionOutput(
                view_embeddings={},
                combined_view=combined,
                delta_h=delta_h,
                gate=gate,
                mask=mask,
                diagnostics={"baseline_recheck": True},
            )

        tensors: list[Any] = []
        view_map: dict[str, Any] = {}
        for name in self.enabled_views:
            output = branch_outputs[name]
            embeddings = output.embeddings if isinstance(output, ArchitectureViewBranchOutput) else output
            embeddings = embeddings.to(device=mask.device, dtype=torch.float32)
            if tuple(embeddings.shape[:2]) != tuple(mask.shape):
                raise ValueError(
                    f"view {name!r} has particle shape {tuple(embeddings.shape[:2])}, expected {tuple(mask.shape)}"
                )
            if int(embeddings.shape[-1]) != int(self.config.view_dim):
                raise ValueError(
                    f"view {name!r} dim {embeddings.shape[-1]} does not match view_dim={self.config.view_dim}"
                )
            view_map[name] = _masked_zero(embeddings, mask)
            tensors.append(view_map[name])
        combined = torch.cat(tensors, dim=-1)
        hidden = self.fuse(combined)
        delta_raw = self.delta_projection(hidden)
        gate = torch.sigmoid(self.gate(combined))
        delta_h = _masked_zero(gate * delta_raw, mask)
        gate = _masked_zero(gate, mask)
        return ArchitectureViewFusionOutput(
            view_embeddings=view_map,
            combined_view=_masked_zero(combined, mask),
            delta_h=delta_h,
            gate=gate,
            mask=mask,
            diagnostics={"baseline_recovery_zero_projection": True},
        )


class ArchitectureViewParticleViews(_ModuleBase):
    """Convenience Step 1 module: raw tokens -> branch outputs -> fused delta_h."""

    def __init__(
        self,
        config: ArchitectureViewConfig | None = None,
        *,
        enabled_views: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ArchitectureViewConfig()
        views = self.config.enabled_views if enabled_views is None else enabled_views
        self.enabled_views = tuple(normalize_architecture_view_branch(v) for v in views)
        self.branch_bank = ArchitectureViewBranchBank(self.config, enabled_views=self.enabled_views)
        self.fusion = ArchitectureViewFusion(self.config, enabled_views=self.enabled_views)

    def forward(self, tokens: Any, mask: Any | None = None) -> ArchitectureViewFusionOutput:
        prepared = sanitize_architecture_view_tokens(tokens, mask)
        branch_outputs = self.branch_bank(prepared.tokens, prepared.mask)
        output = self.fusion(branch_outputs, prepared.mask)
        return ArchitectureViewFusionOutput(
            view_embeddings=output.view_embeddings,
            combined_view=output.combined_view,
            delta_h=output.delta_h,
            gate=output.gate,
            mask=prepared.mask,
            diagnostics={
                **output.diagnostics,
                "contract": ARCHITECTURE_VIEW_PART_CONTRACT,
                "enabled_views": list(self.enabled_views),
            },
        )


def expected_combined_view_dim(config: ArchitectureViewConfig, enabled_views: tuple[str, ...] | None = None) -> int:
    views = config.enabled_views if enabled_views is None else enabled_views
    return int(config.view_dim) * len(tuple(views))


def architecture_view_branch_order() -> tuple[str, ...]:
    return ARCHITECTURE_VIEW_BRANCHES
