"""Version A architecture residual experts for frozen HLT ParT.

The experiment keeps the exact HLT ParT baseline frozen and lets a lightweight
PN/PFN/PCNN-style branch propose only a scalar correction to the ParT two-logit
margin.  A zero-initialized residual head recovers the baseline at step zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import build_hlt_classifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part.config import (
    LOCAL_COMPRESSION_PART_CONTRACT,
    LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
    LocalCompressionFeatureConfig,
)
from teacher_logit_reco.local_compression_part.features import (
    LocalCompressionCanonicalInputs,
    build_local_compression_canonical_inputs,
)

try:  # Keep import cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ARCH_RESIDUAL_MODEL_STEP = "arch_residual_part_version_a_model"
ARCH_RESIDUAL_MODEL_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_arch_residual_expert_v1"
ARCH_RESIDUAL_ARCHITECTURES = ("pfn", "pcnn", "pn")


def normalize_architecture(value: str) -> str:
    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "particle_flow_network": "pfn",
        "particleflow": "pfn",
        "particle_net": "pn",
        "particlenet": "pn",
        "particle_cnn": "pcnn",
        "cnn": "pcnn",
    }
    clean = aliases.get(clean, clean)
    if clean not in ARCH_RESIDUAL_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCH_RESIDUAL_ARCHITECTURES}, got {value!r}")
    return clean


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return value


@dataclass(frozen=True)
class ArchResidualExpertConfig:
    """Config for one residual architecture expert."""

    architecture: str = "pfn"
    num_classes: int = 2
    raw_token_dim: int = RAW_TOKEN_DIM
    canonical_feature_dim: int = 19
    hidden_dim: int = 128
    particle_layers: int = 3
    global_layers: int = 2
    edge_k: int = 16
    dropout: float = 0.05
    condition_on_baseline: bool = True
    baseline_condition_dim: int = 2
    gamma_init: float = 1.0
    residual_scale: float = 1.0
    label_names: tuple[str, ...] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", normalize_architecture(self.architecture))
        for field_name in ("num_classes", "raw_token_dim", "canonical_feature_dim", "hidden_dim", "particle_layers", "global_layers"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if int(self.num_classes) != 2:
            raise ValueError("Version A arch residual is intentionally a binary QCD/Hgg residual")
        if int(self.raw_token_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"raw_token_dim must match RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        if int(self.edge_k) <= 0:
            raise ValueError("edge_k must be positive")
        object.__setattr__(self, "edge_k", int(self.edge_k))
        object.__setattr__(self, "dropout", _validate_probability(self.dropout, name="dropout"))
        object.__setattr__(self, "baseline_condition_dim", 2 if bool(self.condition_on_baseline) else 0)
        if float(self.residual_scale) <= 0.0:
            raise ValueError("residual_scale must be positive")
        object.__setattr__(self, "residual_scale", float(self.residual_scale))
        object.__setattr__(self, "gamma_init", float(self.gamma_init))
        object.__setattr__(self, "label_names", tuple(str(name) for name in self.label_names))
        if len(self.label_names) != 2:
            raise ValueError("label_names must contain two binary labels")

    @property
    def branch_input_dim(self) -> int:
        return int(self.canonical_feature_dim) + int(self.baseline_condition_dim)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    layers: int,
    dropout: float,
    final_activation: bool = False,
) -> Any:
    torch = require_torch()
    modules: list[Any] = []
    dim = int(in_dim)
    for _ in range(max(0, int(layers) - 1)):
        modules.extend(
            [
                torch.nn.Linear(dim, int(hidden_dim)),
                torch.nn.LayerNorm(int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
            ]
        )
        dim = int(hidden_dim)
    modules.append(torch.nn.Linear(dim, int(out_dim)))
    if final_activation:
        modules.extend([torch.nn.GELU(), torch.nn.Dropout(float(dropout))])
    return torch.nn.Sequential(*modules)


def _masked_mean(values: Any, mask: Any) -> Any:
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=values.dtype)
    return (values * mask[:, :, None].to(dtype=values.dtype)).sum(dim=1) / denom


def _masked_max(values: Any, mask: Any) -> Any:
    torch = require_torch()
    masked = values.masked_fill(~mask[:, :, None].bool(), -1.0e9)
    output = masked.max(dim=1).values
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


class _ResidualHead(_ModuleBase):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        layers: int,
        dropout: float,
        gamma_init: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.net = _mlp(int(in_dim), int(hidden_dim), 1, layers=max(1, int(layers)), dropout=float(dropout))
        final = self.net[-1]
        if isinstance(final, torch.nn.Linear):
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
        self.gamma = torch.nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.residual_scale = float(residual_scale)

    def forward(self, features: Any) -> Any:
        residual = self.net(features).squeeze(-1)
        return residual * self.gamma * float(self.residual_scale)


class PFNResidualBranch(_ModuleBase):
    """Particle-flow-network style branch: per-particle phi, pooled rho."""

    def __init__(self, config: ArchResidualExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.phi = _mlp(
            config.branch_input_dim,
            config.hidden_dim,
            config.hidden_dim,
            layers=config.particle_layers,
            dropout=config.dropout,
            final_activation=True,
        )
        self.head = _ResidualHead(
            2 * config.hidden_dim,
            config.hidden_dim,
            layers=config.global_layers,
            dropout=config.dropout,
            gamma_init=config.gamma_init,
            residual_scale=config.residual_scale,
        )

    def forward(self, features: Any, tokens: Any, mask: Any) -> tuple[Any, Mapping[str, Any]]:
        del tokens
        encoded = self.phi(features)
        encoded = encoded * mask[:, :, None].to(dtype=encoded.dtype)
        global_features = require_torch().cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=-1)
        correction = self.head(global_features)
        return correction, {
            "branch_encoded_abs_mean": float(encoded.detach().abs().mean().cpu().item()),
            "gamma": float(self.head.gamma.detach().cpu().item()),
        }


class PCNNResidualBranch(_ModuleBase):
    """Sequence CNN branch over the pT-sorted fixed-HLT particles."""

    def __init__(self, config: ArchResidualExpertConfig) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config
        layers: list[Any] = []
        in_dim = int(config.branch_input_dim)
        for _ in range(int(config.particle_layers)):
            layers.extend(
                [
                    torch.nn.Conv1d(in_dim, int(config.hidden_dim), kernel_size=3, padding=1),
                    torch.nn.GroupNorm(1, int(config.hidden_dim)),
                    torch.nn.GELU(),
                    torch.nn.Dropout(float(config.dropout)),
                ]
            )
            in_dim = int(config.hidden_dim)
        self.conv = torch.nn.Sequential(*layers)
        self.head = _ResidualHead(
            2 * config.hidden_dim,
            config.hidden_dim,
            layers=config.global_layers,
            dropout=config.dropout,
            gamma_init=config.gamma_init,
            residual_scale=config.residual_scale,
        )

    def forward(self, features: Any, tokens: Any, mask: Any) -> tuple[Any, Mapping[str, Any]]:
        del tokens
        x = features * mask[:, :, None].to(dtype=features.dtype)
        encoded = self.conv(x.transpose(1, 2)).transpose(1, 2)
        encoded = encoded * mask[:, :, None].to(dtype=encoded.dtype)
        global_features = require_torch().cat([_masked_mean(encoded, mask), _masked_max(encoded, mask)], dim=-1)
        correction = self.head(global_features)
        return correction, {
            "branch_encoded_abs_mean": float(encoded.detach().abs().mean().cpu().item()),
            "gamma": float(self.head.gamma.detach().cpu().item()),
        }


def _delta_phi(phi_i: Any, phi_j: Any) -> Any:
    torch = require_torch()
    return torch.atan2(torch.sin(phi_i - phi_j), torch.cos(phi_i - phi_j))


def _knn_indices(tokens: Any, mask: Any, k: int) -> tuple[Any, Any]:
    torch = require_torch()
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    deta = eta[:, :, None] - eta[:, None, :]
    dphi = _delta_phi(phi[:, :, None], phi[:, None, :])
    dist = deta.square() + dphi.square()
    valid_pair = mask[:, :, None].bool() & mask[:, None, :].bool()
    eye = torch.eye(tokens.shape[1], device=tokens.device, dtype=torch.bool)[None, :, :]
    valid_pair = valid_pair & ~eye
    dist = dist.masked_fill(~valid_pair, float("inf"))
    k_eff = min(int(k), int(tokens.shape[1]))
    values, indices = torch.topk(dist, k=k_eff, dim=-1, largest=False, sorted=True)
    neighbor_valid = torch.isfinite(values) & mask[:, :, None].bool()
    indices = torch.where(neighbor_valid, indices, torch.zeros_like(indices))
    return indices, neighbor_valid


def _gather_neighbors(values: Any, indices: Any) -> Any:
    batch, particles, k = indices.shape
    dim = values.shape[-1]
    gather_index = indices[:, :, :, None].expand(batch, particles, k, dim)
    source = values[:, None, :, :].expand(batch, particles, values.shape[1], dim)
    return source.gather(dim=2, index=gather_index)


class PNResidualBranch(_ModuleBase):
    """Small ParticleNet-like EdgeConv branch in eta-phi neighborhoods."""

    def __init__(self, config: ArchResidualExpertConfig) -> None:
        super().__init__()
        torch = require_torch()
        self.config = config
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(config.branch_input_dim, config.hidden_dim),
            torch.nn.LayerNorm(config.hidden_dim),
            torch.nn.GELU(),
        )
        self.edge_mlps = torch.nn.ModuleList(
            [
                _mlp(
                    2 * config.hidden_dim + 4,
                    config.hidden_dim,
                    config.hidden_dim,
                    layers=2,
                    dropout=config.dropout,
                    final_activation=True,
                )
                for _ in range(int(config.particle_layers))
            ]
        )
        self.head = _ResidualHead(
            2 * config.hidden_dim,
            config.hidden_dim,
            layers=config.global_layers,
            dropout=config.dropout,
            gamma_init=config.gamma_init,
            residual_scale=config.residual_scale,
        )

    def _edge_geometry(self, tokens: Any, indices: Any, neighbor_valid: Any) -> Any:
        torch = require_torch()
        neighbor_tokens = _gather_neighbors(tokens, indices)
        center = tokens[:, :, None, :]
        deta = center[..., 1:2] - neighbor_tokens[..., 1:2]
        dphi = _delta_phi(center[..., 2:3], neighbor_tokens[..., 2:3])
        dr = torch.sqrt(deta.square() + dphi.square() + 1.0e-12)
        log_pt_ratio = torch.log1p(torch.clamp(center[..., 0:1], min=0.0)) - torch.log1p(
            torch.clamp(neighbor_tokens[..., 0:1], min=0.0)
        )
        geom = torch.cat([deta, dphi, dr, log_pt_ratio], dim=-1)
        return torch.where(neighbor_valid[:, :, :, None], geom, torch.zeros_like(geom))

    def forward(self, features: Any, tokens: Any, mask: Any) -> tuple[Any, Mapping[str, Any]]:
        torch = require_torch()
        h = self.input_proj(features) * mask[:, :, None].to(dtype=features.dtype)
        indices, neighbor_valid = _knn_indices(tokens, mask, int(self.config.edge_k))
        edge_geom = self._edge_geometry(tokens, indices, neighbor_valid)
        for edge_mlp in self.edge_mlps:
            neighbors = _gather_neighbors(h, indices)
            center = h[:, :, None, :].expand_as(neighbors)
            edge_inputs = torch.cat([center, neighbors - center, edge_geom], dim=-1)
            messages = edge_mlp(edge_inputs)
            messages = messages.masked_fill(~neighbor_valid[:, :, :, None], -1.0e9)
            update = messages.max(dim=2).values
            update = torch.where(torch.isfinite(update), update, torch.zeros_like(update))
            h = (h + update) * mask[:, :, None].to(dtype=h.dtype)
        global_features = torch.cat([_masked_mean(h, mask), _masked_max(h, mask)], dim=-1)
        correction = self.head(global_features)
        valid_edges = neighbor_valid.sum().detach().float()
        possible_edges = mask.sum().detach().float().clamp(min=1.0) * float(min(int(self.config.edge_k), int(tokens.shape[1])))
        return correction, {
            "branch_encoded_abs_mean": float(h.detach().abs().mean().cpu().item()),
            "valid_neighbor_fraction": float((valid_edges / possible_edges).cpu().item()),
            "gamma": float(self.head.gamma.detach().cpu().item()),
        }


def _build_branch(config: ArchResidualExpertConfig) -> Any:
    if config.architecture == "pfn":
        return PFNResidualBranch(config)
    if config.architecture == "pcnn":
        return PCNNResidualBranch(config)
    if config.architecture == "pn":
        return PNResidualBranch(config)
    raise ValueError(f"unsupported architecture {config.architecture!r}")


@dataclass(frozen=True)
class FrozenHLTPartOutput:
    logits: Any
    canonical_inputs: LocalCompressionCanonicalInputs


class FrozenHLTPartBaseline(_ModuleBase):
    """Raw-token scorer for the exact frozen HLT ParT baseline."""

    def __init__(self, *, num_classes: int = 2, feature_config: LocalCompressionFeatureConfig | None = None) -> None:
        super().__init__()
        self.part_model = build_hlt_classifier(num_classes=int(num_classes))
        self.feature_config = feature_config or LocalCompressionFeatureConfig()
        self.baseline_checkpoint_report: dict[str, Any] | None = None
        for parameter in self.part_model.parameters():
            parameter.requires_grad = False

    def forward_outputs(self, tokens: Any, mask: Any, *, max_constits: int | None = None) -> FrozenHLTPartOutput:
        canonical = build_local_compression_canonical_inputs(
            tokens,
            mask,
            max_constits=max_constits if max_constits is not None else int(tokens.shape[1]),
            config=self.feature_config,
        )
        logits = self.part_model(
            canonical.points,
            canonical.features,
            canonical.lorentz_vectors,
            canonical.mask,
        )
        return FrozenHLTPartOutput(logits=logits.float(), canonical_inputs=canonical)


@dataclass(frozen=True)
class ArchResidualPartOutput:
    logits: Any
    baseline_logits: Any
    residual_only_logits: Any
    correction: Any
    baseline_margin: Any
    baseline_signal_prob: Any
    canonical_inputs: LocalCompressionCanonicalInputs
    branch_diagnostics: Mapping[str, Any]
    config: ArchResidualExpertConfig

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        mask = self.canonical_inputs.particle_mask
        correction = self.correction.detach()
        output = {
            "contract": ARCH_RESIDUAL_MODEL_CONTRACT,
            "architecture": self.config.architecture,
            "valid_particle_count": int(mask.detach().cpu().sum().item()),
            "correction_abs_mean": float(correction.abs().mean().cpu().item()),
            "correction_abs_p90": float(torch.quantile(correction.abs().float(), 0.90).cpu().item())
            if int(correction.numel()) > 0
            else 0.0,
            "correction_abs_max": float(correction.abs().max().cpu().item()) if int(correction.numel()) > 0 else 0.0,
            "baseline_margin_mean": float(self.baseline_margin.detach().mean().cpu().item()),
            "baseline_signal_prob_mean": float(self.baseline_signal_prob.detach().mean().cpu().item()),
        }
        output.update({str(key): value for key, value in self.branch_diagnostics.items()})
        return output


class ArchResidualPartModel(_ModuleBase):
    """Frozen HLT ParT plus an architecture-specific residual margin expert."""

    def __init__(
        self,
        config: ArchResidualExpertConfig | Mapping[str, Any] | None = None,
        *,
        baseline: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, ArchResidualExpertConfig) else ArchResidualExpertConfig(**dict(config or {}))
        self.baseline = baseline or FrozenHLTPartBaseline(num_classes=int(self.config.num_classes))
        self.branch = _build_branch(self.config)

    @property
    def output_contract(self) -> str:
        return ARCH_RESIDUAL_MODEL_CONTRACT

    def _branch_features(self, baseline_output: FrozenHLTPartOutput) -> Any:
        torch = require_torch()
        features = baseline_output.canonical_inputs.feature_rows()
        if not bool(self.config.condition_on_baseline):
            return features
        logits = baseline_output.logits.detach().float()
        margin = (logits[:, 1] - logits[:, 0])[:, None, None]
        prob = torch.softmax(logits, dim=1)[:, 1][:, None, None]
        cond = torch.cat([margin, prob], dim=-1).expand(features.shape[0], features.shape[1], 2)
        return torch.cat([features, cond.to(dtype=features.dtype)], dim=-1)

    def forward_outputs(self, tokens: Any, mask: Any, *, max_constits: int | None = None) -> ArchResidualPartOutput:
        torch = require_torch()
        with torch.no_grad():
            baseline_output = self.baseline.forward_outputs(tokens, mask, max_constits=max_constits)
            baseline_logits = baseline_output.logits.detach().float()
        branch_features = self._branch_features(baseline_output)
        correction, diagnostics = self.branch(
            branch_features,
            baseline_output.canonical_inputs.selected_tokens,
            baseline_output.canonical_inputs.particle_mask,
        )
        fused = baseline_logits.clone()
        fused[:, 0] = fused[:, 0] - 0.5 * correction
        fused[:, 1] = fused[:, 1] + 0.5 * correction
        residual_only = torch.stack([-0.5 * correction, 0.5 * correction], dim=-1)
        baseline_margin = baseline_logits[:, 1] - baseline_logits[:, 0]
        baseline_prob = torch.softmax(baseline_logits, dim=1)[:, 1]
        return ArchResidualPartOutput(
            logits=fused,
            baseline_logits=baseline_logits,
            residual_only_logits=residual_only,
            correction=correction,
            baseline_margin=baseline_margin,
            baseline_signal_prob=baseline_prob,
            canonical_inputs=baseline_output.canonical_inputs,
            branch_diagnostics=diagnostics,
            config=self.config,
        )

    def forward(
        self,
        tokens: Any,
        mask: Any,
        *,
        return_outputs: bool = False,
        max_constits: int | None = None,
    ) -> Any:
        output = self.forward_outputs(tokens, mask, max_constits=max_constits)
        return output if bool(return_outputs) else output.logits

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "contract": ARCH_RESIDUAL_MODEL_CONTRACT,
            "step": ARCH_RESIDUAL_MODEL_STEP,
            "config": self.config.to_dict(),
            "baseline_checkpoint": dict(getattr(self.baseline, "baseline_checkpoint_report", None) or {}),
            "baseline_is_frozen": all(not param.requires_grad for param in self.baseline.parameters()),
            "label_names": list(self.config.label_names),
        }


def build_arch_residual_part_model(
    config: ArchResidualExpertConfig | Mapping[str, Any] | None = None,
    *,
    baseline: Any | None = None,
) -> ArchResidualPartModel:
    return ArchResidualPartModel(config, baseline=baseline)


__all__ = [
    "ARCH_RESIDUAL_ARCHITECTURES",
    "ARCH_RESIDUAL_MODEL_CONTRACT",
    "ARCH_RESIDUAL_MODEL_STEP",
    "ArchResidualExpertConfig",
    "ArchResidualPartModel",
    "ArchResidualPartOutput",
    "FrozenHLTPartBaseline",
    "FrozenHLTPartOutput",
    "build_arch_residual_part_model",
    "normalize_architecture",
]
