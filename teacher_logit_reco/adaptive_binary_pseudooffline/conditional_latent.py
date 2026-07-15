"""Conditional spline prior and training-only hierarchy posterior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .hierarchy_alignment import HierarchyTargetTensors
from .targets import ABPH_LEVEL_CAPACITIES, ROOT_FEATURE_NAMES


try:  # Keep cache/schema utilities importable without a training environment.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ABPH_CONDITIONAL_LATENT_CONTRACT = "adaptive_binary_pseudooffline_conditional_latent_v1"
ABPH_PRIMARY_LATENT_DIM = 64
ABPH_PRIMARY_SPLINE_LAYERS = 8
ABPH_PRIMARY_STOCHASTIC_HYPOTHESES = 4
ABPH_FIXED_EVALUATION_SEED = 24071


@dataclass(frozen=True)
class ConditionalLatentConfig:
    hlt_evidence_dim: int = 192
    hlt_jet_dim: int = 256
    root_semantic_dim: int = 256
    context_dim: int = 256
    latent_dim: int = ABPH_PRIMARY_LATENT_DIM
    num_context_queries: int = 4
    context_heads: int = 8
    context_blocks: int = 2
    context_ffn_dim: int = 1024
    posterior_blocks: int = 3
    mean_quadrature_samples: int = 64
    spline_layers: int = ABPH_PRIMARY_SPLINE_LAYERS
    spline_bins: int = 8
    spline_hidden_dim: int = 512
    spline_tail_bound: float = 5.0
    minimum_bin_width: float = 1.0e-3
    minimum_bin_height: float = 1.0e-3
    minimum_derivative: float = 1.0e-3
    log_scale_min: float = -6.0
    log_scale_max: float = 3.0
    dropout: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "hlt_evidence_dim",
            "hlt_jet_dim",
            "root_semantic_dim",
            "context_dim",
            "latent_dim",
            "num_context_queries",
            "context_heads",
            "context_blocks",
            "context_ffn_dim",
            "posterior_blocks",
            "mean_quadrature_samples",
            "spline_layers",
            "spline_bins",
            "spline_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.context_dim) % int(self.context_heads):
            raise ValueError("context_dim must be divisible by context_heads")
        if int(self.latent_dim) % 2:
            raise ValueError("spline coupling requires an even latent dimension")
        if int(self.spline_bins) < 4:
            raise ValueError("at least four spline bins are required")
        if float(self.spline_tail_bound) <= 0.0:
            raise ValueError("spline_tail_bound must be positive")
        if float(self.minimum_bin_width) * int(self.spline_bins) >= 1.0:
            raise ValueError("minimum spline widths exhaust the interval")
        if float(self.minimum_bin_height) * int(self.spline_bins) >= 1.0:
            raise ValueError("minimum spline heights exhaust the interval")
        if float(self.minimum_derivative) <= 0.0:
            raise ValueError("minimum_derivative must be positive")
        if float(self.log_scale_min) >= float(self.log_scale_max):
            raise ValueError("latent log-scale bounds are reversed")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "contract": ABPH_CONDITIONAL_LATENT_CONTRACT,
                "prior_inputs": [
                    "hlt_particle_evidence",
                    "hlt_jet_embedding",
                    "semantic_root_queries",
                    "shared_compiled_root_ledger",
                ],
                "posterior_additional_input": "offline_target_hierarchy_training_only",
                "node_local_noise": False,
                "root_sampling": False,
            }
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload


@dataclass(frozen=True)
class ConditionalGaussian:
    mean: Any
    log_scale: Any

    def log_prob(self, value: Any) -> Any:
        torch = require_torch()
        scale = self.log_scale.exp()
        standardized = (value - self.mean) / scale
        return (
            -0.5 * standardized.square()
            - self.log_scale
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)

    def sample(self, *, seed: int | None = None) -> Any:
        torch = require_torch()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.mean.device)
            generator.manual_seed(int(seed))
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + self.log_scale.exp() * noise


@dataclass(frozen=True)
class VariationalLatentSample:
    latent: Any
    posterior_log_prob: Any
    prior_log_prob: Any
    monte_carlo_kl: Any
    sampling_seed: int | None
    diagnostics: Mapping[str, Any]


class _ContextQueryBlock(_ModuleBase):
    def __init__(self, config: ConditionalLatentConfig) -> None:
        torch = require_torch()
        super().__init__()
        kwargs = {
            "embed_dim": config.context_dim,
            "num_heads": config.context_heads,
            "dropout": config.dropout,
            "batch_first": True,
        }
        self.self_attention = torch.nn.MultiheadAttention(**kwargs)
        self.cross_attention = torch.nn.MultiheadAttention(**kwargs)
        self.norm_self = torch.nn.LayerNorm(config.context_dim)
        self.norm_cross = torch.nn.LayerNorm(config.context_dim)
        self.norm_ffn = torch.nn.LayerNorm(config.context_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.context_ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.context_ffn_dim, config.context_dim),
            torch.nn.Dropout(config.dropout),
        )

    def forward(self, queries: Any, evidence: Any, evidence_mask: Any) -> Any:
        normalized = self.norm_self(queries)
        update, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + update
        update, _ = self.cross_attention(
            self.norm_cross(queries),
            evidence,
            evidence,
            key_padding_mask=~evidence_mask,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.norm_ffn(queries))


class ConditionalLatentContextEncoder(_ModuleBase):
    """Deployable HLT-only context for the below-root latent distribution."""

    def __init__(self, config: ConditionalLatentConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.evidence_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(config.hlt_evidence_dim),
            torch.nn.Linear(config.hlt_evidence_dim, config.context_dim),
        )
        self.jet_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(config.hlt_jet_dim),
            torch.nn.Linear(config.hlt_jet_dim, config.context_dim),
        )
        self.root_semantic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(config.root_semantic_dim),
            torch.nn.Linear(config.root_semantic_dim, config.context_dim),
        )
        self.root_ledger_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(len(ROOT_FEATURE_NAMES)),
            torch.nn.Linear(len(ROOT_FEATURE_NAMES), config.context_dim),
        )
        self.query_tokens = torch.nn.Parameter(
            torch.empty(1, config.num_context_queries, config.context_dim)
        )
        torch.nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.blocks = torch.nn.ModuleList(
            [_ContextQueryBlock(config) for _ in range(config.context_blocks)]
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(4 * config.context_dim),
            torch.nn.Linear(4 * config.context_dim, config.context_ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.context_ffn_dim, config.context_dim),
        )

    def forward(
        self,
        hlt_particle_evidence: Any,
        hlt_particle_mask: Any,
        hlt_jet_embedding: Any,
        root_semantic_tokens: Any,
        shared_root_ledger: Any,
    ) -> Any:
        torch = require_torch()
        particles_raw = torch.as_tensor(hlt_particle_evidence)
        mask = torch.as_tensor(hlt_particle_mask, device=particles_raw.device).bool()
        jet = torch.as_tensor(hlt_jet_embedding, device=particles_raw.device)
        semantics = torch.as_tensor(root_semantic_tokens, device=particles_raw.device)
        root = torch.as_tensor(shared_root_ledger, device=particles_raw.device).float()
        if particles_raw.ndim != 3 or particles_raw.shape[-1] != self.config.hlt_evidence_dim:
            raise ValueError(
                f"HLT evidence must have shape [B, N, {self.config.hlt_evidence_dim}]"
            )
        batch = int(particles_raw.shape[0])
        if mask.shape != particles_raw.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("HLT evidence mask is invalid or empty")
        if jet.shape != (batch, self.config.hlt_jet_dim):
            raise ValueError(f"HLT jet embedding must have shape [B, {self.config.hlt_jet_dim}]")
        if semantics.ndim != 3 or semantics.shape[0] != batch or semantics.shape[-1] != self.config.root_semantic_dim:
            raise ValueError("root semantic token shape does not match the latent context")
        if root.shape != (batch, len(ROOT_FEATURE_NAMES)):
            raise ValueError("shared root ledger has the wrong shape")
        evidence = self.evidence_projection(particles_raw)
        queries = self.query_tokens.expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, evidence, mask)
        query_summary = queries.mean(dim=1)
        semantic_summary = self.root_semantic_projection(semantics).mean(dim=1)
        return self.output(
            torch.cat(
                (
                    query_summary,
                    self.jet_projection(jet),
                    semantic_summary,
                    self.root_ledger_projection(root),
                ),
                dim=-1,
            )
        )


def _bounded_log_scale(values: Any, config: ConditionalLatentConfig) -> Any:
    midpoint = 0.5 * (config.log_scale_min + config.log_scale_max)
    half_range = 0.5 * (config.log_scale_max - config.log_scale_min)
    return midpoint + half_range * values.tanh()


def _rational_quadratic_spline(
    inputs: Any,
    parameters: Any,
    *,
    inverse: bool,
    config: ConditionalLatentConfig,
) -> tuple[Any, Any]:
    """Elementwise monotonic RQS with identity linear tails."""

    torch = require_torch()
    bins = int(config.spline_bins)
    expected = 3 * bins - 1
    if parameters.shape != (*inputs.shape, expected):
        raise ValueError("rational-quadratic spline parameter shape mismatch")
    raw_widths = parameters[..., :bins]
    raw_heights = parameters[..., bins : 2 * bins]
    raw_derivatives = parameters[..., 2 * bins :]
    interval = 2.0 * float(config.spline_tail_bound)
    widths = float(config.minimum_bin_width) + (
        1.0 - float(config.minimum_bin_width) * bins
    ) * raw_widths.softmax(dim=-1)
    heights = float(config.minimum_bin_height) + (
        1.0 - float(config.minimum_bin_height) * bins
    ) * raw_heights.softmax(dim=-1)
    widths = widths * interval
    heights = heights * interval
    cumulative_widths = torch.nn.functional.pad(widths.cumsum(dim=-1), (1, 0))
    cumulative_heights = torch.nn.functional.pad(heights.cumsum(dim=-1), (1, 0))
    cumulative_widths = cumulative_widths + -float(config.spline_tail_bound)
    cumulative_heights = cumulative_heights + -float(config.spline_tail_bound)
    internal_derivatives = float(config.minimum_derivative) + torch.nn.functional.softplus(
        raw_derivatives
    )
    derivatives = torch.nn.functional.pad(internal_derivatives, (1, 1), value=1.0)
    inside = inputs.abs() <= float(config.spline_tail_bound)
    bounded = inputs.clamp(
        -float(config.spline_tail_bound) + 1.0e-6,
        float(config.spline_tail_bound) - 1.0e-6,
    )
    knots = cumulative_heights if inverse else cumulative_widths
    bin_index = (bounded[..., None] >= knots[..., 1:-1]).sum(dim=-1)

    def gather(values: Any, offset: int = 0) -> Any:
        return values.gather(-1, (bin_index + offset).unsqueeze(-1)).squeeze(-1)

    input_cumulative_width = gather(cumulative_widths)
    input_bin_width = gather(widths)
    input_cumulative_height = gather(cumulative_heights)
    input_bin_height = gather(heights)
    delta = input_bin_height / input_bin_width
    derivative_left = gather(derivatives)
    derivative_right = gather(derivatives, 1)
    if inverse:
        y_delta = (bounded - input_cumulative_height) / input_bin_height
        common = derivative_left + derivative_right - 2.0 * delta
        quadratic_a = y_delta * common + delta - derivative_left
        quadratic_b = derivative_left - y_delta * common
        quadratic_c = -delta * y_delta
        discriminant = (
            quadratic_b.square() - 4.0 * quadratic_a * quadratic_c
        ).clamp_min(0.0)
        denominator = -quadratic_b - torch.sqrt(discriminant)
        safe_denominator = torch.where(
            denominator.abs() < 1.0e-12,
            torch.where(
                denominator < 0.0,
                torch.full_like(denominator, -1.0e-12),
                torch.full_like(denominator, 1.0e-12),
            ),
            denominator,
        )
        safe_linear = torch.where(
            quadratic_b.abs() < 1.0e-12,
            torch.where(
                quadratic_b < 0.0,
                torch.full_like(quadratic_b, -1.0e-12),
                torch.full_like(quadratic_b, 1.0e-12),
            ),
            quadratic_b,
        )
        theta_quadratic = (2.0 * quadratic_c) / safe_denominator
        theta_linear = -quadratic_c / safe_linear
        theta = torch.where(quadratic_a.abs() < 1.0e-8, theta_linear, theta_quadratic)
        theta = theta.clamp(0.0, 1.0)
        outputs_inside = input_cumulative_width + theta * input_bin_width
    else:
        theta = (bounded - input_cumulative_width) / input_bin_width
        theta_one_minus = theta * (1.0 - theta)
        numerator = input_bin_height * (
            delta * theta.square() + derivative_left * theta_one_minus
        )
        denominator_forward = delta + (
            derivative_left + derivative_right - 2.0 * delta
        ) * theta_one_minus
        outputs_inside = input_cumulative_height + numerator / denominator_forward
    theta_one_minus = theta * (1.0 - theta)
    denominator_derivative = delta + (
        derivative_left + derivative_right - 2.0 * delta
    ) * theta_one_minus
    derivative_numerator = delta.square() * (
        derivative_right * theta.square()
        + 2.0 * delta * theta_one_minus
        + derivative_left * (1.0 - theta).square()
    )
    log_abs_derivative = torch.log(derivative_numerator.clamp_min(1.0e-12)) - 2.0 * torch.log(
        denominator_derivative.clamp_min(1.0e-12)
    )
    if inverse:
        log_abs_derivative = -log_abs_derivative
    outputs = torch.where(inside, outputs_inside, inputs)
    logdet = torch.where(inside, log_abs_derivative, torch.zeros_like(inputs))
    return outputs, logdet


class ConditionalSplineCoupling(_ModuleBase):
    def __init__(self, config: ConditionalLatentConfig, parity: int) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        identity = tuple(range(int(parity) % 2, config.latent_dim, 2))
        transformed = tuple(index for index in range(config.latent_dim) if index not in identity)
        self.register_buffer("identity_indices", torch.tensor(identity, dtype=torch.long))
        self.register_buffer("transformed_indices", torch.tensor(transformed, dtype=torch.long))
        parameters_per_dimension = 3 * config.spline_bins - 1
        self.conditioner = torch.nn.Sequential(
            torch.nn.LayerNorm(len(identity) + config.context_dim),
            torch.nn.Linear(len(identity) + config.context_dim, config.spline_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.spline_hidden_dim, config.spline_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(
                config.spline_hidden_dim,
                len(transformed) * parameters_per_dimension,
            ),
        )
        self.parameters_per_dimension = parameters_per_dimension

    def forward(self, values: Any, context: Any, *, inverse: bool = False) -> tuple[Any, Any]:
        identity = values.index_select(-1, self.identity_indices)
        transformed = values.index_select(-1, self.transformed_indices)
        parameters = self.conditioner(require_torch().cat((identity, context), dim=-1))
        parameters = parameters.reshape(
            *transformed.shape, self.parameters_per_dimension
        )
        output_values, element_logdet = _rational_quadratic_spline(
            transformed,
            parameters,
            inverse=inverse,
            config=self.config,
        )
        output = values.clone()
        output.index_copy_(-1, self.transformed_indices, output_values)
        return output, element_logdet.sum(dim=-1)


class ConditionalSplinePrior(_ModuleBase):
    """Eight-layer conditional RQS prior over one coherent jet latent."""

    def __init__(self, config: ConditionalLatentConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.base_head = torch.nn.Sequential(
            torch.nn.LayerNorm(config.context_dim),
            torch.nn.Linear(config.context_dim, config.spline_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.spline_hidden_dim, 2 * config.latent_dim),
        )
        self.layers = torch.nn.ModuleList(
            [ConditionalSplineCoupling(config, parity=index) for index in range(config.spline_layers)]
        )

    def base_distribution(self, context: Any) -> ConditionalGaussian:
        raw = self.base_head(context)
        mean, log_scale_raw = raw.chunk(2, dim=-1)
        return ConditionalGaussian(mean, _bounded_log_scale(log_scale_raw, self.config))

    def transform(self, base_value: Any, context: Any) -> tuple[Any, Any]:
        value = base_value
        logdet = base_value.new_zeros(base_value.shape[:-1])
        for layer in self.layers:
            value, update = layer(value, context, inverse=False)
            logdet = logdet + update
        return value, logdet

    def inverse(self, latent: Any, context: Any) -> tuple[Any, Any]:
        value = latent
        logdet = latent.new_zeros(latent.shape[:-1])
        for layer in reversed(self.layers):
            value, update = layer(value, context, inverse=True)
            logdet = logdet + update
        return value, logdet

    def center(self, context: Any) -> Any:
        center, _ = self.transform(self.base_distribution(context).mean, context)
        return center

    def deterministic_mean(self, context: Any, *, seed: int = 17011) -> Any:
        """Differentiable fixed-quadrature estimate of E[z | HLT, root]."""

        torch = require_torch()
        count = int(self.config.mean_quadrature_samples)
        half = (count + 1) // 2
        generator = torch.Generator(device=context.device)
        generator.manual_seed(int(seed))
        positive = torch.randn(
            (half, self.config.latent_dim),
            dtype=context.dtype,
            device=context.device,
            generator=generator,
        )
        standard = torch.cat((positive, -positive), dim=0)[:count]
        base = self.base_distribution(context)
        values = (
            base.mean[:, None, :]
            + base.log_scale.exp()[:, None, :] * standard[None, :, :]
        )
        flat_context = context[:, None, :].expand(-1, count, -1).reshape(
            -1, context.shape[-1]
        )
        transformed, _ = self.transform(
            values.reshape(-1, self.config.latent_dim), flat_context
        )
        return transformed.reshape(context.shape[0], count, self.config.latent_dim).mean(dim=1)

    def sample(self, context: Any, *, count: int, seed: int) -> tuple[Any, Any]:
        torch = require_torch()
        if int(count) <= 0:
            raise ValueError("hypothesis sample count must be positive")
        base = self.base_distribution(context)
        generator = torch.Generator(device=context.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            (context.shape[0], int(count), self.config.latent_dim),
            dtype=context.dtype,
            device=context.device,
            generator=generator,
        )
        base_value = base.mean[:, None, :] + base.log_scale.exp()[:, None, :] * noise
        flat_context = context[:, None, :].expand(-1, int(count), -1).reshape(-1, context.shape[-1])
        flat_base = base_value.reshape(-1, self.config.latent_dim)
        latent, _ = self.transform(flat_base, flat_context)
        latent = latent.reshape(context.shape[0], int(count), self.config.latent_dim)
        return latent, self.log_prob(
            latent.reshape(-1, self.config.latent_dim), flat_context
        ).reshape(context.shape[0], int(count))

    def log_prob(self, latent: Any, context: Any) -> Any:
        base_value, inverse_logdet = self.inverse(latent, context)
        return self.base_distribution(context).log_prob(base_value) + inverse_logdet


class OfflineHierarchyPosteriorEncoder(_ModuleBase):
    """Training-only encoder over the complete offline target hierarchy."""

    def __init__(self, config: ConditionalLatentConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.root_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(len(ROOT_FEATURE_NAMES)),
            torch.nn.Linear(len(ROOT_FEATURE_NAMES), config.context_dim),
        )
        group_dim = len(ROOT_FEATURE_NAMES) + 11 + 2
        self.group_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(group_dim),
            torch.nn.Linear(group_dim, config.context_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.context_dim, config.context_dim),
        )
        self.level_embedding = torch.nn.Embedding(len(ABPH_LEVEL_CAPACITIES) + 1, config.context_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.context_dim,
            nhead=config.context_heads,
            dim_feedforward=config.context_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=config.posterior_blocks,
            enable_nested_tensor=False,
        )
        self.output_norm = torch.nn.LayerNorm(config.context_dim)

    def forward(self, targets: HierarchyTargetTensors) -> Any:
        torch = require_torch()
        device = self.level_embedding.weight.device
        dtype = self.level_embedding.weight.dtype
        root = torch.as_tensor(targets.root_ledger, device=device, dtype=dtype)
        tokens = [self.root_projection(root) + self.level_embedding.weight[0]]
        for depth, (ledger, support, mask, topology) in enumerate(
            zip(
                targets.level_ledgers,
                targets.level_supports,
                targets.level_masks,
                targets.level_topology,
            ),
            start=1,
        ):
            ledger = torch.as_tensor(ledger, device=device, dtype=dtype)
            support = torch.as_tensor(support, device=device, dtype=dtype)
            mask = torch.as_tensor(mask, device=device).bool()
            topology = torch.as_tensor(topology, device=device).to(torch.long)
            topology_one_hot = torch.nn.functional.one_hot(
                (topology - 1).clamp(0, 1), num_classes=2
            ).to(dtype)
            group = self.group_projection(torch.cat((ledger, support, topology_one_hot), dim=-1))
            weights = mask.to(group.dtype).unsqueeze(-1)
            pooled_mean = (group * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            pooled_max = group.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
            pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
            tokens.append(0.5 * (pooled_mean + pooled_max) + self.level_embedding.weight[depth])
        stacked = torch.stack(tokens, dim=1)
        return self.output_norm(self.transformer(stacked).mean(dim=1))


class TrainingOnlyHierarchyPosterior(_ModuleBase):
    def __init__(self, config: ConditionalLatentConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.target_encoder = OfflineHierarchyPosteriorEncoder(config)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * config.context_dim),
            torch.nn.Linear(2 * config.context_dim, config.context_ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.context_ffn_dim, 2 * config.latent_dim),
        )

    def distribution(self, prior_context: Any, targets: HierarchyTargetTensors) -> ConditionalGaussian:
        target_context = self.target_encoder(targets)
        raw = self.head(require_torch().cat((prior_context, target_context), dim=-1))
        mean, log_scale_raw = raw.chunk(2, dim=-1)
        return ConditionalGaussian(mean, _bounded_log_scale(log_scale_raw, self.config))

    def sample(
        self,
        prior_context: Any,
        targets: HierarchyTargetTensors,
        prior: ConditionalSplinePrior,
        *,
        seed: int | None = None,
    ) -> VariationalLatentSample:
        posterior = self.distribution(prior_context, targets)
        latent = posterior.sample(seed=seed)
        posterior_log_prob = posterior.log_prob(latent)
        prior_log_prob = prior.log_prob(latent, prior_context)
        kl = posterior_log_prob - prior_log_prob
        return VariationalLatentSample(
            latent=latent,
            posterior_log_prob=posterior_log_prob,
            prior_log_prob=prior_log_prob,
            monte_carlo_kl=kl,
            sampling_seed=None if seed is None else int(seed),
            diagnostics={
                "contract": ABPH_CONDITIONAL_LATENT_CONTRACT,
                "posterior_used_offline_target": True,
                "posterior_deployable": False,
                "finite": bool(
                    require_torch().isfinite(latent).all()
                    and require_torch().isfinite(kl).all()
                ),
            },
        )


__all__ = [
    "ABPH_CONDITIONAL_LATENT_CONTRACT",
    "ABPH_FIXED_EVALUATION_SEED",
    "ABPH_PRIMARY_LATENT_DIM",
    "ABPH_PRIMARY_SPLINE_LAYERS",
    "ABPH_PRIMARY_STOCHASTIC_HYPOTHESES",
    "ConditionalGaussian",
    "ConditionalLatentConfig",
    "ConditionalLatentContextEncoder",
    "ConditionalSplineCoupling",
    "ConditionalSplinePrior",
    "OfflineHierarchyPosteriorEncoder",
    "TrainingOnlyHierarchyPosterior",
    "VariationalLatentSample",
]
