"""Control residual-field sources for the local particle residual-field tagger."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT = "local_particle_residual_field_controls_v1"

RESIDUAL_FIELD_SOURCE_RANDOM = "random"
RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE = "cross_jet_shuffle"
RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE = "within_jet_shuffle"
RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED = "radius_permuted"
RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET = "learned_no_target"

CONTROL_RESIDUAL_FIELD_SOURCES = (
    RESIDUAL_FIELD_SOURCE_RANDOM,
    RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
    RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
)


_CONTROL_ALIASES = {
    "noise": RESIDUAL_FIELD_SOURCE_RANDOM,
    "random_fields": RESIDUAL_FIELD_SOURCE_RANDOM,
    "cross_shuffle": RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    "cross_jet": RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    "cross_jet_shuffled": RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    "shuffle_jets": RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    "within_shuffle": RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    "within_jet": RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    "within_jet_shuffled": RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    "shuffle_particles": RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    "radius_shuffle": RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
    "radius_permutation": RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
    "radius_permuted": RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
    "learned_control": RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    "learned_notarget": RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    "learned_no_target": RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    "no_target": RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
}


def normalize_residual_field_control_source(value: str) -> str:
    clean = str(value).strip().lower().replace("-", "_")
    if clean in CONTROL_RESIDUAL_FIELD_SOURCES:
        return clean
    if clean in _CONTROL_ALIASES:
        return _CONTROL_ALIASES[clean]
    raise ValueError(f"unknown residual-field control source {value!r}")


@dataclass(frozen=True)
class LocalResidualFieldControlConfig:
    """Configuration for non-oracle control residual-field sources."""

    seed: int = 9173
    noise_scale: float = 1.0
    random_match_target_std: bool = True
    learned_hidden_dim: int = 128
    learned_dropout: float = 0.05
    field_names: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))
        scale = float(self.noise_scale)
        if scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        object.__setattr__(self, "noise_scale", scale)
        hidden_dim = int(self.learned_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("learned_hidden_dim must be positive")
        object.__setattr__(self, "learned_hidden_dim", hidden_dim)
        dropout = float(self.learned_dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("learned_dropout must be in [0, 1)")
        object.__setattr__(self, "learned_dropout", dropout)
        object.__setattr__(self, "random_match_target_std", bool(self.random_match_target_std))
        object.__setattr__(self, "field_names", tuple(str(name) for name in self.field_names))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT
        payload["field_names"] = list(self.field_names)
        return payload


@dataclass(frozen=True)
class LocalResidualFieldControlOutput:
    """Output from a control residual-field source."""

    fields: torch.Tensor
    diagnostics: Mapping[str, Any]


def _mask_fields(fields: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return fields
    if mask.ndim == 3:
        mask = mask.squeeze(1)
    return fields * mask.to(device=fields.device, dtype=fields.dtype).unsqueeze(-1)


def _seed_for_batch(seed: int, indices: torch.Tensor | None, *, device: Any) -> torch.Generator:
    generator = torch.Generator(device=device)
    offset = 0
    if indices is not None and int(indices.numel()) > 0:
        safe_indices = indices.detach().to(device="cpu", dtype=torch.long)
        offset = int(safe_indices.sum().item()) % 1_000_003
    generator.manual_seed(int(seed) + offset)
    return generator


def _field_std(fields: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        selected = fields.reshape(-1, fields.shape[-1])
    else:
        if mask.ndim == 3:
            mask = mask.squeeze(1)
        valid = mask.to(device=fields.device, dtype=torch.bool)
        selected = fields[valid] if bool(valid.any().detach().cpu().item()) else fields.reshape(-1, fields.shape[-1])
    if int(selected.numel()) == 0:
        return torch.ones((fields.shape[-1],), device=fields.device, dtype=fields.dtype)
    return selected.detach().float().std(dim=0, unbiased=False).to(device=fields.device, dtype=fields.dtype).clamp_min(1.0e-6)


def _cross_jet_shuffle(fields: torch.Tensor, mask: torch.Tensor | None, generator: torch.Generator) -> torch.Tensor:
    if int(fields.shape[0]) <= 1:
        return _mask_fields(fields.clone(), mask)
    permutation = torch.randperm(int(fields.shape[0]), device=fields.device, generator=generator)
    if bool(torch.equal(permutation, torch.arange(int(fields.shape[0]), device=fields.device))):
        permutation = torch.roll(permutation, shifts=1)
    return _mask_fields(fields[permutation].clone(), mask)


def _within_jet_shuffle(fields: torch.Tensor, mask: torch.Tensor | None, generator: torch.Generator) -> torch.Tensor:
    output = torch.zeros_like(fields)
    if mask is None:
        valid_mask = torch.ones(fields.shape[:2], device=fields.device, dtype=torch.bool)
    else:
        valid_mask = mask.squeeze(1) if mask.ndim == 3 else mask
        valid_mask = valid_mask.to(device=fields.device, dtype=torch.bool)
    for batch_index in range(int(fields.shape[0])):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        if int(valid_indices.numel()) == 0:
            continue
        permutation = valid_indices[torch.randperm(int(valid_indices.numel()), device=fields.device, generator=generator)]
        output[batch_index, valid_indices] = fields[batch_index, permutation]
    return output


def radius_field_groups(field_names: Sequence[str], field_dim: int | None = None) -> tuple[tuple[int, ...], ...]:
    """Return equal-length field-index groups for radius-tagged field names.

    The local residual target builder names radius channels as ``r0p02.foo``,
    ``r0p05.foo``, etc.  Non-radius diagnostic/reliability fields are excluded
    from these groups and therefore remain fixed under radius permutation.
    """

    names = tuple(str(name) for name in field_names)
    if not names:
        return ()
    if field_dim is not None and len(names) != int(field_dim):
        raise ValueError("field_names length must match field_dim when provided")
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        prefix, sep, _suffix = name.partition(".")
        if sep and prefix.startswith("r") and "p" in prefix:
            grouped.setdefault(prefix, []).append(index)
    if len(grouped) < 2:
        return ()
    lengths = {len(values) for values in grouped.values()}
    if len(lengths) != 1:
        raise ValueError(f"radius field groups must have equal lengths, got {sorted(lengths)}")
    return tuple(tuple(grouped[key]) for key in sorted(grouped))


def _radius_permuted_fields(fields: torch.Tensor, field_names: Sequence[str], generator: torch.Generator) -> tuple[torch.Tensor, tuple[int, ...]]:
    groups = radius_field_groups(field_names, int(fields.shape[-1]))
    if len(groups) < 2:
        return fields.clone(), tuple(range(len(groups)))
    permutation = torch.randperm(len(groups), device=fields.device, generator=generator)
    if bool(torch.equal(permutation, torch.arange(len(groups), device=fields.device))):
        permutation = torch.roll(permutation, shifts=1)
    output = fields.clone()
    for target_group_index, source_group_index in enumerate(permutation.detach().cpu().tolist()):
        target_indices = torch.as_tensor(groups[target_group_index], device=fields.device, dtype=torch.long)
        source_indices = torch.as_tensor(groups[int(source_group_index)], device=fields.device, dtype=torch.long)
        output[..., target_indices] = fields[..., source_indices]
    return output, tuple(int(value) for value in permutation.detach().cpu().tolist())


class LocalResidualFieldControlGenerator(torch.nn.Module):
    """Trainable no-target control that emits residual-field-shaped channels."""

    def __init__(
        self,
        *,
        field_dim: int,
        particle_dim: int = RAW_TOKEN_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.field_dim = int(field_dim)
        self.particle_dim = int(particle_dim)
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(int(particle_dim)),
            torch.nn.Linear(int(particle_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(field_dim)),
        )
        if bool(zero_init_output):
            last = self.net[-1]
            if isinstance(last, torch.nn.Linear):
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> LocalResidualFieldControlOutput:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, P, C]")
        fields = self.net(tokens)
        fields = _mask_fields(fields, mask)
        diagnostics = {
            "contract": LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT,
            "control_source": RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
            "learned_no_target": True,
        }
        return LocalResidualFieldControlOutput(fields=fields, diagnostics=diagnostics)


def apply_local_residual_field_control(
    *,
    source: str,
    target_fields: torch.Tensor | None,
    mask: torch.Tensor | None,
    field_names: Sequence[str],
    config: LocalResidualFieldControlConfig | Mapping[str, Any] | None = None,
    tokens: torch.Tensor | None = None,
    generator: LocalResidualFieldControlGenerator | None = None,
    indices: torch.Tensor | None = None,
) -> LocalResidualFieldControlOutput:
    """Create control residual fields with the same shape as the real targets."""

    normalized = normalize_residual_field_control_source(source)
    control_config = config if isinstance(config, LocalResidualFieldControlConfig) else LocalResidualFieldControlConfig(**dict(config or {}))
    if normalized == RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET:
        if generator is None:
            raise ValueError("learned_no_target control requires a generator module")
        if tokens is None:
            raise ValueError("learned_no_target control requires HLT tokens")
        output = generator(tokens, mask)
        diagnostics = dict(output.diagnostics)
        diagnostics["control_config"] = control_config.to_dict()
        return LocalResidualFieldControlOutput(fields=output.fields, diagnostics=diagnostics)
    if target_fields is None:
        raise ValueError(f"{normalized} control requires target_fields for shape/statistics")
    target_fields = _mask_fields(target_fields, mask)
    generator_state = _seed_for_batch(int(control_config.seed), indices, device=target_fields.device)
    diagnostics: dict[str, Any] = {
        "contract": LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT,
        "control_source": normalized,
        "control_config": control_config.to_dict(),
    }
    if normalized == RESIDUAL_FIELD_SOURCE_RANDOM:
        noise = torch.randn(
            target_fields.shape,
            generator=generator_state,
            device=target_fields.device,
            dtype=target_fields.dtype,
        )
        if bool(control_config.random_match_target_std):
            noise = noise * _field_std(target_fields, mask).view(1, 1, -1)
            diagnostics["random_match_target_std"] = True
        fields = noise * float(control_config.noise_scale)
    elif normalized == RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE:
        fields = _cross_jet_shuffle(target_fields, mask, generator_state)
    elif normalized == RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE:
        fields = _within_jet_shuffle(target_fields, mask, generator_state)
    elif normalized == RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED:
        fields, permutation = _radius_permuted_fields(target_fields, field_names, generator_state)
        diagnostics["radius_permutation"] = list(permutation)
    else:  # pragma: no cover - guarded by normalize_residual_field_control_source
        raise ValueError(f"unsupported control source {normalized!r}")
    return LocalResidualFieldControlOutput(fields=_mask_fields(fields, mask), diagnostics=diagnostics)


__all__ = [
    "LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT",
    "RESIDUAL_FIELD_SOURCE_RANDOM",
    "RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE",
    "RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE",
    "RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED",
    "RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET",
    "CONTROL_RESIDUAL_FIELD_SOURCES",
    "LocalResidualFieldControlConfig",
    "LocalResidualFieldControlGenerator",
    "LocalResidualFieldControlOutput",
    "apply_local_residual_field_control",
    "normalize_residual_field_control_source",
    "radius_field_groups",
]
