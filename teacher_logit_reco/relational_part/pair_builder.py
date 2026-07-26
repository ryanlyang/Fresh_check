"""Canonical non-tree relation composition and shared Weaver pair stem."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .normalization import validate_relation_normalization_artifact
from .pair_base import (
    STANDARD_FOUR_CHANNELS,
    build_standard_four_pair_features,
    require_torch,
)
from .relation_pid_charge import (
    CHARGE_ENCODED_DIMENSION,
    PID_ENCODED_DIMENSION,
    ChargeEncoder,
    PIDEncoder,
)
from .relation_pt import PT_ENCODED_DIMENSION, PTEncoder, valid_pair_mask
from .relation_track import TRACK_ENCODED_DIMENSION, TrackEncoder
from .relation_density import DENSITY_ENCODED_DIMENSION, DensityEncoder

try:
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


STEP3_FAMILY_ORDER = ("PT", "PID", "CHARGE")
STEP3_FAMILY_DIMENSIONS = {
    "PT": PT_ENCODED_DIMENSION,
    "PID": PID_ENCODED_DIMENSION,
    "CHARGE": CHARGE_ENCODED_DIMENSION,
}
SUPPORTED_FAMILY_ORDER = ("PT", "TRACK", "PID", "CHARGE", "DENSITY")
SUPPORTED_FAMILY_DIMENSIONS = {
    **STEP3_FAMILY_DIMENSIONS,
    "TRACK": TRACK_ENCODED_DIMENSION,
    "DENSITY": DENSITY_ENCODED_DIMENSION,
}
_EXPECTED_FEATURE_INDICES = {
    "part_charge": 5,
    "part_isChargedHadron": 6,
    "part_isNeutralHadron": 7,
    "part_isPhoton": 8,
    "part_isElectron": 9,
    "part_isMuon": 10,
}


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def _validate_feature_layout() -> None:
    for name, index in _EXPECTED_FEATURE_INDICES.items():
        if len(PF_FEATURE_NAMES) <= index or PF_FEATURE_NAMES[index] != name:
            raise RuntimeError(
                f"canonical Particle Transformer feature layout drifted at {index}: "
                f"expected {name!r}"
            )


def canonical_step3_families(families: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(family) for family in families)
    if len(values) != len(set(values)):
        raise ValueError("relation family list contains duplicates")
    unknown = sorted(set(values) - set(STEP3_FAMILY_ORDER))
    if unknown:
        raise ValueError(f"Step 3 cannot build relation families {unknown}")
    expected = tuple(name for name in STEP3_FAMILY_ORDER if name in values)
    if values != expected:
        raise ValueError(
            f"relation families must follow canonical Step-3 order {expected}"
        )
    if not values:
        raise ValueError("Step-3 relation builder requires at least one family")
    return values


def canonical_supported_families(
    families: Sequence[str],
) -> tuple[str, ...]:
    values = tuple(str(family) for family in families)
    if len(values) != len(set(values)):
        raise ValueError("relation family list contains duplicates")
    unknown = sorted(set(values) - set(SUPPORTED_FAMILY_ORDER))
    if unknown:
        raise ValueError(f"non-tree builder cannot build relation families {unknown}")
    expected = tuple(name for name in SUPPORTED_FAMILY_ORDER if name in values)
    if values != expected:
        raise ValueError(
            f"relation families must follow canonical order {expected}"
        )
    if not values:
        raise ValueError("relation builder requires at least one family")
    return values


class RelationalPairBuilder(_ModuleBase):
    """Build ``[base4, encoded families...]`` without a persistent pair cache."""

    def __init__(
        self,
        families: Sequence[str],
        *,
        normalization_artifact: Mapping[str, Any],
        weaver_module: Any,
    ) -> None:
        torch = require_torch()
        super().__init__()
        _validate_feature_layout()
        self.families = canonical_supported_families(families)
        self.normalization_sha256 = validate_relation_normalization_artifact(
            normalization_artifact
        )
        modules: dict[str, Any] = {}
        if "PT" in self.families:
            modules["PT"] = PTEncoder(normalization_artifact)
        if "PID" in self.families:
            modules["PID"] = PIDEncoder()
        if "CHARGE" in self.families:
            modules["CHARGE"] = ChargeEncoder(normalization_artifact)
        if "TRACK" in self.families:
            modules["TRACK"] = TrackEncoder(normalization_artifact)
        if "DENSITY" in self.families:
            modules["DENSITY"] = DensityEncoder(normalization_artifact)
        self.encoders = torch.nn.ModuleDict(modules)
        object.__setattr__(self, "_weaver_module", weaver_module)
        self.output_dimension = STANDARD_FOUR_CHANNELS + sum(
            SUPPORTED_FAMILY_DIMENSIONS[family] for family in self.families
        )

    def forward(
        self,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        *,
        return_details: bool = False,
    ) -> Any:
        base4 = build_standard_four_pair_features(
            lorentz_vectors,
            mask=mask,
            module=self._weaver_module,
        )
        encoded: dict[str, Any] = {}
        if "PT" in self.encoders:
            encoded["PT"] = self.encoders["PT"](lorentz_vectors, mask)
        if "PID" in self.encoders:
            encoded["PID"] = self.encoders["PID"](features[:, 6:11], mask)
        if "CHARGE" in self.encoders:
            encoded["CHARGE"] = self.encoders["CHARGE"](features[:, 5], mask)
        if "TRACK" in self.encoders or "DENSITY" in self.encoders:
            if (
                not isinstance(raw_tokens, _torch.Tensor)
                or raw_tokens.ndim != 3
                or tuple(raw_tokens.shape[:2])
                != (int(features.shape[0]), int(features.shape[2]))
                or int(raw_tokens.shape[2]) != 14
            ):
                raise ValueError(
                    "TRACK/DENSITY require raw HLT tokens [batch,particles,14]"
                )
            if raw_tokens.device != features.device:
                raise ValueError("raw HLT tokens must share the model device")
        if "TRACK" in self.encoders:
            encoded["TRACK"] = self.encoders["TRACK"](raw_tokens, mask)
        if "DENSITY" in self.encoders:
            encoded["DENSITY"] = self.encoders["DENSITY"](raw_tokens, mask)
        combined = _torch.cat(
            [base4, *(encoded[family] for family in self.families)],
            dim=1,
        )
        if int(combined.shape[1]) != self.output_dimension:
            raise RuntimeError("combined pair tensor dimension drifted")
        if not bool(_torch.isfinite(combined).all()):
            raise FloatingPointError("combined relation tensor is nonfinite")
        if return_details:
            return {
                "base4": base4,
                "encoded": encoded,
                "combined": combined,
                "pair_mask": valid_pair_mask(mask),
            }
        return combined

    def metadata(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "family_dimensions": {
                family: SUPPORTED_FAMILY_DIMENSIONS[family]
                for family in self.families
            },
            "family_raw_feature_names": {
                family: list(self.encoders[family].raw_feature_names)
                for family in self.families
            },
            "canonical_concatenation_order": ["base4", *self.families],
            "base4_dimension": STANDARD_FOUR_CHANNELS,
            "combined_dimension": self.output_dimension,
            "normalization_sha256": self.normalization_sha256,
            "persistent_N_by_N_cache": False,
        }


Step3PairBuilder = RelationalPairBuilder


class SharedDirectionalPairEmbed(_ModuleBase):
    """Use one Weaver pair stem for the complete directional pair tensor."""

    def __init__(self, reference_pair_embed: Any, *, input_dimension: int) -> None:
        torch = require_torch()
        super().__init__()
        if not isinstance(reference_pair_embed, torch.nn.Module):
            raise TypeError("reference_pair_embed must be a torch module")
        if int(getattr(reference_pair_embed, "pairwise_lv_dim", -1)) != 0:
            raise RuntimeError("shared relational pair stem must disable internal LV features")
        if int(getattr(reference_pair_embed, "pairwise_input_dim", -1)) != int(
            input_dimension
        ):
            raise RuntimeError("shared relational pair stem input dimension drifted")
        stem = getattr(reference_pair_embed, "fts_embed", None)
        if not isinstance(stem, torch.nn.Module):
            raise RuntimeError("Weaver pair stem lacks fts_embed")
        self.fts_embed = stem
        self.pairwise_lv_dim = 0
        self.pairwise_input_dim = int(input_dimension)
        self.out_dim = int(getattr(reference_pair_embed, "out_dim"))
        self.remove_self_pair = bool(
            getattr(reference_pair_embed, "remove_self_pair", False)
        )
        if self.remove_self_pair:
            raise RuntimeError("Step-3 shared pair stem must retain self pairs")
        self.is_symmetric = False

    def forward(self, v: Any, uu: Any = None, mask: Any | None = None) -> Any:
        torch = require_torch()
        if not isinstance(uu, torch.Tensor) or uu.ndim != 4:
            raise ValueError("shared pair tensor must have shape [batch,C,N,N]")
        batch, channels, query, context = map(int, uu.shape)
        if channels != self.pairwise_input_dim or query != context:
            raise ValueError("shared pair tensor has an incompatible shape")
        if mask is None:
            if not isinstance(v, torch.Tensor) or tuple(v.shape) != (
                batch,
                4,
                query,
            ):
                raise ValueError("masked vectors are required to infer valid pairs")
            mask = v.ne(0).any(dim=1, keepdim=True)
        if tuple(mask.shape) != (batch, 1, query):
            raise ValueError("shared pair mask has an incompatible shape")
        pair_mask = valid_pair_mask(mask)
        i0, _, i2, i3 = pair_mask.nonzero(as_tuple=True)
        if int(i0.numel()) == 0:
            raise ValueError("shared pair stem received an all-empty batch")
        gathered = uu.permute(0, 2, 3, 1)[i0, i2, i3, :]
        elements = self.fts_embed(gathered.T.unsqueeze(0)).squeeze(0).T
        output = elements.new_zeros(batch, query, query, self.out_dim)
        output[i0, i2, i3, :] = elements
        return output.permute(0, 3, 1, 2).contiguous()


__all__ = [
    "STEP3_FAMILY_DIMENSIONS",
    "STEP3_FAMILY_ORDER",
    "SUPPORTED_FAMILY_DIMENSIONS",
    "SUPPORTED_FAMILY_ORDER",
    "RelationalPairBuilder",
    "SharedDirectionalPairEmbed",
    "Step3PairBuilder",
    "canonical_step3_families",
    "canonical_supported_families",
]
