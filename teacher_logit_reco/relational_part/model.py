"""Exact RPT_BASE wrapper with Weaver standard-four features passed as ``uu``."""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import default_part_config

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .capacity import (
    WIDE_CAPACITY_CONTRACT,
    pair_encoder_parameter_count,
    select_wide_widths,
)
from .determinism import DIAGNOSTIC_BIN_EDGES
from .region_normalization import validate_region_normalization
from .normalization import validate_relation_normalization_artifact
from .pair_builder import (
    SUPPORTED_FAMILY_DIMENSIONS,
    SharedDirectionalPairEmbed,
    RelationalPairBuilder,
    canonical_supported_families,
)
from .pair_base import (
    STANDARD_FOUR_CHANNELS,
    _import_weaver_module,
    build_standard_four_pair_features,
    require_torch,
)
from .registry import (
    CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    resolve_registered_run,
    validate_screening_registry,
)
from .relation_pid_charge import PID_CHARGE_RELATION_CONTRACT, pid_categories
from .relation_pt import PT_RELATION_CONTRACT
from .relation_pt import average_tied_descending_rank, valid_pair_mask
from .relation_track import TRACK_RELATION_CONTRACT
from .relation_density import DENSITY_RELATION_CONTRACT

try:  # Keep contract-only imports usable without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


RPT_BASE_MODEL_CONTRACT = "relational_part_rpt_base_model_v1"
STEP3_RELATIONAL_MODEL_CONTRACT = "relational_part_step3_model_v1"
STEP4_RELATIONAL_MODEL_CONTRACT = "relational_part_step4_model_v1"
STEP5_RELATIONAL_MODEL_CONTRACT = "relational_part_step5_model_v1"
STEP6_RELATIONAL_MODEL_CONTRACT = "relational_part_step6_model_v2"

RPT_BASE_CONFIG: dict[str, Any] = {
    "input_dim": 17,
    "num_classes": 10,
    "pair_input_dim": 4,
    "use_pre_activation_pair": False,
    "embed_dims": [128, 512, 128],
    "pair_embed_dims": [64, 64, 64],
    "num_heads": 8,
    "num_layers": 8,
    "num_cls_layers": 2,
    "block_params": None,
    "cls_block_params": {
        "dropout": 0,
        "attn_dropout": 0,
        "activation_dropout": 0,
    },
    "fc_params": [],
    "activation": "gelu",
    "trim": True,
    "for_inference": False,
}

RPT_BASE_EFFECTIVE_WEAVER_DEFAULTS: dict[str, Any] = {
    "pair_extra_dim": 0,
    "remove_self_pair": False,
    "use_amp": False,
}


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def exact_rpt_base_config() -> dict[str, Any]:
    """Return the locked config and fail if the canonical local wrapper drifts."""

    canonical = default_part_config(num_classes=10, model_size="base")
    if canonical != RPT_BASE_CONFIG:
        raise RuntimeError(
            "jetclass_fresh.hlt_baseline.default_part_config drifted from RPT_BASE"
        )
    return copy.deepcopy(RPT_BASE_CONFIG)


def _state_structure(module: Any) -> dict[str, tuple[tuple[int, ...], Any]]:
    return {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in module.state_dict().items()
    }


class ExplicitStandardFourPairEmbed(_ModuleBase):
    """State-dictionary-compatible replacement for Weaver's symmetric PairEmbed.

    Weaver 0.4 names the physical pair network ``embed`` and the extra ``uu``
    network ``fts_embed``.  Instantiating a second network for ``uu`` would
    change both parameters and state keys.  This adapter retains the original
    ``embed`` module under the same name, but gathers the precomputed standard
    four channels from ``uu`` before applying it.  Its dense and mask-sparse
    paths mirror the installed Weaver triangular contracts.
    """

    def __init__(self, reference_pair_embed: Any) -> None:
        torch = require_torch()
        super().__init__()
        if not isinstance(reference_pair_embed, torch.nn.Module):
            raise TypeError("reference_pair_embed must be a torch.nn.Module")
        pairwise_lv_dim = int(getattr(reference_pair_embed, "pairwise_lv_dim", -1))
        pairwise_input_dim = int(
            getattr(reference_pair_embed, "pairwise_input_dim", -1)
        )
        if pairwise_lv_dim != STANDARD_FOUR_CHANNELS or pairwise_input_dim != 0:
            raise RuntimeError(
                "RPT_BASE requires Weaver PairEmbed(4 physical, 0 extra), got "
                f"({pairwise_lv_dim}, {pairwise_input_dim})"
            )
        if getattr(reference_pair_embed, "is_symmetric", True) is not True:
            raise RuntimeError("Weaver standard-four PairEmbed must be symmetric")
        embed = getattr(reference_pair_embed, "embed", None)
        if not isinstance(embed, torch.nn.Module):
            raise RuntimeError("Weaver standard-four PairEmbed lacks embed module")

        self.embed = embed
        self.out_dim = int(getattr(reference_pair_embed, "out_dim"))
        self.remove_self_pair = bool(
            getattr(reference_pair_embed, "remove_self_pair", False)
        )
        self.pairwise_lv_dim = STANDARD_FOUR_CHANNELS
        self.pairwise_input_dim = 0
        self.is_symmetric = True
        sparse_eval = getattr(reference_pair_embed, "sparse_eval", (False, False))
        if (
            not isinstance(sparse_eval, (tuple, list))
            or len(sparse_eval) != 2
        ):
            raise RuntimeError(f"unsupported Weaver sparse_eval={sparse_eval!r}")
        self.sparse_eval = (bool(sparse_eval[0]), bool(sparse_eval[1]))

        before = _state_structure(reference_pair_embed)
        after = _state_structure(self)
        if before != after:
            raise RuntimeError(
                "explicit pair adapter changed PairEmbed state structure: "
                f"before={before}, after={after}"
            )

    @staticmethod
    def _validate(
        v: Any,
        uu: Any,
        mask: Any | None,
    ) -> tuple[int, int, Any | None]:
        torch = require_torch()
        if uu is None:
            raise ValueError("explicit RPT_BASE pair path requires uu")
        if not isinstance(uu, torch.Tensor) or uu.ndim != 4:
            raise ValueError("uu must have shape [batch,4,query,context]")
        batch, channels, query, context = map(int, uu.shape)
        if channels != STANDARD_FOUR_CHANNELS or query != context:
            raise ValueError(
                "uu must have shape [batch,4,N,N], got " f"{tuple(uu.shape)}"
            )
        if v is not None:
            if (
                not isinstance(v, torch.Tensor)
                or tuple(v.shape) != (batch, 4, query)
            ):
                raise ValueError("v and uu batch/sequence dimensions disagree")
            if v.dtype != uu.dtype or v.device != uu.device:
                raise ValueError("v and uu dtype/device must match")
        if not bool(torch.isfinite(uu).all()):
            raise FloatingPointError("uu contains NaN or infinity")
        if mask is not None:
            if tuple(mask.shape) != (batch, 1, query):
                raise ValueError("mask and uu batch/sequence dimensions disagree")
            mask = mask.bool()
        return batch, query, mask

    def _forward_dense(self, uu: Any, *, batch: int, length: int) -> Any:
        torch = require_torch()
        i, j = torch.tril_indices(
            length,
            length,
            offset=-1 if self.remove_self_pair else 0,
            device=uu.device,
        )
        elements = self.embed(uu[:, :, i, j])
        output = elements.new_zeros(batch, self.out_dim, length, length)
        output[:, :, i, j] = elements
        output[:, :, j, i] = elements
        return output

    def _forward_sparse(
        self,
        uu: Any,
        *,
        batch: int,
        length: int,
        mask: Any,
    ) -> Any:
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        offset = -1 if self.remove_self_pair else 0
        i0, _, i2, i3 = pair_mask.float().tril(offset).nonzero(as_tuple=True)
        if int(i0.numel()) == 0:
            raise ValueError(
                "canonical Particle Transformer inputs must force one valid "
                "particle in every all-empty row"
            )
        gathered = uu.permute(0, 2, 3, 1)[i0, i2, i3, :]
        elements = self.embed(gathered.T.unsqueeze(0)).squeeze(0).T
        output = elements.new_zeros(batch, length, length, self.out_dim)
        output[i0, i2, i3, :] = elements
        output[i0, i3, i2, :] = elements
        return output.permute(0, 3, 1, 2).contiguous()

    def forward(self, v: Any, uu: Any = None, mask: Any | None = None) -> Any:
        batch, length, mask = self._validate(v, uu, mask)
        sparse = self.sparse_eval[0 if self.training else 1]
        if sparse and mask is not None:
            return self._forward_sparse(
                uu, batch=batch, length=length, mask=mask
            )
        return self._forward_dense(uu, batch=batch, length=length)


class RelationalParticleTransformer(_ModuleBase):
    """Campaign-owned exact RPT_BASE using explicit standard-four ``uu``."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        weaver_module: Any | None = None,
        allow_registered_capacity_config: bool = False,
    ) -> None:
        torch = require_torch()
        super().__init__()
        resolved = exact_rpt_base_config() if config is None else dict(config)
        if resolved != RPT_BASE_CONFIG and not bool(allow_registered_capacity_config):
            raise ValueError("Step-2 RPT_BASE config must equal the locked base config")
        module = _import_weaver_module() if weaver_module is None else weaver_module
        transformer = getattr(module, "ParticleTransformer", None)
        if transformer is None:
            raise RuntimeError("Weaver module lacks ParticleTransformer")

        self.config = copy.deepcopy(resolved)
        self.mod = transformer(**resolved)
        effective_defaults = {
            "pair_extra_dim": int(getattr(self.mod, "pair_extra_dim", -1)),
            "remove_self_pair": bool(
                getattr(getattr(self.mod, "pair_embed", None), "remove_self_pair", True)
            ),
            "use_amp": bool(getattr(self.mod, "use_amp", True)),
        }
        if effective_defaults != RPT_BASE_EFFECTIVE_WEAVER_DEFAULTS:
            raise RuntimeError(
                "installed Weaver defaults drifted from RPT_BASE: "
                f"{effective_defaults}"
            )
        before = _state_structure(self.mod)
        original_pair_embed = getattr(self.mod, "pair_embed", None)
        self.mod.pair_embed = ExplicitStandardFourPairEmbed(original_pair_embed)
        after = _state_structure(self.mod)
        if before != after:
            raise RuntimeError("explicit RPT_BASE changed model state structure")
        object.__setattr__(self, "_weaver_module", module)

        forward_parameters = inspect.signature(self.mod.forward).parameters
        if "uu" not in forward_parameters:
            raise RuntimeError("installed Weaver ParticleTransformer lacks uu")

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}

    @staticmethod
    def _validate_batch(
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
    ) -> None:
        torch = require_torch()
        for name, value in (
            ("features", features),
            ("lorentz_vectors", lorentz_vectors),
            ("mask", mask),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if features.ndim != 3 or int(features.shape[1]) != 17:
            raise ValueError("features must have shape [batch,17,constituents]")
        expected_vector = (
            int(features.shape[0]),
            4,
            int(features.shape[2]),
        )
        if tuple(lorentz_vectors.shape) != expected_vector:
            raise ValueError(
                f"lorentz_vectors must have shape {expected_vector}, got "
                f"{tuple(lorentz_vectors.shape)}"
            )
        expected_mask = (
            int(features.shape[0]),
            1,
            int(features.shape[2]),
        )
        if tuple(mask.shape) != expected_mask:
            raise ValueError(
                f"mask must have shape {expected_mask}, got {tuple(mask.shape)}"
            )
        if points is not None and (
            not isinstance(points, torch.Tensor)
            or tuple(points.shape)
            != (int(features.shape[0]), 2, int(features.shape[2]))
        ):
            raise ValueError("points must have shape [batch,2,constituents]")
        if features.dtype != lorentz_vectors.dtype:
            raise ValueError("features and lorentz_vectors dtype must match")
        if features.device != lorentz_vectors.device or mask.device != features.device:
            raise ValueError("features, vectors, and mask must share one device")
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError("features contain NaN or infinity")
        if not bool(torch.isfinite(lorentz_vectors).all()):
            raise FloatingPointError("lorentz_vectors contain NaN or infinity")
        canonical_mask = mask.bool()
        if bool((canonical_mask.sum(dim=-1) == 0).any()):
            raise ValueError(
                "all-empty rows must pass through canonical Particle Transformer "
                "input construction before RPT_BASE"
            )

    def explicit_standard_four(self, lorentz_vectors: Any, mask: Any) -> Any:
        return build_standard_four_pair_features(
            lorentz_vectors,
            mask=mask,
            module=self._weaver_module,
        )

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
    ) -> Any:
        self._validate_batch(points, features, lorentz_vectors, mask)
        uu = self.explicit_standard_four(lorentz_vectors, mask)
        return self.mod(features, v=lorentz_vectors, mask=mask, uu=uu)

    def diagnostics(
        self,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
    ) -> dict[str, Any]:
        from .attention import (
            attention_allocation_diagnostics,
            capture_multihead_attention_weights,
        )

        valid = mask.bool()
        uu = self.explicit_standard_four(lorentz_vectors, valid)
        with require_torch().no_grad():
            captured = capture_multihead_attention_weights(
                self.mod,
                lambda: self.mod(
                    features, v=lorentz_vectors, mask=valid, uu=uu
                ),
            )
        return {
            "families": [],
            "architecture": "shared_directional_pair_bias",
            "attention_allocation": attention_allocation_diagnostics(
                captured, lorentz_vectors, valid
            ),
        }


class WideBaseParticleTransformer(_ModuleBase):
    """Base4-only active pair-stem capacity control."""

    def __init__(
        self,
        *,
        capacity_artifact: Mapping[str, Any] | None = None,
        weaver_module: Any | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        artifact = (
            select_wide_widths()
            if capacity_artifact is None
            else dict(capacity_artifact)
        )
        validate_content_hash(
            artifact, expected_contract=WIDE_CAPACITY_CONTRACT
        )
        if artifact != select_wide_widths():
            raise ValueError("wide capacity artifact differs from the locked search")
        self.run_id = "RPT_BASE_WIDE_MAX"
        widths = tuple(int(value) for value in artifact["selected_widths"])
        config = exact_rpt_base_config()
        config["pair_embed_dims"] = list(widths)
        module = _import_weaver_module() if weaver_module is None else weaver_module
        transformer = getattr(module, "ParticleTransformer", None)
        if transformer is None:
            raise RuntimeError("Weaver module lacks ParticleTransformer")
        self.capacity_artifact = artifact
        self.config = copy.deepcopy(config)
        self.mod = transformer(**config)
        self.mod.pair_embed = ExplicitStandardFourPairEmbed(self.mod.pair_embed)
        observed_pair_parameters = sum(
            parameter.numel() for parameter in self.mod.pair_embed.parameters()
        )
        expected_pair_parameters = pair_encoder_parameter_count(4, widths)
        if observed_pair_parameters != expected_pair_parameters:
            raise RuntimeError(
                "instantiated Weaver pair-stem count differs from the locked "
                f"formula: {observed_pair_parameters} != {expected_pair_parameters}"
            )
        self.verified_pair_encoder_parameters = observed_pair_parameters
        object.__setattr__(self, "_weaver_module", module)

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
    ) -> Any:
        RelationalParticleTransformer._validate_batch(
            points, features, lorentz_vectors, mask
        )
        uu = build_standard_four_pair_features(
            lorentz_vectors, mask=mask, module=self._weaver_module
        )
        return self.mod(features, v=lorentz_vectors, mask=mask, uu=uu)

    diagnostics = RelationalParticleTransformer.diagnostics

    def explicit_standard_four(self, lorentz_vectors: Any, mask: Any) -> Any:
        return build_standard_four_pair_features(
            lorentz_vectors, mask=mask, module=self._weaver_module
        )


class RelationalFamilyParticleTransformer(_ModuleBase):
    """Shared-pair-stem ParT for the Step-3 scientific relation families."""

    def __init__(
        self,
        families: tuple[str, ...] | list[str],
        *,
        normalization_artifact: Mapping[str, Any],
        region_normalization_artifact: Mapping[str, Any] | None = None,
        force_zero_relations: bool = False,
        weaver_module: Any | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.run_id: str | None = None
        self.force_zero_relations = bool(force_zero_relations)
        self.families = canonical_supported_families(families)
        self.normalization_sha256 = validate_relation_normalization_artifact(
            normalization_artifact
        )
        module = _import_weaver_module() if weaver_module is None else weaver_module
        transformer = getattr(module, "ParticleTransformer", None)
        if transformer is None:
            raise RuntimeError("Weaver module lacks ParticleTransformer")

        combined_dimension = STANDARD_FOUR_CHANNELS + sum(
            SUPPORTED_FAMILY_DIMENSIONS[family] for family in self.families
        )
        config = exact_rpt_base_config()
        config["pair_input_dim"] = 0
        config["pair_extra_dim"] = combined_dimension
        self.config = copy.deepcopy(config)
        self.mod = transformer(**config)
        if bool(getattr(self.mod, "use_amp", True)):
            raise RuntimeError("Step-3 Weaver model must keep internal AMP disabled")
        if int(getattr(self.mod, "pair_extra_dim", -1)) != combined_dimension:
            raise RuntimeError("Step-3 Weaver pair-extra dimension drifted")
        self.mod.pair_embed = SharedDirectionalPairEmbed(
            self.mod.pair_embed,
            input_dimension=combined_dimension,
        )
        self.pair_builder = RelationalPairBuilder(
            self.families,
            normalization_artifact=normalization_artifact,
            weaver_module=module,
            region_normalization_artifact=region_normalization_artifact,
        )
        object.__setattr__(self, "_weaver_module", module)
        forward_parameters = inspect.signature(self.mod.forward).parameters
        if "uu" not in forward_parameters:
            raise RuntimeError("installed Weaver ParticleTransformer lacks uu")

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}

    def pair_features(
        self,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
        *,
        return_details: bool = False,
    ) -> Any:
        result = self.pair_builder(
            features,
            lorentz_vectors,
            mask,
            raw_tokens,
            region_trees,
            return_details=return_details,
        )
        if not self.force_zero_relations:
            return result
        if return_details:
            result = dict(result)
            combined = result["combined"].clone()
            combined[:, STANDARD_FOUR_CHANNELS:] = 0
            result["combined"] = combined
            result["relations_forced_zero"] = True
            return result
        output = result.clone()
        output[:, STANDARD_FOUR_CHANNELS:] = 0
        return output

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
        pair_transform: Any | None = None,
    ) -> Any:
        torch = require_torch()
        RelationalParticleTransformer._validate_batch(
            points, features, lorentz_vectors, mask
        )
        valid = mask.bool()
        clean_features = features.masked_fill(~valid, 0.0)
        clean_vectors = lorentz_vectors.masked_fill(~valid, 0.0)
        clean_raw = raw_tokens
        if raw_tokens is not None:
            if (
                not isinstance(raw_tokens, torch.Tensor)
                or raw_tokens.ndim != 3
                or tuple(raw_tokens.shape[:2])
                != (int(features.shape[0]), int(features.shape[2]))
                or int(raw_tokens.shape[2]) != 14
            ):
                raise ValueError("raw HLT tokens must have shape [batch,particles,14]")
            clean_raw = raw_tokens.masked_fill(
                ~valid[:, 0].unsqueeze(-1), 0.0
            )
        uu = self.pair_features(
            clean_features, clean_vectors, valid, clean_raw, region_trees
        )
        if pair_transform is not None:
            if self.training:
                raise RuntimeError(
                    "semantic pair perturbations are inference-only"
                )
            uu = pair_transform(
                uu,
                mask=valid,
                features=clean_features,
                lorentz_vectors=clean_vectors,
                raw_tokens=clean_raw,
                region_trees=region_trees,
            )
            if tuple(uu.shape) != (
                int(features.shape[0]),
                self.pair_builder.output_dimension,
                int(features.shape[2]),
                int(features.shape[2]),
            ):
                raise ValueError("semantic pair transform changed tensor shape")
        return self.mod(
            clean_features,
            v=clean_vectors,
            mask=valid,
            uu=uu,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_contract": (
                STEP4_RELATIONAL_MODEL_CONTRACT
                if any(family in self.families for family in ("TRACK", "DENSITY"))
                else STEP3_RELATIONAL_MODEL_CONTRACT
            ),
            "run_id": self.run_id,
            "families": list(self.families),
            "run_initialization": "from_scratch",
            "pair_builder": self.pair_builder.metadata(),
            "particle_transformer_config": copy.deepcopy(self.config),
            "internal_amp": False,
            "hlt_only": True,
            "relation_input_mode": (
                "forced_zero" if self.force_zero_relations else "active"
            ),
        }

    def diagnostics(
        self,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
        labels: Any | None = None,
    ) -> dict[str, Any]:
        """Return the prespecified family and head-wise pair-bias diagnostics."""

        torch = require_torch()
        if self.training:
            raise RuntimeError("relation diagnostics require eval mode")
        valid = mask.bool()
        clean_features = features.masked_fill(~valid, 0.0)
        clean_vectors = lorentz_vectors.masked_fill(~valid, 0.0)
        clean_raw = raw_tokens
        if raw_tokens is not None:
            clean_raw = raw_tokens.masked_fill(
                ~valid[:, 0].unsqueeze(-1), 0.0
            )
        with torch.no_grad():
            details = self.pair_builder(
                clean_features,
                clean_vectors,
                valid,
                clean_raw,
                region_trees,
                return_details=True,
            )
            bias = self.mod.pair_embed(
                clean_vectors,
                uu=details["combined"],
                mask=valid,
            )
            from .attention import (
                attention_allocation_diagnostics,
                capture_multihead_attention_weights,
            )
            diagnostic_logits: list[Any] = []

            def diagnostic_forward() -> Any:
                value = self.mod(
                    clean_features,
                    v=clean_vectors,
                    mask=valid,
                    uu=details["combined"],
                )
                diagnostic_logits.append(value)
                return value

            captured_attention = capture_multihead_attention_weights(
                self.mod,
                diagnostic_forward,
            )
            logits = diagnostic_logits[0]
        pair_mask = valid_pair_mask(valid)

        def head_means(group: Any) -> list[float | None]:
            selected = pair_mask & group.bool()
            output: list[float | None] = []
            for head in range(int(bias.shape[1])):
                values = bias[:, head].masked_select(selected[:, 0])
                output.append(
                    None
                    if int(values.numel()) == 0
                    else float(values.mean().detach().cpu())
                )
            return output

        def head_mean_mapping(
            groups: Mapping[str, Any],
        ) -> dict[str, Any]:
            values: dict[str, list[float | None]] = {}
            statistics: dict[str, Any] = {}
            for name, group in groups.items():
                selected = pair_mask & group.bool()
                count = int(selected[:, 0].sum().cpu())
                sums = [
                    float(
                        bias[:, head]
                        .masked_select(selected[:, 0])
                        .sum()
                        .detach()
                        .cpu()
                    )
                    for head in range(int(bias.shape[1]))
                ]
                values[str(name)] = [
                    None if count == 0 else value / count
                    for value in sums
                ]
                statistics[str(name)] = {
                    "kind": "ratio",
                    "numerator": sums,
                    "denominator": [count] * len(sums),
                }
            return {
                **values,
                "_population_statistics": statistics,
            }

        def histogram(
            values: Any,
            selected: Any,
            edges: tuple[float, ...],
        ) -> dict[str, Any]:
            finite = values.masked_select(selected & torch.isfinite(values))
            counts = []
            for index in range(len(edges) - 1):
                left, right = edges[index], edges[index + 1]
                inside = (
                    finite.ge(left) & finite.le(right)
                    if index == 0
                    else finite.gt(left) & finite.le(right)
                )
                counts.append(int(inside.sum().cpu()))
            serialized_edges = [
                (
                    "-inf"
                    if value == -float("inf")
                    else "+inf"
                    if value == float("inf")
                    else value
                )
                for value in edges
            ]
            return {
                "bin_edges": serialized_edges,
                "endpoint_policy": (
                    "first bin [left,right], later bins (left,right]"
                ),
                "bin_counts": counts,
                "finite_entry_count": int(finite.numel()),
                "_population_statistics": {
                    "bin_counts": {"kind": "sum", "value": counts},
                    "finite_entry_count": {
                        "kind": "sum",
                        "value": int(finite.numel()),
                    },
                },
            }

        def binned_pair_bias(
            measure: Any,
            edges: tuple[float, ...],
            *,
            applicable: Any | None = None,
        ) -> dict[str, Any]:
            selected_domain = pair_mask[:, 0]
            if applicable is not None:
                selected_domain = selected_domain & applicable.bool()
            counts: list[int] = []
            sums: list[list[float]] = []
            means: list[list[float | None]] = []
            denominators: list[list[int]] = []
            for index in range(len(edges) - 1):
                left, right = edges[index], edges[index + 1]
                selected = selected_domain & (
                    measure.ge(left) & measure.le(right)
                    if index == 0
                    else measure.gt(left) & measure.le(right)
                )
                count = int(selected.sum().cpu())
                counts.append(count)
                head_sums = [
                    float(
                        bias[:, head].masked_select(selected).sum().cpu()
                    )
                    for head in range(int(bias.shape[1]))
                ]
                sums.append(head_sums)
                denominators.append([count] * int(bias.shape[1]))
                means.append(
                    [
                        None if count == 0 else value / count
                        for value in head_sums
                    ]
                )
            serialized_edges = [
                (
                    "-inf"
                    if value == -float("inf")
                    else "+inf"
                    if value == float("inf")
                    else value
                )
                for value in edges
            ]
            return {
                "bin_edges": serialized_edges,
                "endpoint_policy": (
                    "first bin [left,right], later bins (left,right]"
                ),
                "pair_counts": counts,
                "bias_sums_by_head": sums,
                "mean_bias_by_head": means,
                "_population_statistics": {
                    "pair_counts": {"kind": "sum", "value": counts},
                    "bias_sums_by_head": {"kind": "sum", "value": sums},
                    "mean_bias_by_head": {
                        "kind": "ratio",
                        "numerator": sums,
                        "denominator": denominators,
                    },
                },
            }

        predictions = logits.argmax(dim=1)

        def binned_performance(
            measure: Any,
            edges: tuple[float, ...],
        ) -> dict[str, Any] | None:
            if labels is None:
                return None
            truth = labels.long()
            counts: list[int] = []
            correct: list[int] = []
            accuracy: list[float | None] = []
            for index in range(len(edges) - 1):
                left, right = edges[index], edges[index + 1]
                selected = (
                    measure.ge(left) & measure.le(right)
                    if index == 0
                    else measure.gt(left) & measure.le(right)
                )
                count = int(selected.sum().cpu())
                matched = int(
                    (predictions.eq(truth) & selected).sum().cpu()
                )
                counts.append(count)
                correct.append(matched)
                accuracy.append(None if count == 0 else matched / count)
            serialized_edges = [
                (
                    "-inf"
                    if value == -float("inf")
                    else "+inf"
                    if value == float("inf")
                    else value
                )
                for value in edges
            ]
            return {
                "bin_edges": serialized_edges,
                "endpoint_policy": (
                    "first bin [left,right], later bins (left,right]"
                ),
                "event_counts": counts,
                "correct_counts": correct,
                "accuracy": accuracy,
                "_population_statistics": {
                    "event_counts": {"kind": "sum", "value": counts},
                    "correct_counts": {"kind": "sum", "value": correct},
                    "accuracy": {
                        "kind": "ratio",
                        "numerator": correct,
                        "denominator": counts,
                    },
                },
            }

        def categorical_performance(
            categories: Any,
            names: tuple[str, ...],
        ) -> dict[str, Any] | None:
            if labels is None:
                return None
            truth = labels.long()
            counts = [
                int(categories.eq(index).sum().cpu())
                for index in range(len(names))
            ]
            correct = [
                int(
                    (
                        categories.eq(index)
                        & predictions.eq(truth)
                    ).sum().cpu()
                )
                for index in range(len(names))
            ]
            accuracy = [
                None if count == 0 else matched / count
                for count, matched in zip(counts, correct)
            ]
            return {
                "category_order": list(names),
                "event_counts": counts,
                "correct_counts": correct,
                "accuracy": accuracy,
                "_population_statistics": {
                    "event_counts": {"kind": "sum", "value": counts},
                    "correct_counts": {"kind": "sum", "value": correct},
                    "accuracy": {
                        "kind": "ratio",
                        "numerator": correct,
                        "denominator": counts,
                    },
                },
            }

        output: dict[str, Any] = {
            "families": list(self.families),
            "pair_bias_shape": list(bias.shape),
            "valid_directed_pair_count": int(pair_mask.sum().cpu()),
            "pair_bias_finite": bool(torch.isfinite(bias).all()),
            "attention_allocation": attention_allocation_diagnostics(
                captured_attention, clean_vectors, valid
            ),
        }
        if "PT" in self.families:
            pt_encoder = self.pair_builder.encoders["PT"]
            pt_details = pt_encoder(
                clean_vectors, valid, return_details=True
            )
            rank = average_tied_descending_rank(
                torch.hypot(clean_vectors[:, 0], clean_vectors[:, 1]),
                valid,
            )
            rank_bin = torch.clamp((rank * 10.0).floor().long(), 0, 9)
            rank_means = head_mean_mapping({
                str(index): (
                    (rank_bin == index).unsqueeze(1).unsqueeze(-2)
                )
                for index in range(10)
            })
            leading = torch.zeros_like(rank, dtype=torch.bool)
            subleading = torch.zeros_like(rank, dtype=torch.bool)
            soft = torch.zeros_like(rank, dtype=torch.bool)
            for row in range(int(rank.shape[0])):
                values = torch.unique(rank[row][valid[row, 0]], sorted=True)
                if int(values.numel()) >= 1:
                    leading[row] = valid[row, 0] & (rank[row] == values[0])
                if int(values.numel()) >= 2:
                    subleading[row] = valid[row, 0] & (rank[row] == values[1])
                soft[row] = valid[row, 0] & ~(
                    leading[row] | subleading[row]
                )
            difference = bias - bias.transpose(-1, -2)
            swap = []
            for head in range(int(bias.shape[1])):
                values = difference[:, head].masked_select(pair_mask[:, 0])
                value_count = int(values.numel())
                absolute_sum = float(values.abs().sum().cpu())
                square_sum = float(values.square().sum().cpu())
                swap.append(
                    {
                        "mean_absolute": absolute_sum / value_count,
                        "rms": math.sqrt(square_sum / value_count),
                        "_population_statistics": {
                            "mean_absolute": {
                                "kind": "ratio",
                                "numerator": absolute_sum,
                                "denominator": value_count,
                            },
                            "rms": {
                                "kind": "root_mean_square",
                                "square_sum": square_sum,
                                "denominator": value_count,
                            },
                        },
                    }
                )
            output["PT"] = {
                **pt_encoder.diagnostics(clean_vectors, valid),
                "context_rank_decile_rule": (
                    "floor(10*r), clipped to 0..9; r=1 belongs to bin 9"
                ),
                "mean_bias_by_context_rank_decile": rank_means,
                "context_groups": (
                    "leading=lowest_tied_rank; subleading=next_distinct_tied_rank; "
                    "soft=remaining_valid"
                ),
                "headwise_bias": head_mean_mapping({
                    "leading_context": (
                        leading.unsqueeze(1).unsqueeze(-2)
                    ),
                    "subleading_context": (
                        subleading.unsqueeze(1).unsqueeze(-2)
                    ),
                    "soft_context": (
                        soft.unsqueeze(1).unsqueeze(-2)
                    ),
                }),
                "directional_swap": swap,
                "normalized_finite": bool(
                    torch.isfinite(pt_details["normalized"]).all()
                ),
            }
        if "PID" in self.families:
            pid_encoder = self.pair_builder.encoders["PID"]
            pid_details = pid_encoder(
                clean_features[:, 6:11], valid, return_details=True
            )
            pair_index = pid_details["pair_indices"]
            output["PID"] = {
                **pid_encoder.diagnostics(clean_features[:, 6:11], valid),
                "headwise_mean_bias_by_directed_pair": head_mean_mapping({
                    str(index): (
                        (pair_index == index).unsqueeze(1)
                    )
                    for index in range(36)
                }),
                "headwise_pid_pair_type_bias": head_mean_mapping({
                    "same_pid": (
                        pid_details["categories"].unsqueeze(-1)
                        == pid_details["categories"].unsqueeze(-2)
                    ).unsqueeze(1),
                    "mixed_pid": (
                        pid_details["categories"].unsqueeze(-1)
                        != pid_details["categories"].unsqueeze(-2)
                    ).unsqueeze(1),
                }),
            }
        if "CHARGE" in self.families:
            charge_encoder = self.pair_builder.encoders["CHARGE"]
            charge_details = charge_encoder(
                clean_features[:, 5], valid, return_details=True
            )
            raw = charge_details["raw"]
            diagnostic_pid = pid_categories(
                clean_features[:, 6:11],
                valid,
                fail_on_multi_hot=False,
            )
            pid_population = torch.stack(
                [
                    diagnostic_pid.eq(index).logical_and(valid[:, 0]).sum(1)
                    for index in range(6)
                ],
                dim=1,
            )
            dominant_pid = pid_population.argmax(dim=1)
            output["CHARGE"] = {
                **charge_encoder.diagnostics(clean_features[:, 5], valid),
                "headwise_bias": head_mean_mapping({
                    "opposite_sign": raw[:, 7:8].bool(),
                    "same_nonzero_sign": raw[:, 6:7].bool(),
                    "charged_neutral": raw[:, 5:6].bool(),
                    "neutral_neutral": raw[:, 4:5].bool(),
                }),
                "pid_conditioned_performance": categorical_performance(
                    dominant_pid,
                    (
                        "dominant_charged_hadron",
                        "dominant_neutral_hadron",
                        "dominant_photon",
                        "dominant_electron",
                        "dominant_muon",
                        "dominant_unknown",
                    ),
                ),
                "pid_conditioning_definition": (
                    "jet stratum is the most frequent canonical HLT PID "
                    "category; ties resolve by canonical category order"
                ),
            }
        if "TRACK" in self.families:
            if clean_raw is None:
                raise ValueError("TRACK diagnostics require raw HLT tokens")
            track_encoder = self.pair_builder.encoders["TRACK"]
            track_details = track_encoder(
                clean_raw, valid, return_details=True
            )
            raw_displacement = torch.maximum(
                track_details["raw_d0_significance"].abs(),
                track_details["raw_dz_significance"].abs(),
            )
            track_valid = track_details["track_valid"]
            endpoint_min = torch.minimum(
                raw_displacement.unsqueeze(-1),
                raw_displacement.unsqueeze(-2),
            )
            endpoint_displaced = track_valid & (raw_displacement >= 2.0)
            endpoint_prompt = track_valid & ~endpoint_displaced
            both_tracks_valid = (
                track_valid.unsqueeze(-1) & track_valid.unsqueeze(-2)
            )
            track_class_performance = None
            if labels is not None:
                required_indices = (1, 2, 8, 9)
                required_names = ("Hbb", "Hcc", "Tbqq", "Tbl")
                class_counts = [
                    int(labels.eq(index).sum().cpu())
                    for index in required_indices
                ]
                class_correct = [
                    int(
                        (
                            labels.eq(index) & predictions.eq(labels)
                        ).sum().cpu()
                    )
                    for index in required_indices
                ]
                track_class_performance = {
                    "class_order": list(required_names),
                    "event_counts": class_counts,
                    "correct_counts": class_correct,
                    "accuracy": [
                        None if count == 0 else correct / count
                        for count, correct in zip(
                            class_counts, class_correct
                        )
                    ],
                    "_population_statistics": {
                        "event_counts": {
                            "kind": "sum",
                            "value": class_counts,
                        },
                        "correct_counts": {
                            "kind": "sum",
                            "value": class_correct,
                        },
                        "accuracy": {
                            "kind": "ratio",
                            "numerator": class_correct,
                            "denominator": class_counts,
                        },
                    },
                }
            output["TRACK"] = {
                **track_encoder.diagnostics(clean_raw, valid),
                "displaced_threshold_raw_absolute_significance": 2.0,
                "headwise_bias": head_mean_mapping({
                    "prompt_prompt": (
                            endpoint_prompt.unsqueeze(-1)
                            & endpoint_prompt.unsqueeze(-2)
                        ).unsqueeze(1),
                    "prompt_displaced": (
                            endpoint_displaced.unsqueeze(-1)
                            ^ endpoint_displaced.unsqueeze(-2)
                        ).unsqueeze(1),
                    "displaced_displaced": (
                            endpoint_displaced.unsqueeze(-1)
                            & endpoint_displaced.unsqueeze(-2)
                        ).unsqueeze(1),
                }),
                "minimum_absolute_displacement_significance_mean": (
                    float(
                        endpoint_min.masked_select(
                            both_tracks_valid
                        ).mean().cpu()
                    )
                    if bool(both_tracks_valid.any())
                    else None
                ),
                "raw_displacement_distributions": {
                    "d0": histogram(
                        track_details["d0"],
                        track_valid,
                        DIAGNOSTIC_BIN_EDGES["track_raw_displacement"],
                    ),
                    "dz": histogram(
                        track_details["dz"],
                        track_valid,
                        DIAGNOSTIC_BIN_EDGES["track_raw_displacement"],
                    ),
                },
                "absolute_significance_distributions": {
                    "d0": histogram(
                        track_details["raw_d0_significance"].abs(),
                        track_valid,
                        DIAGNOSTIC_BIN_EDGES["track_absolute_significance"],
                    ),
                    "dz": histogram(
                        track_details["raw_dz_significance"].abs(),
                        track_valid,
                        DIAGNOSTIC_BIN_EDGES["track_absolute_significance"],
                    ),
                },
                "asinh_absolute_significance_distributions": {
                    "d0": histogram(
                        track_details["continuous"][:, 4].abs(),
                        track_valid,
                        DIAGNOSTIC_BIN_EDGES[
                            "track_asinh_absolute_significance"
                        ],
                    ),
                    "dz": histogram(
                        track_details["continuous"][:, 5].abs(),
                        track_valid,
                        DIAGNOSTIC_BIN_EDGES[
                            "track_asinh_absolute_significance"
                        ],
                    ),
                },
                "compatibility_chi2_distribution": histogram(
                    track_details["chi2"],
                    both_tracks_valid,
                    DIAGNOSTIC_BIN_EDGES["track_compatibility_chi2"],
                ),
                "bias_by_minimum_absolute_displacement_significance": (
                    binned_pair_bias(
                        endpoint_min,
                        DIAGNOSTIC_BIN_EDGES["track_absolute_significance"],
                        applicable=both_tracks_valid,
                    )
                ),
                "required_class_performance": track_class_performance,
            }
        if "DENSITY" in self.families:
            if clean_raw is None:
                raise ValueError("DENSITY diagnostics require raw HLT tokens")
            density_encoder = self.pair_builder.encoders["DENSITY"]
            density_details = density_encoder(
                clean_raw, valid, return_details=True
            )
            density_audit = density_encoder.diagnostics(clean_raw, valid)
            local_activity = density_details["descriptor"][:, 20]
            particle_valid = valid[:, 0]
            multiplicity = particle_valid.sum(1).to(clean_vectors.dtype)
            particle_pt = torch.hypot(
                clean_vectors[:, 0], clean_vectors[:, 1]
            ).masked_fill(~particle_valid, 0.0)
            leading_fraction = particle_pt.amax(1) / particle_pt.sum(
                1
            ).clamp_min(1.0e-30)
            context_activity = local_activity.unsqueeze(-2).expand(
                -1, int(local_activity.shape[1]), -1
            )
            local_activity_sum = float(
                local_activity.masked_select(particle_valid).sum().cpu()
            )
            valid_particle_count = int(particle_valid.sum().cpu())
            output["DENSITY"] = {
                **density_audit,
                "local_activity_definition": (
                    "valid_neighbor_fraction_R0p40"
                ),
                "mean_local_activity": float(
                    local_activity.masked_select(valid[:, 0]).mean().cpu()
                ),
                "annulus_occupancy_count_distributions": {
                    str(index): histogram(
                        density_details["annulus_masks"][:, index]
                        .sum(dim=-1)
                        .to(clean_vectors.dtype),
                        particle_valid,
                        DIAGNOSTIC_BIN_EDGES[
                            "density_annulus_neighbor_count"
                        ],
                    )
                    for index in range(4)
                },
                "bias_by_context_local_activity_fraction": binned_pair_bias(
                    context_activity,
                    DIAGNOSTIC_BIN_EDGES["density_local_activity"],
                ),
                "performance_by_jet_multiplicity": binned_performance(
                    multiplicity,
                    DIAGNOSTIC_BIN_EDGES["jet_multiplicity"],
                ),
                "performance_by_leading_particle_pt_fraction": (
                    binned_performance(
                        leading_fraction,
                        DIAGNOSTIC_BIN_EDGES[
                            "leading_particle_pt_fraction"
                        ],
                    )
                ),
                "_population_statistics": {
                    **density_audit["_population_statistics"],
                    "mean_local_activity": {
                        "kind": "ratio",
                        "numerator": local_activity_sum,
                        "denominator": valid_particle_count,
                    },
                },
            }
        if "REGION" in self.families:
            if clean_raw is None or region_trees is None:
                raise ValueError("REGION diagnostics require raw tokens and trees")
            region_encoder = self.pair_builder.encoders["REGION"]
            region_details = region_encoder(
                clean_raw, valid, region_trees, return_details=True
            )
            region_raw = region_details["raw"]
            depth_per_event = torch.as_tensor(
                [
                    (
                        int(max(tree["depth"][:tree["n_valid"]]))
                        if int(tree["n_valid"]) else 0
                    )
                    for tree in region_trees
                ],
                device=clean_vectors.device,
                dtype=clean_vectors.dtype,
            )
            hard_prongs = torch.as_tensor(
                [
                    int(tree["actual_cluster_counts"]["8"])
                    for tree in region_trees
                ],
                device=clean_vectors.device,
                dtype=clean_vectors.dtype,
            )
            actual_counts = {
                str(k): [
                    int(tree["actual_cluster_counts"][str(k)])
                    for tree in region_trees
                ]
                for k in (2, 4, 8)
            }
            valid_pair_count = int(pair_mask[:, 0].sum().cpu())
            same_cluster_sums = {
                str(k): float(
                    region_raw[:, index]
                    .masked_select(pair_mask[:, 0])
                    .sum()
                    .cpu()
                )
                for index, k in enumerate((2, 4, 8))
            }
            same_cluster_bias_sums = {}
            same_cluster_bias_denominators = {}
            for index, k in enumerate((2, 4, 8)):
                selected = pair_mask[:, 0] & region_raw[:, index].bool()
                selected_count = int(selected.sum().cpu())
                same_cluster_bias_sums[str(k)] = [
                    float(
                        bias[:, head].masked_select(selected).sum().cpu()
                    )
                    for head in range(int(bias.shape[1]))
                ]
                same_cluster_bias_denominators[str(k)] = [
                    selected_count
                ] * int(bias.shape[1])
            lca_depth_sum = float(
                region_raw[:, 3]
                .masked_select(pair_mask[:, 0])
                .sum()
                .cpu()
            )
            off_diagonal = ~torch.eye(
                int(region_raw.shape[-1]),
                dtype=torch.bool,
                device=region_raw.device,
            ).unsqueeze(0)
            merge_pair_mask = pair_mask[:, 0] & off_diagonal
            output["REGION"] = {
                "normalization_sha256": region_encoder.normalization_sha256,
                "requested_cluster_counts": [2, 4, 8],
                "actual_cluster_counts": {
                    **actual_counts,
                    "_population_statistics": {
                        str(k): {
                            "kind": "concatenate",
                            "values": actual_counts[str(k)],
                        }
                        for k in (2, 4, 8)
                    },
                },
                "node_counts": [int(tree["n_nodes"]) for tree in region_trees],
                "maximum_leaf_depths": [
                    (
                        int(max(tree["depth"][:tree["n_valid"]]))
                        if int(tree["n_valid"]) else 0
                    )
                    for tree in region_trees
                ],
                "same_cluster_fractions": {
                    **{
                        str(k): (
                            same_cluster_sums[str(k)] / valid_pair_count
                        )
                        for k in (2, 4, 8)
                    },
                    "_population_statistics": {
                        str(k): {
                            "kind": "ratio",
                            "numerator": same_cluster_sums[str(k)],
                            "denominator": valid_pair_count,
                        }
                        for k in (2, 4, 8)
                    },
                },
                "mean_normalized_lca_depth": float(
                    region_raw[:, 3]
                    .masked_select(pair_mask[:, 0])
                    .mean()
                    .cpu()
                ),
                "headwise_same_cluster_bias": {
                    **{
                        str(k): head_means(
                            region_raw[:, index:index + 1].bool()
                        )
                        for index, k in enumerate((2, 4, 8))
                    },
                    "_population_statistics": {
                        str(k): {
                            "kind": "ratio",
                            "numerator": same_cluster_bias_sums[str(k)],
                            "denominator": (
                                same_cluster_bias_denominators[str(k)]
                            ),
                        }
                        for k in (2, 4, 8)
                    },
                },
                "lca_distributions": {
                    "normalized_depth": histogram(
                        region_raw[:, 3],
                        pair_mask[:, 0],
                        DIAGNOSTIC_BIN_EDGES["region_lca_depth"],
                    ),
                    "log_merge_delta_r": histogram(
                        region_raw[:, 4],
                        merge_pair_mask,
                        DIAGNOSTIC_BIN_EDGES[
                            "region_log_merge_delta_r"
                        ],
                    ),
                    "log_merge_kt": histogram(
                        region_raw[:, 5],
                        merge_pair_mask,
                        DIAGNOSTIC_BIN_EDGES["region_log_merge_kt"],
                    ),
                    "merge_z": histogram(
                        region_raw[:, 6],
                        merge_pair_mask,
                        DIAGNOSTIC_BIN_EDGES["region_merge_z"],
                    ),
                    "log_merge_mass_fraction": histogram(
                        region_raw[:, 7],
                        merge_pair_mask,
                        DIAGNOSTIC_BIN_EDGES[
                            "region_log_merge_mass_fraction"
                        ],
                    ),
                },
                "cluster_property_distributions": {
                    str(k): {
                        "log_pt_fraction": histogram(
                            region_raw[:, 8 + index * 6],
                            pair_mask[:, 0],
                            DIAGNOSTIC_BIN_EDGES[
                                "region_log_cluster_pt_fraction"
                            ],
                        ),
                        "log_mass_fraction": histogram(
                            region_raw[:, 9 + index * 6],
                            pair_mask[:, 0],
                            DIAGNOSTIC_BIN_EDGES[
                                "region_log_cluster_mass_fraction"
                            ],
                        ),
                        "multiplicity_fraction": histogram(
                            region_raw[:, 10 + index * 6],
                            pair_mask[:, 0],
                            DIAGNOSTIC_BIN_EDGES[
                                "region_cluster_multiplicity_fraction"
                            ],
                        ),
                    }
                    for index, k in enumerate((2, 4, 8))
                },
                "performance_by_tree_depth": binned_performance(
                    depth_per_event,
                    DIAGNOSTIC_BIN_EDGES["region_tree_depth"],
                ),
                "performance_by_hard_prong_count": binned_performance(
                    hard_prongs,
                    DIAGNOSTIC_BIN_EDGES["region_hard_prong_count"],
                ),
                "_population_statistics": {
                    "mean_normalized_lca_depth": {
                        "kind": "ratio",
                        "numerator": lca_depth_sum,
                        "denominator": valid_pair_count,
                    },
                },
            }
        return output


def build_relational_particle_transformer():
    return RelationalParticleTransformer()


def build_registered_step3_model(
    run_id: str,
    *,
    normalization_artifact: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    weaver_module: Any | None = None,
) -> RelationalFamilyParticleTransformer:
    """Resolve and instantiate exactly one registered Step-3 single."""

    validate_screening_registry(screening_registry)
    validate_relation_normalization_artifact(normalization_artifact)
    if normalization_artifact.get(
        "relation_registry_sha256"
    ) != screening_registry.get("relation_registry_sha256"):
        raise ValueError(
            "relation normalizer and screening registry use different "
            "relation registries"
        )
    row = resolve_registered_run(
        run_id,
        screening_registry=screening_registry,
    )
    families = tuple(row.get("new_relation_families", ()))
    if families not in (("PT",), ("PID",), ("CHARGE",)):
        raise ValueError(
            f"{run_id} is not a Step-3 registered single-family run"
        )
    model = RelationalFamilyParticleTransformer(
        families,
        normalization_artifact=normalization_artifact,
        weaver_module=weaver_module,
    )
    model.run_id = str(run_id)
    return model


def build_registered_step4_model(
    run_id: str,
    *,
    normalization_artifact: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    weaver_module: Any | None = None,
) -> RelationalFamilyParticleTransformer:
    """Resolve and instantiate a registered TRACK or DENSITY single."""

    validate_screening_registry(screening_registry)
    validate_relation_normalization_artifact(normalization_artifact)
    if normalization_artifact.get(
        "relation_registry_sha256"
    ) != screening_registry.get("relation_registry_sha256"):
        raise ValueError(
            "relation normalizer and screening registry use different "
            "relation registries"
        )
    row = resolve_registered_run(run_id, screening_registry=screening_registry)
    families = tuple(row.get("new_relation_families", ()))
    if families not in (("TRACK",), ("DENSITY",)):
        raise ValueError(f"{run_id} is not a Step-4 registered single-family run")
    model = RelationalFamilyParticleTransformer(
        families,
        normalization_artifact=normalization_artifact,
        weaver_module=weaver_module,
    )
    model.run_id = str(run_id)
    return model


def build_registered_screening_model(
    run_id: str,
    *,
    normalization_artifact: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    region_normalization_artifact: Mapping[str, Any] | None = None,
    weaver_module: Any | None = None,
) -> RelationalFamilyParticleTransformer:
    """Instantiate any standard shared-bias screening relation row."""

    validate_screening_registry(screening_registry)
    validate_relation_normalization_artifact(normalization_artifact)
    row = resolve_registered_run(run_id, screening_registry=screening_registry)
    if row.get("attention_architecture") == "wide_pair_encoder":
        raise ValueError("RPT_BASE_WIDE_MAX requires the wide-control builder")
    families = tuple(row.get("new_relation_families", ()))
    if not families:
        raise ValueError("RPT_BASE uses the exact base-model builder")
    model = RelationalFamilyParticleTransformer(
        families,
        normalization_artifact=normalization_artifact,
        region_normalization_artifact=region_normalization_artifact,
        force_zero_relations=row.get("relation_input_mode") == "forced_zero",
        weaver_module=weaver_module,
    )
    model.run_id = str(run_id)
    return model


def build_registered_wide_model(
    run_id: str,
    *,
    screening_registry: Mapping[str, Any],
    capacity_artifact: Mapping[str, Any] | None = None,
    weaver_module: Any | None = None,
) -> WideBaseParticleTransformer:
    validate_screening_registry(screening_registry)
    row = resolve_registered_run(run_id, screening_registry=screening_registry)
    if (
        row.get("run_id") != "RPT_BASE_WIDE_MAX"
        or row.get("attention_architecture") != "wide_pair_encoder"
    ):
        raise ValueError(f"{run_id} is not the registered wide capacity control")
    return WideBaseParticleTransformer(
        capacity_artifact=capacity_artifact,
        weaver_module=weaver_module,
    )


def build_confirmation_architecture_model(
    run_id: str,
    *,
    selected_families: tuple[str, ...] | list[str] = (),
    normalization_artifact: Mapping[str, Any] | None = None,
    region_normalization_artifact: Mapping[str, Any] | None = None,
    weaver_module: Any | None = None,
):
    """Instantiate one of the four prespecified Step-6 architecture rows."""

    from .attention import ConfirmationArchitectureParticleTransformer

    offline_aliases = {
        "OFF_RPT_BASE_EDGEVALUE": "RPT_BASE_EDGEVALUE",
        "OFF_RPT_SELECTED_LAYERWISE": "RPT_SELECTED_LAYERWISE",
        "OFF_RPT_SELECTED_EDGEVALUE": "RPT_SELECTED_EDGEVALUE",
    }
    architecture_run_id = offline_aliases.get(run_id, run_id)
    allowed = {
        "RPT_BASE_LAYERWISE": (False, True),
        "RPT_BASE_EDGEVALUE": (True, True),
        "RPT_SELECTED_LAYERWISE": (False, False),
        "RPT_SELECTED_EDGEVALUE": (True, False),
    }
    if architecture_run_id not in allowed:
        raise ValueError(f"{run_id!r} is not a Step-6 architecture run")
    edge_value, base_control = allowed[architecture_run_id]
    families = (
        () if base_control else canonical_supported_families(selected_families)
    )
    if base_control and tuple(selected_families):
        raise ValueError("base4 architecture controls cannot receive relations")
    combined_dimension = STANDARD_FOUR_CHANNELS + sum(
        SUPPORTED_FAMILY_DIMENSIONS[family] for family in families
    )
    module = _import_weaver_module() if weaver_module is None else weaver_module
    transformer = getattr(module, "ParticleTransformer", None)
    if transformer is None:
        raise RuntimeError("Weaver module lacks ParticleTransformer")
    config = exact_rpt_base_config()
    config["pair_input_dim"] = 0
    config["pair_extra_dim"] = combined_dimension
    instantiated = transformer(**config)
    model = ConfirmationArchitectureParticleTransformer(
        transformer=instantiated,
        weaver_module=module,
        families=families,
        normalization_artifact=normalization_artifact,
        region_normalization_artifact=region_normalization_artifact,
        edge_value=edge_value,
    )
    model.run_id = run_id
    model.config = copy.deepcopy(config)
    return model


def build_step6_model_contract(
    run_id: str,
    *,
    selected_families: tuple[str, ...] | list[str],
    confirmation_architecture_registry: Mapping[str, Any],
    relation_normalization_artifact: Mapping[str, Any],
    selected_shared_bias_model_contract_sha256: str,
    global_determinism_sha256: str,
) -> dict[str, Any]:
    """Bind one resolved confirmation architecture before it is trained."""

    registry_sha = validate_content_hash(
        confirmation_architecture_registry,
        expected_contract=CONFIRMATION_ARCHITECTURE_REGISTRY_CONTRACT,
    )
    normalizer_sha = validate_relation_normalization_artifact(
        relation_normalization_artifact
    )
    allowed = {
        "RPT_BASE_LAYERWISE": ("layerwise_bias", False, True),
        "RPT_BASE_EDGEVALUE": (
            "layerwise_bias_and_edge_value",
            True,
            True,
        ),
        "RPT_SELECTED_LAYERWISE": ("layerwise_bias", False, False),
        "RPT_SELECTED_EDGEVALUE": (
            "layerwise_bias_and_edge_value",
            True,
            False,
        ),
    }
    if run_id not in allowed:
        raise ValueError(f"{run_id!r} is not a Step-6 architecture run")
    architecture, edge_value, base_control = allowed[run_id]
    families = (
        () if base_control else canonical_supported_families(selected_families)
    )
    if base_control and tuple(selected_families):
        raise ValueError("base4 architecture controls cannot receive relations")
    family_set_hash = canonical_sha256(list(families))
    return with_content_hash(
        {
            "contract": STEP6_RELATIONAL_MODEL_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "configuration_role": (
                "architecture_control" if base_control else "scientific_finalist"
            ),
            "confirmation_architecture_registry_sha256": registry_sha,
            "relation_normalization_sha256": normalizer_sha,
            "selected_shared_bias_model_contract_sha256": require_sha256(
                selected_shared_bias_model_contract_sha256,
                name="selected_shared_bias_model_contract_sha256",
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "selected_relation_set": list(families),
            "selected_relation_set_sha256": family_set_hash,
            "enabled_relations": ["base4", *families],
            "attention_architecture": architecture,
            "pair_stem_evaluations_per_batch": 1,
            "independent_bias_projection_per_layer": True,
            "layerwise_bias_projection_initialization": (
                "independent_parameters_initialized_as_deep_copies_of_one_"
                "Weaver_initialized_final_projection_tail"
            ),
            "layerwise_bias_split_boundary": (
                "immediately_before_final_Conv1d_or_Linear_head_projection"
            ),
            "layerwise_bias_projection_tail": (
                "final_head_projection_plus_supported_trailing_modules_"
                "including_Weaver_output_BatchNorm1d"
            ),
            "particle_attention_layer_count": 8,
            "edge_value": {
                "enabled": edge_value,
                "per_layer_per_head_linear_no_bias": edge_value,
                "aggregate_before_projection": edge_value,
                "materialize_B_H_N_N_dh": False,
                "norm_ratio_epsilon": 1.0e-6,
                "projection_initialization": (
                    "torch_nn_init_xavier_uniform_under_run_seed"
                ),
            },
            "base4_architecture_control": base_control,
            "every_registered_parameter_active": True,
            "initialization": "from_scratch",
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
        }
    )


def build_step3_model_contract(
    run_id: str,
    *,
    normalization_artifact: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    relation_registry_sha256: str,
    pair_base_sha256: str,
    family_contract: Mapping[str, Any],
    weaver_runtime_sha256: str,
    global_determinism_sha256: str,
) -> dict[str, Any]:
    screening_sha = validate_screening_registry(screening_registry)
    normalizer_sha = validate_relation_normalization_artifact(
        normalization_artifact,
        relation_registry_sha256=relation_registry_sha256,
    )
    if screening_registry.get(
        "relation_registry_sha256"
    ) != require_sha256(
        relation_registry_sha256, name="relation_registry_sha256"
    ):
        raise ValueError(
            "screening registry belongs to another relation registry"
        )
    row = resolve_registered_run(
        run_id,
        screening_registry=screening_registry,
    )
    families = tuple(row.get("new_relation_families", ()))
    if families not in (("PT",), ("PID",), ("CHARGE",)):
        raise ValueError(
            f"{run_id} is not a Step-3 registered single-family run"
        )
    expected_family_contract = (
        PT_RELATION_CONTRACT
        if families == ("PT",)
        else PID_CHARGE_RELATION_CONTRACT
    )
    family_contract_sha = validate_content_hash(
        family_contract,
        expected_contract=expected_family_contract,
    )
    if family_contract.get("relation_registry_sha256") != relation_registry_sha256:
        raise ValueError("family contract belongs to another relation registry")
    if (
        family_contract.get("relation_normalization_sha256")
        != normalizer_sha
    ):
        raise ValueError("family contract belongs to another relation normalizer")
    combined_dimension = STANDARD_FOUR_CHANNELS + sum(
        SUPPORTED_FAMILY_DIMENSIONS[family] for family in families
    )
    return with_content_hash(
        {
            "contract": STEP3_RELATIONAL_MODEL_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "configuration_role": row["configuration_role"],
            "screening_registry_sha256": screening_sha,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "relation_normalization_sha256": normalizer_sha,
            "pair_base_sha256": require_sha256(
                pair_base_sha256, name="pair_base_sha256"
            ),
            "family_contract_sha256": family_contract_sha,
            "weaver_runtime_sha256": require_sha256(
                weaver_runtime_sha256, name="weaver_runtime_sha256"
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256,
                name="global_determinism_sha256",
            ),
            "enabled_relations": ["base4", *families],
            "new_relation_families": list(families),
            "initialization": "from_scratch",
            "base_config": exact_rpt_base_config(),
            "pair_path": {
                "forward_argument": "uu",
                "canonical_concatenation_order": ["base4", *families],
                "combined_dimension": combined_dimension,
                "weaver_pair_input_dim": 0,
                "weaver_pair_extra_dim": combined_dimension,
                "shared_pair_stem": True,
                "pair_embed_dims": [64, 64, 64, 8],
                "invalid_pairs_zero_after_learned_encoders": True,
            },
            "diagnostics": {
                "PT_context_rank_bins": (
                    "floor(10*r)_clipped_0_9_with_r_equals_1_in_bin_9"
                ),
                "PT_context_groups": (
                    "lowest_tied_rank_next_distinct_tied_rank_remaining"
                ),
                "PT_directional_swap_summaries": ["mean_absolute", "rms"],
                "PID_directed_pair_states": 36,
                "CHARGE_pair_groups": [
                    "opposite_sign",
                    "same_nonzero_sign",
                    "charged_neutral",
                    "neutral_neutral",
                ],
                "empty_group_value": None,
            },
            "internal_amp": False,
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
        }
    )


def build_step4_model_contract(
    run_id: str,
    *,
    normalization_artifact: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    relation_registry_sha256: str,
    raw_input_schema_sha256: str,
    pair_base_sha256: str,
    family_contract: Mapping[str, Any],
    weaver_runtime_sha256: str,
    global_determinism_sha256: str,
) -> dict[str, Any]:
    """Bind a TRACK/DENSITY single to all scientific input contracts."""

    screening_sha = validate_screening_registry(screening_registry)
    normalizer_sha = validate_relation_normalization_artifact(
        normalization_artifact,
        relation_registry_sha256=relation_registry_sha256,
        raw_input_schema_sha256=raw_input_schema_sha256,
    )


    if screening_registry.get(
        "relation_registry_sha256"
    ) != require_sha256(
        relation_registry_sha256, name="relation_registry_sha256"
    ):
        raise ValueError("screening registry belongs to another relation registry")
    row = resolve_registered_run(run_id, screening_registry=screening_registry)
    families = tuple(row.get("new_relation_families", ()))
    if families not in (("TRACK",), ("DENSITY",)):
        raise ValueError(f"{run_id} is not a Step-4 registered single-family run")
    expected_contract = (
        TRACK_RELATION_CONTRACT
        if families == ("TRACK",)
        else DENSITY_RELATION_CONTRACT
    )
    family_sha = validate_content_hash(
        family_contract, expected_contract=expected_contract
    )
    if family_contract.get("relation_registry_sha256") != relation_registry_sha256:
        raise ValueError("family contract belongs to another relation registry")
    if family_contract.get("relation_normalization_sha256") != normalizer_sha:
        raise ValueError("family contract belongs to another relation normalizer")
    if families == ("TRACK",) and family_contract.get(
        "raw_input_schema_sha256"
    ) != raw_input_schema_sha256:
        raise ValueError("TRACK contract belongs to another raw-input schema")
    combined_dimension = (
        STANDARD_FOUR_CHANNELS + SUPPORTED_FAMILY_DIMENSIONS[families[0]]
    )
    return with_content_hash(
        {
            "contract": STEP4_RELATIONAL_MODEL_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "configuration_role": row["configuration_role"],
            "screening_registry_sha256": screening_sha,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "raw_input_schema_sha256": require_sha256(
                raw_input_schema_sha256, name="raw_input_schema_sha256"
            ),
            "relation_normalization_sha256": normalizer_sha,
            "pair_base_sha256": require_sha256(
                pair_base_sha256, name="pair_base_sha256"
            ),
            "family_contract_sha256": family_sha,
            "weaver_runtime_sha256": require_sha256(
                weaver_runtime_sha256, name="weaver_runtime_sha256"
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "enabled_relations": ["base4", *families],
            "new_relation_families": list(families),
            "initialization": "from_scratch",
            "base_config": exact_rpt_base_config(),
            "pair_path": {
                "forward_argument": "uu",
                "raw_hlt_forward_argument": "raw_tokens",
                "canonical_concatenation_order": ["base4", *families],
                "combined_dimension": combined_dimension,
                "weaver_pair_input_dim": 0,
                "weaver_pair_extra_dim": combined_dimension,
                "shared_pair_stem": True,
                "pair_embed_dims": [64, 64, 64, 8],
                "invalid_pairs_zero_after_learned_encoders": True,
            },
            "diagnostics": (
                {
                    "validity_states": 4,
                    "displaced_threshold_raw_absolute_significance": 2.0,
                    "bias_groups": [
                        "prompt_prompt",
                        "prompt_displaced",
                        "displaced_displaced",
                    ],
                }
                if families == ("TRACK",)
                else {
                    "annulus_nonempty_fractions": 4,
                    "bias_axis": "valid_neighbor_fraction_R0p40",
                }
            ),
            "internal_amp": False,
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
        }
    )


def build_step5_model_contract(
    run_id: str,
    *,
    normalization_artifact: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    relation_registry_sha256: str,
    pair_base_sha256: str,
    family_contract_sha256: Mapping[str, str],
    weaver_runtime_sha256: str,
    global_determinism_sha256: str,
    region_normalization_artifact: Mapping[str, Any] | None = None,
    wide_capacity_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    screening_sha = validate_screening_registry(screening_registry)
    base_normalizer_sha = validate_relation_normalization_artifact(
        normalization_artifact,
        relation_registry_sha256=relation_registry_sha256,
    )
    row = resolve_registered_run(run_id, screening_registry=screening_registry)
    families = tuple(row.get("new_relation_families", ()))
    required_hashes = {
        family: require_sha256(
            family_contract_sha256.get(family),
            name=f"family_contract_sha256.{family}",
        )
        for family in families
    }
    region_normalizer_sha = None
    if "REGION" in families:
        if region_normalization_artifact is None:
            raise ValueError("REGION model contract requires REGION normalization")
        region_normalizer_sha = validate_region_normalization(
            region_normalization_artifact,
            relation_normalization_sha256=base_normalizer_sha,
        )
    capacity_sha = None
    if run_id == "RPT_BASE_WIDE_MAX":
        if wide_capacity_artifact is None:
            raise ValueError("wide model contract requires capacity search artifact")
        capacity_sha = validate_content_hash(
            wide_capacity_artifact, expected_contract=WIDE_CAPACITY_CONTRACT
        )
    combined_dimension = STANDARD_FOUR_CHANNELS + sum(
        SUPPORTED_FAMILY_DIMENSIONS[family] for family in families
    )
    return with_content_hash(
        {
            "contract": STEP5_RELATIONAL_MODEL_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "configuration_role": row["configuration_role"],
            "screening_registry_sha256": screening_sha,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "pair_base_sha256": require_sha256(
                pair_base_sha256, name="pair_base_sha256"
            ),
            "relation_normalization_sha256": base_normalizer_sha,
            "region_normalization_sha256": region_normalizer_sha,
            "family_contract_sha256": required_hashes,
            "wide_capacity_sha256": capacity_sha,
            "weaver_runtime_sha256": require_sha256(
                weaver_runtime_sha256, name="weaver_runtime_sha256"
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "enabled_relations": ["base4", *families],
            "new_relation_families": list(families),
            "canonical_concatenation_order": ["base4", *families],
            "combined_dimension": combined_dimension,
            "relation_input_mode": row["relation_input_mode"],
            "attention_architecture": row["attention_architecture"],
            "initialization": "from_scratch",
            "full_zero_exact_shape": run_id == "RPT_FULL_ZERO_REL",
            "persistent_N_by_N_cache": False,
            "hlt_only_inference": True,
        }
    )


def build_rpt_base_model_contract(
    *,
    pair_base_sha256: str,
    weaver_runtime_sha256: str,
    global_determinism_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": RPT_BASE_MODEL_CONTRACT,
            "schema_version": 1,
            "pair_base_sha256": require_sha256(
                pair_base_sha256, name="pair_base_sha256"
            ),
            "weaver_runtime_sha256": require_sha256(
                weaver_runtime_sha256, name="weaver_runtime_sha256"
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "run_id": "RPT_BASE",
            "configuration_role": "reference_baseline",
            "initialization": "from_scratch",
            "config": exact_rpt_base_config(),
            "required_effective_weaver_defaults": copy.deepcopy(
                RPT_BASE_EFFECTIVE_WEAVER_DEFAULTS
            ),
            "pair_path": {
                "standard_four_source": "installed_Weaver_helper",
                "forward_argument": "uu",
                "pair_embed_adapter": "state_structure_preserving",
                "new_relation_channels": 0,
            },
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
        }
    )


__all__ = [
    "ExplicitStandardFourPairEmbed",
    "RPT_BASE_CONFIG",
    "RPT_BASE_EFFECTIVE_WEAVER_DEFAULTS",
    "RPT_BASE_MODEL_CONTRACT",
    "STEP3_RELATIONAL_MODEL_CONTRACT",
    "STEP4_RELATIONAL_MODEL_CONTRACT",
    "STEP5_RELATIONAL_MODEL_CONTRACT",
    "STEP6_RELATIONAL_MODEL_CONTRACT",
    "RelationalFamilyParticleTransformer",
    "RelationalParticleTransformer",
    "WideBaseParticleTransformer",
    "build_registered_screening_model",
    "build_confirmation_architecture_model",
    "build_registered_wide_model",
    "build_registered_step3_model",
    "build_registered_step4_model",
    "build_relational_particle_transformer",
    "build_rpt_base_model_contract",
    "build_step3_model_contract",
    "build_step4_model_contract",
    "build_step5_model_contract",
    "build_step6_model_contract",
    "exact_rpt_base_config",
]
