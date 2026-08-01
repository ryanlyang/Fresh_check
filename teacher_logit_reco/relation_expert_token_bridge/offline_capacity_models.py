"""Executable single-model controls for the RETB offline capacity wave."""

from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from teacher_logit_reco.relational_part.model import (
    RelationalParticleTransformer,
    exact_rpt_base_config,
)
from teacher_logit_reco.relational_part.pair_builder import (
    SUPPORTED_FAMILY_DIMENSIONS,
    RelationalPairBuilder,
    SharedDirectionalPairEmbed,
)

from .fusion import GroupedHeadRelationBias

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FAMILIES = ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")
MONOLITHIC_CONFIGURATION_FIELDS = (
    "particle_hidden_width",
    "feed_forward_expansion",
    "attention_head_count",
    "particle_block_count",
    "class_block_count",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for offline capacity models")
    return torch


def build_monolithic_grid() -> list[tuple[int, int, int, int, int]]:
    """Return the frozen dense grid used by both complete-graph selectors."""

    rows = []
    for width in (96, 128, 160, 192, 256):
        for expansion in (2, 4, 6):
            for heads in (4, 8, 16):
                if width % heads:
                    continue
                for particle_blocks in (6, 8, 10, 12):
                    for class_blocks in (1, 2, 3):
                        rows.append(
                            (
                                width,
                                expansion,
                                heads,
                                particle_blocks,
                                class_blocks,
                            )
                        )
    return rows


def monolithic_config(
    configuration: Sequence[int],
) -> dict[str, Any]:
    if len(configuration) != 5:
        raise ValueError("monolithic configuration must have five fields")
    width, expansion, heads, particle_blocks, class_blocks = map(
        int, configuration
    )
    if (
        width % heads
        or width <= 0
        or expansion <= 0
        or particle_blocks <= 0
        or class_blocks <= 0
    ):
        raise ValueError("monolithic configuration is invalid")
    config = exact_rpt_base_config()
    # Weaver's embedding terminates at the transformer width.  The middle
    # embedding layer remains four times the width, matching ordinary ParT.
    config.update(
        {
            "embed_dims": [width, 4 * width, width],
            "num_heads": heads,
            "num_layers": particle_blocks,
            "num_cls_layers": class_blocks,
            "block_params": {
                "dropout": 0.0,
                "attn_dropout": 0.0,
                "activation_dropout": 0.0,
                "scale_fc": True,
                "scale_attn": True,
                "scale_heads": True,
                "scale_resids": True,
            },
        }
    )
    # Weaver derives its feed-forward width from ``scale_fc`` rather than a
    # public dimension argument.  Record the requested expansion and accept
    # only its canonical factors; the explicit profile binds the instantiated
    # graph, so selector evidence cannot silently use the requested value.
    config["retb_feed_forward_expansion"] = expansion
    return config


class OfflineClassifierAdapter(
    torch.nn.Module if torch is not None else object
):
    """Give ordinary Particle Transformers the campaign batch interface."""

    def __init__(
        self,
        classifier: Any,
        *,
        expert_configuration: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self._expert_configuration = (
            None
            if expert_configuration is None
            else dict(expert_configuration)
        )
        if self._expert_configuration is not None:
            config = self._expert_configuration
            # The expert trainer validates these public semantics.  The
            # ordinary Weaver control deliberately has no RetbParticleEncoder,
            # but it must still attest which expert/relation configuration it
            # replaces.
            self.particle_encoder = SimpleNamespace(
                expert_id=config["expert_id"],
                topology=config["topology"],
                particle_dropout=float(config["particle_dropout"]),
                measurement_embedding_enabled=bool(
                    config["measurement_embedding"]
                ),
            )
            self.shape_id = str(config["shape_id"])
            self.token_count = int(config["token_count"])
            self.token_dimension = int(config["token_dimension"])
            self.tokenizer_mode = str(config["tokenizer_mode"])

    def forward(
        self,
        *,
        features: Any,
        vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
        return_details: bool = False,
        **_: Any,
    ) -> Any:
        kwargs = {
            "points": features[:, -2:, :],
            "features": features,
            "lorentz_vectors": vectors,
            "mask": mask,
        }
        if raw_tokens is not None:
            kwargs["raw_tokens"] = raw_tokens
        if region_trees is not None:
            kwargs["region_trees"] = region_trees
        accepted = inspect.signature(self.classifier.forward).parameters
        logits = self.classifier(
            **{name: value for name, value in kwargs.items() if name in accepted}
        )
        if not return_details:
            return logits
        if self._expert_configuration is None:
            raise ValueError(
                "ordinary classifier details require expert configuration"
            )
        # TOK_WEAVER_CLASS has no summary-token bottleneck.  Diagnostics still
        # require a shape-stable tensor, so publish a non-consumable zero
        # sentinel at the registered shape; the classifier logits remain the
        # only scientific output of this control.
        tokens = logits.new_zeros(
            logits.shape[0], self.token_count, self.token_dimension
        )
        return {"tokens": tokens, "logits": logits}


class MonolithicBase4ParticleTransformer(
    torch.nn.Module if torch is not None else object
):
    """Token-free base4 ParT for deterministic parameter/FLOP matching."""

    def __init__(
        self,
        configuration: Sequence[int],
        *,
        weaver_module: Any,
    ) -> None:
        super().__init__()
        resolved = monolithic_config(configuration)
        expansion = int(resolved.pop("retb_feed_forward_expansion"))
        transformer = getattr(weaver_module, "ParticleTransformer", None)
        if transformer is None:
            raise RuntimeError("Weaver module lacks ParticleTransformer")
        self.configuration = tuple(map(int, configuration))
        self.feed_forward_expansion = expansion
        self.config = copy.deepcopy(resolved)
        self.mod = transformer(**resolved)
        module = _require_torch()
        rewritten = 0
        for collection_name in ("blocks", "cls_blocks"):
            collection = getattr(self.mod, collection_name, None)
            if collection is None:
                continue
            for block in collection:
                fc1 = getattr(block, "fc1", None)
                fc2 = getattr(block, "fc2", None)
                if not (
                    isinstance(fc1, module.nn.Linear)
                    and isinstance(fc2, module.nn.Linear)
                    and int(fc1.in_features) == self.configuration[0]
                    and int(fc2.out_features) == self.configuration[0]
                ):
                    raise RuntimeError(
                        "installed Weaver block lacks rewritable fc1/fc2"
                    )
                hidden = self.configuration[0] * expansion
                block.fc1 = module.nn.Linear(self.configuration[0], hidden)
                block.fc2 = module.nn.Linear(hidden, self.configuration[0])
                rewritten += 1
        if rewritten != self.configuration[3] + self.configuration[4]:
            raise RuntimeError("monolithic feed-forward block coverage differs")
        object.__setattr__(self, "_weaver_module", weaver_module)

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
    ) -> Any:
        # This is intentionally the ordinary Weaver physical-pair path.  It
        # has the same token-free interface as O_BASE and changes only dense
        # model capacity.
        return self.mod(features, v=lorentz_vectors, mask=mask)


class _GroupedPairEmbed(
    torch.nn.Module if torch is not None else object
):
    def __init__(self, *, transformer_factory: Any) -> None:
        module = _require_torch()
        super().__init__()
        dimensions = {"base4": 4, **SUPPORTED_FAMILY_DIMENSIONS}
        stems = {}
        for name, dimension in dimensions.items():
            config = exact_rpt_base_config()
            config["pair_input_dim"] = 0
            config["pair_extra_dim"] = int(dimension)
            temporary = transformer_factory(**config)
            stems[name] = SharedDirectionalPairEmbed(
                temporary.pair_embed, input_dimension=int(dimension)
            )
        self.stems = module.nn.ModuleDict(stems)
        self.assembler = GroupedHeadRelationBias()
        self.slices = {}
        offset = 4
        for family in FAMILIES:
            width = int(SUPPORTED_FAMILY_DIMENSIONS[family])
            self.slices[family] = (offset, offset + width)
            offset += width
        self.input_dimension = offset

    def forward(self, v: Any, uu: Any = None, mask: Any | None = None) -> Any:
        if (
            uu is None
            or uu.ndim != 4
            or int(uu.shape[1]) != self.input_dimension
            or mask is None
        ):
            raise ValueError("grouped-head pair input differs")
        base = self.stems["base4"](v, uu=uu[:, :4], mask=mask)
        relation = {
            family: self.stems[family](
                v,
                uu=uu[:, left:right],
                mask=mask,
            )
            for family, (left, right) in self.slices.items()
        }
        return self.assembler(
            base4_bias=base, relation_biases=relation
        )


class GroupedHeadRelationParticleTransformer(
    torch.nn.Module if torch is not None else object
):
    """One ParT whose eight heads receive fixed disjoint relation families."""

    def __init__(
        self,
        *,
        normalization_artifact: Mapping[str, Any],
        region_normalization_artifact: Mapping[str, Any],
        weaver_module: Any,
    ) -> None:
        super().__init__()
        transformer = getattr(weaver_module, "ParticleTransformer", None)
        if transformer is None:
            raise RuntimeError("Weaver module lacks ParticleTransformer")
        self.pair_builder = RelationalPairBuilder(
            FAMILIES,
            normalization_artifact=normalization_artifact,
            region_normalization_artifact=region_normalization_artifact,
            weaver_module=weaver_module,
        )
        config = exact_rpt_base_config()
        config["pair_input_dim"] = 0
        config["pair_extra_dim"] = 0
        config["pair_embed_dims"] = None
        self.mod = transformer(**config)
        self.mod.pair_embed = _GroupedPairEmbed(
            transformer_factory=transformer
        )

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any,
        region_trees: Any,
    ) -> Any:
        RelationalParticleTransformer._validate_batch(
            points, features, lorentz_vectors, mask
        )
        combined = self.pair_builder(
            features,
            lorentz_vectors,
            mask,
            raw_tokens,
            region_trees,
        )
        return self.mod(
            features,
            v=lorentz_vectors,
            mask=mask,
            uu=combined,
        )


def analytical_particle_transformer_flops(
    *,
    configuration: Sequence[int],
    particles: int = 128,
    classes: int = 10,
) -> int:
    """Locked multiply-add formula for dense token-free ParT comparisons."""

    width, expansion, _heads, particle_blocks, class_blocks = map(
        int, configuration
    )
    n = int(particles)
    # Per particle block: QKV+output projections, dense QK/AV attention, and
    # the two feed-forward projections.  Heads cancel in the aggregate.
    block = (
        8 * n * width * width
        + 4 * n * n * width
        + 4 * n * width * (expansion * width)
    )
    # Class attention has one query against N particles plus the same MLP.
    cls = (
        8 * (n + 1) * width * width
        + 4 * n * width
        + 4 * width * (expansion * width)
    )
    embedding = 2 * n * (17 * width + width * 4 * width + 4 * width * width)
    classifier = 2 * width * int(classes)
    total = (
        embedding
        + int(particle_blocks) * block
        + int(class_blocks) * cls
        + classifier
    )
    if total <= 0:
        raise ValueError("analytical ParT FLOPs are not positive")
    return int(total)


__all__ = [
    "FAMILIES",
    "MONOLITHIC_CONFIGURATION_FIELDS",
    "GroupedHeadRelationParticleTransformer",
    "MonolithicBase4ParticleTransformer",
    "OfflineClassifierAdapter",
    "analytical_particle_transformer_flops",
    "build_monolithic_grid",
    "monolithic_config",
]
