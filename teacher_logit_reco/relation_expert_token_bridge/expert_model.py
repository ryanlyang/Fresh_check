"""RETB particle encoder, streamed relation bias, tokenizer, and expert model."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from teacher_logit_reco.relational_part.attention import DirectionalPairStem
from teacher_logit_reco.relational_part.model import exact_rpt_base_config
from teacher_logit_reco.relational_part.pair_base import (
    STANDARD_FOUR_CHANNELS,
    build_standard_four_pair_features,
)
from teacher_logit_reco.relational_part.pair_builder import (
    SUPPORTED_FAMILY_DIMENSIONS,
    RelationalPairBuilder,
    SharedDirectionalPairEmbed,
)

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .layerwise_pair_bias import (
    DUAL_TOPOLOGIES,
    LAYERWISE_PAIR_BIAS_CONTRACT,
    LayerwisePairBiasProvider,
)
from .particle_tap import (
    MeasurementStateEmbedding,
    derive_measurement_states_torch,
)
from .registry import EXPERT_ORDER
from .summary_tokens import (
    CanonicalSummaryTokenizer,
    MultiDepthSummaryTokenizer,
    TokenOnlyExpertHead,
    build_summary_tokenizer,
)
from .token_shape_registry import resolve_uniform_shape

try:
    import torch
    from torch.utils.checkpoint import checkpoint
except ImportError:  # pragma: no cover - environment dependent
    torch = None
    checkpoint = None


EXPERT_ARCHITECTURE_CONTRACT = "retb_expert_token_architecture_v1"
EXPERT_STATE_DICTIONARY_CONTRACT = "retb_expert_state_dictionary_v1"
PAIR_TOPOLOGIES = ("B_CONCAT", *DUAL_TOPOLOGIES)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB expert models")
    return torch


def expert_relation_family(expert_id: str) -> str | None:
    if expert_id not in EXPERT_ORDER:
        raise ValueError(f"unknown RETB expert {expert_id!r}")
    return None if expert_id == "BASE4" else str(expert_id)


def build_expert_architecture_contract(
    *,
    particle_tap_sha256: str,
    layerwise_pair_bias_sha256: str,
    measurement_embedding_sha256: str,
    summary_tokenizer_sha256: str,
    token_only_head_sha256: str,
    token_shape_registry_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": EXPERT_ARCHITECTURE_CONTRACT,
            "schema_version": 1,
            "parents": {
                "particle_tap": require_sha256(
                    particle_tap_sha256, name="particle_tap_sha256"
                ),
                "layerwise_pair_bias": require_sha256(
                    layerwise_pair_bias_sha256,
                    name="layerwise_pair_bias_sha256",
                ),
                "measurement_embedding": require_sha256(
                    measurement_embedding_sha256,
                    name="measurement_embedding_sha256",
                ),
                "summary_tokenizer": require_sha256(
                    summary_tokenizer_sha256,
                    name="summary_tokenizer_sha256",
                ),
                "token_only_head": require_sha256(
                    token_only_head_sha256, name="token_only_head_sha256"
                ),
                "token_shape_registry": require_sha256(
                    token_shape_registry_sha256,
                    name="token_shape_registry_sha256",
                ),
            },
            "particle_encoder": {
                "input_dimension": 17,
                "embedding_dimensions": [128, 512, 128],
                "hidden_dimension": 128,
                "heads": 8,
                "particle_blocks": 8,
                "activation": "GELU",
                "all_particle_fields": True,
                "attention_dropout": 0.0,
                "screened_residual_activation_dropout": [0.0, 0.1],
            },
            "expert_order": list(EXPERT_ORDER),
            "pair_topologies": list(PAIR_TOPOLOGIES),
            "base4_always_present": True,
            "one_additional_relation_per_nonbase_expert": True,
            "dual_base4_capacity_control": True,
            "dual_zero_relation_shape_control": True,
            "intermediate_state_block_one_based": 4,
            "final_state_block_one_based": 8,
            "classification_input": "summary_tokens_only",
            "state_dictionary_contract": EXPERT_STATE_DICTIONARY_CONTRACT,
            "single_bias_state_dictionary_compatible": False,
        }
    )


def validate_expert_architecture_contract(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=EXPERT_ARCHITECTURE_CONTRACT
    )
    parents = payload.get("parents", {})
    expected = build_expert_architecture_contract(
        particle_tap_sha256=parents.get("particle_tap"),
        layerwise_pair_bias_sha256=parents.get("layerwise_pair_bias"),
        measurement_embedding_sha256=parents.get("measurement_embedding"),
        summary_tokenizer_sha256=parents.get("summary_tokenizer"),
        token_only_head_sha256=parents.get("token_only_head"),
        token_shape_registry_sha256=parents.get("token_shape_registry"),
    )
    semantic = dict(payload)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    expected.pop("content_hash")
    if canonical_sha256(semantic) != canonical_sha256(expected):
        raise ValueError("expert architecture differs from the locked contract")
    return digest


def _pair_embed_for_dimension(
    transformer_factory: Any,
    *,
    input_dimension: int,
) -> Any:
    config = exact_rpt_base_config()
    config["pair_input_dim"] = 0
    config["pair_extra_dim"] = int(input_dimension)
    temporary = transformer_factory(**config)
    pair_embed = getattr(temporary, "pair_embed", None)
    if pair_embed is None:
        raise RuntimeError("Weaver failed to construct an extra-feature pair stem")
    return pair_embed


class RetbParticleEncoder(torch.nn.Module if torch is not None else object):
    """Explicit eight-block encoder with optional streamed dual pair bias."""

    def __init__(
        self,
        *,
        expert_id: str,
        topology: str,
        weaver_module: Any,
        normalization_artifact: Mapping[str, Any] | None = None,
        region_normalization_artifact: Mapping[str, Any] | None = None,
        measurement_embedding: bool = False,
        force_zero_relation: bool = False,
        dual_base4_capacity_control: bool = False,
        activation_checkpointing: bool = True,
        particle_dropout: float = 0.0,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if topology not in PAIR_TOPOLOGIES:
            raise ValueError(f"unknown RETB pair topology {topology!r}")
        if expert_id not in EXPERT_ORDER:
            raise ValueError(f"unknown RETB expert {expert_id!r}")
        transformer_factory = getattr(weaver_module, "ParticleTransformer", None)
        if transformer_factory is None:
            raise RuntimeError("Weaver module lacks ParticleTransformer")
        self.expert_id = str(expert_id)
        self.relation_family = expert_relation_family(expert_id)
        self.topology = str(topology)
        self.measurement_embedding_enabled = bool(measurement_embedding)
        self.force_zero_relation = bool(force_zero_relation)
        self.dual_base4_capacity_control = bool(dual_base4_capacity_control)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.particle_dropout = float(particle_dropout)
        # Evaluation-only semantic controls are intentionally excluded from
        # the state dictionary.  They can be enabled only after a checkpoint
        # has been loaded and never alter learned parameters.
        self._semantic_relation_transform = "active"
        if self.particle_dropout not in {0.0, 0.1}:
            raise ValueError("particle dropout must be a registered 0.0 or 0.1")
        if self.dual_base4_capacity_control and self.relation_family is not None:
            raise ValueError("dual base4 capacity control is a BASE4 expert")
        if (
            self.topology in DUAL_TOPOLOGIES
            and self.relation_family is None
            and not self.dual_base4_capacity_control
            and not self.force_zero_relation
        ):
            raise ValueError(
                "dual BASE4 requires the capacity or zero-relation control"
            )
        if self.topology == "B_CONCAT" and (
            self.force_zero_relation or self.dual_base4_capacity_control
        ):
            raise ValueError("dual shape controls require a dual topology")

        relation_dimension = (
            0
            if self.relation_family is None
            else SUPPORTED_FAMILY_DIMENSIONS[self.relation_family]
        )
        combined_dimension = STANDARD_FOUR_CHANNELS + relation_dimension
        config = exact_rpt_base_config()
        config["block_params"] = {
            "dropout": self.particle_dropout,
            "attn_dropout": 0.0,
            "activation_dropout": self.particle_dropout,
        }
        config["pair_input_dim"] = 0
        if self.topology == "B_CONCAT":
            config["pair_extra_dim"] = combined_dimension
        else:
            config["pair_extra_dim"] = 0
            config["pair_embed_dims"] = None
        self.mod = transformer_factory(**config)
        blocks = getattr(self.mod, "blocks", None)
        if not isinstance(blocks, module.nn.ModuleList) or len(blocks) != 8:
            raise RuntimeError("RETB requires exactly eight particle blocks")
        if int(config["num_heads"]) != 8:
            raise RuntimeError("RETB requires exactly eight attention heads")
        if bool(getattr(self.mod, "include_global_token", False)):
            raise RuntimeError("RETB base encoder cannot include a global token")
        # RETB decisions must pass through summary tokens.  Remove the ordinary
        # class-attention path from the active module and its state dictionary.
        self.mod.cls_token = None
        self.mod.cls_blocks = None
        self.mod.norm = module.nn.Identity()
        self.mod.fc = None

        self.pair_builder = None
        if self.relation_family is not None:
            if normalization_artifact is None:
                raise ValueError("relation expert requires a normalization artifact")
            self.pair_builder = RelationalPairBuilder(
                [self.relation_family],
                normalization_artifact=normalization_artifact,
                weaver_module=weaver_module,
                region_normalization_artifact=region_normalization_artifact,
            )
        object.__setattr__(self, "_weaver_module", weaver_module)

        self.concat_pair_embed = None
        self.pair_bias_provider = None
        if self.topology == "B_CONCAT":
            reference = getattr(self.mod, "pair_embed", None)
            self.concat_pair_embed = SharedDirectionalPairEmbed(
                reference, input_dimension=combined_dimension
            )
            self.mod.pair_embed = None
        else:
            self.mod.pair_embed = None
            base_reference = _pair_embed_for_dimension(
                transformer_factory, input_dimension=STANDARD_FOUR_CHANNELS
            )
            relation_input_dimension = (
                STANDARD_FOUR_CHANNELS
                if self.relation_family is None
                else relation_dimension
            )
            relation_reference = _pair_embed_for_dimension(
                transformer_factory, input_dimension=relation_input_dimension
            )
            self.pair_bias_provider = LayerwisePairBiasProvider(
                base_stem=DirectionalPairStem(
                    base_reference, input_dimension=STANDARD_FOUR_CHANNELS
                ),
                relation_stem=DirectionalPairStem(
                    relation_reference,
                    input_dimension=relation_input_dimension,
                ),
                num_layers=8,
                num_heads=8,
                topology=self.topology,
                force_zero_relation=self.force_zero_relation,
            )
        self.measurement_state_embedding = (
            MeasurementStateEmbedding(128)
            if self.measurement_embedding_enabled
            else None
        )
        self._last_diagnostics: dict[str, Any] | None = None

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract": EXPERT_STATE_DICTIONARY_CONTRACT,
            "schema_version": 1,
            "expert_id": self.expert_id,
            "topology": self.topology,
            "measurement_embedding": self.measurement_embedding_enabled,
            "force_zero_relation": self.force_zero_relation,
            "dual_base4_capacity_control": self.dual_base4_capacity_control,
            "particle_dropout": self.particle_dropout,
            "layerwise_contract": (
                LAYERWISE_PAIR_BIAS_CONTRACT
                if self.topology in DUAL_TOPOLOGIES
                else None
            ),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError(
                "particle encoder state dictionary has incompatible RETB semantics"
            )

    def _untrimmed_pair_features(
        self,
        *,
        features: Any,
        vectors: Any,
        mask: Any,
        raw_tokens: Any,
        region_trees: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[Any, Any | None]:
        base4 = build_standard_four_pair_features(
            vectors,
            mask=mask,
            module=self._weaver_module,
        )
        if self.pair_builder is None:
            relation = (
                base4 if self.dual_base4_capacity_control else None
            )
            return base4, relation
        details = self.pair_builder(
            features,
            vectors,
            mask,
            raw_tokens,
            region_trees,
            return_details=True,
        )
        relation = details["encoded"][self.relation_family]
        if self._semantic_relation_transform == "zero":
            relation = _require_torch().zeros_like(relation)
        elif self._semantic_relation_transform == "within_jet_cyclic":
            transformed = relation.clone()
            valid = mask[:, 0].bool()
            for batch_index in range(int(relation.shape[0])):
                indices = valid[batch_index].nonzero(
                    as_tuple=False
                ).flatten()
                if int(indices.numel()) < 2:
                    continue
                # Independent endpoint cycles preserve the valid-pair support
                # while breaking the event's learned relation assignment.
                source = indices.roll(1)
                destination = indices.roll(-1)
                transformed[batch_index][
                    :, indices[:, None], indices[None, :]
                ] = relation[batch_index][
                    :, source[:, None], destination[None, :]
                ]
            relation = transformed
        elif self._semantic_relation_transform != "active":
            raise RuntimeError("unknown evaluation relation transform")
        return details["base4"], relation

    def set_semantic_relation_transform(self, mode: str) -> None:
        """Set a parameter-free evaluation-only relation perturbation."""

        if mode not in {"active", "zero", "within_jet_cyclic"}:
            raise ValueError("semantic relation transform is unregistered")
        if self.relation_family is None and mode != "active":
            raise ValueError("BASE4 has no relation family to perturb")
        self._semantic_relation_transform = str(mode)

    def _trim(
        self,
        *,
        features: Any,
        vectors: Any,
        mask: Any,
        pair_features: Any,
        measurement_states: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        packed_features = _require_torch().cat(
            (
                features,
                measurement_states.to(features.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        trimmer = getattr(self.mod, "trimmer", None)
        if callable(trimmer):
            packed_features, vectors, mask, pair_features = trimmer(
                packed_features, vectors, mask, pair_features
            )
        trimmed_features = packed_features[:, :17]
        trimmed_states = packed_features[:, 17].round().to(_require_torch().int64)
        return trimmed_features, vectors, mask.bool(), pair_features, trimmed_states

    def forward(
        self,
        *,
        features: Any,
        vectors: Any,
        mask: Any,
        raw_tokens: Any,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        module = _require_torch()
        if features.ndim != 3 or int(features.shape[1]) != 17:
            raise ValueError("RETB particle features must have shape [B,17,N]")
        batch, _, particles = map(int, features.shape)
        if tuple(vectors.shape) != (batch, 4, particles):
            raise ValueError("RETB Lorentz-vector shape differs")
        if tuple(mask.shape) != (batch, 1, particles):
            raise ValueError("RETB particle mask shape differs")
        if tuple(raw_tokens.shape) != (batch, particles, 14):
            raise ValueError("RETB raw-token shape differs")
        if not bool(mask.bool().any(dim=-1).all()):
            raise ValueError("all-empty rows require canonical dummy particles")
        valid = mask.bool()
        features = features.masked_fill(~valid, 0.0)
        vectors = vectors.masked_fill(~valid, 0.0)
        raw_tokens = raw_tokens.masked_fill(
            ~valid.transpose(1, 2), 0.0
        )
        measurement_states = derive_measurement_states_torch(raw_tokens, mask)
        base4, relation = self._untrimmed_pair_features(
            features=features,
            vectors=vectors,
            mask=mask,
            raw_tokens=raw_tokens,
            region_trees=region_trees,
        )
        if self.topology == "B_CONCAT":
            pair_features = (
                base4
                if relation is None
                else module.cat((base4, relation), dim=1)
            )
            base_channels = int(pair_features.shape[1])
        else:
            if relation is None:
                relation = module.zeros_like(base4)
            base_channels = int(base4.shape[1])
            pair_features = module.cat((base4, relation), dim=1)
        (
            features,
            vectors,
            mask,
            pair_features,
            measurement_states,
        ) = self._trim(
            features=features,
            vectors=vectors,
            mask=mask,
            pair_features=pair_features,
            measurement_states=measurement_states,
        )
        padding_mask = ~mask[:, 0]
        states = self.mod.embed(features).masked_fill(
            ~mask.transpose(1, 2), 0.0
        )
        if int(states.shape[-1]) != 128:
            raise RuntimeError("RETB particle hidden width differs from 128")
        if self.measurement_state_embedding is not None:
            states = states + self.measurement_state_embedding(
                measurement_states, mask
            )

        intermediate = None
        provider_diagnostics = None
        if self.topology == "B_CONCAT":
            pair_bias = self.concat_pair_embed(
                vectors, uu=pair_features, mask=mask
            )
            for layer_index, block in enumerate(self.mod.blocks):
                states = block(
                    states,
                    padding_mask=padding_mask,
                    attn_mask=pair_bias,
                )
                if layer_index == 3:
                    intermediate = states
        else:
            base_pairs = pair_features[:, :base_channels]
            relation_pairs = pair_features[:, base_channels:]
            base_latent, relation_latent = self.pair_bias_provider.build_latents(
                base_pairs, relation_pairs, mask
            )
            pair_mask = (
                mask.bool().unsqueeze(-1) & mask.bool().unsqueeze(-2)
            )
            self.pair_bias_provider.bind(base_latent, relation_latent, mask)
            try:
                for layer_index, block in enumerate(self.mod.blocks):
                    if (
                        self.training
                        and self.activation_checkpointing
                        and states.requires_grad
                    ):
                        def checked(
                            current: Any,
                            base: Any,
                            relation_value: Any,
                            *,
                            _layer_index: int = layer_index,
                            _block: Any = block,
                        ) -> Any:
                            bias = self.pair_bias_provider.checkpointed_bias_for_layer(
                                _layer_index,
                                base,
                                relation_value,
                                pair_mask,
                            )
                            return _block(
                                current,
                                padding_mask=padding_mask,
                                attn_mask=bias,
                            )

                        states = checkpoint(
                            checked,
                            states,
                            base_latent,
                            relation_latent,
                            use_reentrant=False,
                        )
                        self.pair_bias_provider.record_checkpointed_layer(
                            layer_index,
                            batch_size=batch,
                            sequence_length=int(mask.shape[-1]),
                        )
                    else:
                        bias = self.pair_bias_provider.bias_for_layer(layer_index)
                        states = block(
                            states,
                            padding_mask=padding_mask,
                            attn_mask=bias,
                        )
                        del bias
                    if layer_index == 3:
                        intermediate = states
                provider_diagnostics = self.pair_bias_provider.diagnostics()
            finally:
                self.pair_bias_provider.clear()
        if intermediate is None:
            raise RuntimeError("RETB failed to capture explicit block-4 states")
        final = states.masked_fill(~mask.transpose(1, 2), 0.0)
        intermediate = intermediate.masked_fill(
            ~mask.transpose(1, 2), 0.0
        )
        self._last_diagnostics = {
            "expert_id": self.expert_id,
            "topology": self.topology,
            "particle_state_shapes": {
                "block4": list(intermediate.shape),
                "block8": list(final.shape),
            },
            "measurement_embedding": self.measurement_embedding_enabled,
            "particle_dropout": self.particle_dropout,
            "pair_bias_provider": provider_diagnostics,
            "materialized_B_L_H_N_N": False,
        }
        return {
            "particle_states": final,
            "particle_mask": mask[:, 0],
            "intermediate_particle_states": intermediate,
            "intermediate_particle_mask": mask[:, 0],
            "measurement_states": measurement_states,
        }

    def diagnostics(self) -> dict[str, Any]:
        if self._last_diagnostics is None:
            raise RuntimeError("particle encoder has not completed a forward pass")
        return copy.deepcopy(self._last_diagnostics)


class RetbExpertModel(torch.nn.Module if torch is not None else object):
    """End-to-end expert whose decision is forced through saved tokens."""

    def __init__(
        self,
        *,
        particle_encoder: RetbParticleEncoder,
        shape_id: str | None = None,
        token_count: int | None = None,
        token_dimension: int | None = None,
        tokenizer_mode: str = "TOK_CANONICAL",
    ) -> None:
        super().__init__()
        if shape_id is not None:
            if token_count is not None or token_dimension is not None:
                raise ValueError(
                    "uniform shape ID cannot be combined with explicit shape"
                )
            resolved_count, resolved_dimension = resolve_uniform_shape(shape_id)
            shape_label = str(shape_id)
        else:
            if token_count is None or token_dimension is None:
                raise ValueError("explicit heterogeneous K and D are required")
            resolved_count = int(token_count)
            resolved_dimension = int(token_dimension)
            if resolved_count not in {1, 2, 4, 8, 16} or resolved_dimension != 128:
                raise ValueError("heterogeneous expert shape is not registered")
            shape_label = f"HET_K{resolved_count}_D128"
        self.particle_encoder = particle_encoder
        self.shape_id = shape_label
        self.tokenizer_mode = str(tokenizer_mode)
        self.token_count = resolved_count
        self.token_dimension = resolved_dimension
        if shape_id is None:
            if tokenizer_mode != "TOK_CANONICAL":
                raise ValueError(
                    "heterogeneous Step-3 instantiation uses canonical tokenizer"
                )
            self.tokenizer = CanonicalSummaryTokenizer(
                expert_id=particle_encoder.expert_id,
                token_count=resolved_count,
                token_dimension=resolved_dimension,
            )
        else:
            self.tokenizer = build_summary_tokenizer(
                mode=tokenizer_mode,
                expert_id=particle_encoder.expert_id,
                shape_id=shape_id,
            )
        self.head = TokenOnlyExpertHead(
            token_dimension=resolved_dimension,
            num_classes=10,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract": EXPERT_STATE_DICTIONARY_CONTRACT,
            "schema_version": 1,
            "shape_id": self.shape_id,
            "tokenizer_mode": self.tokenizer_mode,
            "expert_id": self.particle_encoder.expert_id,
            "topology": self.particle_encoder.topology,
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping) or dict(state) != self.get_extra_state():
            raise RuntimeError(
                "expert state dictionary has incompatible RETB semantics"
            )

    def tokenize(self, encoded: Mapping[str, Any]) -> Any:
        if isinstance(self.tokenizer, MultiDepthSummaryTokenizer):
            return self.tokenizer(
                encoded["intermediate_particle_states"],
                encoded["particle_states"],
                encoded["intermediate_particle_mask"],
                encoded["particle_mask"],
            )
        return self.tokenizer(
            encoded["particle_states"], encoded["particle_mask"]
        )

    def forward_from_tokens(self, tokens: Any) -> Any:
        return self.head(tokens)

    def forward(self, *, return_details: bool = False, **batch: Any) -> Any:
        encoded = self.particle_encoder(**batch)
        tokens = self.tokenize(encoded)
        logits = self.forward_from_tokens(tokens)
        if return_details:
            return {
                **encoded,
                "tokens": tokens,
                "logits": logits,
            }
        return logits


__all__ = [
    "EXPERT_ARCHITECTURE_CONTRACT",
    "EXPERT_STATE_DICTIONARY_CONTRACT",
    "PAIR_TOPOLOGIES",
    "RetbExpertModel",
    "RetbParticleEncoder",
    "build_expert_architecture_contract",
    "expert_relation_family",
    "validate_expert_architecture_contract",
]
