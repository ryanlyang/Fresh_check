"""Canonical learned-query summary tokens and token-only expert classifier."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import canonical_sha256, validate_content_hash, with_content_hash
from .registry import EXPERT_ORDER
from .token_shape_registry import resolve_uniform_shape

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


SUMMARY_TOKENIZER_CONTRACT = "retb_summary_tokenizer_v1"
TOKEN_ONLY_HEAD_CONTRACT = "retb_token_only_expert_head_v1"
TOKENIZER_MODES = (
    "TOK_CANONICAL",
    "TOK_MASKED_MEAN",
    "TOK_ONE_QUERY_NO_SELF",
    "TOK_K_QUERY_NO_SELF",
    "TOK_MULTI_DEPTH",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB summary tokens")
    return torch


def tokenizer_heads(token_dimension: int) -> int:
    if int(token_dimension) == 64:
        return 4
    if int(token_dimension) == 128:
        return 8
    raise ValueError("token dimension must be 64 or 128")


def build_summary_tokenizer_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": SUMMARY_TOKENIZER_CONTRACT,
            "schema_version": 1,
            "canonical_mode": "TOK_CANONICAL",
            "registered_modes": list(TOKENIZER_MODES),
            "particle_width": 128,
            "canonical_blocks": 2,
            "block_order": [
                "pre_norm_slot_self_attention",
                "pre_norm_slot_to_particle_cross_attention",
                "pre_norm_gelu_ffn",
            ],
            "heads": {"D64": 4, "D128": 8},
            "mlp_expansion": 4,
            "attention_dropout": 0.0,
            "residual_dropout": 0.1,
            "controls": {
                "TOK_MASKED_MEAN": {
                    "token_count": 1,
                    "particle_projection": "RMSNorm128+Linear128toD",
                },
                "TOK_ONE_QUERY_NO_SELF": {
                    "token_count": 1,
                    "blocks": 2,
                    "slot_self_attention": False,
                },
                "TOK_K_QUERY_NO_SELF": {
                    "token_count": "K",
                    "blocks": 2,
                    "slot_self_attention": False,
                },
            },
            "query_components": [
                "learned_slot_query",
                "learned_expert_type_embedding",
                "learned_slot_index_embedding",
            ],
            "particle_position_embedding": False,
            "padded_particles_are_keys_or_values": False,
            "summary_slots_always_valid": True,
            "diagnostic_attention_statistics": {
                "retained_full_attention_tensor": False,
                "per_block": [
                    "event_count",
                    "entropy_sum_by_head_slot",
                    "maximum_sum_by_head_slot",
                    "probability_square_sum_by_head_slot",
                ],
            },
            "multi_depth": {
                "one_based_particle_blocks": [4, 8],
                "per_depth_projection": "RMSNorm128+Linear128toD",
                "learned_depth_embeddings": ["intermediate", "final"],
                "concatenation": "masked_particle_axis",
                "classification_bypass": False,
            },
        }
    )


def build_token_only_head_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": TOKEN_ONLY_HEAD_CONTRACT,
            "schema_version": 1,
            "input": "summary_tokens_only",
            "class_query_count": 1,
            "class_attention_blocks": 2,
            "classifier": "RMSNorm(D)+Linear(D,10)",
            "class_query_is_reconstruction_target": False,
            "particle_states_accessible": False,
            "raw_particles_accessible": False,
            "weaver_class_token_accessible": False,
            "external_jet_features_accessible": False,
        }
    )


def _validate_contract(
    payload: Mapping[str, Any],
    *,
    contract: str,
    expected: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(payload, expected_contract=contract)
    semantic = dict(payload)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    target = dict(expected)
    target.pop("content_hash", None)
    if canonical_sha256(semantic) != canonical_sha256(target):
        raise ValueError(f"{contract} differs from the locked contract")
    return digest


def validate_summary_tokenizer_contract(payload: Mapping[str, Any]) -> str:
    return _validate_contract(
        payload,
        contract=SUMMARY_TOKENIZER_CONTRACT,
        expected=build_summary_tokenizer_contract(),
    )


def validate_token_only_head_contract(payload: Mapping[str, Any]) -> str:
    return _validate_contract(
        payload,
        contract=TOKEN_ONLY_HEAD_CONTRACT,
        expected=build_token_only_head_contract(),
    )


class SummaryTokenizerBlock(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        *,
        token_dimension: int,
        particle_width: int,
        enable_slot_self_attention: bool,
    ) -> None:
        module = _require_torch()
        super().__init__()
        dimension = int(token_dimension)
        heads = tokenizer_heads(dimension)
        self.enable_slot_self_attention = bool(enable_slot_self_attention)
        self.slot_self_norm = module.nn.RMSNorm(dimension)
        self.slot_self_attention = module.nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.slot_cross_norm = module.nn.RMSNorm(dimension)
        self.particle_norm = module.nn.RMSNorm(int(particle_width))
        self.key_projection = module.nn.Linear(int(particle_width), dimension)
        self.value_projection = module.nn.Linear(int(particle_width), dimension)
        self.cross_attention = module.nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = module.nn.RMSNorm(dimension)
        self.ffn = module.nn.Sequential(
            module.nn.Linear(dimension, 4 * dimension),
            module.nn.GELU(),
            module.nn.Linear(4 * dimension, dimension),
        )
        self.residual_dropout = module.nn.Dropout(0.1)
        self.collect_attention_diagnostics = False
        self._last_attention_statistics: dict[str, Any] | None = None

    def forward(self, slots: Any, particles: Any, particle_mask: Any) -> Any:
        if self.enable_slot_self_attention:
            normalized = self.slot_self_norm(slots)
            update, _ = self.slot_self_attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            slots = slots + self.residual_dropout(update)
        query = self.slot_cross_norm(slots)
        normalized_particles = self.particle_norm(particles)
        key = self.key_projection(normalized_particles)
        value = self.value_projection(normalized_particles)
        if self.collect_attention_diagnostics:
            update, weights = self.cross_attention(
                query,
                key,
                value,
                key_padding_mask=~particle_mask.bool(),
                need_weights=True,
                average_attn_weights=False,
            )
            valid = particle_mask.bool()[:, None, None, :]
            probabilities = weights.float().masked_fill(~valid, 0.0)
            safe = probabilities.clamp_min(1.0e-30)
            entropy = -(probabilities * safe.log()).sum(dim=-1)
            maximum = probabilities.amax(dim=-1)
            square_sum = probabilities.square().sum(dim=-1)
            self._last_attention_statistics = {
                "event_count": int(probabilities.shape[0]),
                "head_count": int(probabilities.shape[1]),
                "slot_count": int(probabilities.shape[2]),
                "entropy_sum_by_head_slot": (
                    entropy.sum(dim=0).detach().cpu().tolist()
                ),
                "maximum_sum_by_head_slot": (
                    maximum.sum(dim=0).detach().cpu().tolist()
                ),
                "probability_square_sum_by_head_slot": (
                    square_sum.sum(dim=0).detach().cpu().tolist()
                ),
            }
        else:
            update, _ = self.cross_attention(
                query,
                key,
                value,
                key_padding_mask=~particle_mask.bool(),
                need_weights=False,
            )
            self._last_attention_statistics = None
        slots = slots + self.residual_dropout(update)
        slots = slots + self.residual_dropout(self.ffn(self.ffn_norm(slots)))
        return slots

    def set_collect_attention_diagnostics(self, enabled: bool) -> None:
        self.collect_attention_diagnostics = bool(enabled)
        if not enabled:
            self._last_attention_statistics = None

    def attention_sufficient_statistics(self) -> dict[str, Any] | None:
        if self._last_attention_statistics is None:
            return None
        return {
            key: value
            for key, value in self._last_attention_statistics.items()
        }


class CanonicalSummaryTokenizer(
    torch.nn.Module if torch is not None else object
):
    """Two-block learned slot tokenizer producing exactly ``[B,K,D]``."""

    def __init__(
        self,
        *,
        expert_id: str,
        token_count: int,
        token_dimension: int,
        particle_width: int = 128,
        blocks: int = 2,
        enable_slot_self_attention: bool = True,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if expert_id not in EXPERT_ORDER:
            raise ValueError(f"unknown expert ID {expert_id!r}")
        if int(token_count) <= 0:
            raise ValueError("token_count must be positive")
        dimension = int(token_dimension)
        tokenizer_heads(dimension)
        if int(blocks) not in {1, 2}:
            raise ValueError("summary tokenizer supports one or two blocks")
        self.expert_id = str(expert_id)
        self.token_count = int(token_count)
        self.token_dimension = dimension
        self.particle_width = int(particle_width)
        self.slot_queries = module.nn.Parameter(
            module.empty(self.token_count, dimension)
        )
        self.expert_type_embedding = module.nn.Parameter(module.empty(dimension))
        self.slot_index_embedding = module.nn.Parameter(
            module.empty(self.token_count, dimension)
        )
        module.nn.init.trunc_normal_(self.slot_queries, std=0.02)
        module.nn.init.trunc_normal_(self.expert_type_embedding, std=0.02)
        module.nn.init.trunc_normal_(self.slot_index_embedding, std=0.02)
        self.blocks = module.nn.ModuleList(
            SummaryTokenizerBlock(
                token_dimension=dimension,
                particle_width=self.particle_width,
                enable_slot_self_attention=enable_slot_self_attention,
            )
            for _ in range(int(blocks))
        )

    def forward(self, particle_states: Any, particle_mask: Any) -> Any:
        module = _require_torch()
        if (
            not isinstance(particle_states, module.Tensor)
            or particle_states.ndim != 3
            or int(particle_states.shape[-1]) != self.particle_width
        ):
            raise ValueError(
                f"particle states must have shape [B,N,{self.particle_width}]"
            )
        if tuple(particle_mask.shape) != tuple(particle_states.shape[:2]):
            raise ValueError("particle tokenizer mask shape differs")
        if bool((particle_mask.bool().sum(dim=1) == 0).any()):
            raise ValueError("summary tokenizer received an all-empty row")
        base = (
            self.slot_queries
            + self.slot_index_embedding
            + self.expert_type_embedding.unsqueeze(0)
        )
        slots = base.unsqueeze(0).expand(int(particle_states.shape[0]), -1, -1)
        for block in self.blocks:
            slots = block(slots, particle_states, particle_mask)
        expected = (
            int(particle_states.shape[0]),
            self.token_count,
            self.token_dimension,
        )
        if tuple(slots.shape) != expected:
            raise RuntimeError("summary tokenizer output shape drifted")
        if not bool(module.isfinite(slots).all()):
            raise FloatingPointError("summary tokens contain NaN or infinity")
        return slots

    def set_collect_attention_diagnostics(self, enabled: bool) -> None:
        for block in self.blocks:
            block.set_collect_attention_diagnostics(enabled)

    def attention_sufficient_statistics(self) -> list[dict[str, Any] | None]:
        return [
            block.attention_sufficient_statistics() for block in self.blocks
        ]


class MultiDepthSummaryTokenizer(
    torch.nn.Module if torch is not None else object
):
    """Canonical tokenizer over explicit block-4 and block-8 particle states."""

    depth_block_numbers = (4, 8)

    def __init__(
        self,
        *,
        expert_id: str,
        token_count: int,
        token_dimension: int,
    ) -> None:
        module = _require_torch()
        super().__init__()
        dimension = int(token_dimension)
        tokenizer_heads(dimension)
        self.depth_norms = module.nn.ModuleList(
            [module.nn.RMSNorm(128), module.nn.RMSNorm(128)]
        )
        self.depth_projections = module.nn.ModuleList(
            [module.nn.Linear(128, dimension), module.nn.Linear(128, dimension)]
        )
        self.depth_embeddings = module.nn.Parameter(module.empty(2, dimension))
        module.nn.init.trunc_normal_(self.depth_embeddings, std=0.02)
        self.tokenizer = CanonicalSummaryTokenizer(
            expert_id=expert_id,
            token_count=token_count,
            token_dimension=dimension,
            particle_width=dimension,
            blocks=2,
            enable_slot_self_attention=True,
        )

    def forward(
        self,
        intermediate_states: Any,
        final_states: Any,
        intermediate_mask: Any,
        final_mask: Any,
    ) -> Any:
        if tuple(intermediate_states.shape) != tuple(final_states.shape):
            raise ValueError("multi-depth particle-state shapes differ")
        if int(intermediate_states.shape[-1]) != 128:
            raise ValueError("multi-depth particle width must be 128")
        if tuple(intermediate_mask.shape) != tuple(intermediate_states.shape[:2]):
            raise ValueError("intermediate depth mask shape differs")
        if tuple(final_mask.shape) != tuple(final_states.shape[:2]):
            raise ValueError("final depth mask shape differs")
        projected = []
        for index, states in enumerate((intermediate_states, final_states)):
            value = self.depth_projections[index](self.depth_norms[index](states))
            projected.append(value + self.depth_embeddings[index].view(1, 1, -1))
        particles = _require_torch().cat(projected, dim=1)
        mask = _require_torch().cat(
            (intermediate_mask.bool(), final_mask.bool()), dim=1
        )
        return self.tokenizer(particles, mask)

    def set_collect_attention_diagnostics(self, enabled: bool) -> None:
        self.tokenizer.set_collect_attention_diagnostics(enabled)

    def attention_sufficient_statistics(self) -> list[dict[str, Any] | None]:
        return self.tokenizer.attention_sufficient_statistics()


class MaskedMeanTokenizer(torch.nn.Module if torch is not None else object):
    def __init__(self, *, token_dimension: int) -> None:
        module = _require_torch()
        super().__init__()
        self.token_count = 1
        self.token_dimension = int(token_dimension)
        tokenizer_heads(self.token_dimension)
        self.projection = module.nn.Sequential(
            module.nn.RMSNorm(128),
            module.nn.Linear(128, self.token_dimension),
        )

    def forward(self, particle_states: Any, particle_mask: Any) -> Any:
        projected = self.projection(particle_states)
        valid = particle_mask.bool().unsqueeze(-1)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        return (projected.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / count)


class ClassAttentionBlock(torch.nn.Module if torch is not None else object):
    def __init__(self, dimension: int) -> None:
        module = _require_torch()
        super().__init__()
        heads = tokenizer_heads(int(dimension))
        self.query_norm = module.nn.RMSNorm(int(dimension))
        self.token_norm = module.nn.RMSNorm(int(dimension))
        self.attention = module.nn.MultiheadAttention(
            int(dimension), heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = module.nn.RMSNorm(int(dimension))
        self.ffn = module.nn.Sequential(
            module.nn.Linear(int(dimension), 4 * int(dimension)),
            module.nn.GELU(),
            module.nn.Linear(4 * int(dimension), int(dimension)),
        )
        self.dropout = module.nn.Dropout(0.1)

    def forward(self, query: Any, tokens: Any) -> Any:
        update, _ = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )
        query = query + self.dropout(update)
        return query + self.dropout(self.ffn(self.ffn_norm(query)))


class TokenOnlyExpertHead(torch.nn.Module if torch is not None else object):
    """Classifier whose public forward accepts only a summary-token bank."""

    def __init__(self, *, token_dimension: int, num_classes: int = 10) -> None:
        module = _require_torch()
        super().__init__()
        dimension = int(token_dimension)
        tokenizer_heads(dimension)
        if int(num_classes) != 10:
            raise ValueError("RETB expert head has exactly ten classes")
        self.token_dimension = dimension
        self.class_query = module.nn.Parameter(module.empty(1, 1, dimension))
        module.nn.init.trunc_normal_(self.class_query, std=0.02)
        self.blocks = module.nn.ModuleList(
            [ClassAttentionBlock(dimension), ClassAttentionBlock(dimension)]
        )
        self.norm = module.nn.RMSNorm(dimension)
        self.classifier = module.nn.Linear(dimension, 10)

    def forward(self, tokens: Any) -> Any:
        module = _require_torch()
        if (
            not isinstance(tokens, module.Tensor)
            or tokens.ndim != 3
            or int(tokens.shape[-1]) != self.token_dimension
        ):
            raise ValueError("token-only head requires [B,K,D] tokens")
        if int(tokens.shape[1]) < 1:
            raise ValueError("token-only head requires at least one token")
        query = self.class_query.expand(int(tokens.shape[0]), -1, -1)
        for block in self.blocks:
            query = block(query, tokens)
        logits = self.classifier(self.norm(query[:, 0]))
        if not bool(module.isfinite(logits).all()):
            raise FloatingPointError("token-only expert logits are nonfinite")
        return logits


def build_summary_tokenizer(
    *,
    mode: str,
    expert_id: str,
    shape_id: str,
) -> Any:
    token_count, token_dimension = resolve_uniform_shape(shape_id)
    if mode == "TOK_CANONICAL":
        return CanonicalSummaryTokenizer(
            expert_id=expert_id,
            token_count=token_count,
            token_dimension=token_dimension,
        )
    if mode == "TOK_MULTI_DEPTH":
        return MultiDepthSummaryTokenizer(
            expert_id=expert_id,
            token_count=token_count,
            token_dimension=token_dimension,
        )
    if mode == "TOK_MASKED_MEAN":
        if token_count != 1:
            raise ValueError("TOK_MASKED_MEAN is defined only for K=1")
        return MaskedMeanTokenizer(token_dimension=token_dimension)
    if mode in {"TOK_ONE_QUERY_NO_SELF", "TOK_K_QUERY_NO_SELF"}:
        if mode == "TOK_ONE_QUERY_NO_SELF" and token_count != 1:
            raise ValueError("TOK_ONE_QUERY_NO_SELF requires K=1")
        return CanonicalSummaryTokenizer(
            expert_id=expert_id,
            token_count=token_count,
            token_dimension=token_dimension,
            blocks=2,
            enable_slot_self_attention=False,
        )
    raise ValueError(f"unknown tokenizer mode {mode!r}")


__all__ = [
    "SUMMARY_TOKENIZER_CONTRACT",
    "TOKENIZER_MODES",
    "TOKEN_ONLY_HEAD_CONTRACT",
    "CanonicalSummaryTokenizer",
    "MaskedMeanTokenizer",
    "MultiDepthSummaryTokenizer",
    "SummaryTokenizerBlock",
    "TokenOnlyExpertHead",
    "build_summary_tokenizer",
    "build_summary_tokenizer_contract",
    "build_token_only_head_contract",
    "tokenizer_heads",
    "validate_summary_tokenizer_contract",
    "validate_token_only_head_contract",
]
