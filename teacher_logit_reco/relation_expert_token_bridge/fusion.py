"""Offline token-fusion architectures and Stage-C capacity controls."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import validate_content_hash, with_content_hash
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


OFFLINE_FUSION_ARCHITECTURE_CONTRACT = "retb_offline_fusion_architecture_v1"
GROUPED_HEAD_RELATION_CONTRACT = "retb_grouped_head_relation_control_v1"
RELATION_SPECIALIZATION_CONTRACT = "retb_relation_specialization_controls_v1"
FUSION_VARIANTS = (
    "F_BEST_SINGLE",
    "F_UNIFORM_LOGIT_MEAN",
    "F_TRAINED_LOGIT_LINEAR",
    "F_POOLED_MLP",
    "F_TOKEN_TRANSFORMER",
    "F_TOKEN_TRANSFORMER_LIGHT_FINETUNE",
    "F_TOKEN_TRANSFORMER_FULL_FINETUNE",
)
EXPERT_INDEX = {name: index for index, name in enumerate(EXPERT_ORDER)}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB fusion")
    return torch


def build_offline_fusion_architecture_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": OFFLINE_FUSION_ARCHITECTURE_CONTRACT,
            "schema_version": 1,
            "expert_order": list(EXPERT_ORDER),
            "variants": list(FUSION_VARIANTS),
            "token_transformer": {
                "width": 128,
                "layers": 3,
                "heads": 8,
                "mlp_expansion": 4,
                "pre_norm": True,
                "attention_dropout": 0.0,
                "residual_dropout": 0.1,
                "classifier": "RMSNorm_then_Linear_128_to_10",
                "source_embedding": "oracle_offline",
                "fusion_class_token_count": 1,
            },
            "bank_projection": {
                "D128": "identity_initialized_trainable_Linear_128_to_128",
                "D64": "RMSNorm_64_then_Linear_64_to_128",
                "belongs_to": "fusion_model",
            },
            "primary_expert_state": "frozen",
            "light_finetune": {
                "trainable": "summary_tokenizers_only",
                "tokenizer_lr_multiplier": 0.1,
            },
            "full_finetune": {"trainable": "complete_experts"},
            "whole_bank_dropout_primary": 0.0,
            "anti_redundancy_dropout": False,
        }
    )


def validate_offline_fusion_architecture_contract(
    payload: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=OFFLINE_FUSION_ARCHITECTURE_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_offline_fusion_architecture_contract()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("offline fusion architecture contract differs")
    return digest


def build_grouped_head_relation_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": GROUPED_HEAD_RELATION_CONTRACT,
            "schema_version": 1,
            "head_mapping": {
                "0": ["base4"],
                "1": ["base4"],
                "2": ["base4", "PT"],
                "3": ["base4", "TRACK"],
                "4": ["base4", "PID"],
                "5": ["base4", "CHARGE"],
                "6": ["base4", "DENSITY"],
                "7": ["base4", "REGION"],
            },
            "softmax_scope": "independent_per_attention_head",
            "relations_summed_before_softmax": False,
            "output_shape": ["B", 8, "N", "N"],
        }
    )


def build_relation_specialization_contract() -> dict[str, Any]:
    summaries = {
        "BASE4": [
            "jet_mass_fraction",
            "multiplicity",
            "deltaR_pair_quartiles",
        ],
        "PT": [
            "leading_four_pt_fractions",
            "pt_entropy",
            "scalar_pt_concentration",
        ],
        "TRACK": [
            "valid_track_fraction",
            "abs_d0_significance_quartiles",
            "abs_dz_significance_quartiles",
            "compatible_pair_fraction_chi2_1_4_9",
        ],
        "PID": ["six_PID_pt_fractions", "six_PID_count_fractions"],
        "CHARGE": [
            "positive_negative_neutral_pt_fractions",
            "normalized_net_charge",
            "opposite_same_sign_valid_pair_fractions",
        ],
        "DENSITY": [
            "four_annular_mean_counts",
            "four_annular_pt_fractions",
            "displaced_local_pt_fraction",
        ],
        "REGION": [
            "K2_4_8_leading_cluster_pt_fractions",
            "cluster_multiplicities",
            "LCA_depth_quartiles",
            "merge_scale_quartiles",
        ],
    }
    return with_content_hash(
        {
            "contract": RELATION_SPECIALIZATION_CONTRACT,
            "schema_version": 1,
            "controls": {
                "S0_NATURAL": {"primary": True, "regularizer": None},
                "S1_FIXED_SCALE": {
                    "topology": "B_DUAL_FIXED",
                    "relation_scale": 1.0,
                },
                "S2_BOUNDED_SCALE": {
                    "topology": "B_DUAL_GATED",
                    "formula": "2*sigmoid(a_layer_head)",
                },
                "S3_RELATION_AUX": {
                    "weight": 0.10,
                    "train_only_standardization": True,
                    "masked_missing_targets": True,
                    "summaries": summaries,
                },
                "S4_RESTRICTED_FIELDS": {
                    "interpretability_only": True,
                    "hide_non_relation_particle_fields": True,
                },
                "S5_CROSSCOV": {
                    "weight": 1.0e-3,
                    "per_bank_centering": True,
                    "off_diagonal_only": True,
                    "nominal_eligibility_requires_primary_metric_improvement": True,
                },
            },
            "primary_diversity_regularizer": None,
            "expert_dropout_as_diversity": False,
        }
    )


class RMSNorm(torch.nn.Module if torch is not None else object):
    def __init__(self, dimension: int, epsilon: float = 1.0e-8) -> None:
        super().__init__()
        self.weight = _require_torch().nn.Parameter(
            _require_torch().ones(int(dimension))
        )
        self.epsilon = float(epsilon)

    def forward(self, values: Any) -> Any:
        scale = values.square().mean(dim=-1, keepdim=True)
        return values * _require_torch().rsqrt(scale + self.epsilon) * self.weight


class BankProjection(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dimension: int) -> None:
        module = _require_torch()
        super().__init__()
        dimension = int(input_dimension)
        if dimension not in {64, 128}:
            raise ValueError("fusion bank dimension must be 64 or 128")
        self.input_dimension = dimension
        self.norm = RMSNorm(dimension) if dimension == 64 else module.nn.Identity()
        self.linear = module.nn.Linear(dimension, 128)
        if dimension == 128:
            with module.no_grad():
                self.linear.weight.copy_(module.eye(128))
                self.linear.bias.zero_()

    def forward(self, values: Any) -> Any:
        return self.linear(self.norm(values))


def _validate_banks(
    token_banks: Mapping[str, Any],
    *,
    dimensions: Mapping[str, int],
) -> tuple[int, Any]:
    module = _require_torch()
    if set(token_banks) != set(EXPERT_ORDER):
        raise ValueError("fusion token banks differ from canonical experts")
    batch = None
    device = None
    for expert in EXPERT_ORDER:
        values = token_banks[expert]
        if values.ndim != 3 or int(values.shape[-1]) != int(dimensions[expert]):
            raise ValueError(f"fusion bank {expert} has the wrong shape")
        if batch is None:
            batch, device = int(values.shape[0]), values.device
        if int(values.shape[0]) != batch or values.device != device:
            raise ValueError("fusion bank batch/device differs")
        if not bool(module.isfinite(values).all()):
            raise FloatingPointError("fusion token bank is nonfinite")
    return int(batch), device


class UniformLogitMean(torch.nn.Module if torch is not None else object):
    def forward(self, *, expert_logits: Mapping[str, Any], **_: Any) -> Any:
        if set(expert_logits) != set(EXPERT_ORDER):
            raise ValueError("expert logits differ from canonical order")
        return _require_torch().stack(
            [expert_logits[name] for name in EXPERT_ORDER], dim=0
        ).mean(dim=0)


class BestSingleFusion(torch.nn.Module if torch is not None else object):
    def __init__(self, *, expert_id: str) -> None:
        super().__init__()
        if expert_id not in EXPERT_ORDER:
            raise ValueError("best-single expert is not registered")
        self.expert_id = str(expert_id)

    def forward(self, *, expert_logits: Mapping[str, Any], **_: Any) -> Any:
        if set(expert_logits) != set(EXPERT_ORDER):
            raise ValueError("expert logits differ from canonical order")
        return expert_logits[self.expert_id]


class TrainedLogitLinear(torch.nn.Module if torch is not None else object):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = _require_torch().nn.Linear(70, 10)

    def forward(self, *, expert_logits: Mapping[str, Any], **_: Any) -> Any:
        if set(expert_logits) != set(EXPERT_ORDER):
            raise ValueError("expert logits differ from canonical order")
        return self.classifier(
            _require_torch().cat(
                [expert_logits[name] for name in EXPERT_ORDER], dim=-1
            )
        )


class PooledMLPFusion(torch.nn.Module if torch is not None else object):
    def __init__(self, *, bank_dimensions: Mapping[str, int]) -> None:
        module = _require_torch()
        super().__init__()
        if set(bank_dimensions) != set(EXPERT_ORDER):
            raise ValueError("fusion dimensions differ from canonical experts")
        self.bank_dimensions = {name: int(bank_dimensions[name]) for name in EXPERT_ORDER}
        self.projections = module.nn.ModuleDict(
            {
                name: BankProjection(self.bank_dimensions[name])
                for name in EXPERT_ORDER
            }
        )
        self.classifier = module.nn.Sequential(
            RMSNorm(7 * 128),
            module.nn.Linear(7 * 128, 512),
            module.nn.GELU(),
            module.nn.Dropout(0.1),
            module.nn.Linear(512, 10),
        )

    def forward(self, *, token_banks: Mapping[str, Any], **_: Any) -> Any:
        _validate_banks(token_banks, dimensions=self.bank_dimensions)
        pooled = [
            self.projections[name](token_banks[name]).mean(dim=1)
            for name in EXPERT_ORDER
        ]
        return self.classifier(_require_torch().cat(pooled, dim=-1))


class FusionTransformerBlock(torch.nn.Module if torch is not None else object):
    def __init__(self) -> None:
        module = _require_torch()
        super().__init__()
        self.norm1 = RMSNorm(128)
        self.attention = module.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.norm2 = RMSNorm(128)
        self.mlp = module.nn.Sequential(
            module.nn.Linear(128, 512),
            module.nn.GELU(),
            module.nn.Dropout(0.1),
            module.nn.Linear(512, 128),
            module.nn.Dropout(0.1),
        )

    def forward(self, values: Any) -> Any:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        values = values + attended
        return values + self.mlp(self.norm2(values))


class TokenTransformerFusion(torch.nn.Module if torch is not None else object):
    def __init__(self, *, bank_dimensions: Mapping[str, int]) -> None:
        module = _require_torch()
        super().__init__()
        if set(bank_dimensions) != set(EXPERT_ORDER):
            raise ValueError("fusion dimensions differ from canonical experts")
        self.bank_dimensions = {name: int(bank_dimensions[name]) for name in EXPERT_ORDER}
        self.projections = module.nn.ModuleDict(
            {
                name: BankProjection(self.bank_dimensions[name])
                for name in EXPERT_ORDER
            }
        )
        self.expert_embedding = module.nn.Embedding(7, 128)
        self.slot_embedding = module.nn.Embedding(16, 128)
        self.source_embedding = module.nn.Parameter(module.zeros(128))
        self.class_token = module.nn.Parameter(module.zeros(1, 1, 128))
        module.nn.init.normal_(self.class_token, std=0.02)
        self.blocks = module.nn.ModuleList(
            [FusionTransformerBlock() for _ in range(3)]
        )
        self.norm = RMSNorm(128)
        self.classifier = module.nn.Linear(128, 10)
        self.whole_bank_dropout_probability = 0.0

    def forward(
        self,
        *,
        token_banks: Mapping[str, Any],
        return_details: bool = False,
        **_: Any,
    ) -> Any:
        batch, device = _validate_banks(
            token_banks, dimensions=self.bank_dimensions
        )
        rows = []
        for expert_index, name in enumerate(EXPERT_ORDER):
            values = self.projections[name](token_banks[name])
            slots = int(values.shape[1])
            if slots > 16:
                raise ValueError("fusion bank exceeds registered slot count")
            slot_ids = _require_torch().arange(slots, device=device)
            values = (
                values
                + self.expert_embedding.weight[expert_index].view(1, 1, -1)
                + self.slot_embedding(slot_ids).view(1, slots, -1)
                + self.source_embedding.view(1, 1, -1)
            )
            rows.append(values)
        sequence = _require_torch().cat(rows, dim=1)
        class_token = self.class_token.expand(batch, -1, -1)
        sequence = _require_torch().cat((class_token, sequence), dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        logits = self.classifier(self.norm(sequence[:, 0]))
        if not bool(_require_torch().isfinite(logits).all()):
            raise FloatingPointError("offline fusion logits are nonfinite")
        if return_details:
            return {"logits": logits, "sequence": sequence}
        return logits


def build_fusion_model(
    variant: str,
    *,
    bank_dimensions: Mapping[str, int],
    best_single_expert: str | None = None,
) -> Any:
    if variant == "F_BEST_SINGLE":
        if best_single_expert is None:
            raise ValueError("F_BEST_SINGLE requires its val_stop-selected expert")
        return BestSingleFusion(expert_id=best_single_expert)
    if variant == "F_UNIFORM_LOGIT_MEAN":
        return UniformLogitMean()
    if variant == "F_TRAINED_LOGIT_LINEAR":
        return TrainedLogitLinear()
    if variant == "F_POOLED_MLP":
        return PooledMLPFusion(bank_dimensions=bank_dimensions)
    if variant in {
        "F_TOKEN_TRANSFORMER",
        "F_TOKEN_TRANSFORMER_LIGHT_FINETUNE",
        "F_TOKEN_TRANSFORMER_FULL_FINETUNE",
    }:
        return TokenTransformerFusion(bank_dimensions=bank_dimensions)
    raise ValueError(f"fusion variant {variant!r} requires a separate control path")


class LiveExpertFusion(torch.nn.Module if torch is not None else object):
    """Run live experts and force the fusion decision through their token banks."""

    def __init__(
        self,
        *,
        experts: Mapping[str, Any],
        fusion: Any,
        variant: str,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if set(experts) != set(EXPERT_ORDER):
            raise ValueError("live fusion experts differ from canonical order")
        self.experts = module.nn.ModuleDict(
            {name: experts[name] for name in EXPERT_ORDER}
        )
        self.fusion = fusion
        self.variant = str(variant)
        self.trainability = configure_expert_trainability(
            self.experts, variant=variant
        )

    def forward(self, **particle_batch: Any) -> Any:
        token_banks = {}
        expert_logits = {}
        for name in EXPERT_ORDER:
            output = self.experts[name](
                return_details=True, **particle_batch
            )
            if not isinstance(output, Mapping) or set(("tokens", "logits")) - set(output):
                raise ValueError("live expert did not expose tokens and logits")
            token_banks[name] = output["tokens"]
            expert_logits[name] = output["logits"]
        return self.fusion(
            token_banks=token_banks,
            expert_logits=expert_logits,
        )


class RelationAuxiliaryHead(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        *,
        token_dimension: int,
        summary_dimension: int,
    ) -> None:
        super().__init__()
        self.projection = _require_torch().nn.Linear(
            int(token_dimension), int(summary_dimension)
        )

    def forward(self, tokens: Any) -> Any:
        if tokens.ndim != 3:
            raise ValueError("relation auxiliary head requires [B,K,D] tokens")
        return self.projection(tokens.mean(dim=1))


def masked_relation_auxiliary_loss(
    prediction: Any,
    target: Any,
    valid_mask: Any,
) -> Any:
    module = _require_torch()
    if prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("relation auxiliary target/mask shape differs")
    selected = valid_mask.bool()
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    values = (prediction - target).square()[selected]
    if not bool(module.isfinite(values).all()):
        raise FloatingPointError("relation auxiliary loss is nonfinite")
    return values.mean()


def configure_expert_trainability(
    experts: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, list[str]]:
    if set(experts) != set(EXPERT_ORDER):
        raise ValueError("expert modules differ from canonical order")
    trainable = []
    frozen = []
    for expert_name in EXPERT_ORDER:
        expert = experts[expert_name]
        for name, parameter in expert.named_parameters():
            allowed = (
                variant == "F_TOKEN_TRANSFORMER_FULL_FINETUNE"
                or variant == "F_TOKEN_TRANSFORMER_LIGHT_FINETUNE"
                and name.startswith("tokenizer.")
            )
            parameter.requires_grad_(allowed)
            (trainable if allowed else frozen).append(f"{expert_name}.{name}")
    if variant == "F_TOKEN_TRANSFORMER" and trainable:
        raise RuntimeError("frozen fusion unexpectedly enabled expert parameters")
    return {"trainable": trainable, "frozen": frozen}


def fusion_parameter_groups(
    model: LiveExpertFusion,
    *,
    fusion_learning_rate: float,
) -> list[dict[str, Any]]:
    learning_rate = float(fusion_learning_rate)
    if learning_rate <= 0:
        raise ValueError("fusion learning rate must be positive")
    fusion_parameters = [
        parameter
        for parameter in model.fusion.parameters()
        if parameter.requires_grad
    ]
    groups = [{"params": fusion_parameters, "lr": learning_rate}]
    expert_parameters = [
        parameter
        for expert in model.experts.values()
        for parameter in expert.parameters()
        if parameter.requires_grad
    ]
    if expert_parameters:
        groups.append(
            {
                "params": expert_parameters,
                "lr": (
                    learning_rate * 0.1
                    if model.variant
                    == "F_TOKEN_TRANSFORMER_LIGHT_FINETUNE"
                    else learning_rate
                ),
            }
        )
    return groups


class GroupedHeadRelationBias(torch.nn.Module if torch is not None else object):
    """Assemble preencoded relation biases into fixed disjoint head groups."""

    _relations = ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")

    def forward(
        self,
        *,
        base4_bias: Any,
        relation_biases: Mapping[str, Any],
    ) -> Any:
        module = _require_torch()
        if base4_bias.ndim != 4 or int(base4_bias.shape[1]) not in {1, 2, 8}:
            raise ValueError("base4 grouped bias has the wrong shape")
        if set(relation_biases) != set(self._relations):
            raise ValueError("grouped relation biases differ")
        base_head = base4_bias[:, 0]
        heads = [base_head, base4_bias[:, min(1, base4_bias.shape[1] - 1)]]
        for name in self._relations:
            value = relation_biases[name]
            if value.ndim != 4 or int(value.shape[1]) < 1:
                raise ValueError("relation grouped bias has the wrong shape")
            if tuple(value.shape[0:1] + value.shape[2:]) != tuple(
                base4_bias.shape[0:1] + base4_bias.shape[2:]
            ):
                raise ValueError("grouped relation pair shapes differ")
            heads.append(base_head + value[:, 0])
        output = module.stack(heads, dim=1)
        if int(output.shape[1]) != 8:
            raise RuntimeError("grouped relation control did not emit eight heads")
        return output


def cross_covariance_penalty(token_banks: Mapping[str, Any]) -> Any:
    module = _require_torch()
    if set(token_banks) != set(EXPERT_ORDER):
        raise ValueError("cross-covariance banks differ")
    representations = []
    batch = None
    for name in EXPERT_ORDER:
        values = token_banks[name].flatten(1)
        if batch is None:
            batch = int(values.shape[0])
        if int(values.shape[0]) != batch:
            raise ValueError("cross-covariance batch differs")
        values = values - values.mean(dim=0, keepdim=True)
        values = values / values.square().mean().sqrt().clamp_min(1.0e-8)
        representations.append(values)
    if int(batch) < 2:
        raise ValueError("cross-covariance requires at least two events")
    penalties = []
    for left in range(7):
        for right in range(left + 1, 7):
            covariance = (
                representations[left].transpose(0, 1)
                @ representations[right]
                / (batch - 1)
            )
            penalties.append(covariance.square().mean())
    return module.stack(penalties).mean()


__all__ = [
    "FUSION_VARIANTS",
    "GROUPED_HEAD_RELATION_CONTRACT",
    "OFFLINE_FUSION_ARCHITECTURE_CONTRACT",
    "RELATION_SPECIALIZATION_CONTRACT",
    "GroupedHeadRelationBias",
    "BestSingleFusion",
    "LiveExpertFusion",
    "PooledMLPFusion",
    "RelationAuxiliaryHead",
    "TokenTransformerFusion",
    "TrainedLogitLinear",
    "UniformLogitMean",
    "build_fusion_model",
    "build_grouped_head_relation_contract",
    "build_offline_fusion_architecture_contract",
    "build_relation_specialization_contract",
    "configure_expert_trainability",
    "cross_covariance_penalty",
    "fusion_parameter_groups",
    "masked_relation_auxiliary_loss",
    "validate_offline_fusion_architecture_contract",
]
