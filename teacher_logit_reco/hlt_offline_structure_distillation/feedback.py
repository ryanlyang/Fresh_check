"""Stage-E predicted-structure feedback graphs and deterministic registry."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .auxiliary import (
    GLOBAL_PHYSICAL_TARGETS,
    PAIR_TARGETS,
    global_auxiliary_loss,
    pair_auxiliary_loss,
    select_utility_row,
)
from .baselines import component_seed
from .contracts import (
    FEEDBACK_INTERFACE_CONTRACT,
    FEEDBACK_RESULT_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_E_PLAN_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .heads import GlobalTargetHead, PairTargetHead
from .taps import HBaseParticleTransformer
from .target_schemas import target_declarations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


MANDATORY_FEEDBACK = (
    ("T_OFFLINE_TRACK_32", "FB_TOKEN"),
    ("T_OFFLINE_TRACK_32", "FB_FILM"),
    ("T_HLT_TRACK_PAIR_13", "FB_PAIR"),
    ("T_HLT_REGION_PAIR_8", "FB_PAIR"),
)
GLOBAL_CONTROLS = (
    "ZERO",
    "DISABLED_LOSS",
    "SHUFFLED_PREDICTION",
    "UNRESTRICTED",
    "MEAN_ONLY",
    "ORACLE_SUB",
    "ORACLE_TRAINED",
)
PAIR_CONTROLS = (
    "AUX_ONLY",
    "ZERO_GATE",
    "DETACHED",
    "NO_SEMANTIC_LOSS",
    "SHUFFLED",
    "UNRESTRICTED_MLP",
    "EXACT_HLT",
)


def feedback_interface_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": FEEDBACK_INTERFACE_CONTRACT,
            "schema_version": 1,
            "tap": "post_block_4",
            "consumer_blocks": [5, 6, 7, 8],
            "token": {
                "slots": 4,
                "dimension": 128,
                "separate_branch": True,
                "live_particle_sequence_expanded": False,
                "gate": "2*tanh(raw_gamma)",
                "raw_gate_initialization": 0.0,
                "output_projection_bias": 0.0,
            },
            "film": {
                "scale": "1+0.1*tanh(s)",
                "shift": "0.1*tanh(b)",
                "final_projection_initialization": 0.0,
            },
            "pair": {
                "gate": "2*tanh(raw_alpha_per_head)",
                "raw_gate_initialization": 0.0,
                "warmup": "min(T,max(1,ceil(0.05*T)))",
                "invalid_and_diagonal": 0.0,
                "offline_pair_feedback_allowed": False,
            },
            "gradient_paths": ["END_TO_END", "DETACHED", "AUX_ONLY"],
            "probabilistic_feedback": {
                "heteroscedastic": "mean_and_clipped_log_variance",
                "categorical": "complete_probability_vector",
                "sampling_primary": False,
            },
            "oracle_primary_or_export_allowed": False,
        }
    )


def gate_warmup_updates(total_updates: int) -> int:
    total = int(total_updates)
    if total <= 0:
        raise ValueError("feedback gate schedule requires positive total updates")
    return min(total, max(1, math.ceil(0.05 * total)))


class ResidualStructureTokenAdapter(
    torch.nn.Module if torch is not None else object
):
    """Four structure tokens cross-attend into particles without joining them."""

    def __init__(
        self,
        target_dimension: int,
        *,
        particle_dimension: int = 128,
        heteroscedastic: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        source_dimension = int(target_dimension) * (2 if heteroscedastic else 1)
        self.heteroscedastic = bool(heteroscedastic)
        self.particle_norm = torch.nn.RMSNorm(particle_dimension)
        self.query_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(particle_dimension, 128, bias=False)
        )
        self.structure_projection = torch.nn.Linear(source_dimension, 4 * 128)
        self.family_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.parameterization_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.source_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.slot_embedding = torch.nn.Parameter(torch.empty(1, 4, 128))
        torch.nn.init.normal_(self.slot_embedding, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.output_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(128, particle_dimension, bias=True)
        )
        if isinstance(self.output_projection, torch.nn.Linear):
            torch.nn.init.zeros_(self.output_projection.bias)
        self.raw_gamma = torch.nn.Parameter(torch.zeros(()))

    @property
    def gamma(self) -> Any:
        return 2.0 * torch.tanh(self.raw_gamma)

    def structure_tokens(self, prediction: Mapping[str, Any]) -> Any:
        mean = prediction.get("mean", prediction["value"])
        source = (
            torch.cat((mean, prediction["log_variance"].clamp(-8.0, 5.0)), dim=-1)
            if self.heteroscedastic
            else mean
        )
        tokens = self.structure_projection(source).reshape(-1, 4, 128)
        return (
            tokens
            + self.family_embedding
            + self.parameterization_embedding
            + self.source_embedding
            + self.slot_embedding
        )

    def forward(
        self, particles: Any, particle_mask: Any, prediction: Mapping[str, Any]
    ) -> Any:
        tokens = self.structure_tokens(prediction)
        query = self.query_projection(self.particle_norm(particles))
        update, _ = self.cross_attention(
            query, tokens, tokens, need_weights=False
        )
        update = self.output_projection(update)
        update = update.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)
        return particles + self.gamma * update


class DirectFourTokenHead(torch.nn.Module if torch is not None else object):
    """Unrestricted four-query trunk whose output is four latent tokens.

    This is deliberately not a structure predictor.  It retains the registered
    global head's four-query masked cross-attention, but emits one learned
    128-dimensional token per query instead of projecting to a named target.
    """

    def __init__(self, *, input_dimension: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.input_projection = (
            torch.nn.Identity()
            if input_dimension == 128
            else torch.nn.Linear(input_dimension, 128)
        )
        self.input_norm = torch.nn.RMSNorm(128)
        self.queries = torch.nn.Parameter(torch.empty(4, 128))
        torch.nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.direct_output = torch.nn.Linear(128, 128)

    def forward(self, particle_states: Any, particle_mask: Any) -> Any:
        if particle_states.ndim != 3 or particle_mask.ndim != 2:
            raise ValueError("unrestricted token head expects [B,N,D] and [B,N]")
        if tuple(particle_states.shape[:2]) != tuple(particle_mask.shape):
            raise ValueError("unrestricted token state and mask shapes differ")
        if bool((particle_mask.bool().sum(dim=1) == 0).any()):
            raise ValueError("unrestricted token head cannot attend to an empty event")
        state = self.input_norm(self.input_projection(particle_states))
        query = self.queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        tokens, _ = self.cross_attention(
            query,
            state,
            state,
            key_padding_mask=~particle_mask.bool(),
            need_weights=False,
        )
        return self.direct_output(tokens)


class DirectStructureTokenAdapter(
    torch.nn.Module if torch is not None else object
):
    """Consume unrestricted tokens through the semantic adapter's same readout."""

    def __init__(self, *, particle_dimension: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.particle_norm = torch.nn.RMSNorm(particle_dimension)
        self.query_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(particle_dimension, 128, bias=False)
        )
        self.family_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.parameterization_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.source_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.slot_embedding = torch.nn.Parameter(torch.empty(1, 4, 128))
        torch.nn.init.normal_(self.slot_embedding, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.output_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(128, particle_dimension, bias=True)
        )
        if isinstance(self.output_projection, torch.nn.Linear):
            torch.nn.init.zeros_(self.output_projection.bias)
        self.raw_gamma = torch.nn.Parameter(torch.zeros(()))

    @property
    def gamma(self) -> Any:
        return 2.0 * torch.tanh(self.raw_gamma)

    def forward(self, particles: Any, particle_mask: Any, tokens: Any) -> Any:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (4, 128):
            raise ValueError("unrestricted feedback must provide [B,4,128] tokens")
        tokens = (
            tokens
            + self.family_embedding
            + self.parameterization_embedding
            + self.source_embedding
            + self.slot_embedding
        )
        query = self.query_projection(self.particle_norm(particles))
        update, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        update = self.output_projection(update)
        update = update.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)
        return particles + self.gamma * update


class DirectTokenFeaturewiseConditioning(
    torch.nn.Module if torch is not None else object
):
    """Map four unrestricted latent tokens to the registered bounded FiLM path."""

    def __init__(self, *, particle_dimension: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.projection = torch.nn.Linear(4 * 128, 2 * particle_dimension)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)

    def parameters_for(self, tokens: Any) -> tuple[Any, Any]:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (4, 128):
            raise ValueError("unrestricted FiLM must receive [B,4,128] tokens")
        scale_raw, shift_raw = self.projection(tokens.flatten(1)).chunk(2, dim=-1)
        return 1.0 + 0.1 * torch.tanh(scale_raw), 0.1 * torch.tanh(shift_raw)

    def forward(self, particles: Any, particle_mask: Any, tokens: Any) -> Any:
        scale, shift = self.parameters_for(tokens)
        result = particles * scale.unsqueeze(1) + shift.unsqueeze(1)
        return result.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)


class TrainableParameterPadding(torch.nn.Module if torch is not None else object):
    """Exact, inert trainable-parameter ledger for matched-capacity controls."""

    def __init__(self, count: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        if int(count) < 0:
            raise ValueError("matched-capacity padding cannot be negative")
        self.count = int(count)
        self.padding = torch.nn.Parameter(torch.zeros(self.count))

    def inert_scalar(self) -> Any:
        # Retains the parameter in the autograd graph without changing logits.
        return self.padding.sum() * 0.0


class BoundedFeaturewiseConditioning(
    torch.nn.Module if torch is not None else object
):
    def __init__(
        self,
        target_dimension: int,
        *,
        particle_dimension: int = 128,
        heteroscedastic: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        source_dimension = int(target_dimension) * (2 if heteroscedastic else 1)
        self.heteroscedastic = bool(heteroscedastic)
        self.projection = torch.nn.Linear(source_dimension, 2 * particle_dimension)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)

    def parameters_for(self, prediction: Mapping[str, Any]) -> tuple[Any, Any]:
        mean = prediction.get("mean", prediction["value"])
        source = (
            torch.cat((mean, prediction["log_variance"].clamp(-8.0, 5.0)), dim=-1)
            if self.heteroscedastic
            else mean
        )
        scale_raw, shift_raw = self.projection(source).chunk(2, dim=-1)
        return 1.0 + 0.1 * torch.tanh(scale_raw), 0.1 * torch.tanh(shift_raw)

    def forward(
        self, particles: Any, particle_mask: Any, prediction: Mapping[str, Any]
    ) -> Any:
        scale, shift = self.parameters_for(prediction)
        result = particles * scale.unsqueeze(1) + shift.unsqueeze(1)
        return result.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)


class PredictedPairAttentionBias(
    torch.nn.Module if torch is not None else object
):
    def __init__(
        self,
        *,
        input_dimension: int,
        pair_dimension: int,
        attention_heads: int = 8,
        symmetric: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.symmetric = bool(symmetric)
        self.predictor = PairTargetHead(
            input_dimension, pair_dimension, symmetric=symmetric
        )
        self.bias_network = torch.nn.Sequential(
            torch.nn.Linear(pair_dimension, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, attention_heads),
        )
        self.raw_alpha = torch.nn.Parameter(torch.zeros(attention_heads))
        self.total_updates = 1
        self.update_ordinal = 0

    @property
    def alpha(self) -> Any:
        if self.update_ordinal <= gate_warmup_updates(self.total_updates):
            return self.raw_alpha * 0.0
        return 2.0 * torch.tanh(self.raw_alpha)

    def set_update(self, update_ordinal: int, total_updates: int) -> None:
        if not 0 <= int(update_ordinal) <= int(total_updates):
            raise ValueError("feedback update ordinal lies outside schedule")
        self.update_ordinal = int(update_ordinal)
        self.total_updates = int(total_updates)

    def forward(
        self,
        states: Any,
        mask: Any,
        *,
        direct_features: Any | None = None,
        detach_consumer: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        predicted, pair_mask = self.predictor(states, mask)
        consumed = predicted if direct_features is None else direct_features
        if detach_consumer:
            consumed = consumed.detach()
        raw = self.bias_network(consumed).permute(0, 3, 1, 2)
        bias = torch.tanh(raw) * self.alpha.view(1, -1, 1, 1)
        if self.symmetric:
            bias = 0.5 * (bias + bias.transpose(2, 3))
        valid = pair_mask.unsqueeze(1)
        bias = bias.masked_fill(~valid, 0)
        return bias, {"value": predicted, "pair_mask": pair_mask}


def _target_dimension(target_id: str) -> int:
    rows = {row.target_id: row for row in target_declarations()}
    if target_id not in rows:
        raise ValueError(f"unknown feedback target {target_id}")
    return len(rows[target_id].components)


def build_feedback_model(
    row: Mapping[str, Any],
    *,
    weaver_module: Any | None = None,
    particle_dimension: int = 128,
) -> "FeedbackHBaseClassifier":
    if str(row.get("target_id")) not in {
        declaration.target_id for declaration in target_declarations()
    }:
        raise ValueError("feedback row target is not registered")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(row["encoder_component_seed"]))
        classifier = HBaseParticleTransformer(weaver_module=weaver_module)
        torch.manual_seed(int(row["feedback_component_seed"]))
        return FeedbackHBaseClassifier(
            classifier,
            target_id=str(row["target_id"]),
            interface=str(row["interface"]),
            gradient_path=str(row["gradient_path"]),
            parameterization=str(row["parameterization"]),
            particle_dimension=particle_dimension,
            control=row.get("control"),
            allow_oracle=row.get("control") in {"ORACLE_SUB", "ORACLE_TRAINED"},
        )


class FeedbackHBaseClassifier(torch.nn.Module if torch is not None else object):
    """H_BASE with one explicitly registered HLT-predicted feedback interface."""

    def __init__(
        self,
        classifier: HBaseParticleTransformer,
        *,
        target_id: str,
        interface: str,
        gradient_path: str = "END_TO_END",
        parameterization: str = "ABS",
        particle_dimension: int = 128,
        attention_heads: int = 8,
        control: str | None = None,
        allow_oracle: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        if interface not in {"FB_TOKEN", "FB_FILM", "FB_PAIR"}:
            raise ValueError("unknown feedback interface")
        if gradient_path not in {"END_TO_END", "DETACHED", "AUX_ONLY"}:
            raise ValueError("unknown feedback gradient path")
        if interface == "FB_PAIR" and target_id not in PAIR_TARGETS:
            raise ValueError("pair feedback is restricted to registered HLT targets")
        if interface != "FB_PAIR" and target_id in PAIR_TARGETS:
            raise ValueError("global feedback cannot consume a pair target")
        if control in {"ORACLE_SUB", "ORACLE_TRAINED"} and not allow_oracle:
            raise ValueError("oracle feedback requires an explicit diagnostic graph")
        self.classifier = classifier
        self.target_id = str(target_id)
        self.interface = str(interface)
        self.gradient_path = str(gradient_path)
        self.parameterization = str(parameterization)
        self.control = control
        self.allow_oracle = bool(allow_oracle)
        self.capacity_padding = None
        self.capacity_ledger = None
        dimension = _target_dimension(target_id)
        heteroscedastic = parameterization == "HET" and control != "MEAN_ONLY"
        if interface == "FB_PAIR":
            self.global_predictor = None
            self.consumer = PredictedPairAttentionBias(
                input_dimension=particle_dimension,
                pair_dimension=dimension,
                attention_heads=attention_heads,
                symmetric=True,
            )
            if control == "ZERO_GATE":
                self.consumer.raw_alpha.requires_grad_(False)
        else:
            if control == "UNRESTRICTED":
                self.global_predictor = DirectFourTokenHead(
                    input_dimension=particle_dimension
                )
                self.consumer = (
                    DirectStructureTokenAdapter(
                        particle_dimension=particle_dimension
                    )
                    if interface == "FB_TOKEN"
                    else DirectTokenFeaturewiseConditioning(
                        particle_dimension=particle_dimension
                    )
                )
                # The reference branch is instantiated only to derive its exact
                # active tensor count.  fork_rng prevents this accounting step
                # from changing the unrestricted branch initialization stream.
                devices = (
                    list(range(torch.cuda.device_count()))
                    if torch.cuda.is_available()
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    reference_predictor = GlobalTargetHead(
                        dimension,
                        input_dimension=particle_dimension,
                        heteroscedastic=parameterization == "HET",
                    )
                    reference_consumer_type = (
                        ResidualStructureTokenAdapter
                        if interface == "FB_TOKEN"
                        else BoundedFeaturewiseConditioning
                    )
                    reference_consumer = reference_consumer_type(
                        dimension,
                        particle_dimension=particle_dimension,
                        heteroscedastic=heteroscedastic,
                    )
                reference_count = sum(
                    value.numel()
                    for module in (reference_predictor, reference_consumer)
                    for value in module.parameters()
                    if value.requires_grad
                )
                unrestricted_count = sum(
                    value.numel()
                    for module in (self.global_predictor, self.consumer)
                    for value in module.parameters()
                    if value.requires_grad
                )
                difference = reference_count - unrestricted_count
                self.capacity_padding = TrainableParameterPadding(difference)
                self.capacity_ledger = {
                    "contract": "hosd_unrestricted_feedback_capacity_ledger_v1",
                    "reference_trainable_parameters": reference_count,
                    "unrestricted_pre_padding_trainable_parameters": (
                        unrestricted_count
                    ),
                    "inert_trainable_padding_parameters": difference,
                    "matched_trainable_parameters": reference_count,
                }
            else:
                self.global_predictor = GlobalTargetHead(
                    dimension,
                    input_dimension=particle_dimension,
                    heteroscedastic=parameterization == "HET",
                )
                consumer_type = (
                    ResidualStructureTokenAdapter
                    if interface == "FB_TOKEN"
                    else BoundedFeaturewiseConditioning
                )
                self.consumer = consumer_type(
                    dimension,
                    particle_dimension=particle_dimension,
                    heteroscedastic=heteroscedastic,
                )
            if control in {"ZERO", "ZERO_GATE"} and hasattr(
                self.consumer, "raw_gamma"
            ):
                self.consumer.raw_gamma.requires_grad_(False)

    def set_update(self, update_ordinal: int, total_updates: int) -> None:
        if isinstance(self.consumer, PredictedPairAttentionBias):
            self.consumer.set_update(update_ordinal, total_updates)

    def shared_parameters(self) -> tuple[Any, ...]:
        return tuple(self.classifier.parameters())

    def head_parameters(self) -> tuple[Any, ...]:
        classifier_ids = {id(value) for value in self.classifier.parameters()}
        return tuple(
            value for value in self.parameters() if id(value) not in classifier_ids
        )

    def forward_with_feedback(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        direct_pair_features: Any | None = None,
        oracle_feedback: Mapping[str, Any] | None = None,
        predicted_feedback_override: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        if oracle_feedback is not None and not self.allow_oracle:
            raise ValueError("primary/deployable feedback cannot consume oracle data")
        prediction_box: dict[str, Any] = {}
        detach = self.gradient_path == "DETACHED" or self.control in {
            "DETACHED",
            "SHUFFLED_PREDICTION",
            "SHUFFLED",
        }
        active = self.gradient_path != "AUX_ONLY" and self.control != "AUX_ONLY"

        def predict_global(state: Any, active_mask: Any) -> Mapping[str, Any]:
            if self.control == "UNRESTRICTED":
                tokens = self.global_predictor(state, active_mask)
                predicted = {
                    "tokens": tokens,
                    # Semantic loss is globally fixed to zero for this row.
                    # These shape-compatible inert values keep the shared
                    # training/evaluation worker contract uniform.
                    "value": tokens.new_zeros(tokens.shape[0], _target_dimension(
                        self.target_id
                    )),
                    "availability_logits": tokens.new_zeros(tokens.shape[0], 1),
                }
            else:
                predicted = self.global_predictor(state, active_mask)
            prediction_box.update(predicted)
            consumed = (
                oracle_feedback
                if oracle_feedback is not None
                else predicted_feedback_override
                if predicted_feedback_override is not None
                else predicted
            )
            if detach:
                consumed = {
                    key: value.detach() if hasattr(value, "detach") else value
                    for key, value in consumed.items()
                }
            return consumed

        if self.interface == "FB_TOKEN":
            def token_transform(state: Any, active_mask: Any) -> Any:
                consumed = predict_global(state, active_mask)
                if not active:
                    return state
                if self.control == "UNRESTRICTED":
                    result = self.consumer(state, active_mask, consumed["tokens"])
                    if self.capacity_padding is not None:
                        result = result + self.capacity_padding.inert_scalar()
                    return result
                return self.consumer(state, active_mask, consumed)

            result = self.classifier.forward_with_taps(
                points,
                features,
                lorentz_vectors,
                mask,
                capture=("TAP_MID",),
                post_mid_transform=token_transform,
            )
        elif self.interface == "FB_FILM":
            consumed_box: dict[str, Mapping[str, Any]] = {}

            def record(state: Any, active_mask: Any) -> Any:
                consumed_box["value"] = predict_global(state, active_mask)
                return state

            def film_transform(_block: int, state: Any, active_mask: Any) -> Any:
                return (
                    self.consumer(
                        state,
                        active_mask,
                        consumed_box["value"]["tokens"]
                        if self.control == "UNRESTRICTED"
                        else consumed_box["value"],
                    )
                    if active
                    else state
                )

            result = self.classifier.forward_with_taps(
                points,
                features,
                lorentz_vectors,
                mask,
                capture=("TAP_MID",),
                post_mid_transform=record,
                later_block_transform=film_transform,
            )
        else:
            def pair_bias(state: Any, active_mask: Any) -> Any:
                direct = (
                    direct_pair_features
                    if self.control == "EXACT_HLT"
                    else predicted_feedback_override["value"]
                    if predicted_feedback_override is not None
                    else None
                )
                bias, predicted = self.consumer(
                    state,
                    active_mask,
                    direct_features=direct,
                    detach_consumer=detach,
                )
                prediction_box.update(predicted)
                return bias if active else bias * 0

            result = self.classifier.forward_with_taps(
                points,
                features,
                lorentz_vectors,
                mask,
                capture=("TAP_MID",),
                later_pair_bias=pair_bias,
            )
        if not prediction_box:
            raise RuntimeError("feedback predictor did not execute")
        return result.logits, prediction_box

    def forward(self, points: Any, features: Any, lorentz_vectors: Any, mask: Any):
        logits, _ = self.forward_with_feedback(
            points, features, lorentz_vectors, mask
        )
        return logits

    def forward_with_aux(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        sampled_pair_indices: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        logits, prediction = self.forward_with_feedback(
            points, features, lorentz_vectors, mask
        )
        if sampled_pair_indices is not None and self.target_id in PAIR_TARGETS:
            index = (
                sampled_pair_indices["event_indices"].long(),
                sampled_pair_indices["left_indices"].long(),
                sampled_pair_indices["right_indices"].long(),
            )
            prediction = {
                **prediction,
                "value": prediction["value"][index],
            }
        return logits, prediction


def _feedback_row(
    *,
    target_id: str,
    interface: str,
    gradient_path: str,
    parameterization: str,
    auxiliary_weight: float,
    row_kind: str,
    control: str | None = None,
) -> dict[str, Any]:
    semantics = {
        "target_id": target_id,
        "interface": interface,
        "gradient_path": gradient_path,
        "parameterization": parameterization,
        "auxiliary_weight": float(auxiliary_weight),
        "row_kind": row_kind,
        "control": control,
    }
    row_id = f"FB_{canonical_sha256(semantics)[:16]}"
    return {
        "row_id": row_id,
        **semantics,
        "pipeline_seed": 101,
        "encoder_component_seed": component_seed(101, "encoder", "H_BASE"),
        "feedback_component_seed": component_seed(101, "feedback", row_id),
        "head_type": "pair" if interface == "FB_PAIR" else "global",
        "resolved": True,
        "selection_eligible": row_kind == "SCIENTIFIC",
        "deployable": control not in {"ORACLE_SUB", "ORACLE_TRAINED"},
        "fixed_epoch_budget": 40,
        "performance_can_omit_or_cancel": False,
    }


def build_stage_e_plan(
    *,
    single_family_selection: Mapping[str, Any],
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        single_family_selection,
        expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
    )
    selected = dict(single_family_selection["selected_row_by_target"])
    required = {
        "T_OFFLINE_TRACK_32",
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    }
    if not required.issubset(selected):
        raise ValueError("Stage-E lock lacks mandatory feedback targets")
    order = [
        str(row["target_id"])
        for row in single_family_selection.get("cross_family_order", ())
        if str(row["target_id"]) in GLOBAL_PHYSICAL_TARGETS
        and str(row["target_id"]) != "T_OFFLINE_TRACK_32"
    ]
    if not order:
        # Older Step-6 locks expose only the selected mapping. Its insertion
        # order is canonical selection order and remains deterministic.
        order = [
            target_id
            for target_id in selected
            if target_id in GLOBAL_PHYSICAL_TARGETS
            and target_id != "T_OFFLINE_TRACK_32"
        ]
    promoted = order[:2]
    base = [*MANDATORY_FEEDBACK]
    for target_id in promoted:
        base.extend(((target_id, "FB_TOKEN"), (target_id, "FB_FILM")))
    scientific = []
    for target_id, interface in base:
        parameterization = str(
            single_family_selection.get("selected_definition_by_target", {})
            .get(target_id, {})
            .get("parameterization", "ABS")
        )
        weight = float(
            single_family_selection.get("selected_definition_by_target", {})
            .get(target_id, {})
            .get("auxiliary_weight", 0.30)
        )
        for path in ("END_TO_END", "DETACHED"):
            scientific.append(
                _feedback_row(
                    target_id=target_id,
                    interface=interface,
                    gradient_path=path,
                    parameterization=parameterization,
                    auxiliary_weight=weight,
                    row_kind="SCIENTIFIC",
                )
            )
    controls = []
    for target_id, interface in MANDATORY_FEEDBACK:
        names = PAIR_CONTROLS if interface == "FB_PAIR" else GLOBAL_CONTROLS
        for control in names:
            controls.append(
                _feedback_row(
                    target_id=target_id,
                    interface=interface,
                    gradient_path=(
                        "AUX_ONLY"
                        if control == "AUX_ONLY"
                        else "DETACHED"
                        if control == "DETACHED"
                        else "END_TO_END"
                    ),
                    parameterization=(
                        "HET"
                        if control == "MEAN_ONLY"
                        else "ABS"
                    ),
                    auxiliary_weight=0.0
                    if control
                    in {"DISABLED_LOSS", "NO_SEMANTIC_LOSS", "UNRESTRICTED",
                        "UNRESTRICTED_MLP"}
                    else 0.30,
                    row_kind="CONTROL",
                    control=control,
                )
            )
    rows = scientific + controls
    if len(base) > 8 or len(scientific) > 16 or len(controls) > 30 or len(rows) > 46:
        raise AssertionError("Stage-E matrix exceeds its immutable bound")
    return with_content_hash(
        {
            "contract": STAGE_E_PLAN_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "single_family_selection_sha256": single_family_selection[
                "content_hash"
            ],
            "promoted_global_targets": promoted,
            "scientific_rows": scientific,
            "control_rows": controls,
            "all_rows": rows,
            "row_count": len(rows),
            "hard_maximum": 46,
            "scientific_row_maximum": 16,
            "control_row_maximum": 30,
            "performance_can_cancel_or_omit": False,
        }
    )


def build_feedback_selection(
    *,
    stage_e_plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(stage_e_plan, expected_contract=STAGE_E_PLAN_CONTRACT)
    by_id = {str(row["row_id"]): row for row in results}
    required = {str(row["row_id"]) for row in stage_e_plan["all_rows"]}
    if set(by_id) != required:
        raise ValueError("feedback selection requires complete Stage-E coverage")
    for result in by_id.values():
        validate_content_hash(result, expected_contract=FEEDBACK_RESULT_CONTRACT)
        if result.get("source") != dict(source):
            raise ValueError("feedback result source differs")
    scientific = [
        by_id[row["row_id"]] for row in stage_e_plan["scientific_rows"]
    ]
    deployable = [
        row
        for row in scientific
        if bool(row.get("selection_eligible")) and bool(row.get("deployable"))
    ]
    winner = select_utility_row(deployable)
    by_interface = {}
    for interface in ("FB_TOKEN", "FB_FILM", "FB_PAIR"):
        candidates = [row for row in deployable if row["interface"] == interface]
        if candidates:
            by_interface[interface] = select_utility_row(candidates)["row_id"]
    return with_content_hash(
        {
            "contract": FEEDBACK_SELECTION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "stage_e_plan_sha256": stage_e_plan["content_hash"],
            "result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "selected_feedback_row_id": winner["row_id"],
            "selected_feedback_definition": {
                key: winner[key]
                for key in (
                    "row_id",
                    "target_id",
                    "interface",
                    "gradient_path",
                    "parameterization",
                    "auxiliary_weight",
                    "control",
                    "deployable",
                )
            },
            "selected_by_interface": by_interface,
            "all_rows_completed": True,
            "negative_gain_can_still_win": True,
            "selection_split": "design_select",
            "oracle_or_control_rows_eligible": False,
        }
    )


__all__ = [
    "BoundedFeaturewiseConditioning",
    "FeedbackHBaseClassifier",
    "DirectFourTokenHead",
    "DirectStructureTokenAdapter",
    "GLOBAL_CONTROLS",
    "MANDATORY_FEEDBACK",
    "PAIR_CONTROLS",
    "PredictedPairAttentionBias",
    "ResidualStructureTokenAdapter",
    "TrainableParameterPadding",
    "build_feedback_selection",
    "build_feedback_model",
    "build_stage_e_plan",
    "feedback_interface_contract",
    "gate_warmup_updates",
]
