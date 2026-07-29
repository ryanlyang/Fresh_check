"""Step-12 refiners and deployable final consumers for RETB."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
from typing import Any

from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


REFINER_VARIANTS = (
    "TR0_NONE",
    "TR1_NATIVE_BASE",
    "TR2_ALL_NATIVE",
    "TR3_ZERO_NATIVE_SHAPE",
)
ADAPTER_VARIANTS = (
    "R0_PREDICTED_ONLY",
    "R1_PREDICTED_PLUS_NATIVE_BASE",
    "R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS",
    "R3_NATIVE_ONLY_MATCHED_TO_R2",
)
UNRESTRICTED_EVIDENCE_VARIANTS = (
    "F_TOKEN_ONLY",
    "F_TOKEN_PLUS_EXPERT_LOGITS",
    "F_TOKEN_ONLY_MATCHED",
)
NATIVE_DROPOUT_MODES = (
    "ND0_NONE",
    "ND1_FIXED",
    "ND2_CONFIDENCE",
)
BYPASS_CONTROLS = (
    "NORMAL",
    "NATIVE_BRANCH_REMOVED",
    "RECONSTRUCTED_BRANCH_REMOVED",
    "NATIVE_BRANCH_DROPPED_AT_EVALUATION",
    "RESIDUAL_GAMMA_ZERO",
    "SOURCE_EMBEDDINGS_SWAPPED",
)
CONSUMER_WIDTH = 256
MATCHED_MLP_WIDTHS = (64, 128, 192, 256, 320, 384)
EXPECTED_CORRUPTION_RATE = 0.10


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB final consumers")
    return torch


def _validate_dimensions(
    dimensions: Mapping[str, int],
    token_counts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    if (
        set(dimensions) != set(EXPERT_ORDER)
        or set(token_counts) != set(EXPERT_ORDER)
    ):
        raise ValueError("final-consumer expert allocation differs")
    dims = {name: int(dimensions[name]) for name in EXPERT_ORDER}
    counts = {name: int(token_counts[name]) for name in EXPERT_ORDER}
    if any(value not in {64, 128} for value in dims.values()) or any(
        value not in {1, 2, 4, 8, 16} for value in counts.values()
    ):
        raise ValueError("final-consumer token allocation is unregistered")
    return dims, counts


def _validate_banks(
    banks: Mapping[str, Any],
    *,
    dimensions: Mapping[str, int],
    token_counts: Mapping[str, int],
) -> tuple[int, Any]:
    module = _require_torch()
    if set(banks) != set(EXPERT_ORDER):
        raise ValueError("final-consumer bank coverage differs")
    batch = None
    device = None
    for expert in EXPERT_ORDER:
        value = banks[expert]
        if (
            not isinstance(value, module.Tensor)
            or value.ndim != 3
            or tuple(value.shape[1:])
            != (token_counts[expert], dimensions[expert])
            or not bool(module.isfinite(value).all())
        ):
            raise ValueError(f"final-consumer {expert} bank differs")
        if batch is None:
            batch, device = int(value.shape[0]), value.device
        elif int(value.shape[0]) != batch or value.device != device:
            raise ValueError("final-consumer bank batch/device differs")
    assert batch is not None
    return batch, device


def deterministic_robust_mixture(
    *,
    identities: Sequence[str],
    zero_based_epoch: int,
) -> list[dict[str, Any]]:
    """Return the exact repeating 25/25/25/25 robust-fusion schedule."""

    if int(zero_based_epoch) < 0 or not identities:
        raise ValueError("robust-fusion schedule inputs differ")
    rows = []
    for index, identity in enumerate(identities):
        mode_index = (index + int(zero_based_epoch)) % 4
        identity = str(identity)
        digest = hashlib.sha256(
            f"retb-of-robust-v1|{identity}|{zero_based_epoch}".encode()
        ).digest()
        if mode_index == 0:
            predicted = []
            mode = "all_oracle"
        elif mode_index == 1:
            predicted = [EXPERT_ORDER[digest[0] % len(EXPERT_ORDER)]]
            mode = "exactly_one_predicted"
        elif mode_index == 2:
            predicted = [
                expert
                for expert_index, expert in enumerate(EXPERT_ORDER)
                if digest[expert_index + 1] & 1
            ]
            mode = "independent_p0.5"
        else:
            predicted = list(EXPERT_ORDER)
            mode = "all_predicted"
        rows.append(
            {
                "identity": identity,
                "mode": mode,
                "predicted_experts": predicted,
            }
        )
    return rows


def materialize_robust_mixture_banks(
    *,
    identities: Sequence[str],
    zero_based_epoch: int,
    oracle_banks: Mapping[str, Any],
    predicted_banks: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module = _require_torch()
    if set(oracle_banks) != set(EXPERT_ORDER) or set(
        predicted_banks
    ) != set(EXPERT_ORDER):
        raise ValueError("robust-fusion mixture bank coverage differs")
    schedule = deterministic_robust_mixture(
        identities=identities, zero_based_epoch=zero_based_epoch
    )
    mixed = {}
    for expert in EXPERT_ORDER:
        selector = module.as_tensor(
            [
                expert in row["predicted_experts"]
                for row in schedule
            ],
            dtype=module.bool,
            device=predicted_banks[expert].device,
        )
        while selector.ndim < predicted_banks[expert].ndim:
            selector = selector.unsqueeze(-1)
        mixed[expert] = module.where(
            selector, predicted_banks[expert], oracle_banks[expert]
        )
    return mixed, schedule


def _waterfill_expected_probability(
    weights: Any, *, expected_rate: float
) -> Any:
    """Scale nonnegative scores to an exact mean with deterministic capping."""

    module = _require_torch()
    rate = float(expected_rate)
    if (
        weights.ndim != 2
        or not 0.0 <= rate <= 1.0
        or not bool(module.isfinite(weights).all())
        or bool((weights < 0).any())
    ):
        raise ValueError("confidence-corruption weights differ")
    result = module.zeros_like(weights)
    active = module.ones_like(weights, dtype=module.bool)
    target = rate * weights.shape[1]
    for _ in range(weights.shape[1] + 1):
        remaining = target - result.sum(dim=1, keepdim=True)
        count = active.sum(dim=1, keepdim=True).clamp_min(1)
        normalized = weights.masked_fill(~active, 0)
        scale = remaining / normalized.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        proposal = normalized * scale
        uniform_rows = normalized.sum(dim=1, keepdim=True) <= 1.0e-12
        proposal = module.where(
            uniform_rows,
            active.to(weights.dtype) * remaining / count,
            proposal,
        )
        capped = proposal >= 1.0
        if not bool((capped & active).any()):
            result = result + proposal
            break
        newly_capped = capped & active
        result = module.where(newly_capped, module.ones_like(result), result)
        active = active & ~newly_capped
    if not bool(
        module.allclose(
            result.mean(dim=1),
            module.full(
                (weights.shape[0],),
                rate,
                dtype=result.dtype,
                device=result.device,
            ),
            atol=2.0e-6,
            rtol=0.0,
        )
    ):
        raise RuntimeError("confidence corruption failed expected-rate lock")
    return result.clamp(0.0, 1.0)


def confidence_corruption_probabilities(
    calibrated_log_variance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build paired predicted/native probabilities with exact mean 0.10."""

    module = _require_torch()
    if set(calibrated_log_variance) != set(EXPERT_ORDER):
        raise ValueError("confidence-corruption expert coverage differs")
    scores = []
    for expert in EXPERT_ORDER:
        value = calibrated_log_variance[expert]
        if value.ndim != 3 or not bool(module.isfinite(value).all()):
            raise ValueError("confidence-corruption uncertainty differs")
        scores.append(value.float().mean(dim=(1, 2)))
    score = module.stack(scores, dim=1)
    centered = score - score.mean(dim=1, keepdim=True)
    predicted_weights = module.exp(centered.clamp(-8.0, 8.0))
    native_weights = module.exp((-centered).clamp(-8.0, 8.0))
    combined = module.cat((predicted_weights, native_weights), dim=1)
    probabilities = _waterfill_expected_probability(
        combined, expected_rate=EXPECTED_CORRUPTION_RATE
    )
    count = len(EXPERT_ORDER)
    return {
        "predicted": {
            expert: probabilities[:, index]
            for index, expert in enumerate(EXPERT_ORDER)
        },
        "native": {
            expert: probabilities[:, count + index]
            for index, expert in enumerate(EXPERT_ORDER)
        },
    }


def sample_native_dropout(
    *,
    mode: str,
    calibrated_log_variance: Mapping[str, Any],
    training: bool,
    evaluation_control: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return per-event availability without rejecting any event."""

    module = _require_torch()
    if mode not in NATIVE_DROPOUT_MODES:
        raise ValueError("native-dropout mode is unregistered")
    first = calibrated_log_variance[EXPERT_ORDER[0]]
    batch, device = int(first.shape[0]), first.device
    ones = {
        source: {
            expert: module.ones(batch, device=device)
            for expert in EXPERT_ORDER
        }
        for source in ("predicted", "native")
    }
    if mode == "ND0_NONE" or not (training or evaluation_control):
        return ones
    if mode == "ND1_FIXED":
        keep_native = (
            module.rand(batch, device=device)
            >= EXPECTED_CORRUPTION_RATE
        ).float()
        return {
            "predicted": ones["predicted"],
            "native": {
                expert: keep_native for expert in EXPERT_ORDER
            },
        }
    probabilities = confidence_corruption_probabilities(
        calibrated_log_variance
    )
    return {
        source: {
            expert: (
                module.rand(batch, device=device)
                >= probabilities[source][expert]
            ).float()
            for expert in EXPERT_ORDER
        }
        for source in ("predicted", "native")
    }


class PreNormalizedConsumerBlock(
    torch.nn.Module if torch is not None else object
):
    def __init__(self, *, width: int = CONSUMER_WIDTH) -> None:
        module = _require_torch()
        super().__init__()
        self.norm1 = module.nn.LayerNorm(width)
        self.attention = module.nn.MultiheadAttention(
            width, 8, dropout=0.0, batch_first=True
        )
        self.norm2 = module.nn.LayerNorm(width)
        self.mlp = module.nn.Sequential(
            module.nn.Linear(width, 4 * width),
            module.nn.GELU(),
            module.nn.Linear(4 * width, width),
        )

    def forward(self, values: Any, *, residual_gate: Any | None = None) -> Any:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        if residual_gate is not None:
            attended = attended * residual_gate
        values = values + attended
        residual = self.mlp(self.norm2(values))
        if residual_gate is not None:
            residual = residual * residual_gate
        return values + residual


class CrossAttentionRefinerBlock(
    torch.nn.Module if torch is not None else object
):
    def __init__(self) -> None:
        module = _require_torch()
        super().__init__()
        self.query_norm = module.nn.LayerNorm(CONSUMER_WIDTH)
        self.memory_norm = module.nn.LayerNorm(CONSUMER_WIDTH)
        self.cross_attention = module.nn.MultiheadAttention(
            CONSUMER_WIDTH, 8, dropout=0.0, batch_first=True
        )
        self.output_norm = module.nn.LayerNorm(CONSUMER_WIDTH)
        self.mlp = module.nn.Sequential(
            module.nn.Linear(CONSUMER_WIDTH, 4 * CONSUMER_WIDTH),
            module.nn.GELU(),
            module.nn.Linear(4 * CONSUMER_WIDTH, CONSUMER_WIDTH),
        )

    def forward(self, queries: Any, memory: Any) -> Any:
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        values = queries + attended
        return values + self.mlp(self.output_norm(values))


class NativeConditionedTokenRefiner(
    torch.nn.Module if torch is not None else object
):
    """Two-cross-attention gated residual refiner in offline coordinates."""

    def __init__(
        self,
        *,
        variant: str,
        bank_dimensions: Mapping[str, int],
        token_counts: Mapping[str, int],
        uncertainty_widths: Mapping[str, int],
    ) -> None:
        module = _require_torch()
        super().__init__()
        if variant not in REFINER_VARIANTS or set(
            uncertainty_widths
        ) != set(EXPERT_ORDER):
            raise ValueError("token-refiner configuration differs")
        self.variant = variant
        self.bank_dimensions, self.token_counts = _validate_dimensions(
            bank_dimensions, token_counts
        )
        self.uncertainty_widths = {
            name: int(uncertainty_widths[name]) for name in EXPERT_ORDER
        }
        if variant == "TR0_NONE":
            return
        self.predicted_projections = module.nn.ModuleDict(
            {
                name: module.nn.Linear(
                    self.bank_dimensions[name], CONSUMER_WIDTH
                )
                for name in EXPERT_ORDER
            }
        )
        self.native_projections = copy.deepcopy(self.predicted_projections)
        self.reliability_projections = module.nn.ModuleDict(
            {
                name: module.nn.Linear(
                    self.uncertainty_widths[name], CONSUMER_WIDTH
                )
                for name in EXPERT_ORDER
            }
        )
        self.expert_embedding = module.nn.Embedding(
            len(EXPERT_ORDER), CONSUMER_WIDTH
        )
        self.slot_embedding = module.nn.Embedding(16, CONSUMER_WIDTH)
        self.source_embedding = module.nn.Embedding(2, CONSUMER_WIDTH)
        self.blocks = module.nn.ModuleList(
            [CrossAttentionRefinerBlock() for _ in range(2)]
        )
        self.delta_heads = module.nn.ModuleDict(
            {
                name: module.nn.Linear(
                    CONSUMER_WIDTH, self.bank_dimensions[name]
                )
                for name in EXPERT_ORDER
            }
        )
        self.gate_heads = module.nn.ModuleDict(
            {
                name: module.nn.Linear(
                    CONSUMER_WIDTH, self.bank_dimensions[name]
                )
                for name in EXPERT_ORDER
            }
        )

    def forward(
        self,
        *,
        predicted_banks: Mapping[str, Any],
        calibrated_log_variance: Mapping[str, Any],
        native_banks: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        batch, device = _validate_banks(
            predicted_banks,
            dimensions=self.bank_dimensions,
            token_counts=self.token_counts,
        )
        if self.variant == "TR0_NONE":
            return {
                "refined_banks": dict(predicted_banks),
                "delta_banks": {
                    name: predicted_banks[name].new_zeros(
                        predicted_banks[name].shape
                    )
                    for name in EXPERT_ORDER
                },
                "gates": {
                    name: predicted_banks[name].new_zeros(
                        predicted_banks[name].shape
                    )
                    for name in EXPERT_ORDER
                },
                "identity_control": True,
            }
        if native_banks is None:
            raise ValueError("active token refiner requires native HLT banks")
        _validate_banks(
            native_banks,
            dimensions=self.bank_dimensions,
            token_counts=self.token_counts,
        )
        memory_rows = []
        native_experts = (
            ("BASE4",)
            if self.variant == "TR1_NATIVE_BASE"
            else EXPERT_ORDER
        )
        for index, expert in enumerate(native_experts):
            value = self.native_projections[expert](native_banks[expert])
            if self.variant == "TR3_ZERO_NATIVE_SHAPE":
                value = value * 0.0
            slots = self.token_counts[expert]
            slot_ids = _require_torch().arange(slots, device=device)
            memory_rows.append(
                value
                + self.expert_embedding.weight[
                    EXPERT_ORDER.index(expert)
                ][None, None]
                + self.slot_embedding(slot_ids)[None]
                + self.source_embedding.weight[1][None, None]
            )
        memory = _require_torch().cat(memory_rows, dim=1)
        refined, deltas, gates = {}, {}, {}
        for index, expert in enumerate(EXPERT_ORDER):
            uncertainty = calibrated_log_variance[expert]
            if (
                uncertainty.shape[:2]
                != predicted_banks[expert].shape[:2]
                or int(uncertainty.shape[-1])
                != self.uncertainty_widths[expert]
            ):
                raise ValueError("token-refiner uncertainty shape differs")
            slot_ids = _require_torch().arange(
                self.token_counts[expert], device=device
            )
            values = (
                self.predicted_projections[expert](
                    predicted_banks[expert]
                )
                + self.reliability_projections[expert](-uncertainty.float())
                + self.expert_embedding.weight[index][None, None]
                + self.slot_embedding(slot_ids)[None]
                + self.source_embedding.weight[0][None, None]
            )
            for block in self.blocks:
                values = block(values, memory)
            delta = self.delta_heads[expert](values)
            gate = _require_torch().sigmoid(self.gate_heads[expert](values))
            refined[expert] = predicted_banks[expert] + gate * delta
            deltas[expert], gates[expert] = delta, gate
        return {
            "refined_banks": refined,
            "delta_banks": deltas,
            "gates": gates,
            "identity_control": False,
        }


def _analytical_transformer_block_flops(
    token_count: int, *, width: int = CONSUMER_WIDTH
) -> int:
    n, d = int(token_count), int(width)
    return 4 * n * d * d + 2 * n * n * d + 8 * n * d * d


def select_matched_token_mlp_width(
    *,
    token_count: int,
    target_incremental_parameters: int | None = None,
    target_incremental_flops: int | None = None,
) -> dict[str, Any]:
    n = int(token_count)
    if n <= 0:
        raise ValueError("matched token-only selector token count differs")
    if target_incremental_parameters is None:
        target_incremental_parameters = 14 * (
            2 * 10 + 10 * CONSUMER_WIDTH + CONSUMER_WIDTH
        )
    if target_incremental_flops is None:
        target_incremental_flops = (
            4
            * (
                _analytical_transformer_block_flops(n + 14)
                - _analytical_transformer_block_flops(n)
            )
            + 14 * 2 * 10 * CONSUMER_WIDTH
        )
    rows = []
    for width in MATCHED_MLP_WIDTHS:
        parameters = 2 * CONSUMER_WIDTH + (
            CONSUMER_WIDTH * width
            + width
            + width * CONSUMER_WIDTH
            + CONSUMER_WIDTH
        )
        flops = n * 2 * (
            CONSUMER_WIDTH * width + width * CONSUMER_WIDTH
        )
        rows.append(
            {
                "hidden_width": width,
                "incremental_parameters": parameters,
                "incremental_flops": flops,
                "parameter_mismatch": abs(
                    parameters - int(target_incremental_parameters)
                ),
                "flop_mismatch": abs(
                    flops - int(target_incremental_flops)
                ),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["parameter_mismatch"],
            row["flop_mismatch"],
            row["hidden_width"],
        ),
    )
    return {
        "target_incremental_parameters": int(
            target_incremental_parameters
        ),
        "target_incremental_flops": int(target_incremental_flops),
        "ranking": [
            "incremental_parameter_mismatch",
            "analytical_inference_FLOP_mismatch",
            "smaller_hidden_width",
        ],
        "candidates": rows,
        "selected_hidden_width": selected["hidden_width"],
    }


class _TypedTokenConsumer(
    torch.nn.Module if torch is not None else object
):
    def _initialize_typed_inputs(
        self,
        *,
        bank_dimensions: Mapping[str, int],
        token_counts: Mapping[str, int],
        uncertainty_widths: Mapping[str, int],
    ) -> None:
        module = _require_torch()
        self.bank_dimensions, self.token_counts = _validate_dimensions(
            bank_dimensions, token_counts
        )
        if set(uncertainty_widths) != set(EXPERT_ORDER):
            raise ValueError("consumer uncertainty coverage differs")
        self.uncertainty_widths = {
            expert: int(uncertainty_widths[expert])
            for expert in EXPERT_ORDER
        }
        self.predicted_projections = module.nn.ModuleDict(
            {
                expert: module.nn.Linear(
                    self.bank_dimensions[expert], CONSUMER_WIDTH
                )
                for expert in EXPERT_ORDER
            }
        )
        self.native_projections = copy.deepcopy(self.predicted_projections)
        self.reliability_projections = module.nn.ModuleDict(
            {
                expert: module.nn.Linear(
                    self.uncertainty_widths[expert], CONSUMER_WIDTH
                )
                for expert in EXPERT_ORDER
            }
        )
        self.expert_embedding = module.nn.Embedding(
            len(EXPERT_ORDER), CONSUMER_WIDTH
        )
        self.slot_embedding = module.nn.Embedding(16, CONSUMER_WIDTH)
        self.source_embedding = module.nn.Embedding(2, CONSUMER_WIDTH)
        self.availability_embedding = module.nn.Embedding(
            2, CONSUMER_WIDTH
        )

    def _project_source(
        self,
        *,
        source: str,
        expert: str,
        values: Any,
        uncertainty: Any,
        availability: Any,
        swap_source_embeddings: bool,
    ) -> tuple[Any, Any]:
        module = _require_torch()
        source_index = 0 if source == "predicted" else 1
        if swap_source_embeddings:
            source_index = 1 - source_index
        projection = (
            self.predicted_projections[expert]
            if source == "predicted"
            else self.native_projections[expert]
        )
        projected = projection(values)
        if source == "predicted":
            projected = projected + self.reliability_projections[expert](
                -uncertainty.float()
            )
        slots = int(values.shape[1])
        slot_ids = module.arange(slots, device=values.device)
        available = availability[:, None, None]
        projected = projected * available
        projected = (
            projected
            + self.expert_embedding.weight[
                EXPERT_ORDER.index(expert)
            ][None, None]
            + self.slot_embedding(slot_ids)[None]
            + self.source_embedding.weight[source_index][None, None]
            + self.availability_embedding(
                availability.long()
            )[:, None]
        )
        return projected, available


class HLTResidualAdapter(_TypedTokenConsumer):
    """Two-layer constrained residual correction over deployable evidence."""

    def __init__(
        self,
        *,
        variant: str,
        native_dropout_mode: str,
        bank_dimensions: Mapping[str, int],
        token_counts: Mapping[str, int],
        uncertainty_widths: Mapping[str, int],
    ) -> None:
        module = _require_torch()
        super().__init__()
        if (
            variant not in ADAPTER_VARIANTS
            or native_dropout_mode not in NATIVE_DROPOUT_MODES
        ):
            raise ValueError("HLT adapter configuration differs")
        self.variant = variant
        self.native_dropout_mode = native_dropout_mode
        self._initialize_typed_inputs(
            bank_dimensions=bank_dimensions,
            token_counts=token_counts,
            uncertainty_widths=uncertainty_widths,
        )
        self.missing_predicted = module.nn.ParameterDict(
            {
                expert: module.nn.Parameter(
                    module.zeros(1, 1, self.bank_dimensions[expert])
                )
                for expert in EXPERT_ORDER
            }
        )
        self.class_token = module.nn.Parameter(
            module.zeros(1, 1, CONSUMER_WIDTH)
        )
        module.nn.init.normal_(self.class_token, std=0.02)
        self.blocks = module.nn.ModuleList(
            [PreNormalizedConsumerBlock() for _ in range(2)]
        )
        self.norm = module.nn.LayerNorm(CONSUMER_WIDTH)
        self.correction_head = module.nn.Linear(CONSUMER_WIDTH, 10)
        self.gamma = module.nn.Parameter(module.zeros(()))

    def forward(
        self,
        *,
        frozen_offline_logits: Any,
        predicted_banks: Mapping[str, Any],
        calibrated_log_variance: Mapping[str, Any],
        native_banks: Mapping[str, Any],
        bypass_control: str = "NORMAL",
        evaluation_dropout_control: bool = False,
    ) -> dict[str, Any]:
        module = _require_torch()
        if bypass_control not in BYPASS_CONTROLS:
            raise ValueError("HLT adapter bypass control is unregistered")
        batch, _ = _validate_banks(
            predicted_banks,
            dimensions=self.bank_dimensions,
            token_counts=self.token_counts,
        )
        _validate_banks(
            native_banks,
            dimensions=self.bank_dimensions,
            token_counts=self.token_counts,
        )
        availability = sample_native_dropout(
            mode=self.native_dropout_mode,
            calibrated_log_variance=calibrated_log_variance,
            training=self.training,
            evaluation_control=evaluation_dropout_control,
        )
        if bypass_control in {
            "NATIVE_BRANCH_REMOVED",
            "NATIVE_BRANCH_DROPPED_AT_EVALUATION",
        }:
            availability["native"] = {
                expert: predicted_banks[expert].new_zeros(batch)
                for expert in EXPERT_ORDER
            }
        if bypass_control == "RECONSTRUCTED_BRANCH_REMOVED":
            availability["predicted"] = {
                expert: predicted_banks[expert].new_zeros(batch)
                for expert in EXPERT_ORDER
            }
        native_sources = {
            "R0_PREDICTED_ONLY": (),
            "R1_PREDICTED_PLUS_NATIVE_BASE": ("BASE4",),
            "R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS": EXPERT_ORDER,
            "R3_NATIVE_ONLY_MATCHED_TO_R2": EXPERT_ORDER,
        }[self.variant]
        rows = []
        swap = bypass_control == "SOURCE_EMBEDDINGS_SWAPPED"
        for expert in EXPERT_ORDER:
            values = predicted_banks[expert]
            predicted_availability = availability["predicted"][expert]
            if self.variant == "R3_NATIVE_ONLY_MATCHED_TO_R2":
                values = self.missing_predicted[expert].expand(
                    batch, self.token_counts[expert], -1
                )
                predicted_availability = values.new_zeros(batch)
            row, _ = self._project_source(
                source="predicted",
                expert=expert,
                values=values,
                uncertainty=calibrated_log_variance[expert],
                availability=predicted_availability,
                swap_source_embeddings=swap,
            )
            rows.append(row)
        for expert in native_sources:
            row, _ = self._project_source(
                source="native",
                expert=expert,
                values=native_banks[expert],
                uncertainty=calibrated_log_variance[expert],
                availability=availability["native"][expert],
                swap_source_embeddings=swap,
            )
            rows.append(row)
        sequence = module.cat(rows, dim=1)
        sequence = module.cat(
            (self.class_token.expand(batch, -1, -1), sequence), dim=1
        )
        for block in self.blocks:
            sequence = block(sequence)
        residual_logits = self.correction_head(
            self.norm(sequence[:, 0])
        )
        gamma = (
            self.gamma * 0.0
            if bypass_control == "RESIDUAL_GAMMA_ZERO"
            else self.gamma
        )
        combined = frozen_offline_logits + gamma * residual_logits
        return {
            "frozen_path_logits": frozen_offline_logits,
            "residual_path_logits": residual_logits,
            "combined_logits": combined,
            "gamma": gamma,
            "availability": availability,
        }


class UnrestrictedHLTFusion(_TypedTokenConsumer):
    """Four-layer maximum-performance fusion over all deployable banks."""

    def __init__(
        self,
        *,
        evidence_variant: str,
        native_dropout_mode: str,
        bank_dimensions: Mapping[str, int],
        token_counts: Mapping[str, int],
        uncertainty_widths: Mapping[str, int],
    ) -> None:
        module = _require_torch()
        super().__init__()
        if (
            evidence_variant not in UNRESTRICTED_EVIDENCE_VARIANTS
            or native_dropout_mode not in NATIVE_DROPOUT_MODES
        ):
            raise ValueError("unrestricted HLT fusion configuration differs")
        self.evidence_variant = evidence_variant
        self.native_dropout_mode = native_dropout_mode
        self._initialize_typed_inputs(
            bank_dimensions=bank_dimensions,
            token_counts=token_counts,
            uncertainty_widths=uncertainty_widths,
        )
        self.reliability_gates = module.nn.ModuleDict(
            {
                expert: module.nn.Linear(
                    self.uncertainty_widths[expert], 1
                )
                for expert in EXPERT_ORDER
            }
        )
        self.class_token = module.nn.Parameter(
            module.zeros(1, 1, CONSUMER_WIDTH)
        )
        module.nn.init.normal_(self.class_token, std=0.02)
        self.blocks = module.nn.ModuleList(
            [PreNormalizedConsumerBlock() for _ in range(4)]
        )
        self.logit_embeddings = (
            module.nn.ModuleDict(
                {
                    f"{source}__{expert}": module.nn.Sequential(
                        module.nn.LayerNorm(10),
                        module.nn.Linear(10, CONSUMER_WIDTH),
                    )
                    for source in ("native", "predicted")
                    for expert in EXPERT_ORDER
                }
            )
            if evidence_variant == "F_TOKEN_PLUS_EXPERT_LOGITS"
            else None
        )
        token_total = 2 * sum(self.token_counts.values())
        self.matched_width_selection = (
            select_matched_token_mlp_width(token_count=token_total)
            if evidence_variant == "F_TOKEN_ONLY_MATCHED"
            else None
        )
        self.matched_residual = (
            module.nn.Sequential(
                module.nn.LayerNorm(CONSUMER_WIDTH),
                module.nn.Linear(
                    CONSUMER_WIDTH,
                    self.matched_width_selection[
                        "selected_hidden_width"
                    ],
                ),
                module.nn.GELU(),
                module.nn.Linear(
                    self.matched_width_selection[
                        "selected_hidden_width"
                    ],
                    CONSUMER_WIDTH,
                ),
            )
            if self.matched_width_selection is not None
            else None
        )
        self.norm = module.nn.LayerNorm(CONSUMER_WIDTH)
        self.classifier = module.nn.Linear(CONSUMER_WIDTH, 10)

    def forward(
        self,
        *,
        token_banks: Mapping[str, Any],
        calibrated_log_variance: Mapping[str, Any],
        native_banks: Mapping[str, Any],
        native_expert_logits: Mapping[str, Any] | None = None,
        predicted_expert_logits: Mapping[str, Any] | None = None,
        bypass_control: str = "NORMAL",
        evaluation_dropout_control: bool = False,
    ) -> dict[str, Any]:
        module = _require_torch()
        if (
            bypass_control not in BYPASS_CONTROLS
            or bypass_control == "RESIDUAL_GAMMA_ZERO"
        ):
            raise ValueError("unrestricted-fusion bypass control differs")
        batch, _ = _validate_banks(
            token_banks,
            dimensions=self.bank_dimensions,
            token_counts=self.token_counts,
        )
        _validate_banks(
            native_banks,
            dimensions=self.bank_dimensions,
            token_counts=self.token_counts,
        )
        availability = sample_native_dropout(
            mode=self.native_dropout_mode,
            calibrated_log_variance=calibrated_log_variance,
            training=self.training,
            evaluation_control=evaluation_dropout_control,
        )
        if bypass_control in {
            "NATIVE_BRANCH_REMOVED",
            "NATIVE_BRANCH_DROPPED_AT_EVALUATION",
        }:
            availability["native"] = {
                expert: token_banks[expert].new_zeros(batch)
                for expert in EXPERT_ORDER
            }
        if bypass_control == "RECONSTRUCTED_BRANCH_REMOVED":
            availability["predicted"] = {
                expert: token_banks[expert].new_zeros(batch)
                for expert in EXPERT_ORDER
            }
        swap = bypass_control == "SOURCE_EMBEDDINGS_SWAPPED"
        rows, gates = [], []
        for source, banks in (
            ("predicted", token_banks),
            ("native", native_banks),
        ):
            for expert in EXPERT_ORDER:
                row, available = self._project_source(
                    source=source,
                    expert=expert,
                    values=banks[expert],
                    uncertainty=calibrated_log_variance[expert],
                    availability=availability[source][expert],
                    swap_source_embeddings=swap,
                )
                reliability_gate = module.sigmoid(
                    self.reliability_gates[expert](
                        -calibrated_log_variance[expert].float()
                    )
                )
                rows.append(row)
                gates.append(reliability_gate * available)
        sequence = module.cat(rows, dim=1)
        residual_gate = module.cat(gates, dim=1)
        if self.matched_residual is not None:
            sequence = sequence + self.matched_residual(sequence)
        if self.logit_embeddings is not None:
            if (
                native_expert_logits is None
                or predicted_expert_logits is None
                or set(native_expert_logits) != set(EXPERT_ORDER)
                or set(predicted_expert_logits) != set(EXPERT_ORDER)
            ):
                raise ValueError(
                    "logit-augmented fusion requires all deployable logits"
                )
            logit_rows = []
            for source, values in (
                ("native", native_expert_logits),
                ("predicted", predicted_expert_logits),
            ):
                source_index = 1 if source == "native" else 0
                if swap:
                    source_index = 1 - source_index
                for expert_index, expert in enumerate(EXPERT_ORDER):
                    logits = values[expert]
                    if logits.shape != (batch, 10):
                        raise ValueError("expert-logit token shape differs")
                    available = availability[source][expert][
                        :, None, None
                    ]
                    logit_rows.append(
                        available
                        * (
                            self.logit_embeddings[
                                f"{source}__{expert}"
                            ](logits)[:, None]
                            + self.expert_embedding.weight[
                                expert_index
                            ][None, None]
                            + self.source_embedding.weight[
                                source_index
                            ][None, None]
                        )
                        + self.availability_embedding.weight[
                            1
                        ][None, None]
                        * available
                        + self.availability_embedding.weight[
                            0
                        ][None, None]
                        * (1.0 - available)
                    )
            sequence = module.cat((sequence, *logit_rows), dim=1)
            residual_gate = module.cat(
                (
                    residual_gate,
                    module.cat(
                        [
                            availability[source][expert][:, None, None]
                            for source in ("native", "predicted")
                            for expert in EXPERT_ORDER
                        ],
                        dim=1,
                    ),
                ),
                dim=1,
            )
        class_token = self.class_token.expand(batch, -1, -1)
        sequence = module.cat((class_token, sequence), dim=1)
        residual_gate = module.cat(
            (
                module.ones(
                    batch,
                    1,
                    1,
                    dtype=residual_gate.dtype,
                    device=residual_gate.device,
                ),
                residual_gate,
            ),
            dim=1,
        )
        for block in self.blocks:
            sequence = block(sequence, residual_gate=residual_gate)
        logits = self.classifier(self.norm(sequence[:, 0]))
        if not bool(module.isfinite(logits).all()):
            raise FloatingPointError("unrestricted HLT logits are nonfinite")
        return {
            "logits": logits,
            "availability": availability,
            "reliability_gates": residual_gate[:, 1:],
            "matched_width_selection": self.matched_width_selection,
        }


class FrozenPredictedOfflineFusion(
    torch.nn.Module if torch is not None else object
):
    """PF_FROZEN: exact frozen OF weights over predicted original tokens."""

    def __init__(self, frozen_offline_fusion: Any) -> None:
        super().__init__()
        self.fusion = frozen_offline_fusion
        for parameter in self.fusion.parameters():
            parameter.requires_grad_(False)
        self.fusion.eval()

    def train(self, mode: bool = True) -> Any:
        super().train(mode)
        self.fusion.eval()
        return self

    def forward(self, *, predicted_banks: Mapping[str, Any]) -> Any:
        return self.fusion(token_banks=predicted_banks)


def clone_robust_offline_fusion(frozen_offline_fusion: Any) -> Any:
    cloned = copy.deepcopy(frozen_offline_fusion)
    for parameter in cloned.parameters():
        parameter.requires_grad_(True)
    return cloned


__all__ = [
    "ADAPTER_VARIANTS",
    "BYPASS_CONTROLS",
    "EXPECTED_CORRUPTION_RATE",
    "FrozenPredictedOfflineFusion",
    "HLTResidualAdapter",
    "MATCHED_MLP_WIDTHS",
    "NATIVE_DROPOUT_MODES",
    "NativeConditionedTokenRefiner",
    "REFINER_VARIANTS",
    "UNRESTRICTED_EVIDENCE_VARIANTS",
    "UnrestrictedHLTFusion",
    "clone_robust_offline_fusion",
    "confidence_corruption_probabilities",
    "deterministic_robust_mixture",
    "materialize_robust_mixture_banks",
    "sample_native_dropout",
    "select_matched_token_mlp_width",
]
