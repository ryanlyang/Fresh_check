"""RETB token predictors, typed HLT evidence, and predictor capacity controls."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

from .fusion import RMSNorm
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PREDICTOR_ARCHITECTURE_CONTRACT = "retb_predictor_architectures_v1"
PREDICTOR_CAPACITY_CONTRACT = "retb_predictor_capacity_report_v1"
ARCHITECTURES = (
    "A0_AFFINE",
    "A1_RESMLP",
    "A2_TOKEN_ENCODER",
    "A3_SLOT_DECODER_DIRECT",
    "A4_SLOT_DECODER_GATED",
)
CONTEXTS = ("C0_SELF", "C1_NATIVE", "C2_ALL", "C3_ALL_PARTICLE")
UNCERTAINTY_HEADS = ("U_SLOT", "U_GROUP4", "U_DIAGONAL")
NORMALIZATION_MODES = ("N_UNCLIPPED", "N_CLIP16", "N_CLIP8")
RELATION_PARTICLE_ORDER = ("PT", "TRACK", "REGION")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB predictors")
    return torch


def uncertainty_width(head: str, token_dimension: int) -> int:
    if head == "U_SLOT":
        return 1
    if head == "U_GROUP4":
        if int(token_dimension) % 4:
            raise ValueError("U_GROUP4 requires four equal channel groups")
        return 4
    if head == "U_DIAGONAL":
        return int(token_dimension)
    raise ValueError("predictor uncertainty head is unregistered")


def build_predictor_architecture_contract() -> dict[str, Any]:
    from .contracts import with_content_hash

    return with_content_hash(
        {
            "contract": PREDICTOR_ARCHITECTURE_CONTRACT,
            "schema_version": 1,
            "architectures": list(ARCHITECTURES),
            "contexts": {
                "C0_SELF": ["corresponding_HLT_token_bank"],
                "C1_NATIVE": [
                    "corresponding_HLT_token_bank",
                    "HE_BASE4_particle_states",
                ],
                "C2_ALL": [
                    "corresponding_HLT_token_bank",
                    "all_seven_HLT_token_banks",
                    "HE_BASE4_particle_states",
                ],
                "C3_ALL_PARTICLE": [
                    "all_seven_HLT_token_banks",
                    "HE_BASE4_particle_states",
                    "PT_particle_states",
                    "TRACK_particle_states",
                    "REGION_particle_states",
                ],
            },
            "decoder": {
                "layers": 3,
                "heads": {"D64": 4, "D128": 8},
                "mlp_expansion": 4,
                "dropout": 0.1,
                "query_initialization": (
                    "copy_offline_slot_queries_without_weight_sharing"
                ),
                "fixed_index_non_autoregressive": True,
            },
            "A4": {
                "anchor": "learned_affine_of_LayerNorm_corresponding_HLT_bank",
                "gate_bias": -2.0,
                "formula": "anchor_plus_sigmoid_gate_times_delta",
            },
            "uncertainty": {
                "heads": list(UNCERTAINTY_HEADS),
                "log_variance_clip": [-8.0, 4.0],
            },
            "normalization_modes": list(NORMALIZATION_MODES),
            "analytical_FLOP_convention": {
                "multiply_add": 2,
                "included": [
                    "linear_projections",
                    "attention_score_and_value_products",
                    "MLP_matrix_products",
                    "typed_evidence_projections",
                    "prediction_and_uncertainty_heads",
                ],
                "excluded": [
                    "normalization",
                    "activation",
                    "softmax",
                    "bias_and_residual_additions",
                ],
            },
            "offline_inputs_permitted_in_forward": False,
            "physical_pair_bias_between_abstract_tokens": False,
            "performance_based_termination": False,
        }
    )


def validate_predictor_architecture_contract(
    payload: Mapping[str, Any],
) -> str:
    from .contracts import validate_content_hash

    digest = validate_content_hash(
        payload, expected_contract=PREDICTOR_ARCHITECTURE_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_predictor_architecture_contract()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("predictor architecture contract semantics differ")
    return digest


class TypedHLTEvidence(torch.nn.Module if torch is not None else object):
    """Build one typed evidence sequence from HLT-derived states only."""

    def __init__(
        self,
        *,
        token_dimension: int,
        target_expert_id: str,
        context: str,
    ) -> None:
        module = _require_torch()
        super().__init__()
        dimension = int(token_dimension)
        if (
            dimension not in {64, 128}
            or target_expert_id not in EXPERT_ORDER
            or context not in CONTEXTS
        ):
            raise ValueError("typed HLT evidence configuration differs")
        self.token_dimension = dimension
        self.target_expert_id = target_expert_id
        self.context = context
        bank_names = (
            EXPERT_ORDER
            if context in {"C2_ALL", "C3_ALL_PARTICLE"}
            else (target_expert_id,)
        )
        self.bank_projections = module.nn.ModuleDict(
            {expert: module.nn.LazyLinear(dimension) for expert in bank_names}
        )
        particle_names = (
            ("BASE4", *RELATION_PARTICLE_ORDER)
            if context == "C3_ALL_PARTICLE"
            else ("BASE4",)
            if context in {"C1_NATIVE", "C2_ALL"}
            else ()
        )
        self.particle_projections = module.nn.ModuleDict(
            {
                name: module.nn.LazyLinear(dimension)
                for name in particle_names
            }
        )
        # corresponding, all-bank, BASE4 particle, relation particle
        self.source_embedding = module.nn.Embedding(4, dimension)
        self.expert_embedding = module.nn.Embedding(
            len(EXPERT_ORDER), dimension
        )
        self.slot_embedding = module.nn.Embedding(16, dimension)
        self.kind_embedding = module.nn.Embedding(2, dimension)

    def _bank(
        self, bank: Any, *, expert: str, source_index: int
    ) -> tuple[Any, Any]:
        module = _require_torch()
        if (
            not isinstance(bank, module.Tensor)
            or bank.ndim != 3
            or not 1 <= int(bank.shape[1]) <= 16
        ):
            raise ValueError("predictor HLT token-bank shape differs")
        slots = module.arange(bank.shape[1], device=bank.device)
        values = (
            self.bank_projections[expert](bank)
            + self.source_embedding.weight[source_index]
            + self.expert_embedding.weight[EXPERT_ORDER.index(expert)]
            + self.slot_embedding(slots)[None]
            + self.kind_embedding.weight[0]
        )
        mask = module.zeros(
            bank.shape[:2], dtype=module.bool, device=bank.device
        )
        return values, mask

    def _particles(
        self, values: Any, mask: Any, *, source: str, source_index: int
    ) -> tuple[Any, Any]:
        module = _require_torch()
        if (
            not isinstance(values, module.Tensor)
            or values.ndim != 3
            or tuple(mask.shape) != tuple(values.shape[:2])
            or bool((mask.bool().sum(dim=1) == 0).any())
        ):
            raise ValueError("predictor HLT particle evidence shape differs")
        projected = (
            self.particle_projections[source](values)
            + self.source_embedding.weight[source_index]
            + self.kind_embedding.weight[1]
        )
        return projected, ~mask.bool()

    def forward(
        self,
        *,
        corresponding_hlt_tokens: Any,
        hlt_token_banks: Mapping[str, Any] | None = None,
        unbiased_particle_states: Any | None = None,
        particle_mask: Any | None = None,
        relation_particle_states: Mapping[str, Any] | None = None,
        relation_particle_masks: Mapping[str, Any] | None = None,
        zero_evidence: bool = False,
        evidence_permutation: Any | None = None,
        evidence_batch_permutation: Any | None = None,
    ) -> tuple[Any, Any]:
        rows, masks = [], []
        batch = int(corresponding_hlt_tokens.shape[0])
        if self.context != "C3_ALL_PARTICLE":
            row, mask = self._bank(
                corresponding_hlt_tokens,
                expert=self.target_expert_id,
                source_index=0,
            )
            rows.append(row)
            masks.append(mask)
        if self.context in {"C2_ALL", "C3_ALL_PARTICLE"}:
            if hlt_token_banks is None or set(hlt_token_banks) != set(
                EXPERT_ORDER
            ):
                raise ValueError("all-bank predictor context coverage differs")
            for expert in EXPERT_ORDER:
                row, mask = self._bank(
                    hlt_token_banks[expert], expert=expert, source_index=1
                )
                if int(row.shape[0]) != batch:
                    raise ValueError("predictor HLT bank batch differs")
                rows.append(row)
                masks.append(mask)
        if self.context in {"C1_NATIVE", "C2_ALL", "C3_ALL_PARTICLE"}:
            if unbiased_particle_states is None or particle_mask is None:
                raise ValueError("predictor context lacks BASE4 particle states")
            row, mask = self._particles(
                unbiased_particle_states,
                particle_mask,
                source="BASE4",
                source_index=2,
            )
            if int(row.shape[0]) != batch:
                raise ValueError("predictor BASE4 particle batch differs")
            rows.append(row)
            masks.append(mask)
        if self.context == "C3_ALL_PARTICLE":
            if (
                relation_particle_states is None
                or relation_particle_masks is None
                or set(relation_particle_states) != set(RELATION_PARTICLE_ORDER)
                or set(relation_particle_masks)
                != set(RELATION_PARTICLE_ORDER)
            ):
                raise ValueError("C3 relation-particle coverage differs")
            for source in RELATION_PARTICLE_ORDER:
                row, mask = self._particles(
                    relation_particle_states[source],
                    relation_particle_masks[source],
                    source=source,
                    source_index=3,
                )
                if int(row.shape[0]) != batch:
                    raise ValueError("C3 relation-particle batch differs")
                rows.append(row)
                masks.append(mask)
        module = _require_torch()
        evidence = module.cat(rows, dim=1)
        padding_mask = module.cat(masks, dim=1)
        if evidence_permutation is not None:
            permutation = module.as_tensor(
                evidence_permutation,
                device=evidence.device,
                dtype=module.long,
            )
            if (
                permutation.ndim != 1
                or len(permutation) != evidence.shape[1]
                or not module.equal(
                    permutation.sort().values,
                    module.arange(len(permutation), device=evidence.device),
                )
            ):
                raise ValueError("predictor evidence permutation differs")
            evidence = evidence[:, permutation]
            padding_mask = padding_mask[:, permutation]
        if zero_evidence:
            evidence = evidence * 0.0
        if evidence_batch_permutation is not None:
            permutation = module.as_tensor(
                evidence_batch_permutation,
                device=evidence.device,
                dtype=module.long,
            )
            if (
                permutation.ndim != 1
                or len(permutation) != batch
                or not module.equal(
                    permutation.sort().values,
                    module.arange(batch, device=evidence.device),
                )
            ):
                raise ValueError("predictor evidence batch permutation differs")
            evidence = evidence[permutation]
            padding_mask = padding_mask[permutation]
        return evidence, padding_mask


class RetbTokenPredictor(torch.nn.Module if torch is not None else object):
    """All registered A0-A4 predictors with a uniform HLT-only interface."""

    def __init__(
        self,
        *,
        architecture: str,
        context: str,
        target_expert_id: str,
        token_count: int,
        token_dimension: int,
        offline_slot_queries: Any,
        uncertainty_head: str = "U_SLOT",
        dropout: float = 0.1,
        zero_evidence_control: bool = False,
        residual_hidden_width: int | None = None,
    ) -> None:
        module = _require_torch()
        super().__init__()
        k, d = int(token_count), int(token_dimension)
        if (
            architecture not in ARCHITECTURES
            or context not in CONTEXTS
            or target_expert_id not in EXPERT_ORDER
            or k not in {1, 2, 4, 8, 16}
            or d not in {64, 128}
            or tuple(offline_slot_queries.shape) != (k, d)
            or float(dropout) not in {0.0, 0.1}
        ):
            raise ValueError("predictor configuration is unregistered")
        if architecture in {"A0_AFFINE", "A1_RESMLP", "A2_TOKEN_ENCODER"}:
            if context != "C0_SELF":
                raise ValueError("A0-A2 are registered only with C0_SELF")
        self.architecture = architecture
        self.context = context
        self.target_expert_id = target_expert_id
        self.token_count = k
        self.token_dimension = d
        self.uncertainty_head = uncertainty_head
        self.zero_evidence_control = bool(zero_evidence_control)
        if architecture in {
            "A3_SLOT_DECODER_DIRECT",
            "A4_SLOT_DECODER_GATED",
        }:
            self.evidence = TypedHLTEvidence(
                token_dimension=d,
                target_expert_id=target_expert_id,
                context=context,
            )
        if architecture == "A0_AFFINE":
            self.affine = module.nn.Linear(d, d)
        if architecture == "A4_SLOT_DECODER_GATED":
            self.anchor_norm = module.nn.LayerNorm(d)
            self.anchor_map = module.nn.Linear(d, d)
        if architecture == "A1_RESMLP":
            width = int(residual_hidden_width or 2 * d)
            if width <= 0:
                raise ValueError("residual MLP width must be positive")
            self.residual_hidden_width = width
            self.residual_norm = RMSNorm(d)
            self.residual_up = module.nn.Linear(d, width)
            self.residual_down = module.nn.Linear(width, d)
        else:
            self.residual_hidden_width = None
        if architecture == "A2_TOKEN_ENCODER":
            layer = module.nn.TransformerEncoderLayer(
                d_model=d,
                nhead=4 if d == 64 else 8,
                dim_feedforward=4 * d,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = module.nn.TransformerEncoder(layer, num_layers=3)
            self.encoder_output = module.nn.Linear(d, d)
        if architecture in {
            "A3_SLOT_DECODER_DIRECT",
            "A4_SLOT_DECODER_GATED",
        }:
            self.target_queries = module.nn.Parameter(
                offline_slot_queries.detach().float().clone()
            )
            layer = module.nn.TransformerDecoderLayer(
                d_model=d,
                nhead=4 if d == 64 else 8,
                dim_feedforward=4 * d,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = module.nn.TransformerDecoder(layer, num_layers=3)
            self.output_norm = module.nn.LayerNorm(d)
        if architecture == "A4_SLOT_DECODER_GATED":
            self.gate_head = module.nn.Linear(d, 1)
            module.nn.init.zeros_(self.gate_head.weight)
            module.nn.init.constant_(self.gate_head.bias, -2.0)
        width = uncertainty_width(uncertainty_head, d)
        self.log_variance_head = module.nn.Linear(d, width)

    def forward(
        self,
        *,
        corresponding_hlt_tokens: Any,
        hlt_token_banks: Mapping[str, Any] | None = None,
        unbiased_particle_states: Any | None = None,
        particle_mask: Any | None = None,
        relation_particle_states: Mapping[str, Any] | None = None,
        relation_particle_masks: Mapping[str, Any] | None = None,
        evidence_permutation: Any | None = None,
        evidence_batch_permutation: Any | None = None,
    ) -> dict[str, Any]:
        module = _require_torch()
        if (
            corresponding_hlt_tokens.ndim != 3
            or tuple(corresponding_hlt_tokens.shape[1:])
            != (self.token_count, self.token_dimension)
        ):
            raise ValueError("corresponding HLT token shape differs")
        corresponding = corresponding_hlt_tokens
        gate = None
        if self.architecture == "A0_AFFINE":
            predicted = self.affine(corresponding)
        elif self.architecture == "A1_RESMLP":
            delta = self.residual_down(
                module.nn.functional.gelu(
                    self.residual_up(self.residual_norm(corresponding))
                )
            )
            predicted = corresponding + delta
        elif self.architecture == "A2_TOKEN_ENCODER":
            predicted = self.encoder_output(
                self.encoder(corresponding)[:, : self.token_count]
            )
        else:
            evidence, padding_mask = self.evidence(
                corresponding_hlt_tokens=corresponding_hlt_tokens,
                hlt_token_banks=hlt_token_banks,
                unbiased_particle_states=unbiased_particle_states,
                particle_mask=particle_mask,
                relation_particle_states=relation_particle_states,
                relation_particle_masks=relation_particle_masks,
                zero_evidence=self.zero_evidence_control,
                evidence_permutation=evidence_permutation,
                evidence_batch_permutation=evidence_batch_permutation,
            )
            queries = self.target_queries[None].expand(
                corresponding.shape[0], -1, -1
            )
            decoded = self.output_norm(
                self.decoder(
                    queries,
                    evidence,
                    memory_key_padding_mask=padding_mask,
                )
            )
            if self.architecture == "A3_SLOT_DECODER_DIRECT":
                predicted = decoded
            else:
                anchor = self.anchor_map(self.anchor_norm(corresponding))
                gate = module.sigmoid(self.gate_head(decoded))
                predicted = anchor + gate * decoded
        log_variance = self.log_variance_head(predicted).clamp(-8.0, 4.0)
        if (
            tuple(predicted.shape)
            != (
                int(corresponding_hlt_tokens.shape[0]),
                self.token_count,
                self.token_dimension,
            )
            or tuple(log_variance.shape)
            != tuple(predicted.shape[:2])
            + (uncertainty_width(self.uncertainty_head, self.token_dimension),)
            or not bool(
                module.isfinite(predicted).all()
                and module.isfinite(log_variance).all()
            )
        ):
            raise FloatingPointError("predictor output contract differs")
        return {
            "predicted_tokens": predicted,
            "log_variance": log_variance,
            "gate": gate,
        }


def select_widened_resmlp_width(
    *, token_dimension: int, target_incremental_parameters: int
) -> dict[str, int]:
    d, target = int(token_dimension), int(target_incremental_parameters)
    if d not in {64, 128} or target <= 0:
        raise ValueError("widened residual-MLP target differs")
    rows = []
    for width in range(d, 64 * d + 1):
        residual_block_parameters = (
            d + d * width + width + width * d + d
        )
        affine_block_parameters = d * d + d
        incremental_parameters = (
            residual_block_parameters - affine_block_parameters
        )
        flops = 4 * d * width
        rows.append(
            (
                abs(incremental_parameters - target),
                flops,
                width,
                residual_block_parameters,
                incremental_parameters,
            )
        )
    mismatch, flops, width, residual_parameters, incremental_parameters = min(
        rows
    )
    return {
        "hidden_width": width,
        "residual_block_parameter_count": residual_parameters,
        "incremental_parameter_count": incremental_parameters,
        "incremental_parameter_mismatch": mismatch,
        "analytical_per_token_linear_flops": flops,
    }


def predictor_analytical_flops(
    *,
    architecture: str,
    batch_size: int,
    token_count: int,
    token_dimension: int,
    evidence_token_count: int,
    uncertainty_width_value: int,
    residual_hidden_width: int | None = None,
    evidence_projection_flops: int = 0,
) -> int:
    b, k, d, m, u = map(
        int,
        (
            batch_size,
            token_count,
            token_dimension,
            evidence_token_count,
            uncertainty_width_value,
        ),
    )
    if architecture not in ARCHITECTURES or min(b, k, d, m, u) <= 0:
        raise ValueError("predictor FLOP configuration differs")
    output_projection = 2 * b * k * d * d
    uncertainty = 2 * b * k * d * u
    evidence_projection = int(evidence_projection_flops)
    if evidence_projection < 0:
        raise ValueError("predictor evidence-projection FLOPs differ")
    if architecture == "A0_AFFINE":
        core = output_projection
    elif architecture == "A1_RESMLP":
        width = int(residual_hidden_width or 2 * d)
        core = 4 * b * k * d * width
    elif architecture == "A2_TOKEN_ENCODER":
        per_layer = b * (
            8 * k * d * d + 4 * k * k * d + 16 * k * d * d
        )
        core = output_projection + 3 * per_layer
    else:
        per_layer = b * (
            8 * k * d * d
            + 4 * k * k * d
            + 4 * (k + m) * d * d
            + 4 * k * m * d
            + 16 * k * d * d
        )
        core = evidence_projection + 3 * per_layer
        if architecture == "A4_SLOT_DECODER_GATED":
            core += 2 * b * k * d * d + 2 * b * k * d
    return int(core + uncertainty)


def profile_predictor(
    model: Any,
    *,
    forward_kwargs: Mapping[str, Any],
    analytical_flops: int,
    warmup_iterations: int = 2,
    measured_iterations: int = 5,
) -> dict[str, Any]:
    module = _require_torch()
    if int(analytical_flops) <= 0:
        raise ValueError("predictor analytical FLOPs must be positive")
    model.eval()
    device = next(model.parameters()).device
    with module.no_grad():
        for _ in range(int(warmup_iterations)):
            model(**forward_kwargs)
        if device.type == "cuda":
            module.cuda.synchronize(device)
            module.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(int(measured_iterations)):
            model(**forward_kwargs)
        if device.type == "cuda":
            module.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "analytical_flops": int(analytical_flops),
        "measured_latency_seconds_mean": elapsed / int(measured_iterations),
        "measured_iterations": int(measured_iterations),
        "peak_memory_bytes": (
            int(module.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "latency_used_for_selection": False,
    }


def build_predictor_capacity_report(
    *,
    run_id: str,
    architecture: str,
    token_dimension: int,
    selected_profile: Mapping[str, Any],
    affine_baseline_parameter_count: int,
    zero_evidence_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from .contracts import with_content_hash

    required = {
        "parameter_count",
        "analytical_flops",
        "measured_latency_seconds_mean",
        "measured_iterations",
        "peak_memory_bytes",
        "latency_used_for_selection",
    }
    if (
        architecture not in ARCHITECTURES
        or int(token_dimension) not in {64, 128}
        or set(selected_profile) != required
        or int(selected_profile["parameter_count"]) <= 0
        or int(selected_profile["analytical_flops"]) <= 0
        or int(affine_baseline_parameter_count) <= 0
        or int(affine_baseline_parameter_count)
        > int(selected_profile["parameter_count"])
    ):
        raise ValueError("predictor capacity profile differs")
    incremental = max(
        1,
        int(selected_profile["parameter_count"])
        - int(affine_baseline_parameter_count),
    )
    widened = select_widened_resmlp_width(
        token_dimension=int(token_dimension),
        target_incremental_parameters=incremental,
    )
    widened["control_total_parameter_count"] = (
        int(affine_baseline_parameter_count)
        + int(widened["incremental_parameter_count"])
    )
    if zero_evidence_profile is not None and set(zero_evidence_profile) != required:
        raise ValueError("zero-evidence capacity profile differs")
    return with_content_hash(
        {
            "contract": PREDICTOR_CAPACITY_CONTRACT,
            "schema_version": 1,
            "run_id": str(run_id),
            "architecture": architecture,
            "selected_predictor": dict(selected_profile),
            "affine_baseline_parameter_count": int(
                affine_baseline_parameter_count
            ),
            "selected_incremental_parameter_count": incremental,
            "matched_widened_residual_MLP": widened,
            "zero_evidence_decoder": (
                None
                if zero_evidence_profile is None
                else {
                    **dict(zero_evidence_profile),
                    "evidence_values_zeroed_after_typed_projection": True,
                    "parameters_removed_or_frozen": False,
                }
            ),
            "matching_order": [
                "incremental_parameter_mismatch",
                "analytical_linear_FLOPs",
                "smaller_hidden_width",
            ],
            "measured_latency_used_for_selection": False,
            "performance_based_termination": False,
        }
    )


__all__ = [
    "ARCHITECTURES",
    "CONTEXTS",
    "NORMALIZATION_MODES",
    "PREDICTOR_ARCHITECTURE_CONTRACT",
    "PREDICTOR_CAPACITY_CONTRACT",
    "RELATION_PARTICLE_ORDER",
    "RetbTokenPredictor",
    "TypedHLTEvidence",
    "UNCERTAINTY_HEADS",
    "build_predictor_architecture_contract",
    "build_predictor_capacity_report",
    "predictor_analytical_flops",
    "profile_predictor",
    "select_widened_resmlp_width",
    "uncertainty_width",
    "validate_predictor_architecture_contract",
]
