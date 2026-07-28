"""Locked Stage-E pilot and bridge-target models, losses, and update rules."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, with_content_hash
from .fusion import RMSNorm
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


BRIDGE_TARGET_CONTRACT = "retb_bridge_target_modes_v1"
PILOT_ARCHITECTURE_CONTRACT = "retb_pilot_t0_architecture_v1"
TARGET_MODES = (
    "T0_PURE",
    "T1_ANCHORED_BRIDGE",
    "T1_TASK_BRIDGE",
    "T2_PROJECT",
    "T3_LOGIT",
)
LAMBDA_PRED_VALUES = (0.05, 0.10, 0.25)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for bridge targets")
    return torch


def build_bridge_target_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": BRIDGE_TARGET_CONTRACT,
            "schema_version": 1,
            "modes": {
                "T0_PURE": {
                    "semantics": "selected_frozen_offline_expert_tokens",
                    "hlt_objective": False,
                },
                "T1_ANCHORED_BRIDGE": {
                    "semantics": "instance_anchored_co_designed_tokens",
                    "representation_claim_requires_content_certification": True,
                },
                "T1_TASK_BRIDGE": {
                    "semantics": "co_designed_task_tokens",
                    "T0_coordinate_claim": False,
                    "representation_claim_requires_content_certification": True,
                },
                "T2_PROJECT": {
                    "semantics": "learned_predictable_bridge_coordinate",
                    "T0_coordinate_claim": False,
                    "bridge_dimensions": [64, 128],
                },
                "T3_LOGIT": {
                    "semantics": "direct_offline_logit_distillation",
                    "token_fidelity_claim": False,
                },
            },
            "lambda_pred_candidates": list(LAMBDA_PRED_VALUES),
            "pilot": {
                "id": "PILOT_T0",
                "hlt_encoder": "seed_matched_HE_OFFLINE_INIT",
                "hlt_realization": "R_MULTI",
                "predictor": "A3_SLOT_DECODER_DIRECT",
                "context": "C2_ALL",
                "objective": "W_TOKEN_HEAVY",
                "uncertainty": "U_SLOT",
                "normalization": "N_UNCLIPPED",
                "learning_rate": 5.0e-4,
                "dropout": 0.0,
                "selected_per_target": False,
            },
            "alternating_updates": {
                "predictor_update": "offline_target_detached",
                "offline_target_update": "predictor_detached",
                "starts_from": "copy_of_PILOT_T0",
            },
            "T1_common_weights": {
                "offline_fusion_CE": 0.50,
                "T0_logit_KL": 0.50,
            },
            "T1_anchor_weights": {
                "normalized_T0_Huber": 0.25,
                "within_class_retrieval": 0.10,
                "relative_slot_covariance": 0.05,
            },
            "T2_weights": {
                "projected_fusion_CE": 0.50,
                "T0_reconstruction": 0.25,
                "decoded_T0_logit_KL": 0.50,
            },
            "performance_based_termination": False,
        }
    )


def build_pilot_architecture_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PILOT_ARCHITECTURE_CONTRACT,
            "schema_version": 1,
            "architecture": "A3_SLOT_DECODER_DIRECT",
            "context": "C2_ALL",
            "layers": 3,
            "heads": {"D64": 4, "D128": 8},
            "mlp_expansion": 4,
            "dropout": 0.1,
            "query_initialization": "copy_offline_slot_queries_no_weight_sharing",
            "evidence": [
                "corresponding_HLT_expert_bank",
                "all_seven_HLT_expert_banks",
                "unbiased_HLT_particle_hidden_states",
            ],
            "embeddings": [
                "source_type",
                "expert_type",
                "slot_index",
                "particle_vs_summary",
            ],
            "uncertainty": {
                "head": "U_SLOT",
                "shape": "B_K_1",
                "log_variance_clip": [-8.0, 4.0],
            },
        }
    )


class PilotSlotDecoderDirect(
    torch.nn.Module if torch is not None else object
):
    """Exact A3/C2 pilot decoder over typed HLT evidence sequences."""

    def __init__(
        self,
        *,
        token_count: int,
        token_dimension: int,
        offline_slot_queries: Any,
        dropout: float = 0.1,
    ) -> None:
        module = _require_torch()
        super().__init__()
        k, d = int(token_count), int(token_dimension)
        if k not in {1, 2, 4, 8, 16} or d not in {64, 128}:
            raise ValueError("pilot token shape is not registered")
        if tuple(offline_slot_queries.shape) != (k, d):
            raise ValueError("offline pilot slot-query shape differs")
        self.token_count, self.token_dimension = k, d
        self.target_queries = module.nn.Parameter(
            offline_slot_queries.detach().float().clone()
        )
        self.bank_projections = module.nn.ModuleDict(
            {name: module.nn.LazyLinear(d) for name in EXPERT_ORDER}
        )
        self.particle_projection = module.nn.LazyLinear(d)
        self.source_embedding = module.nn.Embedding(2, d)
        self.expert_embedding = module.nn.Embedding(len(EXPERT_ORDER), d)
        self.slot_embedding = module.nn.Embedding(16, d)
        self.kind_embedding = module.nn.Embedding(2, d)
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
        self.log_variance = module.nn.Linear(d, 1)

    def _evidence(
        self,
        hlt_token_banks: Mapping[str, Any],
        unbiased_particle_states: Any,
        particle_mask: Any,
    ) -> tuple[Any, Any]:
        module = _require_torch()
        if set(hlt_token_banks) != set(EXPERT_ORDER):
            raise ValueError("pilot HLT expert coverage differs")
        rows, masks = [], []
        batch = None
        for expert_index, expert in enumerate(EXPERT_ORDER):
            bank = hlt_token_banks[expert]
            if bank.ndim != 3 or int(bank.shape[1]) > 16:
                raise ValueError("pilot HLT bank shape differs")
            batch = int(bank.shape[0]) if batch is None else batch
            if int(bank.shape[0]) != batch:
                raise ValueError("pilot HLT bank batch differs")
            projected = self.bank_projections[expert](bank)
            slot_ids = module.arange(bank.shape[1], device=bank.device)
            projected = (
                projected
                + self.source_embedding.weight[0]
                + self.expert_embedding.weight[expert_index]
                + self.slot_embedding(slot_ids)[None]
                + self.kind_embedding.weight[0]
            )
            rows.append(projected)
            masks.append(
                module.zeros(
                    (batch, bank.shape[1]),
                    dtype=module.bool,
                    device=bank.device,
                )
            )
        if (
            unbiased_particle_states.ndim != 3
            or tuple(particle_mask.shape)
            != tuple(unbiased_particle_states.shape[:2])
            or int(unbiased_particle_states.shape[0]) != batch
        ):
            raise ValueError("pilot particle evidence shape differs")
        particles = (
            self.particle_projection(unbiased_particle_states)
            + self.source_embedding.weight[0]
            + self.kind_embedding.weight[1]
        )
        rows.append(particles)
        masks.append(~particle_mask.bool())
        return module.cat(rows, dim=1), module.cat(masks, dim=1)

    def forward(
        self,
        *,
        hlt_token_banks: Mapping[str, Any],
        unbiased_particle_states: Any,
        particle_mask: Any,
    ) -> dict[str, Any]:
        module = _require_torch()
        evidence, padding_mask = self._evidence(
            hlt_token_banks, unbiased_particle_states, particle_mask
        )
        queries = self.target_queries[None].expand(evidence.shape[0], -1, -1)
        decoded = self.output_norm(
            self.decoder(
                queries,
                evidence,
                memory_key_padding_mask=padding_mask,
            )
        )
        log_variance = self.log_variance(decoded).clamp(-8.0, 4.0)
        if not bool(
            module.isfinite(decoded).all()
            and module.isfinite(log_variance).all()
        ):
            raise FloatingPointError("pilot output is nonfinite")
        return {
            "predicted_tokens": decoded,
            "log_variance": log_variance,
            "gate": None,
        }


class BridgeProjection(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dimension: int, bridge_dimension: int) -> None:
        module = _require_torch()
        super().__init__()
        source, bridge = int(input_dimension), int(bridge_dimension)
        if source not in {64, 128} or bridge not in {64, 128}:
            raise ValueError("T2 projection dimensions are not registered")
        self.input_dimension, self.bridge_dimension = source, bridge
        self.input_norm = RMSNorm(source)
        if source == bridge:
            self.base = module.nn.Identity()
        else:
            self.base = module.nn.Linear(source, bridge)
        self.residual_norm = RMSNorm(bridge)
        self.up = module.nn.Linear(bridge, 2 * bridge)
        self.down = module.nn.Linear(2 * bridge, bridge)
        self.decoder = module.nn.Linear(bridge, source)

    def forward(self, values: Any) -> Any:
        module = _require_torch()
        normalized = self.input_norm(values)
        base = values if self.input_dimension == self.bridge_dimension else self.base(
            normalized
        )
        return base + self.down(module.nn.functional.gelu(self.up(self.residual_norm(base))))

    def decode(self, bridge: Any) -> Any:
        return self.decoder(bridge)


def heteroscedastic_huber_loss(
    prediction: Any, target: Any, log_variance: Any
) -> Any:
    module = _require_torch()
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("pilot prediction/target shape differs")
    if tuple(log_variance.shape) != tuple(prediction.shape[:2]) + (1,):
        raise ValueError("U_SLOT log-variance shape differs")
    error = module.nn.functional.huber_loss(
        prediction, target, delta=0.5, reduction="none"
    )
    return (module.exp(-log_variance) * error + log_variance).mean()


def temperature_two_kl(student_logits: Any, teacher_logits: Any) -> Any:
    module = _require_torch()
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError("KL logit shapes differ")
    temperature = 2.0
    return module.nn.functional.kl_div(
        module.log_softmax(student_logits / temperature, dim=-1),
        module.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2


def normalized_huber_anchor(moving: Any, pure: Any) -> Any:
    return _require_torch().nn.functional.huber_loss(
        moving.float(), pure.detach().float(), delta=0.5
    )


def relative_slot_covariance_loss(moving: Any, pure: Any) -> Any:
    module = _require_torch()
    if tuple(moving.shape) != tuple(pure.shape) or moving.ndim != 3:
        raise ValueError("bridge covariance bank shapes differ")
    if int(moving.shape[0]) < 2:
        raise ValueError("bridge covariance requires effective batch >=2")
    moving32, pure32 = moving.float(), pure.detach().float()
    moving_centered = moving32 - moving32.mean(dim=0, keepdim=True)
    pure_centered = pure32 - pure32.mean(dim=0, keepdim=True)
    moving_cov = module.einsum(
        "bkd,bke->kde", moving_centered, moving_centered
    ) / moving.shape[0]
    pure_cov = module.einsum(
        "bkd,bke->kde", pure_centered, pure_centered
    ) / pure.shape[0]
    numerator = (moving_cov - pure_cov).square().sum((-2, -1)).sqrt()
    denominator = pure_cov.square().sum((-2, -1)).sqrt().clamp_min(1.0e-8)
    return (numerator / denominator).mean()


def _identity_seed(namespace: str, pipeline_seed: int, identity: str) -> int:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(int(pipeline_seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(identity).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def deterministic_within_class_negatives(
    *,
    identity: str,
    class_label: int,
    class_rings: Mapping[int, Sequence[str]],
    pipeline_seed: int,
    certification: bool,
    count: int = 31,
) -> tuple[str, ...]:
    ring = sorted(set(str(value) for value in class_rings[int(class_label)]))
    ring = [value for value in ring if value != str(identity)]
    if len(ring) < int(count):
        raise ValueError("within-class identity ring has fewer than 32 identities")
    namespace = (
        "retb_t1_cert_negatives_v1"
        if certification
        else "retb_t1_negatives_v1"
    )
    start = _identity_seed(namespace, pipeline_seed, str(identity)) % len(ring)
    return tuple(ring[(start + offset) % len(ring)] for offset in range(count))


def fp32_cosine_similarity(query: Any, candidates: Any) -> Any:
    module = _require_torch()
    q = query.float().reshape(-1)
    c = candidates.float().reshape(candidates.shape[0], -1)
    qnorm = module.linalg.vector_norm(q)
    cnorm = module.linalg.vector_norm(c, dim=1)
    dot = c @ q
    denominator = qnorm * cnorm
    return module.where(
        denominator == 0,
        module.zeros_like(dot),
        dot / denominator.clamp_min(1.0e-8),
    )


def within_class_retrieval_loss(
    predicted_queries: Any,
    positive_and_negative_banks: Any,
) -> Any:
    module = _require_torch()
    if (
        predicted_queries.ndim != 3
        or positive_and_negative_banks.ndim != 4
        or positive_and_negative_banks.shape[1] != 32
        or tuple(predicted_queries.shape[1:])
        != tuple(positive_and_negative_banks.shape[2:])
    ):
        raise ValueError("within-class retrieval tensors differ")
    rows = []
    for query, candidates in zip(
        predicted_queries, positive_and_negative_banks
    ):
        scores = fp32_cosine_similarity(query, candidates) / 0.1
        rows.append(module.logsumexp(scores.float(), dim=0) - scores[0])
    return module.stack(rows).mean()


def alternating_bridge_update(
    *,
    phase: str,
    predictor_loss: Any,
    target_loss: Any,
    predictor_optimizer: Any,
    target_optimizer: Any,
) -> dict[str, Any]:
    """Apply exactly one phase while proving the opposite graph is detached."""
    module = _require_torch()
    if phase == "predictor":
        predictor_optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        predictor_loss.backward()
        active, inactive = predictor_optimizer, target_optimizer
    elif phase == "offline_target":
        predictor_optimizer.zero_grad(set_to_none=True)
        target_optimizer.zero_grad(set_to_none=True)
        target_loss.backward()
        active, inactive = target_optimizer, predictor_optimizer
    else:
        raise ValueError("bridge alternating phase is unknown")
    inactive_gradients = [
        parameter.grad
        for group in inactive.param_groups
        for parameter in group["params"]
    ]
    if any(gradient is not None for gradient in inactive_gradients):
        raise RuntimeError("alternating bridge update leaked into detached graph")
    norm = module.nn.utils.clip_grad_norm_(
        [
            parameter
            for group in active.param_groups
            for parameter in group["params"]
        ],
        1.0,
    )
    if not bool(module.isfinite(norm)):
        raise FloatingPointError("alternating bridge gradient is nonfinite")
    active.step()
    return {
        "phase": phase,
        "opposite_graph_detached": True,
        "gradient_norm": float(norm.detach().cpu()),
    }


def build_bridge_candidate_contract(
    *,
    target_mode: str,
    pipeline_seed: int,
    expert_id: str,
    shape_id: str,
    lambda_pred: float,
    t0_checkpoint_sha256: str,
    hlt_encoder_checkpoint_sha256: str,
    pilot_checkpoint_sha256: str,
    bridge_dimension: int | None = None,
    unfreeze_final_two_blocks: bool = False,
) -> dict[str, Any]:
    if target_mode not in TARGET_MODES or expert_id not in EXPERT_ORDER:
        raise ValueError("bridge candidate identity is unknown")
    if target_mode in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE", "T2_PROJECT"}:
        if float(lambda_pred) not in LAMBDA_PRED_VALUES:
            raise ValueError("bridge candidate lambda_pred is not registered")
    elif float(lambda_pred) != 0.0:
        raise ValueError("T0/T3 cannot use token-prediction weight")
    if target_mode == "T2_PROJECT":
        if bridge_dimension not in {64, 128}:
            raise ValueError("T2 bridge dimension is not registered")
    elif bridge_dimension is not None:
        raise ValueError("only T2 owns a bridge dimension")
    return with_content_hash(
        {
            "contract": "retb_bridge_target_candidate_v1",
            "schema_version": 1,
            "target_mode": target_mode,
            "pipeline_seed": int(pipeline_seed),
            "expert_id": expert_id,
            "shape_id": str(shape_id),
            "lambda_pred": float(lambda_pred),
            "bridge_dimension": bridge_dimension,
            "unfreeze": (
                "summary_tokenizer_plus_final_two_particle_blocks_at_0p1_lr"
                if unfreeze_final_two_blocks
                else "summary_tokenizer_and_consumers_only"
            ),
            "parents": {
                "T0_checkpoint": require_sha256(
                    t0_checkpoint_sha256, name="t0_checkpoint_sha256"
                ),
                "HLT_encoder_checkpoint": require_sha256(
                    hlt_encoder_checkpoint_sha256,
                    name="hlt_encoder_checkpoint_sha256",
                ),
                "initial_PILOT_T0_checkpoint": require_sha256(
                    pilot_checkpoint_sha256,
                    name="pilot_checkpoint_sha256",
                ),
            },
            "offline_forward_uses_offline_particles_only": True,
            "paired_HLT_appears_only_in_training_losses": True,
            "performance_based_termination": False,
        }
    )


__all__ = [
    "BridgeProjection",
    "PilotSlotDecoderDirect",
    "TARGET_MODES",
    "alternating_bridge_update",
    "build_bridge_candidate_contract",
    "build_bridge_target_contract",
    "build_pilot_architecture_contract",
    "deterministic_within_class_negatives",
    "fp32_cosine_similarity",
    "heteroscedastic_huber_loss",
    "normalized_huber_anchor",
    "relative_slot_covariance_loss",
    "temperature_two_kl",
    "within_class_retrieval_loss",
]
