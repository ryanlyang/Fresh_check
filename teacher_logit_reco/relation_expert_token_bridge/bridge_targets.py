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


BRIDGE_TARGET_CONTRACT = "retb_bridge_target_modes_v2"
PILOT_ARCHITECTURE_CONTRACT = "retb_pilot_t0_architecture_v2"
TOKEN_NORMALIZER_CONTRACT = "retb_bridge_token_normalizer_v1"
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
            "schema_version": 2,
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
                "unbiased_particle_context_encoder": (
                    "seed_and_shape_matched_HE_BASE4"
                ),
                "hlt_realization": "R_MULTI",
                "predictor": "A3_SLOT_DECODER_DIRECT",
                "context": "C2_ALL",
                "objective": "W_TOKEN_HEAVY",
                "uncertainty": "U_SLOT",
                "normalization": "N_UNCLIPPED",
                "learning_rate": 5.0e-4,
                "dropout": 0.0,
                "effective_batch_size": 256,
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
            "token_normalization": {
                "contract": TOKEN_NORMALIZER_CONTRACT,
                "fit_population": "model_train_targets_only",
                "granularity": "per_expert_per_slot_per_channel",
                "standard_deviation_floor": 1.0e-4,
                "primary": "N_UNCLIPPED",
                "inverse_transform_before_frozen_expert_or_fusion": True,
            },
            "performance_based_termination": False,
        }
    )


def build_pilot_architecture_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PILOT_ARCHITECTURE_CONTRACT,
            "schema_version": 2,
            "architecture": "A3_SLOT_DECODER_DIRECT",
            "context": "C2_ALL",
            "layers": 3,
            "heads": {"D64": 4, "D128": 8},
            "mlp_expansion": 4,
            "registered_A3_dropout": 0.1,
            "PILOT_T0_dropout_override": 0.0,
            "query_initialization": "copy_offline_slot_queries_no_weight_sharing",
            "evidence": [
                "corresponding_HLT_expert_bank",
                "all_seven_HLT_expert_banks",
                "unbiased_HLT_particle_hidden_states",
            ],
            "corresponding_bank_is_separately_typed_and_repeated_in_all_bank_context": True,
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


def fit_bridge_token_normalizer(
    tokens: Any,
    *,
    expert_id: str,
    shape_id: str,
    target_checkpoint_sha256: str,
    token_cache_sha256: str,
    identity_manifest_sha256: str,
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(tokens, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] == 0
        or values.shape[1] not in {1, 2, 4, 8, 16}
        or values.shape[2] not in {64, 128}
        or expert_id not in EXPERT_ORDER
        or not np.isfinite(values).all()
    ):
        raise ValueError("bridge token normalizer population differs")
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    normalized = (values.astype(np.float64) - mean) / np.maximum(std, 1.0e-4)
    absolute = np.abs(normalized)
    return with_content_hash(
        {
            "contract": TOKEN_NORMALIZER_CONTRACT,
            "schema_version": 1,
            "expert_id": expert_id,
            "shape_id": str(shape_id),
            "fit_split": "model_train",
            "event_count": int(len(values)),
            "mean": mean.tolist(),
            "standard_deviation": std.tolist(),
            "standard_deviation_floor": 1.0e-4,
            "zero_variance_channel_count": int(np.sum(std == 0)),
            "tail_counts": {
                str(threshold): int(np.sum(absolute > threshold))
                for threshold in (8, 16, 32)
            },
            "control_clipping_fractions": {
                "N_CLIP16": float(np.mean(absolute > 16)),
                "N_CLIP8": float(np.mean(absolute > 8)),
            },
            "primary_mode": "N_UNCLIPPED",
            "nonfinite_values_permitted": False,
            "target_checkpoint_sha256": require_sha256(
                target_checkpoint_sha256,
                name="target_checkpoint_sha256",
            ),
            "token_cache_sha256": require_sha256(
                token_cache_sha256, name="token_cache_sha256"
            ),
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256,
                name="identity_manifest_sha256",
            ),
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
        target_expert_id: str,
        offline_slot_queries: Any,
        dropout: float = 0.1,
    ) -> None:
        module = _require_torch()
        super().__init__()
        k, d = int(token_count), int(token_dimension)
        if (
            k not in {1, 2, 4, 8, 16}
            or d not in {64, 128}
            or target_expert_id not in EXPERT_ORDER
        ):
            raise ValueError("pilot token shape is not registered")
        if tuple(offline_slot_queries.shape) != (k, d):
            raise ValueError("offline pilot slot-query shape differs")
        self.token_count, self.token_dimension = k, d
        self.target_expert_id = target_expert_id
        self.target_queries = module.nn.Parameter(
            offline_slot_queries.detach().float().clone()
        )
        self.bank_projections = module.nn.ModuleDict(
            {name: module.nn.LazyLinear(d) for name in EXPERT_ORDER}
        )
        self.particle_projection = module.nn.LazyLinear(d)
        self.source_embedding = module.nn.Embedding(3, d)
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
        corresponding = hlt_token_banks[self.target_expert_id]
        if corresponding.ndim != 3 or int(corresponding.shape[1]) > 16:
            raise ValueError("pilot corresponding HLT bank shape differs")
        batch = int(corresponding.shape[0])
        target_index = EXPERT_ORDER.index(self.target_expert_id)
        target_slots = module.arange(
            corresponding.shape[1], device=corresponding.device
        )
        rows.append(
            self.bank_projections[self.target_expert_id](corresponding)
            + self.source_embedding.weight[0]
            + self.expert_embedding.weight[target_index]
            + self.slot_embedding(target_slots)[None]
            + self.kind_embedding.weight[0]
        )
        masks.append(
            module.zeros(
                (batch, corresponding.shape[1]),
                dtype=module.bool,
                device=corresponding.device,
            )
        )
        for expert_index, expert in enumerate(EXPERT_ORDER):
            bank = hlt_token_banks[expert]
            if bank.ndim != 3 or int(bank.shape[1]) > 16:
                raise ValueError("pilot HLT bank shape differs")
            if int(bank.shape[0]) != batch:
                raise ValueError("pilot HLT bank batch differs")
            projected = self.bank_projections[expert](bank)
            slot_ids = module.arange(bank.shape[1], device=bank.device)
            projected = (
                projected
                + self.source_embedding.weight[1]
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
            + self.source_embedding.weight[2]
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


class BridgeCandidatePredictor(
    torch.nn.Module if torch is not None else object
):
    """PILOT_T0-initialized token or logit predictor used by Stage-E targets."""

    def __init__(
        self,
        *,
        pilot: PilotSlotDecoderDirect,
        target_mode: str,
        bridge_dimension: int | None = None,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if target_mode not in {
            "T1_ANCHORED_BRIDGE",
            "T1_TASK_BRIDGE",
            "T2_PROJECT",
            "T3_LOGIT",
        }:
            raise ValueError("bridge candidate predictor mode differs")
        self.pilot = pilot
        self.target_mode = str(target_mode)
        source_dimension = int(pilot.token_dimension)
        if target_mode == "T2_PROJECT":
            if bridge_dimension not in {64, 128}:
                raise ValueError("T2 predictor bridge dimension differs")
            self.output_projection = (
                module.nn.Identity()
                if int(bridge_dimension) == source_dimension
                else module.nn.Linear(source_dimension, int(bridge_dimension))
            )
            output_dimension = int(bridge_dimension)
        else:
            if bridge_dimension is not None:
                raise ValueError("only T2 predictor owns a bridge dimension")
            self.output_projection = module.nn.Identity()
            output_dimension = source_dimension
        self.logit_head = (
            module.nn.Linear(output_dimension, 10)
            if target_mode == "T3_LOGIT"
            else None
        )

    def forward(
        self,
        *,
        hlt_token_banks: Mapping[str, Any],
        unbiased_particle_states: Any,
        particle_mask: Any,
    ) -> dict[str, Any]:
        output = self.pilot(
            hlt_token_banks=hlt_token_banks,
            unbiased_particle_states=unbiased_particle_states,
            particle_mask=particle_mask,
        )
        tokens = self.output_projection(output["predicted_tokens"])
        logits = (
            None
            if self.logit_head is None
            else self.logit_head(tokens.mean(dim=1))
        )
        return {
            "predicted_tokens": tokens,
            "log_variance": output["log_variance"],
            "logits": logits,
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


class BridgeOfflineTarget(torch.nn.Module if torch is not None else object):
    """Registered moving offline target plus its coordinate-specific consumers."""

    def __init__(
        self,
        *,
        target_mode: str,
        target_expert_id: str,
        expert_model: Any,
        candidate_fusion: Any,
        projection: BridgeProjection | None = None,
        projected_expert_head: Any | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        if (
            target_mode
            not in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE", "T2_PROJECT"}
            or target_expert_id not in EXPERT_ORDER
        ):
            raise ValueError("moving bridge target mode/expert differs")
        if target_mode == "T2_PROJECT":
            if projection is None or projected_expert_head is None:
                raise ValueError("T2 target requires projection and new expert head")
        elif projection is not None or projected_expert_head is not None:
            raise ValueError("only T2 owns projection consumers")
        self.target_mode = target_mode
        self.target_expert_id = target_expert_id
        self.expert_model = expert_model
        self.candidate_fusion = candidate_fusion
        self.projection = projection
        self.projected_expert_head = projected_expert_head

    def configure_bridge_trainability(
        self, *, unfreeze_final_two_blocks: bool
    ) -> tuple[str, ...]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        trainable_modules = [self.candidate_fusion]
        if self.target_mode == "T2_PROJECT":
            if unfreeze_final_two_blocks:
                raise ValueError("T2 cannot unfreeze offline particle blocks")
            trainable_modules.extend([self.projection, self.projected_expert_head])
        else:
            trainable_modules.extend(
                [self.expert_model.tokenizer, self.expert_model.head]
            )
            if unfreeze_final_two_blocks:
                blocks = self.expert_model.particle_encoder.mod.blocks
                if len(blocks) != 8:
                    raise ValueError("bridge target particle block count differs")
                trainable_modules.extend([blocks[-2], blocks[-1]])
        for child in trainable_modules:
            for parameter in child.parameters():
                parameter.requires_grad_(True)
        names = tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
        if not names:
            raise ValueError("moving bridge target has no trainable parameters")
        return names

    def train(self, mode: bool = True) -> Any:
        super().train(mode)
        if self.target_mode == "T2_PROJECT":
            self.expert_model.eval()
        return self

    def forward(
        self,
        *,
        offline_batch: Mapping[str, Any],
        other_t0_banks: Mapping[str, Any],
    ) -> dict[str, Any]:
        if set(other_t0_banks) != set(EXPERT_ORDER) - {
            self.target_expert_id
        }:
            raise ValueError("moving bridge target other-bank coverage differs")
        if self.target_mode == "T2_PROJECT":
            with _require_torch().no_grad():
                details = self.expert_model(
                    return_details=True, **offline_batch
                )
            pure_tokens = details["tokens"].detach()
            moving_tokens = self.projection(pure_tokens)
            expert_logits = self.projected_expert_head(moving_tokens)
            decoded_tokens = self.projection.decode(moving_tokens)
        else:
            details = self.expert_model(return_details=True, **offline_batch)
            pure_tokens = None
            moving_tokens = details["tokens"]
            expert_logits = details["logits"]
            decoded_tokens = None
        banks = dict(other_t0_banks)
        banks[self.target_expert_id] = moving_tokens
        fusion_logits = self.candidate_fusion(token_banks=banks)
        return {
            "moving_tokens": moving_tokens,
            "expert_logits": expert_logits,
            "fusion_logits": fusion_logits,
            "pure_tokens": pure_tokens,
            "decoded_tokens": decoded_tokens,
        }


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


def directional_token_loss(prediction: Any, target: Any) -> Any:
    module = _require_torch()
    pred = module.nn.functional.normalize(prediction.float(), dim=-1, eps=1.0e-8)
    truth = module.nn.functional.normalize(target.float(), dim=-1, eps=1.0e-8)
    return (1.0 - (pred * truth).sum(dim=-1)).mean()


def token_relation_loss(prediction: Any, target: Any) -> Any:
    module = _require_torch()
    pred = module.nn.functional.normalize(prediction.float(), dim=-1, eps=1.0e-8)
    truth = module.nn.functional.normalize(target.float(), dim=-1, eps=1.0e-8)
    return (pred @ pred.transpose(-1, -2) - truth @ truth.transpose(-1, -2)).square().mean()


def pilot_t0_objective(
    *,
    predicted_tokens: Any,
    target_tokens: Any,
    log_variance: Any,
    predicted_expert_logits: Any,
    target_expert_logits: Any,
    predicted_hybrid_logits: Any,
    target_hybrid_logits: Any,
    labels: Any,
) -> tuple[Any, dict[str, Any]]:
    """Exact W_TOKEN_HEAVY columns: token, cosine, relation, expertKD, swapKD, CE."""
    module = _require_torch()
    pieces = {
        "token": heteroscedastic_huber_loss(
            predicted_tokens, target_tokens.detach(), log_variance
        ),
        "cosine": directional_token_loss(
            predicted_tokens, target_tokens.detach()
        ),
        "relation": token_relation_loss(
            predicted_tokens, target_tokens.detach()
        ),
        "expertKD": temperature_two_kl(
            predicted_expert_logits, target_expert_logits
        ),
        "swapKD": temperature_two_kl(
            predicted_hybrid_logits, target_hybrid_logits
        ),
        "CE": module.nn.functional.cross_entropy(
            predicted_hybrid_logits, labels.long()
        ),
    }
    weights = {
        "token": 1.0,
        "cosine": 0.25,
        "relation": 0.10,
        "expertKD": 0.25,
        "swapKD": 0.25,
        "CE": 0.10,
    }
    total = sum(weights[name] * pieces[name] for name in weights)
    if not bool(module.isfinite(total)):
        raise FloatingPointError("PILOT_T0 objective is nonfinite")
    return total, {name: value.detach() for name, value in pieces.items()}


def bridge_target_objective(
    *,
    target_mode: str,
    offline_expert_loss: Any,
    token_prediction_loss: Any,
    offline_fusion_loss: Any,
    t0_logit_loss: Any,
    lambda_pred: float,
    anchor_loss: Any | None = None,
    retrieval_loss: Any | None = None,
    covariance_loss: Any | None = None,
    t0_project_loss: Any | None = None,
    decoded_t0_logit_loss: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    if target_mode not in {
        "T1_ANCHORED_BRIDGE",
        "T1_TASK_BRIDGE",
        "T2_PROJECT",
    }:
        raise ValueError("bridge target objective mode is not trainable tokens")
    if float(lambda_pred) not in LAMBDA_PRED_VALUES:
        raise ValueError("bridge target prediction weight is not registered")
    common = (
        offline_expert_loss
        + float(lambda_pred) * token_prediction_loss
        + 0.50 * offline_fusion_loss
    )
    if target_mode in {"T1_ANCHORED_BRIDGE", "T1_TASK_BRIDGE"}:
        total = common + 0.50 * t0_logit_loss
        if target_mode == "T1_ANCHORED_BRIDGE":
            if any(
                value is None
                for value in (anchor_loss, retrieval_loss, covariance_loss)
            ):
                raise ValueError("anchored bridge objective lacks content losses")
            total = (
                total
                + 0.25 * anchor_loss
                + 0.10 * retrieval_loss
                + 0.05 * covariance_loss
            )
    else:
        if t0_project_loss is None or decoded_t0_logit_loss is None:
            raise ValueError("T2 objective lacks decoder preservation losses")
        total = (
            common
            + 0.25 * t0_project_loss
            + 0.50 * decoded_t0_logit_loss
        )
    if not bool(_require_torch().isfinite(total)):
        raise FloatingPointError("bridge target objective is nonfinite")
    return total, {
        "target_mode": target_mode,
        "lambda_pred": float(lambda_pred),
        "total": total.detach(),
    }


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
    moving32, pure32 = moving.float(), pure.detach().float()
    count = module.tensor(
        float(moving.shape[0]), device=moving.device, dtype=module.float32
    )
    moving_sum = moving32.sum(dim=0)
    moving_outer = module.einsum("bkd,bke->kde", moving32, moving32)
    pure_sum = pure32.sum(dim=0)
    pure_outer = module.einsum("bkd,bke->kde", pure32, pure32)
    distributed = (
        module.distributed.is_available()
        and module.distributed.is_initialized()
    )
    if distributed:
        # Autograd-aware collectives preserve gradients through the moving
        # sufficient statistics. The detached T0 statistics and count use the
        # ordinary collective.
        from torch.distributed.nn.functional import all_reduce

        moving_sum = all_reduce(moving_sum)
        moving_outer = all_reduce(moving_outer)
        module.distributed.all_reduce(pure_sum)
        module.distributed.all_reduce(pure_outer)
        module.distributed.all_reduce(count)
    if float(count.detach().cpu()) < 2.0:
        raise ValueError("bridge covariance requires global effective batch >=2")
    moving_mean = moving_sum / count
    pure_mean = pure_sum / count
    moving_cov = (
        moving_outer / count
        - module.einsum("kd,ke->kde", moving_mean, moving_mean)
    )
    pure_cov = (
        pure_outer / count
        - module.einsum("kd,ke->kde", pure_mean, pure_mean)
    )
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
    unbiased_particle_encoder_checkpoint_sha256: str,
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
                "unbiased_HLT_particle_encoder_checkpoint": require_sha256(
                    unbiased_particle_encoder_checkpoint_sha256,
                    name="unbiased_particle_encoder_checkpoint_sha256",
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
    "BridgeOfflineTarget",
    "BridgeCandidatePredictor",
    "BridgeProjection",
    "PilotSlotDecoderDirect",
    "TARGET_MODES",
    "alternating_bridge_update",
    "build_bridge_candidate_contract",
    "build_bridge_target_contract",
    "build_pilot_architecture_contract",
    "bridge_target_objective",
    "directional_token_loss",
    "deterministic_within_class_negatives",
    "fp32_cosine_similarity",
    "fit_bridge_token_normalizer",
    "heteroscedastic_huber_loss",
    "normalized_huber_anchor",
    "relative_slot_covariance_loss",
    "temperature_two_kl",
    "pilot_t0_objective",
    "token_relation_loss",
    "within_class_retrieval_loss",
]
