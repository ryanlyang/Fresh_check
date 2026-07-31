"""Exact HOSD Stage-C baseline definitions and factories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping

from jetclass_fresh.heterogeneous_hlt import ParticleNetHLTClassifier
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.relational_part.model import exact_rpt_base_config

from .contracts import (
    BASELINE_REGISTRY_CONTRACT,
    STAGE_C_PLAN_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .taps import HBaseParticleTransformer, TAP_BLOCKS
from .heads import GlobalTargetHead

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


BASELINE_IDS = (
    "H_BASE",
    "H_BASE_BEAM_BUDGET",
    "H_BASE_LONG",
    "H_PARTICLENET",
    "H_KD_LOGIT_O_BASE",
    "H_KD_LOGIT_O_FULLREL",
    "H_NATIVE_REL_AUX",
)
PROBE_TARGET_PREFIXES = (
    "T_OFFLINE_JET_",
    "T_OFFLINE_COMPOSITION_",
    "T_OFFLINE_TRACK_",
    "T_OFFLINE_DENSITY_",
    "T_OFFLINE_CA_TREE_",
    "T_OFFLINE_TRACK_COMPONENT_PROXY_",
    "T_OFFLINE_RELATION_",
    "T_OFFLINE_LOGITS_",
    "T_OFFLINE_POOLED_LATENT",
    "T_HLT_TRACK_PAIR_",
    "T_HLT_REGION_PAIR_",
)


@dataclass(frozen=True)
class HOSDTrainingProtocol:
    maximum_epochs: int = 40
    base_learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 128
    num_workers: int = 0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("unknown HOSD training profile")
        allowed_epochs = {5, 40, 80} if self.campaign_profile == "production" else {2}
        if self.maximum_epochs not in allowed_epochs:
            raise ValueError("HOSD Stage-C epoch budget differs from its profile")
        if self.microbatch_size * self.gradient_accumulation_steps != 128:
            raise ValueError("HOSD effective batch size drifted")
        expected = {
            "base_learning_rate": 1e-3,
            "minimum_learning_rate": 1e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "microbatch_size": 64,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 128,
            "num_workers": 0,
        }
        drift = {
            key: (getattr(self, key), value)
            for key, value in expected.items()
            if getattr(self, key) != value
        }
        if drift:
            raise ValueError(f"HOSD training protocol drifted: {drift}")

    def as_record(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "optimizer": "AdamW",
            "warmup_updates": "min(T,max(1,ceil(0.05*T)))",
            "schedule": "one_based_linear_warmup_then_cosine",
            "precision": "GH200_BF16;authoritative_parity_FP32",
            "checkpoint_selector": (
                "maximum_val_stop_balanced_accuracy_then_lower_CE_then_earlier_epoch"
            ),
            "fixed_epoch_budget": True,
            "early_stopping": False,
            "performance_based_termination": False,
            "production_protocol": self.campaign_profile == "production",
        }


def component_seed(
    campaign_seed: int, component_role: str, canonical_graph_id: str
) -> int:
    payload = (
        f"hosd_component_seed_v1||{int(campaign_seed)}||"
        f"{component_role}||{canonical_graph_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def particle_net_config() -> dict[str, Any]:
    return {
        "architecture": "pn",
        "input_dims": len(PF_FEATURE_NAMES),
        "num_classes": 10,
        "conv_params": [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ],
        "fc_params": [(256, 0.1)],
        "use_fusion": False,
        "use_fts_bn": True,
        "use_counts": True,
        "for_inference": False,
    }


def build_baseline_registry(*, source: Mapping[str, Any]) -> dict[str, Any]:
    hbase = exact_rpt_base_config()
    rows = [
        {
            "baseline_id": "H_BASE",
            "architecture": "exact_standard_four_H_BASE",
            "epochs": 40,
            "teacher_id": None,
        },
        {
            "baseline_id": "H_BASE_BEAM_BUDGET",
            "architecture": "exact_standard_four_H_BASE",
            "epochs": 5,
            "teacher_id": None,
            "selection_eligible": False,
            "purpose": "fixed_five_epoch_stage_f_beam_root",
        },
        {
            "baseline_id": "H_BASE_LONG",
            "architecture": "exact_standard_four_H_BASE",
            "epochs": 80,
            "teacher_id": None,
        },
        {
            "baseline_id": "H_PARTICLENET",
            "architecture": "exact_matched_input_ParticleNet",
            "epochs": 40,
            "teacher_id": None,
        },
        {
            "baseline_id": "H_KD_LOGIT_O_BASE",
            "architecture": "exact_standard_four_H_BASE",
            "epochs": 40,
            "teacher_id": "O_BASE",
        },
        {
            "baseline_id": "H_KD_LOGIT_O_FULLREL",
            "architecture": "exact_standard_four_H_BASE",
            "epochs": 40,
            "teacher_id": "O_FULLREL",
        },
        {
            "baseline_id": "H_NATIVE_REL_AUX",
            "architecture": "exact_standard_four_H_BASE_with_training_only_head",
            "epochs": 40,
            "teacher_id": None,
        },
    ]
    return with_content_hash(
        {
            "contract": BASELINE_REGISTRY_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "baseline_order": list(BASELINE_IDS),
            "baselines": rows,
            "h_base_config": hbase,
            "particle_net_config": particle_net_config(),
            "kd": {
                "temperature": 2.0,
                "cross_entropy_weight": 1.0,
                "kl_weight": 1.0,
                "temperature_squared_multiplier": 4.0,
                "reductions": "ordinary_per_event_means",
                "teacher_logits_detached": True,
            },
            "native_relation_auxiliary": {
                "families": ["BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"],
                "canonical_concatenated_dimension": 545,
                "tap": "TAP_LATE",
                "parameterization": "ABS",
                "loss": "normalized_huber_delta_1",
                "fixed_weight": 0.30,
                "head_discarded_at_deployable_inference": True,
            },
            "training_40": HOSDTrainingProtocol().as_record(),
            "training_5": HOSDTrainingProtocol(maximum_epochs=5).as_record(),
            "training_80": HOSDTrainingProtocol(maximum_epochs=80).as_record(),
            "seed": 101,
            "hlt_realization_policy": "R_MULTI",
            "robustness_replicas": [0, 1, 2, 3],
            "robustness_aggregation": (
                "arithmetic_mean_of_separate_per_replica_metrics_no_logit_averaging"
            ),
        }
    )


def validate_baseline_registry(
    payload: Mapping[str, Any], *, source: Mapping[str, Any] | None = None
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=BASELINE_REGISTRY_CONTRACT
    )
    if source is not None and payload.get("source") != dict(source):
        raise ValueError("baseline registry source differs")
    expected = build_baseline_registry(source=payload["source"])
    if dict(payload) != expected:
        raise ValueError("HOSD baseline registry differs from the exact contract")
    return digest


def build_baseline_model(
    baseline_id: str, *, weaver_module: Any | None = None
) -> Any:
    if baseline_id not in BASELINE_IDS:
        raise ValueError("unknown HOSD baseline")
    if baseline_id == "H_PARTICLENET":
        cfg = particle_net_config()
        cfg.pop("architecture")
        return ParticleNetHLTClassifier(**cfg)
    base = HBaseParticleTransformer(weaver_module=weaver_module)
    if baseline_id == "H_NATIVE_REL_AUX":
        return NativeRelationAuxClassifier(base)
    return base


class NativeRelationAuxClassifier(torch.nn.Module if torch is not None else object):
    """Training-only seven-family HLT relation-summary auxiliary control."""

    target_dimension = 20 + 50 + 100 + 48 + 24 + 110 + 193

    def __init__(self, classifier: HBaseParticleTransformer) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD baselines")
        super().__init__()
        self.classifier = classifier
        self.target_head = GlobalTargetHead(
            self.target_dimension, availability_groups=7
        )

    def forward(self, points: Any, features: Any, lorentz_vectors: Any, mask: Any) -> Any:
        return self.classifier(points, features, lorentz_vectors, mask)

    def forward_with_aux(
        self, points: Any, features: Any, lorentz_vectors: Any, mask: Any
    ) -> tuple[Any, Mapping[str, Any]]:
        result = self.classifier.forward_with_taps(
            points,
            features,
            lorentz_vectors,
            mask,
            capture=("TAP_LATE",),
        )
        prediction = self.target_head(
            result.states["TAP_LATE"], result.masks["TAP_LATE"]
        )
        return result.logits, prediction


def classification_kd_loss(
    student_logits: Any,
    labels: Any,
    teacher_logits: Any | None = None,
    *,
    temperature: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for HOSD baseline losses")
    ce = torch.nn.functional.cross_entropy(student_logits, labels.long())
    if teacher_logits is None:
        return ce, {"cross_entropy": ce.detach(), "temperature_2_kl": None}
    if tuple(teacher_logits.shape) != tuple(student_logits.shape):
        raise ValueError("teacher and student logits differ in shape")
    teacher_probability = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probability = torch.log_softmax(student_logits / temperature, dim=-1)
    per_event = torch.nn.functional.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=-1)
    kd = temperature * temperature * per_event.mean()
    return ce + kd, {
        "cross_entropy": ce.detach(),
        "temperature_2_kl": kd.detach(),
    }


def _probe_target_rows(target_registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    validate_content_hash(
        target_registry, expected_contract="hosd_structure_target_registry_v1"
    )
    rows = []
    for row in target_registry["targets"]:
        target_id = str(row["target_id"])
        if (
            bool(row["executable_current_source"])
            and any(target_id.startswith(prefix) for prefix in PROBE_TARGET_PREFIXES)
            and not target_id.startswith("T_HLT_SELF_")
        ):
            rows.append(row)
    if len(rows) != 18:
        raise ValueError(f"Stage C requires exactly 18 probe target families, got {len(rows)}")
    return rows


def build_stage_c_plan(
    *,
    campaign_spec_sha256: str,
    target_registry: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_baseline_registry(baseline_registry, source=source)
    targets = _probe_target_rows(target_registry)
    probe_rows = []
    raw_prefixes = (
        "T_OFFLINE_JET_",
        "T_OFFLINE_COMPOSITION_",
        "T_OFFLINE_TRACK_",
        "T_OFFLINE_DENSITY_",
        "T_OFFLINE_CA_TREE_",
        "T_OFFLINE_TRACK_COMPONENT_PROXY_",
        "T_OFFLINE_RELATION_",
    )
    for target in targets:
        target_id = str(target["target_id"])
        # One statistics job emits both required train-only statistical
        # references; this keeps the frozen hard bound at nine rows/family.
        probe_rows.append(
            {
                "row_id": f"{target_id}__P_STATISTICAL_REFERENCES",
                "target_id": target_id,
                "probe_kind": "P_STATISTICAL_REFERENCES",
                "emits": ["P_PRIOR", "P_CLASS_CONDITIONAL_ORACLE"],
                "tap": None,
                "deployable": False,
                "selection_eligible": False,
            }
        )
        if target_id.startswith(raw_prefixes):
            probe_rows.append(
                {
                    "row_id": f"{target_id}__P_RAW_MLP",
                    "target_id": target_id,
                    "probe_kind": "P_RAW_MLP",
                    "tap": None,
                    "deployable": True,
                    "selection_eligible": False,
                }
            )
        for probe_kind in ("P_LINEAR", "P_SHALLOW"):
            for tap in TAP_BLOCKS:
                probe_rows.append(
                    {
                        "row_id": f"{target_id}__{probe_kind}__{tap}",
                        "target_id": target_id,
                        "probe_kind": probe_kind,
                        "tap": tap,
                        "deployable": True,
                        "selection_eligible": False,
                    }
                )
        probe_rows.append(
            {
                "row_id": f"{target_id}__P_TARGET_TO_CLASS_ORACLE",
                "target_id": target_id,
                "probe_kind": "P_TARGET_TO_CLASS_ORACLE",
                "tap": None,
                "deployable": False,
                "selection_eligible": False,
            }
        )
    if len(probe_rows) > 18 * 9:
        raise RuntimeError("Stage-C probe plan exceeds its frozen 162-row cap")
    probe_rows = [
        {
            **row,
            "pipeline_seed": 101,
            "component_seed": component_seed(
                101, "stage_c_probe", row["row_id"]
            ),
            "input_materializer": (
                "scripts/materialize_hosd_pair_probe_inputs.py"
                if row["target_id"] in {
                    "T_HLT_TRACK_PAIR_13",
                    "T_HLT_REGION_PAIR_8",
                }
                else "scripts/materialize_hosd_probe_inputs.py"
            ),
        }
        for row in probe_rows
    ]
    return with_content_hash(
        {
            "contract": STAGE_C_PLAN_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "target_registry_sha256": target_registry["content_hash"],
            "baseline_registry_sha256": baseline_registry["content_hash"],
            "baseline_rows": [
                {
                    **row,
                    "row_id": row["baseline_id"],
                    "pipeline_seed": 101,
                    "component_seed": component_seed(
                        101, "stage_c_baseline", row["baseline_id"]
                    ),
                }
                for row in baseline_registry["baselines"]
            ],
            "probe_encoder": {
                "baseline_id": "H_BASE",
                "seed": 101,
                "freeze_before_any_target_result": True,
            },
            "probe_rows": probe_rows,
            "probe_target_order": [row["target_id"] for row in targets],
            "probe_row_count": len(probe_rows),
            "probe_row_hard_cap": 162,
            "all_targets_continue_to_stage_d": True,
            "performance_can_cancel_or_omit": False,
            "plan_sha256_preimage": canonical_sha256(
                {
                    "baselines": list(BASELINE_IDS),
                    "targets": [row["target_id"] for row in targets],
                    "probe_rows": probe_rows,
                }
            ),
        }
    )


__all__ = [
    "BASELINE_IDS",
    "HOSDTrainingProtocol",
    "NativeRelationAuxClassifier",
    "PROBE_TARGET_PREFIXES",
    "build_baseline_model",
    "build_baseline_registry",
    "build_stage_c_plan",
    "classification_kd_loss",
    "component_seed",
    "particle_net_config",
    "validate_baseline_registry",
]
