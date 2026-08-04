"""Authenticated fast-track offline expert-bank and OBASE7 controls.

This module deliberately lives outside the immutable Stage-C registry.  It
binds completed Stage-B checkpoints from an existing parent campaign and
publishes a separate supplemental experiment whose source may advance without
making the parent artifacts appear interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .registry import EXPERT_ORDER
from .step4 import resolve_stage_b_run, validate_stage_b_run_registry


SUPPLEMENTAL_PLAN_CONTRACT = "retb_supplemental_offline_fusion_plan_v1"
SUPPLEMENTAL_BANK_RESULT_CONTRACT = (
    "retb_supplemental_offline_fusion_bank_result_v1"
)
SUPPLEMENTAL_OBASE7_RESULT_CONTRACT = "retb_supplemental_obase7_result_v1"
SUPPLEMENTAL_REPORT_CONTRACT = "retb_supplemental_offline_fusion_report_v1"

SEVEN_SEEDS = (101, 202, 303, 404, 505, 606, 707)
FUSION_VARIANTS = (
    "MEAN_LOGITS",
    "TRAINED_LOGIT_LINEAR",
    "POOLED_MLP",
    "TOKEN_TRANSFORMER",
)

BANK_DEFINITIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "CE4": tuple(
        (expert, "ELOSS_CE")
        for expert in ("BASE4", "PT", "TRACK", "REGION")
    ),
    "CE7": tuple((expert, "ELOSS_CE") for expert in EXPERT_ORDER),
    "KD3": tuple(
        (expert, "ELOSS_KD_DOMINANT")
        for expert in ("BASE4", "PT", "TRACK")
    ),
    "KD4": tuple(
        (expert, "ELOSS_KD_DOMINANT")
        for expert in ("BASE4", "PT", "TRACK", "REGION")
    ),
    "MIXED7": tuple(
        (expert, "ELOSS_KD_DOMINANT")
        for expert in ("BASE4", "PT", "TRACK", "REGION")
    )
    + tuple(
        (expert, "ELOSS_CE")
        for expert in ("PID", "CHARGE", "DENSITY")
    ),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_stage_b_rows(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for section in (
            "primary_shape_screen",
            "tokenizer_controls",
            "dual_topology_controls",
            "representative_expert_loss_rows",
            "full_optimization_grid",
            "fixed_followup_references",
        )
        for row in registry[section]
    ]


def resolve_bank_parent(
    registry: Mapping[str, Any], *, expert_id: str, loss_id: str
) -> dict[str, Any]:
    """Resolve the unique physical S8_128 Stage-B parent for one bank."""

    validate_stage_b_run_registry(registry)
    if expert_id not in EXPERT_ORDER:
        raise ValueError("supplemental bank names an unknown expert")
    if loss_id not in {"ELOSS_CE", "ELOSS_KD_DOMINANT"}:
        raise ValueError("supplemental bank names an unsupported loss")
    matches: dict[str, Mapping[str, Any]] = {}
    for row in _all_stage_b_rows(registry):
        configuration = row["configuration"]
        if (
            configuration.get("expert_id") == expert_id
            and configuration.get("shape_id") == "S8_128"
            and configuration.get("loss_id") == loss_id
            and int(row["seed"]) == 101
        ):
            matches[str(row["run_id"])] = row
    if len(matches) != 1:
        raise ValueError(
            f"cannot uniquely resolve {expert_id}/{loss_id} Stage-B parent"
        )
    return resolve_stage_b_run(registry, run_id=next(iter(matches)))


def _parent_record(
    parent_root: Path,
    registry: Mapping[str, Any],
    *,
    expert_id: str,
    loss_id: str,
) -> dict[str, Any]:
    row = resolve_bank_parent(
        registry, expert_id=expert_id, loss_id=loss_id
    )
    run_root = (
        parent_root
        / "runs"
        / "stage_b"
        / str(row["run_id"])
        / "seed_101"
    )
    registration_path = run_root / "checkpoint_registration.json"
    checkpoint_path = run_root / "best_model_val.pt"
    registration = load_hashed_json(
        registration_path,
        expected_contract="retb_offline_expert_registration_v1",
    )
    if (
        registration.get("run_id") != row["run_id"]
        or registration.get("expert_id") != expert_id
        or registration.get("shape_id") != "S8_128"
        or int(registration.get("seed", -1)) != 101
        or registration.get("fixed_epoch_budget_completed") is not True
        or registration.get("checkpoint_sha256")
        != file_sha256(checkpoint_path)
    ):
        raise ValueError(f"{expert_id}/{loss_id} parent lineage differs")
    return {
        "expert_id": expert_id,
        "loss_id": loss_id,
        "run_id": row["run_id"],
        "configuration": dict(row["configuration"]),
        "registration_path": str(registration_path.resolve()),
        "registration_sha256": registration["content_hash"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": registration["checkpoint_sha256"],
    }


def build_supplemental_plan(
    *,
    parent_root: str | Path,
    supplemental_id: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(parent_root).resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    registry = load_hashed_json(root / "registry" / "retb_stage_b_runs.json")
    registry_sha = validate_stage_b_run_registry(registry)
    if campaign.get("campaign_profile") != "production_500k_scale3m":
        raise ValueError("supplemental fusion requires a production 500k parent")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for definition in BANK_DEFINITIONS.values():
        for expert_id, loss_id in definition:
            records.setdefault(
                (expert_id, loss_id),
                _parent_record(
                    root,
                    registry,
                    expert_id=expert_id,
                    loss_id=loss_id,
                ),
            )
    banks = {
        bank_id: {
            "bank_id": bank_id,
            "expert_order": [expert for expert, _ in definition],
            "members": [records[item] for item in definition],
            "fusion_variants": list(FUSION_VARIANTS),
        }
        for bank_id, definition in BANK_DEFINITIONS.items()
    }
    payload = with_content_hash(
        {
            "contract": SUPPLEMENTAL_PLAN_CONTRACT,
            "schema_version": 1,
            "supplemental_id": str(supplemental_id),
            "parent_campaign_root": str(root),
            "parent_campaign_id": campaign["campaign_id"],
            "parent_campaign_spec_sha256": campaign["content_hash"],
            "parent_source": campaign["source"],
            "parent_stage_b_registry_sha256": registry_sha,
            "training_split": "model_train",
            "checkpoint_selection_split": "val_stop",
            "comparison_split": "val_design",
            "final_test_access": False,
            "shape_id": "S8_128",
            "pipeline_seed": 101,
            "banks": banks,
            "obase7": {
                "control_id": "OBASE7_MEAN_LOGITS",
                "member_model": "ordinary_O_BASE_RelationalParticleTransformer",
                "member_objective": "unweighted_offline_cross_entropy",
                "member_seeds": list(SEVEN_SEEDS),
                "member_epochs": 40,
                "combiner": "arithmetic_mean_logits",
                "learned_combiner": False,
                "reuse_stage_a_seed_101": False,
            },
            "performance_based_termination": False,
            "scientific_underperformance_blocks_execution": False,
            "temporary_bank_policy": (
                "allocation_local_only_deleted_before_success_receipt"
            ),
        }
    )
    return bind_source(payload, source_snapshot=source_snapshot)


def validate_supplemental_plan(
    payload: Mapping[str, Any], *, check_parent_bytes: bool = True
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SUPPLEMENTAL_PLAN_CONTRACT
    )
    if set(payload.get("banks", {})) != set(BANK_DEFINITIONS):
        raise ValueError("supplemental bank coverage differs")
    if payload.get("final_test_access") is not False:
        raise ValueError("supplemental plan may not access final_test")
    if payload.get("performance_based_termination") is not False:
        raise ValueError("supplemental plan may not stop on performance")
    obase = payload.get("obase7", {})
    if (
        obase.get("member_seeds") != list(SEVEN_SEEDS)
        or obase.get("combiner") != "arithmetic_mean_logits"
        or obase.get("reuse_stage_a_seed_101") is not False
    ):
        raise ValueError("OBASE7 control differs")
    for bank_id, definition in BANK_DEFINITIONS.items():
        bank = payload["banks"][bank_id]
        if (
            bank.get("expert_order") != [item[0] for item in definition]
            or bank.get("fusion_variants") != list(FUSION_VARIANTS)
            or [(row["expert_id"], row["loss_id"]) for row in bank["members"]]
            != list(definition)
        ):
            raise ValueError(f"{bank_id} definition differs")
        if check_parent_bytes:
            for row in bank["members"]:
                registration = load_hashed_json(row["registration_path"])
                if (
                    registration["content_hash"]
                    != row["registration_sha256"]
                    or file_sha256(row["checkpoint_path"])
                    != row["checkpoint_sha256"]
                ):
                    raise ValueError("supplemental parent bytes drifted")
    return digest


def mean_logits(logit_banks: Sequence[Any]) -> Any:
    if not logit_banks:
        raise ValueError("mean-logit fusion requires at least one member")
    first = logit_banks[0]
    return first.__class__.stack(logit_banks, dim=0).mean(dim=0) \
        if hasattr(first.__class__, "stack") else None


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:  # pragma: no cover
        pass


def select_fixed_budget_checkpoint(
    rows: Sequence[Mapping[str, Any]], *, accuracy_window: float = 0.0001
) -> int:
    if not rows:
        raise ValueError("fusion checkpoint selection requires epochs")
    checked = []
    for row in rows:
        epoch = int(row["epoch"])
        accuracy = float(row["val_stop"]["accuracy"])
        cross_entropy = float(row["val_stop"]["cross_entropy"])
        if not (math.isfinite(accuracy) and math.isfinite(cross_entropy)):
            raise FloatingPointError("fusion checkpoint metric is nonfinite")
        checked.append((epoch, accuracy, cross_entropy))
    maximum = max(item[1] for item in checked)
    return min(
        (item for item in checked if maximum - item[1] <= accuracy_window),
        key=lambda item: (item[2], item[0]),
    )[0]


__all__ = [
    "BANK_DEFINITIONS",
    "FUSION_VARIANTS",
    "SEVEN_SEEDS",
    "SUPPLEMENTAL_BANK_RESULT_CONTRACT",
    "SUPPLEMENTAL_OBASE7_RESULT_CONTRACT",
    "SUPPLEMENTAL_PLAN_CONTRACT",
    "SUPPLEMENTAL_REPORT_CONTRACT",
    "build_supplemental_plan",
    "file_sha256",
    "resolve_bank_parent",
    "select_fixed_budget_checkpoint",
    "set_deterministic_seed",
    "validate_supplemental_plan",
]
