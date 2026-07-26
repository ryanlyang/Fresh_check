"""Label-exposure and training-compute ledger for particle-view campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .splits import (
    PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    logical_split_binding,
)


PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT = (
    "particle_view_label_exposure_ledger_v1"
)
PARTICLE_VIEW_LABEL_EXPOSURE_RECORD_CONTRACT = (
    "particle_view_label_exposure_record_v1"
)
PARTICLE_VIEW_FAIRNESS_BUDGET_ACCOUNTING_CONTRACT = (
    "particle_view_fairness_budget_accounting_v1"
)
_FAIRNESS_LEDGER_CONTRACT = "particle_view_selected_path_fairness_ledger_v1"


def _nonnegative_integer(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class LabelExposureRecord:
    run_id: str
    component: str
    stage: str
    seed: int
    train_identity_sha256: str
    optimizer_steps: int
    label_bearing_steps: int
    labeled_examples_processed: int
    ce_bearing_steps: int
    teacher_kd_steps: int
    view_supervision_steps: int
    training_flops: int
    retained_in_deployable_path: bool
    train_split: str = "train"

    def to_payload(self) -> dict[str, Any]:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("label-exposure run_id must be non-empty")
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("label-exposure component must be non-empty")
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("label-exposure stage must be non-empty")
        if self.seed not in {101, 202, 303}:
            raise ValueError("label-exposure seed must be 101, 202, or 303")
        if self.train_split != "train":
            raise ValueError("label exposure must use the unified train split")
        require_sha256("train_identity_sha256", self.train_identity_sha256)
        values = {
            "optimizer_steps": _nonnegative_integer(
                "optimizer_steps", self.optimizer_steps
            ),
            "label_bearing_steps": _nonnegative_integer(
                "label_bearing_steps", self.label_bearing_steps
            ),
            "labeled_examples_processed": _nonnegative_integer(
                "labeled_examples_processed", self.labeled_examples_processed
            ),
            "ce_bearing_steps": _nonnegative_integer(
                "ce_bearing_steps", self.ce_bearing_steps
            ),
            "teacher_kd_steps": _nonnegative_integer(
                "teacher_kd_steps", self.teacher_kd_steps
            ),
            "view_supervision_steps": _nonnegative_integer(
                "view_supervision_steps", self.view_supervision_steps
            ),
            "training_flops": _nonnegative_integer(
                "training_flops", self.training_flops
            ),
        }
        if values["label_bearing_steps"] > values["optimizer_steps"]:
            raise ValueError("label_bearing_steps cannot exceed optimizer_steps")
        if values["ce_bearing_steps"] > values["label_bearing_steps"]:
            raise ValueError("ce_bearing_steps cannot exceed label_bearing_steps")
        if not isinstance(self.retained_in_deployable_path, bool):
            raise ValueError("retained_in_deployable_path must be boolean")
        return {
            "contract": PARTICLE_VIEW_LABEL_EXPOSURE_RECORD_CONTRACT,
            "run_id": self.run_id,
            "component": self.component,
            "stage": self.stage,
            "seed": int(self.seed),
            "train_split": self.train_split,
            "train_identity_sha256": self.train_identity_sha256,
            **values,
            "retained_in_deployable_path": self.retained_in_deployable_path,
        }


_TOTAL_FIELDS = (
    "optimizer_steps",
    "label_bearing_steps",
    "labeled_examples_processed",
    "ce_bearing_steps",
    "teacher_kd_steps",
    "view_supervision_steps",
    "training_flops",
)


def _sum_records(
    records: Sequence[Mapping[str, Any]], *, retained_only: bool
) -> dict[str, int]:
    selected = [
        record
        for record in records
        if not retained_only or bool(record["retained_in_deployable_path"])
    ]
    return {
        field: sum(int(record[field]) for record in selected)
        for field in _TOTAL_FIELDS
    }


def build_label_exposure_ledger(
    *,
    unified_split_manifest: Mapping[str, Any],
    pipeline_id: str,
    records: Sequence[LabelExposureRecord],
) -> dict[str, Any]:
    validate_content_hash(
        unified_split_manifest,
        expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    )
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise ValueError("pipeline_id must be non-empty")
    _, train_split_sha256, train_identity_sha256 = logical_split_binding(
        unified_split_manifest, "train"
    )
    serialized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, int]] = set()
    for record in records:
        payload = record.to_payload()
        if payload["train_identity_sha256"] != train_identity_sha256:
            raise ValueError("label-exposure record uses a different train identity")
        identity = (
            payload["run_id"],
            payload["component"],
            payload["stage"],
            payload["seed"],
        )
        if identity in identities:
            raise ValueError(f"duplicate label-exposure record {identity}")
        identities.add(identity)
        serialized.append(payload)
    serialized.sort(
        key=lambda row: (
            row["run_id"],
            row["component"],
            row["stage"],
            row["seed"],
        )
    )
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT,
            "pipeline_id": pipeline_id,
            "unified_split_manifest_sha256": unified_split_manifest[
                "content_hash"
            ],
            "train_split_sha256": train_split_sha256,
            "train_identity_sha256": train_identity_sha256,
            "training_topology": "single_pool_no_crossfit_v1",
            "records": serialized,
            "totals_all_training": _sum_records(serialized, retained_only=False),
            "totals_retained_deployable_path": _sum_records(
                serialized, retained_only=True
            ),
        }
    )
    validate_label_exposure_ledger(
        artifact, unified_split_manifest=unified_split_manifest
    )
    return artifact


def validate_label_exposure_ledger(
    payload: Mapping[str, Any],
    *,
    unified_split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT
    )
    expected_fields = {
        "contract",
        "pipeline_id",
        "unified_split_manifest_sha256",
        "train_split_sha256",
        "train_identity_sha256",
        "training_topology",
        "records",
        "totals_all_training",
        "totals_retained_deployable_path",
        "content_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("label-exposure ledger field inventory mismatch")
    if not isinstance(payload.get("pipeline_id"), str) or not payload["pipeline_id"]:
        raise ValueError("label-exposure pipeline_id is invalid")
    validate_content_hash(
        unified_split_manifest,
        expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    )
    _, train_split_sha256, train_identity_sha256 = logical_split_binding(
        unified_split_manifest, "train"
    )
    if payload.get("unified_split_manifest_sha256") != unified_split_manifest.get(
        "content_hash"
    ):
        raise ValueError("label ledger is bound to a different unified manifest")
    if payload.get("train_split_sha256") != train_split_sha256:
        raise ValueError("label ledger train split hash mismatch")
    if payload.get("train_identity_sha256") != train_identity_sha256:
        raise ValueError("label ledger train identity mismatch")
    if payload.get("training_topology") != "single_pool_no_crossfit_v1":
        raise ValueError("label ledger training topology mismatch")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("label ledger records must be a list")
    records: list[dict[str, Any]] = []
    identities: list[tuple[str, str, str, int]] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("label ledger record must be an object")
        if raw.get("contract") != PARTICLE_VIEW_LABEL_EXPOSURE_RECORD_CONTRACT:
            raise ValueError("label ledger record contract mismatch")
        record = LabelExposureRecord(
            run_id=raw["run_id"],
            component=raw["component"],
            stage=raw["stage"],
            seed=raw["seed"],
            train_split=raw["train_split"],
            train_identity_sha256=raw["train_identity_sha256"],
            optimizer_steps=raw["optimizer_steps"],
            label_bearing_steps=raw["label_bearing_steps"],
            labeled_examples_processed=raw["labeled_examples_processed"],
            ce_bearing_steps=raw["ce_bearing_steps"],
            teacher_kd_steps=raw["teacher_kd_steps"],
            view_supervision_steps=raw["view_supervision_steps"],
            training_flops=raw["training_flops"],
            retained_in_deployable_path=raw["retained_in_deployable_path"],
        )
        canonical = record.to_payload()
        if canonical != dict(raw):
            raise ValueError("label ledger record is not canonical")
        if canonical["train_identity_sha256"] != train_identity_sha256:
            raise ValueError("label ledger mixes training identities")
        identity = (
            canonical["run_id"],
            canonical["component"],
            canonical["stage"],
            canonical["seed"],
        )
        if identity in identities:
            raise ValueError(f"duplicate label-exposure record {identity}")
        identities.append(identity)
        records.append(canonical)
    if identities != sorted(identities):
        raise ValueError("label-exposure records must use canonical order")
    expected_all = _sum_records(records, retained_only=False)
    expected_retained = _sum_records(records, retained_only=True)
    if payload.get("totals_all_training") != expected_all:
        raise ValueError("label ledger all-training totals mismatch")
    if payload.get("totals_retained_deployable_path") != expected_retained:
        raise ValueError("label ledger retained-path totals mismatch")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "record_count": len(records),
        "train_identity_sha256": train_identity_sha256,
        "totals_all_training": expected_all,
        "totals_retained_deployable_path": expected_retained,
    }


def build_fairness_budget_accounting(
    *,
    selected_path_fairness_ledger: Mapping[str, Any],
    label_exposure_ledgers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile every selected-path control budget with its source ledger."""

    validate_content_hash(
        selected_path_fairness_ledger,
        expected_contract=_FAIRNESS_LEDGER_CONTRACT,
    )
    by_hash = {}
    for ledger in label_exposure_ledgers:
        validate_content_hash(
            ledger,
            expected_contract=PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT,
        )
        content_hash = ledger["content_hash"]
        if content_hash in by_hash:
            raise ValueError("duplicate label-exposure ledger")
        by_hash[content_hash] = ledger
    rows = []
    referenced: set[str] = set()
    for entry in selected_path_fairness_ledger["entries"]:
        require_sha256(
            "fairness_entry_sha256", entry["fairness_entry_sha256"]
        )
        for replica in entry["replicas"]:
            ledger_sha = require_sha256(
                "training_ledger_sha256",
                replica["training_ledger_sha256"],
            )
            ledger = by_hash.get(ledger_sha)
            if ledger is None:
                raise ValueError("fairness entry source ledger is missing")
            if (
                ledger["train_identity_sha256"]
                != selected_path_fairness_ledger["train_identity_sha256"]
            ):
                raise ValueError("fairness accounting mixes train identities")
            retained = ledger["totals_retained_deployable_path"]
            total = ledger["totals_all_training"]
            expected = {
                "a0_view_long_deploy_exact_ce_updates": retained[
                    "ce_bearing_steps"
                ],
                "a0_view_total_label_budget_exact_updates": total[
                    "label_bearing_steps"
                ],
                "label_bearing_updates": total["label_bearing_steps"],
                "optimizer_updates_retained_path": retained["optimizer_steps"],
                "training_flops_retained_path": retained["training_flops"],
            }
            for field, value in expected.items():
                if replica[field] != value:
                    raise ValueError(
                        f"fairness budget {field} differs from source ledger"
                    )
            referenced.add(ledger_sha)
            rows.append(
                {
                    "configuration_id": entry["configuration_id"],
                    "fairness_entry_sha256": entry["fairness_entry_sha256"],
                    "seed": replica["seed"],
                    "training_ledger_sha256": ledger_sha,
                    **expected,
                    "labeled_examples_processed_all_training": total[
                        "labeled_examples_processed"
                    ],
                    "teacher_kd_steps_all_training": total[
                        "teacher_kd_steps"
                    ],
                    "view_supervision_steps_all_training": total[
                        "view_supervision_steps"
                    ],
                    "exact_match": True,
                }
            )
    rows.sort(
        key=lambda row: (
            row["configuration_id"],
            row["seed"],
            row["training_ledger_sha256"],
        )
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FAIRNESS_BUDGET_ACCOUNTING_CONTRACT,
            "selected_path_fairness_ledger_sha256": (
                selected_path_fairness_ledger["content_hash"]
            ),
            "train_identity_sha256": selected_path_fairness_ledger[
                "train_identity_sha256"
            ],
            "rows": rows,
            "row_count": len(rows),
            "referenced_label_exposure_ledger_sha256": sorted(referenced),
            "all_control_budgets_exactly_reconciled": True,
        }
    )


__all__ = [
    "PARTICLE_VIEW_FAIRNESS_BUDGET_ACCOUNTING_CONTRACT",
    "PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT",
    "PARTICLE_VIEW_LABEL_EXPOSURE_RECORD_CONTRACT",
    "LabelExposureRecord",
    "build_fairness_budget_accounting",
    "build_label_exposure_ledger",
    "validate_label_exposure_ledger",
]
