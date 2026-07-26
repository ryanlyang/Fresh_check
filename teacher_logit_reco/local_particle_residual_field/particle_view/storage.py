"""Measured storage reservations and bounded retention for particle-view runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .tap_staging import (
    PARTICLE_VIEW_STAGED_TAP_RESERVATION_CONTRACT,
    validate_tap_stage_reservation,
)


PARTICLE_VIEW_STORAGE_RESERVATION_CONTRACT = (
    "particle_view_storage_reservation_v1"
)
PARTICLE_VIEW_DIAGNOSTIC_BUDGET_CONTRACT = (
    "particle_view_diagnostic_budget_v1"
)
PARTICLE_VIEW_RETENTION_PLAN_CONTRACT = "particle_view_retention_plan_v1"
PARTICLE_VIEW_EVICTION_REPORT_CONTRACT = "particle_view_eviction_report_v1"

DEFAULT_DIAGNOSTIC_MAX_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_DIAGNOSTIC_MAX_TOTAL_BYTES = 256 * 1024 * 1024
DEFAULT_ATTENTION_SAMPLE_LIMIT = 8

_RETENTION_KINDS = {
    "json_metric",
    "registry",
    "confirmed_checkpoint",
    "canonical_checkpoint",
    "selected_checkpoint",
    "screen_checkpoint",
    "optimizer_state",
    "attention_diagnostic",
}


def _positive_integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < (0 if allow_zero else 1)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _relative_file(root: Path, path: str | Path) -> tuple[Path, str]:
    root = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("storage artifact is outside the campaign root") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate, relative


def measure_storage_artifacts(
    campaign_root: str | Path,
    artifacts: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    """Measure exact persistent source/checkpoint files before reservation."""

    root = Path(campaign_root)
    if not artifacts:
        raise ValueError("at least one measured artifact is required")
    measured = []
    for role, raw_path in sorted(artifacts.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("storage artifact roles must be non-empty strings")
        path, relative = _relative_file(root, raw_path)
        measured.append(
            {
                "role": role,
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return measured


def build_diagnostic_budget(
    *,
    max_file_bytes: int = DEFAULT_DIAGNOSTIC_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_DIAGNOSTIC_MAX_TOTAL_BYTES,
    max_attention_samples: int = DEFAULT_ATTENTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    max_file_bytes = _positive_integer("max_file_bytes", max_file_bytes)
    max_total_bytes = _positive_integer("max_total_bytes", max_total_bytes)
    max_attention_samples = _positive_integer(
        "max_attention_samples", max_attention_samples, allow_zero=True
    )
    if max_file_bytes > max_total_bytes:
        raise ValueError("diagnostic file cap cannot exceed total cap")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_DIAGNOSTIC_BUDGET_CONTRACT,
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
            "max_attention_samples": max_attention_samples,
            "full_dataset_attention_persistence_forbidden": True,
            "overflow_policy": "reject_before_write",
        }
    )


def validate_diagnostic_inventory(
    budget: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_content_hash(
        budget, expected_contract=PARTICLE_VIEW_DIAGNOSTIC_BUDGET_CONTRACT
    )
    total = 0
    attention_samples = 0
    paths: set[str] = set()
    for row in diagnostics:
        path = str(row.get("relative_path", ""))
        if not path or path in paths:
            raise ValueError("diagnostic paths must be non-empty and unique")
        paths.add(path)
        size = _positive_integer("diagnostic bytes", row.get("bytes"))
        if size > budget["max_file_bytes"]:
            raise ValueError(f"diagnostic {path} exceeds its file cap")
        samples = _positive_integer(
            "attention_samples",
            row.get("attention_samples", 0),
            allow_zero=True,
        )
        if row.get("contains_attention") is not True and samples:
            raise ValueError("non-attention diagnostic declares attention samples")
        total += size
        attention_samples += samples
    if total > budget["max_total_bytes"]:
        raise ValueError("diagnostic inventory exceeds its total byte cap")
    if attention_samples > budget["max_attention_samples"]:
        raise ValueError("diagnostic inventory exceeds its attention sample cap")
    return {
        "ok": True,
        "file_count": len(diagnostics),
        "total_bytes": total,
        "attention_samples": attention_samples,
    }


def write_bounded_json_diagnostic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    budget: Mapping[str, Any],
    existing_total_bytes: int = 0,
    attention_samples: int = 0,
) -> int:
    """Write one diagnostic only after checking its serialized byte budget."""

    validate_content_hash(
        budget, expected_contract=PARTICLE_VIEW_DIAGNOSTIC_BUDGET_CONTRACT
    )
    existing_total_bytes = _positive_integer(
        "existing_total_bytes", existing_total_bytes, allow_zero=True
    )
    attention_samples = _positive_integer(
        "attention_samples", attention_samples, allow_zero=True
    )
    encoded = (
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(encoded) > budget["max_file_bytes"]:
        raise ValueError("diagnostic exceeds its file byte cap")
    if existing_total_bytes + len(encoded) > budget["max_total_bytes"]:
        raise ValueError("diagnostic would exceed the total byte cap")
    if attention_samples > budget["max_attention_samples"]:
        raise ValueError("diagnostic exceeds the attention sample cap")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return len(encoded)


def build_storage_reservation(
    *,
    campaign_root: str | Path,
    measured_artifacts: Mapping[str, str | Path],
    planned_persistent_bytes: Mapping[str, int],
    tap_stage_reservations: Sequence[Mapping[str, Any]],
    persistent_budget_bytes: int,
    filesystem_available_bytes: int,
    allocation_ram_bytes: int,
    transient_ram_bytes: int,
    diagnostic_budget: Mapping[str, Any],
    derivation_evidence: Mapping[str, Any] | None = None,
    safety_margin_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    """Reserve measured disk and RAM before a campaign is submitted."""

    persistent_budget_bytes = _positive_integer(
        "persistent_budget_bytes", persistent_budget_bytes
    )
    filesystem_available_bytes = _positive_integer(
        "filesystem_available_bytes", filesystem_available_bytes
    )
    allocation_ram_bytes = _positive_integer(
        "allocation_ram_bytes", allocation_ram_bytes
    )
    transient_ram_bytes = _positive_integer(
        "transient_ram_bytes", transient_ram_bytes, allow_zero=True
    )
    safety_margin_bytes = _positive_integer(
        "safety_margin_bytes", safety_margin_bytes, allow_zero=True
    )
    validate_content_hash(
        diagnostic_budget,
        expected_contract=PARTICLE_VIEW_DIAGNOSTIC_BUDGET_CONTRACT,
    )
    measured = measure_storage_artifacts(campaign_root, measured_artifacts)
    planned = []
    for role, value in sorted(planned_persistent_bytes.items()):
        if not role:
            raise ValueError("planned persistent role cannot be empty")
        planned.append(
            {"role": str(role), "reserved_bytes": _positive_integer(role, value)}
        )
    if not planned:
        raise ValueError("planned persistent artifact inventory is empty")
    tap_rows = []
    for reservation in tap_stage_reservations:
        validate_content_hash(
            reservation,
            expected_contract=PARTICLE_VIEW_STAGED_TAP_RESERVATION_CONTRACT,
        )
        tap_bytes = validate_tap_stage_reservation(reservation)
        tap_rows.append(
            {
                "reservation_sha256": reservation["content_hash"],
                "source_role": reservation["source_role"],
                "reserved_bytes": tap_bytes,
            }
        )
    measured_bytes = sum(row["bytes"] for row in measured)
    planned_bytes = sum(row["reserved_bytes"] for row in planned)
    persistent_required = (
        measured_bytes
        + planned_bytes
        + int(diagnostic_budget["max_total_bytes"])
        + safety_margin_bytes
    )
    if persistent_required > persistent_budget_bytes:
        raise ValueError("campaign persistent reservation exceeds its budget")
    if persistent_required > filesystem_available_bytes:
        raise ValueError("campaign persistent reservation exceeds free storage")
    staged_tap_bytes = sum(row["reserved_bytes"] for row in tap_rows)
    ram_required = staged_tap_bytes + transient_ram_bytes
    if ram_required > allocation_ram_bytes:
        raise ValueError("teacher-tap staging exceeds allocation RAM")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_STORAGE_RESERVATION_CONTRACT,
            "campaign_root_name": Path(campaign_root).resolve().name,
            "campaign_root_identity_sha256": canonical_sha256(
                {"resolved_campaign_root": str(Path(campaign_root).resolve())}
            ),
            "measured_artifacts": measured,
            "planned_persistent_artifacts": planned,
            "tap_stage_reservations": tap_rows,
            "derivation_evidence": dict(derivation_evidence or {}),
            "diagnostic_budget_sha256": diagnostic_budget["content_hash"],
            "measured_persistent_bytes": measured_bytes,
            "planned_persistent_bytes": planned_bytes,
            "diagnostic_reserved_bytes": diagnostic_budget["max_total_bytes"],
            "safety_margin_bytes": safety_margin_bytes,
            "persistent_required_bytes": persistent_required,
            "persistent_budget_bytes": persistent_budget_bytes,
            "filesystem_available_bytes": filesystem_available_bytes,
            "staged_tap_ram_bytes": staged_tap_bytes,
            "transient_ram_bytes": transient_ram_bytes,
            "allocation_ram_bytes": allocation_ram_bytes,
            "ram_required_bytes": ram_required,
            "offline_contextual_tokens_persisted": False,
            "selected_target_only_persisted": True,
            "preflight_passed": True,
        }
    )


@dataclass(frozen=True)
class RetentionCandidate:
    relative_path: str
    kind: str
    sha256: str
    metrics_finalized: bool
    selected: bool = False
    canonical: bool = False
    three_seed_confirmed: bool = False
    screen_run_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        if (
            not self.relative_path
            or Path(self.relative_path).is_absolute()
            or ".." in Path(self.relative_path).parts
        ):
            raise ValueError("retention path must be campaign-relative")
        if self.kind not in _RETENTION_KINDS:
            raise ValueError(f"unknown retention kind {self.kind!r}")
        require_sha256("retention sha256", self.sha256)
        for name in (
            "metrics_finalized",
            "selected",
            "canonical",
            "three_seed_confirmed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.screen_run_id is not None and not self.screen_run_id:
            raise ValueError("screen_run_id cannot be empty")
        return {
            "relative_path": Path(self.relative_path).as_posix(),
            "kind": self.kind,
            "sha256": self.sha256,
            "metrics_finalized": self.metrics_finalized,
            "selected": self.selected,
            "canonical": self.canonical,
            "three_seed_confirmed": self.three_seed_confirmed,
            "screen_run_id": self.screen_run_id,
        }


def build_retention_plan(
    *,
    campaign_root: str | Path,
    candidates: Sequence[RetentionCandidate],
) -> dict[str, Any]:
    """Apply the locked retention rules without deleting anything yet."""

    root = Path(campaign_root)
    rows = []
    payloads = [candidate.to_payload() for candidate in candidates]
    screen_groups: dict[str, list[dict[str, Any]]] = {}
    for row in payloads:
        if row["kind"] == "screen_checkpoint":
            if row["screen_run_id"] is None:
                raise ValueError("screen checkpoints require screen_run_id")
            screen_groups.setdefault(row["screen_run_id"], []).append(row)
    for run_id, group in screen_groups.items():
        retained = [
            row
            for row in group
            if row["selected"]
            or row["canonical"]
            or row["three_seed_confirmed"]
        ]
        if len(retained) != 1:
            raise ValueError(
                f"screen run {run_id!r} must identify exactly one best checkpoint"
            )
    for row in sorted(payloads, key=lambda item: item["relative_path"]):
        path, relative = _relative_file(root, row["relative_path"])
        if relative != row["relative_path"]:
            raise ValueError("retention path is not canonical")
        if sha256_file(path) != row["sha256"]:
            raise ValueError("retention candidate hash mismatch")
        retain = (
            row["kind"] in {"json_metric", "registry"}
            or row["selected"]
            or row["canonical"]
            or row["three_seed_confirmed"]
        )
        reason = "required_campaign_artifact"
        if row["kind"] == "optimizer_state" and not retain:
            reason = "unselected_optimizer_state_after_metrics"
        elif row["kind"] == "attention_diagnostic" and not retain:
            reason = "batchwise_attention_not_retained"
        elif row["kind"] == "screen_checkpoint":
            reason = (
                "single_best_unconfirmed_checkpoint"
                if retain
                else "extra_unconfirmed_screen_checkpoint"
            )
        if not retain and not row["metrics_finalized"]:
            raise ValueError("cannot evict before metrics are finalized")
        rows.append(
            {
                **row,
                "bytes": path.stat().st_size,
                "action": "retain" if retain else "evict",
                "reason": reason,
            }
        )
    rows.sort(key=lambda row: row["relative_path"])
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_RETENTION_PLAN_CONTRACT,
            "campaign_root_name": root.resolve().name,
            "campaign_root_identity_sha256": canonical_sha256(
                {"resolved_campaign_root": str(root.resolve())}
            ),
            "rows": rows,
            "retain_count": sum(row["action"] == "retain" for row in rows),
            "evict_count": sum(row["action"] == "evict" for row in rows),
            "evict_bytes": sum(
                row["bytes"] for row in rows if row["action"] == "evict"
            ),
            "metrics_and_registries_always_retained": True,
            "at_most_one_unconfirmed_screen_checkpoint": True,
        }
    )


def execute_retention_plan(
    campaign_root: str | Path,
    plan: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Delete only hash-authenticated files explicitly marked for eviction."""

    validate_content_hash(
        plan, expected_contract=PARTICLE_VIEW_RETENTION_PLAN_CONTRACT
    )
    root = Path(campaign_root).resolve()
    if (
        root.name != plan["campaign_root_name"]
        or canonical_sha256({"resolved_campaign_root": str(root)})
        != plan["campaign_root_identity_sha256"]
    ):
        raise ValueError("retention plan belongs to a different campaign root")
    evicted = []
    retained = []
    for row in plan["rows"]:
        path, relative = _relative_file(root, row["relative_path"])
        if relative != row["relative_path"] or sha256_file(path) != row["sha256"]:
            raise ValueError("retention target changed after plan publication")
        if row["action"] == "evict":
            path.unlink()
            evicted.append(
                {
                    "relative_path": relative,
                    "sha256": row["sha256"],
                    "bytes": row["bytes"],
                    "recoverable": False,
                }
            )
        elif row["action"] == "retain":
            retained.append(relative)
        else:
            raise ValueError("retention plan action is invalid")
    report = with_content_hash(
        {
            "contract": PARTICLE_VIEW_EVICTION_REPORT_CONTRACT,
            "retention_plan_sha256": plan["content_hash"],
            "campaign_root_name": root.name,
            "campaign_root_identity_sha256": plan[
                "campaign_root_identity_sha256"
            ],
            "evicted": evicted,
            "retained": retained,
            "evicted_bytes": sum(row["bytes"] for row in evicted),
            "only_plan_authorized_files_removed": True,
        }
    )
    write_immutable_json(output_path, report)
    return report


__all__ = [
    "DEFAULT_ATTENTION_SAMPLE_LIMIT",
    "DEFAULT_DIAGNOSTIC_MAX_FILE_BYTES",
    "DEFAULT_DIAGNOSTIC_MAX_TOTAL_BYTES",
    "PARTICLE_VIEW_DIAGNOSTIC_BUDGET_CONTRACT",
    "PARTICLE_VIEW_EVICTION_REPORT_CONTRACT",
    "PARTICLE_VIEW_RETENTION_PLAN_CONTRACT",
    "PARTICLE_VIEW_STORAGE_RESERVATION_CONTRACT",
    "RetentionCandidate",
    "build_diagnostic_budget",
    "build_retention_plan",
    "build_storage_reservation",
    "execute_retention_plan",
    "measure_storage_artifacts",
    "validate_diagnostic_inventory",
    "write_bounded_json_diagnostic",
]
