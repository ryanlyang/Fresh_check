"""Typed, full-split distributed validation for adaptive-binary training."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from jetclass_fresh.jetclass_data import JetIdentity

from .distributed import (
    DistributedRuntime,
    all_gather_objects,
    all_reduce_float64_pair,
    all_reduce_sum_int,
    broadcast_object,
)
from .distributed_stream import ValidationRangeRow, compile_validation_coverage


ABPH_DISTRIBUTED_VALIDATION_CONTRACT = "adaptive_binary_distributed_validation_v1"
_ADDITIVE_KINDS = frozenset({"mean", "sum", "count", "ratio"})


@dataclass
class TypedValidationAccumulator:
    """Accumulate only explicitly typed validation quantities."""

    additive: dict[str, dict[str, Any]] = field(default_factory=dict)
    non_additive: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    n_jets: int = 0
    n_batches: int = 0

    def add_mean(
        self,
        name: str,
        value: float,
        weight: int,
        *,
        selection_eligible: bool = False,
    ) -> None:
        number = float(value)
        count = int(weight)
        if not math.isfinite(number) or count <= 0:
            raise ValueError(f"invalid mean contribution for {name}")
        row = self.additive.setdefault(
            str(name),
            {
                "kind": "mean",
                "numerator": 0.0,
                "denominator": 0.0,
                "selection_eligible": bool(selection_eligible),
                "denominator_semantics": "jets",
            },
        )
        if row["kind"] != "mean" or bool(row["selection_eligible"]) != bool(
            selection_eligible
        ):
            raise ValueError(f"validation reduction schema changed for {name}")
        row["numerator"] += number * count
        row["denominator"] += count

    def add_non_additive(self, name: str, value: float, weight: int) -> None:
        number = float(value)
        count = int(weight)
        if not math.isfinite(number) or count <= 0:
            raise ValueError(f"invalid diagnostic contribution for {name}")
        self.non_additive.setdefault(str(name), []).append(
            {"value": number, "n_jets": count}
        )

    def add_sum(self, name: str, value: float) -> None:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"invalid sum contribution for {name}")
        row = self.additive.setdefault(
            str(name),
            {
                "kind": "sum",
                "numerator": 0.0,
                "denominator": 0.0,
                "selection_eligible": False,
                "denominator_semantics": None,
            },
        )
        if row["kind"] != "sum":
            raise ValueError(f"validation reduction schema changed for {name}")
        row["numerator"] += number

    def add_count(self, name: str, count: int) -> None:
        value = int(count)
        if value < 0:
            raise ValueError(f"invalid count contribution for {name}")
        row = self.additive.setdefault(
            str(name),
            {
                "kind": "count",
                "numerator": 0.0,
                "denominator": 0.0,
                "selection_eligible": False,
                "denominator_semantics": "events",
            },
        )
        if row["kind"] != "count":
            raise ValueError(f"validation reduction schema changed for {name}")
        row["denominator"] += value

    def add_ratio(
        self,
        name: str,
        numerator: float,
        denominator: float,
        *,
        denominator_semantics: str,
    ) -> None:
        top = float(numerator)
        bottom = float(denominator)
        if (
            not math.isfinite(top)
            or not math.isfinite(bottom)
            or bottom < 0.0
            or not str(denominator_semantics)
        ):
            raise ValueError(f"invalid ratio contribution for {name}")
        row = self.additive.setdefault(
            str(name),
            {
                "kind": "ratio",
                "numerator": 0.0,
                "denominator": 0.0,
                "selection_eligible": False,
                "denominator_semantics": str(denominator_semantics),
            },
        )
        if (
            row["kind"] != "ratio"
            or row["denominator_semantics"] != str(denominator_semantics)
        ):
            raise ValueError(f"validation reduction schema changed for {name}")
        row["numerator"] += top
        row["denominator"] += bottom

    def finish_batch(self, n_jets: int) -> None:
        count = int(n_jets)
        if count <= 0:
            raise ValueError("validation batch size must be positive")
        self.n_jets += count
        self.n_batches += 1

    def local_payload(self) -> dict[str, Any]:
        if self.n_jets <= 0 or self.n_batches <= 0:
            raise RuntimeError("rollout model validation produced no batches")
        return {
            "contract": ABPH_DISTRIBUTED_VALIDATION_CONTRACT,
            "additive": {name: dict(row) for name, row in self.additive.items()},
            "non_additive": {
                name: [dict(item) for item in rows]
                for name, rows in self.non_additive.items()
            },
            "n_jets": int(self.n_jets),
            "n_batches": int(self.n_batches),
        }


def _schema(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, raw in dict(payload["additive"]).items():
        row = dict(raw)
        kind = str(row.get("kind"))
        if kind not in _ADDITIVE_KINDS:
            raise ValueError(f"unsupported additive validation kind {kind}")
        schema[str(name)] = {
            "kind": kind,
            "selection_eligible": bool(row.get("selection_eligible", False)),
            "denominator_semantics": row.get("denominator_semantics"),
        }
    for name in dict(payload["non_additive"]):
        if name in schema:
            raise ValueError(f"validation metric {name} has two reduction kinds")
        schema[str(name)] = {
            "kind": "non_additive_summary",
            "selection_eligible": False,
            "denominator_semantics": None,
        }
    return schema


def finalize_typed_validation(
    accumulator: TypedValidationAccumulator,
    *,
    runtime: DistributedRuntime,
    device: Any,
    required_losses: Sequence[str],
    effective_weights: Mapping[str, float],
    validation_range: ValidationRangeRow | Mapping[str, Any] | None = None,
    expected_jet_ids: Sequence[JetIdentity] | None = None,
    split: str = "model_val",
) -> dict[str, Any]:
    """Reduce one complete validation and attest exact identity coverage."""

    local = accumulator.local_payload()
    local_schema = _schema(local)
    control = {
        "rank": runtime.rank,
        "schema": local_schema,
        "required_losses": [str(value) for value in required_losses],
        "effective_weights": {
            str(name): float(value) for name, value in effective_weights.items()
        },
        "teacher_forcing_probability": 0.0,
        "validation_range": (
            None
            if validation_range is None
            else (
                validation_range.to_dict()
                if isinstance(validation_range, ValidationRangeRow)
                else dict(validation_range)
            )
        ),
        "non_additive": local["non_additive"],
    }
    controls = all_gather_objects(runtime, control)
    reference = controls[0]
    for row in controls:
        if row["schema"] != reference["schema"]:
            raise RuntimeError("validation reduction schema differs across ranks")
        if row["required_losses"] != reference["required_losses"]:
            raise RuntimeError("required validation losses differ across ranks")
        if row["effective_weights"] != reference["effective_weights"]:
            raise RuntimeError("validation objective weights differ across ranks")
        if float(row.get("teacher_forcing_probability", -1.0)) != 0.0:
            raise RuntimeError("distributed validation used nonzero teacher forcing")

    reduced: dict[str, dict[str, Any]] = {}
    metrics: dict[str, float] = {}
    for name, schema in sorted(local_schema.items()):
        kind = schema["kind"]
        if kind == "non_additive_summary":
            continue
        source = dict(local["additive"])[name]
        numerator, denominator = all_reduce_float64_pair(
            runtime,
            float(source["numerator"]),
            float(source["denominator"]),
            device=device,
        )
        if kind in {"mean", "ratio"}:
            if denominator <= 0.0:
                raise RuntimeError(f"validation denominator is empty for {name}")
            value = numerator / denominator
        elif kind == "sum":
            value = numerator
        else:
            value = denominator
        if not math.isfinite(value):
            raise FloatingPointError(f"reduced validation metric {name} is nonfinite")
        reduced[name] = {
            **schema,
            "numerator": numerator,
            "denominator": denominator,
            "value": value,
        }
        metrics[name] = value

    diagnostic_summaries: dict[str, Any] = {}
    for name, schema in sorted(local_schema.items()):
        if schema["kind"] != "non_additive_summary":
            continue
        rank_rows = [dict(row["non_additive"]).get(name, []) for row in controls]
        flattened = [item for rows in rank_rows for item in rows]
        values = [float(item["value"]) for item in flattened]
        if any(not math.isfinite(value) for value in values):
            raise FloatingPointError(f"validation diagnostic {name} is nonfinite")
        diagnostic_summaries[name] = {
            **schema,
            "rank_batch_values": rank_rows,
            "display_mean": sum(values) / len(values) if values else None,
        }

    total_jets = all_reduce_sum_int(runtime, local["n_jets"], device=device)
    total_batches = all_reduce_sum_int(runtime, local["n_batches"], device=device)
    coverage = None
    if runtime.distributed:
        if expected_jet_ids is None or any(
            row.get("validation_range") is None for row in controls
        ):
            raise RuntimeError("distributed validation lacks immutable identity coverage")
        coverage_error = None
        if runtime.is_primary:
            try:
                coverage = compile_validation_coverage(
                    split=str(split),
                    rows=[row["validation_range"] for row in controls],
                    expected_jet_ids=expected_jet_ids,
                    world_size=runtime.world_size,
                )
            except BaseException as exc:
                coverage_error = f"{type(exc).__name__}: {exc}"
        coverage_control = broadcast_object(
            runtime, {"coverage": coverage, "error": coverage_error}
        )
        if coverage_control["error"] is not None:
            raise RuntimeError(
                "rank-zero validation coverage compilation failed: "
                f"{coverage_control['error']}"
            )
        coverage = coverage_control["coverage"]
    elif validation_range is not None and expected_jet_ids is not None:
        coverage = compile_validation_coverage(
            split=str(split),
            rows=(validation_range,),
            expected_jet_ids=expected_jet_ids,
            world_size=1,
        )

    if expected_jet_ids is not None and total_jets != len(expected_jet_ids):
        raise RuntimeError(
            f"validation reduced {total_jets} jets, expected {len(expected_jet_ids)}"
        )
    selection_row = reduced.get("loss.total")
    if selection_row is None or selection_row["kind"] != "mean":
        raise RuntimeError("selection loss lacks the reviewed mean reduction schema")
    if selection_row["denominator"] != float(total_jets):
        raise RuntimeError("selection denominator differs from the reduced jet count")
    if not bool(selection_row["selection_eligible"]):
        raise RuntimeError("selection loss is not marked selection eligible")
    if any(
        row["selection_eligible"] and name != "loss.total"
        for name, row in reduced.items()
    ):
        raise RuntimeError("an unreviewed validation metric is selection eligible")
    required = tuple(reference["required_losses"])
    if set(reference["effective_weights"]) != set(required):
        raise RuntimeError("validation effective weights do not cover required losses")
    missing_required_schema = sorted(
        name
        for loss_name in required
        for name in (f"loss.raw.{loss_name}", f"loss.weighted.{loss_name}")
        if name not in reduced or reduced[name]["kind"] != "mean"
    )
    if missing_required_schema:
        raise RuntimeError(
            "required validation loss schemas are missing: "
            + ", ".join(missing_required_schema)
        )
    return {
        "contract": ABPH_DISTRIBUTED_VALIDATION_CONTRACT,
        "selection_score": float(selection_row["value"]),
        "selection_numerator": float(selection_row["numerator"]),
        "selection_denominator": float(selection_row["denominator"]),
        "n_jets": int(total_jets),
        "n_batches": int(total_batches),
        "metrics": metrics,
        "reduction_schema": reduced,
        "non_additive_diagnostics": diagnostic_summaries,
        "required_losses": list(reference["required_losses"]),
        "effective_weights": dict(reference["effective_weights"]),
        "validation_coverage": coverage,
        "checkpoint_selection_eligible": bool(
            coverage is not None or runtime.world_size == 1
        ),
    }


__all__ = [
    "ABPH_DISTRIBUTED_VALIDATION_CONTRACT",
    "TypedValidationAccumulator",
    "finalize_typed_validation",
]
