"""Distributed execution and promotion contracts for RAM-backed ABPH taggers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity

from .config import canonical_hash
from .distributed import DistributedRuntime, all_gather_objects


ABPH_TAGGER_DDP_FORWARD_CONTRACT = "adaptive_binary_tagger_ddp_tensor_forward_v1"
ABPH_TAGGER_GLOBAL_BATCH_PLAN_CONTRACT = "adaptive_binary_tagger_global_batch_plan_v1"
ABPH_TAGGER_DDP_ACCEPTANCE_CONTRACT = "adaptive_binary_tagger_ddp_acceptance_v1"
ABPH_TAGGER_DDP_MINIMUM_SPEEDUP = 1.5
ABPH_TAGGER_DDP_MAXIMUM_RELATIVE_LOSS_DELTA = 0.01
ABPH_TAGGER_DDP_MAXIMUM_ACCURACY_DELTA = 0.005


def _detach(value: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {str(name): _detach(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_detach(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor
    if isinstance(value, np.generic):
        return value.item()
    return value


class TaggerTrainingModule(__import__("torch").nn.Module):
    """DDP-visible tagger and optional joint reconstructor objective."""

    def __init__(
        self,
        model: Any,
        reconstructor: Any | None,
        step_function: Callable[[Any, Any | None, Any, str, bool], tuple[Any, Any, Any, Any]],
        objective_function: Callable[..., Any],
        objective_config: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.reconstructor = reconstructor
        self.step_function = step_function
        self.objective_function = objective_function
        self.objective_config = objective_config
        self.last_metadata: dict[str, Any] | None = None

    def forward(
        self,
        batch: Any,
        teacher_logits: Any | None,
        split: str,
        validation: bool,
    ) -> Mapping[str, Any]:
        import torch

        self.last_metadata = None
        output, labels, reconstruction, indices = self.step_function(
            self.model,
            self.reconstructor,
            batch,
            str(split),
            bool(validation),
        )
        objective = self.objective_function(
            output,
            labels,
            self.objective_config,
            split=str(split),
            reconstruction_loss=reconstruction,
            auxiliary_logits=output.auxiliary_logits,
            teacher_logits=teacher_logits,
        )
        finite = (
            objective.total,
            output.logits,
            *tuple(objective.raw_terms.values()),
            *tuple(objective.weighted_terms.values()),
        )
        self.last_metadata = {
            "contract": ABPH_TAGGER_DDP_FORWARD_CONTRACT,
            "required_terms": tuple(objective.raw_terms),
            "root_provenance": _detach(output.diagnostics.get("root_provenance")),
            "objective_diagnostics": _detach(objective.diagnostics),
        }
        return {
            "total_loss": objective.total,
            "raw_loss_terms": dict(objective.raw_terms),
            "weighted_loss_terms": dict(objective.weighted_terms),
            "logits": output.logits,
            "labels": labels,
            "indices": indices,
            "finite_check_tensors": finite,
            "batch_size_tensor": torch.tensor(
                int(labels.shape[0]), dtype=torch.int64, device=labels.device
            ),
        }


def require_tagger_tensor_mapping(value: Any) -> Mapping[str, Any]:
    import torch

    required = {
        "total_loss",
        "raw_loss_terms",
        "weighted_loss_terms",
        "logits",
        "labels",
        "indices",
        "finite_check_tensors",
        "batch_size_tensor",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TypeError("DDP tagger forward must return its exact tensor mapping")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif not isinstance(item, torch.Tensor):
            raise TypeError("DDP tagger output contains a non-tensor leaf")

    visit(value)
    if value["total_loss"].numel() != 1:
        raise ValueError("DDP tagger total loss must be scalar")
    return value


def tagger_tensor_mapping_is_finite(value: Mapping[str, Any]) -> bool:
    import torch

    checked = require_tagger_tensor_mapping(value)
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in checked["finite_check_tensors"]
    )


def build_tagger_ddp_wrapper(
    module: TaggerTrainingModule,
    runtime: DistributedRuntime,
    *,
    device: Any,
) -> Any:
    import torch

    if not runtime.distributed:
        return module
    options: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": True,
    }
    if str(getattr(device, "type", device)) == "cuda":
        options.update(device_ids=[runtime.local_rank], output_device=runtime.local_rank)
    return torch.nn.parallel.DistributedDataParallel(module, **options)


def compile_tagger_global_batch_plan(
    runtime: DistributedRuntime,
    *,
    split: str,
    epoch: int,
    global_update: int,
    indices: Sequence[int],
    jet_ids: Sequence[JetIdentity],
    immutable_rank_range: tuple[int, int] | None = None,
    upstream_plan_hash: str | None = None,
) -> dict[str, Any]:
    """Audit one equal-size all-rank optimizer window before backward."""

    local_indices = tuple(int(value) for value in indices)
    if not local_indices:
        raise ValueError("tagger global batch plan cannot contain an empty rank")
    if len(set(local_indices)) != len(local_indices):
        raise ValueError("rank-local tagger batch contains duplicate identities")
    if immutable_rank_range is not None and any(
        not int(immutable_rank_range[0]) <= value < int(immutable_rank_range[1])
        for value in local_indices
    ):
        raise ValueError("tagger batch escaped its immutable rank range")
    identities = tuple(jet_ids[index] for index in local_indices)
    local = {
        "rank": runtime.rank,
        "indices": list(local_indices),
        "n_jets": len(local_indices),
        "ordered_jet_identity_hash": jet_identity_hash(identities),
        "immutable_rank_range": (
            None if immutable_rank_range is None else list(immutable_rank_range)
        ),
        "upstream_plan_hash": upstream_plan_hash,
    }
    rows = tuple(dict(value) for value in all_gather_objects(runtime, local))
    if tuple(int(row["rank"]) for row in rows) != tuple(range(runtime.world_size)):
        raise RuntimeError("tagger batch-plan ranks are not canonically ordered")
    sizes = {int(row["n_jets"]) for row in rows}
    if len(sizes) != 1:
        raise RuntimeError("tagger DDP ranks have unequal local batch sizes")
    flattened = [int(index) for row in rows for index in row["indices"]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("tagger global batch plan overlaps across ranks")
    for row in rows:
        expected = jet_identity_hash(
            tuple(jet_ids[int(index)] for index in row["indices"])
        )
        if expected != row["ordered_jet_identity_hash"]:
            raise RuntimeError("tagger batch-plan identity hash mismatch")
    upstream_hashes = {row.get("upstream_plan_hash") for row in rows}
    if len(upstream_hashes) != 1:
        raise RuntimeError("tagger ranks disagree on their upstream batch plan")
    payload = {
        "contract": ABPH_TAGGER_GLOBAL_BATCH_PLAN_CONTRACT,
        "split": str(split),
        "epoch": int(epoch),
        "global_update": int(global_update),
        "world_size": runtime.world_size,
        "global_effective_batch": len(flattened),
        "rank_plans": list(rows),
        "full_window_identity_hash": jet_identity_hash(
            tuple(jet_ids[index] for index in flattened)
        ),
    }
    payload["plan_hash"] = canonical_hash(payload)
    return payload


def _read_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise ValueError(f"tagger DDP evidence is invalid: {path}")
    return dict(payload)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tagger_ddp_acceptance(
    *,
    single_reports: Mapping[str, str | Path],
    ddp4_reports: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Compile the E7/F0 scientific-parity and speed promotion gate."""

    names = tuple(sorted(set(single_reports) | set(ddp4_reports)))
    if set(single_reports) != set(ddp4_reports) or not names:
        raise ValueError("tagger DDP acceptance requires paired variants")
    rows = []
    checks: dict[str, bool] = {}
    evidence: dict[str, dict[str, dict[str, str]]] = {}
    for name in names:
        single_path = Path(single_reports[name]).resolve()
        ddp_path = Path(ddp4_reports[name]).resolve()
        single = _read_report(single_path)
        ddp = _read_report(ddp_path)
        evidence[name] = {
            "single": {
                "path": str(single_path),
                "sha256": _sha256_file(single_path),
            },
            "ddp4": {
                "path": str(ddp_path),
                "sha256": _sha256_file(ddp_path),
            },
        }
        if single.get("variant_name") != name or ddp.get("variant_name") != name:
            raise ValueError("tagger DDP evidence variant mismatch")
        single_metrics = single["metrics"]["model_val"]
        ddp_metrics = ddp["metrics"]["model_val"]
        single_runtime = single.get("distributed_runtime", {})
        ddp_runtime = ddp.get("distributed_runtime", {})
        if int(single_runtime.get("world_size", -1)) != 1 or int(
            ddp_runtime.get("world_size", -1)
        ) != 4:
            raise ValueError("tagger DDP evidence has the wrong rank topology")
        left_seconds = float(single_runtime.get("training_wall_seconds", math.nan))
        right_seconds = float(ddp_runtime.get("training_wall_seconds", math.nan))
        if not all(math.isfinite(value) and value > 0 for value in (left_seconds, right_seconds)):
            raise ValueError("tagger DDP evidence lacks measured training time")
        speedup = left_seconds / right_seconds
        left_loss = float(single_metrics["loss"])
        right_loss = float(ddp_metrics["loss"])
        relative_loss = abs(right_loss - left_loss) / max(abs(left_loss), 1.0e-12)
        accuracy_delta = abs(
            float(ddp_metrics["accuracy"]) - float(single_metrics["accuracy"])
        )
        provenance_equal = (
            single.get("provenance", {}).get("model_val", {}).get("jet_identity_hash")
            == ddp.get("provenance", {}).get("model_val", {}).get("jet_identity_hash")
        )
        row_checks = {
            "speedup": speedup >= ABPH_TAGGER_DDP_MINIMUM_SPEEDUP,
            "model_val_loss_parity": relative_loss
            <= ABPH_TAGGER_DDP_MAXIMUM_RELATIVE_LOSS_DELTA,
            "model_val_accuracy_parity": accuracy_delta
            <= ABPH_TAGGER_DDP_MAXIMUM_ACCURACY_DELTA,
            "validation_identity_parity": provenance_equal,
            "ddp_validation_coverage": bool(
                ddp_runtime.get("validation_coverage_hash")
            ),
        }
        checks.update({f"{name}.{key}": value for key, value in row_checks.items()})
        rows.append(
            {
                "variant": name,
                "speedup": speedup,
                "relative_model_val_loss_delta": relative_loss,
                "absolute_model_val_accuracy_delta": accuracy_delta,
                "checks": row_checks,
            }
        )
    report = {
        "contract": ABPH_TAGGER_DDP_ACCEPTANCE_CONTRACT,
        "ok": all(checks.values()),
        "production_mode": "ddp4" if all(checks.values()) else "single",
        "fallback_forbids_persistent_pseudo": True,
        "thresholds": {
            "minimum_speedup": ABPH_TAGGER_DDP_MINIMUM_SPEEDUP,
            "maximum_relative_loss_delta": ABPH_TAGGER_DDP_MAXIMUM_RELATIVE_LOSS_DELTA,
            "maximum_accuracy_delta": ABPH_TAGGER_DDP_MAXIMUM_ACCURACY_DELTA,
        },
        "checks": checks,
        "comparisons": rows,
        "evidence": evidence,
    }
    report["content_hash"] = canonical_hash(report)
    return report


def require_tagger_ddp_acceptance(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    saved = payload.pop("content_hash", None)
    evidence_valid = True
    for modes in dict(payload.get("evidence") or {}).values():
        for row in dict(modes).values():
            evidence_path = Path(str(row.get("path", "")))
            evidence_valid = evidence_valid and evidence_path.is_file()
            if evidence_path.is_file():
                evidence_valid = evidence_valid and _sha256_file(
                    evidence_path
                ) == row.get("sha256")
    if (
        payload.get("contract") != ABPH_TAGGER_DDP_ACCEPTANCE_CONTRACT
        or saved != canonical_hash(payload)
        or payload.get("ok") is not True
        or payload.get("production_mode") != "ddp4"
        or not all(dict(payload.get("checks") or {}).values())
        or not payload.get("evidence")
        or not evidence_valid
    ):
        raise PermissionError("tagger DDP acceptance is missing, stale, or failed")
    return {**payload, "content_hash": saved}


@dataclass
class TaggerWallTimer:
    started: float = 0.0

    def start(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        if self.started <= 0:
            raise RuntimeError("tagger wall timer was not started")
        return time.perf_counter() - self.started


__all__ = [
    "ABPH_TAGGER_DDP_ACCEPTANCE_CONTRACT",
    "ABPH_TAGGER_DDP_FORWARD_CONTRACT",
    "ABPH_TAGGER_DDP_MINIMUM_SPEEDUP",
    "ABPH_TAGGER_GLOBAL_BATCH_PLAN_CONTRACT",
    "TaggerTrainingModule",
    "TaggerWallTimer",
    "build_tagger_ddp_acceptance",
    "build_tagger_ddp_wrapper",
    "compile_tagger_global_batch_plan",
    "require_tagger_ddp_acceptance",
    "require_tagger_tensor_mapping",
    "tagger_tensor_mapping_is_finite",
]
