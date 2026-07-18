"""Phase-specific runtime batch calibration and immutable production contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import canonical_hash


ABPH_RUNTIME_BATCH_CONTRACT = "adaptive_binary_pseudooffline_runtime_batch_v1"
ABPH_RUNTIME_BATCH_MEASUREMENT_CONTRACT = (
    "adaptive_binary_pseudooffline_full_step_batch_measurement_v2"
)
ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER = "canonical_slurm_full_optimizer_step_v1"
ABPH_RUNTIME_BATCH_STAGE_FAMILIES = ("root_hierarchy", "renderer_distribution")
ABPH_RUNTIME_BATCH_EFFECTIVE_BATCHES = {
    "root_hierarchy": 1024,
    "renderer_distribution": 512,
}
ABPH_RUNTIME_BATCH_CANDIDATES = {
    "root_hierarchy": (256, 128, 64),
    "renderer_distribution": (128, 64, 32),
}
ABPH_RUNTIME_BATCH_MINIMUM_FREE_FRACTION = 0.15


def exact_accumulation_steps(
    stage_family: str, *, world_size: int, local_batch_size: int
) -> int:
    """Return the only accumulation count preserving the locked global batch."""

    family = str(stage_family)
    if family not in ABPH_RUNTIME_BATCH_EFFECTIVE_BATCHES:
        raise ValueError(f"unknown runtime batch stage family {family!r}")
    world = int(world_size)
    local = int(local_batch_size)
    if world <= 0 or local <= 0:
        raise ValueError("world_size and local_batch_size must be positive")
    denominator = world * local
    effective = int(ABPH_RUNTIME_BATCH_EFFECTIVE_BATCHES[family])
    if effective % denominator:
        raise ValueError(
            f"{family} local batch {local} at world size {world} does not exactly "
            f"divide effective batch {effective}"
        )
    accumulation = effective // denominator
    if accumulation <= 0:
        raise ValueError("calibrated accumulation must be positive")
    return accumulation


@dataclass(frozen=True)
class FullStepBatchMeasurement:
    """Auditable worst-rank evidence from one candidate optimizer step."""

    stage_family: str
    variant_name: str
    resolved_variant_config_hash: str
    runtime_provenance_hash: str
    measurement_producer: str
    slurm_job_id: str
    slurm_job_account: str
    slurm_job_partition: str
    local_batch_size: int
    accumulation_steps: int
    requested_world_size: int
    measured_world_size: int
    rank_count: int
    distributed_backend: str
    successful: bool
    all_ranks_completed: bool
    process_group_initialized: bool
    ddp_gradient_buckets_initialized: bool
    find_unused_parameters_exercised: bool
    active_parameter_groups: tuple[str, ...]
    forward_completed: bool
    backward_completed: bool
    gradients_unscaled: bool
    gradients_clipped: bool
    optimizer_step_completed: bool
    adamw_state_materialized: bool
    online_model_resident: bool
    ema_model_resident: bool
    prefetch_buffer_count: int
    pinned_memory_staging: bool
    largest_path_exercised: bool
    total_device_memory_bytes: int
    peak_device_memory_bytes: int
    free_device_memory_bytes_at_peak: int
    rank_measurement_hashes: tuple[str, ...] = ()
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.stage_family not in ABPH_RUNTIME_BATCH_STAGE_FAMILIES:
            raise ValueError(f"unknown stage family {self.stage_family!r}")
        for name in (
            "variant_name",
            "resolved_variant_config_hash",
            "runtime_provenance_hash",
            "slurm_job_id",
            "slurm_job_account",
            "slurm_job_partition",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.measurement_producer != ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER:
            raise ValueError("runtime batch measurement producer is not canonical")
        expected = exact_accumulation_steps(
            self.stage_family,
            world_size=self.requested_world_size,
            local_batch_size=self.local_batch_size,
        )
        if int(self.accumulation_steps) != expected:
            raise ValueError("measurement accumulation is not exact")
        for name in (
            "requested_world_size",
            "measured_world_size",
            "rank_count",
            "local_batch_size",
            "accumulation_steps",
            "total_device_memory_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= int(self.free_device_memory_bytes_at_peak) <= int(
            self.total_device_memory_bytes
        ):
            raise ValueError("free memory at peak is outside device capacity")

    @property
    def free_fraction_at_peak(self) -> float:
        return float(self.free_device_memory_bytes_at_peak) / float(
            self.total_device_memory_bytes
        )

    def production_rejections(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.measured_world_size != self.requested_world_size:
            failures.append("measured_world_size_mismatch")
        if self.rank_count != self.requested_world_size:
            failures.append("rank_count_mismatch")
        if not self.successful:
            failures.append("candidate_step_failed")
        flags = {
            "all_ranks_completed": self.all_ranks_completed,
            "forward_completed": self.forward_completed,
            "backward_completed": self.backward_completed,
            "gradients_unscaled": self.gradients_unscaled,
            "gradients_clipped": self.gradients_clipped,
            "optimizer_step_completed": self.optimizer_step_completed,
            "adamw_state_materialized": self.adamw_state_materialized,
            "online_model_resident": self.online_model_resident,
            "ema_model_resident": self.ema_model_resident,
            "pinned_memory_staging": self.pinned_memory_staging,
            "largest_path_exercised": self.largest_path_exercised,
        }
        failures.extend(name for name, value in flags.items() if not value)
        if self.prefetch_buffer_count < 2:
            failures.append("fewer_than_two_prefetch_buffers")
        if not self.active_parameter_groups:
            failures.append("active_parameter_groups_missing")
        if not self.slurm_job_id.isdigit():
            failures.append("canonical_slurm_job_id_missing")
        if self.requested_world_size > 1:
            if not self.process_group_initialized:
                failures.append("process_group_not_initialized")
            if self.distributed_backend.lower() != "nccl":
                failures.append("production_distributed_backend_is_not_nccl")
            if not self.ddp_gradient_buckets_initialized:
                failures.append("ddp_gradient_buckets_not_initialized")
            if not self.find_unused_parameters_exercised:
                failures.append("find_unused_parameters_not_exercised")
            if len(self.rank_measurement_hashes) != self.requested_world_size:
                failures.append("per_rank_measurements_incomplete")
        if self.free_fraction_at_peak + 1.0e-12 < ABPH_RUNTIME_BATCH_MINIMUM_FREE_FRACTION:
            failures.append("memory_headroom_below_15_percent")
        return tuple(failures)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = ABPH_RUNTIME_BATCH_MEASUREMENT_CONTRACT
        payload["free_fraction_at_peak"] = self.free_fraction_at_peak
        payload["production_rejections"] = list(self.production_rejections())
        payload["measurement_hash"] = canonical_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FullStepBatchMeasurement":
        values = dict(payload)
        if values.pop("contract", None) != ABPH_RUNTIME_BATCH_MEASUREMENT_CONTRACT:
            raise ValueError("runtime batch measurement contract mismatch")
        saved_hash = values.pop("measurement_hash", None)
        values.pop("free_fraction_at_peak", None)
        values.pop("production_rejections", None)
        values["active_parameter_groups"] = tuple(values["active_parameter_groups"])
        values["rank_measurement_hashes"] = tuple(values.get("rank_measurement_hashes", ()))
        result = cls(**values)
        if saved_hash != result.to_dict()["measurement_hash"]:
            raise ValueError("runtime batch measurement hash mismatch")
        return result


@dataclass(frozen=True)
class RuntimeBatchSelection:
    stage_family: str
    local_batch_size: int
    accumulation_steps: int
    effective_global_batch_size: int
    measurement: FullStepBatchMeasurement

    def __post_init__(self) -> None:
        if self.stage_family != self.measurement.stage_family:
            raise ValueError("selection and measurement stage families differ")
        if self.accumulation_steps != exact_accumulation_steps(
            self.stage_family,
            world_size=self.measurement.requested_world_size,
            local_batch_size=self.local_batch_size,
        ):
            raise ValueError("selection accumulation is not exact")
        if self.effective_global_batch_size != ABPH_RUNTIME_BATCH_EFFECTIVE_BATCHES[
            self.stage_family
        ]:
            raise ValueError("selection changes the locked effective global batch")
        if self.measurement.production_rejections():
            raise ValueError(
                "selected measurement is not production eligible: "
                + ", ".join(self.measurement.production_rejections())
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_family": self.stage_family,
            "local_batch_size": self.local_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "effective_global_batch_size": self.effective_global_batch_size,
            "measurement": self.measurement.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeBatchContract:
    variant_name: str
    resolved_variant_config_hash: str
    runtime_provenance_hash: str
    requested_world_size: int
    selections: Mapping[str, RuntimeBatchSelection]
    attempted_measurement_hashes: Mapping[str, tuple[str, ...]]
    production_approved: bool = True

    def __post_init__(self) -> None:
        if not self.variant_name or not self.resolved_variant_config_hash:
            raise ValueError("variant name and resolved config hash are required")
        if not self.runtime_provenance_hash:
            raise ValueError("runtime provenance hash is required")
        if self.requested_world_size <= 0:
            raise ValueError("requested_world_size must be positive")
        expected = set(ABPH_RUNTIME_BATCH_STAGE_FAMILIES)
        if set(self.selections) != expected or set(self.attempted_measurement_hashes) != expected:
            raise ValueError("runtime contract must cover both stage families")
        for family, selection in self.selections.items():
            if selection.stage_family != family:
                raise ValueError("runtime selection key mismatch")
            if selection.measurement.requested_world_size != self.requested_world_size:
                raise ValueError("runtime selection topology differs from contract")
        if not self.production_approved:
            raise ValueError("an immutable production contract must be approved")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": ABPH_RUNTIME_BATCH_CONTRACT,
            "variant_name": self.variant_name,
            "resolved_variant_config_hash": self.resolved_variant_config_hash,
            "runtime_provenance_hash": self.runtime_provenance_hash,
            "requested_world_size": self.requested_world_size,
            "selections": {name: self.selections[name].to_dict() for name in ABPH_RUNTIME_BATCH_STAGE_FAMILIES},
            "attempted_measurement_hashes": {name: list(self.attempted_measurement_hashes[name]) for name in ABPH_RUNTIME_BATCH_STAGE_FAMILIES},
            "production_approved": True,
            "oom_fallback_allowed": False,
        }
        payload["contract_hash"] = canonical_hash(payload)
        return payload

    @property
    def contract_hash(self) -> str:
        return str(self.to_dict()["contract_hash"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeBatchContract":
        values = dict(payload)
        if values.pop("contract", None) != ABPH_RUNTIME_BATCH_CONTRACT:
            raise ValueError("runtime batch contract type mismatch")
        saved_hash = values.pop("contract_hash", None)
        if values.pop("oom_fallback_allowed", None) is not False:
            raise ValueError("runtime batch contract permits an OOM fallback")
        selections: dict[str, RuntimeBatchSelection] = {}
        for family, raw in dict(values.pop("selections")).items():
            item = dict(raw)
            measurement = FullStepBatchMeasurement.from_dict(item.pop("measurement"))
            selections[family] = RuntimeBatchSelection(measurement=measurement, **item)
        attempts = {
            name: tuple(items)
            for name, items in dict(values.pop("attempted_measurement_hashes")).items()
        }
        result = cls(selections=selections, attempted_measurement_hashes=attempts, **values)
        if saved_hash != result.contract_hash:
            raise ValueError("runtime batch contract hash mismatch")
        return result


def calibrate_runtime_batch_contract(
    *,
    variant_name: str,
    resolved_variant_config_hash: str,
    runtime_provenance_hash: str,
    requested_world_size: int,
    probe: Callable[[str, int, int], FullStepBatchMeasurement],
) -> RuntimeBatchContract:
    """Probe descending candidates and fail if either phase has no safe exact batch."""

    selected: dict[str, RuntimeBatchSelection] = {}
    attempted: dict[str, tuple[str, ...]] = {}
    for family in ABPH_RUNTIME_BATCH_STAGE_FAMILIES:
        hashes: list[str] = []
        chosen: RuntimeBatchSelection | None = None
        for local_batch_size in ABPH_RUNTIME_BATCH_CANDIDATES[family]:
            try:
                accumulation = exact_accumulation_steps(
                    family,
                    world_size=requested_world_size,
                    local_batch_size=local_batch_size,
                )
            except ValueError:
                continue
            measurement = probe(family, local_batch_size, accumulation)
            if measurement.stage_family != family or measurement.local_batch_size != local_batch_size:
                raise ValueError("probe returned evidence for the wrong candidate")
            hashes.append(str(measurement.to_dict()["measurement_hash"]))
            if not measurement.production_rejections():
                chosen = RuntimeBatchSelection(
                    stage_family=family,
                    local_batch_size=local_batch_size,
                    accumulation_steps=accumulation,
                    effective_global_batch_size=ABPH_RUNTIME_BATCH_EFFECTIVE_BATCHES[family],
                    measurement=measurement,
                )
                break
        attempted[family] = tuple(hashes)
        if chosen is None:
            raise RuntimeError(
                f"no {family} candidate satisfied exact batch arithmetic, full-step "
                "evidence, topology, and 15% memory headroom"
            )
        selected[family] = chosen
    return RuntimeBatchContract(
        variant_name=variant_name,
        resolved_variant_config_hash=resolved_variant_config_hash,
        runtime_provenance_hash=runtime_provenance_hash,
        requested_world_size=requested_world_size,
        selections=selected,
        attempted_measurement_hashes=attempted,
    )


def write_runtime_batch_contract(path: str | Path, contract: RuntimeBatchContract) -> Path:
    """Atomically create an immutable contract, accepting only identical reuse."""

    target = Path(path)
    if target.exists():
        existing = load_runtime_batch_contract(target)
        if existing.contract_hash != contract.contract_hash:
            raise FileExistsError(f"refusing to replace immutable runtime batch contract {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def write_full_step_measurement(
    path: str | Path, measurement: FullStepBatchMeasurement
) -> Path:
    """Atomically persist one candidate measurement without permitting mutation."""

    target = Path(path)
    payload = measurement.to_dict()
    if target.exists():
        existing = FullStepBatchMeasurement.from_dict(
            json.loads(target.read_text(encoding="utf-8"))
        )
        if existing.to_dict()["measurement_hash"] != payload["measurement_hash"]:
            raise FileExistsError(f"refusing to replace immutable measurement {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def load_runtime_batch_contract(
    path: str | Path,
    *,
    expected_variant_name: str | None = None,
    expected_resolved_variant_config_hash: str | None = None,
    expected_runtime_provenance_hash: str | None = None,
    expected_world_size: int | None = None,
) -> RuntimeBatchContract:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"runtime batch contract is missing: {target}")
    contract = RuntimeBatchContract.from_dict(
        json.loads(target.read_text(encoding="utf-8"))
    )
    expected = {
        "variant_name": expected_variant_name,
        "resolved_variant_config_hash": expected_resolved_variant_config_hash,
        "runtime_provenance_hash": expected_runtime_provenance_hash,
        "requested_world_size": expected_world_size,
    }
    for name, value in expected.items():
        if value is not None and getattr(contract, name) != value:
            raise ValueError(f"runtime batch contract {name} mismatch")
    return contract


__all__ = [
    "ABPH_RUNTIME_BATCH_CANDIDATES",
    "ABPH_RUNTIME_BATCH_CONTRACT",
    "ABPH_RUNTIME_BATCH_EFFECTIVE_BATCHES",
    "ABPH_RUNTIME_BATCH_MEASUREMENT_CONTRACT",
    "ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER",
    "ABPH_RUNTIME_BATCH_MINIMUM_FREE_FRACTION",
    "ABPH_RUNTIME_BATCH_STAGE_FAMILIES",
    "FullStepBatchMeasurement",
    "RuntimeBatchContract",
    "RuntimeBatchSelection",
    "calibrate_runtime_batch_contract",
    "exact_accumulation_steps",
    "load_runtime_batch_contract",
    "write_runtime_batch_contract",
    "write_full_step_measurement",
]
