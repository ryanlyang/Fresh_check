"""Low-overhead runtime telemetry for ABPH reconstructor training."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterator, Mapping


ABPH_RUNTIME_PROFILE_CONTRACT = "adaptive_binary_pseudooffline_runtime_profile_v2"
ABPH_RUNTIME_PROFILE_BUCKETS: tuple[str, ...] = (
    "optimizer_update_total",
    "target_source_wait",
    "target_shard_decompression",
    "cpu_batch_assembly",
    "pinned_memory_staging",
    "host_to_device",
    "model_step_total",
    "hlt_encode_root_compile",
    "teacher_forced_hierarchy_decode",
    "rollout_hierarchy_decode",
    "particle_render_projection",
    "matching_loss_construction",
    "loss_composition",
    "objective_gradient_diagnostics",
    "backward",
    "gradient_synchronization",
    "optimizer_ema_update",
    "full_validation",
    "validation_source_wait",
    "checkpoint_serialization",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class RuntimeProfileConfig:
    """Sampling controls that do not participate in optimization decisions."""

    enabled: bool = True
    warmup_updates_per_stage: int = 5
    sample_interval: int = 100
    profile_validation: bool = True
    benchmark_mode: bool = False
    benchmark_updates: int = 20

    def __post_init__(self) -> None:
        if int(self.warmup_updates_per_stage) < 0:
            raise ValueError("runtime-profile warmup must be nonnegative")
        if int(self.sample_interval) <= 0:
            raise ValueError("runtime-profile sample interval must be positive")
        if int(self.benchmark_updates) <= 0:
            raise ValueError("runtime-profile benchmark updates must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_RUNTIME_PROFILE_CONTRACT,
            **asdict(self),
        }


@dataclass
class _TimingSample:
    stage_key: str
    scope: str
    bucket: str
    cpu_seconds: float
    cuda_seconds: float | None
    synchronized_wall_seconds: float | None = None


class RuntimeProfiler:
    """Sample spans with one CUDA synchronization at the end of each scope."""

    def __init__(
        self,
        config: RuntimeProfileConfig,
        *,
        device: Any,
        output_dir: str | Path,
        provenance: Mapping[str, Any] | None = None,
        training_config_hash: str | None = None,
        model_metadata_hash: str | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.output_path = Path(output_dir) / "runtime_profile.json"
        self.provenance = _jsonable(dict(provenance or {}))
        self.training_config_hash = training_config_hash
        self.model_metadata_hash = model_metadata_hash
        self._scope: dict[str, Any] | None = None
        self._samples: list[_TimingSample] = []
        self._stage_rows: dict[str, dict[str, Any]] = {}
        self._validation_rows: list[dict[str, Any]] = []
        self._peak_allocated: list[int] = []
        self._peak_reserved: list[int] = []
        self._torch = None
        self._cuda_enabled = False
        if bool(config.enabled):
            try:
                import torch

                self._torch = torch
                self._cuda_enabled = bool(
                    getattr(device, "type", str(device)) == "cuda"
                    and torch.cuda.is_available()
                )
            except ImportError:  # pragma: no cover - torch is required in production
                self._torch = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def active(self) -> bool:
        return self._scope is not None and bool(self._scope.get("sample"))

    def _new_event(self) -> Any | None:
        if not self._cuda_enabled:
            return None
        return self._torch.cuda.Event(enable_timing=True)

    def _record_event(self, event: Any | None) -> None:
        if event is not None:
            event.record()

    def _stage_row(self, state: Any) -> dict[str, Any]:
        key = str(state.stage_key)
        row = self._stage_rows.setdefault(
            key,
            {
                "stage_key": key,
                "phase": int(state.phase),
                "phase_name": str(state.phase_name),
                "stage_maximum_updates": int(state.stage_maximum_updates),
                "stage_nominal_updates": int(
                    getattr(state, "stage_nominal_updates", state.stage_maximum_updates)
                    or state.stage_maximum_updates
                ),
                "stage_hard_max_updates": int(
                    getattr(state, "stage_hard_max_updates", state.stage_maximum_updates)
                    or state.stage_maximum_updates
                ),
                "schedule_contract": str(
                    getattr(state, "schedule_contract", "legacy_fixed_v1")
                ),
                "updates_attempted": 0,
                "updates_completed": 0,
                "updates_failed": 0,
                "sampled_updates": 0,
                "sampled_jets": 0,
                "sampled_jets_scope": "global_effective_batch",
                "local_batch_size": None,
                "gradient_accumulation_steps": None,
                "distributed_world_size": None,
                "sampled_global_batch_plan_hashes": [],
            },
        )
        return row

    def begin_train_update(
        self,
        state: Any,
        *,
        local_batch_size: int | None,
        accumulation_steps: int,
        distributed_world_size: int,
    ) -> None:
        if self._scope is not None:
            raise RuntimeError("runtime profiler already has an active scope")
        row = self._stage_row(state)
        row["updates_attempted"] += 1
        row["local_batch_size"] = (
            None if local_batch_size is None else int(local_batch_size)
        )
        row["gradient_accumulation_steps"] = int(accumulation_steps)
        row["distributed_world_size"] = int(distributed_world_size)
        stage_update = int(state.stage_update)
        after_warmup = stage_update >= int(self.config.warmup_updates_per_stage)
        sample = bool(
            self.enabled
            and after_warmup
            and (
                bool(self.config.benchmark_mode)
                or (
                    stage_update - int(self.config.warmup_updates_per_stage)
                )
                % int(self.config.sample_interval)
                == 0
            )
        )
        start_event = self._new_event() if sample else None
        self._record_event(start_event)
        if sample and self._cuda_enabled:
            self._torch.cuda.reset_peak_memory_stats(self.device)
        self._scope = {
            "kind": "train",
            "stage_key": str(state.stage_key),
            "sample": sample,
            "start_cpu": time.perf_counter(),
            "start_event": start_event,
            "pending": [],
            "global_batch_plan_hashes": [],
        }

    def record_global_batch_plan_hash(self, plan_hash: str | None) -> None:
        if plan_hash is None:
            return
        if self._scope is None or self._scope.get("kind") != "train":
            raise RuntimeError("runtime profiler has no active training update")
        self._scope["global_batch_plan_hashes"].append(str(plan_hash))

    def record_cpu_duration(self, bucket: str, seconds: float) -> None:
        """Attach worker-measured CPU time to the active sampled scope."""

        if bucket not in ABPH_RUNTIME_PROFILE_BUCKETS:
            raise KeyError(f"unknown ABPH runtime bucket {bucket!r}")
        if not self.active:
            return
        duration = float(seconds)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("recorded CPU duration must be finite and nonnegative")
        now = time.perf_counter()
        self._scope["pending"].append(
            {
                "bucket": bucket,
                "start_cpu": now - duration,
                "end_cpu": now,
                "start_event": None,
                "end_event": None,
            }
        )

    def begin_validation(self, state: Any) -> None:
        if self._scope is not None:
            raise RuntimeError("runtime profiler already has an active scope")
        sample = bool(self.enabled and self.config.profile_validation)
        start_event = self._new_event() if sample else None
        self._record_event(start_event)
        if sample and self._cuda_enabled:
            self._torch.cuda.reset_peak_memory_stats(self.device)
        self._scope = {
            "kind": "validation",
            "stage_key": str(state.stage_key),
            "sample": sample,
            "start_cpu": time.perf_counter(),
            "start_event": start_event,
            "pending": [],
        }

    @contextmanager
    def span(self, bucket: str) -> Iterator[None]:
        if bucket not in ABPH_RUNTIME_PROFILE_BUCKETS:
            raise KeyError(f"unknown ABPH runtime bucket {bucket!r}")
        if not self.active:
            yield
            return
        scope = self._scope
        start_cpu = time.perf_counter()
        start_event = self._new_event()
        end_event = self._new_event()
        self._record_event(start_event)
        try:
            yield
        finally:
            self._record_event(end_event)
            scope["pending"].append(
                {
                    "bucket": bucket,
                    "start_cpu": start_cpu,
                    "end_cpu": time.perf_counter(),
                    "start_event": start_event,
                    "end_event": end_event,
                }
            )

    @contextmanager
    def standalone_span(self, bucket: str, *, stage_key: str) -> Iterator[None]:
        if bucket not in ABPH_RUNTIME_PROFILE_BUCKETS:
            raise KeyError(f"unknown ABPH runtime bucket {bucket!r}")
        if not self.enabled:
            yield
            return
        start_cpu = time.perf_counter()
        start_event = self._new_event()
        end_event = self._new_event()
        self._record_event(start_event)
        try:
            yield
        finally:
            self._record_event(end_event)
            if self._cuda_enabled:
                self._torch.cuda.synchronize(self.device)
            synchronized_wall_seconds = max(
                time.perf_counter() - start_cpu, 0.0
            )
            self._samples.append(
                _TimingSample(
                    stage_key=str(stage_key),
                    scope="standalone",
                    bucket=bucket,
                    cpu_seconds=synchronized_wall_seconds,
                    cuda_seconds=(
                        None
                        if start_event is None
                        else max(float(start_event.elapsed_time(end_event)) / 1000.0, 0.0)
                    ),
                    synchronized_wall_seconds=synchronized_wall_seconds,
                )
            )

    def _finish_scope(self, *, total_bucket: str) -> dict[str, Any]:
        if self._scope is None:
            raise RuntimeError("runtime profiler has no active scope")
        scope = self._scope
        end_event = self._new_event() if scope["sample"] else None
        self._record_event(end_event)
        if scope["sample"] and self._cuda_enabled:
            self._torch.cuda.synchronize(self.device)
        end_cpu = time.perf_counter()
        if scope["sample"]:
            for pending in scope["pending"]:
                cuda_seconds = None
                if pending["start_event"] is not None:
                    cuda_seconds = max(
                        float(
                            pending["start_event"].elapsed_time(pending["end_event"])
                        )
                        / 1000.0,
                        0.0,
                    )
                self._samples.append(
                    _TimingSample(
                        stage_key=scope["stage_key"],
                        scope=scope["kind"],
                        bucket=pending["bucket"],
                        cpu_seconds=max(
                            float(pending["end_cpu"] - pending["start_cpu"]), 0.0
                        ),
                        cuda_seconds=cuda_seconds,
                        synchronized_wall_seconds=None,
                    )
                )
            total_cuda = (
                None
                if scope["start_event"] is None
                else max(
                    float(scope["start_event"].elapsed_time(end_event)) / 1000.0,
                    0.0,
                )
            )
            self._samples.append(
                _TimingSample(
                    stage_key=scope["stage_key"],
                    scope=scope["kind"],
                    bucket=total_bucket,
                    cpu_seconds=max(float(end_cpu - scope["start_cpu"]), 0.0),
                    cuda_seconds=total_cuda,
                    synchronized_wall_seconds=max(
                        float(end_cpu - scope["start_cpu"]), 0.0
                    ),
                )
            )
            if self._cuda_enabled:
                self._peak_allocated.append(
                    int(self._torch.cuda.max_memory_allocated(self.device))
                )
                self._peak_reserved.append(
                    int(self._torch.cuda.max_memory_reserved(self.device))
                )
        self._scope = None
        return {
            "sampled": bool(scope["sample"]),
            "cpu_seconds": max(float(end_cpu - scope["start_cpu"]), 0.0),
            "synchronized_wall_seconds": max(
                float(end_cpu - scope["start_cpu"]), 0.0
            ),
        }

    def end_train_update(self, *, success: bool, jets: int) -> None:
        if self._scope is None or self._scope.get("kind") != "train":
            raise RuntimeError("runtime profiler is not timing a training update")
        stage_key = str(self._scope["stage_key"])
        sampled = bool(self._scope["sample"])
        plan_hashes = tuple(self._scope.get("global_batch_plan_hashes", ()))
        self._finish_scope(total_bucket="optimizer_update_total")
        row = self._stage_rows[stage_key]
        if success:
            row["updates_completed"] += 1
        else:
            row["updates_failed"] += 1
        if sampled:
            row["sampled_updates"] += 1
            row["sampled_jets"] += int(jets)
            row["sampled_global_batch_plan_hashes"].append(list(plan_hashes))
            # Persist sparse samples immediately so a wall-time interruption
            # still leaves an actionable profile before the next validation.
            self.write()

    def end_validation(self, *, jets: int, batches: int) -> None:
        if self._scope is None or self._scope.get("kind") != "validation":
            raise RuntimeError("runtime profiler is not timing validation")
        stage_key = str(self._scope["stage_key"])
        summary = self._finish_scope(total_bucket="full_validation")
        self._validation_rows.append(
            {
                "stage_key": stage_key,
                "n_jets": int(jets),
                "n_batches": int(batches),
                "sampled": bool(summary["sampled"]),
                "cpu_seconds": float(summary["cpu_seconds"]),
                "synchronized_wall_seconds": float(
                    summary["synchronized_wall_seconds"]
                ),
                "jets_per_second": (
                    float(jets) / float(summary["synchronized_wall_seconds"])
                    if summary["synchronized_wall_seconds"] > 0.0
                    else None
                ),
            }
        )

    def _bucket_summary(self, bucket: str) -> dict[str, Any]:
        samples = [row for row in self._samples if row.bucket == bucket]
        cpu = [float(row.cpu_seconds) for row in samples]
        cuda = [float(row.cuda_seconds) for row in samples if row.cuda_seconds is not None]
        synchronized_wall = [
            float(row.synchronized_wall_seconds)
            for row in samples
            if row.synchronized_wall_seconds is not None
        ]
        return {
            "status": "measured" if samples else "no_samples",
            "samples": len(samples),
            "cpu_total_seconds": float(sum(cpu)),
            "cpu_median_seconds": float(statistics.median(cpu)) if cpu else None,
            "cuda_total_seconds": float(sum(cuda)) if cuda else None,
            "cuda_median_seconds": float(statistics.median(cuda)) if cuda else None,
            "synchronized_wall_total_seconds": float(sum(synchronized_wall)),
            "synchronized_wall_median_seconds": (
                float(statistics.median(synchronized_wall))
                if synchronized_wall
                else None
            ),
        }

    def _stage_summary(self, row: Mapping[str, Any]) -> dict[str, Any]:
        update_samples = [
            sample.synchronized_wall_seconds
            for sample in self._samples
            if sample.stage_key == row["stage_key"]
            and sample.scope == "train"
            and sample.bucket == "optimizer_update_total"
            and sample.synchronized_wall_seconds is not None
        ]
        median_update = (
            float(statistics.median(update_samples)) if update_samples else None
        )
        sampled_jets = int(row["sampled_jets"])
        sampled_seconds = float(sum(update_samples))
        return {
            **dict(row),
            "median_optimizer_update_seconds": median_update,
            "updates_per_hour": (
                3600.0 / median_update if median_update and median_update > 0.0 else None
            ),
            "sampled_jets_per_second": (
                sampled_jets / sampled_seconds if sampled_seconds > 0.0 else None
            ),
            "projected_maximum_stage_seconds": (
                median_update * int(row["stage_maximum_updates"])
                if median_update is not None
                else None
            ),
            "projected_nominal_stage_seconds": (
                median_update * int(row["stage_nominal_updates"])
                if median_update is not None
                else None
            ),
            "projected_hard_max_stage_seconds": (
                median_update * int(row["stage_hard_max_updates"])
                if median_update is not None
                else None
            ),
        }

    def _device_metadata(self) -> dict[str, Any]:
        result = {
            "device": str(self.device),
            "cuda_enabled": bool(self._cuda_enabled),
        }
        if self._cuda_enabled:
            properties = self._torch.cuda.get_device_properties(self.device)
            result.update(
                {
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": [int(properties.major), int(properties.minor)],
                }
            )
        return result

    @staticmethod
    def _slurm_metadata() -> dict[str, Any]:
        names = (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_NNODES",
            "SLURM_NTASKS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE",
            "SLURM_GPUS",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_PARTITION",
        )
        return {name: os.environ[name] for name in names if name in os.environ}

    def payload(self) -> dict[str, Any]:
        buckets = {
            name: self._bucket_summary(name) for name in ABPH_RUNTIME_PROFILE_BUCKETS
        }
        update_total = float(
            buckets["optimizer_update_total"]["synchronized_wall_total_seconds"]
        )
        data_wait = float(buckets["target_source_wait"]["cpu_total_seconds"])
        communication_bucket = buckets["gradient_synchronization"]
        communication = float(
            communication_bucket["cuda_total_seconds"]
            if communication_bucket["cuda_total_seconds"] is not None
            else communication_bucket["cpu_total_seconds"]
        )
        payload: dict[str, Any] = {
            "contract": ABPH_RUNTIME_PROFILE_CONTRACT,
            "ok": True,
            "config": self.config.to_dict(),
            "training_config_hash": self.training_config_hash,
            "model_metadata_hash": self.model_metadata_hash,
            "provenance": self.provenance,
            "device": self._device_metadata(),
            "slurm": self._slurm_metadata(),
            "required_buckets": list(ABPH_RUNTIME_PROFILE_BUCKETS),
            "buckets": buckets,
            "stages": {
                key: self._stage_summary(row)
                for key, row in sorted(self._stage_rows.items())
            },
            "validations": list(self._validation_rows),
            "summary": {
                "data_wait_fraction": (
                    data_wait / update_total if update_total > 0.0 else None
                ),
                "communication_fraction": (
                    communication / update_total if update_total > 0.0 else None
                ),
                "peak_allocated_bytes": max(self._peak_allocated, default=0),
                "peak_reserved_bytes": max(self._peak_reserved, default=0),
                "sampled_training_updates": sum(
                    int(row["sampled_updates"]) for row in self._stage_rows.values()
                ),
                "validation_count": len(self._validation_rows),
            },
        }
        if not all(
            value is None or math.isfinite(float(value))
            for value in (
                payload["summary"]["data_wait_fraction"],
                payload["summary"]["communication_fraction"],
            )
        ):
            raise FloatingPointError("runtime profile summary is nonfinite")
        payload["profile_content_hash"] = _canonical_hash(payload)
        return payload

    def write(self, *, persist: bool = True) -> dict[str, Any]:
        payload = self.payload()
        if persist:
            _atomic_json(self.output_path, payload)
        return payload


def profile_span(profiler: RuntimeProfiler | None, bucket: str):
    """Return a profiler span or a no-op context for model/source code."""

    if profiler is None:
        return nullcontext()
    return profiler.span(bucket)


__all__ = [
    "ABPH_RUNTIME_PROFILE_BUCKETS",
    "ABPH_RUNTIME_PROFILE_CONTRACT",
    "RuntimeProfileConfig",
    "RuntimeProfiler",
    "profile_span",
]
