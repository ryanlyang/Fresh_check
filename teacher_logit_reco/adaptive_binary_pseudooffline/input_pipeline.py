"""Bounded CPU prefetch and asynchronous transfer for ABPH training batches."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass, replace
import time
from typing import Any, Iterable, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from .distributed_stream import GlobalBatchCursor, GlobalBatchPlan


ABPH_INPUT_PIPELINE_CONTRACT = "adaptive_binary_input_pipeline_v1"
ABPH_PREFETCH_QUEUE_DEPTH = 2


def _rebuild_dataclass(value: Any, converted: Mapping[str, Any]) -> Any:
    return replace(value, **converted)


def prepare_contiguous_cpu_batch(value: Any, *, pin_memory: bool) -> Any:
    """Prepare numeric model inputs while preserving nonnumeric CPU metadata.

    Target batches intentionally carry byte-encoded hierarchy identities alongside
    numeric supervision arrays. PyTorch cannot represent those identity arrays, and
    they are provenance rather than model inputs, so keep them as contiguous NumPy
    arrays throughout staging.
    """

    torch = require_torch()
    if isinstance(value, Mapping):
        return {
            key: prepare_contiguous_cpu_batch(item, pin_memory=pin_memory)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        converted = {
            item.name: prepare_contiguous_cpu_batch(
                getattr(value, item.name), pin_memory=pin_memory
            )
            for item in fields(value)
            if item.init
        }
        return _rebuild_dataclass(value, converted)
    if isinstance(value, tuple):
        return tuple(
            prepare_contiguous_cpu_batch(item, pin_memory=pin_memory) for item in value
        )
    if isinstance(value, list):
        return [
            prepare_contiguous_cpu_batch(item, pin_memory=pin_memory) for item in value
        ]
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        if contiguous.dtype.kind not in "biufc":
            return contiguous
        tensor = torch.from_numpy(contiguous)
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("the CPU prefetch worker received a non-CPU tensor")
        tensor = value.contiguous()
    else:
        return value
    if pin_memory and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor


def _move_nested_to_device(value: Any, *, device: Any, non_blocking: bool) -> Any:
    torch = require_torch()
    if isinstance(value, Mapping):
        return {
            key: _move_nested_to_device(
                item, device=device, non_blocking=non_blocking
            )
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        converted = {
            item.name: _move_nested_to_device(
                getattr(value, item.name),
                device=device,
                non_blocking=non_blocking,
            )
            for item in fields(value)
            if item.init
        }
        return _rebuild_dataclass(value, converted)
    if isinstance(value, tuple):
        return tuple(
            _move_nested_to_device(item, device=device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            _move_nested_to_device(item, device=device, non_blocking=non_blocking)
            for item in value
        ]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    return value


def _record_nested_stream(value: Any, stream: Any) -> None:
    torch = require_torch()
    if isinstance(value, Mapping):
        for item in value.values():
            _record_nested_stream(item, stream)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _record_nested_stream(getattr(value, item.name), stream)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _record_nested_stream(item, stream)
        return
    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
        value.record_stream(stream)


@dataclass(frozen=True)
class PrefetchTiming:
    source_prepare_seconds: float
    target_shard_decompression_seconds: float
    pinned_staging_seconds: float


@dataclass(frozen=True)
class _PreparedBatch:
    plan: GlobalBatchPlan
    batch: Any
    timing: PrefetchTiming


@dataclass(frozen=True)
class _PendingBatch:
    global_update: int
    accumulation_index: int
    plan: GlobalBatchPlan
    future: Future[_PreparedBatch]

    @property
    def key(self) -> tuple[int, int]:
        return (self.global_update, self.accumulation_index)


class PrefetchWorkerError(RuntimeError):
    """Context-rich propagation of an exception raised by the CPU worker."""


class RankLocalBatchPrefetcher:
    """One-worker, two-batch prefetcher with consume-time cursor commits."""

    def __init__(
        self,
        source: Any,
        *,
        queue_depth: int = ABPH_PREFETCH_QUEUE_DEPTH,
        pin_memory: bool,
    ) -> None:
        if int(queue_depth) != ABPH_PREFETCH_QUEUE_DEPTH:
            raise ValueError(
                f"ABPH prefetch queue depth is locked to {ABPH_PREFETCH_QUEUE_DEPTH}"
            )
        required = (
            "current_cursor",
            "derive_next_plan",
            "agree_plan_hash",
            "prepare_planned_batch",
            "commit_planned_batch",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(f"target source lacks prefetch methods: {missing}")
        self.source = source
        self.queue_depth = int(queue_depth)
        self.pin_memory = bool(pin_memory)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="abph-rank-prefetch"
        )
        self._pending: deque[_PendingBatch] = deque()
        self._consumed_uncommitted: deque[GlobalBatchPlan] = deque()
        self._producer_cursor: GlobalBatchCursor = source.current_cursor
        self._closed = False
        self.last_timing: PrefetchTiming | None = None
        self.maximum_resident_batches = 0

    @property
    def resident_batches(self) -> int:
        return len(self._pending)

    def _worker_prepare(self, plan: GlobalBatchPlan) -> _PreparedBatch:
        source_started = time.perf_counter()
        raw_batch = self.source.prepare_planned_batch(plan, background_worker=True)
        source_seconds = time.perf_counter() - source_started
        decompression_seconds = (
            float(self.source.consume_background_decompression_seconds())
            if hasattr(self.source, "consume_background_decompression_seconds")
            else 0.0
        )
        pin_started = time.perf_counter()
        batch = prepare_contiguous_cpu_batch(raw_batch, pin_memory=self.pin_memory)
        pin_seconds = time.perf_counter() - pin_started
        return _PreparedBatch(
            plan=plan,
            batch=batch,
            timing=PrefetchTiming(
                source_prepare_seconds=source_seconds,
                target_shard_decompression_seconds=decompression_seconds,
                pinned_staging_seconds=pin_seconds,
            ),
        )

    def prime(self, requests: Iterable[tuple[int, int]]) -> None:
        """Fill available slots in request order; plan agreement stays on the main thread."""

        if self._closed:
            raise RuntimeError("cannot prime a closed ABPH prefetcher")
        queued_keys = {item.key for item in self._pending}
        for raw_update, raw_accumulation in requests:
            key = (int(raw_update), int(raw_accumulation))
            if key in queued_keys:
                continue
            if len(self._pending) >= self.queue_depth:
                break
            plan = self.source.derive_next_plan(
                global_update=key[0],
                accumulation_index=key[1],
                cursor=self._producer_cursor,
            )
            self.source.agree_plan_hash(plan)
            future = self._executor.submit(self._worker_prepare, plan)
            self._pending.append(
                _PendingBatch(
                    global_update=key[0],
                    accumulation_index=key[1],
                    plan=plan,
                    future=future,
                )
            )
            queued_keys.add(key)
            self._producer_cursor = plan.next_cursor
            self.maximum_resident_batches = max(
                self.maximum_resident_batches, len(self._pending)
            )

    def next_planned_batch(
        self, *, global_update: int, accumulation_index: int
    ) -> Any:
        key = (int(global_update), int(accumulation_index))
        if not self._pending:
            self.prime((key,))
        pending = self._pending[0]
        if pending.key != key:
            raise RuntimeError(
                f"prefetch order mismatch: requested {key}, next prepared key is {pending.key}"
            )
        try:
            prepared = pending.future.result()
        except BaseException as exc:
            self.close(cancel_pending=True)
            rank_plan = pending.plan.rank_plans[self.source.rank]
            shard_ids = sorted({item.shard_id for item in rank_plan.slices})
            raise PrefetchWorkerError(
                "ABPH prefetch worker failed for "
                f"split={self.source.split} grouping={self.source.grouping} "
                f"rank={self.source.rank}/{self.source.world_size} "
                f"update={key[0]} accumulation={key[1]} shards={shard_ids}: {exc}"
            ) from exc
        if prepared.plan.plan_hash != pending.plan.plan_hash:
            raise RuntimeError("prefetch worker returned a different global batch plan")
        self._pending.popleft()
        self._consumed_uncommitted.append(prepared.plan)
        self.last_timing = prepared.timing
        return prepared.batch

    def commit_consumed(self) -> tuple[str, ...]:
        """Commit every dequeued plan only after its optimizer update succeeds."""

        committed: list[str] = []
        while self._consumed_uncommitted:
            plan = self._consumed_uncommitted[0]
            self.source.commit_planned_batch(plan)
            committed.append(plan.plan_hash)
            self._consumed_uncommitted.popleft()
        return tuple(committed)

    def reset_to_committed_cursor(self) -> None:
        """Discard speculative work and restart planning from consumed source state."""

        for pending in self._pending:
            pending.future.cancel()
        self._pending.clear()
        self._consumed_uncommitted.clear()
        self._producer_cursor = self.source.current_cursor

    def close(self, *, cancel_pending: bool = True) -> None:
        if self._closed:
            return
        if cancel_pending:
            for pending in self._pending:
                pending.future.cancel()
        self._pending.clear()
        self._consumed_uncommitted.clear()
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
        self._closed = True

    def __enter__(self) -> "RankLocalBatchPrefetcher":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


@dataclass
class StagedDeviceBatch:
    batch: Any
    event: Any | None
    device: Any
    _waited: bool = False

    def wait(self) -> Any:
        """Make the current compute stream wait immediately before batch use."""

        torch = require_torch()
        if self.event is not None and not self._waited:
            compute_stream = torch.cuda.current_stream(self.device)
            compute_stream.wait_event(self.event)
            _record_nested_stream(self.batch, compute_stream)
            self._waited = True
        return self.batch


class BatchTransferStager:
    """Move a prepared batch on a dedicated CUDA stream with one readiness event."""

    def __init__(self, device: Any, *, non_blocking: bool = True) -> None:
        torch = require_torch()
        self.device = torch.device(device)
        self.non_blocking = bool(non_blocking)
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    def stage(self, batch: Any) -> StagedDeviceBatch:
        torch = require_torch()
        if self._stream is None:
            return StagedDeviceBatch(
                batch=_move_nested_to_device(
                    batch, device=self.device, non_blocking=False
                ),
                event=None,
                device=self.device,
            )
        with torch.cuda.stream(self._stream):
            moved = _move_nested_to_device(
                batch,
                device=self.device,
                non_blocking=self.non_blocking,
            )
            event = torch.cuda.Event()
            event.record(self._stream)
        return StagedDeviceBatch(batch=moved, event=event, device=self.device)


__all__ = [
    "ABPH_INPUT_PIPELINE_CONTRACT",
    "ABPH_PREFETCH_QUEUE_DEPTH",
    "BatchTransferStager",
    "PrefetchTiming",
    "PrefetchWorkerError",
    "RankLocalBatchPrefetcher",
    "StagedDeviceBatch",
    "prepare_contiguous_cpu_batch",
]
