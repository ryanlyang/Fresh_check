from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from teacher_logit_reco.adaptive_binary_pseudooffline.input_pipeline import (
    ABPH_PREFETCH_QUEUE_DEPTH,
    BatchTransferStager,
    PrefetchWorkerError,
    RankLocalBatchPrefetcher,
    prepare_contiguous_cpu_batch,
)


@dataclass(frozen=True)
class _NestedBatch:
    values: object
    mask: object


@dataclass(frozen=True)
class _FakePlan:
    global_update: int
    accumulation_index: int
    start_cursor: int
    next_cursor: int

    @property
    def plan_hash(self) -> str:
        return f"plan-{self.global_update}-{self.accumulation_index}-{self.start_cursor}"

    @property
    def rank_plans(self):
        return (SimpleNamespace(slices=(SimpleNamespace(shard_id=self.start_cursor),)),)


class _FakePlannedSource:
    rank = 0
    world_size = 1
    split = "model_train"
    grouping = "exclusive_kt"

    def __init__(self, *, fail_cursor: int | None = None) -> None:
        self.cursor = 0
        self.fail_cursor = fail_cursor
        self.agreement_threads = []
        self.worker_threads = []

    @property
    def current_cursor(self):
        return self.cursor

    def derive_next_plan(self, *, global_update, accumulation_index, cursor=None):
        start = self.cursor if cursor is None else cursor
        return _FakePlan(global_update, accumulation_index, start, start + 1)

    def agree_plan_hash(self, _plan):
        self.agreement_threads.append(threading.get_ident())

    def prepare_planned_batch(self, plan, *, background_worker=False):
        assert background_worker is True
        self.worker_threads.append(threading.get_ident())
        if plan.start_cursor == self.fail_cursor:
            raise OSError("synthetic shard read failure")
        base = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        return {
            "values": base,
            "global_batch_plan_hash": plan.plan_hash,
        }

    def commit_planned_batch(self, plan):
        if plan.start_cursor != self.cursor:
            raise RuntimeError("out-of-order commit")
        self.cursor = plan.next_cursor


def test_recursive_cpu_preparation_preserves_dataclass_and_contiguity():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    batch = _NestedBatch(values=(source,), mask=np.ones((4, 1), dtype=bool))

    prepared = prepare_contiguous_cpu_batch(batch, pin_memory=False)

    assert isinstance(prepared, _NestedBatch)
    assert isinstance(prepared.values[0], torch.Tensor)
    assert prepared.values[0].is_contiguous()
    assert torch.equal(prepared.values[0], torch.from_numpy(source.copy()))
    assert prepared.mask.dtype == torch.bool


def test_prefetch_is_bounded_ordered_and_commits_only_on_consumption():
    source = _FakePlannedSource()
    main_thread = threading.get_ident()
    prefetcher = RankLocalBatchPrefetcher(
        source, queue_depth=ABPH_PREFETCH_QUEUE_DEPTH, pin_memory=False
    )
    try:
        requests = ((4, 0), (4, 1), (4, 2))
        prefetcher.prime(requests)
        assert prefetcher.resident_batches == 2
        assert prefetcher.maximum_resident_batches == 2
        assert source.cursor == 0
        assert source.agreement_threads == [main_thread, main_thread]

        first = prefetcher.next_planned_batch(global_update=4, accumulation_index=0)
        assert source.cursor == 0
        assert first["values"].is_contiguous()
        prefetcher.prime(requests[1:])
        assert prefetcher.resident_batches == 2

        second = prefetcher.next_planned_batch(global_update=4, accumulation_index=1)
        third = prefetcher.next_planned_batch(global_update=4, accumulation_index=2)
        assert source.cursor == 0
        assert prefetcher.commit_consumed() == (
            "plan-4-0-0",
            "plan-4-1-1",
            "plan-4-2-2",
        )
        assert source.cursor == 3
        assert second["global_batch_plan_hash"].startswith("plan-4-1")
        assert third["global_batch_plan_hash"].startswith("plan-4-2")
        assert source.worker_threads
        assert all(value != main_thread for value in source.worker_threads)
    finally:
        prefetcher.close()


def test_prefetch_worker_error_is_reraised_with_source_context():
    source = _FakePlannedSource(fail_cursor=0)
    prefetcher = RankLocalBatchPrefetcher(source, pin_memory=False)
    prefetcher.prime(((7, 1),))

    with pytest.raises(PrefetchWorkerError, match=r"split=model_train.*rank=0/1.*shards=\[0\]"):
        prefetcher.next_planned_batch(global_update=7, accumulation_index=1)
    assert source.cursor == 0


def test_cpu_transfer_stager_moves_nested_arrays_without_cuda():
    batch = prepare_contiguous_cpu_batch(
        _NestedBatch(
            values=np.arange(6, dtype=np.float32).reshape(2, 3),
            mask=np.ones((2, 1), dtype=bool),
        ),
        pin_memory=False,
    )
    staged = BatchTransferStager(torch.device("cpu")).stage(batch)
    moved = staged.wait()

    assert moved.values.device.type == "cpu"
    assert moved.values.is_contiguous()
    assert staged.event is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_transfer_uses_pinned_inputs_and_a_readiness_event():
    prepared = prepare_contiguous_cpu_batch(
        {"values": np.arange(8, dtype=np.float32)}, pin_memory=True
    )
    assert prepared["values"].is_pinned()
    staged = BatchTransferStager(torch.device("cuda")).stage(prepared)
    moved = staged.wait()
    assert staged.event is not None
    assert moved["values"].device.type == "cuda"
