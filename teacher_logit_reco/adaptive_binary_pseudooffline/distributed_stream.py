"""Deterministic rank-sharded batch plans and validation coverage contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity

from .config import canonical_hash


ABPH_GLOBAL_BATCH_PLAN_CONTRACT = "adaptive_binary_global_batch_plan_v1"
ABPH_GLOBAL_BATCH_CURSOR_CONTRACT = "adaptive_binary_global_batch_cursor_v1"
ABPH_VALIDATION_RANGE_CONTRACT = "adaptive_binary_validation_range_v1"
ABPH_VALIDATION_COVERAGE_CONTRACT = "adaptive_binary_validation_coverage_v1"


@dataclass(frozen=True)
class ShardSlice:
    stream_epoch: int
    shard_id: int
    local_start: int
    local_stop: int

    def __post_init__(self) -> None:
        if self.stream_epoch < 0 or self.shard_id < 0:
            raise ValueError("stream epoch and shard id must be nonnegative")
        if not 0 <= self.local_start < self.local_stop:
            raise ValueError("shard slice must be nonempty and ordered")

    @property
    def n_jets(self) -> int:
        return self.local_stop - self.local_start


@dataclass(frozen=True)
class RankBatchSlicePlan:
    rank: int
    slices: tuple[ShardSlice, ...]
    expected_n_jets: int
    ordered_jet_identity_hash: str

    def __post_init__(self) -> None:
        if self.rank < 0 or self.expected_n_jets <= 0:
            raise ValueError("rank plan has invalid rank or jet count")
        if sum(item.n_jets for item in self.slices) != self.expected_n_jets:
            raise ValueError("rank slices do not sum to the expected local jet count")
        if not self.ordered_jet_identity_hash:
            raise ValueError("rank plan identity hash is required")


@dataclass(frozen=True)
class GlobalBatchCursor:
    epoch: int
    shard_cursor: int
    shard_offset: int
    global_stream_position: int
    shard_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.epoch, self.shard_cursor, self.shard_offset, self.global_stream_position) < 0:
            raise ValueError("global batch cursor values must be nonnegative")
        if not self.shard_order:
            raise ValueError("global batch cursor requires a shard order")
        if not 0 <= self.shard_cursor < len(self.shard_order):
            raise ValueError("global batch cursor shard index is out of range")
        if sorted(self.shard_order) != list(range(len(self.shard_order))):
            raise ValueError("global batch cursor shard order is not a permutation")

    def to_dict(self) -> dict[str, Any]:
        return {"contract": ABPH_GLOBAL_BATCH_CURSOR_CONTRACT, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalBatchCursor":
        values = dict(payload)
        if values.pop("contract", None) != ABPH_GLOBAL_BATCH_CURSOR_CONTRACT:
            raise ValueError("global batch cursor contract mismatch")
        values["shard_order"] = tuple(values["shard_order"])
        return cls(**values)


@dataclass(frozen=True)
class GlobalBatchPlan:
    runtime_contract_hash: str
    split: str
    grouping: str
    epoch: int
    global_update: int
    accumulation_index: int
    global_window_start: int
    global_window_stop: int
    expected_jet_count: int
    expected_rank_local_jet_count: int
    rank_plans: tuple[RankBatchSlicePlan, ...]
    full_window_identity_hash: str
    next_cursor: GlobalBatchCursor

    def __post_init__(self) -> None:
        if not self.runtime_contract_hash or not self.split or not self.grouping:
            raise ValueError("global batch plan provenance is incomplete")
        if min(self.epoch, self.global_update, self.accumulation_index) < 0:
            raise ValueError("global batch plan counters must be nonnegative")
        if self.global_window_stop - self.global_window_start != self.expected_jet_count:
            raise ValueError("global batch plan window/count mismatch")
        if self.expected_jet_count <= 0 or self.expected_rank_local_jet_count <= 0:
            raise ValueError("global batch plan counts must be positive")
        if len(self.rank_plans) * self.expected_rank_local_jet_count != self.expected_jet_count:
            raise ValueError("rank plans do not cover the global window")
        if tuple(item.rank for item in self.rank_plans) != tuple(range(len(self.rank_plans))):
            raise ValueError("rank plans are not canonically ordered")
        if any(item.expected_n_jets != self.expected_rank_local_jet_count for item in self.rank_plans):
            raise ValueError("rank-local batch sizes differ")
        if not self.full_window_identity_hash:
            raise ValueError("full-window identity hash is required")

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "contract": ABPH_GLOBAL_BATCH_PLAN_CONTRACT,
            "runtime_contract_hash": self.runtime_contract_hash,
            "split": self.split,
            "grouping": self.grouping,
            "epoch": self.epoch,
            "global_update": self.global_update,
            "accumulation_index": self.accumulation_index,
            "global_window_start": self.global_window_start,
            "global_window_stop": self.global_window_stop,
            "expected_jet_count": self.expected_jet_count,
            "expected_rank_local_jet_count": self.expected_rank_local_jet_count,
            "rank_plans": [asdict(item) for item in self.rank_plans],
            "full_window_identity_hash": self.full_window_identity_hash,
            "next_cursor": self.next_cursor.to_dict(),
        }

    @property
    def plan_hash(self) -> str:
        return canonical_hash(self.payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_without_hash()
        payload["plan_hash"] = self.plan_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalBatchPlan":
        values = dict(payload)
        saved_hash = values.pop("plan_hash", None)
        if values.pop("contract", None) != ABPH_GLOBAL_BATCH_PLAN_CONTRACT:
            raise ValueError("global batch plan contract mismatch")
        rank_plans = []
        for raw_rank in values.pop("rank_plans"):
            rank_values = dict(raw_rank)
            rank_values["slices"] = tuple(
                ShardSlice(**dict(item)) for item in rank_values["slices"]
            )
            rank_plans.append(RankBatchSlicePlan(**rank_values))
        result = cls(
            rank_plans=tuple(rank_plans),
            next_cursor=GlobalBatchCursor.from_dict(values.pop("next_cursor")),
            **values,
        )
        if saved_hash != result.plan_hash:
            raise ValueError("global batch plan hash mismatch")
        return result


def deterministic_shard_order(
    *, n_shards: int, seed: int, epoch: int, grouping: str, shuffle: bool
) -> tuple[int, ...]:
    order = np.arange(int(n_shards), dtype=np.int64)
    if bool(shuffle):
        grouping_offset = int.from_bytes(
            hashlib.sha256(str(grouping).encode("utf-8")).digest()[:8], "big"
        )
        np.random.default_rng(int(seed) + int(epoch) + grouping_offset).shuffle(order)
    return tuple(int(value) for value in order)


def _shard_bounds(shards: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    expected_start = 0
    for expected_id, raw in enumerate(shards):
        shard_id = int(raw.get("shard_index", -1))
        start = int(raw.get("start", -1))
        stop = int(raw.get("stop", -1))
        if shard_id != expected_id or start != expected_start or stop <= start:
            raise ValueError("target shard metadata is not contiguous and canonical")
        result.append((start, stop))
        expected_start = stop
    if not result:
        raise ValueError("target shard metadata is empty")
    return tuple(result)


def _compress_rank_entries(
    entries: Sequence[tuple[int, int, int, int]],
) -> tuple[ShardSlice, ...]:
    slices: list[ShardSlice] = []
    for stream_epoch, shard_id, local_index, _physical_index in entries:
        if (
            slices
            and slices[-1].stream_epoch == stream_epoch
            and slices[-1].shard_id == shard_id
            and slices[-1].local_stop == local_index
        ):
            previous = slices[-1]
            slices[-1] = ShardSlice(
                stream_epoch=previous.stream_epoch,
                shard_id=previous.shard_id,
                local_start=previous.local_start,
                local_stop=local_index + 1,
            )
        else:
            slices.append(
                ShardSlice(
                    stream_epoch=stream_epoch,
                    shard_id=shard_id,
                    local_start=local_index,
                    local_stop=local_index + 1,
                )
            )
    return tuple(slices)


def derive_global_batch_plan(
    *,
    runtime_contract_hash: str,
    split: str,
    grouping: str,
    cursor: GlobalBatchCursor,
    global_update: int,
    accumulation_index: int,
    rank_count: int,
    local_batch_size: int,
    shards: Sequence[Mapping[str, Any]],
    jet_ids: Sequence[JetIdentity],
    seed: int,
    shuffle: bool,
) -> GlobalBatchPlan:
    """Derive one global window without opening any target shard payload."""

    ranks = int(rank_count)
    local = int(local_batch_size)
    if ranks <= 0 or local <= 0:
        raise ValueError("rank_count and local_batch_size must be positive")
    bounds = _shard_bounds(shards)
    if bounds[-1][1] != len(jet_ids):
        raise ValueError("target shard metadata does not cover the active HLT identities")
    if tuple(cursor.shard_order) != deterministic_shard_order(
        n_shards=len(bounds),
        seed=seed,
        epoch=cursor.epoch,
        grouping=grouping,
        shuffle=shuffle,
    ):
        raise ValueError("cursor shard order differs from the deterministic epoch order")

    required = ranks * local
    stream_epoch = int(cursor.epoch)
    shard_cursor = int(cursor.shard_cursor)
    shard_offset = int(cursor.shard_offset)
    order = tuple(cursor.shard_order)
    entries: list[tuple[int, int, int, int]] = []
    while len(entries) < required:
        shard_id = order[shard_cursor]
        start, stop = bounds[shard_id]
        shard_size = stop - start
        if not 0 <= shard_offset < shard_size:
            raise ValueError("global cursor offset is outside the active shard")
        take = min(required - len(entries), shard_size - shard_offset)
        entries.extend(
            (stream_epoch, shard_id, local_index, start + local_index)
            for local_index in range(shard_offset, shard_offset + take)
        )
        shard_offset += take
        if shard_offset == shard_size:
            shard_cursor += 1
            shard_offset = 0
            if shard_cursor == len(order):
                stream_epoch += 1
                shard_cursor = 0
                order = deterministic_shard_order(
                    n_shards=len(bounds),
                    seed=seed,
                    epoch=stream_epoch,
                    grouping=grouping,
                    shuffle=shuffle,
                )

    rank_plans: list[RankBatchSlicePlan] = []
    full_identities: list[JetIdentity] = []
    for rank in range(ranks):
        rank_entries = entries[rank * local : (rank + 1) * local]
        identities = tuple(jet_ids[item[3]] for item in rank_entries)
        full_identities.extend(identities)
        rank_plans.append(
            RankBatchSlicePlan(
                rank=rank,
                slices=_compress_rank_entries(rank_entries),
                expected_n_jets=local,
                ordered_jet_identity_hash=jet_identity_hash(identities),
            )
        )
    next_cursor = GlobalBatchCursor(
        epoch=stream_epoch,
        shard_cursor=shard_cursor,
        shard_offset=shard_offset,
        global_stream_position=cursor.global_stream_position + required,
        shard_order=order,
    )
    return GlobalBatchPlan(
        runtime_contract_hash=str(runtime_contract_hash),
        split=str(split),
        grouping=str(grouping),
        epoch=cursor.epoch,
        global_update=int(global_update),
        accumulation_index=int(accumulation_index),
        global_window_start=cursor.global_stream_position,
        global_window_stop=cursor.global_stream_position + required,
        expected_jet_count=required,
        expected_rank_local_jet_count=local,
        rank_plans=tuple(rank_plans),
        full_window_identity_hash=jet_identity_hash(tuple(full_identities)),
        next_cursor=next_cursor,
    )


@dataclass(frozen=True)
class ValidationRangeRow:
    rank: int
    split: str
    start: int
    stop: int
    n_jets: int
    range_hash: str

    def __post_init__(self) -> None:
        if self.rank < 0 or not self.split or not 0 <= self.start <= self.stop:
            raise ValueError("validation range is invalid")
        if self.stop - self.start != self.n_jets or not self.range_hash:
            raise ValueError("validation range count/hash is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"contract": ABPH_VALIDATION_RANGE_CONTRACT, **asdict(self)}


def contiguous_validation_range(
    *, n_jets: int, rank: int, world_size: int
) -> tuple[int, int]:
    total = int(n_jets)
    current_rank = int(rank)
    ranks = int(world_size)
    if total < 0 or ranks <= 0 or not 0 <= current_rank < ranks:
        raise ValueError("validation partition inputs are invalid")
    return (total * current_rank // ranks, total * (current_rank + 1) // ranks)


def validation_range_row(
    *, split: str, rank: int, start: int, stop: int, jet_ids: Sequence[JetIdentity]
) -> ValidationRangeRow:
    identities = tuple(jet_ids[int(start) : int(stop)])
    payload = {
        "contract": ABPH_VALIDATION_RANGE_CONTRACT,
        "rank": int(rank),
        "split": str(split),
        "start": int(start),
        "stop": int(stop),
        "n_jets": len(identities),
        "ordered_identities": [item.to_dict() for item in identities],
    }
    return ValidationRangeRow(
        rank=int(rank),
        split=str(split),
        start=int(start),
        stop=int(stop),
        n_jets=len(identities),
        range_hash=canonical_hash(payload),
    )


def compile_validation_coverage(
    *,
    split: str,
    rows: Sequence[ValidationRangeRow | Mapping[str, Any]],
    expected_jet_ids: Sequence[JetIdentity],
    world_size: int,
) -> dict[str, Any]:
    """Verify complete contiguous ranges and their independently expected hashes."""

    normalized: list[ValidationRangeRow] = []
    for raw in rows:
        if isinstance(raw, ValidationRangeRow):
            normalized.append(raw)
            continue
        values = dict(raw)
        values.pop("contract", None)
        normalized.append(ValidationRangeRow(**values))
    ordered = sorted(normalized, key=lambda item: (item.start, item.stop, item.rank))
    if len(ordered) != int(world_size):
        raise ValueError("validation coverage does not contain one row per rank")
    cursor = 0
    for expected_rank, row in enumerate(ordered):
        if row.split != split or row.rank != expected_rank or row.start != cursor:
            raise ValueError("validation ranges contain a gap, overlap, or rank mismatch")
        expected_start, expected_stop = contiguous_validation_range(
            n_jets=len(expected_jet_ids), rank=row.rank, world_size=world_size
        )
        if (row.start, row.stop) != (expected_start, expected_stop):
            raise ValueError("validation range differs from the immutable partition")
        expected = validation_range_row(
            split=split,
            rank=row.rank,
            start=row.start,
            stop=row.stop,
            jet_ids=expected_jet_ids,
        )
        if expected.range_hash != row.range_hash:
            raise ValueError("validation range identity hash mismatch")
        cursor = row.stop
    if cursor != len(expected_jet_ids):
        raise ValueError("validation ranges do not cover the full split")
    canonical_rows = [row.to_dict() for row in ordered]
    coverage = {
        "contract": ABPH_VALIDATION_COVERAGE_CONTRACT,
        "split": str(split),
        "n_jets": len(expected_jet_ids),
        "world_size": int(world_size),
        "ordered_ranges": canonical_rows,
    }
    coverage["validation_coverage_hash"] = canonical_hash(coverage)
    coverage["expected_ordered_row_hash"] = canonical_hash(canonical_rows)
    coverage["selection_eligible"] = True
    return coverage


__all__ = [
    "ABPH_GLOBAL_BATCH_CURSOR_CONTRACT",
    "ABPH_GLOBAL_BATCH_PLAN_CONTRACT",
    "ABPH_VALIDATION_COVERAGE_CONTRACT",
    "ABPH_VALIDATION_RANGE_CONTRACT",
    "GlobalBatchCursor",
    "GlobalBatchPlan",
    "RankBatchSlicePlan",
    "ShardSlice",
    "ValidationRangeRow",
    "compile_validation_coverage",
    "contiguous_validation_range",
    "derive_global_batch_plan",
    "deterministic_shard_order",
    "validation_range_row",
]
