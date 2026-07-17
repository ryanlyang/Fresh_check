from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    GlobalBatchCursor,
    GlobalBatchPlan,
    compile_validation_coverage,
    contiguous_validation_range,
    derive_global_batch_plan,
    deterministic_shard_order,
    validation_range_row,
)
from teacher_logit_reco.adaptive_binary_pseudooffline import production as production_module
from teacher_logit_reco.adaptive_binary_pseudooffline.production import (
    AdaptiveBinaryTargetBatchSource,
    _concatenate_target_batches,
    _slice_target_batch,
)
from tests.test_adaptive_binary_pseudooffline_production import (
    _hlt_batch,
    _reconstruction_batch,
)


def _identities(count: int):
    return tuple(
        JetIdentity(file="HToBB_010.root", entry=index, label=index % 10)
        for index in range(count)
    )


def _shards(*sizes: int):
    result = []
    start = 0
    for shard_id, size in enumerate(sizes):
        result.append(
            {
                "shard_index": shard_id,
                "start": start,
                "stop": start + size,
            }
        )
        start += size
    return tuple(result)


def _cursor(*, n_shards: int, seed: int = 17):
    return GlobalBatchCursor(
        epoch=0,
        shard_cursor=0,
        shard_offset=0,
        global_stream_position=0,
        shard_order=deterministic_shard_order(
            n_shards=n_shards,
            seed=seed,
            epoch=0,
            grouping="exclusive_kt",
            shuffle=False,
        ),
    )


def test_global_plan_stitches_shard_tails_and_partitions_exact_rank_windows():
    identities = _identities(9)
    plan = derive_global_batch_plan(
        runtime_contract_hash="runtime-hash",
        split="model_train",
        grouping="exclusive_kt",
        cursor=_cursor(n_shards=3),
        global_update=7,
        accumulation_index=1,
        rank_count=4,
        local_batch_size=2,
        shards=_shards(3, 2, 4),
        jet_ids=identities,
        seed=17,
        shuffle=False,
    )
    assert plan.expected_jet_count == 8
    assert [row.expected_n_jets for row in plan.rank_plans] == [2, 2, 2, 2]
    assert [(item.shard_id, item.local_start, item.local_stop) for item in plan.rank_plans[1].slices] == [
        (0, 2, 3),
        (1, 0, 1),
    ]
    assert plan.next_cursor.shard_cursor == 2
    assert plan.next_cursor.shard_offset == 3
    assert plan.next_cursor.global_stream_position == 8


def test_global_plan_crosses_epoch_without_dropping_the_old_epoch_tail():
    identities = _identities(5)
    cursor = GlobalBatchCursor(
        epoch=0,
        shard_cursor=1,
        shard_offset=1,
        global_stream_position=4,
        shard_order=(0, 1),
    )
    plan = derive_global_batch_plan(
        runtime_contract_hash="runtime-hash",
        split="model_train",
        grouping="exclusive_kt",
        cursor=cursor,
        global_update=1,
        accumulation_index=0,
        rank_count=2,
        local_batch_size=2,
        shards=_shards(3, 2),
        jet_ids=identities,
        seed=17,
        shuffle=False,
    )
    slices = [item for rank in plan.rank_plans for item in rank.slices]
    assert [(item.stream_epoch, item.shard_id, item.local_start, item.local_stop) for item in slices] == [
        (0, 1, 1, 2),
        (1, 0, 0, 1),
        (1, 0, 1, 3),
    ]
    assert plan.next_cursor == GlobalBatchCursor(
        epoch=1,
        shard_cursor=1,
        shard_offset=0,
        global_stream_position=8,
        shard_order=(0, 1),
    )


def test_global_plan_round_trip_and_hash_tamper_detection():
    plan = derive_global_batch_plan(
        runtime_contract_hash="runtime-hash",
        split="model_train",
        grouping="exclusive_kt",
        cursor=_cursor(n_shards=2),
        global_update=0,
        accumulation_index=0,
        rank_count=2,
        local_batch_size=2,
        shards=_shards(2, 3),
        jet_ids=_identities(5),
        seed=17,
        shuffle=False,
    )
    assert GlobalBatchPlan.from_dict(plan.to_dict()) == plan
    payload = json.loads(json.dumps(plan.to_dict()))
    payload["global_window_stop"] += 1
    with pytest.raises(ValueError):
        GlobalBatchPlan.from_dict(payload)


def test_validation_ranges_are_disjoint_uneven_and_canonically_covered():
    identities = _identities(10)
    assert [contiguous_validation_range(n_jets=10, rank=rank, world_size=3) for rank in range(3)] == [
        (0, 3),
        (3, 6),
        (6, 10),
    ]
    rows = [
        validation_range_row(
            split="model_val", rank=rank, start=start, stop=stop, jet_ids=identities
        )
        for rank, (start, stop) in enumerate(
            contiguous_validation_range(n_jets=10, rank=rank, world_size=3)
            for rank in range(3)
        )
    ]
    coverage = compile_validation_coverage(
        split="model_val", rows=tuple(reversed(rows)), expected_jet_ids=identities, world_size=3
    )
    assert coverage["selection_eligible"] is True
    assert coverage["n_jets"] == 10
    bad = [row.to_dict() for row in rows]
    bad[1]["range_hash"] = "stale"
    with pytest.raises(ValueError, match="identity hash mismatch"):
        compile_validation_coverage(
            split="model_val", rows=bad, expected_jet_ids=identities, world_size=3
        )


def test_rank_sources_load_only_declared_slices_and_resume_at_exact_next_plan(
    monkeypatch, tmp_path
):
    tokens, mask = _hlt_batch()
    tokens = torch.cat((tokens, tokens), dim=0).numpy()
    mask = torch.cat((mask, mask), dim=0).numpy()
    labels = np.arange(4, dtype=np.int64)
    identities = tuple(
        JetIdentity(file="HToBB_010.root", entry=index, label=int(labels[index]))
        for index in range(4)
    )
    targets = _concatenate_target_batches(
        (_reconstruction_batch()["targets"], _reconstruction_batch()["targets"])
    )
    shards = (
        SimpleNamespace(
            targets=_slice_target_batch(targets, 0, 2),
            jet_ids=identities[:2],
            start=0,
        ),
        SimpleNamespace(
            targets=_slice_target_batch(targets, 2, 4),
            jet_ids=identities[2:],
            start=2,
        ),
    )
    hlt_view = SimpleNamespace(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=identities,
        metadata={"hlt_content_hash": "hlt-hash"},
    )
    metadata = {
        "n_shards": 2,
        "shards": list(_shards(2, 2)),
        "hlt_content_hash": "hlt-hash",
        "jet_identity_hash": production_module.jet_identity_hash(identities),
    }
    loaded_shards = []
    monkeypatch.setattr(
        production_module, "load_cached_hlt_view", lambda *_args, **_kwargs: hlt_view
    )
    monkeypatch.setattr(
        production_module,
        "load_adaptive_binary_target_cache_metadata",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        production_module,
        "load_adaptive_binary_target_shard",
        lambda _root, _split, _grouping, index, **_kwargs: (
            loaded_shards.append(index) or shards[index]
        ),
    )

    def make_source(rank):
        source = AdaptiveBinaryTargetBatchSource(
            hlt_cache_dir="unused",
            target_cache_dir="unused",
            split="model_train",
            grouping="exclusive_kt",
            batch_size=2,
            shuffle_shards=False,
            seed=17,
            rank=rank,
            world_size=2,
            runtime_contract_hash="runtime-hash",
        )
        source._agree_plan_hash = lambda _plan: None
        return source

    rank_zero = make_source(0)
    rank_one = make_source(1)
    rank_zero.set_plan_log_dir(tmp_path / "plans")
    plan_zero = rank_zero.derive_next_plan(global_update=0, accumulation_index=0)
    batch_zero = rank_zero.prepare_planned_batch(plan_zero)
    assert rank_zero.state_dict()["global_cursor"]["global_stream_position"] == 0
    rank_zero.commit_planned_batch(plan_zero)
    assert rank_zero.state_dict()["global_cursor"]["global_stream_position"] == 4
    assert batch_zero["indices"].tolist() == [0, 1]
    assert loaded_shards == [0]
    loaded_shards.clear()
    batch_one = rank_one.next_planned_batch(global_update=0, accumulation_index=0)
    assert batch_one["indices"].tolist() == [2, 3]
    assert loaded_shards == [1]
    assert batch_zero["global_batch_plan_hash"] == batch_one["global_batch_plan_hash"]
    saved_plan = json.loads(
        (tmp_path / "plans" / "update_000000000_accum_0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_plan["plan_hash"] == batch_zero["global_batch_plan_hash"]

    saved = rank_zero.state_dict()
    resumed = make_source(0)
    resumed.load_state_dict(saved)
    expected_next = rank_zero.derive_next_plan(global_update=1, accumulation_index=0)
    resumed_next = resumed.derive_next_plan(global_update=1, accumulation_index=0)
    assert resumed_next.plan_hash == expected_next.plan_hash
    assert resumed_next.global_window_start == 4
