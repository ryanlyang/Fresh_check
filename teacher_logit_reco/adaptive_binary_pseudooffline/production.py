"""Concrete end-to-end runtime for the adaptive-binary campaign.

This module composes the independently tested Step 1-11 components into the
actual reconstructor used by the Slurm entry points.  Offline targets enter
only the training step; ``deploy`` accepts HLT tensors and nothing privileged.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.hlt_baseline import (
    build_particle_transformer_classifier,
    require_torch,
)
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .binary_accounting import AccountingState
from .cache import (
    AdaptiveBinaryTargetShard,
    load_adaptive_binary_target_cache_metadata,
    load_adaptive_binary_target_shard,
)
from .config import canonical_hash
from .ram_workspace import RankLocalWorkspace
from .target_mode import (
    ABPH_RANK_LOCAL_TARGET_MODE,
    ABPH_SHARED_TRANSIENT_TARGET_MODE,
    build_rank_local_offline_view,
    load_target_mode_selection,
    rank_local_target_metadata,
)
from .distributed_stream import (
    GlobalBatchCursor,
    GlobalBatchPlan,
    ShardSlice,
    contiguous_validation_range,
    derive_global_batch_plan,
    deterministic_shard_order,
    validation_range_row,
)
from .hierarchy_alignment import (
    HierarchyTargetTensors,
    align_recursive_hierarchy,
    build_teacher_parent_frontiers,
    compute_teacher_forced_level_supervision,
    hierarchy_targets_to_tensors,
)
from .hierarchy_decoder import (
    ABPH_GROUP_SUPPORT_DIM,
    ABPH_HLT_SUPPORT_FEATURE_NAMES,
    RecursiveHierarchyDecoder,
    RecursiveHierarchyDecoderConfig,
    RecursiveHierarchyOutput,
)
from .hypothesis_distribution import (
    HierarchyHypothesis,
    HypothesisLatentSet,
    MultiHypothesisHierarchyOutput,
    MultiHypothesisHierarchyReconstructor,
    compute_distribution_losses,
)
from .particle_matching import (
    compute_global_particle_matching_loss,
    compute_local_particle_matching_loss,
    compute_particle_auxiliary_losses,
)
from .particle_renderer import (
    ConstrainedParticleRenderer,
    ParticleRendererConfig,
    RenderedParticleBatch,
)
from .prediction_cache import package_deployable_pseudo_views
from .pseudo_consumer import (
    ABPH_CONSUMER_FRONTIER_FIELDS,
    ABPH_CONSUMER_PARTICLE_FIELDS,
    ABPH_CONSUMER_PSEUDO_CONTRACT,
    ABPH_RENDERER_ONLY_FRONTIER_FIELDS,
    ABPH_RENDERER_ONLY_PARTICLE_FIELDS,
    consumer_pseudo_schema_hash,
)
from .root_compiler import CompiledRootState, ROOT_FEATURE_NAMES, compile_root_state
from .root_model import SemanticRootPrediction, SemanticRootPredictor, SemanticRootPredictorConfig
from .root_objectives import compute_root_losses, compute_root_metrics
from .root_transforms import build_root_residual_targets, summarize_hlt_root, wrap_phi_tensor
from .runtime_profile import RuntimeProfiler, profile_span
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .tagger import (
    NativeStagewiseParticleTransformer,
    PseudoViewInputs,
    WeaverStagewiseParticleTransformer,
)
from .targets import (
    ABPH_LEVEL_CAPACITIES,
    ABPH_TARGET_BUILDER_CONTRACT,
    ABPH_TARGET_BUILDER_VERSION,
    GROUP_FEATURE_NAMES,
    PARTICLE_TARGET_NAMES,
    TOPOLOGY_ACTIVE_TERMINAL,
    AdaptiveBinaryHierarchyLayout,
    AdaptiveBinaryTargetBatch,
    build_adaptive_binary_targets,
)
from .checkpoints import load_torch_checkpoint, selected_model_state
from .training import (
    ReconstructorStepContext,
    ReconstructorStepResult,
    active_reconstruction_loss_names,
    assemble_reconstruction_loss_terms,
)
from .variants import resolve_variant_config


ABPH_PRODUCTION_RUNTIME_CONTRACT = "adaptive_binary_pseudooffline_production_runtime_v1"
ABPH_DISTRIBUTED_TARGET_SOURCE_STATE_CONTRACT = (
    "adaptive_binary_distributed_target_source_state_v2"
)


def _root_only_renderer_hierarchy(
    output: RecursiveHierarchyOutput,
) -> RecursiveHierarchyOutput:
    """Expose one global accounting group to the D3 particle set decoder."""

    terminal_root = replace(
        output.root_frontier,
        topology=require_torch().full_like(
            output.root_frontier.topology, int(TOPOLOGY_ACTIVE_TERMINAL)
        ),
    )
    return RecursiveHierarchyOutput(
        mode=output.mode,
        root_frontier=terminal_root,
        levels=(),
        final_frontier=terminal_root,
        diagnostics={
            **dict(output.diagnostics),
            "renderer_grouping": "single_global_root_set",
            "hierarchy_groups_exposed_to_renderer": False,
        },
    )


class _DirectEightGroupSetDecoder(require_torch().nn.Module):
    """One-shot eight-query set decoder used only by the C0 control.

    All eight groups are predicted in parallel from the compiled root and the
    complete HLT evidence.  No predicted child is fed into another prediction,
    which keeps C0 distinct from the progressive binary hierarchy.
    """

    def __init__(self, input_dim: int, *, d_model: int = 256, blocks: int = 4) -> None:
        torch = require_torch()
        super().__init__()
        self.evidence_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(int(input_dim)),
            torch.nn.Linear(int(input_dim), int(d_model)),
        )
        self.root_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(len(ROOT_FEATURE_NAMES)),
            torch.nn.Linear(len(ROOT_FEATURE_NAMES), int(d_model)),
        )
        self.queries = torch.nn.Parameter(torch.empty(1, 8, int(d_model)))
        torch.nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=int(d_model),
                    nhead=8,
                    dim_feedforward=4 * int(d_model),
                    dropout=0.10,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(blocks))
            ]
        )
        self.ledger_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(d_model)),
            torch.nn.Linear(int(d_model), 2 * int(d_model)),
            torch.nn.GELU(),
            torch.nn.Linear(2 * int(d_model), len(ROOT_FEATURE_NAMES)),
        )
        self.presence_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(d_model)),
            torch.nn.Linear(int(d_model), 1),
        )

    def forward(self, evidence: Any, mask: Any, root_ledger: Any) -> tuple[Any, Any]:
        torch = require_torch()
        particles = self.evidence_projection(torch.as_tensor(evidence))
        valid = torch.as_tensor(mask, device=particles.device).bool()
        root = torch.as_tensor(root_ledger, device=particles.device).float()
        queries = self.queries.expand(particles.shape[0], -1, -1)
        queries = queries + self.root_projection(root)[:, None, :]
        for block in self.blocks:
            queries = block(queries, particles, memory_key_padding_mask=~valid)
        return self.ledger_head(queries), self.presence_head(queries).squeeze(-1)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slice_target_batch(targets: AdaptiveBinaryTargetBatch, start: int, stop: int) -> AdaptiveBinaryTargetBatch:
    slc = slice(int(start), int(stop))
    return AdaptiveBinaryTargetBatch(
        root_features=targets.root_features[slc],
        root_identities=targets.root_identities[slc],
        level_features=tuple(value[slc] for value in targets.level_features),
        level_masks=tuple(value[slc] for value in targets.level_masks),
        level_topology=tuple(value[slc] for value in targets.level_topology),
        level_parent_indices=tuple(value[slc] for value in targets.level_parent_indices),
        level_membership=tuple(value[slc] for value in targets.level_membership),
        level_identities=tuple(value[slc] for value in targets.level_identities),
        particle_targets=targets.particle_targets[slc],
        particle_mask=targets.particle_mask[slc],
        hlt_axis_eta=targets.hlt_axis_eta[slc],
        hlt_axis_phi=targets.hlt_axis_phi[slc],
        valid_hlt_counts=targets.valid_hlt_counts[slc],
        valid_offline_counts=targets.valid_offline_counts[slc],
        layout=targets.layout,
        diagnostics=dict(targets.diagnostics),
    )


def _concatenate_target_batches(
    batches: Sequence[AdaptiveBinaryTargetBatch],
) -> AdaptiveBinaryTargetBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty target-batch sequence")
    if len(batches) == 1:
        return batches[0]
    first = batches[0]
    layout = first.layout.to_dict()
    level_count = len(first.level_features)
    for batch in batches[1:]:
        if batch.layout.to_dict() != layout:
            raise ValueError("target batches use different hierarchy layouts")
        if len(batch.level_features) != level_count:
            raise ValueError("target batches use different hierarchy depths")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(batch, name)) for batch in batches], axis=0
        )

    def concatenate_levels(name: str) -> tuple[np.ndarray, ...]:
        return tuple(
            np.concatenate(
                [np.asarray(getattr(batch, name)[depth]) for batch in batches],
                axis=0,
            )
            for depth in range(level_count)
        )

    return AdaptiveBinaryTargetBatch(
        root_features=concatenate("root_features"),
        root_identities=concatenate("root_identities"),
        level_features=concatenate_levels("level_features"),
        level_masks=concatenate_levels("level_masks"),
        level_topology=concatenate_levels("level_topology"),
        level_parent_indices=concatenate_levels("level_parent_indices"),
        level_membership=concatenate_levels("level_membership"),
        level_identities=concatenate_levels("level_identities"),
        particle_targets=concatenate("particle_targets"),
        particle_mask=concatenate("particle_mask"),
        hlt_axis_eta=concatenate("hlt_axis_eta"),
        hlt_axis_phi=concatenate("hlt_axis_phi"),
        valid_hlt_counts=concatenate("valid_hlt_counts"),
        valid_offline_counts=concatenate("valid_offline_counts"),
        layout=first.layout,
        diagnostics={
            **dict(first.diagnostics),
            "assembled_target_chunks": len(batches),
        },
    )


def _target_batch_nbytes(targets: AdaptiveBinaryTargetBatch) -> int:
    return sum(int(value.nbytes) for value in targets.array_dict().values())


class AdaptiveBinaryTargetBatchSource:
    """Stateful target-shard reader aligned to the immutable HLT cache."""

    def __init__(
        self,
        *,
        hlt_cache_dir: str | Path,
        target_cache_dir: str | Path,
        split: str,
        grouping: str,
        batch_size: int,
        shuffle_shards: bool,
        seed: int,
        maximum_batches: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        runtime_contract_hash: str = ABPH_PRODUCTION_RUNTIME_CONTRACT,
        target_mode_report: str | Path | None = None,
        offline_cache_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        workspace: RankLocalWorkspace | None = None,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.target_cache_dir = str(target_cache_dir)
        self.split = str(split)
        self.grouping = str(grouping)
        self.batch_size = int(batch_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.maximum_batches = None if maximum_batches is None else int(maximum_batches)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("target source rank/world_size is invalid")
        self.runtime_contract_hash = str(runtime_contract_hash)
        if not self.runtime_contract_hash:
            raise ValueError("target source runtime contract hash is required")
        report_path = target_mode_report or os.environ.get("ABPH_TARGET_MODE_REPORT")
        if report_path and not Path(report_path).is_file():
            if (
                target_mode_report is not None
                or os.environ.get("ABPH_STORAGE_PROFILE") == "streaming_30gb_v1"
            ):
                raise FileNotFoundError(
                    f"immutable target-mode selection is missing: {report_path}"
                )
            report_path = None
        self.target_mode = ABPH_SHARED_TRANSIENT_TARGET_MODE
        self.target_mode_selection: Mapping[str, Any] | None = None
        if report_path:
            campaign_root = Path(target_cache_dir).resolve().parent
            canonical_report = (
                campaign_root / "audits" / "target_mode_selection.json"
            ).resolve()
            if Path(report_path).resolve() != canonical_report:
                raise ValueError(
                    "workers must consume the canonical campaign target-mode selection"
                )
            self.target_mode_selection = load_target_mode_selection(
                report_path, campaign_root=campaign_root
            )
            self.target_mode = str(self.target_mode_selection["selected_mode"])
        self._workspace = workspace
        self._hlt_reservation_id: str | None = None
        self._rank_local_cache: OrderedDict[
            tuple[int, int, int], tuple[AdaptiveBinaryTargetShard, str, int]
        ] = OrderedDict()
        self._rank_local_cache_bytes = 0
        self._manifest_path = None if manifest_path is None else Path(manifest_path)
        self._data_dir = None if data_dir is None else str(data_dir)
        self._offline_cache_dir = (
            Path(offline_cache_dir)
            if offline_cache_dir is not None
            else Path(target_cache_dir).resolve().parent / "inputs" / "offline_cache"
        )
        if report_path:
            self._workspace = self._workspace or RankLocalWorkspace.from_environment(
                rank=self.rank
            )
            hlt_metadata_path = Path(hlt_cache_dir) / f"{split}_fixed_hlt_metadata.json"
            hlt_metadata = json.loads(hlt_metadata_path.read_text(encoding="utf-8"))
            expected_hlt_bytes = int(hlt_metadata["n_jets"]) * (
                128 * 14 * 4 + 128 + 16
            )
            self._hlt_reservation_id = self._workspace.reserve(
                owner=f"target_source:{split}:{grouping}:{self.rank}",
                role="immutable_hlt_source_view",
                expected_bytes=max(1, expected_hlt_bytes),
            )
        try:
            self.hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
            if self._workspace is not None and self._hlt_reservation_id is not None:
                measured_hlt_bytes = sum(
                    int(value.nbytes)
                    for value in (
                        self.hlt_view.tokens,
                        self.hlt_view.mask,
                        self.hlt_view.labels,
                    )
                )
                self._workspace.commit(
                    self._hlt_reservation_id, measured_bytes=measured_hlt_bytes
                )
        except Exception:
            if self._workspace is not None and self._hlt_reservation_id is not None:
                self._workspace.release(self._hlt_reservation_id)
            raise
        if self.target_mode == ABPH_RANK_LOCAL_TARGET_MODE:
            if self.target_mode_selection is None:
                raise ValueError("rank-local targets require an immutable mode selection")
            self._manifest_path = self._manifest_path or (
                Path(target_cache_dir).resolve().parent
                / "inputs"
                / "split_manifest"
                / "split_manifest.json.gz"
            )
            self._data_dir = self._data_dir or str(
                self.target_mode_selection.get("data_dir") or ""
            )
            if not self._manifest_path.is_file() or not self._data_dir:
                raise FileNotFoundError(
                    "rank-local target construction requires the manifest and raw data directory"
                )
            offline_metadata_path = self._offline_cache_dir / f"{split}_offline_metadata.json"
            offline_metadata = json.loads(
                offline_metadata_path.read_text(encoding="utf-8")
            )
            split_provenance = dict(
                self.target_mode_selection.get("source_provenance_by_split", {})
            ).get(split)
            if not isinstance(split_provenance, Mapping):
                raise ValueError(f"rank-local target selection lacks {split} provenance")
            for label, actual, expected in (
                (
                    "manifest",
                    self.hlt_view.metadata.get("source_manifest_hash"),
                    split_provenance.get("source_manifest_hash"),
                ),
                (
                    "HLT cache",
                    self.hlt_view.metadata.get("hlt_content_hash"),
                    split_provenance.get("hlt_content_hash"),
                ),
                (
                    "offline cache",
                    offline_metadata.get("offline_content_hash"),
                    split_provenance.get("offline_content_hash"),
                ),
            ):
                if actual in {None, ""} or actual != expected:
                    raise ValueError(f"rank-local target selection is stale for {label}")
            assert self._workspace is not None
            cache_fraction = float(
                os.environ.get("ABPH_RANK_LOCAL_TARGET_CACHE_FRACTION", "0.25")
            )
            if not 0.0 < cache_fraction <= 0.5:
                raise ValueError("rank-local target cache fraction must lie in (0, 0.5]")
            self._rank_local_cache_limit = int(
                self._workspace.reservation_limit_bytes * cache_fraction
            )
            n_jets = len(self.hlt_view.jet_ids)
            metadata = rank_local_target_metadata(
                selection=self.target_mode_selection,
                split=split,
                grouping=grouping,
                n_jets=n_jets,
                jet_identity_hash_value=jet_identity_hash(self.hlt_view.jet_ids),
            )
        else:
            self._rank_local_cache_limit = 0
            metadata = load_adaptive_binary_target_cache_metadata(
                target_cache_dir, split, grouping
            )
        self.metadata = metadata
        self.n_shards = int(metadata["n_shards"])
        self._shard_metadata = tuple(dict(value) for value in metadata.get("shards", ()))
        if len(self._shard_metadata) != self.n_shards:
            raise ValueError("target source requires canonical aggregate shard metadata")
        self._order = list(range(self.n_shards))
        self._epoch = 0
        self._cursor = 0
        self._offset = 0
        self._active = None
        self._active_shard_id: int | None = None
        self._batches_seen = 0
        self._global_stream_position = 0
        self._last_completed_plan: GlobalBatchPlan | None = None
        self._plan_log_dir: Path | None = None
        self._last_validation_range = None
        self._runtime_profiler: RuntimeProfiler | None = None
        self._last_background_decompression_seconds = 0.0
        self._reshuffle()
        if metadata.get("hlt_content_hash") != self.hlt_view.metadata.get("hlt_content_hash"):
            raise ValueError("target source is not bound to the active HLT cache")
        if metadata.get("jet_identity_hash") != jet_identity_hash(self.hlt_view.jet_ids):
            raise ValueError("target source and HLT cache jet identities differ")

    def _reshuffle(self) -> None:
        self._order = list(
            deterministic_shard_order(
                n_shards=self.n_shards,
                seed=self.seed,
                epoch=self._epoch,
                grouping=self.grouping,
                shuffle=self.shuffle_shards,
            )
        )

    def _load_shard(self, shard_id: int, *, profile_runtime: bool = True) -> Any:
        value = int(shard_id)
        if self._active is None or self._active_shard_id != value:
            profiler = self._runtime_profiler if profile_runtime else None
            with profile_span(profiler, "target_shard_decompression"):
                self._active = load_adaptive_binary_target_shard(
                    self.target_cache_dir,
                    self.split,
                    self.grouping,
                    value,
                    verify_hash=True,
                )
            self._active_shard_id = value
        return self._active

    def _evict_rank_local_until(self, expected_bytes: int) -> None:
        if expected_bytes > self._rank_local_cache_limit:
            raise MemoryError(
                "one rank-local target slice exceeds the bounded target-cache allocation"
            )
        while (
            self._rank_local_cache
            and self._rank_local_cache_bytes + expected_bytes
            > self._rank_local_cache_limit
        ):
            _key, (_shard, reservation_id, measured) = self._rank_local_cache.popitem(
                last=False
            )
            self._rank_local_cache_bytes -= measured
            assert self._workspace is not None
            self._workspace.release(reservation_id)

    def _build_rank_local_slice(self, item: ShardSlice) -> AdaptiveBinaryTargetShard:
        if self.target_mode != ABPH_RANK_LOCAL_TARGET_MODE:
            raise RuntimeError("rank-local target builder used in shared-cache mode")
        key = (int(item.shard_id), int(item.local_start), int(item.local_stop))
        cached = self._rank_local_cache.pop(key, None)
        if cached is not None:
            self._rank_local_cache[key] = cached
            return cached[0]
        shard_row = self._shard_metadata[int(item.shard_id)]
        global_start = int(shard_row["start"]) + int(item.local_start)
        global_stop = int(shard_row["start"]) + int(item.local_stop)
        identities = tuple(self.hlt_view.jet_ids[global_start:global_stop])
        measurement = self.target_mode_selection["grouping_measurements"][self.grouping]
        expected = max(
            1,
            math.ceil(
                float(measurement["logical_bytes_per_jet"])
                * len(identities)
                * 1.25
            )
            + len(identities) * (128 * 14 * 4 + 128),
        )
        self._evict_rank_local_until(expected)
        assert self._workspace is not None and self._manifest_path is not None
        reservation_id = self._workspace.reserve(
            owner=f"target_source:{self.split}:{self.grouping}:{self.rank}",
            role=f"planned_slice:{item.shard_id}:{item.local_start}:{item.local_stop}",
            expected_bytes=expected,
        )
        try:
            offline = build_rank_local_offline_view(
                manifest_path=self._manifest_path,
                split=self.split,
                identities=identities,
                data_dir=self._data_dir,
            )
            if tuple(offline.jet_ids) != identities:
                raise ValueError("rank-local raw ROOT identities differ from the batch plan")
            targets = build_adaptive_binary_targets(
                self.hlt_view.tokens[global_start:global_stop],
                self.hlt_view.mask[global_start:global_stop],
                offline.tokens,
                offline.mask,
                jet_ids=identities,
                layout=AdaptiveBinaryHierarchyLayout(grouping=self.grouping),
            )
            measured = _target_batch_nbytes(targets)
            self._workspace.commit(reservation_id, measured_bytes=measured)
        except Exception:
            self._workspace.release(reservation_id)
            raise
        shard = AdaptiveBinaryTargetShard(
            targets=targets,
            labels=np.asarray(self.hlt_view.labels[global_start:global_stop]),
            jet_ids=identities,
            split=self.split,
            grouping=self.grouping,
            shard_index=int(item.shard_id),
            start=global_start,
            stop=global_stop,
            metadata={**self.metadata, "rank_local_slice": key},
        )
        try:
            self._evict_rank_local_until(measured)
        except Exception:
            self._workspace.release(reservation_id)
            raise
        self._rank_local_cache[key] = (shard, reservation_id, measured)
        self._rank_local_cache_bytes += measured
        return shard

    def set_runtime_profiler(self, profiler: RuntimeProfiler | None) -> None:
        self._runtime_profiler = profiler

    def set_plan_log_dir(self, path: str | Path | None) -> None:
        self._plan_log_dir = None if path is None else Path(path)
        if self._plan_log_dir is not None and self.rank == 0:
            self._plan_log_dir.mkdir(parents=True, exist_ok=True)

    def bind_runtime_contract_hash(self, contract_hash: str) -> None:
        if self._batches_seen or self._global_stream_position:
            raise RuntimeError("cannot change runtime contract after stream consumption")
        value = str(contract_hash)
        if not value:
            raise ValueError("runtime contract hash is required")
        self.runtime_contract_hash = value

    def set_batch_size(self, batch_size: int) -> None:
        """Switch phase-local microbatch size without changing the sample cursor."""

        value = int(batch_size)
        if value <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = value

    def _cursor_state(self) -> GlobalBatchCursor:
        return GlobalBatchCursor(
            epoch=int(self._epoch),
            shard_cursor=int(self._cursor),
            shard_offset=int(self._offset),
            global_stream_position=int(self._global_stream_position),
            shard_order=tuple(int(value) for value in self._order),
        )

    @property
    def current_cursor(self) -> GlobalBatchCursor:
        """Return the committed cursor; speculative prefetch never mutates it."""

        return self._cursor_state()

    def derive_next_plan(
        self,
        *,
        global_update: int,
        accumulation_index: int,
        cursor: GlobalBatchCursor | None = None,
    ) -> GlobalBatchPlan:
        return derive_global_batch_plan(
            runtime_contract_hash=self.runtime_contract_hash,
            split=self.split,
            grouping=self.grouping,
            cursor=self._cursor_state() if cursor is None else cursor,
            global_update=int(global_update),
            accumulation_index=int(accumulation_index),
            rank_count=self.world_size,
            local_batch_size=self.batch_size,
            shards=self._shard_metadata,
            jet_ids=self.hlt_view.jet_ids,
            seed=self.seed,
            shuffle=self.shuffle_shards,
        )

    def _agree_plan_hash(self, plan: GlobalBatchPlan) -> None:
        if self.world_size == 1:
            return
        torch = require_torch()
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("distributed target source requires an initialized process group")
        if int(torch.distributed.get_world_size()) != self.world_size:
            raise RuntimeError("target source world size differs from the process group")
        if int(torch.distributed.get_rank()) != self.rank:
            raise RuntimeError("target source rank differs from the process group")
        gathered: list[str | None] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, plan.plan_hash)
        if any(value != plan.plan_hash for value in gathered):
            raise RuntimeError("global batch plan hash differs across ranks")

    def agree_plan_hash(self, plan: GlobalBatchPlan) -> None:
        """Collectively approve a plan before a worker opens target shards."""

        self._agree_plan_hash(plan)

    def _persist_plan(self, plan: GlobalBatchPlan) -> None:
        if self._plan_log_dir is None or self.rank != 0:
            return
        path = self._plan_log_dir / (
            f"update_{plan.global_update:09d}_accum_{plan.accumulation_index:04d}.json"
        )
        payload = plan.to_dict()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("plan_hash") != plan.plan_hash:
                raise FileExistsError(f"refusing to replace a different global batch plan: {path}")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _materialize_slices(
        self,
        slices: Sequence[ShardSlice],
        *,
        expected_identity_hash: str | None = None,
        plan_hash: str | None = None,
        profile_runtime: bool = True,
    ) -> Mapping[str, Any]:
        torch = require_torch()
        token_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        index_chunks: list[np.ndarray] = []
        target_chunks: list[AdaptiveBinaryTargetBatch] = []
        observed_ids = []
        background_decompression_seconds = 0.0
        for item in slices:
            shard_started = time.perf_counter()
            shard = (
                self._build_rank_local_slice(item)
                if self.target_mode == ABPH_RANK_LOCAL_TARGET_MODE
                else self._load_shard(item.shard_id, profile_runtime=profile_runtime)
            )
            if not profile_runtime:
                background_decompression_seconds += time.perf_counter() - shard_started
            if self.target_mode == ABPH_RANK_LOCAL_TARGET_MODE:
                local_start = 0
                local_stop = int(shard.stop) - int(shard.start)
                global_start = int(shard.start)
                global_stop = int(shard.stop)
            else:
                local_start = int(item.local_start)
                local_stop = int(item.local_stop)
                global_start = int(shard.start) + local_start
                global_stop = int(shard.start) + local_stop
            expected_ids = tuple(self.hlt_view.jet_ids[global_start:global_stop])
            actual_ids = tuple(shard.jet_ids[local_start:local_stop])
            if expected_ids != actual_ids:
                raise ValueError("HLT/target planned slice identity alignment failed")
            observed_ids.extend(actual_ids)
            token_chunks.append(np.asarray(self.hlt_view.tokens[global_start:global_stop], dtype=np.float32))
            mask_chunks.append(np.asarray(self.hlt_view.mask[global_start:global_stop], dtype=bool))
            label_chunks.append(np.asarray(self.hlt_view.labels[global_start:global_stop], dtype=np.int64))
            index_chunks.append(np.arange(global_start, global_stop, dtype=np.int64))
            target_chunks.append(_slice_target_batch(shard.targets, local_start, local_stop))
        if not target_chunks:
            raise RuntimeError("planned target source produced an empty batch")
        observed_hash = jet_identity_hash(tuple(observed_ids))
        if expected_identity_hash is not None and observed_hash != expected_identity_hash:
            raise ValueError("materialized rank-local identities differ from the global batch plan")
        if not profile_runtime:
            self._last_background_decompression_seconds = (
                background_decompression_seconds
            )
        return {
            "hlt_tokens": torch.from_numpy(np.concatenate(token_chunks, axis=0)),
            "hlt_mask": torch.from_numpy(np.concatenate(mask_chunks, axis=0)),
            "labels": torch.from_numpy(np.concatenate(label_chunks, axis=0)),
            "targets": _concatenate_target_batches(target_chunks),
            "indices": torch.from_numpy(np.concatenate(index_chunks, axis=0)),
            "global_batch_plan_hash": plan_hash,
        }

    def consume_background_decompression_seconds(self) -> float:
        value = float(self._last_background_decompression_seconds)
        self._last_background_decompression_seconds = 0.0
        return value

    def prepare_planned_batch(
        self, plan: GlobalBatchPlan, *, background_worker: bool = False
    ) -> Mapping[str, Any]:
        """Materialize this rank's slice without advancing the committed cursor."""

        if plan.runtime_contract_hash != self.runtime_contract_hash:
            raise ValueError("prefetched plan runtime contract differs from the source")
        if plan.split != self.split or plan.grouping != self.grouping:
            raise ValueError("prefetched plan split/grouping differs from the source")
        if len(plan.rank_plans) != self.world_size:
            raise ValueError("prefetched plan rank topology differs from the source")
        rank_plan = plan.rank_plans[self.rank]
        return self._materialize_slices(
            rank_plan.slices,
            expected_identity_hash=rank_plan.ordered_jet_identity_hash,
            plan_hash=plan.plan_hash,
            profile_runtime=not background_worker,
        )

    def commit_planned_batch(self, plan: GlobalBatchPlan) -> None:
        """Commit one successfully consumed plan in exact stream order."""

        expected = self.derive_next_plan(
            global_update=plan.global_update,
            accumulation_index=plan.accumulation_index,
        )
        if expected.plan_hash != plan.plan_hash:
            raise RuntimeError(
                "cannot commit a prefetched plan that is not next at the source cursor"
            )
        self._persist_plan(plan)
        next_cursor = plan.next_cursor
        self._epoch = next_cursor.epoch
        self._cursor = next_cursor.shard_cursor
        self._offset = next_cursor.shard_offset
        self._global_stream_position = next_cursor.global_stream_position
        self._order = list(next_cursor.shard_order)
        self._last_completed_plan = plan
        self._batches_seen += 1

    def next_planned_batch(
        self, *, global_update: int, accumulation_index: int
    ) -> Mapping[str, Any]:
        """Agree one global window, then materialize only this rank's slices."""

        with profile_span(self._runtime_profiler, "cpu_batch_assembly"):
            plan = self.derive_next_plan(
                global_update=global_update, accumulation_index=accumulation_index
            )
            self._agree_plan_hash(plan)
            batch = self.prepare_planned_batch(plan)
            self.commit_planned_batch(plan)
            return batch

    def next_batch(self) -> Mapping[str, Any]:
        return self.next_planned_batch(
            global_update=self._batches_seen,
            accumulation_index=0,
        )

    def iter_epoch(self) -> Iterable[Mapping[str, Any]]:
        start, stop = contiguous_validation_range(
            n_jets=len(self.hlt_view.jet_ids),
            rank=self.rank,
            world_size=self.world_size,
        )
        produced = 0
        observed_indices: list[int] = []
        self._last_validation_range = None
        for batch_start in range(start, stop, self.batch_size):
            if self.maximum_batches is not None and produced >= self.maximum_batches:
                break
            batch_stop = min(batch_start + self.batch_size, stop)
            slices: list[ShardSlice] = []
            for shard_id, metadata in enumerate(self._shard_metadata):
                shard_start = int(metadata["start"])
                shard_stop = int(metadata["stop"])
                overlap_start = max(batch_start, shard_start)
                overlap_stop = min(batch_stop, shard_stop)
                if overlap_start < overlap_stop:
                    slices.append(
                        ShardSlice(
                            stream_epoch=0,
                            shard_id=shard_id,
                            local_start=overlap_start - shard_start,
                            local_stop=overlap_stop - shard_start,
                        )
                    )
            batch = self._materialize_slices(tuple(slices))
            indices = [int(value) for value in batch["indices"].tolist()]
            if indices != list(range(batch_start, batch_stop)):
                raise ValueError("validation batch does not match its contiguous range")
            observed_indices.extend(indices)
            produced += 1
            yield batch
        if observed_indices == list(range(start, stop)):
            self._last_validation_range = validation_range_row(
                split=self.split,
                rank=self.rank,
                start=start,
                stop=stop,
                jet_ids=self.hlt_view.jet_ids,
            )

    @property
    def last_validation_range(self):
        return self._last_validation_range

    def state_dict(self) -> dict[str, Any]:
        cursor = self._cursor_state()
        return {
            "contract": ABPH_DISTRIBUTED_TARGET_SOURCE_STATE_CONTRACT,
            "split": self.split,
            "grouping": self.grouping,
            "batch_size": int(self.batch_size),
            "rank": self.rank,
            "world_size": self.world_size,
            "runtime_contract_hash": self.runtime_contract_hash,
            "target_mode": self.target_mode,
            "target_mode_selection_hash": (
                None
                if self.target_mode_selection is None
                else self.target_mode_selection.get("content_hash")
            ),
            "rank_batch_counter": self._batches_seen,
            "global_cursor": cursor.to_dict(),
            "next_plan_cursor": cursor.to_dict(),
            "last_completed_plan": (
                None
                if self._last_completed_plan is None
                else self._last_completed_plan.to_dict()
            ),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("contract") != ABPH_DISTRIBUTED_TARGET_SOURCE_STATE_CONTRACT:
            raise ValueError("target batch-source checkpoint contract mismatch")
        if payload.get("split") != self.split or payload.get("grouping") != self.grouping:
            raise ValueError("target batch-source split/grouping mismatch")
        if int(payload.get("rank", -1)) != self.rank or int(
            payload.get("world_size", -1)
        ) != self.world_size:
            raise ValueError("target batch-source rank topology mismatch")
        if payload.get("runtime_contract_hash") != self.runtime_contract_hash:
            raise ValueError("target batch-source runtime contract mismatch")
        if payload.get("target_mode", ABPH_SHARED_TRANSIENT_TARGET_MODE) != self.target_mode:
            raise ValueError("target batch-source target mode mismatch")
        expected_selection_hash = (
            None
            if self.target_mode_selection is None
            else self.target_mode_selection.get("content_hash")
        )
        if payload.get("target_mode_selection_hash") != expected_selection_hash:
            raise ValueError("target batch-source target-mode selection mismatch")
        saved_batch_size = int(payload.get("batch_size", self.batch_size))
        if saved_batch_size <= 0:
            raise ValueError("target batch-source checkpoint batch size is invalid")
        self.batch_size = saved_batch_size
        cursor = GlobalBatchCursor.from_dict(payload["global_cursor"])
        next_cursor = GlobalBatchCursor.from_dict(payload["next_plan_cursor"])
        if cursor != next_cursor:
            raise ValueError("checkpoint global cursor differs from the exact next-plan cursor")
        self._epoch = cursor.epoch
        self._cursor = cursor.shard_cursor
        self._offset = cursor.shard_offset
        self._global_stream_position = cursor.global_stream_position
        self._order = list(cursor.shard_order)
        self._batches_seen = int(payload.get("rank_batch_counter", 0))
        raw_plan = payload.get("last_completed_plan")
        self._last_completed_plan = (
            None if raw_plan is None else GlobalBatchPlan.from_dict(raw_plan)
        )
        if (
            self._last_completed_plan is not None
            and self._last_completed_plan.next_cursor != cursor
        ):
            raise ValueError("checkpoint plan does not lead to the saved next cursor")
        self._active = None
        self._active_shard_id = None
        if self._workspace is not None:
            for _shard, reservation_id, _measured in self._rank_local_cache.values():
                self._workspace.release(reservation_id)
        self._rank_local_cache.clear()
        self._rank_local_cache_bytes = 0


class _ParameterGroup:
    def __init__(
        self,
        name: str,
        parameters: Sequence[Any],
        *,
        allow_empty: bool = False,
    ) -> None:
        self.name = str(name)
        self._parameters = tuple(parameters)
        self.allow_empty = bool(allow_empty)
        self.shared_across_depths = False

    def parameters(self) -> Iterable[Any]:
        return iter(self._parameters)


@dataclass(frozen=True)
class AdaptiveBinaryReconstructionOutput:
    root_prediction: SemanticRootPrediction
    compiled_root: CompiledRootState
    root_state: AccountingState
    hlt_particle_embeddings: Any
    hlt_jet_embedding: Any
    hlt_support_features: Any
    hierarchy_output: MultiHypothesisHierarchyOutput
    rendered_views: Mapping[str, tuple[RenderedParticleBatch, ...]]


@dataclass(frozen=True)
class HypothesisRenderBundle:
    """An ordered hypothesis subset and the exact particle objects it rendered."""

    latent_set: HypothesisLatentSet
    hypotheses: tuple[HierarchyHypothesis, ...]
    rendered_views: Mapping[str, tuple[RenderedParticleBatch, ...]]
    hypothesis_indices: tuple[int, ...]


class AdaptiveBinaryReconstructorModel(require_torch().nn.Module):
    """Performance-first HLT -> shared root -> hierarchy -> particle model."""

    def __init__(
        self,
        *,
        hierarchy_names: Sequence[str] = ("exclusive_kt",),
        variant_name: str = "D1_kt32_mh4_particles",
        num_classes: int = 10,
        model_size: str = "large",
        smoke: bool = False,
    ) -> None:
        torch = require_torch()
        super().__init__()
        names = tuple(str(value) for value in hierarchy_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("reconstructor requires unique hierarchy names")
        self.hierarchy_names = names
        self.variant_name = str(variant_name)
        self.resolved_variant = resolve_variant_config(self.variant_name)
        self.variant_run_id = str(self.resolved_variant["variant"]["run_id"])
        root_options = self.resolved_variant["model"]["root_predictor"]
        hierarchy_options = self.resolved_variant["model"]["hierarchy"]
        renderer_options = self.resolved_variant["model"]["renderer"]
        distribution_options = self.resolved_variant["model"]["distribution"]
        self.oracle_root = root_options.get("kind") == "oracle_root"
        self.oracle_parent_rollout = bool(hierarchy_options.get("oracle_parent_rollout"))
        self.oracle_groups = bool(renderer_options.get("oracle_groups"))
        self.oracle_offline_particles = renderer_options.get("kind") == "oracle_offline_particles"
        self.global_particle_set = renderer_options.get("kind") == "global_set"
        self.direct_group_set = hierarchy_options.get("kind") == "direct_set"
        self.constrained_hierarchy = bool(hierarchy_options.get("constrained", True))
        self.sample_root = bool(distribution_options.get("sample_root"))
        self.smoke = bool(smoke)
        if smoke:
            self.hlt_encoder = NativeStagewiseParticleTransformer(
                input_dim=len(PF_FEATURE_NAMES),
                model_dim=32,
                num_layers=2,
                num_heads=4,
                num_classes=int(num_classes),
            )
        else:
            reference = build_particle_transformer_classifier(
                num_classes=int(num_classes), model_size=model_size
            )
            self.hlt_encoder = WeaverStagewiseParticleTransformer(reference)
        model_dim = int(self.hlt_encoder.model_dim)
        self.root_predictor = SemanticRootPredictor(
            SemanticRootPredictorConfig(
                input_dim=model_dim,
                jet_input_dim=model_dim,
                d_model=256,
                num_heads=8,
                query_blocks=int(root_options.get("query_blocks", 4)),
                ffn_dim=1024,
                dropout=0.10,
                attention_dropout=0.10,
                architecture_kind=(
                    "pooled_mlp" if root_options.get("kind") == "pooled_mlp" else "semantic_query"
                ),
                probabilistic=root_options.get("kind") != "semantic_query_deterministic",
            )
        )
        decoder_config = RecursiveHierarchyDecoderConfig(
            hlt_input_dims=(model_dim,),
            d_model=256,
            num_heads=8,
            ffn_dim=1024,
            blocks_per_level=(1 if smoke else 4),
            dropout=(0.0 if smoke else 0.10),
            attention_dropout=(0.0 if smoke else 0.10),
            root_semantic_dim=256,
            latent_dim=64,
        )
        decoders = {name: RecursiveHierarchyDecoder(decoder_config) for name in names}
        if not bool(hierarchy_options.get("level_specific_weights", True)):
            for decoder in decoders.values():
                shared_level = decoder.levels[0]
                decoder.levels = torch.nn.ModuleList(
                    [shared_level for _ in ABPH_LEVEL_CAPACITIES]
                )
        self.hierarchy_reconstructor = MultiHypothesisHierarchyReconstructor(decoders)
        self.direct_set_decoder = (
            _DirectEightGroupSetDecoder(
                model_dim, d_model=256, blocks=(1 if smoke else 4)
            )
            if self.direct_group_set
            else None
        )
        self.unconstrained_child_heads = torch.nn.ModuleList()
        if not self.constrained_hierarchy:
            self.unconstrained_child_heads.extend(
                torch.nn.Sequential(
                    torch.nn.LayerNorm(256),
                    torch.nn.Linear(256, 1024),
                    torch.nn.GELU(),
                    torch.nn.Linear(1024, 2 * (len(ROOT_FEATURE_NAMES) + 1)),
                )
                for _ in ABPH_LEVEL_CAPACITIES
            )
        renderer_config = ParticleRendererConfig(
            hlt_input_dims=(model_dim,),
            d_model=256,
            num_heads=8,
            ffn_dim=1024,
            blocks=(1 if smoke else 4),
            dropout=(0.0 if smoke else 0.10),
            attention_dropout=(0.0 if smoke else 0.10),
            root_semantic_dim=256,
            latent_dim=64,
            phase_space_iterations=(8 if smoke else 64),
            type_sinkhorn_iterations=(8 if smoke else 40),
            exact_nbody_projection=bool(renderer_options.get("exact_nbody_projection", True)),
            local_matching=bool(renderer_options.get("local_matching", True)),
        )
        self.renderers = torch.nn.ModuleDict(
            {name: ConstrainedParticleRenderer(renderer_config) for name in names}
        )

    @property
    def renderer(self) -> ConstrainedParticleRenderer:
        """Single-hierarchy compatibility accessor used by the curriculum."""

        return self.renderers[self.hierarchy_names[0]]

    @staticmethod
    def _support_features(tokens: Any, mask: Any) -> tuple[Any, Any, Any]:
        torch = require_torch()
        raw = torch.as_tensor(tokens).float()
        valid = torch.as_tensor(mask, device=raw.device).bool()
        summary = summarize_hlt_root(raw, valid)
        eta_axis = summary.kinematics.eta
        phi_axis = summary.kinematics.phi
        known = raw[..., 5:10]
        other = (known.max(dim=-1).values <= 0.5).to(raw.dtype).unsqueeze(-1)
        pid = torch.cat((known, other), dim=-1)
        support = torch.cat(
            (
                (raw[..., 1] - eta_axis[:, None]).unsqueeze(-1),
                wrap_phi_tensor(raw[..., 2] - phi_axis[:, None]).unsqueeze(-1),
                torch.log(raw[..., 0].clamp_min(1.0e-8)).unsqueeze(-1),
                pid,
            ),
            dim=-1,
        )
        support = support * valid.unsqueeze(-1)
        if support.shape[-1] != len(ABPH_HLT_SUPPORT_FEATURE_NAMES):
            raise RuntimeError("HLT support feature schema mismatch")
        return support, eta_axis, phi_axis

    def encode_hlt(self, tokens: Any, mask: Any) -> tuple[Any, Any, Any]:
        part = build_part_inputs_torch(tokens, mask, max_constits=ABPH_MAX_PARTICLES)
        state = self.hlt_encoder.prepare(
            part["features"], part["lorentz_vectors"], part["mask"]
        )
        state = self.hlt_encoder.run_layers(state, 0, self.hlt_encoder.num_layers)
        jet, logits = self.hlt_encoder.pool_and_classify(state)
        return state.tokens, jet, logits

    def prepare_shared_reconstruction_forward(
        self,
        hlt_tokens: Any,
        hlt_mask: Any,
        *,
        runtime_profiler: RuntimeProfiler | None = None,
    ) -> Mapping[str, Any]:
        """Compile the one root graph consumed by every hierarchy branch."""

        with profile_span(runtime_profiler, "hlt_encode_root_compile"):
            torch = require_torch()
            tokens = torch.as_tensor(hlt_tokens).float()
            mask = torch.as_tensor(hlt_mask, device=tokens.device).bool()
            particle_embeddings, jet_embedding, _ = self.encode_hlt(tokens, mask)
            prediction = self.root_predictor(
                particle_embeddings, mask, jet_embedding=jet_embedding
            )
            compiled = compile_root_state(prediction, tokens, mask)
            support, axis_eta, axis_phi = self._support_features(tokens, mask)
        return {
            "tokens": tokens,
            "mask": mask,
            "particle_embeddings": particle_embeddings,
            "jet_embedding": jet_embedding,
            "root_prediction": prediction,
            "compiled_root": compiled,
            "support": support,
            "axis_eta": axis_eta,
            "axis_phi": axis_phi,
        }

    def module_groups(self) -> Mapping[str, Any]:
        claimed: set[int] = set()

        def take(parameters: Iterable[Any]) -> tuple[Any, ...]:
            result = []
            for parameter in parameters:
                if id(parameter) not in claimed:
                    result.append(parameter)
                    claimed.add(id(parameter))
            return tuple(result)

        groups: dict[str, Any] = {
            "hlt_encoder": _ParameterGroup("hlt_encoder", take(self.hlt_encoder.parameters())),
            "root": _ParameterGroup("root", take(self.root_predictor.parameters())),
        }
        decoders = tuple(self.hierarchy_reconstructor.decoders.values())
        for index, capacity in enumerate(ABPH_LEVEL_CAPACITIES):
            parameters = list(take(
                parameter
                for decoder in decoders
                for parameter in decoder.levels[index].parameters()
            ))
            if self.direct_set_decoder is not None and int(capacity) == 8:
                parameters.extend(take(self.direct_set_decoder.parameters()))
            if self.unconstrained_child_heads:
                parameters.extend(take(self.unconstrained_child_heads[index].parameters()))
            groups[f"hierarchy_{capacity}"] = _ParameterGroup(
                f"hierarchy_{capacity}",
                tuple(parameters),
                allow_empty=(
                    not bool(
                        self.resolved_variant["model"]["hierarchy"].get(
                            "level_specific_weights", True
                        )
                    )
                    and index > 0
                ),
            )
            if (
                index == 0
                and not bool(
                    self.resolved_variant["model"]["hierarchy"].get(
                        "level_specific_weights", True
                    )
                )
            ):
                groups[f"hierarchy_{capacity}"].shared_across_depths = True
        # Decoder projections/embeddings are root-conditioned shared machinery.
        root_extra = take(
            parameter
            for decoder in decoders
            for name, parameter in decoder.named_parameters()
            if not name.startswith("levels.")
        )
        groups["root"] = _ParameterGroup(
            "root", tuple(groups["root"].parameters()) + root_extra
        )
        groups["renderer"] = _ParameterGroup("renderer", take(self.renderers.parameters()))
        groups["distribution"] = _ParameterGroup(
            "distribution", take(self.hierarchy_reconstructor.latent_model.parameters())
        )
        leftovers = take(self.parameters())
        if leftovers:
            raise RuntimeError(f"production module grouping left {len(leftovers)} parameters unassigned")
        return groups

    def sample_compiled_roots(
        self,
        hlt_tokens: Any,
        hlt_mask: Any,
        *,
        count: int = 4,
        seed: int = 24731,
    ) -> tuple[CompiledRootState, ...]:
        """Draw calibrated B3 root hypotheses without offline information."""

        torch = require_torch()
        if int(count) < 2:
            raise ValueError("sampled-root diagnostic requires at least two roots")
        tokens = torch.as_tensor(hlt_tokens).float()
        mask = torch.as_tensor(hlt_mask, device=tokens.device).bool()
        particle_embeddings, jet_embedding, _ = self.encode_hlt(tokens, mask)
        prediction = self.root_predictor(
            particle_embeddings, mask, jet_embedding=jet_embedding
        )
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(int(seed))

        def noise_like(value: Any) -> Any:
            return torch.randn(
                value.shape,
                generator=generator,
                device=value.device,
                dtype=value.dtype,
            )

        results = []
        for _ in range(int(count)):
            composition_scale = prediction.composition_log_scale.exp().mean(
                dim=-1, keepdim=True
            )
            sampled = replace(
                prediction,
                p4_residual_mean=(
                    prediction.p4_residual_mean
                    + noise_like(prediction.p4_residual_mean)
                    * prediction.p4_residual_log_scale.exp()
                ),
                count_logits=(
                    prediction.count_logits
                    + noise_like(prediction.count_logits)
                    * prediction.delta_count_log_scale.exp()[:, None]
                ),
                type_count_logits=(
                    prediction.type_count_logits
                    + noise_like(prediction.type_count_logits) * composition_scale
                ),
                type_pt_logits=(
                    prediction.type_pt_logits
                    + noise_like(prediction.type_pt_logits) * composition_scale
                ),
                type_energy_logits=(
                    prediction.type_energy_logits
                    + noise_like(prediction.type_energy_logits) * composition_scale
                ),
                charge_logits=(
                    prediction.charge_logits
                    + noise_like(prediction.charge_logits)
                    * prediction.absolute_charge_log_scale.exp()[:, None]
                ),
                shape_raw=(
                    prediction.shape_raw
                    + noise_like(prediction.shape_raw)
                    * torch.cat(
                        (
                            prediction.shape_log_scale.exp(),
                            prediction.shape_log_scale[:, -1:].exp(),
                        ),
                        dim=-1,
                    )
                ),
                diagnostics={
                    **dict(prediction.diagnostics),
                    "sampled_root_hypothesis": True,
                },
            )
            results.append(compile_root_state(sampled, tokens, mask))
        return tuple(results)

    def _render_hypotheses(
        self,
        *,
        root_prediction: SemanticRootPrediction,
        hypotheses: Sequence[HierarchyHypothesis],
        particle_embeddings: Any,
        hlt_mask: Any,
        support: Any,
        axis_eta: Any,
        axis_phi: Any,
        runtime_profiler: RuntimeProfiler | None = None,
    ) -> Mapping[str, tuple[RenderedParticleBatch, ...]]:
        rendered: dict[str, list[RenderedParticleBatch]] = {
            name: [] for name in self.hierarchy_names
        }
        with profile_span(runtime_profiler, "particle_render_projection"):
            for hypothesis in hypotheses:
                hypothesis_index = int(hypothesis.identity.index)
                for name in self.hierarchy_names:
                    renderer_hierarchy = hypothesis.hierarchy_outputs[name]
                    if self.global_particle_set:
                        renderer_hierarchy = _root_only_renderer_hierarchy(
                            renderer_hierarchy
                        )
                    rendered[name].append(
                        self.renderers[name](
                            renderer_hierarchy,
                            root_prediction.query_tokens,
                            particle_embeddings,
                            hlt_mask,
                            support,
                            hypothesis.latent,
                            axis_eta,
                            axis_phi,
                            hypothesis_index=hypothesis_index,
                        )
                    )
        return {name: tuple(values) for name, values in rendered.items()}

    def _prepare_deployment_latent_set(
        self,
        *,
        root_state: AccountingState,
        root_prediction: SemanticRootPrediction,
        particle_embeddings: Any,
        hlt_mask: Any,
        evaluation_seed: int,
    ) -> HypothesisLatentSet:
        return self.hierarchy_reconstructor.prepare_deployment_hypotheses(
            root_state,
            root_prediction.shared_context,
            root_prediction.query_tokens,
            particle_embeddings,
            hlt_mask,
            evaluation_seed=int(evaluation_seed),
        )

    def _rollout_and_render_hypothesis_indices(
        self,
        *,
        root_state: AccountingState,
        root_prediction: SemanticRootPrediction,
        particle_embeddings: Any,
        hlt_mask: Any,
        support: Any,
        axis_eta: Any,
        axis_phi: Any,
        latent_set: HypothesisLatentSet,
        hypothesis_indices: Sequence[int],
        runtime_profiler: RuntimeProfiler | None = None,
    ) -> HypothesisRenderBundle:
        indices = tuple(int(value) for value in hypothesis_indices)
        with profile_span(runtime_profiler, "rollout_hierarchy_decode"):
            hypotheses = self.hierarchy_reconstructor.rollout_deployment_hypotheses(
                root_state,
                root_prediction.shared_context,
                root_prediction.query_tokens,
                particle_embeddings,
                hlt_mask,
                support,
                latent_set=latent_set,
                hypothesis_indices=indices,
            )
        rendered = self._render_hypotheses(
            root_prediction=root_prediction,
            hypotheses=hypotheses,
            particle_embeddings=particle_embeddings,
            hlt_mask=hlt_mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            runtime_profiler=runtime_profiler,
        )
        return HypothesisRenderBundle(
            latent_set=latent_set,
            hypotheses=hypotheses,
            rendered_views=rendered,
            hypothesis_indices=indices,
        )

    def rollout_and_render_hypothesis_zero(
        self,
        *,
        root_state: AccountingState,
        root_prediction: SemanticRootPrediction,
        particle_embeddings: Any,
        hlt_mask: Any,
        support: Any,
        axis_eta: Any,
        axis_phi: Any,
        evaluation_seed: int,
        latent_set: HypothesisLatentSet | None = None,
        runtime_profiler: RuntimeProfiler | None = None,
    ) -> HypothesisRenderBundle:
        """Create the deterministic reference view once for all phase-4 losses."""

        prepared = latent_set or self._prepare_deployment_latent_set(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=hlt_mask,
            evaluation_seed=evaluation_seed,
        )
        if int(prepared.identities[0].index) != 0:
            raise RuntimeError("the prepared reference hypothesis is not index zero")
        return self._rollout_and_render_hypothesis_indices(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=hlt_mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            latent_set=prepared,
            hypothesis_indices=(0,),
            runtime_profiler=runtime_profiler,
        )

    def rollout_and_render_additional_hypotheses(
        self,
        *,
        root_state: AccountingState,
        root_prediction: SemanticRootPrediction,
        particle_embeddings: Any,
        hlt_mask: Any,
        support: Any,
        axis_eta: Any,
        axis_phi: Any,
        latent_set: HypothesisLatentSet,
        start_index: int = 1,
        runtime_profiler: RuntimeProfiler | None = None,
    ) -> HypothesisRenderBundle:
        """Render only the stochastic tail from an existing prepared latent set."""

        start = int(start_index)
        if start != 1:
            raise ValueError("additional deployable hypotheses must begin at index one")
        return self._rollout_and_render_hypothesis_indices(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=hlt_mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            latent_set=latent_set,
            hypothesis_indices=range(start, latent_set.count),
            runtime_profiler=runtime_profiler,
        )

    def assemble_deployment_output(
        self,
        *,
        root_prediction: SemanticRootPrediction,
        compiled_root: CompiledRootState,
        root_state: AccountingState,
        particle_embeddings: Any,
        jet_embedding: Any,
        support: Any,
        hypothesis_zero: HypothesisRenderBundle,
        additional_hypotheses: HypothesisRenderBundle,
    ) -> AdaptiveBinaryReconstructionOutput:
        """Assemble the standard output while preserving hypothesis-zero identity."""

        if hypothesis_zero.latent_set is not additional_hypotheses.latent_set:
            raise ValueError("deployment bundles were built from different latent sets")
        if hypothesis_zero.hypothesis_indices != (0,):
            raise ValueError("the reference deployment bundle is not hypothesis zero")
        if additional_hypotheses.hypothesis_indices != tuple(
            range(1, hypothesis_zero.latent_set.count)
        ):
            raise ValueError("the additional deployment bundle is incomplete or reordered")
        hypotheses = hypothesis_zero.hypotheses + additional_hypotheses.hypotheses
        hierarchy = self.hierarchy_reconstructor.assemble_deployment_rollout(
            root_state,
            latent_set=hypothesis_zero.latent_set,
            hypotheses=hypotheses,
        )
        rendered: dict[str, tuple[RenderedParticleBatch, ...]] = {}
        for name in self.hierarchy_names:
            zero_views = hypothesis_zero.rendered_views[name]
            extra_views = additional_hypotheses.rendered_views[name]
            if len(zero_views) != 1:
                raise RuntimeError("hypothesis-zero bundle must contain one rendered view")
            rendered[name] = zero_views + extra_views
            if rendered[name][0] is not zero_views[0]:
                raise RuntimeError("deployment assembly copied the hypothesis-zero view")
        return AdaptiveBinaryReconstructionOutput(
            root_prediction=root_prediction,
            compiled_root=compiled_root,
            root_state=root_state,
            hlt_particle_embeddings=particle_embeddings,
            hlt_jet_embedding=jet_embedding,
            hlt_support_features=support,
            hierarchy_output=hierarchy,
            rendered_views=rendered,
        )

    def rollout_and_render(
        self,
        *,
        root_state: AccountingState,
        root_prediction: SemanticRootPrediction,
        particle_embeddings: Any,
        hlt_mask: Any,
        support: Any,
        axis_eta: Any,
        axis_phi: Any,
        evaluation_seed: int,
        runtime_profiler: RuntimeProfiler | None = None,
    ) -> tuple[MultiHypothesisHierarchyOutput, Mapping[str, tuple[RenderedParticleBatch, ...]]]:
        """Compatibility API assembled from one reference and one tail rollout."""

        zero = self.rollout_and_render_hypothesis_zero(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=hlt_mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            evaluation_seed=evaluation_seed,
            runtime_profiler=runtime_profiler,
        )
        additional = self.rollout_and_render_additional_hypotheses(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=hlt_mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            latent_set=zero.latent_set,
            start_index=1,
            runtime_profiler=runtime_profiler,
        )
        hierarchy = self.hierarchy_reconstructor.assemble_deployment_rollout(
            root_state,
            latent_set=zero.latent_set,
            hypotheses=zero.hypotheses + additional.hypotheses,
        )
        return hierarchy, {
            name: zero.rendered_views[name] + additional.rendered_views[name]
            for name in self.hierarchy_names
        }

    def deploy(
        self,
        hlt_tokens: Any,
        hlt_mask: Any,
        *,
        evaluation_seed: int,
    ) -> AdaptiveBinaryReconstructionOutput:
        shared_forward = self.prepare_shared_reconstruction_forward(
            hlt_tokens, hlt_mask
        )
        return self.deploy_from_shared_reconstruction_forward(
            shared_forward, evaluation_seed=evaluation_seed
        )

    def deploy_from_shared_reconstruction_forward(
        self,
        shared_forward: Mapping[str, Any],
        *,
        evaluation_seed: int,
    ) -> AdaptiveBinaryReconstructionOutput:
        """Roll out all deployable views from one already-compiled root graph."""

        if self.oracle_root or self.oracle_parent_rollout or self.oracle_groups or self.oracle_offline_particles:
            raise RuntimeError(
                f"{self.variant_name} is an offline-target diagnostic and has no deployable HLT-only path"
            )
        required = {
            "tokens",
            "mask",
            "particle_embeddings",
            "jet_embedding",
            "root_prediction",
            "compiled_root",
            "support",
            "axis_eta",
            "axis_phi",
        }
        missing = sorted(required - set(shared_forward))
        if missing:
            raise ValueError(f"shared reconstruction forward lacks {missing}")
        mask = shared_forward["mask"]
        particle_embeddings = shared_forward["particle_embeddings"]
        jet_embedding = shared_forward["jet_embedding"]
        root_prediction = shared_forward["root_prediction"]
        compiled = shared_forward["compiled_root"]
        root_state = AccountingState.from_ledger(compiled.root_ledger)
        support = shared_forward["support"]
        axis_eta = shared_forward["axis_eta"]
        axis_phi = shared_forward["axis_phi"]
        zero = self.rollout_and_render_hypothesis_zero(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            evaluation_seed=evaluation_seed,
        )
        additional = self.rollout_and_render_additional_hypotheses(
            root_state=root_state,
            root_prediction=root_prediction,
            particle_embeddings=particle_embeddings,
            hlt_mask=mask,
            support=support,
            axis_eta=axis_eta,
            axis_phi=axis_phi,
            latent_set=zero.latent_set,
            start_index=1,
        )
        return self.assemble_deployment_output(
            root_prediction=root_prediction,
            compiled_root=compiled,
            root_state=root_state,
            particle_embeddings=particle_embeddings,
            jet_embedding=jet_embedding,
            support=support,
            hypothesis_zero=zero,
            additional_hypotheses=additional,
        )

    def predict_deployable_batch(self, batch: Any, *, evaluation_seed: int) -> Any:
        output = self.deploy(batch.tokens, batch.mask, evaluation_seed=evaluation_seed)
        return package_deployable_pseudo_views(
            output.hierarchy_output, output.rendered_views
        )


def package_trainable_pseudo_views(
    output: AdaptiveBinaryReconstructionOutput,
    *,
    consumer_only: bool = True,
) -> PseudoViewInputs:
    """Torch-native Step-9 schema that preserves tagger gradients to the renderer."""

    torch = require_torch()
    hierarchy = output.hierarchy_output
    hypotheses = tuple(hierarchy.hypotheses)
    view_names = tuple(item.identity.name for item in hypotheses)
    arrays: dict[str, Any] = {
        "shared_root_ledger": hierarchy.shared_root_ledger,
        "hypothesis_latent": torch.stack([item.latent for item in hypotheses], dim=1),
        "hypothesis_prior_log_prob": torch.stack(
            [item.prior_log_prob for item in hypotheses], dim=1
        ),
    }
    frontier_depths: dict[str, int] = {}
    particle_fields = ABPH_CONSUMER_PARTICLE_FIELDS
    frontier_fields = ABPH_CONSUMER_FRONTIER_FIELDS
    if not consumer_only:
        particle_fields = (*particle_fields, *ABPH_RENDERER_ONLY_PARTICLE_FIELDS)
        frontier_fields = (*frontier_fields, *ABPH_RENDERER_ONLY_FRONTIER_FIELDS)
    for name in output.rendered_views:
        rendered = output.rendered_views[name]
        for field_name in particle_fields:
            arrays[f"particle__{name}__{field_name}"] = torch.stack(
                [getattr(item, field_name) for item in rendered], dim=1
            )
        hierarchy_outputs = [item.hierarchy_outputs[name] for item in hypotheses]
        depth_count = 1 + len(hierarchy_outputs[0].levels)
        frontier_depths[name] = depth_count
        for depth in range(depth_count):
            frontiers = [
                item.root_frontier if depth == 0 else item.levels[depth - 1].next_frontier
                for item in hierarchy_outputs
            ]
            for field_name in frontier_fields:
                arrays[
                    f"frontier__{name}__depth_{depth:02d}__{field_name}"
                ] = torch.stack([getattr(item, field_name) for item in frontiers], dim=1)
    diagnostics = {
        "offline_inputs_loaded": False,
        "teacher_logits_loaded": False,
        "offline_target_selected_hypothesis": False,
        "fixed_evaluation_hypotheses": True,
        "torch_native_joint_training": True,
        "consumer_only_pseudo": bool(consumer_only),
        "consumer_pseudo_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
        "renderer_only_fields_retained": not bool(consumer_only),
    }
    diagnostics["consumer_pseudo_schema_hash"] = consumer_pseudo_schema_hash(
        arrays,
        hierarchy_names=tuple(output.rendered_views),
        frontier_depths=frontier_depths,
    )
    result = PseudoViewInputs(
        arrays=arrays,
        view_names=view_names,
        hierarchy_names=tuple(output.rendered_views),
        frontier_depths=frontier_depths,
        diagnostics=diagnostics,
    )
    result.validate()
    return result


def _truncate_targets(targets: HierarchyTargetTensors, depth_count: int) -> HierarchyTargetTensors:
    count = int(depth_count)
    return HierarchyTargetTensors(
        root_ledger=targets.root_ledger,
        level_ledgers=targets.level_ledgers[:count],
        level_supports=targets.level_supports[:count],
        level_masks=targets.level_masks[:count],
        level_topology=targets.level_topology[:count],
        level_parent_indices=targets.level_parent_indices[:count],
        level_membership=targets.level_membership[:count],
        particle_mask=targets.particle_mask,
    )


def _truncate_output(output: RecursiveHierarchyOutput, depth_count: int) -> RecursiveHierarchyOutput:
    levels = output.levels[: int(depth_count)]
    return replace(output, levels=levels, final_frontier=levels[-1].next_frontier)


def _distribution_observables(rendered: Sequence[RenderedParticleBatch]) -> Any:
    torch = require_torch()
    rows = []
    for value in rendered:
        pt = torch.linalg.vector_norm(value.four_vector[..., 1:3], dim=-1) * value.mask
        ordered = pt.sort(dim=-1, descending=True).values[:, :8]
        total = pt.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        rows.append(torch.cat((ordered / total, value.uncertainty.mean(dim=-1, keepdim=True)), dim=-1))
    return torch.stack(rows, dim=1)


def _target_distribution_observables(targets: AdaptiveBinaryTargetBatch, device: Any) -> Any:
    torch = require_torch()
    values = torch.as_tensor(targets.particle_targets, device=device).float()
    mask = torch.as_tensor(targets.particle_mask, device=device).bool()
    pt_index = PARTICLE_TARGET_NAMES.index("pt")
    pt = values[..., pt_index] * mask
    ordered = pt.sort(dim=-1, descending=True).values[:, :8]
    total = pt.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.cat((ordered / total, torch.zeros_like(total)), dim=-1)


def _one_shot_eight_group_losses(
    model: AdaptiveBinaryReconstructorModel,
    prediction: SemanticRootPrediction,
    compiled: CompiledRootState,
    particle_embeddings: Any,
    mask: Any,
    target_tensors: HierarchyTargetTensors,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """C0 objective: all groups are queried in parallel with no child rollout."""

    torch = require_torch()
    if model.direct_set_decoder is None:
        raise RuntimeError("direct-set loss requested without a direct set decoder")
    depth = ABPH_LEVEL_CAPACITIES.index(8)
    target = target_tensors.level_ledgers[depth]
    target_mask = target_tensors.level_masks[depth]
    predicted_normalized, presence_logits = model.direct_set_decoder(
        particle_embeddings, mask, compiled.root_ledger
    )
    scale = compiled.root_ledger.abs().clamp_min(1.0)[:, None, :]
    target_normalized = target / scale
    valid = target_mask.unsqueeze(-1).to(target.dtype)
    ledger = torch.nn.functional.smooth_l1_loss(
        predicted_normalized * valid,
        target_normalized * valid,
        reduction="sum",
    ) / valid.sum().clamp_min(1.0)
    presence = torch.nn.functional.binary_cross_entropy_with_logits(
        presence_logits, target_mask.to(presence_logits.dtype)
    )
    return ledger + 0.25 * presence, presence, {
        "direct_parallel_queries": 8,
        "predicted_child_conditioning": False,
        "binary_rollout_used": False,
        "target_active_groups": float(target_mask.float().sum(dim=1).mean().detach().cpu()),
    }


def _unconstrained_level_losses(
    model: AdaptiveBinaryReconstructorModel,
    supervised_outputs: Sequence[Any],
    target_tensors: HierarchyTargetTensors,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Raw absolute child-set controls with no accounting compiler in the loss path."""

    torch = require_torch()
    group_losses = []
    topology_losses = []
    for depth, supervised in enumerate(supervised_outputs):
        flat_indices = supervised.flat_parent_indices
        parent_context = supervised.parent_context[supervised.parent_frontier.mask]
        raw = model.unconstrained_child_heads[depth](parent_context).reshape(
            -1, 2, len(ROOT_FEATURE_NAMES) + 1
        )
        normalized_prediction = raw[..., : len(ROOT_FEATURE_NAMES)]
        presence_logits = raw[..., -1]
        target_ledger = target_tensors.level_ledgers[depth]
        target_mask = target_tensors.level_masks[depth]
        target_parents = target_tensors.level_parent_indices[depth]
        normalized_target = torch.zeros_like(normalized_prediction)
        child_present = torch.zeros_like(presence_logits, dtype=torch.bool)
        for flat_index in range(int(flat_indices.shape[0])):
            batch_index = int(flat_indices[flat_index, 0])
            parent_index = int(flat_indices[flat_index, 1])
            selected = torch.nonzero(
                target_mask[batch_index]
                & (target_parents[batch_index] == parent_index),
                as_tuple=False,
            ).flatten()
            if int(selected.numel()) > 2:
                raise RuntimeError("binary target contains more than two children for one parent")
            count = int(selected.numel())
            if count:
                parent = supervised.parent_frontier.ledger[batch_index, parent_index]
                scale = parent.abs().clamp_min(1.0)
                normalized_target[flat_index, :count] = (
                    target_ledger[batch_index, selected] / scale[None, :]
                )
                child_present[flat_index, :count] = True
        valid = child_present.unsqueeze(-1).to(normalized_prediction.dtype)
        ledger_loss = torch.nn.functional.smooth_l1_loss(
            normalized_prediction * valid,
            normalized_target * valid,
            reduction="sum",
        ) / valid.sum().clamp_min(1.0)
        presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            presence_logits, child_present.to(presence_logits.dtype)
        )
        group_losses.append(ledger_loss + 0.25 * presence_loss)
        topology_losses.append(presence_loss)
    return tuple(group_losses), tuple(topology_losses)


def reconstructor_step(
    model: AdaptiveBinaryReconstructorModel,
    batch: Mapping[str, Any],
    context: ReconstructorStepContext,
) -> ReconstructorStepResult:
    """Typed Step-8 objective using the real compiled hierarchy and renderer."""

    torch = require_torch()
    runtime_profiler = context.runtime_profiler
    stochastic_seed = (
        24731
        if context.validation
        else int(context.stochastic_seed if context.stochastic_seed is not None else 24731)
    )
    tokens = torch.as_tensor(batch["hlt_tokens"]).float()
    mask = torch.as_tensor(batch["hlt_mask"], device=tokens.device).bool()
    targets: AdaptiveBinaryTargetBatch = batch["targets"]
    hierarchy_name = str(batch.get("hierarchy_name", model.hierarchy_names[0]))
    if hierarchy_name not in model.hierarchy_names:
        raise ValueError(
            f"requested hierarchy {hierarchy_name!r} is absent from {model.hierarchy_names}"
        )
    with profile_span(runtime_profiler, "matching_loss_construction"):
        target_tensors = hierarchy_targets_to_tensors(targets, device=tokens.device)
    shared_forward = batch.get("shared_reconstructor_forward")
    shared_deployment = batch.get("shared_deployment_output")
    if shared_forward is not None:
        if not isinstance(shared_forward, Mapping):
            raise TypeError("shared_reconstructor_forward must be a mapping")
        if not torch.equal(shared_forward["tokens"], tokens) or not torch.equal(
            shared_forward["mask"], mask
        ):
            raise ValueError("shared reconstruction forward belongs to another HLT batch")
        particle_embeddings = shared_forward["particle_embeddings"]
        jet_embedding = shared_forward["jet_embedding"]
        prediction = shared_forward["root_prediction"]
        compiled = shared_forward["compiled_root"]
        support = shared_forward["support"]
        axis_eta = shared_forward["axis_eta"]
        axis_phi = shared_forward["axis_phi"]
    else:
        shared_forward = model.prepare_shared_reconstruction_forward(
            tokens, mask, runtime_profiler=runtime_profiler
        )
        particle_embeddings = shared_forward["particle_embeddings"]
        jet_embedding = shared_forward["jet_embedding"]
        prediction = shared_forward["root_prediction"]
        compiled = shared_forward["compiled_root"]
        support = shared_forward["support"]
        axis_eta = shared_forward["axis_eta"]
        axis_phi = shared_forward["axis_phi"]
    if shared_deployment is not None:
        if not isinstance(shared_deployment, AdaptiveBinaryReconstructionOutput):
            raise TypeError(
                "shared_deployment_output must be an AdaptiveBinaryReconstructionOutput"
            )
        if (
            shared_deployment.root_prediction is not prediction
            or shared_deployment.compiled_root is not compiled
            or shared_deployment.hlt_particle_embeddings is not particle_embeddings
        ):
            raise ValueError(
                "shared deployment output does not belong to the supplied shared forward"
            )
    root_state = AccountingState.from_ledger(
        target_tensors.root_ledger if model.oracle_root else compiled.root_ledger
    )
    with profile_span(runtime_profiler, "matching_loss_construction"):
        root_targets = build_root_residual_targets(
            tokens, mask, target_tensors.root_ledger
        )
        root_loss = compute_root_losses(prediction, compiled, root_targets)
    hierarchy_supervision = []
    supervised_outputs = []
    rollout_alignment = None
    particle_loss = None
    particle_auxiliary = None
    distribution_loss = None
    rollout_required = False
    hypothesis_zero_reused = False
    tensors_to_check = [prediction.p4_residual_mean, compiled.root_ledger]

    if context.curriculum.phase >= 2:
        depth_count = sum(
            int(capacity) <= int(context.curriculum.active_capacity)
            for capacity in ABPH_LEVEL_CAPACITIES
        )
        decoder = model.hierarchy_reconstructor.decoders[hierarchy_name]
        with profile_span(runtime_profiler, "matching_loss_construction"):
            teachers = build_teacher_parent_frontiers(
                target_tensors, d_model=decoder.config.d_model
            )
        with profile_span(runtime_profiler, "teacher_forced_hierarchy_decode"):
            supervised = decoder(
                root_state,
                prediction.shared_context,
                prediction.query_tokens,
                particle_embeddings,
                mask,
                support,
                mode="teacher_forced",
                teacher_parent_frontiers=teachers,
            )
        if model.direct_group_set:
            with profile_span(runtime_profiler, "matching_loss_construction"):
                direct_group, direct_topology, direct_metrics = (
                    _one_shot_eight_group_losses(
                        model,
                        prediction,
                        compiled,
                        particle_embeddings,
                        mask,
                        target_tensors,
                    )
                )
            terms = assemble_reconstruction_loss_terms(root_loss=root_loss)
            terms.update(
                {
                    "group_8": direct_group,
                    "topology": direct_topology,
                    "frontier": direct_group,
                }
            )
            return ReconstructorStepResult(
                loss_terms=terms,
                metrics={
                    "root": compute_root_metrics(
                        prediction, compiled, root_targets, losses=root_loss
                    ),
                    "mode": "direct_set",
                    "active_capacity": 8,
                    "compiler_ok": compiled.diagnostics["ok"],
                    "direct_set": direct_metrics,
                },
                batch_size=int(tokens.shape[0]),
                tensors_to_check=(
                    prediction.p4_residual_mean,
                    compiled.root_ledger,
                    direct_group,
                ),
            )
        with profile_span(runtime_profiler, "matching_loss_construction"):
            for depth, level in enumerate(supervised.levels[:depth_count]):
                supervised_outputs.append(level)
                hierarchy_supervision.append(
                    compute_teacher_forced_level_supervision(
                        level,
                        target_tensors.level_ledgers[depth],
                        target_tensors.level_supports[depth],
                        target_tensors.level_masks[depth],
                        target_tensors.level_parent_indices[depth],
                        teachers[depth].topology,
                    )
                )
        required_losses = active_reconstruction_loss_names(context)
        rollout_required = bool(
            context.mode == "rollout"
            or context.validation
            or context.curriculum.phase >= 3
            or model.oracle_parent_rollout
            or model.oracle_groups
            or model.oracle_offline_particles
        )
        if not rollout_required and "frontier" in required_losses:
            raise RuntimeError(
                "the required-loss contract requested frontier supervision after "
                "the rollout forward was declared inactive"
            )
        latent = None
        rollout = None
        active_rollout = None
        hypothesis_zero_bundle = None
        if rollout_required:
            if shared_deployment is not None:
                rollout = shared_deployment.hierarchy_output.hypotheses[
                    0
                ].hierarchy_outputs[hierarchy_name]
            elif (
                context.curriculum.phase >= 4
                and not model.oracle_root
                and not model.oracle_parent_rollout
                and not model.oracle_groups
                and not model.oracle_offline_particles
            ):
                hypothesis_zero_bundle = model.rollout_and_render_hypothesis_zero(
                    root_state=root_state,
                    root_prediction=prediction,
                    particle_embeddings=particle_embeddings,
                    hlt_mask=mask,
                    support=support,
                    axis_eta=axis_eta,
                    axis_phi=axis_phi,
                    evaluation_seed=stochastic_seed,
                    runtime_profiler=runtime_profiler,
                )
                rollout = hypothesis_zero_bundle.hypotheses[0].hierarchy_outputs[
                    hierarchy_name
                ]
                latent = hypothesis_zero_bundle.hypotheses[0].latent
            else:
                latent = torch.zeros(
                    tokens.shape[0],
                    decoder.config.latent_dim,
                    device=tokens.device,
                    dtype=particle_embeddings.dtype,
                )
                with profile_span(runtime_profiler, "rollout_hierarchy_decode"):
                    rollout = decoder(
                        root_state,
                        prediction.shared_context,
                        prediction.query_tokens,
                        particle_embeddings,
                        mask,
                        support,
                        mode="rollout",
                        hypothesis_latent=latent,
                    )
            active_rollout = _truncate_output(
                supervised if model.oracle_parent_rollout else rollout,
                depth_count,
            )
            active_targets = _truncate_targets(target_tensors, depth_count)
            if model.oracle_groups and depth_count == len(ABPH_LEVEL_CAPACITIES):
                final = active_rollout.final_frontier
                oracle_final = replace(
                    final,
                    ledger=target_tensors.level_ledgers[-1],
                    support=target_tensors.level_supports[-1],
                    mask=target_tensors.level_masks[-1],
                    topology=target_tensors.level_topology[-1],
                    parent_indices=target_tensors.level_parent_indices[-1],
                )
                active_rollout = replace(
                    active_rollout,
                    mode="rollout",
                    final_frontier=oracle_final,
                    diagnostics={
                        **dict(active_rollout.diagnostics),
                        "oracle_groups_supplied": True,
                    },
                )
            with profile_span(runtime_profiler, "matching_loss_construction"):
                rollout_alignment = align_recursive_hierarchy(
                    active_rollout, active_targets
                )
            tensors_to_check.append(active_rollout.final_frontier.ledger)

        if context.curriculum.phase >= 3:
            if rollout is None or active_rollout is None or rollout_alignment is None:
                raise RuntimeError("renderer supervision requires an active rollout")
            if depth_count != len(ABPH_LEVEL_CAPACITIES):
                raise RuntimeError("renderer phase requires the complete depth-32 hierarchy")
            render_hierarchy = active_rollout if model.oracle_groups else rollout
            if model.global_particle_set:
                render_hierarchy = _root_only_renderer_hierarchy(render_hierarchy)
            if shared_deployment is not None:
                rendered = shared_deployment.rendered_views[hierarchy_name][0]
            elif hypothesis_zero_bundle is not None:
                rendered = hypothesis_zero_bundle.rendered_views[hierarchy_name][0]
            else:
                if latent is None:
                    raise RuntimeError("renderer supervision lacks its hypothesis latent")
                with profile_span(runtime_profiler, "particle_render_projection"):
                    rendered = model.renderers[hierarchy_name](
                        render_hierarchy,
                        prediction.query_tokens,
                        particle_embeddings,
                        mask,
                        support,
                        latent,
                        axis_eta,
                        axis_phi,
                        hypothesis_index=0,
                    )
            with profile_span(runtime_profiler, "matching_loss_construction"):
                if model.renderer.config.local_matching:
                    particle_loss = compute_local_particle_matching_loss(
                        rendered,
                        torch.as_tensor(
                            targets.particle_targets, device=tokens.device
                        ).float(),
                        torch.as_tensor(
                            targets.particle_mask, device=tokens.device
                        ).bool(),
                        rollout_alignment.levels[-1].renderer_target_map,
                    )
                else:
                    particle_loss = compute_global_particle_matching_loss(
                        rendered,
                        torch.as_tensor(
                            targets.particle_targets, device=tokens.device
                        ).float(),
                        torch.as_tensor(
                            targets.particle_mask, device=tokens.device
                        ).bool(),
                    )
                particle_auxiliary = compute_particle_auxiliary_losses(
                    rendered,
                    torch.as_tensor(targets.particle_targets, device=tokens.device).float(),
                    torch.as_tensor(targets.particle_mask, device=tokens.device).bool(),
                )
            tensors_to_check.append(rendered.four_vector)

            if context.curriculum.phase >= 4:
                if shared_deployment is not None:
                    deployment_hierarchy = shared_deployment.hierarchy_output
                    deployment_views = shared_deployment.rendered_views
                elif hypothesis_zero_bundle is not None:
                    additional = model.rollout_and_render_additional_hypotheses(
                        root_state=root_state,
                        root_prediction=prediction,
                        particle_embeddings=particle_embeddings,
                        hlt_mask=mask,
                        support=support,
                        axis_eta=axis_eta,
                        axis_phi=axis_phi,
                        latent_set=hypothesis_zero_bundle.latent_set,
                        start_index=1,
                        runtime_profiler=runtime_profiler,
                    )
                    deployment = model.assemble_deployment_output(
                        root_prediction=prediction,
                        compiled_root=compiled,
                        root_state=root_state,
                        particle_embeddings=particle_embeddings,
                        jet_embedding=jet_embedding,
                        support=support,
                        hypothesis_zero=hypothesis_zero_bundle,
                        additional_hypotheses=additional,
                    )
                    deployment_hierarchy = deployment.hierarchy_output
                    deployment_views = deployment.rendered_views
                else:
                    deployment_hierarchy, deployment_views = model.rollout_and_render(
                        root_state=root_state,
                        root_prediction=prediction,
                        particle_embeddings=particle_embeddings,
                        hlt_mask=mask,
                        support=support,
                        axis_eta=axis_eta,
                        axis_phi=axis_phi,
                        evaluation_seed=stochastic_seed,
                        runtime_profiler=runtime_profiler,
                    )
                rendered_hypotheses = deployment_views[hierarchy_name]
                hypothesis_zero_reused = rendered_hypotheses[0] is rendered
                if not hypothesis_zero_reused:
                    raise RuntimeError(
                        "phase-4 particle and distribution paths did not share the "
                        "exact rendered hypothesis-zero object"
                    )
                latent_context = model.hierarchy_reconstructor.encode_deployment_context(
                    root_state,
                    prediction.shared_context,
                    prediction.query_tokens,
                    particle_embeddings,
                    mask,
                )
                posterior = model.hierarchy_reconstructor.latent_model.training_posterior_sample(
                    latent_context, target_tensors, seed=stochastic_seed
                )
                observables = _distribution_observables(rendered_hypotheses[1:])
                with profile_span(runtime_profiler, "matching_loss_construction"):
                    distribution_loss = compute_distribution_losses(
                        posterior,
                        observables,
                        _target_distribution_observables(targets, tokens.device),
                        split_negative_log_likelihood=(
                            rollout_alignment.total_frontier_loss
                        ),
                        particle_negative_log_likelihood=particle_loss.total,
                    )
                tensors_to_check.append(
                    deployment_hierarchy.hypotheses[0]
                    .hierarchy_outputs[hierarchy_name]
                    .final_frontier.ledger
                )

    terms = assemble_reconstruction_loss_terms(
        root_loss=root_loss,
        hierarchy_supervision=tuple(hierarchy_supervision),
        rollout_alignment=(rollout_alignment if context.mode == "rollout" else None),
        particle_matching=particle_loss,
        particle_auxiliary=particle_auxiliary,
        distribution_loss=distribution_loss,
    )
    if not model.constrained_hierarchy and supervised_outputs:
        raw_groups, raw_topology = _unconstrained_level_losses(
            model, supervised_outputs, target_tensors
        )
        for capacity, loss in zip(ABPH_LEVEL_CAPACITIES, raw_groups):
            terms[f"group_{capacity}"] = loss
        terms["topology"] = torch.stack(raw_topology).mean()
        if context.mode == "rollout":
            terms["frontier"] = torch.stack(raw_groups).mean()
    metrics = {
        "root": compute_root_metrics(prediction, compiled, root_targets, losses=root_loss),
        "mode": context.mode,
        "active_capacity": context.curriculum.active_capacity,
        "compiler_ok": compiled.diagnostics["ok"],
        "oracle_root_supplied": model.oracle_root,
        "oracle_parent_rollout": model.oracle_parent_rollout,
        "oracle_groups_supplied": model.oracle_groups,
        "hierarchy_constraints_in_loss": model.constrained_hierarchy,
        "unconstrained_raw_child_heads": not model.constrained_hierarchy,
        "hierarchy_name": hierarchy_name,
        "rollout_forward_executed": bool(rollout_required),
        "hypothesis_zero_reused": bool(hypothesis_zero_reused),
    }
    return ReconstructorStepResult(
        loss_terms=terms,
        metrics=metrics,
        batch_size=int(tokens.shape[0]),
        tensors_to_check=tuple(tensors_to_check),
    )


def reconstructor_runtime_provenance(
    *,
    variant_name: str,
    target_metadata: Mapping[str, Any],
    hlt_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = resolve_variant_config(variant_name)
    return {
        "contract": ABPH_PRODUCTION_RUNTIME_CONTRACT,
        "variant_name": variant_name,
        "resolved_variant_config_hash": resolved["resolved_config_hash"],
        "source_manifest_hash": target_metadata.get("source_manifest_hash"),
        "hlt_content_hash": hlt_metadata.get("hlt_content_hash"),
        "offline_cache_content_hash": target_metadata.get("offline_content_hash"),
        "hierarchy_target_content_hash": target_metadata.get("target_content_hash"),
        "target_mode": target_metadata.get(
            "target_mode", ABPH_SHARED_TRANSIENT_TARGET_MODE
        ),
        "target_mode_selection_hash": target_metadata.get(
            "target_mode_selection_hash"
        ),
        "hierarchy_target_schema_hash": canonical_hash(
            {
                "root": target_metadata.get("root_feature_names"),
                "group": target_metadata.get("group_feature_names"),
                "particle": target_metadata.get("particle_target_names"),
            }
        ),
        "grouping_algorithm_hash": canonical_hash(target_metadata.get("layout", {})),
        "root_ledger_schema_hash": canonical_hash(
            {"root_feature_names": target_metadata.get("root_feature_names")}
        ),
        "normalization_hash": "identity_physical_units_v1",
        "source_git_commit": "recorded_by_slurm_run_config",
        "source_status_hash": "recorded_by_slurm_run_config",
        "final_test_loaded": False,
        "teacher_logits_loaded": False,
    }


def load_selected_reconstructor(
    checkpoint_path: str | Path,
    *,
    variant_name: str,
    device: str | Any = "cpu",
    smoke: bool = False,
) -> AdaptiveBinaryReconstructorModel:
    """Rebuild and strictly load one model-val-selected reconstructor."""

    torch = require_torch()
    resolved = resolve_variant_config(variant_name)
    grouping = str(resolved["model"]["hierarchy"].get("grouping", "exclusive_kt"))
    model = AdaptiveBinaryReconstructorModel(
        hierarchy_names=(grouping,), variant_name=variant_name, smoke=bool(smoke)
    )
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    state = selected_model_state(payload)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def build_shared_root_dual_reconstructor(
    kt_checkpoint_path: str | Path,
    ca_checkpoint_path: str | Path,
    *,
    device: str | Any = "cpu",
    smoke: bool = False,
) -> AdaptiveBinaryReconstructorModel:
    """Compose kT/C-A branches below one exact selected kT root compiler.

    The kT checkpoint owns the shared HLT encoder, root predictor, and latent
    distribution.  Each hierarchy keeps its independently selected recursive
    decoder and renderer.  No root is recomputed for the C/A branch.
    """

    torch = require_torch()

    def state(path: str | Path) -> Mapping[str, Any]:
        payload = load_torch_checkpoint(path, device=device)
        return selected_model_state(payload)

    kt_state = state(kt_checkpoint_path)
    ca_state = state(ca_checkpoint_path)
    model = AdaptiveBinaryReconstructorModel(
        hierarchy_names=("exclusive_kt", "cambridge_aachen"),
        variant_name="E7_dual_hierarchy_dualcross",
        smoke=bool(smoke),
    )
    merged: dict[str, Any] = {}
    for name, reference in model.state_dict().items():
        source = ca_state if (
            name.startswith("hierarchy_reconstructor.decoders.cambridge_aachen.")
            or name.startswith("renderers.cambridge_aachen.")
        ) else kt_state
        if name not in source:
            raise KeyError(f"dual-reconstructor source lacks parameter {name}")
        value = source[name]
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"dual-reconstructor parameter shape mismatch for {name}")
        merged[name] = value
    model.load_state_dict(merged, strict=True)
    model.to(device)
    return model


__all__ = [
    "ABPH_PRODUCTION_RUNTIME_CONTRACT",
    "AdaptiveBinaryReconstructionOutput",
    "AdaptiveBinaryReconstructorModel",
    "AdaptiveBinaryTargetBatchSource",
    "HypothesisRenderBundle",
    "reconstructor_runtime_provenance",
    "reconstructor_step",
    "package_trainable_pseudo_views",
    "load_selected_reconstructor",
    "build_shared_root_dual_reconstructor",
]
