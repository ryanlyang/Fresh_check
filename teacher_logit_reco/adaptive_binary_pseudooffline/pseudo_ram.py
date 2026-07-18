"""Rank-local RAM sources for frozen ABPH reconstructors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view

from .config import canonical_hash
from .conditional_latent import ABPH_FIXED_EVALUATION_SEED
from .prediction_cache import DeployablePseudoViewBatch, package_deployable_pseudo_views
from .pseudo_consumer import ABPH_CONSUMER_PSEUDO_CONTRACT
from .ram_workspace import RankLocalWorkspace


ABPH_FROZEN_PSEUDO_RAM_CONTRACT = "adaptive_binary_frozen_pseudo_ram_v1"
ABPH_FULL_RANK_PSEUDO_MODE = "full_rank_cache"
ABPH_BOUNDED_LRU_PSEUDO_MODE = "bounded_lru"
ABPH_PSEUDO_EXECUTION_MODES: tuple[str, ...] = (
    "auto",
    ABPH_FULL_RANK_PSEUDO_MODE,
    ABPH_BOUNDED_LRU_PSEUDO_MODE,
)


def _array_bytes(arrays: Mapping[str, Any]) -> int:
    return sum(int(np.asarray(value).nbytes) for value in arrays.values())


def _rank_interval(size: int, rank: int, world_size: int) -> tuple[int, int]:
    total = int(size)
    rank_value = int(rank)
    world = int(world_size)
    if total <= 0 or world <= 0 or not 0 <= rank_value < world:
        raise ValueError("pseudo RAM rank interval inputs are invalid")
    base, remainder = divmod(total, world)
    start = rank_value * base + min(rank_value, remainder)
    stop = start + base + int(rank_value < remainder)
    if stop <= start:
        raise ValueError("pseudo RAM rank owns no jets")
    return start, stop


def _identity_range_hash(identities: Sequence[Any]) -> str:
    return jet_identity_hash(tuple(identities))


class FrozenPseudoGenerator(Protocol):
    source_name: str
    source_hash: str
    checkpoint_hashes: Mapping[str, str]
    evaluation_seed: int

    def generate(self, hlt_tokens: np.ndarray, hlt_mask: np.ndarray) -> DeployablePseudoViewBatch:
        ...


class SelectedReconstructorPseudoGenerator:
    """Teacher-free deployable packaging around one selected reconstructor."""

    def __init__(
        self,
        model: Any,
        *,
        source_name: str,
        checkpoint_hashes: Mapping[str, str],
        device: Any,
        evaluation_seed: int = ABPH_FIXED_EVALUATION_SEED,
    ) -> None:
        import torch

        if not source_name or not checkpoint_hashes:
            raise ValueError("frozen pseudo generator requires source/checkpoint identity")
        self.model = model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.source_name = str(source_name)
        self.checkpoint_hashes = {
            str(name): str(value) for name, value in checkpoint_hashes.items()
        }
        if any(not value for value in self.checkpoint_hashes.values()):
            raise ValueError("frozen pseudo checkpoint hashes cannot be empty")
        self.device = torch.device(device)
        self.evaluation_seed = int(evaluation_seed)
        self.source_hash = canonical_hash(
            {
                "contract": ABPH_FROZEN_PSEUDO_RAM_CONTRACT,
                "source_name": self.source_name,
                "checkpoint_hashes": self.checkpoint_hashes,
                "evaluation_seed": self.evaluation_seed,
                "consumer_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
            }
        )

    def generate(
        self, hlt_tokens: np.ndarray, hlt_mask: np.ndarray
    ) -> DeployablePseudoViewBatch:
        import torch

        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("frozen pseudo generator model is not frozen in eval mode")
        with torch.no_grad():
            tokens = torch.as_tensor(hlt_tokens, device=self.device)
            mask = torch.as_tensor(hlt_mask, device=self.device).bool()
            output = self.model.deploy(
                tokens, mask, evaluation_seed=self.evaluation_seed
            )
            batch = package_deployable_pseudo_views(
                output.hierarchy_output,
                output.rendered_views,
                consumer_only=True,
            )
        batch.validate()
        return batch


@dataclass(frozen=True)
class RamFrozenPseudoBatch:
    hlt_tokens: Any
    hlt_mask: Any
    labels: Any
    indices: Any
    pseudo: DeployablePseudoViewBatch
    independent_roots: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class _RamPseudoShard:
    start: int
    stop: int
    pseudo: DeployablePseudoViewBatch
    independent_roots: Mapping[str, np.ndarray] | None
    binding_hash: str
    identity_range_hash: str
    resident_bytes: int


def _concatenate_batches(
    batches: Sequence[DeployablePseudoViewBatch],
) -> DeployablePseudoViewBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty pseudo batch sequence")
    first = batches[0]
    for batch in batches:
        batch.validate()
        if (
            batch.view_names != first.view_names
            or batch.hierarchy_names != first.hierarchy_names
            or dict(batch.frontier_depths) != dict(first.frontier_depths)
            or set(batch.arrays) != set(first.arrays)
        ):
            raise ValueError("pseudo generation schema changed within one RAM shard")
    diagnostics = dict(first.diagnostics)
    diagnostics.pop("consumer_pseudo_schema_hash", None)
    diagnostics["consumer_only_pseudo"] = False
    result = DeployablePseudoViewBatch(
        arrays={
            name: np.concatenate(
                [np.asarray(batch.arrays[name]) for batch in batches], axis=0
            )
            for name in first.arrays
        },
        view_names=first.view_names,
        hierarchy_names=first.hierarchy_names,
        frontier_depths=dict(first.frontier_depths),
        diagnostics=diagnostics,
    ).to_consumer_only()
    result.validate()
    return result


def _merge_independent_batches(
    first: DeployablePseudoViewBatch,
    second: DeployablePseudoViewBatch,
) -> tuple[DeployablePseudoViewBatch, Mapping[str, np.ndarray]]:
    first = first.to_consumer_only()
    second = second.to_consumer_only()
    if first.batch_size != second.batch_size:
        raise ValueError("independent pseudo branches differ in batch size")
    kt_name = first.hierarchy_names[0]
    ca_name = second.hierarchy_names[0]
    if (kt_name, ca_name) != ("exclusive_kt", "cambridge_aachen"):
        raise ValueError("independent pseudo branches have unexpected hierarchy order")
    arrays = dict(first.arrays)
    arrays.update(
        {
            name: value
            for name, value in second.arrays.items()
            if name.startswith("particle__") or name.startswith("frontier__")
        }
    )
    independent = {
        kt_name: np.asarray(first.arrays["shared_root_ledger"]),
        ca_name: np.asarray(second.arrays["shared_root_ledger"]),
    }
    for field in ("ledger", "uncertainty", "mask"):
        arrays[f"frontier__{ca_name}__depth_00__{field}"] = np.asarray(
            first.arrays[f"frontier__{kt_name}__depth_00__{field}"]
        )
    diagnostics = dict(first.diagnostics)
    diagnostics.pop("consumer_pseudo_schema_hash", None)
    diagnostics["consumer_only_pseudo"] = False
    result = DeployablePseudoViewBatch(
        arrays=arrays,
        view_names=first.view_names,
        hierarchy_names=(kt_name, ca_name),
        frontier_depths={
            kt_name: int(first.frontier_depths[kt_name]),
            ca_name: int(second.frontier_depths[ca_name]),
        },
        diagnostics=diagnostics,
    ).to_consumer_only()
    result.validate()
    return result, independent


class FrozenReconstructorRamSource:
    """Generate and retain consumer-only pseudo shards in bounded process RAM."""

    def __init__(
        self,
        *,
        hlt_cache_dir: str | Path,
        generators: Sequence[FrozenPseudoGenerator],
        split: str,
        batch_size: int,
        shard_size: int,
        generation_batch_size: int,
        execution_mode: str = "auto",
        lru_capacity_bytes: int | None = None,
        workspace: RankLocalWorkspace | None = None,
        rank: int = 0,
        world_size: int = 1,
        independent_roots: bool = False,
        maximum_batches: int | None = None,
    ) -> None:
        mode = str(execution_mode)
        if mode not in ABPH_PSEUDO_EXECUTION_MODES:
            raise ValueError(f"unknown frozen pseudo execution mode {mode!r}")
        if not generators or len(generators) > 2:
            raise ValueError("frozen pseudo RAM source requires one or two generators")
        if len(generators) == 2 and not independent_roots:
            raise ValueError("two frozen generators are restricted to independent-root diagnostics")
        self.hlt = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
        self.generators = tuple(generators)
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.shard_size = int(shard_size)
        self.generation_batch_size = int(generation_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.independent_roots = bool(independent_roots)
        self.maximum_batches = maximum_batches
        self.workspace = workspace
        if min(self.batch_size, self.shard_size, self.generation_batch_size) <= 0:
            raise ValueError("frozen pseudo RAM batch and shard sizes must be positive")
        self.rank_start, self.rank_stop = _rank_interval(
            len(self.hlt.labels), self.rank, self.world_size
        )
        self._descriptors = tuple(
            (start, min(start + self.shard_size, self.rank_stop))
            for start in range(self.rank_start, self.rank_stop, self.shard_size)
        )
        self._cache: OrderedDict[tuple[int, int], _RamPseudoShard] = OrderedDict()
        self._seen: set[tuple[int, int]] = set()
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._reservation_id: str | None = None
        self._metadata: tuple[dict[str, Any], ...] = ()
        self._telemetry = {
            "cache_hits": 0,
            "cache_misses": 0,
            "generated_shards": 0,
            "regenerated_shards": 0,
            "evictions": 0,
            "root_identity_checks": 0,
            "pseudo_representations_written_persistently": False,
            "generator_gpu_models_released": False,
        }
        self.hlt_content_hash = str(self.hlt.metadata.get("hlt_content_hash") or "")
        if not self.hlt_content_hash:
            raise ValueError("HLT source lacks its content hash")
        self.rank_identity_hash = _identity_range_hash(
            self.hlt.jet_ids[self.rank_start : self.rank_stop]
        )
        self.source_hash = canonical_hash(
            {
                "contract": ABPH_FROZEN_PSEUDO_RAM_CONTRACT,
                "split": self.split,
                "hlt_content_hash": self.hlt_content_hash,
                "rank": self.rank,
                "world_size": self.world_size,
                "rank_start": self.rank_start,
                "rank_stop": self.rank_stop,
                "rank_identity_hash": self.rank_identity_hash,
                "generators": [generator.source_hash for generator in self.generators],
                "evaluation_seeds": [
                    int(generator.evaluation_seed) for generator in self.generators
                ],
                "consumer_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
            }
        )

        first_key = self._descriptors[0]
        first = self._generate_shard(*first_key)
        bytes_per_jet = first.resident_bytes / max(1, first.stop - first.start)
        projected_full_bytes = int(np.ceil(bytes_per_jet * (self.rank_stop - self.rank_start)))
        available = (
            None
            if workspace is None
            else max(0, workspace.reservation_limit_bytes - workspace.reserved_bytes)
        )
        if mode == "auto":
            mode = (
                ABPH_FULL_RANK_PSEUDO_MODE
                if available is None or projected_full_bytes <= available
                else ABPH_BOUNDED_LRU_PSEUDO_MODE
            )
        self.execution_mode = mode
        if mode == ABPH_FULL_RANK_PSEUDO_MODE:
            capacity = max(1, projected_full_bytes)
        else:
            default_lru = max(first.resident_bytes, min(projected_full_bytes, 4 * first.resident_bytes))
            capacity = int(lru_capacity_bytes or default_lru)
            if capacity < first.resident_bytes:
                raise MemoryError("bounded pseudo LRU cannot retain one production shard")
        if available is not None and capacity > available:
            if mode == ABPH_FULL_RANK_PSEUDO_MODE and execution_mode == "auto":
                self.execution_mode = ABPH_BOUNDED_LRU_PSEUDO_MODE
                capacity = int(lru_capacity_bytes or max(first.resident_bytes, min(available, 4 * first.resident_bytes)))
            if capacity < first.resident_bytes or capacity > available:
                raise MemoryError("pseudo RAM source cannot preserve mandatory 20% headroom")
        self.capacity_bytes = capacity
        if workspace is not None:
            self._reservation_id = workspace.reserve(
                owner=f"frozen_pseudo:{self.split}:{self.rank}:{self.source_hash}",
                role=f"pseudo_{self.execution_mode}",
                expected_bytes=max(1, self.capacity_bytes),
            )
        self._insert(first_key, first)
        if self.execution_mode == ABPH_FULL_RANK_PSEUDO_MODE:
            for descriptor in self._descriptors[1:]:
                self._insert(descriptor, self._generate_shard(*descriptor))
            if self._reservation_id is not None:
                workspace.commit(
                    self._reservation_id, measured_bytes=self._resident_bytes
                )
        elif self._reservation_id is not None:
            workspace.commit(self._reservation_id, measured_bytes=self.capacity_bytes)

    @property
    def cache_dirs(self) -> tuple[Path, ...]:
        return tuple(Path(generator.source_name) for generator in self.generators)

    @property
    def metadata(self) -> tuple[Mapping[str, Any], ...]:
        if not self._metadata:
            raise RuntimeError("pseudo RAM source generated no metadata")
        return self._metadata

    def _generate_one(
        self, generator: FrozenPseudoGenerator, start: int, stop: int
    ) -> DeployablePseudoViewBatch:
        parts = []
        for offset in range(start, stop, self.generation_batch_size):
            end = min(offset + self.generation_batch_size, stop)
            parts.append(
                generator.generate(
                    np.asarray(self.hlt.tokens[offset:end], dtype=np.float32),
                    np.asarray(self.hlt.mask[offset:end], dtype=bool),
                ).to_consumer_only()
            )
        return _concatenate_batches(parts)

    def _generate_shard(self, start: int, stop: int) -> _RamPseudoShard:
        key = (int(start), int(stop))
        batches = [self._generate_one(generator, start, stop) for generator in self.generators]
        independent = None
        if len(batches) == 2:
            pseudo, independent = _merge_independent_batches(*batches)
        else:
            pseudo = batches[0]
        report = pseudo.validate()
        identity_hash = _identity_range_hash(self.hlt.jet_ids[start:stop])
        binding = canonical_hash(
            {
                "source_hash": self.source_hash,
                "start": start,
                "stop": stop,
                "identity_range_hash": identity_hash,
                "consumer_pseudo_schema_hash": report["consumer_pseudo_schema_hash"],
            }
        )
        resident = _array_bytes(pseudo.arrays)
        if independent is not None:
            resident += _array_bytes(independent)
        self._telemetry["cache_misses"] += 1
        self._telemetry["generated_shards"] += 1
        if key in self._seen:
            self._telemetry["regenerated_shards"] += 1
        self._seen.add(key)
        self._telemetry["root_identity_checks"] += 1
        metadata = tuple(
            {
                "source_name": generator.source_name,
                "source_hash": generator.source_hash,
                "checkpoint_hashes": dict(generator.checkpoint_hashes),
                "evaluation_seed": int(generator.evaluation_seed),
                "split": self.split,
                "n_jets": self.rank_stop - self.rank_start,
                "hlt_content_hash": self.hlt_content_hash,
                "jet_identity_hash": self.rank_identity_hash,
                "prediction_schema": report["schema"],
                "hierarchy_names": list(pseudo.hierarchy_names),
                "frontier_depths": dict(pseudo.frontier_depths),
                "consumer_pseudo_schema_hash": report["consumer_pseudo_schema_hash"],
                "consumer_only_pseudo": True,
                "consumer_pseudo_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
                "pseudo_representations_written_persistently": False,
            }
            for generator in self.generators
        )
        if not self._metadata:
            self._metadata = metadata
        elif any(
            row["consumer_pseudo_schema_hash"] != prior["consumer_pseudo_schema_hash"]
            for row, prior in zip(metadata, self._metadata)
        ):
            raise ValueError("pseudo RAM consumer schema changed during regeneration")
        return _RamPseudoShard(
            start=start,
            stop=stop,
            pseudo=pseudo,
            independent_roots=independent,
            binding_hash=binding,
            identity_range_hash=identity_hash,
            resident_bytes=resident,
        )

    def _insert(self, key: tuple[int, int], shard: _RamPseudoShard) -> None:
        if shard.resident_bytes > self.capacity_bytes:
            raise MemoryError("one pseudo RAM shard exceeds the fixed cache capacity")
        while self._cache and self._resident_bytes + shard.resident_bytes > self.capacity_bytes:
            _, removed = self._cache.popitem(last=False)
            self._resident_bytes -= removed.resident_bytes
            self._telemetry["evictions"] += 1
        self._cache[key] = shard
        self._resident_bytes += shard.resident_bytes
        self._peak_resident_bytes = max(self._peak_resident_bytes, self._resident_bytes)

    def _get(self, start: int, stop: int) -> _RamPseudoShard:
        key = (int(start), int(stop))
        if key in self._cache:
            shard = self._cache.pop(key)
            self._cache[key] = shard
            self._telemetry["cache_hits"] += 1
        else:
            shard = self._generate_shard(*key)
            self._insert(key, shard)
        expected_identity = _identity_range_hash(self.hlt.jet_ids[start:stop])
        expected_binding = canonical_hash(
            {
                "source_hash": self.source_hash,
                "start": start,
                "stop": stop,
                "identity_range_hash": expected_identity,
                "consumer_pseudo_schema_hash": shard.pseudo.validate()[
                    "consumer_pseudo_schema_hash"
                ],
            }
        )
        if shard.identity_range_hash != expected_identity or shard.binding_hash != expected_binding:
            raise ValueError("pseudo RAM cache entry provenance mismatch")
        return shard

    def iter_batches(self, *, shuffle: bool, seed: int) -> Iterator[RamFrozenPseudoBatch]:
        yielded = 0
        for start, stop in self._descriptors:
            shard = self._get(start, stop)
            local_size = stop - start
            order = (
                np.random.default_rng(int(seed) + start).permutation(local_size)
                if shuffle
                else np.arange(local_size, dtype=np.int64)
            )
            for offset in range(0, local_size, self.batch_size):
                chosen = order[offset : offset + self.batch_size]
                global_indices = chosen + start
                pseudo = DeployablePseudoViewBatch(
                    arrays={
                        name: np.asarray(value)[chosen]
                        for name, value in shard.pseudo.arrays.items()
                    },
                    view_names=shard.pseudo.view_names,
                    hierarchy_names=shard.pseudo.hierarchy_names,
                    frontier_depths=dict(shard.pseudo.frontier_depths),
                    diagnostics=dict(shard.pseudo.diagnostics),
                )
                pseudo.validate()
                roots = None
                if shard.independent_roots is not None:
                    roots = {
                        name: np.asarray(value)[chosen]
                        for name, value in shard.independent_roots.items()
                    }
                yield RamFrozenPseudoBatch(
                    hlt_tokens=np.asarray(self.hlt.tokens[global_indices], dtype=np.float32),
                    hlt_mask=np.asarray(self.hlt.mask[global_indices], dtype=bool),
                    labels=np.asarray(self.hlt.labels[global_indices], dtype=np.int64),
                    indices=np.asarray(global_indices, dtype=np.int64),
                    pseudo=pseudo,
                    independent_roots=roots,
                )
                yielded += 1
                if self.maximum_batches is not None and yielded >= int(self.maximum_batches):
                    return

    def telemetry(self) -> dict[str, Any]:
        requests = int(self._telemetry["cache_hits"]) + int(self._telemetry["cache_misses"])
        return {
            "contract": ABPH_FROZEN_PSEUDO_RAM_CONTRACT,
            "execution_mode": self.execution_mode,
            "split": self.split,
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_start": self.rank_start,
            "rank_stop": self.rank_stop,
            "rank_identity_hash": self.rank_identity_hash,
            "source_hash": self.source_hash,
            "source_names": [generator.source_name for generator in self.generators],
            "evaluation_seeds": [
                int(generator.evaluation_seed) for generator in self.generators
            ],
            "consumer_pseudo_schema_hashes": [
                row["consumer_pseudo_schema_hash"] for row in self.metadata
            ],
            "capacity_bytes": int(self.capacity_bytes),
            "resident_bytes": int(self._resident_bytes),
            "peak_resident_bytes": int(self._peak_resident_bytes),
            "cache_hit_rate": (
                None if requests == 0 else float(self._telemetry["cache_hits"]) / requests
            ),
            **self._telemetry,
        }

    def release_generator_gpu_models(self) -> None:
        """Free reconstructor GPU memory after a complete full-rank materialization."""

        if self.execution_mode != ABPH_FULL_RANK_PSEUDO_MODE:
            raise RuntimeError(
                "generator GPU release requires a complete full-rank pseudo cache"
            )
        if set(self._cache) != set(self._descriptors):
            raise RuntimeError("full-rank pseudo cache is incomplete")
        import torch

        for generator in self.generators:
            model = getattr(generator, "model", None)
            if model is not None:
                model.to("cpu")
                if hasattr(generator, "device"):
                    generator.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._telemetry["generator_gpu_models_released"] = True

    def close(self) -> None:
        self._cache.clear()
        self._resident_bytes = 0
        if self.workspace is not None and self._reservation_id is not None:
            self.workspace.release(self._reservation_id)
            self._reservation_id = None


__all__ = [
    "ABPH_BOUNDED_LRU_PSEUDO_MODE",
    "ABPH_FROZEN_PSEUDO_RAM_CONTRACT",
    "ABPH_FULL_RANK_PSEUDO_MODE",
    "ABPH_PSEUDO_EXECUTION_MODES",
    "FrozenPseudoGenerator",
    "FrozenReconstructorRamSource",
    "RamFrozenPseudoBatch",
    "SelectedReconstructorPseudoGenerator",
]
