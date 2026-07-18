from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline.config import canonical_hash
from teacher_logit_reco.adaptive_binary_pseudooffline.hypothesis_distribution import (
    ABPH_PRIMARY_HYPOTHESIS_NAMES,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.prediction_cache import (
    DeployablePseudoViewBatch,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.pseudo_ram import (
    ABPH_BOUNDED_LRU_PSEUDO_MODE,
    ABPH_FULL_RANK_PSEUDO_MODE,
    FrozenReconstructorRamSource,
)


def _hlt_view(n_jets: int = 12) -> JetView:
    tokens = np.zeros((n_jets, 128, RAW_TOKEN_DIM), dtype=np.float32)
    tokens[:, 0, 0] = np.arange(n_jets, dtype=np.float32) + 1.0
    mask = np.zeros((n_jets, 128), dtype=bool)
    mask[:, :4] = True
    labels = np.arange(n_jets, dtype=np.int64) % 10
    identities = [
        JetIdentity(file="HToBB_010.root", entry=index, label=int(labels[index]))
        for index in range(n_jets)
    ]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=identities,
        split="model_train",
        metadata={"hlt_content_hash": "hlt-content", "source_manifest_hash": "manifest"},
    )


def _pseudo_batch(
    hlt_tokens: np.ndarray,
    *,
    hierarchy_names: tuple[str, ...],
    root_offset: float,
) -> DeployablePseudoViewBatch:
    batch = len(hlt_tokens)
    views = len(ABPH_PRIMARY_HYPOTHESIS_NAMES)
    particles = 128
    identity = np.asarray(hlt_tokens[:, 0, 0], dtype=np.float32) + root_offset
    root = np.stack((identity, identity + 0.5, identity + 1.0), axis=-1)
    arrays: dict[str, np.ndarray] = {
        "shared_root_ledger": root,
        "hypothesis_latent": np.broadcast_to(identity[:, None, None], (batch, views, 2)).copy(),
        "hypothesis_prior_log_prob": np.zeros((batch, views), dtype=np.float32),
    }
    for hierarchy_index, hierarchy in enumerate(hierarchy_names):
        base = identity[:, None, None] + float(hierarchy_index)
        arrays.update(
            {
                f"particle__{hierarchy}__canonical_features": np.broadcast_to(
                    base[..., None], (batch, views, particles, 3)
                ).copy(),
                f"particle__{hierarchy}__side_channels": np.zeros(
                    (batch, views, particles, 2), dtype=np.float32
                ),
                f"particle__{hierarchy}__four_vector": np.zeros(
                    (batch, views, particles, 4), dtype=np.float32
                ),
                f"particle__{hierarchy}__mass": np.zeros(
                    (batch, views, particles), dtype=np.float32
                ),
                f"particle__{hierarchy}__mask": np.ones(
                    (batch, views, particles), dtype=bool
                ),
                f"particle__{hierarchy}__group_indices": np.zeros(
                    (batch, views, particles), dtype=np.int64
                ),
                f"particle__{hierarchy}__local_slot_indices": np.broadcast_to(
                    np.arange(particles, dtype=np.int64),
                    (batch, views, particles),
                ).copy(),
                f"particle__{hierarchy}__uncertainty": np.full(
                    (batch, views, particles), 0.25, dtype=np.float32
                ),
                f"particle__{hierarchy}__slot_hidden": np.zeros(
                    (batch, views, particles, 5), dtype=np.float32
                ),
            }
        )
        for depth, capacity in enumerate((1, 2)):
            prefix = f"frontier__{hierarchy}__depth_{depth:02d}"
            ledger = np.broadcast_to(
                root[:, None, None, :], (batch, views, capacity, 3)
            ).copy()
            arrays.update(
                {
                    f"{prefix}__ledger": ledger,
                    f"{prefix}__hidden": np.zeros(
                        (batch, views, capacity, 5), dtype=np.float32
                    ),
                    f"{prefix}__support": np.zeros(
                        (batch, views, capacity, 4), dtype=np.float32
                    ),
                    f"{prefix}__uncertainty": np.full(
                        (batch, views, capacity), 0.1, dtype=np.float32
                    ),
                    f"{prefix}__mask": np.ones(
                        (batch, views, capacity), dtype=bool
                    ),
                    f"{prefix}__topology": np.ones(
                        (batch, views, capacity), dtype=np.int64
                    ),
                    f"{prefix}__parent_indices": np.full(
                        (batch, views, capacity), -1 if depth == 0 else 0, dtype=np.int64
                    ),
                    f"{prefix}__source_child_indices": np.full(
                        (batch, views, capacity), -1, dtype=np.int64
                    ),
                }
            )
    result = DeployablePseudoViewBatch(
        arrays=arrays,
        view_names=ABPH_PRIMARY_HYPOTHESIS_NAMES,
        hierarchy_names=hierarchy_names,
        frontier_depths={name: 2 for name in hierarchy_names},
        diagnostics={
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
        },
    ).to_consumer_only()
    result.validate()
    return result


@dataclass
class _Generator:
    source_name: str
    hierarchy_names: tuple[str, ...]
    root_offset: float = 0.0
    evaluation_seed: int = 1729

    def __post_init__(self) -> None:
        self.checkpoint_hashes = {self.source_name: f"checkpoint-{self.source_name}"}
        self.source_hash = canonical_hash(
            {
                "source": self.source_name,
                "checkpoint": self.checkpoint_hashes,
                "seed": self.evaluation_seed,
            }
        )

    def generate(self, hlt_tokens, hlt_mask):
        del hlt_mask
        return _pseudo_batch(
            np.asarray(hlt_tokens),
            hierarchy_names=self.hierarchy_names,
            root_offset=self.root_offset,
        )


def _source(monkeypatch, *, generators, mode, lru_capacity_bytes=None):
    import teacher_logit_reco.adaptive_binary_pseudooffline.pseudo_ram as module

    monkeypatch.setattr(module, "load_cached_hlt_view", lambda *args, **kwargs: _hlt_view())
    return FrozenReconstructorRamSource(
        hlt_cache_dir=Path("unused"),
        generators=generators,
        split="model_train",
        batch_size=3,
        shard_size=4,
        generation_batch_size=2,
        execution_mode=mode,
        lru_capacity_bytes=lru_capacity_bytes,
    )


def test_full_rank_scoring_source_releases_generators_without_regeneration(
    monkeypatch,
) -> None:
    source = _source(
        monkeypatch,
        generators=(_Generator("D1", ("exclusive_kt",)),),
        mode=ABPH_FULL_RANK_PSEUDO_MODE,
    )
    generated = source.telemetry()["generated_shards"]
    source.release_generator_gpu_models()
    first = list(source.iter_batches(shuffle=False, seed=17))
    second = list(source.iter_batches(shuffle=False, seed=17))
    telemetry = source.telemetry()
    assert telemetry["generator_gpu_models_released"] is True
    assert telemetry["generated_shards"] == generated
    assert telemetry["regenerated_shards"] == 0
    assert [batch.indices.tolist() for batch in first] == [
        batch.indices.tolist() for batch in second
    ]


def _epoch(source):
    rows = list(source.iter_batches(shuffle=False, seed=17))
    return {
        "indices": np.concatenate([row.indices for row in rows]),
        "root": np.concatenate([row.pseudo.arrays["shared_root_ledger"] for row in rows]),
        "particle": np.concatenate(
            [
                row.pseudo.arrays[
                    "particle__exclusive_kt__canonical_features"
                ]
                for row in rows
            ]
        ),
    }


def test_full_rank_and_lru_sources_are_exactly_equivalent(monkeypatch) -> None:
    generator = _Generator("D1_kt32_mh4_particles", ("exclusive_kt",))
    full = _source(
        monkeypatch, generators=(generator,), mode=ABPH_FULL_RANK_PSEUDO_MODE
    )
    first_shard_bytes = next(iter(full._cache.values())).resident_bytes
    lru = _source(
        monkeypatch,
        generators=(generator,),
        mode=ABPH_BOUNDED_LRU_PSEUDO_MODE,
        lru_capacity_bytes=first_shard_bytes,
    )
    for _ in range(2):
        full_epoch = _epoch(full)
        lru_epoch = _epoch(lru)
        assert set(full_epoch) == set(lru_epoch)
        for name in full_epoch:
            assert np.array_equal(full_epoch[name], lru_epoch[name])
    assert full.telemetry()["regenerated_shards"] == 0
    assert full.telemetry()["cache_hit_rate"] > 0.0
    assert lru.telemetry()["regenerated_shards"] > 0
    assert lru.telemetry()["evictions"] > 0
    assert lru.telemetry()["pseudo_representations_written_persistently"] is False
    full.close()
    lru.close()


def test_d1_d2_e7_and_independent_root_contracts(monkeypatch) -> None:
    d1 = _source(
        monkeypatch,
        generators=(_Generator("D1", ("exclusive_kt",)),),
        mode=ABPH_FULL_RANK_PSEUDO_MODE,
    )
    d2 = _source(
        monkeypatch,
        generators=(_Generator("D2", ("cambridge_aachen",)),),
        mode=ABPH_FULL_RANK_PSEUDO_MODE,
    )
    e7 = _source(
        monkeypatch,
        generators=(
            _Generator("E7", ("exclusive_kt", "cambridge_aachen")),
        ),
        mode=ABPH_FULL_RANK_PSEUDO_MODE,
    )
    independent = FrozenReconstructorRamSource(
        hlt_cache_dir=Path("unused"),
        generators=(
            _Generator("D1", ("exclusive_kt",), root_offset=0.0),
            _Generator("D2", ("cambridge_aachen",), root_offset=10.0),
        ),
        split="model_train",
        batch_size=3,
        shard_size=4,
        generation_batch_size=2,
        execution_mode=ABPH_FULL_RANK_PSEUDO_MODE,
        independent_roots=True,
    )
    assert d1.metadata[0]["hierarchy_names"] == ["exclusive_kt"]
    assert d2.metadata[0]["hierarchy_names"] == ["cambridge_aachen"]
    e7_batch = next(e7.iter_batches(shuffle=False, seed=1))
    root = e7_batch.pseudo.arrays["shared_root_ledger"]
    for hierarchy in ("exclusive_kt", "cambridge_aachen"):
        frontier_root = e7_batch.pseudo.arrays[
            f"frontier__{hierarchy}__depth_00__ledger"
        ][:, :, 0]
        assert np.array_equal(frontier_root, np.broadcast_to(root[:, None], frontier_root.shape))
    independent_batch = next(independent.iter_batches(shuffle=False, seed=1))
    assert set(independent_batch.independent_roots) == {
        "exclusive_kt",
        "cambridge_aachen",
    }
    assert not np.array_equal(
        independent_batch.independent_roots["exclusive_kt"],
        independent_batch.independent_roots["cambridge_aachen"],
    )
    for source in (d1, d2, e7, independent):
        telemetry = source.telemetry()
        assert telemetry["source_hash"]
        assert telemetry["rank_identity_hash"]
        assert telemetry["consumer_pseudo_schema_hashes"]
        source.close()
