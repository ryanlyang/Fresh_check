from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_FIXED_EVALUATION_SEED,
    ABPH_HLT_PROFILE,
    ABPH_HLT_PROFILE_VERSION,
    ABPH_PRIMARY_HYPOTHESIS_NAMES,
    ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
    DeployableHLTBatch,
    DeployablePseudoViewBatch,
    DeployablePseudoViewCacheConfig,
    HierarchyFrontier,
    HierarchyHypothesis,
    HypothesisIdentity,
    HypothesisLatentSet,
    MultiHypothesisHierarchyOutput,
    RecursiveHierarchyOutput,
    RenderedParticleBatch,
    abph_hlt_params_dict,
    audit_deployable_pseudo_view_cache,
    canonical_hash,
    describe_reconstructor_model,
    generate_deployable_pseudo_view_cache,
    package_deployable_pseudo_views,
    reassemble_deployable_pseudo_view_cache,
)


class _TinySelectedReconstructor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root = torch.nn.Linear(2, 2)

    def module_groups(self):
        return {"root_predictor": self.root}


class _StreamingSource:
    def __init__(self, n_jets: int = 7, batch_size: int = 2, hlt_hash: str = "hlt-hash"):
        self.tokens = np.arange(n_jets * 128 * 3, dtype=np.float32).reshape(n_jets, 128, 3)
        self.mask = np.ones((n_jets, 128), dtype=bool)
        self.labels = np.arange(n_jets, dtype=np.int64) % len(LABEL_NAMES)
        self.jet_ids = tuple(
            JetIdentity(file=f"class_{int(label)}.root", entry=index + 100, label=int(label))
            for index, label in enumerate(self.labels)
        )
        self.batch_size = int(batch_size)
        self.maximum_yielded_rows = 0
        self.streaming = True
        self.resident_bytes = int(self.tokens.nbytes + self.mask.nbytes + self.labels.nbytes)
        self.provenance = {
            "source_manifest_hash": "manifest-hash",
            "hlt_content_hash": hlt_hash,
            "jet_identity_hash": jet_identity_hash(self.jet_ids),
            "label_hash": hashlib.sha256(
                np.ascontiguousarray(self.labels, dtype=np.int64).tobytes()
            ).hexdigest(),
            "class_mapping_hash": canonical_hash({"label_names": list(LABEL_NAMES)}),
            "hlt_profile": ABPH_HLT_PROFILE,
            "hlt_profile_version": ABPH_HLT_PROFILE_VERSION,
            "hlt_degradation_strength": ABPH_HLT_DEGRADATION_STRENGTH,
            "hlt_params_hash": canonical_hash(abph_hlt_params_dict()),
            "split": "model_val",
            "n_jets": n_jets,
            "target_cache_loaded": False,
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "source_kind": "unit_test_stream",
        }

    def __len__(self):
        return len(self.labels)

    def iter_batches(self):
        for start in range(0, len(self), self.batch_size):
            stop = min(start + self.batch_size, len(self))
            self.maximum_yielded_rows = max(self.maximum_yielded_rows, stop - start)
            yield DeployableHLTBatch(
                tokens=self.tokens[start:stop],
                mask=self.mask[start:stop],
                labels=self.labels[start:stop],
                indices=np.arange(start, stop, dtype=np.int64),
                jet_ids=self.jet_ids[start:stop],
            )


def _selected_checkpoint(
    path: Path,
    model: _TinySelectedReconstructor,
    *,
    role: str = "best_model_val",
) -> Path:
    torch.save(
        {
            "checkpoint_contract": ABPH_RECONSTRUCTOR_TRAINING_CONTRACT,
            "checkpoint_role": role,
            "final_test_loaded": False,
            "teacher_logits_loaded": False,
            "model_state_dict": model.state_dict(),
            "model_metadata": describe_reconstructor_model(model, model.module_groups()),
            "config": {"seed": 17, "curriculum": "unit-test"},
            "provenance": {
                "source_git_commit": "deadbeef",
                "source_status_hash": "clean-status-hash",
            },
        },
        path,
    )
    return path


def _prediction_batch(batch: DeployableHLTBatch) -> DeployablePseudoViewBatch:
    batch_size = batch.batch_size
    views = len(ABPH_PRIMARY_HYPOTHESIS_NAMES)
    particles = 128
    hierarchy = "exclusive_kt"
    indices = np.asarray(batch.indices, dtype=np.float32)
    arrays = {
        "shared_root_ledger": np.stack((indices, indices + 0.5, indices + 1.0), axis=-1),
        "hypothesis_latent": np.broadcast_to(
            indices[:, None, None], (batch_size, views, 2)
        ).copy(),
        "hypothesis_prior_log_prob": np.zeros((batch_size, views), dtype=np.float32),
        f"particle__{hierarchy}__canonical_features": np.zeros(
            (batch_size, views, particles, 3), dtype=np.float32
        ),
        f"particle__{hierarchy}__side_channels": np.zeros(
            (batch_size, views, particles, 2), dtype=np.float32
        ),
        f"particle__{hierarchy}__four_vector": np.zeros(
            (batch_size, views, particles, 4), dtype=np.float32
        ),
        f"particle__{hierarchy}__mass": np.zeros(
            (batch_size, views, particles), dtype=np.float32
        ),
        f"particle__{hierarchy}__mask": np.ones(
            (batch_size, views, particles), dtype=bool
        ),
        f"particle__{hierarchy}__group_indices": np.zeros(
            (batch_size, views, particles), dtype=np.int64
        ),
        f"particle__{hierarchy}__local_slot_indices": np.broadcast_to(
            np.arange(particles, dtype=np.int64), (batch_size, views, particles)
        ).copy(),
        f"particle__{hierarchy}__uncertainty": np.full(
            (batch_size, views, particles), 0.25, dtype=np.float32
        ),
        f"particle__{hierarchy}__slot_hidden": np.zeros(
            (batch_size, views, particles, 5), dtype=np.float32
        ),
    }
    for depth, capacity in enumerate((1, 2)):
        prefix = f"frontier__{hierarchy}__depth_{depth:02d}"
        arrays.update(
            {
                f"{prefix}__ledger": np.zeros(
                    (batch_size, views, capacity, 3), dtype=np.float32
                ),
                f"{prefix}__hidden": np.zeros(
                    (batch_size, views, capacity, 5), dtype=np.float32
                ),
                f"{prefix}__support": np.zeros(
                    (batch_size, views, capacity, 4), dtype=np.float32
                ),
                f"{prefix}__uncertainty": np.full(
                    (batch_size, views, capacity), 0.1, dtype=np.float32
                ),
                f"{prefix}__mask": np.ones(
                    (batch_size, views, capacity), dtype=bool
                ),
                f"{prefix}__topology": np.ones(
                    (batch_size, views, capacity), dtype=np.int64
                ),
                f"{prefix}__parent_indices": np.full(
                    (batch_size, views, capacity), -1 if depth == 0 else 0, dtype=np.int64
                ),
                f"{prefix}__source_child_indices": np.full(
                    (batch_size, views, capacity), -1, dtype=np.int64
                ),
            }
        )
    return DeployablePseudoViewBatch(
        arrays=arrays,
        view_names=ABPH_PRIMARY_HYPOTHESIS_NAMES,
        hierarchy_names=(hierarchy,),
        frontier_depths={hierarchy: 2},
        diagnostics={
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
        },
    )


def _predictor(model, batch, evaluation_seed, device):
    assert isinstance(model, _TinySelectedReconstructor)
    assert evaluation_seed == ABPH_FIXED_EVALUATION_SEED
    assert device == "cpu"
    return _prediction_batch(batch)


def _config(checkpoint: Path, output: Path, **updates) -> DeployablePseudoViewCacheConfig:
    values = {
        "checkpoint_path": checkpoint,
        "output_cache_dir": output,
        "split": "model_val",
        "resolved_variant_config_hash": "variant-hash",
        "campaign_mode": "highdata",
        "device": "cpu",
        "shard_size": 3,
        "feature_dtype": "float16",
    }
    values.update(updates)
    return DeployablePseudoViewCacheConfig(**values)


@pytest.mark.parametrize(
    "field",
    ["offline_cache_dir", "target_cache_dir", "teacher_logits_dir"],
)
def test_deployable_config_rejects_every_privileged_path(tmp_path, field):
    model = _TinySelectedReconstructor()
    checkpoint = _selected_checkpoint(tmp_path / "selected.pt", model)
    with pytest.raises(ValueError, match="rejects privileged paths"):
        _config(checkpoint, tmp_path / "cache", **{field: tmp_path / "privileged"})


def test_deployment_rejects_stage_local_checkpoint(tmp_path):
    model = _TinySelectedReconstructor()
    checkpoint = _selected_checkpoint(
        tmp_path / "stage.pt", model, role="best_stage_model_val"
    )
    with pytest.raises(ValueError, match="completed best_model_val"):
        generate_deployable_pseudo_view_cache(
            model,
            model.module_groups(),
            _StreamingSource(),
            _config(checkpoint, tmp_path / "cache"),
            predictor=_predictor,
        )


def test_streaming_cache_binds_inputs_and_reassembles_order(tmp_path):
    model = _TinySelectedReconstructor()
    checkpoint = _selected_checkpoint(tmp_path / "selected.pt", model)
    source = _StreamingSource(n_jets=7, batch_size=2)
    output = tmp_path / "cache"
    metadata = generate_deployable_pseudo_view_cache(
        model,
        model.module_groups(),
        source,
        _config(checkpoint, output),
        predictor=_predictor,
    )
    assert metadata["n_shards"] == 3
    assert [row["n_jets"] for row in metadata["shards"]] == [3, 3, 1]
    assert metadata["hlt_content_hash"] == "hlt-hash"
    assert metadata["checkpoint_sha256"]
    assert metadata["final_test_attestation"] == {
        "offline_inputs_loaded": False,
        "teacher_logits_loaded": False,
        "offline_target_selected_hypothesis": False,
        "fusion_fitted_on_final_test": False,
    }
    assert metadata["memory_audit"]["full_prediction_materialized"] is False
    assert metadata["memory_audit"]["maximum_buffered_rows"] == 3
    assert metadata["memory_audit"]["bounded_buffer_verified"] is True
    assert source.maximum_yielded_rows == 2

    audit = audit_deployable_pseudo_view_cache(
        output,
        expected_bindings={
            "source_manifest_hash": "manifest-hash",
            "hlt_content_hash": "hlt-hash",
            "jet_identity_hash": source.provenance["jet_identity_hash"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        },
    )
    assert audit["ok"], audit["problems"]
    joined = reassemble_deployable_pseudo_view_cache(output)
    assert joined.jet_ids == source.jet_ids
    assert np.array_equal(joined.indices, np.arange(7, dtype=np.int64))
    assert np.array_equal(joined.labels, source.labels)
    assert np.array_equal(
        joined.arrays["shared_root_ledger"][:, 0], np.arange(7, dtype=np.float16)
    )


def test_partial_and_stale_caches_fail_reuse_closed(tmp_path):
    model = _TinySelectedReconstructor()
    checkpoint = _selected_checkpoint(tmp_path / "selected.pt", model)
    source = _StreamingSource()
    output = tmp_path / "cache"
    generate_deployable_pseudo_view_cache(
        model,
        model.module_groups(),
        source,
        _config(checkpoint, output),
        predictor=_predictor,
    )
    stale = audit_deployable_pseudo_view_cache(
        output, expected_bindings={"hlt_content_hash": "replacement-hlt-cache"}
    )
    assert stale["ok"] is False
    assert any("hlt_content_hash mismatch" in value for value in stale["problems"])

    partial_output = tmp_path / "partial"
    shutil.copytree(output, partial_output)
    (partial_output / "shard_000001.npz").unlink()
    partial = audit_deployable_pseudo_view_cache(partial_output)
    assert partial["ok"] is False
    assert any("missing cache files" in value for value in partial["problems"])

    with pytest.raises(ValueError, match="stale or partial"):
        generate_deployable_pseudo_view_cache(
            model,
            model.module_groups(),
            _StreamingSource(hlt_hash="replacement-hlt-cache"),
            _config(checkpoint, output, reuse_existing=True),
            predictor=_predictor,
        )


def test_tampered_numeric_shard_and_sidecar_are_detected(tmp_path):
    model = _TinySelectedReconstructor()
    checkpoint = _selected_checkpoint(tmp_path / "selected.pt", model)
    output = tmp_path / "cache"
    generate_deployable_pseudo_view_cache(
        model,
        model.module_groups(),
        _StreamingSource(),
        _config(checkpoint, output),
        predictor=_predictor,
    )
    sidecar_path = output / "shard_000000.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = "tampered"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    report = audit_deployable_pseudo_view_cache(output)
    assert report["ok"] is False
    assert any("sidecar hash mismatch" in value for value in report["problems"])


def test_rollout_and_renderer_outputs_package_all_views_frontiers_and_uncertainty():
    batch_size = 2
    views = len(ABPH_PRIMARY_HYPOTHESIS_NAMES)
    root_ledger = torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3)

    def frontier(capacity, parent):
        return HierarchyFrontier(
            ledger=torch.zeros(batch_size, capacity, 3),
            hidden=torch.zeros(batch_size, capacity, 5),
            support=torch.zeros(batch_size, capacity, 4),
            uncertainty=torch.full((batch_size, capacity), 0.2),
            mask=torch.ones(batch_size, capacity, dtype=torch.bool),
            topology=torch.ones(batch_size, capacity, dtype=torch.long),
            parent_indices=torch.full((batch_size, capacity), parent, dtype=torch.long),
            source_child_indices=torch.full((batch_size, capacity), -1, dtype=torch.long),
        )

    identities = tuple(
        HypothesisIdentity(
            index=index,
            name=name,
            kind="mean" if index == 0 else "stochastic",
            sampling_seed=None if index == 0 else 100 + index,
            report_identity=name,
        )
        for index, name in enumerate(ABPH_PRIMARY_HYPOTHESIS_NAMES)
    )
    hypotheses = []
    rendered = []
    for index, identity in enumerate(identities):
        root = frontier(1, -1)
        root = HierarchyFrontier(
            ledger=root_ledger[:, None, :],
            hidden=root.hidden,
            support=root.support,
            uncertainty=root.uncertainty,
            mask=root.mask,
            topology=root.topology,
            parent_indices=root.parent_indices,
            source_child_indices=root.source_child_indices,
        )
        final = frontier(2, 0)
        recursive = RecursiveHierarchyOutput(
            mode="rollout",
            root_frontier=root,
            levels=(type("Level", (), {"next_frontier": final})(),),
            final_frontier=final,
            diagnostics={},
        )
        hypotheses.append(
            HierarchyHypothesis(
                identity=identity,
                latent=torch.full((batch_size, 2), float(index)),
                prior_log_prob=torch.zeros(batch_size),
                hierarchy_outputs={"exclusive_kt": recursive},
            )
        )
        particle_mask = torch.ones(batch_size, 128, dtype=torch.bool)
        rendered.append(
            RenderedParticleBatch(
                four_vector=torch.zeros(batch_size, 128, 4),
                mass=torch.zeros(batch_size, 128),
                mask=particle_mask,
                group_indices=torch.zeros(batch_size, 128, dtype=torch.long),
                local_slot_indices=torch.arange(128).expand(batch_size, 128),
                soft_pid_probabilities=torch.zeros(batch_size, 128, 6),
                hard_pid_indices=torch.zeros(batch_size, 128, dtype=torch.long),
                hard_pid_one_hot=torch.zeros(batch_size, 128, 6),
                charges=torch.zeros(batch_size, 128),
                hard_charges=torch.zeros(batch_size, 128, dtype=torch.long),
                track_features=torch.zeros(batch_size, 128, 4),
                uncertainty=torch.full((batch_size, 128), 0.3),
                canonical_features=torch.zeros(batch_size, 128, 19),
                side_channels=torch.zeros(batch_size, 128, 6),
                slot_hidden=torch.zeros(batch_size, 128, 5),
                diagnostics={
                    "offline_inputs_consumed": False,
                    "hlt_only_deployment_inputs": True,
                    "hypothesis_index": index,
                },
            )
        )
    latent_set = HypothesisLatentSet(
        values=torch.stack([item.latent for item in hypotheses], dim=1),
        prior_log_prob=torch.stack([item.prior_log_prob for item in hypotheses], dim=1),
        identities=identities,
        evaluation_seed=ABPH_FIXED_EVALUATION_SEED,
        diagnostics={},
    )
    output = MultiHypothesisHierarchyOutput(
        hypotheses=tuple(hypotheses),
        latent_set=latent_set,
        shared_root_ledger=root_ledger,
        diagnostics={
            "exact_root_identity_across_all_hypotheses_and_hierarchies": True
        },
    )
    packaged = package_deployable_pseudo_views(
        output, {"exclusive_kt": tuple(rendered)}
    )
    report = packaged.validate()
    assert report["ok"] is True
    assert packaged.arrays["particle__exclusive_kt__uncertainty"].shape == (
        batch_size,
        views,
        128,
    )
    assert packaged.arrays[
        "frontier__exclusive_kt__depth_01__hidden"
    ].shape == (batch_size, views, 2, 5)
    assert np.array_equal(packaged.arrays["shared_root_ledger"], root_ledger.numpy())
    assert packaged.diagnostics["consumer_only_pseudo"] is True
    assert "particle__exclusive_kt__four_vector" not in packaged.arrays
    assert "frontier__exclusive_kt__depth_00__source_child_indices" not in packaged.arrays
    forensic = package_deployable_pseudo_views(
        output, {"exclusive_kt": tuple(rendered)}, consumer_only=False
    )
    assert forensic.diagnostics["consumer_only_pseudo"] is False
    assert "particle__exclusive_kt__four_vector" in forensic.arrays
    assert "frontier__exclusive_kt__depth_00__source_child_indices" in forensic.arrays
