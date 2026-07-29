from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge import (
    CoupledExpertDecoder,
    JointBridgeGraph,
    configure_joint_trainability,
    validate_common_view_metadata,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (
    JointBridgeDataset,
    JointBridgeTrainingConfig,
    evaluate_joint_bridge,
    load_joint_dataset_cache,
    load_joint_graph_template,
    make_joint_bridge_loader,
    publish_joint_dataset_cache,
    publish_joint_graph_template,
    train_joint_bridge,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.predictors import (
    RetbTokenPredictor,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.replicas import replica_for
from teacher_logit_reco.relation_expert_token_bridge.step11 import (
    SEMANTIC_LABELS,
    build_stage_j_policy,
    build_stage_j_registry,
    build_step11_bundle,
    materialize_stage_j_run,
    select_j4_block_count,
    validate_materialized_stage_j_run,
    validate_stage_j_registry,
    validate_step11_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _lock():
    return with_content_hash(
        {
            "contract": PREDICTOR_BUNDLE_LOCK_CONTRACT,
            "schema_version": 1,
            "expert_order": list(EXPERT_ORDER),
            "candidate_hashes": {
                expert: _digest(f"candidate-{expert}")
                for expert in EXPERT_ORDER
            },
            "seed_specific_artifacts": {
                str(seed): {
                    expert: {
                        "predictor_registration": _digest(
                            f"registration-{seed}-{expert}"
                        )
                    }
                    for expert in EXPERT_ORDER
                }
                for seed in (101, 202, 303)
            },
            "selected_candidate_descriptors": {
                expert: {
                    "candidate_id": f"{expert}_selected",
                    "target_mode": (
                        "T1_TASK_BRIDGE"
                        if expert == "PT"
                        else "T0_PURE"
                    ),
                    "configuration": {
                        "objective_id": (
                            "W_LOGIT_ONLY"
                            if expert == "PT"
                            else "W_CANONICAL"
                        )
                    },
                }
                for expert in EXPERT_ORDER
            },
            "configuration_shared_across_pipeline_seeds": True,
            "per_seed_selection_permitted": False,
            "locked_before_joint_training": True,
        }
    )


class _Head(torch.nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.classifier = torch.nn.Linear(dimension, 10)

    def forward(self, tokens):
        return self.classifier(tokens.mean(dim=1))


class _Fusion(torch.nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.classifier = torch.nn.Linear(7 * dimension, 10)

    def forward(self, *, token_banks):
        return self.classifier(
            torch.cat(
                [
                    token_banks[expert].mean(dim=1)
                    for expert in EXPERT_ORDER
                ],
                dim=-1,
            )
        )


def _parts():
    torch.manual_seed(101)
    predictors = {
        expert: RetbTokenPredictor(
            architecture="A3_SLOT_DECODER_DIRECT",
            context="C2_ALL",
            target_expert_id=expert,
            token_count=2,
            token_dimension=64,
            offline_slot_queries=torch.randn(2, 64),
            uncertainty_head="U_SLOT",
            dropout=0.0,
        )
        for expert in EXPERT_ORDER
    }
    heads = {expert: _Head() for expert in EXPERT_ORDER}
    means = {expert: np.zeros((2, 64), np.float32) for expert in EXPERT_ORDER}
    stds = {expert: np.ones((2, 64), np.float32) for expert in EXPERT_ORDER}
    return predictors, heads, means, stds


def _evidence(batch=4):
    torch.manual_seed(102)
    return {
        "hlt_token_banks": {
            expert: torch.randn(batch, 2, 64) for expert in EXPERT_ORDER
        },
        "unbiased_particle_states": torch.randn(batch, 5, 64),
        "particle_mask": torch.ones(batch, 5, dtype=torch.bool),
        "relation_particle_states": {
            expert: torch.randn(batch, 5, 64)
            for expert in ("PT", "TRACK", "REGION")
        },
        "relation_particle_masks": {
            expert: torch.ones(batch, 5, dtype=torch.bool)
            for expert in ("PT", "TRACK", "REGION")
        },
    }


class _ParticleCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(17, 64)
        self.pair_bias_provider = torch.nn.Linear(64, 64)
        self.mod = torch.nn.Module()
        self.mod.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(64, 64) for _ in range(8)]
        )


class _LiveExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.particle_encoder = _ParticleCore()
        self.tokenizer = torch.nn.Linear(64, 64)
        self.head = _Head()

    def forward(
        self,
        *,
        features,
        vectors,
        mask,
        raw_tokens,
        region_trees,
        return_details,
    ):
        del vectors, raw_tokens, region_trees
        states = self.particle_encoder.input(features.transpose(1, 2))
        for block in self.particle_encoder.mod.blocks:
            states = torch.tanh(block(states))
        states = states + self.particle_encoder.pair_bias_provider(states)
        tokens = self.tokenizer(states[:, :2])
        logits = self.head(tokens)
        return {
            "particle_states": states,
            "particle_mask": mask[:, 0].bool(),
            "tokens": tokens,
            "logits": logits,
        }


def _graph(variant: str):
    predictors, heads, means, stds = _parts()
    fusion = _Fusion()
    kwargs = {}
    if variant == "J2_COUPLED_DECODER":
        kwargs["coupled_decoder"] = CoupledExpertDecoder(
            allocation={expert: [2, 64] for expert in EXPERT_ORDER},
            offline_slot_queries={
                expert: predictor.target_queries.detach()
                for expert, predictor in predictors.items()
            },
            uncertainty_widths={expert: 1 for expert in EXPERT_ORDER},
            dropout=0.0,
        )
    if variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}:
        kwargs["hlt_experts"] = {
            expert: _LiveExpert() for expert in EXPERT_ORDER
        }
    if variant == "J5_END_TO_END":
        kwargs["deployable_fusion"] = copy.deepcopy(fusion)
    return JointBridgeGraph(
        variant=variant,
        predictors=predictors,
        frozen_offline_fusion=fusion,
        frozen_expert_heads=heads,
        token_means=means,
        token_standard_deviations=stds,
        **kwargs,
    )


def _shared_view(batch=4):
    return {
        "identities": [f"jet-{index}" for index in range(batch)],
        "replica_ids": torch.arange(batch) % 4,
        "degraded_view_hashes": [
            _digest(f"view-{index}") for index in range(batch)
        ],
        "features": torch.randn(batch, 17, 5),
        "vectors": torch.randn(batch, 4, 5),
        "mask": torch.ones(batch, 1, 5, dtype=torch.bool),
        "raw_tokens": torch.randn(batch, 5, 14),
        "region_trees_by_expert": {
            expert: [None] * batch for expert in EXPERT_ORDER
        },
    }


def test_stage_j_policy_registry_and_bundle_are_exact() -> None:
    lock = _lock()
    policy = build_stage_j_policy()
    registry = build_stage_j_registry(predictor_bundle_lock=lock)
    assert policy["input_policy"] == "R_MULTI"
    assert registry["membership_count"] == 21
    assert registry["selected_bundle_is_task_distilled"]
    assert validate_stage_j_registry(
        registry, predictor_bundle_lock=lock
    ) == registry["content_hash"]
    assert set(registry["semantic_comparison_registry"]) == {
        "FAITHFUL",
        "COORDINATED",
        "LOGIT_DISTILLED",
        "BRIDGE_TUNED",
        "END_TO_END",
    }
    bundle = build_step11_bundle(
        campaign_spec_sha256=SHA_A,
        step10_bundle_sha256=SHA_B,
        predictor_bundle_lock=lock,
        global_determinism_sha256=SHA_C,
        source_snapshot=SOURCE,
    )
    assert (
        validate_step11_bundle(bundle, predictor_bundle_lock=lock)
        == bundle["step11_bundle"]["content_hash"]
    )


def test_materialized_stage_j_run_is_fail_closed() -> None:
    parents = {
        name: _digest(name)
        for name in (
            "model_train_identity_manifest",
            "val_stop_identity_manifest",
            "val_design_identity_manifest",
            "val_design_label_manifest",
            "model_train_R_MULTI_view_cache",
            "val_stop_R_MULTI_view_cache",
            "val_design_fixed_view_cache",
            "offline_target_cache",
            "target_normalizer_set",
            "frozen_offline_fusion",
            "frozen_offline_expert_heads",
            "selected_predictor_seed_artifacts",
            "selected_HLT_expert_seed_artifacts",
        )
    }
    run = materialize_stage_j_run(
        run_id="RETB_J4_BRIDGE_FINETUNE_S101_N2",
        variant="J4_BRIDGE_FINETUNE",
        pipeline_seed=101,
        final_particle_blocks=2,
        predictor_bundle_lock_sha256=SHA_A,
        step11_bundle_sha256=SHA_B,
        parent_hashes=parents,
        semantic_label=SEMANTIC_LABELS["J4_BRIDGE_FINETUNE"],
    )
    assert validate_materialized_stage_j_run(run) == run["content_hash"]
    with pytest.raises(ValueError, match="semantics differ"):
        materialize_stage_j_run(
            **{
                **{
                    "run_id": "bad",
                    "variant": "J4_BRIDGE_FINETUNE",
                    "pipeline_seed": 101,
                    "final_particle_blocks": 3,
                    "predictor_bundle_lock_sha256": SHA_A,
                    "step11_bundle_sha256": SHA_B,
                    "parent_hashes": parents,
                    "semantic_label": SEMANTIC_LABELS[
                        "J4_BRIDGE_FINETUNE"
                    ],
                }
            }
        )


def test_j0_j1_j2_j3_forward_and_trainability_boundaries() -> None:
    evidence = _evidence()
    j0 = _graph("J0_INDEPENDENT")
    out0 = j0(evidence=evidence)
    assert out0["logits"].shape == (4, 10)
    assert not configure_joint_trainability(j0)["optimizer_groups"]

    j1 = _graph("J1_SHARED_CONTEXT")
    out1 = j1(evidence=evidence)
    assert out1["logits"].shape == (4, 10)
    assert j1.shared_context.forward_call_count == 1
    trainability = configure_joint_trainability(j1)
    assert set(trainability["parameter_groups"]) == {
        "predictors",
        "shared_context",
        "shared_memory_projections",
    }

    j2 = _graph("J2_COUPLED_DECODER")
    out2 = j2(evidence=evidence)
    assert set(out2["predicted_tokens"]) == set(EXPERT_ORDER)
    assert all(
        value.shape == (4, 2, 64)
        for value in out2["predicted_tokens"].values()
    )

    j3 = _graph("J3_INDEPENDENT_PLUS_ADAPTER")
    j3.eval()
    output = j3(evidence=evidence)
    base = j3.deployable_fusion(token_banks=output["predicted_tokens"])
    assert torch.equal(output["logits"], base)
    trainability = configure_joint_trainability(j3)
    assert set(trainability["parameter_groups"]) == {"adapter"}
    assert not any(
        parameter.requires_grad for parameter in j3.predictors.parameters()
    )


def test_j4_j5_live_common_view_and_trainability() -> None:
    view = _shared_view()
    j4 = _graph("J4_BRIDGE_FINETUNE")
    output = j4(shared_view=view)
    assert output["logits"].shape == (4, 10)
    trainability = configure_joint_trainability(
        j4, final_particle_blocks=2
    )
    assert trainability["offline_expert_heads_frozen"]
    assert trainability["offline_fusion_frozen"]
    assert any(name.startswith("hlt_tokenizer.") for name in trainability[
        "parameter_groups"
    ])
    assert any(name.startswith("hlt_final_blocks.") for name in trainability[
        "parameter_groups"
    ])

    j5 = _graph("J5_END_TO_END")
    trainability = configure_joint_trainability(j5)
    assert "deployable_fusion" in trainability["parameter_groups"]
    assert trainability["offline_fusion_frozen"]
    assert trainability["final_particle_blocks"] is None


def test_common_view_metadata_rejects_replica_and_identity_drift() -> None:
    with pytest.raises(ValueError, match="metadata differs"):
        validate_common_view_metadata(
            identities=["a", "a"],
            replica_ids=[0, 1],
            degraded_view_hashes=[SHA_A, SHA_B],
        )
    with pytest.raises(ValueError, match="metadata differs"):
        validate_common_view_metadata(
            identities=["a", "b"],
            replica_ids=[0, 4],
            degraded_view_hashes=[SHA_A, SHA_B],
        )


def test_joint_dataset_cycles_one_common_rmulti_replica_per_epoch() -> None:
    dataset = _dataset("model_train")
    first = [int(dataset[index]["replica_id"]) for index in range(len(dataset))]
    dataset.set_epoch(2)
    second = [
        int(dataset[index]["replica_id"]) for index in range(len(dataset))
    ]
    assert second == [(value + 1) % 4 for value in first]
    for index in range(len(dataset)):
        row = dataset[index]
        assert row["degraded_view_hash"] == (
            dataset.degraded_view_hashes_by_replica[row["replica_id"]][index]
        )


def test_j4_block_selector_is_deterministic_when_every_row_loses() -> None:
    rows = []
    for blocks in (2, 4):
        for seed in (101, 202, 303):
            rows.append(
                {
                    "variant": "J4_BRIDGE_FINETUNE",
                    "final_particle_blocks": blocks,
                    "pipeline_seed": seed,
                    "split": "val_design",
                    "accuracy": 0.1,
                    "cross_entropy": 2.3,
                    "normalized_token_error": 1.0,
                    "inference_flops": 100,
                    "parameter_count": 200,
                    "registration_sha256": _digest(
                        f"registration-{blocks}-{seed}"
                    ),
                }
            )
    first = select_j4_block_count(
        rows,
        predictor_bundle_lock_sha256=SHA_A,
        label_manifest_sha256=SHA_B,
    )
    second = select_j4_block_count(
        rows,
        predictor_bundle_lock_sha256=SHA_A,
        label_manifest_sha256=SHA_B,
    )
    assert first == second
    assert first["selected_final_particle_blocks"] == 2
    assert first["all_candidates_worse_than_baseline_does_not_block"]


def _dataset(split: str, count: int = 20):
    rng = np.random.default_rng(1101 if split == "model_train" else 1102)
    identities = [f"{split}-{index:03d}" for index in range(count)]
    evidence = {
        expert: rng.normal(size=(count, 2, 64)).astype(np.float32)
        for expert in EXPERT_ORDER
    }
    training = split == "model_train"

    def replicated(value):
        if not training:
            return value
        return np.stack(
            [value + np.float32(replica * 0.001) for replica in range(4)]
        )

    replica_ids = np.asarray(
        [
            replica_for(
                policy="R_MULTI",
                logical_role=split,
                epoch=0,
                canonical_identity=identity,
            )
            for identity in identities
        ],
        dtype=np.int64,
    )
    view_hashes = (
        [
            _digest(f"{split}-view-{index}")
            for index in range(count)
        ]
        if not training
        else np.asarray(
            [
                [
                    _digest(f"{split}-view-{replica}-{index}")
                    for index in range(count)
                ]
                for replica in range(4)
            ]
        )
    )
    oracle = {
        expert: rng.normal(size=(count, 2, 64)).astype(np.float32)
        for expert in EXPERT_ORDER
    }
    heads = {expert: _Head() for expert in EXPERT_ORDER}
    fusion = _Fusion()
    with torch.no_grad():
        target_logits = {
            expert: heads[expert](torch.from_numpy(oracle[expert])).numpy()
            for expert in EXPERT_ORDER
        }
        fusion_logits = fusion(
            token_banks={
                expert: torch.from_numpy(value)
                for expert, value in oracle.items()
            }
        ).numpy()
    return JointBridgeDataset(
        identities=identities,
        labels=np.arange(count, dtype=np.int64) % 10,
        replica_ids=replica_ids,
        degraded_view_hashes=view_hashes,
        split=split,
        hlt_token_banks={
            expert: replicated(value)
            for expert, value in evidence.items()
        },
        unbiased_particle_states=replicated(
            rng.normal(size=(count, 4, 64)).astype(np.float32)
        ),
        particle_mask=(
            np.stack([np.ones((count, 4), dtype=bool)] * 4)
            if training
            else np.ones((count, 4), dtype=bool)
        ),
        relation_particle_states={
            expert: replicated(
                rng.normal(size=(count, 4, 64)).astype(np.float32)
            )
            for expert in ("PT", "TRACK", "REGION")
        },
        relation_particle_masks={
            expert: (
                np.stack([np.ones((count, 4), dtype=bool)] * 4)
                if training
                else np.ones((count, 4), dtype=bool)
            )
            for expert in ("PT", "TRACK", "REGION")
        },
        target_normalized_banks=oracle,
        oracle_banks=oracle,
        target_expert_logits=target_logits,
        oracle_fusion_logits=fusion_logits,
        shared_raw_view=None,
        lineage_hashes={
            "identity_manifest": _digest("identity"),
            "HLT_view_cache": _digest("view"),
            "offline_target_cache": _digest("targets"),
            "target_normalizer_set": _digest("normalizers"),
        },
    )


def test_miniature_j3_training_runs_fixed_budget_and_reuses(tmp_path) -> None:
    graph = _graph("J3_INDEPENDENT_PLUS_ADAPTER")
    train = _dataset("model_train")
    stop = _dataset("val_stop")
    train_loader = make_joint_bridge_loader(
        train, batch_size=4, seed=101, training=True
    )
    stop_loader = make_joint_bridge_loader(
        stop, batch_size=4, seed=101, training=False
    )
    run = with_content_hash(
        {
            "contract": "retb_test_stage_j_run_v1",
            "run_id": "RETB_J3_TEST",
            "variant": "J3_INDEPENDENT_PLUS_ADAPTER",
            "pipeline_seed": 101,
        }
    )
    kwargs = {
        "graph": graph,
        "train_loader": train_loader,
        "val_stop_loader": stop_loader,
        "objective_by_expert": {
            expert: "W_CANONICAL" for expert in EXPERT_ORDER
        },
        "gradnorm_weights_by_expert": {
            expert: None for expert in EXPERT_ORDER
        },
        "output_dir": tmp_path,
        "run_record": run,
        "step11_bundle_sha256": SHA_A,
        "predictor_bundle_lock_sha256": SHA_B,
        "global_determinism_sha256": SHA_C,
        "lineage_hashes": {"fixture": _digest("fixture")},
        "source_snapshot": SOURCE,
        "config": JointBridgeTrainingConfig(
            seed=101,
            variant="J3_INDEPENDENT_PLUS_ADAPTER",
            maximum_epochs=2,
            microbatch_size=4,
            gradient_accumulation_steps=2,
            effective_batch_size=8,
            campaign_profile="miniature_test",
        ),
        "device": "cpu",
    }
    first = train_joint_bridge(**kwargs)
    selected = torch.load(
        tmp_path / "best_model_val.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert all(
        torch.equal(graph.state_dict()[name].cpu(), value)
        for name, value in selected["model_state_dict"].items()
        if isinstance(value, torch.Tensor)
    )
    second = train_joint_bridge(**kwargs)
    assert first == second
    evaluation = evaluate_joint_bridge(
        graph=graph,
        loader=stop_loader,
        objective_by_expert=kwargs["objective_by_expert"],
        gradnorm_weights_by_expert=kwargs[
            "gradnorm_weights_by_expert"
        ],
        device=torch.device("cpu"),
    )
    assert evaluation["metrics"]["event_count"] == 20


def test_graph_template_and_dataset_cache_round_trip(tmp_path) -> None:
    graph = _graph("J3_INDEPENDENT_PLUS_ADAPTER")
    objectives = {expert: "W_CANONICAL" for expert in EXPERT_ORDER}
    gradnorm = {expert: None for expert in EXPERT_ORDER}
    graph_manifest = publish_joint_graph_template(
        output_dir=tmp_path / "graph",
        graph=graph,
        run_record_sha256=SHA_A,
        predictor_bundle_lock_sha256=SHA_B,
        objective_by_expert=objectives,
        gradnorm_weights_by_expert=gradnorm,
        component_parent_hashes={
            "frozen_offline_fusion": _digest("fusion"),
            "frozen_offline_expert_heads": _digest("heads"),
            "offline_target_cache": _digest("targets"),
            "selected_predictor_seed_artifacts": _digest("predictors"),
            "target_normalizer_set": _digest("normalizers"),
        },
        source_snapshot=SOURCE,
    )
    _, loaded, loaded_objectives, loaded_gradnorm = (
        load_joint_graph_template(
            tmp_path / "graph" / "joint_graph_template.json",
            expected_variant="J3_INDEPENDENT_PLUS_ADAPTER",
            expected_run_record_sha256=SHA_A,
            expected_predictor_bundle_lock_sha256=SHA_B,
            expected_source=SOURCE,
        )
    )
    assert loaded.variant == graph.variant
    assert loaded_objectives == objectives
    assert loaded_gradnorm == gradnorm
    assert graph_manifest["input_policy"] == "R_MULTI"

    dataset = _dataset("val_stop")
    dataset_manifest = publish_joint_dataset_cache(
        output_dir=tmp_path / "dataset",
        dataset=dataset,
        parent_hashes={
            "identity_manifest": _digest("identity"),
            "HLT_view_cache": _digest("view"),
            "offline_target_cache": _digest("targets"),
            "target_normalizer_set": _digest("normalizers"),
        },
        source_snapshot=SOURCE,
    )
    _, loaded_dataset = load_joint_dataset_cache(
        tmp_path / "dataset" / "joint_dataset.json",
        expected_split="val_stop",
        expected_source=SOURCE,
    )
    assert loaded_dataset.identities == dataset.identities
    assert dataset_manifest["replica_ids"] == [0]


def test_j5_materialization_requires_locked_j4_initialization() -> None:
    names = (
        "model_train_identity_manifest",
        "val_stop_identity_manifest",
        "val_design_identity_manifest",
        "val_design_label_manifest",
        "model_train_R_MULTI_view_cache",
        "val_stop_R_MULTI_view_cache",
        "val_design_fixed_view_cache",
        "offline_target_cache",
        "target_normalizer_set",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
        "selected_predictor_seed_artifacts",
        "selected_HLT_expert_seed_artifacts",
        "j4_block_selection",
        "selected_J4_bridge_initialization",
    )
    parents = {name: _digest(name) for name in names}
    run = materialize_stage_j_run(
        run_id="RETB_J5_END_TO_END_S101",
        variant="J5_END_TO_END",
        pipeline_seed=101,
        final_particle_blocks=None,
        predictor_bundle_lock_sha256=SHA_A,
        step11_bundle_sha256=SHA_B,
        parent_hashes=parents,
        semantic_label=SEMANTIC_LABELS["J5_END_TO_END"],
    )
    assert validate_materialized_stage_j_run(run) == run["content_hash"]
    with pytest.raises(ValueError, match="semantics differ"):
        materialize_stage_j_run(
            run_id="RETB_J5_END_TO_END_S101",
            variant="J5_END_TO_END",
            pipeline_seed=101,
            final_particle_blocks=None,
            predictor_bundle_lock_sha256=SHA_A,
            step11_bundle_sha256=SHA_B,
            parent_hashes={
                name: value
                for name, value in parents.items()
                if name != "j4_block_selection"
            },
            semantic_label=SEMANTIC_LABELS["J5_END_TO_END"],
        )


def test_step11_production_entrypoints_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "scripts/build_retb_step11_contracts.py": "build_step11_bundle",
        "scripts/materialize_retb_stage_j_run.py": (
            "materialize_stage_j_run"
        ),
        "scripts/train_retb_joint_bridge.py": "train_joint_bridge",
        "scripts/register_retb_joint_graph_template.py": (
            "publish_joint_graph_template"
        ),
        "scripts/register_retb_joint_dataset_cache.py": (
            "publish_joint_dataset_cache"
        ),
        "scripts/evaluate_retb_j0_independent.py": (
            "evaluate_joint_bridge"
        ),
        "scripts/select_retb_j4_blocks.py": "select_j4_block_count",
        "sbatch/run_retb_build_step11_contracts.sh": (
            "build_retb_step11_contracts.py"
        ),
        "sbatch/run_retb_materialize_stage_j_run.sh": (
            "materialize_retb_stage_j_run.py"
        ),
        "sbatch/run_retb_train_joint_bridge.sh": (
            "train_retb_joint_bridge.py"
        ),
        "sbatch/run_retb_register_joint_graph_template.sh": (
            "register_retb_joint_graph_template.py"
        ),
        "sbatch/run_retb_register_joint_dataset_cache.sh": (
            "register_retb_joint_dataset_cache.py"
        ),
        "sbatch/run_retb_evaluate_j0_independent.sh": (
            "evaluate_retb_j0_independent.py"
        ),
        "sbatch/run_retb_select_j4_blocks.sh": (
            "select_retb_j4_blocks.py"
        ),
    }
    for relative, needle in expected.items():
        path = root / relative
        assert path.is_file()
        assert needle in path.read_text("utf-8")
