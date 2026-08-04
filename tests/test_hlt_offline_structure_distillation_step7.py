from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    BoundedFeaturewiseConditioning,
    AuxiliaryHBaseClassifier,
    FeedbackHBaseClassifier,
    FeedbackInterventionDataset,
    HBaseParticleTransformer,
    HOSDTrainingProtocol,
    PredictedPairAttentionBias,
    ResidualStructureTokenAdapter,
    build_feedback_selection,
    build_stage_e_loader_manifest,
    build_stage_e_plan,
    build_stage_d_loader_manifest,
    data_order_seed,
    feedback_interface_contract,
    gate_warmup_updates,
    global_auxiliary_loss,
    global_feedback_layout,
    pack_global_feedback,
    initialize_feedback_from_auxiliary_checkpoint,
    load_stage_d_loaders_from_manifest,
    evaluate_posthoc_feedback_control,
    feedback_model_flop_ledger,
    train_stage_e_feedback,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    AUXILIARY_CHECKPOINT_CONTRACT,
    AUXILIARY_COMPLETION_CONTRACT,
    AUXILIARY_PREDICTION_CONTRACT,
    CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT,
    CONFIRMATION_TRAINING_COMPLETION_CONTRACT,
    CONFIRMATION_TRAINING_PREDICTION_CONTRACT,
    FEEDBACK_RESULT_CONTRACT,
    SCALE_TRAINING_CHECKPOINT_CONTRACT,
    SCALE_TRAINING_COMPLETION_CONTRACT,
    SCALE_TRAINING_PREDICTION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    validate_content_hash,
    write_immutable_json,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_training import (
    evaluate_auxiliary,
    train_stage_d_auxiliary,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_data_factory import (
    _scale_labels,
)
from teacher_logit_reco.hlt_offline_structure_distillation.auxiliary_data import (
    HLTArrayDataset,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}


def test_scale_dataset_retains_mmap_like_identities_and_lazy_positions() -> None:
    count = 50_000
    identities = np.asarray([f"jet-{index:06d}" for index in range(count)])
    tokens = np.zeros((count, 1, 4), dtype=np.float32)
    mask = np.ones((count, 1), dtype=bool)
    states = np.zeros((count, 1, 4), dtype=np.int8)
    replicas = {
        replica: {
            "tokens": tokens,
            "mask": mask,
            "measurement_states": states,
        }
        for replica in range(4)
    }
    dataset = HLTArrayDataset(
        replica_arrays=replicas,
        labels=np.arange(count, dtype=np.int64) % 10,
        identities=identities,
        logical_role="scale_train",
        realization_policy="R_MULTI",
    )
    assert dataset.identities is identities
    assert all(
        isinstance(indices, range)
        for indices in dataset.source_indices_by_replica.values()
    )
    boundaries = dataset.locality_boundaries()
    assert boundaries[0] == 0 and boundaries[-1] == count
    assert max(
        right - left for left, right in zip(boundaries, boundaries[1:])
    ) <= 2_048


def test_design_subroles_use_the_authenticated_val_design_replica_contract() -> None:
    arrays = {
        0: {
            "tokens": np.zeros((2, 1, 4), dtype=np.float32),
            "mask": np.ones((2, 1), dtype=bool),
            "measurement_states": np.zeros((2, 1, 4), dtype=np.int8),
        }
    }
    for role in ("design_select", "design_confirm"):
        dataset = HLTArrayDataset(
            replica_arrays=arrays,
            labels=np.asarray([0, 1]),
            identities=("jet-a", "jet-b"),
            logical_role=role,
            realization_policy="R_FIXED",
        )
        assert dataset.replica_for_index(0) == 0
        assert dataset.replica_for_index(1) == 0


def test_scale_label_validation_does_not_iterate_authenticated_input_identities(
    tmp_path,
) -> None:
    identities = np.asarray(["jet-a", "jet-b", "jet-c"])
    labels_path = tmp_path / "labels.npz"
    np.savez(labels_path, identities=identities, labels=np.asarray([0, 1, 2]))

    class NoIteration:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return identities[index]

        def __iter__(self):
            raise AssertionError("authenticated input identities were rescanned")

        def __array__(self, *args, **kwargs):
            raise AssertionError("authenticated input identities were converted")

    from teacher_logit_reco.hlt_offline_structure_distillation import (
        identity_order_sha256,
    )

    wrapped, labels, _ = _scale_labels(
        labels_path,
        NoIteration(),
        identity_order_sha256(identities),
    )
    assert wrapped.identity_order_sha256 == identity_order_sha256(identities)
    assert np.array_equal(labels, [0, 1, 2])


class _PairEmbed(torch.nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.pairwise_lv_dim = 4
        self.pairwise_input_dim = 0
        self.is_symmetric = True
        self.out_dim = heads
        self.remove_self_pair = False
        self.sparse_eval = (False, False)
        self.embed = torch.nn.Conv1d(4, heads, 1)

    def forward(self, v, uu=None, mask=None):
        del v, mask
        batch, _, length, _ = uu.shape
        return self.embed(uu.reshape(batch, 4, -1)).reshape(
            batch, -1, length, length
        )


class _Block(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.observed_attention_masks = []
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(dimension)

    def forward(self, x, padding_mask=None, attn_mask=None):
        self.observed_attention_masks.append(
            None if attn_mask is None else attn_mask.detach().clone()
        )
        if attn_mask is not None:
            attn_mask = attn_mask.flatten(0, 1)
        update, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.norm(x + update)


class _ClassBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )

    def forward(self, x, x_cls=None, padding_mask=None):
        context = torch.cat((x_cls, x), dim=1)
        prefix = torch.zeros(
            padding_mask.shape[0], 1, dtype=torch.bool, device=x.device
        )
        value, _ = self.attn(
            x_cls,
            context,
            context,
            key_padding_mask=torch.cat((prefix, padding_mask), dim=1),
            need_weights=False,
        )
        return x_cls + value


class _Transformer(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        dimension, heads = 16, int(config["num_heads"])
        self.pair_extra_dim = int(config.get("pair_extra_dim", 0))
        self.use_amp = False
        self.pair_embed = _PairEmbed(heads)
        self.input = torch.nn.Linear(17, dimension)
        self.blocks = torch.nn.ModuleList(
            [_Block(dimension, heads) for _ in range(8)]
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dimension))
        self.cls_blocks = torch.nn.ModuleList(
            [_ClassBlock(dimension, heads) for _ in range(2)]
        )
        self.norm = torch.nn.LayerNorm(dimension)
        self.fc = torch.nn.Linear(dimension, 10)

    def embed(self, features):
        return self.input(features.transpose(1, 2))

    def forward(self, features, v=None, mask=None, uu=None):
        trimmer = getattr(self, "trimmer", None)
        if trimmer is not None:
            features, v, mask, uu = trimmer(features, v, mask, uu)
        x = self.embed(features)
        padding = ~mask[:, 0].bool()
        bias = self.pair_embed(v, uu=uu, mask=mask)
        for block in self.blocks:
            x = block(x, padding_mask=padding, attn_mask=bias)
        cls = self.cls_token.expand(x.shape[0], 1, -1)
        for block in self.cls_blocks:
            cls = block(x, x_cls=cls, padding_mask=padding)
        return self.fc(self.norm(cls).squeeze(1))


class _DeterministicTrainingTrimmer(torch.nn.Module):
    """Official-Weaver-shaped valid permutation followed by width trimming."""

    def forward(self, x, v=None, mask=None, uu=None):
        length = int(mask.shape[-1])
        rank = torch.arange(length, device=mask.device).view(1, 1, -1)
        rank = rank.expand(mask.shape[0], 1, -1).masked_fill(~mask.bool(), -1)
        permutation = rank.argsort(dim=-1, descending=True)
        mask = mask.gather(-1, permutation)
        x = x.gather(-1, permutation.expand_as(x))
        if v is not None:
            v = v.gather(-1, permutation.expand_as(v))
        if uu is not None:
            uu = uu.gather(-2, permutation.unsqueeze(-1).expand_as(uu))
            uu = uu.gather(-1, permutation.unsqueeze(-2).expand_as(uu))
        maximum = int(mask.sum(dim=-1).max().item())
        return (
            x[..., :maximum],
            None if v is None else v[..., :maximum],
            mask[..., :maximum],
            None if uu is None else uu[..., :maximum, :maximum],
        )


def _weaver():
    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat([base + index for index in range(num_outputs)], dim=1)

    return SimpleNamespace(
        ParticleTransformer=lambda **config: _Transformer(**config),
        pairwise_lv_fts=pairwise,
    )


def _batch():
    torch.manual_seed(37)
    mask = torch.tensor(
        [[[True, True, True, False]], [[True, True, False, False]]]
    )
    return {
        "points": torch.zeros(2, 2, 4),
        "features": torch.randn(2, 17, 4).masked_fill(~mask, 0),
        "lorentz_vectors": torch.randn(2, 4, 4).masked_fill(~mask, 0),
        "mask": mask,
    }


def test_feedback_contract_and_exact_warmup_integer_rule():
    contract = feedback_interface_contract()
    assert contract["token"]["live_particle_sequence_expanded"] is False
    assert contract["pair"]["offline_pair_feedback_allowed"] is False
    assert gate_warmup_updates(1) == 1
    assert gate_warmup_updates(2) == 1
    assert gate_warmup_updates(20) == 1
    assert gate_warmup_updates(21) == 2
    with pytest.raises(ValueError):
        gate_warmup_updates(0)


def test_token_and_film_zero_initialization_are_exact_hbase_noops():
    batch = _batch()
    for interface in ("FB_TOKEN", "FB_FILM"):
        torch.manual_seed(41)
        classifier = HBaseParticleTransformer(weaver_module=_weaver()).eval()
        expected = classifier(**batch)
        model = FeedbackHBaseClassifier(
            classifier,
            target_id="T_OFFLINE_TRACK_32",
            interface=interface,
            particle_dimension=16,
        ).eval()
        actual, prediction = model.forward_with_feedback(**batch)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert prediction["value"].shape == (2, 32)
    token = ResidualStructureTokenAdapter(3, particle_dimension=16)
    assert float(token.gamma) == 0.0
    assert token.structure_tokens({
        "value": torch.zeros(2, 3),
        "availability_logits": torch.zeros(2, 1),
    }).shape == (
        2,
        4,
        128,
    )
    film = BoundedFeaturewiseConditioning(3, particle_dimension=16)
    prediction = {
        "value": torch.randn(5, 3),
        "availability_logits": torch.zeros(5, 1),
    }
    scale, shift = film.parameters_for(prediction)
    torch.testing.assert_close(scale, torch.ones_like(scale), atol=0, rtol=0)
    torch.testing.assert_close(shift, torch.zeros_like(shift), atol=0, rtol=0)
    with torch.no_grad():
        film.projection.weight.fill_(100)
    scale, shift = film.parameters_for({
        "value": torch.ones(5, 3),
        "availability_logits": torch.zeros(5, 1),
    })
    assert bool(((scale >= 0.9) & (scale <= 1.1)).all())
    assert bool(((shift >= -0.1) & (shift <= 0.1)).all())


def test_pair_bias_is_warmup_zero_then_symmetric_masked_and_diagonal_zero():
    module = PredictedPairAttentionBias(
        input_dimension=16,
        pair_dimension=8,
        attention_heads=4,
        symmetric=True,
    )
    states = torch.randn(2, 4, 16)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    with torch.no_grad():
        module.raw_alpha.fill_(1)
    module.set_update(1, 20)
    warmup, _ = module(states, mask)
    torch.testing.assert_close(warmup, torch.zeros_like(warmup), atol=0, rtol=0)
    module.set_update(2, 20)
    active, prediction = module(states, mask)
    torch.testing.assert_close(active, active.transpose(2, 3))
    diagonal = active.diagonal(dim1=2, dim2=3)
    torch.testing.assert_close(diagonal, torch.zeros_like(diagonal), atol=0, rtol=0)
    assert bool((active[0, :, 3] == 0).all())
    assert prediction["value"].shape == (2, 4, 4, 8)


def test_later_feedback_hooks_reach_every_block_and_change_logits():
    batch = _batch()
    classifier = HBaseParticleTransformer(weaver_module=_weaver()).eval()
    expected = classifier(**batch)
    calls = []

    def transform(block, state, active):
        calls.append(block)
        return state + active.unsqueeze(-1).to(state.dtype) * 0.25

    result = classifier.forward_with_taps(
        **batch,
        capture=("TAP_MID",),
        later_block_transform=transform,
    )
    assert calls == [5, 6, 7, 8]
    assert not torch.allclose(result.logits, expected)

    pair_calls = []

    def bias(state, active, particle_indices):
        del state
        pair_calls.append(True)
        assert torch.equal(
            particle_indices,
            torch.arange(active.shape[1]).view(1, -1).expand(active.shape[0], -1),
        )
        valid = active.unsqueeze(1) & active.unsqueeze(2)
        return valid.unsqueeze(1).expand(-1, 8, -1, -1).float() * 0.5

    pair_classifier = HBaseParticleTransformer(weaver_module=_weaver()).eval()
    paired = pair_classifier.forward_with_taps(
        **batch,
        capture=("TAP_MID",),
        later_pair_bias=bias,
    )
    assert pair_calls == [True]
    pair_expected = HBaseParticleTransformer(weaver_module=_weaver()).eval()
    pair_expected.load_state_dict(pair_classifier.state_dict())
    ordinary = pair_expected(**batch)
    assert not torch.allclose(paired.logits, ordinary)
    blocks = pair_classifier.mod.blocks
    reference_mask = blocks[3].observed_attention_masks[-1]
    assert all(
        not torch.equal(block.observed_attention_masks[-1], reference_mask)
        for block in blocks[4:8]
    )


def test_pair_feedback_receives_official_trimmer_particle_correspondence():
    batch = _batch()
    classifier = HBaseParticleTransformer(weaver_module=_weaver()).train()
    classifier.mod.trimmer = _DeterministicTrainingTrimmer()
    observed = []

    def bias(state, active, particle_indices):
        del state
        observed.append(particle_indices.detach().clone())
        valid = active.unsqueeze(1) & active.unsqueeze(2)
        return valid.unsqueeze(1).expand(-1, 8, -1, -1).float()

    result = classifier.forward_with_taps(
        **batch,
        capture=("TAP_MID",),
        later_pair_bias=bias,
    )
    assert result.logits.shape == (2, 10)
    assert len(observed) == 1
    assert torch.equal(
        observed[0],
        torch.tensor([[2, 1, 0], [1, 0, 2]]),
    )
    # The temporary trace wrapper must not remain installed after the forward.
    assert "traced_trimmer" not in classifier.mod.trimmer.forward.__qualname__


def test_feedback_packing_gates_values_and_keeps_availability_and_het():
    layout = global_feedback_layout("T_OFFLINE_TRACK_32", "HET")
    values = torch.ones(2, 32)
    availability_logits = torch.full(
        (2, len(layout["availability_group_order"])), -20.0
    )
    availability_logits[:, 0] = 20.0
    packed = pack_global_feedback(
        {
            "mean": values,
            "value": values,
            "availability_logits": availability_logits,
            "log_variance": torch.full_like(values, 2.0),
        },
        component_to_availability_index=torch.tensor(
            layout["component_to_availability_index"]
        ),
        heteroscedastic_component_mask=torch.tensor(
            layout["heteroscedastic_component_mask"]
        ),
    )
    assert packed.shape == (2, layout["packed_dimension"])
    probabilities = torch.sigmoid(availability_logits)
    indices = torch.tensor(layout["component_to_availability_index"])
    torch.testing.assert_close(packed[:, :32], probabilities[:, indices])
    torch.testing.assert_close(
        packed[:, 32 : 32 + probabilities.shape[1]], probabilities
    )


def test_feedback_initialization_loads_locked_classifier_and_target_head(tmp_path):
    auxiliary = AuxiliaryHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        target_dimension=32,
        input_dimension=16,
        availability_group_count=2,
    )
    with torch.no_grad():
        for ordinal, parameter in enumerate(auxiliary.parameters(), start=1):
            parameter.fill_(ordinal / 1000.0)
    checkpoint_path = tmp_path / "best_model_val.pt"
    checkpoint = {
        "contract": AUXILIARY_CHECKPOINT_CONTRACT,
        "row_id": "selected-A-t",
        "stage_d_plan_sha256": "d" * 64,
        "campaign_spec_sha256": "c" * 64,
        "source": SOURCE,
        "model_state_dict": auxiliary.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = __import__("hashlib").sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    result = with_content_hash(
        {
            "contract": AUXILIARY_PREDICTION_CONTRACT,
            "schema_version": 2,
            "row_id": "selected-A-t",
            "target_id": "T_OFFLINE_TRACK_32",
            "parameterization": "ABS",
            "auxiliary_weight": 0.3,
            "checkpoint_sha256": checkpoint_sha,
            "stage_d_plan_sha256": "d" * 64,
            "campaign_spec_sha256": "c" * 64,
            "source": SOURCE,
        }
    )
    completion = with_content_hash(
        {
            "contract": AUXILIARY_COMPLETION_CONTRACT,
            "schema_version": 2,
            "row_id": "selected-A-t",
            "checkpoint_sha256": checkpoint_sha,
            "stage_d_plan_sha256": "d" * 64,
            "campaign_spec_sha256": "c" * 64,
            "source": SOURCE,
        }
    )
    row = {
        "selected_auxiliary_row_id": "selected-A-t",
        "selected_auxiliary_result_sha256": result["content_hash"],
        "selected_auxiliary_parameterization": "ABS",
        "selected_auxiliary_weight": 0.3,
        "target_id": "T_OFFLINE_TRACK_32",
    }
    feedback = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id=row["target_id"],
        interface="FB_TOKEN",
        particle_dimension=16,
    )
    lineage = initialize_feedback_from_auxiliary_checkpoint(
        feedback,
        row,
        checkpoint_path=checkpoint_path,
        completion=completion,
        result=result,
        stage_d_plan_sha256="d" * 64,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    assert lineage["selected_auxiliary_checkpoint"] == checkpoint_sha
    for key, value in auxiliary.classifier.state_dict().items():
        torch.testing.assert_close(feedback.classifier.state_dict()[key], value)
    for key, value in auxiliary.target_head.state_dict().items():
        torch.testing.assert_close(feedback.global_predictor.state_dict()[key], value)


def test_track_pair_is_directed_and_region_pair_is_exactly_symmetric():
    states = torch.randn(2, 4, 16)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    track = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_HLT_TRACK_PAIR_13",
        interface="FB_PAIR",
        particle_dimension=16,
    ).consumer
    region = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_HLT_REGION_PAIR_8",
        interface="FB_PAIR",
        particle_dimension=16,
    ).consumer
    assert track.symmetric is False
    assert region.symmetric is True
    _, track_prediction = track(states, mask)
    _, region_prediction = region(states, mask)
    assert not torch.allclose(
        track_prediction["value"], track_prediction["value"].transpose(1, 2)
    )
    torch.testing.assert_close(
        region_prediction["value"],
        region_prediction["value"].transpose(1, 2),
    )


def test_detached_feedback_blocks_consumer_gradient_but_target_path_survives():
    model = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_TOKEN",
        gradient_path="DETACHED",
        particle_dimension=16,
    )
    with torch.no_grad():
        model.consumer.raw_gamma.fill_(0.5)
    logits, prediction = model.forward_with_feedback(**_batch())
    logits.sum().backward(retain_graph=True)
    assert all(
        parameter.grad is None or bool((parameter.grad == 0).all())
        for parameter in model.global_predictor.parameters()
    )
    model.zero_grad(set_to_none=True)
    prediction["value"].sum().backward()
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.global_predictor.parameters()
    )
    with pytest.raises(ValueError, match="oracle"):
        model.forward_with_feedback(
            **_batch(), oracle_feedback={"value": torch.zeros(2, 32)}
        )


class _FeedbackTrainingDataset(torch.utils.data.Dataset):
    def __init__(self):
        batch = _batch()
        self.rows = []
        for index in range(10):
            source_index = index % 2
            self.rows.append(
                {
                    key: value[source_index]
                    for key, value in batch.items()
                }
                | {
                    "labels": torch.tensor(index),
                    "identities": f"feedback-jet-{index}",
                    "target": torch.zeros(32),
                    "target_mask": torch.ones(32, dtype=torch.bool),
                }
            )
        self.control_kind = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _feedback_collate(rows):
    return {
        key: (
            [row[key] for row in rows]
            if key == "identities"
            else torch.stack([row[key] for row in rows])
        )
        for key in rows[0]
    }


@pytest.mark.parametrize("training_path", ["stage_e", "confirmation", "scale"])
def test_feedback_training_executes_real_batches_on_every_training_path(
    tmp_path, training_path
):
    row = {
        "row_id": f"feedback-{training_path}",
        "target_id": "T_OFFLINE_TRACK_32",
        "interface": "FB_TOKEN",
        "gradient_path": "END_TO_END",
        "parameterization": "ABS",
        "auxiliary_weight": 0.3,
        "row_kind": "SCIENTIFIC",
        "control": None,
        "pipeline_seed": 101,
        "encoder_component_seed": 102,
        "feedback_component_seed": 103,
        "resolved": True,
        "selection_eligible": True,
        "deployable": True,
    }
    model = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id=row["target_id"],
        interface=row["interface"],
        particle_dimension=16,
    )
    loader = torch.utils.data.DataLoader(
        _FeedbackTrainingDataset(),
        batch_size=10,
        shuffle=False,
        collate_fn=_feedback_collate,
    )
    common = dict(
        model=model,
        train_loader=loader,
        val_stop_loader=loader,
        output_dir=tmp_path / training_path,
        row=row,
        component_group_ids=(
            *("track_availability_observation" for _ in range(4)),
            *("has_valid_track" for _ in range(28)),
        ),
        campaign_spec_sha256="c" * 64,
        lineage_hashes={"input": "d" * 64},
        protocol=HOSDTrainingProtocol(
            maximum_epochs=2, campaign_profile="miniature_test"
        ),
        source=SOURCE,
        deployed_analytical_flops=10.0,
        device="cpu",
        training_gpu_hours_override=0.01,
    )
    if training_path == "stage_e":
        completion = train_stage_e_feedback(
            **common,
            design_select_loader=loader,
            stage_e_plan_sha256="e" * 64,
        )
    else:
        contracts = (
            (
                CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT,
                CONFIRMATION_TRAINING_COMPLETION_CONTRACT,
                CONFIRMATION_TRAINING_PREDICTION_CONTRACT,
                "confirmation_plan_sha256",
            )
            if training_path == "confirmation"
            else (
                SCALE_TRAINING_CHECKPOINT_CONTRACT,
                SCALE_TRAINING_COMPLETION_CONTRACT,
                SCALE_TRAINING_PREDICTION_CONTRACT,
                "scale_execution_plan_sha256",
            )
        )
        completion = train_stage_d_auxiliary(
            **common,
            design_select_loader=loader,
            stage_d_plan_sha256="e" * 64,
            checkpoint_contract=contracts[0],
            completion_contract=contracts[1],
            prediction_contract=contracts[2],
            plan_hash_field=contracts[3],
            completion_filename="training_completion.json",
        )
    assert completion["epochs_completed"] == 2
    assert completion["task_component_seed"] == 103


@pytest.mark.parametrize(
    ("training_role", "evaluation_role"),
    [
        ("model_train", "design_select"),
        ("model_train", "design_confirm"),
        ("scale_train", "design_confirm"),
    ],
)
def test_feedback_production_manifest_loader_uses_graph_independent_data_order(
    tmp_path, monkeypatch, training_role, evaluation_role
):
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        stage_d_data_factory,
    )

    identities = tuple(f"event-{index}" for index in range(10))
    labels = np.arange(10, dtype=np.int64)

    monkeypatch.setattr(
        stage_d_data_factory,
        "_labels",
        lambda path: (identities, labels, "1" * 64),
    )

    def fake_hlt(caches, requested):
        role = str(next(iter(caches.values()))).split("/")[-2]
        replicas = {int(key) for key in caches}
        arrays = {
            replica: {
                "tokens": np.zeros((10, 128, 14), dtype=np.float32),
                "mask": np.ones((10, 128), dtype=bool),
                "measurement_states": np.zeros((10, 128, 14), dtype=np.int8),
            }
            for replica in replicas
        }
        return (
            arrays,
            {replica: np.arange(10, dtype=np.int64) for replica in replicas},
            {f"hlt_replica_{replica}": f"{replica + 2:x}" * 64 for replica in replicas},
            role,
            "R_MULTI" if len(replicas) == 4 else "R_FIXED",
        )

    monkeypatch.setattr(stage_d_data_factory, "_load_hlt", fake_hlt)
    monkeypatch.setattr(
        stage_d_data_factory,
        "_static_targets",
        lambda **kwargs: (
            np.zeros((10, 32), dtype=np.float32),
            np.ones((10, 32), dtype=bool),
            {"target_cache": "f" * 64},
        ),
    )
    row = {
        "row_id": f"feedback-{training_role}-{evaluation_role}",
        "target_id": "T_OFFLINE_TRACK_32",
        "parameterization": "ABS",
        "row_kind": "SCIENTIFIC",
        "pipeline_seed": 101,
        "encoder_component_seed": 102,
        # Deliberately no head_component_seed: this is the Stage-E shape that
        # previously crashed the production manifest loader.
        "feedback_component_seed": 987654,
        "resolved": True,
    }
    roles = {}
    for role in (training_role, "val_stop", evaluation_role):
        replica_keys = range(4) if role == training_role else range(1)
        roles[role] = {
            "labels": f"/labels/{role}.npz",
            "hlt_caches": {
                str(replica): f"/cache/{role}/replica_{replica}"
                for replica in replica_keys
            },
            "target": {
                "mode": "static_cache",
                "caches": {"shared": f"/target/{role}"},
            },
        }
    manifest = build_stage_d_loader_manifest(
        row=row,
        role_definitions=roles,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
        training_role=training_role,
        evaluation_role=evaluation_role,
    )
    path = tmp_path / "loader.json"
    write_immutable_json(path, manifest)
    loaded = load_stage_d_loaders_from_manifest(
        manifest_path=path,
        campaign_root=tmp_path,
        row=row,
        campaign={"source": SOURCE, "content_hash": "c" * 64},
        target_registry={},
    )
    expected = data_order_seed(101, training_role)
    assert manifest["sampler_seed_by_role"][training_role] == expected
    assert loaded["train_loader"].sampler.seed == expected
    assert loaded["data_order_contract"] == "hosd_data_order_v2"
    assert loaded["sampler_contract_by_role"][training_role] == (
        "hosd_scale_shard_aware_sampler_v1"
        if training_role == "scale_train"
        else "retb_deterministic_full_permutation_sampler_v1"
    )


def test_unrestricted_feedback_is_direct_token_and_exact_capacity_matched():
    semantic = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_TOKEN",
        particle_dimension=16,
    )
    unrestricted = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_TOKEN",
        particle_dimension=16,
        control="UNRESTRICTED",
    )
    semantic_count = sum(value.numel() for value in semantic.head_parameters())
    unrestricted_count = sum(
        value.numel() for value in unrestricted.head_parameters()
    )
    assert unrestricted_count == semantic_count
    validate_content_hash(unrestricted.capacity_ledger)
    assert unrestricted.capacity_ledger["contract"] == (
        "hosd_unrestricted_feedback_capacity_ledger_v1"
    )
    assert unrestricted.capacity_ledger["target_id"] == "T_OFFLINE_TRACK_32"
    assert unrestricted.capacity_ledger["interface"] == "FB_TOKEN"
    assert unrestricted.capacity_ledger["reference_trainable_parameters"] == semantic_count
    assert unrestricted.capacity_ledger[
        "unrestricted_pre_padding_trainable_parameters"
    ] == semantic_count - unrestricted.capacity_padding.count
    assert unrestricted.capacity_ledger[
        "inert_trainable_padding_parameters"
    ] == unrestricted.capacity_padding.count
    assert unrestricted.capacity_ledger["matched_trainable_parameters"] == semantic_count
    logits, prediction = unrestricted.forward_with_feedback(**_batch())
    assert logits.shape == (2, 10)
    assert prediction["tokens"].shape == (2, 4, 128)
    layout = global_feedback_layout("T_OFFLINE_TRACK_32", "ABS")
    assert prediction["availability_logits"].shape == (
        2,
        len(layout["availability_group_order"]),
    )
    unrestricted_loss, unrestricted_pieces = global_auxiliary_loss(
        prediction,
        torch.zeros(2, 32),
        torch.ones(2, 32, dtype=torch.bool),
        parameterization="ABS",
        component_group_ids=layout["component_group_ids"],
        target_id="T_OFFLINE_TRACK_32",
    )
    assert torch.isfinite(unrestricted_loss)
    assert unrestricted_pieces["availability_group_order"] == (
        "track_availability_observation",
        "has_valid_track",
    )
    plan = build_stage_e_plan(
        single_family_selection=_single_family_lock(),
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    rows = [row for row in plan["control_rows"] if row["control"] == "UNRESTRICTED"]
    assert rows and all(row["auxiliary_weight"] == 0.0 for row in rows)
    mean_only_rows = [
        row for row in plan["control_rows"] if row["control"] == "MEAN_ONLY"
    ]
    assert mean_only_rows and all(
        row["parameterization"] == "HET" for row in mean_only_rows
    )


@pytest.mark.parametrize("interface", ["FB_TOKEN", "FB_FILM"])
def test_mean_only_is_exactly_parameter_matched_to_het(interface):
    heteroscedastic = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface=interface,
        parameterization="HET",
        particle_dimension=16,
    )
    mean_only = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface=interface,
        parameterization="HET",
        particle_dimension=16,
        control="MEAN_ONLY",
    )
    assert sum(value.numel() for value in mean_only.head_parameters()) == sum(
        value.numel() for value in heteroscedastic.head_parameters()
    )
    validate_content_hash(mean_only.capacity_ledger)
    assert mean_only.capacity_ledger["contract"] == (
        "hosd_mean_only_capacity_ledger_v1"
    )
    assert mean_only.capacity_ledger["reference_control"] == "HET"


def test_exact_hlt_builder_has_separate_hashed_operation_ledger():
    exact = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_HLT_REGION_PAIR_8",
        interface="FB_PAIR",
        particle_dimension=16,
        control="EXACT_HLT",
    )
    profile = feedback_model_flop_ledger(exact)["exact_hlt_builder_profile"]
    validate_content_hash(profile)
    assert profile["normalization_cost_included"]
    assert profile["measured_timing_evidence_required_before_production"]
    assert "reuse_authenticated_same_event_tree" in profile["tree_reuse_policy"]
    assert profile["operation_counts"]["ca_merge_operations"] == 127


def test_unrestricted_film_is_direct_latent_and_exact_capacity_matched():
    semantic = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_FILM",
        particle_dimension=16,
    )
    unrestricted = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_FILM",
        particle_dimension=16,
        control="UNRESTRICTED",
    )
    assert sum(value.numel() for value in unrestricted.head_parameters()) == sum(
        value.numel() for value in semantic.head_parameters()
    )
    assert unrestricted.capacity_ledger["inert_trainable_padding_parameters"] >= 0
    logits, prediction = unrestricted.forward_with_feedback(**_batch())
    assert logits.shape == (2, 10)
    assert prediction["tokens"].shape == (2, 4, 128)
    assert prediction["value"].shape == (2, 32)


def test_wrong_event_posthoc_contract_is_deployable_but_oracle_is_not(
    tmp_path, monkeypatch
):
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        stage_e_training,
    )

    model = torch.nn.Linear(2, 2)
    model.allow_oracle = False
    checkpoint = tmp_path / "best_model_val.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    checkpoint_sha = __import__("hashlib").sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        stage_e_training,
        "evaluate_auxiliary",
        lambda *args, **kwargs: {
            "classification_metrics": {"balanced_accuracy": 0.1},
            "auxiliary_loss": 1.0,
            "event_count": 2,
        },
    )
    monkeypatch.setattr(
        stage_e_training,
        "feedback_model_flop_ledger",
        lambda value: {"deployed_total_flops": 10},
    )
    source_row = {
        "row_id": "source",
        "encoder_component_seed": 11,
        "feedback_component_seed": 12,
    }
    source_completion = {
        "row_id": "source",
        "checkpoint_sha256": checkpoint_sha,
        "stage_e_plan_sha256": "a" * 64,
        "source": SOURCE,
        "content_hash": "b" * 64,
        "selected_epoch": 1,
        "selected_val_stop": {"balanced_accuracy": 0.1},
    }

    def run(control):
        return evaluate_posthoc_feedback_control(
            source_model=model,
            design_select_loader=[],
            output_dir=tmp_path / control,
            control_row={
                "row_id": control,
                "target_id": "T_OFFLINE_TRACK_32",
                "parameterization": "ABS",
                "auxiliary_weight": 0.3,
                "row_kind": "CONTROL",
                "interface": "FB_TOKEN",
                "gradient_path": "END_TO_END",
                "control": control,
                "pipeline_seed": 101,
            },
            source_row=source_row,
            component_group_ids=("track",),
            source_checkpoint_path=checkpoint,
            source_completion=source_completion,
            stage_e_plan_sha256="a" * 64,
            campaign_spec_sha256="c" * 64,
            lineage_hashes={"loader": "d" * 64},
            source=SOURCE,
            device="cpu",
        )

    shuffled = run("SHUFFLED_PREDICTION")
    assert shuffled["deployable"] is True
    assert shuffled["schema_version"] == 4
    oracle = run("ORACLE_SUB")
    assert oracle["deployable"] is False


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_auxiliary_evaluation_moves_model_and_batch_to_declared_device(
    monkeypatch, device_type
):
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        stage_d_training,
    )

    model = torch.nn.Linear(1, 1)
    observed = {}

    def fake_loss_for_batch(*, model, batch, **kwargs):
        del kwargs
        observed["model_device"] = next(model.parameters()).device.type
        observed["batch_device"] = batch["labels"].device.type
        logits = torch.zeros(
            int(batch["labels"].numel()), 10, device=batch["labels"].device
        )
        return (
            logits.sum(),
            {"auxiliary_loss": logits.sum()},
            logits,
        )

    monkeypatch.setattr(stage_d_training, "_loss_for_batch", fake_loss_for_batch)
    result = evaluate_auxiliary(
        model,
        [{"labels": torch.tensor([0, 1], dtype=torch.long)}],
        row={"target_id": "T_OFFLINE_TRACK_32"},
        component_group_ids=("track",),
        split="val_stop",
        device=device_type,
    )
    assert observed == {
        "model_device": device_type,
        "batch_device": device_type,
    }
    assert result["event_count"] == 2


def _single_family_lock():
    target_ids = (
        "T_OFFLINE_JET_10",
        "T_OFFLINE_COMPOSITION_16",
        "T_OFFLINE_TRACK_32",
        "T_OFFLINE_DENSITY_22",
        "T_OFFLINE_CA_TREE_26",
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    )
    selected = {target_id: f"row-{index}" for index, target_id in enumerate(target_ids)}
    definitions = {
        target_id: {
            "target_id": target_id,
            "parameterization": "ABS",
            "auxiliary_weight": 0.3,
            "head_type": "pair" if target_id.startswith("T_HLT_") else "global",
        }
        for target_id in target_ids
    }
    return with_content_hash(
        {
            "contract": SINGLE_FAMILY_SELECTION_CONTRACT,
            "schema_version": 2,
            "source": SOURCE,
            "stage_d_plan_sha256": "d" * 64,
            "phase_lock_sha256": "e" * 64,
            "complete_result_hashes": {
                row_id: (f"{index:064x}"[-64:])
                for index, row_id in enumerate(selected.values(), start=1)
            },
            "selected_row_by_target": selected,
            "selected_definition_by_target": definitions,
            "cross_family_order": [
                {"ordinal": index, "target_id": target_id, "row_id": selected[target_id]}
                for index, target_id in enumerate(target_ids)
            ],
            "global_winner_row_id": selected[target_ids[0]],
            "negative_results_permitted": True,
            "performance_based_termination": False,
        }
    )


def test_stage_e_exact_bound_and_all_negative_selection_complete():
    plan = build_stage_e_plan(
        single_family_selection=_single_family_lock(),
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    assert plan["row_count"] == 44
    assert len(plan["scientific_rows"]) == 16
    assert len(plan["reference_rows"]) == 2
    assert len(plan["control_rows"]) == 26
    assert all(
        row["control"] == "EXACT_HLT"
        and row["selection_eligible"]
        and not row["semantic_loss_enabled"]
        for row in plan["reference_rows"]
    )
    for row in plan["reference_rows"]:
        reference = FeedbackHBaseClassifier(
            HBaseParticleTransformer(weaver_module=_weaver()),
            target_id=row["target_id"],
            interface="FB_PAIR",
            parameterization=row["parameterization"],
            particle_dimension=16,
            control="EXACT_HLT",
        )
        assert reference.consumer.predictor is None
    assert plan["promoted_global_targets"] == [
        "T_OFFLINE_JET_10",
        "T_OFFLINE_COMPOSITION_16",
    ]
    labels = np.tile(np.arange(10), 3)
    logits = np.zeros((30, 10), dtype=np.float64)
    metrics = evaluate_classification(logits, labels, split="design_select")
    perfect = evaluate_classification(
        np.eye(10, dtype=np.float64)[labels] * 20.0,
        labels,
        split="design_select",
    )
    results = [
        with_content_hash(
            {
                "contract": FEEDBACK_RESULT_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                **row,
                "stage_e_plan_sha256": plan["content_hash"],
                "design_select": {
                    "classification_metrics": (
                        perfect if row.get("control") == "EXACT_HLT" else metrics
                    )
                },
                "deployed_analytical_flops": 10.0,
                "deployed_parameter_count": 100,
                "training_gpu_hours": 1.0,
                "deployable": bool(row["deployable"]),
            }
        )
        for row in plan["all_rows"]
    ]
    with pytest.raises(ValueError, match="complete"):
        build_feedback_selection(
            stage_e_plan=plan, results=results[:-1], source=SOURCE
        )
    lock = build_feedback_selection(
        stage_e_plan=plan, results=results, source=SOURCE
    )
    assert lock["all_rows_completed"] is True
    assert lock["negative_gain_can_still_win"] is True
    assert lock["selected_feedback_row_id"]
    assert lock["selected_feedback_definition"].get("control") != "EXACT_HLT"
    assert set(lock["reference_graph_definitions"]) == {
        "REFERENCE_EXACT_TRACK",
        "REFERENCE_EXACT_REGION",
    }


def _exact_pair_normalizers(target_id: str, dimensions: int):
    normalizer = with_content_hash(
        {
            "contract": "hosd_streamed_target_normalizer_v1",
            "schema_version": 1,
            "targets": [
                {
                    "target_id": target_id,
                    "component_count": dimensions,
                    "components": [
                        {
                            "component_index": index,
                            "normalize": index >= 4,
                            "center": float(index) / 10.0,
                            "scale": 1.0 + float(index) / 20.0,
                        }
                        for index in range(dimensions)
                    ],
                }
            ],
            "source": SOURCE,
        }
    )
    relation = with_content_hash(
        {
            "contract": "test_relation_normalizer_v1",
            "schema_version": 1,
            "track_uncertainty_floors": {
                "d0": {"floor": 0.01},
                "dz": {"floor": 0.02},
            },
            "track_sentinel_policy": {
                "d0": None,
                "d0err": None,
                "dz": None,
                "dzerr": None,
            },
            "source": SOURCE,
        }
    )
    return normalizer, relation


def test_exact_hlt_reference_rebuilds_from_raw_inputs_and_survives_state_reload():
    batch = _batch()
    raw = torch.zeros(2, 4, 14)
    raw[:, :, 0] = torch.tensor([4.0, 3.0, 2.0, 0.0])
    raw[:, :, 5] = 1.0
    raw[:, :, 10] = torch.tensor([0.1, -0.3, 0.25, 0.0])
    raw[:, :, 11] = 0.03
    raw[:, :, 12] = torch.tensor([-0.2, 0.15, 0.4, 0.0])
    raw[:, :, 13] = 0.04
    normalizer, relation = _exact_pair_normalizers(
        "T_HLT_TRACK_PAIR_13", 13
    )
    model = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_HLT_TRACK_PAIR_13",
        interface="FB_PAIR",
        particle_dimension=16,
        control="EXACT_HLT",
    )
    model.configure_exact_hlt_runtime(
        target_normalizer=normalizer,
        relation_normalizer=relation,
    )
    model.consumer.raw_alpha.data.fill_(0.5)
    model.set_update(2, 2)
    logits, prediction = model.forward_with_feedback(
        **batch, raw_tokens=raw
    )
    assert logits.shape == (2, 10)
    assert prediction["value"].shape == (2, 4, 4, 13)
    assert prediction["pair_mask"].shape == (2, 4, 4)
    assert not bool(prediction["pair_mask"][:, torch.arange(4), torch.arange(4)].any())
    assert not bool(prediction["pair_mask"][0, 3].any())
    assert not bool(prediction["pair_mask"][1, 2:].any())
    assert not torch.equal(
        prediction["value"], prediction["value"].transpose(1, 2)
    )
    clone = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_HLT_TRACK_PAIR_13",
        interface="FB_PAIR",
        particle_dimension=16,
        control="EXACT_HLT",
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.set_update(2, 2)
    cloned_logits = clone(**batch, raw_tokens=raw)
    torch.testing.assert_close(cloned_logits, logits)
    clone.classifier.mod.trimmer = _DeterministicTrainingTrimmer()
    trimmed_logits, trimmed_prediction = clone.forward_with_feedback(
        **batch, raw_tokens=raw
    )
    assert trimmed_logits.shape == (2, 10)
    assert trimmed_prediction["value"].shape == (2, 3, 3, 13)
    assert trimmed_prediction["pair_mask"].shape == (2, 3, 3)
    with pytest.raises(ValueError, match="raw HLT tokens"):
        clone(**batch)


def test_exact_hlt_loader_contract_has_no_materialized_pair_intervention(tmp_path):
    base = tmp_path / "base_loader.json"
    base.write_text("{}", encoding="utf-8")
    manifest = build_stage_e_loader_manifest(
        row={"row_id": "exact-track", "control": "EXACT_HLT"},
        base_loader_manifest=base,
        intervention_sources=None,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    assert manifest["contract"] == "hosd_stage_e_loader_manifest_v2"
    assert manifest["schema_version"] == 2
    assert manifest["intervention_sources"] == {}


def test_feedback_intervention_joins_scalar_sample_identity():
    class _Base(torch.utils.data.Dataset):
        identities = ("jet-a", "jet-b")

        def __len__(self):
            return len(self.identities)

        def __getitem__(self, index):
            return {"identity": self.identities[index], "payload": index}

    values = {
        "jet-a": {"value": np.asarray([1.0], dtype=np.float32)},
        "jet-b": {"value": np.asarray([2.0], dtype=np.float32)},
    }
    dataset = FeedbackInterventionDataset(
        _Base(),
        intervention="predicted_feedback_override",
        values_by_identity=values,
        donor_identity_by_identity={"jet-a": "jet-b", "jet-b": "jet-a"},
        parent_hashes={"intervention": "a" * 64},
    )
    assert dataset[0]["identity"] == "jet-a"
    np.testing.assert_array_equal(
        dataset[0]["predicted_feedback_override"]["value"],
        values["jet-b"]["value"],
    )
    np.testing.assert_array_equal(
        dataset[1]["predicted_feedback_override"]["value"],
        values["jet-a"]["value"],
    )


def test_feedback_intervention_fails_closed_without_scalar_identity():
    class _Base(torch.utils.data.Dataset):
        identities = ("jet-a", "jet-b")

        def __len__(self):
            return len(self.identities)

        def __getitem__(self, index):
            return {"payload": index}

    dataset = FeedbackInterventionDataset(
        _Base(),
        intervention="oracle_feedback",
        values_by_identity={
            identity: {"value": np.asarray([index], dtype=np.float32)}
            for index, identity in enumerate(_Base.identities)
        },
        parent_hashes={"intervention": "b" * 64},
    )
    with pytest.raises(ValueError, match="scalar event identity"):
        dataset[0]


def test_stage_e_controls_inherit_locked_a_t_semantics():
    raw = copy.deepcopy(_single_family_lock())
    raw.pop("content_hash")
    raw["selected_definition_by_target"]["T_OFFLINE_TRACK_32"].update(
        {"parameterization": "HET", "auxiliary_weight": 0.1}
    )
    lock = with_content_hash(raw)
    plan = build_stage_e_plan(
        single_family_selection=lock,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    rows = [
        row
        for row in plan["control_rows"]
        if row["target_id"] == "T_OFFLINE_TRACK_32"
    ]
    assert rows and all(row["parameterization"] == "HET" for row in rows)
    assert all(
        row["auxiliary_weight"]
        == (0.0 if row["control"] in {"DISABLED_LOSS", "UNRESTRICTED"} else 0.1)
        for row in rows
    )
