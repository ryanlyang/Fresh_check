from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    AuxiliaryHBaseClassifier,
    HOSDTrainingProtocol,
    auxiliary_objective,
    auxiliary_objective_contract,
    build_sampled_pair_batch,
    build_default_stage_d_role_definitions,
    build_single_family_phase_lock,
    build_single_family_selection,
    build_stage_d_loader_manifest,
    build_stage_d_plan,
    global_auxiliary_loss,
    heteroscedastic_component_mask,
    resolve_stage_d_phase_two,
    sampled_pair_auxiliary_loss,
    validate_stage_d_plan,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    AUXILIARY_PREDICTION_CONTRACT,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_training import (
    train_stage_d_auxiliary,
)
from teacher_logit_reco.hlt_offline_structure_distillation.target_schemas import (
    target_declarations,
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
REPO_ROOT = Path(__file__).resolve().parents[1]


def _registry():
    return with_content_hash(
        {
            "contract": "hosd_structure_target_registry_v1",
            "schema_version": 1,
            "source": SOURCE,
            "targets": [
                {
                    "target_id": declaration.target_id,
                    "executable_current_source": (
                        declaration.campaign_status == "current_required"
                    ),
                    "head_type": declaration.head_type,
                }
                for declaration in target_declarations()
            ],
        }
    )


def test_complete_stage_d_matrix_is_fixed_bounded_and_performance_independent():
    plan = build_stage_d_plan(
        campaign_spec_sha256="c" * 64,
        target_registry=_registry(),
        source=SOURCE,
    )
    assert validate_stage_d_plan(plan, target_registry=_registry()) == plan["content_hash"]
    assert plan["row_count"] == 213
    assert plan["hard_maximum"] == 213
    assert plan["matrix_counts"] == {
        "global_abs_res_het_weight": 54,
        "relation_abs_res": 42,
        "locked_best_relation_het": 3,
        "pair": 6,
        "latent": 3,
        "kd": 2,
        "matched_hlt_self": 13,
        "null_controls": 90,
    }
    assert len(plan["scientific_rows"]) == 110
    assert len(plan["hlt_self_rows"]) == 13
    assert len(plan["control_rows"]) == 90
    assert all(not row["performance_can_omit_or_cancel"] for row in plan["all_rows"])
    assert {
        row["row_kind"] for row in plan["control_rows"]
    } == {
        "DISABLED",
        "STOP_ENCODER",
        "TARGET_MEAN",
        "GLOBAL_SHUFFLE",
        "WITHIN_CLASS_SHUFFLE",
    }
    jet_controls = [
        row
        for row in plan["control_rows"]
        if row["target_id"] == "T_OFFLINE_JET_10"
    ]
    assert len({row["head_component_seed"] for row in jet_controls}) == 1
    row = next(
        row
        for row in plan["scientific_rows"]
        if row["phase"] == "PRIMARY"
    )
    roles = {
        role: {
            "labels": f"/bound/{role}/labels.npz",
            "hlt_caches": {"0": f"/bound/{role}/hlt"},
            "target": {
                "mode": "static_cache",
                "caches": {"shared": f"/bound/{role}/target"},
            },
        }
        for role in ("model_train", "val_stop", "design_select")
    }
    loader = build_stage_d_loader_manifest(
        row=row,
        role_definitions=roles,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    assert loader["dense_pair_targets_persisted"] is False
    assert loader["performance_dependent_inputs"] is False


def test_pair_target_roles_bind_the_authenticated_hlt_relation_normalizer(tmp_path):
    row = {
        "target_id": "T_HLT_TRACK_PAIR_13",
        "parameterization": "ABS",
        "row_kind": "SCIENTIFIC",
    }

    def base_roles(training_role, evaluation_role):
        return {
            role: {
                "labels": str(tmp_path / f"{role}.npz"),
                "hlt_caches": {"0": str(tmp_path / f"{role}-hlt")},
            }
            for role in (training_role, "val_stop", evaluation_role)
        }

    roles_500k = build_default_stage_d_role_definitions(
        row=row,
        campaign_root=tmp_path,
        base_role_definitions=base_roles("model_train", "design_select"),
    )
    expected_500k = (
        tmp_path
        / "inputs"
        / "normalization"
        / "hlt_shared_500k"
        / "relation.json"
    )
    assert {
        definition["target"]["relation_normalizer"]
        for definition in roles_500k.values()
    } == {str(expected_500k)}

    roles_scale = build_default_stage_d_role_definitions(
        row=row,
        campaign_root=tmp_path,
        base_role_definitions=base_roles("scale_train", "design_confirm"),
        evaluation_role="design_confirm",
        training_role="scale_train",
    )
    expected_scale = (
        tmp_path
        / "scale_up"
        / "normalization"
        / "shared_hlt_scale"
        / "relation.json"
    )
    assert {
        definition["target"]["relation_normalizer"]
        for definition in roles_scale.values()
    } == {str(expected_scale)}


class _TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(3, 8)
        self.output = torch.nn.Linear(8, 10)

    def _states(self, features, mask):
        state = self.input(features.transpose(1, 2))
        return state.masked_fill(~mask[:, 0].bool().unsqueeze(-1), 0)

    def forward(self, points, features, vectors, mask):
        del points, vectors
        state = self._states(features, mask)
        pooled = state.sum(dim=1) / mask[:, 0].sum(dim=1, keepdim=True).clamp_min(1)
        return self.output(pooled)

    def forward_with_taps(self, points, features, vectors, mask, *, capture):
        state = self._states(features, mask)
        return SimpleNamespace(
            logits=self.forward(points, features, vectors, mask),
            states={capture[0]: state},
            masks={capture[0]: mask[:, 0].bool()},
        )


def _model(*, stop=False):
    torch.manual_seed(4)
    return AuxiliaryHBaseClassifier(
        _TinyClassifier(),
        target_id="T_OFFLINE_JET_10",
        target_dimension=2,
        input_dimension=8,
        availability_group_count=1,
        parameterization="ABS",
        stop_encoder=stop,
    )


def _model_batch():
    torch.manual_seed(5)
    return {
        "points": torch.zeros(4, 2, 5),
        "features": torch.randn(4, 3, 5),
        "vectors": torch.ones(4, 4, 5),
        "mask": torch.ones(4, 1, 5, dtype=torch.bool),
    }


def test_auxiliary_graph_is_classification_isolated_and_stop_encoder_is_exact():
    batch = _model_batch()
    ordinary = _model()
    expected = ordinary.classifier(**batch)
    torch.testing.assert_close(ordinary(**batch), expected, atol=0, rtol=0)
    logits, prediction = ordinary.forward_with_aux(**batch)
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    target = torch.zeros(4, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss, _ = global_auxiliary_loss(
        prediction,
        target,
        mask,
        parameterization="ABS",
        component_group_ids=("valid", "valid"),
        target_id="T_OFFLINE_JET_10",
    )
    gradients = torch.autograd.grad(
        loss, ordinary.classifier.parameters(), allow_unused=True
    )
    assert any(value is not None and bool(value.abs().sum() > 0) for value in gradients)

    stopped = _model(stop=True)
    _, prediction = stopped.forward_with_aux(**batch)
    stopped_loss, _ = global_auxiliary_loss(
        prediction,
        target,
        mask,
        parameterization="ABS",
        component_group_ids=("valid", "valid"),
        target_id="T_OFFLINE_JET_10",
    )
    gradients = torch.autograd.grad(
        stopped_loss, stopped.classifier.parameters(), allow_unused=True
    )
    assert all(value is None or int(value.count_nonzero()) == 0 for value in gradients)
    head_gradients = torch.autograd.grad(stopped_loss, stopped.target_head.parameters())
    assert any(bool(value.abs().sum() > 0) for value in head_gradients)


def test_per_event_losses_availability_het_latent_and_disabled_weight():
    prediction = {
        "value": torch.tensor([[1.0, 3.0], [4.0, 8.0]], requires_grad=True),
        "availability_logits": torch.zeros(2, 2, requires_grad=True),
    }
    target = torch.zeros(2, 2)
    mask = torch.tensor([[True, False], [True, True]])
    family, pieces = global_auxiliary_loss(
        prediction,
        target,
        mask,
        parameterization="ABS",
        component_group_ids=("a", "b"),
        target_id="T_OFFLINE_JET_10",
    )
    # Huber: event means are 0.5 and (3.5+7.5)/2=5.5; jet mean=3.
    assert pieces["value_loss"].item() == pytest.approx(3.0)
    assert pieces["availability_loss"].item() == pytest.approx(np.log(2))
    assert family.item() == pytest.approx(0.5 * (3.0 + np.log(2)))
    logits = torch.randn(2, 10, requires_grad=True)
    total, objective = auxiliary_objective(
        logits=logits,
        labels=torch.tensor([0, 1]),
        prediction=prediction,
        target=target,
        target_mask=mask,
        target_id="T_OFFLINE_JET_10",
        parameterization="ABS",
        auxiliary_weight=0.0,
        component_group_ids=("a", "b"),
    )
    assert total.item() == pytest.approx(objective["classification_loss"].item())
    contract = auxiliary_objective_contract()
    assert contract["classification_isolation"]
    assert contract["heteroscedastic"]["log_variance_clip"] == [-8.0, 5.0]
    relation = next(
        declaration
        for declaration in target_declarations()
        if declaration.target_id == "T_OFFLINE_RELATION_CHARGE"
    )
    het_mask = heteroscedastic_component_mask(
        relation.target_id, relation.components
    )
    assert any(het_mask) and not all(het_mask)


def test_pair_sampling_is_exact_capped_stratified_and_reduced_per_jet():
    size = 50
    target = torch.zeros(2, size, size, 8)
    mask = torch.zeros_like(target, dtype=torch.bool)
    for event in range(2):
        for left in range(size):
            for right in range(left + 1, size):
                mask[event, left, right] = True
                target[event, left, right, 0] = (left + right + event) % 3 == 0
    first = build_sampled_pair_batch(
        target,
        mask,
        identities=("jet-a", "jet-b"),
        epoch=3,
        target_id="T_HLT_REGION_PAIR_8",
    )
    second = build_sampled_pair_batch(
        target,
        mask,
        identities=("jet-a", "jet-b"),
        epoch=3,
        target_id="T_HLT_REGION_PAIR_8",
    )
    assert torch.equal(first.event_indices, second.event_indices)
    assert torch.equal(first.left_indices, second.left_indices)
    for event in range(2):
        selected = first.event_indices == event
        assert int((first.positive & selected).sum()) <= 512
        assert int((~first.positive & selected).sum()) <= 512
    prediction = torch.zeros_like(first.target, requires_grad=True)
    loss, pieces = sampled_pair_auxiliary_loss(
        prediction, first, target_id="T_HLT_REGION_PAIR_8", event_count=2
    )
    assert torch.isfinite(loss)
    assert pieces["pair_reduction"] == (
        "per_jet_equal_positive_negative_strata_when_both_exist"
    )


def _primary_results(plan):
    labels = np.tile(np.arange(10, dtype=np.int64), 3)
    logits = np.full((len(labels), 10), -2.0)
    logits[np.arange(len(labels)), labels] = 3.0
    metrics = evaluate_classification(logits, labels, split="design_select")
    return [
        with_content_hash(
            {
                "contract": AUXILIARY_PREDICTION_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                "stage_d_plan_sha256": plan["content_hash"],
                "campaign_spec_sha256": "c" * 64,
                "row_id": row["row_id"],
                "target_id": row["target_id"],
                "parameterization": row["parameterization"],
                "auxiliary_weight": row["auxiliary_weight"],
                "row_kind": row["row_kind"],
                "selection_eligible": True,
                "design_select": {"classification_metrics": metrics},
                "deployed_analytical_flops": 10.0,
                "deployed_parameter_count": 100,
                "training_gpu_hours": 1.0,
            }
        )
        for row in plan["scientific_rows"]
        if row["phase"] == "PRIMARY"
    ]


def test_phase_lock_resolves_relation_het_and_all_matched_hlt_self_rows():
    plan = build_stage_d_plan(
        campaign_spec_sha256="c" * 64,
        target_registry=_registry(),
        source=SOURCE,
    )
    lock = build_single_family_phase_lock(
        stage_d_plan=plan,
        primary_results=_primary_results(plan),
        source=SOURCE,
    )
    resolved = resolve_stage_d_phase_two(stage_d_plan=plan, phase_lock=lock)
    assert len(resolved) == 16
    assert sum(row["row_kind"] == "SCIENTIFIC" for row in resolved) == 3
    assert sum(row["row_kind"] == "HLT_SELF" for row in resolved) == 13
    assert all(row["resolved"] for row in resolved)
    assert all("phase_lock_sha256" in row for row in resolved)
    metrics = _primary_results(plan)[0]["design_select"]["classification_metrics"]

    def result(row):
        return with_content_hash(
            {
                "contract": AUXILIARY_PREDICTION_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                "stage_d_plan_sha256": plan["content_hash"],
                "campaign_spec_sha256": "c" * 64,
                "row_id": row["row_id"],
                "target_id": row["target_id"],
                "parameterization": row["parameterization"],
                "auxiliary_weight": row["auxiliary_weight"],
                "row_kind": row["row_kind"],
                "selection_eligible": row["row_kind"] == "SCIENTIFIC",
                "design_select": {"classification_metrics": metrics},
                "deployed_analytical_flops": 10.0,
                "deployed_parameter_count": 100,
                "training_gpu_hours": 1.0,
            }
        )

    complete_rows = [
        *[
            row
            for row in plan["scientific_rows"]
            if row["phase"] == "PRIMARY"
        ],
        *plan["control_rows"],
        *resolved,
    ]
    with pytest.raises(ValueError, match="result/control coverage"):
        build_single_family_selection(
            stage_d_plan=plan,
            phase_lock=lock,
            results=[result(row) for row in complete_rows[:-1]],
            source=SOURCE,
        )
    selection = build_single_family_selection(
        stage_d_plan=plan,
        phase_lock=lock,
        results=[result(row) for row in complete_rows],
        source=SOURCE,
    )
    assert len(selection["complete_result_hashes"]) == 213
    assert len(selection["selected_row_by_target"]) == 18


class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(9)
        self.features = torch.randn(20, 3, 5, generator=generator)
        self.labels = torch.arange(20) % 10
        self.control_kind = None
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(int(epoch))

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {
            "points": torch.zeros(2, 5),
            "features": self.features[index],
            "vectors": torch.ones(4, 5),
            "mask": torch.ones(1, 5, dtype=torch.bool),
            "labels": self.labels[index],
            "identities": f"jet-{index}",
            "target": torch.tensor([index / 20, 1.0]),
            "target_mask": torch.ones(2, dtype=torch.bool),
        }


def _collate(rows):
    output = {}
    for key in rows[0]:
        if key == "identities":
            output[key] = [row[key] for row in rows]
        else:
            output[key] = torch.stack([row[key] for row in rows])
    return output


def test_miniature_stage_d_trainer_finishes_all_epochs_on_negative_results(tmp_path):
    plan = build_stage_d_plan(
        campaign_spec_sha256="c" * 64,
        target_registry=_registry(),
        source=SOURCE,
    )
    row = next(
        row
        for row in plan["scientific_rows"]
        if row["target_id"] == "T_OFFLINE_JET_10"
        and row["parameterization"] == "ABS"
        and row["auxiliary_weight"] == 0.1
    )
    data = _Dataset()
    loader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, collate_fn=_collate
    )
    completion = train_stage_d_auxiliary(
        model=_model(),
        train_loader=loader,
        val_stop_loader=loader,
        design_select_loader=loader,
        output_dir=tmp_path / "row",
        row=row,
        component_group_ids=("valid", "valid"),
        stage_d_plan_sha256=plan["content_hash"],
        campaign_spec_sha256="c" * 64,
        lineage_hashes={"target_cache": "d" * 64},
        protocol=HOSDTrainingProtocol(
            maximum_epochs=2, campaign_profile="miniature_test"
        ),
        source=SOURCE,
        deployed_analytical_flops=123.0,
        device="cpu",
        training_gpu_hours_override=0.5,
    )
    assert completion["epochs_completed"] == 2
    assert completion["performance_based_termination"] is False
    assert completion["future_rows_cancelled_for_performance"] is False
    assert data.epochs == [1, 2]
    assert not (tmp_path / "row" / "last.pt").exists()


def test_step6_production_entrypoints_are_concrete():
    for name in (
        "build_hosd_stage_d_loader_manifest.py",
        "train_hosd_auxiliary.py",
        "select_hosd_single_targets.py",
    ):
        text = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "--campaign-root" in text
    train = (REPO_ROOT / "scripts" / "train_hosd_auxiliary.py").read_text(
        encoding="utf-8"
    )
    assert "--loader-manifest" in train
    assert "train_stage_d_auxiliary" in train
