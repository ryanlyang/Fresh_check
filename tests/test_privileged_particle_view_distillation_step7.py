from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from teacher_logit_reco.local_particle_residual_field.particle_view.distillation import (
    DISTILLATION_LINEAGE_FIELDS,
    JOINT_LINEAGE_FIELDS,
    PARTICLE_VIEW_LOSS_IDS,
    TARGET_LOGIT_CACHE_LINEAGE_FIELDS,
    DeployableParticleViewBundle,
    DistillationTrainConfig,
    FrozenTargetLogitCache,
    JointFineTuneConfig,
    audit_teacher_independent_deployment,
    build_distillation_loss_screen,
    build_generalization_report,
    build_schedule_matched_ce_control,
    build_target_loss_interaction_campaign,
    distillation_losses,
    exact_same_consumer_forward,
    joint_finetuning_losses,
    joint_schedule_contract_sha256,
    load_registered_distillation_bundle,
    module_state_sha256,
    publish_target_logit_cache,
    rank_model_val_select_configurations,
    select_distillation_checkpoint,
    stage_e_learning_rate,
    train_frozen_consumer_distillation,
    train_joint_finetuning,
    validate_distillation_registration,
)
from teacher_logit_reco.local_particle_residual_field.particle_view.predictor import (
    ParticleViewPredictorOutput,
)


def _hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


class TinyPredictor(nn.Module):
    def __init__(self, input_dim: int = 3, view_dim: int = 2):
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, view_dim)
        self.variance_layer = nn.Linear(input_dim, view_dim)

    def forward(self, features, lorentz_vectors, mask):
        del lorentz_vectors
        valid = mask[:, 0] if mask.ndim == 3 else mask
        hidden = features.transpose(1, 2)
        mean = self.mean_layer(hidden)
        mean = torch.where(valid[:, :, None], mean, torch.zeros_like(mean))
        log_variance = self.variance_layer(hidden).clamp(-6, 3)
        log_variance = torch.where(
            valid[:, :, None],
            log_variance,
            torch.zeros_like(log_variance),
        )
        return ParticleViewPredictorOutput(
            mean=mean,
            log_variance=log_variance,
            trust=torch.sigmoid(mean[..., :1]),
            balance_loss=mean.new_zeros(()),
            hierarchy=None,
            local_embeddings=hidden,
            decoded_embeddings=hidden,
        )


class TinyConsumer(nn.Module):
    def __init__(self, feature_dim: int = 3, view_dim: int = 2, classes: int = 3):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, classes)
        self.view_projection = nn.Linear(view_dim, classes, bias=False)

    def forward(
        self,
        points,
        features,
        lorentz_vectors,
        mask,
        view,
        *,
        augment_clean_view=False,
    ):
        del points, lorentz_vectors, augment_clean_view
        valid = (mask[:, 0] if mask.ndim == 3 else mask).float()
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled_features = (
            features.transpose(1, 2) * valid[:, :, None]
        ).sum(dim=1) / denominator
        pooled_view = (view * valid[:, :, None]).sum(dim=1) / denominator
        logits = self.feature_projection(pooled_features) + self.view_projection(
            pooled_view
        )
        return SimpleNamespace(
            logits=logits,
            trust_loss=view.abs().mean(),
        )


class TinySharedStemPredictor(nn.Module):
    def __init__(self, shared_stem: nn.Linear):
        super().__init__()
        self.shared_stem = shared_stem
        self.trainable_delta = nn.Linear(3, 3)
        self.variance_layer = nn.Linear(3, 3)

    def forward(self, features, lorentz_vectors, mask):
        del lorentz_vectors
        valid = mask[:, 0] if mask.ndim == 3 else mask
        hidden = features.transpose(1, 2)
        mean = self.shared_stem(hidden) + self.trainable_delta(hidden)
        mean = torch.where(valid[:, :, None], mean, torch.zeros_like(mean))
        log_variance = self.variance_layer(hidden).clamp(-6, 3)
        log_variance = torch.where(
            valid[:, :, None],
            log_variance,
            torch.zeros_like(log_variance),
        )
        return ParticleViewPredictorOutput(
            mean=mean,
            log_variance=log_variance,
            trust=torch.sigmoid(mean[..., :1]),
            balance_loss=mean.new_zeros(()),
            hierarchy=None,
            local_embeddings=hidden,
            decoded_embeddings=hidden,
        )


def _batch(
    *,
    ids=(31, 12, 44, 20),
    input_dim: int = 3,
    particles: int = 5,
    view_dim: int = 2,
    include_true: bool = True,
):
    generator = torch.Generator().manual_seed(7101 + sum(ids))
    batch = len(ids)
    features = torch.randn(
        (batch, input_dim, particles), generator=generator
    )
    momentum = torch.randn((batch, 3, particles), generator=generator)
    energy = torch.sqrt(momentum.square().sum(dim=1, keepdim=True) + 1)
    mask = torch.ones((batch, 1, particles), dtype=torch.bool)
    mask[0, 0, -1] = False
    result = {
        "event_ids": torch.tensor(ids, dtype=torch.long),
        "points": torch.randn((batch, 2, particles), generator=generator),
        "features": features,
        "lorentz_vectors": torch.cat((momentum, energy), dim=1),
        "mask": mask,
        "labels": torch.tensor(
            [index % 3 for index in range(batch)], dtype=torch.long
        ),
    }
    if include_true:
        true_view = torch.randn(
            (batch, particles, view_dim), generator=generator
        )
        true_view = torch.where(
            mask[:, 0, :, None],
            true_view,
            torch.zeros_like(true_view),
        )
        result["true_view"] = true_view
    return result


def _cache_lineage(prefix: str):
    return {
        field: _hash(f"{prefix}:{field}")
        for field in TARGET_LOGIT_CACHE_LINEAGE_FIELDS
    }


def _publish_three_caches(tmp_path: Path, consumer: nn.Module):
    result = {}
    for split, ids in (
        ("train", (1, 2, 3, 4)),
        ("model_val_stop", (11, 12, 13, 14)),
        ("model_val_select", (21, 22, 23, 24)),
    ):
        root = tmp_path / split
        manifest = publish_target_logit_cache(
            frozen_consumer=consumer,
            loader=[_batch(ids=ids)],
            output_dir=root,
            split=split,
            lineage=_cache_lineage("shared"),
        )
        result[split] = (
            FrozenTargetLogitCache(
                root / f"{split}_frozen_consumer_logits.json"
            ),
            manifest,
            [_batch(ids=ids)],
        )
    return result


def test_step7_loss_screen_and_schedule_matched_control_are_complete():
    screen = build_distillation_loss_screen()
    assert tuple(screen) == PARTICLE_VIEW_LOSS_IDS
    assert len(screen) == 17
    primary = screen["L_PRIMARY"]
    assert (
        primary.kd,
        primary.huber,
        primary.cosine,
        primary.relational,
        primary.ce,
        primary.trust,
    ) == (1.0, 0.35, 0.10, 0.10, 0.15, 0.01)
    assert screen["L_PRIMARY_NO_CE"].ce == 0
    assert screen["L_PRIMARY_CE05"].ce == 0.05
    assert screen["L_PRIMARY_CE35"].ce == 0.35
    assert screen["L_PRIMARY_NO_TRUST"].trust == 0
    assert screen["L_UNCERTAINTY"].uncertainty == 0.05
    assert screen["L_CE"].privileged_claim_eligible is False
    control = build_schedule_matched_ce_control(
        primary,
        source_registration_sha256=_hash("source-registration"),
        matched_optimizer_updates=123,
    )
    assert control["same_architecture"]
    assert control["same_freeze_unfreeze_schedule"]
    assert control["same_optimizer_update_budget"]
    assert control["privileged_targets_removed"]
    assert control["matched_optimizer_updates"] == 123


def test_step7_exact_same_consumer_pair_has_zero_kd_at_exact_view_and_gradients():
    torch.manual_seed(7102)
    predictor = TinyPredictor()
    consumer = TinyConsumer()
    batch = _batch()
    # Make the true view exactly equal to the live predictor output.
    with torch.no_grad():
        batch["true_view"] = predictor(
            batch["features"], batch["lorentz_vectors"], batch["mask"]
        ).mean
    state = module_state_sha256(consumer)
    paired = exact_same_consumer_forward(
        predictor=predictor,
        frozen_consumer=consumer,
        batch=batch,
        true_view=batch["true_view"],
    )
    assert torch.allclose(paired.target_logits, paired.live_logits, atol=1e-7)
    assert module_state_sha256(consumer) == state
    assert all(not parameter.requires_grad for parameter in consumer.parameters())
    kd = torch.nn.functional.kl_div(
        torch.log_softmax(paired.live_logits / 2, dim=-1),
        torch.softmax(paired.target_logits / 2, dim=-1),
        reduction="batchmean",
    ) * 4
    assert kd.abs() < 1e-7
    (paired.live_logits.square().mean()).backward()
    assert predictor.mean_layer.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="differ"):
        exact_same_consumer_forward(
            predictor=predictor,
            frozen_consumer=consumer,
            batch=batch,
            true_view=batch["true_view"],
            cached_target_logits=paired.target_logits + 0.1,
        )


def test_step7_primary_and_joint_losses_are_finite_and_semantically_separate():
    torch.manual_seed(7103)
    predictor = TinyPredictor()
    consumer = TinyConsumer()
    batch = _batch()
    output = predictor(
        batch["features"], batch["lorentz_vectors"], batch["mask"]
    )
    live = consumer(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        output.mean,
    )
    target = consumer(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        batch["true_view"],
    ).logits.detach()
    primary = distillation_losses(
        predictor_output=output,
        live_consumer_output=live,
        target_logits=target,
        labels=batch["labels"],
        true_view=batch["true_view"],
        mask=batch["mask"],
        objective=build_distillation_loss_screen()["L_PRIMARY"],
    )
    expected = (
        primary["kd"]
        + 0.35 * primary["huber"]
        + 0.10 * primary["cosine"]
        + 0.10 * primary["relational"]
        + 0.15 * primary["ce"]
        + 0.01 * primary["trust"]
        + primary["balance"]
    )
    assert torch.allclose(primary["total"], expected)
    joint = joint_finetuning_losses(
        predictor_output=output,
        live_consumer_output=live,
        target_logits=target,
        labels=batch["labels"],
        true_view=batch["true_view"],
        mask=batch["mask"],
        config=JointFineTuneConfig(),
    )
    assert joint["relational"].item() == 0
    assert torch.isfinite(joint["total"])
    ce_joint = joint_finetuning_losses(
        predictor_output=output,
        live_consumer_output=live,
        target_logits=None,
        labels=batch["labels"],
        true_view=None,
        mask=batch["mask"],
        config=JointFineTuneConfig(objective_id="JOINT_CE_ONLY"),
    )
    assert torch.allclose(ce_joint["total"], ce_joint["ce"])


def test_step7_target_logit_cache_binds_exact_consumer_and_event_ids(tmp_path):
    consumer = TinyConsumer()
    lineage = _cache_lineage("cache")
    manifest = publish_target_logit_cache(
        frozen_consumer=consumer,
        loader=[_batch(ids=(8, 2, 5, 3))],
        output_dir=tmp_path,
        split="train",
        lineage=lineage,
    )
    cache = FrozenTargetLogitCache(
        tmp_path / "train_frozen_consumer_logits.json"
    )
    assert cache.content_hash == manifest["content_hash"]
    cache.validate_consumer(consumer)
    values = cache.lookup(torch.tensor([5, 2]), device=torch.device("cpu"))
    assert values.shape == (2, 3)
    with pytest.raises(KeyError):
        cache.lookup(torch.tensor([999]), device=torch.device("cpu"))
    with torch.no_grad():
        next(consumer.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="another consumer"):
        cache.validate_consumer(consumer)


def test_step7_checkpoint_selection_uses_accuracy_pool_then_locked_ties():
    rows = [
        {
            "epoch": 1,
            "model_val_stop": {
                "deployable_accuracy": 0.80000,
                "deployable_cross_entropy": 0.50,
                "recovery_status": "finite",
                "recovery_fraction": 0.8,
            },
        },
        {
            "epoch": 2,
            "model_val_stop": {
                "deployable_accuracy": 0.80008,
                "deployable_cross_entropy": 0.49,
                "recovery_status": "undefined",
                "recovery_fraction": None,
            },
        },
        {
            "epoch": 3,
            "model_val_stop": {
                "deployable_accuracy": 0.80100,
                "deployable_cross_entropy": 0.70,
                "recovery_status": "finite",
                "recovery_fraction": 0.1,
            },
        },
    ]
    # Epoch three excludes both earlier rows despite its worse CE.
    assert select_distillation_checkpoint(rows)["epoch"] == 3
    assert select_distillation_checkpoint(rows[:2])["epoch"] == 2
    tied = [dict(rows[0]), dict(rows[0])]
    tied[1]["epoch"] = 4
    assert select_distillation_checkpoint(tied)["epoch"] == 1


def test_step7_campaign_enumerates_full_target_loss_and_joint_interactions():
    campaign = build_target_loss_interaction_campaign(
        target_ids=("canonical", "alternate"),
        canonical_target_id="canonical",
        alternate_target_id="alternate",
    )
    assert campaign["row_count"] == 52
    assert campaign["performance_gates"] is False
    rows = campaign["rows"]
    for target in ("canonical", "alternate"):
        assert {
            row["loss_id"]
            for row in rows
            if row["target_id"] == target and row["mode"] == "frozen"
        } == set(PARTICLE_VIEW_LOSS_IDS)
    assert sum(row["mode"] == "joint" for row in rows) == 2
    assert sum(row["mode"] == "joint_ce_control" for row in rows) == 2


def test_step7_model_val_select_ranking_and_generalization_warning():
    rows = [
        {
            "split": "model_val_select",
            "run_id": "run_b",
            "configuration_id": "b",
            "deployable_accuracy": 0.75,
            "deployable_cross_entropy": 0.6,
            "recovery_status": "undefined",
            "recovery_fraction": None,
            "deployed_parameters": 100,
        },
        {
            "split": "model_val_select",
            "run_id": "run_a",
            "configuration_id": "a",
            "deployable_accuracy": 0.75,
            "deployable_cross_entropy": 0.59,
            "recovery_status": "finite",
            "recovery_fraction": 0.1,
            "deployed_parameters": 200,
        },
    ]
    ranking = rank_model_val_select_configurations(rows)
    assert ranking["winner"]["configuration_id"] == "a"
    base = {
        "kd": 0.2,
        "huber": 0.1,
        "cosine": 0.1,
        "relational": 0.1,
        "deployable_accuracy": 0.70,
    }
    report = build_generalization_report(
        train_metrics={**base, "split": "train", "deployable_accuracy": 0.73},
        model_val_stop_metrics={**base, "split": "model_val_stop"},
        model_val_select_metrics={**base, "split": "model_val_select"},
        configuration_id="config",
        seed=101,
    )
    assert report["warnings_are_non_gating"]
    assert report["quality_warnings"][0]["warning_code"] == (
        "WARN_LARGE_TRAIN_VALIDATION_GAP"
    )


def test_step7_learning_rate_and_teacher_independent_deployment_audit():
    config = DistillationTrainConfig()
    assert stage_e_learning_rate(1, 10_000, config) == pytest.approx(
        3.0e-4 / 2_000
    )
    assert stage_e_learning_rate(2_000, 10_000, config) == pytest.approx(3.0e-4)
    assert stage_e_learning_rate(10_000, 10_000, config) == pytest.approx(
        3.0e-6
    )
    predictor = TinyPredictor()
    consumer = TinyConsumer()
    batch = _batch()
    hlt = {
        name: batch[name]
        for name in ("points", "features", "lorentz_vectors", "mask")
    }
    bundle = DeployableParticleViewBundle(predictor, consumer).eval()
    with torch.no_grad():
        reference = bundle(**hlt)
    audit = audit_teacher_independent_deployment(
        predictor=predictor,
        consumer=consumer,
        hlt_batch=hlt,
        reference_logits=reference,
        dependency_manifest={"inputs": sorted(hlt)},
    )
    assert audit["teacher_independent"]
    with pytest.raises(ValueError, match="forbidden"):
        audit_teacher_independent_deployment(
            predictor=predictor,
            consumer=consumer,
            hlt_batch=hlt,
            dependency_manifest={"teacher_checkpoint": "bad"},
        )


def test_step7_miniature_frozen_consumer_training_registration_and_reload(
    tmp_path,
):
    torch.manual_seed(7104)
    predictor = TinyPredictor()
    consumer = TinyConsumer()
    caches = _publish_three_caches(tmp_path / "caches", consumer)
    cache_lineage = _cache_lineage("shared")
    lineage = {
        field: _hash(f"distill:{field}") for field in DISTILLATION_LINEAGE_FIELDS
    }
    lineage["hlt_preprocessing_sha256"] = cache_lineage[
        "hlt_preprocessing_sha256"
    ]
    lineage["class_order_sha256"] = cache_lineage["class_order_sha256"]
    lineage["coordinate_binding_sha256"] = cache_lineage[
        "coordinate_binding_sha256"
    ]
    lineage["consumer_checkpoint_sha256"] = cache_lineage[
        "consumer_checkpoint_sha256"
    ]
    lineage["initial_predictor_state_sha256"] = module_state_sha256(
        predictor
    )
    lineage["train_target_logit_cache_sha256"] = caches["train"][0].content_hash
    lineage["model_val_stop_target_logit_cache_sha256"] = caches[
        "model_val_stop"
    ][0].content_hash
    lineage["model_val_select_target_logit_cache_sha256"] = caches[
        "model_val_select"
    ][0].content_hash
    registration = train_frozen_consumer_distillation(
        predictor=predictor,
        frozen_consumer=consumer,
        train_loader=caches["train"][2],
        model_val_stop_loader=caches["model_val_stop"][2],
        model_val_select_loader=caches["model_val_select"][2],
        train_target_cache=caches["train"][0],
        model_val_stop_target_cache=caches["model_val_stop"][0],
        model_val_select_target_cache=caches["model_val_select"][0],
        output_dir=tmp_path / "distillation",
        lineage=lineage,
        objective=build_distillation_loss_screen()["L_PRIMARY"],
        configuration_id="tiny_primary",
        run_id="run_tiny_primary",
        deployed_parameters=sum(
            parameter.numel()
            for module in (predictor, consumer)
            for parameter in module.parameters()
        ),
    )
    assert registration["consumer_frozen"]
    assert registration["consumer_state_sha256_before"] == registration[
        "consumer_state_sha256_after"
    ]
    assert registration["model_val_select_evaluation_count"] == 1
    assert registration["teacher_kd_updates"] > 0
    assert registration["view_supervision_updates"] > 0
    assert registration["label_bearing_updates"] == registration[
        "optimizer_updates"
    ]
    assert registration["labeled_examples_processed"] > 0
    registration_path = tmp_path / "distillation" / "distillation_registration.json"
    loaded_payload = json.loads(registration_path.read_text())
    validated = validate_distillation_registration(
        loaded_payload,
        root=registration_path.parent,
        expected_lineage=lineage,
    )
    assert validated["ok"]
    reloaded_predictor = TinyPredictor()
    reloaded_consumer = TinyConsumer()
    reloaded_consumer.load_state_dict(consumer.state_dict())
    load_registered_distillation_bundle(
        reloaded_predictor,
        reloaded_consumer,
        registration_path=registration_path,
        expected_lineage=lineage,
        expected_frozen_consumer_state_sha256=module_state_sha256(consumer),
    )
    assert all(
        not parameter.requires_grad
        for parameter in reloaded_predictor.parameters()
    )
    assert registration["deployment_audit"]["teacher_independent"]


def test_step7_shared_consumer_stem_remains_exactly_frozen(tmp_path):
    torch.manual_seed(71041)
    consumer = TinyConsumer(view_dim=3)
    predictor = TinySharedStemPredictor(consumer.feature_projection)
    consumer_state = module_state_sha256(consumer)
    caches = {}
    for split, ids in (
        ("train", (1, 2, 3, 4)),
        ("model_val_stop", (11, 12, 13, 14)),
        ("model_val_select", (21, 22, 23, 24)),
    ):
        root = tmp_path / "caches" / split
        publish_target_logit_cache(
            frozen_consumer=consumer,
            loader=[_batch(ids=ids, view_dim=3)],
            output_dir=root,
            split=split,
            lineage=_cache_lineage("shared-stem"),
        )
        caches[split] = (
            FrozenTargetLogitCache(
                root / f"{split}_frozen_consumer_logits.json"
            ),
            [_batch(ids=ids, view_dim=3)],
        )
    cache_lineage = _cache_lineage("shared-stem")
    lineage = {
        field: _hash(f"shared-distill:{field}")
        for field in DISTILLATION_LINEAGE_FIELDS
    }
    for field in (
        "hlt_preprocessing_sha256",
        "class_order_sha256",
        "coordinate_binding_sha256",
        "consumer_checkpoint_sha256",
    ):
        lineage[field] = cache_lineage[field]
    lineage["initial_predictor_state_sha256"] = module_state_sha256(
        predictor
    )
    for split, field in (
        ("train", "train_target_logit_cache_sha256"),
        ("model_val_stop", "model_val_stop_target_logit_cache_sha256"),
        ("model_val_select", "model_val_select_target_logit_cache_sha256"),
    ):
        lineage[field] = caches[split][0].content_hash
    registration = train_frozen_consumer_distillation(
        predictor=predictor,
        frozen_consumer=consumer,
        train_loader=caches["train"][1],
        model_val_stop_loader=caches["model_val_stop"][1],
        model_val_select_loader=caches["model_val_select"][1],
        train_target_cache=caches["train"][0],
        model_val_stop_target_cache=caches["model_val_stop"][0],
        model_val_select_target_cache=caches["model_val_select"][0],
        output_dir=tmp_path / "shared_distillation",
        lineage=lineage,
        objective=build_distillation_loss_screen()["L_PRIMARY"],
        configuration_id="shared_stem",
        run_id="run_shared_stem",
        deployed_parameters=sum(
            parameter.numel()
            for parameter in {
                *predictor.parameters(),
                *consumer.parameters(),
            }
        ),
    )
    assert module_state_sha256(consumer) == consumer_state
    assert registration["consumer_state_sha256_before"] == consumer_state
    assert registration["consumer_state_sha256_after"] == consumer_state


def test_step7_joint_branch_and_ce_control_match_updates_and_initialization(
    tmp_path,
):
    torch.manual_seed(7105)
    predictor = TinyPredictor()
    consumer = TinyConsumer()
    caches = _publish_three_caches(tmp_path / "caches", consumer)
    cache_lineage = _cache_lineage("shared")
    config = JointFineTuneConfig()
    lineage = {
        field: _hash(f"joint:{field}") for field in JOINT_LINEAGE_FIELDS
    }
    lineage["initial_predictor_state_sha256"] = module_state_sha256(predictor)
    lineage["initial_consumer_state_sha256"] = module_state_sha256(consumer)
    lineage["schedule_contract_sha256"] = joint_schedule_contract_sha256(
        config
    )
    lineage["oracle_consumer_checkpoint_sha256"] = cache_lineage[
        "consumer_checkpoint_sha256"
    ]
    lineage["coordinate_binding_sha256"] = cache_lineage[
        "coordinate_binding_sha256"
    ]
    lineage["train_target_logit_cache_sha256"] = caches["train"][0].content_hash
    lineage["model_val_stop_target_logit_cache_sha256"] = caches[
        "model_val_stop"
    ][0].content_hash
    lineage["model_val_select_target_logit_cache_sha256"] = caches[
        "model_val_select"
    ][0].content_hash
    _, _, privileged = train_joint_finetuning(
        predictor=predictor,
        selected_frozen_consumer=consumer,
        train_loader=caches["train"][2],
        model_val_stop_loader=caches["model_val_stop"][2],
        model_val_select_loader=caches["model_val_select"][2],
        train_target_cache=caches["train"][0],
        model_val_stop_target_cache=caches["model_val_stop"][0],
        model_val_select_target_cache=caches["model_val_select"][0],
        output_dir=tmp_path / "joint_privileged",
        lineage=lineage,
        configuration_id="joint_privileged",
        run_id="run_joint_privileged",
        config=config,
    )
    assert privileged["mode"] == "Dview_joint"
    assert privileged["recovery_status"] == "undefined"
    privileged_path = (
        tmp_path / "joint_privileged" / "joint_registration.json"
    )
    assert validate_distillation_registration(
        json.loads(privileged_path.read_text()),
        root=privileged_path.parent,
        expected_lineage=lineage,
    )["ok"]
    ce_config = JointFineTuneConfig(objective_id="JOINT_CE_ONLY")
    assert joint_schedule_contract_sha256(ce_config) == lineage[
        "schedule_contract_sha256"
    ]

    def without_true(loader):
        return [
            {name: value for name, value in loader[0].items() if name != "true_view"}
        ]

    _, _, ce_control = train_joint_finetuning(
        predictor=predictor,
        selected_frozen_consumer=consumer,
        train_loader=without_true(caches["train"][2]),
        model_val_stop_loader=without_true(caches["model_val_stop"][2]),
        model_val_select_loader=without_true(caches["model_val_select"][2]),
        train_target_cache=None,
        model_val_stop_target_cache=None,
        model_val_select_target_cache=None,
        output_dir=tmp_path / "joint_ce",
        lineage=lineage,
        configuration_id="joint_ce",
        run_id="run_joint_ce",
        config=ce_config,
        matched_optimizer_updates=privileged["optimizer_updates"],
        schedule_match_source_sha256=privileged["content_hash"],
    )
    assert ce_control["mode"] == "Dview_joint_ce_control"
    assert ce_control["optimizer_updates"] == privileged["optimizer_updates"]
    assert ce_control["exact_schedule_match_completed"]
    assert ce_control["teacher_kd_updates"] == 0
    assert ce_control["view_supervision_updates"] == 0
    assert ce_control["label_bearing_updates"] == ce_control[
        "optimizer_updates"
    ]
    assert ce_control["labeled_examples_processed"] == privileged[
        "labeled_examples_processed"
    ]
    ce_path = tmp_path / "joint_ce" / "joint_registration.json"
    assert validate_distillation_registration(
        json.loads(ce_path.read_text()),
        root=ce_path.parent,
        expected_lineage=lineage,
    )["ok"]
