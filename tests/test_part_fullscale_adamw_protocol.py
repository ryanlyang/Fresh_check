"""The optional two-run AdamW recipe must never alter existing campaigns."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from part_ev_fullscale import protocol as P


ORIGINAL_RECIPE_HASH = "b27c84ee052a8bf1c18a3f7f84fe451db0e88afacbbfd8c3d03312fd9425a106"
ORIGINAL_TASKS_HASH = "e53d14e054530c8d1464efdd97c5df1c781cfff3efd40af8083164f906c3687f"
PLAIN_RECIPE_HASH = "17331d43b4a9d9fe1a69851d2ab6d480d36195ec99ca2c6c86e244171a9fa097"


def write_campaign(output, comparison="standard", training_recipe="original", *, legacy=False):
    config = Path(P.__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
    payload = {
        "schema": P.SCHEMA,
        "recipe": P.recipe_for(comparison, training_recipe),
        "tasks": P.tasks_for(comparison, training_recipe),
        "environment": {"fixture": "pinned runtime"},
        "data_config_sha256": P.file_sha256(config),
    }
    if comparison != "standard" or not legacy:
        payload["comparison"] = comparison
    if not legacy:
        payload["training_recipe"] = training_recipe
    payload["content_hash"] = P.digest(payload)
    P.write_json(output / "campaign.json", payload)
    return payload


def resign(output, payload):
    payload.pop("content_hash", None)
    payload["content_hash"] = P.digest(payload)
    P.write_json(output / "campaign.json", payload)


def test_original_recipes_and_tasks_have_identical_frozen_hashes():
    assert P.digest(P.RECIPE) == ORIGINAL_RECIPE_HASH
    assert P.digest(P.recipe_for()) == ORIGINAL_RECIPE_HASH
    assert P.digest(P.recipe_for("standard", "original")) == ORIGINAL_RECIPE_HASH
    assert P.digest(P.TASKS) == ORIGINAL_TASKS_HASH
    assert P.digest(P.tasks_for()) == ORIGINAL_TASKS_HASH
    assert P.digest(P.recipe_for("plain")) == PLAIN_RECIPE_HASH
    assert len(P.tasks_for("plain")) == 6


def test_adamw_pilot_uses_only_one_prespecified_matched_seed():
    tasks = P.tasks_for(training_recipe="adamw")
    assert tasks == [
        {"architecture": "part", "seed": 101},
        {"architecture": "part_ev", "seed": 101},
    ]
    assert tasks == P.tasks_for("standard", "adamw")
    assert P.SEEDS == (101, 202, 303)
    assert len(P.TASKS) == 6


def test_adamw_recipe_contains_only_the_approved_changes():
    expected_overrides = {
        "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 1e-4, "lookahead_k": None, "lookahead_alpha": None,
        "lr_decay_epochs": [], "lr_schedule": "linear_warmup_then_cosine",
        "warmup_steps": 50000, "gradient_clipping": 1.0,
        "weight_decay_policy": "exclude_bias_normalization_and_cls_token_include_edge_projection",
    }
    assert P.recipe_for(training_recipe="adamw") == {**P.RECIPE, **expected_overrides}
    assert P.recipe_for(training_recipe="adamw")["total_steps"] == 1000000
    assert P.recipe_for(training_recipe="adamw")["global_batch_size"] == 512


def test_recipe_and_task_results_cannot_mutate_global_or_future_profiles():
    before = copy.deepcopy(P.RECIPE)
    recipe = P.recipe_for(training_recipe="adamw")
    recipe["betas"][0] = 0
    recipe["lr_decay_epochs"].append(12)
    recipe["optimizer"] = "another optimizer"
    tasks = P.tasks_for(training_recipe="adamw")
    tasks[0]["seed"] = 999
    tasks.append({"architecture": "part", "seed": 202})
    assert P.RECIPE == before
    assert P.recipe_for(training_recipe="adamw")["betas"] == [0.9, 0.999]
    assert P.recipe_for(training_recipe="adamw")["lr_decay_epochs"] == []
    assert len(P.tasks_for(training_recipe="adamw")) == 2
    assert P.tasks_for(training_recipe="adamw")[0]["seed"] == 101


@pytest.mark.parametrize("helper", ["tasks_for", "recipe_for"])
@pytest.mark.parametrize("comparison,training_recipe", [
    ("standard", "unrecognized"), ("unknown", "adamw"), ("plain", "adamw"),
])
def test_unsupported_combinations_are_rejected(helper, comparison, training_recipe):
    with pytest.raises(ValueError):
        getattr(P, helper)(comparison, training_recipe)


@pytest.mark.parametrize("comparison,training_recipe,legacy", [
    ("standard", "original", True), ("plain", "original", True),
    ("standard", "original", False), ("plain", "original", False),
    ("standard", "adamw", False),
])
def test_loading_keeps_old_and_new_campaign_metadata_byte_identical(
    tmp_path, monkeypatch, comparison, training_recipe, legacy
):
    payload = write_campaign(tmp_path, comparison, training_recipe, legacy=legacy)
    monkeypatch.setattr(P, "environment", lambda: {"fixture": "pinned runtime"})
    before = (tmp_path / "campaign.json").read_bytes()
    observed = P.load_campaign(tmp_path)
    assert observed == payload
    assert (tmp_path / "campaign.json").read_bytes() == before
    assert ("training_recipe" in observed) is not legacy


@pytest.mark.parametrize("field,value", [
    ("optimizer", "Weaver_RAdam_plus_Lookahead"), ("betas", [0.95, 0.999]),
    ("eps", 1e-5), ("weight_decay", 0.0), ("weight_decay", 0.01),
    ("warmup_steps", 10000), ("lr_schedule", "constant"),
    ("gradient_clipping", None), ("lookahead_k", 6),
    ("lr_decay_epochs", list(range(35, 50))),
    ("weight_decay_policy", "all_parameters"),
    ("total_steps", 999999), ("global_batch_size", 128),
])
def test_resigned_adamw_recipe_changes_are_rejected(tmp_path, field, value):
    payload = write_campaign(tmp_path, training_recipe="adamw")
    payload["recipe"][field] = value
    resign(tmp_path, payload)
    with pytest.raises(ValueError, match="frozen"):
        P.load_campaign(tmp_path, check_environment=False)


@pytest.mark.parametrize("mutation", ["missing_profile", "original_profile", "plain_comparison", "six_tasks", "wrong_seed", "wrong_order"])
def test_adamw_campaign_cannot_be_relabelled_or_retasked(tmp_path, mutation):
    payload = write_campaign(tmp_path, training_recipe="adamw")
    if mutation == "missing_profile":
        payload.pop("training_recipe")
    elif mutation == "original_profile":
        payload["training_recipe"] = "original"
    elif mutation == "plain_comparison":
        payload["comparison"] = "plain"
    elif mutation == "six_tasks":
        payload["tasks"] = P.tasks_for()
    elif mutation == "wrong_seed":
        payload["tasks"][0]["seed"] = 202
    else:
        payload["tasks"].reverse()
    resign(tmp_path, payload)
    with pytest.raises(ValueError):
        P.load_campaign(tmp_path, check_environment=False)


def test_original_campaign_cannot_switch_profile_even_with_resigned_metadata(tmp_path):
    payload = write_campaign(tmp_path, legacy=True)
    payload["training_recipe"] = "adamw"
    resign(tmp_path, payload)
    with pytest.raises(ValueError, match="frozen"):
        P.load_campaign(tmp_path, check_environment=False)


def test_unresigned_profile_edit_fails_integrity_check(tmp_path):
    payload = write_campaign(tmp_path, training_recipe="adamw")
    payload["training_recipe"] = "original"
    P.write_json(tmp_path / "campaign.json", payload)
    with pytest.raises(ValueError, match="changed|incompatible"):
        P.load_campaign(tmp_path, check_environment=False)


def test_legacy_learning_rate_is_exactly_unchanged():
    for comparison in ("standard", "plain"):
        recipe = P.recipe_for(comparison)
        for step in (0, 49999, 50000, 699999, 700000, 719999, 720000, 999999):
            epoch = step // 20000
            decays = sum(epoch >= milestone for milestone in range(35, 50))
            expected = 1e-3 * (0.01 ** (1 / 15)) ** decays
            assert P.learning_rate(step) == P.learning_rate(step, recipe) == expected


def test_adamw_warmup_and_cosine_have_the_prespecified_endpoints():
    recipe = P.recipe_for(training_recipe="adamw")
    assert P.learning_rate(0, recipe) == pytest.approx(1e-3 / 50000)
    assert P.learning_rate(24999, recipe) == pytest.approx(5e-4)
    assert P.learning_rate(49998, recipe) < P.learning_rate(49999, recipe)
    assert P.learning_rate(49999, recipe) == 1e-3
    assert P.learning_rate(50000, recipe) < 1e-3
    assert P.learning_rate(524999, recipe) == pytest.approx((1e-3 + 1e-5) / 2)
    assert P.learning_rate(999999, recipe) == 1e-5
    assert P.learning_rate(999998, recipe) > 1e-5


def test_adamw_schedule_is_stateless_and_monotone_within_each_phase():
    recipe = P.recipe_for(training_recipe="adamw")
    warmup = [P.learning_rate(step, recipe) for step in range(0, 50000, 500)]
    decay = [P.learning_rate(step, recipe) for step in range(49999, 1000000, 1000)]
    assert all(a < b for a, b in zip(warmup, warmup[1:]))
    assert all(a > b for a, b in zip(decay, decay[1:]))
    for step in (60000, 124999, 700000, 999999):
        ordinal = step + 1
        expected = 1e-5 + (1e-3 - 1e-5) * (1 + math.cos(math.pi * (ordinal - 50000) / 950000)) / 2
        assert P.learning_rate(step, recipe) == pytest.approx(expected, rel=1e-14)
        assert P.learning_rate(step, recipe) == P.learning_rate(step, recipe)


@pytest.mark.parametrize("step", [-1, 1000000])
def test_schedule_rejects_updates_outside_budget(step):
    for recipe in (None, P.recipe_for(training_recipe="adamw")):
        with pytest.raises(ValueError, match="budget"):
            P.learning_rate(step, recipe)


@pytest.mark.parametrize("warmup", [0, -1, 1000000, 1000001])
def test_cosine_schedule_rejects_invalid_warmup(warmup):
    recipe = P.recipe_for(training_recipe="adamw")
    recipe["warmup_steps"] = warmup
    with pytest.raises(ValueError, match="Warm-up"):
        P.learning_rate(0, recipe)


def test_unknown_lr_schedule_does_not_silently_use_original_milestones():
    recipe = P.recipe_for(training_recipe="adamw")
    recipe["lr_schedule"] = "unrecognized"
    with pytest.raises(ValueError, match="schedule"):
        P.learning_rate(0, recipe)


def test_pilot_task_index_bounds_follow_its_two_task_registry(tmp_path):
    tasks = P.tasks_for(training_recipe="adamw")
    assert P.task_directory(tmp_path, 0, tasks=tasks) == tmp_path / "runs/part/seed_101"
    assert P.task_directory(tmp_path, 1, tasks=tasks) == tmp_path / "runs/part_ev/seed_101"
    for invalid in (-1, 2, 5):
        with pytest.raises(ValueError, match=r"0\.\.1"):
            P.task_directory(tmp_path, invalid, tasks=tasks)
    assert P.task_directory(tmp_path, 5) == tmp_path / "runs/part_ev/seed_303"
