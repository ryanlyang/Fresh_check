"""The three-seed capacity control is separate from existing comparisons."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from part_ev_fullscale import protocol as P


FROZEN_PROFILES = [
    ("standard", "original",
     "b27c84ee052a8bf1c18a3f7f84fe451db0e88afacbbfd8c3d03312fd9425a106",
     "e53d14e054530c8d1464efdd97c5df1c781cfff3efd40af8083164f906c3687f"),
    ("plain", "original",
     "17331d43b4a9d9fe1a69851d2ab6d480d36195ec99ca2c6c86e244171a9fa097",
     "75a2ec5a306d8413539a068f2216b1999c14fde93bc3dc906c211f4a126b3bf8"),
    ("standard", "adamw",
     "efe96479632471975ec5021278b74dc5a6aa7f0a20e1a0c9641b9705a1ea4b55",
     "e1d5802cc464a47da60e0681ed7b84e40b509457a5c1cd4ab2f06199ef7dc23e"),
]


def write_campaign(output, *, comparison="parameter_matched", training_recipe="original"):
    config = Path(P.__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
    payload = {
        "schema": P.SCHEMA, "comparison": comparison, "training_recipe": training_recipe,
        "recipe": P.recipe_for(comparison, training_recipe),
        "tasks": P.tasks_for(comparison, training_recipe),
        "environment": {"fixture": "pinned runtime"},
        "data_config_sha256": P.file_sha256(config),
    }
    resign(output, payload)
    return payload


def resign(output, payload):
    payload.pop("content_hash", None)
    payload["content_hash"] = P.digest(payload)
    P.write_json(output / "campaign.json", payload)


@pytest.mark.parametrize("comparison,training_recipe,recipe_hash,tasks_hash", FROZEN_PROFILES)
def test_existing_profile_hashes_are_unchanged(comparison, training_recipe, recipe_hash, tasks_hash):
    assert P.digest(P.recipe_for(comparison, training_recipe)) == recipe_hash
    assert P.digest(P.tasks_for(comparison, training_recipe)) == tasks_hash
    assert P.recipe_for() == P.RECIPE
    assert P.tasks_for() == P.TASKS


def test_control_has_only_wider_baseline_for_three_matched_seeds():
    assert P.COMPARISONS["parameter_matched"] == ("part_matched",)
    assert P.tasks_for("parameter_matched") == [
        {"architecture": "part_matched", "seed": seed} for seed in (101, 202, 303)
    ]
    assert P.recipe_for("parameter_matched") == {
        **P.RECIPE, "particle_ffn_ratio": 4.25, "class_ffn_ratio": 4.0,
    }
    assert "particle_ffn_ratio" not in P.RECIPE
    assert "class_ffn_ratio" not in P.RECIPE


def test_matched_profile_keeps_original_learning_rate_budget():
    recipe = P.recipe_for("parameter_matched")
    for step in (0, 49999, 699999, 700000, 719999, 720000, 999999):
        assert P.learning_rate(step, recipe) == P.learning_rate(step)


def test_mutating_helpers_cannot_change_other_calls_or_original_recipe():
    original = copy.deepcopy(P.RECIPE)
    expected_tasks = copy.deepcopy(P.tasks_for("parameter_matched"))
    expected_recipe = copy.deepcopy(P.recipe_for("parameter_matched"))
    recipe = P.recipe_for("parameter_matched")
    recipe["particle_ffn_ratio"] = 8.0
    recipe["betas"][0] = 0.0
    recipe["lr_decay_epochs"].clear()
    tasks = P.tasks_for("parameter_matched")
    tasks[0]["seed"] = 999
    tasks.append({"architecture": "part_ev", "seed": 101})
    assert P.RECIPE == original
    assert P.tasks_for("parameter_matched") == expected_tasks
    assert P.recipe_for("parameter_matched") == expected_recipe


@pytest.mark.parametrize("helper", [P.tasks_for, P.recipe_for])
def test_control_does_not_enable_adamw_or_unknown_recipe(helper):
    with pytest.raises(ValueError, match="only for the standard"):
        helper("parameter_matched", "adamw")
    with pytest.raises(ValueError, match="Unknown training recipe"):
        helper("parameter_matched", "unknown")
    with pytest.raises(ValueError, match="parameter_matched"):
        helper("unknown")


@pytest.mark.parametrize("implicit_original", [False, True])
def test_matched_campaign_load_is_read_only(tmp_path, monkeypatch, implicit_original):
    payload = write_campaign(tmp_path)
    if implicit_original:
        payload.pop("training_recipe")
        resign(tmp_path, payload)
    before = (tmp_path / "campaign.json").read_bytes()
    monkeypatch.setattr(P, "environment", lambda: {"fixture": "pinned runtime"})
    assert P.load_campaign(tmp_path) == payload
    assert (tmp_path / "campaign.json").read_bytes() == before


@pytest.mark.parametrize("mutation", [
    "missing_particle_ratio", "missing_class_ratio", "particle_ratio", "class_ratio",
    "pair_bias", "budget", "optimizer", "missing_comparison", "standard_comparison",
    "plain_comparison", "adamw", "six_tasks", "wrong_architecture", "wrong_seed", "wrong_order",
])
def test_resigned_control_metadata_cannot_change_its_protocol(tmp_path, mutation):
    payload = write_campaign(tmp_path)
    if mutation == "missing_particle_ratio":
        payload["recipe"].pop("particle_ffn_ratio")
    elif mutation == "missing_class_ratio":
        payload["recipe"].pop("class_ffn_ratio")
    elif mutation in ("particle_ratio", "class_ratio"):
        payload["recipe"]["particle_ffn_ratio" if mutation == "particle_ratio" else "class_ffn_ratio"] = 4.5
    elif mutation == "pair_bias":
        payload["recipe"]["pair_bias"] = "none"
    elif mutation == "budget":
        payload["recipe"]["total_steps"] -= 1
    elif mutation == "optimizer":
        payload["recipe"]["optimizer"] = "AdamW"
    elif mutation == "missing_comparison":
        payload.pop("comparison")
    elif mutation.endswith("_comparison"):
        payload["comparison"] = mutation.removesuffix("_comparison")
    elif mutation == "adamw":
        payload["training_recipe"] = "adamw"
    elif mutation == "six_tasks":
        payload["tasks"] = P.tasks_for()
    elif mutation == "wrong_architecture":
        payload["tasks"][0]["architecture"] = "part"
    elif mutation == "wrong_seed":
        payload["tasks"][0]["seed"] = 404
    else:
        payload["tasks"].reverse()
    resign(tmp_path, payload)
    with pytest.raises(ValueError):
        P.load_campaign(tmp_path, check_environment=False)


def test_unsigned_control_edit_fails_integrity_check(tmp_path):
    payload = write_campaign(tmp_path)
    payload["recipe"]["particle_ffn_ratio"] = 4.5
    P.write_json(tmp_path / "campaign.json", payload)
    with pytest.raises(ValueError, match="changed|incompatible"):
        P.load_campaign(tmp_path, check_environment=False)


def test_control_has_separate_paths_and_three_task_bounds(tmp_path):
    tasks = P.tasks_for("parameter_matched")
    for index, seed in enumerate((101, 202, 303)):
        assert P.task_directory(tmp_path, index, tasks=tasks) == tmp_path / "runs/part_matched" / f"seed_{seed}"
    for invalid in (-1, 3, 5):
        with pytest.raises(ValueError, match=r"0\.\.2"):
            P.task_directory(tmp_path, invalid, tasks=tasks)
    assert P.task_directory(tmp_path, 5) == tmp_path / "runs/part_ev/seed_303"
