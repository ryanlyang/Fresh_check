from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    build_stack_task_specs,
    canonical_sha256,
    load_hashed_json,
    run_stack_evaluation,
    run_stack_fusion,
    with_content_hash,
)
from teacher_logit_reco.local_particle_residual_field.particle_view import (
    stack_runtime,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _authorization(*hashes: str):
    return with_content_hash(
        {
            "contract": "particle_view_split_authorization_v1",
            "selection_sha256": _sha("selection"),
            "fairness_ledger_sha256": _sha("fairness"),
            "stack_val": {
                "split_sha256": _sha("stack"),
                "authorized_bundles": [
                    {
                        "bundle_sha256": value,
                        "seed": 101,
                        "role": "test",
                        "winner_families": [],
                    }
                    for value in hashes
                ],
                "may_select_or_replace_winner": False,
            },
            "final_test": {
                "split_sha256": _sha("final"),
                "authorized_bundles": [],
                "authorized_fusion_recipes": [],
                "hlt_only_required": True,
                "stage_g_controls_forbidden": True,
            },
        }
    )


class _ToyHLT(torch.nn.Module):
    def __init__(self, offset: tuple[float, float]):
        super().__init__()
        self.register_buffer("offset", torch.tensor(offset))

    def forward(self, points, features, lorentz_vectors, mask):
        return features[:, :2, 0] + self.offset


def _source(name: str, offset: tuple[float, float]):
    return {
        "kind": "direct",
        "model": _ToyHLT(offset),
        "bundle_sha256": _sha(name),
        "configuration_id": name,
        "seed": 101,
    }


def _loader():
    batches = []
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    values = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.2, 0.3],
            [0.3, 0.2],
            [1.0, 0.8],
            [0.8, 1.0],
        ]
    )
    for start in (0, 3):
        selected = slice(start, start + 3)
        features = torch.zeros(3, 2, 1)
        features[:, :, 0] = values[selected]
        batches.append(
            {
                "points": torch.zeros(3, 2, 1),
                "features": features,
                "lorentz_vectors": torch.zeros(3, 4, 1),
                "mask": torch.ones(3, 1, 1, dtype=torch.bool),
                "labels": labels[selected],
                "parent_indices": torch.arange(start, start + 3),
            }
        )
    return batches


def test_pv08_stack_evaluation_is_hlt_only_paired_and_nonselecting(tmp_path):
    baseline = _source("a0", (0.0, 0.0))
    candidate = _source("candidate", (0.0, 0.25))
    authorization = _authorization(
        baseline["bundle_sha256"], candidate["bundle_sha256"]
    )
    run_stack_evaluation(
        evaluations=[
            {
                "candidate": candidate,
                "baseline": baseline,
                "role": "preselected_winner_replica",
                "winner_family": "PRIVILEGED_SCIENTIFIC",
            }
        ],
        loader=_loader(),
        authorization=authorization,
        output_dir=str(tmp_path),
        class_names=("a", "b"),
        stack_split_sha256=_sha("stack"),
        full_stack_identity_sha256=canonical_sha256(
            [str(value) for value in range(6)]
        ),
        expected_stack_count=6,
        device="cpu",
        max_stack_batches=None,
        bootstrap_replicates=25,
    )
    result = load_hashed_json(tmp_path / "stack_evaluation.json")
    assert result["selection_changed"] is False
    assert result["final_test_loaded"] is False
    assert result["warnings"] == []
    assert result["rows"][0]["complete_stack_split"]
    assert result["paired_statistics"][0]["event_count"] == 6


def test_pv08_fusion_uses_disjoint_fit_eval_and_freezes_recipe(tmp_path):
    left = _source("left", (0.0, 0.0))
    right = _source("right", (0.1, -0.1))
    authorization = _authorization(
        left["bundle_sha256"], right["bundle_sha256"]
    )
    run_stack_fusion(
        sources=(left, right),
        loader=_loader(),
        authorization=authorization,
        output_dir=str(tmp_path),
        fusion_id="PRIVILEGED_LINEAR_FUSION",
        method="linear_logit",
        class_names=("a", "b"),
        stack_split_sha256=_sha("stack"),
        expected_stack_count=6,
        device="cpu",
        max_stack_batches=None,
        linear_fusion_steps=10,
    )
    partition = load_hashed_json(tmp_path / "stack_partition.json")
    recipe = load_hashed_json(tmp_path / "fusion_recipe.json")
    report = load_hashed_json(tmp_path / "fusion_report.json")
    assert set(partition["fit_indices"]).isdisjoint(
        partition["evaluation_indices"]
    )
    assert recipe["winner_selection_permitted"] is False
    assert report["evaluation_only"]
    assert report["evaluation_event_count"] == 3


def test_pv08_optional_p7b_absence_warns_and_does_not_fail(tmp_path):
    left = _source("left", (0.0, 0.0))
    right = _source("right", (0.1, -0.1))
    authorization = _authorization(
        left["bundle_sha256"], right["bundle_sha256"]
    )
    run_stack_fusion(
        sources=(left, right),
        loader=_loader(),
        authorization=authorization,
        output_dir=str(tmp_path),
        fusion_id="OPTIONAL_P7B_FUSION",
        method="logit_average",
        class_names=("a", "b"),
        stack_split_sha256=_sha("stack"),
        expected_stack_count=6,
        device="cpu",
        max_stack_batches=None,
        linear_fusion_steps=10,
        optional_p7b_resource=None,
    )
    status = load_hashed_json(tmp_path / "fusion_status.json")
    assert status["status"] == "not_run"
    assert status["warning_is_non_gating"]
    assert not (tmp_path / "fusion_recipe.json").exists()


def test_pv08_task_specs_cover_all_19_registry_runs(monkeypatch, tmp_path):
    config = tmp_path / "stack_factory.json"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        stack_runtime, "load_hashed_json", lambda _path: {}
    )
    monkeypatch.setattr(
        stack_runtime, "validate_stack_factory_config", lambda _payload: {}
    )
    specs = build_stack_task_specs(factory_config_path=config)
    assert len(specs) == 19
    assert specs["STACK_MATCHED_CE_ONLY_COMPARATOR"]["operation"] == (
        "stack_evaluation"
    )
    assert specs["STACK_PRIVILEGED_LINEAR_FUSION"]["operation"] == "fusion"
    assert sum(
        row["operation"] == "stack_evaluation" for row in specs.values()
    ) == 11
    assert sum(row["operation"] == "fusion" for row in specs.values()) == 8


def test_pv08_tigris_wrapper_requests_gpu_and_correct_environment():
    text = Path("sbatch/run_particle_view_fusion.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --account=reu-aisocial" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=1-00:00:00" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "CONDA_ENV:=atlas_kd_tigris" in text
