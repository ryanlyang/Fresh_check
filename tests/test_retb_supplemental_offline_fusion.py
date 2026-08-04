from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.run_retb_supplemental_offline_fusion_bank import (
    _LogitLinear,
    _PooledMLP,
    _TokenTransformer,
    _mean_logits,
)

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (
    build_stage_b_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import (
    BANK_DEFINITIONS,
    FUSION_VARIANTS,
    SEVEN_SEEDS,
    build_supplemental_plan,
    file_sha256,
    resolve_bank_parent,
    select_fixed_budget_checkpoint,
    validate_supplemental_plan,
)


SOURCE = {
    "source_commit": "1" * 40,
    "source_dirty": False,
    "source_status_sha256": "2" * 64,
}


def _fixture_parent(root: Path) -> Path:
    parent = root / "parent"
    campaign = with_content_hash(
        {
            "contract": "fixture_campaign",
            "campaign_id": "fixture-production",
            "campaign_profile": "production_500k_scale3m",
            "source": SOURCE,
        }
    )
    write_immutable_json(parent / "campaign_spec.json", campaign)
    registry = build_stage_b_run_registry()
    write_immutable_json(parent / "registry/retb_stage_b_runs.json", registry)
    for role in ("model_train", "val_stop", "val_design"):
        split = parent / "inputs/offline" / role
        split.mkdir(parents=True, exist_ok=True)
        npz = split / "offline_inputs.npz"
        npz.write_bytes((role + "-fixture").encode())
        write_immutable_json(
            split / "offline_input_manifest.json",
            with_content_hash(
                {
                    "contract": "fixture_offline_input",
                    "npz_filename": npz.name,
                    "npz_sha256": file_sha256(npz),
                }
            ),
        )
    for relative in (
        "inputs/normalization/offline_500k/relation.json",
        "inputs/normalization/offline_500k/region.json",
        "inputs/input_audit.json",
    ):
        write_immutable_json(
            parent / relative,
            with_content_hash({"contract": "fixture_parent_artifact"}),
        )
    unique = {item for definition in BANK_DEFINITIONS.values() for item in definition}
    for expert_id, loss_id in unique:
        row = resolve_bank_parent(
            registry, expert_id=expert_id, loss_id=loss_id
        )
        run = parent / "runs/stage_b" / row["run_id"] / "seed_101"
        run.mkdir(parents=True, exist_ok=True)
        checkpoint = run / "best_model_val.pt"
        checkpoint.write_bytes(f"{expert_id}/{loss_id}".encode())
        write_immutable_json(
            run / "checkpoint_registration.json",
            with_content_hash(
                {
                    "contract": "retb_offline_expert_registration_v1",
                    "run_id": row["run_id"],
                    "expert_id": expert_id,
                    "shape_id": "S8_128",
                    "seed": 101,
                    "fixed_epoch_budget_completed": True,
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "source": SOURCE,
                }
            ),
        )
    return parent


def test_exact_fast_track_banks_and_obase7_seeds() -> None:
    assert [name for name, _ in BANK_DEFINITIONS["CE4"]] == [
        "BASE4", "PT", "TRACK", "REGION"
    ]
    assert all(loss == "ELOSS_CE" for _, loss in BANK_DEFINITIONS["CE7"])
    assert BANK_DEFINITIONS["KD3"] == (
        ("BASE4", "ELOSS_KD_DOMINANT"),
        ("PT", "ELOSS_KD_DOMINANT"),
        ("TRACK", "ELOSS_KD_DOMINANT"),
    )
    assert dict(BANK_DEFINITIONS["MIXED7"])["REGION"] == "ELOSS_KD_DOMINANT"
    assert dict(BANK_DEFINITIONS["MIXED7"])["PID"] == "ELOSS_CE"
    assert SEVEN_SEEDS == (101, 202, 303, 404, 505, 606, 707)
    assert FUSION_VARIANTS == (
        "MEAN_LOGITS",
        "TRAINED_LOGIT_LINEAR",
        "POOLED_MLP",
        "TOKEN_TRANSFORMER",
    )


def test_plan_byte_binds_parent_and_rejects_drift(tmp_path: Path) -> None:
    parent = _fixture_parent(tmp_path)
    plan = build_supplemental_plan(
        parent_root=parent,
        supplemental_id="supplemental-fixture",
        source_snapshot=SOURCE,
    )
    validate_supplemental_plan(plan)
    assert plan["final_test_access"] is False
    assert plan["scientific_underperformance_blocks_execution"] is False
    assert plan["obase7"]["reuse_stage_a_seed_101"] is False
    checkpoint = Path(plan["banks"]["KD4"]["members"][-1]["checkpoint_path"])
    checkpoint.write_bytes(b"drift")
    with pytest.raises(ValueError, match="parent bytes drifted"):
        validate_supplemental_plan(plan)


def test_ready_and_late_plans_split_dependencies_without_overlap(
    tmp_path: Path,
) -> None:
    parent = _fixture_parent(tmp_path)
    ready = build_supplemental_plan(
        parent_root=parent,
        supplemental_id="supplemental-fixture",
        source_snapshot=SOURCE,
        plan_role="ready",
    )
    late = build_supplemental_plan(
        parent_root=parent,
        supplemental_id="supplemental-fixture",
        source_snapshot=SOURCE,
        plan_role="late",
    )
    validate_supplemental_plan(ready)
    validate_supplemental_plan(late)
    assert ready["bank_order"] == ["CE4", "CE7", "KD3"]
    assert late["bank_order"] == ["KD4", "MIXED7"]
    assert ready["obase7"]["member_seeds"] == list(SEVEN_SEEDS)
    assert late["obase7"] is None
    assert set(ready["banks"]).isdisjoint(late["banks"])
    assert set(ready["banks"]) | set(late["banks"]) == set(BANK_DEFINITIONS)


def test_fixed_budget_selection_uses_accuracy_window_then_ce_then_epoch() -> None:
    rows = [
        {"epoch": 1, "val_stop": {"accuracy": 0.8000, "cross_entropy": 0.50}},
        {"epoch": 2, "val_stop": {"accuracy": 0.8001, "cross_entropy": 0.51}},
        {"epoch": 3, "val_stop": {"accuracy": 0.8000, "cross_entropy": 0.49}},
    ]
    assert select_fixed_budget_checkpoint(rows) == 3


def test_submission_has_unthrottled_parallel_arrays_and_dependency() -> None:
    text = Path("sbatch/submit_retb_supplemental_offline_fusion.sh").read_text()
    assert "--array=0-2" in text
    assert "--array=0-1" in text
    assert "--array=0-6" in text
    assert "%" not in "\n".join(
        line for line in text.splitlines() if "--array=" in line
    )
    assert "RETB_REGION_KD_JOB_ID" in text
    assert 'afterok:${RETB_REGION_KD_JOB_ID}' in text
    assert 'afterok:${ready_fusion}:${late_fusion}:${obase}' in text
    assert "status --porcelain" not in text
    assert "resume.unlink(missing_ok=True)" in Path(
        "scripts/train_retb_supplemental_obase7_member.py"
    ).read_text()


def test_workers_never_reference_final_test() -> None:
    paths = (
        "scripts/run_retb_supplemental_offline_fusion_bank.py",
        "scripts/train_retb_supplemental_obase7_member.py",
        "scripts/finalize_retb_supplemental_offline_fusion.py",
    )
    for path in paths:
        text = Path(path).read_text()
        assert "inputs/final_test" not in text
        assert '"final_test"' not in text


@pytest.mark.parametrize("bank_id", tuple(BANK_DEFINITIONS))
def test_all_flexible_fusion_heads_accept_exact_bank(bank_id: str) -> None:
    order = [name for name, _ in BANK_DEFINITIONS[bank_id]]
    batch = {
        "tokens": {
            name: torch.randn(2, 8, 128) for name in order
        },
        "logits": {name: torch.randn(2, 10) for name in order},
        "labels": torch.tensor([0, 1]),
    }
    dimensions = {name: 128 for name in order}
    assert _mean_logits(batch, order).shape == (2, 10)
    assert _LogitLinear(order)(batch).shape == (2, 10)
    assert _PooledMLP(order, dimensions)(batch).shape == (2, 10)
    assert _TokenTransformer(order, dimensions)(batch).shape == (2, 10)
