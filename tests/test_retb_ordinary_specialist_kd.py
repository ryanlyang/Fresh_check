from __future__ import annotations

from pathlib import Path

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_ordinary_specialist_kd import (
    ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT,
    ORDINARY_SPECIALIST_RESUME_CONTRACT,
    ordinary_student_configuration,
    validate_ordinary_specialist_kd_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_specialist_kd import (
    SPECIALIST_CONDITIONS,
    SPECIALIST_EXPERTS,
    STUDENT_COORDINATES,
    specialist_student_configuration,
    specialist_teacher_configuration,
)


def _plan() -> dict:
    records = {
        "compact_plan": {
            "path": "/fixture/plan",
            "file_sha256": "1" * 64,
            "content_hash": "2" * 64,
        },
        "compact_report": {
            "path": "/fixture/report",
            "file_sha256": "3" * 64,
            "content_hash": "4" * 64,
        },
        **{
            f"teacher_{expert}": {
                "path": f"/fixture/{expert}",
                "file_sha256": "5" * 64,
                "content_hash": "6" * 64,
            }
            for expert in ("PT", "TRACK", "REGION")
        },
        **{
            f"teacher_{expert}_{split}": {
                "path": f"/fixture/{expert}/{split}",
                "file_sha256": "9" * 64,
            }
            for expert in ("PT", "TRACK", "REGION")
            for split in ("model_train", "val_stop")
        },
        **{
            f"compact_{condition}_{expert}": {
                "path": f"/fixture/{condition}/{expert}",
                "file_sha256": "7" * 64,
                "content_hash": "8" * 64,
            }
            for condition, expert in STUDENT_COORDINATES
        },
        **{
            f"compact_{condition}_{expert}_{split}": {
                "path": f"/fixture/{condition}/{expert}/{split}",
                "file_sha256": "a" * 64,
            }
            for condition, expert in STUDENT_COORDINATES
            for split in ("val_stop", "val_design")
        },
    }
    return with_content_hash(
        {
            "contract": "retb_ordinary_specialist_kd_plan_v1",
            "schema_version": 1,
            "compact_plan_sha256": "2" * 64,
            "compact_report_sha256": "4" * 64,
            "parent_artifacts": records,
            "experts": list(SPECIALIST_EXPERTS),
            "conditions": list(SPECIALIST_CONDITIONS),
            "student_coordinates": [
                {"condition": condition, "expert": expert, "seed": 101}
                for condition, expert in STUDENT_COORDINATES
            ],
            "student_configurations": {
                f"{condition}:{expert}": ordinary_student_configuration(
                    expert, condition
                )
                for condition, expert in STUDENT_COORDINATES
            },
            "objective": {
                "temperature": 2.0,
                "cross_entropy_weight": 0.25,
                "MATCHED_KD": {
                    "common_teacher_weight": 0.0,
                    "specialist_teacher_weight": 1.0,
                },
                "HYBRID_KD": {
                    "common_teacher_weight": 0.5,
                    "specialist_teacher_weight": 0.5,
                },
            },
            "training_protocol": {
                "maximum_epochs": 40,
                "microbatch_size": 64,
                "gradient_accumulation_steps": 2,
                "effective_batch_size": 128,
                "checkpoint_selection": "val_stop",
                "comparison_split": "val_design",
                "early_stopping": False,
                "performance_based_termination": False,
            },
            "fusion": {
                "method": "MEAN_LOGITS",
                "expert_order": list(SPECIALIST_EXPERTS),
                "learned_parameters": 0,
            },
            "comparison_design": {
                "kd_effect": "ordinary_KD_minus_ordinary_CE_teacher",
                "compression_effect": "compact_KD_minus_ordinary_KD",
                "architectures_differ_only_by_summary_token_bottleneck": True,
                "primary_split": "val_design",
            },
            "final_test_access": False,
            "scientific_underperformance_blocks_execution": False,
        }
    )


def test_ordinary_student_removes_only_the_summary_token_bottleneck() -> None:
    ordinary = ordinary_student_configuration("TRACK", "MATCHED_KD")
    compact = specialist_student_configuration("TRACK", "MATCHED_KD")
    teacher = specialist_teacher_configuration("TRACK")
    assert ordinary["relation_family"] == compact["relation_family"] == "TRACK"
    assert ordinary["tokenizer_mode"] == teacher["tokenizer_mode"] == "TOK_WEAVER_CLASS"
    assert ordinary["ordinary_weaver_head"] is True
    assert ordinary["summary_token_bottleneck_present"] is False
    assert compact["ordinary_weaver_head"] is False
    assert compact["summary_token_bottleneck_present"] is True
    assert ordinary["loss_id"] == compact["loss_id"] == "MATCHED_KD"


def test_ordinary_plan_freezes_the_exact_two_by_four_matrix() -> None:
    plan = _plan()
    assert validate_ordinary_specialist_kd_plan(plan, check_parent_bytes=False)
    assert len(plan["student_coordinates"]) == 8
    changed = dict(plan)
    changed.pop("content_hash")
    changed["final_test_access"] = True
    changed = with_content_hash(changed)
    try:
        validate_ordinary_specialist_kd_plan(changed, check_parent_bytes=False)
    except ValueError as error:
        assert "semantics differ" in str(error)
    else:
        raise AssertionError("final-test access was accepted")


def test_ordinary_checkpoint_contract_cannot_alias_compact_checkpoint(
    tmp_path: Path,
) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Dataset

    from teacher_logit_reco.relation_expert_token_bridge.specialist_kd_training import (
        train_specialist_kd_student,
    )

    class TinyDataset(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict:
            generator = torch.Generator().manual_seed(index)
            return {
                "features": torch.randn(3, 2, generator=generator),
                "labels": torch.tensor(index % 2),
                "teacher_logits": {
                    "COMMON": torch.randn(10, generator=generator),
                    "SPECIALIST": torch.randn(10, generator=generator),
                },
            }

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Linear(3, 10)

        def forward(self, *, features, **_):
            return self.classifier(features.mean(dim=-1))

    loader = DataLoader(TinyDataset(), batch_size=64, shuffle=False)
    train_specialist_kd_student(
        model=TinyModel(),
        train_loader=loader,
        val_stop_loader=loader,
        output_dir=tmp_path,
        condition="MATCHED_KD",
        run_id="ordinary_fixture",
        plan_sha256="9" * 64,
        device=torch.device("cpu"),
        maximum_epochs=1,
        checkpoint_contract=ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT,
        resume_contract=ORDINARY_SPECIALIST_RESUME_CONTRACT,
    )
    checkpoint = torch.load(
        tmp_path / "best_model_val.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["contract"] == ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT
    assert checkpoint["contract"] != "retb_specialist_kd_checkpoint_v1"


def test_submission_runs_all_eight_students_without_array_throttle() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "sbatch/submit_retb_ordinary_specialist_kd.sh").read_text()
    array_lines = [line for line in text.splitlines() if "--array=" in line]
    assert len(array_lines) == 1
    assert "--array=0-7" in array_lines[0]
    assert "%" not in array_lines[0]
    assert 'afterok:${students}' in text
    assert "accuracy" not in text.lower()
    assert "final_test" not in text


def test_compact_ensemble_reproduction_preserves_float32_reduction() -> None:
    import pytest

    np = pytest.importorskip("numpy")
    from scripts.finalize_retb_ordinary_specialist_kd import (
        _compact_mean_logits,
    )

    generator = np.random.default_rng(20260807)
    stacked = None
    legacy = None
    upcast = None
    for _ in range(100):
        candidate = generator.normal(size=(4, 32, 10)).astype(np.float32)
        candidate *= np.float32(
            10.0 ** int(generator.integers(-4, 5))
        )
        candidate_legacy = np.mean(candidate, axis=0).astype(np.float32)
        candidate_upcast = np.mean(
            candidate, axis=0, dtype=np.float64
        ).astype(np.float32)
        if not np.array_equal(candidate_legacy, candidate_upcast):
            stacked = candidate
            legacy = candidate_legacy
            upcast = candidate_upcast
            break
    assert stacked is not None
    result = _compact_mean_logits(
        {
            expert: stacked[index]
            for index, expert in enumerate(SPECIALIST_EXPERTS)
        }
    )
    assert np.array_equal(result, legacy)
    assert not np.array_equal(result, upcast)


def test_workers_use_the_frozen_project_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "run_retb_ordinary_specialist_kd_bootstrap.sh",
        "run_retb_ordinary_specialist_kd_student.sh",
        "run_retb_ordinary_specialist_kd_finalize.sh",
    ):
        text = (root / "sbatch" / name).read_text()
        assert 'source "${PROJECT_DIR}/sbatch/retb_common.sh"' in text
        assert 'dirname "${BASH_SOURCE[0]}"' not in text

    finalizer = (
        root / "sbatch/run_retb_ordinary_specialist_kd_finalize.sh"
    ).read_text()
    assert "RETB_ORDINARY_SPECIALIST_KD_FINALIZER_RECOVERY" in finalizer
    recovery = (
        root / "sbatch/submit_retb_ordinary_specialist_kd_finalize_recovery.sh"
    ).read_text()
    assert 'git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel' in recovery
    assert 'PROJECT_DIR="${SCRIPT_PROJECT_DIR}"' in recovery
    assert 'git -C "${PROJECT_DIR}" rev-parse --show-toplevel' not in recovery
    assert 'source "${SCRIPT_DIR}/retb_common.sh"' in recovery
    assert "worktree add --detach" in recovery
    assert "--array=" not in recovery
    assert "run_retb_ordinary_specialist_kd_finalize.sh" in recovery
    assert "RETB_ORDINARY_SPECIALIST_KD_FINALIZER_RECOVERY=1" in recovery
