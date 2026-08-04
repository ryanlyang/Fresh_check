from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from teacher_logit_reco.relation_expert_token_bridge.specialist_kd_training import (
    train_specialist_kd_student,
)

from teacher_logit_reco.relation_expert_token_bridge.supplemental_specialist_kd import (
    NEW_SPECIALIST_TEACHERS,
    SPECIALIST_CONDITIONS,
    SPECIALIST_EXPERTS,
    STUDENT_COORDINATES,
    pairwise_diversity,
    specialist_kd_objective,
    specialist_student_configuration,
    specialist_teacher_configuration,
)


def test_specialist_teacher_and_student_coordinates_are_exact() -> None:
    assert NEW_SPECIALIST_TEACHERS == ("PT", "TRACK", "REGION")
    assert len(STUDENT_COORDINATES) == 8
    assert set(STUDENT_COORDINATES) == {
        (condition, expert)
        for condition in SPECIALIST_CONDITIONS
        for expert in SPECIALIST_EXPERTS
    }
    teacher = specialist_teacher_configuration("PT")
    student = specialist_student_configuration("PT", "MATCHED_KD")
    assert teacher["tokenizer_mode"] == "TOK_WEAVER_CLASS"
    assert teacher["summary_token_bottleneck_present"] is False
    assert student["tokenizer_mode"] == "TOK_CANONICAL"
    assert student["summary_token_bottleneck_present"] is True


def test_specialist_kd_objectives_match_frozen_equations_and_backpropagate() -> None:
    torch.manual_seed(4)
    logits = torch.randn(5, 10, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 4])
    common = torch.randn(5, 10)
    specialist = torch.randn(5, 10)

    def kd(target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.kl_div(
            torch.log_softmax(logits / 2.0, dim=-1),
            torch.softmax(target / 2.0, dim=-1),
            reduction="batchmean",
        ) * 4.0

    ce = torch.nn.functional.cross_entropy(logits, labels)
    matched, _ = specialist_kd_objective(
        logits,
        labels,
        condition="MATCHED_KD",
        common_teacher_logits=common,
        specialist_teacher_logits=specialist,
    )
    hybrid, _ = specialist_kd_objective(
        logits,
        labels,
        condition="HYBRID_KD",
        common_teacher_logits=common,
        specialist_teacher_logits=specialist,
    )
    assert torch.allclose(matched, 0.25 * ce + kd(specialist))
    assert torch.allclose(hybrid, 0.25 * ce + 0.5 * kd(common) + 0.5 * kd(specialist))
    hybrid.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_pairwise_diversity_has_all_six_pairs() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = {
        "BASE4": np.array([[3, 0], [3, 0], [0, 3], [0, 3]], dtype=float),
        "PT": np.array([[3, 0], [0, 3], [0, 3], [0, 3]], dtype=float),
        "TRACK": np.array([[2, 0], [2, 0], [0, 2], [2, 0]], dtype=float),
        "REGION": np.array([[4, 0], [4, 0], [4, 0], [0, 4]], dtype=float),
    }
    rows = pairwise_diversity(predictions, labels)
    assert len(rows) == 6
    base_pt = next(row for row in rows if row["left"] == "BASE4" and row["right"] == "PT")
    assert base_pt["prediction_disagreement"] == 0.25
    assert base_pt["correctness_disagreement"] == 0.25


def test_submission_maximizes_safe_parallelism_without_performance_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "sbatch/submit_retb_specialist_kd.sh").read_text()
    assert "--array=0-2" in text
    assert "--array=0,4" in text
    assert "--array=1-3,5-7" in text
    assert "%" not in "\n".join(line for line in text.splitlines() if "--array=" in line)
    assert 'afterok:${bootstrap}' in text
    assert 'afterok:${teachers}' in text
    assert 'afterok:${base4_students}:${relation_students}' in text
    assert "accuracy" not in text.lower()
    assert "final_test" not in text


class _TinyDataset(Dataset):
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


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3, 10)

    def forward(self, *, features: torch.Tensor, **_: object) -> torch.Tensor:
        return self.classifier(features.mean(dim=-1))


def test_fixed_budget_specialist_trainer_runs_and_cleans_resume_state(tmp_path: Path) -> None:
    loader = DataLoader(_TinyDataset(), batch_size=64, shuffle=False)
    result = train_specialist_kd_student(
        model=_TinyModel(),
        train_loader=loader,
        val_stop_loader=loader,
        output_dir=tmp_path,
        condition="HYBRID_KD",
        run_id="fixture",
        plan_sha256="1" * 64,
        device=torch.device("cpu"),
        maximum_epochs=2,
    )
    assert len(result["rows"]) == 2
    assert (tmp_path / "best_model_val.pt").is_file()
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / ".checkpoint_frontier").exists()
