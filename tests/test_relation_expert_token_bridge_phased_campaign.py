from pathlib import Path
from types import SimpleNamespace

import pytest

from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (
    build_internal_phase_plan,
    execute_internal_phase,
    execute_phased_controller,
    phase_row_completion_path,
    reusable_internal_phase_completion,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SOURCE = {"git_commit": "1" * 40, "source_tree_sha256": "c" * 64}


def _plan(
    root: Path,
    *,
    phase: str,
    index: int,
    prerequisites: dict[str, str],
    count: int = 2,
):
    rows = []
    for task_index in range(count):
        output = root / "outputs" / phase / f"{task_index}.txt"
        rows.append(
            {
                "task_id": f"{phase}:{task_index}",
                "argv": [
                    "python",
                    "scripts/unit_test_phase_worker.py",
                    str(output),
                ],
                "environment": {},
                "expected_outputs": [str(output)],
                "input_artifact_hashes": {
                    "campaign_spec": SHA_A,
                    **prerequisites,
                },
            }
        )
    return build_internal_phase_plan(
        campaign_root=root,
        campaign_spec_sha256=SHA_A,
        production_graph_sha256=SHA_B,
        controller_id="predictor_campaign",
        phase_id=phase,
        sequence_index=index,
        resource="gpu",
        maximum_concurrent_tasks=2,
        rows=rows,
        prerequisite_completion_hashes=prerequisites,
        source=SOURCE,
    )


def test_internal_phase_executes_reuses_and_rejects_output_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(
        tmp_path, phase="F_SCREEN", index=0, prerequisites={}
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        output = Path(argv[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"row-{len(calls)}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "teacher_logit_reco.relation_expert_token_bridge.phased_campaign."
        "subprocess.run",
        run,
    )
    completion = execute_internal_phase(
        plan=plan,
        repo_root=tmp_path,
        slurm_task_script=tmp_path / "unused.sh",
    )
    assert len(calls) == 2
    assert completion["scientific_result_sign_used_as_completion_gate"] is False
    assert (
        reusable_internal_phase_completion(plan=plan)["content_hash"]
        == completion["content_hash"]
    )
    execute_internal_phase(
        plan=plan,
        repo_root=tmp_path,
        slurm_task_script=tmp_path / "unused.sh",
    )
    assert len(calls) == 2
    Path(plan["rows"][0]["expected_outputs"][0]).write_text(
        "drift", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lineage differs"):
        reusable_internal_phase_completion(plan=plan)


def test_controller_builds_later_phase_only_after_authenticated_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []

    def run(argv, **kwargs):
        output = Path(argv[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("complete", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "teacher_logit_reco.relation_expert_token_bridge.phased_campaign."
        "subprocess.run",
        run,
    )

    def builder(phase, index, prerequisites):
        observed.append((phase, dict(prerequisites)))
        return _plan(
            tmp_path,
            phase=phase,
            index=index,
            prerequisites=dict(prerequisites),
            count=1,
        )

    completion = execute_phased_controller(
        campaign_root=tmp_path,
        controller_id="predictor_campaign",
        phase_ids=("F_SCREEN", "F_SELECT", "G_SCREEN"),
        phase_builder=builder,
        repo_root=tmp_path,
        slurm_task_script=tmp_path / "unused.sh",
        output_path=tmp_path / "predictor_campaign_completion.json",
    )
    assert observed[0] == ("F_SCREEN", {})
    assert set(observed[1][1]) == {"F_SCREEN"}
    assert set(observed[2][1]) == {"F_SCREEN", "F_SELECT"}
    assert completion["phase_order"] == [
        "F_SCREEN",
        "F_SELECT",
        "G_SCREEN",
    ]
    assert completion["scientific_result_sign_used_as_continuation_gate"] is False
    assert phase_row_completion_path(
        tmp_path,
        controller_id="predictor_campaign",
        phase_id="G_SCREEN",
        task_index=0,
    ).is_file()
