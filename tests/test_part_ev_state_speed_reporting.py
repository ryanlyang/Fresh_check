from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from part_ev_state_speed import reporting as R, protocol as P
from tests.test_part_ev_state_speed_protocol import fixture


@pytest.mark.parametrize("mismatched",[False, True])
def test_report_uses_all_full_population_results_not_training_checks(tmp_path,monkeypatch,mismatched):
    spec = fixture(tmp_path)
    spec["state_reference"]["path"] = str(tmp_path/"state")
    state_task = dict(run_id="EV_STATE_RANGER",seed=101,architecture="ev_state",recipe="ranger_short_v1")
    def record(name):
        acc = {"PART_RANGER":.85,"EV_RANGER":.851,"EV_STATE_RANGER":.852}.get(name,.8519)
        return dict(metrics=dict(accuracy=acc,macro_auc_ovo=.98),best_check=10,
                    validation_checks=[dict(accuracy=.99)]*10,training=dict(train_seconds=3600))
    labels = np.array([0,1,2])
    monkeypatch.setattr(R.C,"donor_contract",lambda _: ({},spec["donor"]))
    monkeypatch.setattr(R.C,"registry",lambda *a,**k: (dict(controls={n:record(n) for n in ("PART_RANGER","EV_RANGER")}),
                                                       {n:labels for n in ("PART_RANGER","EV_RANGER")}))
    monkeypatch.setattr(R.historical,"load_campaign",lambda _:dict(tasks=[state_task]))
    monkeypatch.setattr(R.C,"validate_training",lambda *a: {})
    def evaluation(root, campaign, task, *args, **kwargs):
        truth = labels[::-1] if mismatched and task["run_id"] == P.TASKS[2]["run_id"] else labels
        return record(task["run_id"]), truth
    monkeypatch.setattr(R.C,"validate_evaluation",evaluation)
    summary = P.benchmark_summary(tmp_path,spec)
    for bench in summary["models"]:
        bench["jets_per_second"] = 800
    monkeypatch.setattr(P,"require_approval",lambda *a:summary)
    engine = SimpleNamespace(classification_metrics=lambda *a: {})
    if mismatched:
        with pytest.raises(ValueError,match="identities"):
            R.report(tmp_path,spec,engine)
        assert not (tmp_path/"reports/state_speed_report.json").exists()
    else:
        result = R.report(tmp_path,spec,engine)
        assert len(result["models"]) == 6
        assert result["population"] == "official_validation_5M"
        assert result["accuracy_deltas"][P.TASKS[0]["run_id"]]["EV_RANGER"] == pytest.approx(.0009)
        text = (tmp_path/"reports/state_speed_report.md").read_text()
        assert "99.0000" not in text and "85.1900" in text
        assert all(t["run_id"] in text for t in P.TASKS)
        assert "Time / original state" in text
