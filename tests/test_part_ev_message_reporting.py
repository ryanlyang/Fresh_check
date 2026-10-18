from types import SimpleNamespace
import copy
import numpy as np
import pytest
from part_ev_message_study import reporting as R, protocol as P


def test_five_models_vs_both_controls_and_descriptive_interaction():
    values = dict(PART_RANGER=.8,EV_RANGER=.81,EV_STATE_RANGER=.82,
                  EV_SEPARATE_RANGER=.83,EV_STATE_SEPARATE_RANGER=.85,
                  EV_ROUTING_RANGER=.815,EV_VARIANCE_RANGER=.79)
    effects = R.contrasts(values)
    assert len(effects) == 13
    assert effects["state_separate_interaction"] == pytest.approx(.01)
    assert effects["EV_VARIANCE_RANGER_minus_EV_RANGER"] == pytest.approx(-.02)
    assert effects["EV_ROUTING_RANGER_minus_PART_RANGER"] == pytest.approx(.015)


def test_paired_correctness_keeps_common_events(tmp_path):
    a,b = tmp_path/"a",tmp_path/"b"
    a.mkdir()
    b.mkdir()
    truth = np.array([0,1,2,3])
    left,right = np.eye(10)[[0,1,0,0]],np.eye(10)[[0,0,2,0]]
    np.savez(a/"file_000.npz",logits=left,labels=truth)
    np.savez(b/"file_000.npz",logits=right,labels=truth)
    assert R.paired_counts(a,b,1) == dict(new_only_correct=1,control_only_correct=1,both_correct=1,both_wrong=1)
    np.savez(b/"file_000.npz",logits=right,labels=truth[::-1])
    with pytest.raises(ValueError,match="labels"):
        R.paired_counts(a,b,1)


def test_report_join_requests_verified_complete_controls_and_includes_all_models(tmp_path,monkeypatch):
    metrics = dict(accuracy=.8,macro_auc_ovo=.98,macro_auc_ovr=.99,loss=.3,
                   qcd_rejection={"Hbb":{"0.3":dict(rejection=None,qcd_accepted=0)}})
    row = dict(metrics=metrics,best_check=10,validation_checks=[metrics]*10,
               training=dict(train_seconds=10,amp_skipped_steps=0))
    registry = dict(content_hash="controls",controls={name:copy.deepcopy(row) for name in ("PART_RANGER","EV_RANGER")})
    labels = {name:np.array([0,1]) for name in registry["controls"]}
    spec = dict(content_hash="new",donor=dict(path=str(tmp_path/"donor"),runtime={}),
                inventory=dict(splits=dict(val=dict(files=[]))))
    calls = []
    def refresh(output,supplied,**kw):
        calls.append(kw)
        return registry,labels
    monkeypatch.setattr(R.C,"validate_training",lambda *a:{})
    monkeypatch.setattr(R.C,"validate_evaluation",lambda *a,**kw:(copy.deepcopy(row),np.array([0,1])))
    monkeypatch.setattr(R,"paired_counts",lambda *a:dict(new_only_correct=0,control_only_correct=0))
    for task in P.TASKS:
        P.write_json(tmp_path/"benchmarks"/(task["run_id"]+".json"),dict(jets_per_second=10,full_length_peak_bytes=1024**3))
    engine = SimpleNamespace(classification_metrics=lambda *a:None)
    result = R.report(tmp_path,spec,engine,refresh)
    assert calls == [dict(require_complete=True,recompute=True)]
    assert len(result["models"]) == 7
    assert len(result["paired_correctness"]) == 10
    document = (tmp_path/"reports/message_study_report.md").read_text()
    assert "ZB" in document
    assert all(task["run_id"] in document for task in P.TASKS)
    assert result["population"] == "official_validation_5M"
