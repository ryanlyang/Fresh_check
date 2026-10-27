import copy
from pathlib import Path
import pytest
import torch
from part_ev_state_capacity import protocol as P, engine as E, reporting as R
from tests.test_part_ev_state_fullscale_reporting import training_fixture


def fixture(root):
    spec,_,runtime,run=training_fixture(root/'fixture')
    old=dict(spec,content_hash='old',tasks=copy.deepcopy(P.old.TASKS),parameters=P.old.COUNTS)
    new=dict(spec,content_hash='new',tasks=copy.deepcopy(P.TASKS),parameters=P.COUNTS,
        controls=dict(path=str(root/'old'),campaign_sha256='old',runtime={'gpu':'A100'}))
    for parent,source,tasks in ((root/'old',old,old['tasks']),(root/'new',new,new['tasks'][:6])):
        P.write_json(parent/'campaign.json',source)
        for task in tasks:
            target=parent/'runs'/task['run_id']/f"seed_{task['seed']}"
            for check in (run/'validation').glob('*.json'):
                P.write_json(target/'validation'/check.name,P.read_json(check))
            state=torch.load(run/'best_model_val.pt',weights_only=False)
            state.update(task=task,campaign_sha256=source['content_hash'],runtime=dict(runtime,campaign_sha256=source['content_hash']))
            torch.save(state,target/'best_model_val.pt')
            done=P.read_json(run/'training_complete.json')
            done.update(task=task,campaign_sha256=source['content_hash'],runtime=state['runtime'],
                        best_checkpoint_sha256=P.file_sha256(target/'best_model_val.pt'))
            P.write_json(target/'training_complete.json',done)
            opt=P.read_json(run/'optimizer.json'); opt['parameters']=source['parameters'][task['architecture']]
            P.write_json(target/'optimizer.json',opt)
    return new


def test_lock_byte_copies_controls_and_preserves_provenance(tmp_path):
    spec=fixture(tmp_path)
    before={str(p):P.file_sha256(p) for p in (tmp_path/'old').rglob('*') if p.is_file()}
    locked=E.lock_finalists(tmp_path/'new',spec)
    assert len(locked['checkpoints'])==10
    for i,row in enumerate(locked['checkpoints']):
        assert row['training_campaign_sha256']==('new' if i<6 else 'old')
        saved=torch.load(row['checkpoint'],weights_only=False)
        assert saved['campaign_sha256']==row['training_campaign_sha256']
        assert row['reused']==(i>=6)
    assert E.lock_finalists(tmp_path/'new',spec)==locked
    assert {str(p):P.file_sha256(p) for p in (tmp_path/'old').rglob('*') if p.is_file()}==before
    checkpoint=Path(locked['checkpoints'][6]['checkpoint'])
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checkpoint changed'): E.lock_finalists(tmp_path/'new',spec)


def test_incomplete_control_blocks_lock_without_partial_import(tmp_path):
    spec=fixture(tmp_path)
    (tmp_path/'old/runs/EV_RANGER/seed_101/validation/check_50.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)): E.lock_finalists(tmp_path/'new',spec)
    assert not (tmp_path/'new/runs/PART_RANGER').exists()
    assert not (tmp_path/'new/selection/locked_checkpoints.json').exists()


def test_capacity_contrasts_are_three_seed_paired():
    models={}
    for task in P.TASKS:
        acc={'part':.85,'part_ev':.851,'state_fast':.853,'part_state_matched':.852}[task['architecture']]
        models[P.task_key(task)]=dict(task=task,metrics=dict(accuracy=acc,macro_auc_ovo=acc,macro_auc_ovr=acc,loss=1-acc))
    rows=R.summarize(models)['accuracy']
    assert rows['state_minus_matched']['values']==pytest.approx([.001]*3)
    assert rows['state_minus_part']['mean']==pytest.approx(.003)
    assert rows['state_minus_ev_seed101']==pytest.approx(.002)


def test_report_uses_original_training_provenance(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    spec=fixture(tmp_path)
    monkeypatch.setattr(P,'require_approval',lambda *a:{})
    def evaluated(root,spec,task,trained,*,recompute):
        trained=copy.deepcopy(trained)
        trained['training'].update(train_seconds=3600,validation_seconds=60,amp_skipped_steps=2)
        trained['metrics']=dict(accuracy=.85,macro_auc_ovo=.98,macro_auc_ovr=.98,loss=.3,
                                qcd_rejection={'Hbb':{'0.5':dict(rejection=10000)}})
        return trained,np.arange(10)
    monkeypatch.setattr(R,'validate_evaluation',evaluated)
    result=R.report(tmp_path/'new',spec,SimpleNamespace(classification_metrics=None))
    assert len(result['models'])==10 and result['ordinary_ev_seeds']==[101]
    for task in P.TASKS[6:]:
        assert result['models'][P.task_key(task)]['training_source']['campaign_sha256']=='old'
    text=(tmp_path/'new/reports/state_capacity_report.md').read_text()
    assert '| 50 |' in text and 'not independent testing' in text and 'Recorded cost' in text
