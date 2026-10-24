from types import SimpleNamespace
import numpy as np
import pytest
import torch
from part_ev_state_fullscale import protocol as P, reporting as R


def training_fixture(root):
    task = P.TASKS[0]
    runtime = dict(gpu='A100',campaign_sha256='fixture')
    spec = dict(content_hash='fixture',recipes=P.recipes(),parameters=P.COUNTS,tasks=P.TASKS)
    run = P.task_directory(root,0)
    checks = []
    for i in range(1,51):
        check = dict(check=i,step=i*20000,events=1280000,accuracy=.8+min(i,49)/1000,
                     macro_auc_ovo=.98,macro_auc_ovr=.98,loss=.2)
        P.write_json(run/'validation'/f'check_{i:02d}.json',check)
        checks.append(check)
    path = run/'best_model_val.pt'
    torch.save(dict(task=task,campaign_sha256='fixture',runtime=runtime,validation=checks[48],model={}),path)
    complete = dict(task=task,step=1000000,validated_checks=50,best_check=49,
        best_accuracy=checks[48]['accuracy'],campaign_sha256='fixture',runtime=runtime,
        best_checkpoint_sha256=P.file_sha256(path))
    P.write_json(run/'training_complete.json',complete)
    P.write_json(run/'optimizer.json',dict(optimizer='Ranger',parameters=P.COUNTS['part'],
        groups=[dict(weight_decay=0,betas=[.95,.999],eps=1e-5)]))
    P.write_json(root/'campaign.json',spec)
    return spec,task,runtime,run


@pytest.mark.parametrize('failure',['budget','checks','tie','hash','missing','optimizer','runtime'])
def test_complete_fifty_checks_audited(tmp_path,failure):
    spec,task,runtime,run = training_fixture(tmp_path)
    assert R.validate_training(tmp_path,spec,task,R.runtime_identity(runtime))['best_check'] == 49
    complete = P.read_json(run/'training_complete.json')
    if failure == 'budget': complete['step'] = 200000
    if failure == 'checks': complete['validated_checks'] = 10
    if failure == 'tie': complete['best_check'] = 50
    if failure == 'hash': complete['best_checkpoint_sha256'] = 'wrong'
    if failure == 'runtime': complete['runtime']['gpu'] = 'GH200'
    if failure == 'missing': (run/'validation/check_50.json').unlink()
    if failure == 'optimizer':
        row = P.read_json(run/'optimizer.json')
        row['groups'][0]['weight_decay'] = .01
        P.write_json(run/'optimizer.json',row)
    P.write_json(run/'training_complete.json',complete)
    with pytest.raises((ValueError,FileNotFoundError)):
        R.validate_training(tmp_path,spec,task,R.runtime_identity(runtime))


@pytest.mark.parametrize('bad_labels',[False,True])
def test_main_contrast_paired_and_ev_single_seed(tmp_path,monkeypatch,bad_labels):
    spec = dict(tasks=P.TASKS,content_hash='report')
    monkeypatch.setattr(P,'require_approval',lambda *a:dict(runtime=dict(gpu='A100')))
    monkeypatch.setattr(R,'validate_training',lambda *a: {})
    accuracies = [.850,.852,.854,.853,.856,.859,.851]
    def evaluate(root,spec,task,trained,*,recompute):
        index = P.TASKS.index(task)
        metric = dict(accuracy=accuracies[index],macro_auc_ovo=.98,macro_auc_ovr=.98,
                      loss=.2,qcd_rejection={'Hbb':{'0.5':dict(rejection=10000)}})
        return dict(task=task,metrics=metric,best_check=50,
            validation_checks=[dict(accuracy=.99)]*50,
            training=dict(train_seconds=3600,validation_seconds=60,amp_skipped_steps=2)), (
                np.arange(10)[::-1] if bad_labels and index==6 else np.arange(10))
    monkeypatch.setattr(R,'validate_evaluation',evaluate)
    engine = SimpleNamespace(lock_finalists=lambda *a: {},classification_metrics=lambda *a: {})
    if bad_labels:
        with pytest.raises(ValueError,match='labels'): R.report(tmp_path,spec,engine)
        assert not (tmp_path/'reports/state_fullscale_report.json').exists()
        return
    result = R.report(tmp_path,spec,engine)
    assert result['ordinary_ev_seeds'] == [101] and result['main_comparison_seeds'] == [101,202,303]
    effect = result['contrasts']['accuracy']
    assert effect['state_minus_part']['values'] == pytest.approx([.003,.004,.005])
    assert effect['state_minus_part']['mean'] == pytest.approx(.004)
    assert effect['state_minus_part']['sample_std'] == pytest.approx(.001)
    assert effect['state_minus_ev_seed101'] == pytest.approx(.002)
    assert effect['ev_minus_part_seed101'] == pytest.approx(.001)
    text = (tmp_path/'reports/state_fullscale_report.md').read_text()
    assert '| 50 |' in text and 'not independent test' in text and 'seed 101 only' in text
    assert '| PART_RANGER | 101 | 85.0000' in text
