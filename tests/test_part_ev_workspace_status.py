import json
import pytest
from part_ev_workspace import status as S,protocol as P


def test_status_distinguishes_boundary_from_completed_validation(tmp_path):
    run=P.run_directory(tmp_path,'workspace')
    P.write(run/'validation/check_01.json',dict(check=1,step=20000,accuracy=.8))
    log=tmp_path/'logs/evws_tigris_train_workspace_123.out';log.parent.mkdir()
    log.write_text(json.dumps(dict(stage='training',task=P.task('workspace'),step=40000))+'\n')
    result=S.display(tmp_path)['arms']['workspace']
    assert result['progress']['percent_to_boundary']==100
    assert result['progress']['validation_pending']
    assert len(result['checks'])==1


def test_actual_accounting_includes_retry_elapsed_times_gpu_count(tmp_path,monkeypatch):
    P.write(tmp_path/'production_submissions.json',[
        dict(status='recorded',job='123',phase='train_workspace'),
        dict(status='recorded',job='124',phase='evaluate_workspace')])
    P.write(tmp_path/'resume_001_submissions.json',[
        dict(status='recorded',job='125',phase='train_workspace')])
    monkeypatch.setattr(S.subprocess,'check_output',lambda *a,**k:
        '123|TIMEOUT|7200|cpu=8,gres/gpu=2,gres/gpu:a100=2|\n'
        '124|CANCELLED|0||\n'
        '125|COMPLETED|3600|cpu=8,gres/gpu=2|\n')
    assert S.allocated_cost(tmp_path)['gpu_hours']==6
    monkeypatch.setattr(S.subprocess,'check_output',lambda *a,**k:'123|TIMEOUT|7200|cpu=8|\n')
    assert S.allocated_cost(tmp_path)['gpu_hours'] is None
