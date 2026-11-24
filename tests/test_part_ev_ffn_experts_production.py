"""Instrumented CPU final-boundary/evaluation-report plumbing, not 1M training."""
import copy
import numpy as np
import pytest
import torch
from part_ev_ffn_experts import engine as E,protocol as P,runtime as R
from part_ev_ffn_experts.config import COUNTS,recipe

torch.set_num_threads(1)


def prior_counts(tokens):
    return dict(tokens=[tokens]*6,requested=[[tokens,tokens,0,0] for _ in range(6)],
        executed=[[tokens,tokens,0,0] for _ in range(6)],primary=[[tokens,0,0,0] for _ in range(6)],
        secondary=[[0,tokens,0,0] for _ in range(6)])


@pytest.mark.parametrize('arm,tie',[('experts',False),('control',True)])
def test_final_boundary_commits_validation_selection_and_complete_audit(tmp_path,monkeypatch,arm,tie):
    spec=dict(content_hash='instrumented_cpu_only',runtime=dict(platform='cpu_fixture'),parameters=COUNTS,
        recipes={'ranger_full_v1':recipe()},resources=dict(local_batch=8,workers=0),inventory={})
    P.write(tmp_path/'production/campaign.json',spec)
    batch=R.fixed_batch(8,7);saved={};fresh=R.fresh;initial=E.initial_state
    def make(*a,**k):
        result=fresh(*a,**k);saved['base']=result[0];return result
    def start(a):
        state=initial(a);run=P.run_directory(tmp_path,a);checks=[]
        for i in range(1,50):
            row=dict(check=i,step=i*20000,events=1280000,accuracy=.8,task=P.task(a),campaign_sha256=spec['content_hash'])
            P.write(run/'validation'/f'check_{i:02d}.json',row);checks.append(row)
        selected=run/'selected/check_01.pt'
        E.atomic_torch_save(dict(model=saved['base'].state_dict(),task=P.task(a),runtime=E.expected_runtime(spec),
            campaign_sha256=spec['content_hash'],validation=checks[0]),selected)
        n=999999;balance=1.1 if a=='experts' else 0.
        state.update(step=n,validated_checks=49,best_accuracy=.8,best_check=1,best_file='selected/check_01.pt',
            best_checkpoint_sha256=P.sha(selected),loss_sums=dict(ce=2.*n,balance=balance*n,total=(2.+.001*balance)*n),
            routing_totals=prior_counts(8*n) if a=='experts' else None)
        return state
    monkeypatch.setattr(R,'fresh',make);monkeypatch.setattr(E,'initial_state',start)
    monkeypatch.setattr(E,'build_loader',lambda *a,**k:([batch],None))
    monkeypatch.setattr(E,'shutdown_loader',lambda *a:None)
    monkeypatch.setattr(E,'validate',lambda *a,**k:dict(events=1280000,accuracy=.8 if tie else .9,seconds=.1))
    E.train(tmp_path,spec,arm,torch.device('cpu'),0,1)
    audited=E.audit_training(tmp_path,spec,arm)
    assert audited['best_check']==(1 if tie else 50)
    state=audited['training'];assert state['step']==1000000 and state['validated_checks']==50
    E.audit_counters(state,arm)
    assert (P.run_directory(tmp_path,arm)/'last.pt').exists()


def test_paired_report_route_loss_cost_and_zero_background_handling(tmp_path,monkeypatch):
    from part_ev_message_study import controls
    from part_ev_ffn_experts import status
    spec=dict(content_hash='fixture',launch_sha256='launch',benchmark_sha256='bench',
        platform='cpu_fixture',resources=dict(gpus=2))
    rows={}
    for arm in COUNTS:
        state=dict(train_seconds=3600.,validation_seconds=600.,amp_skipped_steps=0,
            routing_totals=prior_counts(8000000) if arm=='experts' else None,
            loss_sums=dict(ce=2000000.,balance=1100000. if arm=='experts' else 0.,
                total=2001100. if arm=='experts' else 2000000.))
        rows[arm]=dict(task=P.task(arm),training=state,best_check=50,
            validation_checks=[dict(check=i,accuracy=.85+(.002 if arm=='experts' else 0.)) for i in range(1,51)],
            metrics=dict(accuracy=.85+(.002 if arm=='experts' else 0.),macro_auc_ovo=.99,
                qcd_rejection={'Hbb':{'0.5':dict(rejection=1100. if arm=='experts' else 1000.)},
                               'Hgg':{'0.5':dict(rejection=None)}}))
    P.write(tmp_path/'benchmark.json',P.hashed(dict(models=[dict(arm=a,gpus=[dict(peak_allocated=10*2**30)]) for a in COUNTS])))
    monkeypatch.setattr(E,'audit_training',lambda root,spec,arm:copy.deepcopy(rows[arm]))
    monkeypatch.setattr(controls,'validate_evaluation',lambda root,spec,task,trained,**k:(trained,np.arange(8)))
    monkeypatch.setattr(status,'allocated_cost',lambda output:dict(gpu_hours=2.,jobs=[]))
    result=E.report(tmp_path,spec)
    assert result['accuracy_delta_pp']==pytest.approx(.2)
    text=(tmp_path/'production/reports/experts_report.md').read_text()
    assert 'Global training route usage' in text and 'ZB' in text and '+10.00' in text
    assert P.checked(tmp_path/'production/reports/experts_report.json')['complete']
