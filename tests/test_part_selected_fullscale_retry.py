import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_selected_fullscale import parity, benchmark as B, protocol as P
from scripts import run_part_selected_fullscale as driver
from tests.test_part_fullscale_model import bounded_cpu_threads
from tests.test_part_selected_fullscale_benchmark import fake_bench, paired_report
from tests.test_part_selected_fullscale_protocol import evidence
from tests.test_part_selected_restart import Toy, step
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state


def restart_result():
    return dict(passed=True, contract='synchronized_exact_restart_v2', steps=[0,1,2,3,4,5,699999,700000,979999,980000],
                warmup_steps=10, trials=['uninterrupted','repeat','resumed'], serialized_restart_after=3,
                optimizer_and_rng_exact=True, exact_roundtrip=True, exact_each_update=True, rtol=0, atol=0)


def pilot(monkeypatch, broken=False):
    seed_everything(101)
    model = Toy().eval()
    batch = ({'x':torch.randn(8,4)}, {'_label_':torch.zeros(8,dtype=torch.long)}, {})
    before = parity._cpu_state(dict(model=model.state_dict(), rng=capture_rank_state(model)))
    steps = []
    def update(*args, **kwargs):
        steps.append(args[-1])
        return step(*args, **kwargs)
    monkeypatch.setattr(parity, 'training_step', update)
    if broken:
        build = parity.build_optimizer
        def optimizer(*args):
            result = build(*args); original = result.load_state_dict
            def load(state):
                original(state)
                result.slow[0].add_(.01)
            result.load_state_dict = load
            return result
        monkeypatch.setattr(parity, 'build_optimizer', optimizer)
    try:
        row = parity.full_recipe_restart(model,batch,torch.device('cpu'))
        B.validate_retry_parity(dict(passed=True,full_recipe_restart=row))
        assert steps == row['steps']*4  # disposable warmup + three matched trials
        return row
    finally:
        parity._exact(before,parity._cpu_state(dict(model=model.state_dict(), rng=capture_rank_state(model))))
        assert not model.training and model._counter == 0


def test_synchronized_exact_restart_restores_model_rng_modes(monkeypatch):
    assert pilot(monkeypatch)['exact_each_update']


def test_missing_lookahead_state_still_fails_and_restores_model(monkeypatch):
    with pytest.raises(AssertionError, match='restored.optimizer.slow'):
        pilot(monkeypatch,broken=True)


@pytest.mark.parametrize('field,value', [('atol',1e-6),('rtol',1e-5),('warmup_steps',0),
    ('exact_roundtrip',False),('exact_each_update',False),('trials',['uninterrupted','resumed'])])
def test_retry_cannot_accept_relaxed_or_incomplete_check(field,value):
    row=restart_result(); row[field]=value
    with pytest.raises(ValueError,match='zero-tolerance'):
        B.validate_retry_parity(dict(passed=True,full_recipe_restart=row))


def test_separate_retry_routes_only_new_standard_evidence(tmp_path,fake_bench,monkeypatch):
    parent=tmp_path/'parent'; parent.mkdir()
    retry=tmp_path/'retry'; retry.mkdir()
    (parent/'original.txt').write_text('keep')
    monkeypatch.setattr(B,'validate_model',lambda *a:{'parameters':2323750})
    proof=dict(plan_sha256='plan')
    row=B.benchmark(parent,0,lambda:None,retry_root=retry,
                    restart_check=lambda *a:restart_result(),retry_provenance=proof)
    assert row['retry_provenance']==proof and row['campaign_sha256']=='campaign'
    assert (retry/'benchmarks/PART_MATCH_STATE_SHARED.json').is_file()
    assert list(parent.iterdir())==[parent/'original.txt']
    assert fake_bench['steps']==1031  # unchanged timing/calibration work
    with pytest.raises(ValueError,match='provenance changed'):
        B.benchmark(parent,0,lambda:None,retry_root=retry,restart_check=lambda *a:restart_result(),
                    retry_provenance={'plan_sha256':'different'})
    with pytest.raises(ValueError,match='only standard'):
        B.benchmark(parent,2,lambda:None,retry_root=retry,restart_check=lambda *a:restart_result(),retry_provenance=proof)


def test_retry_failure_never_starts_timing_or_writes_success(tmp_path,fake_bench,monkeypatch):
    monkeypatch.setattr(B,'validate_model',lambda *a:{})
    def fail(*_): raise AssertionError('controlled mismatch')
    with pytest.raises(AssertionError,match='controlled mismatch'):
        B.benchmark(tmp_path,0,lambda:None,retry_root=tmp_path/'retry',restart_check=fail,retry_provenance={'hash':'a'})
    assert fake_bench['steps']==0
    assert not list((tmp_path/'retry').rglob('*.json'))


def test_combined_review_keeps_original_hashes_and_checks_overlay(tmp_path,monkeypatch):
    parent=tmp_path/'parent'; parent.mkdir()
    retry=tmp_path/'retry'; retry.mkdir()
    proof=dict(plan_sha256='plan')
    monkeypatch.setattr(P,'load_campaign',lambda *a,**k:evidence()[1])
    monkeypatch.setattr(driver.shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    for index,task in enumerate(P.TASKS):
        row,_=evidence(index)
        row.update(schema='selected_fullscale_benchmark_v1',jets_per_second=2000,
                   checkpoint_bytes=40*1024**2,weight_bytes=10*1024**2)
        row['windows']=[dict(steps=1000,seconds=200)]*len(row['windows'])
        root=parent if index in P.SELECTED else retry
        if index in P.SELECTED:
            path=parent/'profiles'/f'{task["run_id"]}.json'; P.write_json(path,paired_report(index))
            row['profile_sha256']=P.file_sha256(path)
        else:
            row['retry_provenance']=proof
            row['parity']=dict(passed=True,full_recipe_restart=restart_result())
        P.write_json(root/'benchmarks'/f'{task["run_id"]}.json',P.hashed(row))
    before={p:P.file_sha256(p) for p in parent.rglob('*') if p.is_file()}
    report=driver.review(parent,save=True,retry_root=retry,retry_provenance=proof,retry_validator=B.validate_retry_parity)
    assert report['complete'] and report['ready_for_resource_review'] and not report['authorizes_training']
    assert [i for i,r in enumerate(report['models']) if r['reused_original_selected']]==[2,5,6]
    assert all(P.file_sha256(p)==sha for p,sha in before.items()) and not (parent/'reports').exists()
    report=driver.review(parent,retry_root=retry,retry_provenance={'plan_sha256':'different'},retry_validator=B.validate_retry_parity)
    assert not report['ready_for_resource_review'] and report['models'][0]['status']=='failed'
