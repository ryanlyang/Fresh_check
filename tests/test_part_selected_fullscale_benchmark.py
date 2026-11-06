"""Exercise orchestration without CUDA, a cluster, or the real JetClass population."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_selected_fullscale import benchmark as B, protocol as P
from scripts import run_part_selected_fullscale as driver
from tests.test_part_selected_fullscale_protocol import evidence


@pytest.fixture
def fake_bench(tmp_path,monkeypatch):
    counters=dict(steps=0,full=0,closed=0,validations=0,fail_full=False)
    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(2,2)
            self.mod=SimpleNamespace(trimmer=SimpleNamespace(enabled=True,_counter=0))
    class Bench:
        def __init__(self,output,index,check_stop):
            _,self.spec=evidence(index)
            self.spec['recipes']=P.recipes()
            self.spec['content_hash']='campaign'
            self.output=Path(output); self.index=index; self.task=P.task(index)
            self.selected=index in P.SELECTED; self.device=torch.device('cpu')
            self.recipe=P.recipes()['ranger_full_v1']; self.runtime=evidence(index)[0]['runtime']
        def storage_check(self): pass
        def model(self,*_): return Model()
        def loader(self,split): return split
        def batches(self,loader):
            for i in range(10000): yield i
        def parity(self,*_): return dict(passed=True)
        def header(self,schema):
            return dict(schema=schema,campaign_sha256='campaign',task=self.task,runtime=self.runtime,
                        execution=P.EXECUTION,authorizes_training=False)
    monkeypatch.setattr(B,'Bench',Bench)
    monkeypatch.setattr(B,'build_optimizer',lambda model,recipe:torch.optim.SGD(model.parameters(),lr=.01))
    scaler=torch.amp.GradScaler
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *args,**kwargs:scaler('cpu',enabled=False))
    for name in ('reset_peak_memory_stats','synchronize','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:30*2**30)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:35*2**30)
    monkeypatch.setattr(B,'host_peak_bytes',lambda:50*2**30)
    class Monitor:
        peak=50*2**30; gpu_reserved=35*2**30
        def __enter__(self): return self
        def __exit__(self,*_): pass
    monkeypatch.setattr(B,'MemoryMonitor',Monitor)
    monkeypatch.setattr(B.selected_data,'batch_fingerprint',lambda batch:str(batch))
    monkeypatch.setattr(B.selected_data,'shutdown_loader',lambda loader:counters.__setitem__('closed',counters['closed']+1))
    def step(*args,full_length=False,**kwargs):
        counters['steps']+=1
        if full_length: counters['full']+=1
        return dict(amp_skipped=bool(full_length and counters['fail_full']),precision=dict(passed=True))
    monkeypatch.setattr(B,'training_step',step)
    def validate(*args,**kwargs):
        counters['validations']+=1
        return dict(events=131072,seconds=100,inference_seconds=90,macro_auc_ovo=.98,
                    macro_auc_ovr=.98,accuracy=.85,precision=dict(passed=True))
    monkeypatch.setattr(B,'validate',validate)
    return counters


def paired_report(index=6):
    row,spec=evidence(index)
    measurements={name:[dict(trial=i,steps=200,seconds=seconds,amp_skips=0,
        input_fingerprints={k:spec['input_reference']['fingerprints'][k] for k in ('0','20')})
        for i in range(3)] for name,seconds in (('reference',120),('candidate',100))}
    return P.hashed(dict(schema='selected_fullscale_profile_v1',complete=True,passed=True,
        campaign_sha256='campaign',task=P.task(index),runtime=row['runtime'],execution=P.EXECUTION,
        authorizes_training=False,parity=dict(passed=True),measurements=measurements,
        median_speedup=1.2,conservative_speedup=1.2))


@pytest.mark.parametrize('index',(0,2,5,6))
def test_complete_measurement_persists_projections_not_training(tmp_path,fake_bench,index):
    if index in P.SELECTED:
        P.write_json(tmp_path/'profiles'/f"{P.task(index)['run_id']}.json",paired_report(index))
    row=B.benchmark(tmp_path,index,lambda:None)
    assert row['passed'] and row['complete'] and row['authorizes_training'] is False
    assert row['measured_steps']==(5000 if index in P.SELECTED else 1000)
    assert fake_bench['steps']==row['measured_steps']+20+11
    assert fake_bench['closed']==2 and fake_bench['validations']==1
    assert row['projection']['validation_checks']==50
    assert not (tmp_path/'runs').exists()
    assert row['checkpoint_bytes']>0 and row['weight_bytes']>0
    # A resume reuses complete evidence, not another round of diagnostic updates.
    before=fake_bench['steps']; B.benchmark(tmp_path,index,lambda:None)
    assert fake_bench['steps']==before


def test_full_length_calibration_failure_stops_admission(tmp_path,fake_bench):
    fake_bench['fail_full']=True
    with pytest.raises(ValueError,match='calibration exhausted'): B.benchmark(tmp_path,0,lambda:None)
    assert fake_bench['full']==8 and fake_bench['validations']==0 and fake_bench['closed']==1
    assert not (tmp_path/'benchmarks'/f'{P.task(0)["run_id"]}.json').exists()


def test_timeout_does_not_save_partial_success(tmp_path,fake_bench):
    calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls>30: raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError): B.benchmark(tmp_path,0,stop)
    assert fake_bench['closed']==1
    assert not (tmp_path/'benchmarks'/f'{P.task(0)["run_id"]}.json').exists()


def test_selected_wrong_profile_cannot_start(tmp_path,fake_bench):
    P.write_json(tmp_path/'profiles'/f'{P.task(6)["run_id"]}.json',paired_report(5))
    with pytest.raises(ValueError,match='mismatched'): B.benchmark(tmp_path,6,lambda:None)
    assert fake_bench['steps']==0


def test_review_pending_and_complete_never_authorize_training(tmp_path,monkeypatch):
    spec=evidence()[1]
    monkeypatch.setattr(P,'load_campaign',lambda *a,**k:spec)
    monkeypatch.setattr(driver.shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    report=driver.review(tmp_path)
    assert not report['complete'] and not report['ready_for_resource_review']
    assert not (tmp_path/'reports').exists()
    for index,task in enumerate(P.TASKS):
        row,_=evidence(index)
        row.update(schema='selected_fullscale_benchmark_v1',jets_per_second=640,
                   checkpoint_bytes=40*1024**2,weight_bytes=10*1024**2)
        if index in P.SELECTED:
            path=tmp_path/'profiles'/f'{task["run_id"]}.json'; P.write_json(path,paired_report(index))
            row['profile_sha256']=P.file_sha256(path)
        P.write_json(tmp_path/'benchmarks'/f'{task["run_id"]}.json',P.hashed(row))
    report=driver.review(tmp_path,save=True)
    assert report['complete'] and report['authorizes_training'] is False
    assert not report['ready_for_resource_review']  # Standard projection exceeds its 216h allocation.
    assert (tmp_path/'reports/benchmark_review.json').is_file()


def test_step_adapter_does_not_modify_historical_lr_dispatch():
    from scripts.run_part_ev_screen import training_step as old
    from part_selected_fullscale.steps import training_step as new
    from part_ev_fullscale.protocol import learning_rate
    assert old is not new and old.__globals__['P'] is not new.__globals__['P']
    assert new.__globals__['P'].learning_rate is learning_rate
    assert new.__kwdefaults__==old.__kwdefaults__


def test_review_reports_aborted_profile_instead_of_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'load_campaign',lambda *a,**k:evidence()[1])
    P.write_json(tmp_path/'failures/profile_6_123.json',dict(phase='profile',task_index=6,
        error='TimeoutError: diagnostic deadline',complete=False,passed=False))
    report=driver.review(tmp_path)
    assert not report['complete'] and not report['ready_for_resource_review']
    assert report['models'][6]['status']=='failed'
    assert report['models'][6]['phase']=='profile'
    assert 'deadline' in report['models'][6]['error']
    assert report['models'][0]['status']=='pending'
