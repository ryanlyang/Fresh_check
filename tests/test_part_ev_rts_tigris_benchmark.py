import copy
from contextlib import nullcontext
import importlib
import sys
from types import SimpleNamespace
import time

import pytest
import torch
from part_ev_rts_tigris import benchmark as G, protocol as P, evidence as E
from part_ev_gpu_scaling import benchmark as B
from tests.test_part_ev_rts_tigris_evidence import evidence
from tests.test_part_ev_state_production_engine import ToyModel
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_actual_native_correctness_has_truthful_device_label(kind):
    C=importlib.import_module(P.PACKAGES[kind]+'.benchmark')
    rows=G.correctness(C,torch.device('cpu'))
    assert rows[0]['passed'] and rows[0]['execution']['world_size']==1
    assert rows[0]['execution']['device_type']=='cpu'
    assert {r['kind'] for r in rows[0]['checks']}=={E.RENAMES.get(k,k) for k in C.P.CHECKS}
    assert not any('DDP' in r['kind'] for r in rows[0]['checks'])
    if kind=='support':
        C.P.validate_particle_path(next(c for c in rows[0]['checks'] if c['kind']=='native_path_at_initialization'))


@pytest.mark.parametrize('kind',list(P.PACKAGES))
@pytest.mark.parametrize('baseline',(False,True))
def test_measure_flow_batch_sizes_cleanup_and_native_probes(tmp_path,monkeypatch,kind,baseline):
    """Exercise orchestration locally; not a substitute for CUDA numerical/timing gates."""
    torch.set_num_threads(1)
    template=evidence(kind)
    science=importlib.import_module(P.PACKAGES[kind]+'.protocol')
    D=importlib.import_module(P.PACKAGES[kind]+'.diagnostics')
    base=ToyModel()
    events=[];closed=[];calls=[];profiles=[]
    class Loader:
        def __iter__(self):
            while True: yield B.fixed_batch(512,17)
    def loader(split,inv,batch,**kw):
        events.append((split,batch,kw));return Loader(),None
    def measure(base,model,opt,scaler,fetch,recipe,device,world,warmup,steps,deadline,**kw):
        values,y=fetch(0)
        assert len(y)==512 and world==1 and base is model
        if 'fixed' in kw['label'] or 'ties' in kw['label']:assert values[0].shape[-1]==128
        calls.append((warmup,steps))
        return copy.deepcopy(template['stream'] if steps==4200 else template['fixed'])
    def restart(*args):
        assert args[-2:]==(0,1) and len(args[5][0])==512
        return dict(science.RESTART,ranks=1),1.
    def inference(base,values,*args):
        assert values[0].shape[:1]==(256,) and values[0].shape[-1]==128
        return template['inference']
    def validate(base,iterator,device,rank,world,count,**kw):
        assert world==1 and rank==0 and count==128000
        next(iterator)
        return template['validation']
    def diagnostic(base,values,device):
        assert len(values[0])==8
        return copy.deepcopy(template['diagnostics'][0])
    gradients=next(template[k] for k in template if k.endswith('_gradient_norms'))
    monkeypatch.setattr(D,'gradient_norms',lambda _:copy.deepcopy(gradients[0]))
    if kind=='triangles':
        class Stats:
            phase=None
            def __init__(self,_):pass
            def close(self):closed.append('stats')
            def report(self):return {k:v for k,v in template['neighbor_statistics'][0].items() if k!='rank'}
        monkeypatch.setattr(D,'NeighborStats',Stats)
        def profile(base,model,values,y,*args,**kw):
            profiles.append((kw['label'],len(y),values[0].shape[-1]))
            assert 'train' not in kw['label'] or closed  # Training workers released first.
            return copy.deepcopy(template['profiles'][0][kw['label']])
        monkeypatch.setattr(D,'stage_profile',profile)
    monkeypatch.setitem(sys.modules,'resource',SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:SimpleNamespace(ru_maxrss=100)))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda _:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(name='GH200',total_memory=96*2**30))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda _:2**30)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda _:2**30)
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *_:SimpleNamespace(state_dict=lambda:{}))
    backend=SimpleNamespace(B=SimpleNamespace(measure=measure,tensors=B.tensors,fixed_batch=B.fixed_batch,
        batch_fingerprint=B.batch_fingerprint,STOP=False),fresh=lambda *a,**kw:(base,base),
        build_optimizer=lambda *a:SimpleNamespace(state_dict=lambda:{}),optimizer_metadata=lambda *a:{},
        build_loader=loader,shutdown_loader=lambda obj:closed.append(obj),seed_everything=B.seed_everything,
        restart_probe=restart,validate=validate,inference=inference,diagnostic_sample=diagnostic,
        finite_state=lambda *a:True,model_precision=lambda *a,**kw:nullcontext(),P=science)
    row=G.measure(backend,tmp_path,dict(model_kind=kind,inventory={},recipe={},content_hash='probe'),
                  torch.device('cpu'),time.monotonic()+60,baseline=baseline)
    assert calls[:2]==[(20,4200),(10,40)] and row['finite_state']
    assert all(batch==512 and kw['workers']==2 for _,batch,kw in events)
    assert len(row['gpus'])==1 and len(closed)>=2
    if not baseline:
        assert row['restart']['ranks']==1 and row['full128_validation_passed']
        if kind=='triangles':
            assert calls[2]==(2,3)
            assert profiles==[('real_train',512,17),('adversarial_train',512,128),('real_validation',8,17)]
            assert row['full128_tied_validation_passed'] and 'stats' in closed
        if kind=='support':assert row['diagnostics'][0]['population']=='first_8_validation_probe'
