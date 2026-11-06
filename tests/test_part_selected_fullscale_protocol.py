import copy
import math
import pytest
from part_selected_fullscale import protocol as P
from part_ev_fullscale.protocol import learning_rate


def evidence(index=6):
    steps=5000 if index in P.SELECTED else 1000
    spec=dict(content_hash='campaign',input_reference=dict(fingerprints={str(i):str(i) for i in (0,20,1020,2020,4020)}))
    runtime=dict(gpu='NVIDIA GH200',gpu_memory_bytes=96*2**30,world_size=1,batch_size=512,workers=2,
                 batchnorm='native',tf32=False,cudnn_benchmark=False,precision=P.F.R.S.POLICY,
                 campaign_sha256='campaign',execution=P.EXECUTION)
    row=dict(task=P.task(index),campaign_sha256='campaign',runtime=runtime,execution=P.EXECUTION,
        passed=True,complete=True,authorizes_training=False,parity=dict(passed=True),measured_steps=steps,
        windows=[dict(steps=1000,seconds=800)]*(steps//1000),
        input_fingerprints={k:v for k,v in spec['input_reference']['fingerprints'].items() if int(k)<steps+20},
        amp_skipped_steps=0,full_length_amp_skips=0,full_length_confirmed=10,training_precision=dict(passed=True),
        validation=dict(events=131072,seconds=200,precision=dict(passed=True)),
        gpu_peak_allocated_bytes=30*2**30,gpu_peak_reserved_bytes=35*2**30,host_peak_bytes=50*2**30,
        startup_seconds=30,checkpoint_seconds=.5,weight_save_seconds=.2)
    return row,spec


def test_seven_tasks_full_budget_and_matching_controls():
    assert len(P.TASKS)==7 and len(set(t['run_id'] for t in P.TASKS))==7
    assert P.SELECTED==(2,5,6)
    assert all(t['seed']==101 and t['recipe']=='ranger_full_v1' for t in P.TASKS)
    recipe=P.recipes()['ranger_full_v1']
    assert recipe['total_steps']==1000000 and recipe['epochs']==50
    assert recipe['global_batch_size']==512 and recipe['validation_events_per_epoch']==1280000
    assert learning_rate(699999,recipe)==.001
    assert learning_rate(700000,recipe)<.001
    assert math.isclose(learning_rate(980000,recipe),1e-5)
    assert P.COUNTS['part_match_shared']-P.COUNTS['state_shared']==40
    assert P.COUNTS['selected_match_layerwise']-P.COUNTS['selected_state_layerwise']==432
    assert P.storage_projection()['required_free_bytes']<18*2**30


@pytest.mark.parametrize('index',range(7))
def test_projection_counts_fifty_validations_and_full_checkpoint_budget(index):
    row,spec=evidence(index)
    result=P.validate_benchmark(row,spec,index)
    expected=30+1000000*.8+50*200*1280000/131072+1152*.5+50*.2
    assert result['projected_training_hours']==pytest.approx(expected/3600)
    assert result['checkpoint_writes']==1152
    assert result['recommended_train_hours']==math.ceil(expected*1.3/3600)


@pytest.mark.parametrize('key,value',[
    ('complete',False),('passed',False),('authorizes_training',True),('task',P.task(5)),
    ('measured_steps',1000),('windows',[dict(steps=5000,seconds=4000)]),
    ('input_fingerprints',{}),('full_length_confirmed',9),('full_length_amp_skips',1),
    ('amp_skipped_steps',50),('gpu_peak_reserved_bytes',90*2**30),
    ('host_peak_bytes',120*2**30),('parity',dict(passed=False)),
    ('startup_seconds',float('nan'))])
def test_incomplete_mismatched_or_unhealthy_measurements_cannot_pass(key,value):
    row,spec=evidence(); row[key]=value
    with pytest.raises(ValueError): P.validate_benchmark(row,spec,6)


def test_frozen_hash_detects_mutations(tmp_path):
    path=tmp_path/'report.json'; P.write_json(path,P.hashed(dict(schema='test',passed=True)))
    value=P.read_json(path); value['passed']=False; P.write_json(path,value)
    with pytest.raises(ValueError): P.checked_json(path,'test')
