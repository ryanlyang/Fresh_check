import copy
import math
import pytest
from part_ev_state_fullscale import protocol as P
from part_ev_fullscale import protocol as F
from part_ev_screen import protocol as S
from tests.test_part_ev_message_protocol import parity_fixture
from tests.test_part_ev_screen_protocol import precision_record


def hashed(row):
    row['content_hash'] = P.digest({k:v for k,v in row.items() if k != 'content_hash'})
    return row


def fixture(root):
    spec = dict(content_hash='full')
    runtime = dict(gpu='A100', gpu_memory_bytes=1000, campaign_sha256='full')
    for index in P.BENCHMARK_INDICES:
        task = copy.deepcopy(P.TASKS[index])
        parity = parity_fixture()
        if index == 3:
            parity['state_gradients'] = dict(passed=True, checkpoint_backward=True, tensors=19,
                forward=dict(passed=True), max_absolute_error=1e-6)
        windows = [dict(steps=n, seconds=n*.1) for n in (1000,1000,1000,1000,200)]
        val = dict(events=128000, seconds=100, precision=precision_record('torch.float32'))
        row = dict(passed=True, task=task, campaign_sha256='full', runtime=runtime,
            initial_parity=parity, measured_steps=4200, amp_skipped_steps=2,
            training_precision=precision_record('torch.float16'), validation=val,
            windows=windows, checkpoint_seconds=1., cold_fetch_seconds=20.,
            native_peak_bytes=100, full_length_peak_bytes=200, validation_peak_bytes=300,
            input_fingerprints={str(i):str(i) for i in (0,20,1020,2020,4020)},
            full_length_attempts=[dict(attempt=i+1, loss=1., state_finite=True, amp_skipped=i==0,
                                      gradient_norm=None if i==0 else 1.) for i in range(18)])
        row['projected_training_hours'] = P.projected_hours(P.recipes()['ranger_full_v1'], windows, val, 1., 20.)
        P.write_json(root/'benchmarks'/(task['run_id']+'.json'), hashed(row))
    return spec, runtime


def test_seven_runs_full_recipe_and_unchanged_old_protocol():
    assert [(t['architecture'],t['seed']) for t in P.TASKS] == [
        ('part',101),('part',202),('part',303),('state_late4',101),
        ('state_late4',202),('state_late4',303),('part_ev',101)]
    assert len({P.task_key(t) for t in P.TASKS}) == 7
    assert P.BENCHMARK_INDICES == (0,3,6)
    recipe = P.recipes()['ranger_full_v1']
    original = F.recipe_for()
    for key in ('epochs','total_steps','steps_per_epoch','global_batch_size','validation_events_per_epoch',
                'optimizer','betas','eps','weight_decay','lookahead_k','lookahead_alpha','gradient_clipping',
                'start_lr','final_lr','lr_decay_epochs','checkpoint_selection'):
        assert recipe[key] == original[key]
    assert recipe['epochs'] == 50 and recipe['total_steps'] == 1000000
    for step in (0,139999,140000,199999,200000,699999,700000,719999,720000,980000,999999):
        assert P.learning_rate(step,recipe) == F.learning_rate(step,original)
    assert P.learning_rate(699999,recipe) == .001
    assert P.learning_rate(999999,recipe) == pytest.approx(.00001)
    assert S.recipes()['ranger_short_v1']['total_steps'] == 200000
    for bad in (-1,7,None,True):
        with pytest.raises(ValueError): P.task_directory('/root',bad)


def test_projection_scales_validation_and_checkpoint_io():
    recipe = P.recipes()['ranger_full_v1']
    hours = P.projected_hours(recipe,[dict(steps=1000,seconds=100)],dict(events=128000,seconds=100),1,20)
    assert hours == pytest.approx((100000+50000+1152+20)/3600)


@pytest.mark.parametrize('failure', ['missing','hash','task','runtime','input','input_missing',
    'precision','memory','confirmation','calibration_last','skip_rate','parity','gradients',
    'ten_check_projection','bad_windows','bad_val','timing'])
def test_preflight_is_fail_closed(tmp_path,failure):
    spec,runtime = fixture(tmp_path)
    assert P.benchmark_summary(tmp_path,spec,runtime)['recommended_evaluate_hours'] >= 5
    path = tmp_path/'benchmarks/EV_STATE_LATE4_RANGER.json'
    row = P.read_json(path)
    if failure == 'missing': path.unlink()
    else:
        if failure == 'task': row['task']['seed'] = 202
        if failure == 'runtime': row['runtime']['gpu'] = 'H100'
        if failure == 'input': row['input_fingerprints']['0'] = 'changed'
        if failure == 'input_missing': row['input_fingerprints'].pop('4020')
        if failure == 'precision': row['training_precision']['passed'] = False
        if failure == 'memory': row['validation_peak_bytes'] = 950
        if failure == 'confirmation': row['full_length_attempts'][-1]['amp_skipped'] = True
        if failure == 'calibration_last': row['full_length_attempts'][7]['amp_skipped'] = True
        if failure == 'skip_rate': row['amp_skipped_steps'] = 210
        if failure == 'parity': row['initial_parity']['modes'] = {}
        if failure == 'gradients': row['initial_parity']['state_gradients']['passed'] = False
        if failure == 'ten_check_projection': row['projected_training_hours'] /= 5
        if failure == 'bad_windows': row['windows'] = []
        if failure == 'bad_val': row['validation']['events'] = 10
        if failure == 'timing': row['checkpoint_seconds'] = -1
        hashed(row)
        if failure == 'hash': row['passed'] = False
        P.write_json(path,row)
    with pytest.raises((ValueError,FileNotFoundError)): P.benchmark_summary(tmp_path,spec,runtime)


def test_approval_and_evidence_freeze(tmp_path):
    spec,_ = fixture(tmp_path)
    with pytest.raises(ValueError): P.approve(tmp_path,spec,reviewed=False)
    P.approve(tmp_path,spec,reviewed=True)
    assert P.require_approval(tmp_path,spec)['passed']
    path = tmp_path/'benchmarks/EV_RANGER.json'
    row = P.read_json(path)
    row['amp_skipped_steps'] = 1
    P.write_json(path,hashed(row))
    with pytest.raises(ValueError,match='approval'): P.require_approval(tmp_path,spec)


@pytest.mark.parametrize('changed',['recipes','tasks','environment','source','launch','contents'])
def test_campaign_frozen_and_no_donor_dependency(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(P,'runtime_environment',lambda:dict(version='fixture'))
    inventory = dict(splits={})
    inventory['inventory_sha256'] = P.digest(inventory)
    launch,contents = dict(partition='tier3'),dict(files={})
    spec = P.create_spec(inventory,launch,contents)
    assert 'donor' not in spec and 'state_reference' not in spec
    P.write_json(tmp_path/'campaign.json',spec)
    P.write_json(tmp_path/'launch.json',launch)
    P.write_json(tmp_path/'data_contents.json',contents)
    assert P.load_campaign(tmp_path)['tasks'] == P.TASKS
    if changed == 'launch': P.write_json(tmp_path/'launch.json',dict(partition='debug'))
    elif changed == 'contents': P.write_json(tmp_path/'data_contents.json',dict(files={'bad':1}))
    else:
        spec[changed] = {} if isinstance(spec[changed],dict) else []
        P.write_json(tmp_path/'campaign.json',hashed(spec))
    with pytest.raises(ValueError): P.load_campaign(tmp_path)
