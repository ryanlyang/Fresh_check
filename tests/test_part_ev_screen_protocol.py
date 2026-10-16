import copy
import pytest
from part_ev_screen import protocol as P
from part_ev_fullscale import protocol as old


def test_seven_tasks_and_explicit_short_recipes_do_not_change_old_contract():
    before = copy.deepcopy(old.RECIPE)
    recipes = P.recipes()
    assert len(P.TASKS) == len({t['run_id'] for t in P.TASKS}) == 7
    assert {t['seed'] for t in P.TASKS} == {101}
    assert sum(t['recipe'] == 'ranger_short_v1' for t in P.TASKS) == 5
    assert len({P.task_directory('/output', i) for i in range(7)}) == 7
    for recipe in recipes.values():
        assert recipe['total_steps'] == 200000
        assert recipe['total_steps'] * recipe['global_batch_size'] == 102400000
        assert recipe['epochs'] * recipe['steps_per_epoch'] == 200000
        assert recipe['validation_events_per_epoch'] == 1280000
        assert recipe['evaluation_population'] == 5000000
        assert 'test_population' not in recipe
    assert old.RECIPE == before


@pytest.mark.parametrize('step,decays', [(0,0),(139999,0),(140000,1),(195999,14),(196000,15),(199999,15)])
def test_ranger_boundaries(step, decays):
    assert P.learning_rate(step, P.recipes()['ranger_short_v1']) == pytest.approx(1e-3 * .01**(decays/15))


def test_adamw_boundaries_and_groups_contract():
    r = P.recipes()['adamw_short_v1']
    assert P.learning_rate(0,r) == pytest.approx(1e-7)
    assert P.learning_rate(9999,r) == pytest.approx(1e-3)
    assert 1e-5 < P.learning_rate(10000,r) < 1e-3
    assert P.learning_rate(199999,r) == pytest.approx(1e-5)
    assert r['weight_decay'] == 1e-4 and r['gradient_clipping'] == 1.
    for step in (-1,200000):
        with pytest.raises(ValueError): P.learning_rate(step,r)


def benchmark_files(tmp_path):
    spec = {'content_hash':'test'}
    runtime = {'gpu':'A100'}
    for task in P.TASKS:
        P.write_json(tmp_path/'benchmarks'/f"{task['run_id']}.json", dict(task=task, passed=True,
            campaign_sha256='test', runtime=runtime, input_fingerprints={'0':'same'}, projected_training_hours=10,
            initial_parity={'passed':True}, training_precision=precision_record('torch.float16'),
            validation={'precision':precision_record('torch.float32')}))
    return spec, runtime


def precision_record(dtype):
    return dict(passed=True, expected_activation_dtype=dtype, parameter_dtypes=['torch.float32'],
                observed_activations={'mod.blocks.0.fc1':dtype})


@pytest.mark.parametrize('failure', ['missing','failed','runtime','fingerprint','task','campaign'])
def test_all_seven_benchmarks_fail_closed(tmp_path, failure):
    spec, runtime = benchmark_files(tmp_path)
    assert P.benchmark_summary(tmp_path,spec,runtime)['recommended_train_hours'] == 13
    path = tmp_path/'benchmarks/EV_CONTEXT_RANGER.json'
    row = P.read_json(path)
    if failure == 'missing': path.unlink()
    else:
        if failure == 'failed': row['passed'] = False
        if failure == 'runtime': row['runtime']['gpu'] = 'H100'
        if failure == 'fingerprint': row['input_fingerprints']['0'] = 'different'
        if failure == 'task': row['task']['seed'] = 202
        if failure == 'campaign': row['campaign_sha256'] = 'different'
        P.write_json(path,row)
    with pytest.raises((ValueError,FileNotFoundError)): P.benchmark_summary(tmp_path,spec,runtime)


def test_manifest_mutation_rejected(tmp_path, monkeypatch):
    spec = P.create_spec({}, P.Path(__file__).resolve().parents[1], {})
    P.write_json(tmp_path/'campaign.json',spec)
    assert P.load_campaign(tmp_path)['tasks'] == P.TASKS
    spec['recipes']['ranger_short_v1']['total_steps'] = 100
    spec['content_hash'] = P.digest({k:v for k,v in spec.items() if k!='content_hash'})
    P.write_json(tmp_path/'campaign.json',spec)
    with pytest.raises(ValueError,match='protocol'): P.load_campaign(tmp_path)


@pytest.mark.parametrize('field', ['training_precision', 'validation'])
@pytest.mark.parametrize('failure', ['missing', 'failed', 'unobserved', 'wrong_dtype', 'half_parameters'])
def test_precision_evidence_is_required_before_production(tmp_path, field, failure):
    spec, runtime = benchmark_files(tmp_path)
    path = tmp_path/'benchmarks/EV_RANGER.json'
    row = P.read_json(path)
    if failure == 'missing':
        row.pop(field)
    else:
        audit = row[field] if field == 'training_precision' else row[field]['precision']
        if failure == 'failed': audit['passed'] = False
        if failure == 'unobserved': audit['observed_activations'] = {}
        if failure == 'wrong_dtype': audit['observed_activations']['mod.blocks.0.fc1'] = 'torch.float64'
        if failure == 'half_parameters': audit['parameter_dtypes'] = ['torch.float16']
    P.write_json(path, row)
    with pytest.raises(ValueError, match='precision'): P.benchmark_summary(tmp_path, spec, runtime)


def test_v1_campaign_cannot_resume_as_corrected_precision(tmp_path):
    spec = P.create_spec({}, P.Path(__file__).resolve().parents[1], {})
    assert spec['precision'] == P.PRECISION_POLICY
    spec['schema'] = 'part_ev_context_gate_screen_v1'
    spec['content_hash'] = P.digest({k:v for k,v in spec.items() if k!='content_hash'})
    P.write_json(tmp_path/'campaign.json',spec)
    with pytest.raises(ValueError, match='schema'): P.load_campaign(tmp_path)
