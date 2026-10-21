import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_selected_screen import protocol as P
from part_selected_screen.normalization import SETTINGS, identity_normalization
from part_ev_screen.protocol import recipes


def test_exact_four_tasks_recipe_and_controls():
    assert len(P.TASKS) == 4
    assert all(t['seed'] == 101 and t['recipe'] == 'ranger_short_v1' for t in P.TASKS)
    assert set(P.MATCHED.values()) == set(P.SHARED+P.LAYERWISE)
    assert not set(P.MATCHED) & set(P.SHARED+P.LAYERWISE)
    assert recipes()['ranger_short_v1']['total_steps'] == 200000
    for index in (-1, 4, True, None):
        with pytest.raises(ValueError): P.task_directory('/unused', index)


def test_prepare_and_frozen_contract_reject_drift(tmp_path, monkeypatch):
    donor = dict(environment={'fixture': 1}, inventory={'inventory_sha256': 'inventory'})
    layerwise = {'fixture': 'layerwise'}
    monkeypatch.setattr(P, 'donor_contract', lambda *a, **kw: donor)
    monkeypatch.setattr(P, 'require_environment', lambda env: None)
    monkeypatch.setattr(P, 'verify_dataset_metadata', lambda inventory: None)
    monkeypatch.setattr(P, 'layerwise_contract', lambda *a: layerwise)
    P.write_json(tmp_path/'launch.json', dict(donor='/shared', layerwise_donor='/layerwise'))
    normal = P.hashed(dict(schema='part_selected_normalization_v1', inventory_sha256='inventory',
        settings=SETTINGS, population='training_only', **identity_normalization()))
    P.write_json(tmp_path/'inputs/normalization.json', normal)
    P.write_json(tmp_path/'backend/backend_manifest.json', {'fixture': True})
    (tmp_path/'inputs/selected.yaml').write_text('fixture')
    spec = P.create_spec(tmp_path)
    P.write_json(tmp_path/'experiment_spec.json', spec)
    assert P.load_campaign(tmp_path) == spec
    for key in ('tasks', 'model', 'parameters', 'precision', 'benchmark', 'recipes', 'source'):
        row = copy.deepcopy(spec)
        row.pop('content_hash')
        row[key] = [] if key == 'tasks' else {}
        P.write_json(tmp_path/'experiment_spec.json', P.hashed(row))
        with pytest.raises(ValueError): P.load_campaign(tmp_path)
    P.write_json(tmp_path/'experiment_spec.json', spec)
    (tmp_path/'inputs/selected.yaml').write_text('changed')
    with pytest.raises(ValueError, match='artifacts changed'): P.load_campaign(tmp_path)


def benchmark_fixture(tmp_path):
    runtime = dict(gpu='A100', campaign_sha256='selected')
    spec = dict(content_hash='selected', donor=dict(runtime={'gpu': 'A100'}, input_fingerprints={'0': 'same'}),
                launch=dict(train_hours='96', evaluate_hours='12'))
    rows = []
    for task in P.TASKS:
        row = dict(passed=True, task=task, campaign_sha256='selected', runtime=runtime,
            input_fingerprints={'0': 'same'}, projected_training_hours=20.,
            validation=dict(events=131072, seconds=20.))
        rows.append(row)
        P.write_json(tmp_path/'benchmarks'/f"{task['run_id']}.json", row)
    return spec, rows


@pytest.mark.parametrize('change', ['failed', 'wrong_seed', 'wrong_runtime', 'wrong_stream', 'too_slow', 'nan', 'eval_nan'])
def test_gate_checks_all_four_models_and_runtime_budget(tmp_path, change):
    spec, rows = benchmark_fixture(tmp_path)
    assert len(P.require_approval(tmp_path, spec, {'gpu': 'A100'})) == 4
    row = copy.deepcopy(rows[3])
    if change == 'failed': row['passed'] = False
    if change == 'wrong_seed': row['task']['seed'] = 202
    if change == 'wrong_runtime': row['runtime']['gpu'] = 'GH200'
    if change == 'wrong_stream': row['input_fingerprints']['0'] = 'different'
    if change == 'too_slow': row['projected_training_hours'] = 100
    if change == 'nan': row['projected_training_hours'] = float('nan')
    if change == 'eval_nan': row['validation']['seconds'] = float('nan')
    # JSON writer correctly rejects NaN; exercise the gate against an untrusted reader fixture.
    if 'nan' in change:
        from unittest.mock import patch
        original = P.read_json
        with patch.object(P, 'read_json', side_effect=lambda p: row if str(p).endswith('SELECTED_EV_LAYERWISE.json') else original(p)):
            with pytest.raises(ValueError): P.require_approval(tmp_path, spec)
    else:
        P.write_json(tmp_path/'benchmarks/SELECTED_EV_LAYERWISE.json', row)
        with pytest.raises(ValueError): P.require_approval(tmp_path, spec)


def test_engine_data_adapter_is_isolated_not_global_mutation(tmp_path, monkeypatch):
    import part_screen_common.engine as common
    from scripts import run_part_selected_screen as driver
    original = common.build_loader, common.batch_fingerprint, common.finite_validation_loader
    monkeypatch.setattr(driver, 'load_backend', lambda path: 'reference_for_tests')
    monkeypatch.setattr(P, 'checked_json', lambda *a: identity_normalization())
    engine = driver.engine_for(tmp_path)
    assert original == (common.build_loader, common.batch_fingerprint, common.finite_validation_loader)
    assert engine.train.__func__.__globals__['build_loader'] is not common.build_loader
    assert engine.train.__func__.__globals__['training_step'] is common.training_step
    assert engine.evaluate.__func__.__globals__['classification_metrics'] is common.classification_metrics


def test_completed_layerwise_control_identity_must_match(tmp_path, monkeypatch):
    from part_layerwise_screen import protocol as L
    shared = dict(environment={'python': 'same'}, inventory={'files': 'same'}, runtime={'gpu': 'A100'})
    spec = dict(schema=L.SCHEMA, tasks=L.TASKS, model=L.CONFIG, parameters=L.COUNTS,
        precision=P.POLICY, donor=shared, environment=shared['environment'], inventory=shared['inventory'],
        recipes={'ranger_short_v1': recipes()['ranger_short_v1']}, admission=L.ADMISSION, source={'fixture': 'hash'})
    P.write_json(tmp_path/'experiment_spec.json', P.hashed(spec))
    monkeypatch.setattr(L, 'source_manifest', lambda source: {'fixture': 'hash'})
    monkeypatch.setattr(P, 'verify_source', lambda *a: None)
    contract = P.layerwise_contract(tmp_path, shared, tmp_path)
    assert contract['controls'] == P.LAYERWISE
    for key in ('tasks', 'donor', 'environment', 'recipes', 'source'):
        row = copy.deepcopy(spec)
        row[key] = [] if key == 'tasks' else {}
        P.write_json(tmp_path/'experiment_spec.json', P.hashed(row))
        with pytest.raises(ValueError): P.layerwise_contract(tmp_path, shared, tmp_path)


def test_eight_model_report_matched_pair_counts_and_immutable_lock(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from part_selected_screen import reporting as R
    labels = np.arange(20, dtype=np.int64) % 10
    names = P.SHARED+P.LAYERWISE+list(P.MATCHED)
    rows = {}
    controls_root = tmp_path/'read_only_controls'
    controls_root.mkdir()
    for i, name in enumerate(names):
        pred = labels.copy()
        pred[:8-i] = (pred[:8-i]+1)%10
        logits = np.eye(10, dtype=np.float32)[pred]
        path = controls_root/f'{name}.npz'
        np.savez(path, logits=logits, labels=labels)
        count = P.COUNTS[P.TASKS[i-4]['architecture']] if i >= 4 else 2000000+i
        training = dict(checkpoint=f'/checkpoints/{name}.pt', checkpoint_sha256=f'hash{i}',
            best_check=10, optimizer={'parameters': count})
        metrics = dict(accuracy=float((pred == labels).mean()), macro_auc_ovo=.98+i*.001,
            macro_auc_ovr=.98+i*.001, loss=.8-i*.01, qcd_rejection={'Hbb': {'0.5': {'rejection': 100.+i}}})
        evaluation = dict(metrics=metrics, metrics_sha256=f'metrics{i}',
            labels_by_file={'file': 'same'}, predictions=[dict(name='file', path=str(path))])
        rows[name] = dict(training=training, evaluation=evaluation)
    before = {p.name: p.read_bytes() for p in controls_root.iterdir()}
    output = tmp_path/'new_campaign'
    spec = dict(content_hash='campaign', tasks=P.TASKS, recipes=recipes(), donor={'runtime': {}},
                inventory={'splits': {'val': {'files': []}}})
    monkeypatch.setattr(P, 'load_campaign', lambda *a: spec)
    monkeypatch.setattr(R, 'controls', lambda *a: {n: rows[n] for n in names[:4]})
    monkeypatch.setattr(R, 'selected_training', lambda run, task, *a: rows[task['run_id']]['training'])
    monkeypatch.setattr(R, 'validate_predictions', lambda run, task, *a: rows[task['run_id']]['evaluation'])
    lock = dict(campaign_sha256='campaign', checkpoints=[dict(task=t,
        checkpoint=rows[t['run_id']]['training']['checkpoint'], sha256=rows[t['run_id']]['training']['checkpoint_sha256']) for t in P.TASKS])
    P.write_json(output/'selection/locked_checkpoints.json', lock)
    R.report(output)
    result = P.read_json(output/'reports/selected_screen_report.json')
    assert result['complete'] and len(result['models']) == 8
    assert sum(r['reused'] for r in result['models'].values()) == 4
    assert result['population'] == 'official_validation_5M' and result['single_seed_exploratory']
    assert all(r['events'] == 20 for r in result['paired_accuracy'].values())
    assert before == {p.name: p.read_bytes() for p in controls_root.iterdir()}
    R.report(output)  # unchanged idempotent publication
    lock['checkpoints'][0]['sha256'] = 'changed'
    P.write_json(output/'selection/locked_checkpoints.json', lock)
    with pytest.raises(ValueError, match='identity changed'): R.report(output)
