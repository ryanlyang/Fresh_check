import copy
import importlib
import math
from pathlib import Path
import sys

import pytest

from part_ev_extensions_tigris import protocol as P, evidence as E


def fixture(kind):
    science = importlib.import_module(P.PACKAGES[kind]+'.protocol')
    helper = importlib.import_module('tests.test_'+P.PACKAGES[kind]+'_protocol')
    row = helper.evidence()
    row.update(authorizes_training=False, runtime=P.runtime(), parameters=P.COUNTS[kind], extension_active=True)
    for name in ('stream', 'fixed'): row[name]['ranks'] = row[name]['ranks'][:1]
    row['correctness'] = row['correctness'][:1]
    row['correctness'][0]['execution'] = dict(world_size=1, batchnorm='native', distributed=False)
    for c in row['correctness'][0]['checks']:
        if c['kind'] == 'active_SyncBN_DDP': c['kind'] = 'active_native_global_batch'
    row['diagnostics'] = row['diagnostics'][:1]
    row['restart'].update(ranks=1, rng_and_trimmers=True)
    row['gpus'] = [dict(rank=0, name='NVIDIA GH200 480GB', total_memory=96*2**30,
        peak_allocated=50*2**30, peak_reserved=60*2**30, cold_fetch_seconds=100.)]
    row['inference'].update(batch1_iterations=100, batch1_seconds=2.)
    spec = dict(content_hash='spec', model_kind=kind, parameters=science.PARAMETERS,
                recipe=science.recipes()['ranger_full_v1'], task=science.TASK, parity=science.PARITY)
    return spec, row


@pytest.mark.parametrize('kind', list(P.PACKAGES))
def test_full_schedule_native_summary_never_authorizes(kind):
    spec, row = fixture(kind)
    before = copy.deepcopy(row)
    result = E.summarize(spec, row)
    assert result['passed'] and result['complete'] and not result['authorizes_training']
    assert result['parameters'] == P.COUNTS[kind]
    assert result['projected_train_hours'] == pytest.approx(500000/3600)
    assert result['projected_validation_hours'] == pytest.approx(15000/3600)
    assert result['projected_checkpoint_hours'] == pytest.approx(1152/3600)
    assert result['recommended_train_hours'] == math.ceil(1.3*(500000+15000+1152+100)/3600)
    assert result['fixed128_inference_jets_per_second'] == 1280
    assert result['fixed128_batch1_latency_ms'] == 20
    assert result['runtime']['world_size'] == 1 and result['runtime']['local_batch'] == 512
    assert result['runtime']['workers_per_rank'] == 2 and row == before


@pytest.mark.parametrize('kind', list(P.PACKAGES))
@pytest.mark.parametrize('fault', ['hash', 'runtime', 'workers', 'batch', 'params', 'rank', 'gpu',
    'vram', 'reserved', 'headroom', 'precision', 'restart', 'rng', 'checks', 'window', 'skips',
    'short', 'diagnostics', 'inactive', 'finite', 'latency', 'cold', 'cpu_parity', 'ddp_label', 'authorize'])
def test_gh200_evidence_fails_closed(kind, fault):
    spec, row = fixture(kind)
    if fault == 'hash': row['campaign_sha256'] = 'wrong'
    if fault == 'runtime': row['runtime']['batchnorm'] = 'SyncBatchNorm'
    if fault == 'workers': row['runtime']['workers_per_rank'] = 1
    if fault == 'batch': row['runtime']['local_batch'] = 256
    if fault == 'params': row['parameters'] += 1
    if fault == 'rank': row['stream']['ranks'] *= 2
    if fault == 'gpu': row['gpus'][0]['name'] = 'A100'
    if fault == 'vram': row['gpus'][0]['total_memory'] = 40*2**30
    if fault == 'reserved': row['gpus'][0]['peak_reserved'] = 100*2**30
    if fault == 'headroom': row['gpus'][0]['peak_allocated'] = 90*2**30
    if fault == 'precision': row['validation']['precision']['passed'] = False
    if fault == 'restart': row['restart']['full_state'] = False
    if fault == 'rng': row['restart']['rng_and_trimmers'] = False
    if fault == 'checks': row['correctness'][0]['checks'].pop()
    if fault == 'window': row['stream']['ranks'][0]['windows'].pop()
    if fault == 'skips': row['fixed']['skips'] = 2
    if fault == 'short': row['stream']['measured_steps'] = 20
    if fault == 'diagnostics': row['diagnostics'][0]['layers'].pop()
    if fault == 'inactive': row['extension_active'] = False
    if fault == 'finite': row['finite_state'] = False
    if fault == 'latency': row['inference']['batch1_seconds'] = 0
    if fault == 'cold': row['gpus'][0]['cold_fetch_seconds'] = float('nan')
    if fault == 'cpu_parity': row['correctness'][0]['initialization']['device_type'] = 'cpu'
    if fault == 'ddp_label': row['correctness'][0]['execution']['distributed'] = True
    if fault == 'authorize': row['authorizes_training'] = True
    with pytest.raises(ValueError): E.summarize(spec, row)


@pytest.mark.parametrize('kind', ['multiply', 'memory'])
def test_delayed_learning_required(kind):
    spec, row = fixture(kind)
    row['correctness'][0]['delayed_activation']['successful_updates'] = 2
    with pytest.raises(ValueError, match='Delayed'): E.summarize(spec, row)


@pytest.mark.parametrize('fault', ['history', 'boundaries', 'diagnostics'])
def test_memory_specific_safety_is_not_lost(fault):
    spec, row = fixture('memory')
    if fault == 'history': row['correctness'][0]['checks'][-2]['final_reader_first_writer_gradient'] = False
    if fault == 'boundaries': row['correctness'][0]['checks'][-1]['overflow_rejected'] = False
    if fault == 'diagnostics': row['diagnostics'][0]['layers'][0]['memory_rms'] = 2
    with pytest.raises(ValueError): E.summarize(spec, row)


def test_slow_model_reported_not_silently_shortened():
    spec, row = fixture('memory')
    for window in row['stream']['ranks'][0]['windows']: window['seconds'] *= 10
    row['stream']['seconds'] *= 10; row['stream']['jets_per_second'] /= 10
    report = E.summarize(spec, row)
    assert report['recommended_train_hours'] > 480 and not report['within_20_day_training_ceiling']
    assert not report['authorizes_training']


def test_missing_and_failed_never_admit(tmp_path):
    assert P.summary(tmp_path)['missing'] == ['campaign.json', 'benchmark.json']
    spec, row = fixture('bias'); row['passed'] = False
    assert E.summarize(spec, row)['passed'] is False


@pytest.fixture(params=list(P.PACKAGES))
def prepared(request, tmp_path, monkeypatch):
    kind = request.param
    science = importlib.import_module(P.PACKAGES[kind]+'.protocol')
    helper = importlib.import_module('tests.test_'+P.PACKAGES[kind]+'_protocol')
    donor = tmp_path/'donor'; donor.mkdir()
    parent = dict(content_hash='spec', task=science.TASK, parameters=science.PARAMETERS,
        model=science.CONFIG, recipe=science.recipes()['ranger_full_v1'],
        inventory={'splits': {'train': {}, 'val': {}}},
        environment={'weaver_sources': {'model': 'pinned'}}, launch=dict(nccl_environment={}))
    monkeypatch.setattr(P, 'activate', lambda *a: science)
    monkeypatch.setattr(science, 'load', lambda *a, **kw: parent)
    monkeypatch.setattr(science, 'verify_inventory_metadata', lambda *a: None)
    for name in ('campaign.json', 'launch.json', 'source_manifest.json', 'data_contents.json'):
        P.write(donor/name, dict(frozen=name))
    P.write(donor/'benchmark.json', P.hashed(helper.evidence()))
    P.write(donor/'summary.json', science.review(donor))
    env = dict(architecture='aarch64', torch_cuda='12.6', installed_distributions={'torch': 'arm'})
    monkeypatch.setattr(P, 'runtime_environment', lambda *a: env.copy())
    output = tmp_path/'tigris'; output.mkdir()
    for name in P.FILES:
        path = output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# frozen adapter')
    launch = P.hashed(dict(schema=P.SCHEMA, model_kind=kind, benchmark=str(donor.resolve()),
        output=str(output.resolve()), python=sys.executable, policy=P.POLICY, partition='tigris',
        source=P.inventory(output/'source'), admission=P.inspect(donor, kind), authorizes_training=False))
    P.write(output/'launch.json', launch)
    monkeypatch.setattr(P, 'check_storage', lambda _: 6*2**30)
    return output, donor, parent, kind, env


def test_prepare_reuses_frozen_science_without_donor_writes(prepared):
    output, donor, parent, kind, _ = prepared
    before = P.inventory(donor)
    spec = P.prepare(output)
    assert spec == P.load(output)
    assert spec['parameters'] == P.COUNTS[kind] and spec['model'] == parent['model']
    assert spec['recipe']['total_steps'] == 1000000 and spec['recipe']['epochs'] == 50
    assert spec['inventory'] == parent['inventory'] and spec['environment'] != parent['environment']
    assert not spec['authorizes_training'] and not spec['cross_platform_restart']
    assert not (output/'data_contents.json').exists() and not (output/'runs').exists()
    assert P.inventory(donor) == before
    with pytest.raises(FileExistsError): P.prepare(output)


@pytest.mark.parametrize('fault', ['donor', 'source', 'environment', 'policy', 'recipe', 'launch', 'interpreter'])
def test_campaign_identity_rechecked(prepared, fault):
    output, donor, _, _, env = prepared
    spec = P.prepare(output)
    if fault == 'donor': P.write(donor/'summary.json', {'passed': True})
    if fault == 'source': (output/'source'/P.FILES[0]).write_text('changed')
    if fault == 'environment': env['torch_cuda'] = 'changed'
    if fault == 'policy': spec['policy'] = dict(P.POLICY, local_batch=256)
    if fault == 'recipe': spec['recipe'] = dict(spec['recipe'], total_steps=200000)
    if fault in ('policy', 'recipe'):
        P.write(output/'campaign.json', P.hashed({k: v for k, v in spec.items() if k != 'content_hash'}))
    if fault in ('launch', 'interpreter'):
        launch = P.checked(output/'launch.json'); launch.pop('content_hash')
        if fault == 'launch': launch['policy'] = dict(P.POLICY, world_size=2)
        else: launch['python'] = '/different/python'
        P.write(output/'launch.json', P.hashed(launch))
    with pytest.raises(ValueError): P.load(output)


def test_environment_admits_arm_versions_but_not_weaver_changes(monkeypatch):
    from part_ev_fullscale import model
    from types import SimpleNamespace
    monkeypatch.setattr(model, 'reference_metadata', lambda: {})
    current = dict(architecture='aarch64', torch_cuda='12.6', weaver_sources={'source': 'pinned'},
                   installed_distributions={'torch': 'ARM version'})
    science = SimpleNamespace(runtime_environment=lambda: current)
    parent = dict(environment=dict(weaver_sources={'source': 'pinned'}))
    assert P.runtime_environment(science, parent) == current
    current['weaver_sources'] = {'source': 'wrong'}
    with pytest.raises(ValueError, match='Weaver'): P.runtime_environment(science, parent)
    current['weaver_sources'] = parent['environment']['weaver_sources']
    current['architecture'] = 'x86_64'
    with pytest.raises(ValueError, match='ARM'): P.runtime_environment(science, parent)
