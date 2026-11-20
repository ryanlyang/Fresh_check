import copy
import importlib
import math
import pytest
from part_ev_rts_tigris import protocol as P, evidence as E

FIXTURES = {'readout':'test_part_ev_pair_readout_protocol',
            'triangles':'test_part_ev_triangles_protocol','support':'test_part_ev_support_protocol'}


def evidence(kind):
    row = importlib.import_module('tests.'+FIXTURES[kind]).evidence()
    # Synthetic test fixture conversion only; production never fabricates ranks.
    for key in ('stream','fixed','baseline_stream','baseline_fixed','adversarial_ties'):
        if key in row: row[key]['ranks'] = row[key]['ranks'][:1]
    for key in ('correctness','gpus','baseline_gpus','diagnostics','profiles','neighbor_statistics',
                'readout_gradient_norms','triangle_gradient_norms','support_gradient_norms'):
        if key in row: row[key] = row[key][:1]
    for group in ('gpus','baseline_gpus'):
        row[group][0].update(name='NVIDIA GH200',total_memory=96*2**30,cold_fetch_seconds=100.)
    row['correctness'][0]['execution'] = dict(E.EXECUTION)
    for check in row['correctness'][0]['checks']:
        check['kind'] = E.RENAMES.get(check['kind'],check['kind'])
    if kind == 'triangles':
        for phase in ('real_train','adversarial_train','real_validation'):
            row['profiles'][0][phase]['sample']['source_events'] = 512
    row['host_peak_rss_bytes'] = [2**30]
    row['runtime'] = P.runtime()
    row['restart']['ranks'] = 1
    return row


def spec(kind):
    return dict(content_hash='spec',model_kind=kind,parameter_inventory_sha256='inventory')


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_native_full_gates_and_projection(kind):
    result = E.summarize(spec(kind),evidence(kind))
    assert result['passed'] and result['complete'] and not result['authorizes_training']
    assert result['parameters'] == P.COUNTS[kind]
    assert result['recommended_train_hours'] == math.ceil(1.3*(500000+15000+1152+100)/3600)
    assert result['runtime']['batchnorm'] == 'native'
    assert result['training_time_vs_state_ev'] == 1


def mutations():
    for kind,name in FIXTURES.items():
        test = importlib.import_module('tests.'+name).test_fail_closed
        for mark in test.pytestmark:
            if mark.name != 'parametrize': continue
            for path,value in mark.args[1]:
                if 1 in path: continue  # Removed second rank; other mutations tested below.
                if path == ('baseline_gpus',0,'name'): value='NVIDIA A100'
                yield kind,path,value


@pytest.mark.parametrize('kind,path,value',list(mutations()))
def test_original_model_gates_still_fail_closed(kind,path,value):
    row=evidence(kind); target=row
    for key in path[:-1]: target=target[key]
    target[path[-1]]=value
    with pytest.raises((ValueError,KeyError)): E.summarize(spec(kind),row)


@pytest.mark.parametrize('kind',list(P.PACKAGES))
@pytest.mark.parametrize('fault',('two_ranks','cpu','syncbn','low_vram','wrong_batch','reserved','profile_source'))
def test_native_specific_gates(kind,fault):
    row=evidence(kind)
    if fault=='two_ranks':row['stream']['ranks']*=2
    if fault=='cpu':row['correctness'][0]['execution']['device_type']='cpu'
    if fault=='syncbn':row['runtime']['batchnorm']='SyncBatchNorm'
    if fault=='low_vram':row['gpus'][0]['total_memory']=40*2**30
    if fault=='wrong_batch':row['runtime']['local_batch']=256
    if fault=='reserved':row['gpus'][0]['peak_reserved']=1000*2**30
    if fault=='profile_source':
        if kind!='triangles':return
        row['profiles'][0]['adversarial_train']['sample']['source_events']=256
    with pytest.raises(ValueError):E.summarize(spec(kind),row)


@pytest.mark.parametrize('kind',list(P.PACKAGES))
def test_failure_is_not_admission(kind):
    row=evidence(kind);row['passed']=False
    assert not E.summarize(spec(kind),row)['passed']
    row=evidence(kind);row['complete']=False
    assert not E.summarize(spec(kind),row)['complete']
