import pytest
import torch
from part_ev_compact import protocol as P
from part_ev_compact import flops as C
from part_ev_compact.model import build_model
from part_ev_fullscale.protocol import learning_rate
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_real_operator_counts_and_recomputation():
    from part_ev_fullscale.model import build_model as native
    assert C.measure(native('part',trim=False),128,torch.device('cpu'),backward=False)['per_jet']==669519360
    costs={}
    for arch,expected in [('compact',942738944),('matched',699928064)]:
        result=C.curve(build_model(arch,trim=False),torch.device('cpu'))
        assert result['samples']['128']['forward']['per_jet']==expected
        # Also cover unsampled lengths and chunk boundaries.
        for n in (2,9,33,66):
            measured=C.measure(build_model(arch,trim=False),n,torch.device('cpu'),backward=True)['per_jet']
            inferred=C.mean_cost(result['coefficients']['training'],{str(n):1},singleton_cost=1)
            assert inferred==measured
        costs[arch]=C.mean_cost(result['coefficients']['training'],{'128':4200},
                               singleton_cost=result['samples']['1']['training']['per_jet'])
    assert costs['compact']==2861771264
    assert costs['matched']==2099784192
    report=C.budget(costs['compact'],costs['matched'])
    assert report['matched_steps']==1362850
    assert report['matched_steps_per_check']==27257
    assert report['matched_training_flops']<=report['compact_training_flops']


def test_same_fifty_checks_and_fractional_decay_schedule():
    base=P.recipe('compact'); longer=P.recipe('matched',1362850)
    assert base['epochs']==longer['epochs']==50
    for check in range(50):
        assert learning_rate(check*20000,base)==learning_rate(check*27257,longer)
    assert longer['steps_per_epoch']*50==longer['total_steps']
    assert longer['pair_bias']=='shared' and base['pair_bias']=='layerwise'
    with pytest.raises(ValueError): P.recipe('matched',1000001)
    with pytest.raises(ValueError): P.recipe('compact',1362850)


@pytest.mark.parametrize('a,b',[(0,1),(1,0),(-1,1),(float('nan'),1),(float('inf'),1),(1,10),(100,1)])
def test_bad_budget_rejected(a,b):
    with pytest.raises((ValueError,OverflowError)): C.budget(a,b)


def test_curves_and_histograms_fail_closed():
    with pytest.raises(ValueError): C.quadratic({'1':1,'2':4,'3':9,'4':17})
    with pytest.raises(ValueError): C.mean_cost(['0','1','1'],{},singleton_cost=1)
    with pytest.raises(ValueError): C.mean_cost(['0','1','1'],{'129':4},singleton_cost=1)
    assert C.mean_cost(['0','1','1'],{'1':1,'2':1},singleton_cost=10)==8


def test_installed_distribution_inventory_uses_explicit_install_roots(monkeypatch):
    from types import SimpleNamespace
    import part_ev_fullscale.protocol as original
    monkeypatch.setattr(original,'environment',lambda:{'test':'core'})
    observed=[]
    def distributions(*,path):
        observed.append(path)
        return [SimpleNamespace(metadata={'Name':'Example'},version='1')]
    monkeypatch.setattr(P.importlib.metadata,'distributions',distributions)
    assert P.runtime_environment()['installed_distributions']=={'example':'1'}
    assert observed and all('_vendor' not in p for p in observed[0])


@pytest.mark.parametrize('order', ('new_first', 'old_first', 'interleaved'))
def test_duplicate_distribution_records_match_donor_convention(monkeypatch, order):
    """RC's five duplicate records must not make an unchanged inventory drift."""
    from types import SimpleNamespace
    import part_ev_fullscale.protocol as original
    pairs = [('fsspec', '2026.6.0', '2026.4.0'),
             ('numpy', '2.2.6', '1.26.4'),
             ('pillow', '12.3.0', '12.2.0'),
             ('setuptools', '81.0.0', '78.1.0'),
             ('typing_extensions', '4.16.0', '4.15.0')]
    records = [(name, version) for name, new, old in pairs for version in (new, old)]
    if order == 'old_first': records.reverse()
    elif order == 'interleaved': records = records[::2] + records[1::2]
    core = {'packages': {'numpy': '2.2.6'}, 'weaver_sources': {'model': 'unchanged'}}
    monkeypatch.setattr(original, 'environment', lambda: dict(core))
    calls = []
    def distributions(*, path):
        calls.append(path)
        return [SimpleNamespace(metadata={'Name': name}, version=version) for name, version in records]
    monkeypatch.setattr(P.importlib.metadata, 'distributions', distributions)
    current = P.runtime_environment()
    # This is precisely the donor's serialization convention, not selection of
    # a preferred version based on what would make the current check pass.
    assert current['installed_distributions'] == dict(sorted(records))
    assert current['installed_distributions'] == {name: new for name, new, old in pairs}
    assert current['packages'] == core['packages']
    assert current['weaver_sources'] == core['weaver_sources']
    assert calls and all('_vendor' not in path for path in calls[0])


def test_duplicate_fix_preserves_lexical_rule_and_real_core_differences(monkeypatch):
    from types import SimpleNamespace
    import part_ev_fullscale.protocol as original
    # Deliberately unlike semantic ordering. Preserve the frozen donor format;
    # do not invent a new "highest version" or suppress actual core drift.
    records = [('example', '10.0'), ('example', '9.0')]
    monkeypatch.setattr(P.importlib.metadata, 'distributions', lambda *, path:
        [SimpleNamespace(metadata={'Name': name}, version=version) for name, version in records])
    monkeypatch.setattr(original, 'environment', lambda: {'packages': {'numpy': '1.26.4'}})
    current = P.runtime_environment()
    assert current['installed_distributions'] == {'example': '9.0'}
    assert current['packages']['numpy'] == '1.26.4'
