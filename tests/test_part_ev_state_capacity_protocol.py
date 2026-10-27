import copy
import pytest
from part_ev_state_capacity import protocol as P
from part_ev_state_fullscale import protocol as old
from part_ev_message_study.controls import runtime_identity
from tests.test_part_ev_state_fullscale_protocol import fixture as old_fixture, hashed


def fixture(root,monkeypatch):
    source, runtime = old_fixture(root/'old')
    old_summary = old.benchmark_summary(root/'old',source)
    monkeypatch.setattr(old,'require_approval',lambda *a:old_summary)
    old.write_json(root/'old/campaign.json',source)
    spec = dict(content_hash='new',controls=dict(path=str(root/'old'),runtime=runtime_identity(runtime)))
    for index in P.BENCHMARK_INDICES:
        row = copy.deepcopy(old_summary['models'][0])
        row.update(task=P.TASKS[index],campaign_sha256='new')
        row['runtime']['campaign_sha256'] = 'new'
        parity = row['initial_parity']
        if index == 0:
            parity.update(reference_kind='ordinary_EV_zero_state_branch',state_gradients=dict(
                passed=True,checkpoint_backward=True,tensors=19,forward=dict(passed=True),max_absolute_error=1e-6))
        else:
            parity.update(reference_kind='independent_width599_ParT',capacity_control=dict(
                parameters=2323750,target=2323710,delta=40,particle_ffn_width=599,class_ffn_width=512))
        P.write_json(root/'benchmarks'/(P.TASKS[index]['run_id']+'.json'),hashed(row))
    return spec,old_summary


def test_full_recipe_and_ten_tasks_no_old_mutation():
    assert len(P.TASKS)==10 and P.TRAIN_INDICES==tuple(range(6))
    assert P.BENCHMARK_INDICES == (0,3)
    assert P.TASKS[6:] == [old.TASKS[i] for i in (0,1,2,6)]
    assert [t['seed'] for t in P.TASKS[:6]] == [101,202,303]*2
    assert [t['architecture'] for t in P.TASKS[:6]] == ['state_fast']*3+['part_state_matched']*3
    assert P.recipes()==old.recipes()
    assert P.recipes()['ranger_full_v1']['total_steps']==1000000
    assert P.recipes()['ranger_full_v1']['epochs']==50
    assert len(old.TASKS)==7 and old.BENCHMARK_INDICES==(0,3,6)
    assert 'part_ev_state_capacity/model.py' in P.source_manifest(P.ROOT)
    assert 'part_ev_state_capacity/model.py' not in old.source_manifest(P.ROOT)


@pytest.mark.parametrize('failure',['none','gradient','wide','runtime','old_stream','new_stream','missing','approval'])
def test_two_probe_admission_and_reused_runtime(tmp_path,monkeypatch,failure):
    spec,old_summary=fixture(tmp_path,monkeypatch)
    assert P.benchmark_summary(tmp_path,spec)['passed']
    if failure=='none': return
    if failure=='old_stream': old_summary['models'][0]['input_fingerprints']['0']='changed'
    elif failure=='runtime': spec['controls']['runtime']['gpu']='GH200'
    else:
        path=tmp_path/'benchmarks'/('PART_STATE_MATCHED_RANGER.json' if failure=='wide' else 'EV_STATE_FAST_RANGER.json')
        if failure=='missing': path.unlink()
        else:
            if failure=='approval': P.approve(tmp_path,spec,reviewed=True)
            row=P.read_json(path)
            if failure=='gradient': row['initial_parity']['state_gradients']['passed']=False
            if failure=='wide': row['initial_parity']['capacity_control']['parameters']+=1
            if failure=='new_stream': row['input_fingerprints']['0']='changed'
            if failure=='approval': row['amp_skipped_steps']=1
            P.write_json(path,hashed(row))
    with pytest.raises((ValueError,FileNotFoundError)):
        if failure=='approval': P.require_approval(tmp_path,spec)
        else: P.benchmark_summary(tmp_path,spec)


def test_frozen_campaign_includes_controls(tmp_path,monkeypatch):
    contents=dict(files={})
    inventory=dict(splits={}); inventory['inventory_sha256']=P.digest(inventory)
    launch=dict(controls='/original',python='/python',data_root='/data',gpu='a100',
                partition='tier3',account='account',cpus='8',memory='64G')
    original=dict(inventory=inventory,data_contents_sha256=P.digest(contents),launch=launch)
    contract=dict(path='/original',campaign_sha256='old',jobs={'train':'123'})
    monkeypatch.setattr(P,'control_contract',lambda *a,**k:(original,contract))
    monkeypatch.setattr(P._base,'runtime_environment',lambda:dict(version='fixture'))
    spec=P.create_spec(inventory,launch,contents)
    P.write_json(tmp_path/'campaign.json',spec); P.write_json(tmp_path/'launch.json',launch)
    P.write_json(tmp_path/'data_contents.json',contents)
    assert P.load_campaign(tmp_path)['controls']==contract
    contract['campaign_sha256']='changed'
    with pytest.raises(ValueError,match='provenance'): P.load_campaign(tmp_path)


@pytest.mark.parametrize('key',['python','data_root','gpu','partition','account','cpus','memory'])
def test_new_resources_must_match_controls(key,monkeypatch):
    inventory={}; contents={}
    launch={k:'same' for k in ('python','data_root','gpu','partition','account','cpus','memory','controls')}
    source=dict(inventory=inventory,data_contents_sha256=P.digest(contents),launch=dict(launch))
    monkeypatch.setattr(P,'control_contract',lambda *a:(source,{}))
    launch[key]='different'
    with pytest.raises(ValueError,match='resource differs'): P.create_spec(inventory,launch,contents)


@pytest.mark.parametrize('failure',['none','source','duplicate','missing','nonnumeric'])
def test_original_control_contract_is_readonly_and_fail_closed(tmp_path,monkeypatch,failure):
    frozen=tmp_path/'source/original.py'; frozen.parent.mkdir()
    frozen.write_text('# frozen\n')
    spec=dict(content_hash='old',source={'original.py':P.file_sha256(frozen)})
    P.write_json(tmp_path/'campaign.json',spec)
    P.write_json(tmp_path/'production_approval.json',dict(reviewed=True))
    ledger=tmp_path/'submitted_job_ids.tsv'
    ledger.write_text('train\t100\nevaluate\t101\nreport\t102\n')
    monkeypatch.setattr(old,'load_campaign',lambda *a,**k:spec)
    monkeypatch.setattr(old,'require_approval',lambda *a:dict(runtime=dict(gpu='A100',campaign_sha256='old')))
    before={p:P.file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert P.control_contract(tmp_path)[1]['jobs']==dict(train='100',evaluate='101',report='102')
    assert {p:P.file_sha256(p) for p in before}==before
    if failure=='none': return
    if failure=='source': frozen.write_text('# changed\n')
    if failure=='duplicate': ledger.write_text(ledger.read_text()+'train\t110\n')
    if failure=='missing': ledger.write_text('train\t100\nreport\t102\n')
    if failure=='nonnumeric': ledger.write_text('train\t100_0\nevaluate\t101\nreport\t102\n')
    with pytest.raises(ValueError): P.control_contract(tmp_path)
