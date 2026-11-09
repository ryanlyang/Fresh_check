import copy
from pathlib import Path
import pytest
from part_ev_compact import protocol as P
from part_ev_fullscale.data import CLASSES, SPLITS


@pytest.fixture
def prepared_inputs(tmp_path,monkeypatch):
    source=tmp_path/'source';reference=tmp_path/'reference';output=tmp_path/'new_campaign'
    files=('part_ev_fullscale/model.py','part_ev_fullscale/data.py','part_ev_fullscale/optimizer.py',
           'part_ev_fullscale/checkpoint.py','part_ev_screen/precision.py','part_layerwise_screen/model.py',
           'scripts/run_part_ev_screen.py','particle_transformer/data/JetClass/JetClass_full.yaml')
    manifest={}
    for name in files:
        for parent in (source,reference/'source'):
            path=parent/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# unchanged native fixture\n')
        manifest[name]=P.file_sha256(source/name)
    inventory={'splits':{}}
    for split in ('train','val'):
        rows=[]
        directory=tmp_path/'data'/split;directory.mkdir(parents=True)
        for cls in CLASSES:
            for index in SPLITS[split][1]:
                path=directory/f'{cls}_{index:03d}.root';path.write_bytes(b'fixture')
                stat=path.stat()
                rows.append(dict(name=path.name,path=str(path),size_bytes=stat.st_size,
                                 mtime_ns=stat.st_mtime_ns,num_entries=100000))
        inventory['splits'][split]={'files':rows}
    inventory['inventory_sha256']=P.digest(inventory)
    env=dict(architecture='aarch64',installed_distributions={'torch':'fixture'})
    stream=P.hashed(dict(inventory_sha256=inventory['inventory_sha256'],environment=env,
                        fingerprints={str(k):str(k) for k in (0,20,1020,2020,4020)}))
    P.write_json(reference/'inputs/native_stream.json',stream)
    old=P.hashed(dict(schema='selected_fullscale_benchmarks_v1',source=manifest,inventory=inventory,
                      environment=env,input_reference=stream,
                      inputs={'inputs/native_stream.json':P.file_sha256(reference/'inputs/native_stream.json')}))
    P.write_json(reference/'experiment_spec.json',old)
    P.write_json(output/'launch.json',dict(reference=str(reference)))
    P.write_json(output/'snapshot.json',{'fixture':'inventory'})
    monkeypatch.setattr(P,'ROOT',source)
    monkeypatch.setattr(P,'runtime_environment',lambda:copy.deepcopy(env))
    monkeypatch.setattr(P,'require_space',lambda *_:None)
    return output,reference,inventory,env


def test_prepare_read_only_donor_then_load_without_donor(prepared_inputs):
    output,reference,inventory,env=prepared_inputs
    before={str(p):P.file_sha256(p) for p in reference.rglob('*') if p.is_file()}
    spec=P.prepare(output)
    assert spec['inventory']==inventory and spec['environment']==env
    assert P.prepare(output)==spec
    assert before=={str(p):P.file_sha256(p) for p in reference.rglob('*') if p.is_file()}
    reference.rename(reference.with_name('donor_not_needed_after_prepare'))
    assert P.load(output)==spec


@pytest.mark.parametrize('change',('data_size','data_entries','duplicates','environment','source','stream','snapshot','recipe'))
def test_prepare_and_load_reject_drift(prepared_inputs,change):
    output,reference,inventory,env=prepared_inputs
    if change=='data_size':Path(inventory['splits']['train']['files'][0]['path']).write_bytes(b'changed data size')
    elif change in ('data_entries','duplicates'):
        row=inventory['splits']['train']['files'][0]
        if change=='data_entries':row['num_entries']=1
        else:inventory['splits']['train']['files'][1]=row
        inventory['inventory_sha256']=P.digest({k:v for k,v in inventory.items() if k!='inventory_sha256'})
        with pytest.raises(ValueError):P.verify_inventory(inventory)
        return
    elif change=='environment':env['installed_distributions']['torch']='changed'
    elif change=='source':(reference/'source/part_ev_fullscale/model.py').write_text('changed')
    elif change=='stream':P.write_json(reference/'inputs/native_stream.json',P.hashed({'different':'stream'}))
    elif change in ('snapshot','recipe'):
        P.prepare(output)
        if change=='snapshot':P.write_json(output/'snapshot.json',{'unexpected':'file'})
        else:
            spec=P.read_json(output/'campaign.json');spec.pop('content_hash');spec['compact_recipe']['total_steps']=10
            P.write_json(output/'campaign.json',P.hashed(spec))
        with pytest.raises(ValueError):P.load(output)
        return
    with pytest.raises(ValueError):P.prepare(output)
    assert not (output/'campaign.json').exists()
