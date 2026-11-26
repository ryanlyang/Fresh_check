from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import submit_part_ev_masked_speed as C
from part_ev_masked_speed import protocol as D
from part_ev_masked_speed.config import SCHEMA, VARIANTS, POLICY


def test_requests_only_one_gh200_and_cpu_summary_without_early_signal(tmp_path):
    launch=dict(account='test')
    benchmark=C.request(tmp_path,launch,'benchmark')
    assert '--gres=gpu:gh200:1' in benchmark and '--time=330' in benchmark
    assert '--mem=128G' in benchmark and '--cpus-per-task=12' in benchmark
    summary=C.request(tmp_path,launch,'summary','afterany:123')
    assert '--time=15' in summary and '--dependency=afterany:123' in summary
    assert not any(s.startswith(('--gres','--signal')) for s in summary)
    with pytest.raises(ValueError):C.request(tmp_path,launch,'train')


@pytest.fixture
def launch_fixture(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();donor=tmp_path/'donor';donor.mkdir()
    (donor/'preserve.txt').write_text('unchanged')
    output=repo/'checkpoints/part_ev_masked_speed/diagnostic'
    original=dict(python='/env/bin/python',resources=D.P.PLATFORMS['tigris'],account='test')
    inspection=dict(launch=original,donor_files={},review=dict(runtime=dict(gpu={}),models=[dict(arm='masked')]))
    monkeypatch.setattr(C,'ROOT',repo)
    monkeypatch.setattr(C.P,'load_launch',lambda *a,**k:original)
    monkeypatch.setattr(C.D,'check_storage',lambda *a:6*2**30)
    calls=[];frozen=[]
    monkeypatch.setattr(C.S,'command',lambda *a,**k:json.dumps(inspection))
    @contextmanager
    def locks(*a):yield
    monkeypatch.setattr(C.S,'locks',locks)
    monkeypatch.setattr(C.S,'preflight',lambda *a:None)
    def freeze(repo,output,proposal):
        frozen.append(proposal)
        for name in ('part_ev_masked_speed/benchmark.py','part_ev_masked_speed/step.py',
                     'scripts/run_part_ev_masked_speed.py','sbatch/run_part_ev_masked_speed.sh'):
            p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
        return proposal
    monkeypatch.setattr(C.S,'freeze',freeze)
    def submit(out,launch,phase,dependency=None):
        calls.append((phase,dependency));return str(100+len(calls))
    monkeypatch.setattr(C,'submit',submit)
    return SimpleNamespace(mode='benchmark',from_campaign=donor,output=output),calls,frozen


def test_plan_is_readonly_and_submission_does_not_queue_production(launch_fixture):
    args,calls,frozen=launch_fixture
    args.mode='plan';C.launch(args)
    assert not calls and not frozen and not args.output.exists()
    args.mode='benchmark';C.launch(args)
    assert calls==[('benchmark',None),('summary','afterany:101')]
    assert frozen[0]['variants']==VARIANTS and not frozen[0]['authorizes_training']
    assert (args.from_campaign/'preserve.txt').read_text()=='unchanged'
    with pytest.raises(FileExistsError):C.launch(args)


def test_durable_unknown_submission_is_not_duplicated(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(C.S,'command',lambda *a,**k:calls.append(a) or 'unknown')
    with pytest.raises(ValueError,match='Ambiguous'):C.submit(tmp_path,dict(account='x',python='/env/python'),'benchmark')
    with pytest.raises(ValueError,match='Uncertain'):C.submit(tmp_path,dict(account='x',python='/env/python'),'benchmark')
    assert len(calls)==1


def test_parser_has_no_production_or_old_campaign_destination(tmp_path,monkeypatch):
    monkeypatch.setattr(C,'ROOT',tmp_path)
    for mode in ('dispatch','production','pipeline','resume'):
        with pytest.raises(SystemExit):C.parse(['--mode',mode,'--output',str(tmp_path/'out')])
    with pytest.raises(SystemExit):C.parse(['--mode','plan','--output',str(tmp_path/'checkpoints/old_campaign')])
    with pytest.raises(SystemExit):C.parse(['--mode','plan','--output',str(tmp_path/'checkpoints/part_ev_masked_speed/new')])


def test_reference_byte_drift_rejected_without_altering_donor(tmp_path,monkeypatch):
    old=tmp_path/'donor';old.mkdir();source=tmp_path/'source';source.mkdir()
    p=source/'part_ev_masked_predictive/step.py';p.parent.mkdir();p.write_text('original')
    manifest={'part_ev_masked_predictive/step.py':D.P.sha(p)}
    D.P.write(old/'source_manifest.json',manifest)
    for name in ('launch.json','campaign.json','data_contents.json','benchmark.json'):(old/name).write_text('{}')
    monkeypatch.setattr(D.P,'load_launch',lambda *a:dict(platform='tigris'))
    monkeypatch.setattr(D.P,'load',lambda *a:{})
    monkeypatch.setattr(D.P,'review',lambda *a:dict(passed=True))
    _,_,_,files=D.donor(old,source)
    p.write_text('different')
    with pytest.raises(ValueError,match='Reference implementation differs'):D.donor(old,source)
    assert all(D.P.sha(old/n)==h for n,h in files.items())


def test_launch_contract_does_not_accept_resource_or_science_drift(tmp_path,monkeypatch):
    output=tmp_path/'checkpoints/part_ev_masked_speed/new';output.mkdir(parents=True)
    row=dict(schema=SCHEMA,output=str(output),repo=str(tmp_path),variants=VARIANTS,policy=POLICY,
        platform='tigris',resources=D.P.PLATFORMS['tigris'],authorizes_training=False,source_sha256='source')
    monkeypatch.setattr(D.P,'verify_snapshot',lambda *a:'source')
    def write():D.P.write(output/'launch.json',D.P.hashed(row))
    write();assert D.load(output)['schema']==SCHEMA
    row['resources']=dict(row['resources'],gpus=2);write()
    with pytest.raises(ValueError):D.load(output)


def test_storage_guard_does_not_delete(tmp_path,monkeypatch):
    keep=tmp_path/'keep';keep.write_text('keep')
    monkeypatch.setattr(D.shutil,'disk_usage',lambda *a:SimpleNamespace(free=1))
    with pytest.raises(OSError,match='nothing deleted'):D.check_storage(tmp_path/'new')
    assert keep.read_text()=='keep' and not (tmp_path/'new').exists()
