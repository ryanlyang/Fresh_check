import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_selected_fullscale as S
from tests.test_part_fullscale_launcher import BASH,shell_path


def args(tmp,extra=()):
    return S.parse(['--reference',str(tmp/'reference'),'--output',str(tmp/'output'),'--python',sys.executable,
                    '--baseline-control',str(tmp/'baseline'),'--baseline-evaluation',str(tmp/'evaluation'),
                    '--state-control',str(tmp/'state'),*extra])


def test_dry_run_only_and_no_production_modes(tmp_path,capsys):
    S.launch(args(tmp_path,['--dry-run']))
    assert not list(tmp_path.iterdir())
    assert 'No files changed or jobs submitted' in capsys.readouterr().out
    for mode in ('train','run','evaluate','gate'):
        with pytest.raises(SystemExit): args(tmp_path,['--mode',mode])
    assert S.graph('benchmark')==[('prepare',5,None),('profile',5,'2,5,6'),('benchmark',5,'0-6'),('summary',.5,None)]
    assert all(hours<6 for _,hours,_ in S.graph('benchmark'))


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    root=tmp_path/'repo'; root.mkdir()
    for name in ('part_selected_fullscale/model.py','scripts/run_part_selected_fullscale.py',
                 'scripts/submit_part_selected_fullscale.py','sbatch/run_part_selected_fullscale_tigris.sh'):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# fixture\n')
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture'],check=True,capture_output=True)
    (tmp_path/'reference').mkdir(); (tmp_path/'reference/experiment_spec.json').write_text('{}')
    monkeypatch.setattr(S,'ROOT',root); monkeypatch.setattr(platform,'machine',lambda:'aarch64')
    old_which=shutil.which
    monkeypatch.setattr(shutil,'which',lambda x:'/fake/'+x if x in ('sbatch','scontrol','squeue') else old_which(x))
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    monkeypatch.setenv('USER','test'); monkeypatch.setenv('SBATCH_GRES','gpu:wrong:99')
    old_command=S.command
    state=dict(jobs=[],live='',fail=None,maximum='20-00:00:00')
    def command(argv,**kwargs):
        if argv[0]=='scontrol': return 'PartitionName=tigris MaxTime='+state['maximum']
        if argv[0]=='squeue': return state['live']
        if argv[0]=='sbatch':
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,argv)
            job=str(1000+len(state['jobs'])); state['jobs'].append((job,argv,dict(kwargs['env']))); return job+';tigris'
        return old_command(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    return state


def test_exact_benchmark_dependency_graph_and_snapshot(tmp_path,scheduler):
    a=args(tmp_path); S.launch(a)
    assert len(scheduler['jobs'])==4
    for i,(job,argv,env) in enumerate(scheduler['jobs']):
        phase=S.graph('benchmark')[i][0]
        assert env['SELECTED_FULL_PHASE']==phase
        assert not any(k.startswith('SBATCH_') for k in env)
        if i: assert '--dependency='+('afterany:' if i==3 else 'afterok:')+str(999+i) in argv
        if i in (1,2): assert '--gres=gpu:gh200:1' in argv
        else: assert not any(s.startswith('--gres') for s in argv)
    assert '--array=2,5,6%1' in scheduler['jobs'][1][1]
    assert '--array=0-6%1' in scheduler['jobs'][2][1]
    S.verify_snapshot(a.output)
    (S.ROOT/'part_selected_fullscale/model.py').write_text('changed live checkout')
    S.verify_snapshot(a.output)
    assert (a.reference/'experiment_spec.json').read_text()=='{}'


def test_duplicates_failed_submission_and_corrupt_snapshot(tmp_path,scheduler):
    scheduler['fail']=2
    with pytest.raises(subprocess.CalledProcessError): S.launch(args(tmp_path))
    assert (tmp_path/'output/submitted_job_ids.tsv').read_text().splitlines()==['prepare\t1000','profile\t1001']
    scheduler['fail']=None; scheduler['live']='1001_[2,5-6%1]'
    with pytest.raises(ValueError,match='active'): S.launch(args(tmp_path,['--resume']))
    scheduler['live']=''
    (tmp_path/'output/source/part_selected_fullscale/model.py').write_text('corrupt')
    with pytest.raises(ValueError,match='Frozen source changed'): S.launch(args(tmp_path,['--resume']))


def test_resource_and_dirty_source_fail_before_writes(tmp_path,scheduler):
    scheduler['maximum']='04:00:00'
    with pytest.raises(ValueError,match='MaxTime'): S.launch(args(tmp_path))
    scheduler['maximum']='20-00:00:00'
    (S.ROOT/'part_selected_fullscale/model.py').write_text('dirty')
    with pytest.raises(ValueError,match='Commit/push'): S.launch(args(tmp_path))
    assert not (tmp_path/'output').exists() and not scheduler['jobs']


def test_committed_bytecode_is_not_source(tmp_path,scheduler):
    # git archive includes already tracked caches even when .gitignore excludes
    # new ones. These mirror the real repository, not just a clean toy checkout.
    caches=('__pycache__/jetclass_fixed_hlt.cpython-310.pyc',
            'scripts/__pycache__/demo_load_and_score_models_no_fusion.cpython-310.pyc',
            'tests/__pycache__/test_sbatch_scripts.cpython-313.pyc',
            'legacy.pyc')
    for name in caches:
        path=S.ROOT/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'fixture bytecode')
    subprocess.run(['git','-C',str(S.ROOT),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(S.ROOT),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','tracked cache regression'],check=True,capture_output=True)
    a=args(tmp_path); S.launch(a)
    S.verify_snapshot(a.output)
    manifest=json.loads((a.output/'snapshot.json').read_text())
    assert 'part_selected_fullscale/model.py' in manifest
    for name in caches:
        assert name not in manifest
        assert not (a.output/'source'/name).exists()
        assert (S.ROOT/name).is_file()  # Never modify/delete the user's originals.


def test_snapshot_is_verified_before_first_submission(tmp_path,scheduler,monkeypatch):
    def reject(_): raise ValueError('test snapshot verification failure')
    monkeypatch.setattr(S,'verify_snapshot',reject)
    with pytest.raises(ValueError,match='test snapshot verification failure'):
        S.launch(args(tmp_path))
    assert not scheduler['jobs']
    assert not (tmp_path/'output/submitted_job_ids.tsv').exists()


@pytest.mark.parametrize('change',('added','missing','changed'))
def test_real_source_inventory_and_hash_changes_still_rejected(tmp_path,scheduler,change):
    a=args(tmp_path); S.launch(a)
    model=a.output/'source/part_selected_fullscale/model.py'
    if change=='added':
        (model.parent/'unexpected.py').write_text('# added source\n')
    elif change=='missing':
        model.rename(model.with_suffix('.missing'))
    else:
        model.write_text('# changed source\n')
    with pytest.raises(ValueError,match='Frozen source'):
        S.verify_snapshot(a.output)


@pytest.mark.skipif(BASH is None,reason='Bash unavailable')
def test_bash_syntax():
    root=Path(__file__).resolve().parents[1]
    for name in ('submit','run'):
        subprocess.run([BASH,'-n',shell_path(root/f'sbatch/{name}_part_selected_fullscale_tigris.sh')],check=True)
