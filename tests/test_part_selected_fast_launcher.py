import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_selected_fast as S
from tests.test_part_fullscale_launcher import BASH,shell_path


def arguments(tmp_path,mode='benchmark',extra=()):
    return S.parse(['--mode',mode,'--reference',str(tmp_path/'reference'),
                    '--output',str(tmp_path/'output'),'--python',sys.executable,*extra])


def test_dry_run_is_read_only_and_all_checks_under_six_hours(tmp_path,capsys):
    for mode in ('profile','benchmark','run','train','evaluate','report'):
        args=arguments(tmp_path,mode,['--dry-run']); S.launch(args)
        assert not list(tmp_path.iterdir())
        for phase,hours,array in S.graph(mode,args):
            if phase not in ('train','evaluate'): assert hours<6
            assert array==(phase in ('profile','benchmark','train','evaluate'))
    assert 'No files changed or jobs submitted' in capsys.readouterr().out


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    root=tmp_path/'repo'; root.mkdir()
    for name in ('scripts/submit_part_selected_fast.py','part_selected_fast/engine.py',
                 'part_selected_fast/csrc/tree_core.h','sbatch/run_part_selected_fast_tigris.sh'):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# fixture\n')
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','test fixture'],check=True,capture_output=True)
    reference=tmp_path/'reference'; reference.mkdir(); (reference/'experiment_spec.json').write_text('{}')
    monkeypatch.setattr(S,'ROOT',root); monkeypatch.setattr(platform,'machine',lambda:'aarch64')
    original_which=shutil.which
    monkeypatch.setattr(shutil,'which',lambda name: '/fake/'+name if name in ('sbatch','squeue','scontrol') else original_which(name))
    monkeypatch.setenv('USER','test'); monkeypatch.setenv('SBATCH_ARRAY_INX','0-200')
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    original_command=S.command
    state={'jobs':[],'live':'','maxtime':'20-00:00:00','fail':None}
    def command(args,**kwargs):
        if args[0]=='scontrol': return 'PartitionName=tigris MaxTime='+state['maxtime']
        if args[0]=='squeue': return state['live']
        if args[0]=='sbatch':
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,args)
            job=str(1000+len(state['jobs']))
            state['jobs'].append((job,list(args),dict(kwargs['env'])))
            return job+';tigris'
        return original_command(args,**kwargs)
    monkeypatch.setattr(S,'command',command)
    return state


def test_full_dependency_graph_immutable_source_and_no_controls(tmp_path,scheduler):
    args=arguments(tmp_path,'run'); S.launch(args)
    expected=['prepare','profile','benchmark','gate','train','lock','evaluate','report']
    assert len(scheduler['jobs'])==len(expected)
    for index,(job,argv,env) in enumerate(scheduler['jobs']):
        assert env['PART_SELECTED_FAST_PHASE']==expected[index]
        assert not any(k.startswith('SBATCH_') for k in env)
        if index:
            assert '--dependency=afterok:'+str(999+index) in argv
            assert '--kill-on-invalid-dep=yes' in argv
        if expected[index] in ('profile','benchmark','train','evaluate'):
            assert '--array=0-3%2' in argv and '--gpus-per-node=1' in argv
        else: assert not any(x.startswith('--gpus') for x in argv)
        assert str(args.reference) not in argv[-1]
    S.verify_snapshot(args.output)
    # A later main-repo edit cannot modify the frozen copy or resumed worker.
    (S.ROOT/'part_selected_fast/engine.py').write_text('# changed live checkout\n')
    S.verify_snapshot(args.output)
    assert (args.reference/'experiment_spec.json').read_text()=='{}'
    assert set(p.name for p in args.reference.iterdir())=={'experiment_spec.json'}


def test_retry_guards_active_jobs_settings_snapshot_and_gate(tmp_path,scheduler):
    args=arguments(tmp_path); S.launch(args)
    scheduler['live']='1001_[0-3%2]'
    with pytest.raises(ValueError,match='active'):
        S.launch(arguments(tmp_path,'train',['--resume']))
    scheduler['live']=''
    with pytest.raises(ValueError,match='Settings differ'):
        S.launch(arguments(tmp_path,'train',['--resume','--cache-gib','1']))
    S.launch(arguments(tmp_path,'train',['--resume']))
    assert scheduler['jobs'][4][2]['PART_SELECTED_FAST_PHASE']=='gate'
    (args.output/'source/part_selected_fast/engine.py').write_text('changed')
    with pytest.raises(ValueError,match='Frozen source changed'):
        S.launch(arguments(tmp_path,'train',['--resume']))


def test_failed_submission_retains_ledger_and_never_cancels(tmp_path,scheduler):
    scheduler['fail']=2
    with pytest.raises(subprocess.CalledProcessError): S.launch(arguments(tmp_path,'run'))
    ledger=(tmp_path/'output/submitted_job_ids.tsv').read_text()
    assert ledger.splitlines()==['prepare\t1000','profile\t1001']
    scheduler['live']='1001_0'; scheduler['fail']=None
    with pytest.raises(ValueError,match='active'): S.launch(arguments(tmp_path,'run',['--resume']))
    assert len(scheduler['jobs'])==2


def test_partition_limit_prevents_any_writes_or_submission(tmp_path,scheduler):
    scheduler['maxtime']='1-00:00:00'
    with pytest.raises(ValueError,match='MaxTime'): S.launch(arguments(tmp_path,'run'))
    assert not (tmp_path/'output').exists() and not scheduler['jobs']


def test_dirty_source_and_path_overlap_refused(tmp_path,scheduler):
    (S.ROOT/'part_selected_fast/engine.py').write_text('dirty')
    with pytest.raises(ValueError,match='Commit/push'): S.launch(arguments(tmp_path))
    assert not (tmp_path/'output').exists()
    with pytest.raises(SystemExit):
        S.parse(['--reference',str(tmp_path),'--output',str(tmp_path/'child')])


@pytest.mark.skipif(BASH is None,reason='Bash unavailable')
def test_shell_syntax():
    root=Path(__file__).resolve().parents[1]
    for name in ('submit','run'):
        subprocess.run([BASH,'-n',shell_path(root/f'sbatch/{name}_part_selected_fast_tigris.sh')],check=True)
