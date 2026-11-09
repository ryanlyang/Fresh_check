import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_ev_compact as S
from tests.test_part_fullscale_launcher import BASH, shell_path


def args(tmp, extra=()):
    return S.parse(['--output',str(S.ROOT/'checkpoints/part_ev_compact/test'),
                    '--reference',str(tmp/'reference'),'--python',sys.executable,*extra])


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    for name in ('part_ev_compact/model.py','part_ev_compact/benchmark.py','scripts/run_part_ev_compact.py',
                 'sbatch/run_part_ev_compact_tigris.sh','__pycache__/old.pyc'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','fixture'],check=True,capture_output=True)
    (tmp_path/'reference').mkdir();(tmp_path/'reference/experiment_spec.json').write_text('{}')
    monkeypatch.setattr(S,'ROOT',root)
    monkeypatch.setattr(S.platform,'machine',lambda:'aarch64')
    monkeypatch.setattr(S.P.shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    monkeypatch.setenv('SBATCH_GRES','gpu:wrong:99')
    original_command,original_run=S.command,subprocess.run
    state=dict(jobs=[],tests=[],fail=None,test_fail=False)
    def command(argv,**kw):
        if argv[0]=='sbatch':
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,argv)
            job=str(100+len(state['jobs']));state['jobs'].append((job,argv,kw['env']));return job+';tigris'
        return original_command(argv,**kw)
    def run(argv,**kw):
        if argv[0]=='sbatch':
            assert '--test-only' in argv
            state['tests'].append(argv)
            if state['test_fail']: raise subprocess.CalledProcessError(1,argv)
            return SimpleNamespace(returncode=0)
        return original_run(argv,**kw)
    monkeypatch.setattr(S,'command',command);monkeypatch.setattr(subprocess,'run',run)
    return state


def test_dry_run_no_files_and_no_production(tmp_path,scheduler,capsys):
    a=args(tmp_path,['--dry-run']);S.launch(a)
    assert not a.output.exists() and not scheduler['jobs']
    assert 'Benchmarks ONLY' in capsys.readouterr().out
    for mode in ('train','production','evaluate'):
        with pytest.raises(SystemExit): args(tmp_path,['--mode',mode])


def test_frozen_isolated_graph_and_resource_requests(tmp_path,scheduler):
    a=args(tmp_path);S.launch(a)
    assert len(scheduler['tests'])==3 and len(scheduler['jobs'])==3
    for i,(job,argv,env) in enumerate(scheduler['jobs']):
        assert '--partition=tigris' in argv
        assert '--time='+('00:30:00' if i==2 else '05:00:00') in argv
        assert not any(k.startswith('SBATCH_') for k in env)
        if i==1:
            assert '--array=0-1%2' in argv and '--gres=gpu:gh200:1' in argv
            assert '--dependency=afterok:100' in argv
        else:
            assert not any(s.startswith('--gres') for s in argv)
        if i==2: assert '--dependency=afterany:101' in argv
        assert str(a.output/'source/sbatch/run_part_ev_compact_tigris.sh') in argv
    S.verify_snapshot(a.output)
    assert not (a.output/'source/__pycache__/old.pyc').exists()
    (S.ROOT/'part_ev_compact/model.py').write_text('new live checkout')
    S.verify_snapshot(a.output)
    assert (a.reference/'experiment_spec.json').read_text()=='{}'
    assert (a.output/'submitted_job_ids.tsv').read_text().splitlines()==['prepare\t100','benchmark\t101','summary\t102']
    with pytest.raises(ValueError,match='already exists'): S.launch(a)


def test_uncertain_partial_submission_journal(tmp_path,scheduler):
    scheduler['fail']=1;a=args(tmp_path)
    with pytest.raises(subprocess.CalledProcessError): S.launch(a)
    assert json.loads((a.output/'submission_prepare.json').read_text())['status']=='accepted'
    assert json.loads((a.output/'submission_benchmark.json').read_text())['status']=='submitting'
    assert not (a.output/'submission_summary.json').exists()
    with pytest.raises(ValueError,match='already exists'): S.launch(a)


def test_resource_rejection_submits_nothing(tmp_path,scheduler):
    scheduler['test_fail']=True
    with pytest.raises(subprocess.CalledProcessError): S.launch(args(tmp_path))
    assert not scheduler['jobs']


def test_dirty_worktree_and_low_storage_are_nonmutating(tmp_path,scheduler,monkeypatch):
    a=args(tmp_path)
    monkeypatch.setattr(S.P.shutil,'disk_usage',lambda _:SimpleNamespace(free=5*2**30))
    with pytest.raises(OSError): S.launch(a)
    assert not a.output.exists()
    (S.ROOT/'part_ev_compact/model.py').write_text('uncommitted')
    with pytest.raises(ValueError,match='clean worktree'): S.launch(a)
    assert not a.output.exists() and not scheduler['jobs']


@pytest.mark.parametrize('change',('add','change','rename'))
def test_frozen_source_mutation_blocks(tmp_path,scheduler,change):
    a=args(tmp_path);S.launch(a);p=a.output/'source/part_ev_compact/model.py'
    if change=='add': (p.parent/'extra.py').write_text('x')
    elif change=='change':p.write_text('changed')
    else:p.rename(p.with_suffix('.gone'))
    with pytest.raises(ValueError,match='Frozen source'): S.verify_snapshot(a.output)


def test_output_cannot_overlap_reference_or_repo(tmp_path,scheduler):
    for path in (S.ROOT,tmp_path/'reference',tmp_path/'reference/nested',tmp_path):
        with pytest.raises(SystemExit): args(tmp_path,['--output',str(path)])


@pytest.mark.skipif(BASH is None,reason='Bash not available')
def test_shell_syntax():
    root=Path(__file__).resolve().parents[1]
    for name in ('submit','run'):
        subprocess.run([BASH,'-n',shell_path(root/f'sbatch/{name}_part_ev_compact_tigris.sh')],check=True)
