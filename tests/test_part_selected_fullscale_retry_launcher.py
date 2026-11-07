import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import pytest
from scripts import retry_part_selected_fullscale as S
from part_selected_fullscale import protocol as P
from tests.test_part_fullscale_launcher import BASH, shell_path


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'campaign'; (root/'source').mkdir(parents=True)
    (root/'source/frozen.py').write_text('# original science\n')
    (root/'snapshot.json').write_text(json.dumps({'frozen.py':S.sha(root/'source/frozen.py')}))
    launch=dict(partition='tigris',account='reu-aisocial',python=sys.executable)
    P.write_json(root/'launch.json',launch)
    spec=P.hashed(dict(schema='selected_fullscale_benchmarks_v1',launch=launch,tasks=P.TASKS,authorizes_training=False))
    P.write_json(root/'experiment_spec.json',spec)
    for name in S.SELECTED_IDS:
        for kind in ('profiles','benchmarks'):
            P.write_json(root/kind/(name+'.json'),P.hashed(dict(complete=True,passed=True,campaign_sha256=spec['content_hash'])))
    diagnostic=root/'diagnostics/restart.test'; diagnostic.mkdir(parents=True)
    (diagnostic/'probe.py').write_text('# source\n')
    P.write_json(diagnostic/'plan.json',dict(schema='selected_restart_diagnostic_v1',campaign=str(root),
        campaign_sha256=spec['content_hash'],diagnostic_source={'probe.py':S.sha(diagnostic/'probe.py')},
        artifacts={n:S.sha(root/n) for n in ('experiment_spec.json','launch.json','snapshot.json')}))
    P.write_json(diagnostic/'report.json',dict(complete=True,authorizes_training=False,authorizes_benchmark_retry=False))
    for i in S.INDICES:
        P.write_json(diagnostic/f'task_{i}.json',dict(task=spec['tasks'][i],campaign_sha256=spec['content_hash'],
            complete=True,exact_roundtrip=True,identical_initial_states=True,uninterrupted_repeat_exact=True,
            resumed_exact=True,update_sequences_equal=True))
    return root,diagnostic


def args(campaign,extra=()):
    return S.parse(['--campaign',str(campaign[0]),'--diagnostic',str(campaign[1]),*extra])


def test_dry_run_requires_passed_diagnostic_and_preserves_all_files(campaign,capsys):
    root,diagnostic=campaign
    before={p:S.sha(p) for p in root.rglob('*') if p.is_file()}
    S.submit(args(campaign,['--dry-run']))
    assert 'five hours' in capsys.readouterr().out
    assert before=={p:S.sha(p) for p in root.rglob('*') if p.is_file()}
    assert not (root/'benchmark_retries').exists()
    row=S.read(diagnostic/'task_3.json'); row['exact_roundtrip']=False
    P.write_json(diagnostic/'task_3.json',row)
    with pytest.raises(ValueError,match='did not establish'):
        S.submit(args(campaign,['--dry-run']))


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    repo=tmp_path/'repo'; repo.mkdir()
    for name in S.FILES:
        path=repo/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(S.ROOT/name,path)
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-m','fixture'],check=True,capture_output=True)
    monkeypatch.setattr(S,'ROOT',repo)
    monkeypatch.setattr(platform,'machine',lambda:'aarch64')
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    old_which=shutil.which
    monkeypatch.setattr(shutil,'which',lambda n:'/fake/'+n if n in ('sbatch','squeue') else old_which(n))
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    monkeypatch.setenv('SBATCH_GRES','gpu:wrong:99')
    real=S.command
    state=dict(jobs=[],live='',fail=None)
    def command(argv,**kwargs):
        if argv[0]=='squeue': return state['live']
        if argv[0]=='sbatch':
            if state['fail']==len(state['jobs']): raise subprocess.CalledProcessError(1,argv)
            job=str(1000+len(state['jobs'])); state['jobs'].append((argv,kwargs['env'])); return job+';tigris'
        return real(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    return state


def test_only_four_arm_array_and_afterany_summary(campaign,scheduler):
    root,_=campaign
    before={p:S.sha(p) for p in root.rglob('*') if p.is_file()}
    S.submit(args(campaign))
    assert len(scheduler['jobs'])==2
    array,env=scheduler['jobs'][0]
    assert '--array=0,1,3,4%1' in array and '--time=05:00:00' in array
    assert '--gres=gpu:gh200:1' in array and '--mem=128G' in array
    assert not any(k.startswith('SBATCH_') for k in env)
    summary,_=scheduler['jobs'][1]
    assert '--dependency=afterany:1000' in summary and '--time=00:30:00' in summary
    assert not any(x.startswith(('--array','--gres')) for x in summary)
    output=Path(array[-2])
    plan=S.load_plan(output)
    assert plan['indices']==[0,1,3,4] and not plan['authorizes_training']
    assert all(S.sha(p)==digest for p,digest in before.items())
    scheduler['live']='1000_[0-4%1]'
    with pytest.raises(ValueError,match='active'):
        S.submit(args(campaign))
    assert len(scheduler['jobs'])==2
    (output/'parity.py').write_text('# edited')
    with pytest.raises(ValueError,match='Retry snapshot changed'):
        S.load_plan(output)


def test_partial_submission_guard_and_original_evidence_tampering(campaign,scheduler):
    scheduler['fail']=1
    with pytest.raises(subprocess.CalledProcessError): S.submit(args(campaign))
    output=Path(scheduler['jobs'][0][0][-2])
    assert (output/'submitted_job_ids.tsv').read_text().startswith('benchmark\t1000\t')
    scheduler['live']='1000_0'
    with pytest.raises(ValueError,match='active'): S.submit(args(campaign))
    original=campaign[0]/'benchmarks'/f'{S.SELECTED_IDS[0]}.json'
    row=S.read(original); row['changed']=True; P.write_json(original,P.hashed({k:v for k,v in row.items() if k!='content_hash'}))
    with pytest.raises(ValueError,match='Original evidence changed'): S.load_plan(output)


def test_old_source_and_diagnostic_cannot_change(campaign,scheduler):
    S.submit(args(campaign)); output=Path(scheduler['jobs'][0][0][-2])
    path=campaign[0]/'source/frozen.py'; path.write_text('# modified original science')
    with pytest.raises(ValueError,match='Original source changed'): S.load_plan(output)


def test_overlay_loads_original_committed_science_in_fresh_process(tmp_path):
    """Exercise the real import boundary, not just a mocked retry callback."""
    archive=tmp_path/'original.tar'; source=tmp_path/'source'; source.mkdir()
    subprocess.run(['git','-C',str(S.ROOT),'archive','--format=tar','--output',str(archive),'HEAD'],check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(source,filter='data')
    # The overlay files live outside the archived package, as they do on RC.
    for name in S.FILES:
        shutil.copyfile(S.ROOT/name,tmp_path/Path(name).name)
    code = '''
import importlib.util, sys
from pathlib import Path
source, overlay = map(Path, sys.argv[1:])
sys.path.insert(0,str(source))
import torch
torch.set_num_threads(2)
from part_selected_fullscale import protocol as P
before=P.source_manifest()
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module); return module
parity=load('part_selected_fullscale._retry_parity',overlay/'parity.py')
bench=load('part_selected_fullscale._retry_benchmark',overlay/'benchmark.py')
reviewer=load('_retry_review',overlay/'run_part_selected_fullscale.py')
assert bench.P is reviewer.P is P
assert P.ROOT == source
assert bench.full_recipe_restart.__module__ == 'part_selected_fullscale.parity'
assert parity.training_step is bench.training_step
assert Path(sys.modules[bench.build_model.__module__].__file__).is_relative_to(source)
assert Path(sys.modules[bench.build_optimizer.__module__].__file__).is_relative_to(source)
from tests.test_part_selected_fullscale_model import factory, batch_for
result=parity.full_recipe_restart(factory('part_match_shared'),batch_for('part_match_shared'),torch.device('cpu'))
bench.validate_retry_parity(dict(passed=True,full_recipe_restart=result))
assert P.source_manifest() == before
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(source),str(tmp_path)],
                           cwd=tmp_path,capture_output=True,text=True,timeout=120)
    assert result.returncode==0, result.stdout+'\n'+result.stderr


@pytest.mark.skipif(BASH is None,reason='Bash unavailable')
def test_bash_syntax():
    for name in ('submit','run'):
        subprocess.run([BASH,'-n',shell_path(S.ROOT/f'sbatch/{name}_part_selected_fullscale_retry.sh')],check=True)
