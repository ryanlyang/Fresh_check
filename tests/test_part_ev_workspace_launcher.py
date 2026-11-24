import copy
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from part_ev_workspace import protocol as P, submission as S
from part_ev_workspace.config import *
from scripts import submit_part_ev_workspace as CLI
from tests.test_part_ev_workspace_protocol import evidence


@pytest.fixture
def setup(tmp_path,monkeypatch):
    repo=tmp_path/'repository with spaces';repo.mkdir()
    (repo/'.gitignore').write_text('checkpoints/\n')
    for name in ('part_ev_workspace/model.py','scripts/run_part_ev_workspace.py','sbatch/run_part_ev_workspace.sh'):
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    subprocess.run(['git','init',str(repo)],capture_output=True,check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],capture_output=True,check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture'],capture_output=True,check=True)
    output=repo/'checkpoints/part_ev_workspace/campaign'
    proposal=dict(schema=SCHEMA,science_sha256=science_hash(),platform='sporc',output=str(output),repo=str(repo),
        python=sys.executable,data=str(tmp_path/'data'),account='reu-aisocial',resources=PLATFORMS['sporc'],
        tasks=TASKS,policy=POLICY,environment={},auto_production=True,reservation_bytes=campaign_budget(),
        max_train_hours=480,max_evaluate_hours=24)
    output.parent.mkdir(parents=True)
    launch=S.freeze(repo,output,proposal)
    calls=[];state=dict(fail=False,ambiguous=False)
    real=S.command
    def command(argv,**kwargs):
        if argv[0]=='sbatch':
            if '--test-only' in argv:
                if state['fail']:raise subprocess.CalledProcessError(1,argv)
                return 'valid'
            calls.append((argv,kwargs))
            if state['ambiguous']:return 'unrecognized reply'
            return str(1000+len(calls))+';cluster'
        if argv[0]=='scontrol':return 'State=UP MaxTime=20-00:00:00'
        if argv[0]=='squeue':return ''
        return real(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    old=shutil.which
    monkeypatch.setattr(shutil,'which',lambda n:'/fake/'+n if n in ('sbatch','scontrol','squeue') else old(n))
    return repo,output,launch,calls,state


@pytest.mark.parametrize('platform',['sporc','tigris'])
def test_phase_resources_and_no_cross_cluster_dependencies(platform,tmp_path):
    launch=dict(platform=platform,resources=PLATFORMS[platform],account='reu-aisocial')
    b=S.request(tmp_path,launch,'benchmark')
    assert '--time=330' in b and '--mem=128G' in b
    assert '--gres=gpu:'+PLATFORMS[platform]['gpu']+':'+str(PLATFORMS[platform]['gpus']) in b
    plan=dict(models=[dict(arm=a,train_hours=200,evaluate_hours=3) for a in COUNTS])
    e=S.request(tmp_path,launch,'evaluate',arm='workspace',plan=plan,dependency='afterok:123')
    assert '--gres=gpu:'+PLATFORMS[platform]['gpu']+':1' in e
    assert '--dependency=afterok:123' in e
    assert not any('array' in arg for arg in b)


def test_environment_sanitizes_submit_overrides(monkeypatch):
    for k in ('SBATCH_GRES','SLURM_MEM_PER_CPU','RANK','WORLD_SIZE','PYTHONPATH','NCCL_DEBUG','CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(k,'bad')
    env=S.environment(sys.executable,'sporc')
    assert not any(k in env for k in ('SBATCH_GRES','SLURM_MEM_PER_CPU','RANK','WORLD_SIZE','PYTHONPATH','NCCL_DEBUG','CUDA_VISIBLE_DEVICES'))
    assert env['TORCH_NCCL_ASYNC_ERROR_HANDLING']=='1'


def test_freeze_unchanged_by_live_pull_and_detects_worker_drift(setup):
    repo,output,launch,_,_=setup
    P.verify_snapshot(output)
    (repo/'part_ev_workspace/model.py').write_text('later checkout change')
    assert (output/'source/part_ev_workspace/model.py').read_text()=='# fixture\n'
    P.verify_snapshot(output)
    (output/'source/part_ev_workspace/model.py').write_text('tampered')
    with pytest.raises(ValueError):P.verify_snapshot(output)


def test_ambiguous_submission_is_not_retried(setup):
    _,output,launch,calls,state=setup;state['ambiguous']=True
    with pytest.raises(ValueError):S.submit_one(output,launch,'prepare',prefix='pipeline')
    with pytest.raises(ValueError):S.submit_one(output,launch,'prepare',prefix='pipeline')
    assert len(calls)==1


def test_production_dispatch_is_gated_paired_and_idempotent(setup,monkeypatch):
    repo,output,launch,calls,_=setup
    plan=P.hashed(dict(models=[dict(arm=a,train_hours=200,evaluate_hours=3) for a in COUNTS]))
    monkeypatch.setattr(P,'load_launch',lambda *a,**k:launch)
    monkeypatch.setattr(P,'production_spec',lambda *a:copy.deepcopy(plan))
    monkeypatch.setattr(P,'check_storage',lambda *a,**k:None)
    result=S.dispatch(output)
    assert len(calls)==5 and result['status']=='submitted'
    assert not any('--dependency=' in v for v in calls[0][0])
    assert not any('--dependency=' in v for v in calls[2][0])
    assert '--dependency=afterok:1001' in calls[1][0]
    assert '--dependency=afterok:1003' in calls[3][0]
    assert '--dependency=afterok:1002:1004' in calls[4][0]
    assert S.dispatch(output)['status']=='already_submitted' and len(calls)==5


def test_failed_admission_or_test_only_submits_nothing(setup,monkeypatch):
    _,output,launch,calls,state=setup
    monkeypatch.setattr(P,'load_launch',lambda *a,**k:launch)
    monkeypatch.setattr(P,'production_spec',lambda *a:(_ for _ in ()).throw(ValueError('failed benchmark')))
    with pytest.raises(ValueError):S.dispatch(output)
    assert not calls
    state['fail']=True
    with pytest.raises(subprocess.CalledProcessError):S.preflight(output,launch,[S.request(output,launch,'benchmark')])
    assert not calls


def test_readonly_plan_and_explicit_auto_authority(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(CLI,'ROOT',tmp_path)
    output=tmp_path/'checkpoints/part_ev_workspace/new'
    CLI.launch(CLI.parse(['--mode','plan','--platform','tigris','--output',str(output)]))
    assert not output.exists() and 'No files or jobs changed' in capsys.readouterr().out
    with pytest.raises(SystemExit):CLI.parse(['--mode','pipeline','--platform','sporc','--output',str(output)])


def test_explicit_resume_checks_old_jobs_and_relinks_only_this_arm(setup,monkeypatch):
    _,output,launch,calls,_=setup
    plan=P.hashed(dict(models=[dict(arm=a,train_hours=200,evaluate_hours=3) for a in COUNTS]))
    monkeypatch.setattr(P,'load_launch',lambda *a,**k:launch)
    monkeypatch.setattr(P,'production_spec',lambda *a:copy.deepcopy(plan))
    monkeypatch.setattr(P,'load_production',lambda *a,**k:copy.deepcopy(plan))
    monkeypatch.setattr(P,'check_storage',lambda *a,**k:None)
    S.dispatch(output)
    monkeypatch.setattr(S,'checkpoint_for_retry',lambda *a:False)
    monkeypatch.setattr(S,'job_states',lambda jobs:{j:'RUNNING' for j in jobs})
    with pytest.raises(ValueError,match='still active'):S.resume(output,'workspace')
    assert len(calls)==5
    monkeypatch.setattr(S,'job_states',lambda jobs:{j:'CANCELLED' for j in jobs})
    result=S.resume(output,'workspace')
    assert len(calls)==8 and result['status']=='recovery_submitted'
    assert '--dependency=afterok:1006' in calls[6][0]
    assert '--dependency=afterok:1007:1002' in calls[7][0]
    assert not any('--dependency=' in x for x in calls[5][0])


def test_accounting_uncertainty_is_not_terminal_permission(monkeypatch):
    monkeypatch.setattr(S,'command',lambda *a,**k:'123|FAILED|\n')
    with pytest.raises(ValueError,match='incomplete'):S.job_states({'123','124'})


def test_dirty_source_names_with_spaces_are_not_ignored(setup):
    repo,output,launch,_,_=setup
    (repo/'source with spaces.py').write_text('uncommitted')
    with pytest.raises(ValueError,match='Commit/push/pull'):
        S.freeze(repo,output.parent/'fresh',launch)


def test_bash_syntax():
    from tests.test_part_fullscale_launcher import BASH,shell_path
    if BASH is None:pytest.skip('Bash unavailable')
    for name in ('run_part_ev_workspace.sh','submit_part_ev_workspace.sh'):
        result=subprocess.run([BASH,'-n',shell_path(CLI.ROOT/'sbatch'/name)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_fake_slurm_pipeline_queues_only_benchmark_and_reviewed_controller(setup,monkeypatch):
    import json
    repo,output,_,calls,_=setup
    fresh=output.parent/'pipeline-fresh';old=S.command
    def command(argv,**kwargs):
        if '--phase' in argv and 'inspect' in argv:return json.dumps({})
        return old(argv,**kwargs)
    monkeypatch.setattr(S,'command',command)
    monkeypatch.setattr(CLI,'ROOT',repo)
    monkeypatch.setattr(CLI.platform,'system',lambda:'Linux')
    monkeypatch.setattr(CLI.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(P,'check_storage',lambda *a,**k:None)
    args=CLI.parse(['--mode','pipeline','--platform','sporc','--output',str(fresh),
        '--python',sys.executable,'--data',str(repo/'dataset'),'--reviewed'])
    CLI.launch(args)
    assert len(calls)==4
    assert '--dependency=afterok:1001' in calls[1][0]
    assert '--dependency=afterany:1002' in calls[2][0]
    assert '--dependency=afterok:1003' in calls[3][0]
    assert not (fresh/'production.json').exists()
    assert P.checked(fresh/'launch.json')['auto_production'] is True
