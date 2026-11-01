"""Offline Slurm fixtures; never contact the cluster or submit real jobs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH,reason='Bash unavailable')


@pytest.fixture
def launcher(tmp_path):
    repo=tmp_path/'repo space';campaign=tmp_path/'old campaign';bin_dir=tmp_path/'bin'
    chain=campaign/'segment_chains/first'
    for directory in (repo/'sbatch',repo/'scripts',repo/'part_ev_ddp_segments',campaign,bin_dir):
        directory.mkdir(parents=True)
    for name in ('submit_part_ev_ddp_segments.sh','run_part_ev_ddp_segments.sh'):
        shutil.copyfile(REPO/'sbatch'/name,repo/'sbatch'/name)
    (repo/'scripts/run_part_ev_ddp_segments.py').write_text('# driver fixture\n')
    for name in ('protocol.py','engine.py'):
        (repo/'part_ev_ddp_segments'/name).write_text('# fixture\n')
    calls=tmp_path/'calls.txt'
    stubs=dict(
        uname='echo "${TEST_ARCH:-x86_64}"\n',
        squeue='[[ "${TEST_QUEUE_FAIL:-}" != 1 ]] || exit 1\nprintf "%s\\n" "${TEST_LIVE:-}"\n',
        flock='[[ "${TEST_LOCK_BUSY:-}" != 1 ]]\n',
        scontrol='echo "PartitionName=tier3 MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        sbatch='''count=0
[[ ! -f "$TEST_CALLS" ]] || count="$(wc -l < "$TEST_CALLS")"
[[ "${TEST_SUBMIT_FAIL_AT:-}" != "$((count+1))" ]] || exit 1
printf '%s|%s|%s|%s\n' "$EVSEG_PHASE" "$EVSEG_INDEX" "$EVSEG_CHAIN" "$*" >> "$TEST_CALLS"
echo "$((101+count))"
''',
        python='''if [[ "$1" == -c ]]; then
  echo 5
elif [[ "$*" == *'--phase prepare'* ]]; then
  [[ "${TEST_PREPARE_FAIL:-}" != 1 ]] || exit 1
  echo '{"evaluate_hours":5}'
elif [[ "$*" == *'--phase gate'* ]]; then
  [[ "${TEST_GATE_FAIL:-}" != 1 ]] || exit 1
else
  printf 'PYTHON %s\n' "$*"
  printf 'RANK=%s WORLD=%s OMP=%s\n' "${RANK:-}" "${WORLD_SIZE:-}" "${OMP_NUM_THREADS:-}"
fi
''')
    for name,body in stubs.items():
        path=bin_dir/name
        path.write_text('#!/usr/bin/env bash\n'+body,newline='\n');path.chmod(0o755)
    setup=subprocess.run([BASH,'-c',
        'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture && '
        'printf "EVDDP_PYTHON=%q\\nEVDDP_ACCOUNT=reu-aisocial\\nEVDDP_PARTITION=tier3\\n" "$3/python" > "$2/config.sh" && '
        '(cd "$2" && sha256sum config.sh > config.sha256)',
        'fixture',shell_path(repo),shell_path(campaign),shell_path(bin_dir)],capture_output=True,text=True,timeout=30)
    assert setup.returncode==0,setup.stderr
    (campaign/'submitted_job_ids.tsv').write_text('preflight\t21977762\ntrain\t21988787\nevaluate\t21988788\nreport\t21988789\n', newline='\n')
    (campaign/'campaign.json').write_text('{"unchanged":true}\n')
    def run(*args,env=None,worker=False):
        environment=dict(os.environ,TEST_CALLS=shell_path(calls))
        environment.update(env or {})
        script=(chain/'source' if worker else repo)/'sbatch'/(
            'run_part_ev_ddp_segments.sh' if worker else 'submit_part_ev_ddp_segments.sh')
        defaults=[] if worker else ['--campaign',shell_path(campaign),'--chain',shell_path(chain)]
        return subprocess.run([BASH,'--noprofile','--norc','-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"','fixture',shell_path(bin_dir),shell_path(script),
            *defaults,*args],env=environment,capture_output=True,text=True,timeout=40)
    return run,repo,campaign,chain,calls


def test_four_segments_then_completion_gate_eval_report_and_frozen_workers(launcher):
    run,repo,campaign,chain,calls=launcher
    old={p:p.read_bytes() for p in campaign.iterdir() if p.is_file()}
    result=run('--mode','submit','--reviewed')
    assert result.returncode==0,result.stderr
    lines=calls.read_text().splitlines()
    assert len(lines)==7
    for index,line in enumerate(lines):
        assert '--partition=tier3' in line and '--account=reu-aisocial' in line
        assert '--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G' in line
        assert '--array' not in line and 'scancel' not in line
        if index:
            assert f'--dependency=afterok:{100+index}' in line
            assert '--kill-on-invalid-dep=yes' in line
        else: assert '--dependency' not in line
    for i in range(4):
        assert lines[i].startswith(f'segment|{i}|')
        assert '--gres=gpu:a100:2' in lines[i] and '--time=72:00:00' in lines[i]
    assert lines[4].startswith('gate|') and '--gres' not in lines[4] and '--time=00:15:00' in lines[4]
    assert lines[5].startswith('evaluate|') and '--gres=gpu:a100:1' in lines[5] and '--time=5:00:00' in lines[5]
    assert lines[6].startswith('report|') and '--gres' not in lines[6]
    assert all(p.read_bytes()==value for p,value in old.items())
    (repo/'part_ev_ddp_segments/engine.py').write_text('# dirty live repo\n')
    env=dict(EVSEG_CHAIN=shell_path(chain),EVSEG_PHASE='segment',EVSEG_INDEX='1',
             SLURM_JOB_START_TIME='1',SLURM_JOB_END_TIME='259201',RANK='99',WORLD_SIZE='99')
    result=run(worker=True,env=env)
    assert result.returncode==0,result.stderr
    assert '--nproc-per-node=2' in result.stdout and '--index 1' in result.stdout
    assert 'RANK= WORLD= OMP=2' in result.stdout
    for phase in ('gate','report','evaluate'):
        result=run(worker=True,env=dict(env,EVSEG_PHASE=phase))
        assert result.returncode==0,result.stderr
        assert ('torch.distributed.run' in result.stdout)==(phase=='evaluate')
    result=run(worker=True,env=dict(env,SLURM_JOB_END_TIME=''))
    assert result.returncode!=0


def test_dry_run_and_review_read_only(launcher):
    run,_,campaign,chain,calls=launcher
    before={p:p.read_bytes() for p in campaign.iterdir()}
    assert run('--mode','submit','--dry-run').returncode==0
    assert run('--mode','review').returncode==0
    assert not chain.exists() and not calls.exists()
    assert before=={p:p.read_bytes() for p in campaign.iterdir()}


@pytest.mark.parametrize('args,env',[
    ([],{}),(['--reviewed'],{'TEST_LIVE':'21988787'}),
    (['--reviewed'],{'TEST_LIVE':'21988788'}),(['--reviewed'],{'TEST_LIVE':'21988789'}),
    (['--reviewed'],{'TEST_LOCK_BUSY':'1'}),(['--reviewed'],{'TEST_QUEUE_FAIL':'1'}),
    (['--reviewed'],{'TEST_MAXTIME':'2-00:00:00'}),(['--reviewed'],{'TEST_PREPARE_FAIL':'1'}),
    (['--reviewed','--segments','0'],{}),(['--reviewed','--segments','9'],{}),
    (['--reviewed'],{'TEST_ARCH':'aarch64'}),
])
def test_invalid_submit_never_queues(launcher,args,env):
    run,_,_,_,calls=launcher
    assert run('--mode','submit',*args,env=env).returncode!=0
    assert not calls.exists()


def test_segment_ledger_guard_and_partial_submission(launcher):
    run,_,campaign,chain,calls=launcher
    result=run('--mode','submit','--reviewed',env={'TEST_SUBMIT_FAIL_AT':'3'})
    assert result.returncode!=0 and 'already queued' in result.stderr
    assert len(calls.read_text().splitlines())==2
    assert (chain/'submitted_job_ids.tsv').read_text().splitlines()==['segment_0\t101','segment_1\t102']
    result=run('--mode','submit','--reviewed','--chain',shell_path(campaign/'segment_chains/next'),env={'TEST_LIVE':'102'})
    assert result.returncode!=0 and 'still active' in result.stderr
    assert not (campaign/'segment_chains/next').exists()


def test_retry_gate_and_existing_source_tamper(launcher):
    run,_,_,chain,calls=launcher
    assert run('--mode','submit','--reviewed').returncode==0
    assert run('--mode','submit','--reviewed').returncode!=0
    assert run('--mode','evaluate',env={'TEST_GATE_FAIL':'1'}).returncode!=0
    assert len(calls.read_text().splitlines())==7
    result=run('--mode','evaluate')
    assert result.returncode==0,result.stderr
    assert len(calls.read_text().splitlines())==9
    (chain/'source/part_ev_ddp_segments/engine.py').write_text('# tamper\n')
    assert run('--mode','report').returncode!=0
    assert len(calls.read_text().splitlines())==9


def test_dirty_uncommitted_adapter_not_frozen(launcher):
    run,repo,_,chain,calls=launcher
    (repo/'part_ev_ddp_segments/engine.py').write_text('# uncommitted\n')
    assert run('--mode','submit','--reviewed').returncode!=0
    assert not calls.exists() and not chain.exists()
