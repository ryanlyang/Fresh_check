"""Offline Slurm stubs: no real jobs or cluster connections."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not BASH,reason='Bash unavailable')


@pytest.fixture
def launcher(tmp_path):
    repo,bench,bin_dir=(tmp_path/name for name in ('repo space','benchmark','bin'))
    output=repo/'checkpoints/part_ev_tigris_segments/benchmark_723341'
    for path in (repo/'sbatch',repo/'scripts',repo/'part_ev_tigris_segments',bench,bin_dir,output.parent):
        path.mkdir(parents=True,exist_ok=True)
    for name in ('submit_part_ev_tigris_segments.sh','run_part_ev_tigris_segments.sh'):
        shutil.copyfile(REPO/'sbatch'/name,repo/'sbatch'/name)
    for name in ('protocol.py','engine.py'):
        (repo/'part_ev_tigris_segments'/name).write_text('# fixture\n')
    (repo/'scripts/run_part_ev_tigris_segments.py').write_text('# fixture\n')
    calls=tmp_path/'calls.txt'
    stubs=dict(
        uname='echo "${TEST_ARCH:-aarch64}"\n',
        flock='[[ "${TEST_LOCK_BUSY:-}" != 1 ]]\n',
        squeue='[[ "${TEST_QUEUE_FAIL:-}" != 1 ]] || exit 1; printf "%s\\n" "${TEST_LIVE-723341}"\n',
        sacct='printf "723341|%s|%s|\\n" "${TEST_BENCH_STATE:-COMPLETED}" "${TEST_BENCH_EXIT:-0:0}"\n',
        scontrol='echo "PartitionName=tigris MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        sbatch='''count=0
[[ ! -f "$TEST_CALLS" ]] || count="$(wc -l < "$TEST_CALLS")"
[[ "${TEST_FAIL_AT:-}" != "$((count+1))" ]] || exit 1
printf '%s|%s|%s|%s|%s\n' "$EVTG_PHASE" "$EVTG_INDEX" "$EVTG_CHAIN" "${SBATCH_GPUS:-}" "$*" >> "$TEST_CALLS"
echo "$((101+count))"
''',
        python='''if [[ "$1" == - ]]; then
  cat >/dev/null; printf '{}\n' > "$2"
else
  [[ "${TEST_PREPARE_FAIL:-}" != 1 || "$*" != *'--phase prepare'* ]] || exit 1
  [[ "${TEST_GATE_FAIL:-}" != 1 || "$*" != *'--phase gate'* ]] || exit 1
  printf 'PYTHON %s\nRANK=%s WORLD=%s OMP=%s\n' "$*" "${RANK:-}" "${WORLD_SIZE:-}" "${OMP_NUM_THREADS:-}"
fi
''')
    for name,body in stubs.items():
        path=bin_dir/name
        path.write_text('#!/usr/bin/env bash\n'+body,newline='\n');path.chmod(0o755)
    setup=subprocess.run([BASH,'-c',
        'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test '
        '-c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture && '
        'printf "EVGH_PYTHON=%q\\nEVGH_ACCOUNT=reu-aisocial\\nEVGH_PARTITION=tigris\\n" "$3/python" > "$2/launch.env" && '
        '(cd "$2" && sha256sum launch.env > launch.sha256)',
        'fixture',shell_path(repo),shell_path(bench),shell_path(bin_dir)],capture_output=True,text=True,timeout=30)
    assert setup.returncode==0,setup.stderr
    (bench/'submitted_job_ids.tsv').write_text('prepare\t723340\nbenchmark\t723341\n')
    (bench/'campaign.json').write_text('{"unchanged":true}\n')
    def run(*args,mode='submit',env=None,worker=False,chain=None,auto=False):
        environment=dict(os.environ,TEST_CALLS=shell_path(calls))
        environment.update(env or {})
        script=(chain/'source' if worker else repo)/'sbatch'/(
            'run_part_ev_tigris_segments.sh' if worker else 'submit_part_ev_tigris_segments.sh')
        defaults=[] if worker else ['--mode',mode]
        if not worker:
            if mode=='submit':
                defaults+=['--benchmark-job','723341']
                if not auto: defaults+=['--benchmark',shell_path(bench)]
            elif mode in ('continue','review'): defaults+=['--campaign',shell_path(output)]
            else: defaults+=['--chain',shell_path(chain)]
        return subprocess.run([BASH,'--noprofile','--norc','-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"','fixture',shell_path(bin_dir),shell_path(script),
            *defaults,*args],env=environment,capture_output=True,text=True,timeout=45)
    return run,repo,bench,output,calls


def test_pending_benchmark_to_admission_four_slots_gate_eval_report(launcher):
    run,repo,bench,output,calls=launcher
    original={p:p.read_bytes() for p in bench.iterdir()}
    result=run('--reviewed',env={'SBATCH_GPUS':'4'})
    assert result.returncode==0,result.stderr
    rows=calls.read_text().splitlines()
    assert len(rows)==8
    assert rows[0].startswith('admit|') and '--dependency=afterok:723341' in rows[0]
    for index,row in enumerate(rows):
        assert '--partition=tigris --account=reu-aisocial' in row
        assert '--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G' in row
        assert '||' in row and '--array' not in row and 'scancel' not in row
        assert '--kill-on-invalid-dep=yes' in row
        if index: assert f'--dependency=afterok:{100+index}' in row
    for i in range(1,5):
        assert rows[i].startswith(f'segment|{i-1}|')
        assert '--time=72:00:00' in rows[i] and '--gpus-per-node=1' in rows[i]
    for i,time in ((0,'00:30:00'),(5,'00:15:00'),(7,'04:30:00')):
        assert f'--time={time}' in rows[i] and '--gpus' not in rows[i]
    assert '--time=05:00:00' in rows[6] and '--gpus-per-node=1' in rows[6]
    assert original=={p:p.read_bytes() for p in bench.iterdir()}
    chain=next((output/'segment_chains').iterdir())
    (repo/'part_ev_tigris_segments/engine.py').write_text('# unrelated later pull\n')
    env=dict(EVTG_CHAIN=shell_path(chain),EVTG_PHASE='segment',EVTG_INDEX='0',
             SLURM_JOB_START_TIME='1',SLURM_JOB_END_TIME='259201',RANK='9',WORLD_SIZE='2')
    result=run(worker=True,chain=chain,env=env)
    assert result.returncode==0,result.stderr
    assert 'RANK= WORLD= OMP=4' in result.stdout and '--index 0' in result.stdout
    assert 'torch.distributed.run' not in result.stdout
    for phase in ('admit','gate','evaluate','report'):
        assert run(worker=True,chain=chain,env=dict(env,EVTG_PHASE=phase)).returncode==0
    assert run(worker=True,chain=chain,env=dict(env,SLURM_JOB_END_TIME='')).returncode!=0


def test_completed_retired_benchmark_needs_no_old_job_dependency(launcher):
    run,_,_,_,calls=launcher
    result=run('--reviewed',env={'TEST_LIVE':''})
    assert result.returncode==0,result.stderr
    assert '--dependency' not in calls.read_text().splitlines()[0]


@pytest.mark.parametrize('args,env',[
    ([],{}),(['--reviewed'],{'TEST_ARCH':'x86_64'}),(['--reviewed'],{'TEST_LOCK_BUSY':'1'}),
    (['--reviewed'],{'TEST_QUEUE_FAIL':'1'}),(['--reviewed'],{'TEST_MAXTIME':'2-00:00:00'}),
    (['--reviewed'],{'TEST_LIVE':'','TEST_BENCH_STATE':'FAILED','TEST_BENCH_EXIT':'1:0'}),
    (['--reviewed'],{'TEST_PREPARE_FAIL':'1'}),(['--reviewed','--segments','0'],{}),
    (['--reviewed','--segments','9'],{}),(['--reviewed','--benchmark-job','999'],{}),
])
def test_invalid_admission_never_submits(launcher,args,env):
    run,_,_,_,calls=launcher
    assert run(*args,env=env).returncode!=0
    assert not calls.exists()


def test_dry_run_readonly_review_and_no_active_duplicate(launcher):
    run,_,_,output,calls=launcher
    assert run('--dry-run').returncode==0
    assert not output.exists() and not calls.exists()
    assert run('--reviewed').returncode==0
    before={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert run(mode='review').returncode==0
    assert before=={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert run('--reviewed').returncode!=0
    assert run('--reviewed',mode='continue',env={'TEST_LIVE':'104'}).returncode!=0
    assert len(calls.read_text().splitlines())==8


def test_continuation_and_partial_failure_preserve_ledger(launcher):
    run,_,_,output,calls=launcher
    result=run('--reviewed',env={'TEST_FAIL_AT':'4'})
    assert result.returncode!=0 and 'already queued' in result.stderr
    chain=next((output/'segment_chains').iterdir())
    assert (chain/'submitted_job_ids.tsv').read_text().splitlines()==['admit\t101','segment_0\t102','segment_1\t103']
    assert run('--reviewed',mode='continue',env={'TEST_LIVE':'102'}).returncode!=0
    result=run('--reviewed','--segments','2',mode='continue',env={'TEST_LIVE':''})
    assert result.returncode==0,result.stderr
    assert len(list((output/'segment_chains').iterdir()))==2
    assert len(calls.read_text().splitlines())==9


def test_downstream_retries_require_gate_and_frozen_source(launcher):
    run,_,_,output,calls=launcher
    assert run('--reviewed').returncode==0
    chain=next((output/'segment_chains').iterdir())
    assert run(mode='evaluate',chain=chain,env={'TEST_LIVE':'','TEST_GATE_FAIL':'1'}).returncode!=0
    result=run(mode='evaluate',chain=chain,env={'TEST_LIVE':''})
    assert result.returncode==0,result.stderr
    assert len(calls.read_text().splitlines())==10
    (chain/'source/part_ev_tigris_segments/engine.py').write_text('# tamper\n')
    assert run(mode='report',chain=chain,env={'TEST_LIVE':''}).returncode!=0
    assert len(calls.read_text().splitlines())==10


def test_dirty_relevant_code_not_frozen(launcher):
    run,repo,_,_,calls=launcher
    (repo/'part_ev_tigris_segments/protocol.py').write_text('# changed\n')
    assert run('--reviewed').returncode!=0
    assert not calls.exists()


def test_benchmark_directory_is_found_by_ledger_not_pending_stdout(launcher):
    run,repo,bench,_,calls=launcher
    assert run('--reviewed',auto=True).returncode!=0
    target=repo/'checkpoints/part_ev_state_tigris/gh200_original'
    shutil.copytree(bench,target)
    assert not (target/'logs').exists()
    assert run('--reviewed',auto=True).returncode==0
    assert len(calls.read_text().splitlines())==8


def test_ambiguous_ledger_does_not_guess(launcher):
    run,repo,bench,_,calls=launcher
    for name in ('one','two'):
        shutil.copytree(bench,repo/'checkpoints/part_ev_state_tigris'/name)
    result=run('--reviewed',auto=True)
    assert result.returncode!=0 and 'exactly one' in result.stderr
    assert not calls.exists()
