"""Fake Slurm only: no cluster access or real submissions."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH, reason='Bash unavailable')


@pytest.fixture
def launcher(tmp_path):
    repo, donor, bin_dir = (tmp_path/name for name in ('repo space','donor','bin'))
    output, calls = tmp_path/'new diagnostic', tmp_path/'calls.txt'
    for directory in (repo/'sbatch',repo/'scripts',repo/'part_ev_state_tigris',donor,bin_dir):
        directory.mkdir(parents=True)
    for name in ('submit_part_ev_state_tigris_benchmark.sh','run_part_ev_state_tigris_benchmark.sh'):
        shutil.copyfile(REPO/'sbatch'/name,repo/'sbatch'/name)
    for name in ('protocol.py','benchmark.py'):
        (repo/'part_ev_state_tigris'/name).write_text('# fixture\n')
    (repo/'scripts/benchmark_part_ev_state_tigris.py').write_text('# fixture\n')
    (donor/'campaign.json').write_text('{"untouched":true}\n')
    stubs = dict(
        uname='echo "${TEST_ARCH:-aarch64}"\n',
        flock='[[ "${TEST_LOCK_BUSY:-}" != 1 ]]\n',
        squeue='echo "Must not query or change SPORC queue" >&2; exit 1\n',
        scontrol='echo "PartitionName=tigris MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        sbatch='''count=0
[[ ! -f "$TEST_CALLS" ]] || count="$(wc -l < "$TEST_CALLS")"
[[ "${TEST_SUBMIT_FAIL_AT:-}" != "$((count+1))" ]] || exit 1
printf '%s|%s|%s|%s\n' "$EVGH_PHASE" "$EVGH_OUTPUT" "${SBATCH_GPUS:-}" "$*" >> "$TEST_CALLS"
echo "$((101+count))"
''',
        python='''if [[ "$1" == - ]]; then
  cat >/dev/null
  printf '{}\n' > "$2"
else
  printf 'PYTHON %s\n' "$*"
  printf 'RANK=%s WORLD=%s OMP=%s\n' "${RANK:-}" "${WORLD_SIZE:-}" "${OMP_NUM_THREADS:-}"
fi
''')
    for name, body in stubs.items():
        path = bin_dir/name
        path.write_text('#!/usr/bin/env bash\n'+body,newline='\n');path.chmod(0o755)
    result = subprocess.run([BASH,'-c',
        'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test '
        '-c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture',
        'fixture',shell_path(repo)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    def run(*args, env=None, worker=False):
        environment = dict(os.environ,TEST_CALLS=shell_path(calls))
        environment.update(env or {})
        script=(output/'source' if worker else repo)/'sbatch'/(
            'run_part_ev_state_tigris_benchmark.sh' if worker else 'submit_part_ev_state_tigris_benchmark.sh')
        defaults=[] if worker else ['--output',shell_path(output),'--donor',shell_path(donor),
                                   '--python',shell_path(bin_dir/'python')]
        return subprocess.run([BASH,'--noprofile','--norc','-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"','fixture',shell_path(bin_dir),
            shell_path(script),*defaults,*args],env=environment,capture_output=True,text=True,timeout=40)
    return run,repo,donor,output,calls


def test_two_bounded_jobs_no_production_and_frozen_worker(launcher):
    run,repo,donor,output,calls=launcher
    old={p:p.read_bytes() for p in donor.iterdir()}
    result=run(env={'SBATCH_GPUS':'4'})
    assert result.returncode==0,result.stderr
    rows=calls.read_text().splitlines()
    assert len(rows)==2
    for row in rows:
        assert '--partition=tigris --account=reu-aisocial' in row
        assert '--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=02:00:00' in row
        assert '--array' not in row and 'scancel' not in row and '||' in row
        assert '--signal=B:USR1@180' in row
    assert rows[0].startswith('prepare|') and '--gpus' not in rows[0] and '--dependency' not in rows[0]
    assert rows[1].startswith('benchmark|') and '--gpus-per-node=1' in rows[1]
    assert '--dependency=afterok:101 --kill-on-invalid-dep=yes' in rows[1]
    assert (output/'submitted_job_ids.tsv').read_text().splitlines()==['prepare\t101','benchmark\t102']
    assert old=={p:p.read_bytes() for p in donor.iterdir()}
    (repo/'part_ev_state_tigris/benchmark.py').write_text('# dirty live source\n')
    for phase in ('prepare','benchmark'):
        result=run(worker=True,env=dict(EVGH_OUTPUT=shell_path(output),EVGH_PHASE=phase,RANK='99',WORLD_SIZE='2'))
        assert result.returncode==0,result.stderr
        assert f'--phase {phase}' in result.stdout and 'RANK= WORLD= OMP=4' in result.stdout
        assert 'torch.distributed.run' not in result.stdout
    result=run(worker=True,env=dict(EVGH_OUTPUT=shell_path(output),EVGH_PHASE='train'))
    assert result.returncode!=0


def test_dry_run_and_review_never_submit(launcher):
    run,_,_,output,calls=launcher
    assert run('--dry-run').returncode==0
    assert not output.exists() and not calls.exists()
    assert run().returncode==0
    before={p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result=run('--mode','review')
    assert result.returncode==0,result.stderr
    assert '--phase review' in result.stdout
    assert before=={p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert len(calls.read_text().splitlines())==2


@pytest.mark.parametrize('args,env',[
    (['--mode','train'],{}),([],{'TEST_ARCH':'x86_64'}),
    ([],{'TEST_MAXTIME':'01:59:00'}),([],{'TEST_MAXTIME':'unknown'}),
    ([],{'TEST_LOCK_BUSY':'1'}),(['--partition','bad name'],{}),
])
def test_invalid_request_does_not_submit(launcher,args,env):
    run,_,_,_,calls=launcher
    assert run(*args,env=env).returncode!=0
    assert not calls.exists()


def test_dirty_source_and_donor_overlap_rejected(launcher):
    run,repo,donor,output,calls=launcher
    assert run('--output',shell_path(donor/'child')).returncode!=0
    (repo/'part_ev_state_tigris/benchmark.py').write_text('# changed\n')
    assert run().returncode!=0
    assert not calls.exists() and not output.exists() and not (donor/'child').exists()


def test_partial_submission_is_recorded_and_not_blindly_retried(launcher):
    run,_,_,output,calls=launcher
    result=run(env={'TEST_SUBMIT_FAIL_AT':'2'})
    assert result.returncode!=0 and 'already submitted' in result.stderr
    assert (output/'submitted_job_ids.tsv').read_text()=='prepare\t101\n'
    assert run().returncode!=0 and len(calls.read_text().splitlines())==1


@pytest.mark.parametrize('tamper',['launch.env','source/part_ev_state_tigris/benchmark.py'])
def test_tampered_frozen_source_or_config_blocks_review_and_worker(launcher,tamper):
    run,_,_,output,calls=launcher
    assert run().returncode==0
    (output/tamper).write_text('# tampered\n')
    assert run('--mode','review').returncode!=0
    assert run(worker=True,env=dict(EVGH_OUTPUT=shell_path(output),EVGH_PHASE='benchmark')).returncode!=0
    assert len(calls.read_text().splitlines())==2
