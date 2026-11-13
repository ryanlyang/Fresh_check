"""Fake Slurm only: no network or cluster writes."""
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
    repo,output,reference,bin_dir=[tmp_path/n for n in ('repo space','new campaign','old reference','bin')]
    for p in (repo/'sbatch',repo/'scripts',repo/'part_ev_pair_memory',reference,bin_dir): p.mkdir(parents=True)
    for name in ('submit_part_ev_pair_memory.sh','run_part_ev_pair_memory.sh'):
        shutil.copyfile(REPO/'sbatch'/name,repo/'sbatch'/name)
    for name in ('model.py','benchmark.py','protocol.py','oracle.py'): (repo/'part_ev_pair_memory'/name).write_text('# fixture')
    (repo/'scripts/run_part_ev_pair_memory.py').write_text('# fixture')
    (reference/'campaign.json').write_text('{"untouched":true}')
    calls=tmp_path/'calls.txt'
    stubs=dict(uname='echo x86_64\n',flock='exit 0\n',
      scontrol='echo "PartitionName=tier3 MaxTime=${TEST_MAXTIME:-14-00:00:00} State=${TEST_STATE:-UP}"\n',
      sbatch='printf "%s|%s|%s\\n" "$PAIRMEM_PHASE" "$PAIRMEM_OUTPUT" "$*" >> "$TEST_CALLS"\necho "$((100+$(wc -l < "$TEST_CALLS")))"\n',
      python='''[[ "$1" != -B ]] || shift
if [[ "$1" == - ]]; then
  printf '{}\\n' > "$2/source_manifest.json"
  printf '{}\\n' > "$2/launch.json"
elif [[ "$1" == -c ]]; then
  exit 0
else
  printf 'PYTHON %s\\n' "$*"
  printf 'WORLD=%s OMP=%s\\n' "${WORLD_SIZE:-}" "${OMP_NUM_THREADS:-}"
fi
''')
    for name,body in stubs.items():
        path=bin_dir/name;path.write_text('#!/usr/bin/env bash\n'+body,newline='\n');path.chmod(0o755)
    result=subprocess.run([BASH,'-c','git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture','init',shell_path(repo)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    def run(*args,env=None,worker=False):
        environ=dict(os.environ,TEST_CALLS=shell_path(calls))
        environ.update(env or {})
        script=(output/'source' if worker else repo)/'sbatch'/('run_part_ev_pair_memory.sh' if worker else 'submit_part_ev_pair_memory.sh')
        options=[] if worker else ['--output',shell_path(output),'--reference',shell_path(reference),'--python',shell_path(bin_dir/'python')]
        return subprocess.run([BASH,'--noprofile','--norc','-c','export PATH="$1:$PATH"; shift; exec bash "$@"','test',shell_path(bin_dir),shell_path(script),*options,*args],env=environ,capture_output=True,text=True,timeout=35)
    return run,repo,output,reference,calls


def test_chain_resources_frozen_source_and_worker(launcher):
    run,repo,output,reference,calls=launcher
    old=(reference/'campaign.json').read_bytes()
    result=run('--mode','prepare')
    assert result.returncode==0,result.stderr
    lines=calls.read_text().splitlines()
    assert len(lines)==3
    assert '--time=05:00:00' in lines[0] and '--gres' not in lines[0]
    assert '--time=05:30:00' in lines[1] and '--gres=gpu:a100:2' in lines[1]
    assert '--dependency=afterok:101' in lines[1]
    assert '--dependency=afterany:102' in lines[2] and '--time=00:15:00' in lines[2]
    assert '--cpus-per-task=8' in lines[1] and '--mem=64G' in lines[1]
    assert (reference/'campaign.json').read_bytes()==old
    frozen=(output/'source/scripts/run_part_ev_pair_memory.py').read_bytes()
    (repo/'scripts/run_part_ev_pair_memory.py').write_text('changed checkout')
    assert (output/'source/scripts/run_part_ev_pair_memory.py').read_bytes()==frozen
    review=run('--mode','review');assert review.returncode==0,review.stderr
    assert len(calls.read_text().splitlines())==3
    result=run(worker=True,env=dict(PAIRMEM_OUTPUT=shell_path(output),PAIRMEM_PHASE='benchmark',WORLD_SIZE='99',RANK='99'))
    assert result.returncode==0,result.stderr
    assert '--nproc-per-node=2' in result.stdout and 'WORLD= OMP=2' in result.stdout
    assert '--max-restarts=0' in result.stdout


def test_dry_run_no_files_no_jobs_and_reject_training(launcher):
    run,repo,output,reference,calls=launcher
    assert run('--mode','prepare','--dry-run').returncode==0
    assert not output.exists() and not calls.exists()
    assert run('--mode','train').returncode!=0
    assert not calls.exists()


@pytest.mark.parametrize('fault',['existing','overlap','dirty','down','limit'])
def test_submission_guards(launcher,fault):
    run,repo,output,reference,calls=launcher
    args=[];env={}
    if fault=='existing':output.mkdir()
    if fault=='overlap':args=['--output',shell_path(reference/'nested')]
    if fault=='dirty':(repo/'part_ev_pair_memory/model.py').write_text('uncommitted')
    if fault=='down':env={'TEST_STATE':'DOWN'}
    if fault=='limit':env={'TEST_MAXTIME':'05:00:00'}
    result=run('--mode','prepare',*args,env=env)
    assert result.returncode!=0,result.stdout
    assert not calls.exists()
