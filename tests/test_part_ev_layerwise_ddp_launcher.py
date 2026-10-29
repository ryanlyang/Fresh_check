"""Exercise frozen launchers against fake Slurm; never contact an RC cluster."""
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
    source, output, scaling, parent, bin_dir = (tmp_path/n for n in
        ('repo with space', 'new production', 'old scaling', 'old parent', 'bin'))
    for path in (source/'sbatch', source/'scripts', source/'part_ev_layerwise_ddp', scaling, parent, bin_dir):
        path.mkdir(parents=True)
    (parent/'campaign.json').write_text('{"untouched":true}\n')
    (scaling/'gpus_2.json').write_text('{"untouched":true}\n')
    (scaling/'plan.json').write_text('{}\n')
    for name in ('submit_part_ev_layerwise_ddp.sh', 'run_part_ev_layerwise_ddp.sh'):
        shutil.copyfile(REPO/'sbatch'/name, source/'sbatch'/name)
    (source/'scripts/run_part_ev_layerwise_ddp.py').write_text('# driver fixture\n')
    for name in ('protocol.py', 'engine.py'):
        (source/'part_ev_layerwise_ddp'/name).write_text('# fixture\n')
    calls = tmp_path/'calls.txt'
    stubs = dict(
        uname='echo "${TEST_ARCH:-x86_64}"\n',
        squeue='printf "%s\\n" "${TEST_LIVE:-}"\n',
        flock='[[ "${TEST_LOCK_BUSY:-}" != 1 ]]\n',
        scontrol='echo "PartitionName=tier3 MaxTime=${TEST_MAXTIME:-14-00:00:00}"\n',
        sbatch='printf "%s|%s|%s\\n" "$EVDDP_PHASE" "$EVDDP_OUTPUT" "$*" >> "$TEST_CALLS"\necho "$((100+$(wc -l < "$TEST_CALLS")))"\n',
        python='''if [[ "$1" == -c ]]; then
  case "$2" in
    *parent*) echo "$TEST_PARENT" ;;
    *evaluate_hours*) echo 5 ;;
    *train_hours*) echo 190 ;;
    *) exit 9 ;;
  esac
elif [[ "$*" == *'--phase review'* ]]; then
  [[ "${TEST_REVIEW_FAIL:-}" != 1 ]] || exit 1
  echo '{"train_hours":190,"evaluate_hours":5}'
else
  printf 'PYTHON %s\\n' "$*"
  printf 'RANK=%s WORLD=%s OMP=%s\\n' "${RANK:-}" "${WORLD_SIZE:-}" "${OMP_NUM_THREADS:-}"
  [[ "$*" != *'--phase prepare'* || "${TEST_PREPARE_FAIL:-}" != 1 ]] || exit 1
fi
''')
    for name, body in stubs.items():
        path = bin_dir/name
        path.write_text('#!/usr/bin/env bash\n'+body, newline='\n')
        path.chmod(0o755)
    result = subprocess.run([BASH, '-c',
        'git init -q "$1" && git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid -c commit.gpgsign=false commit -qm fixture',
        'init', shell_path(source)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

    def run(*args, env=None, worker=False):
        environment = dict(os.environ, TEST_CALLS=shell_path(calls), TEST_PARENT=shell_path(parent))
        environment.update(env or {})
        script = (output/'source' if worker else source)/'sbatch'/(
            'run_part_ev_layerwise_ddp.sh' if worker else 'submit_part_ev_layerwise_ddp.sh')
        defaults = [] if worker else ['--output', shell_path(output), '--scaling', shell_path(scaling),
                                     '--python', shell_path(bin_dir/'python')]
        return subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"', 'test', shell_path(bin_dir), shell_path(script),
            *defaults, *args], env=environment, capture_output=True, text=True, timeout=40)
    return run, source, output, scaling, parent, calls


def test_prepare_review_train_single_job_chain_and_frozen_worker(launcher):
    run, source, output, scaling, parent, calls = launcher
    originals = {p: p.read_bytes() for directory in (parent, scaling) for p in directory.iterdir()}
    result = run('--mode', 'prepare')
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith('preflight|') and '--gres=gpu:a100:2' in lines[0]
    assert '--time=04:30:00' in lines[0]
    assert run('--mode', 'review').returncode == 0
    assert len(calls.read_text().splitlines()) == 1
    result = run('--mode', 'train', '--reviewed')
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 4
    for line in lines:
        assert '--nodes=1' in line and '--ntasks=1' in line
        assert '--cpus-per-task=8' in line and '--mem=64G' in line
        assert '--array' not in line and 'scancel' not in line
    assert lines[1].startswith('train|') and '--gres=gpu:a100:2' in lines[1]
    assert '--time=190:00:00' in lines[1] and '--dependency' not in lines[1]
    assert lines[2].startswith('evaluate|') and '--gres=gpu:a100:1' in lines[2]
    assert '--time=5:00:00' in lines[2] and '--dependency=afterok:102' in lines[2]
    assert lines[3].startswith('report|') and '--gres' not in lines[3]
    assert '--dependency=afterok:103' in lines[3]
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert len(list(parent.iterdir())) == 1 and len(list(scaling.iterdir())) == 2
    (source/'part_ev_layerwise_ddp/engine.py').write_text('# changed live repo\n')
    env = dict(EVDDP_OUTPUT=shell_path(output), EVDDP_PHASE='train', RANK='99', WORLD_SIZE='77')
    result = run(worker=True, env=env)
    assert result.returncode == 0, result.stderr
    assert '--nproc-per-node=2' in result.stdout and '--standalone --nnodes=1' in result.stdout
    assert 'RANK= WORLD= OMP=2' in result.stdout
    assert (output/'source/part_ev_layerwise_ddp/engine.py').read_text() == '# fixture\n'
    result = run(worker=True, env=dict(env, EVDDP_PHASE='evaluate'))
    assert result.returncode == 0 and '--nproc-per-node=1' in result.stdout
    result = run(worker=True, env=dict(env, EVDDP_PHASE='report'))
    assert result.returncode == 0 and 'torch.distributed.run' not in result.stdout


def test_dry_run_does_not_write_or_submit(launcher):
    run, _, output, _, _, calls = launcher
    result = run('--mode', 'prepare', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--gres=gpu:a100:2' in result.stdout and '04:30:00' in result.stdout
    assert not output.exists() and not calls.exists()
    assert run('--help').returncode == 0


@pytest.mark.parametrize('args,env', [
    ([], {'TEST_ARCH':'aarch64'}), ([], {'TEST_MAXTIME':'04:00:00'}),
    ([], {'TEST_PREPARE_FAIL':'1'}), (['--partition','bad name'], {}), (['--resume'], {}),
])
def test_invalid_prepare_never_submits(launcher, args, env):
    run, _, _, _, _, calls = launcher
    assert run('--mode', 'prepare', *args, env=env).returncode != 0
    assert not calls.exists()


@pytest.mark.parametrize('args,env', [
    ([], {}), (['--reviewed','--train-hours','189'], {}),
    (['--reviewed','--evaluate-hours','4'], {}), (['--reviewed'], {'TEST_MAXTIME':'7-00:00:00'}),
    (['--reviewed'], {'TEST_REVIEW_FAIL':'1'}), (['--reviewed'], {'TEST_LIVE':'101'}),
    (['--reviewed','--partition','sporc'], {}),
    (['--reviewed'], {'TEST_LOCK_BUSY':'1'}),
])
def test_train_admission_failures_never_queue_chain(launcher, args, env):
    run, _, _, _, _, calls = launcher
    assert run('--mode', 'prepare').returncode == 0
    assert run('--mode', 'train', *args, env=env).returncode != 0
    assert len(calls.read_text().splitlines()) == 1


def test_duplicates_and_active_jobs_guarded_even_with_resume(launcher):
    run, _, _, _, _, calls = launcher
    assert run('--mode', 'prepare').returncode == 0
    assert run('--mode', 'train', '--reviewed').returncode == 0
    assert run('--mode', 'train', '--reviewed').returncode != 0
    assert run('--mode', 'train', '--reviewed', '--resume', env={'TEST_LIVE':'103'}).returncode != 0
    assert len(calls.read_text().splitlines()) == 4
    assert run('--mode', 'train', '--reviewed', '--resume').returncode == 0
    assert len(calls.read_text().splitlines()) == 7


def test_old_directories_snapshot_tampering_and_dirty_source_refused(launcher):
    run, source, output, scaling, parent, calls = launcher
    for directory in (parent, scaling):
        assert run('--mode', 'prepare', '--output', shell_path(directory/'nested')).returncode != 0
        assert not (directory/'nested').exists()
    assert not calls.exists()
    path = source/'part_ev_layerwise_ddp/engine.py'
    path.write_text('# uncommitted\n')
    assert run('--mode', 'prepare').returncode != 0
    assert not output.exists() and not calls.exists()
    path.write_text('# fixture\n')
    assert run('--mode', 'prepare').returncode == 0
    assert run('--mode', 'prepare').returncode != 0
    (output/'source/part_ev_layerwise_ddp/engine.py').write_text('# tampered\n')
    assert run('--mode', 'review').returncode != 0
    assert run(worker=True, env=dict(EVDDP_OUTPUT=shell_path(output), EVDDP_PHASE='train')).returncode != 0
