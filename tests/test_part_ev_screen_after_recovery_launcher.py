"""Mock Slurm coverage: no remote submissions, deletions or cancellations."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from tests.test_part_fullscale_launcher import BASH, shell_path
from tests.test_part_ev_screen_amp_recovery_launcher import launcher as recovery_launcher

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not BASH, reason='Bash unavailable')


@pytest.fixture
def launcher(recovery_launcher):
    _, repo, campaign, calls = recovery_launcher
    fakebin = repo.parent/'bin'
    diagnostic = campaign/'diagnostics/amp_recovery.fixture'
    diagnostic.mkdir(parents=True)
    (campaign/'diagnostics/amp_recovery_jobs.tsv').write_text(
        f'run\t21904258\t{shell_path(diagnostic)}\nsummary\t21904259\t{shell_path(diagnostic)}\n', newline='\n')
    for name in ('run_part_ev_screen_after_recovery.sh', 'submit_part_ev_screen_after_recovery.sh'):
        shutil.copyfile(REPO/'sbatch'/name, repo/'sbatch'/name)
    shutil.copyfile(REPO/'scripts/run_part_ev_screen_after_recovery.py', repo/'scripts/run_part_ev_screen_after_recovery.py')
    fakebin.joinpath('python').write_text('''#!/usr/bin/env bash
if [[ "$*" == *run_part_ev_screen_after_recovery.py* ]]; then
  phase= approval= recovery=
  for ((i=1;i<=$#;i++)); do
    j=$((i+1))
    case "${!i}" in
      --phase) phase="${!j}" ;; --approval) approval="${!j}" ;; --recovery) recovery="${!j}" ;;
    esac
  done
  case "$phase" in
    approve)
      [[ "${TEST_EVIDENCE_BAD:-}" != 1 ]] || exit 1
      printf '{"campaign":"%s","recovery":"%s"}\\n' "$TEST_CAMPAIGN" "$recovery" > "$approval/approval.json" ;;
    review)
      [[ "${TEST_EVIDENCE_BAD:-}" != 1 ]] || exit 1
      echo '{"recommended_train_hours":65,"recommended_evaluate_hours":3}' ;;
    *) printf 'FROZEN %s\\n' "$*" ;;
  esac
else
  exec "$TEST_PYTHON" "$@"
fi
''', newline='\n')
    initialized = subprocess.run([BASH, '-c',
        'git -C "$1" add . && git -C "$1" -c user.name=Test -c user.email=test@example.invalid '
        '-c commit.gpgsign=false commit -qm approval_fixture', 'init', shell_path(repo)],
        capture_output=True, text=True, timeout=30)
    assert initialized.returncode == 0, initialized.stderr
    def run(*options, env=None, worker=False):
        approval = campaign/'production_approval'
        extra = dict(os.environ, TEST_CALLS=shell_path(calls), TEST_CAMPAIGN=shell_path(campaign),
                     TEST_PYTHON=shell_path(sys.executable))
        extra.update(env or {})
        if worker:
            script = approval/'run_part_ev_screen_after_recovery.sh'
            args = [shell_path(approval), *options]
        else:
            script = repo/'sbatch/submit_part_ev_screen_after_recovery.sh'
            args = ['--campaign', shell_path(campaign), '--recovery-job', '21904258', *options]
        return subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$1:$PATH"; shift; exec bash "$@"', 'test', shell_path(fakebin), shell_path(script), *args],
            env=extra, capture_output=True, text=True, timeout=40)
    return run, repo, campaign, calls


def test_dry_run_is_read_only_and_no_default_review_authority(launcher):
    run, _, campaign, calls = launcher
    result = run('--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--array=0-6%2' in result.stdout
    assert not (campaign/'production_approval').exists() and not calls.exists()
    assert run().returncode != 0
    assert not calls.exists()


def test_seven_fresh_trainings_then_seven_evaluations_and_report(launcher):
    run, _, campaign, calls = launcher
    result = run('--reviewed', '--concurrency', '3')
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 3
    assert '--job-name=evscr_train' in rows[0] and '--time=65:00:00' in rows[0]
    assert '--dependency' not in rows[0]
    assert '--job-name=evscr_evaluate' in rows[1] and '--dependency=afterok:750001' in rows[1]
    assert '--time=3:00:00' in rows[1]
    assert '--job-name=evscr_report' in rows[2] and '--dependency=afterok:750002' in rows[2]
    assert '--time=2:00:00' in rows[2] and '--gres' not in rows[2] and '--array' not in rows[2]
    assert all('--array=0-6%3' in r and '--gres=gpu:a100:1' in r for r in rows[:2])
    assert all('--partition=tier3' in r and '--account=reu-aisocial' in r for r in rows)
    assert rows[0].endswith(' train 0')
    assert 'benchmark\t123456' in (campaign/'submitted_job_ids.tsv').read_text()
    assert 'train\t750001' in (campaign/'submitted_job_ids.tsv').read_text()


@pytest.mark.parametrize('env', [{'TEST_LIVE':'123456_3'}, {'TEST_LIVE':'21904258_0'},
    {'TEST_LIVE':'21904259'}, {'TEST_ARCH':'aarch64'}, {'TEST_QUEUE_FAIL':'1'}, {'TEST_EVIDENCE_BAD':'1'}])
def test_no_submission_on_bad_evidence_or_active_campaign(launcher, env):
    run, _, _, calls = launcher
    assert run('--reviewed', env=env).returncode != 0
    assert not calls.exists()


def test_resume_requires_terminal_jobs_and_is_explicit(launcher):
    run, _, _, calls = launcher
    assert run('--reviewed').returncode == 0
    assert run('--reviewed').returncode != 0
    assert run('--reviewed', '--resume', env={'TEST_LIVE':'750001_[0-6%2]'}).returncode != 0
    assert run('--reviewed', '--resume', env={'TEST_LIVE':'750003'}).returncode != 0
    assert len(calls.read_text().splitlines()) == 3
    result = run('--reviewed', '--resume')
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 6 and rows[3].endswith(' train 1')
    assert '--dependency=afterok:750004' in rows[4]


def test_frozen_worker_survives_live_pull_but_rejects_tampering(launcher):
    run, repo, campaign, _ = launcher
    assert run('--reviewed').returncode == 0
    (repo/'scripts/run_part_ev_screen_after_recovery.py').write_text('# updated later\n')
    result = run('train', '0', worker=True, env={'SLURM_ARRAY_TASK_ID':'4'})
    assert result.returncode == 0, result.stderr
    assert '--phase train' in result.stdout and '--task-index 4' in result.stdout
    assert '--resume' not in result.stdout
    result = run('train', '1', worker=True, env={'SLURM_ARRAY_TASK_ID':'4'})
    assert result.returncode == 0 and '--resume' in result.stdout
    assert run('train', '0', worker=True, env={'SLURM_ARRAY_TASK_ID':'7'}).returncode != 0
    assert run('benchmark', '0', worker=True).returncode != 0
    assert run('report', '0', worker=True).returncode == 0
    (campaign/'production_approval/run_part_ev_screen_after_recovery.py').write_text('# tampered\n')
    assert run('report', '0', worker=True).returncode != 0


def test_review_and_evaluation_only_do_not_requeue_training(launcher):
    run, _, campaign, calls = launcher
    assert run('--reviewed').returncode == 0
    assert run('--mode', 'review', env={'TEST_LIVE':'750001_1'}).returncode == 0
    assert len(calls.read_text().splitlines()) == 3
    result = run('--mode', 'evaluate', '--resume')
    assert result.returncode == 0, result.stderr
    rows = calls.read_text().splitlines()
    assert len(rows) == 5 and '--job-name=evscr_evaluate' in rows[3]
    assert '--dependency' not in rows[3]
    assert '--dependency=afterok:750004' in rows[4]


def test_lower_walltime_dirty_code_and_resource_changes_fail_closed(launcher):
    run, repo, campaign, calls = launcher
    assert run('--reviewed', '--train-hours', '2').returncode != 0
    assert not calls.exists()
    assert run('--reviewed', '--train-hours', '72').returncode == 0
    assert '--time=72:00:00' in calls.read_text().splitlines()[0]
    (campaign/'launch.env').write_text((campaign/'launch.env').read_text().replace('64G', '128G'), newline='\n')
    assert run('--mode', 'review').returncode != 0


def test_dirty_approval_code_prevents_first_launch(launcher):
    run, repo, _, calls = launcher
    (repo/'scripts/run_part_ev_screen_after_recovery.py').write_text('# dirty\n')
    assert run('--reviewed').returncode != 0
    assert not calls.exists()
