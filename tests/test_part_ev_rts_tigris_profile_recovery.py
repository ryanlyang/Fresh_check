"""Saved GH200 evidence can be revalidated, never relabeled or bypassed."""
import copy
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from part_ev_rts_tigris import protocol as P, submission as Q
from scripts import submit_part_ev_rts_tigris as S, run_part_ev_rts_tigris as W
from tests.test_part_ev_rts_tigris_launcher import setup
from tests.test_part_ev_rts_tigris_evidence import evidence
from tests.test_part_fullscale_launcher import BASH


@pytest.fixture
def prepared(setup, monkeypatch):
    repo, donor, original, args, state = setup
    science = importlib.import_module('part_ev_triangles.protocol')
    parent = dict(task=science.TASK, parameters=P.COUNTS['triangles'],
        recipe=science.recipes()['ranger_full_v1'], model=science.CONFIG, inventory={'dataset': 'sha'})
    identity = dict(path=str(donor), content_hash='donor', evidence={'campaign.json': 'same'})
    env = dict(architecture='aarch64', torch_cuda='12.6', weaver_sources={'model': 'same'})
    admission = dict(donor=identity, environment=env, task=parent['task'], parameters=parent['parameters'])
    monkeypatch.setattr(P, 'donor_contract', lambda *_: (science, parent, identity))
    monkeypatch.setattr(P, 'runtime_environment', lambda *_: env)
    args[1] = 'triangles'
    original.parent.mkdir(parents=True)
    S.freeze(S.parse(['--mode', 'plan', *args]), admission)
    spec = P.prepare(original)
    row = evidence('triangles')
    row['campaign_sha256'] = spec['content_hash']
    for phase in ('real_train', 'adversarial_train', 'real_validation'):
        row['profiles'][0][phase]['stages']['triangles.factor_projections']['cpu_us'] = 0.
    P.write(original/'benchmark.json', P.hashed(row))
    P.write(original/'benchmark_error.json', dict(error='ValueError: Missing triangle stage profiling'))
    P.write(original/'summary_error.json', dict(error='ValueError: Missing triangle stage profiling'))
    (original/'pipeline_job_ids.tsv').write_text('prepare\t901\nbenchmark\t902\nsummary\t903\ndispatch\t904\n')
    new = original.parent/'profile_recovered'
    retry = ['--from-campaign', str(original), '--output', str(new), '--python', sys.executable]
    state['active'] = ''
    previous = S.command
    def command(argv, **kw):
        if 'inspect_profile_recovery' in argv:
            old, origin, summary = P.inspect_profile_origin(original)
            return json.dumps(dict(admission=old['launch']['admission'], origin=origin, summary=summary))
        if argv[0] == 'squeue': return state['active']
        return previous(argv, **kw)
    monkeypatch.setattr(S, 'command', command)
    monkeypatch.setattr(Q, 'command', command)
    which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda name: '/fake/squeue' if name == 'squeue' else which(name))
    return SimpleNamespace(repo=repo, donor=donor, old=original, new=new, args=retry,
                           state=state, environment=env)


def queue(p):
    S.launch(S.parse(['--mode', 'recover-profile', '--reviewed', *p.args]))


def rewrite(path, mutate):
    row = P.read(path)
    row.pop('content_hash', None)
    mutate(row)
    P.write(path, P.hashed(row))


def test_plan_read_only_and_submit_cpu_only(prepared, capsys):
    p = prepared
    before = P.inventory(p.repo)
    S.launch(S.parse(['--mode', 'plan-profile-recovery', *p.args]))
    assert P.inventory(p.repo) == before and not p.state['jobs'] and not p.state['checks']
    assert 'CPU-only 30-minute' in capsys.readouterr().out
    queue(p)
    assert len(p.state['jobs']) == len(p.state['checks']) == 1
    request, env = p.state['jobs'][0]
    assert '--time=30' in request and '--mem=16G' in request and '--cpus-per-task=4' in request
    assert not any(v.startswith(('--gres', '--dependency')) for v in request)
    assert request[-1] == 'recover_profile' and 'WORLD_SIZE' not in env
    assert Q.ledger(p.new, 'recovery') == {'recover_profile': '1000'}
    assert not Q.ledger(p.new, 'pipeline') and not Q.ledger(p.new, 'production')
    assert not (p.new/'campaign.json').exists()  # Worker, not submission, creates admission.
    assert P.launch_config(p.new)['max_train_hours'] == P.launch_config(p.old)['max_train_hours']


def test_saved_measurements_survive_recovery_byte_for_byte(prepared):
    p = prepared
    before, donor_before = P.inventory(p.old), P.inventory(p.donor)
    queue(p)
    assert W.main(['--phase', 'recover_profile', '--output', str(p.new)]) == 0
    summary = P.summary(p.new)
    assert summary == P.read(p.new/'summary.json') and summary['passed']
    assert summary['authorizes_training'] is False
    assert P.inventory(p.old) == before and P.inventory(p.donor) == donor_before
    assert (p.old/'benchmark.json').read_bytes() == (p.new/'benchmark.json').read_bytes()
    old, new = P.load(p.old), P.load(p.new)
    assert new['content_hash'] != old['content_hash']
    assert P.checked(p.new/'benchmark.json')['campaign_sha256'] == old['content_hash']
    assert summary['profile_recovery']['artifacts']['benchmark.json'] == P.sha(p.old/'benchmark.json')
    assert not (p.new/'production.json').exists() and len(p.state['jobs']) == 1
    recovered = P.inventory(p.new)
    with pytest.raises(FileExistsError): P.recover_profile(p.new)
    assert P.inventory(p.new) == recovered
    assert W.main(['--phase', 'status', '--output', str(p.new)]) == 0
    assert P.inventory(p.new) == recovered
    for phase in ('prepare', 'benchmark'):
        with pytest.raises(ValueError, match='may not be replaced'):
            W.main(['--phase', phase, '--output', str(p.new)])
    assert P.inventory(p.new) == recovered


def test_recovered_dispatch_recomputes_evidence_and_is_idempotent(prepared):
    p = prepared
    queue(p); P.recover_profile(p.new)
    before = P.inventory(p.old)
    plan = P.production_spec(p.new)
    assert plan['profile_recovery'] == P.checked(p.new/'launch.json')['profile_recovery']
    assert plan['recipes'][plan['task']['recipe']]['total_steps'] == 1000000
    assert plan['policy']['validation_checks'] == 50
    assert Q.dispatch(p.new)['status'] == 'submitted'
    assert len(p.state['jobs']) == 4  # One CPU recovery, then train/evaluate/report.
    assert '--gres=gpu:gh200:1' in p.state['jobs'][1][0]
    assert Q.dispatch(p.new)['status'] == 'already_submitted'
    assert P.load_production(p.new) == plan
    assert len(p.state['jobs']) == 4 and P.inventory(p.old) == before
    # The old output cannot subsequently dispatch a duplicate of the same model.
    P.write(p.old/'summary.json', P.summary(p.old))
    with pytest.raises(ValueError, match='already owns'): Q.dispatch(p.old)
    assert len(p.state['jobs']) == 4


@pytest.mark.parametrize('fault', ('failed', 'restart', 'wrong_campaign', 'missing_stage', 'no_device_only'))
def test_other_benchmark_failures_cannot_be_recovered(prepared, fault):
    p = prepared
    def mutate(row):
        if fault == 'failed': row['passed'] = False
        if fault == 'restart': row['restart']['passed'] = False
        if fault == 'wrong_campaign': row['campaign_sha256'] = 'other'
        if fault == 'missing_stage': del row['profiles'][0]['real_train']['stages']['triangles.ordinary_mix']
        if fault == 'no_device_only':
            for phase in ('real_train', 'adversarial_train', 'real_validation'):
                for stage in row['profiles'][0][phase]['stages'].values(): stage['cpu_us'] = 1.
    rewrite(p.old/'benchmark.json', mutate)
    with pytest.raises(ValueError): queue(p)
    assert not p.state['jobs'] and not p.new.exists()


@pytest.mark.parametrize('fault', ('active', 'owner', 'production', 'storage', 'partition', 'sbatch', 'engine', 'optin'))
def test_recovery_submission_safety(prepared, monkeypatch, fault):
    p = prepared
    if fault == 'active': p.state['active'] = '904'
    if fault == 'owner':
        P.write(p.old.parent/(P.digest(dict(model='triangles', donor='donor'))+'.production-owner.json'),
                dict(output='existing'))
    if fault == 'production': P.write(p.old/'production_intent.json', dict(status='unknown'))
    if fault == 'storage': monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    if fault == 'partition': p.state['partition'] = 'State=UP MaxTime=00:15:00'
    if fault == 'sbatch': p.state['test_fail'] = True
    if fault == 'engine':
        path = p.repo/'part_ev_rts_tigris/production.py'
        path.write_text('# changed production engine')
        subprocess.run(['git','-C',str(p.repo), 'add', '.'], check=True, capture_output=True)
        subprocess.run(['git','-C',str(p.repo), '-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-m','change engine'], check=True, capture_output=True)
    with pytest.raises((ValueError, OSError, subprocess.CalledProcessError, SystemExit)):
        if fault == 'optin': S.parse(['--mode','recover-profile', *p.args])
        else: queue(p)
    assert not p.state['jobs'] and not p.new.exists()


def test_ambiguous_recovery_submission_retains_intent(prepared):
    p = prepared
    p.state['fail'] = 0
    with pytest.raises(subprocess.CalledProcessError): queue(p)
    assert P.read(p.new/'recovery_intent.json')['status'] == 'unknown'
    before = P.inventory(p.new)
    with pytest.raises(FileExistsError): queue(p)
    assert P.inventory(p.new) == before and not p.state['jobs']


@pytest.mark.parametrize('fault', ('origin_payload', 'origin_source', 'copied_payload', 'relabeled', 'runtime'))
def test_recovery_rejects_post_queue_or_post_recovery_changes(prepared, fault):
    p = prepared
    queue(p)
    if fault == 'origin_payload':
        rewrite(p.old/'benchmark.json', lambda r: r.update(checkpoint_seconds=2.))
    elif fault == 'origin_source':
        (p.old/'source/part_ev_rts_tigris/evidence.py').write_text('# changed')
    elif fault == 'runtime':
        p.environment['torch_cuda'] = 'different'
    else:
        P.recover_profile(p.new)
        if fault == 'relabeled':
            rewrite(p.new/'benchmark.json', lambda r: r.update(campaign_sha256=P.load(p.new)['content_hash']))
        else:
            rewrite(p.new/'benchmark.json', lambda r: r.update(checkpoint_seconds=2.))
    with pytest.raises(ValueError):
        if fault in ('copied_payload', 'relabeled'): P.production_spec(p.new)
        else: P.recover_profile(p.new)
    assert not (p.new/'production.json').exists() and len(p.state['jobs']) == 1


@pytest.mark.parametrize('field', ('max_train_hours', 'account', 'model', 'environment', 'engine'))
def test_recovery_identity_cannot_change_science_or_authority(prepared, field):
    p = prepared
    queue(p); P.recover_profile(p.new)
    spec = copy.deepcopy(P.load(p.new))
    if field in ('max_train_hours', 'account'):
        spec['launch'][field] = 'changed'
    elif field == 'engine':
        spec['launch']['source']['part_ev_rts_tigris/production.py'] = 'changed'
    else: spec[field] = {'changed': True}
    with pytest.raises(ValueError): P.verify_profile_recovery(p.new, spec)


def test_no_recursive_recovery_or_bypassed_production_storage(prepared, monkeypatch):
    p = prepared
    queue(p); P.recover_profile(p.new)
    with pytest.raises(ValueError, match='original'): P.inspect_profile_origin(p.new)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='headroom'): Q.dispatch(p.new)
    assert len(p.state['jobs']) == 1 and not (p.new/'production.json').exists()


def test_recovery_documented_commands_are_valid_bash():
    repo = Path(__file__).resolve().parents[1]
    text = (repo/'part_ev_rts_tigris/PROFILE_RECOVERY.md').read_text(encoding='utf-8')
    blocks = re.findall('```bash\n(.*?)```', text, re.S)
    assert len(blocks) == 3
    for block in blocks:
        subprocess.run([BASH, '-n'], input=block, text=True, check=True, capture_output=True)
    assert '--mode recover-profile' in blocks[0] and '--reviewed' in blocks[0]
    assert '--mode dispatch' in blocks[2] and '--reviewed' in blocks[2]
    assert 'triangles_20261120T210558Z' in blocks[0]
