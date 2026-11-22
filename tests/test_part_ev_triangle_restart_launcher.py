import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from part_ev_triangle_restart import protocol as P
from scripts import submit_part_ev_triangles_restart as S
from tests.test_part_fullscale_launcher import BASH, shell_path


def hashed(path, row):
    row = dict(row)
    row['content_hash'] = P.digest(row)
    P.write_new(path, row)
    return row


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path/'campaign'
    (root/'source').mkdir(parents=True)
    (root/'probes').mkdir()
    (root/'source/frozen.py').write_text('# old source\n')
    (root/'probes/restart.pt').write_bytes(b'checkpoint identity fixture')
    inventory = {'frozen.py': P.sha(root/'source/frozen.py')}
    P.write_new(root/'source_manifest.json', inventory)
    launch = hashed(root/'launch.json', dict(output=str(root), python=sys.executable,
        source_sha256=P.digest(inventory), nccl_environment={}))
    hashed(root/'campaign.json', dict(schema='part_ev_triangles_benchmark_v1',
        launch=launch, source_sha256=P.digest(inventory), parameters=2334662,
        task=dict(architecture='state_triangles_layerwise', seed=101),
        policy=dict(world_size=2, global_batch=512, training_precision='fp16'),
        authorizes_training=False))
    return root


def args(tmp_path, campaign, mode='plan', reviewed=False):
    return SimpleNamespace(mode=mode, reviewed=reviewed, campaign=campaign,
        output=tmp_path/'out', account='reu-aisocial')


def fingerprints(path):
    return {p.relative_to(path).as_posix(): P.sha(p) for p in path.rglob('*') if p.is_file()}


def test_plan_is_readonly_and_uses_saved_probe(campaign, tmp_path, capsys):
    before = fingerprints(tmp_path)
    S.launch(args(tmp_path, campaign))
    assert fingerprints(tmp_path) == before
    assert 'debug job' in capsys.readouterr().out
    (campaign/'probes/restart.pt').unlink()
    with pytest.raises(ValueError, match='required'):
        S.launch(args(tmp_path, campaign))


def test_donor_source_and_identity_are_strict(campaign):
    spec, _, artifacts = P.donor(campaign)
    assert set(artifacts) == set(P.ARTIFACTS)
    (campaign/'source/frozen.py').write_text('# changed')
    with pytest.raises(ValueError, match='inventory'):
        P.donor(campaign)


@pytest.mark.parametrize('field,value', [('world_size', 1), ('global_batch', 256), ('training_precision', 'fp32')])
def test_reject_wrong_runtime(campaign, field, value):
    path = campaign/'campaign.json'
    row = P.read(path)
    row['policy'][field] = value
    row['content_hash'] = P.digest({k:v for k,v in row.items() if k != 'content_hash'})
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='two-A100'):
        P.donor(campaign)


def test_checkpoint_shape_identity_and_cursors():
    spec = dict(content_hash='campaign')
    good = dict(campaign_sha256='campaign', rank_states=[{}, {}], model={}, scaler={},
                optimizer={'schema': 'full_ranger_v1'}, progress=dict(attempt=19997, checks=0, fixture_cursor=0))
    P.validate_checkpoint(good, spec, 2)
    for key, value in [('campaign_sha256', 'wrong'), ('rank_states', [{}]), ('progress', {}), ('optimizer', {})]:
        bad = copy.deepcopy(good)
        bad[key] = value
        with pytest.raises(ValueError, match='fixture'):
            P.validate_checkpoint(bad, spec, 2)


@pytest.mark.parametrize('text,allowed', [
    ('PartitionName=debug State=UP MaxTime=01:00:00', True),
    ('PartitionName=debug State=UP MaxTime=06:00:00', True),
    ('PartitionName=debug State=UP MaxTime=1-00:00:00', True),
    ('PartitionName=debug State=UP MaxTime=UNLIMITED', True),
    ('PartitionName=debug State=UP MaxTime=00:30:00', False),
    ('PartitionName=debug State=DOWN MaxTime=06:00:00', False),
    ('PartitionName=tier3 State=UP MaxTime=06:00:00', False),
    ('PartitionName=debug State=UP MaxTime=N/A', False),
])
def test_live_partition_limit(monkeypatch, text, allowed):
    monkeypatch.setattr(S, 'command', lambda argv: text)
    if allowed:
        S.check_partition()
    else:
        with pytest.raises(ValueError):
            S.check_partition()


def test_resources_one_debug_job_only(tmp_path):
    cmd = S.request(tmp_path, 'reu-aisocial')
    assert '--partition=debug' in cmd and '--gres=gpu:a100:2' in cmd
    assert '--time=01:00:00' in cmd and '--mem=128G' in cmd
    assert '--cpus-per-task=8' in cmd and '--no-requeue' in cmd
    assert not any(v.startswith(('--dependency', '--array')) for v in cmd)
    assert P.POLICY['authorizes_training'] is False and not P.POLICY['changes_tolerances']


@pytest.fixture
def submit_fixture(tmp_path, campaign, monkeypatch):
    repo = tmp_path/'repo'
    for name in P.FILES:
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S.ROOT/name, path)
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'core.autocrlf', 'false'], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    monkeypatch.setattr(S, 'ROOT', repo)
    monkeypatch.setattr(S.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(S.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(S.shutil, 'which', lambda tool: '/tools/'+tool)
    monkeypatch.setattr(S.shutil, 'disk_usage', lambda _: SimpleNamespace(free=30*2**30))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setenv('SBATCH_PARTITION', 'wrong')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    real = S.command
    calls = []
    def command(argv, **kwargs):
        if argv[0] == 'scontrol':
            return 'PartitionName=debug State=UP MaxTime=06:00:00'
        if argv[0] == 'sbatch':
            calls.append((argv, kwargs['env']))
            return 'Estimated start: tomorrow' if '--test-only' in argv else '555;cluster'
        return real(argv, **kwargs)
    monkeypatch.setattr(S, 'command', command)
    a = S.parse(['--mode', 'submit', '--campaign', str(campaign), '--output',
                 str(repo/'checkpoints/part_ev_triangle_restart/probe'), '--reviewed'])
    return a, calls


def test_frozen_submission_no_donor_changes_and_review(campaign, submit_fixture, capsys):
    a, calls = submit_fixture
    before = fingerprints(campaign)
    S.launch(a)
    assert fingerprints(campaign) == before
    assert len(calls) == 2 and '--test-only' in calls[0][0] and '--test-only' not in calls[1][0]
    assert not any(k.startswith(('SBATCH_', 'SLURM_')) for _, env in calls for k in env)
    plan = P.load_plan(a.output)
    assert set(plan['diagnostic_source']) == set(P.FILES)
    assert (a.output/'submitted_job_ids.tsv').read_text() == 'diagnostic\t555\n'
    # Moving the checkout cannot affect this frozen diagnostic.
    (S.ROOT/'part_ev_triangle_restart/engine.py').write_text('# changed later')
    P.load_plan(a.output)
    a.mode = 'review'
    S.launch(a)
    assert 'No complete report' in capsys.readouterr().out
    a.mode = 'submit'
    with pytest.raises(FileExistsError):
        S.launch(a)
    assert len(calls) == 2
    (a.output/'source/part_ev_triangle_restart/engine.py').write_text('# tamper')
    with pytest.raises(ValueError, match='inventory'):
        P.load_plan(a.output)


def test_fail_closed_reviewed_and_storage(submit_fixture, monkeypatch):
    a, calls = submit_fixture
    a.reviewed = False
    with pytest.raises(ValueError, match='reviewed'):
        S.launch(a)
    assert not calls and not a.output.exists()
    a.reviewed = True
    monkeypatch.setattr(S.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10))
    with pytest.raises(OSError, match='2 GiB'):
        S.launch(a)
    assert all('--test-only' in cmd for cmd, _ in calls) and not a.output.exists()


def test_ambiguous_submission_retains_intent_and_blocks_retry(submit_fixture, monkeypatch):
    a, calls = submit_fixture
    real = S.command
    def command(argv, **kwargs):
        if argv[0] == 'sbatch' and '--test-only' not in argv:
            return 'unexpected reply'
        return real(argv, **kwargs)
    monkeypatch.setattr(S, 'command', command)
    with pytest.raises(ValueError, match='Ambiguous'):
        S.launch(a)
    assert (a.output/'submission_intent.json').exists()
    assert not (a.output/'submitted_job_ids.tsv').exists()
    with pytest.raises(FileExistsError):
        S.launch(a)


def test_saved_probe_change_blocks_runtime(submit_fixture, campaign):
    a, _ = submit_fixture
    S.launch(a)
    (campaign/'probes/restart.pt').write_bytes(b'different')
    with pytest.raises(ValueError, match='artifacts changed'):
        P.load_plan(a.output)


def test_parse_rejects_broad_output_and_injection(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'ROOT', tmp_path)
    with pytest.raises(SystemExit):
        S.parse(['--mode', 'plan', '--campaign', str(tmp_path/'c'), '--output', str(tmp_path/'c')])
    with pytest.raises(SystemExit):
        S.parse(['--mode', 'plan', '--campaign', str(tmp_path/'c'), '--output',
                 str(tmp_path/'checkpoints/part_ev_triangle_restart/x'), '--account', 'bad;command'])


@pytest.mark.skipif(BASH is None, reason='Bash unavailable')
def test_shell_syntax_and_no_hidden_production_path():
    for kind in ('run', 'submit'):
        path = S.ROOT/f'sbatch/{kind}_part_ev_triangles_restart.sh'
        subprocess.run([BASH, '-n', shell_path(path)], check=True)
    text = (S.ROOT/'sbatch/run_part_ev_triangles_restart.sh').read_text()
    assert '--nproc-per-node=2' in text and '--max-restarts=0' in text
    assert 'diagnostic.sha256' in text and 'NCCL_' in text
    assert 'sbatch ' not in text and 'scancel' not in text
