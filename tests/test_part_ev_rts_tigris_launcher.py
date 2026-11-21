import ast
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch  # Load before the fake Linux platform, even when this test file runs alone.

from part_ev_rts_tigris import protocol as P
from scripts import submit_part_ev_rts_tigris as S
from scripts import run_part_ev_rts_tigris as W
from part_ev_rts_tigris import submission as Q
from tests.test_part_fullscale_launcher import BASH, shell_path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir()
    for name in P.FILES:
        path = repo/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S.ROOT/name, path)
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    donor = tmp_path/'sporc_benchmark'; donor.mkdir(); (donor/'unchanged').write_text('passed')
    sporc = repo/'checkpoints/part_ev_state_production'; sporc.mkdir(parents=True)
    (sporc/'unchanged').write_text('queued SPORC source marker')
    output = repo/'checkpoints/part_ev_rts_tigris/readout'
    monkeypatch.setattr(S, 'ROOT', repo)
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=24*2**30))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    old_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda name: '/fake/'+name if name in ('sbatch', 'scontrol') else old_which(name))
    monkeypatch.setattr(S, 'inspect', lambda a: dict(parameters=P.COUNTS[a.model],
        donor={'model': a.model}, environment={'arch': 'arm'}))
    monkeypatch.setenv('SBATCH_GRES', 'gpu:wrong:99'); monkeypatch.setenv('SLURM_MEM_PER_CPU', '99999')
    monkeypatch.setenv('RANK', '4'); monkeypatch.setenv('WORLD_SIZE', '2'); monkeypatch.setenv('NCCL_DEBUG', 'INFO')
    state = dict(jobs=[], checks=[], fail=None, test_fail=False, partition='State=UP MaxTime=20-00:00:00')
    real = S.command

    def command(argv, **kwargs):
        if argv[0] == 'scontrol': return state['partition']
        if argv[0] == 'sbatch':
            if '--test-only' in argv:
                state['checks'].append(argv)
                if state['test_fail']: raise subprocess.CalledProcessError(1, argv)
                return 'Valid'
            if state['fail'] == len(state['jobs']): raise subprocess.CalledProcessError(1, argv)
            job = str(1000+len(state['jobs']))
            state['jobs'].append((argv, kwargs['env']))
            return job+';tigris'
        return real(argv, **kwargs)

    monkeypatch.setattr(S, 'command', command)
    monkeypatch.setattr(Q, 'command', command)
    args = ['--model', 'readout', '--benchmark', str(donor), '--output', str(output), '--python', sys.executable]
    return repo, donor, output, args, state


def test_plan_readonly(setup, capsys):
    repo, donor, output, args, state = setup
    before = P.inventory(repo); old = P.inventory(donor)
    S.launch(S.parse(['--mode', 'plan', *args]))
    assert not output.exists() and not state['jobs'] and not state['checks']
    assert P.inventory(repo) == before and P.inventory(donor) == old
    assert 'No files/jobs changed' in capsys.readouterr().out


def test_three_independent_pipelines_single_gpu_and_immutable_snapshots(setup):
    repo, donor, output, args, state = setup
    sporc = repo/'checkpoints/part_ev_state_production'
    old_sporc, old_donor = P.inventory(sporc), P.inventory(donor)
    for kind in P.PACKAGES:
        new_args = args.copy(); new_args[1] = kind; new_args[5] = str(output.parent/kind)
        S.launch(S.parse(['--mode', 'pipeline', '--reviewed', *new_args]))
    assert len(state['jobs']) == len(state['checks']) == 12
    for index, kind in enumerate(P.PACKAGES):
        prepare, _ = state['jobs'][index*4]
        benchmark, env = state['jobs'][index*4+1]
        summary, _ = state['jobs'][index*4+2]
        dispatch, _ = state['jobs'][index*4+3]
        assert '--dependency=afterok:'+str(1002+index*4) in dispatch
        assert not any(s.startswith('--gres') for s in dispatch)
        assert '--time=120' in prepare and not any(s.startswith(('--gres', '--dependency')) for s in prepare)
        assert '--time=330' in benchmark and '--gres=gpu:gh200:1' in benchmark
        assert '--partition=tigris' in benchmark and '--cpus-per-task=8' in benchmark and '--mem=128G' in benchmark
        assert '--dependency=afterok:'+str(1000+index*4) in benchmark
        assert '--time=15' in summary and '--dependency=afterany:'+str(1001+index*4) in summary
        assert not any(s.startswith('--gres') for s in summary)
        assert not any(s.startswith('--array') for s in benchmark)
        assert not any(k.startswith(('SBATCH_', 'SLURM_', 'NCCL_')) for k in env)
        assert 'RANK' not in env and 'WORLD_SIZE' not in env
        frozen = P.launch_config(output.parent/kind, interpreter=False)
        assert frozen['model_kind'] == kind and frozen['authorizes_training'] is False
        assert set(frozen['source']) == set(P.FILES)
        assert not (output.parent/kind/'runs').exists()
    assert P.inventory(sporc) == old_sporc and P.inventory(donor) == old_donor
    source = P.inventory(output/'source')
    (repo/'part_ev_rts_tigris/benchmark.py').write_text('# later checkout pull')
    assert P.inventory(output/'source') == source


@pytest.mark.parametrize('fault', ['exists', 'dirty', 'storage', 'partition', 'account'])
def test_submission_fails_before_jobs(setup, monkeypatch, fault):
    repo, _, output, args, state = setup
    if fault == 'exists': output.mkdir(parents=True)
    if fault == 'dirty': (repo/P.FILES[0]).write_text('uncommitted')
    if fault == 'storage': monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=2**30))
    if fault == 'partition': state['partition'] = 'State=UP MaxTime=03:00:00'
    if fault == 'account': state['test_fail'] = True
    with pytest.raises((ValueError, OSError, subprocess.CalledProcessError)):
        S.launch(S.parse(['--mode', 'pipeline', '--reviewed', *args]))
    assert not state['jobs']


def test_partial_submission_retained_never_cancelled_or_duplicated(setup):
    _, _, output, args, state = setup
    state['fail'] = 1
    with pytest.raises(subprocess.CalledProcessError): S.launch(S.parse(['--mode', 'pipeline', '--reviewed', *args]))
    assert (output/'pipeline_job_ids.tsv').read_text().strip() == 'prepare\t1000'
    assert P.read(output/'pipeline_intent.json')['status'] == 'unknown'
    before = P.inventory(output)
    with pytest.raises(FileExistsError): S.launch(S.parse(['--mode', 'pipeline', '--reviewed', *args]))
    assert P.inventory(output) == before and len(state['jobs']) == 1


def test_storage_counts_sporc_and_other_tigris_outputs(tmp_path):
    registry = tmp_path/'checkpoints/part_ev_rts_tigris'
    output = registry/'new'
    sporc = tmp_path/'checkpoints/part_ev_state_production'
    active = sporc/'readout'
    P.write(active/'campaign.json', P.hashed(dict(storage=dict(required_free_bytes=7*2**30, reserve_bytes=4*2**30))))
    P.write(sporc/'readout.owner.json', {'output': str(active)})
    other = registry/'triangles'
    P.write(other/'launch.json', P.hashed(dict(output_budget_bytes=P.BENCHMARK_BYTES)))
    required = P.required_storage(tmp_path, output)
    assert required == 9*2**30-P.used_bytes(active)-P.used_bytes(other)
    P.write(other/'summary.json', dict(passed=True))
    assert P.required_storage(tmp_path, output) == 8*2**30-P.used_bytes(active)
    P.write(active/'reports/production_report.json', {})
    assert P.required_storage(tmp_path, output) == 5*2**30


@pytest.mark.parametrize('kind', list(P.PACKAGES))
def test_actual_frozen_import_rejects_tampering_and_live_science(tmp_path, kind):
    donor = tmp_path/'donor'; source = donor/'source'
    package = source/P.PACKAGES[kind]; package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'protocol.py').write_text('MARKER="frozen"')
    P.write(donor/'source_manifest.json', P.inventory(source))
    program = '\n'.join(['from part_ev_rts_tigris.protocol import activate',
        f'science=activate({str(donor)!r},{kind!r})', 'assert science.MARKER=="frozen"',
        f'assert science.__file__.startswith({str(source)!r})'])
    subprocess.run([sys.executable, '-B', '-c', program], cwd=S.ROOT, check=True, capture_output=True)
    bad = 'import part_ev_fullscale.protocol\n'+program
    result = subprocess.run([sys.executable, '-B', '-c', bad], cwd=S.ROOT, capture_output=True, text=True)
    assert result.returncode and 'outside frozen' in result.stderr
    (package/'protocol.py').write_text('MARKER="changed"')
    result = subprocess.run([sys.executable, '-B', '-c', program], cwd=S.ROOT, capture_output=True, text=True)
    assert result.returncode and 'source changed' in result.stderr



def test_source_syntax_and_shell_interface():
    for name in P.FILES:
        path = S.ROOT/name
        if name.endswith('.py'): ast.parse(path.read_text(encoding='utf-8'), filename=name, feature_version=(3,10))
        if name.endswith('.sh'): subprocess.run([BASH,'-n',shell_path(path)],check=True,capture_output=True)
    shell=(S.ROOT/'sbatch/run_part_ev_rts_tigris.sh').read_text()
    assert 'torchrun' not in shell and 'aarch64' in shell and 'sha256sum' in shell


def test_optin_wrong_cluster(setup,monkeypatch):
    _,_,_,args,state=setup
    with pytest.raises(SystemExit):S.parse(['--mode','pipeline',*args])
    monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    with pytest.raises(ValueError,match='Tigris ARM'):S.launch(S.parse(['--mode','pipeline','--reviewed',*args]))
    assert not state['jobs']


def test_documented_commands_have_valid_bash_syntax():
    import re
    document=(S.ROOT/'part_ev_rts_tigris/README.md').read_text(encoding='utf-8')
    fence=chr(96)*3
    blocks=re.findall(fence+'bash\n(.*?)'+fence,document,re.S)
    assert len(blocks)==3
    for block in blocks:
        subprocess.run([BASH,'-n'],input=block,text=True,check=True,capture_output=True)
    assert 'submitted_job_ids.tsv' in blocks[0]  # Does not need a pending GPU log.
    assert 'EVRTS_SUPPORT_BENCHMARK_JOB:?' in blocks[0]
    assert 'evrts_reference part_ev_support 22114967' not in blocks[0]


def test_support_retry_documentation_has_valid_bash_and_new_donor_guard():
    import re
    document=(S.ROOT/'part_ev_support/README.md').read_text(encoding='utf-8')
    fence=chr(96)*3
    blocks=re.findall(fence+'bash\n(.*?)'+fence,document,re.S)
    for block in blocks:
        subprocess.run([BASH,'-n'],input=block,text=True,check=True,capture_output=True)
    retry=next(b for b in blocks if '--mode pipeline --model support' in b)
    assert 'SUPPORT_ROOT:?' in retry and 'support_parityfix_' in retry
    assert '--reviewed' in retry and 'test -f "${SUPPORT_ROOT}/campaign.json"' in retry

