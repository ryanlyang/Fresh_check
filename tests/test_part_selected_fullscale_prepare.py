"""Preparation regression: compiler imports must not change parent metadata discovery."""
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from part_selected_fast import trees
from part_selected_fullscale import model as models
from part_selected_screen import region
from scripts import run_part_selected_fullscale as D

P = D.P


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    output = tmp_path / 'campaign'; output.mkdir()
    reference = tmp_path / 'reference'; reference.mkdir()
    vendor = tmp_path / 'setuptools' / '_vendor'
    dist = vendor / 'sel7_compile_only-9.0.dist-info'; dist.mkdir(parents=True)
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: sel7-compile-only\nVersion: 9.0\n')
    package_version = tmp_path / 'installed-version.txt'; package_version.write_text('original')
    native = object()
    state = dict(references=0, loads=0, models=0, expanded=False, fail=False,
                 change_after=False, corrupt=False)
    old = dict(environment={}, inventory={}, precision={}, input_reference={})
    for name in ('inputs/normalization.json', 'inputs/native_stream.json', 'inputs/selected.yaml'):
        content = (P.hashed(dict(schema='part_selected_normalization_v1', fixture=True))
                   if name.endswith('normalization.json') else dict(fixture=True))
        P.write_json(reference / name, content)
    old['inputs'] = {name: P.file_sha256(reference / name) for name in
                     ('inputs/normalization.json', 'inputs/native_stream.json', 'inputs/selected.yaml')}
    P.write_json(reference / 'experiment_spec.json', {'fixture': 'reference'})
    P.write_json(output / 'launch.json', dict(reference=str(reference),
        baseline_control=str(tmp_path / 'baseline'), baseline_evaluation=str(tmp_path / 'evaluation'),
        state_control=str(tmp_path / 'state')))

    def inventory():
        return dict(packages={'torch': package_version.read_text()},
            installed_distributions={d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                                     if d.metadata['Name'] == 'sel7-compile-only'})

    old['environment'] = inventory()
    assert old['environment']['installed_distributions'] == {}

    def checked_reference(path, *, environment=True):
        assert Path(path) == reference
        assert environment  # Never suppress either strict environment check.
        state['references'] += 1
        if old['environment'] != inventory():
            raise ValueError('Use the unchanged Tigris preparation environment')
        return copy.deepcopy(old)

    # Regression oracle: this is what compiling in the parent did before the
    # fix. Use real importlib.metadata discovery, not an allowlist mock.
    def contaminated_build(directory):
        monkeypatch.syspath_prepend(str(vendor))
        state['expanded'] = True
        P.write_json(directory / 'manifest.json', {'backend': 'fixture'})
        return native

    def load(directory):
        state['loads'] += 1
        if P.read_json(directory / 'manifest.json') != {'backend': 'fixture'}:
            raise ValueError('Backend binary/hash verification failed')
        return native

    def make_model(*args, **kwargs):
        state['models'] += 1
        return object()

    monkeypatch.setattr(P.F, 'reference', checked_reference)
    monkeypatch.setattr(D.shutil, 'disk_usage', lambda _: SimpleNamespace(free=24*2**30))
    monkeypatch.setattr(trees, 'build_backend', contaminated_build)
    monkeypatch.setattr(trees, 'load_backend', load)
    monkeypatch.setattr(trees, 'backend_parity', lambda *a: dict(passed=True, exact=True))
    monkeypatch.setattr(region, 'load_backend', lambda _: 'reference_backend')
    monkeypatch.setattr(models, 'build_model', make_model)
    monkeypatch.setattr(models, 'validate_model', lambda model, architecture: dict(parameters=P.COUNTS[architecture]))

    # Real subprocess simulates the compiler's import-path side effect. It uses
    # the same command/phase interface without needing Linux/OpenMP on the test host.
    child = tmp_path / 'build child.py'
    child.write_text('''import importlib.metadata, json, os, sys
from pathlib import Path
root = Path(__file__).parent
output = Path(sys.argv[sys.argv.index('--output')+1])
config = json.loads((root/'child-options.json').read_text())
sys.path.insert(0, str(root/'setuptools/_vendor'))
(output/'child.json').write_text(json.dumps(dict(pid=os.getpid(), executable=sys.executable,
    args=sys.argv[1:], version=importlib.metadata.version('sel7-compile-only'),
    bytecode_disabled=sys.dont_write_bytecode)))
if config['fail']: raise SystemExit(7)
backend = output/'backend'; backend.mkdir(exist_ok=True)
(backend/'manifest.json').write_text(json.dumps({'backend': 'bad' if config['corrupt'] else 'fixture'}))
if config['change_after']: (root/'installed-version.txt').write_text('changed')
''')
    monkeypatch.setattr(D, '__file__', str(child))

    def configure(**updates):
        state.update(updates)
        P.write_json(tmp_path / 'child-options.json', state)

    configure()
    return SimpleNamespace(output=output, old=old, state=state, configure=configure,
                           inventory=inventory, paths=list(sys.path), package_version=package_version)


def test_prepare_rechecks_unchanged_environment_after_isolated_compilation(preparation):
    f = preparation
    D.prepare(f.output)
    assert f.state['references'] == 2  # Before compile AND while freezing spec.
    assert f.state['loads'] == 1 and f.state['models'] == 7
    assert not f.state['expanded'] and sys.path == f.paths
    assert f.inventory() == f.old['environment']
    child = P.read_json(f.output / 'child.json')
    assert child['pid'] != os.getpid() and Path(child['executable']) == Path(sys.executable)
    assert child['args'] == ['--phase', 'build-backend', '--output', str(f.output)]
    assert child['version'] == '9.0' and child['bytecode_disabled']
    spec = P.checked_json(f.output / 'experiment_spec.json', P.SCHEMA)
    assert spec['environment'] == f.old['environment']
    assert spec['authorizes_training'] is False and spec['tasks'] == P.TASKS


def test_actual_startup_environment_change_blocks_before_compilation(preparation):
    f = preparation; f.package_version.write_text('changed')
    with pytest.raises(ValueError, match='unchanged Tigris'):
        D.prepare(f.output)
    assert not (f.output / 'child.json').exists() and f.state['models'] == 0
    assert not (f.output / 'experiment_spec.json').exists()


def test_actual_environment_change_during_compilation_still_blocks(preparation):
    f = preparation; f.configure(change_after=True)
    with pytest.raises(ValueError, match='unchanged Tigris'):
        D.prepare(f.output)
    assert f.state['references'] == 2 and (f.output / 'child.json').is_file()
    assert not (f.output / 'experiment_spec.json').exists()


def test_failed_compiler_never_finishes_preparation(preparation):
    f = preparation; f.configure(fail=True)
    with pytest.raises(subprocess.CalledProcessError) as error:
        D.prepare(f.output)
    assert error.value.returncode == 7
    assert f.state['loads'] == 0 and f.state['models'] == 0
    assert not (f.output / 'experiment_spec.json').exists()


def test_compiled_backend_is_authenticated_in_parent(preparation):
    f = preparation; f.configure(corrupt=True)
    with pytest.raises(ValueError, match='Backend binary/hash'):
        D.prepare(f.output)
    assert f.state['loads'] == 1 and f.state['models'] == 0
    assert not (f.output / 'experiment_spec.json').exists()


def test_internal_build_phase_only_builds_backend(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trees, 'build_backend', lambda path: calls.append(path))
    monkeypatch.setattr(D, 'prepare', lambda _: pytest.fail('Compiler child must not recurse into prepare'))
    monkeypatch.setattr(P, 'create_spec', lambda _: pytest.fail('Compiler child must not freeze the campaign'))
    assert D.main(['--phase', 'build-backend', '--output', str(tmp_path)]) == 0
    assert calls == [tmp_path / 'backend']
    assert not (tmp_path / 'experiment_spec.json').exists()


def test_compiler_deadline_never_finishes_preparation(preparation, monkeypatch):
    def timeout(command, *, check, timeout):
        assert check and timeout == 17400
        raise subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(D.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        D.prepare(preparation.output)
    assert preparation.state['loads'] == 0 and preparation.state['models'] == 0
    assert not (preparation.output / 'experiment_spec.json').exists()
