import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_selected_compiler_recovery import protocol as P, backend as B, engine as E
from teacher_logit_reco.relational_part.contracts import canonical_json_bytes


def native_identity():
    return dict(contract_id='relational_ca_tree_v1', schema_version=3, compiler_flags=['-O3'],
        compiler_family='gcc', compiler_major_version=11, compiler_version='11.5.0',
        platform_architecture='aarch64', pytorch_cxx11_abi=True, openmp_available=True)


def manifest():
    row = native_identity()
    return dict(contract_id=row['contract_id'], backend_schema_version=row['schema_version'],
        compiler_flags=row['compiler_flags'], compiler_identity=row['compiler_family'],
        compiler_major_version=11, compiler_version='11.5.0', platform_architecture='aarch64',
        pytorch_cxx11_abi=True, openmp_available=True, python_major_minor='3.10', pytorch_version='original',
        self_test_sha256=hashlib.sha256(canonical_json_bytes({'passed': True})).hexdigest(),
        compiler_driver_version_line=P.OLD_BANNER, compiler_executable='/usr/bin/c++')


def test_exact_compiler_exception(monkeypatch):
    monkeypatch.delenv('CXX', raising=False)
    monkeypatch.setattr(B.shutil, 'which', lambda _: '/usr/bin/c++')
    monkeypatch.setattr(B.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=P.NEW_BANNER+'\n'))
    monkeypatch.setattr(P, 'sha', lambda _: 'driverhash')
    before = manifest()
    got = B.compiler_identity(before)
    assert got['old_banner'] == P.OLD_BANNER and got['new_banner'] == P.NEW_BANNER
    assert before == manifest()  # Never forge or rewrite the historical manifest.
    for cxx in ('clang++', 'ccache c++', 'c++ -O3', '/other/c++', ''):
        monkeypatch.setenv('CXX', cxx)
        with pytest.raises(ValueError):
            B.compiler_identity(before)
    monkeypatch.delenv('CXX')
    for banner in (P.OLD_BANNER, P.NEW_BANNER.replace('-15)', '-16)'), 'g++ 12.0'):
        monkeypatch.setattr(B.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=banner+'\n'))
        with pytest.raises(ValueError, match='reviewed -15'):
            B.compiler_identity(before)
    with pytest.raises(ValueError, match='original compiler'):
        B.compiler_identity(dict(before, compiler_executable='/another/c++'))


@pytest.mark.parametrize('field', ['contract_id', 'schema_version', 'compiler_flags', 'compiler_family',
    'compiler_major_version', 'compiler_version', 'platform_architecture', 'pytorch_cxx11_abi', 'openmp_available'])
def test_every_embedded_build_check_retained(field):
    current = {k: manifest()[k] for k in ('platform_architecture', 'python_major_minor', 'pytorch_version', 'pytorch_cxx11_abi')}
    values = native_identity()
    module = SimpleNamespace(backend_manifest=lambda: values, self_test=lambda: {'passed': True})
    B.validate_runtime(manifest(), module, current)
    values[field] = -1 if field in ('schema_version', 'compiler_major_version') else ([] if field == 'compiler_flags' else 'changed')
    with pytest.raises(RuntimeError, match='Loaded tree backend differs'):
        B.validate_runtime(manifest(), module, current)


@pytest.mark.parametrize('field', ['platform_architecture', 'python_major_minor', 'pytorch_version', 'pytorch_cxx11_abi'])
def test_runtime_abi_drift_rejected(field):
    module = SimpleNamespace(backend_manifest=native_identity, self_test=lambda: {'passed': True})
    current = {k: manifest()[k] for k in ('platform_architecture', 'python_major_minor', 'pytorch_version', 'pytorch_cxx11_abi')}
    current[field] = 'changed'
    with pytest.raises(RuntimeError, match=field):
        B.validate_runtime(manifest(), module, current)


def test_self_test_is_still_required():
    module = SimpleNamespace(backend_manifest=native_identity, self_test=lambda: {'passed': False})
    with pytest.raises(RuntimeError, match='self-test'):
        B.validate_runtime(manifest(), module, {})


def test_historical_smoke_hashes(monkeypatch):
    import teacher_logit_reco.relational_part.ca_tree as ca
    from teacher_logit_reco.relational_part.region_tree import build_reference_tree
    monkeypatch.setattr(ca, 'build_compiled_tree', lambda _, *args: build_reference_tree(*args))
    # Gather deterministic test oracle hashes without compiling anything.
    from teacher_logit_reco.relational_part.region_tree import tree_content_sha256
    result = []
    def capture(tree):
        value = tree_content_sha256(tree); result.append(value); return value
    import teacher_logit_reco.relational_part.region_tree as region
    monkeypatch.setattr(region, 'tree_content_sha256', capture)
    with pytest.raises(ValueError, match='Historical smoke'):
        B.smoke(None, dict(compiled_smoke_tree_sha256='wrong', reference_smoke_tree_sha256='wrong'))
    expected = dict(compiled_smoke_tree_sha256=result[0], reference_smoke_tree_sha256=result[1])
    assert B.smoke(None, expected) == expected


def test_worker_rechecks_exact_admission_evidence(monkeypatch):
    import part_selected_fast.trees as trees
    reference, fast = object(), object()
    monkeypatch.setattr(B, 'load_reference', lambda _: (reference, {}, {'compiler': {'new': '-15'}}))
    monkeypatch.setattr(B, 'smoke', lambda *a: {'compiled': 'unchanged'})
    monkeypatch.setattr(B, 'libraries', lambda: {'libgomp': 'unchanged'})
    monkeypatch.setattr(trees, 'load_backend', lambda _: fast)
    monkeypatch.setattr(trees, 'backend_parity', lambda *a, **k: {'passed': True, 'exact': True, 'invalid_input_rejected': True})
    a, b, result = B.verify(dict(benchmark='benchmark'))
    assert a is reference and b is fast
    B.verify(dict(benchmark='benchmark'), expected=result)
    monkeypatch.setattr(B, 'libraries', lambda: {'libgomp': 'changed'})
    with pytest.raises(ValueError, match='evidence changed'):
        B.verify(dict(benchmark='benchmark'), expected=result)
    monkeypatch.setattr(trees, 'backend_parity', lambda *a, **k: (_ for _ in ()).throw(ValueError('parity failed')))
    with pytest.raises(ValueError, match='parity failed'):
        B.verify(dict(benchmark='benchmark'))


def test_private_loader_binding_does_not_change_original(tmp_path):
    def init(self, old, new):
        self.old = reference_backend(old)
        self.fast = load_backend(new)
    # Controlled globals make real isolation testable without Linux extensions.
    import types
    globals_ = dict(init.__globals__, reference_backend=lambda _: 'original_reference',
                    load_backend=lambda _: 'original_fast')
    init = types.FunctionType(init.__code__, globals_)
    base = type('Original', (), {'__init__': init})
    old, new = tmp_path/'old/backend', tmp_path/'new/backend'
    plan = dict(benchmark=str(tmp_path/'new'), science=dict(reference=dict(path=str(tmp_path/'old'))))
    private = E.admitted_bench(base, plan, 'admitted_reference', 'admitted_fast')
    assert private(old, new).old == 'admitted_reference'
    assert private(old, new).fast == 'admitted_fast'
    assert base(old, new).old == 'original_reference'
    assert base.__init__ is init and globals_['load_backend'](None) == 'original_fast'
    with pytest.raises(ValueError, match='different directory'):
        private(tmp_path/'other', new)


@pytest.mark.parametrize('i', P.INDICES)
def test_recovered_composition_matches_original_model_and_training_contract(tmp_path, monkeypatch, i):
    from part_selected_fullscale import benchmark, protocol as science, steps
    from tests.test_part_selected_fullscale_model import factory
    task = science.TASKS[i]
    spec = dict(content_hash='parent', tasks=science.TASKS, recipes=science.recipes(), inventory={})
    runtime = dict(campaign_sha256='parent', gpu='cpu-test')
    def init(self, output, index, check_stop):
        self.output, self.index, self.spec, self.task = output, index, spec, task
        self.runtime, self.device, self.check_stop = runtime, torch.device('cpu'), check_stop
    monkeypatch.setattr(benchmark.Bench, '__init__', init)
    monkeypatch.setattr(benchmark.Bench, 'model', lambda self: factory(task['architecture'], fast=True))
    evidence = tmp_path/'evidence.json'; P.write(evidence, dict(runtime=runtime))
    plan = dict(benchmark=str(tmp_path), science=dict(spec, reference=dict(path=str(tmp_path))),
                content_hash='production', review=dict(models=[dict(evidence_path=str(evidence))]*7))
    worker = E.create_engine(tmp_path, plan, i, object(), object())
    model = worker.make_model(task, torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters()) == science.COUNTS[task['architecture']]
    assert worker.training_step is steps.training_step
    assert worker.recipe['total_steps'] == 1000000 and worker.recipe['epochs'] == 50
    assert worker.P.load_campaign(tmp_path)['content_hash'] == 'production'
    assert worker.validate.__globals__['check_stop'] == worker.check_stop
    assert benchmark.Bench.__init__ is init
    from part_selected_production.engine import create_engine as original_engine
    from part_ev_fullscale.optimizer import build_optimizer
    from scripts.diagnose_part_selected_restart import compare_tree
    from tests.test_part_selected_fullscale_model import batch_for
    old_worker = original_engine(tmp_path, plan, i)
    old_model = old_worker.make_model(task, torch.device('cpu'))
    assert compare_tree(model.state_dict(), old_model.state_dict())['exact']
    batch = batch_for(task['architecture'], particles=5)
    observations = []
    for current, engine in ((old_model, old_worker), (model, worker)):
        opt = build_optimizer(current, engine.recipe)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        torch.manual_seed(4001)
        engine.training_step(current, opt, scaler, batch, engine.device, engine.recipe, 0)
        observations.append(dict(model=current.state_dict(), optimizer=opt.state_dict()))
    assert compare_tree(*observations)['exact']
    with pytest.raises(ValueError, match='non-selected'):
        E.create_engine(tmp_path, plan, 3, object(), object())


def test_approval_rejects_wrong_scope_or_failed_report(tmp_path):
    row = dict(content_hash='recovery')
    good = dict(schema=P.SCHEMA, recovery_sha256='recovery', indices=list(P.INDICES), passed=True, authorizes_recovery=True)
    P.write(tmp_path/'compatibility.json', P.hashed(good))
    assert P.approval(tmp_path, row)['passed']
    for field, value in [('indices', [0, 2, 5, 6]), ('passed', False), ('authorizes_recovery', False),
                         ('recovery_sha256', 'other'), ('schema', 'other')]:
        P.write(tmp_path/'compatibility.json', P.hashed(dict(good, **{field: value})))
        with pytest.raises(ValueError):
            P.approval(tmp_path, row)


def test_recovery_source_and_artifacts_are_pinned(tmp_path, monkeypatch):
    for n in P.FILES:
        path = tmp_path/'source'/n; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# frozen')
    artifact = tmp_path/'binary.so'; artifact.write_bytes(b'original')
    artifacts = {str(artifact): P.sha(artifact)}
    plan = dict(content_hash='production', python='original')
    row = P.hashed(dict(schema=P.SCHEMA, output=str(tmp_path), campaign='donor', indices=list(P.INDICES),
        source={n: P.sha(tmp_path/'source'/n) for n in P.FILES}, artifacts=artifacts,
        transition=dict(old=P.OLD_BANNER, new=P.NEW_BANNER), production_sha256='production', python='original'))
    P.write(tmp_path/'recovery.json', row)
    monkeypatch.setattr(P, 'activate', lambda _: (None, plan))
    monkeypatch.setattr(P, 'evidence', lambda _: artifacts)
    assert P.load(tmp_path)[0] == row
    artifact.write_bytes(b'new')
    with pytest.raises(ValueError, match='artifact changed'):
        P.load(tmp_path)
    artifact.write_bytes(b'original')
    (tmp_path/'source/extra.py').write_text('# extra')
    with pytest.raises(ValueError, match='inventory'):
        P.load(tmp_path)


def test_failed_check_cannot_publish_admission(tmp_path, monkeypatch):
    from scripts import run_part_selected_compiler_recovery as worker
    monkeypatch.setattr(P, 'load', lambda _: (dict(campaign=str(tmp_path)), None, {}))
    def fail(_):
        raise ValueError('Backend parity failed')
    monkeypatch.setattr(B, 'verify', fail)
    with pytest.raises(ValueError, match='parity failed'):
        worker.main(['--phase', 'check', '--output', str(tmp_path)])
    assert not (tmp_path/'compatibility.json').exists()
