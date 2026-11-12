import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from part_selected_compact import protocol as P
from part_selected_compact.flops import CONVENTION, LENGTHS


def evidence(arch):
    cost = 30 if arch == 'compact' else 20
    samples = {str(n): {k: dict(per_jet=cost*n*n, operators_per_batch={'aten.bmm': 2*cost*n*n})
                        for k in ('forward', 'training')} for n in LENGTHS}
    row = dict(task=next(t for t in P.TASKS if t['architecture'] == arch), campaign_sha256='campaign',
        complete=True, passed=True, authorizes_training=False, execution=P.EXECUTION, parameters=P.COUNTS[arch],
        backend_evidence={'approved': True}, measured_steps=4200, windows=[dict(steps=s, seconds=s/10) for s in (1000,1000,1000,1000,200)],
        amp_skips=2, startup_seconds=20, length_histogram={'128': 4200},
        input_fingerprints={str(i): str(i) for i in (0,20,1020,2020,4020)},
        selected_input_fingerprints={str(i): 'raw'+str(i) for i in (0,20,1020,2020,4020)},
        checkpoint_seconds=1, checkpoint_bytes=1000000, validation=dict(seconds=20, events=131072, precision={'passed': True}),
        training_precision={'passed': True}, full_length_confirmed=10,
        peak_reserved_bytes=30*2**30, host_peak_bytes=32*2**30, gpu_memory_bytes=96*2**30, gpu='GH200',
        parity={'passed': True, 'directed_pairs': True}, restart={'passed': True, 'exact_each_update': True},
        flops=dict(convention=CONVENTION, samples=samples,
            coefficients={k: ['0','0',str(cost)] for k in ('forward','training')},
            directed_pairs=True, selected_encoders_included=True, checkpoint_recomputation_included=True,
            cpu_tree_work_included=False))
    spec = dict(content_hash='campaign', input_reference={'fingerprints': row['input_fingerprints']},
                backend_evidence=row['backend_evidence'])
    return spec, row


@pytest.mark.parametrize('field,value', [
    ('passed', False), ('complete', False), ('authorizes_training', True), ('parameters', 2272871),
    ('backend_evidence', {}), ('measured_steps', 4199), ('amp_skips', 42),
    ('full_length_confirmed', 9), ('peak_reserved_bytes', 95*2**30), ('host_peak_bytes', 128*2**30),
    ('gpu', 'A100'), ('checkpoint_seconds', float('nan')), ('checkpoint_bytes', 0),
    ('length_histogram', {'129': 4200}), ('parity', {'passed': True}), ('restart', {'passed': True}),
    ('selected_input_fingerprints', {}), ('windows', [dict(steps=4200, seconds=100)]),
])
def test_reject_incomplete_or_changed_evidence(field, value):
    spec, row = evidence('compact')
    row[field] = value
    with pytest.raises((ValueError, KeyError)):
        P.validate_benchmark(row, spec, P.TASKS[0])


def test_reject_changed_selected_cost_curve():
    spec, row = evidence('compact')
    row['flops']['samples']['65']['training']['per_jet'] += 1
    with pytest.raises(ValueError): P.validate_benchmark(row, spec, P.TASKS[0])


def test_summary_budget_no_production_admission_and_no_writes(tmp_path, monkeypatch):
    spec, compact = evidence('compact')
    _, matched = evidence('matched')
    monkeypatch.setattr(P, 'load', lambda _: spec)
    for name, row in (('compact', compact), ('matched', matched)):
        P.write_json(tmp_path/'benchmarks'/f'{name}.json', P.hashed(row))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = P.summarize(tmp_path)
    assert result['passed'] and not result['authorizes_training']
    assert result['budget']['matched_steps'] == 1500000
    assert result['recipes']['matched']['steps_per_epoch'] == 30000
    assert all(r['checks'] == 50 for r in result['models'])
    assert result['required_free_gib_for_future_production'] < 18
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    matched['selected_input_fingerprints']['20'] = 'different raw track value'
    P.write_json(tmp_path/'benchmarks/matched.json', P.hashed(matched))
    with pytest.raises(ValueError, match='Arms differ'): P.summarize(tmp_path)


def test_summary_does_not_treat_missing_as_pass(tmp_path, monkeypatch):
    spec, _ = evidence('compact')
    monkeypatch.setattr(P, 'load', lambda _: spec)
    with pytest.raises(FileNotFoundError): P.summarize(tmp_path)


def test_backend_recheck_requires_exact_approved_evidence(monkeypatch, tmp_path):
    import part_selected_compiler_recovery.backend as B
    artifact = tmp_path/'binary.so'
    artifact.write_bytes(b'pinned')
    spec = dict(donor_files={str(artifact): P.file_sha256(artifact)},
                backend_context={'context': 1}, backend_evidence={'approved': 2})
    calls = []
    def verify(context, *, expected):
        calls.append((context, expected))
        return 'reference', 'fast', expected
    monkeypatch.setattr(B, 'verify', verify)
    assert P.verify_backends(spec) == ('reference', 'fast', spec['backend_evidence'])
    assert calls == [(spec['backend_context'], spec['backend_evidence'])]
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Pinned donor'): P.verify_backends(spec)
    assert len(calls) == 1


def test_prepare_verifies_before_writes_and_copies_only_small_inputs(tmp_path, monkeypatch):
    reference, output = tmp_path/'reference', tmp_path/'new'
    output.mkdir()
    P.write_json(output/'launch.json', {'reference': str(reference)})
    P.write_json(output/'snapshot.json', {'source.py': 'sha'})
    from part_selected_screen.data import write_config
    write_config(reference/'inputs/selected.yaml')
    P.write_json(reference/'inputs/normalization.json', {'test': 1})
    P.write_json(reference/'inputs/native_stream.json', {'test': 2})
    old = dict(environment={'architecture': 'aarch64'}, inventory={'no_files_copied': True}, input_reference={})
    monkeypatch.setattr(P, 'require_space', lambda *_: None)
    monkeypatch.setattr(P, 'donor', lambda _: (old, {}, {}, {}))
    def reject(_): raise ValueError('backend changed')
    monkeypatch.setattr(P, 'verify_backends', reject)
    with pytest.raises(ValueError, match='backend changed'): P.prepare(output)
    assert not (output/'inputs').exists() and not (output/'campaign.json').exists()
    monkeypatch.setattr(P, 'verify_backends', lambda _: ('ref', 'fast', {}))
    result = P.prepare(output)
    assert result['authorizes_training'] is False and result['tasks'] == P.TASKS
    for name in P.INPUTS:
        assert (output/name).read_bytes() == (reference/name).read_bytes()
    assert not (output/'backend').exists() and not (output/'runs').exists()
    with pytest.raises(FileExistsError): P.prepare(output)


def test_small_space_check_never_deletes(tmp_path, monkeypatch):
    import part_ev_compact.protocol as old
    monkeypatch.setattr(old.shutil, 'disk_usage', lambda _: SimpleNamespace(free=5*2**30))
    with pytest.raises(OSError, match='nothing deleted'):
        P.require_space(tmp_path/'future')
    assert not list(tmp_path.iterdir())


@pytest.fixture
def donor_fixture(tmp_path, monkeypatch):
    from part_selected_compiler_recovery import protocol as R
    live, reference, recovery, production = (tmp_path/n for n in ('live', 'reference', 'recovery', 'production'))
    source = {}
    for name in R.FILES:
        path = recovery/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('reviewed '+name)
        source[name] = P.file_sha256(path)
        current = live/name
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_bytes(path.read_bytes())
    (live/'native.py').write_text('unchanged')
    (reference/'source').mkdir(parents=True)
    (reference/'source/native.py').write_text('unchanged')
    env = {'architecture': 'aarch64'}
    stream = P.hashed(dict(environment=env, inventory_sha256='inv',
                          fingerprints={str(i):str(i) for i in (0,20,1020,2020,4020)}))
    for name, content in zip(P.INPUTS, ({'yaml': 1}, {'normal': 2}, stream)):
        P.write_json(reference/name, content)
    old = P.hashed(dict(schema='selected_fullscale_benchmarks_v1', environment=env,
        source={'native.py': P.file_sha256(live/'native.py')}, inventory={'inventory_sha256': 'inv'},
        input_reference=stream, inputs={n:P.file_sha256(reference/n) for n in P.INPUTS},
        reference={'path':str(tmp_path/'historical')}))
    P.write_json(reference/'experiment_spec.json', old)
    plan = P.hashed(dict(output=str(production), benchmark=str(reference), science=old, python=sys.executable))
    P.write_json(production/'production.json', plan)
    artifacts = {str(production/'production.json'): P.file_sha256(production/'production.json')}
    row = R.hashed(dict(schema=R.SCHEMA, output=str(recovery), campaign=str(production),
        source=source, indices=list(R.INDICES), transition=dict(old=R.OLD_BANNER,new=R.NEW_BANNER),
        production_sha256=plan['content_hash'], python=sys.executable, artifacts=artifacts))
    R.write(recovery/'recovery.json', row)
    R.write(recovery/'compatibility.json', R.hashed(dict(schema=R.SCHEMA, recovery_sha256=row['content_hash'],
        indices=list(R.INDICES), passed=True, authorizes_recovery=True, evidence={'same': True})))
    monkeypatch.setattr(P, 'ROOT', live)
    monkeypatch.setattr(P, 'NUMERICAL_FILES', ('native.py',))
    monkeypatch.setattr(P, 'runtime_environment', lambda: env)
    monkeypatch.setattr(P, 'verify_inventory', lambda _: None)
    monkeypatch.setattr(R, 'evidence', lambda _: artifacts)
    return dict(reference=str(reference), recovery=str(recovery), python=sys.executable), live


def test_donor_evidence_is_read_only_and_does_not_activate_old_imports(donor_fixture):
    launch, live = donor_fixture
    paths_before = list(sys.path)
    before = {p:p.read_bytes() for p in live.parent.rglob('*') if p.is_file()}
    old, files, context, proof = P.donor(launch)
    assert old['input_reference']['fingerprints']['4020'] == '4020'
    assert proof == {'same': True} and context['benchmark'] == launch['reference']
    assert str(Path(launch['recovery'])/'compatibility.json') in files
    assert paths_before == sys.path
    assert before == {p:p.read_bytes() for p in live.parent.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind', ['local_math', 'reference_math', 'local_verifier', 'failed_approval', 'normalization', 'environment'])
def test_changed_donor_or_environment_cannot_be_admitted(donor_fixture, monkeypatch, kind):
    launch, live = donor_fixture
    if kind == 'local_math': (live/'native.py').write_text('changed')
    elif kind == 'reference_math': (Path(launch['reference'])/'source/native.py').write_text('changed')
    elif kind == 'local_verifier': (live/'part_selected_compiler_recovery/backend.py').write_text('bypass')
    elif kind == 'normalization': (Path(launch['reference'])/'inputs/normalization.json').write_text('{}')
    elif kind == 'environment': monkeypatch.setattr(P,'runtime_environment',lambda:{'architecture':'aarch64','changed':True})
    else:
        path = Path(launch['recovery'])/'compatibility.json'
        row = P.read_json(path)
        row.pop('content_hash')
        row['passed'] = False
        P.write_json(path, P.hashed(row))
    with pytest.raises(ValueError): P.donor(launch)
