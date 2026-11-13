"""CPU export tests use synthetic predictions only; no datasets or scheduler."""
from contextlib import nullcontext
import copy
import hashlib
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from part_ev_oscar import common as C, export as E, launcher as L, saved_validation as V


@pytest.fixture(scope='module')
def predictions():
    payloads = {}
    for label in range(10):
        truth = np.full(100000, label, np.int64)
        logits = np.zeros((100000, 10), np.float32)
        logits[:, label] = 5
        f = io.BytesIO()
        np.savez_compressed(f, logits=logits, labels=truth)
        payloads[label] = (f.getvalue(), hashlib.sha256(truth.tobytes()).hexdigest())
    return payloads


@pytest.fixture
def reference(tmp_path, predictions):
    S = E.donor_support()
    root = tmp_path/'evals/completed'
    (root/'source').mkdir(parents=True)
    for name in S.SOURCES:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# original evaluator\n')
    files = sorted([dict(name=f'{prefix}_{i:03d}.root', num_entries=100000,
                         path=str(tmp_path/'data'/f'{prefix}_{i:03d}.root'), size_bytes=1, mtime_ns=2)
                    for prefix in C.PREFIXES for i in range(120, 125)], key=lambda r: r['name'])
    (tmp_path/'data').mkdir()
    for record in files:
        Path(record['path']).write_bytes(b'x')
        record.update(E.record(record['path'], 100000))
    donors = {key: dict(key=key, root=str(tmp_path/('donor_'+key)), python=sys.executable,
        best_check=C.SELECTED[key], checkpoint_sha256=key+'weights', campaign_sha256=key+'campaign',
        training_runtime={'world_size': 1 if key == 'part' else 2}, files=files,
        task={'architecture': 'part' if key == 'part' else 'state_layerwise', 'seed': 101})
        for key in C.SELECTED}
    for key, donor in donors.items():
        original = Path(donor['root'])
        (original/'source').mkdir(parents=True)
        (original/'source/model.py').write_text('# frozen donor source\n')
        (original/'source.sha256').write_text(C.sha(original/'source/model.py')+'  model.py\n')
        C.write(original/'campaign.json', {'expected_gpu': {'nccl_environment': {}}})
        (original/'weights.pt').write_bytes(key.encode())
        donor['pins'] = {'weights.pt': C.sha(original/'weights.pt')}
    plan = C.seal(dict(schema='part_ev_rejection_eval_v1', donors=donors, population='official_validation_5M',
        independent_test=False, precision='FP32', batch_size=512,
        source={name: C.sha(root/'source'/name) for name in S.SOURCES}))
    C.write(root/'plan.json', plan)
    for key, donor in donors.items():
        (root/key).mkdir()
        receipts = []
        for i, record in enumerate(files):
            label = C.PREFIXES[record['name'].rsplit('_', 1)[0]]
            payload, labels_sha = predictions[label]
            path = root/key/f'file_{i:03d}.npz'
            path.write_bytes(payload)
            side = dict(contract=V.contract(plan, donor, record), sha256=C.sha(path),
                labels_sha256=labels_sha, precision={'passed': True, 'expected_activation_dtype': 'torch.float32'})
            C.write(path.with_suffix('.json'), side)
            receipts.append(side)
        C.write(root/key/'metrics.json', C.seal(dict(model=key, plan_sha256=plan['content_hash'],
            campaign_sha256=donor['campaign_sha256'], checkpoint_sha256=donor['checkpoint_sha256'],
            best_check=donor['best_check'], task=donor['task'], training_runtime=donor['training_runtime'],
            evaluation_runtime=dict(gpu='A100', world_size=1, precision='FP32', batch_size=512),
            metrics=dict(events=5000000, accuracy=1., class_support={n: 500000 for n in C.CLASSES}),
            population='official_validation_5M', independent_test=False, identity_manifest=receipts)))
    return root, donors


def reseal(path, change):
    value = C.checked(path)
    del value['content_hash']
    change(value)
    C.write(path, C.seal(value))


def test_discovery_is_read_only_and_rejects_ambiguous_matches(reference, tmp_path):
    root, donors = reference
    before = C.tree_hashes(root)
    found = V.discover(donors, search_root=root.parent)
    assert found['part']['root'] == str(root.resolve())
    assert C.tree_hashes(root) == before
    shutil.copytree(root, root.parent/'duplicate')
    with pytest.raises(ValueError, match='Found 2'): V.discover(donors, search_root=root.parent)
    assert V.discover(donors, explicit=root) == found


def test_discovery_missing_results_never_falls_back_to_gpu(tmp_path):
    with pytest.raises(ValueError, match='validation-results'):
        V.discover({}, search_root=tmp_path)


@pytest.mark.parametrize('change', ['checkpoint', 'selection', 'population', 'precision', 'batch',
                                   'receipt', 'source', 'missing', 'donor'])
def test_metadata_rejects_wrong_reference(reference, change):
    root, donors = reference
    metrics = root/'part/metrics.json'
    if change == 'checkpoint': reseal(metrics, lambda r: r.update(checkpoint_sha256='other'))
    if change == 'selection': reseal(metrics, lambda r: r.update(best_check=45))
    if change == 'population': reseal(metrics, lambda r: r.update(population='test'))
    if change == 'precision': reseal(metrics, lambda r: r['evaluation_runtime'].update(precision='FP16'))
    if change == 'batch': reseal(metrics, lambda r: r['evaluation_runtime'].update(batch_size=256))
    if change == 'receipt': C.write(root/'part/file_000.json', {})
    if change == 'source': (root/'source/scripts/run_part_ev_rejection_eval.py').write_text('changed')
    if change == 'missing': (root/'part/file_000.npz').unlink()
    if change == 'donor': donors = copy.deepcopy(donors); donors['part']['checkpoint_sha256'] = 'other'
    with pytest.raises((ValueError, FileNotFoundError)): V.describe(root, donors)


def test_worker_audits_all_50_and_preserves_exact_saved_probe_values(reference):
    root, donors = reference
    descriptor = V.describe(root, donors)['part']
    before = C.tree_hashes(root)
    probes = V.audit_payloads(descriptor, donors['part'])
    assert len(probes) == 10 and C.tree_hashes(root) == before
    for name, probe in probes.items():
        label = C.PREFIXES[name.removesuffix('_120.root')]
        assert probe['logits'].shape == (512, 10)
        assert np.all(probe['logits'][:, label] == 5) and np.all(probe['labels'] == label)
    # A non-probe file also must pass: no verification of only convenient slices.
    (root/'part/file_001.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bytes changed'):
        V.audit_payloads(descriptor, donors['part'])


def test_worker_recomputes_accuracy_and_rejects_inconsistent_report(reference):
    root, donors = reference
    reseal(root/'part/metrics.json', lambda r: r['metrics'].update(accuracy=.5))
    descriptor = V.describe(root, donors)['part']
    with pytest.raises(ValueError, match='accuracy'):
        V.audit_payloads(descriptor, donors['part'])


def test_portable_provenance_has_no_full_prediction_payloads(reference, tmp_path):
    root, donors = reference
    descriptor = V.describe(root, donors)['part']
    probes = V.audit_payloads(descriptor, donors['part'])
    directory = tmp_path/'portable/part'
    V.copy_provenance(descriptor, directory/'saved_validation')
    model = dict(key='part', export_gpu=None, reference_evaluation=descriptor,
                 checkpoint_sha256=donors['part']['checkpoint_sha256'],
                 campaign_sha256=donors['part']['campaign_sha256'], best_check=46,
                 probes=[dict(original_name=name, saved_prediction=p['evidence'], precision=p['precision'])
                         for name, p in probes.items()])
    V.verify_portable_provenance(directory, model)
    assert not list(directory.rglob('*.npz'))
    assert not list(directory.rglob('*.root'))
    model['probes'][0]['saved_prediction']['prediction_sha256'] = '0'*64
    with pytest.raises(ValueError): V.verify_portable_provenance(directory, model)


def test_cpu_scheduler_requests_zero_gpu_and_snapshot_worker(tmp_path):
    plan = dict(export_mode=V.MODE, donors={'part': {'python': '/env/bin/python'}},
                partition='tier3', cpu_partition='tier3', account='reu-aisocial')
    args = L.request(tmp_path, plan, 'export-cpu', 'part')
    assert not any(x.startswith(('--gres', '--gpu')) for x in args)
    assert '--partition=tier3' in args and '--cpus-per-task=4' in args and '--mem=64G' in args
    assert '--time=2:00:00' in args and '--tool-root' in args
    assert 'export-cpu' in args and '/env/bin/python' in args


def test_new_export_plan_and_submission_default_cpu(reference, tmp_path, monkeypatch, capsys):
    root, donors = reference
    monkeypatch.setattr(E, 'describe', lambda *_: donors)
    monkeypatch.setattr(E.donor_support(), 'refuse_active', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(L, 'no_active', lambda _: None)
    monkeypatch.setattr(C, 'lock', lambda _: nullcontext())
    jobs = []
    monkeypatch.setattr(L, 'submit', lambda o,p,phase,key: jobs.append((phase,key,p)) or str(len(jobs)))
    output = tmp_path/'new_export'
    base = ['--output', str(output), '--validation-results', str(root)]
    L.main(['export-plan', *base])
    assert not output.exists() and not jobs
    assert 'zero GPUs' in capsys.readouterr().out
    with pytest.raises(ValueError, match='GRES'):
        L.main(['export-submit', *base, '--gres', 'gpu:a100:1'])
    assert not output.exists() and not jobs
    L.main(['export-submit', *base])
    assert [(phase,key) for phase,key,_ in jobs] == [('export-cpu','part'), ('export-cpu','state')]
    assert all(p['export_mode'] == V.MODE for _,_,p in jobs)
    C.verify_files(output/'source', C.checked(output/'export_plan.json')['tools'])


def test_cpu_export_refuses_gpu_reference_execution(tmp_path):
    tools = L.snapshot(tmp_path/'source')
    C.write(tmp_path/'export_plan.json', C.seal(dict(tools=tools, export_mode=V.MODE)))
    with pytest.raises(ValueError, match='CPU-only'): E.export_model(tmp_path, 'part')


def test_cpu_export_complete_handoff_never_calls_model_or_gpu(reference, tmp_path, monkeypatch):
    import torch
    from part_ev_oscar import results as M
    root, donors = reference
    refs = V.describe(root, donors)
    original_bytes = {key: C.tree_hashes(d['root']) for key, d in donors.items()}
    before = C.tree_hashes(root)
    monkeypatch.setattr(C, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(E.donor_support(), 'refuse_active', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(E, 'environment', lambda: {'python': '3.10.21', 'packages': {}})
    def forbidden(*_args, **_kwargs):
        raise AssertionError('No GPU reference generation or model forward is allowed')
    monkeypatch.setattr(torch.cuda, 'get_device_name', forbidden)

    class Model(torch.nn.Module):
        def __init__(self, key):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(C.COUNTS[key]))
            self.register_buffer('running_mean', torch.ones(4))
        forward = forbidden

    def finite_loader(inv, rec):
        label = C.PREFIXES[rec['name'].removesuffix('_120.root')]
        return [({name: torch.ones(512, 1, 2) for name in C.INPUTS},
                 {'_label_': torch.full((512,), label, dtype=torch.int64)})]
    D = SimpleNamespace(atomic_torch_save=torch.save, atomic_npz=M.atomic_npz,
                        finite_validation_loader=finite_loader)
    monkeypatch.setattr(V, 'load_cpu_runtime', lambda d: (D, {'inventory': {}}, Model(d['key'])))
    def context(original, output):
        output.write_bytes(b'raw context')
        return dict(policy='weaver_file_wrap_prefix_v1', target_entries=512, padded_length=128,
                    raw_entries=1024, original_entries=100000, flattened_particles=65536)
    monkeypatch.setattr(E, 'write_context_probe', context)
    output = tmp_path/'cpu_bundle'
    tools = L.snapshot(output/'source')
    C.write(output/'export_plan.json', C.seal(dict(tools=tools, donors=donors,
        export_mode=V.MODE, validation_results=refs)))
    for key in C.SELECTED:
        E.export_model(output, key, cpu=True)
        manifest = C.checked(output/key/'model.json')
        assert manifest['export_gpu'] is None and manifest['reference_origin'] == V.MODE
        assert not list((output/key/'saved_validation').rglob('*.npz'))
        for probe in manifest['probes']:
            with np.load(output/key/probe['reference'], allow_pickle=False) as values:
                assert np.all(values['logits'][:, probe['label']] == 5)
                assert np.all(values['labels'] == probe['label'])
        weights = torch.load(output/key/'weights.pt', weights_only=True)['model']
        assert torch.equal(weights['running_mean'], torch.ones(4))
    E.finalize(output)
    relocated = tmp_path/'scratch/pair'
    shutil.copytree(output, relocated)
    C.load_bundle(relocated)
    assert C.tree_hashes(root) == before
    assert {key: C.tree_hashes(d['root']) for key, d in donors.items()} == original_bytes


@pytest.mark.parametrize('key', ['part', 'state'])
def test_cpu_runtime_preserves_original_admission_and_restores_real_model(tmp_path, monkeypatch, key):
    import torch
    from part_ev_oscar import runtime as R
    from part_ev_state_fullscale import protocol as F, reporting
    from part_ev_layerwise_ddp import protocol as P
    torch.set_num_threads(2)
    model = R.build_architecture(key).cpu().eval()
    torch.save({'model': model.state_dict()}, tmp_path/'best_model_val.pt')
    donor = dict(root=str(tmp_path), run=str(tmp_path), key=key, task={'seed': 101},
                 best_check=C.SELECTED[key], checkpoint_sha256=C.sha(tmp_path/'best_model_val.pt'))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for name in V.FROZEN_PACKAGES:
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], '__file__', str(tmp_path/'source'/name/'__init__.py'))
    calls = []
    spec, runtime = {'fixture': 'spec'}, {'world_size': 1 if key == 'part' else 2}
    def check(label, result=None):
        def invoke(*_args, **_kwargs): calls.append(label); return result
        return invoke
    monkeypatch.setattr(F, 'load_campaign', check('load', spec))
    monkeypatch.setattr(F, 'verify_inventory_metadata', check('inventory'))
    monkeypatch.setattr(F, 'require_approval', check('approval', {'runtime': runtime}))
    monkeypatch.setattr(P, 'load', check('load', spec))
    monkeypatch.setattr(P.G.C, 'verify_inventory_metadata', check('inventory'))
    monkeypatch.setattr(P, 'require_approval', check('approval'))
    monkeypatch.setattr(P, 'read_json', check('preflight', {'runtime': runtime}))
    monkeypatch.setattr(reporting, 'validate_training', check('training_audit', donor))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    _, returned, restored = V.load_cpu_runtime(donor)
    assert returned == spec
    assert calls[:3] == ['load', 'inventory', 'approval'] and calls[-1] == 'training_audit'
    assert not torch.cuda.is_initialized()
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert sum(p.numel() for p in restored.parameters()) == C.COUNTS[key]
    assert all(v.device.type == 'cpu' for v in restored.state_dict().values())
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
