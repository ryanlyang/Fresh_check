"""Synthetic/local tests only: no research datasets, SSH, or real scheduler calls."""
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from part_ev_oscar import common as C, export as E, runtime as R, results as M, launcher as L


def test_sealed_json_detects_edit(tmp_path):
    p = tmp_path/'r.json'
    C.write(p, C.seal({'answer': 42}))
    assert C.checked(p)['answer'] == 42
    row = C.read(p)
    row['answer'] = 43
    C.write(p, row)
    with pytest.raises(ValueError, match='changed'):
        C.checked(p)


@pytest.mark.parametrize('name', ['../outside', '/root', 'C:/x', 'folder\\file', 'a/../../b'])
def test_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        C.child(tmp_path, name)


def test_expected_test_population():
    files = C.expected_files()
    assert len(files) == 200
    assert all(sum(x == i for x in files.values()) == 20 for i in range(10))
    assert not any('_120.' in p for p in files)


@pytest.mark.parametrize('change', ['shape', 'nan', 'dtype', 'labels', 'class'])
def test_prediction_contract_rejects_invalid_arrays(change):
    logits, labels = np.zeros((3, 10), np.float32), np.zeros(3, np.int64)
    if change == 'shape': logits = logits[:, :9]
    if change == 'nan': logits[0, 0] = np.nan
    if change == 'dtype': logits = logits.astype(np.float16)
    if change == 'labels': labels[0] = 10
    if change == 'class': labels[0] = 1
    with pytest.raises(ValueError): C.validate_arrays(logits, labels, 3, 0)


def test_parity_thresholds_and_argmax_gate():
    x = np.array([[0., 1.], [2., 0.]], np.float32)
    assert C.comparison(x.copy(), x)['passed']
    assert not C.comparison(x+1, x)['passed']
    assert not C.comparison(np.full_like(x, np.nan), x)['passed']
    # Even close values fail when class ordering changes.
    a = np.array([[0., 1e-7]], np.float32)
    assert not C.comparison(a[:, ::-1], a)['passed']


@pytest.fixture
def archived(tmp_path, monkeypatch):
    data = tmp_path/'test_20M'
    data.mkdir()
    archive = tmp_path/'test.tar'
    with tarfile.open(archive, 'w') as tar:
        directory = tarfile.TarInfo('test_20M')
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name in C.expected_files():
            content = ('fake root '+name).encode()
            (data/name).write_bytes(content)
            entry = tarfile.TarInfo('test_20M/'+name)
            entry.size = len(content)
            tar.addfile(entry, io.BytesIO(content))
    import uproot
    monkeypatch.setattr(uproot, 'open', lambda _: nullcontext({'tree': SimpleNamespace(classname='TTree', num_entries=100000)}))
    return archive, data, C.sha(archive)


def test_archive_verifies_exact_200_files_without_writes(archived):
    archive, data, digest = archived
    before = C.tree_hashes(data)
    rows = R.verify_test_archive(archive, data, digest)
    assert len(rows) == 200 and sum(r['num_entries'] for r in rows) == 20000000
    assert C.tree_hashes(data) == before


@pytest.mark.parametrize('change', ['bytes', 'extra', 'missing', 'archive_hash'])
def test_archive_rejects_incomplete_or_changed_extraction(archived, change):
    archive, data, digest = archived
    first = next(iter(data.iterdir()))
    if change == 'bytes': first.write_bytes(b'x'*first.stat().st_size)
    if change == 'extra': (data/'other.root').write_bytes(b'x')
    if change == 'missing': first.unlink()
    if change == 'archive_hash': digest = '0'*64
    with pytest.raises(ValueError): R.verify_test_archive(archive, data, digest)


def test_real_probe_root_is_ttree_and_preserves_arrays(tmp_path):
    import awkward as ak
    import uproot
    source = tmp_path/'original.root'
    arrays = dict(part_energy=ak.Array([[1., 2.], [3.], [4., 5., 6.]]),
                  label_QCD=ak.Array([1, 1, 1]), jet_pt=ak.Array([2., 3., 4.]))
    with uproot.recreate(source) as f:
        f.mktree('tree', {k: v.type.content for k, v in arrays.items()})
        f['tree'].extend(arrays)
    target = tmp_path/'probe.root'
    E.write_probe_root(source, target, events=2)
    with uproot.open(target) as f:
        assert f['tree'].classname == 'TTree'
        for k, v in arrays.items():
            assert ak.almost_equal(f['tree'][k].array(), v[:2])


def test_actual_preprocessing_matches_validation_slice_and_test_loader(tmp_path):
    import awkward as ak
    import uproot
    from part_ev_screen.data import finite_validation_loader
    events = 19
    n = 6
    ones = ak.unflatten(np.ones(events * n, np.float32), np.full(events, n, np.int64))
    arrays = {k: ones for k in ('part_px', 'part_py', 'part_pz', 'part_charge',
        'part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron',
        'part_isMuon', 'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr')}
    arrays.update(part_energy=ones*3, part_deta=ones*.1, part_dphi=ones*.2)
    for name in ('jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4'):
        arrays[name] = ak.Array(np.full(events, 100, np.float32))
    for i, name in enumerate(C.CLASSES):
        arrays['label_'+name] = ak.Array(np.full(events, int(i == 0), np.int32))
    original = tmp_path/'ZJetsToNuNu_120.root'
    with uproot.recreate(original) as f:
        f.mktree('tree', {k: v.type.content for k, v in arrays.items()})
        f['tree'].extend(arrays)
    original_record = E.record(original, events)
    reference = next(iter(finite_validation_loader({'splits': {'val': {'files': [original_record]}}}, original_record, batch_size=16)))
    probe = tmp_path/'probe.root'
    E.write_probe_root(original, probe, events=16)
    candidate = list(R.loader(E.record(probe, 16), 7))
    assert [len(b[1]['_label_']) for b in candidate] == [7, 7, 2]
    for name in C.INPUTS:
        joined = np.concatenate([batch[0][name].numpy() for batch in candidate])
        assert np.array_equal(joined, reference[0][name].numpy())
    assert np.array_equal(np.concatenate([b[1]['_label_'] for b in candidate]), reference[1]['_label_'])


@pytest.mark.parametrize('events', [40, 600])
def test_nonconstant_wrap_padding_context_roundtrip(tmp_path, events):
    """Small files require their whole table; large files only a bounded prefix."""
    import awkward as ak
    import torch
    import uproot
    from part_ev_screen.data import finite_validation_loader
    lengths = 2 + np.arange(events) % 10
    varied = ak.unflatten(np.arange(lengths.sum(), dtype=np.float32)/10000 + .01, lengths)
    arrays = {k: varied for k in ('part_px', 'part_py', 'part_pz', 'part_charge',
        'part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron',
        'part_isMuon', 'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr')}
    arrays.update(part_energy=varied*3, part_deta=varied*.1, part_dphi=varied*.2)
    for name in ('jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles',
                 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4'):
        arrays[name] = ak.Array(np.arange(events, dtype=np.float32) + 100)
    for i, name in enumerate(C.CLASSES):
        arrays['label_'+name] = ak.Array(np.full(events, int(i == 0), np.int32))
    original = tmp_path/'ZJetsToNuNu_120.root'
    with uproot.recreate(original) as f:
        f.mktree('tree', {k: v.type.content for k, v in arrays.items()})
        f['tree'].extend(arrays)
    r = E.record(original, events)
    reference = next(iter(finite_validation_loader({'splits': {'val': {'files': [r]}}}, r, batch_size=16)))
    too_small = tmp_path/'old_probe.root'
    E.write_probe_root(original, too_small, events=16)
    old = next(iter(R.loader(E.record(too_small, 16), 16)))
    assert torch.equal(reference[0]['pf_mask'], old[0]['pf_mask'])
    valid = reference[0]['pf_mask'].bool().expand_as(reference[0]['pf_vectors'])
    assert torch.equal(reference[0]['pf_vectors'][valid], old[0]['pf_vectors'][valid])
    assert not torch.equal(reference[0]['pf_vectors'], old[0]['pf_vectors'])
    with pytest.raises(ValueError, match='unequal values'):
        E.assert_same_batch(reference, old)

    probe = tmp_path/'context_probe.root'
    context = E.write_context_probe(original, probe, events=16)
    assert context['raw_entries'] > 16
    assert context['raw_entries'] == events if events == 40 else context['raw_entries'] < events
    pr = E.record(probe, context['raw_entries'])
    restored = next(iter(finite_validation_loader({'splits': {'val': {'files': [pr]}}}, pr, batch_size=16)))
    E.assert_same_batch(reference, restored)
    actual = next(iter(R.loader(pr, 16)))
    E.assert_same_batch(reference, actual)
    # Real valid-slot changes are still rejected; no mask-only/tolerance bypass.
    actual[0]['pf_vectors'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='pf_vectors'):
        E.assert_same_batch(reference, actual)


@pytest.mark.parametrize('limit', [None, 16])
def test_forward_scores_only_probe_targets_but_all_test_events(monkeypatch, limit):
    import torch
    import part_ev_screen.precision as precision
    seen = []
    batches = []
    for start in (0, 7, 14):
        batch = {name: torch.arange(start, min(start+7, 19)).reshape(-1, 1, 1).float()
                 for name in C.INPUTS}
        batches.append((batch, {'_label_': torch.zeros(len(batch['pf_mask']), dtype=torch.int64)}))
    monkeypatch.setattr(R, 'loader', lambda record, batch: iter(batches))
    monkeypatch.setattr(precision, 'model_precision', lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(precision, 'audit_precision', lambda *_args, **_kwargs: nullcontext({'passed': True}))
    def model(*values):
        seen.extend(values[0].flatten().tolist())
        return torch.zeros(len(values[0]), 10)
    record = dict(num_entries=19, label=0)
    scores, labels, _, inputs = R.forward_file(model, 'cpu', record, 7, capture=True, event_limit=limit)
    expected = limit or 19
    assert scores.shape == (expected, 10) and len(labels) == expected
    assert seen == list(range(expected))
    assert inputs['pf_mask'].flatten().tolist() == seen
    if limit is None:
        # Production still consumes and checks the full stream, never silently
        # truncating an unexpected excess to the registered population size.
        with pytest.raises(ValueError, match='prediction shape'):
            R.forward_file(model, 'cpu', dict(num_entries=18, label=0), 7)


@pytest.mark.parametrize('change', ['short', 'range', 'inconsistent', 'policy'])
def test_bundle_rejects_bad_probe_padding_context(change):
    probe = dict(entry_start=0, entry_stop=512, raw_entries=2048, padding_context=dict(
        policy='weaver_file_wrap_prefix_v1', target_entries=512, padded_length=128,
        raw_entries=2048, original_entries=100000, flattened_particles=65536))
    if change == 'short': probe['padding_context']['flattened_particles'] = 100
    if change == 'range': probe['entry_stop'] = 2048
    if change == 'inconsistent': probe['padding_context']['raw_entries'] = 4096
    if change == 'policy': probe['padding_context']['policy'] = 'ignore_padding'
    with pytest.raises(ValueError): C.verify_probe_context(probe)


def test_environment_tag_normalization_keeps_release_cuda_and_sources_strict(monkeypatch):
    expected = dict(python='3.10.21', architecture='x86_64', packages={'torch': '2.5.1'},
                    torch_cuda='12.4', weaver_sources={'model': 'abc'})
    actual = dict(expected, python='3.10.20', packages={'torch': '2.5.1+cu124'})
    monkeypatch.setattr(C, 'checked', lambda _: {'environment': expected})
    monkeypatch.setattr(R, 'environment', lambda: actual)
    assert R.environment_check({'bundle': '/fake'}) == actual
    actual['torch_cuda'] = '12.6'
    with pytest.raises(ValueError, match='dependencies differ'):
        R.environment_check({'bundle': '/fake'})


@pytest.mark.parametrize('change', ['failed', 'missing_rows', 'wrong_plan', 'changed_policy'])
def test_admission_requires_both_complete_matching_parities(tmp_path, monkeypatch, change):
    plan = {'content_hash': 'plan'}
    monkeypatch.setattr(R, 'inventory', lambda *_: {'content_hash': 'inventory'})
    for key in C.SELECTED:
        row = dict(plan_sha256='plan', model=key, policy=C.PARITY, passed=True,
                   events=5120, rows=[{'passed': True} for _ in range(10)])
        if key == 'state':
            if change == 'failed': row['passed'] = False
            if change == 'missing_rows': row['rows'].pop()
            if change == 'wrong_plan': row['plan_sha256'] = 'other'
            if change == 'changed_policy': row['policy'] = {}
        C.write(tmp_path/f'parity_{key}.json', C.seal(row))
    with pytest.raises(ValueError, match='parity reports'):
        M.admission_record(tmp_path, plan)


def test_evaluation_submission_requires_review_and_separate_dependencies(tmp_path, monkeypatch):
    p = plan()
    p['content_hash'] = 'plan'
    monkeypatch.setattr(C, 'load_plan', lambda _: p)
    monkeypatch.setattr(C, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(L, 'no_active', lambda _: None)
    monkeypatch.setattr(M, 'admission_record', lambda *_: C.seal({'plan_sha256': 'plan'}))
    calls = []
    def submit(output, plan, phase, key=None, dependencies=()):
        calls.append((phase, key, list(dependencies)))
        return str(len(calls))
    monkeypatch.setattr(L, 'submit', submit)
    args = SimpleNamespace(output=tmp_path, command='evaluate-submit', reviewed=False)
    with pytest.raises(ValueError, match='reviewed'): L.campaign_submit(args)
    assert calls == []
    args.reviewed = True
    L.campaign_submit(args)
    assert calls == [('evaluate', 'part', []), ('metrics', 'part', ['1']),
                     ('evaluate', 'state', []), ('metrics', 'state', ['3']), ('report', None, ['2', '4'])]


def test_resume_skips_completed_inference_but_audits_in_cpu_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'load_plan', lambda _: plan())
    monkeypatch.setattr(C, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(L, 'no_active', lambda _: None)
    monkeypatch.setattr(M, 'admission_record', lambda *_: C.seal({'plan_sha256': 'plan'}))
    calls = []
    monkeypatch.setattr(L, 'submit', lambda o,p,phase,key=None,dependencies=(): calls.append((phase,key)) or str(len(calls)))
    for key in C.SELECTED:
        (tmp_path/key).mkdir()
        (tmp_path/key/'inference_complete.json').write_text('{}')
    L.campaign_submit(SimpleNamespace(output=tmp_path, command='resume', reviewed=True))
    assert calls == [('metrics', 'part'), ('metrics', 'state'), ('report', None)]


def test_prepare_submission_stops_at_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'load_plan', lambda _: plan())
    monkeypatch.setattr(C, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(L, 'no_active', lambda _: None)
    calls = []
    monkeypatch.setattr(L, 'submit', lambda o,p,phase,key=None,dependencies=(): calls.append((phase,key,list(dependencies))) or str(len(calls)))
    L.campaign_submit(SimpleNamespace(output=tmp_path, command='prepare-submit'))
    assert calls == [('prepare', None, []), ('parity', 'part', ['1']), ('parity', 'state', ['1'])]


def chunk(tmp_path):
    path = tmp_path/'file_000.npz'
    scores, labels = np.zeros((10, 10), np.float32), np.zeros(10, np.int64)
    contract = dict(entry_stop=10, file={'label': 0})
    M.atomic_npz(path, logits=scores, labels=labels)
    receipt = C.seal(dict(contract=contract, sha256=C.sha(path),
        labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        precision=dict(passed=True, expected_activation_dtype='torch.float32')))
    C.write(path.with_suffix('.json'), receipt)
    return path, contract


def test_resume_reuses_verified_committed_chunk(tmp_path):
    path, contract = chunk(tmp_path)
    before = path.stat().st_mtime_ns
    logits, labels, _ = R.read_chunk(path, contract)
    assert logits.shape == (10, 10) and len(labels) == 10
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize('change', ['payload', 'contract', 'sidecar', 'missing_payload', 'precision'])
def test_resume_fails_closed(tmp_path, change):
    path, contract = chunk(tmp_path)
    if change == 'payload': path.write_bytes(b'corrupt')
    if change == 'contract': contract = dict(contract, model='state')
    if change == 'sidecar': path.with_suffix('.json').write_text('{}')
    if change == 'missing_payload': path.unlink()
    if change == 'precision':
        row = C.checked(path.with_suffix('.json'))
        del row['content_hash']
        row['precision']['passed'] = False
        C.write(path.with_suffix('.json'), C.seal(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        R.read_chunk(path, contract)


def test_metrics_match_full_population_sklearn_and_ties():
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score
    labels = np.repeat(np.arange(10, dtype=np.int64), 30)
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(300, 10)).astype(np.float32)
    result = M.calculate(logits, labels)
    probabilities = softmax(logits.astype(np.float64), axis=1)
    for mode in ('ovo', 'ovr'):
        assert result['macro_auc_'+mode] == roc_auc_score(labels, probabilities, multi_class=mode, average='macro')
    assert set(result['qcd_rejection']['Hqql']) == {'0.3', '0.5', '0.99'}
    assert set(result['qcd_rejection']['Tbl']) == {'0.3', '0.5', '0.995'}
    tied = M.calculate(np.zeros_like(logits), labels)
    assert tied['macro_auc_ovo'] == .5
    assert tied['qcd_rejection']['Hbb']['0.5']['achieved_signal_efficiency'] == 1
    assert tied['qcd_rejection']['Hbb']['0.5']['rejection'] == 1


def test_zero_background_not_infinity_and_report_percent():
    y = np.repeat(np.arange(10, dtype=np.int64), 10)
    x = np.eye(10, dtype=np.float32)[y]*10
    values = M.calculate(x, y)
    assert values['qcd_rejection']['Hbb']['0.5']['rejection'] is None
    rows = {k: dict(metrics=values, best_check=C.SELECTED[k], parameters=C.COUNTS[k]) for k in C.SELECTED}
    text, changes = M.table(rows)
    assert 'ZB' in text and changes['Hbb']['0.5'] is None
    assert 'Single-seed' in text and 'SyncBatchNorm' in text
    import copy
    rows = {key: copy.deepcopy(row) for key, row in rows.items()}
    rows['part']['metrics']['qcd_rejection']['Hbb']['0.5']['rejection'] = 100.
    rows['state']['metrics']['qcd_rejection']['Hbb']['0.5']['rejection'] = 125.
    _, changes = M.table(rows)
    assert changes['Hbb']['0.5'] == 25.


@pytest.mark.parametrize('key', ['part', 'state'])
def test_actual_architecture_weights_only_roundtrip_cpu(tmp_path, key):
    import torch
    torch.set_num_threads(2)
    a = R.build_architecture(key).eval()
    assert sum(p.numel() for p in a.parameters()) == C.COUNTS[key]
    if key == 'state':
        assert any(isinstance(m, torch.nn.SyncBatchNorm) for m in a.modules())
        from part_ev_state_compare.model import state_branches
        assert len(state_branches(a)) == 8
        with torch.no_grad():
            for branch in state_branches(a): branch.output.normal_(std=.01)
    path = tmp_path/'weights.pt'
    torch.save({'model': a.state_dict()}, path)
    b = R.build_architecture(key).eval()
    b.load_state_dict(torch.load(path, weights_only=True)['model'], strict=True)
    spatial = torch.randn(2, 3, 8)
    vectors = torch.cat((spatial, spatial.square().sum(1, keepdim=True).sqrt()+1), 1)
    values = [torch.zeros(2, 2, 8), torch.randn(2, 17, 8), vectors, torch.ones(2, 1, 8, dtype=torch.bool)]
    with torch.no_grad():
        assert torch.equal(a(*values), b(*values))


def plan():
    return dict(python='/scratch/env/bin/python', partition='gpu', cpu_partition='batch',
                account=None, gres='gpu:nvidia_rtx_a6000:1', evaluate_hours=24)


@pytest.mark.parametrize('phase', ['prepare', 'parity', 'evaluate', 'metrics', 'report'])
def test_scheduler_resources_and_dependencies(tmp_path, phase):
    args = L.request(tmp_path, plan(), phase, 'part' if phase != 'report' else None, ['123'])
    assert '--dependency=afterok:123' in args and '--kill-on-invalid-dep=yes' in args
    assert ('--gres=gpu:nvidia_rtx_a6000:1' in args) == (phase in ('parity', 'evaluate'))
    assert '--time=24:00:00' in args if phase == 'evaluate' else '--time=24:00:00' not in args
    assert '--tool-root' in args and str(tmp_path/'source') in args


def test_live_jobs_block_duplicate_submission(tmp_path, monkeypatch):
    C.write(tmp_path/'jobs.json', [dict(job='123', phase='parity', model='part')])
    monkeypatch.setattr(L, 'command', lambda _: '123\n456')
    with pytest.raises(ValueError, match='active'): L.no_active(tmp_path)


def test_tool_snapshot_is_independent_of_later_repo_changes(tmp_path):
    source, destination = tmp_path/'repo', tmp_path/'frozen'
    for name in C.TOOLS:
        p = source/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    pins = L.snapshot(destination, source)
    (source/C.TOOLS[0]).write_text('changed')
    C.verify_files(destination, pins)
    (destination/C.TOOLS[0]).write_text('tampered')
    with pytest.raises(ValueError): C.verify_files(destination, pins)


def test_export_to_relocated_bundle_and_independent_plan(tmp_path, monkeypatch):
    """Exercise the handoff with synthetic models/probes; GPU donor audit is mocked."""
    import torch
    donors = {}
    for key in C.SELECTED:
        root = tmp_path/('donor_'+key)
        source = root/'source'
        source.mkdir(parents=True)
        (source/'model.py').write_text('# original frozen source\n')
        (root/'source.sha256').write_text(C.sha(source/'model.py')+'  model.py\n')
        C.write(root/'campaign.json', {'expected_gpu': {'nccl_environment': {}}})
        (root/'original_weights.pt').write_bytes(b'original checkpoint for '+key.encode())
        files = []
        for prefix in C.PREFIXES:
            path = root/(prefix+'_120.root')
            path.write_bytes(prefix.encode())
            files.append(E.record(path, 100000))
        donors[key] = dict(key=key, root=str(root), python=sys.executable, files=files,
            task={'architecture': 'part' if key == 'part' else 'state_layerwise'},
            pins={'original_weights.pt': C.sha(root/'original_weights.pt')},
            best_check=C.SELECTED[key], checkpoint_sha256=C.sha(root/'original_weights.pt'),
            campaign_sha256='original_'+key, training_runtime={'world_size': 1 if key == 'part' else 2})
    originals = {k: C.tree_hashes(d['root']) for k, d in donors.items()}
    support = SimpleNamespace(refuse_active=lambda *_: None,
        verify_pins=lambda d: C.verify_files(d['root'], d['pins']), verify_frozen_source=lambda *_: None)
    monkeypatch.setattr(E, 'donor_support', lambda: support)
    monkeypatch.setattr(C, 'lock', lambda _: nullcontext())
    monkeypatch.setattr(E, 'environment', lambda: {'python': '3.10.21', 'packages': {}})
    def context_probe(original, output):
        output.write_bytes(Path(original).read_bytes())
        return dict(policy='weaver_file_wrap_prefix_v1', target_entries=512, padded_length=128,
                    raw_entries=1024, original_entries=100000, flattened_particles=65536)
    monkeypatch.setattr(E, 'write_context_probe', context_probe)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'mock export GPU')

    class Model(torch.nn.Module):
        def __init__(self, key):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(C.COUNTS[key]))
            self.register_buffer('running_mean', torch.ones(3))
        def forward(self, *values):
            return torch.zeros(len(values[0]), 10)

    def finite_loader(inv, rec):
        label = C.PREFIXES[rec['name'].removesuffix('_120.root')]
        return [({name: torch.ones(512, 1, 2) for name in C.INPUTS},
                 {'_label_': torch.full((512,), label, dtype=torch.int64)})]
    D = SimpleNamespace(atomic_torch_save=torch.save, finite_validation_loader=finite_loader,
        tensors=lambda batch, device: ([batch[0][name] for name in C.INPUTS], batch[1]['_label_']),
        model_precision=lambda *_args, **_kwargs: nullcontext(),
        audit_precision=lambda *_args, **_kwargs: nullcontext({'passed': True}), atomic_npz=M.atomic_npz)
    def load_runtime(donor):
        return D, {'inventory': {}}, Model(donor['key']), 'cpu', {}, lambda: None
    monkeypatch.setitem(sys.modules, 'run_part_ev_rejection_eval', SimpleNamespace(load_runtime=load_runtime))

    output = tmp_path/'export'
    tools = L.snapshot(output/'source')
    C.write(output/'export_plan.json', C.seal({'donors': donors, 'tools': tools}))
    for key in C.SELECTED:
        E.export_model(output, key)
    E.finalize(output)
    assert C.load_bundle(output)['models'].keys() == C.SELECTED.keys()
    assert {k: C.tree_hashes(d['root']) for k, d in donors.items()} == originals
    for key in C.SELECTED:
        weights = torch.load(output/key/'weights.pt', weights_only=True)['model']
        assert torch.equal(weights['running_mean'], torch.ones(3))
        assert len(C.checked(output/key/'model.json')['probes']) == 10

    scratch = tmp_path/'scratch'
    relocated = scratch/'models/pair'
    shutil.copytree(output, relocated)
    data, archive = scratch/'data/test_20M', scratch/'test.tar'
    data.mkdir(parents=True)
    archive.write_bytes(b'synthetic; full byte verification happens in prepare')
    python = scratch/'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'fixture interpreter')
    target = scratch/'results/test'
    args = SimpleNamespace(scratch_root=scratch, output=target, bundle=relocated,
        python=python, batch_size=512, evaluate_hours=24, test_dir=data, archive=archive,
        partition='gpu', cpu_partition='batch', gres='gpu:nvidia_rtx_a6000:1', account=None)
    L.init(args)
    assert C.load_plan(target)['bundle'] == str(relocated.resolve())
    assert not (target/'jobs.json').exists()
    # The transferable bundle's actual artifacts, not the original paths, are audited.
    (relocated/'part/probes/HToBB_120.root').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Artifact changed'):
        C.load_plan(target)


def test_setup_env_refuses_home_or_existing_environment(tmp_path):
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    outside = tmp_path/'home'
    outside.mkdir()
    args = SimpleNamespace(scratch_root=scratch, bundle=tmp_path, venv=outside/'env')
    original = C.load_bundle
    try:
        C.load_bundle = lambda _: {}
        with pytest.raises(ValueError, match='NEW venv'): L.setup_env(args)
    finally:
        C.load_bundle = original


@pytest.mark.parametrize('phase', ['evaluate', 'export-cpu'])
def test_worker_relocated_and_cache_paths(tmp_path, phase):
    from tests.test_part_fullscale_launcher import BASH, shell_path
    if not BASH: pytest.skip('Bash unavailable')
    spool = tmp_path/'spool/slurm_script'
    spool.parent.mkdir()
    shutil.copyfile(L.REPO/'sbatch/run_part_ev_oscar.sh', spool)
    output = tmp_path/'scratch output'
    tool = output/'source/scripts/part_ev_oscar.py'
    tool.parent.mkdir(parents=True)
    tool.write_text('# fake driver\n')
    # Native fixture creation also works in Windows sandboxes where MSYS cannot
    # create directories in the host's temp directory. The worker still checks
    # and exports these exact scratch locations.
    for name in ('tmp', 'cache/pip', 'cache/torch', 'cache/cuda', 'cache/extensions', 'cache/matplotlib'):
        (output/name).mkdir(parents=True, exist_ok=True)
    binary = tmp_path/'bin/python'
    binary.parent.mkdir()
    binary.write_text('#!/usr/bin/env bash\nset -eu\n[[ "$1" == -B && -f "$2" ]]\nprintf "CUDA_VISIBLE_DEVICES=<%s>\\n" "${CUDA_VISIBLE_DEVICES-unset}"\nprintf "%s\\n" "$@" "$TMPDIR" "$PYTHONNOUSERSITE" "$PYTHONDONTWRITEBYTECODE"\n', newline='\n')
    binary.chmod(0o755)
    result = subprocess.run([BASH, '--noprofile', '--norc', '-c',
                'export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=0; exec bash "$@"', 'fixture',
                shell_path(spool), '--python', shell_path(binary),
                '--tool-root', shell_path(output/'source'), '--output', shell_path(output),
                '--command', phase, '--model', 'state'], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert ('CUDA_VISIBLE_DEVICES=<>' if phase == 'export-cpu' else 'CUDA_VISIBLE_DEVICES=<0>') in result.stdout
    assert shell_path(tool) in result.stdout and shell_path(output/'tmp') in result.stdout
    assert result.stdout.endswith('1\n1\n')
