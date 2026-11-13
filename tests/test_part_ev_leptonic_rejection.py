import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('leptonic', ROOT / 'scripts/inspect_part_ev_leptonic_rejection.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
S = R.S


def test_working_points_match_original_metrics():
    from part_ev_fullscale.metrics import calculate_metrics
    from part_ev_fullscale.protocol import CLASS_NAMES
    assert tuple(CLASS_NAMES) == R.CLASS_NAMES
    rng = np.random.default_rng(15)
    logits = rng.normal(size=(4000, 10)).astype(np.float32)
    y = np.repeat(np.arange(10), 400)
    expected = calculate_metrics(logits, y)
    for name, efficiency in R.TARGETS.items():
        idx = CLASS_NAMES.index(name)
        score = logits[:, idx].astype(np.float64) - logits[:, 0].astype(np.float64)
        assert R.working_point(score[y == idx], score[y == 0], efficiency) == expected['qcd_rejection'][name][str(efficiency)]


def test_zero_and_ties():
    result = R.working_point(np.ones(1000), np.zeros(1000), .995)
    assert result['rejection'] is None and result['qcd_accepted'] == 0
    assert result['achieved_signal_efficiency'] == 1
    result = R.working_point(np.zeros(1000), np.zeros(1000), .99)
    assert result['rejection'] == 1


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    # Same production file/manifest logic, smaller synthetic file size only.
    monkeypatch.setattr(R, 'ENTRIES_PER_FILE', 20)
    names = R.CLASS_NAMES
    files = [dict(name=f'file{i}.root', num_entries=20) for i in range(50)]
    donors = {k: dict(key=k, task={'seed': 101, 'run_id': k}, checkpoint_sha256=k+'checkpoint',
                      campaign_sha256=k+'campaign', best_check=46 if k == 'part' else 49,
                      files=files) for k in ('part', 'state')}
    source = {}
    for name in S.SOURCES:
        dest = tmp_path / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
        source[name] = S.sha(dest)
    plan = S.seal(dict(schema='part_ev_rejection_eval_v1', donors=donors, source=source))
    S.write(tmp_path / 'plan.json', plan)
    rng = np.random.default_rng(15)
    for key, donor in donors.items():
        directory = tmp_path / key
        directory.mkdir()
        records = []
        for i, file in enumerate(files):
            logits = rng.normal(size=(20, 10)).astype(np.float32)
            y = np.tile(np.arange(10), 2)
            path = directory / f'file_{i:03d}.npz'
            np.savez(path, logits=logits, labels=y)
            row = dict(contract=R.chunk_contract(plan, donor, file), sha256=S.sha(path),
                labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                precision=dict(passed=True, expected_activation_dtype='torch.float32'))
            S.write(path.with_suffix('.json'), row)
            records.append(row)
        S.write(directory / 'metrics.json', S.seal(dict(model=key, task=donor['task'],
            plan_sha256=plan['content_hash'], campaign_sha256=donor['campaign_sha256'],
            checkpoint_sha256=donor['checkpoint_sha256'], best_check=donor['best_check'],
            population='official_validation_5M', independent_test=False,
            metrics=dict(events=1000, class_support={n: 100 for n in names}), identity_manifest=records)))
    return tmp_path


def test_read_only_complete_comparison(evaluation):
    before = {str(p): S.sha(p) for p in evaluation.rglob('*') if p.is_file()}
    text = R.comparison(evaluation)
    assert 'Hlvqq (Hqql) | 99%' in text
    assert 'Tblv (Tbl) | 99.5%' in text
    assert 'Change (%)' in text and 'Read-only CPU' in text
    assert before == {str(p): S.sha(p) for p in evaluation.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad', ['predictions', 'checkpoint', 'population', 'missing_file', 'receipt'])
def test_changed_artifacts_fail_closed(evaluation, bad):
    key = evaluation / 'state'
    if bad == 'predictions':
        with (key / 'file_000.npz').open('ab') as f:
            f.write(b'changed')
    elif bad == 'missing_file':
        (key / 'file_000.npz').unlink()
    elif bad == 'receipt':
        path = key / 'file_000.json'
        row = S.read(path)
        row['labels_sha256'] = 'changed'
        S.write(path, row)
    else:
        path = key / 'metrics.json'
        row = S.read(path)
        row.pop('content_hash')
        if bad == 'checkpoint': row['checkpoint_sha256'] = 'wrong'
        else: row['metrics']['events'] -= 1
        S.write(path, S.seal(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        R.comparison(evaluation)


def test_no_heavy_model_imports_in_fresh_cli():
    import subprocess
    code = f"import runpy,sys; runpy.run_path({str(ROOT / 'scripts/inspect_part_ev_leptonic_rejection.py')!r},run_name='inspect_only'); assert 'torch' not in sys.modules; assert 'uproot' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
