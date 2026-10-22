import copy
import pytest
from part_selected_tigris import inputs as I, protocol as P


def test_reference_uses_native_loader_and_retains_cross_platform_mismatch(tmp_path, monkeypatch):
    donor = dict(inventory={'inventory_sha256': 'same_files'}, input_fingerprints={'0': 'x86_bytes'})
    before = copy.deepcopy(donor)
    environment = {'architecture': 'aarch64'}
    monkeypatch.setattr(I, 'INDICES', (0, 2, 4))
    monkeypatch.setattr(I, 'native_prefix', lambda inv: iter(range(5)))
    monkeypatch.setattr(I, 'batch_fingerprint', lambda batch: f'arm_batch_{batch}')
    monkeypatch.setattr(I, 'shutdown_loader', lambda loader: None)
    result = I.prepare_reference(tmp_path, donor, environment)
    assert result['byte_identical_to_sporc'] is False
    assert result['fingerprints'] == {'0': 'arm_batch_0', '2': 'arm_batch_2', '4': 'arm_batch_4'}
    assert result['donor_fingerprints'] == donor['input_fingerprints']
    assert donor == before
    monkeypatch.setattr(I, 'native_prefix', lambda *a: pytest.fail('must reuse immutable reference'))
    assert I.prepare_reference(tmp_path, donor, environment) == result
    with pytest.raises(ValueError, match='changed'):
        I.prepare_reference(tmp_path, donor, {'architecture': 'different'})


def test_short_native_reference_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(I, 'INDICES', (0, 2, 4))
    monkeypatch.setattr(I, 'native_prefix', lambda inv: iter(range(3)))
    monkeypatch.setattr(I, 'batch_fingerprint', str)
    monkeypatch.setattr(I, 'shutdown_loader', lambda loader: None)
    with pytest.raises(ValueError, match='expected prefix'):
        I.prepare_reference(tmp_path, {'inventory': {}}, {})
    assert not (tmp_path/'inputs/native_stream.json').exists()


def test_native_finite_reference_matches_original_two_worker_stream(tmp_path, monkeypatch):
    from tests.test_part_selected_screen import write_root, fixture_inventory
    from part_ev_fullscale.data import CLASSES
    from part_ev_screen.data import build_loader, shutdown_loader, batch_fingerprint
    files = [tmp_path/f'train_100M/{label}_{index:03d}.root'
             for label in CLASSES for index in (0, 1)]
    for path in files:
        write_root(path, 20)
    inventory = fixture_inventory('train', files)
    monkeypatch.setattr(I, 'INDICES', (0, 2, 4))
    finite = I.native_prefix(inventory, batch_size=4)
    original, _ = build_loader('train', inventory, 4, seed=101)
    try:
        iterator = iter(original)
        expected = [batch_fingerprint(next(iterator)) for _ in range(5)]
        finite_iterator = iter(finite)
        assert [batch_fingerprint(batch) for batch in finite_iterator] == expected
        assert all(not w.is_alive() and w.exitcode == 0 for w in finite_iterator._workers)
    finally:
        shutdown_loader(original)
        shutdown_loader(finite)
