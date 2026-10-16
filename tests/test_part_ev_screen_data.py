import numpy as np
import torch
from part_ev_screen.data import finite_validation_loader, batch_fingerprint, build_loader, shutdown_loader
from tests.test_part_fullscale_data import write_root, fixture_inventory


def test_finite_validation_is_ordered_complete_and_retains_partial_batch(tmp_path):
    path = tmp_path/'val_5M/HToBB_120.root'
    write_root(path, events=23)
    inventory = fixture_inventory('val',[path])
    loader = finite_validation_loader(inventory,inventory['splits']['val']['files'][0],batch_size=7)
    batches = list(loader)
    assert [len(b[1]['_label_']) for b in batches] == [7,7,7,2]
    np.testing.assert_array_equal(torch.cat([b[1]['_label_'] for b in batches]), np.arange(23)%10)
    again = list(finite_validation_loader(inventory,inventory['splits']['val']['files'][0],batch_size=7))
    assert [batch_fingerprint(b) for b in batches] == [batch_fingerprint(b) for b in again]
    assert 'test' not in inventory['splits']


def test_stream_matches_across_model_rng_and_replays(tmp_path):
    path = tmp_path/'val_5M/HToBB_120.root'
    write_root(path,events=60)
    inventory = fixture_inventory('val',[path])
    a,_ = build_loader('val',inventory,4,workers=0,seed=101)
    b,_ = build_loader('val',inventory,4,workers=0,seed=101)
    try:
        ai,bi = iter(a),iter(b)
        expected = [batch_fingerprint(next(ai)) for _ in range(7)]
        for _ in range(3): next(bi)
        torch.manual_seed(999)
        assert [batch_fingerprint(next(bi)) for _ in range(4)] == expected[3:]
    finally:
        shutdown_loader(a); shutdown_loader(b)
