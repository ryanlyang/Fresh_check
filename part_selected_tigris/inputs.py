"""Native ARM stream reference: no GPU, training, or donor mutation.

ARM and x86 floating point preprocessing need not be byte-identical. The
extended selected loader must instead exactly reproduce the *native* loader
on this platform, at the same five training-stream positions as the screen.
Cross-platform hashes are retained for disclosure, not silently substituted.
"""
from torch.utils.data import DataLoader
from part_ev_screen.data import build_loader, shutdown_loader, batch_fingerprint
from part_selected_screen.data import _BatchPrefix
from part_ev_fullscale.checkpoint import seed_everything
from . import protocol as P

INDICES = (0, 20, 1020, 2020, 4020)


def native_prefix(inventory, *, batch_size=512):
    original, _ = build_loader('train', inventory, batch_size, seed=101)
    # Preserve worker seeds, sharding, batching and round-robin order; let the
    # finite prefix exhaust naturally instead of interrupting infinite reads.
    return DataLoader(_BatchPrefix(original.dataset, batch_size, max(INDICES)+1),
        batch_size=batch_size, drop_last=True, pin_memory=original.pin_memory,
        num_workers=2, persistent_workers=False, multiprocessing_context='spawn',
        worker_init_fn=original.worker_init_fn, generator=original.generator)


def prepare_reference(output, donor, environment):
    path = output/'inputs/native_stream.json'
    if path.exists():
        row = P.checked_json(path, 'part_selected_tigris_inputs_v1')
        if (row['environment'] != environment
                or row['inventory_sha256'] != donor['inventory']['inventory_sha256']
                or row['donor_fingerprints'] != donor['input_fingerprints']
                or set(row['fingerprints']) != {str(i) for i in INDICES}):
            raise ValueError('Prepared native input reference changed')
        return row
    loader = native_prefix(donor['inventory'])
    fingerprints = {}
    seed_everything(101)
    count = 0
    try:
        for index, batch in enumerate(loader):
            count += 1
            if index in INDICES:
                fingerprints[str(index)] = batch_fingerprint(batch)
                print(f'Native Tigris input reference: batch {index}', flush=True)
        if count != max(INDICES)+1:
            raise ValueError('Native input reference did not exhaust the expected prefix')
    finally:
        shutdown_loader(loader)
    row = P.hashed(dict(schema='part_selected_tigris_inputs_v1',
        inventory_sha256=donor['inventory']['inventory_sha256'], environment=environment,
        fingerprints=fingerprints, donor_fingerprints=donor['input_fingerprints'],
        byte_identical_to_sporc=fingerprints == donor['input_fingerprints'],
        reference='unchanged_native_ParT_loader_on_Tigris_no_model_training'))
    P.write_json(path, row)
    return row
