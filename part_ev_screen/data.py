"""Reuse the reference training streams, but explicitly evaluate validation."""
from __future__ import annotations

import hashlib
import torch
from torch.utils.data import DataLoader
from part_ev_fullscale.data import (build_loader, replay_batches, shutdown_loader,
    _SeededDataset, _stream_seed, _validated_config, _file_entry_counts, rank_file_dict)


def finite_validation_loader(inventory, file_record, batch_size=512):
    from weaver.utils.dataset import SimpleIterDataset
    selected = {"splits": {"val": {"files": [file_record]}}}
    files = rank_file_dict("val", selected)
    options = dict(file_dict=files, data_config_file=str(_validated_config(None)),
        for_training=False, load_range_and_fraction=((0, 1), 1), fetch_by_files=True,
        fetch_step=1, file_fraction=1, remake_weights=False, async_load=False,
        infinity_mode=False, in_memory=False, name="screen_finite_validation")
    dataset = SimpleIterDataset(**options)
    seed = _stream_seed(101, "val", 0, 0)
    entries = _file_entry_counts(inventory, "val", files)
    return DataLoader(_SeededDataset(dataset, options, entries, seed, False),
        batch_size=batch_size, drop_last=False, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed))


def batch_fingerprint(batch):
    h = hashlib.sha256()
    for mapping in batch[:2]:
        for key in sorted(mapping):
            value = mapping[key].detach().cpu().contiguous().numpy()
            h.update(key.encode())
            h.update(str((value.shape, value.dtype)).encode())
            h.update(value.tobytes())
    return h.hexdigest()
