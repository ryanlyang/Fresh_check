"""Native streams with four additional, UNCLIPPED track measurements.

The first 17 feature channels, every other input, labels, sampling, workers and
RNG are unchanged. Aliases prevent Weaver's in-place error clipping from also
clipping the relation inputs. The appended channels travel through the SAME
native trimmer; they never enter the particle embedding.
"""
import copy
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from part_ev_fullscale.data import (DATA_CONFIG, _SeededDataset, _stream_seed,
    _seed_worker, _file_entry_counts, rank_file_dict, shutdown_loader)
from part_ev_screen.data import batch_fingerprint as native_fingerprint

RAW_BRANCHES = ('part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr')


def config_document():
    document = yaml.safe_load(DATA_CONFIG.read_text())
    document = copy.deepcopy(document)
    for i, branch in enumerate(RAW_BRANCHES):
        alias = f'selected_raw_track_{i}'
        document['new_variables'][alias] = f'{branch} * 1.0'
        document['inputs']['pf_features']['vars'].append([alias, None])
    return document


def write_config(path):
    path = Path(path)
    content = yaml.safe_dump(config_document(), sort_keys=False)
    if path.exists() and path.read_text() != content:
        raise ValueError('Selected input configuration changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)


def validate_config(path):
    path = Path(path)
    if yaml.safe_load(path.read_text()) != config_document():
        raise ValueError('Only the four raw track aliases may extend the native YAML')
    if list(path.parent.glob(path.stem + '.*.auto.yaml')):
        raise ValueError('Auto-generated YAML must not override preprocessing')
    return path


def standard_batch(batch):
    inputs = dict(batch[0])
    inputs['pf_features'] = inputs['pf_features'][:, :17].contiguous()
    return (inputs, *batch[1:])


def batch_fingerprint(batch):
    return native_fingerprint(standard_batch(batch))


class _BatchPrefix(IterableDataset):
    """Finite prefix of the SAME round-robin worker/batch stream.

    A per-worker quota in whole batches preserves native file sharding, seeds,
    preprocessing and interleaving. Exhausting this dataset lets every worker
    finish instead of interrupting infinite prefetched reads at shutdown. This
    is used only for normalization collection, never for production training.
    """
    def __init__(self, dataset, batch_size, batches):
        self.dataset, self.batch_size, self.batches = dataset, batch_size, batches

    def __iter__(self):
        worker = get_worker_info()
        workers, index = (worker.num_workers, worker.id) if worker is not None else (1, 0)
        batches = self.batches // workers + int(index < self.batches % workers)
        if batches == 0:
            return
        source = iter(self.dataset)
        for _ in range(batches * self.batch_size):
            try:
                item = next(source)
            except StopIteration as exc:
                raise RuntimeError('Training stream ended before its normalization prefix') from exc
            yield item


def build_loader(split, inventory, batch_size, *, config, workers=2, seed=101, max_batches=None):
    from weaver.utils.dataset import SimpleIterDataset
    if type(batch_size) is not int or batch_size <= 0 or type(workers) is not int or workers < 0:
        raise ValueError('batch_size must be positive and workers nonnegative')
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError('max_batches must be a positive integer or None')
    if split not in ('train', 'val'):
        raise ValueError('This screen never opens a test population')
    config = validate_config(config)
    files = rank_file_dict(split, inventory)
    if workers and any(len(paths) < workers for paths in files.values()):
        raise ValueError('Each file group needs at least one file per worker')
    options = dict(file_dict=files, data_config_file=str(config), for_training=True,
        load_range_and_fraction=((0, 1), 1), fetch_by_files=False, fetch_step=.01,
        file_fraction=1, remake_weights=False, async_load=False, infinity_mode=True,
        in_memory=False, name=f'fullscale_{split}_rank0')
    dataset = SimpleIterDataset(**options)
    if dataset.config._missing_standardization_info or dataset.config.weight_name is not None:
        raise ValueError('Expected manual unweighted preprocessing')
    stream_seed = _stream_seed(seed, split, 0, 0)
    entries = _file_entry_counts(inventory, split, files)
    stream = _SeededDataset(dataset, options, entries, stream_seed, True)
    if max_batches is not None:
        stream = _BatchPrefix(stream, batch_size, max_batches)
    return DataLoader(stream,
        batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=workers,
        persistent_workers=bool(workers and max_batches is None), multiprocessing_context='spawn' if workers else None,
        worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(stream_seed)), dataset.config


def finite_validation_loader(inventory, file_record, batch_size=512, *, config):
    from weaver.utils.dataset import SimpleIterDataset
    selected = {'splits': {'val': {'files': [file_record]}}}
    files = rank_file_dict('val', selected)
    options = dict(file_dict=files, data_config_file=str(validate_config(config)),
        for_training=False, load_range_and_fraction=((0, 1), 1), fetch_by_files=True,
        fetch_step=1, file_fraction=1, remake_weights=False, async_load=False,
        infinity_mode=False, in_memory=False, name='screen_finite_validation')
    dataset = SimpleIterDataset(**options)
    seed = _stream_seed(101, 'val', 0, 0)
    entries = _file_entry_counts(inventory, 'val', files)
    return DataLoader(_SeededDataset(dataset, options, entries, seed, False),
        batch_size=batch_size, drop_last=False, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed))


def raw_tokens(features, vectors, mask):
    """Historical PT, eta, phi, energy, charge, five PID bits, four raw tracks."""
    if features.shape[1] != 21:
        raise ValueError('Selected models require 17 native + 4 raw track channels')
    work = vectors.float()
    pt = torch.hypot(work[:, 0], work[:, 1])
    raw = torch.cat((pt[:, None], torch.asinh(work[:, 2] / pt.clamp_min(1e-30))[:, None],
        torch.atan2(work[:, 1], work[:, 0])[:, None], work[:, 3:4],
        features[:, 5:11].float(), features[:, 17:21].float()), dim=1)
    return raw.transpose(1, 2).contiguous().masked_fill(~mask[:, 0, :, None].bool(), 0)
