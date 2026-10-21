"""Fit only on 50k training-stream examples; never inspect validation labels."""
import hashlib
import numpy as np
import torch
from part_screen_common.contracts import hashed, checked_json
from part_ev_fullscale.protocol import write_json
from teacher_logit_reco.relational_part.relation_track import track_validity
from .data import build_loader, shutdown_loader, raw_tokens, batch_fingerprint
from .features import CHANNELS, raw_features

SCHEMA = 'part_selected_normalization_v1'
SETTINGS = dict(jets=50000, pairs_per_jet=64, seed=101,
    sampling='first_50000_examples_of_unchanged_seed101_training_stream',
    pair_sampling='PCG64_771903_without_replacement_in_stream_order',
    uncertainty_floor='first_percentile_valid_positive_errors_in_same_training_subset',
    quantiles='linear', scale='max(IQR/1.349,1e-6)', clip=[-8, 8])


def identity_normalization():
    """Synthetic constants for unit tests / parameter counting, never production."""
    return dict(floors=dict(d0=1e-6, dz=1e-6), families={k: dict(names=list(names),
        robust=list(robust), center=[0.]*len(names), scale=[1.]*len(names))
        for k, (names, robust) in CHANNELS.items()})


def fit(output, donor, backend):
    path = output / 'inputs/normalization.json'
    if path.exists():
        row = checked_json(path, SCHEMA)
        if row['settings'] != SETTINGS or row['inventory_sha256'] != donor['inventory']['inventory_sha256']:
            raise ValueError('Training normalization belongs to another population')
        return row
    loader, _ = build_loader('train', donor['inventory'], 512,
        config=output/'inputs/selected.yaml', seed=101)
    features, vectors, masks = [], [], []
    digest = hashlib.sha256()
    count = 0
    try:
        for batch in loader:
            fingerprint = batch_fingerprint(batch)
            if count == 0 and fingerprint != donor['input_fingerprints']['0']:
                raise ValueError('Extended loader changed the donor training stream')
            digest.update(fingerprint.encode())
            take = min(512, SETTINGS['jets']-count)
            features.append(batch[0]['pf_features'][:take].clone())
            vectors.append(batch[0]['pf_vectors'][:take].clone())
            masks.append(batch[0]['pf_mask'][:take].clone())
            count += take
            if count == SETTINGS['jets']:
                break
    finally:
        shutdown_loader(loader)
    f, v, m = map(torch.cat, (features, vectors, masks))
    del features, vectors, masks
    r = raw_tokens(f, v, m)
    valid = track_validity(r, m)
    floors = {}
    for name, index in (('d0', 11), ('dz', 13)):
        values = r[:, :, index][valid].numpy()
        if not len(values):
            raise ValueError('No valid training tracks for uncertainty floors')
        floors[name] = max(1e-6, float(np.quantile(values, .01, method='linear')))
    samples = {k: [] for k in CHANNELS}
    rng = np.random.Generator(np.random.PCG64(771903))
    for start in range(0, count, 16):
        vv, rr, mm = v[start:start+16], r[start:start+16], m[start:start+16]
        vals, track = raw_features(vv, rr, mm, floors, backend)
        for i in range(len(vv)):
            indices = torch.where(mm[i, 0].bool())[0].numpy()
            pairs = np.array(np.meshgrid(indices, indices, indexing='ij')).reshape(2, -1).T
            if len(pairs) > 64:
                pairs = pairs[rng.choice(len(pairs), 64, replace=False)]
            for family in ('PT', 'REGION'):
                if len(pairs):
                    samples[family].append(vals[family][i, :, pairs[:, 0], pairs[:, 1]].T.numpy())
            if len(pairs):
                applicable = track['both_tracks_valid'][i, 0, pairs[:, 0], pairs[:, 1]].numpy()
                kept = pairs[applicable]
                if len(kept):
                    samples['TRACK_PAIR'].append(vals['TRACK_PAIR'][i, :, kept[:, 0], kept[:, 1]].T.numpy())
            nodes = track['track_valid'][i]
            if nodes.any():
                samples['TRACK_NODE'].append(vals['TRACK_NODE'][i, :, nodes].T.numpy())
        if start % 1024 == 0:
            print(f'Normalization: {start}/{count} training examples', flush=True)
    families = {}
    for key, (names, robust) in CHANNELS.items():
        if not samples[key]:
            raise ValueError(f'No applicable normalization samples for {key}')
        values = np.concatenate(samples[key])
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite training normalization values: {key}')
        q25, median, q75 = np.quantile(values, [.25, .5, .75], axis=0, method='linear')
        center = [float(median[i]) if name in robust else 0. for i, name in enumerate(names)]
        scale = [max(1e-6, float((q75[i]-q25[i])/1.349)) if name in robust else 1. for i, name in enumerate(names)]
        families[key] = dict(names=list(names), robust=list(robust), center=center, scale=scale, samples=len(values))
    row = hashed(dict(schema=SCHEMA, settings=SETTINGS, population='training_only',
        inventory_sha256=donor['inventory']['inventory_sha256'], stream_sha256=digest.hexdigest(),
        floors=floors, families=families))
    write_json(path, row)
    return row
