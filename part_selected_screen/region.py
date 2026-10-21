"""Historical angular trees, compact CPU packing and batched REGION features.

No persistent 100M-jet tree/pair cache. Production requires the authenticated
compiled backend; the Python oracle is explicitly for local tests only.
"""
from pathlib import Path
import numpy as np
import torch
from teacher_logit_reco.relational_part.ca_tree import load_tree_backend, build_compiled_tree
from teacher_logit_reco.relational_part.region_tree import build_reference_tree, validate_tree
from teacher_logit_reco.relational_part.relation_region import _node_values

SOURCE = Path(__file__).resolve().parents[1] / 'teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp'


def lowest_common_ancestors(p):
    """Lift by the SET BITS of depth difference, not by every smaller power.

    The legacy vectorized helper used difference >= 2**level, which over-lifts
    differences such as 2 or 4. Keep historical files immutable; this screen
    implements the intended LCA and tests it against a direct parent walk.
    """
    leaves, parent, depth = p['leaf_to_node'], p['parent'], p['depth']
    jumps = [parent]
    for _ in range(1, max(1, (parent.shape[1]-1).bit_length())):
        jumps.append(_node_values(jumps[-1], jumps[-1]))
    left = leaves[:, :, None].expand(-1, -1, leaves.shape[1]).clone()
    right = leaves[:, None, :].expand(-1, leaves.shape[1], -1).clone()
    difference = _node_values(depth, left)-_node_values(depth, right)
    for level, jump in enumerate(jumps):
        bit = difference.abs().bitwise_and(1 << level).ne(0)
        left = torch.where(bit & (difference > 0), _node_values(jump, left), left)
        right = torch.where(bit & (difference < 0), _node_values(jump, right), right)
    for jump in reversed(jumps):
        a, b = _node_values(jump, left), _node_values(jump, right)
        different = a.ne(b)
        left, right = torch.where(different, a, left), torch.where(different, b, right)
    return torch.where(left.eq(right), left, _node_values(parent, left))


def load_backend(directory):
    directory = Path(directory)
    manifest = directory / 'backend_manifest.json'
    import json
    row = json.loads(manifest.read_text())
    # The builder records the authenticated filename; do not guess an ABI suffix.
    binaries = list(directory.glob('relational_ca_tree_v1_ext*.so'))
    if len(binaries) != 1 or binaries[0].name != row['binary_filename']:
        raise ValueError('Expected exactly one compiled REGION backend')
    return load_tree_backend(binaries[0], manifest, source_path=SOURCE)


def trees_for(vectors, raw, mask, backend):
    v = vectors.detach().float().transpose(1, 2).cpu().numpy()
    r = raw.detach().float().cpu().numpy()
    m = mask[:, 0].detach().bool().cpu().numpy()
    if backend == 'reference_for_tests':
        return [build_reference_tree(a, b, c) for a, b, c in zip(v, r, m)]
    if backend is None:
        raise ValueError('Production requires a compiled tree backend; no silent fallback')
    return [build_compiled_tree(backend, a, b, c) for a, b, c in zip(v, r, m)]


def pack(trees, particles, device):
    batch, nodes = len(trees), max(1, 2 * particles - 1)
    result = {name: np.zeros((batch, nodes), np.int64) for name in ('parent', 'depth', 'multiplicity')}
    result.update({name: np.zeros((batch, nodes), np.float32)
        for name in ('pt', 'mass', 'merge_delta_r', 'merge_kt', 'merge_z', 'merge_mass')})
    result.update(vectors=np.zeros((batch, nodes, 4), np.float32),
        leaf_to_node=np.zeros((batch, particles), np.int64),
        assignments=np.zeros((batch, 3, particles), np.int64),
        roots=np.zeros(batch, np.int64), n_valid=np.zeros(batch, np.int64),
        leaf_ranks=np.zeros((batch, 3, particles), np.float32))
    for i, tree in enumerate(trees):
        validate_tree(tree)
        if tree['n_particles'] != particles:
            raise ValueError('Tree and trimmed particle count differ')
        count = tree['n_nodes']
        result['n_valid'][i] = tree['n_valid']
        if not count:
            continue
        for name in ('parent', 'depth', 'multiplicity', 'pt', 'mass', 'merge_delta_r',
                     'merge_kt', 'merge_z', 'merge_mass', 'vectors'):
            result[name][i, :count] = tree[name]
        root = tree['root']
        result['roots'][i] = root
        result['parent'][i, root] = root
        result['leaf_to_node'][i] = np.maximum(tree['leaf_to_node'], 0)
        valid = tree['leaf_to_node'] >= 0
        for k, resolution in enumerate((2, 4, 8)):
            assignments = np.maximum(tree['assignments'][str(resolution)], 0)
            result['assignments'][i, k] = assignments
            cluster_nodes = np.unique(assignments[valid])
            pt = tree['pt'][cluster_nodes]
            ranks = ((pt[None, :] > pt[:, None]).sum(-1)
                + .5 * ((pt[None, :] == pt[:, None]).sum(-1) - 1)) / max(1, len(pt)-1)
            lookup = np.zeros(nodes, np.float32)
            lookup[cluster_nodes] = ranks
            result['leaf_ranks'][i, k] = lookup[assignments]
    # One host-to-device copy per compact field, not thousands of small GPU copies.
    return {k: torch.from_numpy(v).to(device) for k, v in result.items()}


def region_features(trees, raw, mask):
    p = pack(trees, raw.shape[1], raw.device)
    lca = lowest_common_ancestors(p)
    b, n = raw.shape[:2]
    valid = mask[:, 0].bool()
    jet_pt = _node_values(p['pt'], p['roots']).view(b, 1, 1)
    jet_mass = _node_values(p['mass'], p['roots']).view(b, 1, 1)
    max_depth = _node_values(p['depth'], p['leaf_to_node']).masked_fill(~valid, 0).amax(1).clamp_min(1).view(b, 1, 1)
    channels = [a[:, :, None].eq(a[:, None, :]).float() for a in p['assignments'].unbind(1)]
    channels.append(_node_values(p['depth'], lca).float()/max_depth)
    diagonal = torch.eye(n, dtype=torch.bool, device=raw.device)[None]
    channels.extend(x.masked_fill(diagonal, 0) for x in (
        (_node_values(p['merge_delta_r'], lca)/.8 + 1e-6).log(),
        (_node_values(p['merge_kt'], lca)/(jet_pt*.8 + 1e-6) + 1e-6).log(),
        _node_values(p['merge_z'], lca),
        ((_node_values(p['merge_mass'], lca)+1e-6)/(jet_mass+1e-6)).log()))
    def endpoints(x):
        return (x[:, :, None].expand(-1, -1, n), x[:, None, :].expand(-1, n, -1))
    for nodes in p['assignments'].unbind(1):
        descriptors = (
            ((_node_values(p['pt'], nodes)+1e-6)/(jet_pt[:, :, 0]+1e-6)).log(),
            ((_node_values(p['mass'], nodes)+1e-6)/(jet_mass[:, :, 0]+1e-6)).log(),
            _node_values(p['multiplicity'], nodes).float()/p['n_valid'].clamp_min(1)[:, None])
        for endpoint in (0, 1):
            channels.extend(endpoints(x)[endpoint] for x in descriptors)
    for nodes in p['assignments'].unbind(1):
        channels.extend(endpoints(raw[:, :, 0]/(_node_values(p['pt'], nodes)+1e-6)))
    for nodes in p['assignments'].unbind(1):
        v = _node_values(p['vectors'], nodes)
        eta = torch.asinh(v[:, :, 2]/torch.hypot(v[:, :, 0], v[:, :, 1]).clamp_min(1e-30))
        dphi = raw[:, :, 2]-torch.atan2(v[:, :, 1], v[:, :, 0])
        dphi = torch.atan2(dphi.sin(), dphi.cos())
        channels.extend(endpoints(torch.hypot(raw[:, :, 1]-eta, dphi)/.8))
    channels.extend(r[:, None, :]-r[:, :, None] for r in p['leaf_ranks'].unbind(1))
    if len(channels) != 41:
        raise AssertionError('REGION channel schema drifted')
    return torch.stack(channels, 1).masked_fill(~(valid[:, None, :, None] & valid[:, None, None, :]), 0)
