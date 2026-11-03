"""Batched CPU trees with the reference GPU REGION arithmetic unchanged."""
import hashlib
import importlib.util
from pathlib import Path
import platform
import sys
import types
import numpy as np
import torch
from part_ev_fullscale.protocol import file_sha256, read_json, write_json
from part_screen_common.contracts import hashed, checked_json
from part_selected_screen import region as reference

NAME = 'selected_tree_batch_v1_ext'
SCHEMA = 'selected_fast_backend_v1'
SOURCE = Path(__file__).parent / 'csrc'
FLAGS = ['-O3', '-std=c++17', '-fopenmp', '-fno-fast-math', '-fno-associative-math', '-ffp-contract=off']


def sources():
    return {p.name: file_sha256(p) for p in sorted(SOURCE.glob('*')) if p.suffix in ('.cpp', '.h')}


def abi():
    return dict(architecture=platform.machine(), python=list(sys.version_info[:2]), torch=torch.__version__, numpy=np.__version__)


def build_backend(directory):
    directory = Path(directory).resolve()
    if (directory/'manifest.json').exists():
        return load_backend(directory)
    if sys.platform != 'linux':
        raise RuntimeError('Build this OpenMP backend on Linux; no silent compiler/algorithm fallback')
    from torch.utils.cpp_extension import load
    build = directory/'build'
    build.mkdir(parents=True, exist_ok=True)
    module = load(NAME, [str(SOURCE/'selected_tree_batch.cpp')], extra_cflags=FLAGS,
                  extra_ldflags=['-fopenmp'], build_directory=str(build), verbose=True)
    binary = Path(module.__file__).resolve()
    if not module.openmp or module.contract != 'selected_tree_batch_v1':
        raise ValueError('Unexpected native backend')
    write_json(directory/'manifest.json', hashed(dict(schema=SCHEMA, sources=sources(), abi=abi(),
        flags=FLAGS, binary=binary.relative_to(directory).as_posix(), binary_sha256=file_sha256(binary))))
    return module


def load_backend(directory):
    directory = Path(directory).resolve()
    row = checked_json(directory/'manifest.json', SCHEMA)
    binary = (directory/row['binary']).resolve(strict=True)
    if (not binary.is_relative_to(directory) or row['sources'] != sources() or row['abi'] != abi()
            or row['flags'] != FLAGS or file_sha256(binary) != row['binary_sha256']):
        raise ValueError('Batched tree backend source/binary/runtime changed')
    spec = importlib.util.spec_from_file_location(NAME, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.openmp or module.contract != 'selected_tree_batch_v1':
        raise ValueError('Unexpected native backend')
    return module


def backend_parity(module, old_backend, threads):
    """CPU pre-admission against the authenticated original compiled backend."""
    rng=np.random.default_rng(81027)
    cases=[]
    for n in (1,2,11,32,64,128):
        v=rng.normal(size=(5,n,4)).astype(np.float32)
        v[:,:,3]=np.sqrt((v[:,:,:3]**2).sum(-1))+1
        raw=rng.normal(size=(5,n,14)).astype(np.float32)
        mask=np.ones((5,n),bool)
        mask[0]=False; mask[1,1:]=False; mask[2,n//2:]=False
        # Repeated particles and exact ties, plus padded nonfinite values.
        v[3]=v[3,:1]; raw[3]=raw[3,:1]
        v[0]=np.nan; raw[0]=np.nan
        expected=oracle_batch(v,raw,mask,old_backend)
        counts=sorted({1,threads,*[c for c in (2,4,8,16) if c<=threads]})
        for count in counts:
            observed=module.build_batch(np.ascontiguousarray(v,dtype=np.float64),
                np.ascontiguousarray(raw,dtype=np.float64),mask,count)
            for a,b in zip(observed,expected):
                if a.dtype!=b.dtype or not np.array_equal(a,b):
                    raise ValueError(f'Compiled batch parity failed: width={n}, threads={count}')
        cases.append(dict(particles=n,events=5,threads=counts))
    bad=np.zeros((1,2,4),np.float64); bad[0,0,0]=np.nan
    try: module.build_batch(bad,np.zeros((1,2,14),np.float64),np.ones((1,2),bool),threads)
    except RuntimeError: pass
    else: raise ValueError('Native backend accepted a nonfinite valid particle')
    return dict(passed=True,exact=True,cases=cases,invalid_input_rejected=True)


def widths(n):
    if type(n) is not int or not 1 <= n <= 128:
        raise ValueError('Particle width must be 1..128')
    return 3*(2*n-1)+4*n+2, 10*(2*n-1)+3*n


def decode(integer, floating, n, device):
    """Two host copies; reference int64 indexing and exact float32 storage."""
    iw, fw = widths(n)
    if (integer.dtype != np.int32 or floating.dtype != np.float32 or integer.ndim != 2
            or floating.shape != (len(integer), fw) or integer.shape[1] != iw):
        raise ValueError('Batched tree packed schema differs')
    integer = torch.from_numpy(integer).to(device=device, dtype=torch.int64)
    floating = torch.from_numpy(floating).to(device=device)
    m, b = 2*n-1, len(integer)
    p = {name: integer[:, i*m:(i+1)*m] for i, name in enumerate(('parent', 'depth', 'multiplicity'))}
    p.update({name: floating[:, i*m:(i+1)*m] for i, name in enumerate(
        ('pt', 'mass', 'merge_delta_r', 'merge_kt', 'merge_z', 'merge_mass'))})
    p.update(vectors=floating[:, 6*m:10*m].reshape(b,m,4),
        leaf_ranks=floating[:, 10*m:].reshape(b,3,n), leaf_to_node=integer[:, 3*m:3*m+n],
        assignments=integer[:, 3*m+n:3*m+4*n].reshape(b,3,n),
        roots=integer[:, -2], n_valid=integer[:, -1])
    return p


def encode(p):
    """Reference oracle / tests only. Production native backend packs directly."""
    integers = [p[k].reshape(len(p['roots']), -1) for k in
                ('parent', 'depth', 'multiplicity', 'leaf_to_node', 'assignments', 'roots', 'n_valid')]
    floats = [p[k].reshape(len(p['roots']), -1) for k in
              ('pt', 'mass', 'merge_delta_r', 'merge_kt', 'merge_z', 'merge_mass', 'vectors', 'leaf_ranks')]
    return (torch.cat(integers, 1).cpu().numpy().astype(np.int32),
            torch.cat(floats, 1).cpu().numpy().astype(np.float32))


def oracle_batch(v, r, mask, backend):
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree
    from teacher_logit_reco.relational_part.region_tree import build_reference_tree
    fn = build_reference_tree if backend == 'reference_for_tests' else lambda a,b,c: build_compiled_tree(backend,a,b,c)
    return encode(reference.pack([fn(a,b,c) for a,b,c in zip(v,r,mask)], v.shape[1], 'cpu'))


# Isolate a pure reference function with only its already-computed pack input
# substituted. No mutation of the reference module or global monkeypatch.
region_from_packed = types.FunctionType(reference.region_features.__code__,
    dict(reference.region_features.__globals__, pack=lambda p, particles, device: p), 'region_from_packed')


class PackedTrees:
    def __init__(self, backend, *, threads=4, cache=None, namespace='', oracle=None):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError('Tree threads must be 1..64')
        if backend is None and oracle is None:
            raise ValueError('Native backend required; oracle must be explicit for tests')
        self.backend, self.threads, self.cache, self.namespace, self.oracle = backend, threads, cache, namespace, oracle

    def __call__(self, vectors, raw, mask, unused_backend=None, *, use_cache=False):
        with torch.profiler.record_function('selected_fast/d2h'):
            # Match the reference float32 inputs before exact conversion to double.
            v = np.ascontiguousarray(vectors.detach().float().transpose(1,2).cpu().numpy(), dtype=np.float64)
            r = np.ascontiguousarray(raw.detach().float().cpu().numpy(), dtype=np.float64)
            m = np.ascontiguousarray(mask[:,0].detach().bool().cpu().numpy())
        b, n = v.shape[:2]
        iw, fw = widths(n)
        with torch.profiler.record_function('selected_fast/cpu_tree_and_pack'):
            def compute(indices):
                if self.oracle is not None:
                    return oracle_batch(v[indices], r[indices], m[indices], self.oracle)
                return self.backend.build_batch(v[indices],r[indices],m[indices],self.threads)
            if not use_cache or self.cache is None:
                integer, floating = compute(slice(None))
            else:
                integer, floating = np.empty((b,iw),np.int32), np.empty((b,fw),np.float32)
                keys, missing = [], []
                for i in range(b):
                    key = hashlib.sha256(self.namespace.encode()+v[i].tobytes()+r[i].tobytes()+m[i].tobytes()).digest()
                    keys.append(key)
                    record = self.cache.get(key)
                    if record is None:
                        missing.append(i)
                    else:
                        expected = (iw+fw)*4
                        if len(record) != expected:
                            raise ValueError('Corrupt tree cache row')
                        integer[i] = np.frombuffer(record, np.int32, iw)
                        floating[i] = np.frombuffer(record, np.float32, fw, offset=iw*4)
                if missing:
                    a, c = compute(missing)
                    for j,i in enumerate(missing):
                        integer[i], floating[i] = a[j], c[j]
                        self.cache.put(keys[i], a[j].tobytes()+c[j].tobytes())
        with torch.profiler.record_function('selected_fast/h2d_pack'):
            return decode(integer, floating, n, vectors.device)
