"""Portable contracts. No Torch, dataset, scheduler, or filesystem writes on import."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

SCHEMA = 'part_ev_oscar_test_v1'
CLASSES = ('QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl')
PREFIXES = dict(ZJetsToNuNu=0, HToBB=1, HToCC=2, HToGG=3, HToWW4Q=4,
                HToWW2Q1L=5, ZToQQ=6, WToQQ=7, TTBar=8, TTBarLep=9)
SELECTED = dict(part=46, state=49)
COUNTS = dict(part=2143486, state=2327462)
ARCHIVE_SHA = '0545be548366fbc8c4131ac8147c87660bdc038ad35189e01ec4867ab6db76a4'
INPUTS = ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask')
PACKAGES = ('torch', 'weaver-core', 'numpy', 'awkward', 'uproot', 'PyYAML', 'scipy', 'scikit-learn')
WEAVER_FILES = ('nn/model/ParticleTransformer.py', 'utils/dataset.py',
                'utils/data/config.py', 'utils/data/tools.py', 'utils/data/fileio.py')
PARITY = dict(input_rtol=1e-6, input_atol=1e-6, logit_rtol=1e-4,
              logit_atol=2e-5, maximum_logit_error=1e-3, argmax_disagreements=0)
TOOLS = ('part_ev_oscar/__init__.py', 'part_ev_oscar/common.py',
         'part_ev_oscar/export.py', 'part_ev_oscar/runtime.py',
         'part_ev_oscar/saved_validation.py',
         'part_ev_oscar/results.py', 'part_ev_oscar/launcher.py',
         'scripts/part_ev_oscar.py', 'scripts/setup_part_ev_oscar.sh',
         'scripts/part_ev_rejection_eval_support.py', 'scripts/run_part_ev_rejection_eval.py',
         'sbatch/run_part_ev_oscar.sh')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def seal(row):
    if 'content_hash' in row:
        raise ValueError('Already sealed')
    return dict(row, content_hash=digest(row))


def checked(path):
    row = read(path)
    if row.get('content_hash') != digest({k: v for k, v in row.items() if k != 'content_hash'}):
        raise ValueError(f'Manifest changed: {path}')
    return row


def write(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(row, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def child(root, relative):
    name = PurePosixPath(relative)
    if (name.is_absolute() or '..' in name.parts or not name.parts
            or '\\' in relative or ':' in relative):
        raise ValueError(f'Unsafe relative path: {relative}')
    root = Path(root).resolve()
    path = root.joinpath(*name.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f'Linked/outside artifact: {path}')
    return path


def verify_files(root, files):
    if not files:
        raise ValueError('Empty file manifest')
    for name, checksum in files.items():
        path = child(root, name)
        if not path.is_file() or sha(path) != checksum:
            raise ValueError(f'Artifact changed: {path}')


def tree_hashes(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Links not allowed in portable bundle: {path}')
        if path.is_file():
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def verify_probe_context(probe):
    """Keep scored-event identity distinct from extra preprocessing context."""
    raw = probe.get('raw_entries', 512)
    if (probe['entry_start'] != 0 or probe['entry_stop'] != 512
            or type(raw) is not int or raw < 512):
        raise ValueError('Invalid validation probe event range')
    if 'raw_entries' not in probe:
        return  # Legacy bundles already proved exact 512-row preprocessing.
    context = probe['padding_context']
    if (context['policy'] != 'weaver_file_wrap_prefix_v1'
            or context['target_entries'] != 512 or context['padded_length'] != 128
            or context['raw_entries'] != raw
            or type(context['original_entries']) is not int or context['original_entries'] < raw
            or type(context['flattened_particles']) is not int or context['flattened_particles'] <= 0
            or (context['flattened_particles'] < 512*128 and raw != context['original_entries'])):
        raise ValueError('Insufficient or inconsistent validation padding context')


@contextmanager
def lock(path):
    """No persistent stale PID lock; Linux workers release this on termination."""
    import fcntl
    with Path(path).open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_bundle(root):
    root = Path(root)
    row = checked(root / 'bundle.json')
    if row['schema'] != SCHEMA or row['class_names'] != list(CLASSES):
        raise ValueError('Wrong portable bundle schema/classes')
    if set(row['tools']) != set(TOOLS) or set(row['models']) != set(SELECTED):
        raise ValueError('Incomplete bundle tool/model manifest')
    verify_files(root / 'source', row['tools'])
    for key in SELECTED:
        if sha(root / key / 'model.json') != row['models'][key]:
            raise ValueError('Model manifest changed')
        model = checked(root / key / 'model.json')
        if model['key'] != key or model['best_check'] != SELECTED[key]:
            raise ValueError('Unexpected checkpoint selection')
        if (model['parameters'] != COUNTS[key] or model['class_names'] != list(CLASSES)
                or len(model['probes']) != 10 or model['environment'] != row['environment']):
            raise ValueError('Portable model protocol differs')
        for probe in model['probes']:
            verify_probe_context(probe)
        verify_files(root / key, model['files'])
        if model.get('reference_origin') == 'saved_validation_cpu_v1':
            from .saved_validation import verify_portable_provenance
            verify_portable_provenance(root / key, model)
    return row


def load_plan(output):
    plan = checked(Path(output) / 'plan.json')
    if plan['schema'] != SCHEMA or plan['parity'] != PARITY or plan['precision'] != 'FP32':
        raise ValueError('Evaluation protocol changed')
    if set(plan['tools']) != set(TOOLS):
        raise ValueError('Incomplete campaign tool manifest')
    verify_files(Path(output) / 'source', plan['tools'])
    bundle = load_bundle(plan['bundle'])
    if bundle['content_hash'] != plan['bundle_sha256']:
        raise ValueError('Bundle changed since plan')
    return plan


def expected_files():
    return {f'{label}_{index:03d}.root': value
            for label, value in PREFIXES.items() for index in range(100, 120)}


def validate_arrays(logits, labels, events, label=None):
    import numpy as np
    if (logits.shape != (events, 10) or logits.dtype != np.float32
            or labels.shape != (events,) or labels.dtype != np.int64
            or not np.isfinite(logits).all()
            or np.any((labels < 0) | (labels >= 10))
            or (label is not None and np.any(labels != label))):
        raise ValueError('Invalid prediction shape/dtype/labels/finiteness')


def comparison(actual, expected, *, inputs=False):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return dict(passed=False, reason='shape or nonfinite')
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    prefix = 'input' if inputs else 'logit'
    passed = bool(np.allclose(actual, expected, rtol=PARITY[prefix+'_rtol'],
                              atol=PARITY[prefix+'_atol']))
    row = dict(passed=passed, maximum_absolute_error=float(delta.max(initial=0)))
    if not inputs:
        row['argmax_disagreements'] = int(np.count_nonzero(actual.argmax(1) != expected.argmax(1)))
        row['passed'] &= (row['maximum_absolute_error'] <= PARITY['maximum_logit_error']
                          and row['argmax_disagreements'] == 0)
    return row
