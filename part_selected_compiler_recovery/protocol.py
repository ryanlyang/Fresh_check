"""Independent recovery overlay; never rewrite historical source or manifests."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SCHEMA = 'selected_compiler_recovery_v1'
INDICES = (2, 5, 6)
OLD_BANNER = 'c++ (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)'
NEW_BANNER = 'c++ (GCC) 11.5.0 20240719 (Red Hat 11.5.0-15)'
FILES = (
    'part_selected_compiler_recovery/__init__.py',
    'part_selected_compiler_recovery/protocol.py',
    'part_selected_compiler_recovery/backend.py',
    'part_selected_compiler_recovery/engine.py',
    'scripts/run_part_selected_compiler_recovery.py',
    'scripts/submit_part_selected_compiler_recovery.py',
    'sbatch/run_part_selected_compiler_recovery.sh',
    'sbatch/submit_part_selected_compiler_recovery.sh',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def hashed(row):
    return dict(row, content_hash=hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())


def checked(path):
    row = read(path)
    if row != hashed({k: v for k, v in row.items() if k != 'content_hash'}):
        raise ValueError('Metadata hash changed: ' + str(path))
    return row


def write(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def check_files(files):
    for filename, checksum in files.items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file() or sha(path) != checksum:
            raise ValueError('Pinned artifact changed: ' + str(path))


def activate(campaign):
    """Load only the original authenticated production implementation."""
    campaign = Path(campaign).resolve(strict=True)
    plan = checked(campaign/'production.json')
    source = campaign/'source'
    check_files({str(source/n): s for n, s in plan['source'].items()})
    if Path(sys.executable).resolve() != Path(plan['python']).resolve():
        raise ValueError('Use the original production Python')
    sys.path.insert(0, str(source))
    import part_selected_production.protocol as original
    if Path(original.__file__).resolve() != (source/'part_selected_production/protocol.py').resolve():
        raise ValueError('Production imports must come from its frozen source')
    plan = original.load(campaign)  # Full environment/source/evidence checks remain enabled.
    expected = {
        2: 'SELECTED_PART_MATCH_STATE_LAYERWISE',
        5: 'SELECTED_PART_SHARED', 6: 'SELECTED_STATE_EV_LAYERWISE',
    }
    if len(plan['science']['tasks']) != 7 or any(
        plan['science']['tasks'][i]['run_id'] != name for i, name in expected.items()
    ):
        raise ValueError('This recovery is only for the original seven-run selected tasks')
    return original, plan


def evidence(plan):
    """Pin both existing binaries before executing either. Never compile here."""
    campaign = Path(plan['output'])
    benchmark = Path(plan['benchmark'])
    reference = Path(plan['science']['reference']['path'])/'backend'
    manifest = read(reference/'backend_manifest.json')
    candidates = list(reference.glob('relational_ca_tree_v1_ext*.so'))
    if len(candidates) != 1 or candidates[0].name != manifest['binary_filename']:
        raise ValueError('Expected exactly the recorded reference backend binary')
    binary = candidates[0]
    if binary.is_symlink() or sha(binary) != manifest['binary_sha256']:
        raise ValueError('Reference backend binary changed')
    fast = read(benchmark/'backend/manifest.json')
    fast_binary = (benchmark/'backend'/fast['binary']).resolve(strict=True)
    if (not fast_binary.is_relative_to((benchmark/'backend').resolve()) or
            sha(fast_binary) != fast['binary_sha256']):
        raise ValueError('Optimized backend binary changed')
    source = benchmark/'source/teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp'
    if sha(source) != manifest['source_sha256']:
        raise ValueError('Reference backend source changed')
    paths = [campaign/'production.json', reference/'backend_manifest.json', binary, source,
             benchmark/'backend/manifest.json', fast_binary]
    return {str(p): sha(p) for p in paths}


def load(output):
    output = Path(output).resolve(strict=True)
    row = checked(output/'recovery.json')
    if (row['schema'] != SCHEMA or row['output'] != str(output) or
            row['indices'] != list(INDICES) or set(row['source']) != set(FILES) or
            row['transition'] != dict(old=OLD_BANNER, new=NEW_BANNER)):
        raise ValueError('Recovery scope/source/policy changed')
    actual = {p.relative_to(output/'source').as_posix() for p in (output/'source').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(FILES):
        raise ValueError('Recovery source inventory changed')
    check_files({str(output/'source'/n): s for n, s in row['source'].items()})
    check_files(row['artifacts'])
    original, plan = activate(row['campaign'])
    if (plan['content_hash'] != row['production_sha256'] or plan['python'] != row['python'] or
            evidence(plan) != row['artifacts']):
        raise ValueError('Recovery belongs to different production/artifacts')
    return row, original, plan


def approval(output, row):
    result = checked(Path(output)/'compatibility.json')
    if (result.get('schema') != SCHEMA or result.get('recovery_sha256') != row['content_hash'] or
            result.get('passed') is not True or result.get('authorizes_recovery') is not True or
            result.get('indices') != list(INDICES)):
        raise ValueError('A passed, matching compatibility check is required')
    return result


def original_driver(plan):
    path = Path(plan['output'])/'source/scripts/run_part_selected_production.py'
    spec = importlib.util.spec_from_file_location('_recovery_original_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
