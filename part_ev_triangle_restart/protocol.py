"""Standard-library-only admission and immutable diagnostic identities."""
import hashlib
import json
from pathlib import Path

SCHEMA = 'triangle_restart_diagnostic_v1'
POLICY = dict(partition='debug', walltime_seconds=3600, shutdown_margin_seconds=300,
              gpus=2, gpu_type='a100', cpus=8, host_memory_gib=128,
              steps=7, resume_after=3, global_batch=512, particles=128,
              output_budget_bytes=1024**3, authorizes_training=False,
              authorizes_benchmark_retry=False, changes_tolerances=False,
              real_data_stream_resume_verified=False)
ARTIFACTS = ('campaign.json', 'launch.json', 'source_manifest.json', 'probes/restart.pt')
FILES = ('part_ev_triangle_restart/__init__.py', 'part_ev_triangle_restart/protocol.py',
         'part_ev_triangle_restart/engine.py', 'scripts/diagnose_part_ev_triangles_restart.py',
         'scripts/submit_part_ev_triangles_restart.py', 'sbatch/run_part_ev_triangles_restart.sh',
         'sbatch/submit_part_ev_triangles_restart.sh')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def checked(path):
    row = read(path)
    if row.get('content_hash') != digest({k: v for k, v in row.items() if k != 'content_hash'}):
        raise ValueError('Invalid content hash: '+str(path))
    return row


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.absolute() != path.resolve():
        raise ValueError('Use absolute paths without symlinks or traversal')
    return path


def verify_inventory(directory, inventory):
    actual = {}
    for path in Path(directory).rglob('*'):
        if path.is_symlink():
            raise ValueError('Source symlinks are forbidden')
        if path.is_file():
            actual[path.relative_to(directory).as_posix()] = sha(path)
    if not actual or actual != inventory:
        raise ValueError('Frozen source inventory/content changed')


def donor(campaign):
    campaign = canonical(campaign)
    spec, launch = checked(campaign/'campaign.json'), checked(campaign/'launch.json')
    if (spec.get('schema') != 'part_ev_triangles_benchmark_v1'
            or spec.get('task', {}).get('architecture') != 'state_triangles_layerwise'
            or spec['task'].get('seed') != 101 or spec.get('parameters') != 2334662
            or spec.get('policy', {}).get('world_size') != 2
            or spec['policy'].get('global_batch') != 512
            or spec['policy'].get('training_precision') != 'fp16'
            or spec.get('authorizes_training') is not False or spec.get('launch') != launch
            or launch.get('output') != str(campaign)):
        raise ValueError('Need the prepared two-A100 triangle benchmark, not a production campaign')
    inventory = read(campaign/'source_manifest.json')
    verify_inventory(campaign/'source', inventory)
    if spec['source_sha256'] != digest(inventory) or launch['source_sha256'] != digest(inventory):
        raise ValueError('Donor source digest differs')
    probe = campaign/'probes/restart.pt'
    if probe.is_symlink() or probe.resolve() != probe or not probe.is_file():
        raise ValueError('Saved probes/restart.pt is required; no fallback to a fresh model')
    return spec, launch, {name: sha(campaign/name) for name in ARTIFACTS}


def load_plan(output, *, verify_donor=True):
    output = canonical(output)
    p = checked(output/'plan.json')
    if (p.get('schema') != SCHEMA or p.get('policy') != POLICY or p.get('output') != str(output)
            or p.get('authorizes_training') is not False
            or set(p.get('diagnostic_source', {})) != set(FILES)):
        raise ValueError('Diagnostic policy or identity changed')
    verify_inventory(output/'source', p['diagnostic_source'])
    if verify_donor:
        spec, launch, artifacts = donor(Path(p['campaign']))
        if (artifacts != p['artifacts'] or spec['content_hash'] != p['campaign_sha256']
                or launch['python'] != p['python']):
            raise ValueError('Donor artifacts changed since diagnostic submission')
    return p


def validate_checkpoint(saved, spec, world):
    if (saved.get('campaign_sha256') != spec['content_hash']
            or len(saved.get('rank_states', [])) != world
            or saved.get('progress') != dict(attempt=19997, checks=0, fixture_cursor=0)
            or saved.get('optimizer', {}).get('schema') != 'full_ranger_v1'
            or not all(k in saved for k in ('model', 'optimizer', 'scaler'))):
        raise ValueError('Wrong/incomplete saved triangle restart fixture')
