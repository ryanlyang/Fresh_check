"""Read-only reuse of completed evrej GPU predictions; no model inference here."""
import hashlib
from pathlib import Path
from . import common as C

MODE = 'saved_validation_cpu_v1'
FROZEN_PACKAGES = ('part_ev_fullscale', 'part_ev_screen', 'part_ev_message_study',
                   'part_ev_state_speed', 'part_ev_state_compare',
                   'part_ev_state_fullscale', 'part_ev_layerwise_ddp')


def reference_plan(root):
    from .export import donor_support
    plan = donor_support().verify_snapshot(root)
    if (plan['schema'] != 'part_ev_rejection_eval_v1'
            or plan['population'] != 'official_validation_5M'
            or plan['independent_test'] is not False or plan['precision'] != 'FP32'
            or plan['batch_size'] != 512 or set(plan['donors']) != set(C.SELECTED)):
        raise ValueError('Need the completed, FP32, batch-512 evrej validation campaign')
    return plan


def contract(plan, donor, record):
    return dict(plan_sha256=plan['content_hash'], campaign_sha256=donor['campaign_sha256'],
                checkpoint_sha256=donor['checkpoint_sha256'], file=record,
                entry_start=0, entry_stop=record['num_entries'], precision='FP32')


def describe_model(root, plan, key, donor):
    """Metadata audit only on the submit host; workers hash/read all payloads."""
    root = Path(root).resolve(strict=True)
    if plan['donors'][key] != donor:
        raise ValueError(f'{key}: saved validation belongs to different donor artifacts')
    path = root/key/'metrics.json'
    row = C.checked(path)
    expected = dict(model=key, plan_sha256=plan['content_hash'],
                    campaign_sha256=donor['campaign_sha256'], checkpoint_sha256=donor['checkpoint_sha256'],
                    task=donor['task'], best_check=donor['best_check'],
                    training_runtime=donor['training_runtime'],
                    population='official_validation_5M', independent_test=False)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{key}: completed validation identity/selection differs')
    runtime = row['evaluation_runtime']
    if (runtime.get('precision') != 'FP32' or runtime.get('batch_size') != 512
            or runtime.get('world_size') != 1 or not runtime.get('gpu')):
        raise ValueError('Saved reference must be single-GPU FP32 inference at batch 512')
    if (row['metrics']['events'] != 5_000_000 or row['metrics']['class_support'] !=
            {name: 500_000 for name in C.CLASSES}):
        raise ValueError('Saved evaluation is not the full balanced 5M validation population')
    names = {f'{prefix}_{i:03d}.root' for prefix in C.PREFIXES for i in range(120, 125)}
    records = donor['files']
    if len(records) != 50 or [r['name'] for r in records] != sorted(names):
        raise ValueError('Unexpected validation file inventory/order')
    receipts = row['identity_manifest']
    if len(receipts) != 50:
        raise ValueError('Incomplete saved validation manifest')
    files = {'plan.json': C.sha(root/'plan.json'), f'{key}/metrics.json': C.sha(path)}
    files.update({'source/'+name: value for name, value in plan['source'].items()})
    for i, (record, receipt) in enumerate(zip(records, receipts)):
        npz, side = f'{key}/file_{i:03d}.npz', f'{key}/file_{i:03d}.json'
        if (record['num_entries'] != 100_000 or receipt['contract'] != contract(plan, donor, record)
                or C.read(root/side) != receipt or not (root/npz).is_file()):
            raise ValueError(f'{key}: missing or mismatched validation chunk {i}')
        precision = receipt['precision']
        if precision.get('passed') is not True or precision.get('expected_activation_dtype') != 'torch.float32':
            raise ValueError('Missing saved FP32 precision evidence')
        for field in ('sha256', 'labels_sha256'):
            value = receipt[field]
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Malformed saved validation hash')
        files[side], files[npz] = C.sha(root/side), receipt['sha256']
    return dict(root=str(root), model=key, plan_sha256=plan['content_hash'],
                evaluation_runtime=runtime, files=files)


def describe(root, donors):
    root = Path(root).resolve(strict=True)
    plan = reference_plan(root)
    result = {key: describe_model(root, plan, key, donors[key]) for key in C.SELECTED}
    p = C.checked(root/'part/metrics.json')['identity_manifest']
    s = C.checked(root/'state/metrics.json')['identity_manifest']
    if [r['labels_sha256'] for r in p] != [r['labels_sha256'] for r in s]:
        raise ValueError('Saved model evaluations have different event label order')
    return result


def discover(donors, explicit=None, search_root=None):
    if explicit is not None:
        return describe(explicit, donors)
    from .export import donor_support
    parent = Path(search_root) if search_root is not None else donor_support().BASE/'part_ev_rejection_eval'
    matches, failures = [], []
    for path in sorted(parent.glob('*/plan.json')):
        root = path.parent
        if not all((root/key/'metrics.json').is_file() for key in C.SELECTED):
            continue
        try:
            matches.append(describe(root, donors))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append(f'{root.name}: {exc}')
    if len(matches) != 1:
        details = [m['part']['root'] for m in matches] if matches else failures[:3]
        raise ValueError(f'Found {len(matches)} matching completed validation campaigns under {parent}. '
                         f'Pass --validation-results /exact/evrej/campaign. Details: {details}')
    return matches[0]


def inactive(references, donors):
    from .export import donor_support
    ledgers = {Path(r['root'])/'submitted_job_ids.tsv' for r in references.values()}
    donor_support().refuse_active(donors, extra_ledgers=sorted(ledgers))


def audit_payloads(reference, donor):
    """Check all 50 saved chunks, retaining only the ten 512-event probes in RAM."""
    import numpy as np
    from .runtime import stopping
    root, key = Path(reference['root']), reference['model']
    metadata = {k: v for k, v in reference['files'].items() if not k.endswith('.npz')}
    C.verify_files(root, metadata)
    plan = reference_plan(root)
    if describe_model(root, plan, key, donor) != reference:
        raise ValueError('Saved evaluation changed after export submission')
    row = C.checked(root/key/'metrics.json')
    probes, correct, labels_seen = {}, 0, np.zeros(10, np.int64)
    for i, (record, receipt) in enumerate(zip(donor['files'], row['identity_manifest'])):
        stopping()
        name = f'{key}/file_{i:03d}.npz'
        path = C.child(root, name)
        if C.sha(path) != reference['files'][name]:
            raise ValueError(f'Saved prediction bytes changed: {path}')
        with np.load(path, allow_pickle=False) as arrays:
            scores, labels = arrays['logits'], arrays['labels']
        label = C.PREFIXES[record['name'].rsplit('_', 1)[0]]
        C.validate_arrays(scores, labels, 100_000, label)
        if hashlib.sha256(labels.tobytes()).hexdigest() != receipt['labels_sha256']:
            raise ValueError('Saved prediction label hash differs')
        correct += int(np.count_nonzero(scores.argmax(1) == labels))
        labels_seen += np.bincount(labels, minlength=10)
        if record['name'].endswith('_120.root'):
            probes[record['name']] = dict(logits=scores[:512].copy(), labels=labels[:512].copy(),
                precision=receipt['precision'], evidence=dict(
                    prediction_file=name, prediction_sha256=receipt['sha256'],
                    sidecar_file=f'{key}/file_{i:03d}.json', sidecar_sha256=reference['files'][f'{key}/file_{i:03d}.json'],
                    reference_plan_sha256=plan['content_hash'], entry_start=0, entry_stop=512))
        print(f'{key}: verified saved GPU predictions {i+1}/50 (CPU only)', flush=True)
    if (not np.array_equal(labels_seen, np.full(10, 500_000)) or len(probes) != 10
            or correct/5_000_000 != row['metrics']['accuracy']):
        raise ValueError('Saved validation coverage/accuracy does not reproduce its completed report')
    C.verify_files(root, metadata)
    return probes


def load_cpu_runtime(donor):
    """Original CPU-readable admission/training audits, never gpu_setup/forward."""
    import os
    import sys
    from types import SimpleNamespace
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    source = Path(donor['root'])/'source'
    for name in FROZEN_PACKAGES:
        module = sys.modules.get(name)
        if module and not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError('Another donor model namespace was imported before CPU export')
    sys.path.insert(0, str(source))
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU export must not have an initialized CUDA runtime')
    torch.set_num_threads(4)
    from part_ev_state_fullscale.reporting import validate_training
    from part_ev_message_study.controls import runtime_identity
    root = Path(donor['root'])
    if donor['key'] == 'part':
        from part_ev_state_fullscale import protocol as P
        spec = P.load_campaign(root)
        P.verify_inventory_metadata(spec)
        approved = P.require_approval(root, spec)
        expected = approved['runtime']
    else:
        from part_ev_layerwise_ddp import protocol as P
        spec = P.load(root)
        P.G.C.verify_inventory_metadata(spec)
        P.require_approval(root, spec)
        expected = P.read_json(root/'preflight/result.json')['runtime']
    audited = validate_training(root, spec, donor['task'], runtime_identity(expected))
    if (audited['checkpoint_sha256'] != donor['checkpoint_sha256']
            or audited['best_check'] != donor['best_check']):
        raise ValueError('CPU audit found a different selected checkpoint')
    from part_ev_fullscale.checkpoint import trusted_load, atomic_torch_save
    from part_ev_screen.data import finite_validation_loader
    from .runtime import build_architecture
    from .results import atomic_npz
    model = build_architecture(donor['key']).cpu().eval()
    model.load_state_dict(trusted_load(Path(donor['run'])/'best_model_val.pt')['model'], strict=True)
    if (sum(p.numel() for p in model.parameters()) != C.COUNTS[donor['key']]
            or any(t.device.type != 'cpu' for t in model.state_dict().values())):
        raise ValueError('CPU-export model capacity/device changed')
    return SimpleNamespace(atomic_torch_save=atomic_torch_save, atomic_npz=atomic_npz,
                           finite_validation_loader=finite_validation_loader), spec, model


def copy_provenance(reference, destination):
    """Copy small provenance only, not the full 5M prediction payloads."""
    import shutil
    root, destination = Path(reference['root']), Path(destination)
    for name, checksum in reference['files'].items():
        if name.endswith('.npz'):
            continue
        source, target = C.child(root, name), C.child(destination, name)
        if C.sha(source) != checksum:
            raise ValueError('Saved evaluation provenance changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    C.verify_files(destination, {k: v for k, v in reference['files'].items() if not k.endswith('.npz')})
    C.write(destination/'reference.json', C.seal(reference))


def verify_portable_provenance(directory, manifest):
    """Oscar-side audit of CPU export provenance; never open original RIT paths."""
    directory = Path(directory)
    reference = manifest['reference_evaluation']
    root = directory/'saved_validation'
    if (manifest['export_gpu'] is not None or reference['model'] != manifest['key']
            or C.checked(root/'reference.json') != C.seal(reference)):
        raise ValueError('CPU reference provenance differs')
    C.verify_files(root, {k: v for k, v in reference['files'].items() if not k.endswith('.npz')})
    plan = C.checked(root/'plan.json')
    result = C.checked(root/manifest['key']/'metrics.json')
    if (plan['content_hash'] != reference['plan_sha256']
            or result['plan_sha256'] != plan['content_hash']
            or result['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or result['campaign_sha256'] != manifest['campaign_sha256']
            or result['best_check'] != manifest['best_check']):
        raise ValueError('Portable reference selected checkpoint differs')
    for probe in manifest['probes']:
        saved = probe['saved_prediction']
        if (saved['reference_plan_sha256'] != reference['plan_sha256']
                or saved['entry_start'] != 0 or saved['entry_stop'] != 512
                or saved['prediction_sha256'] != reference['files'][saved['prediction_file']]
                or saved['sidecar_sha256'] != reference['files'][saved['sidecar_file']]):
            raise ValueError('Portable saved prediction identity differs')
        receipt = C.read(C.child(root, saved['sidecar_file']))
        if (receipt['sha256'] != saved['prediction_sha256'] or receipt['precision'] != probe['precision']
                or receipt['contract']['file']['name'] != probe['original_name']
                or receipt['contract']['checkpoint_sha256'] != manifest['checkpoint_sha256']):
            raise ValueError('Portable probe does not correspond to the saved validation chunk')
