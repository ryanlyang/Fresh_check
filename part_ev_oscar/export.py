"""Runs on RIT only. Audits original donors before writing a separate bundle."""
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

from . import common as C


def environment():
    import platform
    import torch
    import weaver
    package = Path(weaver.__file__).parent
    return dict(python=platform.python_version(), architecture=platform.machine(),
                packages={name: importlib.metadata.version(name) for name in C.PACKAGES},
                torch_cuda=torch.version.cuda,
                weaver_sources={name: C.sha(package/name) for name in C.WEAVER_FILES})


def donor_support():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import part_ev_rejection_eval_support as support
    return support


def describe(part=None, state=None):
    S = donor_support()
    donors = {key: S.describe(root, key) for key, root in
              (('part', part or S.PART), ('state', state or S.STATE))}
    for key, donor in donors.items():
        if (donor['best_check'] != C.SELECTED[key]
                or donor['task']['architecture'] != ('part' if key == 'part' else 'state_layerwise')):
            raise ValueError('Refusing a different selected checkpoint or architecture')
        S.verify_frozen_source(donor)
    if donors['part']['files'] != donors['state']['files']:
        raise ValueError('Donor validation populations differ')
    return donors


def copy_frozen(donor, destination):
    source = Path(donor['root']) / 'source'
    files = {}
    for line in (Path(donor['root'])/'source.sha256').read_text().splitlines():
        checksum, relative = line.split(maxsplit=1)
        relative = Path(relative.lstrip('*')).as_posix()
        path = C.child(source, relative)
        relative = path.relative_to(source).as_posix()
        if '__pycache__' in Path(relative).parts or relative.endswith('.pyc'):
            continue
        if relative in files or C.sha(path) != checksum:
            raise ValueError('Duplicate or changed donor source')
        target = C.child(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[relative] = checksum
    C.verify_files(destination, files)


def write_probe_root(original, output, events=512):
    """Copy only validation entries 0:512, explicitly writing a TTree, not RNTuple."""
    import uproot
    with uproot.open(original) as f:
        arrays = f['tree'].arrays(entry_start=0, entry_stop=events, library='ak', how=dict)
    with uproot.recreate(output) as f:
        f.mktree('tree', {name: values.type.content for name, values in arrays.items()})
        f['tree'].extend(arrays)
    with uproot.open(output) as f:
        if f['tree'].classname != 'TTree' or f['tree'].num_entries != events:
            raise ValueError('Probe TTree roundtrip failed')


def write_context_probe(original, output, events=512, padded_length=128):
    """Preserve Weaver's file-wide wrap-padding prefix for the target events.

    _repeat_pad flattens the entire fetched table before tiling/reshaping it.
    The first N padded jets therefore need the first N*L constituent values,
    not just the constituents belonging to the first N jets. Retain a prefix
    containing that many particles (or the entire original file if shorter).
    Context jets are preprocessing inputs only, never extra parity predictions.
    """
    import awkward as ak
    import numpy as np
    import uproot
    if type(events) is not int or events <= 0 or type(padded_length) is not int or padded_length <= 0:
        raise ValueError('Positive integer probe events and padded length required')
    needed = events * padded_length
    with uproot.open(original) as f:
        tree = f['tree']
        total = int(tree.num_entries)
        if total < events:
            raise ValueError('Original validation file is shorter than the parity probe')
        read_stop = events
        while True:
            energies = tree['part_energy'].array(entry_start=0, entry_stop=read_stop, library='ak')
            counts = ak.to_numpy(ak.num(energies, axis=1)).astype(np.int64)
            cumulative = np.cumsum(counts)
            if cumulative[-1] >= needed:
                raw_entries = max(events, int(np.searchsorted(cumulative, needed)) + 1)
                break
            if read_stop == total:
                raw_entries = total
                break
            read_stop = min(total, read_stop * 2)
        if cumulative[-1] == 0:
            raise ValueError('Cannot reproduce wrap padding from an empty constituent population')
        # All official per-particle fields must share these offsets. Reject a
        # malformed file rather than guessing context for mismatched branches.
        arrays = tree.arrays(entry_start=0, entry_stop=raw_entries, library='ak', how=dict)
        for name, values in arrays.items():
            if name.startswith('part_') and (values.ndim != 2 or not np.array_equal(
                    ak.to_numpy(ak.num(values, axis=1)), counts[:raw_entries])):
                raise ValueError(f'Inconsistent validation constituent lengths: {name}')
    write_probe_root(original, output, events=raw_entries)
    return dict(policy='weaver_file_wrap_prefix_v1', target_entries=events,
                padded_length=padded_length, raw_entries=raw_entries,
                original_entries=total, flattened_particles=int(cumulative[raw_entries-1]))


def assert_same_batch(original, sliced):
    """Exact comparison, with actionable field-level diagnostics on failure."""
    import torch
    for group, (expected, actual) in enumerate(zip(original[:2], sliced[:2])):
        if expected.keys() != actual.keys():
            raise ValueError(f'Validation ROOT slice changed fields in group {group}')
        for name in expected:
            a, b = expected[name], actual[name]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError(f'Validation ROOT slice changed {name}: '
                                 f'{tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}')
            if not torch.equal(a, b):
                differences = int(torch.count_nonzero(a != b).item())
                raise ValueError('Validation ROOT slice changed preprocessing/order: '
                                 f'{name}, {differences}/{a.numel()} unequal values; '
                                 'exact equality is required, including padding')


def record(path, events):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(name=path.name, path=str(path), size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, num_entries=events)


def export_model(output, key):
    output = Path(output)
    plan = C.checked(output/'export_plan.json')
    C.verify_files(output/'source', plan['tools'])
    donor = plan['donors'][key]
    if Path(sys.executable).resolve() != Path(donor['python']).resolve():
        raise ValueError('Export must use the original donor Python')
    S = donor_support()
    S.refuse_active(plan['donors'])
    S.verify_pins(donor)
    S.verify_frozen_source(donor)
    with C.lock(output/f'.export_{key}.lock'):
        directory = output/key
        if (directory/'model.json').exists():
            saved = C.checked(directory/'model.json')
            if saved['export_plan_sha256'] != plan['content_hash']:
                raise ValueError('Wrong existing export')
            C.verify_files(directory, saved['files'])
            print(f'{key}: already exported and verified', flush=True)
            return
        # An interrupted export is kept for diagnosis; no automatic recursive deletion.
        directory.mkdir()
        for name in list(os.environ):
            if name.startswith('NCCL_'):
                del os.environ[name]
        if key == 'state':
            spec_before_import = C.read(Path(donor['root'])/'campaign.json')
            os.environ.update(spec_before_import['expected_gpu']['nccl_environment'])
        import run_part_ev_rejection_eval as R
        # This is the unchanged, hardware/environment/selection-audited RIT path.
        D, spec, model, device, runtime, stop = R.load_runtime(donor)
        import numpy as np
        import torch
        env = environment()
        copy_frozen(donor, directory/'source')
        for relative in donor['pins']:
            target = C.child(directory/'provenance', relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(C.child(donor['root'], relative), target)
        # Only tensors and a plain dictionary: Oscar uses weights_only=True.
        D.atomic_torch_save({'model': {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}}, directory/'weights.pt')
        probes = []
        (directory/'probes').mkdir()
        for prefix in C.PREFIXES:
            from .runtime import stopping
            stopping()
            stop()
            original = next(r for r in donor['files'] if r['name'] == prefix+'_120.root')
            first = next(iter(D.finite_validation_loader(spec['inventory'], original)))
            values, labels = D.tensors(first, device)
            if len(labels) != 512:
                raise ValueError('Reference probe must be one full 512-event batch')
            probe_path = directory/'probes'/original['name']
            context = write_context_probe(original['path'], probe_path)
            r = record(probe_path, context['raw_entries'])
            inv = {'splits': {'val': {'files': [r]}}}
            roundtrip = next(iter(D.finite_validation_loader(inv, r)))
            assert_same_batch(first, roundtrip)
            tensors = {name: value.detach().cpu().numpy().copy()
                       for name, value in zip(C.INPUTS, values)}
            with torch.no_grad(), D.model_precision(model, device, training=False), \
                    D.audit_precision(model, expected_dtype=torch.float32) as audit:
                logits = model(*values).float().cpu().numpy()
            truth = labels.cpu().numpy().astype(np.int64)
            C.validate_arrays(logits, truth, 512, C.PREFIXES[prefix])
            reference = directory/'probes'/(prefix+'.npz')
            D.atomic_npz(reference, logits=logits, labels=truth, **tensors)
            probes.append(dict(root='probes/'+probe_path.name, reference='probes/'+reference.name,
                               original_name=original['name'], original_sha256=C.sha(original['path']),
                               entry_start=0, entry_stop=512, raw_entries=context['raw_entries'],
                               padding_context=context, label=C.PREFIXES[prefix], precision=audit))
            print(f"{key}: exported validation probe {prefix} ({len(probes)}/10); "
                  f"512 scored jets, {context['raw_entries']} raw context jets", flush=True)
        S.verify_pins(donor)
        S.verify_frozen_source(donor)
        files = C.tree_hashes(directory)
        row = C.seal(dict(schema=C.SCHEMA, key=key, export_plan_sha256=plan['content_hash'],
            best_check=donor['best_check'], checkpoint_sha256=donor['checkpoint_sha256'],
            campaign_sha256=donor['campaign_sha256'], task=donor['task'],
            parameters=sum(p.numel() for p in model.parameters()),
            class_names=list(C.CLASSES), environment=env, training_runtime=donor['training_runtime'],
            export_gpu=torch.cuda.get_device_name(device), probes=probes,
            weights_sha256=C.sha(directory/'weights.pt'), files=files))
        if row['parameters'] != C.COUNTS[key]:
            raise ValueError('Wrong exported model size')
        C.write(directory/'model.json', row)


def finalize(output):
    """Small bundle manifest is the commit point; no training or test data involved."""
    output = Path(output)
    plan = C.checked(output/'export_plan.json')
    C.verify_files(output/'source', plan['tools'])
    models = {key: C.checked(output/key/'model.json') for key in C.SELECTED}
    for key, row in models.items():
        if (row['key'] != key or row['export_plan_sha256'] != plan['content_hash']
                or row['best_check'] != C.SELECTED[key] or row['parameters'] != C.COUNTS[key]):
            raise ValueError('Export provenance differs')
        C.verify_files(output/key, row['files'])
    if models['part']['environment'] != models['state']['environment']:
        raise ValueError('Donor dependency environments differ; needs explicit compatibility investigation')
    p, s = models['part']['probes'], models['state']['probes']
    if len(p) != 10 or len(s) != 10 or any(
            (a['original_name'], a['original_sha256'], a['entry_stop'], a.get('raw_entries', 512), a.get('padding_context')) !=
            (b['original_name'], b['original_sha256'], b['entry_stop'], b.get('raw_entries', 512), b.get('padding_context'))
            for a, b in zip(p, s)):
        raise ValueError('Donor validation probes do not identify the same events')
    row = C.seal(dict(schema=C.SCHEMA, class_names=list(C.CLASSES),
                     tools=plan['tools'], environment=models['part']['environment'],
                     models={k: C.sha(output/k/'model.json') for k in C.SELECTED}))
    if (output/'bundle.json').exists() and C.checked(output/'bundle.json') != row:
        raise ValueError('Existing bundle manifest differs')
    C.write(output/'bundle.json', row)
    print('Portable bundle complete. No test inference authorized yet.', flush=True)
