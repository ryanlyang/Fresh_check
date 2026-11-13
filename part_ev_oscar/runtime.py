"""Oscar admission and FP32 inference. No training, automatic tuning, or donor writes."""
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

from . import common as C
from .export import environment, record

STOP = False


def stop_signal(*_):
    global STOP
    STOP = True


def stopping():
    if STOP:
        raise SystemExit(75)


def configure(output):
    """All application caches/temp files belong to the new scratch campaign."""
    output = Path(output)
    for name, folder in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'),
                         ('TORCH_HOME', 'cache/torch'), ('TORCH_EXTENSIONS_DIR', 'cache/extensions'),
                         ('CUDA_CACHE_PATH', 'cache/cuda'), ('MPLCONFIGDIR', 'cache/matplotlib')]:
        path = output/folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ('SIGTERM', 'SIGUSR1'):
        signal.signal(getattr(signal, name), stop_signal)


def import_source(plan, key):
    source = Path(plan['bundle'])/key/'source'
    # One model per fresh process: never mix two historical namespaces.
    for name in ('part_ev_fullscale', 'part_ev_state_compare', 'part_ev_screen'):
        module = sys.modules.get(name)
        if module and not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError('Another model source namespace is already imported')
    sys.path.insert(0, str(source))


def environment_check(plan):
    expected = C.checked(Path(plan['bundle'])/'bundle.json')['environment']
    actual = environment()
    # Python patch release can differ; the minor version and scientific dependencies cannot.
    left, right = dict(expected), dict(actual)
    left['packages'], right['packages'] = dict(left['packages']), dict(right['packages'])
    # Conda and pip may label the same PyTorch release differently. CUDA version
    # is checked separately and actual full build strings stay in runtime evidence.
    for value in (left, right):
        value['packages']['torch'] = value['packages']['torch'].split('+')[0]
    left['python'] = '.'.join(left['python'].split('.')[:2])
    right['python'] = '.'.join(right['python'].split('.')[:2])
    if left != right:
        differences = {k: dict(expected=left.get(k), actual=right.get(k))
                       for k in set(left)|set(right) if left.get(k) != right.get(k)}
        raise ValueError(f'Oscar dependencies differ from bundle: {differences}')
    return actual


def verify_test_archive(archive, directory, expected_sha=C.ARCHIVE_SHA):
    """Check every extracted file against the approved tar bytes, not just filenames."""
    archive, directory = Path(archive), Path(directory)
    if C.sha(archive) != expected_sha:
        raise ValueError('Test archive SHA-256 differs from the locally verified upload')
    expected = C.expected_files()
    if directory.is_symlink() or {p.name for p in directory.iterdir()} != set(expected):
        raise ValueError('Test directory must contain exactly the official 200 ROOT files')
    records, seen = [], set()
    import uproot
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            stopping()
            if member.name.rstrip('/') == 'test_20M' and member.isdir():
                continue
            relative = member.name.removeprefix('test_20M/')
            if (member.name != 'test_20M/'+relative or relative not in expected
                    or not member.isfile() or relative in seen):
                raise ValueError(f'Unexpected/duplicate tar member: {member.name}')
            seen.add(relative)
            path = directory/relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != member.size:
                raise ValueError(f'Wrong extracted file type/size: {path}')
            checksum = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    checksum.update(block)
            if C.sha(path) != checksum.hexdigest():
                raise ValueError(f'Extracted ROOT bytes differ: {path}')
            with uproot.open(path) as f:
                if f['tree'].classname != 'TTree' or f['tree'].num_entries != 100_000:
                    raise ValueError(f'Unexpected test TTree/event count: {path}')
            r = record(path, 100_000)
            r.update(sha256=checksum.hexdigest(), label=expected[relative])
            records.append(r)
            print(f'Verified test file {len(records)}/200: {relative}', flush=True)
    if seen != set(expected):
        raise ValueError('Incomplete archive population')
    return sorted(records, key=lambda r: r['name'])


def prepare(output):
    output = Path(output)
    plan = C.load_plan(output)
    env = environment_check(plan)
    with C.lock(output/'.prepare.lock'):
        if (output/'inventory.json').exists():
            inventory(output, plan, hash_files=True)
            print('Existing inventory verified; no changes.', flush=True)
            return
        rows = verify_test_archive(plan['archive'], plan['test_directory'])
        C.write(output/'inventory.json', C.seal(dict(plan_sha256=plan['content_hash'],
                archive_sha256=C.ARCHIVE_SHA, events=20_000_000, files=rows, environment=env)))


def inventory(output, plan, hash_files=False):
    row = C.checked(Path(output)/'inventory.json')
    expected = C.expected_files()
    files = row['files']
    if (row['plan_sha256'] != plan['content_hash'] or row['events'] != 20_000_000
            or row['archive_sha256'] != C.ARCHIVE_SHA or len(files) != 200
            or [r['name'] for r in files] != sorted(expected)):
        raise ValueError('Wrong test inventory coverage/identity')
    for r in files:
        path = Path(plan['test_directory'])/r['name']
        if (r['path'] != str(path) or r['num_entries'] != 100_000 or r['label'] != expected[r['name']]
                or path.is_symlink() or path.stat().st_size != r['size_bytes']
                or path.stat().st_mtime_ns != r['mtime_ns']
                or (hash_files and C.sha(path) != r['sha256'])):
            raise ValueError(f'Test file changed: {path}')
    return row


def load_model(plan, key):
    env = environment_check(plan)
    import_source(plan, key)
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or int(os.environ.get('WORLD_SIZE', '1')) != 1):
        raise RuntimeError('Request exactly one visible CUDA GPU; no DDP for inference')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    from part_ev_fullscale.checkpoint import seed_everything
    seed_everything(101)
    directory = Path(plan['bundle'])/key
    manifest = C.checked(directory/'model.json')
    model = build_architecture(key)
    if C.sha(directory/'weights.pt') != manifest['weights_sha256']:
        raise ValueError('Portable weights changed')
    saved = torch.load(directory/'weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != C.COUNTS[key]:
        raise ValueError('Model capacity changed')
    device = torch.device('cuda:0')
    model.to(device).eval()
    for module in model.modules():
        if hasattr(module, '_counter'):
            module._counter = 5
    prop = torch.cuda.get_device_properties(0)
    import subprocess
    drivers = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    runtime = dict(environment=env, gpu=prop.name, capability=[prop.major, prop.minor],
                   driver_versions=sorted(set(drivers)),
                   precision='FP32', batch_size=plan['batch_size'], tf32=False,
                   world_size=1, batchnorm='native' if key == 'part' else 'SyncBatchNorm_eval')
    return model, device, manifest, runtime


def build_architecture(key):
    """Called only after selecting the donor source (also CPU-testable)."""
    import torch
    if key == 'part':
        from part_ev_state_fullscale.model import build_model
        model = build_model('part', seed=101, trim=True, use_amp=False)
    elif key == 'state':
        from part_ev_state_compare.model import build_model
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            build_model('state_layerwise', seed=101, trim=True, use_amp=False))
    else:
        raise ValueError('Unknown architecture')
    return model


def loader(file_record, batch_size):
    from part_ev_fullscale.data import build_loader
    inv = {'splits': {'test': {'files': [file_record]}}}
    result, _ = build_loader('test', inv, batch_size, workers=0, seed=101)
    return result


def forward_file(model, device, file_record, batch_size, *, capture=False):
    import numpy as np
    import torch
    from part_ev_screen.precision import model_precision, audit_precision
    scores, labels, inputs = [], [], {name: [] for name in C.INPUTS}
    precision = None
    with torch.no_grad(), model_precision(model, device, training=False):
        for i, batch in enumerate(loader(file_record, batch_size)):
            stopping()
            values = [batch[0][name].to(device) for name in C.INPUTS]
            if capture:
                for name in C.INPUTS:
                    inputs[name].append(batch[0][name].numpy().copy())
            audit = audit_precision(model, expected_dtype=torch.float32) if i == 0 else nullcontext()
            with audit as observed:
                prediction = model(*values).float().cpu().numpy()
            if i == 0:
                precision = observed
            scores.append(prediction)
            labels.append(batch[1]['_label_'].numpy().astype(np.int64))
    if not scores or not precision or not precision['passed']:
        raise ValueError('Empty data or failed FP32 audit')
    logits, truth = np.concatenate(scores), np.concatenate(labels)
    C.validate_arrays(logits, truth, file_record['num_entries'], file_record.get('label'))
    return logits, truth, precision, {k: np.concatenate(v) for k, v in inputs.items()} if capture else {}


def probe(plan, key, model, device, manifest, runtime):
    import numpy as np
    import torch
    rows = []
    directory = Path(plan['bundle'])/key
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for p in manifest['probes']:
        r = record(directory/p['root'], 512)
        r['label'] = p['label']
        t = time.monotonic()
        logits, labels, precision, inputs = forward_file(model, device, r, plan['batch_size'], capture=True)
        with np.load(directory/p['reference'], allow_pickle=False) as expected:
            checks = {name: C.comparison(inputs[name], expected[name], inputs=True) for name in C.INPUTS}
            checks['labels'] = dict(passed=bool(np.array_equal(labels, expected['labels'])))
            checks['logits'] = C.comparison(logits, expected['logits'])
        rows.append(dict(file=p['original_name'], checks=checks, precision=precision,
                         seconds=time.monotonic()-t, passed=all(v['passed'] for v in checks.values())))
    return C.seal(dict(plan_sha256=plan['content_hash'], model=key, runtime=runtime,
                       policy=C.PARITY, passed=len(rows) == 10 and all(r['passed'] for r in rows),
                       events=5120, rows=rows, seconds=time.monotonic()-started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                       note='Validation parity only. Timing includes tiny-file preprocessing; not a full-test bound.'))


def parity(output, key):
    output = Path(output)
    plan = C.load_plan(output)
    inventory(output, plan)  # prepare dependency must have completed
    with C.lock(output/f'.parity_{key}.lock'):
        target = output/f'parity_{key}.json'
        if target.exists():
            raise FileExistsError('Parity evidence already exists; inspect it, do not overwrite')
        model, device, manifest, runtime = load_model(plan, key)
        result = probe(plan, key, model, device, manifest, runtime)
        C.write(target, result)
        if not result['passed']:
            raise ValueError('Oscar/RIT validation parity failed; no test evaluation allowed')
        print(f'{key}: all ten validation probes passed', flush=True)


def chunk_contract(plan, inv, key, file_record, runtime):
    return dict(plan_sha256=plan['content_hash'], inventory_sha256=inv['content_hash'],
                model=key, file=file_record, entry_start=0, entry_stop=file_record['num_entries'],
                runtime=runtime)


def read_chunk(path, contract):
    import numpy as np
    path = Path(path)
    side = C.checked(path.with_suffix('.json'))
    if side['contract'] != contract or side['sha256'] != C.sha(path):
        raise ValueError('Saved prediction contract/hash mismatch')
    if side['precision'].get('passed') is not True or side['precision'].get('expected_activation_dtype') != 'torch.float32':
        raise ValueError('Missing FP32 evidence')
    with np.load(path, allow_pickle=False) as arrays:
        logits, labels = arrays['logits'], arrays['labels']
    C.validate_arrays(logits, labels, contract['entry_stop'], contract['file']['label'])
    if side['labels_sha256'] != hashlib.sha256(labels.tobytes()).hexdigest():
        raise ValueError('Saved labels changed')
    return logits, labels, side


def evaluate(output, key):
    from .results import require_admission
    output = Path(output)
    plan = C.load_plan(output)
    inv = inventory(output, plan)
    require_admission(output, plan)
    directory = output/key
    directory.mkdir(exist_ok=True)
    with C.lock(directory/'.evaluate.lock'):
        if (directory/'inference_complete.json').exists():
            # Never trust the marker alone; CPU metric path audits all chunks.
            from .results import verified_predictions
            for _ in verified_predictions(output, plan, key):
                pass
            print(f'{key}: complete predictions verified', flush=True)
            return
        model, device, manifest, runtime = load_model(plan, key)
        admission = C.checked(output/f'parity_{key}.json')
        if runtime != admission['runtime']:
            raise ValueError('GPU/dependency/batch runtime differs from admitted parity; use a new plan')
        # Recheck on the actual evaluation node, without changing the frozen admission.
        evidence = probe(plan, key, model, device, manifest, runtime)
        if not evidence['passed']:
            raise ValueError('Worker validation parity failed; test data not inferred')
        receipts = []
        started = time.monotonic()
        for i, r in enumerate(inv['files']):
            stopping()
            if C.sha(r['path']) != r['sha256']:
                raise ValueError('Test ROOT content changed after preparation')
            path = directory/f'file_{i:03d}.npz'
            sidecar = path.with_suffix('.json')
            contract = chunk_contract(plan, inv, key, r, runtime)
            if sidecar.exists() and not path.exists():
                raise ValueError('Committed prediction file missing')
            reused = sidecar.exists()
            if not reused:
                logits, labels, precision, _ = forward_file(model, device, r, plan['batch_size'])
                from .results import atomic_npz
                atomic_npz(path, logits=logits, labels=labels)
                C.write(sidecar, C.seal(dict(contract=contract, sha256=C.sha(path),
                        labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(), precision=precision)))
            _, _, receipt = read_chunk(path, contract)
            receipts.append(receipt['content_hash'])
            print(f'{key}: {i+1}/200 files, {(i+1)*100000:,}/20,000,000 jets; reused={reused}', flush=True)
        C.write(directory/'inference_complete.json', C.seal(dict(plan_sha256=plan['content_hash'],
                inventory_sha256=inv['content_hash'], model=key, runtime=runtime,
                events=20_000_000, receipts=receipts, worker_parity=evidence,
                seconds=time.monotonic()-started)))
