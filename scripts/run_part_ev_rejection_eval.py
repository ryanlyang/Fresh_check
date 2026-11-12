#!/usr/bin/env python3
"""Evaluation-only adapter. Donor source, checkpoints and approvals stay read-only."""
import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part_ev_rejection_eval_support as S


def bootstrap(output, key, plan):
    donor = plan['donors'][key]
    S.verify_pins(donor)
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('NCCL_') or name in (
                'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
                'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'):
            del env[name]
    spec = S.read(Path(donor['root']) / 'campaign.json')
    if key == 'state':
        env.update(spec['expected_gpu']['nccl_environment'])
    binary = Path(donor['python'])
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(Path(donor['root']) / 'source'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    env['PATH'] = str(binary.parent) + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(binary.parent.parent / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    os.execve(str(binary), [str(binary), '-B', str(Path(__file__).resolve()),
              '--phase', 'evaluate', '--output', str(output), '--model', key], env)


def load_runtime(donor):
    """Import exactly one donor namespace in this fresh process; no source patching."""
    source = Path(donor['root']) / 'source'
    sys.path.insert(0, str(source))
    root = Path(donor['root'])
    key = donor['key']
    if key == 'part':
        from scripts import run_part_ev_state_fullscale as original
        P, D = original.P, original.D
        spec = P.load_campaign(root)
        P.verify_inventory_metadata(spec)
        P.require_approval(root, spec)
        device, runtime = D.gpu_setup(spec)
        P.benchmark_summary(root, spec, runtime)
        from part_ev_state_fullscale.reporting import validate_training
        from part_ev_message_study.controls import runtime_identity
        audited = validate_training(root, spec, donor['task'], runtime_identity(runtime))
        model = D.make_model(donor['task'], device).eval()
        stop = D.check_stop
    else:
        from part_ev_layerwise_ddp import protocol as P
        spec = P.load(root)
        P.G.C.verify_inventory_metadata(spec)
        P.require_approval(root, spec)
        from part_ev_layerwise_ddp import engine as E
        from scripts import run_part_ev_screen as D
        _, _, device = E.B.setup(1)
        runtime = E.runtime_record(spec, device, 1)
        audited = E.audit_training(root, spec)
        D.seed_everything(donor['task']['seed'])
        model = D.torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            E.B.build_model('state_layerwise', use_amp=False)).to(device).eval()
        def stop():
            if E.B.STOP:
                raise SystemExit(75)
    # Strong donor audit includes best weights, all 50 checks, optimizer metadata,
    # source/environment and original admission. Unrelated runs are not required.
    if (audited['checkpoint_sha256'] != donor['checkpoint_sha256']
            or audited['best_check'] != donor['best_check']):
        raise ValueError('Selected training checkpoint changed')
    saved = D.trusted_load(Path(donor['run']) / 'best_model_val.pt')
    model.load_state_dict(saved['model'], strict=True)
    expected_count = spec['parameters'][donor['task']['architecture']]
    if sum(p.numel() for p in model.parameters()) != expected_count:
        raise ValueError('Model parameter count changed')
    for module in model.modules():
        if hasattr(module, '_counter'):
            module._counter = 5
    return D, spec, model, device, runtime, stop


def chunk_contract(plan, donor, record):
    return dict(plan_sha256=plan['content_hash'], campaign_sha256=donor['campaign_sha256'],
                checkpoint_sha256=donor['checkpoint_sha256'], file=record,
                entry_start=0, entry_stop=record['num_entries'], precision='FP32')


def read_chunk(path, contract):
    import numpy as np
    sidecar = path.with_suffix('.json')
    saved = S.read(sidecar)
    if saved['contract'] != contract or saved['sha256'] != S.sha(path):
        raise ValueError(f'Prediction contract/hash mismatch: {path}')
    if (saved['precision']['passed'] is not True
            or saved['precision']['expected_activation_dtype'] != 'torch.float32'):
        raise ValueError('Missing observed FP32 inference evidence')
    with np.load(path, allow_pickle=False) as arrays:
        logits, labels = arrays['logits'], arrays['labels']
    S.validate_arrays(logits, labels, contract['entry_stop'])
    if saved['labels_sha256'] != hashlib.sha256(labels.tobytes()).hexdigest():
        raise ValueError('Prediction labels changed')
    return logits, labels, saved


def evaluate_file(D, spec, model, device, record, path, contract, stop):
    """File-level commit boundary: reuse exact predictions or recompute an orphan."""
    import numpy as np
    sidecar = path.with_suffix('.json')
    if sidecar.exists() and not path.exists():
        raise ValueError('Committed prediction file is missing; inspect before retrying')
    reused = path.exists() and sidecar.exists()
    if not reused:
        scores, truth = [], []
        with D.torch.no_grad(), D.model_precision(model, device, training=False):
            for batch_index, batch in enumerate(D.finite_validation_loader(spec['inventory'], record)):
                stop()
                values, y = D.tensors(batch, device)
                audit = (D.audit_precision(model, expected_dtype=D.torch.float32)
                         if batch_index == 0 else nullcontext())
                with audit as observed:
                    logits = model(*values).float().cpu().numpy()
                if batch_index == 0:
                    precision = observed
                if not np.isfinite(logits).all():
                    raise ValueError('Nonfinite inference output')
                scores.append(logits)
                truth.append(y.cpu().numpy())
        if not scores:
            raise ValueError('Empty validation file')
        logits, y = np.concatenate(scores), np.concatenate(truth)
        S.validate_arrays(logits, y, record['num_entries'])
        D.atomic_npz(path, logits=logits, labels=y)
        S.write(sidecar, dict(contract=contract, sha256=S.sha(path),
                             labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                             precision=precision))
    return (*read_chunk(path, contract), reused)


def evaluate(output, key, plan):
    # Workers also guard against a legacy chain being resumed after submission.
    S.refuse_active(plan['donors'])
    donor = plan['donors'][key]
    S.verify_pins(donor)
    S.verify_frozen_source(donor)
    directory = output / key
    directory.mkdir(exist_ok=True)
    import fcntl
    with (directory / '.evaluation.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        D, spec, model, device, runtime, stop = load_runtime(donor)
        import numpy as np
        from part_ev_fullscale.protocol import CLASS_NAMES
        predictions, labels, records = [], [], []
        started = time.monotonic()
        for i, record in enumerate(donor['files']):
            stop()
            path = directory / f'file_{i:03d}.npz'
            contract = chunk_contract(plan, donor, record)
            logits, y, receipt, reused = evaluate_file(D, spec, model, device, record, path, contract, stop)
            predictions.append(logits)
            labels.append(y)
            records.append(receipt)
            D.progress('rejection_validation_file', model=key, completed=i+1, total=50,
                       events=(i+1)*100_000, reused=reused, seconds=time.monotonic()-started)
        stop()
        logits, y = np.concatenate(predictions), np.concatenate(labels)
        if len(y) != 5_000_000 or not np.array_equal(np.bincount(y, minlength=10), np.full(10, 500_000)):
            raise ValueError('Expected full 5M validation, 500k events per class')
        # Avoid multi-class AUC: user requested rejection, which needs only log odds.
        metrics = S.rejection_metrics(logits, y, CLASS_NAMES)
        S.verify_pins(donor)
        row = S.seal(dict(model=key, plan_sha256=plan['content_hash'],
            campaign_sha256=donor['campaign_sha256'], checkpoint_sha256=donor['checkpoint_sha256'],
            best_check=donor['best_check'], task=donor['task'], identity_manifest=records,
            evaluation_runtime=dict(gpu=D.torch.cuda.get_device_name(device), world_size=1,
                                    precision='FP32', batch_size=512),
            training_runtime=donor['training_runtime'], metrics=metrics,
            population='official_validation_5M', independent_test=False,
            seconds=time.monotonic()-started))
        S.write(directory / 'metrics.json', row)
        print(S.comparison_text({key: row}), flush=True)


def report(output, plan, persist=False):
    rows = {}
    for key, donor in plan['donors'].items():
        path = output / key / 'metrics.json'
        if not path.exists():
            print(f'{key}: {len(list((output/key).glob("file_*.json")))}/50 chunk sidecars saved')
            continue
        row = S.checked(S.read(path))
        if (row['plan_sha256'] != plan['content_hash'] or row['model'] != key
                or row['checkpoint_sha256'] != donor['checkpoint_sha256']
                or row['metrics']['events'] != 5_000_000 or row['best_check'] != donor['best_check']):
            raise ValueError('Result identity/coverage mismatch')
        if len(row['identity_manifest']) != 50:
            raise ValueError('Incomplete result manifest')
        for i, (record, receipt) in enumerate(zip(donor['files'], row['identity_manifest'])):
            pred = output / key / f'file_{i:03d}.npz'
            if (receipt['contract'] != chunk_contract(plan, donor, record)
                    or S.read(pred.with_suffix('.json')) != receipt or S.sha(pred) != receipt['sha256']):
                raise ValueError('Stored predictions/coverage changed')
        rows[key] = row
    if len(rows) == 2:
        # Metadata identifies exact same files/entries; both evaluations validated labels/support.
        if plan['donors']['part']['files'] != plan['donors']['state']['files']:
            raise ValueError('Comparison populations differ')
        if ([r['labels_sha256'] for r in rows['part']['identity_manifest']] !=
                [r['labels_sha256'] for r in rows['state']['identity_manifest']]):
            raise ValueError('Prediction event labels differ between models')
    text = S.comparison_text(rows)
    print(text)
    if persist:
        if len(rows) != 2:
            raise ValueError('Both evaluations must complete before the final report')
        S.write(output / 'report.json', S.seal(dict(plan_sha256=plan['content_hash'], models=rows,
                                                  independent_test=False)))
        (output / 'report.md').write_text(text + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('worker', 'evaluate', 'report', 'status'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=('part', 'state'))
    args = parser.parse_args(argv)
    output = args.output.resolve(strict=True)
    plan = S.verify_snapshot(output)
    if args.phase in ('worker', 'evaluate'):
        if args.model is None:
            parser.error('--model required')
        if args.phase == 'worker':
            bootstrap(output, args.model, plan)
        else:
            evaluate(output, args.model, plan)
    else:
        report(output, plan, persist=args.phase == 'report')


if __name__ == '__main__':
    main()
