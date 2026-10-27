"""New output namespace; reused controls and their frozen metadata stay read-only."""
from pathlib import Path
import os
import shutil
from part_ev_state_fullscale.engine import create_engine as full_engine
from part_ev_state_fullscale.reporting import validate_training
from part_ev_message_study.controls import runtime_identity
from . import protocol as P, model


def training_origin(output, spec, index):
    if index in P.TRAIN_INDICES:
        return Path(output), spec, spec['tasks'][index]
    root = Path(spec['controls']['path'])
    original = P.read_json(root / 'campaign.json')
    task = original['tasks'][P.CONTROL_INDICES[index - len(P.TRAIN_INDICES)]]
    if original['content_hash'] != spec['controls']['campaign_sha256'] or task != spec['tasks'][index]:
        raise ValueError('Control task identity changed')
    return root, original, task


def audited_training(output, spec, index):
    root, source, task = training_origin(output, spec, index)
    record = validate_training(root, source, task, spec['controls']['runtime'])
    return root, source, record


def lock_finalists(output, spec):
    output = Path(output)
    with P.file_lock(output / '.selection.lock'):
        # Audit ALL five first; do not partially import unfinished controls.
        records = [audited_training(output, spec, i) for i in range(len(P.TASKS))]
        rows = []
        for i, (root, source, trained) in enumerate(records):
            task = spec['tasks'][i]
            origin = root / 'runs' / task['run_id'] / f"seed_{task['seed']}" / 'best_model_val.pt'
            checkpoint = P.task_directory(output, i) / 'best_model_val.pt'
            checksum = trained['checkpoint_sha256']
            if i not in P.TRAIN_INDICES:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                if not checkpoint.exists():
                    temp = checkpoint.with_suffix(f'.import.{os.getpid()}.tmp')
                    shutil.copyfile(origin, temp)
                    if P.file_sha256(temp) != checksum:
                        raise ValueError('Control checkpoint changed during byte-identical import')
                    os.replace(temp, checkpoint)
            if P.file_sha256(checkpoint) != checksum:
                raise ValueError('Selected/imported checkpoint changed')
            rows.append(dict(task=task, checkpoint=str(checkpoint.resolve()), sha256=checksum,
                training_root=str(root.resolve()), training_campaign_sha256=source['content_hash'],
                training_runtime=trained['training']['runtime'],
                training_artifact_sha256=trained['artifact_sha256'], reused=i not in P.TRAIN_INDICES))
        payload = dict(campaign_sha256=spec['content_hash'], checkpoints=rows)
        path = output / 'selection/locked_checkpoints.json'
        if path.exists():
            if P.read_json(path) != payload:
                raise ValueError('Finalist identities or training evidence changed')
        else:
            P.write_json(path, payload)
        return payload


def create_engine():
    engine = full_engine()
    engine.P = P
    engine.build_model = model.build_model
    engine.diagnostic_mode = model.diagnostic_mode
    engine.diagnostics = model.diagnostics
    engine.lock_finalists = lock_finalists
    return engine


def evaluate(args, engine):
    """Same finite-file FP32 evaluation, with explicit original checkpoint identity."""
    np, torch = engine.np, engine.torch
    spec = P.load_campaign(args.output)
    task = spec['tasks'][args.task_index]
    device, runtime = engine.gpu_setup(spec)
    P.benchmark_summary(args.output, spec, runtime)
    selected = lock_finalists(args.output, spec)['checkpoints'][args.task_index]
    directory = P.task_directory(args.output, args.task_index) / 'full_validation'
    with P.file_lock(directory / '.evaluation.lock', blocking=False):
        candidate = engine.make_model(task, device).eval()
        saved = engine.trusted_load(selected['checkpoint'])
        if (saved['task'] != task or saved['runtime'] != selected['training_runtime']
                or saved['campaign_sha256'] != selected['training_campaign_sha256']
                or runtime_identity(saved['runtime']) != runtime_identity(runtime)):
            raise ValueError('Selected checkpoint origin/runtime mismatch')
        candidate.load_state_dict(saved['model'], strict=True)
        for module in candidate.modules():
            if hasattr(module, '_counter'):
                module._counter = 5
        predictions, labels, records = [], [], []
        files = sorted(spec['inventory']['splits']['val']['files'], key=lambda r: r['name'])
        for i, record in enumerate(files):
            engine.check_stop()
            path = directory / f'file_{i:03d}.npz'
            sidecar = path.with_suffix('.json')
            contract = dict(campaign_sha256=spec['content_hash'], checkpoint_sha256=selected['sha256'],
                            file=record, entry_start=0, entry_stop=record['num_entries'])
            if path.exists() and sidecar.exists():
                committed = P.read_json(sidecar)
                if committed['contract'] != contract or committed['sha256'] != P.file_sha256(path):
                    raise ValueError('Committed validation prediction changed')
            else:
                scores, truth = [], []
                loader = engine.finite_validation_loader(spec['inventory'], record)
                with torch.no_grad(), engine.model_precision(candidate, device, training=False):
                    for batch in loader:
                        engine.check_stop()
                        inputs, y = engine.tensors(batch, device)
                        values = candidate(*inputs).float().cpu().numpy()
                        if not np.isfinite(values).all():
                            raise FloatingPointError('Nonfinite validation logits')
                        scores.append(values)
                        truth.append(y.cpu().numpy())
                score, y = np.concatenate(scores), np.concatenate(truth)
                if len(y) != record['num_entries']:
                    raise ValueError('Incomplete validation file')
                engine.atomic_npz(path, logits=score, labels=y)
                P.write_json(sidecar, dict(contract=contract, sha256=P.file_sha256(path)))
            with np.load(path, allow_pickle=False) as data:
                predictions.append(data['logits'])
                labels.append(data['labels'])
            records.append(P.read_json(sidecar))
            engine.progress('full_validation_file', **task, completed=i+1, total=len(files))
        logits, truth = np.concatenate(predictions), np.concatenate(labels)
        if (len(truth) != 5000000
                or not np.array_equal(np.bincount(truth, minlength=10), np.full(10, 500000))):
            raise ValueError('Expected all 5M validation events, 500k/class')
        metrics = engine.classification_metrics(logits, truth, rejection=True)
        P.write_json(directory / 'metrics.json', dict(task=task, campaign_sha256=spec['content_hash'],
            checkpoint_sha256=selected['sha256'], identity_manifest=records, metrics=metrics))
        engine.progress('full_validation_complete', **task, accuracy=metrics['accuracy'])
