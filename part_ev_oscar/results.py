"""Full-population CPU metrics and integrity-checked reports, never batch averages."""
import math
import os
from pathlib import Path
import tempfile
from . import common as C


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def admission_record(output, plan):
    from .runtime import inventory
    inv = inventory(output, plan)
    checks = {}
    for key in C.SELECTED:
        row = C.checked(Path(output)/f'parity_{key}.json')
        if (row['plan_sha256'] != plan['content_hash'] or row['model'] != key
                or row['policy'] != C.PARITY or row['passed'] is not True
                or row['events'] != 5120 or len(row['rows']) != 10
                or not all(r['passed'] for r in row['rows'])):
            raise ValueError('Both complete successful validation parity reports are required')
        checks[key] = row['content_hash']
    return C.seal(dict(plan_sha256=plan['content_hash'], inventory_sha256=inv['content_hash'],
                       parity=checks, checkpoints_fixed_before_test=True))


def require_admission(output, plan):
    expected = admission_record(output, plan)
    if C.checked(Path(output)/'admission.json') != expected:
        raise ValueError('Test admission changed')
    return expected


def verified_predictions(output, plan, key):
    from .runtime import inventory, read_chunk, chunk_contract
    output = Path(output)
    require_admission(output, plan)
    inv = inventory(output, plan)
    done = C.checked(output/key/'inference_complete.json')
    parity = C.checked(output/f'parity_{key}.json')
    if (done['plan_sha256'] != plan['content_hash'] or done['model'] != key
            or done['inventory_sha256'] != inv['content_hash'] or done['events'] != 20_000_000
            or len(done['receipts']) != 200 or done['runtime'] != parity['runtime']
            or done['worker_parity']['passed'] is not True):
        raise ValueError('Inference completion contract differs')
    for i, r in enumerate(inv['files']):
        contract = chunk_contract(plan, inv, key, r, done['runtime'])
        logits, labels, side = read_chunk(output/key/f'file_{i:03d}.npz', contract)
        if done['receipts'][i] != side['content_hash']:
            raise ValueError('Completion receipt differs from saved predictions')
        yield logits, labels, side


def calculate(logits, truth):
    import numpy as np
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score
    C.validate_arrays(logits, truth, len(truth))
    counts = np.bincount(truth, minlength=10)
    if np.any(counts == 0):
        raise ValueError('Need all ten classes for macro AUC')
    probabilities = softmax(logits.astype(np.float64), axis=1)
    result = dict(events=len(truth), accuracy=float((logits.argmax(1) == truth).mean()),
        class_support=dict(zip(C.CLASSES, counts.tolist())),
        macro_auc_ovo=float(roc_auc_score(truth, probabilities, multi_class='ovo', average='macro')),
        macro_auc_ovr=float(roc_auc_score(truth, probabilities, multi_class='ovr', average='macro')),
        qcd_rejection={})
    del probabilities
    for label, name in enumerate(C.CLASSES[1:], 1):
        score = logits[:, label].astype(np.float64) - logits[:, 0].astype(np.float64)
        signal = np.sort(score[truth == label])
        background = score[truth == 0]
        efficiencies = (0.3, 0.5) + ((0.99,) if name == 'Hqql' else (0.995,) if name == 'Tbl' else ())
        rows = {}
        for efficiency in efficiencies:
            threshold = float(signal[-math.ceil(efficiency*len(signal))])
            accepted = int(np.count_nonzero(background >= threshold))
            passed = int(np.count_nonzero(signal >= threshold))
            rows[str(efficiency)] = dict(target_signal_efficiency=efficiency,
                achieved_signal_efficiency=passed/len(signal), signal_accepted=passed,
                signal_total=len(signal), threshold_log_odds=threshold, qcd_accepted=accepted,
                qcd_total=len(background), rejection=len(background)/accepted if accepted else None)
        result['qcd_rejection'][name] = rows
    return result


def metrics(output, key):
    import numpy as np
    from .runtime import environment_check
    output = Path(output)
    plan = C.load_plan(output)
    environment_check(plan)
    with C.lock(output/key/'.metrics.lock'):
        logits, labels, receipts = [], [], []
        for scores, truth, side in verified_predictions(output, plan, key):
            logits.append(scores)
            labels.append(truth)
            receipts.append(side['content_hash'])
        scores, truth = np.concatenate(logits), np.concatenate(labels)
        del logits, labels
        if len(truth) != 20_000_000 or not np.array_equal(np.bincount(truth, minlength=10), np.full(10, 2_000_000)):
            raise ValueError('Not the full balanced 20M test population')
        path = output/key/'metrics.json'
        if path.exists():
            saved = C.checked(path)
            if saved['plan_sha256'] != plan['content_hash'] or saved['receipts'] != receipts:
                raise ValueError('Existing metrics provenance differs')
            print(f'{key}: existing full-test metrics verified', flush=True)
            return
        manifest = C.checked(Path(plan['bundle'])/key/'model.json')
        values = calculate(scores, truth)
        C.write(path, C.seal(dict(plan_sha256=plan['content_hash'], model=key,
                checkpoint_sha256=manifest['checkpoint_sha256'], best_check=manifest['best_check'],
                parameters=manifest['parameters'], training_runtime=manifest['training_runtime'],
                population='official_test_20M', independent_test=True, metrics=values,
                receipts=receipts, selection='fixed_highest_validation_accuracy_first_tie')))
        print(f"{key}: test accuracy {100*values['accuracy']:.4f}%, macro OVO AUC {values['macro_auc_ovo']:.6f}", flush=True)


def table(rows):
    lines = ['# Full 20M JetClass test: ParT versus layerwise State-EV', '',
             'Seed 101; checkpoints selected on validation before test evaluation.',
             'ParT trained on one GPU/native BatchNorm; State-EV trained on two GPUs/SyncBatchNorm.',
             'Single-seed comparison; training-runtime differences are not isolated architecture effects.', '',
             '| Model | Selected check | Parameters | Accuracy (%) | Macro OVO AUC | Macro OVR AUC |',
             '|---|---:|---:|---:|---:|---:|']
    for key in C.SELECTED:
        row, m = rows[key], rows[key]['metrics']
        lines.append(f"| {'ParT' if key == 'part' else 'State-EV layerwise'} | {row['best_check']} | "
                     f"{row['parameters']:,} | {100*m['accuracy']:.4f} | {m['macro_auc_ovo']:.6f} | {m['macro_auc_ovr']:.6f} |")
    p, s = rows['part']['metrics'], rows['state']['metrics']
    lines += ['', f"Accuracy change: {100*(s['accuracy']-p['accuracy']):+.4f} percentage points.",
              f"Macro OVO AUC change: {s['macro_auc_ovo']-p['macro_auc_ovo']:+.6f} (0–1 scale).", '',
              '| Signal | Efficiency | ParT rejection | State-EV rejection | Change (%) | QCD accepted P/S |',
              '|---|---:|---:|---:|---:|---:|']
    changes = {}
    for name, targets in p['qcd_rejection'].items():
        changes[name] = {}
        for efficiency, a in targets.items():
            b = s['qcd_rejection'][name][efficiency]
            rp, rs = a['rejection'], b['rejection']
            change = None if rp is None or rs is None else 100*(rs/rp-1)
            changes[name][efficiency] = change
            fmt = lambda x: 'ZB' if x is None else f'{x:,.2f}'
            delta = '--' if change is None else f'{change:+.2f}'
            label = {'Hqql': 'Hlvqq (Hqql)', 'Tbl': 'Tblv (Tbl)'}.get(name, name)
            lines.append(f"| {label} | {100*float(efficiency):g}% | {fmt(rp)} | {fmt(rs)} | {delta} | {a['qcd_accepted']}/{b['qcd_accepted']} |")
    lines += ['', 'Rejection = N_QCD / accepted QCD at the stated signal efficiency (ties included).',
              'Thresholds are test-ROC operating points, not deployment thresholds selected on validation.',
              'Positive change means higher rejection; ZB means zero observed QCD acceptances, not measured infinity.',
              'Counts and achieved efficiencies are retained in JSON; small tail counts can give unstable percentage changes.']
    return '\n'.join(lines)+'\n', changes


def report(output):
    output = Path(output)
    plan = C.load_plan(output)
    rows, label_hashes = {}, {}
    with C.lock(output/'.report.lock'):
        for key in C.SELECTED:
            row = C.checked(output/key/'metrics.json')
            if (row['plan_sha256'] != plan['content_hash'] or row['model'] != key
                    or row['population'] != 'official_test_20M' or row['metrics']['events'] != 20_000_000):
                raise ValueError('Metrics identity/coverage differs')
            hashes, labels = [], []
            for _, _, side in verified_predictions(output, plan, key):
                hashes.append(side['content_hash'])
                labels.append(side['labels_sha256'])
            if hashes != row['receipts']:
                raise ValueError('Metrics not bound to these predictions')
            rows[key], label_hashes[key] = row, labels
        if label_hashes['part'] != label_hashes['state']:
            raise ValueError('Models do not have identical event labels/order')
        text, changes = table(rows)
        C.write(output/'report.json', C.seal(dict(plan_sha256=plan['content_hash'], models=rows,
                independent_test=True, rejection_change_percent=changes,
                accuracy_delta_pp=100*(rows['state']['metrics']['accuracy']-rows['part']['metrics']['accuracy']),
                macro_auc_ovo_delta=rows['state']['metrics']['macro_auc_ovo']-rows['part']['metrics']['macro_auc_ovo'])))
        (output/'report.md').write_text(text, encoding='utf-8')
        print(text, flush=True)
