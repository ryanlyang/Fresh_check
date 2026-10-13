"""Full-population metrics, not averages of batch/file metrics."""
from __future__ import annotations

import math
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from .protocol import CLASS_NAMES


def calculate_metrics(logits, truth):
    logits, truth = np.asarray(logits), np.asarray(truth)
    if logits.shape != (len(truth), 10) or not np.isfinite(logits).all():
        raise ValueError("Need finite [events,10] logits")
    support = np.bincount(truth.astype(np.int64), minlength=10)
    if len(support) != 10 or np.any(support == 0):
        raise ValueError("Need all ten classes")
    probabilities = softmax(logits.astype(np.float64), axis=1)
    metrics = {"events": len(truth), "class_support": dict(zip(CLASS_NAMES, support.tolist())),
               "accuracy": float((logits.argmax(1) == truth).mean()),
               # Original ParT/Weaver macro AUC is one-versus-one, not OVR.
               "macro_auc_ovo": float(roc_auc_score(truth, probabilities, multi_class="ovo", average="macro")),
               "macro_auc_ovr": float(roc_auc_score(truth, probabilities, multi_class="ovr", average="macro")),
               "qcd_rejection": {}}
    for label, name in enumerate(CLASS_NAMES[1:], 1):
        # Monotonic log odds exactly implements p_signal/(p_signal+p_QCD),
        # without saturation from an early sigmoid conversion.
        score = logits[:, label].astype(np.float64) - logits[:, 0].astype(np.float64)
        signal = np.sort(score[truth == label])
        background = score[truth == 0]
        targets = (0.99,) if name == "Hqql" else (0.995,) if name == "Tbl" else (0.3, 0.5)
        rows = {}
        for efficiency in targets:
            required = math.ceil(efficiency * len(signal))
            threshold = float(signal[-required])
            signal_passed = int(np.count_nonzero(signal >= threshold))
            background_passed = int(np.count_nonzero(background >= threshold))
            rows[str(efficiency)] = {
                "threshold_log_odds": threshold, "target_signal_efficiency": efficiency,
                "achieved_signal_efficiency": signal_passed / len(signal),
                "signal_accepted": signal_passed, "signal_total": len(signal),
                "qcd_accepted": background_passed, "qcd_total": len(background),
                "rejection": len(background) / background_passed if background_passed else None,
            }
        metrics["qcd_rejection"][name] = rows
    return metrics
