"""Small, explicitly requested eval diagnostics; no persistent pair tensors."""
import torch
from torch.nn import functional as F


def projection_statistics(scores, values, keys, valid):
    if torch.is_grad_enabled(): raise ValueError('Diagnostic must not build a training graph')
    count = valid.sum(-1)
    present = count > 0
    masked = scores.masked_fill(~valid[:, None], -torch.inf)
    # A dummy finite softmax for absent jets, then erase its weights.
    masked = torch.where(present[:, None, None], masked, torch.zeros_like(masked))
    weights = masked.softmax(-1).masked_fill(~valid[:, None], 0)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    normalized = entropy / count.clamp_min(2).log()[:, None]
    support = weights.square().sum(-1).clamp_min(1e-30).reciprocal()
    eligible = int(present.sum())
    def mean(x): return float(x[present].mean()) if eligible else 0.
    result = dict(key_rms=float(keys[valid].square().mean().sqrt()) if eligible else 0.,
        value_rms=float(values[valid].square().mean().sqrt()) if eligible else 0.,
        pair_weight_similarity=mean(F.cosine_similarity(weights[:, 0], weights[:, 1], dim=-1)),
        mean_pair_count=mean(count.float()))
    for a in range(2):
        result.update({f'query{a}_normalized_entropy': mean(normalized[:, a]),
            f'query{a}_max_weight': mean(weights[:, a].amax(-1)),
            f'query{a}_effective_support': mean(support[:, a])})
    return result


def gradient_norms(model):
    """Read already-computed, unscaled gradients; no backward/optimizer call."""
    return {n: float(p.grad.detach().float().norm()) if p.grad is not None else None
            for n, p in model.named_parameters() if n.startswith('mod.pair_readout.')}
