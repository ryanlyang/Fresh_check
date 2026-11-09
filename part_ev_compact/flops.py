"""Measured dense-operator training FLOPs, including checkpoint recomputation.

Multiply-add counts as two FLOPs. This is NOT all arithmetic or GPU time:
normalization, activations, softmax, indexing, optimizer, I/O and validation are
excluded. Both arms use the same convention and observed sequence population.
"""
from fractions import Fraction
import math
import torch

CONVENTION = "pytorch_dense_matmul_convolution_fwd_bwd_recompute_mac2_v1"


class GlobalOnly:
    parents = {"Global"}
    def __enter__(self): return self
    def __exit__(self, *args): pass


def synthetic(batch, n, device):
    # Nondegenerate physical vectors; no zero-energy padding in cost probes.
    g = torch.Generator(device=device).manual_seed(910+n)
    xyz = torch.randn(batch, 3, n, device=device, generator=g)
    return (torch.randn(batch, 2, n, device=device, generator=g),
            torch.randn(batch, 17, n, device=device, generator=g),
            torch.cat((xyz, (xyz.square().sum(1, keepdim=True)+.25).sqrt()), 1),
            torch.ones(batch, 1, n, dtype=torch.bool, device=device))


def measure(model, n, device, *, backward):
    from torch.utils.flop_counter import FlopCounterMode
    from part_ev_screen.precision import model_precision
    model.train(backward)
    model.zero_grad(set_to_none=True)
    values = synthetic(2, n, device)
    counter = FlopCounterMode(display=False)
    # Module attribution is unnecessary and can fail on class-token views.
    counter.mod_tracker = GlobalOnly()
    with counter, torch.set_grad_enabled(backward), model_precision(model, device, training=backward):
        result = model(*values)
        if backward: result.sum().backward()
    count = counter.get_total_flops()
    operators = {str(k): v for k, v in counter.get_flop_counts()['Global'].items()}
    if not operators or count <= 0 or not torch.isfinite(result).all():
        raise ValueError("Missing/nonfinite FLOP probe")
    allowed = {'aten.mm','aten.addmm','aten.bmm','aten.baddbmm','aten.convolution','aten.convolution_backward'}
    if not set(operators).issubset(allowed):
        raise ValueError('New counted operator needs a FLOP convention audit: '+str(set(operators)-allowed))
    if not all(k in operators for k in ('aten.baddbmm','aten.bmm','aten.convolution')):
        raise ValueError("Dense/attention or pair-convolution counter coverage missing")
    model.zero_grad(set_to_none=True)
    return dict(per_jet=count//2, operators_per_batch=operators)


def quadratic(samples):
    # Fit exact rational coefficients; independent points MUST agree exactly.
    xs = sorted(int(k) for k in samples)
    if len(xs) < 4: raise ValueError("Need fitted and independent FLOP samples")
    coeff = [Fraction(0) for _ in range(3)]  # constant, linear, quadratic
    for x in xs[:3]:
        u, v = [y for y in xs[:3] if y != x]
        factor = Fraction(samples[str(x)], (x-u)*(x-v))
        for j, val in enumerate((u*v, -u-v, 1)): coeff[j] += factor*val
    for x in xs:
        if sum(c*x**i for i, c in enumerate(coeff)) != samples[str(x)]:
            raise ValueError("FLOP curve is not quadratic; measure a full length table before budgeting")
    return [str(c) for c in coeff]


def curve(model, device):
    model.mod.trimmer.enabled = False
    records = {}
    for n in (1, 17, 32, 64, 65, 96, 127, 128):
        records[str(n)] = {kind: measure(model, n, device, backward=kind == 'training')
                          for kind in ('forward', 'training')}
    # N=1 einsums can degenerate into elementwise products (outside the dense
    # counter convention). Retain its measured value rather than extrapolating.
    coefficients = {kind: quadratic({n: row[kind]['per_jet'] for n, row in records.items() if n != '1'})
                    for kind in ('forward', 'training')}
    return dict(convention=CONVENTION, samples=records, coefficients=coefficients,
                checkpoint_recomputation_included=True)


def mean_cost(coefficients, histogram, *, singleton_cost):
    if not histogram: raise ValueError("Missing real-stream length histogram")
    coeff = [Fraction(x) for x in coefficients]
    total = Fraction(0)
    for n, count in histogram.items():
        if not 1 <= int(n) <= 128 or type(count) is not int or count <= 0:
            raise ValueError("Invalid length histogram")
        cost = Fraction(singleton_cost) if int(n) == 1 else sum(c*int(n)**i for i, c in enumerate(coeff))
        if cost <= 0: raise ValueError("Nonpositive FLOP estimate")
        total += cost*count
    return total / sum(histogram.values())


def budget(compact_cost, matched_cost):
    if not (math.isfinite(float(compact_cost)) and math.isfinite(float(matched_cost))
            and compact_cost > 0 and matched_cost > 0):
        raise ValueError("Invalid FLOP costs")
    # Round down to preserve the <= candidate budget and exactly 50 checks.
    steps = int(Fraction(1000000)*compact_cost/matched_cost)//50*50
    if not 1000000 <= steps <= 3000000:
        raise ValueError("Unexpected matched exposure; review measurements")
    return dict(compact_steps=1000000, matched_steps=steps, validation_checks=50,
                matched_steps_per_check=steps//50, compact_steps_per_check=20000,
                compact_training_flops=float(compact_cost*512*1000000),
                matched_training_flops=float(matched_cost*512*steps),
                exposure_ratio=steps/1000000,
                convention=CONVENTION, validation_compute_included=False)
