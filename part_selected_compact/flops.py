"""Count selected dense fwd/bwd/recompute work, NOT CPU tree time or all ops."""
import torch
from part_ev_compact.flops import (GlobalOnly, CONVENTION, quadratic, mean_cost,
                                  budget, synthetic as native_synthetic)

LENGTHS = (1, 17, 32, 64, 65, 96, 127, 128)


def synthetic(batch, n, device):
    points, features, vectors, mask = native_synthetic(batch, n, device)
    features[:, 5:11] = 0
    features[:, 5:7] = 1
    tracks = torch.stack((features[:, 0]*.02, torch.full_like(features[:, 0], .03),
                          features[:, 1]*.04, torch.full_like(features[:, 0], .05)), 1)
    return points, torch.cat((features, tracks), 1), vectors, mask


def measure(model, n, device, *, backward):
    from torch.utils.flop_counter import FlopCounterMode
    from part_ev_screen.precision import model_precision
    model.train(backward)
    model.zero_grad(set_to_none=True)
    values = synthetic(2, n, device)
    counter = FlopCounterMode(display=False)
    counter.mod_tracker = GlobalOnly()
    with counter, torch.set_grad_enabled(backward), model_precision(model, device, training=backward):
        result = model(*values)
        if backward:
            result.sum().backward()
    count = counter.get_total_flops()
    operators = {str(k): v for k, v in counter.get_flop_counts()['Global'].items()}
    allowed = {'aten.mm', 'aten.addmm', 'aten.bmm', 'aten.baddbmm',
               'aten.convolution', 'aten.convolution_backward'}
    if (not operators or count <= 0 or not torch.isfinite(result).all()
            or not set(operators).issubset(allowed)
            or not {'aten.bmm', 'aten.baddbmm', 'aten.convolution'}.issubset(operators)):
        raise ValueError('Incomplete or changed dense FLOP coverage')
    model.zero_grad(set_to_none=True)
    return dict(per_jet=count//2, operators_per_batch=operators)


def curve(model, device):
    was_enabled = model.mod.trimmer.enabled
    model.mod.trimmer.enabled = False
    try:
        samples = {str(n): {kind: measure(model, n, device, backward=kind == 'training')
                           for kind in ('forward', 'training')} for n in LENGTHS}
        coefficients = {kind: quadratic({n: r[kind]['per_jet'] for n, r in samples.items() if n != '1'})
                        for kind in ('forward', 'training')}
        return dict(convention=CONVENTION, samples=samples, coefficients=coefficients,
                    checkpoint_recomputation_included=True, selected_encoders_included=True,
                    directed_pairs=True, cpu_tree_work_included=False)
    finally:
        model.mod.trimmer.enabled = was_enabled
