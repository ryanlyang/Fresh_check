"""Numerical admission check for allocation-only selected-model changes.

The reference execution is never a new training configuration. Compare it on
a small prefix before the unchanged full-batch memory/throughput benchmark.
Restore all model state and RNG; do not advance the loader or an optimizer.
"""
import copy
from contextlib import nullcontext
import torch
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state
from part_ev_screen.precision import model_precision


EXECUTION = dict(version='selected_memory_v1',
    ev_layout='shared_BIJD_storage_with_logical_BDIJ_axes',
    descriptors='sequential_raw_families_and_amp_fixed_track_channels',
    activation_checkpointing=False, microbatching=False)
PARITY = dict(events=8,
    fp32=dict(rtol=1e-5, atol=1e-6, gradient_relative_l2=2e-5, gradient_absolute=1e-6),
    fp16=dict(rtol=2e-3, atol=2e-3, gradient_relative_l2=5e-3, gradient_absolute=2e-5))


def compare(actual, expected, limits, *, gradients=False):
    """Elementwise logits, norm-relative gradients; always reject nonfinite data."""
    if actual.shape != expected.shape or not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError('Memory execution parity: shape/nonfinite mismatch')
    difference = (actual.double()-expected.double())
    absolute = float(difference.abs().max())
    relative = float(difference.norm()/expected.double().norm().clamp_min(1e-12))
    if gradients:
        # Biases before BatchNorm can have mathematically zero gradients and
        # roundoff-sized measured ones; a relative-only test is meaningless
        # there. Use an explicit per-element absolute floor in the L2 bound.
        tolerance = (limits['gradient_relative_l2']*expected.double().norm()
                     + limits['gradient_absolute']*expected.numel()**.5)
        passed = bool(difference.norm() <= tolerance)
    else:
        passed = bool(torch.allclose(actual, expected, rtol=limits['rtol'], atol=limits['atol']))
    if not passed:
        raise ValueError(f'Memory execution parity failed: gradients={gradients}, '
                         f'max_abs={absolute:.6g}, relative_l2={relative:.6g}')
    return dict(passed=True, maximum_absolute_error=absolute, relative_l2_error=relative)


def execution_parity(model, batch, device):
    """Real-data FP32 evaluation + FP32/AMP backward, no production mutation.

    Eight jets retain the full constituent axis; disabling trimming ONLY during
    this diagnostic keeps identical inputs in both executions. Small differences
    in gradients are possible from summing shared-storage branches in a different
    order, especially in FP16. This is not a guarantee of identical training paths.
    """
    pair = model.mod.pair_embed
    if not (pair.memory_optimized and pair.relations.memory_optimized):
        raise ValueError('Benchmark must use memory-optimized production execution')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Execution parity must run before any training step')
    features, labels, _ = batch
    n = min(PARITY['events'], len(labels['_label_']))
    if n < 2:
        raise ValueError('Execution parity needs at least two real training jets')
    values = [features[k][:n].to(device) for k in ('pf_points', 'pf_features', 'pf_vectors', 'pf_mask')]
    target = labels['_label_'][:n].long().to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = capture_rank_state(model)
    training = [(module, module.training) for module in model.modules()]
    trimmers = [(module, module.enabled) for module in model.modules() if type(module).__name__ == 'SequenceTrimmer']
    flags = pair.memory_optimized, pair.relations.memory_optimized
    results = {}
    try:
        for module, _ in trimmers:
            module.enabled = False
        for mode, training_mode, amp in (('fp32_eval', False, False), ('fp32_train', True, False), ('fp16_train', True, True)):
            observations = []
            for optimized in (False, True):
                model.load_state_dict(state, strict=True)
                model.zero_grad(set_to_none=True)
                restore_rank_state(model, rng)
                model.train(training_mode)
                pair.memory_optimized = pair.relations.memory_optimized = optimized
                # Production CUDA precision is exactly the shared screen's;
                # CPU autocast only supports the local regression tests.
                cpu_amp = torch.autocast('cpu', dtype=torch.float16) if amp and device.type == 'cpu' else nullcontext()
                with torch.set_grad_enabled(training_mode), model_precision(model, device, training=amp), cpu_amp:
                    logits = model(*values)
                    loss = torch.nn.functional.cross_entropy(logits.float(), target) if training_mode else None
                if training_mode:
                    loss.backward()
                gradients = {}
                if training_mode:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            raise ValueError(f'Missing gradient during execution parity: {name}')
                        gradients[name] = parameter.grad.detach().float().cpu().clone()
                observations.append((logits.detach().float().cpu(), gradients))
            reference, actual = observations
            limits = PARITY['fp16' if amp else 'fp32']
            row = dict(logits=compare(actual[0], reference[0], limits))
            if training_mode:
                # Check each parameter-owning module separately: relation MLPs
                # cannot hide behind the backbone. Combine a Linear/Conv's
                # weight and bias: pre-BatchNorm bias gradients are zero in
                # exact arithmetic, so a bias-only relative norm measures
                # cancellation noise rather than the module's gradient error.
                groups = {}
                for key in reference[1]:
                    groups.setdefault(key.rpartition('.')[0], []).append(key)
                checks = {}
                for group, keys in groups.items():
                    try:
                        checks[group] = compare(torch.cat([actual[1][k].flatten() for k in keys]),
                                                torch.cat([reference[1][k].flatten() for k in keys]),
                                                limits, gradients=True)
                    except ValueError as exc:
                        raise ValueError(f'{mode} gradient parity at {group}: {exc}') from exc
                row['gradients'] = dict(passed=True, parameters=len(reference[1]), groups=len(checks),
                    maximum_absolute_error=max(r['maximum_absolute_error'] for r in checks.values()),
                    maximum_module_relative_l2=max(r['relative_l2_error'] for r in checks.values()))
            results[mode] = row
        return dict(passed=True, execution=copy.deepcopy(EXECUTION), policy=copy.deepcopy(PARITY), events=n, modes=results)
    finally:
        model.load_state_dict(state, strict=True)
        model.zero_grad(set_to_none=True)
        pair.memory_optimized, pair.relations.memory_optimized = flags
        pair.clear()
        for module, enabled in trimmers:
            module.enabled = enabled
        for module, was_training in training:
            module.training = was_training
        restore_rank_state(model, rng)
