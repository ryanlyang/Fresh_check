"""Same-math execution checks; not evidence that rank reduction retains accuracy."""
import copy
import torch
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_screen.precision import model_precision
from scripts.run_part_ev_screen import tensors
from .model import build_model

# Frozen before RC execution. AMP changes the placement of rounding when the
# output GEMMs are merged. FP32 and AMP are tested and reported separately.
TOLERANCES = {'fp32': dict(rtol=1e-4, atol=1e-5), 'amp': dict(rtol=.03, atol=.006)}


def execution(batch, device):
    values, truth = tensors(({k:v[:4] for k,v in batch[0].items()},
                            {'_label_':batch[1]['_label_'][:4]}), device)
    records = []
    for nonzero in (False, True):
        seed_everything(101)
        candidate = build_model('compact', trim=False).to(device)
        seed_everything(101)
        oracle = build_model('compact', trim=False, optimized=False).to(device)
        if nonzero:
            with torch.no_grad():
                for block in candidate.mod.blocks:
                    block.attn.attention.state.output.normal_(0, .03)
        oracle.load_state_dict(candidate.state_dict())
        initial = copy.deepcopy(candidate.state_dict())
        for mode in ('eval', 'train_fp32', 'train_amp'):
            if mode == 'train_amp' and device.type != 'cuda': continue
            training, amp = mode != 'eval', mode == 'train_amp'
            outputs, gradients = [], []
            for model in (candidate, oracle):
                model.load_state_dict(initial); model.train(training); model.zero_grad(set_to_none=True)
                seed_everything(818)
                with torch.set_grad_enabled(training), model_precision(model, device, training=amp):
                    logits = model(*values)
                    loss = torch.nn.functional.cross_entropy(logits, truth)
                if training:
                    loss.backward()
                    grad = {}
                    for name, p in model.named_parameters():
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise ValueError('Missing/nonfinite gradient: '+name)
                        if nonzero and ('state.' in name or name.startswith('shared_state_edge')) and not p.grad.count_nonzero():
                            raise ValueError('Disconnected state gradient: '+name)
                        grad[name] = p.grad.detach().cpu().clone()
                    gradients.append(grad)
                outputs.append(logits.detach().cpu())
                if model._compact_cache.stem is not None: raise ValueError('Pair cache leaked')
            tolerance = TOLERANCES['amp' if amp else 'fp32']
            torch.testing.assert_close(*outputs, **tolerance)
            relative_gradient_error = None
            if training:
                a = torch.cat([g.flatten().double() for g in gradients[0].values()])
                b = torch.cat([g.flatten().double() for g in gradients[1].values()])
                relative_gradient_error = float((a-b).norm()/b.norm().clamp_min(1e-12))
                if relative_gradient_error > (.03 if amp else 1e-4):
                    raise ValueError(f'Execution gradient discrepancy: {relative_gradient_error}')
            records.append(dict(mode=mode, nonzero_state=nonzero,
                                maximum_logit_error=float((outputs[0]-outputs[1]).abs().max()),
                                gradient_relative_l2=relative_gradient_error))
        del candidate, oracle
    return dict(passed=True, tolerances=TOLERANCES, comparisons=records,
                accuracy_retention_established=False)


def restart(model, batch, device):
    # This established generic pilot saves CPU observations after synchronization
    # and checks model, full Ranger slow weights, scaler, RNG and native trimmer.
    # It neither invokes selected-relation code nor changes any old artifact.
    from part_selected_fullscale.parity import full_recipe_restart
    return full_recipe_restart(model, batch, device)
