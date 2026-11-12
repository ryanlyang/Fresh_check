"""Composed selected compact oracle/gradient checks, including directed inputs."""
import copy
import torch
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_screen.precision import model_precision
from part_ev_compact.parity import TOLERANCES
from part_selected_fullscale.parity import full_recipe_restart
from scripts.run_part_ev_screen import tensors


def execution(factory, architecture, batch, device):
    small = ({k: v[:4] for k, v in batch[0].items()}, {'_label_': batch[1]['_label_'][:4]})
    values, truth = tensors(small, device)
    records = []
    compact = architecture == 'compact'
    for nonzero in ((False, True) if compact else (False,)):
        candidate = factory(architecture, optimized=True, fast=True, trim=False)
        oracle = factory(architecture, optimized=False, fast=False, trim=False)
        if nonzero:
            with torch.no_grad():
                for block in candidate.mod.blocks:
                    block.attn.attention.state.output.normal_(0, .03)
        oracle.load_state_dict(candidate.state_dict())
        initial = copy.deepcopy(candidate.state_dict())
        for mode in ('eval', 'train_fp32', 'train_amp'):
            if mode == 'train_amp' and device.type != 'cuda':
                continue
            training, amp = mode != 'eval', mode == 'train_amp'
            outputs, gradients = [], []
            for model in (candidate, oracle):
                model.load_state_dict(initial)
                model.train(training)
                model.zero_grad(set_to_none=True)
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
                        if nonzero and ('.state.' in name or name.startswith('shared_state_edge')) and not p.grad.count_nonzero():
                            raise ValueError('Disconnected state gradient: '+name)
                        grad[name] = p.grad.detach().cpu().clone()
                    for family in ('pt', 'track_node', 'track_pair', 'region'):
                        if not any(p.grad.count_nonzero() for p in getattr(model.mod.pair_embed.relations, family).parameters()):
                            raise ValueError('Disconnected selected family: '+family)
                    gradients.append(grad)
                outputs.append(logits.detach().cpu())
                if compact and model._compact_cache.stem is not None:
                    raise ValueError('Directed cache leaked')
            tolerance = TOLERANCES['amp' if amp else 'fp32']
            torch.testing.assert_close(*outputs, **tolerance)
            error = None
            if training:
                a = torch.cat([g.flatten().double() for g in gradients[0].values()])
                b = torch.cat([g.flatten().double() for g in gradients[1].values()])
                error = float((a-b).norm()/b.norm().clamp_min(1e-12))
                if error > (.03 if amp else 1e-4):
                    raise ValueError('Execution gradients differ: '+str(error))
            records.append(dict(mode=mode, nonzero_state=nonzero,
                                maximum_logit_error=float((outputs[0]-outputs[1]).abs().max()),
                                gradient_relative_l2=error))
        del candidate, oracle
    return dict(passed=True, comparisons=records, tolerances=TOLERANCES,
                directed_pairs=True, selected_family_gradients=True,
                accuracy_retention_established=False)


def restart(model, batch, device):
    return full_recipe_restart(model, batch, device)
