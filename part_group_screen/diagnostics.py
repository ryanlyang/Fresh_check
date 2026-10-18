"""State-isolated grouping diagnostics and native parity checks."""
import time
import torch
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state, seed_everything
from part_ev_fullscale.model import build_model as native_model
from part_ev_screen.precision import model_precision
from part_ev_screen.data import batch_fingerprint
from scripts.run_part_ev_screen import tensors


def initial_parity(model, task, batch, device):
    rng, training = capture_rank_state(model), model.training
    branch = model.group_branch
    disabled = branch.disabled
    try:
        seed_everything(task["seed"])
        reference = native_model("part", trim=True, use_amp=False).to(device).eval()
        for key, value in reference.state_dict().items():
            if not torch.equal(model.state_dict()[key], value):
                raise ValueError(f"Unchanged backbone initialization differs: {key}")
        model.eval()
        branch.disabled = True
        values, _ = tensors(batch, device)
        with torch.no_grad(), model_precision(model, device, training=False):
            expected = reference(*(v[:8] for v in values))
            actual = model(*(v[:8] for v in values))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return dict(passed=True, branch_disabled=True, maximum_absolute_error=float((actual - expected).abs().max()))
    finally:
        branch.disabled = disabled
        restore_rank_state(model, rng)
        model.train(training)


def collect(model, batch, device):
    state, training = capture_rank_state(model), model.training
    branch = model.group_branch
    previous = branch.collect_diagnostics
    timing = {}
    def start(_module, _inputs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing["start"] = time.monotonic()
    def finish(_module, _inputs, _output):
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing["seconds"] = time.monotonic() - timing["start"]
    hooks = [branch.register_forward_pre_hook(start), branch.register_forward_hook(finish)]
    try:
        model.eval()
        branch.collect_diagnostics = True
        with torch.no_grad(), model_precision(model, device, training=False):
            values, _ = tensors(batch, device)
            model(*(v[:16] for v in values))
        return dict(input_fingerprint=batch_fingerprint(batch), diagnostic_events=min(16, len(values[0])),
                    group_forward_seconds=timing["seconds"], **branch.last_diagnostics)
    finally:
        for hook in hooks:
            hook.remove()
        branch.collect_diagnostics = previous
        branch.last_diagnostics = None
        restore_rank_state(model, state)
        model.train(training)
