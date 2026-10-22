"""Nonzero-message equivalence and same-device, same-input forward timings."""
from __future__ import annotations

import copy
import gc
import math
import statistics
import time
import torch
from part_ev_message_study import model as old_model
from part_ev_message_study.parity import compare, initial_parity as ev_parity, ParityError
from part_ev_fullscale.model import build_model as baseline_model
from . import model as M, protocol as P


def gradient_equivalence(architecture, device):
    """Exercise nonzero residuals and checkpoint backward, including shared LN grads."""
    fast = M.build_model(architecture, trim=False).to(device).eval()
    slow = M.build_model(architecture, trim=False, optimized=False).to(device).eval()
    with torch.no_grad():
        for block in fast.mod.blocks:
            if block.attn.state is not None:
                block.attn.state.output.normal_(std=.02)
    slow.load_state_dict(fast.state_dict())
    # Use real-shaped states/edges, independent of four-vector preprocessing.
    n = 67  # Cross both old 16-query and new 64-query boundaries.
    inputs = (torch.rand(2, 8, n, n, device=device),
              torch.randn(2, 64, n, n, device=device),
              torch.randn(n, 2, 128, device=device))
    valid = torch.arange(n, device=device)[None] < torch.tensor([n, n-9], device=device)[:, None]
    results = []
    gradients = []
    for target in (slow, fast):
        args = [x.detach().clone().requires_grad_() for x in inputs]
        branches = [b.attn.state.train() for b in target.mod.blocks if b.attn.state is not None]
        outputs = [branch(*args, valid) for branch in branches]
        loss = sum(out.square().mean() for out in outputs)
        loss.backward()
        results.append(torch.stack([o.detach() for o in outputs]))
        gradients.append([v.grad for v in args] + [p.grad for b in branches for p in b.parameters()])
        target._state_pair_cache.clear()
    forward = compare(results[1], results[0])
    if not forward["passed"]:
        raise AssertionError("Nonzero state branch forward differs: " + str(forward))
    for actual, expected in zip(gradients[1], gradients[0]):
        if actual is None or expected is None or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("Missing/nonfinite optimized state gradient")
        torch.testing.assert_close(actual, expected, rtol=P.SPEED_POLICY["gradient_rtol"],
                                   atol=P.SPEED_POLICY["gradient_atol"])
    return dict(passed=True, forward=forward, tensors=len(gradients[0]), particles=n,
                checkpoint_backward=True, max_absolute_error=max(
                    float((a-b).abs().max()) for a,b in zip(gradients[1], gradients[0])))


def trained_equivalence(engine, spec, batch, device):
    """The optimized full model must also preserve an actual trained checkpoint."""
    saved = engine.trusted_load(spec["state_reference"]["checkpoint"])
    reference = old_model.build_model("ev_state", trim=False).to(device).eval()
    candidate = M.build_model("state_fast", trim=False).to(device).eval()
    reference.load_state_dict(saved["model"], strict=True)
    candidate.load_state_dict(saved["model"], strict=True)
    if not any(b.attn.state.output.detach().count_nonzero().item() for b in reference.mod.blocks):
        raise ValueError("Trained state reference has only zero residuals")
    values, _ = engine.tensors(batch, device)
    values = [x[:P.SPEED_POLICY["equivalent_events"]] for x in values]
    with torch.no_grad(), engine.model_precision(candidate, device, training=False):
        # Fixed warmup, not an adaptive retry.
        for _ in range(2):
            reference(*values)
            candidate(*values)
        expected, actual = reference(*values), candidate(*values)
    row = compare(actual, expected)
    row.update(events=len(actual), checkpoint_sha256=spec["state_reference"]["checkpoint_sha256"],
               argmax_disagreements=int((actual.argmax(1) != expected.argmax(1)).sum()))
    if not row["passed"]:
        raise AssertionError("Trained checkpoint optimization parity failed: " + str(row))
    return row


def initial_parity(engine, spec, candidate, task, batch, device):
    report = ev_parity(engine, candidate, task, batch, device, baseline_model)
    rng = engine.capture_rank_state(candidate)
    try:
        engine.seed_everything(task["seed"] + 991)
        report["state_speed_equivalence"] = dict(passed=True, policy=P.SPEED_POLICY,
            nonzero_branch=True, gradients=gradient_equivalence(task["architecture"], device),
            trained_checkpoint=trained_equivalence(engine, spec, batch, device))
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise ParityError(report["error"], report) from error
    finally:
        engine.restore_rank_state(candidate, rng)
    return report


def summarize_samples(samples, batch):
    seconds = statistics.median(samples)
    return dict(seconds_per_batch=seconds, milliseconds_per_batch=seconds*1000,
                jets_per_second=batch/seconds, samples_seconds=samples)


def validate_speed_report(report, spec, task):
    if (report.get("passed") is not True or report.get("campaign_sha256") != spec["content_hash"]
            or report.get("task") != task or report.get("policy") != P.SPEED_POLICY
            or report.get("device_type") != "cuda" or report.get("runtime") != spec["donor"]["runtime"]
            or report.get("particle_slots") != 128 or not 0 < report.get("valid_particles", 0) <= 512*128
            or report.get("weights") != "fresh_seeded_weights_for_speed_only_not_accuracy"
            or set(report.get("models", {})) != {"ParT", "EV", "STATE_original", task["run_id"]}):
        raise ValueError("Wrong/incomplete matched speed evidence")
    for model in report["models"].values():
        if set(model) != {str(b) for b in P.SPEED_POLICY["batch_sizes"]}:
            raise ValueError("Missing throughput or latency measurement")
        for batch, row in model.items():
            samples = row.get("samples_seconds", [])
            if (len(samples) != P.SPEED_POLICY["repeats"] or
                    any(not math.isfinite(x) or x <= 0 for x in samples)
                    or any(row[k] != v for k,v in summarize_samples(samples, int(batch)).items())
                    or not 0 < row["peak_bytes"] < spec["donor"]["runtime"]["gpu_memory_bytes"]*.90):
                raise ValueError("Invalid inference timing/memory evidence")


def speed_probe(engine, spec, task, batch, device, runtime):
    """All four models receive identical resident tensors; no ROOT/metric work timed."""
    values, _ = engine.tensors(batch, device)
    if len(values[0]) != 512:
        raise ValueError("Speed probe requires the common 512-event batch")
    factories = {
        "ParT": lambda: baseline_model("part", trim=False),
        "EV": lambda: baseline_model("part_ev", trim=False),
        "STATE_original": lambda: old_model.build_model("ev_state", trim=False),
        task["run_id"]: lambda: M.build_model(task["architecture"], trim=False),
    }
    rows = {name: {} for name in factories}
    # Rotate execution order between rounds to reduce consistent ordering bias.
    for repeat in range(P.SPEED_POLICY["repeats"]):
        names = list(factories)
        names = names[repeat:] + names[:repeat]
        for name in names:
            engine.seed_everything(task["seed"])
            target = factories[name]().to(device).eval()
            with torch.no_grad(), engine.model_precision(target, device, training=False):
                for size in P.SPEED_POLICY["batch_sizes"]:
                    subset = [x[:size] for x in values]
                    for _ in range(P.SPEED_POLICY["warmup"]):
                        prediction = target(*subset)
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite inference probe output")
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    for _ in range(P.SPEED_POLICY["steps"]):
                        prediction = target(*subset)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-start)/P.SPEED_POLICY["steps"]
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite timed output")
                    record = rows[name].setdefault(str(size), dict(samples_seconds=[], peak_bytes=0))
                    record["samples_seconds"].append(elapsed)
                    record["peak_bytes"] = max(record["peak_bytes"], torch.cuda.max_memory_allocated())
            del target, prediction
            gc.collect()
            torch.cuda.empty_cache()
    for result in rows.values():
        for batch_size, row in result.items():
            row.update(summarize_samples(row["samples_seconds"], int(batch_size)))
    report = dict(passed=True, task=task, campaign_sha256=spec["content_hash"],
        policy=P.SPEED_POLICY, device_type=device.type,
        runtime={k:v for k,v in runtime.items() if k != "campaign_sha256"}, models=rows,
        input_fingerprint=engine.batch_fingerprint(batch),
        weights="fresh_seeded_weights_for_speed_only_not_accuracy",
        particle_slots=values[-1].shape[-1], valid_particles=int(values[-1].sum()))
    validate_speed_report(report, spec, task)
    report["content_hash"] = P.digest(report)
    return report
