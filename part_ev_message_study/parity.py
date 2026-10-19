"""Bounded, state-isolated initialization checks; never optimize or retry to pass."""
from __future__ import annotations

import copy
import math
import torch


POLICY = dict(version="isolated_warmup_v1", events=8, rtol=1e-5, atol=1e-6,
              modes=["eval_fp32", "train_fp32", "train_fp16"],
              warmup_pairs_per_mode=2, measured_orders=["reference_first", "new_first"],
              optimizer_steps=0)


def validate_report(report):
    """Admission requires the complete CUDA evidence, not just a passed flag."""
    if (report.get("passed") is not True or report.get("policy") != POLICY
            or report.get("device_type") != "cuda" or report.get("events") != POLICY["events"]
            or report.get("optimizer_steps") != 0 or report.get("initial_backbone_exact") is not True
            or set(report.get("modes", {})) != set(POLICY["modes"])):
        raise ValueError("Incomplete or incompatible initialization parity evidence")
    for mode in POLICY["modes"]:
        row = report["modes"][mode]
        if (len(row.get("warmup", [])) != POLICY["warmup_pairs_per_mode"]
                or [r.get("order") for r in row.get("measured", [])] != POLICY["measured_orders"]):
            raise ValueError("Missing fixed warm-up or execution-order parity checks")
        for index, record in enumerate(row["warmup"]):
            comparison = record.get("comparison", {})
            error = comparison.get("maximum_absolute_error", -1)
            if (record.get("pair") != index + 1 or record.get("rng_matches") is not True
                    or not isinstance(comparison.get("passed"), bool)
                    or not isinstance(error, (float, int)) or not math.isfinite(error) or error < 0):
                raise ValueError("Invalid warm-up parity evidence")
        for record in row["measured"]:
            dtype = "torch.float16" if mode == "train_fp16" else "torch.float32"
            if record.get("rng_matches") is not True or record.get("dtype") != dtype:
                raise ValueError("Parity RNG or precision evidence failed")
            for key in ("comparison", "reference_repeat", "new_repeat"):
                comparison = record.get(key, {})
                error = comparison.get("maximum_absolute_error", -1)
                if (comparison.get("passed") is not True or not isinstance(error, (float, int))
                        or not math.isfinite(error) or error < 0):
                    raise ValueError("Initialization parity or repeatability failed")


class ParityError(AssertionError):
    """Carry partial evidence so a failed benchmark can retain it."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def compare(actual, expected):
    # Outputs have already been checked for shape, dtype and finiteness.
    row = dict(passed=True, exact=torch.equal(actual, expected),
               maximum_absolute_error=float((actual.float() - expected.float()).abs().max()))
    try:
        torch.testing.assert_close(actual, expected, rtol=POLICY["rtol"], atol=POLICY["atol"])
    except AssertionError as error:
        row.update(passed=False, error=str(error))
    return row


def same_rng(actual, expected):
    return (torch.equal(actual["torch"], expected["torch"])
            and len(actual["cuda"]) == len(expected["cuda"])
            and all(torch.equal(a, b) for a, b in zip(actual["cuda"], expected["cuda"])))


def forward_once(engine, model, snapshot, values, device, *, mode, seed):
    """Restore buffers, RNG and flags independently for EVERY forward, even on error."""
    rng = engine.capture_rank_state(model)
    flags = [(module, module.training) for module in model.modules()]
    trimmers = [(m, m.enabled) for m in model.modules() if type(m).__name__ == "SequenceTrimmer"]
    switches = [(m, m.disable_new) for m in model.modules() if hasattr(m, "disable_new")]
    try:
        model.load_state_dict(snapshot)
        model.train(mode != "eval_fp32")
        for module, _ in trimmers:
            module.enabled = False
        for module, _ in switches:
            module.disable_new = False  # Never admit a model with its extension disabled.
        engine.seed_everything(seed)
        with torch.no_grad(), engine.model_precision(model, device, training=mode == "train_fp16"):
            logits = model(*values)
        expected_dtype = torch.float16 if mode == "train_fp16" and device.type == "cuda" else torch.float32
        if logits.dtype != expected_dtype or logits.shape != (len(values[0]), 10):
            raise ValueError(f"{mode}: unexpected logits shape/dtype {logits.shape}, {logits.dtype}")
        if not torch.isfinite(logits).all():
            raise ValueError(f"{mode}: nonfinite parity logits (including warm-up)")
        return logits.detach().cpu().clone(), engine.capture_rank_state(model)
    finally:
        model.load_state_dict(snapshot)
        for module, enabled in trimmers:
            module.enabled = enabled
        for module, disabled in switches:
            module.disable_new = disabled
        engine.restore_rank_state(model, rng)
        for module, training in flags:
            module.training = training


def initial_parity(engine, model, task, batch, device, reference_factory):
    """A fixed warm-up, then strict comparisons in both orders and repeatability checks.

    Warm-up discrepancies remain visible, not evidence of a proven root cause.
    Their finite numerical differences do not decide admission; EVERY measured
    comparison must pass. There is no adaptive warm-up, tolerance increase or
    fallback that disables the new branch. Production uses fresh model state.
    """
    report = dict(passed=False, policy=copy.deepcopy(POLICY), modes={},
                  device_type=device.type, optimizer_steps=0)
    original = copy.deepcopy(model.state_dict())
    rng = engine.capture_rank_state(model)
    flags = [(module, module.training) for module in model.modules()]
    try:
        engine.seed_everything(task["seed"])
        reference = reference_factory("part_ev", trim=False, use_amp=False).to(device)
        snapshots = dict(new=original, reference=copy.deepcopy(reference.state_dict()))
        for name, tensor in snapshots["reference"].items():
            if name not in original or not torch.equal(original[name], tensor):
                raise ValueError(f"Changed initial EV/backbone parameter or buffer: {name}")
        if any(p.dtype != torch.float32 for target in (model, reference) for p in target.parameters()):
            raise ValueError("Parity requires FP32 model parameters")
        report["initial_backbone_exact"] = True
        values, _ = engine.tensors(batch, device)
        values = [v[:POLICY["events"]] for v in values]
        report["events"] = len(values[0])
        if not report["events"]:
            raise ValueError("Empty parity input")

        def run(which, mode):
            return forward_once(engine, model if which == "new" else reference,
                                snapshots[which], values, device, mode=mode, seed=task["seed"] + 771)

        for mode in POLICY["modes"]:
            row = report["modes"][mode] = dict(warmup=[], measured=[])
            for index in range(POLICY["warmup_pairs_per_mode"]):
                expected, ref_rng = run("reference", mode)
                actual, new_rng = run("new", mode)
                evidence = dict(pair=index + 1, comparison=compare(actual, expected),
                                rng_matches=same_rng(new_rng, ref_rng))
                row["warmup"].append(evidence)
                if not evidence["rng_matches"]:
                    raise ValueError(f"{mode} warm-up: new mechanism consumes extra training RNG")
            first = None
            for order in POLICY["measured_orders"]:
                results = {which: run(which, mode) for which in
                           (("reference", "new") if order == "reference_first" else ("new", "reference"))}
                actual, new_rng = results["new"]
                expected, ref_rng = results["reference"]
                if first is None:
                    first = results
                evidence = dict(order=order, dtype=str(actual.dtype),
                                comparison=compare(actual, expected),
                                reference_repeat=compare(expected, first["reference"][0]),
                                new_repeat=compare(actual, first["new"][0]),
                                rng_matches=(same_rng(new_rng, ref_rng)
                                    and same_rng(new_rng, first["new"][1])
                                    and same_rng(ref_rng, first["reference"][1])))
                row["measured"].append(evidence)
                if not evidence["rng_matches"]:
                    raise ValueError(f"{mode} {order}: training RNG mismatch")
                for key in ("comparison", "reference_repeat", "new_repeat"):
                    if not evidence[key]["passed"]:
                        raise AssertionError(f"{mode} {order} {key}: {evidence[key]['error']}")
        report["passed"] = True
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise ParityError(report["error"], report) from error
    finally:
        model.load_state_dict(original)
        engine.restore_rank_state(model, rng)
        for module, training in flags:
            module.training = training
