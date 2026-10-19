#!/usr/bin/env python3
"""Diagnostic only: reproduce and localize frozen message-study parity failures."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

SCHEMA = "part_ev_message_parity_diagnostic_v1"
POLICY = dict(events=8, rtol=1e-5, atol=1e-6,
              modes=["eval_fp32", "train_fp32", "train_fp16"],
              authorizes_training=False, optimizer_steps=0)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def create_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def frozen_driver(campaign):
    source = (campaign / "source").resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("part_ev_message_study", "part_ev_screen", "part_ev_fullscale", "scripts"):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("Refusing already-imported non-campaign code: " + name)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "frozen_message_driver", source / "scripts/run_part_ev_message_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(campaign, output, index):
    if not output.is_relative_to(campaign / "diagnostics") or output == campaign / "diagnostics":
        raise ValueError("Use a new directory below campaign/diagnostics")
    M = frozen_driver(campaign)
    spec = M.P.load_campaign(campaign)
    M.P.task_directory(campaign, index)
    if read(output / "launch.json") != spec["launch"]:
        raise ValueError("Diagnostic resources differ from frozen campaign")
    names = ("campaign.json", "launch.env", "launch.json", "launch.sha256", "source.sha256")
    plan = dict(schema=SCHEMA, policy=POLICY, campaign=str(campaign),
                campaign_sha256=spec["content_hash"], task_index=index, task=spec["tasks"][index],
                artifacts={name: sha(campaign / name) for name in names},
                diagnostic_source_sha256=sha(__file__),
                diagnostic_launch_sha256=sha(output / "launch.env"))
    plan["content_hash"] = digest(plan)
    create_json(output / "diagnostic_plan.json", plan)
    print(json.dumps(dict(stage="parity_diagnostic_prepared", task=plan["task"],
                          authorizes_training=False)), flush=True)


def load_plan(output):
    plan = read(output / "diagnostic_plan.json")
    if (plan.get("content_hash") != digest({k: v for k, v in plan.items() if k != "content_hash"})
            or plan["schema"] != SCHEMA or plan["policy"] != POLICY
            or plan["diagnostic_source_sha256"] != sha(__file__)
            or plan["diagnostic_launch_sha256"] != sha(output / "launch.env")):
        raise ValueError("Diagnostic source/plan/resources changed")
    verify_artifacts(plan)
    return plan


def verify_artifacts(plan):
    for name, checksum in plan["artifacts"].items():
        if sha(Path(plan["campaign"]) / name) != checksum:
            raise ValueError("Frozen campaign artifact changed: " + name)


def tensor_record(value):
    return dict(value=value.detach().to("cpu", copy=True), shape=list(value.shape),
                dtype=str(value.dtype), stride=list(value.stride()), contiguous=value.is_contiguous())


def compare(actual, expected):
    """Keep the original tolerance; a mismatch is evidence, not a fatal diagnostic."""
    import torch
    result = dict(shape_matches=actual.shape == expected.shape,
                  actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                  passed=False, exact=False)
    if not result["shape_matches"]:
        return result
    a, b = actual.detach().cpu(), expected.detach().cpu()
    result.update(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()), elements=a.numel())
    if not result["finite"]:
        return result
    # torch.testing.assert_close checks dtype as well as numerical closeness.
    close = torch.isclose(a, b, rtol=POLICY["rtol"], atol=POLICY["atol"]) if a.dtype == b.dtype else torch.zeros_like(a, dtype=torch.bool)
    error = (a.double() - b.double()).abs()
    result.update(passed=bool(close.all()), exact=bool(torch.equal(a, b) and a.dtype == b.dtype),
                  outside_tolerance_count=int((~close).sum()),
                  maximum_absolute_error=float(error.max()) if a.numel() else 0.,
                  mean_absolute_error=float(error.mean()) if a.numel() else 0.)
    if a.numel():
        worst = int(error.reshape(-1).argmax())
        result.update(worst_flat_index=worst, actual_at_worst=float(a.reshape(-1)[worst]),
                      expected_at_worst=float(b.reshape(-1)[worst]))
    if a.ndim == 2 and a.shape[-1] == 10:
        result["argmax_disagreements"] = int((a.argmax(-1) != b.argmax(-1)).sum())
    return result


@contextmanager
def capture_trace(model):
    """Temporary hooks and EV output-projection tap, in this diagnostic process only.

    CPU copies can synchronize CUDA; untraced forwards are also compared so the
    report explicitly checks whether instrumentation changed the result.
    """
    import torch
    from torch.nn import functional as F
    records, handles = {}, []
    original_linear = F.linear

    def record(name, value):
        records[name] = tensor_record(value)

    def hook(name):
        return lambda _module, _inputs, value: record(name, value)

    def native_hook(name):
        def collect(_module, _inputs, value):
            record(name + ".native_output", value[0])
            record(name + ".weights", value[1])
        return collect

    def pair_hook(module, _inputs, value):
        record("pair.bias", value)
        record("pair.value_stem", module._edges[0]._relation_stem)

    projections = {}
    try:
        handles.append(model.mod.embed.register_forward_hook(hook("particle_embedding")))
        pair = model.mod.pair_embed
        handles.append(pair.reference.embed.prefix.register_forward_hook(hook("pair.bias_encoder")))
        handles.append(pair.reference.embed.projection.register_forward_hook(hook("pair.bias_projection")))
        handles.append(pair.register_forward_hook(pair_hook))
        for i, block in enumerate(model.mod.blocks):
            name = f"particle.{i}"
            handles.append(block.attn.reference.register_forward_pre_hook(
                lambda _m, args, name=name: record(name + ".query", args[0])))
            handles.append(block.attn.reference.register_forward_hook(native_hook(name)))
            handles.append(block.attn.register_forward_hook(
                lambda _m, _a, result, name=name: record(name + ".attention_output", result[0])))
            handles.append(block.register_forward_hook(hook(name + ".block_output")))
            projections[id(block.attn.reference.out_proj.weight)] = name
            for branch in ("state", "routing", "variance"):
                module = getattr(block.attn, branch, None)
                if module is not None:
                    handles.append(module.register_forward_hook(hook(name + ".extension." + branch)))
        for i, block in enumerate(model.mod.cls_blocks):
            handles.append(block.register_forward_hook(hook(f"class.{i}.block_output")))
        handles.append(model.mod.norm.register_forward_hook(hook("class.norm")))
        handles.append(model.mod.fc.register_forward_hook(hook("logits")))

        def traced_linear(x, weight, bias=None):
            result = original_linear(x, weight, bias)
            # Native MHA's output projection is 2D; added EV uses [N,B,C].
            name = projections.get(id(weight))
            if name is not None and bias is None and x.ndim == 3:
                record(name + ".ev_projection_input", x)
                record(name + ".ev_projection_output", result)
            return result

        F.linear = traced_linear
        yield records
    finally:
        F.linear = original_linear
        for handle in handles:
            handle.remove()


def compare_traces(actual, expected):
    rows = []
    for name, ref in expected.items():
        if name not in actual:
            rows.append(dict(name=name, missing=True, passed=False, exact=False))
            continue
        row = compare(actual[name]["value"], ref["value"])
        row.update(name=name, actual_layout={k: v for k, v in actual[name].items() if k != "value"},
                   expected_layout={k: v for k, v in ref.items() if k != "value"})
        rows.append(row)
    return dict(first_not_exact=next((r["name"] for r in rows if not r["exact"]), None),
                first_outside_tolerance=next((r["name"] for r in rows if not r["passed"]), None),
                stages=rows)


def branch_evidence(trace):
    import torch
    result = {}
    for name, row in trace.items():
        if ".extension." not in name:
            continue
        value = row["value"]
        evidence = dict(dtype=row["dtype"], finite=bool(torch.isfinite(value).all()),
                        nonzero_elements=int(value.count_nonzero()))
        if evidence["finite"]:
            evidence["maximum_absolute_value"] = float(value.abs().max())
        if name.endswith(".routing"):
            evidence["versus_native_weights"] = compare(value, trace[name.split(".extension.")[0] + ".weights"]["value"])
        result[name] = evidence
    return result


def rng_comparison(actual, expected):
    import torch
    return dict(torch_cpu_exact=torch.equal(actual["torch"], expected["torch"]),
                torch_cuda_exact=len(actual["cuda"]) == len(expected["cuda"]) and
                all(torch.equal(a, b) for a, b in zip(actual["cuda"], expected["cuda"])))


def forward_once(D, model, original, values, device, *, mode, seed, disabled=False, trace=False):
    import torch
    saved_rng, was_training = D.capture_rank_state(model), model.training
    trimmers = [(m, m.enabled) for m in model.modules() if type(m).__name__ == "SequenceTrimmer"]
    switches = [(m, m.disable_new) for m in model.modules() if hasattr(m, "disable_new")]
    try:
        model.load_state_dict(original)
        model.train(mode != "eval_fp32")
        for m, _ in trimmers:
            m.enabled = False
        for m, _ in switches:
            m.disable_new = disabled
        D.seed_everything(seed)
        with capture_trace(model) if trace else nullcontext({}) as stages:
            with torch.no_grad(), D.model_precision(model, device, training=mode == "train_fp16"):
                logits = model(*values)
        return dict(logits=logits.detach().cpu(), rng=D.capture_rank_state(model), trace=stages)
    finally:
        model.load_state_dict(original)
        for m, enabled in trimmers:
            m.enabled = enabled
        for m, previous in switches:
            m.disable_new = previous
        D.restore_rank_state(model, saved_rng)
        model.train(was_training)


def analyze_mode(D, model, reference, originals, values, device, mode, seed):
    def run(which, **kwargs):
        target = reference if which == "reference" else model
        return forward_once(D, target, originals[which], values, device, mode=mode, seed=seed, **kwargs)

    ref = run("reference")
    ref_repeat = run("reference")
    enabled = run("new")
    disabled = run("new", disabled=True)
    enabled_repeat = run("new")
    traced_ref = run("reference", trace=True)
    result = dict(
        reference_repeat=compare(ref_repeat["logits"], ref["logits"]),
        new_repeat=compare(enabled_repeat["logits"], enabled["logits"]),
        enabled_vs_reference=compare(enabled["logits"], ref["logits"]),
        disabled_vs_reference=compare(disabled["logits"], ref["logits"]),
        enabled_vs_disabled=compare(enabled["logits"], disabled["logits"]),
        rng_enabled_vs_reference=rng_comparison(enabled["rng"], ref["rng"]),
        rng_disabled_vs_reference=rng_comparison(disabled["rng"], ref["rng"]),
        reference_instrumentation_effect=compare(traced_ref["logits"], ref["logits"]))
    for label, switch, plain in (("enabled", False, enabled), ("disabled", True, disabled)):
        traced = run("new", trace=True, disabled=switch)
        result[label + "_instrumentation_effect"] = compare(traced["logits"], plain["logits"])
        result[label + "_trace"] = compare_traces(traced["trace"], traced_ref["trace"])
        if not switch:
            result["branch_evidence"] = branch_evidence(traced["trace"])
    logits = {"reference": ref["logits"], "reference_repeat": ref_repeat["logits"],
              "enabled": enabled["logits"], "enabled_repeat": enabled_repeat["logits"],
              "disabled": disabled["logits"]}
    return result, logits


def write_report(output, report):
    create_json(output / "parity_report.json", report)
    original = report.get("original_parity_probe", {}).get("passed")
    original_status = "not completed" if original is None else ("PASS" if original else "FAIL")
    lines = ["# EV initialization parity diagnostic", "",
             "Diagnostic only. No training authorized, no tolerances or campaign files changed.", "",
             "Frozen original parity probe: " + original_status + ".", "",
             "Disabled means the added state/routing/variance branch is disabled; a separate pair encoder stays in place.", "",
             "| Mode | Enabled max error | Disabled max error | Reference repeat max error | First enabled stage outside original tolerance |",
             "|---|---:|---:|---:|---|"]
    for mode, row in report.get("modes", {}).items():
        cells = [row[k].get("maximum_absolute_error") for k in
                 ("enabled_vs_reference", "disabled_vs_reference", "reference_repeat")]
        lines.append(f"| {mode} | " + " | ".join("n/a" if c is None else f"{c:.9g}" for c in cells) +
                     " | " + str(row["enabled_trace"]["first_outside_tolerance"]) + " |")
    lines += ["", "| Mode | Enabled / reference dtype | RNG matches | Traced outputs exactly match untraced |",
              "|---|---|---|---|"]
    for mode, row in report.get("modes", {}).items():
        types = row["enabled_vs_reference"]
        rng = all(row["rng_enabled_vs_reference"].values()) and all(row["rng_disabled_vs_reference"].values())
        traced = all(row[k]["exact"] for k in ("reference_instrumentation_effect",
                                               "enabled_instrumentation_effect", "disabled_instrumentation_effect"))
        lines.append(f"| {mode} | {types['actual_dtype']} / {types['expected_dtype']} | {rng} | {traced} |")
    if "error" in report:
        lines += ["", "Diagnostic error:", "```text", report["error"], "```"]
    lines += ["", "Detailed JSON includes dtypes, exact/strict comparisons, tensor strides, branch-zero checks,",
              "RNG checks and traced-versus-untraced comparisons. A completed job means evidence was",
              "collected, not that the model passed admission. If tracing changes logits, its first",
              "divergence is not yet an explanation of the untraced failure. No optimizer steps were taken.", ""]
    with (output / "parity_report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run_diagnostic(output):
    import numpy as np
    if any((output / name).exists() for name in ("parity_report.json", "parity_report.md", "logits.npz")):
        raise FileExistsError("Diagnostic evidence already exists; submit a new diagnostic directory")
    plan = load_plan(output)
    campaign = Path(plan["campaign"])
    M = frozen_driver(campaign)
    spec = M.P.load_campaign(campaign)
    if spec["content_hash"] != plan["campaign_sha256"] or spec["tasks"][plan["task_index"]] != plan["task"]:
        raise ValueError("Campaign/task changed")
    M.P.verify_inventory_metadata(spec)
    # Requires CUDA on SPORC; there is no CPU fallback in the queued diagnostic.
    device, runtime = M.gpu_setup(spec)
    D, task = M.D, plan["task"]
    report = dict(schema=SCHEMA, plan_sha256=plan["content_hash"], task=task, runtime=runtime,
                  authorizes_training=False, optimizer_steps=0, completed=False, modes={})
    loader = None
    try:
        # Exact original benchmark construction/iterator/seed ordering.
        loader, _ = D.build_loader("train", spec["inventory"], 512, seed=task["seed"])
        model = D.make_model(task, device)
        iterator = iter(loader)
        D.seed_everything(task["seed"])
        first = next(iterator)
        fingerprint = D.batch_fingerprint(first)
        donor = read(Path(spec["donor"]["path"]) / "benchmarks/EV_RANGER.json")
        if fingerprint != donor["input_fingerprints"]["0"]:
            raise ValueError("Recreated first batch differs from the reused EV control")
        report["first_batch_fingerprint"] = fingerprint
        try:
            report["original_parity_probe"] = M.initial_parity(model, task, first, device)
        except AssertionError as error:
            report["original_parity_probe"] = dict(passed=False, error=str(error))
        D.seed_everything(task["seed"])
        reference = M.reference_model("part_ev", trim=False, use_amp=False).to(device)
        originals = {"new": copy.deepcopy(model.state_dict()), "reference": copy.deepcopy(reference.state_dict())}
        differing = [name for name, value in originals["reference"].items()
                     if name not in originals["new"] or not D.torch.equal(value, originals["new"][name])]
        report["initial_backbone_parameters_exact"] = not differing
        if differing:
            raise ValueError("Initial backbone parameters differ: " + ", ".join(differing[:10]))
        values, _ = D.tensors(first, device)
        values = [v[:POLICY["events"]] for v in values]
        saved_logits = {}
        for mode in POLICY["modes"]:
            D.check_stop()
            D.progress("parity_diagnostic_mode", mode=mode, task=task)
            row, logits = analyze_mode(D, model, reference, originals, values, device, mode, task["seed"] + 771)
            report["modes"][mode] = row
            saved_logits.update({mode + "_" + name: value.numpy() for name, value in logits.items()})
            D.progress("parity_diagnostic_comparison", mode=mode,
                       enabled=row["enabled_vs_reference"], disabled=row["disabled_vs_reference"],
                       first_divergence=row["enabled_trace"]["first_outside_tolerance"])
        with (output / "logits.npz").open("xb") as stream:
            np.savez_compressed(stream, **saved_logits)
        report["logits_sha256"] = sha(output / "logits.npz")
        verify_artifacts(plan)
        report["completed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            write_report(output, report)
        finally:
            if loader is not None:
                D.shutdown_loader(loader)
    print((output / "parity_report.md").read_text(encoding="utf-8"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0, choices=range(5))
    args = parser.parse_args(argv)
    output = args.output.resolve(strict=True)
    if args.phase == "prepare":
        if args.campaign is None:
            parser.error("prepare requires --campaign")
        prepare(args.campaign.resolve(strict=True), output, args.task_index)
    else:
        run_diagnostic(output)


if __name__ == "__main__":
    main()
