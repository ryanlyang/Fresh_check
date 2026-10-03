#!/usr/bin/env python3
"""Isolated numerical diagnosis, never an inference-authorization receipt.

Use the failed campaign's authenticated runtime without editing its source,
settings, checkpoints or predictions. A new diagnostic directory is required.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import time
import traceback

VERSION = "relational_part_sporc_parity_diagnostic_v1"
EVENTS = 640
BATCH = 64


def difference(actual, expected, *, atol, rtol):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return {"passed": False, "shape_matches": False,
                "actual_shape": list(a.shape), "reference_shape": list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result = {"shape_matches": True, "finite": finite, "elements": int(a.size)}
    if not finite:
        return {**result, "passed": False}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    outside = delta > atol + rtol * np.abs(b.astype(np.float64))
    return {**result, "passed": not bool(outside.any()),
            "maximum_absolute_error": float(delta.max()) if delta.size else 0.,
            "mean_absolute_error": float(delta.mean()) if delta.size else 0.,
            "root_mean_square_error": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.,
            "outside_tolerance_count": int(outside.sum())}


def logit_difference(actual, expected, *, atol, rtol):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.ndim != 2 or expected.ndim != 2 or not actual.shape[1] or not expected.shape[1]:
        raise ValueError("Logits must have shape [events,classes] with at least one class")
    result = difference(actual, expected, atol=atol, rtol=rtol)
    if result.get("finite") and result["shape_matches"]:
        result["argmax_disagreements"] = int(np.sum(actual.argmax(1) != expected.argmax(1)))
        result["per_batch"] = [{"entry_start": start, "entry_stop": min(start + BATCH, len(actual)),
            **difference(actual[start:start+BATCH], expected[start:start+BATCH], atol=atol, rtol=rtol)}
            for start in range(0, len(actual), BATCH)]
    return result


def pair_difference(actual, expected, mask, *, atol, rtol):
    import numpy as np
    actual, expected, valid = np.asarray(actual), np.asarray(expected), np.asarray(mask, dtype=bool)
    if (actual.ndim != 4 or expected.shape != actual.shape or actual.shape[1] != 4
            or actual.shape[2] != actual.shape[3] or actual.shape[2] == 0):
        raise ValueError("Pair tensors must both have shape [batch,4,particles,particles]")
    if valid.ndim == 3:
        if valid.shape[1] != 1:
            raise ValueError("Pair mask must have one channel")
        valid = valid[:, 0, :]
    if valid.shape != (actual.shape[0], actual.shape[2]):
        raise ValueError("Pair mask dimensions differ from pair tensors")
    pairs = valid[:, :, None] & valid[:, None, :]
    diagonal = np.eye(valid.shape[1], dtype=bool)[None, :, :]
    return {name: {group: difference(actual[:, i][selected], expected[:, i][selected], atol=atol, rtol=rtol)
                  for group, selected in (("valid_diagonal", pairs & diagonal),
                                          ("valid_off_diagonal", pairs & ~diagonal))}
            for i, name in enumerate(("lnkt", "lnz", "lndelta", "lnm2"))}


@contextmanager
def diagnostic_math(*, tf32_disabled=False, math_attention=False):
    """Temporary process-local A/B switches, restored even if a probe fails."""
    import torch
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    switches = []
    try:
        if tf32_disabled:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if math_attention:
            cuda = torch.backends.cuda
            for getter, setter, value in (
                ("flash_sdp_enabled", "enable_flash_sdp", False),
                ("mem_efficient_sdp_enabled", "enable_mem_efficient_sdp", False),
                ("math_sdp_enabled", "enable_math_sdp", True),
                ("cudnn_sdp_enabled", "enable_cudnn_sdp", False),
            ):
                if hasattr(cuda, getter) and hasattr(cuda, setter):
                    change = getattr(cuda, setter)
                    switches.append((change, getattr(cuda, getter)()))
                    change(value)
            if hasattr(torch.backends, "mha"):
                switches.append((torch.backends.mha.set_fastpath_enabled,
                                 torch.backends.mha.get_fastpath_enabled()))
                torch.backends.mha.set_fastpath_enabled(False)
        yield
    finally:
        for change, value in reversed(switches):
            change(value)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def baseline_forward(model, batches, device, reset, *, cpu_pairs=False):
    """Replay BASE's explicit-uu forward; no REGION trees are consumed by BASE."""
    import numpy as np
    import torch
    reset([model])
    results = []
    with torch.no_grad():
        for batch in batches:
            tensors = {k: torch.from_numpy(v).to(device).clone() for k, v in batch.items()}
            vectors, mask = tensors["lorentz_vectors"], tensors["mask"]
            pair = (model.explicit_standard_four(vectors.cpu(), mask.cpu()).to(device)
                    if cpu_pairs else model.explicit_standard_four(vectors, mask))
            value = model.mod(tensors["features"], v=vectors, mask=mask, uu=pair)
            results.append(value.detach().float().cpu().numpy())
    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.campaign.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new diagnostic directory; existing evidence is immutable")
    source_root = root / "source"
    sys.path.insert(0, str(source_root))
    D = importlib.import_module("scripts.run_relational_part_sporc_inference")
    if Path(D.__file__).resolve() != source_root / "scripts/run_relational_part_sporc_inference.py":
        raise RuntimeError("Another campaign's runtime was imported")
    spec, source, plan, backend = D.load_ready(root)
    D.require_separate_output(output, [source_root, root / "test", root / "backend",
        Path(spec["source_campaign"]), Path(spec["previous_evaluation"]),
        Path(spec["original_parent"]), Path(plan["data_dir"])])
    D.configure_math()
    gpu = D.gpu_identity()
    F = D.F
    import numpy as np
    import torch
    import uproot
    from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, LABEL_BRANCHES, _tokens_from_arrays
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

    reference = F.read_json(root / "reference_plan.json")
    old_report = F.read_json(root / "reference_report.json")
    task = reference["tasks"][0]
    if (task["run_id"], task["seed"]) != ("OFF_RPT_BASE", 101):
        raise ValueError("Diagnostic is fixed to historical BASE seed 101")
    index = spec["probe_file_indices"][0]
    row = plan["files"][index]
    if row["label"] != 0 or row["entries"] < EVENTS or reference["chunk_size"] < EVENTS:
        raise ValueError("Expected the same first 640 QCD events as failed validation")
    data = Path(plan["data_dir"]) / row["name"]
    if F.digest(data) != row["sha256"]:
        raise ValueError("ROOT input bytes changed")
    golden, golden_hash = D.golden_chunk(spec, reference, old_report, index)
    expected = golden["logits_00"][:EVENTS].copy()
    del golden
    with uproot.open(data) as file:
        full = file["tree"].arrays(PARTICLE_READ_BRANCHES + LABEL_BRANCHES,
            entry_start=0, entry_stop=reference["chunk_size"], library="ak")
    arrays = full[:EVENTS]
    tokens, masks = _tokens_from_arrays(arrays, 128)
    all_tokens, all_masks = _tokens_from_arrays(full, 128)
    atol, rtol = spec["validation_policy"]["logit_atol"], spec["validation_policy"]["logit_rtol"]
    chunk_check = {"tokens": difference(tokens, all_tokens[:EVENTS], atol=0, rtol=0),
                   "masks_exact": bool(np.array_equal(masks, all_masks[:EVENTS]))}
    del full, all_tokens, all_masks
    batches = []
    for start in range(0, EVENTS, BATCH):
        inputs = build_particle_transformer_inputs_from_tokens(tokens[start:start+BATCH],
            masks[start:start+BATCH], source_view="offline")
        batches.append({"points": inputs.pf_points, "features": inputs.pf_features,
                        "lorentz_vectors": inputs.pf_vectors, "mask": inputs.pf_mask.astype(bool)})
    output.mkdir(parents=True)
    result = {"contract": VERSION, "campaign_sha256": spec["content_hash"],
        "plan_sha256": plan["content_hash"], "reference_plan_sha256": reference["content_hash"],
        "golden_chunk_sha256": golden_hash, "diagnostic_script_sha256": F.digest(__file__),
        "task": task, "file": row, "events": EVENTS, "batch_size": BATCH,
        "environment": D.runtime_environment(), "historical_environment": reference["environment"],
        "historical_driver_sha256": reference["driver_sha256"], "adapter_driver_sha256": F.digest(F.__file__),
        "gpu": gpu, "original_math_options": spec["math_options"],
        "comparison_atol": atol, "comparison_rtol": rtol, "chunk_prefix_check": chunk_check,
        "authorizes_inference": False, "used_for_model_selection": False,
        "production_settings_changed": False, "variants": {}, "pair_features_cpu_vs_gpu": []}
    device = torch.device("cuda")
    gpu_model = D.load_models(spec, source, [task], device, historical=True)[0]
    result["attention_implementations"] = sorted({type(module).__module__ + "." + type(module).__name__
        for name, module in gpu_model.named_modules() if name.split(".")[-1] == "attn"})
    predictions = {"historical": expected}
    # Reference call reproduces the exact frozen inference helper, including
    # mask checks and trimmer resets. Direct forwards must reproduce this control.
    F.progress("diagnostic_canonical_gpu", events=EVENTS)
    predictions["canonical_gpu"] = F.infer_arrays(arrays, 0, [gpu_model], backend, device)["logits_00"]
    result["canonical_gpu_vs_historical"] = logit_difference(predictions["canonical_gpu"], expected, atol=atol, rtol=rtol)
    snapshots = {**batches[0], "raw_tokens": tokens[:BATCH], "raw_mask": masks[:BATCH]}
    with torch.no_grad():
        for start, batch in zip(range(0, EVENTS, BATCH), batches):
            vectors, mask = torch.from_numpy(batch["lorentz_vectors"]), torch.from_numpy(batch["mask"])
            cpu = gpu_model.explicit_standard_four(vectors, mask).numpy()
            cuda = gpu_model.explicit_standard_four(vectors.to(device), mask.to(device)).cpu().numpy()
            result["pair_features_cpu_vs_gpu"].append({"entry_start": start,
                "channels": pair_difference(cuda, cpu, batch["mask"], atol=atol, rtol=rtol)})
            if start == 0:
                snapshots.update(pair_cpu=cpu.copy(), pair_gpu=cuda.copy())
    with F.atomic_file(output / "first_batch_inputs_and_pairs.npz") as stream:
        np.savez_compressed(stream, **snapshots)

    cpu_model = None
    variants = (
        ("gpu_campaign_math", False, False, False, False),
        ("gpu_tf32_disabled", True, False, False, False),
        ("gpu_tf32_disabled_math_attention", True, True, False, False),
        ("gpu_cpu_pair_features", False, False, True, False),
        ("cpu", False, False, False, True),
    )
    for name, disable_tf32, math_attention, cpu_pairs, on_cpu in variants:
        F.progress("diagnostic_variant", variant=name)
        start = time.monotonic()
        try:
            if on_cpu and cpu_model is None:
                cpu_model = D.load_models(spec, source, [task], torch.device("cpu"), historical=True)[0]
            with diagnostic_math(tf32_disabled=disable_tf32, math_attention=math_attention):
                value = baseline_forward(cpu_model if on_cpu else gpu_model, batches,
                    torch.device("cpu") if on_cpu else device, F.reset_transient_trimmers, cpu_pairs=cpu_pairs)
            predictions[name] = value
            result["variants"][name] = {"seconds": time.monotonic() - start,
                "vs_historical": logit_difference(value, expected, atol=atol, rtol=rtol),
                "vs_canonical_gpu": logit_difference(value, predictions["canonical_gpu"], atol=atol, rtol=rtol)}
        except Exception as exc:
            result["variants"][name] = {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "seconds": time.monotonic() - start}
    control = result["variants"]["gpu_campaign_math"]
    result["direct_forward_reproduces_frozen_helper"] = control.get("vs_canonical_gpu", {}).get("passed", False)
    with F.atomic_file(output / "diagnostic_logits.npz") as stream:
        np.savez_compressed(stream, **predictions)
    result["artifacts"] = {name: F.digest(output / name) for name in
                           ("first_batch_inputs_and_pairs.npz", "diagnostic_logits.npz")}
    F.write_json(output / "diagnostic_report.json", F.hashed(result))
    F.progress("diagnostic_complete", report=str(output / "diagnostic_report.json"), authorizes_inference=False,
        canonical_vs_historical=result["canonical_gpu_vs_historical"],
        variants={name: {"historical_max_error": value.get("vs_historical", {}).get("maximum_absolute_error"),
                         "canonical_gpu_max_error": value.get("vs_canonical_gpu", {}).get("maximum_absolute_error"),
                         "error": value.get("error")} for name, value in result["variants"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
