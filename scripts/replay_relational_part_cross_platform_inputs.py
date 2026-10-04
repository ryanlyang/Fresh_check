#!/usr/bin/env python3
"""Isolate preprocessing/pair/runtime drift; never authorize production inference.

Capture one historical BASE-101 batch on Tigris CPU, then replay its exact
inputs on SPORC GPU. All old checkpoints, source, and results are read-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
from pathlib import Path
import sys

VERSION = "relational_part_matched_input_replay_v1"
BATCH = 64
INPUT_KEYS = ("features", "points", "lorentz_vectors", "mask")


def load_probe_helpers():
    path = Path(__file__).with_name("diagnose_relational_part_sporc_parity.py")
    spec = importlib.util.spec_from_file_location("matched_input_probe_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(model, inputs, device, reset, *, pairs=None):
    """Identical fresh-trimmer forwards; caller-owned tensors are never mutated."""
    import torch
    reset([model])
    with torch.no_grad():
        batch = {key: torch.from_numpy(inputs[key]).to(device).clone() for key in INPUT_KEYS}
        pair = (model.explicit_standard_four(batch["lorentz_vectors"], batch["mask"])
                if pairs is None else torch.from_numpy(pairs).to(device).clone())
        # Clone here too: a forward must not change the diagnostic pair snapshot.
        saved_pair = pair.detach().float().cpu().numpy().copy()
        value = model.mod(batch["features"], v=batch["lorentz_vectors"],
                          mask=batch["mask"], uu=pair)
        return value.detach().float().cpu().numpy().copy(), saved_pair


def read_arrays(F, directory, report, name):
    import numpy as np
    path = directory / name
    if F.digest(path) != report["artifacts"][name]:
        raise ValueError(f"Diagnostic artifact bytes changed: {path}")
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def validate_snapshot(values):
    import numpy as np
    shapes = {"features": (BATCH, 17, 128), "points": (BATCH, 2, 128),
              "lorentz_vectors": (BATCH, 4, 128), "mask": (BATCH, 1, 128)}
    for key, shape in shapes.items():
        value = values[key]
        dtype = np.bool_ if key == "mask" else np.float32
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid input snapshot: {key}")
    if not values["mask"].any(axis=-1).all():
        raise ValueError("Input snapshot contains an empty jet")


def require_capture_binding(capture, spec, diagnostic):
    if (capture["contract"] != VERSION or capture["phase"] != "capture"
            or capture["architecture"] != "aarch64" or capture["device"] != "cpu"
            or capture["events"] != BATCH
            or capture["campaign_sha256"] != spec["content_hash"]
            or capture["diagnostic_sha256"] != diagnostic["content_hash"]
            or capture["authorizes_inference"] is not False):
        raise ValueError("Tigris capture belongs to another diagnostic/runtime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "replay"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True, help="Previous diagnostic results directory")
    parser.add_argument("--capture", type=Path, help="Completed Tigris capture results directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.phase == "replay") != (args.capture is not None):
        parser.error("--capture is required only for the SPORC replay phase")
    architecture = platform.machine()
    if architecture != ("aarch64" if args.phase == "capture" else "x86_64"):
        raise ValueError("Capture runs on Tigris ARM CPU; replay runs on SPORC x86 GPU")
    root, diagnostic, output = args.campaign.resolve(), args.diagnostic.resolve(), args.output.resolve()
    if (not output.is_relative_to(root / "diagnostics") or output == root / "diagnostics"
            or output.exists() or output.is_relative_to(diagnostic)):
        raise ValueError("Use a NEW results directory under this campaign's diagnostics/")
    H = load_probe_helpers()
    sys.path.insert(0, str(root / "source"))
    from scripts import run_relational_part_sporc_inference as D
    if Path(D.__file__).resolve() != root / "source/scripts/run_relational_part_sporc_inference.py":
        raise ValueError("Imported the wrong frozen source")
    F = D.F
    spec = F.read_json(root / "sporc_campaign.json")
    D.authenticate_files(root / "source", spec["source_files"])
    reference = F.read_json(root / "reference_plan.json")
    previous = F.read_json(diagnostic / "diagnostic_report.json")
    task = reference["tasks"][0]
    if (reference["content_hash"] != spec["reference_plan_sha256"]
            or previous["campaign_sha256"] != spec["content_hash"]
            or previous["reference_plan_sha256"] != reference["content_hash"]
            or previous["task"] != task or (task["run_id"], task["seed"]) != ("OFF_RPT_BASE", 101)
            or previous["file"] != reference["files"][spec["probe_file_indices"][0]]
            or previous["events"] != 640 or previous["batch_size"] != BATCH):
        raise ValueError("Historical diagnostic lineage or batch differs")
    import numpy as np
    import torch
    D.configure_math()
    torch.set_num_threads(2)
    environment = D.runtime_environment()
    if environment["weaver_particle_transformer_sha256"] != reference["environment"]["weaver_particle_transformer_sha256"]:
        raise ValueError("Installed Weaver source differs from the historical model")
    device = torch.device("cpu" if args.phase == "capture" else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("SPORC replay requires a GPU allocation")
    saved = read_arrays(F, diagnostic, previous, "first_batch_inputs_and_pairs.npz")
    logits = read_arrays(F, diagnostic, previous, "diagnostic_logits.npz")
    validate_snapshot(saved)
    model = D.load_models(spec, None, [task], device, historical=True)[0]
    atol, rtol = spec["validation_policy"]["logit_atol"], spec["validation_policy"]["logit_rtol"]
    result = {"contract": VERSION, "phase": args.phase, "architecture": architecture,
              "device": str(device), "events": BATCH, "task": task,
              "campaign_sha256": spec["content_hash"], "diagnostic_sha256": previous["content_hash"],
              "environment": environment, "comparison_atol": atol, "comparison_rtol": rtol,
              "authorizes_inference": False, "production_settings_changed": False,
              "used_for_model_selection": False, "script_sha256": F.digest(__file__),
              "helper_sha256": F.digest(H.__file__), "comparisons": {}}
    compare = lambda a, b: H.logit_difference(a, b, atol=atol, rtol=rtol)
    reset = F.reset_transient_trimmers
    output_values = {}
    if args.phase == "capture":
        import uproot
        from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, _tokens_from_arrays
        from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
        for name in ("jetclass_fresh/jetclass_data.py", "jetclass_fresh/part_inputs.py"):
            if spec["source_files"][name] != reference["source_files"][name]:
                raise ValueError("Input-building source differs from the historical evaluation")
        data = Path(reference["data_dir"]) / previous["file"]["name"]
        if F.digest(data) != previous["file"]["sha256"]:
            raise ValueError("ROOT input bytes changed")
        with uproot.open(data) as file:
            arrays = file["tree"].arrays(PARTICLE_READ_BRANCHES, entry_start=0,
                                        entry_stop=640, library="ak")
        tokens, mask = _tokens_from_arrays(arrays, 128)
        built = build_particle_transformer_inputs_from_tokens(tokens[:BATCH], mask[:BATCH], source_view="offline")
        native = dict(features=built.pf_features, points=built.pf_points,
                      lorentz_vectors=built.pf_vectors, mask=built.pf_mask)
        validate_snapshot(native)
        if (not np.array_equal(native["mask"], saved["mask"])
                or not np.array_equal(tokens[:BATCH, :, 4:10], saved["raw_tokens"][:, :, 4:10])):
            raise ValueError("Masks or charge/PID differ; this is not only a continuous numerical probe")
        native_logits, native_pairs = replay(model, native, device, reset)
        swapped_logits, swapped_pairs = replay(model, saved, device, reset)
        fixed_logits, _ = replay(model, saved, device, reset, pairs=saved["pair_cpu"])
        result["comparisons"] = {
            "native_inputs_vs_historical": compare(native_logits, logits["historical"][:BATCH]),
            "sporc_inputs_vs_historical": compare(swapped_logits, logits["historical"][:BATCH]),
            "input_swap_effect_same_runtime": compare(swapped_logits, native_logits),
            "sporc_inputs_vs_sporc_cpu": compare(swapped_logits, logits["cpu"][:BATCH]),
            "sporc_inputs_and_pairs_vs_sporc_cpu": compare(fixed_logits, logits["cpu"][:BATCH]),
        }
        result["input_differences"] = {key: H.difference(native[key], saved[key], atol=0, rtol=0) for key in INPUT_KEYS}
        result["pair_input_swap_effect"] = H.pair_difference(native_pairs, swapped_pairs, native["mask"], atol=atol, rtol=rtol)
        result["pair_same_inputs_cross_platform"] = H.pair_difference(swapped_pairs, saved["pair_cpu"], native["mask"], atol=atol, rtol=rtol)
        output_values.update(native, pairs=native_pairs, native_logits=native_logits,
                             sporc_inputs_logits=swapped_logits, sporc_inputs_fixed_pairs_logits=fixed_logits)
    else:
        capture_dir = args.capture.resolve()
        if output.is_relative_to(capture_dir) or capture_dir.is_relative_to(output):
            raise ValueError("Replay output must be separate from the Tigris capture")
        capture = F.read_json(capture_dir / "replay_report.json")
        require_capture_binding(capture, spec, previous)
        native = read_arrays(F, capture_dir, capture, "replay_arrays.npz")
        validate_snapshot(native)
        result["capture_sha256"] = capture["content_hash"]
        result["gpu"] = D.gpu_identity()
        for name, inputs, pairs, target in (
            ("sporc_inputs", saved, None, logits["canonical_gpu"][:BATCH]),
            ("tigris_inputs", native, None, native["native_logits"]),
            ("tigris_inputs_and_pairs", native, native["pairs"], native["native_logits"]),
            ("sporc_inputs_and_pairs", saved, saved["pair_cpu"], logits["cpu"][:BATCH]),
        ):
            value, _ = replay(model, inputs, device, reset, pairs=pairs)
            output_values[name] = value
            result["comparisons"][name] = {"vs_historical": compare(value, logits["historical"][:BATCH]),
                                           "vs_reference_runtime": compare(value, target)}
    output.mkdir(parents=True, exist_ok=False)
    with F.atomic_file(output / "replay_arrays.npz") as stream:
        np.savez_compressed(stream, **output_values)
    result["artifacts"] = {"replay_arrays.npz": F.digest(output / "replay_arrays.npz")}
    F.write_json(output / "replay_report.json", F.hashed(result))
    F.progress("matched_input_replay_complete", phase=args.phase, authorizes_inference=False,
               report=str(output / "replay_report.json"), comparisons=result["comparisons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
