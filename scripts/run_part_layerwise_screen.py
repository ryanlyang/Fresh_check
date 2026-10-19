#!/usr/bin/env python3
"""Two fresh layerwise Ranger runs, with read-only shared-bias controls."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from part_layerwise_screen import protocol as P
from part_layerwise_screen.model import build_model, BASE
from part_screen_common.engine import ScreenEngine, training_step, tensors, progress
from part_screen_common.contracts import runtime_identity
from part_ev_screen.data import build_loader, shutdown_loader, batch_fingerprint
from part_ev_screen.precision import model_precision, audit_precision
from part_ev_fullscale.model import build_model as reference_model
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer


def prepare(output):
    launch = P.read_json(output / "launch.json")
    donor = Path(launch["donor"]).resolve()
    if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError("Campaign and donor must be disjoint")
    # Both array tasks may start together. Only one creates the campaign.
    with P.file_lock(output / ".prepare.lock"):
        if (output / "experiment_spec.json").exists():
            spec = P.load_campaign(output)
            if spec["launch"] != launch:
                raise ValueError("Launch settings changed")
            return spec
        if shutil.disk_usage(output).free < 3 * 1024**3:
            raise OSError("Need 3 GiB free for new results/checkpoints; nothing deleted")
        spec = P.create_spec(Path(__file__).resolve().parents[1], launch)
        P.write_json(output / "experiment_spec.json", spec)
        progress("prepared", campaign_sha256=spec["content_hash"], admission=spec["admission"])
        return spec


def initial_parity(model, task, batch, device):
    rng, training = capture_rank_state(model), model.training
    try:
        seed_everything(task["seed"])
        reference = reference_model(BASE[task["architecture"]], trim=True).to(device).eval()
        model.eval()
        values, _ = tensors(batch, device)
        with torch.no_grad(), model_precision(model, device, training=False):
            expected = reference(*(v[:8] for v in values))
            actual = model(*(v[:8] for v in values))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return dict(passed=True, events=len(actual), max_absolute_error=float((actual-expected).abs().max()))
    finally:
        restore_rank_state(model, rng)
        model.train(training)


def diagnostics(model, batch, device):
    # Read-only parameter divergence; no extra forwards or RNG/data consumption.
    heads = model.mod.pair_embed.reference.embed.heads
    first = torch.cat([p.detach().float().flatten() for p in heads[0].parameters()])
    return dict(bias_head_distance_from_layer0=[float((torch.cat([
        p.detach().float().flatten() for p in head.parameters()])-first).norm()) for head in heads])


class DirectEngine(ScreenEngine):
    def startup(self, output, spec, index):
        """One real-batch check inside training; discard every probe state."""
        task = spec["tasks"][index]
        device, runtime = self.gpu_setup(spec)
        run = P.task_directory(output, index)
        with P.file_lock(run / ".startup.lock", blocking=False):
            model = self.make_model(task, device)
            if sum(p.numel() for p in model.parameters()) != spec["parameters"][task["architecture"]]:
                raise ValueError("Parameter count mismatch")
            loader, _ = build_loader("train", spec["inventory"], 512, seed=task["seed"])
            try:
                iterator = iter(loader)
                seed_everything(task["seed"])
                batch = next(iterator)
                fingerprint = batch_fingerprint(batch)
                if fingerprint != spec["donor"]["input_fingerprints"]["0"]:
                    raise ValueError("First training batch differs from donor")
                parity = initial_parity(model, task, batch, device)
                optimizer = build_optimizer(model, spec["recipes"][task["recipe"]])
                scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
                step = training_step(model, optimizer, scaler, batch, device,
                    spec["recipes"][task["recipe"]], 0, check_precision=True)
                model.eval()
                values, _ = tensors(batch, device)
                with torch.no_grad(), model_precision(model, device, training=False), audit_precision(
                        model, expected_dtype=torch.float32) as fp32:
                    logits = model(*(v[:8] for v in values))
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Nonfinite FP32 startup logits")
                record = dict(task=task, campaign_sha256=spec["content_hash"], runtime=runtime,
                    input_fingerprint=fingerprint, initial_parity=parity, training=step, validation_precision=fp32,
                    probe_discarded=True, sustained_benchmark=False)
                P.write_json(run / "startup.json", record)
                progress("startup_checked", **record)
            finally:
                shutdown_loader(loader)

    def train(self, args):
        spec = prepare(args.output)
        self.startup(args.output, spec, args.task_index)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # The shared engine seeds and constructs fresh model/optimizer/scaler,
        # or restores last.pt. The probe is never used to initialize training.
        super().train(args)


ENGINE = DirectEngine(P, build_model, initial_parity, diagnostics)


def status(output):
    spec = P.load_campaign(output, check_environment=False)
    entries = [(t, P.task_directory(output, i)) for i, t in enumerate(spec["tasks"])]
    entries += [(t, Path(spec["donor"]["path"]) / "runs" / name / f"seed_{t['seed']}")
                for name, t in spec["donor"]["tasks"].items()]
    rows = {t["run_id"]: {c["check"]: c for c in (P.read_json(p) for p in sorted(
        (run / "validation").glob("check_*.json")))} for t, run in entries}
    print("Checkpoint validation accuracy (not final 5M validation):")
    print("Check" + "".join(f" {name:>23s}" for name in rows))
    for check in range(1, 11):
        values = [f"{100*r[check]['accuracy']:.4f}%" if check in r else "--" for r in rows.values()]
        print(f"{check:5d}" + "".join(f" {v:>23s}" for v in values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "lock", "evaluate", "report", "status"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase in ("train", "evaluate"):
        P.task_directory(args.output, args.task_index)
        getattr(ENGINE, args.phase)(args)
    elif args.phase == "lock":
        ENGINE.lock(args.output, P.load_campaign(args.output))
    elif args.phase == "status":
        status(args.output)
    else:
        from part_layerwise_screen.reporting import report
        return 0 if report(args.output) else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
