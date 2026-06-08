#!/usr/bin/env python3
"""Print V2 Step 3 reconstruction-loss diagnostics for one real fixed-HLT batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.reconstructor import (  # noqa: E402
    RECONSTRUCTOR_VARIANT_NAMES,
    PairedReconstructionDataset,
    StageAReconstructorTrainConfig,
    build_reconstructor,
    get_reconstructor_variant_config,
    load_stage_a_views,
    make_reconstruction_loader,
    reconstruction_loss,
)
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Five-way split manifest path")
    parser.add_argument("--hlt-cache-dir", required=True, help="Fixed-HLT cache directory")
    parser.add_argument("--data-dir", default=None, help="JetClass data dir override")
    parser.add_argument("--variant", choices=RECONSTRUCTOR_VARIANT_NAMES, default="m2_base")
    parser.add_argument("--split", default="model_train", choices=["model_train", "model_val"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-jets", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def scalarize(value):
    if hasattr(value, "detach"):
        value = value.detach()
        if value.numel() == 1:
            return float(value.item())
        return value.cpu().numpy().tolist()
    return value


def main() -> int:
    args = parse_args()
    torch = require_torch()
    set_training_seed(args.seed)
    device = resolve_device(args.device)
    variant_config = get_reconstructor_variant_config(args.variant)
    train_config = StageAReconstructorTrainConfig(
        output_dir="/tmp/v2_step3_diagnostic_unused",
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        variant=args.variant,
        batch_size=args.batch_size,
        device=args.device,
    )
    hlt_view, offline_view = load_stage_a_views(train_config, args.split)
    dataset = PairedReconstructionDataset(hlt_view, offline_view, max_jets=args.max_jets)
    loader = make_reconstruction_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        seed=args.seed,
    )
    batch = next(iter(loader))
    batch = {key: value.to(device) for key, value in batch.items()}
    model = build_reconstructor(variant_config).to(device)
    model.eval()
    with torch.no_grad():
        output = model(batch["hlt_tokens"], batch["hlt_mask"])
        loss, diagnostics = reconstruction_loss(
            output,
            hlt_tokens=batch["hlt_tokens"],
            hlt_mask=batch["hlt_mask"],
            offline_tokens=batch["offline_tokens"],
            offline_mask=batch["offline_mask"],
            config=variant_config,
        )
    payload = {
        "experiment_step": "v2_step3_reconstruction_loss_real_batch_diagnostic",
        "variant": args.variant,
        "split": args.split,
        "batch_size": int(batch["hlt_tokens"].shape[0]),
        "loss": float(loss.detach().item()),
        "diagnostics": {key: scalarize(value) for key, value in sorted(diagnostics.items())},
        "matching_mode": variant_config.matching_mode,
        "max_matching_candidates": int(variant_config.max_matching_candidates),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
