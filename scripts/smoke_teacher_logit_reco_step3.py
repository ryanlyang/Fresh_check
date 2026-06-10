#!/usr/bin/env python3
"""Smoke-test Step 3 global transformer reconstructor on paired views."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json  # noqa: E402
from teacher_logit_reco.global_transformer import (  # noqa: E402
    GlobalTransformerReconstructor,
    GlobalTransformerReconstructorConfig,
)
from teacher_logit_reco.views import load_paired_jet_views, summarize_paired_jet_views, summarize_soft_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="model_val")
    parser.add_argument("--max-jets", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-extra-candidates", type=int, default=8)
    parser.add_argument("--output-json", default="checkpoints/teacher_logit_reco_step3_smoke/smoke_report.json")
    return parser.parse_args()


def main() -> int:
    torch = require_torch()
    args = parse_args()
    device = resolve_device(args.device)
    paired = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.split,
        data_dir=args.data_dir,
        max_jets=args.max_jets,
    )
    config = GlobalTransformerReconstructorConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_extra_candidates=args.num_extra_candidates,
        dropout=0.0,
    )
    model = GlobalTransformerReconstructor(config).to(device).eval()
    hlt_tokens = torch.as_tensor(paired.hlt.tokens, dtype=torch.float32, device=device)
    hlt_mask = torch.as_tensor(paired.hlt.mask, dtype=torch.bool, device=device)
    labels = torch.as_tensor(paired.labels, dtype=torch.long, device=device)
    with torch.no_grad():
        soft = model(
            hlt_tokens,
            hlt_mask,
            labels=labels,
            jet_ids=paired.jet_ids,
            split=paired.split,
        )
    report = {
        "ok": True,
        "purpose": "teacher_logit_reco_step3_global_transformer_forward_smoke",
        "config": config.to_dict(),
        "paired_views": summarize_paired_jet_views(paired),
        "soft_view": summarize_soft_view(soft),
        "aux_diagnostics": dict(soft.aux.get("diagnostics", {})),
        "leakage_note": "This smoke test runs HLT-only reconstruction forward pass only; no training or selection.",
    }
    output_path = Path(args.output_json)
    save_json(output_path, report)
    print(f"wrote {output_path}")
    print(f"soft_tokens_shape={list(soft.tokens.shape)}")
    print(f"soft_weights_min={float(soft.weights.min().detach().cpu())}")
    print(f"soft_weights_max={float(soft.weights.max().detach().cpu())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
