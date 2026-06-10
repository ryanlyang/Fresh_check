#!/usr/bin/env python3
"""Smoke-test Step 1 teacher-logit reconstruction view interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.part_inputs import summarize_particle_transformer_inputs  # noqa: E402
from teacher_logit_reco.views import (  # noqa: E402
    load_paired_jet_views,
    make_identity_soft_view,
    summarize_paired_jet_views,
    summarize_soft_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="model_train")
    parser.add_argument("--max-jets", type=int, default=16)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default="checkpoints/teacher_logit_reco_step1_smoke/smoke_report.json",
    )
    parser.add_argument("--verify-label-branches", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paired = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.split,
        data_dir=args.data_dir,
        max_jets=args.max_jets,
        verify_label_branches=bool(args.verify_label_branches),
    )
    identity_soft_hlt = make_identity_soft_view(
        paired.hlt,
        metadata={"smoke_script": Path(__file__).name},
    )
    soft_inputs = identity_soft_hlt.to_particle_transformer_inputs(
        weight_threshold=float(args.weight_threshold),
    )
    offline_soft = make_identity_soft_view(
        paired.offline,
        metadata={"smoke_script": Path(__file__).name, "source": "offline_identity"},
    )
    offline_inputs = offline_soft.to_particle_transformer_inputs(
        weight_threshold=float(args.weight_threshold),
    )
    report = {
        "ok": True,
        "purpose": "teacher_logit_reco_step1_view_interface_smoke",
        "paired_views": summarize_paired_jet_views(paired),
        "identity_soft_hlt": summarize_soft_view(identity_soft_hlt),
        "identity_soft_hlt_part_inputs": summarize_particle_transformer_inputs(soft_inputs),
        "identity_soft_offline": summarize_soft_view(offline_soft),
        "identity_soft_offline_part_inputs": summarize_particle_transformer_inputs(offline_inputs),
        "leakage_note": (
            "This smoke test may load offline for interface validation only. "
            "No training, selection, or final-test evaluation is performed."
        ),
    }
    output_path = Path(args.output_json)
    save_json(output_path, report)
    print(f"wrote {output_path}")
    print(f"n_jets={report['paired_views']['n_jets']}")
    print(f"hlt_shape={report['paired_views']['hlt_shape']}")
    print(f"offline_shape={report['paired_views']['offline_shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
