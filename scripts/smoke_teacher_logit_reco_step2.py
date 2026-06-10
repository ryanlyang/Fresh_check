#!/usr/bin/env python3
"""Smoke-test Step 2 frozen teacher adapters on paired views."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.teachers import (  # noqa: E402
    TEACHER_ARCHITECTURES,
    assert_teacher_frozen,
    load_frozen_teacher,
    summarize_teacher_forward,
)
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
    parser.add_argument("--split", default="model_val")
    parser.add_argument("--max-jets", type=int, default=8)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default="checkpoints/teacher_logit_reco_step2_smoke/smoke_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paired = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.split,
        data_dir=args.data_dir,
        max_jets=args.max_jets,
    )
    teacher = load_frozen_teacher(
        args.teacher_checkpoint,
        architecture=args.teacher_architecture,
        device=args.device,
        max_constits=args.max_constits,
        weight_threshold=args.weight_threshold,
    )
    assert_teacher_frozen(teacher)

    offline_logits = teacher.forward_jet_view_no_grad(paired.offline)
    identity_soft_offline = make_identity_soft_view(
        paired.offline,
        metadata={"smoke_script": Path(__file__).name, "source": "offline_identity"},
    )
    soft_offline_logits = teacher.forward_soft_view_no_grad(identity_soft_offline)
    identity_soft_hlt = make_identity_soft_view(
        paired.hlt,
        metadata={"smoke_script": Path(__file__).name, "source": "hlt_identity"},
    )
    soft_hlt_logits = teacher.forward_soft_view_no_grad(identity_soft_hlt)

    report = {
        "ok": True,
        "purpose": "teacher_logit_reco_step2_frozen_teacher_adapter_smoke",
        "teacher": teacher.metadata,
        "teacher_parameters_frozen": teacher.parameters_frozen(),
        "trainable_parameter_count": teacher.trainable_parameter_count(),
        "paired_views": summarize_paired_jet_views(paired),
        "identity_soft_offline": summarize_soft_view(identity_soft_offline),
        "identity_soft_hlt": summarize_soft_view(identity_soft_hlt),
        "forward_summaries": {
            "offline_view": summarize_teacher_forward(offline_logits, name="offline_view"),
            "identity_soft_offline": summarize_teacher_forward(soft_offline_logits, name="identity_soft_offline"),
            "identity_soft_hlt": summarize_teacher_forward(soft_hlt_logits, name="identity_soft_hlt"),
        },
        "leakage_note": (
            "This smoke test may load offline for teacher-adapter validation only. "
            "No training, selection, or final-test evaluation is performed."
        ),
    }
    output_path = Path(args.output_json)
    save_json(output_path, report)
    print(f"wrote {output_path}")
    print(f"teacher_architecture={teacher.metadata['architecture']}")
    print(f"offline_logits_shape={report['forward_summaries']['offline_view']['shape']}")
    print(f"soft_hlt_logits_shape={report['forward_summaries']['identity_soft_hlt']['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
