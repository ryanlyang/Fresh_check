#!/usr/bin/env python3
"""Materialize deterministic Step-8 structural particle-view controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    STRUCTURAL_CONTROL_IDS,
    apply_particle_view_control,
    with_content_hash,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--control-id", choices=STRUCTURAL_CONTROL_IDS, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with np.load(args.input_npz, allow_pickle=False) as source:
        view = torch.from_numpy(np.asarray(source["view"]))
        mask = torch.from_numpy(np.asarray(source["mask"]))
        particle_pt = (
            torch.from_numpy(np.asarray(source["particle_pt"]))
            if "particle_pt" in source
            else None
        )
        labels = (
            torch.from_numpy(np.asarray(source["labels"]))
            if "labels" in source
            else None
        )
    controlled, diagnostics = apply_particle_view_control(
        view,
        mask,
        control_id=args.control_id,
        seed=args.seed,
        particle_pt=particle_pt,
        labels=labels,
    )
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        view=controlled.cpu().numpy(),
        mask=mask.cpu().numpy(),
    )
    report = with_content_hash(
        {
            "contract": "particle_view_control_materialization_v1",
            "input_npz": str(Path(args.input_npz).resolve()),
            "output_npz": str(output.resolve()),
            "diagnostics": diagnostics,
        }
    )
    write_immutable_json(args.report, report)
    print(json.dumps({"control_id": args.control_id, "report_hash": report["content_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
