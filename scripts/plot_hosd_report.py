#!/usr/bin/env python3
"""Render deterministic HOSD accuracy and rejection comparison plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    HOSD_REPORT_CONTRACT,
    PLOT_BUNDLE_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    report = load_hashed_json(args.report, expected_contract=HOSD_REPORT_CONTRACT)
    if report.get("source") != campaign["source"]:
        raise ValueError("plot report source differs")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(report["rows"])
    names = [str(row["graph_id"]) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        (
            "balanced_accuracy_difference",
            "accuracy_difference_vs_h_base",
            "Balanced accuracy difference vs H_BASE",
        ),
        (
            "mean_log_rejection_difference",
            "mean_log_rejection_difference_vs_h_base",
            "Mean log-rejection difference vs H_BASE",
        ),
    )
    outputs = {}
    for filename, field, ylabel in definitions:
        values = [float(row.get(field, 0.0)) for row in rows]
        figure, axis = plt.subplots(figsize=(max(8, 0.45 * len(rows)), 5))
        colors = ["#2b8cbe" if value >= 0 else "#de2d26" for value in values]
        axis.bar(range(len(rows)), values, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(rows)), names, rotation=75, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(report["title"])
        figure.tight_layout()
        path = args.output_dir / f"{filename}.png"
        figure.savefig(
            path,
            dpi=160,
            metadata={"Software": "HOSD deterministic plotter v1"},
        )
        plt.close(figure)
        outputs[filename] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_order": names,
            "field": field,
        }
    artifact = with_content_hash(
        {
            "contract": PLOT_BUNDLE_CONTRACT,
            "schema_version": 1,
            "source": campaign["source"],
            "report_sha256": report["content_hash"],
            "plots": outputs,
            "positive_color": "#2b8cbe",
            "negative_color": "#de2d26",
            "negative_rows_rendered": True,
            "manual_edits": False,
        }
    )
    publication = write_immutable_json(args.manifest, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
