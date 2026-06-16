#!/usr/bin/env python3
"""Summarize split cross-architecture fusion reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", nargs="+", default=["frozen", "adapted"])
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[
            "hlt4",
            "all16",
            "all16_plus_hlt4",
            "cross12_plus_hlt4",
            "part_teacher4_plus_hlt4",
            "pn_teacher4_plus_hlt4",
            "mixed4_plus_hlt4",
        ],
    )
    parser.add_argument("--bundles", nargs="+", default=["main", "gated", "controls"])
    parser.add_argument("--require-expected", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    group_rows = []
    best_row: dict[str, Any] | None = None
    for report_group_name, group_report in sorted((report.get("groups") or {}).items()):
        fusers = group_report.get("fusers") or {}
        for fuser_name, fuser_report in sorted(fusers.items()):
            metrics = ((fuser_report.get("metrics") or {}).get("final_test") or {})
            accuracy = metrics.get("accuracy")
            row = {
                "report_group_name": report_group_name,
                "fuser": fuser_name,
                "status": fuser_report.get("status", "ok"),
                "final_test_accuracy": accuracy,
                "final_test_cross_entropy": metrics.get("cross_entropy"),
                "final_test_macro_ovr_auc": metrics.get("macro_ovr_auc"),
            }
            group_rows.append(row)
            if accuracy is not None and (best_row is None or float(accuracy) > float(best_row["final_test_accuracy"])):
                best_row = row
    return {
        "ok": bool(report.get("ok", False)),
        "controls_ok": bool((report.get("controls_summary") or {}).get("ok", True)),
        "audit_ok": bool((report.get("audit_summary") or {}).get("ok", False)),
        "suspicious_flag_count": int((report.get("controls_summary") or {}).get("suspicious_flag_count", 0)),
        "best_fuser": best_row,
        "fusers": group_rows,
    }


def main() -> int:
    args = parse_args()
    split_root = Path(args.split_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for family in args.families:
        for group in args.groups:
            for bundle in args.bundles:
                report_path = split_root / family / group / bundle / "fusion_report.json"
                if not report_path.exists():
                    missing.append(
                        {
                            "family": family,
                            "group": group,
                            "bundle": bundle,
                            "report_path": str(report_path),
                        }
                    )
                    continue
                report = read_json(report_path)
                summary = summarize_report(report)
                rows.append(
                    {
                        "family": family,
                        "group": group,
                        "bundle": bundle,
                        "report_path": str(report_path),
                        **summary,
                    }
                )

    overall_ok = not missing and all(bool(row["ok"]) for row in rows)
    payload = {
        "ok": bool(overall_ok),
        "split_root": str(split_root),
        "expected": {
            "families": list(args.families),
            "groups": list(args.groups),
            "bundles": list(args.bundles),
            "n_reports": int(len(args.families) * len(args.groups) * len(args.bundles)),
        },
        "found_reports": int(len(rows)),
        "missing_reports": missing,
        "rows": rows,
    }

    json_path = output_dir / "crossarch_split_fusion_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Crossarch Split Fusion Summary",
        "",
        f"ok: {payload['ok']}",
        f"found_reports: {payload['found_reports']}",
        f"missing_reports: {len(missing)}",
        "",
        "| family | group | bundle | ok | controls_ok | best_fuser | final_test_acc | flags |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        best = row.get("best_fuser") or {}
        accuracy = best.get("final_test_accuracy")
        accuracy_text = "" if accuracy is None else f"{float(accuracy):.6f}"
        lines.append(
            "| {family} | {group} | {bundle} | {ok} | {controls_ok} | {best_fuser} | {acc} | {flags} |".format(
                family=row["family"],
                group=row["group"],
                bundle=row["bundle"],
                ok=row["ok"],
                controls_ok=row["controls_ok"],
                best_fuser=best.get("fuser", ""),
                acc=accuracy_text,
                flags=row["suspicious_flag_count"],
            )
        )
    if missing:
        lines.extend(["", "## Missing Reports", ""])
        for item in missing:
            lines.append(f"- {item['family']}/{item['group']}/{item['bundle']}: {item['report_path']}")
    md_path = output_dir / "crossarch_split_fusion_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.require_expected and missing:
        raise SystemExit(f"Missing {len(missing)} expected split fusion reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
