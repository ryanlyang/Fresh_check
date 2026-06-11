#!/usr/bin/env python3
"""Write a compact cross-architecture final report from frozen JSON outputs.

This intentionally does not load model checkpoints.  It summarizes the Step 8/9
fusion report enough for the Step 10 Slurm graph to end in inspectable files.
The richer paper-facing report is expanded in the next implementation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def best_fusers_by_group(fusion_report: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for group_name, group in (fusion_report.get("groups") or {}).items():
        candidates = []
        for fuser_name, fuser in (group.get("fusers") or {}).items():
            if fuser.get("status") == "skipped":
                continue
            metrics = fuser.get("metrics") or {}
            stack_val = (metrics.get("stack_val") or {}).get("accuracy")
            final_test = (metrics.get("final_test") or {}).get("accuracy")
            if stack_val is None:
                continue
            candidates.append(
                {
                    "fuser": fuser_name,
                    "stack_val_accuracy": float(stack_val),
                    "final_test_accuracy": None if final_test is None else float(final_test),
                }
            )
        candidates.sort(key=lambda row: (row["stack_val_accuracy"], row["final_test_accuracy"] or -1.0), reverse=True)
        rows[group_name] = {
            "ok": bool(group.get("ok", False)),
            "n_models": int(group.get("n_models", 0)),
            "best_by_stack_val": candidates[0] if candidates else None,
            "all_candidates": candidates,
        }
    return rows


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Cross-Architecture 16x4 Experiment",
        "",
        f"ok: `{report.get('ok')}`",
        f"fusion_ok: `{report.get('fusion_ok')}`",
        f"controls_ok: `{report.get('controls_ok')}`",
        f"audit_ok: `{report.get('audit_ok')}`",
        "",
        "## Best Fusers By Group",
        "",
        "| group | ok | n_models | best_fuser | stack_val_acc | final_test_acc |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for group_name, row in report.get("best_fusers_by_group", {}).items():
        best = row.get("best_by_stack_val") or {}
        lines.append(
            "| {group} | {ok} | {n_models} | {fuser} | {stack_val} | {final_test} |".format(
                group=group_name,
                ok=row.get("ok"),
                n_models=row.get("n_models"),
                fuser=best.get("fuser", ""),
                stack_val="" if best.get("stack_val_accuracy") is None else f"{best['stack_val_accuracy']:.6f}",
                final_test="" if best.get("final_test_accuracy") is None else f"{best['final_test_accuracy']:.6f}",
            )
        )
    suspicious = report.get("suspicious_flags") or []
    lines.extend(["", "## Suspicious Flags", ""])
    if suspicious:
        for flag in suspicious:
            lines.append(f"- `{flag.get('severity', 'warning')}` {flag.get('name')} {flag}")
    else:
        lines.append("No suspicious flags were reported.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--prediction-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fusion_path = Path(args.fusion_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_report = load_json(fusion_path)
    final_report = {
        "ok": bool(fusion_report.get("ok", False)),
        "experiment": "crossarch_16x4_final_report",
        "root_dir": args.root_dir,
        "prediction_dir": args.prediction_dir,
        "fusion_report": str(fusion_path),
        "fusion_ok": bool(fusion_report.get("ok", False)),
        "controls_ok": bool((fusion_report.get("controls_summary") or {}).get("ok", False)),
        "audit_ok": bool((fusion_report.get("audit_summary") or {}).get("ok", False)),
        "best_fusers_by_group": best_fusers_by_group(fusion_report),
        "suspicious_flags": list(fusion_report.get("suspicious_flags") or []),
        "source": "frozen JSON reports only; no model checkpoints loaded",
    }
    json_path = output_dir / "crossarch_final_report.json"
    md_path = output_dir / "crossarch_final_report.md"
    json_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, final_report)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
