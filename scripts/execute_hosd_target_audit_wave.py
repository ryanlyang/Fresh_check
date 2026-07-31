#!/usr/bin/env python3
"""Audit every Stage-B cache by split and publish one fail-closed summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_hosd_targets import main as audit_main  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
    load_hashed_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_AUDIT_CONTRACT,
    TARGET_AUDIT_WAVE_CONTRACT,
    with_content_hash,
    write_immutable_json,
)


SPLITS = ("model_train", "val_stop", "val_design")


def _labels(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        split, separator, raw = value.partition("=")
        if not separator or split in output:
            raise ValueError("--label-manifest requires unique SPLIT=PATH")
        output[split] = Path(raw)
    if set(output) != set(SPLITS):
        raise ValueError("target-audit label split coverage differs")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--label-manifest", action="append", default=[], required=True
    )
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    labels = _labels(args.label_manifest)
    rows = []
    for split in SPLITS:
        replicas = range(4) if split == "model_train" else (0,)
        target_caches = [
            ("canonical", root / "targets" / "canonical" / split),
            *[
                (
                    f"hlt_replica_{replica}",
                    root
                    / "targets"
                    / "hlt_analogues"
                    / split
                    / f"replica_{replica}",
                )
                for replica in replicas
            ],
            (
                "teacher_o_base",
                root / "teachers" / "outputs" / split / "O_BASE",
            ),
            (
                "teacher_o_fullrel",
                root / "teachers" / "outputs" / split / "O_FULLREL",
            ),
        ]
        residual_caches = [
            (
                f"residual_replica_{replica}",
                root
                / "targets"
                / "residuals"
                / split
                / f"replica_{replica}",
            )
            for replica in replicas
        ]
        for kind, caches, normalizer in (
            (
                "targets",
                target_caches,
                root
                / "normalization"
                / "target_500k"
                / "normalizer_manifest.json",
            ),
            (
                "residuals",
                residual_caches,
                root
                / "normalization"
                / "residual_500k"
                / "normalizer_manifest.json",
            ),
        ):
            output = (
                root
                / "targets"
                / "audits"
                / f"{split}__{kind}.json"
            )
            command = [
                "--campaign-root",
                str(root),
                "--normalizer",
                str(normalizer),
                "--label-manifest",
                str(labels[split]),
                "--output",
                str(output),
            ]
            for name, path in caches:
                command.extend(["--cache", f"{name}={path}"])
            audit_main(command)
            artifact = load_hashed_json(
                output, expected_contract=TARGET_AUDIT_CONTRACT
            )
            if artifact.get("source") != campaign["source"]:
                raise ValueError("target audit wave source differs")
            rows.append(
                {
                    "split": split,
                    "audit_kind": kind,
                    "audit_sha256": artifact["content_hash"],
                    "cache_count": len(artifact["caches"]),
                    "unusual_correlation_count": len(
                        artifact["unusual_correlation_reports"]
                    ),
                }
            )
    summary = with_content_hash(
        {
            "contract": TARGET_AUDIT_WAVE_CONTRACT,
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "rows": rows,
            "row_count": len(rows),
            "split_order": list(SPLITS),
            "all_cache_families_audited": True,
            "unusual_correlations_change_target_semantics": False,
            "scientific_underperformance_can_fail_or_cancel": False,
        }
    )
    output = root / "targets" / "target_audit.json"
    write_immutable_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
