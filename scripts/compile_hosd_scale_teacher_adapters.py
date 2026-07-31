#!/usr/bin/env python3
"""Compile source-bound Stage-J teacher-inference adapter configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    SCALE_INPUT_COMPLETION_CONTRACT,
    SCALE_NORMALIZER_COMPLETION_CONTRACT,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TEACHER_LOCK_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--screening-registry", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    lock = load_hashed_json(
        root / "scale_up" / "teachers" / "teacher_lock.json",
        expected_contract=TEACHER_LOCK_CONTRACT,
    )
    inputs = load_hashed_json(
        root / "scale_up" / "inputs" / "completion.json",
        expected_contract=SCALE_INPUT_COMPLETION_CONTRACT,
    )
    normalizers = load_hashed_json(
        root / "scale_up" / "normalization" / "completion.json",
        expected_contract=SCALE_NORMALIZER_COMPLETION_CONTRACT,
    )
    screening = load_hashed_json(args.screening_registry)
    for name, artifact in (
        ("teacher lock", lock),
        ("input completion", inputs),
        ("normalizer completion", normalizers),
        ("screening registry", screening),
    ):
        if (
            artifact.get("source") is not None
            and artifact.get("source") != campaign["source"]
        ):
            raise ValueError(f"scale adapter {name} source differs")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "teacher_ids": ["O_BASE", "O_FULLREL"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_root = root / "scale_up" / "teacher_outputs" / "adapter_configs"
    offline_row = next(
        row
        for row in inputs["rows"]
        if row["view_id"] == "offline:scale_train"
    )
    common = {
        "input_npz": str(
            (
                root
                / "scale_up"
                / "inputs"
                / "offline"
                / "scale_train.npz"
            ).resolve()
        ),
        "input_npz_sha256": offline_row["npz_sha256"],
        "screening_registry": str(args.screening_registry.resolve()),
        "relation_normalizer": normalizers["artifact_paths"][
            "offline_relation"
        ],
        "device": str(args.device),
        "campaign_spec_sha256": campaign["content_hash"],
        "teacher_lock_sha256": lock["content_hash"],
        "scale_input_completion_sha256": inputs["content_hash"],
        "scale_normalizer_completion_sha256": normalizers["content_hash"],
    }
    configs = {
        "O_BASE": with_content_hash(
            {
                "contract": "hosd_scale_teacher_adapter_config_v1",
                "schema_version": 1,
                "source": campaign["source"],
                "teacher_id": "O_BASE",
                **common,
            }
        ),
        "O_FULLREL": with_content_hash(
            {
                "contract": "hosd_scale_teacher_adapter_config_v1",
                "schema_version": 1,
                "source": campaign["source"],
                "teacher_id": "O_FULLREL",
                **common,
                "region_normalizer": normalizers["artifact_paths"][
                    "offline_region"
                ],
                "tree_cache_dir": str(
                    (
                        root
                        / "scale_up"
                        / "trees"
                        / "offline"
                        / "scale_train_exclusive_ca_v1"
                    ).resolve()
                ),
            }
        ),
    }
    paths = {}
    for teacher_id, config in configs.items():
        path = output_root / f"{teacher_id}.json"
        write_immutable_json(path, config)
        paths[teacher_id] = {
            "path": str(path.resolve()),
            "content_hash": config["content_hash"],
        }
    completion = with_content_hash(
        {
            "contract": "hosd_scale_teacher_adapter_wave_v1",
            "schema_version": 1,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "teacher_lock_sha256": lock["content_hash"],
            "scale_input_completion_sha256": inputs["content_hash"],
            "scale_normalizer_completion_sha256": normalizers["content_hash"],
            "screening_registry_sha256": screening["content_hash"],
            "configs": paths,
            "teacher_ids": ["O_BASE", "O_FULLREL"],
            "label_access": False,
        }
    )
    output = output_root / "completion.json"
    write_immutable_json(output, completion)
    print(
        json.dumps(
            {
                "completion_sha256": completion["content_hash"],
                "output": str(output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
