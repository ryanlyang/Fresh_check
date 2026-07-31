#!/usr/bin/env python3
"""Finalize complete locked O_BASE/O_FULLREL teacher-output cache lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_teacher_output_manifest,
    load_and_validate_campaign,
    load_hashed_json,
    validate_teacher_lock,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STAGE_B_WAVE_COMPLETION_CONTRACT,
    TEACHER_LOCK_CONTRACT,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--teacher-lock", required=True, type=Path)
    parser.add_argument("--wave-completion", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    lock = load_hashed_json(args.teacher_lock, expected_contract=TEACHER_LOCK_CONTRACT)
    validate_teacher_lock(lock, source=campaign["source"])
    wave = load_hashed_json(
        args.wave_completion,
        expected_contract=STAGE_B_WAVE_COMPLETION_CONTRACT,
    )
    if (
        wave.get("source") != campaign["source"]
        or wave.get("wave_kind") != "teacher_output"
        or not wave.get("exact_coordinate_coverage")
    ):
        raise ValueError("teacher output wave completion semantics differ")
    rows = {
        (row["split"], row["teacher_id"]): row for row in wave["rows"]
    }
    artifact = build_teacher_output_manifest(
        teacher_lock=lock,
        cache_manifest_hashes_by_split={
            split: {
                "T_OFFLINE_LOGITS_O_BASE": rows[
                    (split, "O_BASE")
                ]["cache_manifest_sha256"],
                "T_OFFLINE_POOLED_LATENT": rows[
                    (split, "O_BASE")
                ]["cache_manifest_sha256"],
                "T_OFFLINE_LOGITS_O_FULLREL": rows[
                    (split, "O_FULLREL")
                ]["cache_manifest_sha256"],
            }
            for split in ("model_train", "val_stop", "val_design")
        },
        source=campaign["source"],
    )
    output = args.output or (
        args.campaign_root / "teachers" / "teacher_output_manifest.json"
    )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({**publication, "teacher_output_manifest_sha256": artifact["content_hash"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
