#!/usr/bin/env python3
"""Lock both produced seed-101 HOSD offline teachers without producing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_teacher_lock,
    load_and_validate_campaign,
    load_hashed_json,
    validate_teacher_lock,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TEACHER_TRAINING_MANIFEST_CONTRACT,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--o-base-completion", required=True, type=Path)
    parser.add_argument("--o-fullrel-completion", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    training = load_hashed_json(
        args.training_manifest,
        expected_contract=TEACHER_TRAINING_MANIFEST_CONTRACT,
    )
    completions = {
        "O_BASE": load_hashed_json(
            args.o_base_completion,
            expected_contract="hosd_teacher_training_completion_v1",
        ),
        "O_FULLREL": load_hashed_json(
            args.o_fullrel_completion,
            expected_contract="hosd_teacher_training_completion_v1",
        ),
    }
    lock = build_teacher_lock(
        training, completions=completions, source=campaign["source"]
    )
    validate_teacher_lock(lock, source=campaign["source"])
    output = args.output or args.campaign_root / "teachers" / "teacher_lock.json"
    publication = write_immutable_json(output, lock)
    print(
        json.dumps(
            {
                **publication,
                "teacher_lock_sha256": lock["content_hash"],
                "teacher_order": lock["teacher_order"],
                "lock_is_not_a_checkpoint_producer": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
