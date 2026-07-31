#!/usr/bin/env python3
"""Prepare final inputs or write the second HOSD final-test lock."""

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
    authorize_access,
    build_final_execution_lock,
    build_final_input_preparation,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    FINALIST_LOCK_CONTRACT,
    FINAL_INPUT_PREPARATION_CONTRACT,
    POSTLOCK_ORACLE_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def _pairs(values):
    output = {}
    for value in values:
        key, separator, digest = value.partition("=")
        if not separator or key in output:
            raise ValueError("arguments must be unique NAME=SHA256")
        output[key] = digest
    return output


def _artifact_hashes(values, *, source):
    output = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or key in output:
            raise ValueError(
                "final inputs must be unique NAME=ARTIFACT_PATH"
            )
        path = Path(raw_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"final input artifact is absent: {path}"
            )
        if path.suffix == ".json":
            artifact = load_hashed_json(path)
            if artifact.get("source") not in (None, source):
                raise ValueError("final input artifact source differs")
        output[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("prepare", "lock"), required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    finalist = load_hashed_json(args.campaign_root / "selection" / "locked_hosd_finalists.json", expected_contract=FINALIST_LOCK_CONTRACT)
    if args.mode == "prepare":
        authorize_access(worker_role="final_input_preparer", requested_resource="final_test_raw_inputs")
        artifact = build_final_input_preparation(
            finalist_lock=finalist,
            input_hashes=_artifact_hashes(
                args.input, source=campaign["source"]
            ),
            source=campaign["source"],
        )
        output = args.output or args.campaign_root / "final_test" / "prepared_inputs.json"
    else:
        oracle = load_hashed_json(args.campaign_root / "postlock_oracle_diagnostics" / "completion.json", expected_contract=POSTLOCK_ORACLE_CONTRACT)
        prepared = load_hashed_json(args.campaign_root / "final_test" / "prepared_inputs.json", expected_contract=FINAL_INPUT_PREPARATION_CONTRACT)
        control_completion = load_hashed_json(
            args.campaign_root
            / "selection"
            / "finalist_control_completion.json"
        )
        artifact = build_final_execution_lock(
            finalist_lock=finalist,
            postlock_oracle=oracle,
            prepared_inputs=prepared,
            finalist_control_hashes=_pairs(args.control),
            finalist_control_rows=control_completion.get(
                "final_control_rows", ()
            ),
            source=campaign["source"],
        )
        output = args.output or args.campaign_root / "selection" / "final_test_execution_lock.json"
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
