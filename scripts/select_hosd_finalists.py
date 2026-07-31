#!/usr/bin/env python3
"""Join stack labels separately, select dual finalists, and write their lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_finalist_lock,
    load_and_validate_campaign,
    select_stack_finalists,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STACK_PREDICTION_MANIFEST_CONTRACT,
    STACK_SELECTOR_TRACE_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def _pairs(values):
    output = {}
    for value in values:
        key, separator, digest = value.partition("=")
        if not separator or key in output:
            raise ValueError("arguments must be unique NAME=VALUE")
        output[key] = digest
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("select", "lock"), required=True)
    parser.add_argument("--prediction", action="append", default=[], type=Path)
    parser.add_argument("--labels-npz", type=Path)
    parser.add_argument("--label-manifest-sha256")
    parser.add_argument("--capacity-json", type=Path)
    parser.add_argument("--lineage", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    authorize_access(
        worker_role="stack_selector",
        requested_resource="final_select_label_manifest",
    )
    if args.mode == "select":
        if args.labels_npz is None or args.capacity_json is None:
            raise ValueError("stack selection requires labels and capacity")
        with np.load(args.labels_npz, allow_pickle=False) as payload:
            if set(payload.files) != {"identities", "labels"}:
                raise ValueError("selector label NPZ semantics differ")
            identities = [str(value) for value in payload["identities"].tolist()]
            labels = payload["labels"]
        capacity_payload = json.loads(
            args.capacity_json.read_text(encoding="utf-8")
        )
        capacity_by_graph = capacity_payload.get(
            "capacity_by_graph", capacity_payload
        )
        artifact = select_stack_finalists(
            predictions=[
                load_hashed_json(
                    path, expected_contract=STACK_PREDICTION_MANIFEST_CONTRACT
                )
                for path in args.prediction
            ],
            label_identities=identities,
            labels=labels,
            label_manifest_sha256=args.label_manifest_sha256,
            capacity_by_graph=capacity_by_graph,
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "selection" / "stack_selector_trace.json"
        )
    else:
        trace = load_hashed_json(
            args.campaign_root / "selection" / "stack_selector_trace.json",
            expected_contract=STACK_SELECTOR_TRACE_CONTRACT,
        )
        artifact = build_finalist_lock(
            selector_trace=trace,
            campaign_spec_sha256=campaign["content_hash"],
            required_lineage_hashes=_pairs(args.lineage),
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "selection" / "locked_hosd_finalists.json"
        )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
