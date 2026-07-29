#!/usr/bin/env python3
"""Compute the frozen 10,000-resample RETB paired confirmation statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    require_sha256,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.paired_statistics import (  # noqa: E402
    build_paired_confirmation_statistics,
    validate_paired_confirmation_statistics,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--paired-input", required=True, type=Path)
    parser.add_argument("--paired-input-sha256", required=True)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    expected = require_sha256(
        args.paired_input_sha256, name="paired_input_sha256"
    )
    if (
        not args.paired_input.is_file()
        or args.paired_input.is_symlink()
        or _sha256(args.paired_input) != expected
    ):
        raise ValueError("paired-statistics input bytes differ")
    with np.load(args.paired_input, allow_pickle=False) as payload:
        if set(payload.files) != {
            "identities",
            "labels",
            "candidate_logits",
            "baseline_logits",
        }:
            raise ValueError("paired-statistics NPZ fields differ")
        arrays = {
            name: np.asarray(payload[name]) for name in payload.files
        }
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if set(configuration) != {
        "candidate_graph_id",
        "baseline_graph_id",
        "pipeline_seed",
        "candidate_prediction_sha256",
        "baseline_prediction_sha256",
    }:
        raise ValueError("paired-statistics configuration fields differ")
    artifact = bind_source(
        build_paired_confirmation_statistics(
            identities=arrays["identities"].tolist(),
            labels=arrays["labels"],
            candidate_logits=arrays["candidate_logits"],
            baseline_logits=arrays["baseline_logits"],
            candidate_graph_id=configuration["candidate_graph_id"],
            baseline_graph_id=configuration["baseline_graph_id"],
            pipeline_seed=configuration["pipeline_seed"],
            candidate_prediction_sha256=configuration[
                "candidate_prediction_sha256"
            ],
            baseline_prediction_sha256=configuration[
                "baseline_prediction_sha256"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_paired_confirmation_statistics(artifact)
    result = {
        "dry_run": args.dry_run,
        "paired_statistics_sha256": artifact["content_hash"],
        "bootstrap_replicates": artifact["bootstrap"]["replicates"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
