#!/usr/bin/env python3
"""Publish one label-free deployable 3M stack-val prediction shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_COMPLETION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    PREDICTION_PARENT_KEYS,
    publish_stack_selection_prediction,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--scale-completion", required=True, type=Path)
    parser.add_argument("--inference-output-npz", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="stage_n_selection_inference",
        requested_resource="stack_val_features",
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    completion = load_hashed_json(
        args.scale_completion,
        expected_contract=SCALE_COMPLETION_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if set(configuration) != {
        "graph_id",
        "pipeline_seed",
        "parent_hashes",
    } or set(configuration["parent_hashes"]) != set(PREDICTION_PARENT_KEYS):
        raise ValueError("stack-val inference configuration differs")
    if any(
        row.get("source") != campaign.get("source")
        for row in (shortlist, completion)
    ):
        raise ValueError("stack-val inference source differs")
    key = (
        configuration["graph_id"],
        int(configuration["pipeline_seed"]),
    )
    run_map = {
        (row["graph_id"], row["pipeline_seed"]): row
        for row in completion["runs"]
    }
    parents = configuration["parent_hashes"]
    if (
        key not in run_map
        or parents["campaign_spec"] != campaign["content_hash"]
        or parents["locked_scale_shortlist"] != shortlist["content_hash"]
        or parents["scale_completion"] != completion["content_hash"]
        or parents["scale_graph_run"]
        != run_map[key]["scale_graph_run_sha256"]
        or parents["deployable_export"]
        != run_map[key]["deployable_export_sha256"]
    ):
        raise ValueError("stack-val inference graph/run parents differ")
    if not args.inference_output_npz.is_file() or args.inference_output_npz.is_symlink():
        raise FileNotFoundError("inference output NPZ is absent or unsafe")
    with np.load(args.inference_output_npz, allow_pickle=False) as payload:
        if set(payload.files) not in (
            {"identities", "logits"},
            {"identities", "logits", "probabilities"},
        ):
            raise ValueError(
                "inference output may contain only identities/logits/probabilities"
            )
        identities = np.asarray(payload["identities"]).tolist()
        logits = np.asarray(payload["logits"])
        probabilities = (
            None
            if "probabilities" not in payload.files
            else np.asarray(payload["probabilities"])
        )
    if args.dry_run:
        result = {
            "dry_run": True,
            "graph_id": configuration["graph_id"],
            "pipeline_seed": configuration["pipeline_seed"],
            "identity_count": len(identities),
            "would_publish_label_free_shard": True,
        }
    else:
        publication = publish_stack_selection_prediction(
            output_dir=args.output_dir,
            identities=identities,
            graph_id=configuration["graph_id"],
            pipeline_seed=configuration["pipeline_seed"],
            logits=logits,
            probabilities=probabilities,
            parent_hashes=configuration["parent_hashes"],
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        result = {
            "dry_run": False,
            "prediction_manifest_sha256": publication["manifest"][
                "content_hash"
            ],
            "contains_labels": False,
            "publication": publication,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
