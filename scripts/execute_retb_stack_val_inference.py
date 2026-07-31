#!/usr/bin/env python3
"""Execute one HLT-only stack-val inference plan and publish its shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    execute_plan_steps,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_COMPLETION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    STACK_INFERENCE_EXECUTION_PLAN_CONTRACT,
    validate_stack_inference_execution_plan,
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
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
        args.scale_completion, expected_contract=SCALE_COMPLETION_CONTRACT
    )
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=STACK_INFERENCE_EXECUTION_PLAN_CONTRACT,
    )
    validate_stack_inference_execution_plan(
        plan,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    key = (plan["graph_id"], int(plan["pipeline_seed"]))
    run_map = {
        (row["graph_id"], int(row["pipeline_seed"])): row
        for row in completion["runs"]
    }
    parents = plan["parent_hashes"]
    if (
        shortlist.get("source") != campaign["source"]
        or completion.get("source") != campaign["source"]
        or set(parents) != set(PREDICTION_PARENT_KEYS)
        or plan["locked_scale_shortlist_sha256"]
        != shortlist["content_hash"]
        or plan["scale_completion_sha256"] != completion["content_hash"]
        or key not in run_map
        or parents["campaign_spec"] != campaign["content_hash"]
        or parents["locked_scale_shortlist"] != shortlist["content_hash"]
        or parents["scale_completion"] != completion["content_hash"]
        or parents["scale_graph_run"]
        != run_map[key]["scale_graph_run_sha256"]
        or parents["deployable_export"]
        != run_map[key]["deployable_export_sha256"]
    ):
        raise ValueError("stack-val execution lineage differs")
    receipts = execute_plan_steps(
        plan["steps"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
        forbidden_terms=(),
        forbidden_entrypoints=frozenset(
            {
                "infer_retb_scale_stack_val.py",
                "execute_retb_stack_val_inference.py",
            }
        ),
    )
    inference = Path(plan["inference_output_npz"])
    if not inference.is_file() or inference.is_symlink():
        raise FileNotFoundError("stack-val inference output is absent")
    with np.load(inference, allow_pickle=False) as payload:
        if set(payload.files) not in (
            {"identities", "logits"},
            {"identities", "logits", "probabilities"},
        ):
            raise ValueError("stack-val inference NPZ fields differ")
        identities = np.asarray(payload["identities"]).tolist()
        logits = np.asarray(payload["logits"])
        probabilities = (
            None
            if "probabilities" not in payload.files
            else np.asarray(payload["probabilities"])
        )
    actual_parents = dict(parents)
    inference_steps = [
        step
        for step in plan["steps"]
        if "scripts/run_retb_deployable_inference.py"
        in [str(value).replace("\\", "/") for value in step["argv"]]
    ]
    if len(inference_steps) != 1:
        raise ValueError("stack-val deployable inference step differs")
    argv_values = [str(value) for value in inference_steps[0]["argv"]]
    manifest_values = [
        Path(argv_values[index + 1])
        for index, value in enumerate(argv_values[:-1])
        if value == "--input-manifest"
    ]
    if len(manifest_values) != 1:
        raise ValueError("stack-val HLT input manifest argument differs")
    input_manifest = load_hashed_json(manifest_values[0])
    if (
        input_manifest.get("source") != campaign["source"]
        or input_manifest.get("split") != "stack_val"
        or input_manifest.get("graph_id") != key[0]
        or int(input_manifest.get("pipeline_seed", -1)) != key[1]
        or input_manifest.get("contains_labels") is not False
        or input_manifest.get("contains_offline_or_oracle_values") is not False
    ):
        raise ValueError("stack-val generated HLT input lineage differs")
    actual_parents["stack_val_HLT_input_manifest"] = input_manifest[
        "content_hash"
    ]
    publication = publish_stack_selection_prediction(
        output_dir=args.output_dir,
        identities=identities,
        graph_id=key[0],
        pipeline_seed=key[1],
        logits=logits,
        probabilities=probabilities,
        parent_hashes=actual_parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    print(
        json.dumps(
            {
                "prediction_manifest_sha256": publication["manifest"][
                    "content_hash"
                ],
                "execution_receipts": receipts,
                "contains_labels": False,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
