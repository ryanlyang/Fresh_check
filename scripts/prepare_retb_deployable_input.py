#!/usr/bin/env python3
"""Materialize one label-free HLT-only deployable inference payload."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from scripts.train_retb_native_hlt_expert import _trees  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    canonical_sha256,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    publish_deployable_inference_input,
    publish_deployable_inference_input_binding,
    validate_shared_deployable_inference_payload,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def build_hlt_only_inference_inputs(
    *, campaign_root: Path, split: str
) -> tuple[list[str], dict[str, object]]:
    """Build the one graph-independent HLT payload for a split."""

    root = campaign_root.resolve()
    authorize_dataset_access(
        worker_role=(
            "stage_n_selection_inference"
            if split == "stack_val"
            else "final_test_input_preparation"
        ),
        requested_resource=(
            "stack_val_features"
            if split == "stack_val"
            else "final_test_inputs"
        ),
    )
    cache = (
        root
        / "inputs"
        / "hlt_v3"
        / split
        / "replica_0"
        / "R_FIXED"
        / "D_NOMINAL"
    )
    arrays, metadata = load_hlt_v3_cache(cache)
    identities = [str(value) for value in arrays["identities"].tolist()]
    order = np.argsort(np.asarray(identities, dtype=np.str_))
    identities = [identities[int(index)] for index in order]
    tokens = np.asarray(arrays["tokens"])[order]
    mask = np.asarray(arrays["mask"])[order]
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=np.zeros(len(identities), dtype=np.int64),
        source_view="hlt",
    )
    trees = _trees(
        root / "inputs" / "region_tree" / "hlt",
        logical_role=split,
        replicas=(0,),
        identities=tuple(identities),
        realization_policy="R_FIXED",
    )[0]
    hlt_inputs = {
        "replica_ids": torch.zeros(len(identities), dtype=torch.int64),
        "degraded_view_hashes": [
            canonical_sha256(
                {
                    "identity": identity,
                    "replica": 0,
                    "HLT_cache": metadata["content_hash"],
                }
            )
            for identity in identities
        ],
        "features": inputs.pf_features,
        "vectors": inputs.pf_vectors,
        "mask": inputs.pf_mask,
        "raw_tokens": tokens,
        "region_trees_by_expert": {
            expert: trees for expert in EXPERT_ORDER
        },
    }
    return identities, hlt_inputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--split", required=True, choices=("stack_val", "final_test")
    )
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shared-payload-manifest", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    if args.shared_payload_manifest is not None:
        shared = load_hashed_json(args.shared_payload_manifest)
        validate_shared_deployable_inference_payload(
            shared, manifest_path=args.shared_payload_manifest
        )
        if (
            shared.get("source") != campaign["source"]
            or shared.get("split") != args.split
            or args.output_dir.resolve()
            != args.shared_payload_manifest.resolve().parent
        ):
            raise ValueError("shared deployable input lineage differs")
        publish_deployable_inference_input_binding(
            output_dir=args.output_dir,
            shared_payload_manifest=shared,
            shared_payload_manifest_path=args.shared_payload_manifest,
            graph_id=args.graph_id,
            pipeline_seed=args.pipeline_seed,
        )
        return 0
    identities, hlt_inputs = build_hlt_only_inference_inputs(
        campaign_root=root, split=args.split
    )
    publish_deployable_inference_input(
        output_dir=args.output_dir,
        split=args.split,
        graph_id=args.graph_id,
        pipeline_seed=args.pipeline_seed,
        identities=identities,
        hlt_inputs=hlt_inputs,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
