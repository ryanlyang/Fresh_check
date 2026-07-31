#!/usr/bin/env python3
"""Attest checkpoint-free prelock final-test input preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    build_prelock_final_test_inputs,
    validate_prelock_final_test_inputs,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    HLT_V3_CACHE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    publish_shared_deployable_inference_payload,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (  # noqa: E402
    STAGE_A_TREE_INDEX_CONTRACT,
    validate_stage_a_tree_index,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_access import (  # noqa: E402
    PRELOCK_INPUT_CONFIGURATION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from scripts.prepare_retb_deployable_input import (  # noqa: E402
    build_hlt_only_inference_inputs,
)


def _live_shared_input_sources(
    campaign_root: Path,
    campaign: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, Mapping[str, Any]],
    Mapping[str, Any],
]:
    hlt_by_split = {
        split: load_hashed_json(
            campaign_root
            / "inputs"
            / "hlt_v3"
            / split
            / "replica_0"
            / "R_FIXED"
            / "D_NOMINAL"
            / "hlt_v3_metadata.json",
            expected_contract=HLT_V3_CACHE_CONTRACT,
        )
        for split in ("stack_val", "final_test")
    }
    tree_index = load_hashed_json(
        campaign_root / "inputs" / "region_tree" / "tree_cache_index.json",
        expected_contract=STAGE_A_TREE_INDEX_CONTRACT,
    )
    validate_stage_a_tree_index(tree_index)
    if (
        tree_index.get("source") != campaign.get("source")
        or tree_index.get("campaign_spec_sha256") != campaign["content_hash"]
    ):
        raise ValueError("prelock REGION index campaign lineage differs")
    tree_by_id = {row["view_id"]: row for row in tree_index["views"]}
    sources: dict[str, dict[str, str]] = {}
    for split, hlt in hlt_by_split.items():
        tree = tree_by_id[f"hlt:{split}:r0:R_FIXED"]
        if (
            hlt.get("source") != campaign.get("source")
            or hlt.get("split_manifest_sha256")
            != campaign["parent_artifact_hashes"]["split_manifest"]
            or tree["view_content_sha256"]
            != hlt["array_content_sha256"]
            or tree["cache_metadata_sha256"] != hlt["content_hash"]
            or tree["identity_manifest_sha256"]
            != hlt["identity_manifest_sha256"]
        ):
            raise ValueError(f"prelock {split} HLT/REGION lineage differs")
        sources[split] = {
            "hlt_cache_sha256": hlt["content_hash"],
            "hlt_array_content_sha256": hlt["array_content_sha256"],
            "region_tree_manifest_sha256": tree["tree_manifest_sha256"],
            "identity_manifest_sha256": hlt["identity_manifest_sha256"],
        }
    return sources, hlt_by_split, tree_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="final_test_input_preparation",
        requested_resource="final_test_inputs",
    )
    configuration = load_hashed_json(
        args.configuration,
        expected_contract=PRELOCK_INPUT_CONFIGURATION_CONTRACT,
    )
    live_sources, hlt_by_split, tree_index = _live_shared_input_sources(
        args.campaign_root, campaign
    )
    profile = load_hashed_json(
        args.campaign_root / "inputs" / "hlt_v3_profile.json"
    )
    final_hlt = hlt_by_split["final_test"]
    values = {
        name: configuration[name]
        for name in (
            "split_manifest_sha256",
            "degradation_profile_sha256",
            "input_hashes",
        )
    }
    if set(values) != {
        "split_manifest_sha256",
        "degradation_profile_sha256",
        "input_hashes",
    } or (
        int(configuration.get("schema_version", -1)) != 2
        or configuration.get("source") != campaign.get("source")
        or configuration.get("shared_input_sources") != live_sources
        or configuration.get("region_tree_index_sha256")
        != tree_index["content_hash"]
        or values["split_manifest_sha256"]
        != campaign["parent_artifact_hashes"]["split_manifest"]
        or values["degradation_profile_sha256"] != profile["content_hash"]
        or profile.get("source") != campaign.get("source")
        or any(
            hlt.get("profile_contract_sha256") != profile["content_hash"]
            for hlt in hlt_by_split.values()
        )
        or values["input_hashes"].get("final_test_identity_manifest")
        != final_hlt["identity_manifest_sha256"]
        or values["input_hashes"].get("final_test_raw_inputs")
        != final_hlt["raw_input_sha256"]
        or values["input_hashes"].get("final_test_HLT_inputs")
        != final_hlt["content_hash"]
        or values["input_hashes"].get("final_test_relation_sidecars")
        != canonical_sha256(
            {
                "HLT_cache": final_hlt["content_hash"],
                "sidecar": "relation",
            }
        )
        or values["input_hashes"].get("final_test_REGION_sidecars")
        != live_sources["final_test"]["region_tree_manifest_sha256"]
    ):
        raise ValueError("prelock final input configuration differs")
    shared_root = args.campaign_root / "inputs" / "stage_n" / "shared"
    shared = {}
    if not args.dry_run:
        for split in ("stack_val", "final_test"):
            identities, hlt_inputs = build_hlt_only_inference_inputs(
                campaign_root=args.campaign_root,
                split=split,
            )
            shared[split] = publish_shared_deployable_inference_payload(
                output_dir=shared_root,
                split=split,
                identities=identities,
                hlt_inputs=hlt_inputs,
                source_snapshot=source_snapshot(REPO_ROOT),
            )
        values["input_hashes"] = dict(values["input_hashes"])
        values["input_hashes"]["final_test_HLT_inputs"] = shared[
            "final_test"
        ]["manifest"]["content_hash"]
    artifact = bind_source(
        build_prelock_final_test_inputs(
            campaign_spec_sha256=campaign["content_hash"],
            **values,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_prelock_final_test_inputs(artifact)
    result = {
        "dry_run": args.dry_run,
        "prelock_final_inputs_sha256": artifact["content_hash"],
        "checkpoint_loading_allowed": False,
        "model_outputs_emitted": False,
        "shared_HLT_payloads": {
            split: publication["manifest"]["content_hash"]
            for split, publication in shared.items()
        },
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
