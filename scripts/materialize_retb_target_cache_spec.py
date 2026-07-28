#!/usr/bin/env python3
"""Resolve a locked Step-7 coordinate into a per-seed target-cache spec."""

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
    load_hashed_json,
    validate_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step8 import (  # noqa: E402
    build_locked_target_cache_specification,
    build_selected_target_lineage,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _load_map(
    values: Mapping[str, Any], *, nullable: bool = False
) -> dict[str, Mapping[str, Any] | None]:
    if set(values) != set(EXPERT_ORDER):
        raise ValueError("target-cache input expert coverage differs")
    output = {}
    for expert in EXPERT_ORDER:
        value = values[expert]
        if value is None and nullable:
            output[expert] = None
        elif value is None:
            raise ValueError(f"target-cache input for {expert} is absent")
        else:
            output[expert] = load_hashed_json(Path(value))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--coordinate-contract-sha256", required=True)
    parser.add_argument("--output-lineage", required=True, type=Path)
    parser.add_argument("--output-specification", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    configuration = json.loads(args.inputs.read_text("utf-8"))
    required = {
        "allocation",
        "coordinate_selection",
        "target_registrations",
        "slot_query_hashes",
        "eligibility_artifacts",
        "content_certifications",
        "noninferiority_artifacts",
        "fusion_registration",
        "normalizer_set",
        "identity_manifest",
    }
    if set(configuration) != required:
        raise ValueError("target-cache materialization input fields differ")
    coordinate = load_hashed_json(Path(configuration["coordinate_selection"]))
    registrations = _load_map(configuration["target_registrations"])
    eligibility = _load_map(configuration["eligibility_artifacts"])
    content = _load_map(
        configuration["content_certifications"], nullable=True
    )
    noninferiority = _load_map(
        configuration["noninferiority_artifacts"], nullable=True
    )
    fusion = load_hashed_json(Path(configuration["fusion_registration"]))
    normalizer = load_hashed_json(Path(configuration["normalizer_set"]))
    identity = load_hashed_json(Path(configuration["identity_manifest"]))
    loaded = [
        coordinate,
        *registrations.values(),
        *eligibility.values(),
        *(row for row in content.values() if row is not None),
        *(row for row in noninferiority.values() if row is not None),
        fusion,
        normalizer,
        identity,
    ]
    if any(row.get("source") != campaign.get("source") for row in loaded):
        raise ValueError("target-cache materialization source lineage differs")
    locked = [
        row
        for row in coordinate["locked_coordinate_systems"]
        if row["coordinate_contract_sha256"]
        == args.coordinate_contract_sha256
    ]
    if len(locked) != 1:
        raise ValueError("target-cache coordinate is absent/duplicated")
    target_tuple = locked[0]["target_tuple"]
    snapshot = source_snapshot(REPO_ROOT)
    lineage = bind_source(
        build_selected_target_lineage(
            pipeline_seed=args.pipeline_seed,
            shape_id=args.shape_id,
            target_tuple=target_tuple,
            target_registrations=registrations,
            slot_query_hashes=configuration["slot_query_hashes"],
            eligibility_artifacts=eligibility,
            content_certifications=content,
            noninferiority_artifacts=noninferiority,
        ),
        source_snapshot=snapshot,
    )
    specification = bind_source(
        build_locked_target_cache_specification(
            split=args.split,
            pipeline_seed=args.pipeline_seed,
            shape_id=args.shape_id,
            allocation=configuration["allocation"],
            coordinate_selection=coordinate,
            coordinate_contract_sha256=args.coordinate_contract_sha256,
            target_lineage=lineage,
            fusion_registration=fusion,
            normalizer_set=normalizer,
            identity_manifest_sha256=validate_content_hash(identity),
            identity_order_sha256=identity["identity_order_sha256"],
            event_count=int(identity["event_count"]),
        ),
        source_snapshot=snapshot,
    )
    result = {
        "lineage": write_immutable_json(args.output_lineage, lineage),
        "specification": write_immutable_json(
            args.output_specification, specification
        ),
        "selected_target_lineage_sha256": lineage["content_hash"],
        "target_cache_specification_sha256": specification["content_hash"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
