#!/usr/bin/env python3
"""Register a byte-authenticated RETB final-consumer model template."""

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

import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    require_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    publish_final_consumer_template,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumers import (  # noqa: E402
    FrozenPredictedOfflineFusion,
    HLTResidualAdapter,
    NativeConditionedTokenRefiner,
    UnrestrictedHLTFusion,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    FINAL_CONSUMER_RUN_CONTRACT,
    validate_materialized_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _states_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(
            left_state[name].detach().cpu(),
            right_state[name].detach().cpu(),
        )
        for name in left_state
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepared-template", required=True, type=Path)
    parser.add_argument("--prepared-template-sha256", required=True)
    parser.add_argument("--component-parents", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=FINAL_CONSUMER_RUN_CONTRACT
    )
    validate_materialized_final_consumer_run(run)
    expected_hash = require_sha256(
        args.prepared_template_sha256, name="prepared_template_sha256"
    )
    if (
        run.get("source") != campaign.get("source")
        or not args.prepared_template.is_file()
        or args.prepared_template.is_symlink()
        or _sha256(args.prepared_template) != expected_hash
    ):
        raise ValueError("prepared final-consumer template lineage differs")
    payload = torch.load(
        args.prepared_template, map_location="cpu", weights_only=False
    )
    required = {
        "model",
        "frozen_expert_heads",
        "frozen_offline_fusion",
        "refiner",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or set(payload["frozen_expert_heads"]) != set(EXPERT_ORDER)
    ):
        raise ValueError("prepared final-consumer template semantics differ")
    kind = run["consumer_kind"]
    model = payload["model"]
    frozen_parameters = [
        *payload["frozen_offline_fusion"].parameters(),
        *(
            parameter
            for head in payload["frozen_expert_heads"].values()
            for parameter in head.parameters()
        ),
    ]
    if any(parameter.requires_grad for parameter in frozen_parameters):
        raise ValueError("offline template components are not frozen")
    if payload["refiner"] is not None and any(
        parameter.requires_grad
        for parameter in payload["refiner"].parameters()
    ):
        raise ValueError("selected token refiner is not frozen")
    if (
        (
            kind == "PF_FROZEN"
            and (
                not isinstance(model, FrozenPredictedOfflineFusion)
                or not _states_equal(
                    model.fusion, payload["frozen_offline_fusion"]
                )
                or any(
                    parameter.requires_grad
                    for parameter in model.parameters()
                )
            )
        )
        or (
            kind == "OF_ROBUST"
            and (
                type(model) is not type(payload["frozen_offline_fusion"])
                or not _states_equal(
                    model, payload["frozen_offline_fusion"]
                )
                or {
                    id(parameter) for parameter in model.parameters()
                }
                & {
                    id(parameter)
                    for parameter in payload[
                        "frozen_offline_fusion"
                    ].parameters()
                }
                or not all(
                    parameter.requires_grad
                    for parameter in model.parameters()
                )
            )
        )
        or (
            kind == "TR_REFINE"
            and (
                not isinstance(model, NativeConditionedTokenRefiner)
                or model.variant != run["model_variant"]
            )
        )
        or (
            kind == "HF_ADAPTER"
            and (
                not isinstance(model, HLTResidualAdapter)
                or model.variant != run["model_variant"]
                or model.native_dropout_mode
                != run["native_dropout_mode"]
            )
        )
        or (
            kind == "HF_UNRESTRICTED"
            and (
                not isinstance(model, UnrestrictedHLTFusion)
                or model.evidence_variant != run["model_variant"]
                or model.native_dropout_mode
                != run["native_dropout_mode"]
            )
        )
        or (
            run["token_input"] == "TOKEN_REFINED_SELECTED"
            and not isinstance(
                payload["refiner"], NativeConditionedTokenRefiner
            )
        )
        or (
            run["token_input"] == "TOKEN_PREDICTED"
            and payload["refiner"] is not None
        )
    ):
        raise ValueError("prepared final-consumer model type differs")
    parents = json.loads(args.component_parents.read_text("utf-8"))
    names = {
        "joint_prediction_checkpoint",
        "native_HLT_checkpoint_bundle",
        "uncertainty_calibration",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
    }
    if payload["refiner"] is not None:
        names.add("selected_token_refiner")
    expected_parents = {
        name: run["parent_hashes"][name] for name in names
    }
    if parents != expected_parents:
        raise ValueError("final-consumer template parents differ")
    manifest = publish_final_consumer_template(
        output_dir=args.output_dir,
        model=model,
        frozen_expert_heads=payload["frozen_expert_heads"],
        frozen_offline_fusion=payload["frozen_offline_fusion"],
        refiner=payload["refiner"],
        run_record_sha256=run["content_hash"],
        component_parent_hashes=parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    print(
        json.dumps(
            {
                "template_sha256": manifest["content_hash"],
                "prepared_template_sha256": expected_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
