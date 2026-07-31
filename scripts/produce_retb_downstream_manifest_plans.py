#!/usr/bin/env python3
"""Invoke registered real plan factories for one completed producer node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.early_continuation import (  # noqa: E402
    EARLY_PLAN_FACTORIES,
)
from teacher_logit_reco.relation_expert_token_bridge.manifest_orchestration import (  # noqa: E402
    producer_targets,
)
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (  # noqa: E402
    LATE_PLAN_FACTORIES,
)
from teacher_logit_reco.relation_expert_token_bridge.final_plan_factories import (  # noqa: E402
    FINAL_PLAN_FACTORIES,
)
from teacher_logit_reco.relation_expert_token_bridge.middle_plan_factories import (  # noqa: E402
    MIDDLE_PLAN_FACTORIES,
)
from teacher_logit_reco.relation_expert_token_bridge.middle_dynamic_plan_factories import (  # noqa: E402
    MIDDLE_DYNAMIC_PLAN_FACTORIES,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)

# Literal symbols make the source-level operational audit fail closed if a
# Closure-2 factory is accidentally removed from this entry point.
FACTORY_SYMBOLS = (
    "build_offline_optimization_selector_manifest_plan",
    "build_offline_shape_selector_manifest_plan",
    "build_offline_complementarity_manifest_plan",
    "build_offline_capacity_controls_manifest_plan",
    "build_bridge_target_training_manifest_plan",
    "build_bridge_content_certification_manifest_plan",
    "build_target_coordinate_selector_manifest_plan",
    "build_target_cache_build_manifest_plan",
    "build_target_normalizers_manifest_plan",
    "build_predictor_training_manifest_plan",
    "build_uncertainty_calibration_manifest_plan",
    "build_predictor_bundle_selector_manifest_plan",
    "build_oracle_substitutions_manifest_plan",
    "build_joint_predictor_training_manifest_plan",
    "build_joint_predictor_selector_manifest_plan",
    "build_final_consumer_training_manifest_plan",
    "build_deployable_export_manifest_plan",
    "build_robustness_controls_manifest_plan",
    "build_semantic_controls_manifest_plan",
    "build_stage_l_graph_registration_manifest_plan",
    "build_confirmation_500k_manifest_plan",
    "build_confirmation_summary_manifest_plan",
    "build_bridge_shape_selector_manifest_plan",
    "build_scale_shortlist_selector_manifest_plan",
    "build_scale_refits_manifest_plan",
    "build_scale_graph_training_manifest_plan",
    "build_scale_completion_manifest_plan",
    "build_prelock_final_inputs_manifest_plan",
    "build_stack_val_inference_manifest_plan",
    "build_accuracy_finalist_selector_manifest_plan",
    "build_postlock_oracle_targets_manifest_plan",
    "build_finalist_controls_manifest_plan",
    "build_final_test_execution_lock_manifest_plan",
    "build_sealed_final_test_manifest_plan",
    "build_final_report_manifest_plan",
)
PUBLICATION_SYMBOL = "publish_manifest_materialization_plan"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--producer-node-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    targets = producer_targets(args.producer_node_id)
    factories = {
        **EARLY_PLAN_FACTORIES,
        **MIDDLE_PLAN_FACTORIES,
        **MIDDLE_DYNAMIC_PLAN_FACTORIES,
        **LATE_PLAN_FACTORIES,
        **FINAL_PLAN_FACTORIES,
    }
    selected = [target for target in targets if target in factories]
    unsupported = [target for target in targets if target not in factories]
    if unsupported and selected:
        raise RuntimeError(
            "one producer mixes implemented and unimplemented plan factories: "
            f"{unsupported}"
        )
    if unsupported:
        raise NotImplementedError(
            "registered downstream factories are not implemented by the "
            f"current closure block: {unsupported}"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "producer_node_id": args.producer_node_id,
                    "targets": selected,
                    "factory_symbols": list(FACTORY_SYMBOLS),
                    "publication_symbol": PUBLICATION_SYMBOL,
                    "dry_run": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    results = [
        factories[target](
            campaign_root=args.campaign_root,
            campaign=campaign,
            production_graph=graph,
            producer_node_id=args.producer_node_id,
        )
        for target in selected
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
