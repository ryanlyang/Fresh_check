#!/usr/bin/env python3
"""Run the complete restart-safe Stage-K degradation campaign."""

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
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (  # noqa: E402
    build_internal_phase_plan,
    configured_phase_concurrency,
    execute_phased_controller,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    STAGE_E_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


PROFILES = (
    "D_OFFLINE_IDENTITY",
    "D_KIN_ONLY",
    "D_TRACK_ONLY",
    "D_MISSING_ONLY",
    "D_MILD",
    "D_NOMINAL",
    "D_SEVERE",
    "D_LEGACY_V1",
    "D_LEGACY_V2",
)
REPLICAS = (0, 1, 2, 3)
PHASES = ("ROBUSTNESS_VIEW_WAVE", "ROBUSTNESS_EVALUATION_WAVE", "ROBUSTNESS_FINALIZE")
CONTRACT = "retb_stage_k_robustness_bundle_v1"


def _selected_run(role: str, seed: int) -> str:
    return (
        f"RETB_{role}_HF_UNRESTRICTED_F_TOKEN_PLUS_EXPERT_LOGITS_"
        f"ND0_NONE_TOKEN_REFINED_SELECTED_S{seed}"
    )


class Planner:
    def __init__(
        self,
        *,
        root: Path,
        campaign: Mapping[str, Any],
        graph: Mapping[str, Any],
    ) -> None:
        self.root = root.resolve()
        self.campaign = campaign
        self.campaign_sha = validate_content_hash(campaign)
        self.graph_sha = validate_content_hash(graph)
        self.export_index = load_hashed_json(
            self.root / "selection" / "deployable_export_index.json"
        )

    def phase_plan(
        self,
        phase_id: str,
        sequence_index: int,
        completions: Mapping[str, str],
    ) -> dict[str, Any]:
        rows = []
        if phase_id == "ROBUSTNESS_VIEW_WAVE":
            for profile in PROFILES:
                for replica in REPLICAS:
                    output = (
                        self.root
                        / "controls"
                        / "robustness"
                        / "views"
                        / profile
                        / f"replica_{replica}"
                    )
                    rows.append(
                        {
                            "task_id": f"{phase_id}:{len(rows)}",
                            "argv": [
                                "python",
                                "scripts/prepare_retb_robustness_view.py",
                                "--campaign-root",
                                str(self.root),
                                "--profile",
                                profile,
                                "--replica",
                                str(replica),
                                "--output-dir",
                                str(output),
                            ],
                            "environment": {
                                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                            },
                            "expected_outputs": [
                                str(output / "robustness_view.json"),
                                str(output / "robustness_view.pt"),
                            ],
                            "input_artifact_hashes": {
                                "campaign_spec": self.campaign_sha,
                                "production_graph": self.graph_sha,
                                "deployable_export_index": self.export_index[
                                    "content_hash"
                                ],
                            },
                        }
                    )
            resource = "cpu"
        elif phase_id == "ROBUSTNESS_EVALUATION_WAVE":
            for role in STAGE_E_SHAPES:
                for seed in (101, 202, 303):
                    run_id = _selected_run(role, seed)
                    export = self.export_index["exports"].get(run_id)
                    if export is None:
                        raise ValueError(
                            f"robustness export is absent: {run_id}"
                        )
                    for profile in PROFILES:
                        for replica in REPLICAS:
                            view = (
                                self.root
                                / "controls"
                                / "robustness"
                                / "views"
                                / profile
                                / f"replica_{replica}"
                                / "robustness_view.json"
                            )
                            view_payload = load_hashed_json(view)
                            output = (
                                self.root
                                / "controls"
                                / "robustness"
                                / "evaluations"
                                / role
                                / f"seed_{seed}"
                                / profile
                                / f"replica_{replica}"
                            )
                            rows.append(
                                {
                                    "task_id": f"{phase_id}:{len(rows)}",
                                    "argv": [
                                        "python",
                                        "scripts/evaluate_retb_deployable_robustness.py",
                                        "--campaign-root",
                                        str(self.root),
                                        "--graph-id",
                                        run_id,
                                        "--export-manifest",
                                        export["export_path"],
                                        "--view-manifest",
                                        str(view),
                                        "--output-dir",
                                        str(output),
                                    ],
                                    "environment": {
                                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                                    },
                                    "expected_outputs": [
                                        str(output / "predictions.npz"),
                                        str(output / "metrics.json"),
                                        str(output / "evaluation.json"),
                                    ],
                                    "input_artifact_hashes": {
                                        "campaign_spec": self.campaign_sha,
                                        "production_graph": self.graph_sha,
                                        "deployable_export": export[
                                            "export_sha256"
                                        ],
                                        "robustness_view": view_payload[
                                            "content_hash"
                                        ],
                                    },
                                }
                            )
            resource = "gpu"
        else:
            output = (
                self.root
                / "controls"
                / "robustness"
                / "robustness_bundle.json"
            )
            rows = [
                {
                    "task_id": f"{phase_id}:0",
                    "argv": [
                        "python",
                        "scripts/finalize_retb_robustness_campaign.py",
                        "--campaign-root",
                        str(self.root),
                        "--output",
                        str(output),
                    ],
                    "environment": {
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                    },
                    "expected_outputs": [str(output)],
                    "input_artifact_hashes": {
                        "campaign_spec": self.campaign_sha,
                        "production_graph": self.graph_sha,
                        "deployable_export_index": self.export_index[
                            "content_hash"
                        ],
                    },
                }
            ]
            resource = "cpu"
        return build_internal_phase_plan(
            campaign_root=self.root,
            campaign_spec_sha256=self.campaign_sha,
            production_graph_sha256=self.graph_sha,
            controller_id="robustness_campaign",
            phase_id=phase_id,
            sequence_index=sequence_index,
            resource=resource,
            maximum_concurrent_tasks=configured_phase_concurrency(
                resource=resource,
                family="final",
                row_count=len(rows),
            ),
            rows=rows,
            prerequisite_completion_hashes=completions,
            source=self.campaign["source"],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json"
    )
    planner = Planner(root=args.campaign_root, campaign=campaign, graph=graph)
    artifact = execute_phased_controller(
        campaign_root=args.campaign_root,
        controller_id="robustness_campaign",
        phase_ids=PHASES,
        phase_builder=planner.phase_plan,
        repo_root=REPO_ROOT,
        slurm_task_script=REPO_ROOT / "sbatch" / "run_retb_internal_phase_task.sh",
        output_path=args.output,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
