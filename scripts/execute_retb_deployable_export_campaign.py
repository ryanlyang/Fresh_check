#!/usr/bin/env python3
"""Execute and authenticate every Step-12 HLT-only deployable export."""

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
    load_hashed_json,
    validate_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (  # noqa: E402
    build_internal_phase_plan,
    configured_phase_concurrency,
    execute_phased_controller,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STAGE_J_CONSUMER_REGISTRY_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


PHASES = ("EXPORT_WAVE", "EXPORT_FINALIZE", "STAGE_K_FACTORY_INPUTS")


class ExportPlanner:
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
        self.registry = load_hashed_json(
            self.root / "registry" / "retb_final_consumer_registry.json",
            expected_contract=STAGE_J_CONSUMER_REGISTRY_CONTRACT,
        )

    def phase_plan(
        self,
        phase_id: str,
        sequence_index: int,
        completions: Mapping[str, str],
    ) -> dict[str, Any]:
        if phase_id == "EXPORT_WAVE":
            rows = []
            for index, row in enumerate(self.registry["rows"]):
                run_root = (
                    self.root / "runs" / "final_consumers" / row["run_id"]
                )
                run = load_hashed_json(run_root / "assets" / "run.json")
                output = self.root / "exports" / row["run_id"]
                rows.append(
                    {
                        "task_id": f"{phase_id}:{index}",
                        "argv": [
                            "python",
                            "scripts/execute_retb_deployable_export_row.py",
                            "--campaign-root",
                            str(self.root),
                            "--run-root",
                            str(run_root),
                            "--output-dir",
                            str(output),
                        ],
                        "environment": {},
                        "expected_outputs": [
                            str(output / "deployable_retb_graph.json"),
                            str(output / "deployable_retb_graph.pt"),
                            str(output / "research_graph_parity.json"),
                        ],
                        "input_artifact_hashes": {
                            "campaign_spec": self.campaign_sha,
                            "production_graph": self.graph_sha,
                            "final_consumer_registry": self.registry[
                                "content_hash"
                            ],
                            "final_consumer_run": run["content_hash"],
                        },
                    }
                )
            resource = "gpu"
        elif phase_id == "EXPORT_FINALIZE":
            output = (
                self.root / "selection" / "deployable_export_index.json"
            )
            rows = [
                {
                    "task_id": f"{phase_id}:0",
                    "argv": [
                        "python",
                        "scripts/finalize_retb_deployable_exports.py",
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
                        "final_consumer_registry": self.registry[
                            "content_hash"
                        ],
                    },
                }
            ]
            resource = "cpu"
        else:
            robustness = (
                self.root
                / "job_ledgers"
                / "factory_inputs"
                / "robustness_controls.json"
            )
            semantics = (
                self.root
                / "job_ledgers"
                / "factory_inputs"
                / "semantic_controls.json"
            )
            rows = [
                {
                    "task_id": f"{phase_id}:0",
                    "argv": [
                        "python",
                        "scripts/emit_retb_stage_k_factory_inputs.py",
                        "--campaign-root",
                        str(self.root),
                    ],
                    "environment": {
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                    },
                    "expected_outputs": [str(robustness), str(semantics)],
                    "input_artifact_hashes": {
                        "campaign_spec": self.campaign_sha,
                        "production_graph": self.graph_sha,
                        "deployable_export_index": load_hashed_json(
                            self.root
                            / "selection"
                            / "deployable_export_index.json"
                        )["content_hash"],
                    },
                }
            ]
            resource = "cpu"
        return build_internal_phase_plan(
            campaign_root=self.root,
            campaign_spec_sha256=self.campaign_sha,
            production_graph_sha256=self.graph_sha,
            controller_id="deployable_export_campaign",
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
    planner = ExportPlanner(
        root=args.campaign_root, campaign=campaign, graph=graph
    )
    artifact = execute_phased_controller(
        campaign_root=args.campaign_root,
        controller_id="deployable_export_campaign",
        phase_ids=PHASES,
        phase_builder=planner.phase_plan,
        repo_root=REPO_ROOT,
        slurm_task_script=(
            REPO_ROOT / "sbatch" / "run_retb_internal_phase_task.sh"
        ),
        output_path=args.output,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
