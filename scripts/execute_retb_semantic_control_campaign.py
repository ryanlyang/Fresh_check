#!/usr/bin/env python3
"""Run and authenticate the complete Stage-K semantic-control campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json, validate_content_hash  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (  # noqa: E402
    build_internal_phase_plan,
    configured_phase_concurrency,
    execute_phased_controller,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import STAGE_E_SHAPES  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source  # noqa: E402


PHASES = (
    "SEMANTIC_BYPASS_WAVE",
    "RELATION_PREDICTOR_WAVE",
    "BIAS_SCALE_WAVE",
    "SEMANTIC_FINALIZE",
)
CONTRACT = "retb_stage_k_semantic_controls_bundle_v5"


def semantic_controls_bundle_path(root: str | Path) -> Path:
    """Return the single canonical Stage-K bundle path consumed by Stage L."""

    return Path(root).resolve() / "controls" / "semantics" / "semantic_controls_bundle.json"


def _run_id(role: str, seed: int) -> str:
    return (
        f"RETB_{role}_HF_UNRESTRICTED_F_TOKEN_PLUS_EXPERT_LOGITS_"
        f"ND0_NONE_TOKEN_REFINED_SELECTED_S{seed}"
    )


def _adapter_run_id(role: str, seed: int) -> str:
    return (
        f"RETB_{role}_HF_ADAPTER_R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS_"
        f"ND0_NONE_TOKEN_PREDICTED_S{seed}"
    )


class Planner:
    def __init__(self, *, root: Path, campaign: Mapping[str, Any], graph: Mapping[str, Any]) -> None:
        self.root = root.resolve()
        self.campaign = campaign
        self.campaign_sha = validate_content_hash(campaign)
        self.graph_sha = validate_content_hash(graph)

    def phase_plan(self, phase_id: str, sequence_index: int, completions: Mapping[str, str]) -> dict[str, Any]:
        rows = []
        if phase_id == "SEMANTIC_BYPASS_WAVE":
            for role in STAGE_E_SHAPES:
                for seed in (101, 202, 303):
                    for consumer_kind, run_id in (
                        ("HF_UNRESTRICTED", _run_id(role, seed)),
                        ("HF_ADAPTER", _adapter_run_id(role, seed)),
                    ):
                        run_root = self.root / "runs" / "final_consumers" / run_id
                        run = load_hashed_json(run_root / "assets" / "run.json")
                        template = run_root / "assets" / "template" / "final_consumer_template.json"
                        registration = run_root / "registration.json"
                        output = (
                            self.root / "controls" / "semantics" / "bypass"
                            / role / f"seed_{seed}" / consumer_kind
                        )
                        rows.append({
                            "task_id": f"{phase_id}:{len(rows)}",
                            "argv": [
                                "python",
                                "scripts/evaluate_retb_final_consumer_bypass_controls.py",
                                "--campaign-root",
                                str(self.root),
                                "--run",
                                str(run_root / "assets" / "run.json"),
                                "--template",
                                str(template),
                                "--registration",
                                str(registration),
                                "--checkpoint",
                                str(run_root / "best_model_val.pt"),
                                "--val-design-cache",
                                str(
                                    self.root
                                    / "inputs"
                                    / "final_consumers"
                                    / role
                                    / f"seed_{seed}"
                                    / "val_design"
                                    / "final_consumer_dataset.json"
                                ),
                                "--output-dir",
                                str(output),
                            ],
                            "environment": {
                                "RETB_SEMANTIC_CONTROL_KIND": f"BYPASS_{consumer_kind}",
                                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                            },
                            "expected_outputs": [
                                str(output / "bypass_control_logits.npz"),
                                str(output / "bypass_controls.json"),
                            ],
                            "input_artifact_hashes": {
                                "campaign_spec": self.campaign_sha,
                                "production_graph": self.graph_sha,
                                "final_consumer_run": run["content_hash"],
                                "final_consumer_template": load_hashed_json(template)["content_hash"],
                                "final_consumer_registration": load_hashed_json(registration)["content_hash"],
                            },
                        })
            resource = "gpu"
        elif phase_id == "RELATION_PREDICTOR_WAVE":
            for role in STAGE_E_SHAPES:
                for seed in (101, 202, 303):
                    run_id = _run_id(role, seed)
                    export = (
                        self.root / "exports" / run_id
                        / "deployable_retb_graph.json"
                    )
                    cache = (
                        self.root / "inputs" / "final_consumers" / role
                        / f"seed_{seed}" / "val_design"
                        / "final_consumer_dataset.json"
                    )
                    output = (
                        self.root / "controls" / "semantics"
                        / "relation_predictor" / role / f"seed_{seed}.json"
                    )
                    rows.append({
                        "task_id": f"{phase_id}:{len(rows)}",
                        "argv": [
                            "python",
                            "scripts/evaluate_retb_relation_predictor_semantics.py",
                            "--campaign-root", str(self.root),
                            "--deployable-export", str(export),
                            "--val-design-cache", str(cache),
                            "--graph-id", run_id,
                            "--pipeline-seed", str(seed),
                            "--output", str(output),
                        ],
                        "environment": {
                            "RETB_SEMANTIC_CONTROL_KIND": "RELATION_PREDICTOR",
                            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                        },
                        "expected_outputs": [str(output)],
                        "input_artifact_hashes": {
                            "campaign_spec": self.campaign_sha,
                            "production_graph": self.graph_sha,
                            "deployable_export": load_hashed_json(export)["content_hash"],
                            "val_design_cache": load_hashed_json(cache)["content_hash"],
                        },
                    })
            resource = "gpu"
        elif phase_id == "BIAS_SCALE_WAVE":
            output = self.root / "controls" / "semantics" / "bias_scale.json"
            registry = load_hashed_json(
                self.root / "registry" / "retb_stage_b_runs.json"
            )
            rows = [{
                "task_id": f"{phase_id}:0",
                "argv": [
                    "python", "scripts/evaluate_retb_bias_scale_semantics.py",
                    "--campaign-root", str(self.root),
                    "--output", str(output),
                ],
                "environment": {
                    "RETB_SEMANTIC_CONTROL_KIND": "BIAS_SCALE",
                    "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                },
                "expected_outputs": [str(output)],
                "input_artifact_hashes": {
                    "campaign_spec": self.campaign_sha,
                    "production_graph": self.graph_sha,
                    "stage_b_registry": registry["content_hash"],
                },
            }]
            resource = "gpu"
        else:
            output = semantic_controls_bundle_path(self.root)
            rows = [
                {
                    "task_id": f"{phase_id}:0",
                    "argv": [
                        "python",
                        "scripts/finalize_retb_semantic_control_campaign.py",
                        "--campaign-root",
                        str(self.root),
                        "--output",
                        str(output),
                    ],
                    "environment": {
                        "RETB_SEMANTIC_CONTROL_KIND": "RECONSTRUCTION",
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                    },
                    "expected_outputs": [str(output)],
                    "input_artifact_hashes": {
                        "campaign_spec": self.campaign_sha,
                        "production_graph": self.graph_sha,
                    },
                }
            ]
            resource = "cpu"
        return build_internal_phase_plan(
            campaign_root=self.root,
            campaign_spec_sha256=self.campaign_sha,
            production_graph_sha256=self.graph_sha,
            controller_id="semantic_control_campaign",
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
    campaign = load_and_validate_campaign_source(args.campaign_root, repo_root=REPO_ROOT)
    graph = load_hashed_json(args.campaign_root / "job_ledgers" / "production_graph.json")
    planner = Planner(root=args.campaign_root, campaign=campaign, graph=graph)
    artifact = execute_phased_controller(
        campaign_root=args.campaign_root,
        controller_id="semantic_control_campaign",
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
