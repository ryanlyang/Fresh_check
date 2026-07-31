#!/usr/bin/env python3
"""Execute every Step-12 final consumer in ordered restart-safe phases."""

from __future__ import annotations

import argparse
import hashlib
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
from teacher_logit_reco.relation_expert_token_bridge.predictor_campaign import (  # noqa: E402
    FINAL_CONSUMER_PHASE_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STAGE_J_CONSUMER_REGISTRY_CONTRACT,
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


def _dataset_outputs(root: Path) -> list[str]:
    return [
        str(root / split / filename)
        for split in ("model_train", "val_stop", "val_design")
        for filename in (
            "final_consumer_dataset.json",
            "final_consumer_dataset.pt",
        )
    ]


def _row_outputs(root: Path, *, trainable: bool) -> list[str]:
    common = [
        root / "assets" / "run.json",
        root / "assets" / "template" / "final_consumer_template.json",
        root / "assets" / "template" / "final_consumer_template.pt",
    ]
    if trainable:
        common.extend(
            (
                root / "registration.json",
                root / "training_curves.json",
                root / "best_model_val.pt",
                root / "val_design" / "metrics.json",
                root
                / "val_design"
                / "final_consumer_predictions_manifest.json",
                root / "val_design" / "final_consumer_predictions.npz",
            )
        )
    else:
        common.extend(
            (
                root / "reference_registration.json",
                root / "reference_metrics.json",
                root / "final_consumer_predictions_manifest.json",
                root / "final_consumer_predictions.npz",
            )
        )
    return [str(path) for path in common]


class FinalConsumerPlanner:
    def __init__(
        self,
        *,
        root: Path,
        campaign: Mapping[str, Any],
        graph: Mapping[str, Any],
    ) -> None:
        self.root = root.resolve()
        self.campaign = campaign
        self.graph = graph
        self.campaign_sha = validate_content_hash(campaign)
        self.graph_sha = validate_content_hash(graph)
        self.registry = load_hashed_json(
            self.root / "registry" / "retb_final_consumer_registry.json",
            expected_contract=STAGE_J_CONSUMER_REGISTRY_CONTRACT,
        )
        self.step12 = load_hashed_json(
            self.root
            / "registry"
            / "retb_step12_final_consumers_bundle.json"
        )
        self.dataset_root = self.root / "inputs" / "final_consumers"

    def _consumer_row(
        self, phase: str, index: int, row: Mapping[str, Any]
    ) -> dict[str, Any]:
        output = self.root / "runs" / "final_consumers" / row["run_id"]
        inputs = {
            "campaign_spec": self.campaign_sha,
            "production_graph": self.graph_sha,
            "step12_bundle": self.step12["content_hash"],
            "final_consumer_registry": self.registry["content_hash"],
        }
        if row["token_input"] == "TOKEN_REFINED_SELECTED":
            lock = (
                self.root
                / "selection"
                / "final_consumers"
                / row["carried_shape_role"]
                / "token_refiner_lock.json"
            )
            inputs["selected_token_refiner"] = load_hashed_json(lock)[
                "content_hash"
            ]
        for split in ("model_train", "val_stop", "val_design"):
            path = (
                self.dataset_root
                / row["carried_shape_role"]
                / f"seed_{row['pipeline_seed']}"
                / split
                / "final_consumer_dataset.json"
            )
            inputs[f"{split}_dataset"] = load_hashed_json(path)[
                "content_hash"
            ]
        return {
            "task_id": f"{phase}:{index}",
            "argv": [
                "python",
                "scripts/execute_retb_final_consumer_row.py",
                "--campaign-root",
                str(self.root),
                "--run-id",
                row["run_id"],
                "--dataset-root",
                str(self.dataset_root),
                "--output-dir",
                str(output),
            ],
            "environment": {
                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
            },
            "expected_outputs": _row_outputs(
                output, trainable=bool(row["trainable"])
            ),
            "input_artifact_hashes": inputs,
        }

    def phase_plan(
        self,
        phase_id: str,
        sequence_index: int,
        completions: Mapping[str, str],
    ) -> dict[str, Any]:
        if phase_id == "FINAL_DATASET_PREP":
            rows = []
            joint_lock = load_hashed_json(
                self.root
                / "selection"
                / "joint"
                / "joint_campaign_lock.json"
            )
            for role in self.registry["carried_predictor_bundle_locks"]:
              for seed in (101, 202, 303):
                output = self.dataset_root / role / f"seed_{seed}"
                rows.append(
                    {
                        "task_id": f"{phase_id}:{len(rows)}",
                        "argv": [
                            "python",
                            "scripts/prepare_retb_final_consumer_seed.py",
                            "--campaign-root",
                            str(self.root),
                            "--pipeline-seed",
                            str(seed),
                            "--carried-shape-role",
                            role,
                            "--output-dir",
                            str(output),
                        ],
                        "environment": {},
                        "expected_outputs": _dataset_outputs(output),
                        "input_artifact_hashes": {
                            "campaign_spec": self.campaign_sha,
                            "production_graph": self.graph_sha,
                            "joint_campaign_lock": joint_lock[
                                "content_hash"
                            ],
                            "selected_j5_registration": joint_lock[
                                "carried_by_shape_role"
                            ][role]["selected_j5_by_seed"][str(seed)][
                                "registration_sha256"
                            ],
                        },
                    }
                )
            resource = "gpu"
        elif phase_id == "TOKEN_REFINER_WAVE":
            selected = [
                row
                for row in self.registry["rows"]
                if row["consumer_kind"] == "TR_REFINE"
            ]
            rows = [
                self._consumer_row(phase_id, index, row)
                for index, row in enumerate(selected)
            ]
            resource = "gpu"
        elif phase_id == "TOKEN_REFINER_SELECT":
            rows = []
            for role in self.registry["carried_predictor_bundle_locks"]:
                output = (
                    self.root
                    / "selection"
                    / "final_consumers"
                    / role
                    / "token_refiner_lock.json"
                )
                metrics = [
                    self.root
                    / "runs"
                    / "final_consumers"
                    / (
                        f"RETB_{role}_TR_REFINE_{variant}_ND0_NONE_"
                        f"TOKEN_PREDICTED_S{seed}"
                    )
                    / "val_design"
                    / "metrics.json"
                    for variant in ("TR1_NATIVE_BASE", "TR2_ALL_NATIVE")
                    for seed in (101, 202, 303)
                ]
                rows.append({
                    "task_id": f"{phase_id}:{len(rows)}",
                    "argv": [
                        "python",
                        "scripts/select_retb_token_refiner.py",
                        "--campaign-root",
                        str(self.root),
                        "--carried-shape-role",
                        role,
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
                        **{
                            f"eligible_metric_{index}": _sha256(path)
                            for index, path in enumerate(metrics)
                        },
                    },
                })
            resource = "cpu"
        else:
            selected = [
                row
                for row in self.registry["rows"]
                if row["consumer_kind"] != "TR_REFINE"
            ]
            rows = [
                self._consumer_row(phase_id, index, row)
                for index, row in enumerate(selected)
            ]
            resource = "gpu"
        return build_internal_phase_plan(
            campaign_root=self.root,
            campaign_spec_sha256=self.campaign_sha,
            production_graph_sha256=self.graph_sha,
            controller_id="final_consumer_campaign",
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
    planner = FinalConsumerPlanner(
        root=args.campaign_root, campaign=campaign, graph=graph
    )
    artifact = execute_phased_controller(
        campaign_root=args.campaign_root,
        controller_id="final_consumer_campaign",
        phase_ids=FINAL_CONSUMER_PHASE_ORDER,
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
