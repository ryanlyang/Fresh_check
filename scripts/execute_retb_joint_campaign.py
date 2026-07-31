#!/usr/bin/env python3
"""Execute the complete restart-safe RETB J1--J5 joint bridge campaign."""

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
    execute_phased_controller,
    phase_plan_path,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_campaign import (  # noqa: E402
    JOINT_PHASE_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    STAGE_E_SHAPES,
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


def _outputs(root: Path) -> list[str]:
    values = [
        root / "assets" / "run.json",
        root / "assets" / "graph" / "joint_graph_template.json",
        root / "assets" / "graph" / "joint_graph_template.pt",
        root / "registration.json",
        root / "training_curves.json",
        root / "best_model_val.pt",
        root / "val_design" / "joint_predictions.npz",
        root / "val_design" / "joint_predictions_manifest.json",
        root / "val_design" / "metrics.json",
    ]
    for split in ("model_train", "val_stop", "val_design"):
        values.extend(
            (
                root
                / "assets"
                / "datasets"
                / split
                / "joint_dataset.json",
                root
                / "assets"
                / "datasets"
                / split
                / "joint_dataset.pt",
            )
        )
    return [str(value) for value in values]


class JointPlanner:
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
        carried_root = self.root / "selection" / "predictor_bundle" / "carried"
        self.lock_paths = {
            role: carried_root / f"{role}.json" for role in STAGE_E_SHAPES
        }
        self.locks = {
            role: load_hashed_json(path)
            for role, path in self.lock_paths.items()
        }

    def _training_row(
        self,
        *,
        phase: str,
        index: int,
        carried_shape_role: str,
        variant: str,
        seed: int,
        blocks: int | None = None,
        selection: Path | None = None,
        selected_j4: Path | None = None,
    ) -> dict[str, Any]:
        suffix = f"_N{blocks}" if blocks is not None else ""
        lock = self.locks[carried_shape_role]
        lock_path = self.lock_paths[carried_shape_role]
        output = (
            self.root
            / "runs"
            / "joint"
            / carried_shape_role
            / f"RETB_{variant}_S{seed}{suffix}"
        )
        argv = [
            "python",
            "scripts/execute_retb_joint_training_row.py",
            "--campaign-root",
            str(self.root),
            "--variant",
            variant,
            "--pipeline-seed",
            str(seed),
            "--predictor-bundle-lock",
            str(lock_path),
            "--output-dir",
            str(output),
        ]
        inputs = {
            "campaign_spec": self.campaign_sha,
            "production_graph": self.graph_sha,
            "predictor_bundle_lock": lock["content_hash"],
        }
        if blocks is not None:
            argv.extend(["--final-particle-blocks", str(blocks)])
        if selection is not None and selected_j4 is not None:
            selection_payload = load_hashed_json(selection)
            registration = load_hashed_json(
                selected_j4 / "registration.json"
            )
            argv.extend(
                [
                    "--j4-selection",
                    str(selection),
                    "--selected-j4-output",
                    str(selected_j4),
                ]
            )
            inputs["j4_block_selection"] = selection_payload[
                "content_hash"
            ]
            inputs["selected_j4_registration"] = registration[
                "content_hash"
            ]
        return {
            "task_id": f"{phase}:{index}",
            "argv": argv,
            "environment": {
                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
            },
            "expected_outputs": _outputs(output),
            "input_artifact_hashes": inputs,
        }

    def phase_plan(
        self,
        phase_id: str,
        sequence_index: int,
        completions: Mapping[str, str],
    ) -> dict[str, Any]:
        if phase_id == "J0_J4_CANDIDATES":
            configurations = [
                (role, variant, seed, blocks)
                for role in STAGE_E_SHAPES
                for variant in (
                    "J1_SHARED_CONTEXT",
                    "J2_COUPLED_DECODER",
                    "J3_INDEPENDENT_PLUS_ADAPTER",
                    "J4_BRIDGE_FINETUNE",
                )
                for blocks in (
                    (2, 4) if variant == "J4_BRIDGE_FINETUNE" else (None,)
                )
                for seed in (101, 202, 303)
            ]
            rows = [
                self._training_row(
                    phase=phase_id,
                    index=index,
                    carried_shape_role=role,
                    variant=variant,
                    seed=seed,
                    blocks=blocks,
                )
                for index, (role, variant, seed, blocks) in enumerate(
                    configurations
                )
            ]
            resource = "gpu"
        elif phase_id == "J4_BLOCK_SELECT":
            rows = []
            for index, role in enumerate(STAGE_E_SHAPES):
                lock = self.locks[role]
                metric_paths = [
                    self.root
                    / "runs"
                    / "joint"
                    / role
                    / f"RETB_J4_BRIDGE_FINETUNE_S{seed}_N{blocks}"
                    / "val_design"
                    / "metrics.json"
                    for blocks in (2, 4)
                    for seed in (101, 202, 303)
                ]
                selection_root = self.root / "selection" / "joint" / role
                output = selection_root / "j4_blocks.json"
                configuration = (
                    selection_root / "j4_selector_configuration.json"
                )
                payload = {
                    "metric_artifact_paths": [
                        str(path) for path in metric_paths
                    ],
                    "predictor_bundle_lock_sha256": lock["content_hash"],
                    "label_manifest_hashes_by_seed": lock[
                        "selection_data_hashes"
                    ]["label_manifests"],
                }
                configuration.parent.mkdir(parents=True, exist_ok=True)
                encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
                if configuration.exists():
                    if configuration.read_text("utf-8") != encoded:
                        raise FileExistsError(
                            "J4 selector configuration differs"
                        )
                else:
                    configuration.write_text(encoded, encoding="utf-8")
                rows.append({
                    "task_id": f"{phase_id}:{index}",
                    "argv": [
                        "python",
                        "scripts/select_retb_j4_blocks.py",
                        "--campaign-root",
                        str(self.root),
                        "--configuration",
                        str(configuration),
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
                        "predictor_bundle_lock": lock["content_hash"],
                        "selector_configuration": _sha256(configuration),
                        **{
                            f"metric_{index}": _sha256(path)
                            for index, path in enumerate(metric_paths)
                        },
                    },
                })
            resource = "cpu"
        else:
            rows = []
            for role in STAGE_E_SHAPES:
                selection = (
                    self.root / "selection" / "joint" / role / "j4_blocks.json"
                )
                selected = load_hashed_json(selection)
                blocks = int(selected["selected_final_particle_blocks"])
                for seed in (101, 202, 303):
                    selected_j4 = (
                        self.root
                        / "runs"
                        / "joint"
                        / role
                        / f"RETB_J4_BRIDGE_FINETUNE_S{seed}_N{blocks}"
                    )
                    rows.append(
                        self._training_row(
                            phase=phase_id,
                            index=len(rows),
                            carried_shape_role=role,
                            variant="J5_END_TO_END",
                            seed=seed,
                            selection=selection,
                            selected_j4=selected_j4,
                        )
                    )
            resource = "gpu"
        return build_internal_phase_plan(
            campaign_root=self.root,
            campaign_spec_sha256=self.campaign_sha,
            production_graph_sha256=self.graph_sha,
            controller_id="joint_predictor_campaign",
            phase_id=phase_id,
            sequence_index=sequence_index,
            resource=resource,
            maximum_concurrent_tasks=4 if resource == "gpu" else 1,
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
    planner = JointPlanner(
        root=args.campaign_root, campaign=campaign, graph=graph
    )
    artifact = execute_phased_controller(
        campaign_root=args.campaign_root,
        controller_id="joint_predictor_campaign",
        phase_ids=JOINT_PHASE_ORDER,
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
