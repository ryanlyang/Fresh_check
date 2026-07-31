#!/usr/bin/env python3
"""Execute the ordered, restart-safe Stage F--H predictor campaign."""

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

from scripts.execute_retb_target_cache_row import (  # noqa: E402
    _checkpoint_path,
    _selected_templates,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    validate_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.phased_campaign import (  # noqa: E402
    build_internal_phase_plan,
    configured_phase_concurrency,
    execute_phased_controller,
    phase_plan_path,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_campaign import (  # noqa: E402
    PREDICTOR_PHASE_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.predictors import (  # noqa: E402
    RetbTokenPredictor,
    select_widened_resmlp_width,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_d_selection import (  # noqa: E402
    EVIDENCE_MODES,
)
from teacher_logit_reco.relation_expert_token_bridge.step9 import (  # noqa: E402
    build_stage_f_registry,
    build_stage_g_registry,
    materialize_predictor_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.middle_dynamic_plan_factories import (  # noqa: E402
    build_middle_dynamic_factory_input,
    publish_middle_dynamic_factory_input,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    task_manifest_path_for_graph,
)

import torch  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(campaign: Mapping[str, Any]) -> dict[str, Any]:
    source = campaign["source"]
    return {
        "source_commit": source["commit"],
        "source_status_sha256": source["status_sha256"],
        "source_dirty": bool(source["dirty"]),
    }


class PredictorCampaignPlanner:
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
        self.selection = load_hashed_json(
            self.root / "selection" / "locked_bridge_coordinates.json"
        )
        self.certification = load_hashed_json(
            self.root
            / "selection"
            / "stage_e"
            / "bridge_certification_index.json"
        )
        self.templates = _selected_templates(self.certification)
        self.step9 = load_hashed_json(
            self.root / "registry" / "retb_step9_predictor_bundle.json"
        )

    def _coordinate_index(
        self, *, expert: str, target_mode: str
    ) -> int:
        position = EXPERT_ORDER.index(expert)
        candidates = [
            (index, row)
            for index, row in enumerate(
                self.selection["locked_coordinate_systems"]
            )
            if row["target_tuple"][position] == target_mode
        ]
        homogeneous = [
            pair
            for pair in candidates
            if len(set(pair[1]["target_tuple"])) == 1
        ]
        retained = homogeneous or candidates
        if not retained:
            raise ValueError(
                f"no locked coordinate for {expert}/{target_mode}"
            )
        return min(
            retained,
            key=lambda pair: pair[1]["coordinate_contract_sha256"],
        )[0]

    def _artifacts(
        self,
        *,
        shape: str,
        expert: str,
        seed: int,
        target_mode: str,
        evidence_mode: str = "selected_native_HLT_evidence",
    ) -> dict[str, Any]:
        coordinate_index = self._coordinate_index(
            expert=expert, target_mode=target_mode
        )
        cache_root = (
            self.root
            / "inputs"
            / "target_caches"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{seed}"
        )
        manifests = {
            split: load_hashed_json(
                cache_root / split / "target_cache_manifest.json"
            )
            for split in ("model_train", "val_stop", "val_design")
        }
        specifications = {
            split: load_hashed_json(
                cache_root
                / split
                / "target_cache_specification.json"
            )
            for split in ("model_train", "val_stop", "val_design")
        }
        descriptor = manifests["model_train"]["target_descriptors"][
            expert
        ]
        checkpoint = _checkpoint_path(
            self.root,
            shape=shape,
            expert=expert,
            seed=seed,
            mode=target_mode,
            templates=self.templates,
        )
        fusion_registration = load_hashed_json(
            cache_root / "model_train" / "fusion_registration.json"
        )
        fusion = Path(fusion_registration["checkpoint_path"]).resolve()
        if (
            _sha256(fusion)
            != fusion_registration["checkpoint_sha256"]
            or fusion_registration["shape_id"] != shape
            or int(fusion_registration["pipeline_seed"]) != seed
        ):
            raise ValueError("predictor coordinate fusion lineage differs")
        if evidence_mode in EVIDENCE_MODES and shape in {
            "SHAPE_COMPACT",
            "SHAPE_HIGH",
        }:
            evidence_root = (
                self.root
                / "inputs"
                / "selected_hlt_evidence"
                / shape
                / evidence_mode
                / f"seed_{seed}"
            )
            evidence_manifest = load_hashed_json(
                evidence_root / "evidence_manifest.json"
            )
            registration = load_hashed_json(
                Path(
                    evidence_manifest["expert_parents"][expert][
                        "registration_path"
                    ]
                )
            )
            evidence = {
                split: evidence_root
                / evidence_manifest["splits"][split]["relative_path"]
                for split in ("model_train", "val_stop", "val_design")
            }
        else:
            evidence_root = (
                self.root
                / "selection"
                / "stage_e_parents"
                / shape
                / expert
                / f"seed_{seed}"
            )
            registration = load_hashed_json(
                evidence_root / "hlt_encoder_registration.json"
            )
            evidence = {
                split: evidence_root / f"{split}_pilot_dataset.npz"
                for split in ("model_train", "val_stop", "val_design")
            }
        normalizer = (
            self.root
            / "inputs"
            / "target_normalizers"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{seed}"
            / f"target_normalizer_{expert}.json"
        )
        normalizer_payload = load_hashed_json(normalizer)
        for split in manifests:
            if (
                manifests[split]["target_descriptors"][expert]
                != descriptor
                or specifications[split][
                    "offline_fusion_checkpoint_sha256"
                ]
                != _sha256(fusion)
            ):
                raise ValueError(
                    "predictor split target/fusion lineage differs"
                )
        return {
            "coordinate_index": coordinate_index,
            "manifests": manifests,
            "specifications": specifications,
            "checkpoint": checkpoint,
            "fusion": fusion,
            "evidence": evidence,
            "evidence_root": evidence_root,
            "normalizer": normalizer,
            "normalizer_payload": normalizer_payload,
            "native_registration": registration,
            "descriptor": descriptor,
        }

    def _training_row(
        self,
        *,
        phase_id: str,
        index: int,
        configuration: Mapping[str, Any],
    ) -> dict[str, Any]:
        config = dict(configuration)
        run_id = str(config["run_id"])
        stage = str(config["stage"])
        seed = int(config["pipeline_seed"])
        expert = str(config["expert_id"])
        shape = str(config["shape_alias"])
        target_mode = str(config.get("target_mode", "T0_PURE"))
        artifacts = self._artifacts(
            shape=shape,
            expert=expert,
            seed=seed,
            target_mode=target_mode,
            evidence_mode=str(
                config.get(
                    "hlt_evidence_mode",
                    "selected_native_HLT_evidence",
                )
            ),
        )
        manifests = artifacts["manifests"]
        evidence = artifacts["evidence"]
        parent_hashes = {
            "step9_bundle": self.step9["content_hash"],
            "model_train_target_cache": manifests["model_train"][
                "content_hash"
            ],
            "val_stop_target_cache": manifests["val_stop"]["content_hash"],
            "val_design_target_cache": manifests["val_design"][
                "content_hash"
            ],
            "target_normalizer": artifacts["normalizer_payload"][
                "content_hash"
            ],
            "slot_queries": artifacts["descriptor"][
                "slot_query_sha256"
            ],
            "offline_target_checkpoint": _sha256(
                artifacts["checkpoint"]
            ),
            "offline_fusion": _sha256(artifacts["fusion"]),
            "native_hlt_expert": artifacts["native_registration"][
                "content_hash"
            ],
            "model_train_hlt_evidence_cache": _sha256(
                evidence["model_train"]
            ),
            "val_stop_hlt_evidence_cache": _sha256(
                evidence["val_stop"]
            ),
            "val_design_hlt_evidence_cache": _sha256(
                evidence["val_design"]
            ),
            "model_train_identity_manifest": load_hashed_json(
                Path(manifests["model_train"]["identity_manifest_path"])
                if "identity_manifest_path" in manifests["model_train"]
                else (
                    self.root
                    / "inputs"
                    / "target_caches"
                    / f"coordinate_{artifacts['coordinate_index']:03d}"
                    / shape
                    / f"seed_{seed}"
                    / "model_train"
                    / "identity_manifest.json"
                )
            )["content_hash"],
            "val_stop_identity_manifest": load_hashed_json(
                self.root
                / "inputs"
                / "target_caches"
                / f"coordinate_{artifacts['coordinate_index']:03d}"
                / shape
                / f"seed_{seed}"
                / "val_stop"
                / "identity_manifest.json"
            )["content_hash"],
            "val_design_identity_manifest": load_hashed_json(
                self.root
                / "inputs"
                / "target_caches"
                / f"coordinate_{artifacts['coordinate_index']:03d}"
                / shape
                / f"seed_{seed}"
                / "val_design"
                / "identity_manifest.json"
            )["content_hash"],
        }
        allocation = manifests["model_train"]["allocation"][expert]
        run = bind_source(
            materialize_predictor_run(
                run_id=run_id,
                stage=stage,
                pipeline_seed=seed,
                expert_id=expert,
                shape_id=shape,
                token_count=int(allocation[0]),
                token_dimension=int(allocation[1]),
                architecture=str(config["architecture"]),
                context=str(config["context"]),
                objective_id=str(config["objective_id"]),
                uncertainty_head=str(
                    config.get("uncertainty_head", "U_SLOT")
                ),
                normalization_mode=str(
                    config.get("normalization_mode", "N_UNCLIPPED")
                ),
                target_mode=target_mode,
                hlt_evidence_mode=str(
                    config.get(
                        "hlt_evidence_mode",
                        "selected_native_HLT_evidence",
                    )
                ),
                learning_rate=float(config["learning_rate"]),
                dropout=float(config["dropout"]),
                role=str(config["role"]),
                parent_hashes=parent_hashes,
                control_variant=str(
                    config.get("control_variant", "STANDARD")
                ),
                residual_hidden_width=config.get(
                    "residual_hidden_width"
                ),
            ),
            source_snapshot=_snapshot(self.campaign),
        )
        run_path = (
            self.root
            / "registry"
            / "predictor_runs"
            / f"{run_id}.json"
        )
        write_immutable_json(run_path, run)
        output = self.root / "runs" / "predictors" / run_id
        argv = [
            "python",
            "scripts/execute_retb_predictor_training_row.py",
            "--campaign-root",
            str(self.root),
            "--run",
            str(run_path),
        ]
        for cli, split in (
            ("model-train", "model_train"),
            ("val-stop", "val_stop"),
            ("val-design", "val_design"),
        ):
            argv.extend(
                [
                    f"--{cli}-target-cache",
                    str(
                        self.root
                        / "inputs"
                        / "target_caches"
                        / (
                            f"coordinate_{artifacts['coordinate_index']:03d}"
                        )
                        / shape
                        / f"seed_{seed}"
                        / split
                        / "target_cache_manifest.json"
                    ),
                    f"--{cli}-evidence",
                    str(evidence[split]),
                ]
            )
        argv.extend(
            [
                "--target-normalizer",
                str(artifacts["normalizer"]),
                "--target-checkpoint",
                str(artifacts["checkpoint"]),
                "--fusion-checkpoint",
                str(artifacts["fusion"]),
                "--output-dir",
                str(output),
            ]
        )
        expected = [
            *[
                str(output / "prepared" / f"{split}.{suffix}")
                for split in ("model_train", "val_stop", "val_design")
                for suffix in ("npz", "json")
            ],
            str(output / "training" / "worker_registration.json"),
            str(output / "training" / "best_model_val.pt"),
            str(output / "training" / "capacity_report.json"),
            str(output / "val_design" / "predictor_outputs.npz"),
            str(
                output
                / "val_design"
                / "predictor_outputs_manifest.json"
            ),
            str(output / "val_design" / "val_design_metrics.json"),
        ]
        return {
            "task_id": f"{phase_id}:{index}",
            "argv": argv,
            "environment": {
                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
            },
            "expected_outputs": expected,
            "input_artifact_hashes": {
                "campaign_spec": self.campaign_sha,
                "production_graph": self.graph_sha,
                "run": run["content_hash"],
                **parent_hashes,
            },
        }

    def _selector_row(
        self,
        *,
        phase_id: str,
        selection_phase: str,
        metrics: Sequence[Path],
        output: Path,
        registry: Path | None = None,
        followup: Path | None = None,
    ) -> dict[str, Any]:
        argv = [
            "python",
            "scripts/select_retb_predictor_phase.py",
            "--campaign-root",
            str(self.root),
            "--phase",
            selection_phase,
        ]
        for path in metrics:
            argv.extend(["--metric", str(path)])
        if registry is not None:
            argv.extend(["--registry", str(registry)])
        argv.extend(["--output", str(output)])
        outputs = [str(output)]
        if followup is not None:
            argv.extend(
                ["--followup-registry-output", str(followup)]
            )
            outputs.append(str(followup))
        return {
            "task_id": f"{phase_id}:0",
            "argv": argv,
            "environment": {
                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
            },
            "expected_outputs": outputs,
            "input_artifact_hashes": {
                "campaign_spec": self.campaign_sha,
                "production_graph": self.graph_sha,
                **{
                    f"metric_{index:04d}": _sha256(path)
                    for index, path in enumerate(metrics)
                },
            },
        }

    @staticmethod
    def _metric_paths(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
        return [
            root
            / "runs"
            / "predictors"
            / str(row["run_id"])
            / "val_design"
            / "val_design_metrics.json"
            for row in rows
        ]

    def phase_plan(
        self,
        phase_id: str,
        sequence_index: int,
        completions: Mapping[str, str],
    ) -> dict[str, Any]:
        selection_root = self.root / "selection" / "predictor_phases"
        f_registry = build_stage_f_registry()
        if phase_id == "F_ARCHITECTURE_SCREEN":
            rows = [
                self._training_row(
                    phase_id=phase_id, index=index, configuration=row
                )
                for index, row in enumerate(f_registry["rows"])
            ]
            resource = "gpu"
        elif phase_id == "F_ARCHITECTURE_SELECT":
            rows = [
                self._selector_row(
                    phase_id=phase_id,
                    selection_phase="F_ARCHITECTURE",
                    metrics=self._metric_paths(
                        self.root, f_registry["rows"]
                    ),
                    output=selection_root
                    / "architecture_selection.json",
                    followup=selection_root
                    / "optimizer_followup_registry.json",
                )
            ]
            resource = "cpu"
        elif phase_id == "F_OPTIMIZER_SCREEN":
            registry = load_hashed_json(
                selection_root / "optimizer_followup_registry.json"
            )
            rows = [
                self._training_row(
                    phase_id=phase_id,
                    index=index,
                    configuration={
                        **row,
                        "stage": "F",
                        "shape_alias": row["shape_alias"],
                        "objective_id": "W_CANONICAL",
                        "uncertainty_head": "U_SLOT",
                        "normalization_mode": "N_UNCLIPPED",
                        "target_mode": "T0_PURE",
                        "hlt_evidence_mode": (
                            "selected_native_HLT_evidence"
                        ),
                        "role": "scientific_candidate",
                    },
                )
                for index, row in enumerate(registry["rows"])
            ]
            resource = "gpu"
        elif phase_id == "F_OPTIMIZER_SELECT":
            registry_path = (
                selection_root / "optimizer_followup_registry.json"
            )
            registry = load_hashed_json(registry_path)
            rows = [
                self._selector_row(
                    phase_id=phase_id,
                    selection_phase="F_OPTIMIZER",
                    metrics=self._metric_paths(
                        self.root, registry["rows"]
                    ),
                    output=selection_root / "optimizer_selection.json",
                    registry=registry_path,
                )
            ]
            resource = "cpu"
        elif phase_id == "G_OBJECTIVE_SCREEN":
            architecture = load_hashed_json(
                selection_root / "architecture_selection.json"
            )
            optimizer = load_hashed_json(
                selection_root / "optimizer_selection.json"
            )
            g_registry = build_stage_g_registry()
            rows = []
            for index, template in enumerate(g_registry["templates"]):
                family_key = (
                    "selected_direct"
                    if template["architecture_family"]
                    == "SELECTED_DIRECT"
                    else "selected_gated"
                )
                rows.append(
                    self._training_row(
                        phase_id=phase_id,
                        index=index,
                        configuration={
                            **template,
                            "run_id": template["template_id"],
                            "architecture": architecture[
                                "selected_families"
                            ][family_key]["architecture"],
                            "context": architecture[
                                "selected_families"
                            ][family_key]["context"],
                            "learning_rate": optimizer[
                                "selected_families"
                            ][family_key]["learning_rate"],
                            "dropout": optimizer[
                                "selected_families"
                            ][family_key]["dropout"],
                            "target_mode": "T0_PURE",
                            "hlt_evidence_mode": (
                                "selected_native_HLT_evidence"
                            ),
                            "role": (
                                "semantic_control"
                                if template["objective_id"]
                                == "W_LOGIT_ONLY"
                                else "scientific_candidate"
                            ),
                        },
                    )
                )
            resource = "gpu"
        elif phase_id == "G_CONFIGURATION_SELECT":
            g_registry = build_stage_g_registry()
            rows = [
                self._selector_row(
                    phase_id=phase_id,
                    selection_phase="G_CONFIGURATION",
                    metrics=self._metric_paths(
                        self.root,
                        [
                            {
                                **row,
                                "run_id": row["template_id"],
                            }
                            for row in g_registry["templates"]
                        ],
                    ),
                    output=selection_root
                    / "g_configuration_selection.json",
                )
            ]
            resource = "cpu"
        elif phase_id == "H_EVIDENCE_SELECT":
            output = selection_root / "stage_d_evidence_selection.json"
            confirmation = (
                selection_root / "stage_d_evidence_confirmations.json"
            )
            rows = [
                {
                    "task_id": f"{phase_id}:0",
                    "argv": [
                        "python",
                        "scripts/select_retb_stage_d_evidence_modes.py",
                        "--campaign-root",
                        str(self.root),
                        "--output",
                        str(output),
                        "--confirmation-output",
                        str(confirmation),
                    ],
                    "environment": {
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                    },
                    "expected_outputs": [str(output), str(confirmation)],
                    "input_artifact_hashes": {
                        "campaign_spec": self.campaign_sha,
                        "production_graph": self.graph_sha,
                        "stage_d_registry": load_hashed_json(
                            self.root
                            / "registry"
                            / "retb_stage_d_runs.json"
                        )["content_hash"],
                    },
                }
            ]
            resource = "cpu"
        elif phase_id == "H_EVIDENCE_CONFIRM":
            rows = self._h_evidence_confirmation_rows(
                phase_id=phase_id, selection_root=selection_root
            )
            resource = "gpu"
        elif phase_id == "H_EVIDENCE_CACHE":
            rows = self._h_evidence_cache_rows(
                phase_id=phase_id, selection_root=selection_root
            )
            resource = "cpu"
        elif phase_id == "H_EVIDENCE_FUSION_CONFIRM":
            rows = self._h_evidence_fusion_rows(
                phase_id=phase_id, selection_root=selection_root
            )
            resource = "gpu"
        else:
            rows = self._h_confirmation_rows(
                phase_id=phase_id,
                selection_root=selection_root,
            )
            resource = "gpu"
        return build_internal_phase_plan(
            campaign_root=self.root,
            campaign_spec_sha256=self.campaign_sha,
            production_graph_sha256=self.graph_sha,
            controller_id="predictor_campaign",
            phase_id=phase_id,
            sequence_index=sequence_index,
            resource=resource,
            maximum_concurrent_tasks=configured_phase_concurrency(
                resource=resource,
                family="predictor",
                row_count=len(rows),
            ),
            rows=rows,
            prerequisite_completion_hashes=completions,
            source=self.campaign["source"],
        )

    def _h_evidence_confirmation_rows(
        self, *, phase_id: str, selection_root: Path
    ) -> list[dict[str, Any]]:
        selection = load_hashed_json(
            selection_root / "stage_d_evidence_selection.json"
        )
        confirmation_path = (
            selection_root / "stage_d_evidence_confirmations.json"
        )
        confirmation = load_hashed_json(confirmation_path)
        target_index = load_hashed_json(
            self.root / "inputs" / "stage_d_offline_targets" / "index.json"
        )
        selected_configs = [
            row["configuration"] for row in selection["selected_rows"]
        ]
        rows = []
        for confirmation_row in confirmation["rows"]:
            config = confirmation_row["configuration"]
            seed = int(confirmation_row["seed"])
            if (
                confirmation_row["component"] != "HLT_EXPERT"
                or config not in selected_configs
                or seed == 101
            ):
                continue
            expert = str(config["expert_id"])
            shape = str(config["shape_id"])
            run_id = str(confirmation_row["run_id"])
            output = (
                self.root
                / "runs"
                / "stage_d"
                / "hlt_experts"
                / run_id
                / f"seed_{seed}"
            )
            offline_root = (
                self.root
                / "selection"
                / "offline_experts"
                / shape
                / expert
                / f"seed_{seed}"
            )
            offline_registration = load_hashed_json(
                offline_root / "checkpoint_registration.json"
            )
            argv = [
                "python",
                "scripts/train_retb_native_hlt_expert.py",
                "--campaign-root",
                str(self.root),
                "--run-id",
                run_id,
                "--confirmation-registry",
                str(confirmation_path),
            ]
            for replica in range(4):
                argv.extend(
                    [
                        "--train-cache",
                        (
                            f"{replica}="
                            f"{self.root}/inputs/hlt_v3/model_train/"
                            f"replica_{replica}/R_MULTI/D_NOMINAL/"
                            "hlt_v3_metadata.json"
                        ),
                    ]
                )
            argv.extend(
                [
                    "--val-stop-cache",
                    (
                        f"0={self.root}/inputs/hlt_v3/val_stop/"
                        "replica_0/R_FIXED/D_NOMINAL/hlt_v3_metadata.json"
                    ),
                    "--val-design-cache",
                    (
                        f"0={self.root}/inputs/hlt_v3/val_design/"
                        "replica_0/R_FIXED/D_NOMINAL/hlt_v3_metadata.json"
                    ),
                    "--train-labels",
                    str(
                        self.root
                        / "inputs"
                        / "offline"
                        / "model_train"
                        / "offline_inputs.npz"
                    ),
                    "--val-stop-labels",
                    str(
                        self.root
                        / "inputs"
                        / "offline"
                        / "val_stop"
                        / "offline_inputs.npz"
                    ),
                    "--val-design-labels",
                    str(
                        self.root
                        / "inputs"
                        / "offline"
                        / "val_design"
                        / "offline_inputs.npz"
                    ),
                    "--uniform-shapes",
                    str(
                        self.root
                        / "selection"
                        / "retb_offline_shapes.json"
                    ),
                    "--offline-registration",
                    str(offline_root / "checkpoint_registration.json"),
                    "--offline-checkpoint",
                    str(offline_root / "best_model_val.pt"),
                    "--relation-normalization",
                    str(
                        self.root
                        / "inputs"
                        / "normalization"
                        / "hlt_shared_500k"
                        / "relation.json"
                    ),
                    "--region-normalization",
                    str(
                        self.root
                        / "inputs"
                        / "normalization"
                        / "hlt_shared_500k"
                        / "region.json"
                    ),
                    "--region-tree-root",
                    str(self.root / "inputs" / "region_tree" / "hlt"),
                    "--output-dir",
                    str(output),
                ]
            )
            hashes = {
                "campaign_spec": self.campaign_sha,
                "production_graph": self.graph_sha,
                "stage_d_evidence_selection": selection["content_hash"],
                "stage_d_confirmation_registry": confirmation[
                    "content_hash"
                ],
                "offline_registration": offline_registration["content_hash"],
                "offline_checkpoint": _sha256(
                    offline_root / "best_model_val.pt"
                ),
            }
            if config["mode"] == "HE_DUAL_OBJECTIVE":
                target = (
                    self.root
                    / "inputs"
                    / "stage_d_offline_targets"
                    / shape
                    / expert
                    / f"seed_{seed}.npz"
                )
                argv.extend(["--offline-train-targets", str(target)])
                hashes["offline_train_targets"] = _sha256(target)
                hashes["offline_target_index"] = target_index[
                    "content_hash"
                ]
            rows.append(
                {
                    "task_id": f"{phase_id}:{len(rows)}",
                    "argv": argv,
                    "environment": {
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                    },
                    "expected_outputs": [
                        str(output / "checkpoint_registration.json"),
                        str(output / "best_model_val.pt"),
                        str(output / "native_output_manifest.json"),
                    ],
                    "input_artifact_hashes": hashes,
                }
            )
        if len(rows) != 84:
            raise RuntimeError(
                f"Stage-H evidence confirmations differ: {len(rows)}"
            )
        return rows

    def _h_evidence_fusion_rows(
        self, *, phase_id: str, selection_root: Path
    ) -> list[dict[str, Any]]:
        selection_path = (
            selection_root / "stage_d_evidence_selection.json"
        )
        confirmation_path = (
            selection_root / "stage_d_evidence_confirmations.json"
        )
        selection = load_hashed_json(selection_path)
        confirmation = load_hashed_json(confirmation_path)
        rows = []
        for row in confirmation["rows"]:
            config = row["configuration"]
            if row["component"] != "NATIVE_HLT_FUSION":
                continue
            seed = int(row["seed"])
            run_id = str(row["run_id"])
            shape = str(config["shape_id"])
            cache_root = (
                self.root
                / "inputs"
                / "selected_native_fusion"
                / shape
                / config["fusion_variant"]
                / f"seed_{seed}"
            )
            train_cache = (
                cache_root / "model_train_native_hlt_tokens.json"
            )
            stop_cache = cache_root / "val_stop_native_hlt_tokens.json"
            output = (
                self.root
                / "runs"
                / "stage_d"
                / "native_fusions"
                / run_id
                / f"seed_{seed}"
            )
            rows.append(
                {
                    "task_id": f"{phase_id}:{len(rows)}",
                    "argv": [
                        "python",
                        "scripts/train_retb_native_hlt_fusion.py",
                        "--campaign-root",
                        str(self.root),
                        "--run-id",
                        run_id,
                        "--confirmation-registry",
                        str(confirmation_path),
                        "--model-train-cache",
                        str(train_cache),
                        "--val-stop-cache",
                        str(stop_cache),
                        "--output-dir",
                        str(output),
                    ],
                    "environment": {
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                    },
                    "expected_outputs": [
                        str(train_cache),
                        str(
                            cache_root
                            / "model_train_native_hlt_tokens.npz"
                        ),
                        str(stop_cache),
                        str(
                            cache_root / "val_stop_native_hlt_tokens.npz"
                        ),
                        str(output / "fusion_registration.json"),
                        str(output / "best_model_val.pt"),
                        str(output / "training_curves.json"),
                    ],
                    "input_artifact_hashes": {
                        "campaign_spec": self.campaign_sha,
                        "production_graph": self.graph_sha,
                        "stage_d_evidence_selection": selection[
                            "content_hash"
                        ],
                        "stage_d_confirmation_registry": confirmation[
                            "content_hash"
                        ],
                    },
                }
            )
        if len(rows) != 12:
            raise RuntimeError(
                "selected native-fusion confirmation coverage differs: "
                f"{len(rows)}"
            )
        return rows

    def _h_evidence_cache_rows(
        self, *, phase_id: str, selection_root: Path
    ) -> list[dict[str, Any]]:
        selection_path = (
            selection_root / "stage_d_evidence_selection.json"
        )
        confirmation_path = (
            selection_root / "stage_d_evidence_confirmations.json"
        )
        selection = load_hashed_json(selection_path)
        confirmation = load_hashed_json(confirmation_path)
        rows = []
        for shape in ("SHAPE_COMPACT", "SHAPE_HIGH"):
            for mode in EVIDENCE_MODES:
                for seed in (101, 202, 303):
                    output = (
                        self.root
                        / "inputs"
                        / "selected_hlt_evidence"
                        / shape
                        / mode
                        / f"seed_{seed}"
                    )
                    rows.append(
                        {
                            "task_id": f"{phase_id}:{len(rows)}",
                            "argv": [
                                "python",
                                "scripts/materialize_retb_selected_hlt_evidence.py",
                                "--campaign-root",
                                str(self.root),
                                "--selection",
                                str(selection_path),
                                "--confirmation-registry",
                                str(confirmation_path),
                                "--shape-id",
                                shape,
                                "--evidence-mode",
                                mode,
                                "--pipeline-seed",
                                str(seed),
                                "--output-dir",
                                str(output),
                            ],
                            "environment": {
                                "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                            },
                            "expected_outputs": [
                                str(output / f"{split}_evidence.npz")
                                for split in (
                                    "model_train",
                                    "val_stop",
                                    "val_design",
                                )
                            ]
                            + [str(output / "evidence_manifest.json")],
                            "input_artifact_hashes": {
                                "campaign_spec": self.campaign_sha,
                                "production_graph": self.graph_sha,
                                "stage_d_evidence_selection": selection[
                                    "content_hash"
                                ],
                                "stage_d_confirmation_registry": confirmation[
                                    "content_hash"
                                ],
                            },
                        }
                    )
        return rows

    def _h_confirmation_rows(
        self, *, phase_id: str, selection_root: Path
    ) -> list[dict[str, Any]]:
        architecture = load_hashed_json(
            selection_root / "architecture_selection.json"
        )
        optimizer = load_hashed_json(
            selection_root / "optimizer_selection.json"
        )
        selected = load_hashed_json(
            selection_root / "g_configuration_selection.json"
        )
        shapes = tuple(
            str(value)
            for value in load_hashed_json(
                self.root / "registry" / "retb_stage_e_templates.json"
            )["shapes"]
        )
        uniform_shapes = ("SHAPE_COMPACT", "SHAPE_HIGH")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(
            *,
            family: str,
            expert: str,
            shape: str,
            seed: int,
            target_mode: str = "T0_PURE",
            context: str | None = None,
            architecture_override: str | None = None,
            objective_override: str | None = None,
            role_override: str | None = None,
            control_variant: str = "STANDARD",
            residual_hidden_width: int | None = None,
            evidence_mode: str = "HE_OFFLINE_INIT",
            tag: str = "PRIMARY",
        ) -> None:
            family_key = family
            family_upper = (
                "selected_direct"
                if family == "selected_direct"
                else "selected_gated"
            )
            architecture_row = architecture["selected_families"][
                family_upper
            ]
            g_row = selected["selected_families"][family_key]
            objective = objective_override or g_row["objective_id"]
            role = role_override or (
                "semantic_control"
                if objective == "W_LOGIT_ONLY"
                else "scientific_candidate"
            )
            run_id = "_".join(
                (
                    "RETB",
                    "H",
                    tag,
                    family.upper(),
                    expert,
                    shape,
                    target_mode,
                    evidence_mode,
                    f"S{seed}",
                )
            )
            if run_id in seen:
                return
            seen.add(run_id)
            configuration = {
                "run_id": run_id,
                "stage": "H",
                "pipeline_seed": seed,
                "expert_id": expert,
                "shape_alias": shape,
                "architecture": (
                    architecture_override
                    or architecture_row["architecture"]
                ),
                "context": context or architecture_row["context"],
                "objective_id": objective,
                "uncertainty_head": g_row["uncertainty_head"],
                "normalization_mode": g_row["normalization_mode"],
                "target_mode": target_mode,
                "hlt_evidence_mode": evidence_mode,
                "learning_rate": optimizer["selected_families"][
                    family_upper
                ]["learning_rate"],
                "dropout": optimizer["selected_families"][
                    family_upper
                ]["dropout"],
                "role": role,
                "control_variant": control_variant,
                "residual_hidden_width": residual_hidden_width,
            }
            rows.append(
                self._training_row(
                    phase_id=phase_id,
                    index=len(rows),
                    configuration=configuration,
                )
            )

        for family in ("selected_direct", "selected_gated"):
            for expert in EXPERT_ORDER:
                eligible_modes = sorted(
                    {
                        str(system["target_tuple"][
                            EXPERT_ORDER.index(expert)
                        ])
                        for system in self.selection[
                            "locked_coordinate_systems"
                        ]
                    }
                )
                for shape in shapes:
                    for seed in (101, 202, 303):
                        add(
                            family=family,
                            expert=expert,
                            shape=shape,
                            seed=seed,
                        )
                        for target_mode in eligible_modes:
                            if target_mode != "T0_PURE":
                                add(
                                    family=family,
                                    expert=expert,
                                    shape=shape,
                                    seed=seed,
                                    target_mode=target_mode,
                                    tag="TARGET",
                                )

            for expert in ("PT", "TRACK", "REGION"):
                for shape in uniform_shapes:
                    for seed in (101, 202, 303):
                        add(
                            family=family,
                            expert=expert,
                            shape=shape,
                            seed=seed,
                            context="C3_ALL_PARTICLE",
                            tag="C3",
                        )

            for expert in EXPERT_ORDER:
                for shape in uniform_shapes:
                    allocation = self._artifacts(
                        shape=shape,
                        expert=expert,
                        seed=101,
                        target_mode="T0_PURE",
                        evidence_mode="HE_OFFLINE_INIT",
                    )["manifests"]["model_train"]["allocation"][expert]
                    query = torch.zeros(
                        int(allocation[0]), int(allocation[1])
                    )
                    selected_model = RetbTokenPredictor(
                        architecture=architecture["selected_families"][
                            family
                        ]["architecture"],
                        context=architecture["selected_families"][family][
                            "context"
                        ],
                        target_expert_id=expert,
                        token_count=int(allocation[0]),
                        token_dimension=int(allocation[1]),
                        offline_slot_queries=query,
                        uncertainty_head=selected["selected_families"][
                            family
                        ]["uncertainty_head"],
                        dropout=float(
                            optimizer["selected_families"][family][
                                "dropout"
                            ]
                        ),
                    )
                    affine = RetbTokenPredictor(
                        architecture="A0_AFFINE",
                        context="C0_SELF",
                        target_expert_id=expert,
                        token_count=int(allocation[0]),
                        token_dimension=int(allocation[1]),
                        offline_slot_queries=query,
                        uncertainty_head=selected["selected_families"][
                            family
                        ]["uncertainty_head"],
                        dropout=0.0,
                    )
                    incremental = max(
                        1,
                        sum(p.numel() for p in selected_model.parameters())
                        - sum(p.numel() for p in affine.parameters()),
                    )
                    widened = select_widened_resmlp_width(
                        token_dimension=int(allocation[1]),
                        target_incremental_parameters=incremental,
                    )["hidden_width"]
                    for seed in (101, 202, 303):
                        for evidence_mode in (
                            "HE_SCRATCH_CE",
                            "HE_DUAL_OBJECTIVE",
                        ):
                            add(
                                family=family,
                                expert=expert,
                                shape=shape,
                                seed=seed,
                                evidence_mode=evidence_mode,
                                tag="EVIDENCE",
                            )
                        add(
                            family=family,
                            expert=expert,
                            shape=shape,
                            seed=seed,
                            architecture_override="A1_RESMLP",
                            context="C0_SELF",
                            objective_override="W_CANONICAL",
                            role_override="capacity_control",
                            control_variant="MATCHED_WIDENED_RESMLP",
                            residual_hidden_width=int(widened),
                            tag="WIDE",
                        )
                        add(
                            family=family,
                            expert=expert,
                            shape=shape,
                            seed=seed,
                            role_override="semantic_control",
                            control_variant="ZERO_EVIDENCE",
                            tag="ZERO",
                        )
        if not rows:
            raise RuntimeError("Stage-H confirmation coverage is empty")
        return rows


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
    planner = PredictorCampaignPlanner(
        root=args.campaign_root, campaign=campaign, graph=graph
    )
    artifact = execute_phased_controller(
        campaign_root=args.campaign_root,
        controller_id="predictor_campaign",
        phase_ids=PREDICTOR_PHASE_ORDER,
        phase_builder=planner.phase_plan,
        repo_root=REPO_ROOT,
        slurm_task_script=REPO_ROOT
        / "sbatch"
        / "run_retb_internal_phase_task.sh",
        output_path=args.output,
    )
    h_plan = load_hashed_json(
        phase_plan_path(
            args.campaign_root,
            controller_id="predictor_campaign",
            phase_id="H_CONFIRMATION",
        )
    )
    rows = []
    for source_row in h_plan["rows"]:
        run_index = source_row["argv"].index("--run") + 1
        run = load_hashed_json(Path(source_row["argv"][run_index]))
        output_root = (
            args.campaign_root / "runs" / "predictors" / run["run_id"]
        )
        registration = load_hashed_json(
            output_root / "training" / "worker_registration.json"
        )
        inference = load_hashed_json(
            output_root
            / "val_design"
            / "predictor_outputs_manifest.json"
        )
        artifacts = planner._artifacts(
            shape=run["shape_id"],
            expert=run["expert_id"],
            seed=int(run["pipeline_seed"]),
            target_mode=run["target_mode"],
            evidence_mode=run["hlt_evidence_mode"],
        )
        target_cache = (
            args.campaign_root
            / "inputs"
            / "target_caches"
            / f"coordinate_{artifacts['coordinate_index']:03d}"
            / run["shape_id"]
            / f"seed_{run['pipeline_seed']}"
            / "val_design"
            / "target_cache_manifest.json"
        )
        output = (
            output_root / "val_design" / "uncertainty_calibration.json"
        )
        rows.append(
            {
                "task_id": f"uncertainty_calibration:{len(rows)}",
                "argv": [
                    "python",
                    "scripts/calibrate_retb_uncertainty.py",
                    "--campaign-root",
                    str(args.campaign_root),
                    "--predictor-inference",
                    str(
                        output_root
                        / "val_design"
                        / "predictor_outputs_manifest.json"
                    ),
                    "--predictor-registration",
                    str(
                        output_root
                        / "training"
                        / "worker_registration.json"
                    ),
                    "--target-cache-manifest",
                    str(target_cache),
                    "--output",
                    str(output),
                ],
                "expected_outputs": [str(output)],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "production_graph": graph["content_hash"],
                    "predictor_run": run["content_hash"],
                    "predictor_registration": registration["content_hash"],
                    "predictor_inference": inference["content_hash"],
                    "target_cache_manifest": artifacts["manifests"][
                        "val_design"
                    ]["content_hash"],
                },
                "environment": {
                    "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                },
            }
        )
    predictor_manifest = load_hashed_json(
        task_manifest_path_for_graph(
            graph,
            node_id="predictor_training",
            campaign_root=args.campaign_root,
        )
    )
    factory_input = build_middle_dynamic_factory_input(
        target_node_id="uncertainty_calibration",
        producer_node_id="predictor_training",
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        producer_execution_sha256=predictor_manifest["content_hash"],
        rows=rows,
        coverage={
            "all_predeclared_rows_present": True,
            "scientific_metric_used_for_membership": False,
            "incomplete_wave_permitted": False,
            "predictor_phase_completion_sha256": artifact["content_hash"],
            "required_run_count": len(rows),
        },
        source=campaign["source"],
    )
    publish_middle_dynamic_factory_input(
        campaign_root=args.campaign_root, payload=factory_input
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
