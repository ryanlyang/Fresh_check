#!/usr/bin/env python3
"""Train every shared Stage-M component once for one matched scale seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_retb_predictor_campaign import (  # noqa: E402
    PredictorCampaignPlanner,
)
from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
    TOKEN_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    build_scale_component_index,
    build_scale_predictor_run,
    validate_scale_component_index,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    build_scale_refit_bundle,
    validate_scale_refit_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.target_coordinates import (  # noqa: E402
    target_slot_queries,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


SEED_REFIT_INDEX_CONTRACT = "retb_scale_seed_refit_index_v2"
REFIT_PHASE_COMPLETION_CONTRACT = "retb_scale_refit_phase_completion_v2"
REFIT_PHASES = (
    "normalizers", "teachers", "offline_experts", "targets",
    "native", "native_fusion", "predictors", "calibrations", "complete",
)
TARGET_CONFIGURATION_CONTRACT = (
    "retb_scale_target_coordinate_configuration_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_marker_filename(phase: str, component_id: str | None) -> str:
    if component_id is None:
        return f"{phase}.json"
    suffix = hashlib.sha256(component_id.encode("utf-8")).hexdigest()[:16]
    return f"{phase}__{suffix}.json"


def _run(arguments: Sequence[str], *, expected: Sequence[Path]) -> None:
    outputs = [Path(path) for path in expected]
    if outputs and all(path.is_file() and not path.is_symlink() for path in outputs):
        for path in outputs:
            if path.suffix == ".json":
                load_hashed_json(path)
        return
    completed = subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"scale seed component failed ({completed.returncode}): "
            f"{' '.join(map(str, arguments))}"
        )
    if any(not path.is_file() or path.is_symlink() for path in outputs):
        raise RuntimeError("scale seed component omitted an expected output")


def _named_shape(k: int, d: int) -> str:
    matches = [
        name
        for name, row in TOKEN_SHAPES.items()
        if int(row["K"]) == int(k) and int(row["D"]) == int(d)
    ]
    if len(matches) != 1:
        raise ValueError(f"scale target shape K={k},D={d} is unregistered")
    return matches[0]


def _source_shape(checkpoint: Path, mode: str) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = (
        payload.get("model_state_dict", payload)
        if mode == "T0_PURE"
        else payload.get("offline_target_state_dict")
    )
    if not isinstance(state, Mapping):
        raise ValueError("locked target state is absent")
    rows = [
        value
        for name, value in state.items()
        if str(name).endswith("tokenizer.slot_queries")
        and isinstance(value, torch.Tensor)
    ]
    if len(rows) != 1 or rows[0].ndim != 2:
        raise ValueError("locked target source shape is ambiguous")
    return _named_shape(int(rows[0].shape[0]), int(rows[0].shape[1]))


def _stage_c_expert_run(
    registry: Mapping[str, Any], *, shape: str, expert: str, seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in registry["expert_confirmation_rows"]
        if int(row["seed"]) == seed
        and row["configuration"]["shape_id"] == shape
        and row["configuration"]["expert_id"] == expert
    ]
    if len(rows) != 1:
        raise ValueError("scale offline expert parent is absent or duplicated")
    return rows[0]


def _stage_c_fusion_run(
    registry: Mapping[str, Any], *, seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in registry["canonical_fusion_rows"]
        if int(row["seed"]) == seed
        and row["configuration"]["fusion_variant"]
        == "F_TOKEN_TRANSFORMER"
    ]
    if not rows:
        raise ValueError("scale fusion recipe is absent")
    return min(rows, key=lambda row: row["run_id"])


def _stage_d_expert_run(
    registry: Mapping[str, Any],
    *,
    role: str,
    expert: str,
    seed: int,
) -> Mapping[str, Any]:
    rows = [
        row
        for row in registry["bridge_parent_expert_rows"]
        if int(row["seed"]) == seed
        and row["configuration"]["shape_id"] == role
        and row["configuration"]["expert_id"] == expert
        and row["configuration"]["mode"] == "HE_OFFLINE_INIT"
        and row["configuration"]["realization_policy"] == "R_MULTI"
    ]
    if len(rows) != 1:
        raise ValueError("scale native expert recipe is absent or duplicated")
    return rows[0]


def _stage_d_fusion_run(
    registry: Mapping[str, Any], *, role: str
) -> Mapping[str, Any]:
    rows = [
        row
        for row in registry["native_fusion_rows"]
        if row["configuration"]["shape_id"] == role
        and row["configuration"]["fusion_variant"] == "HF_NATIVE"
    ]
    if len(rows) != 1:
        raise ValueError("scale native fusion recipe is absent or duplicated")
    return rows[0]


def _hlt_cache(root: Path, split: str, replica: int) -> Path:
    policy = "R_MULTI" if split == "scale_train" else "R_FIXED"
    return (
        root
        / "inputs"
        / "hlt_v3"
        / split
        / f"replica_{replica}"
        / policy
        / "D_NOMINAL"
    )


def _target_registration_path(
    root: Path,
    *,
    shape: str,
    expert: str,
    seed: int,
    mode: str,
    checkpoint: Path,
) -> Path:
    if mode == "T0_PURE":
        return (
            root
            / "selection"
            / "stage_e_parents"
            / shape
            / expert
            / f"seed_{seed}"
            / "t0_registration.json"
        )
    return checkpoint.parent / "checkpoint_registration.json"


def _base_predictor_run(
    root: Path,
    *,
    lock: Mapping[str, Any],
    expert: str,
    seed: int,
) -> tuple[Path, Mapping[str, Any]]:
    configuration = json.loads(
        (
            root
            / "selection"
            / "predictor_bundle"
            / "inputs"
            / "selector_configuration.json"
        ).read_text("utf-8")
    )
    candidate = lock["selected_candidate_descriptors"][expert][
        "candidate_id"
    ]
    value = configuration["materialized_run_paths"][candidate]
    path = Path(value.get(str(seed), value.get(seed)))
    return path, load_hashed_json(path)


def _build_role_record(
    *,
    root: Path,
    role: str,
    seed: int,
    planner: PredictorCampaignPlanner,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    lock = load_hashed_json(
        root
        / "selection"
        / "predictor_bundle"
        / "carried"
        / f"{role}.json"
    )
    coordinate_name, shape = str(lock["coordinate_id"]).split(":", 1)
    coordinate_index = int(coordinate_name.rsplit("_", 1)[1])
    coordinate = selection["locked_coordinate_systems"][coordinate_index]
    if (
        list(coordinate["target_tuple"])
        != [lock["target_modes"][expert] for expert in EXPERT_ORDER]
    ):
        raise ValueError("carried predictor coordinate lineage differs")
    artifacts = {}
    for expert in EXPERT_ORDER:
        mode = lock["target_modes"][expert]
        row = planner._artifacts(
            shape=shape,
            expert=expert,
            seed=seed,
            target_mode=mode,
            evidence_mode=lock["selected_candidate_descriptors"][expert][
                "configuration"
            ].get("hlt_evidence_mode", "selected_native_HLT_evidence"),
        )
        exact_spec = load_hashed_json(
            root
            / "inputs"
            / "target_caches"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{seed}"
            / "model_train"
            / "target_cache_specification.json"
        )
        exact_manifest = load_hashed_json(
            root
            / "inputs"
            / "target_caches"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{seed}"
            / "model_train"
            / "target_cache_manifest.json"
        )
        row["descriptor"] = exact_manifest["target_descriptors"][expert]
        row["specification"] = exact_spec
        row["source_shape"] = _source_shape(row["checkpoint"], mode)
        artifacts[expert] = row
    return {
        "role": role,
        "shape": shape,
        "coordinate_index": coordinate_index,
        "coordinate": coordinate,
        "lock": lock,
        "artifacts": artifacts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--locked-scale-shortlist", required=True, type=Path
    )
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-after", choices=REFIT_PHASES, default="complete")
    parser.add_argument("--component-id")
    args = parser.parse_args(argv)
    if args.pipeline_seed not in {101, 202, 303}:
        raise ValueError("scale refit seed differs")
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    if shortlist.get("source") != campaign.get("source"):
        raise ValueError("scale shortlist source differs")
    for resource in ("scale_train", "val_stop", "val_design"):
        authorize_dataset_access(
            worker_role=(
                "scale_training_worker"
                if resource != "val_design"
                else "design_worker"
            ),
            requested_resource=resource,
        )
    output = args.output_dir.resolve()
    component_phases = {
        "offline_experts", "targets", "native", "native_fusion",
        "predictors", "calibrations",
    }
    if (args.stop_after in component_phases) != (args.component_id is not None):
        raise ValueError(
            "bounded scale-refit component phases require exactly one component ID"
        )
    def finish_phase(phase: str, artifacts: Mapping[str, str]) -> int:
        marker = bind_source(
            with_content_hash(
                {
                    "contract": REFIT_PHASE_COMPLETION_CONTRACT,
                    "schema_version": 2,
                    "phase": phase,
                    "component_id": args.component_id,
                    "pipeline_seed": args.pipeline_seed,
                    "locked_scale_shortlist_sha256": shortlist["content_hash"],
                    "artifact_hashes": dict(sorted(artifacts.items())),
                    "bounded_continuation": phase != "complete",
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        path = output / "phase_completions" / _phase_marker_filename(
            phase, args.component_id
        )
        publication = write_immutable_json(path, marker)
        print(json.dumps(publication, indent=2, sort_keys=True))
        return 0
    completed_path = output / "scale_seed_refit_index.json"
    if completed_path.is_file():
        existing = load_hashed_json(
            completed_path, expected_contract=SEED_REFIT_INDEX_CONTRACT
        )
        if (
            existing.get("source") != campaign.get("source")
            or int(existing["pipeline_seed"]) != args.pipeline_seed
            or existing["locked_scale_shortlist_sha256"]
            != shortlist["content_hash"]
        ):
            raise ValueError("reusable scale seed refit differs")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0

    graph = load_hashed_json(root / "job_ledgers" / "production_graph.json")
    planner = PredictorCampaignPlanner(
        root=root, campaign=campaign, graph=graph
    )
    coordinate_selection = load_hashed_json(
        root / "selection" / "locked_bridge_coordinates.json"
    )
    definitions = shortlist["locked_graph_definitions"]
    roles = sorted(
        {
            definitions[graph_id]["configuration"][
                "source_carried_shape_role"
            ]
            for graph_id in shortlist.get(
                "SCALE_TRAINING_GRAPHS", shortlist["SCALE_SHORTLIST"]
            )
        }
    )
    role_records = {
        role: _build_role_record(
            root=root,
            role=role,
            seed=args.pipeline_seed,
            planner=planner,
            selection=coordinate_selection,
        )
        for role in roles
    }
    selected_role: str | None = None
    selected_expert: str | None = None
    if args.component_id is not None:
        if args.stop_after in {"targets", "native_fusion"}:
            selected_role = args.component_id
        elif args.stop_after in {"native", "predictors", "calibrations"}:
            pieces = args.component_id.split("@", 1)
            if len(pieces) != 2:
                raise ValueError("scale-refit role/expert component is malformed")
            selected_role, selected_expert = pieces
            if selected_expert not in EXPERT_ORDER:
                raise ValueError("scale-refit component expert is unregistered")
        if selected_role is not None and selected_role not in role_records:
            raise ValueError("scale-refit component role is unregistered")

    normalization_root = output / "normalization"
    normalizer_bundle_path = (
        normalization_root / "scale_normalizer_bundle.json"
    )
    _run(
        [
            "scripts/fit_retb_normalizers.py",
            "--campaign-root",
            str(root),
            "--population",
            "scale",
            "--output-root",
            str(normalization_root),
            "--output",
            str(normalizer_bundle_path),
        ],
        expected=[normalizer_bundle_path],
    )
    normalizer_bundle = load_hashed_json(normalizer_bundle_path)
    if args.stop_after == "normalizers":
        return finish_phase(
            "normalizers", {"normalizer_bundle": normalizer_bundle["content_hash"]}
        )
    offline_relation = (
        normalization_root / "offline_scale" / "relation.json"
    )
    offline_region = (
        normalization_root / "offline_scale" / "region.json"
    )
    hlt_relation = (
        normalization_root / "hlt_shared_scale" / "relation.json"
    )
    hlt_region = normalization_root / "hlt_shared_scale" / "region.json"

    teachers_root = output / "teachers"
    _run(
        [
            "scripts/train_retb_scale_teachers.py",
            "--campaign-root",
            str(root),
            "--pipeline-seed",
            str(args.pipeline_seed),
            "--output-dir",
            str(teachers_root),
            "--device",
            args.device,
        ],
        expected=[teachers_root / "scale_teacher_bundle.json"],
    )
    teacher_bundle = load_hashed_json(
        teachers_root / "scale_teacher_bundle.json"
    )
    if args.stop_after == "teachers":
        return finish_phase(
            "teachers",
            {
                "normalizer_bundle": normalizer_bundle["content_hash"],
                "scale_teacher_bundle": teacher_bundle["content_hash"],
            },
        )

    stage_c = load_hashed_json(root / "registry" / "retb_stage_c_runs.json")
    required_pairs = set()
    for record in role_records.values():
        for expert in EXPERT_ORDER:
            required_pairs.add(
                (expert, record["artifacts"][expert]["source_shape"])
            )
            K, D = record["lock"]["allocation"][expert]
            required_pairs.add((expert, _named_shape(int(K), int(D))))
    offline_outputs = {}
    for expert, shape in sorted(required_pairs):
        if (
            args.stop_after == "offline_experts"
            and args.component_id != f"{expert}@{shape}"
        ):
            continue
        row = _stage_c_expert_run(
            stage_c, shape=shape, expert=expert, seed=args.pipeline_seed
        )
        target = output / "offline_experts" / shape / expert
        arguments = [
            "scripts/train_retb_offline_expert.py",
            "--campaign-root",
            str(root),
            "--run-id",
            row["run_id"],
            "--registry-stage",
            "C",
            "--training-role",
            "scale_train",
            "--train-npz",
            str(
                root
                / "inputs"
                / "offline"
                / "scale_train"
                / "offline_inputs.npz"
            ),
            "--val-stop-npz",
            str(
                root
                / "inputs"
                / "offline"
                / "val_stop"
                / "offline_inputs.npz"
            ),
            "--relation-normalization",
            str(offline_relation),
            "--region-normalization",
            str(offline_region),
            "--region-tree-root",
            str(root / "inputs" / "region_tree" / "offline"),
            "--output-dir",
            str(target),
            "--device",
            args.device,
        ]
        _run(
            arguments,
            expected=[
                target / "checkpoint_registration.json",
                target / "best_model_val.pt",
            ],
        )
        registration = load_hashed_json(
            target / "checkpoint_registration.json"
        )
        offline_outputs[(expert, shape)] = {
            "checkpoint": str((target / "best_model_val.pt").resolve()),
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "registration": str(
                (target / "checkpoint_registration.json").resolve()
            ),
            "registration_sha256": registration["content_hash"],
        }

    if args.stop_after == "offline_experts":
        return finish_phase(
            "offline_experts",
            {
                f"{expert}:{shape}": row["registration_sha256"]
                for (expert, shape), row in offline_outputs.items()
            },
        )

    role_results = {}
    phase_role_hashes: dict[str, str] = {}
    stage_d = load_hashed_json(root / "registry" / "retb_stage_d_runs.json")
    predictor_lock = load_hashed_json(
        root / "selection" / "predictor_bundle" / "predictor_bundle_lock.json"
    )
    for role, record in role_records.items():
        if selected_role is not None and role != selected_role:
            continue
        role_root = output / "roles" / role
        target_config = bind_source(
            with_content_hash(
                {
                    "contract": TARGET_CONFIGURATION_CONTRACT,
                    "schema_version": 1,
                    "shape_role": role,
                    "resolved_shape_id": record["shape"],
                    "pipeline_seed": args.pipeline_seed,
                    "coordinate_selection_sha256": coordinate_selection[
                        "content_hash"
                    ],
                    "coordinate_contract_sha256": record["coordinate"][
                        "coordinate_contract_sha256"
                    ],
                    "target_cache_namespace": record["coordinate"][
                        "target_cache_namespace"
                    ],
                    "locked_coordinate_normalizer_set_sha256": record[
                        "artifacts"
                    ][EXPERT_ORDER[0]]["specification"][
                        "normalizer_set_sha256"
                    ],
                    "offline_relation_normalizer": str(
                        offline_relation.resolve()
                    ),
                    "offline_region_normalizer": str(
                        offline_region.resolve()
                    ),
                    "target_modes": {
                        expert: record["lock"]["target_modes"][expert]
                        for expert in EXPERT_ORDER
                    },
                    "scale_experts": {
                        expert: offline_outputs[
                            (
                                expert,
                                record["artifacts"][expert][
                                    "source_shape"
                                ],
                            )
                        ]
                        for expert in EXPERT_ORDER
                    },
                    "locked_targets": {
                        expert: {
                            "checkpoint": str(
                                record["artifacts"][expert][
                                    "checkpoint"
                                ].resolve()
                            ),
                            "checkpoint_sha256": _sha256(
                                record["artifacts"][expert]["checkpoint"]
                            ),
                            "registration": str(
                                _target_registration_path(
                                    root,
                                    shape=record["shape"],
                                    expert=expert,
                                    seed=args.pipeline_seed,
                                    mode=record["lock"]["target_modes"][
                                        expert
                                    ],
                                    checkpoint=record["artifacts"][expert][
                                        "checkpoint"
                                    ],
                                ).resolve()
                            ),
                            "registration_sha256": record["artifacts"][
                                expert
                            ]["descriptor"]["registration_sha256"],
                            "descriptor": record["artifacts"][expert][
                                "descriptor"
                            ],
                        }
                        for expert in EXPERT_ORDER
                    },
                    "locked_target_weights_reselected": False,
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        target_config_path = role_root / "target_configuration.json"
        write_immutable_json(target_config_path, target_config)
        coordinate_roots = {}
        for split in ("scale_train", "val_stop", "val_design"):
            coordinate_root = role_root / "coordinates" / split
            _run(
                [
                    "scripts/prepare_retb_scale_target_coordinates.py",
                    "--campaign-root",
                    str(root),
                    "--configuration",
                    str(target_config_path),
                    "--split",
                    split,
                    "--output-dir",
                    str(coordinate_root),
                    "--device",
                    args.device,
                ],
                expected=[
                    coordinate_root
                    / "scale_target_coordinate_index.json",
                    coordinate_root / f"{split}_frozen_tokens.json",
                ],
            )
            coordinate_roots[split] = coordinate_root

        fusion_row = _stage_c_fusion_run(
            stage_c, seed=args.pipeline_seed
        )
        fusion_root = role_root / "offline_fusion"
        _run(
            [
                "scripts/train_retb_offline_fusion.py",
                "--campaign-root",
                str(root),
                "--run-id",
                fusion_row["run_id"],
                "--training-role",
                "scale_train",
                "--model-train-cache",
                str(
                    coordinate_roots["scale_train"]
                    / "scale_train_frozen_tokens.json"
                ),
                "--val-stop-cache",
                str(
                    coordinate_roots["val_stop"]
                    / "val_stop_frozen_tokens.json"
                ),
                "--val-design-cache",
                str(
                    coordinate_roots["val_design"]
                    / "val_design_frozen_tokens.json"
                ),
                "--output-dir",
                str(fusion_root),
                "--device",
                args.device,
            ],
            expected=[
                fusion_root / "fusion_registration.json",
                fusion_root / "best_model_val.pt",
            ],
        )
        fusion_registration = load_hashed_json(
            fusion_root / "fusion_registration.json"
        )
        target_cache_root = role_root / "target_cache"
        for split in ("scale_train", "val_stop", "val_design"):
            split_output = target_cache_root / split
            _run(
                [
                    "scripts/finalize_retb_scale_target_cache.py",
                    "--campaign-root",
                    str(root),
                    "--configuration",
                    str(target_config_path),
                    "--coordinate-index",
                    str(
                        coordinate_roots[split]
                        / "scale_target_coordinate_index.json"
                    ),
                    "--coordinate-cache",
                    str(
                        coordinate_roots[split]
                        / f"{split}_frozen_tokens.json"
                    ),
                    "--fusion-registration",
                    str(fusion_root / "fusion_registration.json"),
                    "--fusion-checkpoint",
                    str(fusion_root / "best_model_val.pt"),
                    "--output-dir",
                    str(split_output),
                ],
                expected=[
                    split_output / "target_cache_manifest.json",
                    split_output / "identity_manifest.json",
                ],
            )
        scale_manifest = load_hashed_json(
            target_cache_root
            / "scale_train"
            / "target_cache_manifest.json"
        )
        scale_spec = load_hashed_json(
            target_cache_root
            / "scale_train"
            / "target_cache_specification.json"
        )
        target_normalizers = role_root / "target_normalizers"
        _run(
            [
                "scripts/fit_retb_target_normalizers.py",
                "--campaign-root",
                str(root),
                "--target-cache-manifest",
                str(
                    target_cache_root
                    / "scale_train"
                    / "target_cache_manifest.json"
                ),
                "--pipeline-seed",
                str(args.pipeline_seed),
                "--training-role",
                "scale_train",
                "--specification-sha256",
                scale_spec["content_hash"],
                "--output-dir",
                str(target_normalizers),
            ],
            expected=[
                target_normalizers / "target_normalizer_set.json"
            ],
        )
        target_normalizer_set = load_hashed_json(
            target_normalizers / "target_normalizer_set.json"
        )
        phase_role_hashes[f"targets:{role}"] = canonical_sha256(
            {
                "target_normalizer_set": target_normalizer_set["content_hash"],
                "offline_fusion": fusion_registration["content_hash"],
                "scale_target_cache": scale_manifest["content_hash"],
            }
        )
        if args.stop_after == "targets":
            return finish_phase(
                "targets", {role: phase_role_hashes[f"targets:{role}"]}
            )

        native_outputs = {}
        for expert in EXPERT_ORDER:
            if args.stop_after == "native" and expert != selected_expert:
                continue
            native_row = _stage_d_expert_run(
                stage_d,
                role=role,
                expert=expert,
                seed=args.pipeline_seed,
            )
            K, D = record["lock"]["allocation"][expert]
            offline = offline_outputs[
                (expert, _named_shape(int(K), int(D)))
            ]
            native_root = role_root / "native_hlt" / expert
            native_args = [
                "scripts/train_retb_native_hlt_expert.py",
                "--campaign-root",
                str(root),
                "--run-id",
                native_row["run_id"],
                "--training-role",
                "scale_train",
                "--train-labels",
                str(
                    root
                    / "inputs"
                    / "offline"
                    / "scale_train"
                    / "offline_inputs.npz"
                ),
                "--val-stop-labels",
                str(
                    root
                    / "inputs"
                    / "offline"
                    / "val_stop"
                    / "offline_inputs.npz"
                ),
                "--val-design-labels",
                str(
                    root
                    / "inputs"
                    / "offline"
                    / "val_design"
                    / "offline_inputs.npz"
                ),
                "--uniform-shapes",
                str(root / "selection" / "retb_offline_shapes.json"),
                "--heterogeneous-shapes",
                str(
                    root / "selection" / "retb_heterogeneous_shapes.json"
                ),
                "--offline-registration",
                offline["registration"],
                "--offline-checkpoint",
                offline["checkpoint"],
                "--relation-normalization",
                str(hlt_relation),
                "--region-normalization",
                str(hlt_region),
                "--region-tree-root",
                str(root / "inputs" / "region_tree" / "hlt"),
                "--output-dir",
                str(native_root),
                "--device",
                args.device,
            ]
            for replica in range(4):
                native_args.extend(
                    [
                        "--train-cache",
                        f"{replica}={_hlt_cache(root, 'scale_train', replica)}",
                    ]
                )
            native_args.extend(
                [
                    "--val-stop-cache",
                    f"0={_hlt_cache(root, 'val_stop', 0)}",
                    "--val-design-cache",
                    f"0={_hlt_cache(root, 'val_design', 0)}",
                ]
            )
            _run(
                native_args,
                expected=[
                    native_root / "checkpoint_registration.json",
                    native_root / "native_output_manifest.json",
                    native_root / "best_model_val.pt",
                ],
            )
            native_registration = load_hashed_json(
                native_root / "checkpoint_registration.json"
            )
            native_outputs[expert] = {
                "output_root": str(native_root.resolve()),
                "checkpoint_sha256": native_registration[
                    "checkpoint_sha256"
                ],
                "registration_sha256": native_registration[
                    "content_hash"
                ],
            }

        if args.stop_after == "native":
            return finish_phase(
                "native",
                {
                    f"{role}@{selected_expert}": native_outputs[
                        str(selected_expert)
                    ]["registration_sha256"]
                },
            )

        evidence = {}
        for split in ("scale_train", "val_stop", "val_design"):
            evidence_root = role_root / "predictor_evidence" / split
            evidence_args = [
                "scripts/build_retb_scale_predictor_evidence.py",
                "--campaign-root",
                str(root),
                "--split",
                split,
                "--pipeline-seed",
                str(args.pipeline_seed),
                "--shape-role",
                role,
                "--output",
                str(evidence_root / "evidence.npz"),
                "--metadata-output",
                str(evidence_root / "evidence.json"),
            ]
            for expert in EXPERT_ORDER:
                evidence_args.extend(
                    [
                        "--native-output",
                        (
                            f"{expert}="
                            f"{native_outputs[expert]['output_root']}/"
                            "native_output_manifest.json"
                        ),
                    ]
                )
            _run(
                evidence_args,
                expected=[
                    evidence_root / "evidence.npz",
                    evidence_root / "evidence.json",
                ],
            )
            evidence[split] = {
                "npz": evidence_root / "evidence.npz",
                "manifest": load_hashed_json(
                    evidence_root / "evidence.json"
                ),
            }

        native_cache_roots = {}
        for split in ("scale_train", "val_stop"):
            replicas = range(4) if split == "scale_train" else (0,)
            cache_root = role_root / "native_fusion_cache" / split
            cache_args = [
                "scripts/build_retb_native_hlt_fusion_cache.py",
                "--campaign-root",
                str(root),
                "--split",
                split,
                "--pipeline-seed",
                str(args.pipeline_seed),
                "--shape-id",
                role,
                "--realization-policy",
                "R_MULTI",
                "--identity-manifest",
                str(target_cache_root / split / "identity_manifest.json"),
                "--label-manifest",
                str(target_cache_root / split / "identity_manifest.json"),
                "--output-dir",
                str(cache_root),
            ]
            for expert in EXPERT_ORDER:
                native_root = Path(native_outputs[expert]["output_root"])
                cache_args.extend(
                    [
                        "--expert-registration",
                        f"{expert}={native_root / 'checkpoint_registration.json'}",
                        "--expert-output-manifest",
                        f"{expert}={native_root / 'native_output_manifest.json'}",
                    ]
                )
                manifest = load_hashed_json(
                    native_root / "native_output_manifest.json"
                )
                for replica in replicas:
                    cache_args.extend(
                        [
                            "--expert-output",
                            (
                                f"{expert}:{replica}="
                                f"{native_root / manifest['files'][f'{split}_replica_{replica}']['relative_path']}"
                            ),
                        ]
                    )
            for replica in replicas:
                cache_args.extend(
                    [
                        "--hlt-cache-manifest",
                        (
                            f"{replica}="
                            f"{_hlt_cache(root, split, replica) / 'hlt_v3_metadata.json'}"
                        ),
                    ]
                )
            _run(
                cache_args,
                expected=[cache_root / "native_fusion_cache.json"],
            )
            native_cache_roots[split] = cache_root
        native_fusion_row = _stage_d_fusion_run(stage_d, role=role)
        native_fusion_root = role_root / "native_fusion"
        _run(
            [
                "scripts/train_retb_native_hlt_fusion.py",
                "--campaign-root",
                str(root),
                "--run-id",
                native_fusion_row["run_id"],
                "--pipeline-seed-override",
                str(args.pipeline_seed),
                "--training-role",
                "scale_train",
                "--model-train-cache",
                str(
                    native_cache_roots["scale_train"]
                    / "native_fusion_cache.json"
                ),
                "--val-stop-cache",
                str(
                    native_cache_roots["val_stop"]
                    / "native_fusion_cache.json"
                ),
                "--output-dir",
                str(native_fusion_root),
                "--device",
                args.device,
            ],
            expected=[native_fusion_root / "fusion_registration.json"],
        )
        native_fusion_registration = load_hashed_json(
            native_fusion_root / "fusion_registration.json"
        )
        phase_role_hashes[f"native:{role}"] = canonical_sha256(
            {
                "experts": native_outputs,
                "fusion": native_fusion_registration["content_hash"],
            }
        )
        if args.stop_after == "native_fusion":
            return finish_phase(
                "native_fusion",
                {role: phase_role_hashes[f"native:{role}"]},
            )

        predictors, calibrations = {}, {}
        scale_target_index = load_hashed_json(
            coordinate_roots["scale_train"]
            / "scale_target_coordinate_index.json"
        )
        step9 = load_hashed_json(
            root / "registry" / "retb_step9_predictor_bundle.json"
        )
        for expert in EXPERT_ORDER:
            if (
                args.stop_after in {"predictors", "calibrations"}
                and expert != selected_expert
            ):
                continue
            base_path, base_run = _base_predictor_run(
                root,
                lock=record["lock"],
                expert=expert,
                seed=args.pipeline_seed,
            )
            target_row = scale_target_index["target_checkpoints"][expert]
            normalizer = load_hashed_json(
                target_normalizers
                / f"target_normalizer_{expert}.json"
            )
            parent_hashes = {
                "scale_train_target_cache": scale_manifest[
                    "content_hash"
                ],
                "val_stop_target_cache": load_hashed_json(
                    target_cache_root
                    / "val_stop"
                    / "target_cache_manifest.json"
                )["content_hash"],
                "val_design_target_cache": load_hashed_json(
                    target_cache_root
                    / "val_design"
                    / "target_cache_manifest.json"
                )["content_hash"],
                "target_normalizer": normalizer["content_hash"],
                "slot_queries": target_row["slot_query_sha256"],
                "offline_target_checkpoint": target_row["sha256"],
                "offline_fusion": fusion_registration[
                    "checkpoint_sha256"
                ],
                "native_hlt_expert": native_outputs[expert][
                    "registration_sha256"
                ],
                "scale_train_hlt_evidence_cache": evidence[
                    "scale_train"
                ]["manifest"]["content_hash"],
                "val_stop_hlt_evidence_cache": evidence["val_stop"][
                    "manifest"
                ]["content_hash"],
                "val_design_hlt_evidence_cache": evidence["val_design"][
                    "manifest"
                ]["content_hash"],
                "scale_train_identity_manifest": load_hashed_json(
                    target_cache_root
                    / "scale_train"
                    / "identity_manifest.json"
                )["content_hash"],
                "val_stop_identity_manifest": load_hashed_json(
                    target_cache_root
                    / "val_stop"
                    / "identity_manifest.json"
                )["content_hash"],
                "val_design_identity_manifest": load_hashed_json(
                    target_cache_root
                    / "val_design"
                    / "identity_manifest.json"
                )["content_hash"],
                "step9_bundle": step9["content_hash"],
            }
            scale_run = bind_source(
                build_scale_predictor_run(
                    base_run=base_run,
                    scale_parent_hashes=parent_hashes,
                ),
                source_snapshot=source_snapshot(REPO_ROOT),
            )
            predictor_root = role_root / "predictors" / expert
            run_path = predictor_root / "run.json"
            write_immutable_json(run_path, scale_run)
            _run(
                [
                    "scripts/execute_retb_predictor_training_row.py",
                    "--campaign-root",
                    str(root),
                    "--run",
                    str(run_path),
                    "--training-role",
                    "scale_train",
                    "--model-train-target-cache",
                    str(
                        target_cache_root
                        / "scale_train"
                        / "target_cache_manifest.json"
                    ),
                    "--model-train-evidence",
                    str(evidence["scale_train"]["npz"]),
                    "--val-stop-target-cache",
                    str(
                        target_cache_root
                        / "val_stop"
                        / "target_cache_manifest.json"
                    ),
                    "--val-stop-evidence",
                    str(evidence["val_stop"]["npz"]),
                    "--val-design-target-cache",
                    str(
                        target_cache_root
                        / "val_design"
                        / "target_cache_manifest.json"
                    ),
                    "--val-design-evidence",
                    str(evidence["val_design"]["npz"]),
                    "--target-normalizer",
                    str(
                        target_normalizers
                        / f"target_normalizer_{expert}.json"
                    ),
                    "--target-checkpoint",
                    target_row["path"],
                    "--fusion-checkpoint",
                    str(fusion_root / "best_model_val.pt"),
                    "--output-dir",
                    str(predictor_root),
                    "--device",
                    args.device,
                ],
                expected=[
                    predictor_root / "training" / "registration.json",
                    predictor_root / "training" / "best_model_val.pt",
                    predictor_root
                    / "val_design"
                    / "predictor_outputs_manifest.json",
                ],
            )
            predictor_registration = load_hashed_json(
                predictor_root / "training" / "registration.json"
            )
            predictors[expert] = {
                "run_path": str(run_path.resolve()),
                "output_root": str(predictor_root.resolve()),
                "checkpoint_sha256": predictor_registration[
                    "checkpoint_sha256"
                ],
                "registration_sha256": predictor_registration[
                    "content_hash"
                ],
            }
            if args.stop_after == "predictors":
                return finish_phase(
                    "predictors",
                    {
                        f"{role}@{expert}": predictor_registration[
                            "content_hash"
                        ]
                    },
                )
            calibration_path = (
                role_root / "uncertainty" / f"{expert}.json"
            )
            _run(
                [
                    "scripts/calibrate_retb_uncertainty.py",
                    "--campaign-root",
                    str(root),
                    "--predictor-inference",
                    str(
                        predictor_root
                        / "val_design"
                        / "predictor_outputs_manifest.json"
                    ),
                    "--predictor-registration",
                    str(
                        predictor_root / "training" / "registration.json"
                    ),
                    "--target-cache-manifest",
                    str(
                        target_cache_root
                        / "val_design"
                        / "target_cache_manifest.json"
                    ),
                    "--output",
                    str(calibration_path),
                ],
                expected=[calibration_path],
            )
            calibrations[expert] = str(calibration_path.resolve())

            if args.stop_after == "calibrations":
                return finish_phase(
                    "calibrations",
                    {
                        f"{role}@{expert}": load_hashed_json(
                            calibration_path
                        )["content_hash"]
                    },
                )

        phase_role_hashes[f"predictors:{role}"] = canonical_sha256(predictors)
        if args.stop_after == "predictors":
            continue
        phase_role_hashes[f"calibrations:{role}"] = canonical_sha256(
            {
                expert: load_hashed_json(path)["content_hash"]
                for expert, path in calibrations.items()
            }
        )

        role_results[role] = {
            "target_cache_root": str(target_cache_root.resolve()),
            "target_normalizer_root": str(target_normalizers.resolve()),
            "target_normalizer_set_sha256": target_normalizer_set[
                "content_hash"
            ],
            "offline_fusion_checkpoint": str(
                (fusion_root / "best_model_val.pt").resolve()
            ),
            "offline_fusion_registration_sha256": fusion_registration[
                "content_hash"
            ],
            "offline_fusion_checkpoint_sha256": fusion_registration[
                "checkpoint_sha256"
            ],
            "native_hlt_experts": native_outputs,
            "native_hlt_fusion_registration_sha256": (
                native_fusion_registration["content_hash"]
            ),
            "predictors": predictors,
            "uncertainty_calibrations": calibrations,
            "target_checkpoints": scale_target_index[
                "target_checkpoints"
            ],
            "coordinate_configuration_sha256": target_config[
                "content_hash"
            ],
        }

    if args.stop_after in {
        "targets", "native", "native_fusion", "predictors", "calibrations"
    }:
        prefix = args.stop_after
        owned = {
            key: value
            for key, value in phase_role_hashes.items()
            if key.startswith(f"{prefix}:")
        }
        if set(key.split(":", 1)[1] for key in owned) != set(roles):
            raise ValueError(f"scale refit {prefix} role coverage differs")
        return finish_phase(prefix, owned)

    component_indexes = {}
    refit_bundles = {}
    profile = load_hashed_json(root / "inputs" / "hlt_v3_profile.json")
    base_normalizers = load_hashed_json(
        root / "inputs" / "normalization" / "stage_a_normalizer_bundle.json"
    )
    joint_lock = load_hashed_json(
        root / "selection" / "joint" / "joint_campaign_lock.json"
    )
    for graph_id in shortlist.get(
        "SCALE_TRAINING_GRAPHS", shortlist["SCALE_SHORTLIST"]
    ):
        definition = definitions[graph_id]
        role = definition["configuration"]["source_carried_shape_role"]
        record = role_records[role]
        result = role_results[role]
        base_j5 = joint_lock["carried_by_shape_role"][role][
            "selected_j5_by_seed"
        ][str(args.pipeline_seed)]["output_root"]
        base_j5_run = Path(base_j5) / "assets" / "run.json"
        target_shape_offline = {
            expert: {
                name: value
                for name, value in offline_outputs[
                    (
                        expert,
                        _named_shape(
                            int(record["lock"]["allocation"][expert][0]),
                            int(record["lock"]["allocation"][expert][1]),
                        ),
                    )
                ].items()
                if name
                in {"checkpoint", "checkpoint_sha256", "registration"}
            }
            for expert in EXPERT_ORDER
        }
        target_checkpoints = {
            expert: {
                "path": result["target_checkpoints"][expert]["path"],
                "sha256": result["target_checkpoints"][expert]["sha256"],
                "registration_sha256": result["target_checkpoints"][
                    expert
                ]["registration_sha256"],
                "target_mode": result["target_checkpoints"][expert][
                    "target_mode"
                ],
            }
            for expert in EXPERT_ORDER
        }
        component_hashes = {
            "scale_teacher_bundle": teacher_bundle["content_hash"],
            "normalizer_bundle": normalizer_bundle["content_hash"],
            "offline_fusion": result[
                "offline_fusion_registration_sha256"
            ],
            "target_normalizers": result[
                "target_normalizer_set_sha256"
            ],
            "native_hlt_fusion": result[
                "native_hlt_fusion_registration_sha256"
            ],
            "predictor_set": canonical_sha256(result["predictors"]),
            "uncertainty_set": canonical_sha256(
                {
                    expert: load_hashed_json(path)["content_hash"]
                    for expert, path in result[
                        "uncertainty_calibrations"
                    ].items()
                }
            ),
        }
        index = bind_source(
            build_scale_component_index(
                graph_id=graph_id,
                pipeline_seed=args.pipeline_seed,
                base_j5_run_path=base_j5_run,
                target_cache_root=result["target_cache_root"],
                target_normalizer_root=result[
                    "target_normalizer_root"
                ],
                scale_normalizer_bundle_sha256=normalizer_bundle[
                    "content_hash"
                ],
                hlt_relation_normalizer_sha256=normalizer_bundle[
                    "artifact_hashes"
                ]["shared_hlt_relation"],
                hlt_region_normalizer_sha256=normalizer_bundle[
                    "artifact_hashes"
                ]["shared_hlt_region"],
                degradation_profile_sha256=profile["content_hash"],
                offline_fusion_checkpoint=result[
                    "offline_fusion_checkpoint"
                ],
                offline_fusion_registration_sha256=result[
                    "offline_fusion_registration_sha256"
                ],
                offline_experts=target_shape_offline,
                native_hlt_experts=result["native_hlt_experts"],
                target_checkpoints=target_checkpoints,
                predictors=result["predictors"],
                uncertainty_calibrations=result[
                    "uncertainty_calibrations"
                ],
                selected_token_refiner=None,
                component_hashes=component_hashes,
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        validate_scale_component_index(index)
        index_path = output / "component_indexes" / f"{graph_id}.json"
        write_immutable_json(index_path, index)
        component_indexes[graph_id] = {
            "path": str(index_path.resolve()),
            "content_hash": index["content_hash"],
        }
        scale_identity = campaign["parent_artifact_hashes"][
            "scale_train_manifest"
        ]
        design_identity = load_hashed_json(
            Path(result["target_cache_root"])
            / "val_design"
            / "identity_manifest.json"
        )["content_hash"]
        uncertainty_sha = component_hashes["uncertainty_set"]
        normalizer_recipe = normalizer_bundle[
            "stage_a_contract_bundle_sha256"
        ]
        refit_values = {
            "offline_input": (
                normalizer_bundle["content_hash"],
                "offline_scale",
                normalizer_recipe,
                [],
            ),
            "offline_relation": (
                normalizer_bundle["artifact_hashes"]["offline_relation"],
                "offline_scale",
                normalizer_recipe,
                [],
            ),
            "offline_REGION": (
                normalizer_bundle["artifact_hashes"]["offline_region"],
                "offline_scale",
                normalizer_recipe,
                [],
            ),
            "shared_HLT_input": (
                normalizer_bundle["content_hash"],
                "shared_hlt_scale",
                normalizer_recipe,
                [0, 1, 2, 3],
            ),
            "shared_HLT_relation": (
                normalizer_bundle["artifact_hashes"][
                    "shared_hlt_relation"
                ],
                "shared_hlt_scale",
                normalizer_recipe,
                [0, 1, 2, 3],
            ),
            "shared_HLT_REGION": (
                normalizer_bundle["artifact_hashes"]["shared_hlt_region"],
                "shared_hlt_scale",
                normalizer_recipe,
                [0, 1, 2, 3],
            ),
            "target_token": (
                result["target_normalizer_set_sha256"],
                "scale_train_offline_targets",
                result["coordinate_configuration_sha256"],
                [],
            ),
            "uncertainty_calibrator": (
                uncertainty_sha,
                "val_design_label_free",
                predictor_lock["content_hash"],
                [],
            ),
        }
        refits = {
            name: {
                "artifact_sha256": artifact_sha,
                "population": population,
                "identity_manifest_sha256": (
                    design_identity
                    if name == "uncertainty_calibrator"
                    else scale_identity
                ),
                "recipe_sha256": recipe_sha,
                "fitted_values_sha256": artifact_sha,
                "labels_consumed": False,
                "replica_ids": replicas,
            }
            for name, (
                artifact_sha,
                population,
                recipe_sha,
                replicas,
            ) in refit_values.items()
        }
        refit = bind_source(
            build_scale_refit_bundle(
                graph_id=graph_id,
                pipeline_seed=args.pipeline_seed,
                locked_scale_shortlist_sha256=shortlist["content_hash"],
                scale_train_manifest_sha256=scale_identity,
                val_design_identity_manifest_sha256=design_identity,
                refits=refits,
                five_hundred_k_artifact_hashes=[
                    base_normalizers["content_hash"],
                    predictor_lock["content_hash"],
                ],
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        validate_scale_refit_bundle(refit)
        refit_path = output / "refit_bundles" / f"{graph_id}.json"
        write_immutable_json(refit_path, refit)
        refit_bundles[graph_id] = {
            "path": str(refit_path.resolve()),
            "content_hash": refit["content_hash"],
        }
    final = bind_source(
        with_content_hash(
            {
                "contract": SEED_REFIT_INDEX_CONTRACT,
                "schema_version": 2,
                "pipeline_seed": args.pipeline_seed,
                "locked_scale_shortlist_sha256": shortlist["content_hash"],
                "scale_train_manifest_sha256": campaign[
                    "parent_artifact_hashes"
                ]["scale_train_manifest"],
                "normalizer_bundle_sha256": normalizer_bundle[
                    "content_hash"
                ],
                "scale_teacher_bundle_sha256": teacher_bundle[
                    "content_hash"
                ],
                "roles": role_results,
                "component_indexes": component_indexes,
                "refit_bundles": refit_bundles,
                "all_shortlisted_graphs_covered": (
                    set(component_indexes)
                    == set(
                        shortlist.get(
                            "SCALE_TRAINING_GRAPHS",
                            shortlist["SCALE_SHORTLIST"],
                        )
                    )
                    == set(refit_bundles)
                ),
                "shared_components_trained_once_per_seed": True,
                "scientific_metric_used_for_membership": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(completed_path, final)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
