#!/usr/bin/env python3
"""Retrain, evaluate, and HLT-only export one locked Stage-J graph/seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_hosd_confirmation_row import (  # noqa: E402
    _capacity,
    _device,
    _inference,
    _predictions,
    _row_with_seed,
    _selected_loader_path,
)
from scripts.train_hosd_baseline import _dataset, _mapping  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    HOSDTrainingProtocol,
    build_auxiliary_model,
    auxiliary_model_flop_ledger,
    build_baseline_model,
    build_combination_loader_manifest,
    build_combination_model,
    combination_model_flop_ledger,
    build_feedback_model,
    feedback_model_flop_ledger,
    build_label_free_hlt_loader,
    build_scale_row_result,
    build_stage_d_loader_manifest,
    component_seed,
    export_deployable_graph,
    exact_trainable_parameter_count,
    initialize_feedback_from_auxiliary_checkpoint,
    load_and_validate_campaign,
    load_combination_loaders,
    NATIVE_RELATION_TARGET_CONTRACT,
    SCALE_NATIVE_RELATION_WAVE_CONTRACT,
)
from teacher_logit_reco.hlt_offline_structure_distillation.combination_runtime import (  # noqa: E402
    train_combination,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_GRAPH_WAVE_COMPLETION_CONTRACT,
    SCALE_ROW_RESULT_CONTRACT,
    SCALE_TARGET_COMPLETION_CONTRACT,
    SCALE_TRAINING_CHECKPOINT_CONTRACT,
    SCALE_TRAINING_COMPLETION_CONTRACT,
    SCALE_TRAINING_PREDICTION_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.deployment_runtime import (  # noqa: E402
    DEPLOYABLE_GRAPH_EXPORT_CONTRACT,
)
from teacher_logit_reco.hlt_offline_structure_distillation.scale_runtime import (  # noqa: E402
    build_scale_graph_wave_completion,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_c_training import (  # noqa: E402
    train_stage_c_baseline,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_data_factory import (  # noqa: E402
    load_stage_d_loaders_from_manifest,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_training import (  # noqa: E402
    train_stage_d_auxiliary,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    make_native_hlt_expert_loader,
)


def _labels(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"identities", "labels"}.issubset(payload.files):
            raise ValueError("scale graph labels lack identities/labels")
        identities = tuple(str(value) for value in payload["identities"].tolist())
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if (
        not identities
        or len(identities) != len(set(identities))
        or labels.shape != (len(identities),)
    ):
        raise ValueError("scale graph label population differs")
    return identities, labels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_lineage(*mappings):
    output = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key in output and output[key] != value:
                raise ValueError(f"scale graph lineage key differs: {key}")
            output[key] = value
    return output


def _required_scale_resources(definition):
    kind = str(definition["graph_kind"])
    target_ids = set()
    if kind in {"AUXILIARY", "FEEDBACK"}:
        target_ids.add(str(definition["row"]["target_id"]))
    elif kind == "COMBINATION":
        target_ids.update(
            str(member["target_id"])
            for member in definition["graph"]["members"]
        )
    baseline_id = str(definition.get("baseline_id", ""))
    return {
        "trees": "T_HLT_REGION_PAIR_8" in target_ids,
        "targets": bool(target_ids),
        "teacher_outputs": baseline_id in {
            "H_KD_LOGIT_O_BASE",
            "H_KD_LOGIT_O_FULLREL",
        },
        "native_relations": (
            baseline_id == "H_NATIVE_REL_AUX"
            or (
                kind == "COMBINATION"
                and definition["graph"].get("native_relation_auxiliary")
                is not None
            )
        ),
    }


def _teacher_logits_npz(root: Path, teacher_id: str, labels_path: Path) -> Path:
    from teacher_logit_reco.hlt_offline_structure_distillation import load_target_cache

    cache_root = root / "scale_up" / "teacher_outputs" / teacher_id
    spec = load_hashed_json(cache_root / "cache_spec.json")
    cache = load_target_cache(cache_root, cache_spec=spec)
    coordinate = f"T_OFFLINE_LOGITS_{teacher_id}"
    identities, _ = _labels(labels_path)
    lookup = {identity: index for index, identity in enumerate(cache.identities)}
    if set(identities) != set(lookup):
        raise ValueError("scale teacher logits and scale labels differ")
    logits = cache.values[coordinate][
        np.asarray([lookup[value] for value in identities], dtype=np.int64)
    ]
    output = root / "scale_up" / "teacher_outputs" / f"{teacher_id}_training_logits.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    def validate_existing() -> None:
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("scale teacher-logit destination is unsafe")
        with np.load(output, allow_pickle=False) as existing:
            if (
                tuple(str(value) for value in existing["identities"].tolist())
                != identities
                or not np.array_equal(existing["logits"], logits)
            ):
                raise FileExistsError("reusable scale teacher logits differ")

    if output.exists():
        validate_existing()
        return output
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            identities=np.asarray(identities, dtype="U"),
            logits=np.asarray(logits, dtype=np.float32),
        )
        try:
            os.link(temporary, output)
        except FileExistsError:
            validate_existing()
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _replace_statistics(target, definition):
    output = dict(target)
    for field in ("normalizer", "whitening"):
        if field in definition:
            output[field] = definition[field]
        elif field in output and field not in definition:
            output.pop(field)
    return output


def _scale_member_manifest(
    *,
    root: Path,
    base_path: Path,
    row,
    completion,
    scale_caches: dict[int, Path],
    scale_trees: dict[int, Path],
    scale_labels: Path,
    design_confirm_labels: Path,
    campaign,
    output: Path,
):
    base = load_hashed_json(base_path)
    if base.get("source") != campaign["source"]:
        raise ValueError("scale base loader source differs")
    parameterization = str(row["parameterization"])
    definitions_by_parameterization = completion["training_target_definitions"]
    if parameterization not in definitions_by_parameterization:
        raise ValueError("scale target completion lacks graph parameterization")
    scale_target = dict(definitions_by_parameterization[parameterization])
    discovery_target = dict(base["roles"]["design_select"]["target"])
    validation_target = _replace_statistics(
        base["roles"]["val_stop"]["target"], scale_target
    )
    confirmation_target = _replace_statistics(discovery_target, scale_target)
    scale_definition = {
        **dict(base["roles"]["model_train"]),
        "labels": str(scale_labels.resolve()),
        "hlt_caches": {
            str(key): str(value.resolve()) for key, value in sorted(scale_caches.items())
        },
        "target": scale_target,
    }
    if scale_trees:
        tree_completion = load_hashed_json(
            root / "scale_up" / "trees" / "completion.json"
        )
        scale_definition["tree_caches"] = {
            str(key): str(value.resolve()) for key, value in sorted(scale_trees.items())
        }
        scale_definition["tree_expected_parents"] = {
            str(key): {
                "hlt_content_sha256": load_hashed_json(
                    value.with_suffix(value.suffix + ".json")
                )["npz_sha256"],
                "tree_resource_sha256": tree_completion["tree_resource_sha256"],
                "backend_manifest_sha256": tree_completion[
                    "backend_manifest_sha256"
                ],
            }
            for key, value in sorted(scale_caches.items())
        }
    elif "tree_caches" in scale_definition:
        scale_definition.pop("tree_caches")
        scale_definition.pop("tree_expected_parents", None)
    definitions = {
        "scale_train": scale_definition,
        "val_stop": {
            **dict(base["roles"]["val_stop"]),
            "target": validation_target,
        },
        "design_confirm": {
            **dict(base["roles"]["design_select"]),
            "labels": str(design_confirm_labels.resolve()),
            "target": confirmation_target,
        },
    }
    manifest = build_stage_d_loader_manifest(
        row=row,
        role_definitions=definitions,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        evaluation_role="design_confirm",
        training_role="scale_train",
    )
    write_immutable_json(output, manifest)
    return output


def _target_completion(root: Path, target_id: str, plan, source):
    completion = load_hashed_json(
        root / "scale_up" / "targets" / target_id / "completion.json",
        expected_contract=SCALE_TARGET_COMPLETION_CONTRACT,
    )
    if (
        completion.get("source") != source
        or completion.get("scale_execution_plan_sha256") != plan["content_hash"]
    ):
        raise ValueError("scale target completion lineage differs")
    return completion


def _scale_native_relation_targets(root: Path, campaign) -> dict[int, Path]:
    completion = load_hashed_json(
        root
        / "scale_up"
        / "targets"
        / "native_relations"
        / "completion.json",
        expected_contract=SCALE_NATIVE_RELATION_WAVE_CONTRACT,
    )
    if (
        completion.get("source") != campaign["source"]
        or completion.get("campaign_spec_sha256") != campaign["content_hash"]
        or completion.get("replicas") != [0, 1, 2, 3]
    ):
        raise ValueError("scale native-relation completion lineage differs")
    outputs = {}
    for replica in range(4):
        output = (
            root
            / "scale_up"
            / "targets"
            / "native_relations"
            / f"replica_{replica}.npz"
        )
        artifact = load_hashed_json(
            output.with_suffix(".manifest.json"),
            expected_contract=NATIVE_RELATION_TARGET_CONTRACT,
        )
        if (
            artifact.get("source") != campaign["source"]
            or artifact.get("campaign_spec_sha256")
            != campaign["content_hash"]
            or artifact.get("split") != "scale_train"
            or int(artifact.get("hlt_replica_id", -1)) != replica
            or artifact.get("content_hash")
            != completion["artifact_hashes_by_replica"][str(replica)]
            or _sha256_file(output) != artifact.get("npz_sha256")
        ):
            raise ValueError("scale native-relation artifact lineage differs")
        outputs[replica] = output
    return outputs


def _load_single(
    *,
    root,
    definition,
    seed,
    loader_root,
    scale_caches,
    scale_trees,
    scale_labels,
    design_confirm_labels,
    campaign,
    registry,
    run,
):
    row = _row_with_seed(definition, seed)
    if definition["graph_kind"] == "AUXILIARY":
        base_path = loader_root / f"{definition['graph_id']}.json"
    else:
        selection = load_hashed_json(
            root / "auxiliary" / "locked_single_family_choices.json"
        )
        base_path = loader_root / (
            f"{selection['selected_row_by_target'][row['target_id']]}.json"
        )
    completion = _target_completion(root, row["target_id"], plan=load_hashed_json(
        root / "scale_up" / "execution_plan.json"
    ), source=campaign["source"])
    path = _scale_member_manifest(
        root=root,
        base_path=base_path,
        row=row,
        completion=completion,
        scale_caches=scale_caches,
        scale_trees=scale_trees,
        scale_labels=scale_labels,
        design_confirm_labels=design_confirm_labels,
        campaign=campaign,
        output=run / "loader.json",
    )
    loaded = load_stage_d_loaders_from_manifest(
        manifest_path=path,
        campaign_root=root,
        row=row,
        campaign=campaign,
        target_registry=registry,
    )
    return row, loaded


def _load_combination(
    *,
    root,
    definition,
    seed,
    loader_root,
    scale_caches,
    scale_trees,
    scale_labels,
    design_confirm_labels,
    campaign,
    registry,
    run,
    scale_native_relations,
    design_confirm_native_relation,
    plan,
):
    graph = dict(definition["graph"])
    graph["graph_id"] = definition["graph_id"]
    members = {}
    for member in graph["members"]:
        row = {
            **dict(member),
            "row_id": member["selected_row_id"],
            "row_kind": "SCIENTIFIC",
            "resolved": True,
        }
        completion = _target_completion(
            root, row["target_id"], plan, campaign["source"]
        )
        path = run / "members" / f"{row['target_id']}.json"
        _scale_member_manifest(
            root=root,
            base_path=_selected_loader_path(loader_root, member),
            row=row,
            completion=completion,
            scale_caches=scale_caches,
            scale_trees=scale_trees,
            scale_labels=scale_labels,
            design_confirm_labels=design_confirm_labels,
            campaign=campaign,
            output=path,
        )
        members[row["target_id"]] = path
    native = None
    if graph.get("native_relation_auxiliary") is not None:
        discovery = load_hashed_json(
            root / "job_ledgers" / "loaders" / "combinations" / f"{definition['graph_id']}.json"
        )
        if scale_native_relations is None:
            raise ValueError("scale native-relation combination lacks scale target")
        native = {
            **{
                f"scale_train:{replica}": str(path.resolve())
                for replica, path in sorted(scale_native_relations.items())
            },
            "val_stop:0": discovery["native_relation_target_files"][
                "val_stop"
            ]["0"]["path"],
            "design_confirm:0": (
                str(design_confirm_native_relation.resolve())
                if design_confirm_native_relation is not None
                else discovery["native_relation_target_files"][
                    "design_select"
                ]["0"]["path"]
            ),
        }
    manifest = build_combination_loader_manifest(
        graph=graph,
        member_loader_manifests=members,
        native_relation_target_files=native,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        evaluation_role="design_confirm",
        training_role="scale_train",
    )
    path = run / "loader.json"
    write_immutable_json(path, manifest)
    return graph, load_combination_loaders(
        manifest=manifest,
        graph=graph,
        campaign_root=root,
        campaign=campaign,
        target_registry=registry,
    )


def _finalize(root, plan, source):
    results = []
    for row in plan["graph_rows"]:
        path = (
            root / "scale_up" / "results"
            / f"{row['graph_id']}__seed_{row['seed']}.json"
        )
        if not path.is_file():
            return None
        results.append(load_hashed_json(path, expected_contract=SCALE_ROW_RESULT_CONTRACT))
    artifact = build_scale_graph_wave_completion(
        scale_plan=plan, results=results, source=source
    )
    write_immutable_json(root / "scale_up" / "graph_completion.json", artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scale-train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--design-confirm-cache", action="append", default=[])
    parser.add_argument("--scale-train-tree", action="append", default=[])
    parser.add_argument("--scale-train-labels", required=True, type=Path)
    parser.add_argument("--val-stop-labels", required=True, type=Path)
    parser.add_argument("--design-confirm-labels", required=True, type=Path)
    parser.add_argument("--stage-d-loader-root", required=True, type=Path)
    parser.add_argument(
        "--scale-native-relation",
        action="append",
        default=[],
        metavar="REPLICA=NPZ",
    )
    parser.add_argument("--design-confirm-native-relation", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    matches = [
        row for row in plan["graph_rows"]
        if row["graph_id"] == args.graph_id and int(row["seed"]) == args.seed
    ]
    if len(matches) != 1:
        raise ValueError("scale graph row is absent or duplicated")
    definition = dict(matches[0]["graph_definition"])
    required_resources = _required_scale_resources(definition)
    if args.dry_run:
        print(json.dumps({"graph_id": args.graph_id, "seed": args.seed, "executed": False}))
        return 0
    scale_caches = _mapping(args.scale_train_cache, name="--scale-train-cache")
    val_caches = _mapping(
        args.val_stop_cache,
        name="--val-stop-cache",
        required_replicas={0},
    )
    confirm_caches = _mapping(
        args.design_confirm_cache,
        name="--design-confirm-cache",
        required_replicas={0},
    )
    scale_trees = {}
    if required_resources["trees"]:
        scale_trees = (
            _mapping(
                args.scale_train_tree,
                name="--scale-train-tree",
                required_replicas=set(range(4)),
            )
            if args.scale_train_tree
            else {
                replica: root / "scale_up" / "trees" / "hlt" / f"replica_{replica}"
                for replica in range(4)
            }
        )
    if required_resources["trees"] and not scale_trees:
        raise ValueError("scale REGION graph lacks its tree resources")
    explicit_native = (
        _mapping(
            args.scale_native_relation,
            name="--scale-native-relation",
            required_replicas=set(range(4)),
        )
        if args.scale_native_relation and required_resources["native_relations"]
        else None
    )
    design_confirm_native_relation = (
        args.design_confirm_native_relation
        if args.design_confirm_native_relation is not None
        else (
            root
            / "targets"
            / "native_relations"
            / "design_confirm"
            / "replica_0.npz"
        )
        if required_resources["native_relations"]
        else None
    )
    common_lineage = {"scale_execution_plan": plan["content_hash"]}
    required_completions = [
        ("scale_input_completion", "scale_up/inputs/completion.json")
    ]
    if required_resources["trees"]:
        required_completions.append(
            ("scale_tree_completion", "scale_up/trees/completion.json")
        )
    if required_resources["targets"]:
        required_completions.extend(
            [
                (
                    "scale_normalizer_completion",
                    "scale_up/normalization/completion.json",
                ),
                ("scale_target_completion", "scale_up/target_completion.json"),
            ]
        )
    if required_resources["teacher_outputs"]:
        required_completions.append(
            (
                "scale_teacher_output_completion",
                "scale_up/teacher_outputs/completion.json",
            )
        )
    if required_resources["native_relations"]:
        required_completions.append(
            (
                "scale_native_relation_completion",
                "scale_up/targets/native_relations/completion.json",
            )
        )
    for key, relative in required_completions:
        artifact = load_hashed_json(root / relative)
        if artifact.get("source") != campaign["source"]:
            raise ValueError(f"scale graph {key} source differs")
        common_lineage[key] = artifact["content_hash"]
    common_lineage.update(
        {
            "scale_train_labels": _sha256_file(args.scale_train_labels),
            "val_stop_labels": _sha256_file(args.val_stop_labels),
            "design_confirm_labels": _sha256_file(
                args.design_confirm_labels
            ),
        }
    )
    for role, caches in (
        ("val_stop", val_caches),
        ("design_confirm", confirm_caches),
    ):
        for replica, cache in sorted(caches.items()):
            metadata = load_hashed_json(cache / "hlt_v3_metadata.json")
            common_lineage[f"{role}_hlt_replica_{replica}"] = metadata[
                "content_hash"
            ]
    if (
        required_resources["native_relations"]
        and design_confirm_native_relation is not None
    ):
        common_lineage["design_confirm_native_relation"] = _sha256_file(
            design_confirm_native_relation
        )
    registry = load_hashed_json(
        root / "registry" / "structure_target_registry.json"
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    protocol = HOSDTrainingProtocol(
        maximum_epochs=2 if miniature else 40,
        campaign_profile="miniature_test" if miniature else "production",
    )
    run = root / "scale_up" / "runs" / f"{args.graph_id}__seed_{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    kind = definition["graph_kind"]
    device = _device(args.device)
    flops, parameters = _capacity(root, args.graph_id)
    if kind == "BASELINE":
        baseline_id = definition["baseline_id"]
        teacher_id = (
            "O_BASE"
            if baseline_id == "H_KD_LOGIT_O_BASE"
            else "O_FULLREL"
            if baseline_id == "H_KD_LOGIT_O_FULLREL"
            else None
        )
        teacher_logits = (
            None
            if teacher_id is None
            else _teacher_logits_npz(root, teacher_id, args.scale_train_labels)
        )
        train = _dataset(
            scale_caches,
            args.scale_train_labels,
            role="scale_train",
            teacher_logits=teacher_logits,
            native_relation_targets=(
                (
                    explicit_native
                    if explicit_native is not None
                    else _scale_native_relation_targets(root, campaign)
                )
                if baseline_id == "H_NATIVE_REL_AUX"
                else None
            ),
        )
        val = _dataset(
            val_caches,
            args.val_stop_labels,
            role="val_stop",
            teacher_logits=None,
            native_relation_targets=None,
        )
        confirm = _dataset(
            confirm_caches,
            args.design_confirm_labels,
            role="design_confirm",
            teacher_logits=None,
            native_relation_targets=None,
        )
        task_seed = component_seed(args.seed, "baseline", baseline_id)
        train_loader = make_native_hlt_expert_loader(
            train, seed=task_seed, training=True, batch_size=64
        )
        val_loader = make_native_hlt_expert_loader(
            val, seed=task_seed, training=False, batch_size=64
        )
        evaluation_loader = make_native_hlt_expert_loader(
            confirm, seed=task_seed, training=False, batch_size=64
        )
        model = build_baseline_model(baseline_id, weaver_module=module)
        completion = train_stage_c_baseline(
            model=model,
            train_loader=train_loader,
            val_stop_loader=val_loader,
            output_dir=run,
            baseline_id=baseline_id,
            seed=args.seed,
            component_seed=task_seed,
            baseline_registry_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=_merge_lineage(
                common_lineage,
                (
                    {}
                    if teacher_logits is None
                    else {
                        "scale_teacher_training_logits": _sha256_file(
                            teacher_logits
                        )
                    }
                ),
            ),
            protocol=protocol,
            teacher_id=teacher_id,
            teacher_logit_key=(
                "offline_target_logits" if teacher_id is not None else None
            ),
            device=device,
            source=campaign["source"],
            checkpoint_contract=SCALE_TRAINING_CHECKPOINT_CONTRACT,
            completion_contract=SCALE_TRAINING_COMPLETION_CONTRACT,
            completion_filename="training_completion.json",
            curves_contract="hosd_scale_training_curves_v1",
            stage_label="Stage-J-scale",
        )
    elif kind in {"AUXILIARY", "FEEDBACK"}:
        row, loaded = _load_single(
            root=root,
            definition=definition,
            seed=args.seed,
            loader_root=args.stage_d_loader_root.resolve(),
            scale_caches=scale_caches,
            scale_trees=scale_trees,
            scale_labels=args.scale_train_labels,
            design_confirm_labels=args.design_confirm_labels,
            campaign=campaign,
            registry=registry,
            run=run,
        )
        training_lineage = _merge_lineage(
            common_lineage, loaded["lineage_hashes"]
        )
        if kind == "AUXILIARY":
            model = build_auxiliary_model(row, weaver_module=module)[0]
        else:
            # Scale feedback is a continuation experiment too: refit the
            # locked A_t definition on scale_train, then add the registered
            # feedback consumer for a second fixed 40-epoch phase.
            auxiliary_row = {
                **row,
                "row_id": f"{row['row_id']}__SCALE_A_T",
                "parameterization": row[
                    "selected_auxiliary_parameterization"
                ],
                "auxiliary_weight": row["selected_auxiliary_weight"],
                "row_kind": "SCIENTIFIC",
                "selection_eligible": False,
                "control": None,
                "semantic_loss_enabled": True,
                "head_component_seed": component_seed(
                    args.seed,
                    "scale_auxiliary_head",
                    row["selected_auxiliary_row_id"],
                ),
                "resolved": True,
            }
            auxiliary_model = build_auxiliary_model(
                auxiliary_row, weaver_module=module
            )[0]
            auxiliary_root = run / "auxiliary_initialization"
            auxiliary_completion = train_stage_d_auxiliary(
                model=auxiliary_model,
                train_loader=loaded["train_loader"],
                val_stop_loader=loaded["val_stop_loader"],
                design_select_loader=loaded["design_confirm_loader"],
                output_dir=auxiliary_root,
                row=auxiliary_row,
                component_group_ids=loaded["component_group_ids"],
                stage_d_plan_sha256=plan["content_hash"],
                campaign_spec_sha256=campaign["content_hash"],
                lineage_hashes=training_lineage,
                protocol=protocol,
                source=campaign["source"],
                deployed_analytical_flops=float(
                    auxiliary_model_flop_ledger(auxiliary_model)[
                        "deployed_total_flops"
                    ]
                ),
                deployed_parameter_count=exact_trainable_parameter_count(
                    auxiliary_model.classifier
                ),
                device=device,
                checkpoint_contract=SCALE_TRAINING_CHECKPOINT_CONTRACT,
                completion_contract=SCALE_TRAINING_COMPLETION_CONTRACT,
                prediction_contract=SCALE_TRAINING_PREDICTION_CONTRACT,
                plan_hash_field="scale_execution_plan_sha256",
                stage_label="Stage-J-scale-A_t",
                completion_filename="training_completion.json",
                curves_contract="hosd_scale_auxiliary_initialization_curves_v1",
                evaluation_split="design_confirm",
            )
            auxiliary_result = load_hashed_json(
                auxiliary_root / "design_confirm_result.json",
                expected_contract=SCALE_TRAINING_PREDICTION_CONTRACT,
            )
            row = {
                **row,
                "selected_auxiliary_row_id": auxiliary_row["row_id"],
                "selected_auxiliary_result_sha256": auxiliary_result[
                    "content_hash"
                ],
            }
            model = build_feedback_model(row, weaver_module=module)
            if row.get("control") == "EXACT_HLT":
                runtime = loaded.get("exact_hlt_runtime")
                if runtime is None:
                    raise ValueError("exact HLT scale row lacks runtime normalization")
                model.configure_exact_hlt_runtime(**runtime)
            initialization = initialize_feedback_from_auxiliary_checkpoint(
                model,
                row,
                checkpoint_path=auxiliary_root / "best_model_val.pt",
                completion=auxiliary_completion,
                result=auxiliary_result,
                stage_d_plan_sha256=plan["content_hash"],
                campaign_spec_sha256=campaign["content_hash"],
                source=campaign["source"],
                checkpoint_contract=SCALE_TRAINING_CHECKPOINT_CONTRACT,
                completion_contract=SCALE_TRAINING_COMPLETION_CONTRACT,
                prediction_contract=SCALE_TRAINING_PREDICTION_CONTRACT,
                plan_hash_field="scale_execution_plan_sha256",
            )
            training_lineage = _merge_lineage(
                training_lineage, initialization
            )
        completion = train_stage_d_auxiliary(
            model=model,
            train_loader=loaded["train_loader"],
            val_stop_loader=loaded["val_stop_loader"],
            design_select_loader=loaded["design_confirm_loader"],
            output_dir=run,
            row=row,
            component_group_ids=loaded["component_group_ids"],
            stage_d_plan_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=training_lineage,
            protocol=protocol,
            source=campaign["source"],
            deployed_analytical_flops=flops,
            deployed_parameter_count=parameters,
            deployed_operation_profile=(
                feedback_model_flop_ledger(model).get(
                    "exact_hlt_builder_profile"
                )
                if kind == "FEEDBACK"
                else None
            ),
            device=device,
            checkpoint_contract=SCALE_TRAINING_CHECKPOINT_CONTRACT,
            completion_contract=SCALE_TRAINING_COMPLETION_CONTRACT,
            prediction_contract=SCALE_TRAINING_PREDICTION_CONTRACT,
            plan_hash_field="scale_execution_plan_sha256",
            stage_label="Stage-J-scale",
            completion_filename="training_completion.json",
            curves_contract="hosd_scale_training_curves_v1",
            evaluation_split="design_confirm",
        )
        evaluation_loader = loaded["design_confirm_loader"]
    elif kind == "COMBINATION":
        graph, loaded = _load_combination(
            root=root,
            definition=definition,
            seed=args.seed,
            loader_root=args.stage_d_loader_root.resolve(),
            scale_caches=scale_caches,
            scale_trees=scale_trees,
            scale_labels=args.scale_train_labels,
            design_confirm_labels=args.design_confirm_labels,
            campaign=campaign,
            registry=registry,
            run=run,
            scale_native_relations=(
                explicit_native
                if explicit_native is not None
                else _scale_native_relation_targets(root, campaign)
                if definition["graph"].get("native_relation_auxiliary")
                is not None
                else None
            ),
            design_confirm_native_relation=design_confirm_native_relation,
            plan=plan,
        )
        model = build_combination_model(graph, seed=args.seed, weaver_module=module)
        completion = train_combination(
            model=model,
            train_loader=loaded["train_loader"],
            val_stop_loader=loaded["val_stop_loader"],
            design_select_loader=loaded["design_confirm_loader"],
            graph=graph,
            output_dir=run,
            stage_f_plan_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=_merge_lineage(
                common_lineage, loaded["lineage_hashes"]
            ),
            protocol=protocol,
            source=campaign["source"],
            deployed_analytical_flops=flops,
            deployed_parameter_count=parameters,
            device=device,
            evaluation_split="design_confirm",
            checkpoint_contract=SCALE_TRAINING_CHECKPOINT_CONTRACT,
            completion_contract=SCALE_TRAINING_COMPLETION_CONTRACT,
            result_contract=SCALE_TRAINING_PREDICTION_CONTRACT,
            plan_hash_field="scale_execution_plan_sha256",
            completion_filename="training_completion.json",
        )
        evaluation_loader = loaded["design_confirm_loader"]
    else:
        raise ValueError("scale graph kind differs")
    identities, logits, labels = _inference(model, evaluation_loader, device=device)
    metrics = evaluate_classification(logits, labels, split="design_confirm")
    prediction_sha = _predictions(
        run / "design_confirm_predictions.npz", identities, logits
    )
    checkpoint_path = run / "best_model_val.pt"
    descriptor = {**definition, "seed": args.seed}
    export_loader = build_label_free_hlt_loader(
        cache_paths={0: val_caches[0]},
        identities=_labels(args.val_stop_labels)[0][:8],
        logical_role="val_stop",
        realization_policy="R_FIXED",
        batch_size=8,
    )
    export_path = (
        root / "scale_up" / "exports" / f"{args.graph_id}__seed_{args.seed}.pt"
    )
    export = export_deployable_graph(
        descriptor=descriptor,
        research_model=model,
        representative_batch=next(iter(export_loader)),
        output_path=export_path,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        lineage_hashes={
            "scale_execution_plan": plan["content_hash"],
            "training_completion": completion["content_hash"],
            "design_confirm_predictions": prediction_sha,
        },
        source=campaign["source"],
        weaver_module=module,
        analytical_inference_flops_batch1_n128=int(flops),
        deployed_operation_profile=(
            feedback_model_flop_ledger(model).get(
                "exact_hlt_builder_profile"
            )
            if kind == "FEEDBACK"
            else None
        ),
    )
    result = build_scale_row_result(
        scale_plan=plan,
        graph_id=args.graph_id,
        seed=args.seed,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        deployable_export_sha256=export["content_hash"],
        classification_metrics=metrics,
        analytical_forward_flops_by_role=(
            (
                lambda ledger: {
                    "model_train": int(ledger["training_forward_flops"]),
                    "val_stop": int(ledger["evaluation_forward_flops"]),
                    "design_confirm": int(flops),
                }
            )(auxiliary_model_flop_ledger(model))
            if kind == "AUXILIARY"
            or (
                kind == "BASELINE"
                and definition["baseline_id"] == "H_NATIVE_REL_AUX"
            )
            else (
                lambda ledger: {
                    "model_train": int(ledger["deployed_total_flops"]),
                    "val_stop": int(ledger["deployed_total_flops"]),
                    "design_confirm": int(flops),
                }
            )(feedback_model_flop_ledger(model))
            if kind == "FEEDBACK"
            else (
                lambda ledger: {
                    "model_train": int(ledger["training_total_flops"]),
                    "val_stop": int(ledger["training_total_flops"]),
                    "design_confirm": int(flops),
                }
            )(combination_model_flop_ledger(model))
            if kind == "COMBINATION"
            else {
                "model_train": int(flops),
                "val_stop": int(flops),
                "design_confirm": int(flops),
            }
        ),
        deployed_operation_profile=(
            feedback_model_flop_ledger(model).get(
                "exact_hlt_builder_profile"
            )
            if kind == "FEEDBACK"
            else None
        ),
        source=campaign["source"],
    )
    output = (
        root / "scale_up" / "results"
        / f"{args.graph_id}__seed_{args.seed}.json"
    )
    write_immutable_json(output, result)
    wave = _finalize(root, plan, campaign["source"])
    print(
        json.dumps(
            {
                "graph_id": args.graph_id,
                "seed": args.seed,
                "result_sha256": result["content_hash"],
                "export_manifest_sha256": export["content_hash"],
                "coverage_complete": wave is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
