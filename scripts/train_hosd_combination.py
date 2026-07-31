#!/usr/bin/env python3
"""Compile Stage F, lock beam waves, or select complete combination fits."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    advance_combination_beam,
    authorize_access,
    build_combination_beam_completion,
    build_combination_loader_manifest,
    build_combination_selection,
    build_combination_wave_completion,
    build_stage_f_plan,
    combination_model_flop_ledger,
    exact_trainable_parameter_count,
    expand_combination_beam,
    build_combination_model,
    HOSDTrainingProtocol,
    load_combination_loaders,
    train_combination,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    COMBINATION_RESULT_CONTRACT,
    COMBINATION_BEAM_COMPLETION_CONTRACT,
    COMBINATION_BEAM_PROMOTION_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_F_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def _deployed_capacity(model, args) -> tuple[float, int]:
    ledger = combination_model_flop_ledger(model)
    flops = float(ledger["deployed_total_flops"])
    parameters = exact_trainable_parameter_count(model.classifier)
    if (
        args.deployed_analytical_flops is not None
        and float(args.deployed_analytical_flops) != flops
    ):
        raise ValueError("supplied combination deployed FLOPs differ analytically")
    if (
        args.deployed_parameter_count is not None
        and int(args.deployed_parameter_count) != parameters
    ):
        raise ValueError("supplied combination deployed parameter count differs")
    return flops, parameters


def _member_loader_paths(graph, root: Path) -> dict[str, Path]:
    result = {
        row["target_id"]: root / f"{row['selected_row_id']}.json"
        for row in graph["members"]
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "combination member loader manifests are absent: "
            + ", ".join(missing)
        )
    return result


def _keyed_paths(values: list[str], *, name: str) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"{name} requires unique KEY=PATH values")
        result[key] = Path(path)
    return result


def _native_relation_paths(
    campaign_root: Path, *, evaluation_role: str
) -> dict[str, Path]:
    split = {
        "design_select": "val_design",
        "design_confirm": "val_design",
    }[evaluation_role]
    return {
        **{
            f"model_train:{replica}": campaign_root
            / "targets"
            / "native_relations"
            / "model_train"
            / f"replica_{replica}.npz"
            for replica in range(4)
        },
        "val_stop:0": (
            campaign_root
            / "targets"
            / "native_relations"
            / "val_stop"
            / "replica_0.npz"
        ),
        f"{evaluation_role}:0": (
            campaign_root
            / "targets"
            / "native_relations"
            / split
            / "replica_0.npz"
        ),
    }


def _try_finalize_training_wave(
    *,
    campaign_root: Path,
    plan,
    graph,
    seed: int,
) -> None:
    pcgrad = graph["graph_id"] == plan["pcgrad_control"]["graph_id"]
    beam = None
    if pcgrad:
        expected = [plan["pcgrad_control"]]
        output = campaign_root / "combinations" / "pcgrad_completion.json"
        wave_kind = "PCGRAD"
    else:
        beam = load_hashed_json(
            campaign_root / "combinations" / "beam_completion.json",
            expected_contract=COMBINATION_BEAM_COMPLETION_CONTRACT,
        )
        expected = [
            *plan["mandatory_combinations"],
            *beam["promotion"]["promoted_graphs"],
        ]
        output = campaign_root / "combinations" / "full_completion.json"
        wave_kind = "FULL"
    paths = [
        campaign_root
        / "combinations"
        / row["graph_id"]
        / f"seed_{int(seed)}"
        / "design_select_result.json"
        for row in expected
    ]
    if not all(path.is_file() for path in paths):
        return
    results = [
        load_hashed_json(path, expected_contract=COMBINATION_RESULT_CONTRACT)
        for path in paths
    ]
    artifact = build_combination_wave_completion(
        stage_f_plan=plan,
        wave_kind=wave_kind,
        results=results,
        beam_completion=beam,
    )
    write_immutable_json(output, artifact)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "compile",
            "execute",
            "beam-run",
            "beam-expand",
            "beam-wave",
            "full-select",
        ),
        default="compile",
    )
    parser.add_argument("--family")
    parser.add_argument("--candidate", action="append", default=[], type=Path)
    parser.add_argument("--result", action="append", default=[], type=Path)
    parser.add_argument("--beam-winner", action="append", default=[])
    parser.add_argument("--completed-new-fit-count", type=int, default=0)
    parser.add_argument("--expansion", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--graph-json", type=Path)
    parser.add_argument("--graph-id")
    parser.add_argument("--loader-manifest", type=Path)
    parser.add_argument("--member-loader-root", type=Path)
    parser.add_argument(
        "--native-relation-target", action="append", default=[]
    )
    parser.add_argument("--beam-root-result", type=Path)
    parser.add_argument("--deployed-analytical-flops", type=float)
    parser.add_argument("--deployed-parameter-count", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    if args.mode == "compile":
        single = load_hashed_json(
            args.campaign_root / "auxiliary" / "locked_single_family_choices.json",
            expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
        )
        feedback = load_hashed_json(
            args.campaign_root / "feedback" / "locked_feedback_choices.json",
            expected_contract=FEEDBACK_SELECTION_CONTRACT,
        )
        artifact = build_stage_f_plan(
            single_family_selection=single,
            feedback_selection=feedback,
            campaign_spec_sha256=campaign["content_hash"],
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "job_ledgers" / "stage_f_execution_plan.json"
        )
    else:
        authorize_access(
            worker_role="design_selector",
            requested_resource="design_select_predictions",
        )
        plan_path = (
            args.campaign_root / "job_ledgers" / "stage_f_execution_plan.json"
        )
        if args.mode == "beam-run" and not plan_path.exists():
            single = load_hashed_json(
                args.campaign_root
                / "auxiliary"
                / "locked_single_family_choices.json",
                expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
            )
            feedback = load_hashed_json(
                args.campaign_root
                / "feedback"
                / "locked_feedback_choices.json",
                expected_contract=FEEDBACK_SELECTION_CONTRACT,
            )
            write_immutable_json(
                plan_path,
                build_stage_f_plan(
                    single_family_selection=single,
                    feedback_selection=feedback,
                    campaign_spec_sha256=campaign["content_hash"],
                    source=campaign["source"],
                ),
            )
        plan = load_hashed_json(
            plan_path,
            expected_contract=STAGE_F_PLAN_CONTRACT,
        )
        if args.mode == "execute":
            if args.graph_json is not None:
                graph = json.loads(args.graph_json.read_text(encoding="utf-8"))
            else:
                promoted = []
                beam_path = (
                    args.campaign_root
                    / "combinations"
                    / "beam_completion.json"
                )
                if beam_path.is_file():
                    beam = load_hashed_json(
                        beam_path,
                        expected_contract=COMBINATION_BEAM_COMPLETION_CONTRACT,
                    )
                    if beam.get("stage_f_plan_sha256") != plan["content_hash"]:
                        raise ValueError("combination beam source plan differs")
                    promoted = list(beam["promotion"]["promoted_graphs"])
                matches = [
                    graph
                    for graph in [
                        *plan["mandatory_combinations"],
                        plan["pcgrad_control"],
                        *promoted,
                    ]
                    if graph["graph_id"] == args.graph_id
                ]
                if len(matches) != 1:
                    raise ValueError("combination graph is absent or duplicated")
                graph = matches[0]
            if args.loader_manifest is None:
                if args.member_loader_root is None:
                    raise ValueError(
                        "combination execution requires loader or member-loader root"
                    )
                loader_artifact = build_combination_loader_manifest(
                    graph=graph,
                    member_loader_manifests=_member_loader_paths(
                        graph, args.member_loader_root
                    ),
                    native_relation_target_files=(
                        _keyed_paths(
                            args.native_relation_target,
                            name="--native-relation-target",
                        )
                        or (
                            _native_relation_paths(
                                args.campaign_root,
                                evaluation_role="design_select",
                            )
                            if graph.get("native_relation_auxiliary")
                            is not None
                            else None
                        )
                    ),
                    campaign_spec_sha256=campaign["content_hash"],
                    source=campaign["source"],
                )
                args.loader_manifest = (
                    args.campaign_root
                    / "combinations"
                    / "loaders"
                    / f"{graph['graph_id']}.json"
                )
                write_immutable_json(args.loader_manifest, loader_artifact)
            target_registry = load_hashed_json(
                args.campaign_root / "registry" / "structure_target_registry.json",
                expected_contract="hosd_structure_target_registry_v1",
            )
            loader_manifest = load_hashed_json(
                args.loader_manifest,
            )
            loaded = load_combination_loaders(
                manifest=loader_manifest,
                graph=graph,
                campaign_root=args.campaign_root,
                campaign=campaign,
                target_registry=target_registry,
            )
            module = importlib.import_module("weaver.nn.model.ParticleTransformer")
            model = build_combination_model(
                graph, seed=args.seed, weaver_module=module
            )
            deployed_flops, deployed_parameters = _deployed_capacity(
                model, args
            )
            import torch

            miniature = campaign["campaign_profile"] == "miniature_test"
            completion = train_combination(
                model=model,
                train_loader=loaded["train_loader"],
                val_stop_loader=loaded["val_stop_loader"],
                design_select_loader=loaded["design_select_loader"],
                graph=graph,
                output_dir=args.output
                or args.campaign_root
                / "combinations"
                / graph["graph_id"]
                / f"seed_{args.seed}",
                stage_f_plan_sha256=plan["content_hash"],
                campaign_spec_sha256=campaign["content_hash"],
                lineage_hashes=loaded["lineage_hashes"],
                protocol=HOSDTrainingProtocol(
                    maximum_epochs=(
                        2
                        if miniature
                        else int(graph["fixed_epoch_budget"])
                    ),
                    campaign_profile=(
                        "miniature_test" if miniature else "production"
                    ),
                ),
                source=campaign["source"],
                deployed_analytical_flops=deployed_flops,
                deployed_parameter_count=deployed_parameters,
                device=(
                    "cuda"
                    if args.device == "auto" and torch.cuda.is_available()
                    else "cpu"
                    if args.device == "auto"
                    else args.device
                ),
            )
            _try_finalize_training_wave(
                campaign_root=args.campaign_root,
                plan=plan,
                graph=graph,
                seed=args.seed,
            )
            print(
                json.dumps(
                    {
                        "mode": "execute",
                        "graph_id": graph["graph_id"],
                        "completion_sha256": completion["content_hash"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "beam-run":
            if args.member_loader_root is None:
                raise ValueError("beam-run mode requires --member-loader-root")
            root_result_path = args.beam_root_result or (
                args.campaign_root
                / "baselines"
                / "H_BASE_BEAM_BUDGET"
                / f"seed_{args.seed}"
                / "design_select_result.json"
            )
            root_result = load_hashed_json(
                root_result_path,
                expected_contract=COMBINATION_RESULT_CONTRACT,
            )
            if (
                root_result.get("source") != campaign["source"]
                or root_result.get("graph_id") != "H_BASE_BEAM_BUDGET"
                or int(root_result.get("fixed_epoch_budget", -1)) != 5
            ):
                raise ValueError("Stage-F beam root result semantics differ")
            target_registry = load_hashed_json(
                args.campaign_root
                / "registry"
                / "structure_target_registry.json",
                expected_contract="hosd_structure_target_registry_v1",
            )
            module = importlib.import_module("weaver.nn.model.ParticleTransformer")
            import torch

            resolved_device = (
                "cuda"
                if args.device == "auto" and torch.cuda.is_available()
                else "cpu"
                if args.device == "auto"
                else args.device
            )
            result_by_graph = {root_result["graph_id"]: root_result}
            beam = []
            completed = 0
            expansions, waves = [], []
            for family in plan["family_order"]:
                expansion = expand_combination_beam(
                    stage_f_plan=plan,
                    family=family,
                    current_beam=beam,
                    completed_new_fit_count=completed,
                )
                expansion_path = (
                    args.campaign_root
                    / "combinations"
                    / f"beam_expansion_{family}.json"
                )
                write_immutable_json(expansion_path, expansion)
                expansions.append(expansion)
                for graph in expansion["new_fit_candidates"]:
                    manifest = build_combination_loader_manifest(
                        graph=graph,
                        member_loader_manifests=_member_loader_paths(
                            graph, args.member_loader_root
                        ),
                        campaign_spec_sha256=campaign["content_hash"],
                        source=campaign["source"],
                    )
                    manifest_path = (
                        args.campaign_root
                        / "combinations"
                        / "loaders"
                        / f"{graph['graph_id']}.json"
                    )
                    write_immutable_json(manifest_path, manifest)
                    loaded = load_combination_loaders(
                        manifest=manifest,
                        graph=graph,
                        campaign_root=args.campaign_root,
                        campaign=campaign,
                        target_registry=target_registry,
                    )
                    model = build_combination_model(
                        graph, seed=args.seed, weaver_module=module
                    )
                    deployed_flops, deployed_parameters = _deployed_capacity(
                        model, args
                    )
                    output_dir = (
                        args.campaign_root
                        / "combinations"
                        / "beam"
                        / graph["graph_id"]
                        / f"seed_{args.seed}"
                    )
                    train_combination(
                        model=model,
                        train_loader=loaded["train_loader"],
                        val_stop_loader=loaded["val_stop_loader"],
                        design_select_loader=loaded["design_select_loader"],
                        graph=graph,
                        output_dir=output_dir,
                        stage_f_plan_sha256=plan["content_hash"],
                        campaign_spec_sha256=campaign["content_hash"],
                        lineage_hashes=loaded["lineage_hashes"],
                        protocol=HOSDTrainingProtocol(
                            maximum_epochs=(
                                2
                                if campaign["campaign_profile"]
                                == "miniature_test"
                                else 5
                            ),
                            campaign_profile=(
                                "miniature_test"
                                if campaign["campaign_profile"]
                                == "miniature_test"
                                else "production"
                            ),
                        ),
                        source=campaign["source"],
                        deployed_analytical_flops=deployed_flops,
                        deployed_parameter_count=deployed_parameters,
                        device=resolved_device,
                    )
                    result_by_graph[graph["graph_id"]] = load_hashed_json(
                        output_dir / "design_select_result.json",
                        expected_contract=COMBINATION_RESULT_CONTRACT,
                    )
                wave_results = [
                    result_by_graph[row["graph_id"]]
                    for row in expansion["all_candidates"]
                ]
                wave = advance_combination_beam(
                    stage_f_plan=plan,
                    family=family,
                    expansion=expansion,
                    reduced_budget_results=wave_results,
                )
                write_immutable_json(
                    args.campaign_root
                    / "combinations"
                    / f"beam_wave_{family}.json",
                    wave,
                )
                waves.append(wave)
                beam = wave["surviving_candidates"]
                completed = int(wave["completed_new_fit_count"])
            artifact = build_combination_beam_completion(
                stage_f_plan=plan,
                expansions=expansions,
                waves=waves,
                reduced_budget_results=list(result_by_graph.values()),
            )
            output = args.output or (
                args.campaign_root
                / "combinations"
                / "beam_completion.json"
            )
            write_immutable_json(
                args.campaign_root
                / "combinations"
                / "beam_promotion.json",
                artifact["promotion"],
            )
            publication = write_immutable_json(output, artifact)
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "output": str(output.resolve()),
                        "content_hash": artifact["content_hash"],
                        "publication": publication["status"],
                        "new_fit_count": artifact["new_fit_count"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        results = [
            load_hashed_json(path, expected_contract=COMBINATION_RESULT_CONTRACT)
            for path in args.result
        ]
        if args.mode == "beam-expand":
            if args.family is None:
                raise ValueError("beam-expand mode requires --family")
            candidates = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.candidate
            ]
            artifact = expand_combination_beam(
                stage_f_plan=plan,
                family=args.family,
                current_beam=candidates,
                completed_new_fit_count=args.completed_new_fit_count,
            )
            output = args.output or (
                args.campaign_root
                / "combinations"
                / f"beam_expansion_{args.family}.json"
            )
        elif args.mode == "beam-wave":
            if args.family is None:
                raise ValueError("beam-wave mode requires --family")
            if args.expansion is None:
                raise ValueError("beam-wave mode requires --expansion")
            expansion = load_hashed_json(
                args.expansion,
                expected_contract="hosd_combination_beam_expansion_v1",
            )
            artifact = advance_combination_beam(
                stage_f_plan=plan,
                family=args.family,
                expansion=expansion,
                reduced_budget_results=results,
            )
            output = args.output or (
                args.campaign_root
                / "combinations"
                / f"beam_wave_{args.family}.json"
            )
        else:
            artifact = build_combination_selection(
                stage_f_plan=plan,
                full_results=results,
                beam_winner_graph_ids=args.beam_winner,
                source=campaign["source"],
            )
            output = args.output or (
                args.campaign_root
                / "combinations"
                / "locked_combination_choices.json"
            )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(output.resolve()),
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
