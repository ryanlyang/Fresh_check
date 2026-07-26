#!/usr/bin/env python3
"""Submit or print the full Step-10 graph with logical-node recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_particle_view_production_graph,
    build_low_data_campaign_inventory,
    build_low_data_campaign_registry,
    build_runtime_command_catalog,
    build_runtime_data_config,
    build_runtime_execution_manifest,
    build_runtime_handler_catalog,
    load_hashed_json,
    publish_full_pilot_scientific_bootstrap,
    reconcile_particle_view_production_graph,
    submit_particle_view_graph,
    with_content_hash,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph")
    source.add_argument("--registry")
    source.add_argument("--unified-manifest")
    parser.add_argument("--command-catalog")
    parser.add_argument(
        "--handler-commands",
        help=(
            "Category-to-argv-template JSON; generates the runtime manifest "
            "and graph command catalog automatically."
        ),
    )
    parser.add_argument("--runtime-python-executable", default="python")
    parser.add_argument(
        "--bootstrap-scientific",
        action="store_true",
        help=(
            "Build every production factory, task spec, scientific catalog, "
            "and handler command before constructing/submitting the graph."
        ),
    )
    parser.add_argument("--runtime-data-config")
    parser.add_argument("--parent-manifest")
    parser.add_argument("--hlt-cache-dir")
    parser.add_argument("--offline-cache-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--linear-fusion-steps", type=int, default=300)
    parser.add_argument("--optional-p7b-resource")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--existing-checkpoint")
    parser.add_argument("--existing-observed-train-identity-sha256")
    parser.add_argument("--existing-serialized-recipe")
    parser.add_argument(
        "--existing-recipe-reproduced-exactly", action="store_true"
    )
    parser.add_argument("--existing-provenance-metadata-sha256")
    parser.add_argument(
        "--existing-description",
        default="pre-existing offline particle teacher",
    )
    parser.add_argument("--artifact-root")
    parser.add_argument("--source-commit")
    parser.add_argument("--graph-id", default="particle_view_full_pilot_v1")
    parser.add_argument("--existing-teacher-compatible", action="store_true")
    parser.add_argument("--teacher-mix-compatible", action="store_true")
    parser.add_argument("--ledger-output")
    parser.add_argument(
        "--existing-job",
        action="append",
        default=[],
        metavar="NODE=JOB_ID:STATE",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--print-only", action="store_true")
    return parser


def _existing(values: list[str]):
    result = {}
    for value in values:
        try:
            node_id, job_state = value.split("=", 1)
            job_id, state = job_state.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "--existing-job must use NODE=JOB_ID:STATE"
            ) from exc
        if node_id in result:
            raise ValueError(f"duplicate existing logical node {node_id}")
        result[node_id] = {"job_id": job_id, "state": state}
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "execute" if args.execute else "print_only" if args.print_only else "dry_run"
    if args.graph:
        if args.bootstrap_scientific:
            raise ValueError("--bootstrap-scientific cannot modify an existing graph")
        graph_path = Path(args.graph).resolve()
        graph = load_hashed_json(graph_path)
        artifact_root = Path(graph["artifact_root"])
    else:
        missing = [
            name
            for name, value in (
                ("--artifact-root", args.artifact_root),
                ("--source-commit", args.source_commit),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "registry bootstrap also requires " + ", ".join(missing)
            )
        supplied_catalogs = sum(
            bool(value)
            for value in (
                args.command_catalog,
                args.handler_commands,
                args.bootstrap_scientific,
            )
        )
        if supplied_catalogs != 1:
            raise ValueError(
                "registry bootstrap requires exactly one of "
                "--command-catalog, --handler-commands, or "
                "--bootstrap-scientific"
            )
        if args.registry:
            registry = load_hashed_json(args.registry)
            inventory = None
        else:
            unified = load_hashed_json(args.unified_manifest)
            registry = build_low_data_campaign_registry(
                unified_split_manifest=unified,
                existing_teacher_compatible=args.existing_teacher_compatible,
                teacher_mix_compatible=args.teacher_mix_compatible,
            )
            inventory = build_low_data_campaign_inventory(registry)
        artifact_root = Path(args.artifact_root).resolve()
        runtime_artifacts = None
        bootstrap_index = None
        if args.bootstrap_scientific:
            if args.runtime_data_config:
                if any(
                    value
                    for value in (
                        args.parent_manifest,
                        args.hlt_cache_dir,
                        args.offline_cache_dir,
                    )
                ):
                    raise ValueError(
                        "--runtime-data-config cannot be combined with raw "
                        "runtime source arguments"
                    )
                runtime_data = load_hashed_json(args.runtime_data_config)
            else:
                missing_runtime = [
                    name
                    for name, value in (
                        ("--parent-manifest", args.parent_manifest),
                        ("--hlt-cache-dir", args.hlt_cache_dir),
                        ("--offline-cache-dir", args.offline_cache_dir),
                    )
                    if not value
                ]
                if missing_runtime or not args.unified_manifest:
                    raise ValueError(
                        "scientific bootstrap from raw sources requires "
                        "--unified-manifest and "
                        + ", ".join(missing_runtime or ["runtime sources"])
                    )
                runtime_data = build_runtime_data_config(
                    parent_manifest_path=args.parent_manifest,
                    unified_manifest_path=args.unified_manifest,
                    hlt_cache_dir=args.hlt_cache_dir,
                    offline_cache_dir=args.offline_cache_dir,
                )
            optional_p7b = None
            if args.optional_p7b_resource:
                optional_p7b = json.loads(
                    Path(args.optional_p7b_resource).read_text(
                        encoding="utf-8"
                    )
                )
                if not isinstance(optional_p7b, dict):
                    raise ValueError(
                        "optional P7b resource must be a JSON object"
                    )
            bootstrap_root = (
                artifact_root / "preflight" / "scientific_bootstrap"
            )
            bootstrap_index = publish_full_pilot_scientific_bootstrap(
                output_dir=bootstrap_root,
                registry=registry,
                runtime_data_config=runtime_data,
                source_commit=args.source_commit,
                device=args.device,
                num_workers=args.num_workers,
                amp=not args.no_amp,
                batch_size=args.batch_size,
                bootstrap_replicates=args.bootstrap_replicates,
                linear_fusion_steps=args.linear_fusion_steps,
                optional_p7b_resource=optional_p7b,
                existing_checkpoint_path=args.existing_checkpoint,
                existing_observed_train_identity_sha256=(
                    args.existing_observed_train_identity_sha256
                ),
                existing_serialized_recipe_path=(
                    args.existing_serialized_recipe
                ),
                existing_recipe_reproduced_exactly=(
                    args.existing_recipe_reproduced_exactly
                ),
                existing_provenance_metadata_sha256=(
                    args.existing_provenance_metadata_sha256
                ),
                existing_description=args.existing_description,
                existing_teacher_compatible=args.existing_teacher_compatible,
                teacher_mix_compatible=args.teacher_mix_compatible,
                python_executable=sys.executable,
                handler_python_executable=(
                    args.runtime_python_executable
                ),
                production=True,
            )
            args.handler_commands = bootstrap_index[
                "scientific_handler_commands"
            ]["path"]
        if args.command_catalog:
            catalog = json.loads(
                Path(args.command_catalog).read_text(encoding="utf-8")
            )
        else:
            handler_commands = json.loads(
                Path(args.handler_commands).read_text(encoding="utf-8")
            )
            if not isinstance(handler_commands, dict):
                raise ValueError("handler commands must be a JSON object")
            handler_catalog = build_runtime_handler_catalog(handler_commands)
            preflight = artifact_root / "preflight"
            registry_path = (
                Path(args.registry).resolve()
                if args.registry
                else preflight / "low_data_campaign_registry.json"
            )
            handler_catalog_path = (
                preflight / "runtime_handler_catalog.json"
            )
            execution_manifest_path = (
                preflight / "runtime_execution_manifest.json"
            )
            execution_manifest = build_runtime_execution_manifest(
                registry=registry,
                registry_path=str(registry_path),
                handler_catalog=handler_catalog,
                handler_catalog_path=str(handler_catalog_path),
                artifact_root=str(artifact_root),
            )
            catalog = build_runtime_command_catalog(
                execution_manifest_path=str(execution_manifest_path),
                python_executable=args.runtime_python_executable,
            )
            runtime_artifacts = (
                handler_catalog_path,
                handler_catalog,
                execution_manifest_path,
                execution_manifest,
            )
        graph = build_particle_view_production_graph(
            registry=registry,
            artifact_root=str(artifact_root),
            source_commit=args.source_commit,
            command_catalog=catalog,
            graph_id=args.graph_id,
        )
        reconciliation = reconcile_particle_view_production_graph(
            graph=graph, registry=registry
        )
        graph_path = artifact_root / "preflight" / "production_graph.json"
        if mode != "print_only":
            if inventory is not None:
                write_immutable_json(
                    artifact_root
                    / "preflight"
                    / "low_data_campaign_registry.json",
                    registry,
                )
                write_immutable_json(
                    artifact_root
                    / "preflight"
                    / "low_data_campaign_inventory.json",
                    inventory,
                )
            if bootstrap_index is not None:
                write_immutable_json(
                    artifact_root
                    / "preflight"
                    / "full_scientific_bootstrap.json",
                    bootstrap_index,
                )
            if runtime_artifacts is not None:
                (
                    handler_catalog_path,
                    handler_catalog,
                    execution_manifest_path,
                    execution_manifest,
                ) = runtime_artifacts
                write_immutable_json(handler_catalog_path, handler_catalog)
                write_immutable_json(
                    execution_manifest_path,
                    execution_manifest,
                )
            write_immutable_json(graph_path, graph)
            write_immutable_json(
                artifact_root / "preflight" / "graph_reconciliation.json",
                reconciliation,
            )
    progress_dir = (
        artifact_root
        / "job_ledgers"
        / "submission_progress"
        / graph["content_hash"][:16]
    )

    def persist_progress(records) -> None:
        if mode != "execute":
            return
        snapshot = with_content_hash(
            {
                "contract": "particle_view_submission_progress_v1",
                "graph_sha256": graph["content_hash"],
                "graph_path": str(graph_path),
                "records": list(records),
                "last_node_id": records[-1]["node_id"],
                "record_count": len(records),
            }
        )
        write_immutable_json(
            progress_dir
            / (
                f"{len(records):02d}_{records[-1]['node_id']}_"
                f"{snapshot['content_hash'][:16]}.json"
            ),
            snapshot,
        )

    ledger = submit_particle_view_graph(
        graph=graph,
        graph_path=str(graph_path),
        existing_jobs=_existing(args.existing_job),
        mode=mode,
        progress_callback=persist_progress,
    )
    for row in ledger["records"]:
        if row["command"] is None:
            print(f"{row['node_id']}: {row['action']} job={row['job_id']}")
        else:
            print(f"{row['node_id']}: {shlex.join(row['command'])}")
    if mode != "print_only":
        ledger_output = (
            Path(args.ledger_output)
            if args.ledger_output
            else artifact_root
            / "job_ledgers"
            / (
                f"submission_{graph['content_hash'][:16]}_{mode}_"
                f"{ledger['content_hash'][:16]}.json"
            )
        )
        write_immutable_json(ledger_output, ledger)
    print(
        f"mode={mode} planned={ledger['planned_submit_count']} "
        f"submitted={ledger['submitted_count']} "
        f"content_hash={ledger['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
