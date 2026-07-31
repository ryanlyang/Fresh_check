"""Authenticated manifest-plan factories for sealed Stage N."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    bind_source,
    canonical_json_bytes,
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .early_continuation import (
    _context,
    _producer_completion,
    _publish,
    _row,
)
from .final_seal import (
    CONTROL_KINDS,
    build_finalist_controls,
    validate_finalist_controls,
)
from .hlt_capacity_controls import validate_hlt_capacity_control_row
from .provenance import source_snapshot
from .production import FINAL_NODE_ENTRYPOINTS
from .late_plan_factories import producer_plan_identity_sha256
from .stage_n_execution import (
    build_deployable_inference_input_binding,
    validate_control_evidence,
    validate_shared_deployable_inference_payload,
)
from .stage_n_access import validate_prelock_input_row_access


FINAL_PLAN_FACTORY_INPUT_CONTRACT = "retb_stage_n_factory_input_v2"
FINAL_PLAN_FACTORY_CONTRACT = "retb_stage_n_plan_factories_v4"
FINAL_PLAN_FACTORY_TARGETS = (
    "prelock_final_inputs",
    "stack_val_inference",
    "accuracy_finalist_selector",
    "postlock_oracle_targets",
    "finalist_controls",
    "final_test_execution_lock",
    "sealed_final_test",
    "final_report",
)
PIPELINE_SEEDS = (101, 202, 303)


def final_factory_input_path(
    campaign_root: str | Path, *, target_node_id: str
) -> Path:
    return (
        Path(campaign_root).resolve()
        / "job_ledgers"
        / "factory_inputs"
        / f"{target_node_id}.json"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_row(
    *,
    target: str,
    index: int,
    raw: Mapping[str, Any],
    campaign_spec_sha256: str,
    production_graph_sha256: str,
) -> dict[str, Any]:
    required = {
        "task_id",
        "argv",
        "expected_outputs",
        "input_artifact_hashes",
        "environment",
    }
    if set(raw) != required:
        raise ValueError("Stage-N factory-input row fields differ")
    argv = [str(value) for value in raw["argv"]]
    scripts = {
        value.replace("\\", "/")
        for value in argv
        if value.replace("\\", "/").startswith("scripts/")
    }
    allowed = set(FINAL_NODE_ENTRYPOINTS[target])
    lowered = " ".join(argv).lower()
    hashes = dict(raw["input_artifact_hashes"])
    environment = {
        str(name): str(value)
        for name, value in raw["environment"].items()
    }
    if (
        str(raw["task_id"]) != f"{target}:{index}"
        or len(scripts & allowed) != 1
        or "--dry-run" in argv
        or not raw["expected_outputs"]
        or hashes.get("campaign_spec") != campaign_spec_sha256
        or hashes.get("production_graph") != production_graph_sha256
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in hashes.values()
        )
        or environment.get(
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION"
        )
        != "0"
    ):
        raise ValueError("Stage-N factory-input row semantics differ")
    if target == "prelock_final_inputs":
        try:
            validate_prelock_input_row_access(
                argv, raw["expected_outputs"]
            )
        except ValueError as exc:
            raise ValueError(
                "prelock final-input access semantics differ"
            ) from exc
        if environment.get("RETB_PRELOCK_MODEL_OUTPUTS_EMITTED") != "0":
            raise ValueError("prelock final-input access semantics differ")
    if target == "stack_val_inference" and (
        any(
            term in lowered
            for term in ("final_test", "label", "offline", "oracle", "target")
        )
        or environment.get("RETB_INFERENCE_SPLIT") != "stack_val"
        or environment.get("RETB_PREDICTION_SHARDS_CONTAIN_LABELS") != "0"
    ):
        raise ValueError("stack-val inference access semantics differ")
    if target == "accuracy_finalist_selector" and (
        "final_test" in lowered
        or environment.get("RETB_SELECTOR_SPLIT") != "stack_val"
        or environment.get("RETB_LABEL_JOIN_LOCATION") != "selector_only"
    ):
        raise ValueError("scale-finalist selector access semantics differ")
    if target == "postlock_oracle_targets" and (
        environment.get("RETB_CREATED_AFTER_FINALIST_LOCK") != "1"
        or environment.get("RETB_SELECTION_ELIGIBLE") != "0"
    ):
        raise ValueError("postlock oracle-target timing differs")
    if target == "finalist_controls" and (
        environment.get("RETB_CREATED_AFTER_FINALIST_LOCK") != "1"
        or environment.get("RETB_CONTROLS_MAY_REPLACE_FINALIST") != "0"
    ):
        raise ValueError("finalist-control timing differs")
    if target == "final_test_execution_lock" and (
        environment.get("RETB_ALL_POSTLOCK_EVIDENCE_COMPLETE") != "1"
        or environment.get("RETB_FINAL_TEST_MODEL_OUTPUT_AUTHORIZED") != "1"
    ):
        raise ValueError("final-test execution-lock semantics differ")
    if target == "sealed_final_test" and (
        environment.get("RETB_EXACTLY_ONCE_FINAL_TEST") != "1"
        or environment.get("RETB_BOTH_FINAL_LOCKS_REQUIRED") != "1"
    ):
        raise ValueError("sealed final-test execution semantics differ")
    if target == "final_report" and environment.get(
        "RETB_TEST_RESULT_MAY_REPLACE_FINALIST"
    ) != "0":
        raise ValueError("final report selection semantics differ")
    return {
        "task_id": str(raw["task_id"]),
        "argv": argv,
        "expected_outputs": [
            str(Path(value)) for value in raw["expected_outputs"]
        ],
        "input_artifact_hashes": hashes,
        "environment": environment,
    }


def build_final_factory_input(
    *,
    target_node_id: str,
    producer_node_id: str,
    campaign_spec_sha256: str,
    production_graph_sha256: str,
    producer_task_manifest_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(target_node_id)
    if target not in FINAL_PLAN_FACTORY_TARGETS or not rows:
        raise ValueError("Stage-N factory-input target/rows differ")
    checked = [
        _checked_row(
            target=target,
            index=index,
            raw=raw,
            campaign_spec_sha256=campaign_spec_sha256,
            production_graph_sha256=production_graph_sha256,
        )
        for index, raw in enumerate(rows)
    ]
    coverage_row = dict(coverage)
    if (
        coverage_row.get("all_predeclared_rows_present") is not True
        or coverage_row.get("scientific_metric_used_for_membership") is not False
        or coverage_row.get("incomplete_wave_permitted") is not False
    ):
        raise ValueError("Stage-N coverage attestation differs")
    singleton_targets = {
        "prelock_final_inputs",
        "accuracy_finalist_selector",
        "final_test_execution_lock",
        "sealed_final_test",
        "final_report",
    }
    if target in singleton_targets and len(checked) != 1:
        raise ValueError(f"{target} requires exactly one row")
    if target == "stack_val_inference":
        expected = {
            (str(graph_id), seed)
            for graph_id in coverage_row.get("shortlisted_graph_ids", [])
            for seed in PIPELINE_SEEDS
        }
        actual = {
            (
                row["environment"].get("RETB_GRAPH_ID"),
                int(row["environment"].get("RETB_PIPELINE_SEED", -1)),
            )
            for row in checked
        }
        if actual != expected:
            raise ValueError("stack-val graph/seed coverage differs")
    if target == "postlock_oracle_targets":
        expected = {
            (str(graph_id), seed, split)
            for graph_id in coverage_row.get("finalist_graph_ids", [])
            for seed in PIPELINE_SEEDS
            for split in ("stack_val", "final_test")
        }
        actual = {
            (
                row["environment"].get("RETB_GRAPH_ID"),
                int(row["environment"].get("RETB_PIPELINE_SEED", -1)),
                row["environment"].get("RETB_TARGET_SPLIT"),
            )
            for row in checked
        }
        if actual != expected:
            raise ValueError("postlock target graph/seed/split coverage differs")
    if target == "accuracy_finalist_selector" and coverage_row.get(
        "prediction_graph_seed_coverage_complete"
    ) is not True:
        raise ValueError("finalist selector lacks complete predictions")
    if target == "finalist_controls" and set(
        coverage_row.get("required_control_kinds", [])
    ) != set(CONTROL_KINDS):
        raise ValueError("finalist-control kind coverage differs")
    if target == "finalist_controls":
        expected = {
            (str(graph_id), kind, seed)
            for graph_id in coverage_row.get("finalist_graph_ids", [])
            for kind in ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG")
            for seed in PIPELINE_SEEDS
        }
        actual = {
            (
                row["environment"].get("RETB_OWNER_FINALIST_GRAPH_ID"),
                row["environment"].get("RETB_CONTROL_KIND"),
                int(row["environment"].get("RETB_PIPELINE_SEED", -1)),
            )
            for row in checked
        }
        if actual != expected:
            raise ValueError("finalist-control graph/kind/seed coverage differs")
    if target == "final_test_execution_lock" and not all(
        coverage_row.get(name) is True
        for name in (
            "prelock_inputs_complete",
            "postlock_targets_complete",
            "finalist_controls_complete",
        )
    ):
        raise ValueError("final-test execution-lock coverage differs")
    if target == "sealed_final_test" and (
        coverage_row.get("task_count") != 1
        or coverage_row.get("execution_claim_precedes_model_access")
        is not True
    ):
        raise ValueError("sealed final-test exactly-once coverage differs")
    return with_content_hash(
        {
            "contract": FINAL_PLAN_FACTORY_INPUT_CONTRACT,
            "schema_version": 2,
            "target_node_id": target,
            "producer_node_id": str(producer_node_id),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "production_graph_sha256": require_sha256(
                production_graph_sha256, name="production_graph_sha256"
            ),
            "producer_task_manifest_sha256": require_sha256(
                producer_task_manifest_sha256,
                name="producer_task_manifest_sha256",
            ),
            "row_count": len(checked),
            "rows": checked,
            "coverage": coverage_row,
            "generated_by_authenticated_upstream_producer": True,
            "operator_supplied_row_json_permitted": False,
            "source": dict(source),
        }
    )


def validate_final_factory_input(
    payload: Mapping[str, Any],
    *,
    target_node_id: str,
    producer_node_id: str,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_task_manifest_sha256: str,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_PLAN_FACTORY_INPUT_CONTRACT
    )
    expected = build_final_factory_input(
        target_node_id=target_node_id,
        producer_node_id=producer_node_id,
        campaign_spec_sha256=validate_content_hash(campaign),
        production_graph_sha256=validate_content_hash(production_graph),
        producer_task_manifest_sha256=producer_task_manifest_sha256,
        rows=payload.get("rows", []),
        coverage=payload.get("coverage", {}),
        source=campaign["source"],
    )
    if dict(payload) != expected:
        raise ValueError("Stage-N factory-input semantics differ")
    return digest


def publish_final_factory_input(
    *,
    campaign_root: str | Path,
    payload: Mapping[str, Any],
    target_node_id: str,
    producer_node_id: str,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_task_manifest_sha256: str,
) -> dict[str, Any]:
    validate_final_factory_input(
        payload,
        target_node_id=target_node_id,
        producer_node_id=producer_node_id,
        campaign=campaign,
        production_graph=production_graph,
        producer_task_manifest_sha256=producer_task_manifest_sha256,
    )
    return write_immutable_json(
        final_factory_input_path(
            campaign_root, target_node_id=target_node_id
        ),
        payload,
    )


def _build(
    *,
    target: str,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str,
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    input_path = final_factory_input_path(
        root, target_node_id=target
    )
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    authenticated_outputs = {
        str(Path(path).resolve()): digest
        for row in completion["rows"]
        for path, digest in row["output_hashes"].items()
    }
    resolved = str(input_path.resolve())
    if (
        resolved not in authenticated_outputs
        or not input_path.is_file()
        or input_path.is_symlink()
        or _file_sha256(input_path) != authenticated_outputs[resolved]
    ):
        raise ValueError(
            "Stage-N factory input is not an authenticated producer output"
        )
    payload = load_hashed_json(
        input_path, expected_contract=FINAL_PLAN_FACTORY_INPUT_CONTRACT
    )
    input_sha = validate_final_factory_input(
        payload,
        target_node_id=target,
        producer_node_id=producer_node_id,
        campaign=campaign,
        production_graph=production_graph,
        producer_task_manifest_sha256=producer_plan_identity_sha256(
            producer_node_id=producer_node_id,
            production_graph=production_graph,
            completion=completion,
        ),
    )
    rows = []
    for index, source_row in enumerate(payload["rows"]):
        extra = {
            name: digest
            for name, digest in source_row[
                "input_artifact_hashes"
            ].items()
            if name not in {"campaign_spec", "production_graph"}
        }
        extra["final_factory_input"] = input_sha
        rows.append(
            _row(
                target=target,
                index=index,
                argv=source_row["argv"],
                outputs=source_row["expected_outputs"],
                campaign_sha256=campaign_sha,
                graph_sha256=graph_sha,
                producer_completion_sha256=completion_sha,
                extra_input_hashes=extra,
                environment=source_row["environment"],
            )
        )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=rows,
    )


def _factory(target: str, producer: str) -> Callable[..., dict[str, Any]]:
    def build(
        *,
        campaign_root: str | Path,
        campaign: Mapping[str, Any],
        production_graph: Mapping[str, Any],
        producer_node_id: str = producer,
    ) -> dict[str, Any]:
        return _build(
            target=target,
            campaign_root=campaign_root,
            campaign=campaign,
            production_graph=production_graph,
            producer_node_id=producer_node_id,
        )

    build.__name__ = f"build_{target}_manifest_plan"
    return build


def build_prelock_final_inputs_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "input_audit",
) -> dict[str, Any]:
    """Prepare only checkpoint-free final-test input lineage."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    profile = load_hashed_json(root / "inputs" / "hlt_v3_profile.json")
    hlt = load_hashed_json(
        root
        / "inputs"
        / "hlt_v3"
        / "final_test"
        / "replica_0"
        / "R_FIXED"
        / "D_NOMINAL"
        / "hlt_v3_metadata.json"
    )
    final_identity = canonical_sha256(
        {
            "split_manifest": campaign["parent_artifact_hashes"][
                "split_manifest"
            ],
            "split": "final_test",
        }
    )
    configuration = with_content_hash(
        {
            "contract": "retb_prelock_final_input_configuration_v1",
            "schema_version": 1,
            "split_manifest_sha256": campaign["parent_artifact_hashes"][
                "split_manifest"
            ],
            "degradation_profile_sha256": profile["content_hash"],
            "input_hashes": {
                "final_test_identity_manifest": final_identity,
                "final_test_raw_inputs": canonical_sha256(
                    {
                        "raw_input_schema": campaign[
                            "parent_artifact_hashes"
                        ]["raw_input_schema"],
                        "split": "final_test",
                    }
                ),
                "final_test_HLT_inputs": hlt["content_hash"],
                "final_test_relation_sidecars": canonical_sha256(
                    {
                        "HLT_cache": hlt["content_hash"],
                        "sidecar": "relation",
                    }
                ),
                "final_test_REGION_sidecars": canonical_sha256(
                    {
                        "HLT_cache": hlt["content_hash"],
                        "sidecar": "REGION_tree",
                    }
                ),
            },
            "source": dict(campaign["source"]),
        }
    )
    config_path = (
        root / "inputs" / "stage_n" / "prelock_input_configuration.json"
    )
    write_immutable_json(config_path, configuration)
    output = root / "inputs" / "stage_n" / "prelock_final_inputs.json"
    shared_root = root / "inputs" / "stage_n" / "shared"
    shared_outputs = [
        shared_root / f"retb_{split}_shared_HLT_inputs{suffix}"
        for split in ("stack_val", "final_test")
        for suffix in (".json", ".pt")
    ]
    row = _row(
        target="prelock_final_inputs",
        index=0,
        argv=[
            "python",
            "scripts/prepare_retb_final_test_inputs.py",
            "--campaign-root",
            str(root),
            "--configuration",
            str(config_path),
            "--output",
            str(output),
        ],
        outputs=[str(output), *(str(path) for path in shared_outputs)],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "configuration": configuration["content_hash"],
            "raw_HLT_cache": hlt["content_hash"],
            "degradation_profile": profile["content_hash"],
        },
        environment={
            "RETB_PRELOCK_MODEL_OUTPUTS_EMITTED": "0",
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="prelock_final_inputs",
        rows=[row],
    )


def build_stack_val_inference_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "scale_completion",
) -> dict[str, Any]:
    """Derive all label-free stack rows from the complete scale registry."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    shortlist = load_hashed_json(
        root / "selection" / "locked_scale_shortlist.json",
        expected_contract="retb_locked_scale_shortlist_v2",
    )
    scale = load_hashed_json(
        root / "selection" / "scale_completion.json",
        expected_contract="retb_scale_completion_v2",
    )
    run_map = {
        (row["graph_id"], int(row["pipeline_seed"])): row
        for row in scale["runs"]
    }
    expected = {
        (graph_id, seed)
        for graph_id in shortlist["SCALE_SHORTLIST"]
        for seed in PIPELINE_SEEDS
    }
    training_expected = {
        (graph_id, seed)
        for graph_id in shortlist["SCALE_TRAINING_GRAPHS"]
        for seed in PIPELINE_SEEDS
    }
    if set(run_map) != training_expected or not expected <= set(run_map):
        raise ValueError("stack-val scale-run coverage differs")
    raw_metadata = load_hashed_json(
        root
        / "inputs"
        / "hlt_v3"
        / "stack_val"
        / "replica_0"
        / "R_FIXED"
        / "D_NOMINAL"
        / "hlt_v3_metadata.json"
    )
    normalizers = {
        seed: load_hashed_json(
            root
            / "runs"
            / "scale"
            / "refits"
            / f"seed_{seed}"
            / "normalization"
            / "scale_normalizer_bundle.json"
        )
        for seed in PIPELINE_SEEDS
    }
    shared_root = root / "inputs" / "stage_n" / "shared"
    shared_manifest_path = (
        shared_root / "retb_stack_val_shared_HLT_inputs.json"
    )
    shared_manifest = load_hashed_json(shared_manifest_path)
    validate_shared_deployable_inference_payload(
        shared_manifest, manifest_path=shared_manifest_path
    )
    if shared_manifest.get("source") != campaign["source"]:
        raise ValueError("stack-val shared HLT input source differs")
    rows = []
    for graph_id, seed in sorted(expected):
        run = run_map[(graph_id, seed)]
        working = (
            root
            / "selection"
            / "stack_val"
            / graph_id
            / f"seed_{seed}"
        )
        safe_graph = graph_id.replace(":", "__")
        input_manifest = (
            shared_root / f"stack_val_{safe_graph}_seed{seed}.json"
        )
        raw_output = working / "deployable_logits.npz"
        prediction_manifest = (
            working / f"{graph_id}_seed{seed}_stack_val_predictions.json"
        )
        prediction_payload = (
            working / f"{graph_id}_seed{seed}_stack_val_predictions.npz"
        )
        export = (
            root
            / "runs"
            / "scale"
            / "graphs"
            / graph_id
            / f"seed_{seed}"
            / "export"
            / "deployable_retb_graph.json"
        )
        parents = {
            "campaign_spec": campaign_sha,
            "locked_scale_shortlist": shortlist["content_hash"],
            "scale_completion": scale["content_hash"],
            "scale_graph_run": run["scale_graph_run_sha256"],
            "deployable_export": run["deployable_export_sha256"],
            # Replaced with the actual manifest hash after the preparation
            # step; this value binds the immutable raw source it must use.
            "stack_val_HLT_input_manifest": raw_metadata["content_hash"],
            "deployable_scale_normalizer_bundle": normalizers[seed][
                "content_hash"
            ],
            "degradation_profile": shortlist["parent_hashes"][
                "degradation_profile"
            ],
        }
        plan = with_content_hash(
            {
                "contract": "retb_stack_val_inference_execution_plan_v1",
                "schema_version": 1,
                "graph_id": graph_id,
                "pipeline_seed": seed,
                "locked_scale_shortlist_sha256": shortlist["content_hash"],
                "scale_completion_sha256": scale["content_hash"],
                "parent_hashes": parents,
                "steps": [
                    {
                        "step_id": "step_000",
                        "argv": [
                            "python",
                            "scripts/prepare_retb_deployable_input.py",
                            "--campaign-root",
                            str(root),
                            "--split",
                            "stack_val",
                            "--graph-id",
                            graph_id,
                            "--pipeline-seed",
                            str(seed),
                            "--output-dir",
                            str(shared_root),
                            "--shared-payload-manifest",
                            str(shared_manifest_path),
                        ],
                        "expected_outputs": [str(input_manifest)],
                    },
                    {
                        "step_id": "step_001",
                        "argv": [
                            "python",
                            "scripts/run_retb_deployable_inference.py",
                            "--campaign-root",
                            str(root),
                            "--deployable-export",
                            str(export),
                            "--input-manifest",
                            str(input_manifest),
                            "--split",
                            "stack_val",
                            "--graph-id",
                            graph_id,
                            "--pipeline-seed",
                            str(seed),
                            "--scale-completion",
                            str(root / "selection" / "scale_completion.json"),
                            "--output",
                            str(raw_output),
                        ],
                        "expected_outputs": [str(raw_output)],
                    },
                ],
                "inference_output_npz": str(raw_output),
                "source": dict(campaign["source"]),
            }
        )
        plan_path = (
            root
            / "job_ledgers"
            / "execution_plans"
            / "stage_n"
            / "stack_val"
            / f"{graph_id}_seed{seed}.json"
        )
        write_immutable_json(plan_path, plan)
        rows.append(
            _row(
                target="stack_val_inference",
                index=len(rows),
                argv=[
                    "python",
                    "scripts/execute_retb_stack_val_inference.py",
                    "--campaign-root",
                    str(root),
                    "--locked-scale-shortlist",
                    str(root / "selection" / "locked_scale_shortlist.json"),
                    "--scale-completion",
                    str(root / "selection" / "scale_completion.json"),
                    "--execution-plan",
                    str(plan_path),
                    "--output-dir",
                    str(working),
                ],
                outputs=[str(prediction_manifest), str(prediction_payload)],
                campaign_sha256=campaign_sha,
                graph_sha256=graph_sha,
                producer_completion_sha256=completion_sha,
                extra_input_hashes={
                    "locked_scale_shortlist": shortlist["content_hash"],
                    "scale_completion": scale["content_hash"],
                    "scale_graph_run": run["scale_graph_run_sha256"],
                    "execution_plan": plan["content_hash"],
                    "raw_HLT_cache": raw_metadata["content_hash"],
                    "shared_HLT_payload": shared_manifest["content_hash"],
                },
                environment={
                    "RETB_GRAPH_ID": graph_id,
                    "RETB_PIPELINE_SEED": str(seed),
                    "RETB_INFERENCE_SPLIT": "stack_val",
                    "RETB_PREDICTION_SHARDS_CONTAIN_LABELS": "0",
                    "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                },
            )
        )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="stack_val_inference",
        rows=rows,
    )
def build_accuracy_finalist_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "stack_val_inference",
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    predictions = sorted(
        Path(path)
        for row in completion["rows"]
        for path in row["output_hashes"]
        if Path(path).name.endswith("_stack_val_predictions.json")
    )
    shortlist = load_hashed_json(
        root / "selection" / "locked_scale_shortlist.json"
    )
    scale = load_hashed_json(root / "selection" / "scale_completion.json")
    expected_count = len(shortlist["SCALE_SHORTLIST"]) * len(PIPELINE_SEEDS)
    if len(predictions) != expected_count:
        raise ValueError("stack-val prediction completion coverage differs")
    step14 = load_hashed_json(
        root / "registry" / "retb_step14_scale_final_seal_bundle.json"
    )
    profile = load_hashed_json(root / "inputs" / "hlt_v3_profile.json")
    raw = load_hashed_json(
        root
        / "inputs"
        / "hlt_v3"
        / "stack_val"
        / "replica_0"
        / "R_FIXED"
        / "D_NOMINAL"
        / "hlt_v3_metadata.json"
    )
    normalizer = load_hashed_json(
        root
        / "runs"
        / "scale"
        / "refits"
        / "seed_101"
        / "normalization"
        / "scale_normalizer_bundle.json"
    )
    offline_shapes = load_hashed_json(
        root / "selection" / "retb_offline_shapes.json"
    )
    heterogeneous = load_hashed_json(
        root / "selection" / "retb_heterogeneous_shapes.json"
    )
    coordinate = load_hashed_json(
        root / "selection" / "locked_bridge_coordinates.json"
    )
    predictor = load_hashed_json(
        root / "selection" / "predictor_bundle" / "predictor_bundle_lock.json"
    )
    lineage = {
        "campaign_spec": campaign_sha,
        "step14_bundle": step14["content_hash"],
        "locked_scale_shortlist": shortlist["content_hash"],
        "scale_completion": scale["content_hash"],
        "split_manifest": campaign["parent_artifact_hashes"][
            "split_manifest"
        ],
        "validation_partition_manifest": campaign[
            "parent_artifact_hashes"
        ]["validation_partition_manifest"],
        "scale_train_manifest": campaign["parent_artifact_hashes"][
            "scale_train_manifest"
        ],
        "hlt_replica_manifest": campaign["parent_artifact_hashes"][
            "hlt_replica_manifest"
        ],
        "hlt_cache_manifest": raw["content_hash"],
        "realization_policy": canonical_sha256(
            {"stack_val": "R_FIXED", "replica": 0}
        ),
        "degradation_profile": profile["content_hash"],
        "degradation_parameters": canonical_sha256(
            profile.get("profiles", profile)
        ),
        "offline_scale_normalizer_bundle": normalizer["content_hash"],
        "shared_hlt_scale_normalizer_bundle": normalizer["content_hash"],
        "shape_registry": canonical_sha256(
            {
                "offline": offline_shapes["content_hash"],
                "heterogeneous": heterogeneous["content_hash"],
            }
        ),
        "target_coordinate_lock": coordinate["content_hash"],
        "predictor_bundle_lock": predictor["content_hash"],
    }
    shapes = {
        "SHAPE_HIGH": offline_shapes["SHAPE_HIGH"],
        "SHAPE_COMPACT": offline_shapes["SHAPE_COMPACT"],
        "SHAPE_BRIDGE": shortlist["SHAPE_BRIDGE"],
        "HET_PHYSICS": heterogeneous.get(
            "HET_PHYSICS", {"allocation": "predeclared"}
        ),
        "HET_SELECTED": heterogeneous["HET_SELECTED"],
        "HET_BEAM": heterogeneous["HET_BEAM"],
    }
    config_root = root / "selection" / "stack_val" / "selector_inputs"
    lineage_path = config_root / "lineage_hashes.json"
    shapes_path = config_root / "shape_assignments.json"
    write_immutable_bytes(
        lineage_path, canonical_json_bytes(lineage) + b"\n"
    )
    write_immutable_bytes(
        shapes_path, canonical_json_bytes(shapes) + b"\n"
    )
    output = root / "selection" / "locked_scale_finalists.json"
    argv = [
        "python",
        "scripts/select_retb_scale_finalists.py",
        "--campaign-root",
        str(root),
        "--locked-scale-shortlist",
        str(root / "selection" / "locked_scale_shortlist.json"),
        "--scale-completion",
        str(root / "selection" / "scale_completion.json"),
        "--final-select-label-manifest",
        str(root / "inputs" / "final_select_label_manifest.json.gz"),
        "--lineage-hashes",
        str(lineage_path),
        "--shape-assignments",
        str(shapes_path),
    ]
    for path in predictions:
        argv.extend(["--prediction-manifest", str(path)])
    row = _row(
        target="accuracy_finalist_selector",
        index=0,
        argv=argv,
        outputs=[
            str(output),
            str(root / "selection" / "accuracy_selection_trace.json"),
            str(root / "selection" / "rejection_selection_trace.json"),
        ],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "locked_scale_shortlist": shortlist["content_hash"],
            "scale_completion": scale["content_hash"],
            "selector_lineage": canonical_sha256(lineage),
            "shape_assignments": canonical_sha256(shapes),
            **{
                f"prediction_{index:03d}": load_hashed_json(path)[
                    "content_hash"
                ]
                for index, path in enumerate(predictions)
            },
        },
        environment={
            "RETB_SELECTOR_SPLIT": "stack_val",
            "RETB_LABEL_JOIN_LOCATION": "selector_only",
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="accuracy_finalist_selector",
        rows=[row],
    )
def build_postlock_oracle_targets_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "accuracy_finalist_selector",
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    finalists = load_hashed_json(
        root / "selection" / "locked_scale_finalists.json",
        expected_contract="retb_locked_scale_finalists_v1",
    )
    prelock = load_hashed_json(
        root / "inputs" / "stage_n" / "prelock_final_inputs.json"
    )
    rows = []
    for graph_id in finalists["finalist_graph_ids"]:
        for seed in PIPELINE_SEEDS:
            stack_prediction = load_hashed_json(
                root
                / "selection"
                / "stack_val"
                / graph_id
                / f"seed_{seed}"
                / f"{graph_id}_seed{seed}_stack_val_predictions.json"
            )
            for split in ("stack_val", "final_test"):
                working = (
                    root
                    / "evaluations"
                    / "postlock_targets"
                    / graph_id
                    / f"seed_{seed}"
                    / split
                )
                evidence = working / "generation_evidence.json"
                output = working / "postlock_target.json"
                input_hash = (
                    stack_prediction["parent_hashes"][
                        "stack_val_HLT_input_manifest"
                    ]
                    if split == "stack_val"
                    else prelock["input_hashes"]["final_test_HLT_inputs"]
                )
                plan = with_content_hash(
                    {
                        "contract": "retb_postlock_target_execution_plan_v1",
                        "schema_version": 1,
                        "graph_id": graph_id,
                        "pipeline_seed": seed,
                        "split": split,
                        "locked_scale_finalists_sha256": finalists[
                            "content_hash"
                        ],
                        "steps": [
                            {
                                "step_id": "step_000",
                                "argv": [
                                    "python",
                                    "scripts/generate_retb_postlock_target_evidence.py",
                                    "--campaign-root",
                                    str(root),
                                    "--locked-scale-finalists",
                                    str(
                                        root
                                        / "selection"
                                        / "locked_scale_finalists.json"
                                    ),
                                    "--graph-id",
                                    graph_id,
                                    "--pipeline-seed",
                                    str(seed),
                                    "--split",
                                    split,
                                    "--input-manifest-sha256",
                                    input_hash,
                                    "--output-dir",
                                    str(working),
                                    "--output",
                                    str(evidence),
                                ],
                                "expected_outputs": [str(evidence)],
                            }
                        ],
                        "target_evidence": str(evidence),
                        "source": dict(campaign["source"]),
                    }
                )
                plan_path = (
                    root
                    / "job_ledgers"
                    / "execution_plans"
                    / "stage_n"
                    / "postlock_targets"
                    / f"{graph_id}_seed{seed}_{split}.json"
                )
                write_immutable_json(plan_path, plan)
                rows.append(
                    _row(
                        target="postlock_oracle_targets",
                        index=len(rows),
                        argv=[
                            "python",
                            "scripts/execute_retb_postlock_oracle_target.py",
                            "--campaign-root",
                            str(root),
                            "--locked-scale-finalists",
                            str(
                                root
                                / "selection"
                                / "locked_scale_finalists.json"
                            ),
                            "--execution-plan",
                            str(plan_path),
                            "--output",
                            str(output),
                        ],
                        outputs=[str(output)],
                        campaign_sha256=campaign_sha,
                        graph_sha256=graph_sha,
                        producer_completion_sha256=completion_sha,
                        extra_input_hashes={
                            "locked_scale_finalists": finalists[
                                "content_hash"
                            ],
                            "execution_plan": plan["content_hash"],
                            "input_manifest": input_hash,
                        },
                        environment={
                            "RETB_GRAPH_ID": graph_id,
                            "RETB_PIPELINE_SEED": str(seed),
                            "RETB_TARGET_SPLIT": split,
                            "RETB_CREATED_AFTER_FINALIST_LOCK": "1",
                            "RETB_SELECTION_ELIGIBLE": "0",
                            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                        },
                    )
                )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="postlock_oracle_targets",
        rows=rows,
    )
def build_finalist_controls_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "accuracy_finalist_selector",
) -> dict[str, Any]:
    """Train every graph-specific post-lock control in parallel."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    finalists = load_hashed_json(
        root / "selection" / "locked_scale_finalists.json",
        expected_contract="retb_locked_scale_finalists_v1",
    )
    scale = load_hashed_json(root / "selection" / "scale_completion.json")
    rows = []
    for graph_id in finalists["finalist_graph_ids"]:
        for kind in ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG"):
            for seed in PIPELINE_SEEDS:
                output = (
                    root
                    / "runs"
                    / "scale"
                    / "finalist_controls"
                    / graph_id
                    / kind
                    / f"seed_{seed}"
                )
                rows.append(
                    _row(
                        target="finalist_controls",
                        index=len(rows),
                        argv=[
                            "python",
                            "scripts/train_retb_scale_finalist_control.py",
                            "--campaign-root",
                            str(root),
                            "--locked-scale-finalists",
                            str(
                                root
                                / "selection"
                                / "locked_scale_finalists.json"
                            ),
                            "--owner-finalist-graph-id",
                            graph_id,
                            "--control-kind",
                            kind,
                            "--pipeline-seed",
                            str(seed),
                            "--output-dir",
                            str(output),
                        ],
                        outputs=[
                            str(output / "control_row.json"),
                            str(output / "deployable_control.json"),
                            str(
                                output
                                / "training"
                                / "training_registration.json"
                            ),
                            str(output / "training" / "best_model_val.pt"),
                        ],
                        campaign_sha256=campaign_sha,
                        graph_sha256=graph_sha,
                        producer_completion_sha256=completion_sha,
                        extra_input_hashes={
                            "locked_scale_finalists": finalists[
                                "content_hash"
                            ],
                            "scale_completion": scale["content_hash"],
                        },
                        environment={
                            "RETB_OWNER_FINALIST_GRAPH_ID": graph_id,
                            "RETB_CONTROL_KIND": kind,
                            "RETB_PIPELINE_SEED": str(seed),
                            "RETB_CREATED_AFTER_FINALIST_LOCK": "1",
                            "RETB_CONTROLS_MAY_REPLACE_FINALIST": "0",
                            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                        },
                    )
                )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="finalist_controls",
        rows=rows,
    )


def _publish_finalist_control_aggregate(
    *,
    root: Path,
    campaign: Mapping[str, Any],
    finalists: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    scale = load_hashed_json(root / "selection" / "scale_completion.json")
    control_paths = sorted(
        Path(path)
        for row in completion["rows"]
        for path in row["output_hashes"]
        if Path(path).name == "control_row.json"
    )
    controls = {}
    for path in control_paths:
        row = load_hashed_json(path)
        validate_hlt_capacity_control_row(row)
        if row.get("source") != campaign["source"]:
            raise ValueError("finalist-control row source differs")
        key = (
            row["owner_finalist_graph_id"],
            row["control_kind"],
            int(row["pipeline_seed"]),
        )
        if key in controls:
            raise ValueError("finalist-control row is duplicated")
        controls[key] = (row, path)
    expected = {
        (graph_id, kind, seed)
        for graph_id in finalists["finalist_graph_ids"]
        for kind in ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG")
        for seed in PIPELINE_SEEDS
    }
    if set(controls) != expected:
        raise ValueError("finalist-control completion coverage differs")
    runs = {
        (row["graph_id"], int(row["pipeline_seed"])): row
        for row in scale["runs"]
    }
    definition_registry = load_hashed_json(
        root / "selection" / "stage_l" / "graph_registry.json"
    )
    definitions = {
        row["graph_id"]: row for row in definition_registry["definitions"]
    }

    def actual_update_count(graph_id: str, seed: int) -> int:
        role = definitions[graph_id]["configuration"][
            "source_carried_shape_role"
        ]
        roots = (
            root
            / "runs"
            / "scale"
            / "refits"
            / f"seed_{seed}"
            / "roles"
            / role,
            root
            / "runs"
            / "scale"
            / "graphs"
            / graph_id
            / f"seed_{seed}",
        )
        excluded = {
            "offline_experts",
            "offline_fusion",
            "target_cache",
            "target_normalizers",
            "coordinates",
            "uncertainty_calibrations",
        }
        counts = {}
        for parent in roots:
            for path in sorted(parent.rglob("*.json")):
                if {part.lower() for part in path.parts} & excluded:
                    continue
                try:
                    payload = load_hashed_json(path)
                except (OSError, ValueError, TypeError):
                    continue
                count = int(
                    payload.get(
                        "optimizer_updates_completed",
                        payload.get("optimizer_update_counts", {}).get(
                            "total_optimizer_updates", 0
                        ),
                    )
                )
                if count > 0:
                    counts[payload["content_hash"]] = count
        total = sum(counts.values())
        if total <= 0:
            raise ValueError(
                "scale graph lacks authenticated optimizer-update evidence"
            )
        return total

    aggregate_rows = []
    optimizer_counts = {}
    for graph_id in finalists["finalist_graph_ids"]:
        baseline_id = definitions[graph_id]["named_baseline_graph_id"]
        evaluations = []
        for kind in CONTROL_KINDS:
            for seed in PIPELINE_SEEDS:
                if kind in {"FINALIST", "NAMED_BASELINE"}:
                    resolved_graph = (
                        graph_id if kind == "FINALIST" else baseline_id
                    )
                    run = runs[(resolved_graph, seed)]
                    export = load_hashed_json(
                        root
                        / "runs"
                        / "scale"
                        / "graphs"
                        / resolved_graph
                        / f"seed_{seed}"
                        / "export"
                        / "deployable_retb_graph.json"
                    )
                    checkpoint_sha = export["graph_state_sha256"]
                    optimizer_counts[
                        f"{graph_id}:{kind}:seed{seed}"
                    ] = actual_update_count(resolved_graph, seed)
                else:
                    resolved_graph = f"{kind}::{graph_id}"
                    control, _ = controls[(graph_id, kind, seed)]
                    checkpoint_sha = control["checkpoint_sha256"]
                    optimizer_counts[
                        f"{graph_id}:{kind}:seed{seed}"
                    ] = control["optimizer_updates_completed"]
                evaluations.append(
                    {
                        "row_id": f"{graph_id}:{kind}:seed{seed}",
                        "owner_finalist_graph_id": graph_id,
                        "kind": kind,
                        "graph_id": resolved_graph,
                        "pipeline_seed": seed,
                        "checkpoint_sha256": checkpoint_sha,
                    }
                )
        capacity = load_hashed_json(
            root
            / "runs"
            / "scale"
            / "graphs"
            / graph_id
            / "seed_101"
            / "export"
            / "complete_graph_capacity.json"
        )
        aggregate_rows.append(
            {
                "finalist_graph_id": graph_id,
                "named_baseline_graph_id": baseline_id,
                "complete_graph_capacity_sha256": capacity["content_hash"],
                "H_MONO_PARAM_control_sha256": canonical_sha256(
                    [
                        controls[(graph_id, "H_MONO_PARAM", seed)][0][
                            "content_hash"
                        ]
                        for seed in PIPELINE_SEEDS
                    ]
                ),
                "H_MONO_FLOP_control_sha256": canonical_sha256(
                    [
                        controls[(graph_id, "H_MONO_FLOP", seed)][0][
                            "content_hash"
                        ]
                        for seed in PIPELINE_SEEDS
                    ]
                ),
                "H_BASE_LONG_control_sha256": canonical_sha256(
                    [
                        controls[(graph_id, "H_BASE_LONG", seed)][0][
                            "content_hash"
                        ]
                        for seed in PIPELINE_SEEDS
                    ]
                ),
                "evaluation_rows": evaluations,
            }
        )
    artifact = bind_source(
        build_finalist_controls(
            locked_scale_finalists=finalists, rows=aggregate_rows
        ),
        source_snapshot=source_snapshot(Path(__file__).resolve().parents[2]),
    )
    validate_finalist_controls(
        artifact, locked_scale_finalists=finalists
    )
    write_immutable_json(
        root / "selection" / "finalist_controls.json", artifact
    )
    evidence = bind_source(
        with_content_hash(
            {
                **{
                    key: value
                    for key, value in artifact.items()
                    if key not in {"contract", "schema_version", "content_hash", "source"}
                },
                "contract": "retb_finalist_control_execution_evidence_v1",
                "schema_version": 1,
                "optimizer_update_counts": optimizer_counts,
                "actual_training_executed": True,
            }
        ),
        source_snapshot=source_snapshot(Path(__file__).resolve().parents[2]),
    )
    validate_control_evidence(
        evidence, locked_scale_finalists=finalists
    )
    write_immutable_json(
        root / "selection" / "finalist_control_execution_evidence.json",
        evidence,
    )
    return artifact


def build_final_test_execution_lock_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "stage_n_evidence_join",
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    _, control_completion = _producer_completion(
        root, producer_node_id="finalist_controls"
    )
    finalists = load_hashed_json(
        root / "selection" / "locked_scale_finalists.json"
    )
    controls = _publish_finalist_control_aggregate(
        root=root,
        campaign=campaign,
        finalists=finalists,
        completion=control_completion,
    )
    prelock = load_hashed_json(
        root / "inputs" / "stage_n" / "prelock_final_inputs.json"
    )
    postlock_completion = load_hashed_json(
        root
        / "job_ledgers"
        / "completions"
        / "postlock_oracle_targets"
        / "manifest_completion.json"
    )
    targets = sorted(
        Path(path)
        for row in postlock_completion["rows"]
        for path in row["output_hashes"]
        if Path(path).name == "postlock_target.json"
    )
    expected_targets = (
        len(finalists["finalist_graph_ids"])
        * len(PIPELINE_SEEDS)
        * 2
    )
    if len(targets) != expected_targets:
        raise ValueError("execution lock lacks postlock target coverage")
    step14 = load_hashed_json(
        root / "registry" / "retb_step14_scale_final_seal_bundle.json"
    )
    configuration = {
        "parent_hashes": {
            "campaign_spec": campaign_sha,
            "step14_bundle": step14["content_hash"],
            "locked_scale_finalists": finalists["content_hash"],
            "finalist_controls": controls["content_hash"],
            "prelock_final_test_inputs": prelock["content_hash"],
        },
        "final_input_hashes": prelock["input_hashes"],
    }
    config_path = (
        root / "selection" / "final_test_execution_lock_inputs.json"
    )
    write_immutable_bytes(
        config_path, canonical_json_bytes(configuration) + b"\n"
    )
    output = root / "selection" / "final_test_execution_lock.json"
    argv = [
        "python",
        "scripts/write_retb_final_test_execution_lock.py",
        "--campaign-root",
        str(root),
        "--locked-scale-finalists",
        str(root / "selection" / "locked_scale_finalists.json"),
        "--finalist-controls",
        str(root / "selection" / "finalist_controls.json"),
        "--prelock-final-inputs",
        str(root / "inputs" / "stage_n" / "prelock_final_inputs.json"),
        "--configuration",
        str(config_path),
        "--output",
        str(output),
    ]
    for target in targets:
        argv.extend(["--postlock-target", str(target)])
    row = _row(
        target="final_test_execution_lock",
        index=0,
        argv=argv,
        outputs=[str(output)],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "locked_scale_finalists": finalists["content_hash"],
            "finalist_controls": controls["content_hash"],
            "prelock_final_inputs": prelock["content_hash"],
            "postlock_target_set": canonical_sha256(
                [load_hashed_json(path)["content_hash"] for path in targets]
            ),
            "lock_configuration": canonical_sha256(configuration),
        },
        environment={
            "RETB_ALL_POSTLOCK_EVIDENCE_COMPLETE": "1",
            "RETB_FINAL_TEST_MODEL_OUTPUT_AUTHORIZED": "1",
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="final_test_execution_lock",
        rows=[row],
    )
def build_sealed_final_test_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "final_test_execution_lock",
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    lock = load_hashed_json(
        root / "selection" / "final_test_execution_lock.json"
    )
    output_dir = root / "evaluations" / "final_test"
    claim = output_dir / "retb_final_test_execution_claim.json"
    label_manifest = output_dir / "retb_final_test_labels.json"
    label_payload = output_dir / "retb_final_test_labels.npz"
    shared_root = root / "inputs" / "stage_n" / "shared"
    shared_manifest_path = (
        shared_root / "retb_final_test_shared_HLT_inputs.json"
    )
    shared_manifest = load_hashed_json(shared_manifest_path)
    validate_shared_deployable_inference_payload(
        shared_manifest, manifest_path=shared_manifest_path
    )
    if (
        shared_manifest.get("source") != campaign["source"]
        or lock["final_input_hashes"]["final_test_HLT_inputs"]
        != shared_manifest["content_hash"]
    ):
        raise ValueError("sealed final-test shared input lineage differs")
    steps = [
        {
            "step_id": "step_000",
            "argv": [
                "python",
                "scripts/prepare_retb_final_labels.py",
                "--campaign-root",
                str(root),
                "--execution-lock",
                str(root / "selection" / "final_test_execution_lock.json"),
                "--execution-claim",
                str(claim),
                "--output-dir",
                str(output_dir),
            ],
            "expected_outputs": [str(label_manifest), str(label_payload)],
        }
    ]
    prediction_rows = []
    for row_id, record in sorted(
        lock["eligible_evaluation_rows"].items()
    ):
        owner = record["owner_finalist_graph_id"]
        kind = record["kind"]
        graph_id = record["graph_id"]
        seed = int(record["pipeline_seed"])
        safe = row_id.replace(":", "__")
        working = output_dir / "rows" / safe
        safe_graph = graph_id.replace(":", "__")
        input_manifest = (
            shared_root / f"final_test_{safe_graph}_seed{seed}.json"
        )
        expected_input_binding = build_deployable_inference_input_binding(
            output_dir=shared_root,
            shared_payload_manifest=shared_manifest,
            shared_payload_manifest_path=shared_manifest_path,
            graph_id=graph_id,
            pipeline_seed=seed,
        )
        raw_output = working / "deployable_logits.npz"
        inference_attestation = working / "inference_attestation.json"
        prediction_manifest = working / "prediction_manifest.json"
        if kind in {"FINALIST", "NAMED_BASELINE"}:
            export = (
                root
                / "runs"
                / "scale"
                / "graphs"
                / graph_id
                / f"seed_{seed}"
                / "export"
                / "deployable_retb_graph.json"
            )
        else:
            export = (
                root
                / "runs"
                / "scale"
                / "finalist_controls"
                / owner
                / kind
                / f"seed_{seed}"
                / "deployable_control.json"
            )
        prepare_step = len(steps)
        steps.append(
            {
                "step_id": f"step_{prepare_step:03d}",
                "argv": [
                    "python",
                    "scripts/prepare_retb_deployable_input.py",
                    "--campaign-root",
                    str(root),
                    "--split",
                    "final_test",
                    "--graph-id",
                    graph_id,
                    "--pipeline-seed",
                    str(seed),
                    "--output-dir",
                    str(shared_root),
                    "--shared-payload-manifest",
                    str(shared_manifest_path),
                ],
                "expected_outputs": [str(input_manifest)],
            }
        )
        inference_step = len(steps)
        # The plan and claim paths are stable before the plan is hashed. The
        # claim itself is published by the outer executor before this command.
        plan_path = (
            root
            / "job_ledgers"
            / "execution_plans"
            / "stage_n"
            / "sealed_final_test.json"
        )
        steps.append(
            {
                "step_id": f"step_{inference_step:03d}",
                "argv": [
                    "python",
                    "scripts/run_retb_deployable_inference.py",
                    "--campaign-root",
                    str(root),
                    "--deployable-export",
                    str(export),
                    "--input-manifest",
                    str(input_manifest),
                    "--split",
                    "final_test",
                    "--graph-id",
                    graph_id,
                    "--pipeline-seed",
                    str(seed),
                    "--row-id",
                    row_id,
                    "--checkpoint-sha256",
                    record["checkpoint_sha256"],
                    "--execution-lock",
                    str(
                        root / "selection" / "final_test_execution_lock.json"
                    ),
                    "--execution-claim",
                    str(claim),
                    "--execution-plan",
                    str(plan_path),
                    "--output",
                    str(raw_output),
                    "--attestation-output",
                    str(inference_attestation),
                ],
                "expected_outputs": [
                    str(raw_output), str(inference_attestation)
                ],
            }
        )
        prediction_rows.append(
            {
                "row_id": row_id,
                "graph_id": graph_id,
                "pipeline_seed": seed,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "deployable_export": str(export),
                "deployable_export_sha256": load_hashed_json(export)[
                    "content_hash"
                ],
                "inference_output_npz": str(raw_output),
                "inference_attestation_output": str(inference_attestation),
                "prediction_manifest_output": str(prediction_manifest),
                "input_manifest": str(input_manifest),
                "input_manifest_sha256": expected_input_binding[
                    "content_hash"
                ],
                "shared_payload_manifest": str(shared_manifest_path),
                "shared_payload_manifest_sha256": shared_manifest[
                    "content_hash"
                ],
                "shared_payload_sha256": shared_manifest["payload_sha256"],
                "locked_final_test_HLT_inputs_sha256": lock[
                    "final_input_hashes"
                ]["final_test_HLT_inputs"],
            }
        )
    plan = with_content_hash(
        {
            "contract": "retb_sealed_final_test_execution_plan_v3",
            "schema_version": 3,
            "final_test_execution_lock_sha256": lock["content_hash"],
            "steps": steps,
            "final_labels_manifest": str(label_manifest),
            "prediction_rows": prediction_rows,
            "source": dict(campaign["source"]),
        }
    )
    plan_path = (
        root
        / "job_ledgers"
        / "execution_plans"
        / "stage_n"
        / "sealed_final_test.json"
    )
    write_immutable_json(plan_path, plan)
    output = output_dir / "retb_sealed_final_test_evaluation.json"
    row = _row(
        target="sealed_final_test",
        index=0,
        argv=[
            "python",
            "scripts/execute_retb_sealed_final_test.py",
            "--campaign-root",
            str(root),
            "--execution-lock",
            str(root / "selection" / "final_test_execution_lock.json"),
            "--execution-plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
        ],
        outputs=[str(output), str(claim)],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "final_test_execution_lock": lock["content_hash"],
            "execution_plan": plan["content_hash"],
            "shared_final_test_HLT_payload": shared_manifest[
                "content_hash"
            ],
        },
        environment={
            "RETB_EXACTLY_ONCE_FINAL_TEST": "1",
            "RETB_BOTH_FINAL_LOCKS_REQUIRED": "1",
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="sealed_final_test",
        rows=[row],
    )


def build_final_report_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "sealed_final_test",
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    scale = load_hashed_json(root / "selection" / "scale_completion.json")
    finalists = load_hashed_json(
        root / "selection" / "locked_scale_finalists.json"
    )
    lock = load_hashed_json(
        root / "selection" / "final_test_execution_lock.json"
    )
    final = load_hashed_json(
        root
        / "evaluations"
        / "final_test"
        / "retb_sealed_final_test_evaluation.json"
    )
    semantic = load_hashed_json(
        root / "controls" / "semantics" / "semantic_controls_bundle.json",
        expected_contract="retb_stage_k_semantic_controls_bundle_v5",
    )
    output_dir = root / "reports"
    row = _row(
        target="final_report",
        index=0,
        argv=[
            "python",
            "scripts/write_retb_step14_report.py",
            "--campaign-root",
            str(root),
            "--scale-completion",
            str(root / "selection" / "scale_completion.json"),
            "--locked-scale-finalists",
            str(root / "selection" / "locked_scale_finalists.json"),
            "--execution-lock",
            str(root / "selection" / "final_test_execution_lock.json"),
            "--final-evaluation",
            str(
                root
                / "evaluations"
                / "final_test"
                / "retb_sealed_final_test_evaluation.json"
            ),
            "--semantic-controls",
            str(
                root / "controls" / "semantics"
                / "semantic_controls_bundle.json"
            ),
            "--output-dir",
            str(output_dir),
        ],
        outputs=[
            str(output_dir / "retb_stage_mn_final_report.json"),
            str(output_dir / "retb_stage_mn_final_report.md"),
        ],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "scale_completion": scale["content_hash"],
            "locked_scale_finalists": finalists["content_hash"],
            "final_test_execution_lock": lock["content_hash"],
            "sealed_final_test_evaluation": final["content_hash"],
            "semantic_controls": semantic["content_hash"],
        },
        environment={
            "RETB_TEST_RESULT_MAY_REPLACE_FINALIST": "0",
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="final_report",
        rows=[row],
    )

FINAL_PLAN_FACTORIES: dict[str, Callable[..., dict[str, Any]]] = {
    target: globals()[f"build_{target}_manifest_plan"]
    for target in FINAL_PLAN_FACTORY_TARGETS
}


__all__ = [
    "FINAL_PLAN_FACTORIES",
    "FINAL_PLAN_FACTORY_CONTRACT",
    "FINAL_PLAN_FACTORY_INPUT_CONTRACT",
    "FINAL_PLAN_FACTORY_TARGETS",
    "build_final_factory_input",
    "final_factory_input_path",
    "publish_final_factory_input",
    "validate_final_factory_input",
    *[
        f"build_{target}_manifest_plan"
        for target in FINAL_PLAN_FACTORY_TARGETS
    ],
]
