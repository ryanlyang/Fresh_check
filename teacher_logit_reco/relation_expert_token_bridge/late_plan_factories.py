"""Authenticated Stage-K--M manifest-plan factories.

Late scientific waves have data-dependent cardinality (registered graphs and
the locked scale shortlist).  Their producer writes one source-bound factory
input beside its completion.  This module validates that input against the
campaign, production graph, producer completion, allowed worker registry, and
stage-specific coverage before publishing the downstream manifest plan.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .early_continuation import (
    _context,
    _producer_completion,
    _publish,
    _row,
)
from .production import LATE_NODE_ENTRYPOINTS
from .registry import EXPERT_ORDER, TOKEN_SHAPES


LATE_PLAN_FACTORY_INPUT_CONTRACT = "retb_stage_k_m_factory_input_v3"
LATE_PLAN_FACTORY_CONTRACT = "retb_stage_k_m_plan_factories_v4"
LATE_PLAN_FACTORY_TARGETS = (
    "robustness_controls",
    "semantic_controls",
    "stage_l_graph_registration",
    "confirmation_500k",
    "confirmation_summary",
    "bridge_shape_selector",
    "scale_shortlist_selector",
    "shortlisted_500k_control_training",
    "shortlisted_500k_controls",
    "scale_refit_normalizers",
    "scale_refit_teachers",
    "scale_refit_offline_experts",
    "scale_refit_targets",
    "scale_refit_native",
    "scale_refit_native_fusion",
    "scale_refit_predictors",
    "scale_refit_calibrations",
    "scale_refits",
    "scale_joint_training",
    "scale_graph_datasets",
    "scale_refiner_training",
    "scale_final_consumer_training",
    "scale_graph_export",
    "scale_graph_training",
    "scale_completion",
)
PIPELINE_SEEDS = (101, 202, 303)
ROBUSTNESS_PROFILES = (
    "D_OFFLINE_IDENTITY",
    "D_KIN_ONLY",
    "D_TRACK_ONLY",
    "D_MISSING_ONLY",
    "D_MILD",
    "D_NOMINAL",
    "D_SEVERE",
    "D_LEGACY_V1",
    "D_LEGACY_V2",
)
ROBUSTNESS_REPLICAS = (0, 1, 2, 3)
SEMANTIC_CONTROL_KINDS = (
    "RELATION_ZERO",
    "RELATION_WITHIN_JET_SHUFFLE",
    "RELATION_WRONG_EVENT_MATCHED_MULTIPLICITY",
    "RELATION_DIRECTIONAL_ENDPOINT_SWAP",
    "RELATION_FIXED_VS_LEARNED_SCALE",
    "TOKEN_SLOT_PERMUTE",
    "TOKEN_WITHIN_CLASS_MEAN_BANK",
    "TOKEN_WRONG_EVENT_SAME_CLASS_BANK",
    "TOKEN_WRONG_EVENT_UNMATCHED_BANK",
    "TOKEN_ZERO_BANK",
    "TOKEN_MATCHED_GAUSSIAN_NOISE",
    "TOKEN_NORM_COVARIANCE",
    "PREDICTOR_ZERO_HLT_EVIDENCE",
    "PREDICTOR_SHUFFLE_HLT_EVIDENCE",
    "PREDICTOR_REMOVE_NATIVE_PARTICLE_CONTEXT",
    "PREDICTOR_REMOVE_NONCORRESPONDING_BANKS",
    "PREDICTOR_DIRECT_VS_GATED",
    "PREDICTOR_W_LOGIT_ONLY",
    "BYPASS_NATIVE_REMOVED",
    "BYPASS_RECONSTRUCTED_REMOVED",
    "BYPASS_NATIVE_DROPPED_EVAL",
    "BYPASS_GAMMA_ZERO",
    "BYPASS_SOURCE_EMBEDDINGS_SWAPPED",
)


def late_factory_input_path(
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


def producer_plan_identity_sha256(
    *,
    producer_node_id: str,
    production_graph: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> str:
    """Return the non-circular producer identity bound by a factory input."""

    if completion.get("contract") == "retb_direct_node_completion_v1":
        node = next(
            row
            for row in production_graph["nodes"]
            if row["node_id"] == producer_node_id
        )
        execution = next(
            row
            for row in production_graph["node_execution_registry"]["entries"]
            if row["node_id"] == producer_node_id
        )
        return canonical_sha256(
            {
                "kind": "direct_worker_plan_identity_v1",
                "node": node,
                "execution": execution,
            }
        )
    return require_sha256(
        completion["task_manifest_sha256"],
        name="producer_task_manifest_sha256",
    )


def build_late_factory_input(
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
    if target not in LATE_PLAN_FACTORY_TARGETS or not rows:
        raise ValueError("Stage-K--M factory-input target/rows differ")
    checked = []
    allowed = set(LATE_NODE_ENTRYPOINTS[target])
    for index, raw in enumerate(rows):
        required = {
            "task_id",
            "argv",
            "expected_outputs",
            "input_artifact_hashes",
            "environment",
        }
        if set(raw) != required:
            raise ValueError("Stage-K--M factory-input row fields differ")
        argv = [str(value) for value in raw["argv"]]
        scripts = {
            value.replace("\\", "/")
            for value in argv
            if value.replace("\\", "/").startswith("scripts/")
        }
        lowered = " ".join(argv).lower()
        if (
            str(raw["task_id"]) != f"{target}:{index}"
            or len(scripts & allowed) != 1
            or "--dry-run" in argv
            or "final_test" in lowered
            or (target != "scale_completion" and "stack_val" in lowered)
            or not raw["expected_outputs"]
        ):
            raise ValueError("Stage-K--M factory-input row semantics differ")
        hashes = dict(raw["input_artifact_hashes"])
        if (
            hashes.get("campaign_spec") != campaign_spec_sha256
            or hashes.get("production_graph") != production_graph_sha256
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in hashes.values()
            )
        ):
            raise ValueError("Stage-K--M row lineage differs")
        environment = {
            str(name): str(value)
            for name, value in raw["environment"].items()
        }
        if environment.get(
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION"
        ) != "0":
            raise ValueError("Stage-K--M performance gate differs")
        checked.append(
            {
                "task_id": str(raw["task_id"]),
                "argv": argv,
                "expected_outputs": [
                    str(Path(value)) for value in raw["expected_outputs"]
                ],
                "input_artifact_hashes": hashes,
                "environment": environment,
            }
        )
    coverage_row = dict(coverage)
    if (
        coverage_row.get("all_predeclared_rows_present") is not True
        or coverage_row.get("scientific_metric_used_for_membership") is not False
        or coverage_row.get("incomplete_wave_permitted") is not False
    ):
        raise ValueError("Stage-K--M coverage attestation differs")
    if target == "robustness_controls":
        required = {
            (profile, replica)
            for profile in ROBUSTNESS_PROFILES
            for replica in ROBUSTNESS_REPLICAS
        }
        observed = {
            (
                row["environment"].get("RETB_DEGRADATION_PROFILE"),
                int(row["environment"].get("RETB_DEGRADATION_REPLICA", -1)),
            )
            for row in checked
        }
        declared = {
            (str(row[0]), int(row[1]))
            for row in coverage_row.get(
                "required_profile_replica_coordinates", []
            )
        }
        if not required <= (observed | declared):
            raise ValueError("Stage-K robustness profile/replica coverage differs")
    elif target == "semantic_controls":
        observed = {
            row["environment"].get("RETB_SEMANTIC_CONTROL_KIND")
            for row in checked
            if row["environment"].get("RETB_SEMANTIC_CONTROL_KIND")
            is not None
        }
        declared = set(coverage_row.get("required_semantic_control_kinds", []))
        if observed | declared != set(SEMANTIC_CONTROL_KINDS):
            raise ValueError("Stage-K semantic-control coverage differs")
    elif target in {"confirmation_500k", "scale_graph_training"}:
        coordinates = {
            (
                row["environment"].get("RETB_GRAPH_ID"),
                int(row["environment"].get("RETB_PIPELINE_SEED", -1)),
            )
            for row in checked
        }
        graph_ids = {
            str(value) for value in coverage_row.get("required_graph_ids", [])
        }
        if coordinates != {
            (graph_id, seed)
            for graph_id in graph_ids
            for seed in PIPELINE_SEEDS
        }:
            raise ValueError("Stage-K--M graph/seed coverage differs")
    elif target == "scale_refits":
        seeds = {
            int(row["environment"].get("RETB_PIPELINE_SEED", -1))
            for row in checked
        }
        if seeds != set(PIPELINE_SEEDS) or len(checked) != len(
            PIPELINE_SEEDS
        ):
            raise ValueError("Stage-M shared refit seed coverage differs")
    elif len(checked) != 1:
        raise ValueError(f"{target} requires exactly one aggregation/lock row")
    return with_content_hash(
        {
            "contract": LATE_PLAN_FACTORY_INPUT_CONTRACT,
            "schema_version": 3,
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


def validate_late_factory_input(
    payload: Mapping[str, Any],
    *,
    target_node_id: str,
    producer_node_id: str,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_task_manifest_sha256: str,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=LATE_PLAN_FACTORY_INPUT_CONTRACT
    )
    expected = build_late_factory_input(
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
        raise ValueError("Stage-K--M factory-input semantics differ")
    return digest


def publish_late_factory_input(
    *,
    campaign_root: str | Path,
    payload: Mapping[str, Any],
    target_node_id: str,
    producer_node_id: str,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_task_manifest_sha256: str,
) -> dict[str, Any]:
    """Publish a producer-generated input before its row completion is sealed."""

    validate_late_factory_input(
        payload,
        target_node_id=target_node_id,
        producer_node_id=producer_node_id,
        campaign=campaign,
        production_graph=production_graph,
        producer_task_manifest_sha256=producer_task_manifest_sha256,
    )
    return write_immutable_json(
        late_factory_input_path(
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
    input_path = late_factory_input_path(root, target_node_id=target)
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    authenticated_outputs = {
        str(Path(path).resolve()): digest
        for row in completion["rows"]
        for path, digest in row["output_hashes"].items()
    }
    resolved_input = str(input_path.resolve())
    if (
        resolved_input not in authenticated_outputs
        or not input_path.is_file()
        or input_path.is_symlink()
        or _file_sha256(input_path) != authenticated_outputs[resolved_input]
    ):
        raise ValueError(
            "Stage-K--M factory input is not an authenticated producer output"
        )
    payload = load_hashed_json(
        input_path,
        expected_contract=LATE_PLAN_FACTORY_INPUT_CONTRACT,
    )
    input_sha = validate_late_factory_input(
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
            for name, digest in source_row["input_artifact_hashes"].items()
            if name
            not in {"campaign_spec", "production_graph"}
        }
        extra["late_factory_input"] = input_sha
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


def _factory(target: str, default_producer: str) -> Callable[..., dict[str, Any]]:
    def build(
        *,
        campaign_root: str | Path,
        campaign: Mapping[str, Any],
        production_graph: Mapping[str, Any],
        producer_node_id: str = default_producer,
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


build_robustness_controls_manifest_plan = _factory(
    "robustness_controls", "deployable_export"
)
build_semantic_controls_manifest_plan = _factory(
    "semantic_controls", "deployable_export"
)
build_stage_l_graph_registration_manifest_plan = _factory(
    "stage_l_graph_registration", "step13_confirmation_contracts"
)
build_confirmation_500k_manifest_plan = _factory(
    "confirmation_500k", "stage_l_graph_registration"
)
def build_confirmation_summary_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "confirmation_500k",
) -> dict[str, Any]:
    """Derive the complete aggregation row from authenticated array outputs."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    confirmation_paths = sorted(
        Path(path)
        for row in completion["rows"]
        for path in row["output_hashes"]
        if Path(path).name == "confirmation.json"
    )
    registry = load_hashed_json(
        root / "selection" / "stage_l" / "graph_registry.json",
        expected_contract="retb_stage_l_graph_registry_v2",
    )
    expected_count = int(registry["definition_count"]) * len(PIPELINE_SEEDS)
    if len(confirmation_paths) != expected_count:
        raise ValueError("500k confirmation completion coverage differs")
    confirmation_hashes = {
        f"seed_confirmation_{index:03d}": require_sha256(
            next(
                digest
                for row in completion["rows"]
                for path, digest in row["output_hashes"].items()
                if Path(path) == confirmation_path
            ),
            name=f"seed_confirmation_{index:03d}",
        )
        for index, confirmation_path in enumerate(confirmation_paths)
    }
    first = load_hashed_json(
        confirmation_paths[0],
        expected_contract="retb_500k_seed_confirmation_v1",
    )
    output = root / "selection" / "stage_l" / "confirmation_summary.json"
    shape_config = (
        root / "selection" / "stage_l" / "bridge_shape_configuration.json"
    )
    next_input = late_factory_input_path(
        root, target_node_id="bridge_shape_selector"
    )
    argv = [
        "python",
        "scripts/aggregate_retb_confirmation.py",
        "--campaign-root",
        str(root),
        "--graph-registry",
        str(root / "selection" / "stage_l" / "graph_registry.json"),
        "--val-design-label-manifest-sha256",
        first["val_design_label_manifest_sha256"],
        "--output",
        str(output),
    ]
    for path in confirmation_paths:
        argv.extend(["--seed-confirmation", str(path)])
    row = _row(
        target="confirmation_summary",
        index=0,
        argv=argv,
        outputs=[str(output), str(shape_config), str(next_input)],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "stage_l_graph_registry": registry["content_hash"],
            **confirmation_hashes,
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="confirmation_summary",
        rows=[row],
    )
build_bridge_shape_selector_manifest_plan = _factory(
    "bridge_shape_selector", "confirmation_summary"
)
def build_scale_shortlist_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "bridge_shape_selector",
) -> dict[str, Any]:
    """Create the shortlist selector only after the shape lock completes."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    shape_path = (
        root / "selection" / "stage_l" / "bridge_shape_selection.json"
    )
    shape = load_hashed_json(
        shape_path, expected_contract="retb_bridge_shape_selection_v1"
    )
    registry = load_hashed_json(
        root / "selection" / "stage_l" / "graph_registry.json",
        expected_contract="retb_stage_l_graph_registry_v2",
    )
    summary = load_hashed_json(
        root / "selection" / "stage_l" / "confirmation_summary.json",
        expected_contract="retb_500k_confirmation_summary_v2",
    )
    step13 = load_hashed_json(
        root / "registry" / "retb_step13_confirmation_shortlist_bundle.json"
    )
    normalizers = load_hashed_json(
        root / "inputs" / "normalization" / "stage_a_normalizer_bundle.json"
    )
    profile = load_hashed_json(root / "inputs" / "hlt_v3_profile.json")
    predictor = load_hashed_json(
        root / "selection" / "predictor_bundle" / "predictor_bundle_lock.json"
    )
    sample_definition = registry["definitions"][0]
    sample_run = load_hashed_json(
        root
        / "runs"
        / "final_consumers"
        / sample_definition["configuration"]["run_ids_by_seed"]["101"]
        / "assets"
        / "run.json"
    )
    parents = {
        "campaign_spec": campaign_sha,
        "step12_bundle": registry["step12_bundle_sha256"],
        "step13_bundle": step13["content_hash"],
        "graph_registry": registry["content_hash"],
        "confirmation_summary": summary["content_hash"],
        "bridge_shape_selection": shape["content_hash"],
        "validation_partition_manifest": campaign[
            "parent_artifact_hashes"
        ]["validation_partition_manifest"],
        "val_design_identity_manifest": sample_run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        "val_design_label_manifest": summary[
            "val_design_label_manifest_sha256"
        ],
        "hlt_replica_manifest": campaign["parent_artifact_hashes"][
            "hlt_replica_manifest"
        ],
        "degradation_profile": profile["content_hash"],
        "offline_normalizer_bundle": normalizers["content_hash"],
        "shared_hlt_normalizer_bundle": normalizers["content_hash"],
        "predictor_bundle_lock": predictor["content_hash"],
    }
    parent_artifact = with_content_hash(
        {
            "contract": "retb_scale_shortlist_parent_input_v1",
            "schema_version": 1,
            "parent_hashes": parents,
            "source": dict(campaign["source"]),
        }
    )
    parent_path = (
        root / "selection" / "stage_l" / "scale_shortlist_parents.json"
    )
    write_immutable_json(parent_path, parent_artifact)
    output = root / "selection" / "locked_scale_shortlist.json"
    row = _row(
        target="scale_shortlist_selector",
        index=0,
        argv=[
            "python",
            "scripts/select_retb_scale_shortlist.py",
            "--campaign-root",
            str(root),
            "--graph-registry",
            str(root / "selection" / "stage_l" / "graph_registry.json"),
            "--confirmation-summary",
            str(root / "selection" / "stage_l" / "confirmation_summary.json"),
            "--bridge-shape-selection",
            str(shape_path),
            "--parent-hashes",
            str(parent_path),
            "--output",
            str(output),
        ],
        outputs=[str(output)],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "bridge_shape_selection": shape["content_hash"],
            "confirmation_summary": summary["content_hash"],
            "graph_registry": registry["content_hash"],
            "parent_input": parent_artifact["content_hash"],
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="scale_shortlist_selector",
        rows=[row],
    )


def build_shortlisted_500k_control_training_manifest_plan(
    *, campaign_root: str | Path, campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "scale_shortlist_selector",
) -> dict[str, Any]:
    """Train three real controls for every shortlist graph and matched seed."""
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root, campaign=campaign,
        production_graph=production_graph, producer_node_id=producer_node_id,
    )
    shortlist_path = root / "selection" / "locked_scale_shortlist.json"
    shortlist = load_hashed_json(
        shortlist_path, expected_contract="retb_locked_scale_shortlist_v2"
    )
    rows = []
    for graph_id in shortlist["SCALE_SHORTLIST"]:
        definition_sha = shortlist["locked_graph_definitions"][graph_id][
            "complete_graph_definition_sha256"
        ]
        for kind in ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG"):
            for seed in PIPELINE_SEEDS:
                output = (
                    root / "controls" / "shortlisted_500k" / "runs"
                    / graph_id / kind / f"seed_{seed}"
                )
                rows.append(_row(
                    target="shortlisted_500k_control_training",
                    index=len(rows),
                    argv=[
                        "python", "scripts/train_retb_scale_finalist_control.py",
                        "--campaign-root", str(root),
                        "--locked-scale-shortlist", str(shortlist_path),
                        "--owner-finalist-graph-id", graph_id,
                        "--control-kind", kind,
                        "--pipeline-seed", str(seed),
                        "--output-dir", str(output),
                    ],
                    outputs=[
                        str(output / "control_row.json"),
                        str(output / "deployable_control.json"),
                        str(output / "training" / "registration.json"),
                        str(output / "training" / "val_design_metrics.json"),
                    ],
                    campaign_sha256=campaign_sha, graph_sha256=graph_sha,
                    producer_completion_sha256=completion_sha,
                    extra_input_hashes={
                        "locked_scale_shortlist": shortlist["content_hash"],
                        "graph_definition": definition_sha,
                    },
                    environment={
                        "RETB_GRAPH_ID": graph_id,
                        "RETB_CONTROL_KIND": kind,
                        "RETB_PIPELINE_SEED": str(seed),
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                    },
                ))
    return _publish(
        campaign_root=root, campaign=campaign,
        production_graph=production_graph, producer_node_id=producer_node_id,
        target_node_id="shortlisted_500k_control_training", rows=rows,
    )


def build_shortlisted_500k_controls_manifest_plan(
    *, campaign_root: str | Path, campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "shortlisted_500k_control_training",
) -> dict[str, Any]:
    """Aggregate authenticated training outputs and publish the Stage-L report."""
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root, campaign=campaign,
        production_graph=production_graph, producer_node_id=producer_node_id,
    )
    _, completion = _producer_completion(root, producer_node_id=producer_node_id)
    control_paths = sorted(
        Path(path) for row in completion["rows"]
        for path in row["output_hashes"] if Path(path).name == "control_row.json"
    )
    shortlist_path = root / "selection" / "locked_scale_shortlist.json"
    shortlist = load_hashed_json(
        shortlist_path, expected_contract="retb_locked_scale_shortlist_v2"
    )
    if len(control_paths) != len(shortlist["SCALE_SHORTLIST"]) * 9:
        raise ValueError("shortlisted 500k control completion coverage differs")
    output = root / "selection" / "stage_l" / "shortlisted_500k_controls.json"
    argv = [
        "python", "scripts/aggregate_retb_shortlisted_500k_controls.py",
        "--campaign-root", str(root),
        "--locked-scale-shortlist", str(shortlist_path),
        "--confirmation-summary",
        str(root / "selection" / "stage_l" / "confirmation_summary.json"),
        "--output", str(output),
        "--report-output-dir", str(root / "reports" / "stage_l"),
    ]
    for path in control_paths:
        argv.extend(["--control-row", str(path)])
    row = _row(
        target="shortlisted_500k_controls", index=0, argv=argv,
        outputs=[
            str(output),
            str(root / "reports" / "stage_l" / "retb_stage_l_report.json"),
            str(root / "reports" / "stage_l" / "retb_stage_l_report.md"),
        ],
        campaign_sha256=campaign_sha, graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "locked_scale_shortlist": shortlist["content_hash"],
            "control_rows": canonical_sha256(
                [load_hashed_json(path)["content_hash"] for path in control_paths]
            ),
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root, campaign=campaign,
        production_graph=production_graph, producer_node_id=producer_node_id,
        target_node_id="shortlisted_500k_controls", rows=[row],
    )


def _build_scale_refit_phase_manifest_plan(
    *, target_node_id: str, stop_after: str, producer_node_id: str,
    campaign_root: str | Path, campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, Any]:
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root, campaign=campaign,
        production_graph=production_graph, producer_node_id=producer_node_id,
    )
    shortlist_path = root / "selection" / "locked_scale_shortlist.json"
    shortlist = load_hashed_json(
        shortlist_path, expected_contract="retb_locked_scale_shortlist_v2"
    )
    definitions = shortlist["locked_graph_definitions"]
    roles = sorted({
        definitions[graph_id]["configuration"]["source_carried_shape_role"]
        for graph_id in shortlist.get(
            "SCALE_TRAINING_GRAPHS", shortlist["SCALE_SHORTLIST"]
        )
    })

    def named_shape(k: int, d: int) -> str:
        matches = [
            name for name, row in TOKEN_SHAPES.items()
            if int(row["K"]) == int(k) and int(row["D"]) == int(d)
        ]
        if len(matches) != 1:
            raise ValueError("scale-refit factory shape is unregistered")
        return matches[0]

    if stop_after in {"normalizers", "teachers"}:
        components: list[str | None] = [None]
    elif stop_after == "offline_experts":
        pairs = set()
        for role in roles:
            lock = load_hashed_json(
                root / "selection" / "predictor_bundle" / "carried"
                / f"{role}.json"
            )
            source_shape = str(lock["coordinate_id"]).split(":", 1)[1]
            for expert in EXPERT_ORDER:
                pairs.add((expert, source_shape))
                K, D = lock["allocation"][expert]
                pairs.add((expert, named_shape(int(K), int(D))))
        components = [f"{expert}@{shape}" for expert, shape in sorted(pairs)]
    elif stop_after in {"targets", "native_fusion"}:
        components = list(roles)
    elif stop_after in {"native", "predictors", "calibrations"}:
        components = [
            f"{role}@{expert}" for role in roles for expert in EXPERT_ORDER
        ]
    else:
        raise ValueError("scale-refit factory phase is unregistered")

    rows = []
    for seed in PIPELINE_SEEDS:
        for component_id in components:
            marker = f"{stop_after}.json"
            if component_id is not None:
                suffix = hashlib.sha256(
                    component_id.encode("utf-8")
                ).hexdigest()[:16]
                marker = f"{stop_after}__{suffix}.json"
            output = root / "runs" / "scale" / "refits" / f"seed_{seed}"
            expected = [output / "phase_completions" / marker]
            argv = [
                "python", "scripts/execute_retb_scale_seed_refit.py",
                "--campaign-root", str(root),
                "--locked-scale-shortlist", str(shortlist_path),
                "--pipeline-seed", str(seed),
                "--output-dir", str(output),
                "--stop-after", stop_after,
            ]
            if component_id is not None:
                argv.extend(["--component-id", component_id])
            rows.append(
                _row(
                    target=target_node_id,
                    index=len(rows),
                    argv=argv,
                    outputs=[str(path) for path in expected],
                    campaign_sha256=campaign_sha,
                    graph_sha256=graph_sha,
                    producer_completion_sha256=completion_sha,
                    extra_input_hashes={
                        "locked_scale_shortlist": shortlist["content_hash"],
                        "phase_recipe": canonical_sha256(
                            {
                                "phase": stop_after,
                                "pipeline_seed": seed,
                                "component_id": component_id,
                            }
                        ),
                    },
                    environment={
                        "RETB_PIPELINE_SEED": str(seed),
                        "RETB_SCALE_COMPONENT_PHASE": stop_after,
                        "RETB_SCALE_COMPONENT_ID": component_id or "ALL",
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                    },
                )
            )
    return _publish(
        campaign_root=root, campaign=campaign,
        production_graph=production_graph, producer_node_id=producer_node_id,
        target_node_id=target_node_id, rows=rows,
    )


def build_scale_refit_normalizers_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_normalizers", stop_after="normalizers",
        producer_node_id=kwargs.pop("producer_node_id", "step14_scale_final_contracts"),
        **kwargs,
    )


def build_scale_refit_teachers_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_teachers", stop_after="teachers",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refit_normalizers"),
        **kwargs,
    )


def build_scale_refit_offline_experts_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_offline_experts", stop_after="offline_experts",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refit_teachers"),
        **kwargs,
    )


def build_scale_refit_targets_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_targets", stop_after="targets",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refit_offline_experts"),
        **kwargs,
    )


def build_scale_refit_native_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_native", stop_after="native",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refit_targets"),
        **kwargs,
    )


def build_scale_refit_native_fusion_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_native_fusion", stop_after="native_fusion",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refit_native"),
        **kwargs,
    )


def build_scale_refit_predictors_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_predictors", stop_after="predictors",
        producer_node_id=kwargs.pop(
            "producer_node_id", "scale_refit_native_fusion"
        ),
        **kwargs,
    )


def build_scale_refit_calibrations_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_refit_phase_manifest_plan(
        target_node_id="scale_refit_calibrations", stop_after="calibrations",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refit_predictors"),
        **kwargs,
    )


def build_scale_refits_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "scale_refit_calibrations",
) -> dict[str, Any]:
    """Train shared scale components once per seed, never once per graph."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    shortlist_path = root / "selection" / "locked_scale_shortlist.json"
    shortlist = load_hashed_json(
        shortlist_path, expected_contract="retb_locked_scale_shortlist_v2"
    )
    step14 = load_hashed_json(
        root / "registry" / "retb_step14_scale_final_seal_bundle.json"
    )
    rows = []
    for index, seed in enumerate(PIPELINE_SEEDS):
        output = root / "runs" / "scale" / "refits" / f"seed_{seed}"
        expected = [output / "scale_seed_refit_index.json"]
        expected.extend(
            output / directory / f"{graph_id}.json"
            for graph_id in shortlist["SCALE_TRAINING_GRAPHS"]
            for directory in ("component_indexes", "refit_bundles")
        )
        rows.append(
            _row(
                target="scale_refits",
                index=index,
                argv=[
                    "python",
                    "scripts/execute_retb_scale_seed_refit.py",
                    "--campaign-root",
                    str(root),
                    "--locked-scale-shortlist",
                    str(shortlist_path),
                    "--pipeline-seed",
                    str(seed),
                    "--output-dir",
                    str(output),
                    "--stop-after",
                    "complete",
                ],
                outputs=[str(path) for path in expected],
                campaign_sha256=campaign_sha,
                graph_sha256=graph_sha,
                producer_completion_sha256=completion_sha,
                extra_input_hashes={
                    "locked_scale_shortlist": shortlist["content_hash"],
                    "step14_bundle": step14["content_hash"],
                },
                environment={
                    "RETB_PIPELINE_SEED": str(seed),
                    "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                },
            )
        )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="scale_refits",
        rows=rows,
    )


def _build_scale_graph_phase_manifest_plan(
    *,
    target_node_id: str,
    stop_after: str,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str,
) -> dict[str, Any]:
    """Derive every graph/seed row from complete authenticated seed refits."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    _, completion = _producer_completion(
        root, producer_node_id="scale_refits"
    )
    indexes = [
        Path(path)
        for row in completion["rows"]
        for path in row["output_hashes"]
        if Path(path).name == "scale_seed_refit_index.json"
    ]
    if len(indexes) != len(PIPELINE_SEEDS):
        raise ValueError("scale seed-refit completion coverage differs")
    shortlist_path = root / "selection" / "locked_scale_shortlist.json"
    shortlist = load_hashed_json(
        shortlist_path, expected_contract="retb_locked_scale_shortlist_v2"
    )
    by_seed = {}
    for path in indexes:
        payload = load_hashed_json(
            path, expected_contract="retb_scale_seed_refit_index_v2"
        )
        seed = int(payload["pipeline_seed"])
        if (
            seed in by_seed
            or payload.get("source") != campaign["source"]
            or payload["locked_scale_shortlist_sha256"]
            != shortlist["content_hash"]
            or set(payload["component_indexes"])
            != set(shortlist["SCALE_TRAINING_GRAPHS"])
            or set(payload["refit_bundles"])
            != set(shortlist["SCALE_TRAINING_GRAPHS"])
        ):
            raise ValueError("scale seed-refit index semantics differ")
        by_seed[seed] = payload
    if set(by_seed) != set(PIPELINE_SEEDS):
        raise ValueError("scale seed-refit seed coverage differs")
    rows = []
    for graph_id in shortlist["SCALE_TRAINING_GRAPHS"]:
        for seed in PIPELINE_SEEDS:
            index = by_seed[seed]
            component = index["component_indexes"][graph_id]
            refit = index["refit_bundles"][graph_id]
            output = (
                root
                / "runs"
                / "scale"
                / "graphs"
                / graph_id
                / f"seed_{seed}"
            )
            rows.append(
                _row(
                    target="scale_graph_training",
                    index=len(rows),
                    argv=[
                        "python",
                        "scripts/execute_retb_scale_graph_pipeline.py",
                        "--campaign-root",
                        str(root),
                        "--locked-scale-shortlist",
                        str(shortlist_path),
                        "--component-index",
                        component["path"],
                        "--scale-refit-bundle",
                        refit["path"],
                        "--graph-id",
                        graph_id,
                        "--pipeline-seed",
                        str(seed),
                        "--output-dir",
                        str(output),
                        "--stop-after",
                        stop_after,
                    ],
                    outputs=(
                        [
                            str(output / "scale_graph_run.json"),
                            str(output / "pre_stack_metrics.json"),
                            str(output / "export" / "deployable_retb_graph.json"),
                            str(output / "export" / "complete_graph_capacity.json"),
                            str(output / "export" / "research_graph_parity.json"),
                        ]
                        if stop_after == "complete"
                        else [
                            str(output / "phase_completions" / f"{stop_after}.json")
                        ]
                    ),
                    campaign_sha256=campaign_sha,
                    graph_sha256=graph_sha,
                    producer_completion_sha256=completion_sha,
                    extra_input_hashes={
                        "locked_scale_shortlist": shortlist["content_hash"],
                        "scale_component_index": component["content_hash"],
                        "scale_refit_bundle": refit["content_hash"],
                    },
                    environment={
                        "RETB_GRAPH_ID": graph_id,
                        "RETB_PIPELINE_SEED": str(seed),
                        "RETB_SCALE_GRAPH_PHASE": stop_after,
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                    },
                )
            )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target_node_id,
        rows=rows,
    )


def build_scale_joint_training_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_graph_phase_manifest_plan(
        target_node_id="scale_joint_training", stop_after="joint",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refits"), **kwargs,
    )


def build_scale_graph_datasets_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_graph_phase_manifest_plan(
        target_node_id="scale_graph_datasets", stop_after="datasets",
        producer_node_id=kwargs.pop("producer_node_id", "scale_joint_training"), **kwargs,
    )


def build_scale_refiner_training_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_graph_phase_manifest_plan(
        target_node_id="scale_refiner_training", stop_after="refiner",
        producer_node_id=kwargs.pop("producer_node_id", "scale_graph_datasets"), **kwargs,
    )


def build_scale_final_consumer_training_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_graph_phase_manifest_plan(
        target_node_id="scale_final_consumer_training", stop_after="final_consumer",
        producer_node_id=kwargs.pop("producer_node_id", "scale_refiner_training"), **kwargs,
    )


def build_scale_graph_export_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_graph_phase_manifest_plan(
        target_node_id="scale_graph_export", stop_after="export",
        producer_node_id=kwargs.pop("producer_node_id", "scale_final_consumer_training"), **kwargs,
    )


def build_scale_graph_training_manifest_plan(**kwargs: Any) -> dict[str, Any]:
    return _build_scale_graph_phase_manifest_plan(
        target_node_id="scale_graph_training", stop_after="complete",
        producer_node_id=kwargs.pop("producer_node_id", "scale_graph_export"), **kwargs,
    )


def build_scale_completion_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "scale_graph_training",
) -> dict[str, Any]:
    """Aggregate only after all locked graph/seed artifacts authenticate."""

    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    run_paths = sorted(
        Path(path)
        for row in completion["rows"]
        for path in row["output_hashes"]
        if Path(path).name == "scale_graph_run.json"
    )
    shortlist = load_hashed_json(
        root / "selection" / "locked_scale_shortlist.json",
        expected_contract="retb_locked_scale_shortlist_v2",
    )
    coordinates = set()
    hashes = {}
    for index, path in enumerate(run_paths):
        run = load_hashed_json(path, expected_contract="retb_scale_graph_run_v1")
        coordinates.add((run["graph_id"], int(run["pipeline_seed"])))
        hashes[f"scale_graph_run_{index:03d}"] = run["content_hash"]
    expected = {
        (graph_id, seed)
        for graph_id in shortlist["SCALE_TRAINING_GRAPHS"]
        for seed in PIPELINE_SEEDS
    }
    if coordinates != expected or len(run_paths) != len(expected):
        raise ValueError("scale graph completion coverage differs")
    output = root / "selection" / "scale_completion.json"
    argv = [
        "python",
        "scripts/aggregate_retb_scale_completion.py",
        "--campaign-root",
        str(root),
        "--locked-scale-shortlist",
        str(root / "selection" / "locked_scale_shortlist.json"),
        "--scale-train-manifest-sha256",
        campaign["parent_artifact_hashes"]["scale_train_manifest"],
        "--output",
        str(output),
    ]
    for path in run_paths:
        argv.extend(["--scale-run", str(path)])
    row = _row(
        target="scale_completion",
        index=0,
        argv=argv,
        outputs=[str(output)],
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "locked_scale_shortlist": shortlist["content_hash"],
            **hashes,
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id="scale_completion",
        rows=[row],
    )

LATE_PLAN_FACTORIES: dict[str, Callable[..., dict[str, Any]]] = {
    target: globals()[f"build_{target}_manifest_plan"]
    for target in LATE_PLAN_FACTORY_TARGETS
}


__all__ = [
    "LATE_PLAN_FACTORIES",
    "LATE_PLAN_FACTORY_CONTRACT",
    "LATE_PLAN_FACTORY_INPUT_CONTRACT",
    "LATE_PLAN_FACTORY_TARGETS",
    "ROBUSTNESS_PROFILES",
    "ROBUSTNESS_REPLICAS",
    "SEMANTIC_CONTROL_KINDS",
    "build_late_factory_input",
    "late_factory_input_path",
    "publish_late_factory_input",
    "producer_plan_identity_sha256",
    "validate_late_factory_input",
    *[
        f"build_{target}_manifest_plan"
        for target in LATE_PLAN_FACTORY_TARGETS
    ],
]
