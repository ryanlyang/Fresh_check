"""Automatic, source-bound command factories for the HOSD production DAG.

The runtime manifest may bind infrastructure paths and ordinary resource
options.  It may not supply scientific row IDs, graph IDs, selectors, seeds,
or result lists; those are derived from immutable campaign plans and locks by
the registered factory.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import string
from typing import Any, Mapping, Sequence

from .contracts import (
    NODE_FACTORY_REGISTRY_CONTRACT,
    RUNTIME_MANIFEST_CONTRACT,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .registry import validate_stage_job_registry


FACTORY_ENTRYPOINT = "scripts/run_hosd_node_factory.py"

# Hard coordinate counts are infrastructure upper bounds, never
# performance-dependent pruning. Coordinates beyond a dynamically compiled
# row set attest an inactive slot and do not replace a scientific row.
NODE_COORDINATE_LIMITS = {
    "offline_teacher_train": 2,
    "canonical_target_build": 3,
    "hlt_analogue_target_build": 6,
    "teacher_target_inference": 6,
    "residual_target_build": 6,
    "baseline_train": 7,
    "probe_input_materialization": 162,
    "probe_train": 162,
    "auxiliary_train": 110,
    "relation_het_auxiliary_train": 3,
    "hlt_self_auxiliary_train": 13,
    "auxiliary_controls": 90,
    "feedback_train": 46,
    "feedback_controls": 46,
    "combination_beam": 1,
    "combination_train": 10,
    "pcgrad_control": 1,
    "robustness_evaluation": 270,
    "robustness_cache_build": 36,
    "discovery_export": 16,
    "confirmation_train": 48,
    "capacity_controls": 48,
    "scale_input_prepare": 5,
    "scale_tree_build": 5,
    "scale_teacher_train": 2,
    "scale_teacher_target_inference": 2,
    "scale_target_build": 31,
    "scale_native_relation_build": 4,
    "scale_graph_train": 21,
    "scale_efficiency": 21,
    "stack_inference": 21,
}

# A real miniature must traverse every registered logical coordinate, but it
# does not need one Slurm allocation per tiny coordinate.  Coordinates are
# executed serially in immutable batches inside one allocation.  Production
# remains one scientific coordinate per allocation for independent recovery.
MINIATURE_COORDINATE_SPAN = 16

FORBIDDEN_RUNTIME_OPTIONS = frozenset(
    {
        "--baseline-id",
        "--beam-winner",
        "--candidate",
        "--check",
        "--final-row-id",
        "--graph-id",
        "--graph-json",
        "--mode",
        "--prediction",
        "--result",
        "--row-id",
        "--seed",
        "--target-id",
        "--training-result",
        "--capacity-result",
    }
)

# Only genuinely external scalar controls belong in this manifest.  Paths to
# authenticated parents and dependency-owned outputs are derived from the
# campaign root by ``node_runtime``.  In particular, a future DAG output must
# never be hashed or requested during pre-execution preparation.
REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE = {
    "storage_measurement": ("--available-storage-bytes",),
    "scale_efficiency": (
        "--production-batch-size",
        "--clock-power-mode",
    ),
}

REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS: dict[str, dict[str, int]] = {}

REQUIRED_INFRASTRUCTURE_OPTION_KEYS: dict[
    str, dict[str, frozenset[str]]
] = {}

# All required infrastructure values except these three scalar controls are
# external paths.  They must be referenced through an authenticated
# ``{file_*}`` or ``{directory_*}`` binding; literal paths would otherwise
# evade manifest hashing and pre-execution drift checks.
SCALAR_INFRASTRUCTURE_OPTIONS = frozenset(
    {
        "--available-storage-bytes",
        "--clock-power-mode",
        "--production-batch-size",
    }
)
DIRECTORY_INFRASTRUCTURE_OPTIONS = frozenset(
    {
        "--cache",
        "--design-confirm-cache",
        "--design-select-cache",
        "--model-train-tree",
        "--scale-hlt-cache",
        "--scale-offline-cache",
        "--scale-hlt-tree",
        "--scale-train-cache",
        "--scale-train-tree",
        "--scale-tree-root",
        "--train-cache",
        "--tree-cache",
        "--tree-cache-dir",
        "--tree-root",
        "--validation-tree-root",
        "--val-stop-cache",
    }
)
PATH_INFRASTRUCTURE_OPTIONS = frozenset(
    option
    for options in REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.values()
    for option in options
    if option not in SCALAR_INFRASTRUCTURE_OPTIONS
)

UNRESOLVED_RUNTIME_MARKERS = (
    "<REPLACE",
    "__REQUIRED_",
    "CHANGEME",
    "TODO",
)


def _present_options(values: Sequence[str]) -> frozenset[str]:
    return frozenset(
        value.split("=", 1)[0]
        for value in values
        if value.startswith("--")
    )


def _option_counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value.startswith("--"):
            option = value.split("=", 1)[0]
            counts[option] = counts.get(option, 0) + 1
    return counts


def _option_payloads(values: Sequence[str], option: str) -> list[str]:
    payloads = []
    index = 0
    while index < len(values):
        value = str(values[index])
        if value == option:
            if index + 1 >= len(values):
                payloads.append("")
            else:
                payloads.append(str(values[index + 1]))
            index += 2
            continue
        prefix = option + "="
        if value.startswith(prefix):
            payloads.append(value[len(prefix) :])
        index += 1
    return payloads


def _file_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"runtime input is absent or unsafe: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
    }


def _validate_runtime_argument_templates(
    arguments: Mapping[str, Sequence[str]],
    *,
    file_keys: Sequence[str],
    directory_keys: Sequence[str],
) -> None:
    allowed_fields = {
        "campaign_root",
        "coordinate",
        *(f"file_{key}" for key in file_keys),
        *(f"directory_{key}" for key in directory_keys),
    }
    formatter = string.Formatter()
    for node_id, values in sorted(arguments.items()):
        for value in values:
            text = str(value)
            if not text or any(marker in text.upper() for marker in UNRESOLVED_RUNTIME_MARKERS):
                raise ValueError(
                    f"runtime argument contains an unresolved placeholder: {node_id}"
                )
            try:
                fields = {
                    field_name
                    for _, field_name, _, _ in formatter.parse(text)
                    if field_name is not None
                }
            except ValueError as error:
                raise ValueError(
                    f"runtime argument has malformed formatting: {node_id}"
                ) from error
            unknown = fields - allowed_fields
            if unknown:
                raise ValueError(
                    f"runtime argument uses unbound placeholders: "
                    f"{node_id}: {sorted(unknown)}"
                )


def _validate_path_bindings(
    arguments: Mapping[str, Sequence[str]],
) -> None:
    formatter = string.Formatter()
    for node_id, required in sorted(
        REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.items()
    ):
        values = arguments.get(node_id, ())
        for option in required:
            if option not in PATH_INFRASTRUCTURE_OPTIONS:
                continue
            required_prefix = (
                "directory_"
                if option in DIRECTORY_INFRASTRUCTURE_OPTIONS
                else "file_"
            )
            for payload in _option_payloads(values, option):
                path_payload = (
                    payload.split("=", 1)[1]
                    if "=" in payload
                    else payload
                )
                fields = {
                    field_name
                    for _, field_name, _, _ in formatter.parse(path_payload)
                    if field_name is not None
                }
                exact_reference = (
                    len(fields) == 1
                    and path_payload == "{" + next(iter(fields)) + "}"
                )
                if not exact_reference or any(
                    not field.startswith(required_prefix)
                    for field in fields
                ):
                    raise ValueError(
                        "runtime path option is not authenticated through "
                        f"a {required_prefix.rstrip('_')} binding: "
                        f"{node_id} {option}"
                    )


def build_runtime_manifest(
    *,
    campaign_spec_sha256: str,
    files: Mapping[str, str | Path],
    directories: Mapping[str, str | Path],
    infrastructure_arguments_by_node: Mapping[str, Sequence[str]],
    source: Mapping[str, Any],
    runtime_support_sha256: str | None = None,
) -> dict[str, Any]:
    file_rows = {key: _file_record(value) for key, value in sorted(files.items())}
    directory_rows = {}
    for key, value in sorted(directories.items()):
        path = Path(value).resolve()
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(
                f"runtime directory is absent or unsafe: {path}"
            )
        directory_rows[key] = str(path)
    arguments = {}
    for node_id, values in sorted(infrastructure_arguments_by_node.items()):
        row = [str(value) for value in values]
        if any(
            value in FORBIDDEN_RUNTIME_OPTIONS
            for value in row
            if value.startswith("--")
        ):
            raise ValueError(
                f"runtime manifest attempts scientific row injection: {node_id}"
            )
        allowed = frozenset(
            REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.get(str(node_id), ())
        )
        observed = _present_options(row)
        if not observed.issubset(allowed):
            raise ValueError(
                "runtime manifest contains an unauthorized option: "
                f"{node_id}: {sorted(observed - allowed)}"
            )
        arguments[str(node_id)] = row
    _validate_runtime_argument_templates(
        arguments,
        file_keys=tuple(file_rows),
        directory_keys=tuple(directory_rows),
    )
    _validate_path_bindings(arguments)
    missing = {
        node_id: [
            option
            for option in required
            if option not in _present_options(arguments.get(node_id, ()))
        ]
        for node_id, required in sorted(
            REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.items()
        )
    }
    missing = {
        node_id: values for node_id, values in missing.items() if values
    }
    for node_id, requirements in sorted(
        REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS.items()
    ):
        counts = _option_counts(arguments.get(node_id, ()))
        for option, required_count in sorted(requirements.items()):
            observed = counts.get(option, 0)
            if observed < required_count:
                missing.setdefault(node_id, []).append(
                    f"{option} (need {required_count}, found {observed})"
                )
    for node_id, option_keys in sorted(
        REQUIRED_INFRASTRUCTURE_OPTION_KEYS.items()
    ):
        values = arguments.get(node_id, ())
        for option, required_keys in sorted(option_keys.items()):
            payloads = _option_payloads(values, option)
            observed_keys = {
                payload.split("=", 1)[0]
                for payload in payloads
                if "=" in payload
            }
            if (
                observed_keys != required_keys
                or len(payloads) != len(required_keys)
                or any("=" not in payload for payload in payloads)
            ):
                missing.setdefault(node_id, []).append(
                    f"{option} requires exact keys {sorted(required_keys)}; "
                    f"found {sorted(observed_keys)}"
                )
    for node_id, option in (
        ("storage_measurement", "--available-storage-bytes"),
        ("scale_efficiency", "--production-batch-size"),
    ):
        payloads = _option_payloads(arguments.get(node_id, ()), option)
        if payloads:
            try:
                positive = len(payloads) == 1 and int(payloads[0]) > 0
            except ValueError:
                positive = False
            if not positive:
                missing.setdefault(node_id, []).append(
                    f"{option} must be one positive integer"
                )
    payload = {
            "contract": RUNTIME_MANIFEST_CONTRACT,
            "schema_version": 4,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "files": file_rows,
            "directories": directory_rows,
            "infrastructure_arguments_by_node": arguments,
            "scientific_row_arguments_allowed": False,
            "manual_command_argv_allowed": False,
            "unresolved_placeholders_rejected": True,
            "required_infrastructure_options_by_node": {
                key: list(value)
                for key, value in sorted(
                    REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.items()
                )
            },
            "missing_required_options_by_node": missing,
            "execution_ready": not missing,
        }
    if runtime_support_sha256 is not None:
        payload["runtime_support_sha256"] = require_sha256(
            runtime_support_sha256, name="runtime_support_sha256"
        )
    return with_content_hash(payload)


def build_node_factory_registry(
    *,
    stage_job_registry: Mapping[str, Any],
    source: Mapping[str, Any],
    execution_profile: str = "production_500k_scale3m",
) -> dict[str, Any]:
    validate_stage_job_registry(stage_job_registry)
    if execution_profile not in {"miniature_test", "production_500k_scale3m"}:
        raise ValueError("node-factory execution profile differs")
    entries = []
    for node in stage_job_registry["nodes"]:
        coordinate_limit = int(NODE_COORDINATE_LIMITS.get(node["node_id"], 1))
        coordinate_span = (
            min(MINIATURE_COORDINATE_SPAN, coordinate_limit)
            if execution_profile == "miniature_test"
            else 1
        )
        entries.append(
            {
                "node_id": node["node_id"],
                "worker_entrypoint": node["entrypoint"],
                "factory_entrypoint": FACTORY_ENTRYPOINT,
                "coordinate_limit": coordinate_limit,
                "coordinate_span": coordinate_span,
                "scheduled_coordinate_count": math.ceil(
                    coordinate_limit / coordinate_span
                ),
                "row_resolution": (
                    "immutable_upstream_plan_or_lock"
                    if node["node_id"] in NODE_COORDINATE_LIMITS
                    else "singleton"
                ),
                "performance_can_change_coordinate_count": False,
                "manual_row_json_allowed": False,
            }
        )
    return with_content_hash(
        {
            "contract": NODE_FACTORY_REGISTRY_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "execution_profile": execution_profile,
            "stage_job_registry_sha256": stage_job_registry["content_hash"],
            "factory_entrypoint": FACTORY_ENTRYPOINT,
            "entries": entries,
            "entry_count": len(entries),
            "all_nodes_have_registered_factory": True,
        }
    )


def build_registered_command_matrix(
    *,
    stage_job_registry: Mapping[str, Any],
    factory_registry: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    campaign_root: str | Path,
) -> dict[str, list[list[str]]]:
    validate_stage_job_registry(stage_job_registry)
    validate_content_hash(
        factory_registry, expected_contract=NODE_FACTORY_REGISTRY_CONTRACT
    )
    validate_content_hash(
        runtime_manifest, expected_contract=RUNTIME_MANIFEST_CONTRACT
    )
    if (
        factory_registry["stage_job_registry_sha256"]
        != stage_job_registry["content_hash"]
        or factory_registry["source"] != runtime_manifest["source"]
    ):
        raise ValueError("node-factory lineage differs")
    root = str(Path(campaign_root).resolve())
    by_id = {row["node_id"]: row for row in factory_registry["entries"]}
    if set(by_id) != {
        row["node_id"] for row in stage_job_registry["nodes"]
    }:
        raise ValueError("node-factory coverage differs")
    return {
        node_id: [
            [
                "python",
                FACTORY_ENTRYPOINT,
                "--campaign-root",
                root,
                "--node-id",
                node_id,
                *(
                    ["--coordinate", str(coordinate)]
                    if int(row["coordinate_span"]) == 1
                    else [
                        "--coordinate-start",
                        str(coordinate * int(row["coordinate_span"])),
                        "--coordinate-stop",
                        str(
                            min(
                                int(row["coordinate_limit"]),
                                (coordinate + 1) * int(row["coordinate_span"]),
                            )
                        ),
                    ]
                ),
            ]
            for coordinate in range(int(row["scheduled_coordinate_count"]))
        ]
        for node_id, row in by_id.items()
    }


__all__ = [
    "FACTORY_ENTRYPOINT",
    "FORBIDDEN_RUNTIME_OPTIONS",
    "DIRECTORY_INFRASTRUCTURE_OPTIONS",
    "NODE_COORDINATE_LIMITS",
    "MINIATURE_COORDINATE_SPAN",
    "PATH_INFRASTRUCTURE_OPTIONS",
    "REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE",
    "REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS",
    "REQUIRED_INFRASTRUCTURE_OPTION_KEYS",
    "SCALAR_INFRASTRUCTURE_OPTIONS",
    "build_node_factory_registry",
    "build_registered_command_matrix",
    "build_runtime_manifest",
]
