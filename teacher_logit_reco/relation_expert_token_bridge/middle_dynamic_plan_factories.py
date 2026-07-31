"""Authenticated data-dependent factories for the Stage F--J waves."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
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
from .production import MIDDLE_NODE_ENTRYPOINTS


MIDDLE_DYNAMIC_FACTORY_INPUT_CONTRACT = (
    "retb_stage_f_j_dynamic_factory_input_v1"
)
MIDDLE_DYNAMIC_TARGETS = (
    "uncertainty_calibration",
)
DEFAULT_PRODUCERS = {
    "uncertainty_calibration": "predictor_training",
}


def middle_dynamic_factory_input_path(
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


def build_middle_dynamic_factory_input(
    *,
    target_node_id: str,
    producer_node_id: str,
    campaign_spec_sha256: str,
    production_graph_sha256: str,
    producer_execution_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(target_node_id)
    if target not in MIDDLE_DYNAMIC_TARGETS or not rows:
        raise ValueError("Stage F--J dynamic factory target/rows differ")
    checked = []
    for index, raw in enumerate(rows):
        if set(raw) != {
            "task_id",
            "argv",
            "expected_outputs",
            "input_artifact_hashes",
            "environment",
        }:
            raise ValueError("Stage F--J dynamic factory row fields differ")
        argv = [str(value) for value in raw["argv"]]
        environment = {
            str(name): str(value)
            for name, value in raw["environment"].items()
        }
        hashes = dict(raw["input_artifact_hashes"])
        if (
            str(raw["task_id"]) != f"{target}:{index}"
            or len(argv) < 2
            or argv[1].replace("\\", "/")
            != MIDDLE_NODE_ENTRYPOINTS[target]
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
            raise ValueError("Stage F--J dynamic factory row semantics differ")
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
        raise ValueError("Stage F--J dynamic coverage differs")
    if target in {
        "predictor_bundle_selector",
        "joint_predictor_selector",
    } and len(checked) != 1:
        raise ValueError(f"{target} requires one complete selector row")
    return with_content_hash(
        {
            "contract": MIDDLE_DYNAMIC_FACTORY_INPUT_CONTRACT,
            "schema_version": 1,
            "target_node_id": target,
            "producer_node_id": str(producer_node_id),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "production_graph_sha256": require_sha256(
                production_graph_sha256, name="production_graph_sha256"
            ),
            "producer_execution_sha256": require_sha256(
                producer_execution_sha256,
                name="producer_execution_sha256",
            ),
            "row_count": len(checked),
            "rows": checked,
            "coverage": coverage_row,
            "operator_supplied_row_json_permitted": False,
            "scientific_underperformance_blocks_continuation": False,
            "source": dict(source),
        }
    )


def publish_middle_dynamic_factory_input(
    *, campaign_root: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=MIDDLE_DYNAMIC_FACTORY_INPUT_CONTRACT
    )
    return write_immutable_json(
        middle_dynamic_factory_input_path(
            campaign_root, target_node_id=payload["target_node_id"]
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
    input_path = middle_dynamic_factory_input_path(
        root, target_node_id=target
    )
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    outputs = {
        str(Path(path).resolve()): digest
        for row in completion["rows"]
        for path, digest in row["output_hashes"].items()
    }
    if (
        str(input_path.resolve()) not in outputs
        or not input_path.is_file()
        or input_path.is_symlink()
        or _file_sha256(input_path) != outputs[str(input_path.resolve())]
    ):
        raise ValueError(
            "Stage F--J factory input is not an authenticated producer output"
        )
    payload = load_hashed_json(
        input_path, expected_contract=MIDDLE_DYNAMIC_FACTORY_INPUT_CONTRACT
    )
    expected = build_middle_dynamic_factory_input(
        target_node_id=target,
        producer_node_id=producer_node_id,
        campaign_spec_sha256=campaign_sha,
        production_graph_sha256=graph_sha,
        producer_execution_sha256=completion["task_manifest_sha256"],
        rows=payload["rows"],
        coverage=payload["coverage"],
        source=campaign["source"],
    )
    if payload != expected:
        raise ValueError("Stage F--J dynamic factory input differs")
    rows = []
    for index, source_row in enumerate(payload["rows"]):
        extras = {
            name: digest
            for name, digest in source_row[
                "input_artifact_hashes"
            ].items()
            if name not in {"campaign_spec", "production_graph"}
        }
        extras["middle_factory_input"] = payload["content_hash"]
        rows.append(
            _row(
                target=target,
                index=index,
                argv=source_row["argv"],
                outputs=source_row["expected_outputs"],
                campaign_sha256=campaign_sha,
                graph_sha256=graph_sha,
                producer_completion_sha256=completion_sha,
                extra_input_hashes=extras,
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


def _factory(target: str) -> Callable[..., dict[str, Any]]:
    def build(
        *,
        campaign_root: str | Path,
        campaign: Mapping[str, Any],
        production_graph: Mapping[str, Any],
        producer_node_id: str = DEFAULT_PRODUCERS[target],
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


for _target in MIDDLE_DYNAMIC_TARGETS:
    globals()[f"build_{_target}_manifest_plan"] = _factory(_target)

MIDDLE_DYNAMIC_PLAN_FACTORIES = {
    target: globals()[f"build_{target}_manifest_plan"]
    for target in MIDDLE_DYNAMIC_TARGETS
}


__all__ = [
    "MIDDLE_DYNAMIC_FACTORY_INPUT_CONTRACT",
    "MIDDLE_DYNAMIC_PLAN_FACTORIES",
    "MIDDLE_DYNAMIC_TARGETS",
    "build_middle_dynamic_factory_input",
    "middle_dynamic_factory_input_path",
    "publish_middle_dynamic_factory_input",
    *[f"build_{target}_manifest_plan" for target in MIDDLE_DYNAMIC_TARGETS],
]
