"""Authenticated executable plans and prediction evidence for Stage N."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .contracts import (
    bind_source,
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .final_seal import CONTROL_KINDS
from .predictor_bundle import PIPELINE_SEEDS
from .scale_execution import validate_execution_steps


STACK_INFERENCE_EXECUTION_PLAN_CONTRACT = (
    "retb_stack_val_inference_execution_plan_v1"
)
POSTLOCK_TARGET_EXECUTION_PLAN_CONTRACT = (
    "retb_postlock_target_execution_plan_v1"
)
FINALIST_CONTROLS_EXECUTION_PLAN_CONTRACT = (
    "retb_finalist_controls_execution_plan_v1"
)
SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT = (
    "retb_sealed_final_test_execution_plan_v1"
)
FINAL_TEST_PREDICTION_CONTRACT = "retb_final_test_prediction_v1"
FINAL_TEST_LABEL_MANIFEST_CONTRACT = "retb_final_test_label_manifest_v1"
DEPLOYABLE_INFERENCE_INPUT_CONTRACT = (
    "retb_deployable_inference_input_v1"
)
DEPLOYABLE_INFERENCE_INPUT_BINDING_CONTRACT = (
    "retb_deployable_inference_input_v2"
)
SHARED_DEPLOYABLE_INFERENCE_PAYLOAD_CONTRACT = (
    "retb_shared_deployable_inference_payload_v1"
)
DEPLOYABLE_INFERENCE_ENTRYPOINT = (
    "scripts/run_retb_deployable_inference.py"
)

_STAGE_N_FORBIDDEN_ENTRYPOINTS = frozenset(
    {
        "infer_retb_scale_stack_val.py",
        "build_retb_postlock_oracle_targets.py",
        "attest_retb_finalist_controls.py",
        "evaluate_retb_final_test.py",
        "execute_retb_stack_val_inference.py",
        "execute_retb_postlock_oracle_target.py",
        "execute_retb_finalist_controls.py",
        "execute_retb_sealed_final_test.py",
    }
)


def _inside(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Stage-N execution path escapes campaign") from error
    return resolved


def _steps(
    payload: Mapping[str, Any], *, root: Path, repo_root: Path
) -> None:
    validate_execution_steps(
        payload["steps"],
        campaign_root=root,
        repo_root=repo_root,
        forbidden_terms=(),
        forbidden_entrypoints=_STAGE_N_FORBIDDEN_ENTRYPOINTS,
    )


def _actual_inference_steps(
    steps: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> list[Mapping[str, Any]]:
    return [
        step
        for step in steps
        if (
            len(step.get("argv", ())) >= 2
            and str(step["argv"][1]).replace("\\", "/")
            == DEPLOYABLE_INFERENCE_ENTRYPOINT
        )
    ]


def _argument_value(argv: list[Any], flag: str) -> str | None:
    values = [str(value) for value in argv]
    matches = [
        values[index + 1]
        for index, value in enumerate(values[:-1])
        if value == flag
    ]
    return matches[0] if len(matches) == 1 else None


def _validate_inference_step(
    step: Mapping[str, Any],
    *,
    split: str,
    graph_id: str,
    pipeline_seed: int,
    output: Path,
) -> None:
    argv = list(step["argv"])
    required_flags = {
        "--split": split,
        "--graph-id": graph_id,
        "--pipeline-seed": str(int(pipeline_seed)),
        "--output": str(output),
    }
    if any(
        _argument_value(argv, flag) != expected
        for flag, expected in required_flags.items()
    ):
        raise ValueError("deployable inference command lineage differs")
    if split == "stack_val":
        if (
            _argument_value(argv, "--scale-completion") is None
            or "--execution-lock" in argv
            or "--execution-claim" in argv
        ):
            raise ValueError("stack-val inference authorization differs")
    elif (
        _argument_value(argv, "--execution-lock") is None
        or _argument_value(argv, "--execution-claim") is None
        or _argument_value(argv, "--execution-plan") is None
        or "--scale-completion" in argv
    ):
        raise ValueError("final-test inference authorization differs")


def validate_stack_inference_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STACK_INFERENCE_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "locked_scale_shortlist_sha256",
        "scale_completion_sha256",
        "parent_hashes",
        "steps",
        "inference_output_npz",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in PIPELINE_SEEDS
        or any(
            term in json.dumps(payload["steps"]).lower()
            for term in ("label", "offline", "oracle", "target")
        )
    ):
        raise ValueError("stack-val execution-plan semantics differ")
    root = Path(campaign_root).resolve()
    _steps(payload, root=root, repo_root=Path(repo_root).resolve())
    inference_output = _inside(payload["inference_output_npz"], root)
    actual_steps = _actual_inference_steps(payload["steps"])
    if (
        inference_output.suffix != ".npz"
        or len(actual_steps) != 1
        or str(inference_output)
        not in {
            str(Path(path).resolve())
            for path in actual_steps[0]["expected_outputs"]
        }
    ):
        raise ValueError("stack-val inference output must be NPZ")
    _validate_inference_step(
        actual_steps[0],
        split="stack_val",
        graph_id=str(payload["graph_id"]),
        pipeline_seed=int(payload["pipeline_seed"]),
        output=inference_output,
    )
    return digest


def validate_postlock_target_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=POSTLOCK_TARGET_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "split",
        "locked_scale_finalists_sha256",
        "steps",
        "target_evidence",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in PIPELINE_SEEDS
        or payload["split"] not in {"stack_val", "final_test"}
    ):
        raise ValueError("postlock target execution-plan semantics differ")
    root = Path(campaign_root).resolve()
    _steps(payload, root=root, repo_root=Path(repo_root).resolve())
    if _inside(payload["target_evidence"], root).suffix != ".json":
        raise ValueError("postlock target evidence must be JSON")
    return digest


def validate_finalist_controls_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINALIST_CONTROLS_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "locked_scale_finalists_sha256",
        "steps",
        "control_evidence",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
    ):
        raise ValueError("finalist-controls execution-plan semantics differ")
    root = Path(campaign_root).resolve()
    _steps(payload, root=root, repo_root=Path(repo_root).resolve())
    if _inside(payload["control_evidence"], root).suffix != ".json":
        raise ValueError("finalist-control evidence must be JSON")
    return digest


def validate_sealed_final_test_execution_plan(
    payload: Mapping[str, Any],
    *,
    execution_lock: Mapping[str, Any],
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "final_test_execution_lock_sha256",
        "steps",
        "final_labels_manifest",
        "prediction_rows",
        "source",
        "content_hash",
    }
    expected_rows = execution_lock["eligible_evaluation_rows"]
    records = payload.get("prediction_rows", [])
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or payload["final_test_execution_lock_sha256"]
        != execution_lock["content_hash"]
        or not isinstance(records, list)
        or len(records) != len(expected_rows)
    ):
        raise ValueError("sealed final-test execution-plan semantics differ")
    root = Path(campaign_root).resolve()
    _steps(payload, root=root, repo_root=Path(repo_root).resolve())
    if _inside(payload["final_labels_manifest"], root).suffix != ".json":
        raise ValueError("final-test labels evidence must be JSON")
    seen = set()
    actual_steps = _actual_inference_steps(payload["steps"])
    output_steps: dict[str, list[Mapping[str, Any]]] = {}
    for step in actual_steps:
        for path in step["expected_outputs"]:
            output_steps.setdefault(str(Path(path).resolve()), []).append(step)
    required_record = {
        "row_id",
        "graph_id",
        "pipeline_seed",
        "checkpoint_sha256",
        "inference_output_npz",
        "prediction_manifest_output",
    }
    for record in records:
        row_id = str(record.get("row_id"))
        expected = expected_rows.get(row_id)
        if (
            set(record) != required_record
            or expected is None
            or row_id in seen
            or record["graph_id"] != expected["graph_id"]
            or int(record["pipeline_seed"]) != expected["pipeline_seed"]
            or record["checkpoint_sha256"]
            != expected["checkpoint_sha256"]
            or _inside(record["inference_output_npz"], root).suffix != ".npz"
            or _inside(
                record["prediction_manifest_output"], root
            ).suffix
            != ".json"
        ):
            raise ValueError("sealed final-test prediction-row coverage differs")
        inference_path = _inside(record["inference_output_npz"], root)
        producers = output_steps.get(str(inference_path), [])
        if len(producers) != 1:
            raise ValueError(
                "sealed final-test prediction has no unique inference worker"
            )
        _validate_inference_step(
            producers[0],
            split="final_test",
            graph_id=str(record["graph_id"]),
            pipeline_seed=int(record["pipeline_seed"]),
            output=inference_path,
        )
        seen.add(row_id)
    if seen != set(expected_rows):
        raise ValueError("sealed final-test prediction rows are incomplete")
    return digest


def validate_deployable_inference_input(
    payload: Mapping[str, Any], *, manifest_path: str | Path
) -> str:
    """Validate one label-free, source-bound HLT inference payload."""

    digest = validate_content_hash(
        payload, expected_contract=DEPLOYABLE_INFERENCE_INPUT_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "split",
        "graph_id",
        "pipeline_seed",
        "identity_count",
        "identity_order_sha256",
        "payload_filename",
        "payload_sha256",
        "contains_HLT_inputs_only",
        "contains_labels",
        "contains_offline_or_oracle_values",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["split"] not in {"stack_val", "final_test"}
        or int(payload["pipeline_seed"]) not in PIPELINE_SEEDS
        or int(payload["identity_count"]) <= 0
        or Path(payload["payload_filename"]).name
        != payload["payload_filename"]
        or payload["contains_HLT_inputs_only"] is not True
        or payload["contains_labels"] is not False
        or payload["contains_offline_or_oracle_values"] is not False
    ):
        raise ValueError("deployable inference-input semantics differ")
    path = Path(manifest_path).resolve().parent / payload["payload_filename"]
    if (
        not path.is_file()
        or path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != payload["payload_sha256"]
    ):
        raise ValueError("deployable inference-input payload differs")
    return digest


def validate_shared_deployable_inference_payload(
    payload: Mapping[str, Any], *, manifest_path: str | Path
) -> str:
    """Validate one graph-independent, label-free HLT inference payload."""

    digest = validate_content_hash(
        payload,
        expected_contract=SHARED_DEPLOYABLE_INFERENCE_PAYLOAD_CONTRACT,
    )
    required = {
        "contract",
        "schema_version",
        "split",
        "identity_count",
        "identity_order_sha256",
        "payload_filename",
        "payload_sha256",
        "contains_HLT_inputs_only",
        "contains_labels",
        "contains_offline_or_oracle_values",
        "graph_independent_payload",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["split"] not in {"stack_val", "final_test"}
        or int(payload["identity_count"]) <= 0
        or Path(payload["payload_filename"]).name
        != payload["payload_filename"]
        or payload["contains_HLT_inputs_only"] is not True
        or payload["contains_labels"] is not False
        or payload["contains_offline_or_oracle_values"] is not False
        or payload["graph_independent_payload"] is not True
    ):
        raise ValueError("shared deployable inference payload semantics differ")
    path = Path(manifest_path).resolve().parent / payload["payload_filename"]
    if (
        not path.is_file()
        or path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != payload["payload_sha256"]
    ):
        raise ValueError("shared deployable inference payload differs")
    return digest


def _reject_privileged_inference_keys(
    value: Any, *, prefix: str = ""
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            if any(
                term in lowered
                for term in (
                    "label",
                    "offline",
                    "oracle",
                    "target",
                    "constituent_match",
                )
            ):
                raise ValueError(
                    f"deployable inference input contains privileged field {path}"
                )
            _reject_privileged_inference_keys(child, prefix=path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_privileged_inference_keys(
                child, prefix=f"{prefix}[{index}]"
            )


def publish_deployable_inference_input(
    *,
    output_dir: str | Path,
    split: str,
    graph_id: str,
    pipeline_seed: int,
    identities: list[str],
    hlt_inputs: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one immutable, label-free HLT input payload for real scoring."""

    import torch

    ids = [str(value) for value in identities]
    if (
        split not in {"stack_val", "final_test"}
        or int(pipeline_seed) not in PIPELINE_SEEDS
        or re.fullmatch(r"[A-Za-z0-9_.:-]+", str(graph_id)) is None
        or not ids
        or ids != sorted(ids)
        or len(ids) != len(set(ids))
        or not isinstance(hlt_inputs, Mapping)
    ):
        raise ValueError("deployable inference input identity differs")
    _reject_privileged_inference_keys(hlt_inputs)
    stream = io.BytesIO()
    torch.save(
        {"identities": ids, "hlt_inputs": dict(hlt_inputs)}, stream
    )
    root = Path(output_dir)
    safe_graph_id = str(graph_id).replace(":", "__")
    filename = (
        f"{split}_{safe_graph_id}_seed{int(pipeline_seed)}_HLT_inputs.pt"
    )
    payload_publication = write_immutable_bytes(
        root / filename, stream.getvalue()
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": DEPLOYABLE_INFERENCE_INPUT_CONTRACT,
                "schema_version": 1,
                "split": split,
                "graph_id": str(graph_id),
                "pipeline_seed": int(pipeline_seed),
                "identity_count": len(ids),
                "identity_order_sha256": canonical_sha256(ids),
                "payload_filename": filename,
                "payload_sha256": payload_publication["file_sha256"],
                "contains_HLT_inputs_only": True,
                "contains_labels": False,
                "contains_offline_or_oracle_values": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    manifest_path = (
        root / f"{split}_{safe_graph_id}_seed{int(pipeline_seed)}.json"
    )
    manifest_publication = write_immutable_json(manifest_path, manifest)
    validate_deployable_inference_input(
        manifest, manifest_path=manifest_path
    )
    return {
        "manifest": manifest,
        "payload_publication": payload_publication,
        "manifest_publication": manifest_publication,
    }


def publish_shared_deployable_inference_payload(
    *,
    output_dir: str | Path,
    split: str,
    identities: list[str],
    hlt_inputs: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the split payload once, independently of graph and seed."""

    import torch

    ids = [str(value) for value in identities]
    if (
        split not in {"stack_val", "final_test"}
        or not ids
        or ids != sorted(ids)
        or len(ids) != len(set(ids))
        or not isinstance(hlt_inputs, Mapping)
    ):
        raise ValueError("shared deployable inference identity differs")
    _reject_privileged_inference_keys(hlt_inputs)
    stream = io.BytesIO()
    torch.save(
        {"identities": ids, "hlt_inputs": dict(hlt_inputs)}, stream
    )
    root = Path(output_dir)
    filename = f"retb_{split}_shared_HLT_inputs.pt"
    payload_publication = write_immutable_bytes(
        root / filename, stream.getvalue()
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": SHARED_DEPLOYABLE_INFERENCE_PAYLOAD_CONTRACT,
                "schema_version": 1,
                "split": split,
                "identity_count": len(ids),
                "identity_order_sha256": canonical_sha256(ids),
                "payload_filename": filename,
                "payload_sha256": payload_publication["file_sha256"],
                "contains_HLT_inputs_only": True,
                "contains_labels": False,
                "contains_offline_or_oracle_values": False,
                "graph_independent_payload": True,
            }
        ),
        source_snapshot=source_snapshot,
    )
    manifest_path = root / f"retb_{split}_shared_HLT_inputs.json"
    manifest_publication = write_immutable_json(manifest_path, manifest)
    validate_shared_deployable_inference_payload(
        manifest, manifest_path=manifest_path
    )
    return {
        "manifest": manifest,
        "payload_publication": payload_publication,
        "manifest_publication": manifest_publication,
    }


def publish_deployable_inference_input_binding(
    *,
    output_dir: str | Path,
    shared_payload_manifest: Mapping[str, Any],
    shared_payload_manifest_path: str | Path,
    graph_id: str,
    pipeline_seed: int,
) -> dict[str, Any]:
    """Bind one graph/seed identity to an existing shared split payload."""

    shared_sha = validate_shared_deployable_inference_payload(
        shared_payload_manifest,
        manifest_path=shared_payload_manifest_path,
    )
    split = str(shared_payload_manifest["split"])
    if (
        int(pipeline_seed) not in PIPELINE_SEEDS
        or re.fullmatch(r"[A-Za-z0-9_.:-]+", str(graph_id)) is None
        or Path(output_dir).resolve()
        != Path(shared_payload_manifest_path).resolve().parent
    ):
        raise ValueError("deployable inference binding identity differs")
    safe_graph_id = str(graph_id).replace(":", "__")
    # v1 graph manifests owned a distinct payload. The v2 binding records the
    # graph identity while reusing one explicitly authenticated split payload.
    manifest = with_content_hash(
        {
            "contract": DEPLOYABLE_INFERENCE_INPUT_BINDING_CONTRACT,
            "schema_version": 2,
            "split": split,
            "graph_id": str(graph_id),
            "pipeline_seed": int(pipeline_seed),
            "identity_count": int(
                shared_payload_manifest["identity_count"]
            ),
            "identity_order_sha256": shared_payload_manifest[
                "identity_order_sha256"
            ],
            "payload_filename": shared_payload_manifest[
                "payload_filename"
            ],
            "payload_sha256": shared_payload_manifest["payload_sha256"],
            "contains_HLT_inputs_only": True,
            "contains_labels": False,
            "contains_offline_or_oracle_values": False,
            "shared_payload_manifest_filename": Path(
                shared_payload_manifest_path
            ).name,
            "shared_payload_manifest_sha256": shared_sha,
            "source": dict(shared_payload_manifest["source"]),
        }
    )
    manifest_path = (
        Path(output_dir)
        / f"{split}_{safe_graph_id}_seed{int(pipeline_seed)}.json"
    )
    publication = write_immutable_json(manifest_path, manifest)
    validate_deployable_inference_input_binding(
        manifest, manifest_path=manifest_path
    )
    return {
        "manifest": manifest,
        "manifest_publication": publication,
        "shared_payload_manifest": dict(shared_payload_manifest),
    }


def validate_deployable_inference_input_binding(
    payload: Mapping[str, Any], *, manifest_path: str | Path
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=DEPLOYABLE_INFERENCE_INPUT_BINDING_CONTRACT,
    )
    required = {
        "contract",
        "schema_version",
        "split",
        "graph_id",
        "pipeline_seed",
        "identity_count",
        "identity_order_sha256",
        "payload_filename",
        "payload_sha256",
        "contains_HLT_inputs_only",
        "contains_labels",
        "contains_offline_or_oracle_values",
        "shared_payload_manifest_filename",
        "shared_payload_manifest_sha256",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 2
        or payload["split"] not in {"stack_val", "final_test"}
        or int(payload["pipeline_seed"]) not in PIPELINE_SEEDS
        or int(payload["identity_count"]) <= 0
        or Path(payload["payload_filename"]).name
        != payload["payload_filename"]
        or payload["contains_HLT_inputs_only"] is not True
        or payload["contains_labels"] is not False
        or payload["contains_offline_or_oracle_values"] is not False
        or Path(payload["shared_payload_manifest_filename"]).name
        != payload["shared_payload_manifest_filename"]
    ):
        raise ValueError("deployable inference-input binding semantics differ")
    require_sha256(
        payload["shared_payload_manifest_sha256"],
        name="shared_payload_manifest_sha256",
    )
    parent = Path(manifest_path).resolve().parent
    shared_path = parent / payload["shared_payload_manifest_filename"]
    shared = load_hashed_json(
        shared_path,
        expected_contract=SHARED_DEPLOYABLE_INFERENCE_PAYLOAD_CONTRACT,
    )
    validate_shared_deployable_inference_payload(
        shared, manifest_path=shared_path
    )
    if (
        shared["content_hash"] != payload["shared_payload_manifest_sha256"]
        or shared["split"] != payload["split"]
        or shared["identity_count"] != payload["identity_count"]
        or shared["identity_order_sha256"]
        != payload["identity_order_sha256"]
        or shared["payload_filename"] != payload["payload_filename"]
        or shared["payload_sha256"] != payload["payload_sha256"]
        or shared["source"] != payload["source"]
    ):
        raise ValueError("deployable inference shared-payload lineage differs")
    path = parent / payload["payload_filename"]
    if (
        not path.is_file()
        or path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != payload["payload_sha256"]
    ):
        raise ValueError("deployable inference-input binding payload differs")
    return digest


def validate_final_labels_manifest(
    payload: Mapping[str, Any], *, manifest_path: str | Path
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FINAL_TEST_LABEL_MANIFEST_CONTRACT
    )
    if (
        set(payload)
        != {
            "contract",
            "schema_version",
            "split",
            "identity_count",
            "identity_order_sha256",
            "npz_filename",
            "npz_sha256",
            "contains_labels",
            "source",
            "content_hash",
        }
        or int(payload["schema_version"]) != 1
        or payload["split"] != "final_test"
        or payload["contains_labels"] is not True
        or Path(payload["npz_filename"]).name != payload["npz_filename"]
    ):
        raise ValueError("final-test label-manifest semantics differ")
    path = Path(manifest_path).resolve().parent / payload["npz_filename"]
    if (
        not path.is_file()
        or path.is_symlink()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != payload["npz_sha256"]
    ):
        raise ValueError("final-test label NPZ differs")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"identities", "labels"}:
            raise ValueError("final-test label NPZ fields differ")
        identities = np.asarray(arrays["identities"]).tolist()
        labels = np.asarray(arrays["labels"])
    if (
        len(identities) != int(payload["identity_count"])
        or canonical_sha256(identities)
        != payload["identity_order_sha256"]
        or labels.shape != (len(identities),)
    ):
        raise ValueError("final-test label identities differ")
    return digest


def publish_final_labels_manifest(
    *,
    output_dir: str | Path,
    identities: list[str],
    labels: np.ndarray,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    ids = [str(value) for value in identities]
    truth = np.asarray(labels, dtype=np.int64)
    if (
        ids != sorted(ids)
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or bool(((truth < 0) | (truth >= 10)).any())
        or len(set(np.bincount(truth, minlength=10).tolist())) != 1
    ):
        raise ValueError("final-test labels are not canonical and balanced")
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        identities=np.asarray(ids, dtype=np.str_),
        labels=truth,
    )
    root = Path(output_dir)
    filename = "retb_final_test_labels.npz"
    npz_publication = write_immutable_bytes(
        root / filename, stream.getvalue()
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": FINAL_TEST_LABEL_MANIFEST_CONTRACT,
                "schema_version": 1,
                "split": "final_test",
                "identity_count": len(ids),
                "identity_order_sha256": canonical_sha256(ids),
                "npz_filename": filename,
                "npz_sha256": npz_publication["file_sha256"],
                "contains_labels": True,
            }
        ),
        source_snapshot=source_snapshot,
    )
    manifest_publication = write_immutable_json(
        root / "retb_final_test_labels.json", manifest
    )
    return {
        "manifest": manifest,
        "npz_publication": npz_publication,
        "manifest_publication": manifest_publication,
    }


def build_final_test_prediction(
    *,
    row: Mapping[str, Any],
    execution_lock: Mapping[str, Any],
    identities: list[str],
    logits: np.ndarray,
    probabilities: np.ndarray,
    npz_filename: str,
    npz_sha256: str,
) -> dict[str, Any]:
    expected = execution_lock["eligible_evaluation_rows"].get(row["row_id"])
    if (
        expected is None
        or row["graph_id"] != expected["graph_id"]
        or int(row["pipeline_seed"]) != expected["pipeline_seed"]
        or row["checkpoint_sha256"] != expected["checkpoint_sha256"]
        or Path(npz_filename).name != npz_filename
    ):
        raise ValueError("final-test prediction is not execution-locked")
    values = np.asarray(logits)
    probability = np.asarray(probabilities)
    if (
        values.dtype != np.float32
        or probability.dtype != np.float32
        or values.shape != (len(identities), 10)
        or probability.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(probability).all()
    ):
        raise ValueError("final-test prediction arrays differ")
    return with_content_hash(
        {
            "contract": FINAL_TEST_PREDICTION_CONTRACT,
            "schema_version": 1,
            "final_test_execution_lock_sha256": execution_lock[
                "content_hash"
            ],
            "row_id": row["row_id"],
            "graph_id": row["graph_id"],
            "pipeline_seed": int(row["pipeline_seed"]),
            "checkpoint_sha256": require_sha256(
                row["checkpoint_sha256"], name="checkpoint_sha256"
            ),
            "degradation_profile_sha256": execution_lock[
                "degradation_profile_sha256"
            ],
            "identity_order_sha256": canonical_sha256(identities),
            "npz_filename": npz_filename,
            "npz_sha256": require_sha256(npz_sha256, name="npz_sha256"),
            "contains_labels": False,
            "created_after_execution_lock": True,
            "test_result_may_select_replacement": False,
        }
    )


def validate_control_evidence(
    payload: Mapping[str, Any],
    *,
    locked_scale_finalists: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(payload)
    expected_pairs = {
        (graph_id, kind, seed)
        for graph_id in locked_scale_finalists["finalist_graph_ids"]
        for kind in CONTROL_KINDS
        for seed in PIPELINE_SEEDS
    }
    counts = payload.get("optimizer_update_counts", {})
    rows = payload.get("rows", [])
    actual_pairs = {
        (
            str(item["owner_finalist_graph_id"]),
            str(item["kind"]),
            int(item["pipeline_seed"]),
        )
        for row in rows
        for item in row.get("evaluation_rows", [])
    }
    if (
        payload.get("source") != locked_scale_finalists.get("source")
        or payload.get("locked_scale_finalists_sha256")
        != locked_scale_finalists["content_hash"]
        or payload.get("actual_training_executed") is not True
        or actual_pairs != expected_pairs
        or set(counts) != {
            f"{graph}:{kind}:seed{seed}"
            for graph, kind, seed in expected_pairs
        }
        or any(int(value) <= 0 for value in counts.values())
    ):
        raise ValueError("finalist-control execution evidence differs")
    return digest


__all__ = [
    "DEPLOYABLE_INFERENCE_ENTRYPOINT",
    "DEPLOYABLE_INFERENCE_INPUT_BINDING_CONTRACT",
    "DEPLOYABLE_INFERENCE_INPUT_CONTRACT",
    "FINALIST_CONTROLS_EXECUTION_PLAN_CONTRACT",
    "FINAL_TEST_LABEL_MANIFEST_CONTRACT",
    "FINAL_TEST_PREDICTION_CONTRACT",
    "POSTLOCK_TARGET_EXECUTION_PLAN_CONTRACT",
    "SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT",
    "STACK_INFERENCE_EXECUTION_PLAN_CONTRACT",
    "SHARED_DEPLOYABLE_INFERENCE_PAYLOAD_CONTRACT",
    "build_final_test_prediction",
    "publish_final_labels_manifest",
    "publish_deployable_inference_input",
    "publish_deployable_inference_input_binding",
    "publish_shared_deployable_inference_payload",
    "validate_control_evidence",
    "validate_deployable_inference_input",
    "validate_deployable_inference_input_binding",
    "validate_final_labels_manifest",
    "validate_finalist_controls_execution_plan",
    "validate_postlock_target_execution_plan",
    "validate_sealed_final_test_execution_plan",
    "validate_stack_inference_execution_plan",
    "validate_shared_deployable_inference_payload",
]
