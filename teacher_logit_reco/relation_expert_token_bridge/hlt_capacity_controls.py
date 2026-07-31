"""Scale-finalist HLT capacity-control selection and deployable exports."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .offline_capacity_models import (
    MonolithicBase4ParticleTransformer,
    OfflineClassifierAdapter,
)
from teacher_logit_reco.relational_part.model import (
    RelationalParticleTransformer,
)


HLT_CAPACITY_CONTROL_EXPORT_CONTRACT = (
    "retb_hlt_capacity_control_export_v2"
)
HLT_CAPACITY_CONTROL_ROW_CONTRACT = "retb_hlt_capacity_control_row_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_hlt_capacity_control_model(
    *,
    control_kind: str,
    configuration: Sequence[int] | None,
    weaver_module: Any | None = None,
) -> Any:
    weaver = weaver_module or importlib.import_module(
        "weaver.nn.model.ParticleTransformer"
    )
    if control_kind in {"H_MONO_PARAM", "H_MONO_FLOP"}:
        if configuration is None:
            raise ValueError("monolithic HLT control lacks a configuration")
        return OfflineClassifierAdapter(
            MonolithicBase4ParticleTransformer(
                configuration, weaver_module=weaver
            )
        )
    if control_kind == "H_BASE_LONG":
        if configuration not in (None, (), []):
            raise ValueError("H_BASE_LONG cannot change H_BASE topology")
        return OfflineClassifierAdapter(
            RelationalParticleTransformer(weaver_module=weaver)
        )
    raise ValueError("HLT capacity-control kind is unknown")


def publish_hlt_capacity_control_export(
    *,
    output: str | Path,
    owner_finalist_graph_id: str,
    control_kind: str,
    pipeline_seed: int,
    configuration: Sequence[int] | None,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_registration_sha256: str,
    capacity_selection_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    manifest_path = Path(output).resolve()
    try:
        checkpoint_relative = str(
            checkpoint.relative_to(manifest_path.parent)
        )
    except ValueError as error:
        raise ValueError(
            "HLT capacity-control checkpoint must be colocated with export"
        ) from error
    expected = require_sha256(
        checkpoint_sha256, name="checkpoint_sha256"
    )
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or _sha256(checkpoint) != expected
    ):
        raise ValueError("HLT capacity-control checkpoint differs")
    artifact = bind_source(
        with_content_hash(
            {
                "contract": HLT_CAPACITY_CONTROL_EXPORT_CONTRACT,
                "schema_version": 2,
                "owner_finalist_graph_id": str(
                    owner_finalist_graph_id
                ),
                "control_kind": str(control_kind),
                "pipeline_seed": int(pipeline_seed),
                "configuration": (
                    None
                    if configuration is None
                    else [int(value) for value in configuration]
                ),
                "checkpoint_path": checkpoint_relative,
                "checkpoint_sha256": expected,
                "training_registration_sha256": require_sha256(
                    training_registration_sha256,
                    name="training_registration_sha256",
                ),
                "capacity_selection_sha256": require_sha256(
                    capacity_selection_sha256,
                    name="capacity_selection_sha256",
                ),
                "input_domain": "HLT_v3",
                "pair_relations": ["base4"],
                "contains_offline_or_oracle_inputs": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(manifest_path, artifact)
    return artifact


def load_hlt_capacity_control_export(
    manifest_path: str | Path, *, expected_source: Mapping[str, Any]
) -> Any:
    import torch

    from .contracts import load_hashed_json

    manifest = load_hashed_json(
        manifest_path,
        expected_contract=HLT_CAPACITY_CONTROL_EXPORT_CONTRACT,
    )
    manifest_parent = Path(manifest_path).resolve().parent
    relative_checkpoint = Path(manifest["checkpoint_path"])
    checkpoint = (manifest_parent / relative_checkpoint).resolve()
    try:
        checkpoint.relative_to(manifest_parent)
    except ValueError as error:
        raise ValueError(
            "HLT capacity-control checkpoint escapes export"
        ) from error
    if (
        manifest.get("source") != expected_source
        or relative_checkpoint.is_absolute()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or _sha256(checkpoint) != manifest["checkpoint_sha256"]
    ):
        raise ValueError("HLT capacity-control export lineage differs")
    model = build_hlt_capacity_control_model(
        control_kind=manifest["control_kind"],
        configuration=manifest["configuration"],
    )
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return model


def build_hlt_capacity_control_row(
    *,
    owner_finalist_graph_id: str,
    control_kind: str,
    pipeline_seed: int,
    checkpoint_sha256: str,
    deployable_export_sha256: str,
    training_registration_sha256: str,
    optimizer_updates_completed: int,
    labeled_example_presentations: int,
    capacity_selection_sha256: str,
) -> dict[str, Any]:
    if (
        control_kind
        not in {"H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG"}
        or int(optimizer_updates_completed) <= 0
        or int(labeled_example_presentations) <= 0
    ):
        raise ValueError("HLT capacity-control row is incomplete")
    return with_content_hash(
        {
            "contract": HLT_CAPACITY_CONTROL_ROW_CONTRACT,
            "schema_version": 1,
            "owner_finalist_graph_id": str(owner_finalist_graph_id),
            "control_kind": control_kind,
            "pipeline_seed": int(pipeline_seed),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "deployable_export_sha256": require_sha256(
                deployable_export_sha256,
                name="deployable_export_sha256",
            ),
            "training_registration_sha256": require_sha256(
                training_registration_sha256,
                name="training_registration_sha256",
            ),
            "optimizer_updates_completed": int(
                optimizer_updates_completed
            ),
            "labeled_example_presentations": int(
                labeled_example_presentations
            ),
            "capacity_selection_sha256": require_sha256(
                capacity_selection_sha256,
                name="capacity_selection_sha256",
            ),
            "actual_training_executed": True,
            "performance_based_termination": False,
        }
    )


def validate_hlt_capacity_control_row(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=HLT_CAPACITY_CONTROL_ROW_CONTRACT
    )
    expected = build_hlt_capacity_control_row(
        **{
            key: payload[key]
            for key in (
                "owner_finalist_graph_id",
                "control_kind",
                "pipeline_seed",
                "checkpoint_sha256",
                "deployable_export_sha256",
                "training_registration_sha256",
                "optimizer_updates_completed",
                "labeled_example_presentations",
                "capacity_selection_sha256",
            )
        }
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("HLT capacity-control row semantics differ")
    return digest


__all__ = [
    "HLT_CAPACITY_CONTROL_EXPORT_CONTRACT",
    "HLT_CAPACITY_CONTROL_ROW_CONTRACT",
    "build_hlt_capacity_control_model",
    "build_hlt_capacity_control_row",
    "load_hlt_capacity_control_export",
    "publish_hlt_capacity_control_export",
    "validate_hlt_capacity_control_row",
]
