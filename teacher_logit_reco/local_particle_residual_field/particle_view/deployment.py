"""HLT-only deployment manifest schema for particle-view bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from jetclass_fresh.jetclass_data import LABEL_NAMES, MAX_CONSTITUENTS

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PARTICLE_VIEW_DEPLOYMENT_CONTRACT = "particle_view_hlt_only_deployment_v1"
PARTICLE_VIEW_DEPLOYMENT_INPUTS = (
    "hlt_pf_features",
    "hlt_pf_vectors",
    "hlt_mask",
)
PARTICLE_VIEW_BUNDLE_KINDS = (
    "frozen_consumer",
    "joint_hlt_only",
    "ce_only_control",
    "hlt_memory_control",
)
PARTICLE_VIEW_DEPLOYMENT_WINNER_FAMILIES = (
    "privileged_scientific",
    "pre_stage_g_deployable",
)
PARTICLE_VIEW_FORBIDDEN_DEPENDENCIES = (
    "offline_particles",
    "offline_teacher",
    "gview",
    "true_views",
    "cached_privileged_logits",
    "training_labels",
    "oracle_attention_maps",
    "offline_normalizer",
)
PARTICLE_VIEW_BUNDLE_EXPORT_CONTRACT = "particle_view_hlt_only_bundle_export_v1"
PARTICLE_VIEW_RELOAD_FIXTURE_CONTRACT = "particle_view_hlt_reload_fixture_v1"
PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT = (
    "particle_view_fresh_process_reload_audit_v1"
)
PARTICLE_VIEW_BUNDLE_ARCHIVE_NAME = "particle_view_bundle.ts"
PARTICLE_VIEW_BUNDLE_INPUT_NAMES = (
    "points",
    "features",
    "lorentz_vectors",
    "mask",
)


class _HLTOnlyLogitBundle(nn.Module):
    def __init__(self, predictor: nn.Module, consumer: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.consumer = consumer

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.predictor(features, lorentz_vectors, mask)
        view = (
            prediction
            if isinstance(prediction, torch.Tensor)
            else prediction.mean
        )
        output = self.consumer(
            points,
            features,
            lorentz_vectors,
            mask,
            view,
            augment_clean_view=False,
        )
        return output.logits if hasattr(output, "logits") else output


def _tensor_logical_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_hlt_inputs(
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if set(inputs) != set(PARTICLE_VIEW_BUNDLE_INPUT_NAMES):
        raise ValueError("bundle export requires exactly four HLT tensors")
    points = inputs["points"]
    features = inputs["features"]
    vectors = inputs["lorentz_vectors"]
    mask = inputs["mask"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (points, features, vectors, mask)
    ):
        raise TypeError("bundle export inputs must be tensors")
    if mask.dtype is not torch.bool or mask.ndim != 3 or mask.shape[1] != 1:
        raise ValueError("bundle export mask must be bool [B,1,P]")
    if any(value.shape[0] != mask.shape[0] for value in (points, features, vectors)):
        raise ValueError("bundle export batch dimensions differ")
    if any(value.shape[-1] != mask.shape[-1] for value in (points, features, vectors)):
        raise ValueError("bundle export particle dimensions differ")
    if mask.shape[-1] != MAX_CONSTITUENTS:
        raise ValueError("bundle export requires exactly 128 padded particles")
    for value in (points, features, vectors):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("bundle export HLT tensors must be finite floating point")
    return tuple(
        value.detach().cpu().contiguous()
        for value in (points, features, vectors, mask)
    )


def build_particle_view_deployment_manifest(
    *,
    bundle_id: str,
    bundle_kind: str,
    winner_family: str,
    unified_split_manifest_sha256: str,
    train_identity_sha256: str,
    predictor_config_sha256: str,
    predictor_checkpoint_sha256: str,
    consumer_config_sha256: str,
    consumer_checkpoint_sha256: str,
    hlt_preprocessing_sha256: str,
    hlt_schema_sha256: str,
    view_normalizer_sha256: str,
    coordinate_binding_sha256: str,
    resource_profile_sha256: str,
    source_commit: str,
    bottleneck_width: int,
    class_names: Sequence[str] = LABEL_NAMES,
    max_particles: int = MAX_CONSTITUENTS,
) -> dict[str, Any]:
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError("bundle_id must be a non-empty string")
    if bundle_kind not in PARTICLE_VIEW_BUNDLE_KINDS:
        raise ValueError(f"unknown bundle_kind {bundle_kind!r}")
    if winner_family not in PARTICLE_VIEW_DEPLOYMENT_WINNER_FAMILIES:
        raise ValueError(f"unknown deployment winner_family {winner_family!r}")
    artifacts = {
        "unified_split_manifest_sha256": require_sha256(
            "unified_split_manifest_sha256", unified_split_manifest_sha256
        ),
        "train_identity_sha256": require_sha256(
            "train_identity_sha256", train_identity_sha256
        ),
        "predictor_config_sha256": require_sha256(
            "predictor_config_sha256", predictor_config_sha256
        ),
        "predictor_checkpoint_sha256": require_sha256(
            "predictor_checkpoint_sha256", predictor_checkpoint_sha256
        ),
        "consumer_config_sha256": require_sha256(
            "consumer_config_sha256", consumer_config_sha256
        ),
        "consumer_checkpoint_sha256": require_sha256(
            "consumer_checkpoint_sha256", consumer_checkpoint_sha256
        ),
        "hlt_preprocessing_sha256": require_sha256(
            "hlt_preprocessing_sha256", hlt_preprocessing_sha256
        ),
        "hlt_schema_sha256": require_sha256(
            "hlt_schema_sha256", hlt_schema_sha256
        ),
        "view_normalizer_sha256": require_sha256(
            "view_normalizer_sha256", view_normalizer_sha256
        ),
        "coordinate_binding_sha256": require_sha256(
            "coordinate_binding_sha256", coordinate_binding_sha256
        ),
        "resource_profile_sha256": require_sha256(
            "resource_profile_sha256", resource_profile_sha256
        ),
    }
    if (
        not isinstance(bottleneck_width, int)
        or isinstance(bottleneck_width, bool)
        or bottleneck_width not in {1, 2, 4, 8}
    ):
        raise ValueError("bottleneck_width must be one of 1, 2, 4, or 8")
    if tuple(class_names) != tuple(LABEL_NAMES):
        raise ValueError("deployment class order mismatch")
    if max_particles != MAX_CONSTITUENTS:
        raise ValueError("deployment must use 128 particles")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source_commit must be a 40- or 64-character hex digest")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DEPLOYMENT_CONTRACT,
            "bundle_id": bundle_id,
            "bundle_kind": bundle_kind,
            "winner_family": winner_family,
            "required_inputs": list(PARTICLE_VIEW_DEPLOYMENT_INPUTS),
            "forbidden_dependencies": list(PARTICLE_VIEW_FORBIDDEN_DEPENDENCIES),
            "oracle_dependencies": [],
            "requires_oracle": False,
            "artifacts": artifacts,
            "class_names": list(class_names),
            "max_particles": int(max_particles),
            "bottleneck_width": int(bottleneck_width),
            "source_commit": source_commit,
            "final_test_hlt_only": True,
            "fresh_process_reload_required": True,
        }
    )
    validate_particle_view_deployment_manifest(artifact)
    return artifact


def validate_particle_view_deployment_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_DEPLOYMENT_CONTRACT
    )
    expected_fields = {
        "contract",
        "bundle_id",
        "bundle_kind",
        "winner_family",
        "required_inputs",
        "forbidden_dependencies",
        "oracle_dependencies",
        "requires_oracle",
        "artifacts",
        "class_names",
        "max_particles",
        "bottleneck_width",
        "source_commit",
        "final_test_hlt_only",
        "fresh_process_reload_required",
        "content_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("deployment manifest schema mismatch")
    if not isinstance(payload.get("bundle_id"), str) or not payload["bundle_id"]:
        raise ValueError("deployment bundle_id is invalid")
    if payload.get("bundle_kind") not in PARTICLE_VIEW_BUNDLE_KINDS:
        raise ValueError("deployment bundle_kind mismatch")
    if payload.get("winner_family") not in PARTICLE_VIEW_DEPLOYMENT_WINNER_FAMILIES:
        raise ValueError("deployment winner family mismatch")
    if payload.get("required_inputs") != list(PARTICLE_VIEW_DEPLOYMENT_INPUTS):
        raise ValueError("deployable bundle requires non-HLT inputs")
    if payload.get("forbidden_dependencies") != list(
        PARTICLE_VIEW_FORBIDDEN_DEPENDENCIES
    ):
        raise ValueError("deployment forbidden-dependency declaration mismatch")
    if payload.get("oracle_dependencies") != []:
        raise ValueError("deployable bundle contains oracle dependencies")
    if payload.get("requires_oracle") is not False:
        raise ValueError("deployable bundle cannot require an oracle")
    if payload.get("class_names") != list(LABEL_NAMES):
        raise ValueError("deployment class order mismatch")
    if payload.get("max_particles") != MAX_CONSTITUENTS:
        raise ValueError("deployment maximum-particle mismatch")
    if payload.get("bottleneck_width") not in {1, 2, 4, 8}:
        raise ValueError("deployment bottleneck width mismatch")
    if payload.get("final_test_hlt_only") is not True:
        raise ValueError("final-test deployable rows must be HLT-only")
    if payload.get("fresh_process_reload_required") is not True:
        raise ValueError("deployment must require a fresh-process reload audit")
    artifacts = payload.get("artifacts")
    expected_artifacts = {
        "unified_split_manifest_sha256",
        "train_identity_sha256",
        "predictor_config_sha256",
        "predictor_checkpoint_sha256",
        "consumer_config_sha256",
        "consumer_checkpoint_sha256",
        "hlt_preprocessing_sha256",
        "hlt_schema_sha256",
        "view_normalizer_sha256",
        "coordinate_binding_sha256",
        "resource_profile_sha256",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("deployment artifact set mismatch")
    for name in sorted(expected_artifacts):
        require_sha256(f"artifacts.{name}", artifacts[name])
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("deployment source_commit is invalid")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "bundle_id": payload["bundle_id"],
        "bundle_kind": payload["bundle_kind"],
        "winner_family": payload["winner_family"],
        "hlt_only": True,
    }


def export_hlt_only_particle_view_bundle(
    output_dir: str | Path,
    *,
    predictor: nn.Module,
    consumer: nn.Module,
    exemplar_hlt_inputs: Mapping[str, torch.Tensor],
    deployment_manifest: Mapping[str, Any],
    source_bundle_sha256: str,
    predictor_config: Mapping[str, Any],
    consumer_config: Mapping[str, Any],
    lineage_graph_sha256: str,
    fairness_ledger_sha256: str,
    split_authorization_sha256: str,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Export a self-contained TorchScript graph with an HLT-only boundary."""

    validate_particle_view_deployment_manifest(deployment_manifest)
    for name, value in (
        ("source_bundle_sha256", source_bundle_sha256),
        ("lineage_graph_sha256", lineage_graph_sha256),
        ("fairness_ledger_sha256", fairness_ledger_sha256),
        ("split_authorization_sha256", split_authorization_sha256),
    ):
        require_sha256(name, value)
    if tolerance != 1.0e-6:
        raise ValueError("bundle export tolerance changed")
    if not isinstance(predictor_config, Mapping) or not isinstance(
        consumer_config, Mapping
    ):
        raise TypeError("bundle configs must be mappings")
    predictor_config_sha = canonical_sha256(dict(predictor_config))
    consumer_config_sha = canonical_sha256(dict(consumer_config))
    artifacts = deployment_manifest["artifacts"]
    if artifacts["predictor_config_sha256"] != predictor_config_sha:
        raise ValueError("predictor config differs from deployment manifest")
    if artifacts["consumer_config_sha256"] != consumer_config_sha:
        raise ValueError("consumer config differs from deployment manifest")
    inputs = _canonical_hlt_inputs(exemplar_hlt_inputs)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / PARTICLE_VIEW_BUNDLE_ARCHIVE_NAME
    manifest_path = root / "particle_view_bundle_manifest.json"
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite particle-view bundle")
    bundle = _HLTOnlyLogitBundle(predictor, consumer).cpu().eval()
    with torch.no_grad():
        reference = bundle(*inputs).detach().cpu().float().contiguous()
    if reference.ndim != 2 or not torch.isfinite(reference).all():
        raise ValueError("bundle reference logits must be finite rank two")
    traced = torch.jit.trace(bundle, inputs, strict=True, check_trace=True)
    traced = torch.jit.freeze(traced.eval())
    torch.jit.save(traced, archive_path)
    loaded = torch.jit.load(str(archive_path), map_location="cpu").eval()
    with torch.no_grad():
        reloaded = loaded(*inputs).detach().cpu().float().contiguous()
    maximum = float((reference - reloaded).abs().max().item())
    if maximum > tolerance:
        raise RuntimeError("exported bundle does not reproduce reference logits")
    input_schema = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        for name, value in zip(PARTICLE_VIEW_BUNDLE_INPUT_NAMES, inputs)
    ]
    export = with_content_hash(
        {
            "contract": PARTICLE_VIEW_BUNDLE_EXPORT_CONTRACT,
            "deployment_manifest": dict(deployment_manifest),
            "deployment_manifest_sha256": deployment_manifest["content_hash"],
            "source_bundle_sha256": source_bundle_sha256,
            "lineage_graph_sha256": lineage_graph_sha256,
            "fairness_ledger_sha256": fairness_ledger_sha256,
            "split_authorization_sha256": split_authorization_sha256,
            "archive_file": archive_path.name,
            "archive_sha256": sha256_file(archive_path),
            "archive_format": "torchscript_frozen_hlt_only_v1",
            "predictor_config": dict(predictor_config),
            "predictor_config_sha256": predictor_config_sha,
            "consumer_config": dict(consumer_config),
            "consumer_config_sha256": consumer_config_sha,
            "input_schema": input_schema,
            "required_inputs": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            "reference_logits_sha256": _tensor_logical_sha256(reference),
            "in_process_reload_max_abs_difference": maximum,
            "reload_tolerance": tolerance,
            "requires_oracle": False,
            "oracle_dependencies": [],
            "final_test_hlt_only": True,
            "fresh_process_reload_required": True,
        }
    )
    write_immutable_json(manifest_path, export)
    validate_particle_view_bundle_export(manifest_path)
    return export


def validate_particle_view_bundle_export(
    manifest_path: str | Path,
    *,
    expected_source_bundle_sha256: str | None = None,
    expected_lineage_graph_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_BUNDLE_EXPORT_CONTRACT
    )
    validate_particle_view_deployment_manifest(payload["deployment_manifest"])
    if (
        payload["deployment_manifest"]["content_hash"]
        != payload["deployment_manifest_sha256"]
    ):
        raise ValueError("nested deployment manifest hash mismatch")
    for name in (
        "source_bundle_sha256",
        "lineage_graph_sha256",
        "fairness_ledger_sha256",
        "split_authorization_sha256",
        "archive_sha256",
        "predictor_config_sha256",
        "consumer_config_sha256",
        "reference_logits_sha256",
    ):
        require_sha256(name, payload[name])
    if expected_source_bundle_sha256 is not None:
        require_sha256(
            "expected_source_bundle_sha256", expected_source_bundle_sha256
        )
        if payload["source_bundle_sha256"] != expected_source_bundle_sha256:
            raise ValueError("export belongs to a different selected bundle")
    if expected_lineage_graph_sha256 is not None:
        require_sha256(
            "expected_lineage_graph_sha256", expected_lineage_graph_sha256
        )
        if payload["lineage_graph_sha256"] != expected_lineage_graph_sha256:
            raise ValueError("export belongs to a different lineage graph")
    if canonical_sha256(payload["predictor_config"]) != payload[
        "predictor_config_sha256"
    ]:
        raise ValueError("exported predictor config hash mismatch")
    if canonical_sha256(payload["consumer_config"]) != payload[
        "consumer_config_sha256"
    ]:
        raise ValueError("exported consumer config hash mismatch")
    if payload["required_inputs"] != list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES):
        raise ValueError("exported graph boundary is not HLT-only")
    if (
        payload["requires_oracle"] is not False
        or payload["oracle_dependencies"] != []
        or payload["final_test_hlt_only"] is not True
    ):
        raise ValueError("exported graph contains an oracle dependency")
    archive = path.parent / payload["archive_file"]
    if sha256_file(archive) != payload["archive_sha256"]:
        raise ValueError("exported TorchScript archive hash mismatch")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "source_bundle_sha256": payload["source_bundle_sha256"],
        "archive_path": str(archive),
        "winner_family": payload["deployment_manifest"]["winner_family"],
        "bundle_kind": payload["deployment_manifest"]["bundle_kind"],
        "hlt_only": True,
    }


def load_exported_particle_view_bundle(
    manifest_path: str | Path,
    *,
    expected_source_bundle_sha256: str | None = None,
    expected_lineage_graph_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> torch.jit.ScriptModule:
    validation = validate_particle_view_bundle_export(
        manifest_path,
        expected_source_bundle_sha256=expected_source_bundle_sha256,
        expected_lineage_graph_sha256=expected_lineage_graph_sha256,
    )
    module = torch.jit.load(validation["archive_path"], map_location=device)
    module.eval()
    return module


def write_hlt_reload_fixture(
    output_dir: str | Path,
    *,
    exemplar_hlt_inputs: Mapping[str, torch.Tensor],
    reference_logits: torch.Tensor,
    bundle_export_sha256: str,
) -> dict[str, Any]:
    """Write a label-free temporary fixture for the fresh-process audit."""

    require_sha256("bundle_export_sha256", bundle_export_sha256)
    inputs = _canonical_hlt_inputs(exemplar_hlt_inputs)
    logits = reference_logits.detach().cpu().float().contiguous()
    if logits.ndim != 2 or logits.shape[0] != inputs[0].shape[0]:
        raise ValueError("reload reference logits shape mismatch")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "hlt_reload_fixture.npz"
    manifest_path = root / "hlt_reload_fixture.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError("reload fixture already exists")
    arrays = {
        name: value.numpy()
        for name, value in zip(PARTICLE_VIEW_BUNDLE_INPUT_NAMES, inputs)
    }
    arrays["reference_logits"] = logits.numpy()
    np.savez(data_path, **arrays)
    manifest = with_content_hash(
        {
            "contract": PARTICLE_VIEW_RELOAD_FIXTURE_CONTRACT,
            "bundle_export_sha256": bundle_export_sha256,
            "data_file": data_path.name,
            "data_file_sha256": sha256_file(data_path),
            "required_inputs": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            "reference_logits_sha256": _tensor_logical_sha256(logits),
            "contains_labels": False,
            "contains_offline_inputs": False,
        }
    )
    write_immutable_json(manifest_path, manifest)
    return manifest


def run_fresh_process_reload_audit(
    *,
    bundle_manifest_path: str | Path,
    fixture_manifest_path: str | Path,
    output_path: str | Path,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Run the exported graph in a new interpreter with only HLT fixture data."""

    bundle_path = Path(bundle_manifest_path).resolve()
    fixture_path = Path(fixture_manifest_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    script = Path(__file__).resolve().parents[3] / "scripts" / (
        "audit_particle_view_bundle_reload.py"
    )
    command = [
        str(python_executable or sys.executable),
        str(script),
        "--bundle-manifest",
        str(bundle_path),
        "--fixture-manifest",
        str(fixture_path),
        "--output",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "fresh-process reload audit failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    audit = json.loads(destination.read_text(encoding="utf-8"))
    validate_content_hash(
        audit, expected_contract=PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT
    )
    return audit


__all__ = [
    "PARTICLE_VIEW_BUNDLE_ARCHIVE_NAME",
    "PARTICLE_VIEW_BUNDLE_EXPORT_CONTRACT",
    "PARTICLE_VIEW_BUNDLE_INPUT_NAMES",
    "PARTICLE_VIEW_BUNDLE_KINDS",
    "PARTICLE_VIEW_DEPLOYMENT_CONTRACT",
    "PARTICLE_VIEW_DEPLOYMENT_INPUTS",
    "PARTICLE_VIEW_DEPLOYMENT_WINNER_FAMILIES",
    "PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT",
    "PARTICLE_VIEW_FORBIDDEN_DEPENDENCIES",
    "PARTICLE_VIEW_RELOAD_FIXTURE_CONTRACT",
    "build_particle_view_deployment_manifest",
    "export_hlt_only_particle_view_bundle",
    "load_exported_particle_view_bundle",
    "run_fresh_process_reload_audit",
    "validate_particle_view_deployment_manifest",
    "validate_particle_view_bundle_export",
    "write_hlt_reload_fixture",
]
