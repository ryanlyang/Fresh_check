"""Repository-owned Step 9 selection, deployment, and HLT-only evaluation.

The policy module defines immutable decisions and a generic bundle.  This
module connects those contracts to the concrete R0, correction, and local
residual-field consumer checkpoints produced by the prediction-anchored
campaign.  Privileged tensors are never accepted by its inference APIs.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jetclass_fresh.hlt_baseline import resolve_device
from jetclass_fresh.jetclass_data import load_split_manifest, manifest_hash
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

from .bridge_campaign import PAIRED_SEED_IDS, resolve_registry_run, validate_campaign_registry
from .bridge_campaign_policy import (
    PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
    DeployableReplicaEvidence,
    PredictionAnchoredDeployableBundle,
    aggregate_deployable_configuration,
    authorize_deployable_confirmation,
    authorize_final_test,
    build_deployable_bundle_manifest,
    build_deployable_confirmation,
    clean_hlt_only_reload_audit,
    export_deployable_bundle,
    finalize_deployable_confirmation,
    load_deployable_bundle,
    select_deployable_preconfirmation,
    validate_final_test_request,
)
from .bridge_evaluation import (
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    classification_metrics,
)
from .bridge_contracts import (
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_reconstruction_execution import (
    PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
    build_reconstruction_model,
    resolve_reconstruction_run,
)
from .bridge_splits import PREDICTION_ANCHORED_SPLIT_CONTRACT
from .fusion import load_local_residual_field_tagger_from_checkpoint
from .model import LocalResidualFieldReconstructorConfig, build_local_residual_field_reconstructor
from .tagger import LocalResidualFieldAugmentedParT


PREDICTION_ANCHORED_DEPLOYABLE_SEMANTIC_REPLICA_CONTRACT = (
    "prediction_anchored_deployable_semantic_replica_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_EVIDENCE_SET_CONTRACT = (
    "prediction_anchored_deployable_evidence_set_v1"
)
PREDICTION_ANCHORED_DEPLOY_CONFIRMATION_EXECUTION_CONTRACT = (
    "prediction_anchored_deploy_confirmation_execution_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_EXPORT_CONTRACT = (
    "prediction_anchored_repository_deployable_export_v1"
)
PREDICTION_ANCHORED_FINAL_TEST_REPORT_CONTRACT = (
    "prediction_anchored_hlt_only_final_test_report_v1"
)


def _regular_file(path: str | Path, *, label: str) -> Path:
    value = Path(path).resolve()
    if value.is_symlink() or not value.is_file():
        raise FileNotFoundError(f"{label} is absent or unsafe: {value}")
    return value


def _finite(value: Any, *, label: str) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be finite")
    return output


def build_deployable_semantic_replica_evidence(
    *,
    run_id: str,
    seed_id: int,
    perturbation_mean_accuracy_loss: float,
    perturbation_worst_accuracy_loss: float,
    alignment_finite: bool,
    distribution_distance_finite: bool,
    provenance_valid: bool = True,
    leakage_audit_passed: bool = True,
    mask_audit_passed: bool = True,
) -> dict[str, Any]:
    """Build the compact, threshold-bearing Step 8 evidence for one replica."""

    if int(seed_id) not in PAIRED_SEED_IDS:
        raise ValueError("semantic evidence requires paired seed 101, 202, or 303")
    mean_loss = _finite(
        perturbation_mean_accuracy_loss, label="perturbation mean accuracy loss"
    )
    worst_loss = _finite(
        perturbation_worst_accuracy_loss, label="perturbation worst accuracy loss"
    )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_SEMANTIC_REPLICA_CONTRACT,
            "run_id": str(run_id),
            "seed_id": int(seed_id),
            "perturbation_audit_seeds": [9101, 9102, 9103, 9104],
            "perturbation_mean_accuracy_loss": mean_loss,
            "perturbation_worst_accuracy_loss": worst_loss,
            "perturbation_threshold_passed": mean_loss <= 0.002 and worst_loss <= 0.003,
            "alignment_finite": bool(alignment_finite),
            "distribution_distance_finite": bool(distribution_distance_finite),
            "alignment_selection_threshold": None,
            "distribution_selection_threshold": None,
            "provenance_valid": bool(provenance_valid),
            "leakage_audit_passed": bool(leakage_audit_passed),
            "mask_audit_passed": bool(mask_audit_passed),
            "final_test_accessed": False,
        }
    )


def _tensor_parameter_count(payload: Mapping[str, Any]) -> int:
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint has no model_state_dict")
    return sum(int(value.numel()) for value in state.values() if isinstance(value, torch.Tensor))


def _semantic_for_replica(
    metrics: Mapping[str, Any],
    *,
    semantic_root: Path,
    run_id: str,
    seed_id: int,
) -> Mapping[str, Any]:
    embedded = metrics.get("semantic_evidence")
    if not isinstance(embedded, Mapping) and isinstance(
        metrics.get("model_val_select"), Mapping
    ):
        embedded = metrics["model_val_select"].get("semantic_evidence")
    if isinstance(embedded, Mapping):
        artifact = dict(embedded)
    else:
        artifact = load_hashed_json(
            semantic_root / run_id / f"seed{seed_id}.json",
            expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_SEMANTIC_REPLICA_CONTRACT,
        )
    validate_content_hash(
        artifact, expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_SEMANTIC_REPLICA_CONTRACT
    )
    if artifact.get("run_id") != run_id or int(artifact.get("seed_id", -1)) != int(seed_id):
        raise ValueError("semantic evidence identity differs from reconstruction replica")
    return artifact


def assemble_deployable_replica_evidence(
    registry: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    r0_checkpoint_path: str | Path,
    selected_consumer_path: str | Path,
    semantic_evidence_root: str | Path,
) -> dict[str, Any]:
    """Translate real paired publications into the strict Step 9 selector input."""

    validate_campaign_registry(registry)
    root = Path(artifact_root)
    r0_path = _regular_file(r0_checkpoint_path, label="R0 checkpoint")
    selected = load_hashed_json(
        selected_consumer_path,
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    if selected.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("deployable evidence refuses an unconfirmed or guessed consumer")
    teacher_path = _regular_file(selected["checkpoint_path"], label="selected T10 checkpoint")
    teacher_sha = sha256_file(teacher_path)
    if teacher_sha != selected.get("checkpoint_sha256"):
        raise ValueError("selected T10 checkpoint hash changed")
    r0_payload = torch.load(r0_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    frozen_parameters = _tensor_parameter_count(r0_payload) + _tensor_parameter_count(
        teacher_payload
    )
    binding = load_hashed_json(root / "bindings" / "primary.json")
    if binding.get("checkpoint_sha256") != teacher_sha:
        raise ValueError("primary binding is not the selected T10 checkpoint")

    replicas: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    excluded_runs: list[dict[str, str]] = []
    semantic_root = Path(semantic_evidence_root)
    for registry_row in registry["runs"]:
        if not bool(registry_row["selectable_for_primary_deployment"]):
            continue
        run_id = str(registry_row["canonical_run_id"])
        run_spec = resolve_reconstruction_run(run_id)
        l0_postteacher = run_id == "D10_L0_bridge_only"
        if run_spec.direct or (
            run_spec.binding_kind != "primary" and not l0_postteacher
        ):
            raise ValueError(f"selectable run {run_id} is not an R0+correction+primary-T10 graph")
        publication_root = root / "reconstructors" / run_id
        if not (publication_root / "aggregate_metrics.json").is_file() or not (
            publication_root / "publication.json"
        ).is_file():
            excluded_runs.append(
                {"run_id": run_id, "reason": "missing_or_failed_reconstruction_publication"}
            )
            continue
        aggregate = load_hashed_json(
            publication_root / "aggregate_metrics.json",
            expected_contract=PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
        )
        publication = load_hashed_json(
            publication_root / "publication.json",
            expected_contract=PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
        )
        if aggregate.get("run_id") != run_id or publication.get("run_id") != run_id:
            raise ValueError("reconstruction publication identity changed")
        if l0_postteacher and (
            aggregate.get("aggregation_phase") != "postteacher_common_model_val_select"
            or publication.get("l0_postteacher_common_evaluation") is not True
        ):
            raise ValueError("L0 publication lacks its verified common post-teacher evaluation")
        if publication.get("aggregate_sha256") != aggregate["content_hash"]:
            raise ValueError("reconstruction publication is bound to another aggregate")
        median_path = _regular_file(
            publication_root / publication["retained_checkpoint"],
            label=f"{run_id} median checkpoint",
        )
        if sha256_file(median_path) != publication["retained_checkpoint_sha256"]:
            raise ValueError("retained reconstruction checkpoint bytes changed")
        median_payload = torch.load(median_path, map_location="cpu", weights_only=False)
        deployed_parameters = frozen_parameters + _tensor_parameter_count(median_payload)
        parent_hashes = median_payload.get("parent_hashes", {})
        if parent_hashes.get("teacher_checkpoint_sha256") != teacher_sha:
            raise ValueError("reconstruction publication used a different selected T10")
        if parent_hashes.get("r0_checkpoint_sha256") != sha256_file(r0_path):
            raise ValueError("reconstruction publication used a different R0")

        run_replicas = []
        run_exclusion: str | None = None
        for replica_record in aggregate["replica_metrics"]:
            seed_id = int(replica_record["seed_id"])
            metrics = replica_record["metrics"]
            select = metrics.get("model_val_select")
            if not isinstance(select, Mapping):
                run_exclusion = "missing_common_model_val_select_evaluation"
                break
            try:
                semantic = _semantic_for_replica(
                    metrics,
                    semantic_root=semantic_root,
                    run_id=run_id,
                    seed_id=seed_id,
                )
            except FileNotFoundError:
                run_exclusion = "missing_required_semantic_evidence"
                break
            source_sha = replica_record.get("source_checkpoint_sha256")
            is_median = seed_id == int(publication["median_seed_id"])
            checkpoint_sha = publication["retained_checkpoint_sha256"] if is_median else source_sha
            if not isinstance(checkpoint_sha, str):
                raise ValueError("paired aggregate lacks a nonmedian source checkpoint hash")
            evidence = DeployableReplicaEvidence(
                run_id=run_id,
                seed_id=seed_id,
                accuracy=select["accuracy"],
                macro_per_class_accuracy=select["macro_per_class_accuracy"],
                cross_entropy=select["cross_entropy"],
                baseline_accuracy=select["f0_accuracy"],
                teacher_bridge_accuracy=select["privileged_bridge_accuracy"],
                epoch=int(metrics["checkpoint_selection"]["selected_epoch"]),
                checkpoint=str(median_path) if is_median else "RAM_ONLY_RETIRED",
                checkpoint_sha256=checkpoint_sha,
                scaler_sha256=parent_hashes["physical45_scaler_sha256"],
                teacher_sha256=teacher_sha,
                recipe_sha256=binding["bridge_recipe_sha256"],
                deployed_parameter_count=deployed_parameters,
                persistent_bytes=int(publication["measured_state_bytes"]),
                reserved_bytes=int(registry_row["measured_retained_bytes"]),
                recovery_fraction=select.get("recovery_fraction"),
                saturation_fraction=select["trust_saturation_fraction"],
                perturbation_mean_accuracy_loss=semantic[
                    "perturbation_mean_accuracy_loss"
                ],
                perturbation_worst_accuracy_loss=semantic[
                    "perturbation_worst_accuracy_loss"
                ],
                provenance_valid=bool(semantic["provenance_valid"]),
                leakage_audit_passed=bool(semantic["leakage_audit_passed"]),
                mask_audit_passed=bool(semantic["mask_audit_passed"]),
                reload_audit_passed=bool(publication["weights_payload_reload_verified"]),
                reliability_pass_through_exact=bool(
                    select["reliability_channels_exact_pass_through"]
                ),
                alignment_finite=bool(semantic["alignment_finite"]),
                distribution_distance_finite=bool(
                    semantic["distribution_distance_finite"]
                ),
            ).to_artifact()
            run_replicas.append(evidence)
        if run_exclusion is not None:
            excluded_runs.append({"run_id": run_id, "reason": run_exclusion})
            continue
        replicas.extend(run_replicas)
        aggregates.append(aggregate_deployable_configuration(registry, run_replicas))
    if not replicas:
        raise PermissionError("registry contains no selectable deployable publications")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_EVIDENCE_SET_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "selected_consumer_sha256": selected["content_hash"],
            "r0_checkpoint_sha256": sha256_file(r0_path),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "replicas": replicas,
            "aggregates": aggregates,
            "excluded_selectable_runs": excluded_runs,
            "all_nonmedian_weights_metrics_only": True,
            "selection_split": "model_val_select",
        }
    )


def select_deployable_from_publications(
    registry: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    r0_checkpoint_path: str | Path,
    selected_consumer_path: str | Path,
    semantic_evidence_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = assemble_deployable_replica_evidence(
        registry,
        artifact_root=artifact_root,
        r0_checkpoint_path=r0_checkpoint_path,
        selected_consumer_path=selected_consumer_path,
        semantic_evidence_root=semantic_evidence_root,
    )
    return evidence, select_deployable_preconfirmation(registry, evidence["aggregates"])


def _load_source_components(
    *,
    run_id: str,
    r0_checkpoint_path: str | Path,
    correction_checkpoint_path: str | Path,
    consumer_checkpoint_path: str | Path,
    physical45_scaler: Mapping[str, Any],
    device: Any,
) -> tuple[PredictionAnchoredDeployableBundle, dict[str, Any]]:
    spec = resolve_reconstruction_run(run_id)
    l0_postteacher = str(run_id) == "D10_L0_bridge_only"
    if (
        spec.direct
        or (spec.binding_kind != "primary" and not l0_postteacher)
        or spec.channel_policy != "physical45"
    ):
        raise PermissionError("primary deployment requires physical45 R0+correction+selected-T10")
    r0_path = _regular_file(r0_checkpoint_path, label="bundle R0 checkpoint")
    correction_path = _regular_file(
        correction_checkpoint_path, label="bundle correction checkpoint"
    )
    consumer_path = _regular_file(consumer_checkpoint_path, label="bundle consumer checkpoint")
    from .tagger import load_local_residual_reconstructor_from_checkpoint

    r0, r0_payload = load_local_residual_reconstructor_from_checkpoint(
        r0_path, map_location=device
    )
    correction_payload = torch.load(correction_path, map_location=device, weights_only=False)
    correction_config = dict(correction_payload.get("model_config") or {})
    correction_inner = dict(correction_config.get("config") or correction_config)
    correction, _ = build_reconstruction_model(
        run_id,
        physical45_scaler=physical45_scaler,
        c0_model_width=int(correction_inner.get("d_model", 160)),
        dropout=float(correction_inner.get("dropout", 0.0)),
    )
    correction.load_state_dict(correction_payload["model_state_dict"], strict=True)
    consumer, consumer_payload = load_local_residual_field_tagger_from_checkpoint(
        consumer_path, device=device
    )
    bundle = PredictionAnchoredDeployableBundle(r0, correction, consumer).to(device).eval()
    return bundle, {
        "r0_payload": r0_payload,
        "correction_payload": correction_payload,
        "consumer_payload": consumer_payload,
        "component_sha256": {
            "r0": sha256_file(r0_path),
            "correction": sha256_file(correction_path),
            "consumer": sha256_file(consumer_path),
        },
    }


def repository_bundle_factory(
    manifest: Mapping[str, Any], *, device: Any = "cpu"
) -> PredictionAnchoredDeployableBundle:
    """Instantiate a bundle using only metadata embedded in its checkpoint."""

    validate_content_hash(manifest)
    architecture = manifest["architecture_manifest"]
    r0_config = dict(architecture["r0_model_config"])
    r0_config.pop("contract", None)
    r0 = build_local_residual_field_reconstructor(
        LocalResidualFieldReconstructorConfig(**r0_config)
    )
    correction_config = dict(architecture["correction_model_config"])
    correction_inner = dict(correction_config.get("config") or correction_config)
    correction, _ = build_reconstruction_model(
        manifest["selected_run_id"],
        physical45_scaler=manifest["residual_normalization"],
        c0_model_width=int(correction_inner.get("d_model", 160)),
        dropout=float(correction_inner.get("dropout", 0.0)),
    )
    consumer_config = dict(architecture["consumer_model_config"])
    consumer_config.pop("contract", None)
    consumer_config.pop("augmented_feature_dim", None)
    consumer = LocalResidualFieldAugmentedParT(consumer_config)
    return PredictionAnchoredDeployableBundle(r0, correction, consumer).to(device).eval()


def _hlt_mapping(tokens: np.ndarray, mask: np.ndarray, *, device: Any) -> dict[str, torch.Tensor]:
    inputs = build_particle_transformer_inputs_from_tokens(
        np.asarray(tokens, dtype=np.float32), np.asarray(mask, dtype=bool), source_view="fixed_hlt"
    )
    return {
        "tokens": torch.as_tensor(tokens, dtype=torch.float32, device=device),
        "raw_mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "points": torch.as_tensor(inputs.pf_points, dtype=torch.float32, device=device),
        "features": torch.as_tensor(inputs.pf_features, dtype=torch.float32, device=device),
        "lorentz_vectors": torch.as_tensor(inputs.pf_vectors, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(inputs.pf_mask, dtype=torch.bool, device=device),
    }


def _read_bound_hlt_source(source: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = _regular_file(source["hlt_npz"]["path"], label="bound HLT NPZ")
    if sha256_file(path) != source["hlt_npz"]["sha256"]:
        raise ValueError("bound HLT NPZ hash changed")
    metadata_path = _regular_file(source["hlt_metadata"]["path"], label="bound HLT metadata")
    if sha256_file(metadata_path) != source["hlt_metadata"]["sha256"]:
        raise ValueError("bound HLT metadata hash changed")
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        arrays = {
            "tokens": np.asarray(data["tokens"], dtype=np.float32).copy(),
            "mask": np.asarray(data["mask"], dtype=bool).copy(),
            "labels": np.asarray(data["labels"], dtype=np.int64).copy(),
        }
    if arrays["tokens"].shape[:2] != arrays["mask"].shape:
        raise ValueError("HLT token/mask shapes differ")
    if arrays["tokens"].shape[0] != int(source["n_events"]):
        raise ValueError("HLT source event count changed")
    if not np.isfinite(arrays["tokens"]).all():
        raise ValueError("HLT source contains non-finite tokens")
    return arrays


def _evaluate_bundle(
    bundle: PredictionAnchoredDeployableBundle,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    class_order: Sequence[str],
    batch_size: int,
    device: Any,
    include_baseline: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    logits: list[np.ndarray] = []
    baseline_logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    bundle.eval()
    with torch.no_grad():
        for start in range(0, int(indices.size), int(batch_size)):
            selected = indices[start : start + int(batch_size)]
            mapping = _hlt_mapping(arrays["tokens"][selected], arrays["mask"][selected], device=device)
            logits.append(bundle(mapping).detach().float().cpu().numpy())
            if include_baseline:
                r0_output = bundle.r0(mapping["tokens"], mapping["raw_mask"])
                f0, _ = bundle._r0_values(r0_output)
                output = bundle.consumer(
                    mapping["points"], mapping["features"], mapping["lorentz_vectors"],
                    mapping["mask"], tokens=mapping["tokens"], raw_mask=mapping["raw_mask"],
                    residual_fields=f0,
                )
                baseline_logits.append(
                    getattr(output, "logits", output).detach().float().cpu().numpy()
                )
            labels.append(arrays["labels"][selected])
    if not logits:
        raise ValueError("authorized HLT evaluation split is empty")
    joined_labels = np.concatenate(labels)
    deploy = classification_metrics(np.concatenate(logits), joined_labels, class_order=class_order)
    baseline = None
    if include_baseline:
        baseline = classification_metrics(
            np.concatenate(baseline_logits), joined_labels, class_order=class_order
        )
    return deploy, baseline


def confirm_deployable_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    preconfirmation_path: str | Path,
    r0_checkpoint_path: str | Path,
    physical45_scaler_path: str | Path,
    selected_consumer_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: int = 512,
) -> dict[str, Any]:
    """Open stack_val_deploy once after locking and confirm the exact median."""

    spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(spec, verify_file_hashes=False)
    pre = load_hashed_json(
        preconfirmation_path,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    )
    child = load_hashed_json(
        spec["child_manifest"]["path"], expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    parent = load_split_manifest(spec["parent_manifest"]["path"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt = authorize_deployable_confirmation(
        child,
        parent=parent,
        preconfirmation=pre,
        receipt_path=output / "access_receipts" / "stack_val_deploy.json",
    )
    # authorize_deployable_confirmation returns the publication when a path is
    # supplied; load the exact immutable receipt for the confirmation contract.
    access = load_hashed_json(output / "access_receipts" / "stack_val_deploy.json")
    selected = load_hashed_json(
        selected_consumer_path,
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    consumer_path = _regular_file(selected["checkpoint_path"], label="selected T10 checkpoint")
    if sha256_file(consumer_path) != pre["teacher_sha256"]:
        raise ValueError("deployable preconfirmation selected another T10")
    correction_path = _regular_file(pre["checkpoint"], label="selected correction checkpoint")
    if sha256_file(correction_path) != pre["checkpoint_sha256"]:
        raise ValueError("selected correction checkpoint bytes changed")
    scaler = load_hashed_json(physical45_scaler_path)
    torch_device = torch.device(resolve_device(str(device)))
    bundle, _ = _load_source_components(
        run_id=pre["selected_run_id"],
        r0_checkpoint_path=r0_checkpoint_path,
        correction_checkpoint_path=correction_path,
        consumer_checkpoint_path=consumer_path,
        physical45_scaler=scaler,
        device=torch_device,
    )
    arrays = _read_bound_hlt_source(spec["sources"]["stack_val"])
    indices = np.asarray(
        child["children"]["stack_val_deploy"]["parent_row_indices"], dtype=np.int64
    )
    deploy, baseline = _evaluate_bundle(
        bundle,
        arrays,
        indices,
        class_order=spec["class_names"],
        batch_size=int(batch_size),
        device=torch_device,
        include_baseline=True,
    )
    assert baseline is not None
    confirmation = build_deployable_confirmation(
        pre,
        access_receipt=access,
        deployable_gain=float(deploy["accuracy"] - baseline["accuracy"]),
        accuracy=deploy["accuracy"],
        cross_entropy=deploy["cross_entropy"],
        provenance_valid=True,
    )
    write_immutable_json(output / "deployable_confirmation_metrics.json", confirmation)
    final = finalize_deployable_confirmation(pre, confirmation)
    destination = (
        output / "locked_deployable.json"
        if final.get("status") == "CONFIRMED_LOCKED"
        else output / "stopped_deployable.json"
    )
    write_immutable_json(destination, final)
    return {
        "contract": PREDICTION_ANCHORED_DEPLOY_CONFIRMATION_EXECUTION_CONTRACT,
        "ok": final.get("status") == "CONFIRMED_LOCKED",
        "status": final["status"],
        "decision": str(destination),
        "access_receipt": str(output / "access_receipts" / "stack_val_deploy.json"),
        "access_receipt_publication": receipt,
        "deployable_accuracy": deploy["accuracy"],
        "baseline_accuracy": baseline["accuracy"],
        "deployable_gain": float(deploy["accuracy"] - baseline["accuracy"]),
        "offline_npz_opened": False,
        "oracle_diagnostics_evaluated": False,
    }


def export_repository_deployable_bundle(
    execution_spec_path: str | Path,
    *,
    locked_deployable_path: str | Path,
    r0_checkpoint_path: str | Path,
    physical45_scaler_path: str | Path,
    selected_consumer_path: str | Path,
    output_dir: str | Path,
    bundle_reservation_bytes: int,
    device: str = "cpu",
) -> dict[str, Any]:
    spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    locked = load_hashed_json(
        locked_deployable_path, expected_contract=PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT
    )
    selected = load_hashed_json(
        selected_consumer_path,
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    consumer_path = _regular_file(selected["checkpoint_path"], label="selected T10 checkpoint")
    correction_path = _regular_file(locked["checkpoint"], label="locked correction checkpoint")
    scaler = load_hashed_json(physical45_scaler_path)
    torch_device = torch.device(resolve_device(str(device)))
    source, payloads = _load_source_components(
        run_id=locked["selected_run_id"],
        r0_checkpoint_path=r0_checkpoint_path,
        correction_checkpoint_path=correction_path,
        consumer_checkpoint_path=consumer_path,
        physical45_scaler=scaler,
        device=torch_device,
    )
    if payloads["component_sha256"]["correction"] != locked["checkpoint_sha256"]:
        raise ValueError("bundle correction differs from locked median")
    if payloads["component_sha256"]["consumer"] != locked["teacher_sha256"]:
        raise ValueError("bundle consumer differs from locked selected T10")
    manifest = build_deployable_bundle_manifest(
        locked,
        component_sha256=payloads["component_sha256"],
        preprocessing={"sha256": spec["preprocessing_sha256"], "source": "fixed_hlt"},
        residual_normalization=scaler,
        target_schema={"sha256": spec["target_schema_sha256"], "field_dim": 50},
        class_order=spec["class_names"],
        architecture_manifest={
            "r0_model_config": deepcopy(dict(payloads["r0_payload"]["model_config"])),
            "correction_model_config": deepcopy(
                dict(payloads["correction_payload"]["model_config"])
            ),
            "consumer_model_config": deepcopy(
                dict(payloads["consumer_payload"]["model_config"])
            ),
        },
        bundle_reservation_bytes=int(bundle_reservation_bytes),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("deployable output directory must be empty")
    checkpoint = output / "deployable_bundle.pt"
    publication = export_deployable_bundle(source, manifest=manifest, output_path=checkpoint)
    write_immutable_json(output / "bundle_manifest.json", manifest)

    arrays = _read_bound_hlt_source(spec["sources"]["stack_val"])
    fixed_count = min(8, arrays["tokens"].shape[0])
    fixed = _hlt_mapping(
        arrays["tokens"][:fixed_count], arrays["mask"][:fixed_count], device=torch_device
    )
    audit = clean_hlt_only_reload_audit(
        source,
        checkpoint_path=checkpoint,
        locked_deployable=locked,
        bundle_factory=lambda value: repository_bundle_factory(value, device=torch_device),
        fixed_hlt_batch=fixed,
        privileged_source_paths=(
            execution_spec_path,
            locked_deployable_path,
            r0_checkpoint_path,
            physical45_scaler_path,
            selected_consumer_path,
            correction_path,
            consumer_path,
            *(
                record[key]["path"]
                for record in spec["sources"].values()
                for key in ("offline_npz", "offline_metadata")
            ),
        ),
    )
    if not audit["passed"]:
        raise RuntimeError("clean HLT-only bundle reload changed logits")
    write_immutable_json(output / "clean_reload_audit.json", audit)
    result = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_EXPORT_CONTRACT,
            "locked_deployable_sha256": locked["content_hash"],
            "bundle_manifest_sha256": manifest["content_hash"],
            "bundle_checkpoint_sha256": publication["sha256"],
            "clean_reload_audit_sha256": audit["content_hash"],
            "component_sha256": payloads["component_sha256"],
            "hlt_only": True,
            "privileged_dependencies_included": False,
            "cached_target_logits_included": False,
            "offline_inputs_included": False,
            "single_inference_entrypoint": True,
        }
    )
    write_immutable_json(output / "export.json", result)
    return result


def evaluate_repository_bundle_final_test(
    *,
    bundle_checkpoint_path: str | Path,
    locked_deployable_path: str | Path,
    clean_reload_audit_path: str | Path,
    child_manifest_path: str | Path,
    parent_manifest_path: str | Path,
    final_hlt_npz_path: str | Path,
    final_hlt_metadata_path: str | Path,
    output_path: str | Path,
    hlt_only: bool,
    evaluation_flags: Mapping[str, Any] | None = None,
    device: str = "auto",
    batch_size: int = 512,
) -> dict[str, Any]:
    if hlt_only is not True:
        raise PermissionError("final-test executor requires explicit HLT-only mode")
    locked = load_hashed_json(
        locked_deployable_path, expected_contract=PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT
    )
    audit = load_hashed_json(clean_reload_audit_path)
    validate_final_test_request(
        locked, evaluation_flags=evaluation_flags, clean_reload_audit=audit
    )
    child = load_hashed_json(
        child_manifest_path, expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    parent = load_split_manifest(parent_manifest_path)
    receipt_path = Path(output_path).parent / "access_receipts" / "final_test.json"
    receipt = authorize_final_test(
        child,
        parent=parent,
        locked_deployable=locked,
        clean_reload_audit=audit,
        evaluation_flags=evaluation_flags,
        receipt_path=receipt_path,
    )
    hlt_path = _regular_file(final_hlt_npz_path, label="final-test HLT NPZ")
    metadata_path = _regular_file(final_hlt_metadata_path, label="final-test HLT metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_manifest_hash") not in (None, manifest_hash(parent)):
        raise ValueError("final-test HLT cache belongs to another parent manifest")
    from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash

    if metadata.get("jet_identity_hash") != jet_identity_hash(parent.splits["final_test"]):
        raise ValueError("final-test HLT cache event order changed")
    with hlt_path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        arrays = {
            "tokens": np.asarray(data["tokens"], dtype=np.float32).copy(),
            "mask": np.asarray(data["mask"], dtype=bool).copy(),
            "labels": np.asarray(data["labels"], dtype=np.int64).copy(),
            "jet_file_indices": np.asarray(data["jet_file_indices"], dtype=np.int32).copy(),
            "jet_entries": np.asarray(data["jet_entries"], dtype=np.int64).copy(),
        }
    actual_content_hash = hash_arrays(arrays)
    if actual_content_hash != metadata.get("hlt_content_hash"):
        raise ValueError("final-test HLT cache content hash changed")
    if arrays["tokens"].shape[0] != len(parent.splits["final_test"]):
        raise ValueError("final-test HLT event count changed")
    torch_device = torch.device(resolve_device(str(device)))
    bundle, manifest = load_deployable_bundle(
        bundle_checkpoint_path,
        bundle_factory=lambda value: repository_bundle_factory(value, device=torch_device),
        map_location=torch_device,
    )
    if manifest["locked_deployable_sha256"] != locked["content_hash"]:
        raise ValueError("final-test bundle belongs to another locked deployment")
    metrics, _ = _evaluate_bundle(
        bundle,
        arrays,
        np.arange(arrays["tokens"].shape[0], dtype=np.int64),
        class_order=manifest["class_order"],
        batch_size=int(batch_size),
        device=torch_device,
        include_baseline=False,
    )
    report = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_FINAL_TEST_REPORT_CONTRACT,
            "locked_deployable_sha256": locked["content_hash"],
            "bundle_checkpoint_sha256": sha256_file(bundle_checkpoint_path),
            "bundle_manifest_sha256": manifest["content_hash"],
            "access_receipt_sha256": load_hashed_json(receipt_path)["content_hash"],
            "selected_run_id": locked["selected_run_id"],
            "median_seed_id": locked["median_seed_id"],
            "event_count": arrays["tokens"].shape[0],
            "metrics": metrics,
            "hlt_only": True,
            "offline_npz_opened": False,
            "oracle_diagnostics_evaluated": False,
            "bridge_diagnostics_evaluated": False,
            "target_logits_loaded": False,
            "selection_changes_allowed": False,
            "final_test_used_for_ranking": False,
        }
    )
    write_immutable_json(output_path, report)
    return {"ok": True, "report": report, "access_receipt_publication": receipt}


__all__ = [
    "PREDICTION_ANCHORED_DEPLOYABLE_SEMANTIC_REPLICA_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_EVIDENCE_SET_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOY_CONFIRMATION_EXECUTION_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_EXPORT_CONTRACT",
    "PREDICTION_ANCHORED_FINAL_TEST_REPORT_CONTRACT",
    "build_deployable_semantic_replica_evidence",
    "assemble_deployable_replica_evidence",
    "select_deployable_from_publications",
    "repository_bundle_factory",
    "confirm_deployable_from_execution_spec",
    "export_repository_deployable_bundle",
    "evaluate_repository_bundle_final_test",
]
