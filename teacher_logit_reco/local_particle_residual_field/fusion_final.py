"""Step 10 selection-gated final evaluation for the P7b fusion campaign."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from jetclass_fresh.fusion import PredictionBlock, load_prediction_block, validate_prediction_alignment
from jetclass_fresh.hlt_baseline import amp_autocast_context, resolve_device
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .curriculum import LocalResidualFieldCurriculumJointModel
from .data import move_local_particle_residual_field_batch_to_device
from .fusion import (
    LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
    LocalResidualFieldPredictionConfig,
    _make_prediction_loader,
    _prediction_dataset,
    cache_local_residual_field_tagger_predictions,
    load_local_residual_field_tagger_from_checkpoint,
)
from .fusion_campaign import (
    FUSION_FINAL_SPLIT,
    FUSION_FINAL_TEST_STATUS,
    FUSION_MEMBER_A0,
    FusionCampaignConfig,
    SelectedFusionArtifact,
    default_fusion_candidate_specs,
    stable_fusion_json_hash,
)
from .fusion_features import (
    FUSION_FEATURE_AMP_LOGITS_ATOL,
    FUSION_FEATURE_FP32_LOGITS_ATOL,
    FUSION_FEATURE_LOGITS_RTOL,
    PreClassifierEmbeddingCapture,
    _campaign_forward,
    require_development_prediction_sources,
)
from .fusion_late import apply_late_fusion_candidate
from .fusion_metrics import (
    freeze_binary_projection_thresholds,
    local_residual_field_binary_projection_metrics,
    local_residual_field_complementarity_metrics,
    local_residual_field_multiclass_metrics,
    paired_binary_projection_bootstrap,
    paired_multiclass_bootstrap,
)
from .fusion_seed_control import sha256_file
from .fusion_selection import (
    FUSION_HEADLINE_SIGNALS,
    _validate_candidate_report,
    load_selected_fusion_set,
)
from .fusion_sources import require_fusion_source_artifact_audit
from .fusion_train import (
    RepresentationFusionSplitData,
    load_representation_fusion_head_from_checkpoint,
    predict_representation_fusion_head,
)


LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT = "local_residual_field_fusion_final_evaluation_v1"


def _try_lock_file(handle) -> bool:
    """Take a non-blocking, process-owned lock on the first byte of a file."""

    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return False
    return True


def _unlock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _final_evaluation_lock(published_root: Path):
    """Own one selection's publication with an OS-released advisory lock."""

    published_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = published_root.with_name(f".{published_root.name}.lock")
    # The byte is required by Windows' msvcrt.locking implementation.  Keep
    # the lock file permanently: deleting a locked pathname would reintroduce
    # a race in which two processes lock different inodes with the same name.
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        if not _try_lock_file(handle):
            raise RuntimeError(f"final evaluation lock is already held: {lock_path}")
        try:
            yield
        finally:
            _unlock_file(handle)
    finally:
        handle.close()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _development_feature_config(feature_root: Path, member: str) -> dict[str, Any]:
    path = feature_root / member / "representation_manifest.json"
    manifest = _read_json(path)
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_hash", None)
    if stored != stable_fusion_json_hash(unsigned) or manifest.get("final_test_opened") is not False:
        raise ValueError(f"invalid development representation manifest for {member}")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"representation manifest lacks extraction config for {member}")
    if sha256_file(config["checkpoint"]) != manifest.get("checkpoint_hash"):
        raise ValueError(f"member checkpoint changed after representation extraction: {member}")
    for split, row in dict(manifest.get("splits") or {}).items():
        if not isinstance(row, Mapping):
            raise ValueError(f"malformed representation row for {member}/{split}")
        representation_path = Path(str(row.get("representation_path") or ""))
        if not representation_path.is_file():
            representation_path = feature_root / member / f"{split}_representations.npz"
        if sha256_file(representation_path) != row.get("representation_sha256"):
            raise ValueError(f"development representation changed after candidate fitting: {member}/{split}")
    return dict(config)


def _load_or_cache_final_prediction(
    *,
    member: str,
    original_root: str | Path,
    feature_config: Mapping[str, Any],
    expected_checkpoint_hash: str,
    fallback_root: Path,
) -> PredictionBlock:
    try:
        block = load_prediction_block(original_root, member, FUSION_FINAL_SPLIT, verify_hash=True)
    except FileNotFoundError:
        cache_local_residual_field_tagger_predictions(
            LocalResidualFieldPredictionConfig(
                checkpoint=str(feature_config["checkpoint"]), prediction_dir=str(fallback_root), model_name=member,
                hlt_cache_dir=str(feature_config["hlt_cache_dir"]), target_cache_dir=None,
                manifest_path=str(feature_config["manifest_path"]), splits=(FUSION_FINAL_SPLIT,),
                batch_size=int(feature_config.get("batch_size", 128)),
                num_workers=int(feature_config.get("num_workers", 0)),
                device=str(feature_config.get("device", "auto")), amp=bool(feature_config.get("amp", True)),
                confirm_final_test=True, allow_oracle_final_test=False, overwrite=False,
            )
        )
        block = load_prediction_block(fallback_root, member, FUSION_FINAL_SPLIT, verify_hash=True)
    metadata = block.metadata
    checkpoint_hash = metadata.get("checkpoint_hash") or metadata.get("student_checkpoint_hash")
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError(f"final prediction checkpoint hash mismatch for {member}")
    if metadata.get("runtime_inputs") != "HLT_only" or metadata.get("deployable") is not True:
        raise ValueError(f"final prediction is not deployable HLT-only for {member}")
    if any(bool(metadata.get(key)) for key in ("uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime")):
        raise ValueError(f"final prediction uses privileged runtime inputs for {member}")
    if metadata.get("selection_allowed") is not False:
        raise ValueError(f"final prediction incorrectly permits selection for {member}")
    return block


def _extract_final_embedding(
    *,
    member: str,
    block: PredictionBlock,
    feature_config: Mapping[str, Any],
    output_root: Path,
) -> np.ndarray:
    device = resolve_device(str(feature_config.get("device", "auto")))
    amp_enabled = bool(feature_config.get("amp", True) and getattr(device, "type", str(device)) == "cuda")
    model, _payload = load_local_residual_field_tagger_from_checkpoint(feature_config["checkpoint"], device=device)
    if isinstance(model, LocalResidualFieldCurriculumJointModel) and model.oracle_consumer is not None:
        raise ValueError("selected deployable member unexpectedly loaded an oracle consumer")
    capture = PreClassifierEmbeddingCapture(model, num_classes=len(LABEL_NAMES))
    prediction_config = LocalResidualFieldPredictionConfig(
        checkpoint=str(feature_config["checkpoint"]), prediction_dir="unused", model_name=member,
        hlt_cache_dir=str(feature_config["hlt_cache_dir"]), target_cache_dir=None,
        manifest_path=str(feature_config["manifest_path"]), splits=(FUSION_FINAL_SPLIT,),
        batch_size=int(feature_config.get("batch_size", 128)), num_workers=int(feature_config.get("num_workers", 0)),
        device=str(feature_config.get("device", "auto")), amp=bool(feature_config.get("amp", True)),
        confirm_final_test=True, allow_oracle_final_test=False,
    )
    dataset = _prediction_dataset(prediction_config, FUSION_FINAL_SPLIT, model=model)
    loader = _make_prediction_loader(
        dataset, batch_size=prediction_config.batch_size, num_workers=prediction_config.num_workers,
        seed=0, hlt_only=True,
    )
    embedding_rows, logit_rows, label_rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_local_particle_residual_field_batch_to_device(batch, device)
            with amp_autocast_context(amp_enabled):
                logits, embedding = capture.capture(lambda: _campaign_forward(model, batch))
            embedding_rows.append(embedding.detach().float().cpu().numpy())
            logit_rows.append(logits.detach().float().cpu().numpy())
            label_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    embeddings = np.concatenate(embedding_rows).astype(np.float32)
    logits = np.concatenate(logit_rows).astype(np.float32)
    labels = np.concatenate(label_rows).astype(np.int64)
    logits_atol = float(
        feature_config.get(
            "logits_atol",
            FUSION_FEATURE_AMP_LOGITS_ATOL if amp_enabled else FUSION_FEATURE_FP32_LOGITS_ATOL,
        )
    )
    logits_rtol = float(feature_config.get("logits_rtol", FUSION_FEATURE_LOGITS_RTOL))
    labels_agree = np.array_equal(labels, block.labels)
    logits_agree = logits.shape == block.logits.shape and np.allclose(
        logits,
        block.logits,
        atol=logits_atol,
        rtol=logits_rtol,
    )
    if not labels_agree or not logits_agree:
        max_abs = (
            float(np.max(np.abs(logits - block.logits)))
            if logits.shape == block.logits.shape
            else None
        )
        raise ValueError(
            f"final representation forward does not reproduce selected predictions for {member}; "
            f"max_abs={max_abs}, atol={logits_atol}, rtol={logits_rtol}"
        )
    path = output_root / "representations" / member / "final_test_representations.npz"
    _atomic_npz(path, jet_embedding=embeddings.astype(np.float16), labels=labels)
    return embeddings


def _validate_selection_dependencies(_selection_path: Path, selection: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = FusionCampaignConfig(campaign_id=selection["campaign_id"])
    registry_path = Path(selection["candidate_registry_path"])
    if sha256_file(registry_path) != selection.get("candidate_registry_sha256"):
        raise ValueError("candidate registry bytes changed after selection")
    registry_artifact = _read_json(registry_path)
    unsigned_registry = dict(registry_artifact)
    stored_registry_hash = unsigned_registry.pop("artifact_hash", None)
    if stored_registry_hash != stable_fusion_json_hash(unsigned_registry):
        raise ValueError("candidate registry logical hash mismatch")
    if registry_artifact.get("candidate_registry_hash") != campaign.candidate_registry_hash:
        raise ValueError("candidate registry does not match the selected campaign")
    audit = require_fusion_source_artifact_audit(selection["source_artifact_audit_path"])
    if audit.get("audit_hash") != selection.get("source_artifact_audit_hash"):
        raise ValueError("source artifact audit differs from selected_fusion.json")
    prediction_registry = require_development_prediction_sources(
        selection["prediction_sources_path"], source_artifact_audit=selection["source_artifact_audit_path"],
    )
    if prediction_registry.get("manifest_hash") != selection.get("prediction_sources_hash"):
        raise ValueError("prediction sources differ from selected_fusion.json")
    records = {
        (row["group_id"], row["champion_role"]): SelectedFusionArtifact(**row)
        for row in selection["selections"]
    }
    for binding in selection["selection_bindings"]:
        path = Path(binding["candidate_report_path"])
        if sha256_file(path) != binding.get("candidate_report_sha256"):
            raise ValueError(f"selected candidate report bytes changed: {path}")
        report = _validate_candidate_report(path)
        if report.get("artifact_hash") != binding.get("candidate_report_artifact_hash"):
            raise ValueError(f"selected candidate report logical hash changed: {path}")
        record = records[(binding["group_id"], binding["champion_role"])]
        if record.fit_artifact_hashes.get("candidate_report") != report["artifact_hash"]:
            raise ValueError("selection record does not bind its candidate report")
        if report.get("candidate_id") != record.candidate_id or report.get("member_checkpoint_hashes") != dict(record.member_checkpoint_hashes):
            raise ValueError("candidate report identity differs from selection record")
        fit_artifacts = report.get("fit_artifacts")
        if not isinstance(fit_artifacts, list) or not fit_artifacts:
            raise ValueError("selected candidate report does not bind any fit artifacts")
        for fit_artifact in fit_artifacts:
            fit_path = Path(str(fit_artifact.get("path") or ""))
            if not fit_path.is_file() or sha256_file(fit_path) != fit_artifact.get("sha256"):
                raise ValueError(f"selected fit artifact changed or disappeared: {fit_path}")
    return audit, prediction_registry


def _final_status(selection: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    pristine = selection.get("pristine_confirmation_manifest")
    if pristine is None:
        return FUSION_FINAL_TEST_STATUS, None
    path = Path(pristine["path"])
    if sha256_file(path) != pristine.get("sha256"):
        raise ValueError("pristine confirmation manifest changed after selection")
    payload = _read_json(path)
    if payload.get("ok") is not True or payload.get("pristine_confirmation") is not True or payload.get("split") != FUSION_FINAL_SPLIT:
        raise ValueError("pristine confirmation manifest does not certify final_test")
    return "pristine_confirmation", payload


def _evaluate_selected_fusion_final_impl(
    selected_fusion_json: str | Path,
    *,
    selection: Mapping[str, Any],
    selection_sha256: str,
    staging_paths: list[Path],
) -> dict[str, Any]:
    """Evaluate only the immutable choices in selected_fusion.json; no override API exists."""

    selection_path = Path(selected_fusion_json).resolve()
    audit, prediction_registry = _validate_selection_dependencies(selection_path, selection)
    final_status, pristine = _final_status(selection)
    published_root = selection_path.parent.parent / "final_evaluation" / selection["artifact_hash"][:16]
    published_root.parent.mkdir(parents=True, exist_ok=True)
    if published_root.exists():
        if (published_root / "final_evaluation.json").is_file():
            raise FileExistsError(f"immutable final result directory already exists: {published_root}")
        quarantine = published_root.with_name(
            f"{published_root.name}.partial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        published_root.replace(quarantine)
    for abandoned in published_root.parent.glob(f".{published_root.name}.staging_*"):
        # The exclusive advisory lock proves that no cooperating evaluator can
        # still own one of these staging directories.
        if abandoned.is_dir():
            shutil.rmtree(abandoned, ignore_errors=True)
    result_root = Path(tempfile.mkdtemp(prefix=f".{published_root.name}.staging_", dir=str(published_root.parent)))
    staging_paths.append(result_root)
    feature_root = Path(selection["feature_root"])
    all_checkpoint_hashes: dict[str, str] = {}
    for row in selection["selections"]:
        for member, value in row["member_checkpoint_hashes"].items():
            previous = all_checkpoint_hashes.setdefault(member, value)
            if previous != value:
                raise ValueError(f"selection contains conflicting checkpoint hashes for {member}")
    feature_configs = {
        member: _development_feature_config(feature_root, member) for member in all_checkpoint_hashes
    }
    blocks: dict[str, PredictionBlock] = {}
    for member, checkpoint_hash in all_checkpoint_hashes.items():
        blocks[member] = _load_or_cache_final_prediction(
            member=member, original_root=selection["member_prediction_roots"][member],
            feature_config=feature_configs[member], expected_checkpoint_hash=checkpoint_hash,
            fallback_root=result_root / "input_predictions",
        )
        alignment = blocks[member].metadata.get("dataset_metadata", {}).get("alignment_report", {})
        if alignment.get("source_manifest_hash") != audit.get("split_manifest_hash"):
            raise ValueError(f"final prediction split-manifest mismatch for {member}")
        expected_hlt = audit.get("hlt_content_hashes", {}).get(FUSION_FINAL_SPLIT)
        if not expected_hlt or alignment.get("hlt_content_hash") != expected_hlt:
            raise ValueError(f"final prediction HLT-content mismatch for {member}")
    validate_prediction_alignment(list(blocks.values()))
    labels = next(iter(blocks.values())).labels
    embeddings: dict[str, np.ndarray] = {}
    candidate_specs = {spec.candidate_id: spec for spec in default_fusion_candidate_specs()}
    binding_lookup = {
        (row["group_id"], row["champion_role"]): row for row in selection["selection_bindings"]
    }
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for selected_row in selection["selections"]:
        key = (selected_row["group_id"], selected_row["candidate_id"])
        unique.setdefault(key, {"roles": [], "selected": selected_row})["roles"].append(selected_row["champion_role"])
    result_rows: list[dict[str, Any]] = []
    for (group_id, candidate_id), item in unique.items():
        selected_row = item["selected"]
        member_a, member_b = selected_row["member_ids"]
        binding = binding_lookup[(group_id, item["roles"][0])]
        candidate_report = _validate_candidate_report(binding["candidate_report_path"])
        spec = candidate_specs[candidate_id]
        if spec.family == "late":
            fused_logits = apply_late_fusion_candidate(
                candidate_id, selected_row["hyperparameters"], blocks[member_a].logits, blocks[member_b].logits,
            )
            head_diagnostics: dict[str, Any] = {}
        else:
            for member in (member_a, member_b):
                if member not in embeddings:
                    embeddings[member] = _extract_final_embedding(
                        member=member, block=blocks[member], feature_config=feature_configs[member], output_root=result_root,
                    )
            data = RepresentationFusionSplitData(
                embedding_a=embeddings[member_a], embedding_b=embeddings[member_b],
                logits_a=blocks[member_a].logits, logits_b=blocks[member_b].logits, labels=labels,
            )
            head_logits, head_diagnostics_rows = [], []
            deployed_heads = (
                candidate_report["head_artifacts"][:1]
                if candidate_id == "R0_linear_embeddings"
                else candidate_report["head_artifacts"]
            )
            for head in deployed_heads:
                if sha256_file(head["checkpoint_path"]) != head["checkpoint_hash"]:
                    raise ValueError("selected fusion-head checkpoint changed before final evaluation")
                model, payload = load_representation_fusion_head_from_checkpoint(head["checkpoint_path"])
                if payload["source_hashes"]["prediction_sources_hash"] != selection["prediction_sources_hash"]:
                    raise ValueError("selected fusion head does not bind the active prediction sources")
                logits, diagnostics = predict_representation_fusion_head(model, data)
                head_logits.append(logits)
                head_diagnostics_rows.append({"seed": head["seed"], **diagnostics})
            fused_logits = np.mean(np.stack(head_logits).astype(np.float64), axis=0).astype(np.float32)
            head_diagnostics = {
                "heads": head_diagnostics_rows, "averaged_head_count": len(head_logits),
                "deployment_rule": (
                    "single_fixed_seed_linear_head"
                    if candidate_id == "R0_linear_embeddings"
                    else "mean_head_logits"
                ),
            }
        thresholds = freeze_binary_projection_thresholds(selected_row["selection_metrics"]["binary_projection"])
        multiclass = local_residual_field_multiclass_metrics(fused_logits, labels, label_names=LABEL_NAMES)
        binary = local_residual_field_binary_projection_metrics(
            fused_logits, labels, label_names=LABEL_NAMES, frozen_thresholds=thresholds,
        )
        bootstrap = paired_multiclass_bootstrap(
            blocks[FUSION_MEMBER_A0].logits, fused_logits, labels, label_names=LABEL_NAMES,
        )
        binary_bootstrap = {
            signal: paired_binary_projection_bootstrap(
                blocks[FUSION_MEMBER_A0].logits, fused_logits, labels, label_names=LABEL_NAMES, signal_name=signal,
            )
            for signal in FUSION_HEADLINE_SIGNALS
        }
        logits_path = result_root / "fused_logits" / f"{group_id}__{candidate_id}.npz"
        published_logits_path = published_root / "fused_logits" / f"{group_id}__{candidate_id}.npz"
        _atomic_npz(logits_path, logits=fused_logits, labels=labels)
        result_rows.append({
            "run_id": f"{group_id}/{candidate_id}", "group_id": group_id, "candidate_id": candidate_id,
            "champion_roles": item["roles"], "member_ids": [member_a, member_b],
            "multiclass": multiclass, "binary_projection": binary,
            "stack_val_frozen_thresholds": thresholds,
            "member_complementarity": local_residual_field_complementarity_metrics(
                blocks[member_a].logits, blocks[member_b].logits, labels,
                label_names=LABEL_NAMES, member_a=member_a, member_b=member_b,
            ),
            "paired_bootstrap_vs_A0": bootstrap, "paired_binary_bootstrap_vs_A0": binary_bootstrap,
            "head_diagnostics": head_diagnostics,
            "logits_path": str(published_logits_path.resolve()), "logits_sha256": sha256_file(logits_path),
            "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False, "deployable": True,
        })
    member_metrics = {
        member: {
            "multiclass": local_residual_field_multiclass_metrics(block.logits, block.labels, label_names=LABEL_NAMES),
            "binary_projection": local_residual_field_binary_projection_metrics(block.logits, block.labels, label_names=LABEL_NAMES),
            "checkpoint_hash": all_checkpoint_hashes[member],
            "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False, "deployable": True,
        }
        for member, block in blocks.items()
    }
    report: dict[str, Any] = {
        "ok": True, "contract": LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT,
        "campaign_id": selection["campaign_id"], "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_fusion_path": str(selection_path), "selected_fusion_sha256": selection_sha256,
        "selected_fusion_artifact_hash": selection["artifact_hash"],
        "selection_source": "stack_val", "final_split": FUSION_FINAL_SPLIT,
        "final_test_status": final_status, "pristine_confirmation": pristine,
        "member_metrics": member_metrics, "selected_results": result_rows,
        "source_artifact_audit_hash": selection["source_artifact_audit_hash"],
        "prediction_sources_hash": prediction_registry["manifest_hash"],
        "runtime_inputs": "HLT_only", "uses_true_fields": False, "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False, "deployable": True,
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    if sha256_file(selection_path) != selection_sha256:
        raise ValueError("selected fusion artifact changed during final evaluation")
    _atomic_json(result_root / "final_evaluation.json", report)
    os.replace(result_root, published_root)
    return report


def evaluate_selected_fusion_final(selected_fusion_json: str | Path) -> dict[str, Any]:
    """Evaluate a frozen selection and remove unpublished staging data after failures."""

    staging_paths: list[Path] = []
    selection_path = Path(selected_fusion_json).resolve()
    selection_sha256 = sha256_file(selection_path)
    selection = load_selected_fusion_set(selection_path)
    if sha256_file(selection_path) != selection_sha256:
        raise ValueError("selected fusion artifact changed while being loaded")
    published_root = selection_path.parent.parent / "final_evaluation" / selection["artifact_hash"][:16]
    try:
        with _final_evaluation_lock(published_root):
            if sha256_file(selection_path) != selection_sha256:
                raise ValueError("selected fusion artifact changed before final evaluation lock acquisition")
            return _evaluate_selected_fusion_final_impl(
                selection_path,
                selection=selection,
                selection_sha256=selection_sha256,
                staging_paths=staging_paths,
            )
    except BaseException:
        for staging_path in staging_paths:
            if staging_path.is_dir():
                shutil.rmtree(staging_path, ignore_errors=True)
        raise


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FUSION_FINAL_EVALUATION_CONTRACT",
    "evaluate_selected_fusion_final",
]
