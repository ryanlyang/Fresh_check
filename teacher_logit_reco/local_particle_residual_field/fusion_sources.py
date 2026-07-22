"""Hash-attested source and prediction gates for the P7b fusion campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from jetclass_fresh.fusion import load_prediction_block, prediction_paths, validate_prediction_alignment
from jetclass_fresh.jetclass_data import load_split_manifest, manifest_hash

from .curriculum import (
    LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
    load_selected_consumer_record,
)
from .fusion import (
    LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
    LocalResidualFieldPredictionConfig,
    cache_local_residual_field_tagger_predictions,
    load_local_residual_field_tagger_from_checkpoint,
)
from .fusion_atomic import publish_temporary_file
from .fusion_campaign import (
    FUSION_FIT_SPLIT,
    FUSION_MEMBER_A0,
    FUSION_MEMBER_A0_SEED1,
    FUSION_MEMBER_P7B,
    FUSION_SELECTION_SPLIT,
    stable_fusion_json_hash,
)
from .fusion_seed_control import sha256_file
from .fusion_metrics import local_residual_field_binary_projection_metrics
from jetclass_fresh.jetclass_data import LABEL_NAMES


LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT_CONTRACT = "local_residual_field_fusion_source_artifact_audit_v1"
LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT = "local_residual_field_fusion_prediction_sources_v1"
FUSION_DEVELOPMENT_SPLITS = (FUSION_FIT_SPLIT, FUSION_SELECTION_SPLIT)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {source}")
    return dict(payload)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    output = Path(path)
    if output.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, output, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime_contract_problems(metadata: Mapping[str, Any], *, context: str) -> list[str]:
    expected = {
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
    }
    return [
        f"{context} {key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]


def _split_from_hlt_path(value: Any) -> str | None:
    name = Path(str(value or "")).name
    suffix = "_fixed_hlt_metadata.json"
    return name[: -len(suffix)] if name.endswith(suffix) else None


def _hlt_content_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    audits = payload.get("audits")
    if isinstance(audits, Mapping) and isinstance(audits.get("hlt_cache"), list):
        for row in audits["hlt_cache"]:
            if not isinstance(row, Mapping):
                continue
            split = _split_from_hlt_path(row.get("path"))
            hashes = row.get("content_hashes")
            value = hashes.get("hlt_content_hash") if isinstance(hashes, Mapping) else None
            if split and value:
                output[split] = str(value)
    split_reports = payload.get("split_reports")
    if isinstance(split_reports, Mapping):
        for split, row in split_reports.items():
            if isinstance(row, Mapping) and row.get("hlt_content_hash"):
                output[str(split)] = str(row["hlt_content_hash"])
    return output


def _attested_binary_projection_metrics(block: Any) -> tuple[dict[str, Any], str]:
    """Return deterministic projections, checking any metrics already in the cache."""

    computed = local_residual_field_binary_projection_metrics(
        block.logits,
        block.labels,
        label_names=LABEL_NAMES,
    )
    stored = block.metadata.get("binary_projection_metrics")
    if stored is None:
        return computed, "computed_from_hash_validated_legacy_predictions"
    if not isinstance(stored, Mapping):
        raise ValueError("binary_projection_metrics is not a JSON object")
    stored_payload = {
        "contract": block.metadata.get(
            "binary_projection_metrics_contract",
            computed["contract"],
        ),
        "label_names": list(LABEL_NAMES),
        "signal_efficiencies": block.metadata.get(
            "binary_projection_signal_efficiencies",
            computed["signal_efficiencies"],
        ),
        "projections": dict(stored),
    }
    if stable_fusion_json_hash(stored_payload) != stable_fusion_json_hash(computed):
        raise ValueError("stored binary-projection metrics do not match cached logits")
    return computed, "prediction_metadata_verified_against_logits"


@dataclass
class FusionSourceArtifactAuditConfig:
    """Exact source paths that must be frozen before campaign GPU work."""

    output_path: str
    a0_checkpoint: str
    a0_report: str
    p7b_checkpoint: str
    p7b_report: str
    selected_consumer_json: str
    manifest_path: str
    hlt_cache_manifest: str
    a0_prediction_dir: str
    p7b_prediction_dir: str
    required_splits: tuple[str, ...] = FUSION_DEVELOPMENT_SPLITS
    verify_prediction_hash: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "output_path",
            "a0_checkpoint",
            "a0_report",
            "p7b_checkpoint",
            "p7b_report",
            "selected_consumer_json",
            "manifest_path",
            "hlt_cache_manifest",
            "a0_prediction_dir",
            "p7b_prediction_dir",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            setattr(self, field_name, value)
        self.required_splits = tuple(str(value) for value in self.required_splits)
        if self.required_splits != FUSION_DEVELOPMENT_SPLITS:
            raise ValueError(f"source preflight splits are locked to {FUSION_DEVELOPMENT_SPLITS}")
        self.verify_prediction_hash = bool(self.verify_prediction_hash)
        self.overwrite = bool(self.overwrite)


def audit_fusion_source_artifacts(
    config: FusionSourceArtifactAuditConfig,
    *,
    model_loader: Callable[..., tuple[Any, Mapping[str, Any]]] = load_local_residual_field_tagger_from_checkpoint,
) -> dict[str, Any]:
    """Audit frozen inputs, always writing a pass/fail artifact unless the output is immutable."""

    if Path(config.output_path).exists() and not config.overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {config.output_path}")
    problems: list[str] = []
    hashed_files: list[dict[str, Any]] = []

    def record(role: str, path_value: str | Path) -> dict[str, Any]:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
        row = {"role": role, "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        hashed_files.append(row)
        return row

    artifacts: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    paths = {
        "a0_checkpoint": config.a0_checkpoint,
        "a0_report": config.a0_report,
        "p7b_checkpoint": config.p7b_checkpoint,
        "p7b_report": config.p7b_report,
        "selected_consumer_json": config.selected_consumer_json,
        "split_manifest": config.manifest_path,
        "hlt_cache_manifest": config.hlt_cache_manifest,
    }
    for role, path in paths.items():
        try:
            artifacts[role] = record(role, path)
            if role not in {"a0_checkpoint", "p7b_checkpoint", "split_manifest"}:
                loaded[role] = _read_json(path)
        except Exception as exc:
            problems.append(str(exc))

    manifest_sha: str | None = None
    try:
        manifest_sha = manifest_hash(load_split_manifest(config.manifest_path))
    except Exception as exc:
        problems.append(f"invalid split manifest: {exc}")

    selected = None
    try:
        selected = load_selected_consumer_record(config.selected_consumer_json)
    except Exception as exc:
        problems.append(f"invalid selected_consumer.json: {exc}")

    hlt_payload = loaded.get("hlt_cache_manifest", {})
    hlt_hashes = _hlt_content_hashes(hlt_payload)
    if hlt_payload.get("ok") is not True:
        problems.append("HLT-cache manifest is not an ok=true audit")
    hlt_manifest_sha = hlt_payload.get("source_manifest_hash") or hlt_payload.get("manifest_hash")
    if manifest_sha and str(hlt_manifest_sha or "") != manifest_sha:
        problems.append(f"HLT-cache manifest source hash {hlt_manifest_sha!r} != split manifest hash {manifest_sha!r}")
    missing_hlt_splits = [split for split in config.required_splits if split not in hlt_hashes]
    if missing_hlt_splits:
        problems.append(f"HLT-cache manifest is missing development splits: {missing_hlt_splits}")

    checkpoint_hashes = {
        FUSION_MEMBER_A0: artifacts.get("a0_checkpoint", {}).get("sha256"),
        FUSION_MEMBER_P7B: artifacts.get("p7b_checkpoint", {}).get("sha256"),
    }
    for member, report_key in ((FUSION_MEMBER_A0, "a0_report"), (FUSION_MEMBER_P7B, "p7b_report")):
        report = loaded.get(report_key, {})
        if report.get("ok") is not True:
            problems.append(f"{member} report is not ok=true")
        stored_hash = report.get("checkpoint_hash")
        if stored_hash and str(stored_hash) != str(checkpoint_hashes[member]):
            problems.append(f"{member} report checkpoint hash does not match the checkpoint bytes")
    a0_report = loaded.get("a0_report", {})
    if str(a0_report.get("field_source") or "") != "hlt_only":
        problems.append("A0 report must identify field_source='hlt_only'")
    p7b_report = loaded.get("p7b_report", {})
    problems.extend(_runtime_contract_problems(p7b_report, context="P7b report"))
    if selected is not None:
        if p7b_report.get("selected_consumer_id") != selected.selected_consumer_id:
            problems.append("P7b report selected consumer does not match selected_consumer.json")
        try:
            endpoint_matches = abs(float(p7b_report.get("selected_alpha_endpoint")) - float(selected.selected_alpha_endpoint)) <= 1.0e-12
        except (TypeError, ValueError):
            endpoint_matches = False
        if not endpoint_matches:
            problems.append("P7b report selected alpha endpoint does not match selected_consumer.json")

    oracle_free_load: dict[str, Any] = {"ok": False, "checkpoint": str(Path(config.p7b_checkpoint).resolve())}
    try:
        model, payload = model_loader(config.p7b_checkpoint, device="cpu")
        if payload.get("contract") != LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT:
            raise ValueError("P7b checkpoint is not a curriculum deployable checkpoint")
        if payload.get("oracle_consumer_included") is not False:
            raise ValueError("P7b deployable checkpoint does not explicitly exclude the oracle consumer")
        if getattr(model, "oracle_consumer", None) is not None:
            raise ValueError("loaded P7b deployable model contains an oracle consumer")
        payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        if payload_metadata.get("runtime_inputs") not in (None, "HLT_only"):
            raise ValueError("P7b checkpoint metadata is not HLT-only")
        oracle_free_load = {
            "ok": True,
            "checkpoint": str(Path(config.p7b_checkpoint).resolve()),
            "checkpoint_hash": checkpoint_hashes[FUSION_MEMBER_P7B],
            "contract": payload.get("contract"),
            "oracle_consumer_included": False,
            "oracle_consumer_after_load": False,
            "device": "cpu",
        }
    except Exception as exc:
        problems.append(f"P7b oracle-free deployable load failed: {exc}")
        oracle_free_load["error"] = str(exc)

    prediction_roots = {
        FUSION_MEMBER_A0: Path(config.a0_prediction_dir).resolve(),
        FUSION_MEMBER_P7B: Path(config.p7b_prediction_dir).resolve(),
    }
    reusable_predictions: dict[str, Any] = {}
    split_blocks: dict[str, list[Any]] = {split: [] for split in config.required_splits}
    member_identity_sets: dict[str, set[tuple[str, int, int]]] = {}
    for member, root in prediction_roots.items():
        member_rows: dict[str, Any] = {}
        manifest_path = root / member / "prediction_manifest.json"
        try:
            prediction_manifest = record(f"{member}_prediction_manifest", manifest_path)
            manifest_payload = _read_json(manifest_path)
            if manifest_payload.get("ok") is not True or manifest_payload.get("contract") != LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT:
                problems.append(f"{member} prediction manifest has an invalid contract or ok flag")
        except Exception as exc:
            prediction_manifest = None
            problems.append(str(exc))
        for split in config.required_splits:
            try:
                npz_path, metadata_path = prediction_paths(root, member, split)
                npz_artifact = record(f"{member}_{split}_predictions", npz_path)
                metadata_artifact = record(f"{member}_{split}_prediction_metadata", metadata_path)
                block = load_prediction_block(root, member, split, verify_hash=config.verify_prediction_hash)
                metadata = block.metadata
                if str(metadata.get("checkpoint_hash") or metadata.get("student_checkpoint_hash") or "") != str(checkpoint_hashes[member]):
                    raise ValueError(f"{member}/{split} prediction checkpoint hash does not match audited checkpoint")
                runtime_problems = _runtime_contract_problems(metadata, context=f"{member}/{split} predictions")
                if runtime_problems:
                    raise ValueError("; ".join(runtime_problems))
                dataset_metadata = metadata.get("dataset_metadata")
                alignment = dataset_metadata.get("alignment_report") if isinstance(dataset_metadata, Mapping) else {}
                if manifest_sha and str(alignment.get("source_manifest_hash") or "") != manifest_sha:
                    raise ValueError(f"{member}/{split} source manifest hash does not match")
                if str(alignment.get("hlt_content_hash") or "") != str(hlt_hashes.get(split) or ""):
                    raise ValueError(f"{member}/{split} HLT content hash does not match HLT-cache manifest")
                binary_metrics, binary_metrics_source = _attested_binary_projection_metrics(block)
                split_blocks[split].append(block)
                identities = {(item.file, int(item.entry), int(item.label)) for item in block.jet_ids}
                member_identity_sets[f"{member}:{split}"] = identities
                member_rows[split] = {
                    "prediction_path": str(npz_path.resolve()),
                    "prediction_sha256": npz_artifact["sha256"],
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": metadata_artifact["sha256"],
                    "prediction_content_hash": metadata.get("prediction_content_hash"),
                    "jet_identity_hash": metadata.get("jet_identity_hash"),
                    "hlt_content_hash": alignment.get("hlt_content_hash"),
                    "source_manifest_hash": alignment.get("source_manifest_hash"),
                    "binary_projection_metrics": binary_metrics,
                    "binary_projection_metrics_hash": stable_fusion_json_hash(binary_metrics),
                    "binary_projection_metrics_source": binary_metrics_source,
                    "n_jets": int(len(block.labels)),
                }
            except Exception as exc:
                problems.append(f"invalid reusable prediction {member}/{split}: {exc}")
        reusable_predictions[member] = {
            "prediction_root": str(root),
            "prediction_manifest": prediction_manifest,
            "splits": member_rows,
        }
    for split, blocks in split_blocks.items():
        if len(blocks) != 2:
            continue
        try:
            validate_prediction_alignment(blocks)
        except Exception as exc:
            problems.append(f"reusable prediction alignment failed for {split}: {exc}")
    for member in (FUSION_MEMBER_A0, FUSION_MEMBER_P7B):
        left = member_identity_sets.get(f"{member}:{FUSION_FIT_SPLIT}", set())
        right = member_identity_sets.get(f"{member}:{FUSION_SELECTION_SPLIT}", set())
        if left.intersection(right):
            problems.append(f"{member} stack_train and stack_val prediction identities overlap")

    report: dict[str, Any] = {
        "ok": not problems,
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT_CONTRACT,
        "config": asdict(config),
        "problems": problems,
        "artifacts": artifacts,
        "hashed_files": hashed_files,
        "checkpoint_hashes": checkpoint_hashes,
        "split_manifest_hash": manifest_sha,
        "hlt_content_hashes": hlt_hashes,
        "selected_consumer": None if selected is None else selected.to_dict(),
        "p7b_oracle_free_load": oracle_free_load,
        "reusable_predictions": reusable_predictions,
        "development_splits": list(FUSION_DEVELOPMENT_SPLITS),
        "final_test_opened": False,
    }
    report["audit_hash"] = stable_fusion_json_hash(report)
    _atomic_write_json(config.output_path, report, overwrite=config.overwrite)
    return report


def require_fusion_source_artifact_audit(path: str | Path) -> dict[str, Any]:
    """Re-hash every frozen input immediately before a dependent GPU command."""

    audit = _read_json(path)
    if audit.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT_CONTRACT:
        raise ValueError("source artifact audit contract mismatch")
    if audit.get("ok") is not True:
        raise ValueError(f"source artifact audit failed: {audit.get('problems')}")
    stored_hash = audit.get("audit_hash")
    unsigned = dict(audit)
    unsigned.pop("audit_hash", None)
    if stored_hash != stable_fusion_json_hash(unsigned):
        raise ValueError("source artifact audit hash mismatch")
    for row in audit.get("hashed_files", []):
        if not isinstance(row, Mapping):
            raise ValueError("source artifact audit contains a malformed hashed-file row")
        file_path = Path(str(row.get("path") or ""))
        if not file_path.is_file():
            raise FileNotFoundError(f"audited source artifact disappeared: {file_path}")
        actual = sha256_file(file_path)
        if actual != row.get("sha256"):
            raise ValueError(f"audited source artifact changed: {file_path}")
    return audit


def cache_a0_seed1_development_predictions(
    config: LocalResidualFieldPredictionConfig,
    *,
    source_artifact_audit: str | Path,
) -> dict[str, Any]:
    """Cache only development predictions, then align them to audited A0/P7b blocks."""

    audit = require_fusion_source_artifact_audit(source_artifact_audit)
    if config.model_name != FUSION_MEMBER_A0_SEED1:
        raise ValueError(f"campaign prediction command only accepts model_name={FUSION_MEMBER_A0_SEED1!r}")
    if tuple(config.splits) != FUSION_DEVELOPMENT_SPLITS:
        raise ValueError(f"development prediction splits are locked to {FUSION_DEVELOPMENT_SPLITS}")
    if config.confirm_final_test or config.allow_oracle_final_test:
        raise ValueError("development prediction caching cannot enable final-test or oracle-final-test flags")
    if config.target_cache_dir:
        raise ValueError("A0_seed1 development prediction caching must be HLT-only with no target cache")

    seed_completion_path = Path(config.checkpoint).resolve().parent / "seed_control_completion.json"
    seed_completion = _read_json(seed_completion_path)
    unsigned_completion = dict(seed_completion)
    stored_completion_hash = unsigned_completion.pop("artifact_hash", None)
    if (
        seed_completion.get("ok") is not True
        or seed_completion.get("contract") != "local_residual_field_a0_seed1_completion_v1"
        or stored_completion_hash != stable_fusion_json_hash(unsigned_completion)
        or seed_completion.get("checkpoint_sha256") != sha256_file(config.checkpoint)
    ):
        raise ValueError("A0_seed1 checkpoint is not bound to a valid seed-control completion artifact")
    report = cache_local_residual_field_tagger_predictions(config)
    source_rows: dict[str, Any] = {
        FUSION_MEMBER_A0: audit["reusable_predictions"][FUSION_MEMBER_A0],
        FUSION_MEMBER_P7B: audit["reusable_predictions"][FUSION_MEMBER_P7B],
    }
    seed_rows: dict[str, Any] = {}
    seed_checkpoint_hash = sha256_file(config.checkpoint)
    for split in FUSION_DEVELOPMENT_SPLITS:
        seed_block = load_prediction_block(config.prediction_dir, FUSION_MEMBER_A0_SEED1, split, verify_hash=True)
        if _runtime_contract_problems(seed_block.metadata, context=f"A0_seed1/{split} predictions"):
            raise ValueError(f"A0_seed1/{split} predictions are not deployable HLT-only predictions")
        if str(seed_block.metadata.get("checkpoint_hash") or seed_block.metadata.get("student_checkpoint_hash") or "") != seed_checkpoint_hash:
            raise ValueError(f"A0_seed1/{split} prediction checkpoint hash does not match A0_seed1 checkpoint")
        dataset_metadata = seed_block.metadata.get("dataset_metadata")
        alignment_metadata = dataset_metadata.get("alignment_report") if isinstance(dataset_metadata, Mapping) else {}
        if alignment_metadata.get("source_manifest_hash") != audit.get("split_manifest_hash"):
            raise ValueError(f"A0_seed1/{split} prediction manifest hash does not match the audited split manifest")
        if alignment_metadata.get("hlt_content_hash") != audit.get("hlt_content_hashes", {}).get(split):
            raise ValueError(f"A0_seed1/{split} HLT content hash does not match the audited HLT cache")
        binary_metrics, binary_metrics_source = _attested_binary_projection_metrics(seed_block)
        aligned = [seed_block]
        for member in (FUSION_MEMBER_A0, FUSION_MEMBER_P7B):
            root = audit["reusable_predictions"][member]["prediction_root"]
            aligned.append(load_prediction_block(root, member, split, verify_hash=True))
        validate_prediction_alignment(aligned)
        npz_path, metadata_path = prediction_paths(config.prediction_dir, FUSION_MEMBER_A0_SEED1, split)
        seed_rows[split] = {
            "prediction_path": str(npz_path.resolve()),
            "prediction_sha256": sha256_file(npz_path),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "prediction_content_hash": seed_block.metadata.get("prediction_content_hash"),
            "jet_identity_hash": seed_block.metadata.get("jet_identity_hash"),
            "binary_projection_metrics": binary_metrics,
            "binary_projection_metrics_hash": stable_fusion_json_hash(binary_metrics),
            "binary_projection_metrics_source": binary_metrics_source,
            "n_jets": int(len(seed_block.labels)),
        }
    source_rows[FUSION_MEMBER_A0_SEED1] = {
        "prediction_root": str(Path(config.prediction_dir).resolve()),
        "splits": seed_rows,
    }
    source_manifest = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT,
        "source_artifact_audit": str(Path(source_artifact_audit).resolve()),
        "source_artifact_audit_path": str(Path(source_artifact_audit).resolve()),
        "source_artifact_audit_hash": audit["audit_hash"],
        "seed_control_completion_path": str(seed_completion_path),
        "seed_control_completion_hash": stored_completion_hash,
        "checkpoint_path": str(Path(config.checkpoint).resolve()),
        "checkpoint_hash": seed_checkpoint_hash,
        "development_splits": list(FUSION_DEVELOPMENT_SPLITS),
        "final_test_opened": False,
        "members": source_rows,
    }
    source_manifest["manifest_hash"] = stable_fusion_json_hash(source_manifest)
    output = Path(config.prediction_dir) / "development_prediction_sources.json"
    _atomic_write_json(output, source_manifest, overwrite=config.overwrite)
    return {**report, "source_artifact_audit_hash": audit["audit_hash"], "prediction_sources": str(output)}


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT",
    "FUSION_DEVELOPMENT_SPLITS",
    "FusionSourceArtifactAuditConfig",
    "audit_fusion_source_artifacts",
    "require_fusion_source_artifact_audit",
    "cache_a0_seed1_development_predictions",
]
