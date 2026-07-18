"""Hash-bound storage waves, consumer receipts, and exact-path cleanup."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bundled_scoring import validate_logit_only_npz
from .cache import adaptive_binary_target_cache_paths
from .config import canonical_hash
from .storage_quota import (
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    campaign_storage_audit,
    reconcile_storage_accounting,
    resolve_storage_profile,
    storage_paths,
    write_campaign_storage_audit,
    write_quota_managed_json,
)


ABPH_ARTIFACT_MANIFEST_CONTRACT = "adaptive_binary_storage_artifact_manifest_v1"
ABPH_CONSUMER_RECEIPT_CONTRACT = "adaptive_binary_storage_consumer_receipt_v1"
ABPH_CLEANUP_RECEIPT_CONTRACT = "adaptive_binary_storage_cleanup_receipt_v1"
ABPH_CLEANUP_INTENT_CONTRACT = "adaptive_binary_storage_cleanup_intent_v1"
ABPH_WAVE_RECEIPT_CONTRACT = "adaptive_binary_storage_wave_receipt_v1"

ABPH_PRIVILEGED_CLEANUP_ROLES = frozenset(
    {"offline_cache_payload", "target_cache_shard"}
)
ABPH_DEPLOYABLE_CLEANUP_ROLES = frozenset({"hlt_cache_payload"})
ABPH_SCORING_SPLITS = ("model_val", "stack_train", "stack_val")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return payload


def _payload_without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != "content_hash"}


def _require_content_hash(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("content_hash") != canonical_hash(_payload_without_hash(payload)):
        raise ValueError(f"{label} content hash mismatch")


def _inside_root(path: str | Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes campaign root: {resolved}") from exc
    return resolved


def _relative_entry(
    root: Path,
    path: Path,
    *,
    role: str,
    cleanup_barrier: str | None,
    retained: bool,
) -> dict[str, Any]:
    resolved = _inside_root(path, root, label=role)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"manifest artifact is absent or unsafe: {resolved}")
    return {
        "relative_path": resolved.relative_to(root).as_posix(),
        "role": role,
        "cleanup_barrier": cleanup_barrier,
        "retained": bool(retained),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def artifact_manifest_path(campaign_root: str | Path) -> Path:
    return storage_paths(campaign_root)["root"] / "artifact_manifest.json"


def consumer_receipt_path(campaign_root: str | Path, consumer: str) -> Path:
    safe = str(consumer).replace("/", "_").replace(":", "__")
    return storage_paths(campaign_root)["root"] / "consumer_receipts" / f"{safe}.json"


def cleanup_receipt_path(campaign_root: str | Path, barrier: str) -> Path:
    names = {
        "privileged": "cleanup_privileged_inputs.json",
        "deployable": "cleanup_deployable_inputs.json",
    }
    try:
        filename = names[str(barrier)]
    except KeyError as exc:
        raise ValueError(f"unknown cleanup barrier {barrier!r}") from exc
    return storage_paths(campaign_root)["cleanup_receipts"] / filename


def cleanup_intent_path(campaign_root: str | Path, barrier: str) -> Path:
    return storage_paths(campaign_root)["root"] / "cleanup_intents" / f"{barrier}.json"


def build_artifact_manifest(
    campaign_root: str | Path,
    *,
    data_dir: str | Path,
    profile: str = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    """Freeze every shared payload that a cleanup barrier may remove."""

    root = Path(campaign_root).resolve()
    destination = artifact_manifest_path(root)
    if destination.exists():
        return require_artifact_manifest(root)
    artifacts: list[dict[str, Any]] = []
    source_contracts: dict[str, dict[str, Any]] = {
        "hlt": {},
        "offline": {},
        "targets": {},
    }
    hlt_dir = root / "inputs" / "hlt_cache"
    for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test"):
        metadata_path = hlt_dir / f"{split}_fixed_hlt_metadata.json"
        metadata = _read_json(metadata_path)
        payload_path = hlt_dir / f"{split}_fixed_hlt.npz"
        final_test = split == "final_test"
        artifacts.append(
            _relative_entry(
                root,
                payload_path,
                role=("hlt_final_test_payload" if final_test else "hlt_cache_payload"),
                cleanup_barrier=(None if final_test else "deployable"),
                retained=final_test,
            )
        )
        if metadata.get("split") not in (None, split):
            raise ValueError(f"HLT metadata split mismatch for {split}")
        source_contracts["hlt"][split] = {
            name: metadata.get(name)
            for name in (
                "source_manifest_hash",
                "hlt_content_hash",
                "jet_identity_hash",
                "label_hash",
            )
        }

    offline_dir = root / "inputs" / "offline_cache"
    for split in ("model_train", "model_val"):
        metadata = _read_json(offline_dir / f"{split}_offline_metadata.json")
        if metadata.get("split") not in (None, split):
            raise ValueError(f"offline metadata split mismatch for {split}")
        source_contracts["offline"][split] = {
            name: metadata.get(name)
            for name in (
                "source_manifest_hash",
                "offline_content_hash",
                "jet_identity_hash",
                "label_hash",
            )
        }
        artifacts.append(
            _relative_entry(
                root,
                offline_dir / f"{split}_offline.npz",
                role="offline_cache_payload",
                cleanup_barrier="privileged",
                retained=False,
            )
        )

    target_mode_path = root / "audits" / "target_mode_selection.json"
    target_mode = "shared_transient_compact"
    target_mode_hash: str | None = None
    if target_mode_path.is_file():
        target_mode_payload = _read_json(target_mode_path)
        target_mode = str(target_mode_payload.get("selected_mode", ""))
        target_mode_hash = str(target_mode_payload.get("content_hash", ""))
    if target_mode != "rank_local_build":
        target_dir = root / "targets"
        for split in ("model_train", "model_val"):
            for grouping in ("exclusive_kt", "cambridge_aachen"):
                shard_dir, metadata_path = adaptive_binary_target_cache_paths(
                    target_dir, split, grouping
                )
                metadata = _read_json(metadata_path)
                source_contracts["targets"][f"{split}/{grouping}"] = {
                    name: metadata.get(name)
                    for name in (
                        "source_manifest_hash",
                        "offline_content_hash",
                        "target_content_hash",
                        "jet_identity_hash",
                    )
                }
                for shard in metadata.get("shards", []):
                    path = shard_dir / str(shard["filename"])
                    entry = _relative_entry(
                        root,
                        path,
                        role="target_cache_shard",
                        cleanup_barrier="privileged",
                        retained=False,
                    )
                    if entry["sha256"] != shard.get("file_sha256"):
                        raise ValueError(f"target shard hash differs from metadata: {path}")
                    artifacts.append(entry)
                forensic = metadata.get("forensic_identity_sample")
                if isinstance(forensic, Mapping):
                    path = shard_dir / str(forensic["filename"])
                    entry = _relative_entry(
                        root,
                        path,
                        role="forensic_target_sample",
                        cleanup_barrier=None,
                        retained=True,
                    )
                    if entry["sha256"] != forensic.get("file_sha256"):
                        raise ValueError(f"forensic sample hash differs from metadata: {path}")
                    artifacts.append(entry)

    required_source_fields = {
        "hlt": ("source_manifest_hash", "hlt_content_hash", "jet_identity_hash"),
        "offline": (
            "source_manifest_hash",
            "offline_content_hash",
            "jet_identity_hash",
        ),
    }
    for source_kind, fields in required_source_fields.items():
        for split, contract in source_contracts[source_kind].items():
            missing = [name for name in fields if contract.get(name) in (None, "")]
            if missing:
                raise ValueError(
                    f"{source_kind} source contract {split} lacks {missing}"
                )
    manifest_hashes = {
        contract["source_manifest_hash"]
        for contracts in (source_contracts["hlt"], source_contracts["offline"])
        for contract in contracts.values()
    }
    if len(manifest_hashes) != 1:
        raise ValueError(
            "HLT/offline source contracts do not share one manifest hash: "
            f"{sorted(manifest_hashes)}"
        )
    for name, contract in source_contracts["targets"].items():
        missing = [
            field
            for field in (
                "source_manifest_hash",
                "offline_content_hash",
                "target_content_hash",
                "jet_identity_hash",
            )
            if contract.get(field) in (None, "")
        ]
        if missing:
            raise ValueError(f"target source contract {name} lacks {missing}")
        split = name.split("/", 1)[0]
        if contract["source_manifest_hash"] != next(iter(manifest_hashes)):
            raise ValueError(f"target source contract {name} uses another manifest")
        if contract["offline_content_hash"] != source_contracts["offline"][split][
            "offline_content_hash"
        ]:
            raise ValueError(f"target source contract {name} uses another offline cache")
        if contract["jet_identity_hash"] != source_contracts["offline"][split][
            "jet_identity_hash"
        ]:
            raise ValueError(f"target source contract {name} uses other jet identities")
    relative_paths = [row["relative_path"] for row in artifacts]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("artifact manifest contains duplicate paths")
    payload: dict[str, Any] = {
        "contract": ABPH_ARTIFACT_MANIFEST_CONTRACT,
        "campaign_root": str(root),
        "storage_profile": resolve_storage_profile(profile).name,
        "created_at": _utc_now(),
        "target_mode": target_mode,
        "target_mode_hash": target_mode_hash,
        "artifacts": sorted(artifacts, key=lambda row: row["relative_path"]),
        "source_contracts": source_contracts,
        "rebuild_instructions": {
            "data_dir": str(Path(data_dir).resolve()),
            "manifest": "inputs/split_manifest/split_manifest.json.gz",
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_degradation_strength": 2.5,
            "offline_splits": ["model_train", "model_val"],
            "target_splits": ["model_train", "model_val"],
        },
    }
    payload["content_hash"] = canonical_hash(payload)
    write_quota_managed_json(
        root,
        destination,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_artifact_manifest",
        source_provenance_hash=str(payload["content_hash"]),
        run_id="wave2_artifact_manifest",
        profile=profile,
    )
    return payload


def require_artifact_manifest(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    payload = _read_json(artifact_manifest_path(root))
    if payload.get("contract") != ABPH_ARTIFACT_MANIFEST_CONTRACT:
        raise ValueError("storage artifact manifest contract mismatch")
    _require_content_hash(payload, label="storage artifact manifest")
    if Path(str(payload.get("campaign_root", ""))).resolve() != root:
        raise ValueError("storage artifact manifest belongs to another campaign")
    rows = payload.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("storage artifact manifest lacks artifacts")
    for row in rows:
        _inside_root(root / str(row["relative_path"]), root, label="manifest artifact")
    return payload


def write_consumer_receipt(
    campaign_root: str | Path,
    *,
    consumer: str,
    run_report: str | Path,
    consumer_kind: str = "target_consumer",
    profile: str = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    manifest = require_artifact_manifest(root)
    report_path = _inside_root(run_report, root, label="consumer run report")
    report = _read_json(report_path)
    if report.get("ok") is not True:
        raise ValueError(f"consumer {consumer} run report has ok!=true")
    reported_name = report.get("variant_name")
    if reported_name is None and isinstance(report.get("variant"), Mapping):
        reported_name = report["variant"].get("name")
    allowed_names = {str(consumer), str(consumer).split("__seed", 1)[0]}
    if reported_name not in allowed_names:
        raise ValueError(
            f"consumer receipt identity mismatch: {reported_name!r} not in "
            f"{sorted(allowed_names)}"
        )
    if consumer_kind == "target_consumer":
        provenance_rows = report.get("provenance")
        provenance = (
            provenance_rows.get("model_val")
            if isinstance(provenance_rows, Mapping)
            else None
        )
        required_target_fields = (
            "source_manifest_hash",
            "hlt_content_hash",
            "offline_cache_content_hash",
            "hierarchy_target_content_hash",
        )
        missing_target_fields = [
            name
            for name in required_target_fields
            if not isinstance(provenance, Mapping) or provenance.get(name) in (None, "")
        ]
        if missing_target_fields:
            raise ValueError(
                f"consumer {consumer} lacks target provenance {missing_target_fields}"
            )
        contracts = manifest.get("source_contracts")
        if not isinstance(contracts, Mapping):
            raise ValueError("artifact manifest lacks source contracts")
        offline_contract = dict(contracts.get("offline", {})).get("model_val", {})
        hlt_contract = dict(contracts.get("hlt", {})).get("model_val", {})
        expected_manifest_hash = hlt_contract.get("source_manifest_hash")
        if expected_manifest_hash in (None, "") or offline_contract.get(
            "source_manifest_hash"
        ) != expected_manifest_hash:
            raise ValueError("artifact manifest has conflicting model_val sources")
        if provenance.get("source_manifest_hash") != expected_manifest_hash:
            raise ValueError(f"consumer {consumer} source manifest hash mismatch")
        expected_hlt_hash = hlt_contract.get("hlt_content_hash")
        if expected_hlt_hash in (None, "") or provenance.get(
            "hlt_content_hash"
        ) != expected_hlt_hash:
            raise ValueError(f"consumer {consumer} HLT cache hash mismatch")
        expected_offline_hash = offline_contract.get("offline_content_hash")
        if expected_offline_hash not in (None, "") and provenance.get(
            "offline_cache_content_hash"
        ) != expected_offline_hash:
            raise ValueError(f"consumer {consumer} offline cache hash mismatch")
        target_contracts = dict(contracts.get("targets", {}))
        model_val_targets = {
            key.split("/", 1)[1]: value.get("target_content_hash")
            for key, value in target_contracts.items()
            if key.startswith("model_val/")
            and isinstance(value, Mapping)
            and value.get("target_content_hash") not in (None, "")
        }
        allowed_target_hashes = set(model_val_targets.values())
        if len(model_val_targets) > 1:
            allowed_target_hashes.add(canonical_hash(model_val_targets))
        if allowed_target_hashes and provenance.get(
            "hierarchy_target_content_hash"
        ) not in allowed_target_hashes:
            raise ValueError(f"consumer {consumer} hierarchy target hash mismatch")
    required_roles = (
        ABPH_PRIVILEGED_CLEANUP_ROLES
        if consumer_kind == "target_consumer"
        else frozenset({"offline_cache_payload"})
    )
    verified_inputs: list[dict[str, Any]] = []
    for row in manifest["artifacts"]:
        if row["role"] not in required_roles:
            continue
        path = _inside_root(root / row["relative_path"], root, label="consumer input")
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ValueError(f"consumer input is missing or changed: {path}")
        verified_inputs.append(
            {"relative_path": row["relative_path"], "sha256": row["sha256"]}
        )
    if not verified_inputs:
        raise ValueError(f"consumer {consumer} verified no privileged inputs")
    payload: dict[str, Any] = {
        "contract": ABPH_CONSUMER_RECEIPT_CONTRACT,
        "campaign_root": str(root),
        "consumer": str(consumer),
        "consumer_kind": str(consumer_kind),
        "completed_at": _utc_now(),
        "run_report": report_path.relative_to(root).as_posix(),
        "run_report_sha256": _sha256_file(report_path),
        "artifact_manifest_hash": manifest["content_hash"],
        "verified_inputs": verified_inputs,
        "success": True,
    }
    payload["content_hash"] = canonical_hash(payload)
    destination = consumer_receipt_path(root, consumer)
    if destination.exists():
        existing = require_consumer_receipt(root, consumer)
        if existing["run_report_sha256"] != payload["run_report_sha256"]:
            raise ValueError(f"consumer receipt conflicts with current run: {consumer}")
        return existing
    write_quota_managed_json(
        root,
        destination,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_consumer_receipt",
        source_provenance_hash=str(manifest["content_hash"]),
        run_id=str(consumer),
        profile=profile,
    )
    return payload


def require_consumer_receipt(
    campaign_root: str | Path, consumer: str
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    payload = _read_json(consumer_receipt_path(root, consumer))
    if payload.get("contract") != ABPH_CONSUMER_RECEIPT_CONTRACT:
        raise ValueError(f"consumer receipt contract mismatch: {consumer}")
    _require_content_hash(payload, label=f"consumer receipt {consumer}")
    if payload.get("consumer") != consumer or payload.get("success") is not True:
        raise ValueError(f"consumer receipt identity/status mismatch: {consumer}")
    manifest = require_artifact_manifest(root)
    if payload.get("artifact_manifest_hash") != manifest["content_hash"]:
        raise ValueError(f"consumer receipt uses another artifact manifest: {consumer}")
    report = _inside_root(root / payload["run_report"], root, label="consumer report")
    if not report.is_file() or _sha256_file(report) != payload.get("run_report_sha256"):
        raise ValueError(f"consumer report changed after receipt: {consumer}")
    return payload


def verify_bundled_logits(
    campaign_root: str | Path, members: Sequence[str]
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    verified: list[dict[str, Any]] = []
    for member in members:
        for split in ABPH_SCORING_SPLITS:
            array_path = root / "logit_predictions" / member / f"{split}.npz"
            metadata_path = root / "logit_predictions" / member / f"{split}_metadata.json"
            artifact = validate_logit_only_npz(array_path)
            metadata = _read_json(metadata_path)
            if metadata.get("ok") is not True:
                raise ValueError(f"logit metadata has ok!=true: {member}/{split}")
            if metadata.get("prediction_sha256") != artifact["sha256"]:
                raise ValueError(f"logit artifact hash mismatch: {member}/{split}")
            provenance = metadata.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"logit artifact lacks provenance: {member}/{split}")
            if provenance.get("persisted_identity_hash") != artifact.get(
                "persisted_identity_hash"
            ):
                raise ValueError(f"logit identity hash mismatch: {member}/{split}")
            verified.append(
                {
                    "member": member,
                    "split": split,
                    "sha256": artifact["sha256"],
                    "source_generation_hash": provenance.get(
                        "source_generation_hash"
                    ),
                }
            )
    return {"verified": verified, "verification_hash": canonical_hash({"rows": verified})}


def execute_cleanup_barrier(
    campaign_root: str | Path,
    *,
    barrier: str,
    expected_consumers: Sequence[str] = (),
    scoring_members: Sequence[str] = (),
    profile: str = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    destination = cleanup_receipt_path(root, barrier)
    if destination.exists():
        return require_cleanup_receipt(root, barrier)
    manifest = require_artifact_manifest(root)
    consumers = [
        require_consumer_receipt(root, consumer) for consumer in expected_consumers
    ]
    scoring_verification: Mapping[str, Any] | None = None
    roles = (
        ABPH_PRIVILEGED_CLEANUP_ROLES
        if barrier == "privileged"
        else ABPH_DEPLOYABLE_CLEANUP_ROLES
    )
    if barrier == "deployable":
        if not scoring_members:
            raise ValueError("deployable cleanup requires frozen scoring membership")
        scoring_verification = verify_bundled_logits(root, scoring_members)
    elif barrier != "privileged":
        raise ValueError(f"unknown cleanup barrier {barrier!r}")
    elif not expected_consumers:
        raise ValueError("privileged cleanup requires frozen consumer membership")
    planned = [
        {
            "relative_path": row["relative_path"],
            "role": row["role"],
            "sha256": row["sha256"],
            "bytes": int(row["bytes"]),
        }
        for row in manifest["artifacts"]
        if row["role"] in roles
        and row.get("cleanup_barrier") == barrier
        and row.get("retained") is not True
    ]
    if not planned:
        raise ValueError(f"cleanup barrier {barrier} selected no artifacts")
    intent_payload: dict[str, Any] = {
        "contract": ABPH_CLEANUP_INTENT_CONTRACT,
        "campaign_root": str(root),
        "barrier": barrier,
        "artifact_manifest_hash": manifest["content_hash"],
        "expected_consumers": list(expected_consumers),
        "scoring_verification_hash": (
            scoring_verification.get("verification_hash")
            if isinstance(scoring_verification, Mapping)
            else None
        ),
        "planned": planned,
    }
    intent_payload["content_hash"] = canonical_hash(intent_payload)
    intent_path = cleanup_intent_path(root, barrier)
    if intent_path.exists():
        existing_intent = _read_json(intent_path)
        if existing_intent.get("contract") != ABPH_CLEANUP_INTENT_CONTRACT:
            raise ValueError(f"cleanup intent contract mismatch: {barrier}")
        _require_content_hash(existing_intent, label=f"cleanup intent {barrier}")
        if existing_intent != intent_payload:
            raise ValueError(f"cleanup intent changed between attempts: {barrier}")
    else:
        write_quota_managed_json(
            root,
            intent_path,
            intent_payload,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role="storage_cleanup_intent",
            source_provenance_hash=str(manifest["content_hash"]),
            run_id=f"cleanup_intent_{barrier}",
            profile=profile,
        )
    before = campaign_storage_audit(root, profile=profile)
    deleted: list[dict[str, Any]] = []
    for row in planned:
        path = _inside_root(root / row["relative_path"], root, label="cleanup artifact")
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"cleanup artifact is unsafe: {path}")
            actual_hash = _sha256_file(path)
            if actual_hash != row["sha256"]:
                raise ValueError(f"cleanup artifact hash changed: {path}")
            path.unlink()
        deleted.append(
            {
                "relative_path": row["relative_path"],
                "role": row["role"],
                "sha256": row["sha256"],
                "bytes": int(row["bytes"]),
            }
        )
    reconciliation = reconcile_storage_accounting(
        root,
        allowed_missing_paths=(root / row["relative_path"] for row in deleted),
        profile=profile,
    )
    after = campaign_storage_audit(root, profile=profile)
    payload: dict[str, Any] = {
        "contract": ABPH_CLEANUP_RECEIPT_CONTRACT,
        "campaign_root": str(root),
        "barrier": barrier,
        "completed_at": _utc_now(),
        "artifact_manifest_hash": manifest["content_hash"],
        "cleanup_intent_hash": intent_payload["content_hash"],
        "expected_consumers": list(expected_consumers),
        "consumer_receipt_hashes": {
            row["consumer"]: row["content_hash"] for row in consumers
        },
        "scoring_verification": scoring_verification,
        "deleted": deleted,
        "deleted_bytes": sum(int(row["bytes"]) for row in deleted),
        "before_measured_bytes": int(before["measured_persistent_bytes"]),
        "after_measured_bytes": int(after["measured_persistent_bytes"]),
        "ledger_reconciliation": reconciliation,
        "ok": True,
    }
    payload["content_hash"] = canonical_hash(payload)
    write_quota_managed_json(
        root,
        destination,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_cleanup_receipt",
        source_provenance_hash=str(manifest["content_hash"]),
        run_id=f"cleanup_{barrier}",
        profile=profile,
    )
    return payload


def require_cleanup_receipt(
    campaign_root: str | Path, barrier: str
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    payload = _read_json(cleanup_receipt_path(root, barrier))
    if payload.get("contract") != ABPH_CLEANUP_RECEIPT_CONTRACT:
        raise ValueError(f"cleanup receipt contract mismatch: {barrier}")
    _require_content_hash(payload, label=f"cleanup receipt {barrier}")
    if payload.get("barrier") != barrier or payload.get("ok") is not True:
        raise ValueError(f"cleanup receipt identity/status mismatch: {barrier}")
    manifest = require_artifact_manifest(root)
    if payload.get("artifact_manifest_hash") != manifest["content_hash"]:
        raise ValueError(f"cleanup receipt uses another manifest: {barrier}")
    intent = _read_json(cleanup_intent_path(root, barrier))
    if intent.get("contract") != ABPH_CLEANUP_INTENT_CONTRACT:
        raise ValueError(f"cleanup intent contract mismatch: {barrier}")
    _require_content_hash(intent, label=f"cleanup intent {barrier}")
    if payload.get("cleanup_intent_hash") != intent.get("content_hash"):
        raise ValueError(f"cleanup receipt uses another intent: {barrier}")
    for row in payload.get("deleted", []):
        path = _inside_root(root / row["relative_path"], root, label="deleted artifact")
        if path.exists():
            raise ValueError(f"cleanup receipt claims a path that still exists: {path}")
    return payload


def write_wave_receipt(
    campaign_root: str | Path,
    *,
    wave: int,
    profile: str = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    audit_path, audit = write_campaign_storage_audit(
        root, profile=profile, label=f"wave_{int(wave)}"
    )
    payload: dict[str, Any] = {
        "contract": ABPH_WAVE_RECEIPT_CONTRACT,
        "campaign_root": str(root),
        "wave": int(wave),
        "created_at": _utc_now(),
        "storage_audit": audit_path.relative_to(root).as_posix(),
        "storage_audit_hash": audit["content_hash"],
        "measured_persistent_bytes": audit["measured_persistent_bytes"],
        "ok": audit["ok"],
    }
    payload["content_hash"] = canonical_hash(payload)
    destination = storage_paths(root)["root"] / "wave_receipts" / f"wave_{wave}.json"
    write_quota_managed_json(
        root,
        destination,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_wave_receipt",
        source_provenance_hash=str(audit["content_hash"]),
        run_id=f"wave_{wave}",
        profile=profile,
    )
    return payload


__all__ = [
    "ABPH_ARTIFACT_MANIFEST_CONTRACT",
    "ABPH_CLEANUP_RECEIPT_CONTRACT",
    "ABPH_CLEANUP_INTENT_CONTRACT",
    "ABPH_CONSUMER_RECEIPT_CONTRACT",
    "ABPH_WAVE_RECEIPT_CONTRACT",
    "artifact_manifest_path",
    "build_artifact_manifest",
    "cleanup_receipt_path",
    "cleanup_intent_path",
    "consumer_receipt_path",
    "execute_cleanup_barrier",
    "require_artifact_manifest",
    "require_cleanup_receipt",
    "require_consumer_receipt",
    "verify_bundled_logits",
    "write_consumer_receipt",
    "write_wave_receipt",
]
