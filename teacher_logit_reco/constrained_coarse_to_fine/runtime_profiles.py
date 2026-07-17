"""Immutable C2F runtime-profile closure validation and per-run resolution."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

from .runtime import collect_code_environment


RUNTIME_APPROVED_CONTRACT = "constrained_c2f_accelerated_approved_v1"
_PROFILE_HASH_FIELD = {
    "accelerated_candidate_v1": "candidate_profile_hash",
    "accelerated_approved_v1": "approved_profile_hash",
}
_SINGLE_VIEW_RUNS = {
    "C0", "C1", "C2", "C3", "C5", "C5-B1", "C5-B2", "C5-B3", "C5-no-slot", "C5-unconstrained", "Cdirect-unconstrained",
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"runtime profile {path} must be a JSON object")
    return payload


def _manifest_hash(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else path.open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as handle:
        payload = json.load(handle)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_profile_hash(profile: Mapping[str, Any], *, status: str) -> None:
    field = _PROFILE_HASH_FIELD.get(status)
    if field is None:
        raise ValueError(f"unsupported runtime profile status {status!r}")
    declared = profile.get(field)
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"runtime profile lacks {field}")
    observed = _canonical_hash({key: value for key, value in profile.items() if key != field})
    if observed != declared:
        raise ValueError(f"runtime profile {field} does not match its content")


def _require_execution_map(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = profile.get("runtime_profiles_by_variant")
    if not isinstance(rows, Mapping):
        raise ValueError("runtime profile lacks runtime_profiles_by_variant")
    selected = profile.get("execution_by_variant")
    if not isinstance(selected, Mapping):
        raise ValueError("runtime profile lacks execution_by_variant")
    for variant in ("C1", "C5-B3", "C6", "C4"):
        row = rows.get(variant)
        if not isinstance(row, Mapping) or not isinstance(row.get("runtime_profile_hash"), str):
            raise ValueError(f"runtime profile lacks a resolved {variant} execution")
        selected_row = selected.get(variant)
        if not isinstance(selected_row, Mapping):
            raise ValueError(f"runtime profile lacks selected {variant} execution metadata")
        if selected_row.get("runtime_profile_hash") != row["runtime_profile_hash"]:
            raise ValueError(f"runtime profile selected {variant} hash mismatch")
    return rows


def validate_runtime_profile(
    path: str | Path,
    *,
    expected_status: str,
    require_current_environment: bool = True,
) -> dict[str, Any]:
    """Fail closed on a malformed, stale, or code-environment-mismatched profile."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    profile = _read_json(profile_path)
    if profile.get("status") != expected_status:
        raise ValueError(f"expected runtime profile {expected_status}, got {profile.get('status')!r}")
    _verify_profile_hash(profile, status=expected_status)
    _require_execution_map(profile)
    environment = profile.get("code_environment")
    environment_hash = profile.get("code_environment_hash")
    if not isinstance(environment, Mapping) or not isinstance(environment_hash, str):
        raise ValueError("runtime profile lacks code environment fingerprint")
    if environment.get("code_environment_hash") != environment_hash:
        raise ValueError("runtime profile code environment hash is internally inconsistent")
    if not bool(environment.get("source_tree_clean")):
        raise ValueError("runtime profile was not created from a clean source tree")
    closure: Mapping[str, Any] | None = None
    if expected_status == "accelerated_approved_v1":
        if profile.get("contract") != RUNTIME_APPROVED_CONTRACT:
            raise ValueError("approved runtime profile has the wrong contract")
        closure_raw = profile.get("closure")
        if not isinstance(closure_raw, Mapping):
            raise ValueError("approved runtime profile lacks an immutable closure")
        closure = closure_raw
        candidate = closure.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("approved runtime profile lacks candidate closure metadata")
        candidate_path = Path(str(candidate.get("path", "")))
        if _file_hash(candidate_path) != candidate.get("sha256"):
            raise ValueError("approved runtime profile candidate file hash mismatch")
        candidate_profile = _read_json(candidate_path)
        _verify_profile_hash(candidate_profile, status="accelerated_candidate_v1")
        if candidate_profile.get("candidate_profile_hash") != candidate.get("candidate_profile_hash"):
            raise ValueError("approved runtime profile candidate hash mismatch")
        if candidate_profile.get("code_environment_hash") != environment_hash:
            raise ValueError("approved runtime profile candidate environment mismatch")
        artifacts = closure.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("approved runtime profile lacks closure artifacts")
        expected_artifacts = {
            "c5_ten_epoch_certification", "c6_ten_epoch_certification",
            "c5_fp32_reference", "c6_fp32_reference",
            "c5_tagger_sanity", "c6_tagger_sanity",
        }
        if set(artifacts) != expected_artifacts:
            raise ValueError("approved runtime profile closure artifact membership mismatch")
        for name in sorted(expected_artifacts):
            row = artifacts[name]
            if not isinstance(row, Mapping):
                raise ValueError(f"approved runtime profile artifact {name} is malformed")
            artifact_path = Path(str(row.get("path", "")))
            if _file_hash(artifact_path) != row.get("sha256"):
                raise ValueError(f"approved runtime profile artifact {name} hash mismatch")
            payload = _read_json(artifact_path)
            _validate_approval_evidence(
                name=name,
                payload=payload,
                evidence_path=artifact_path,
                candidate_hash=str(candidate_profile["candidate_profile_hash"]),
                environment_hash=environment_hash,
                candidate_profile=candidate_profile,
                candidate_profile_path=candidate_path,
                candidate_profile_file_sha256=str(candidate.get("sha256")),
            )
        pilot_inputs = closure.get("pilot_inputs")
        if not isinstance(pilot_inputs, Mapping):
            raise ValueError("approved runtime profile lacks pilot input closure")
        observed_inputs = _input_provenance(
            manifest_path=Path(str(pilot_inputs.get("manifest_path", ""))),
            hlt_cache_dir=Path(str(pilot_inputs.get("hlt_cache_dir", ""))),
            offline_cache_dir=Path(str(pilot_inputs.get("offline_cache_dir", ""))),
            target_cache_dir=Path(str(pilot_inputs.get("target_cache_dir", ""))),
        )
        if observed_inputs != dict(pilot_inputs):
            raise ValueError("approved runtime profile pilot input closure mismatch")
    if require_current_environment:
        current = collect_code_environment()
        if current.get("code_environment_hash") != environment_hash:
            raise ValueError("active code/environment fingerprint differs from runtime profile")
        if not bool(current.get("source_tree_clean")):
            raise ValueError("runtime-profile submission requires a clean source tree")
    return {
        "path": str(profile_path.resolve()),
        "file_sha256": _file_hash(profile_path),
        "status": expected_status,
        "profile": dict(profile),
        "closure": None if closure is None else dict(closure),
    }


def execution_variant_for_run(run_id: str) -> str:
    """Map the full campaign's B/C variants onto the calibrated execution families."""

    normalized = str(run_id).strip()
    if normalized == "C4":
        return "C4"
    if normalized == "C6":
        return "C6"
    if normalized in _SINGLE_VIEW_RUNS or normalized.startswith("B"):
        return "C5-B3" if normalized.startswith("C5") else "C1"
    return "C1"


def resolve_execution(profile: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """Return the hash-bound runtime settings a campaign runner must use for one run."""

    variant = execution_variant_for_run(run_id)
    rows = _require_execution_map(profile)
    row = rows[variant]
    if not isinstance(row, Mapping):
        raise AssertionError("validated runtime profile execution row is not a mapping")
    return {"execution_variant": variant, **dict(row)}


def execution_shell_exports(profile: Mapping[str, Any], run_id: str) -> str:
    """Render only validated numeric/string settings as shell-safe exports for Slurm."""

    execution = resolve_execution(profile, run_id)
    precision = execution["precision"]
    batch = execution["batch"]
    pipeline = execution["input_pipeline"]
    optimizer = execution["optimizer"]
    scheduler = execution["scheduler"]
    hungarian = execution["hungarian"]
    values = {
        # An approved artifact reuses the candidate's physical settings but
        # must still label produced outputs with its approved contract.
        "CONSTRAINED_C2F_RUNTIME_PROFILE": str(profile["status"]),
        "CONSTRAINED_C2F_RECO_PRECISION_MODE": precision["precision_mode"],
        "CONSTRAINED_C2F_RECO_BATCH_SIZE": batch["train"],
        "CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE": batch["eval"],
        "CONSTRAINED_C2F_RECO_NUM_WORKERS": pipeline["num_workers"],
        "CONSTRAINED_C2F_RECO_PREFETCH_FACTOR": "" if pipeline["prefetch_factor"] is None else pipeline["prefetch_factor"],
        "CONSTRAINED_C2F_RECO_LEARNING_RATE": optimizer["learning_rate"],
        "CONSTRAINED_C2F_RECO_HLT_ENCODER_LR_SCALE": optimizer["hlt_encoder_lr_scale"],
        "CONSTRAINED_C2F_RECO_WEIGHT_DECAY": optimizer["weight_decay"],
        "CONSTRAINED_C2F_RECO_GRAD_CLIP_NORM": optimizer["grad_clip_norm"],
        "CONSTRAINED_C2F_RECO_LR_SCHEDULE": scheduler["name"],
        "CONSTRAINED_C2F_RECO_WARMUP_FRACTION": scheduler["warmup_fraction"],
        "CONSTRAINED_C2F_RECO_MIN_LR_RATIO": scheduler["min_lr_ratio"],
        "CONSTRAINED_C2F_RECO_MIN_EPOCHS": scheduler["min_epochs"],
        "CONSTRAINED_C2F_RECO_EPOCHS": scheduler["max_epochs"],
        "CONSTRAINED_C2F_RECO_FIXED_HORIZON": "1" if scheduler["fixed_horizon"] else "0",
        "CONSTRAINED_C2F_RECO_EARLY_STOP_PATIENCE": scheduler["early_stop_patience"],
        "CONSTRAINED_C2F_HUNGARIAN_WORKERS": hungarian["workers"],
        "CONSTRAINED_C2F_HUNGARIAN_EXECUTOR": hungarian["executor"],
    }
    return "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in values.items())


def _validate_approval_evidence(
    *,
    name: str,
    payload: Mapping[str, Any],
    evidence_path: Path,
    candidate_hash: str,
    environment_hash: str,
    candidate_profile: Mapping[str, Any],
    candidate_profile_path: Path,
    candidate_profile_file_sha256: str,
) -> None:
    """Require the exact certification or paired-tagger gate in the closure."""

    expected = {
        "c5_ten_epoch_certification": ("C5-B3", "ten_epoch_certification"),
        "c6_ten_epoch_certification": ("C6", "ten_epoch_certification"),
        "c5_fp32_reference": ("C5-B3", "fp32_reference_promotion"),
        "c6_fp32_reference": ("C6", "fp32_reference_promotion"),
        "c5_tagger_sanity": ("C5-B3", "tagger_sanity"),
        "c6_tagger_sanity": ("C6", "tagger_sanity"),
    }
    expected_path, expected_kind = expected[name]
    if not bool(payload.get("ok")):
        raise ValueError(f"approval evidence {name} has ok=false")
    if payload.get("path") != expected_path:
        raise ValueError(f"approval evidence {name} path mismatch")
    if payload.get("candidate_profile_hash") != candidate_hash:
        raise ValueError(f"approval evidence {name} candidate profile mismatch")
    if payload.get("code_environment_hash") != environment_hash:
        raise ValueError(f"approval evidence {name} code environment mismatch")
    if payload.get("promotion_evidence_kind") != expected_kind:
        raise ValueError(f"approval evidence {name} has wrong promotion evidence kind")
    gate = payload.get("promotion_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("ok")):
        raise ValueError(f"approval evidence {name} lacks a passing promotion gate")
    expected_contract = (
        "constrained_c2f_runtime_tagger_sanity_v1"
        if name.endswith("tagger_sanity")
        else "constrained_c2f_runtime_reconstructor_certification_v1"
    )
    if payload.get("contract") != expected_contract:
        raise ValueError(f"approval evidence {name} has wrong contract")
    if name.endswith("tagger_sanity"):
        from .runtime_tagger_sanity import (
            _reconstructor_checkpoint_metadata,
            validate_tagger_sanity_report,
        )

        sanity = validate_tagger_sanity_report(evidence_path)
        if sanity["report"] != dict(payload):
            raise ValueError(f"approval evidence {name} differs from its validated report")

        declared_candidate_profile = Path(str(payload.get("candidate_profile_path", "")))
        if declared_candidate_profile.resolve() != candidate_profile_path.resolve():
            raise ValueError(f"approval evidence {name} candidate profile path mismatch")
        if payload.get("candidate_profile_file_sha256") != candidate_profile_file_sha256:
            raise ValueError(f"approval evidence {name} candidate profile file hash mismatch")
        expected_execution_hash = str(resolve_execution(candidate_profile, expected_path)["runtime_profile_hash"])
        if payload.get("candidate_execution_runtime_profile_hash") != expected_execution_hash:
            raise ValueError(f"approval evidence {name} candidate execution hash mismatch")

        sources = payload.get("source_artifacts")
        if not isinstance(sources, Mapping):
            raise ValueError(f"approval evidence {name} lacks tagger source artifacts")
        provenance = payload.get("prediction_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"approval evidence {name} lacks prediction provenance")
        candidate_checkpoint = Path(str(sources.get("candidate_reconstructor_checkpoint", "")))
        reference_checkpoint = Path(str(sources.get("fp32_reconstructor_checkpoint", "")))
        if _file_hash(candidate_checkpoint) != provenance.get("accelerated_reconstructor_checkpoint_sha256"):
            raise ValueError(f"approval evidence {name} accelerated checkpoint hash mismatch")
        if _file_hash(reference_checkpoint) != provenance.get("fp32_reconstructor_checkpoint_sha256"):
            raise ValueError(f"approval evidence {name} FP32 checkpoint hash mismatch")
        candidate_checkpoint_metadata = _reconstructor_checkpoint_metadata(
            candidate_checkpoint,
            expected_variant=expected_path,
            expected_runtime_profile="accelerated_candidate_v1",
            expected_runtime_profile_hash=expected_execution_hash,
            require_thirty_epoch_reference=False,
        )
        reference_checkpoint_metadata = _reconstructor_checkpoint_metadata(
            reference_checkpoint,
            expected_variant=expected_path,
            expected_runtime_profile="fp32_reference",
            expected_runtime_profile_hash=None,
            require_thirty_epoch_reference=True,
        )
        if not candidate_checkpoint_metadata["ok"] or not reference_checkpoint_metadata["ok"]:
            raise ValueError(f"approval evidence {name} reconstructor checkpoint contract mismatch")
        if candidate_checkpoint_metadata["code_environment"] != candidate_profile.get("code_environment"):
            raise ValueError(f"approval evidence {name} candidate checkpoint environment mismatch")
    if not name.endswith("tagger_sanity"):
        expected_reference_epochs = 10 if name.endswith("ten_epoch_certification") else 30
        if payload.get("candidate_epochs") != 10 or payload.get("fp32_reference_epochs") != expected_reference_epochs:
            raise ValueError(f"approval evidence {name} has an invalid certification horizon")
        for label in ("candidate", "fp32_reference"):
            files = payload.get(label)
            if not isinstance(files, Mapping):
                raise ValueError(f"approval evidence {name} lacks {label} files")
            for path_field, hash_field in (("run_report_path", "run_report_sha256"), ("checkpoint_path", "checkpoint_sha256")):
                artifact_path = Path(str(files.get(path_field, "")))
                if not artifact_path.is_file() or _file_hash(artifact_path) != files.get(hash_field):
                    raise ValueError(f"approval evidence {name} {label} {path_field} hash mismatch")
            run_report = _read_json(Path(str(files["run_report_path"])))
            expected_profile = "accelerated_candidate_v1" if label == "candidate" else "fp32_reference"
            expected_epochs = 10 if label == "candidate" else expected_reference_epochs
            runtime = run_report.get("runtime_profile")
            config = run_report.get("training_config")
            if not bool(run_report.get("ok")) or run_report.get("variant") != expected_path:
                raise ValueError(f"approval evidence {name} {label} run report identity mismatch")
            if not isinstance(runtime, Mapping) or runtime.get("name") != expected_profile:
                raise ValueError(f"approval evidence {name} {label} run report runtime mismatch")
            if not isinstance(config, Mapping) or config.get("fixed_horizon") is not True:
                raise ValueError(f"approval evidence {name} {label} is not fixed horizon")
            if int(run_report.get("completed_epochs", -1)) != expected_epochs or int(config.get("epochs", -1)) != expected_epochs:
                raise ValueError(f"approval evidence {name} {label} epoch horizon mismatch")
            if run_report.get("checkpoint_sha256") != files["checkpoint_sha256"]:
                raise ValueError(f"approval evidence {name} {label} checkpoint declaration mismatch")
        # Recompute the certification from the immutable source reports rather
        # than treating this evidence JSON as the authority for its own gate.
        from .runtime_certification import validate_reconstructor_certification

        recomputed = validate_reconstructor_certification(
            path=expected_path,
            mode=expected_kind,
            candidate_profile_path=str(payload.get("candidate_profile_path", "")),
            candidate_run_dir=str(Path(str(payload["candidate"]["run_report_path"])).parent),
            fp32_reference_run_dir=str(Path(str(payload["fp32_reference"]["run_report_path"])).parent),
        )
        if recomputed["promotion_gate"] != payload.get("promotion_gate"):
            raise ValueError(f"approval evidence {name} promotion gate does not match recomputation")


def _input_provenance(
    *,
    manifest_path: Path,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    target_cache_dir: Path,
) -> dict[str, Any]:
    manifest_sha = _manifest_hash(manifest_path)
    rows: dict[str, Any] = {}
    for split in ("model_train", "model_val"):
        hlt = _read_json(hlt_cache_dir / f"{split}_fixed_hlt_metadata.json")
        offline = _read_json(offline_cache_dir / f"{split}_offline_metadata.json")
        target = _read_json(target_cache_dir / f"{split}_hierarchy_targets_metadata.json")
        values = {
            "source_manifest_hash": manifest_sha,
            "hlt_content_hash": hlt.get("hlt_content_hash"),
            "offline_content_hash": offline.get("offline_content_hash") or offline.get("content_hash"),
            "target_content_hash": target.get("target_content_hash"),
            "jet_identity_hash": target.get("jet_identity_hash"),
        }
        if any(value in (None, "") for value in values.values()):
            raise ValueError(f"pilot input provenance is incomplete for {split}")
        if hlt.get("source_manifest_hash") != manifest_sha or offline.get("source_manifest_hash") != manifest_sha:
            raise ValueError(f"pilot input manifest mismatch for {split}")
        if target.get("source_manifest_hash") != manifest_sha:
            raise ValueError(f"pilot target manifest mismatch for {split}")
        if target.get("hlt_content_hash") != values["hlt_content_hash"]:
            raise ValueError(f"pilot target HLT provenance mismatch for {split}")
        if target.get("offline_content_hash") != values["offline_content_hash"]:
            raise ValueError(f"pilot target offline provenance mismatch for {split}")
        if hlt.get("jet_identity_hash") != values["jet_identity_hash"] or offline.get("jet_identity_hash") != values["jet_identity_hash"]:
            raise ValueError(f"pilot input identity mismatch for {split}")
        rows[split] = values
    target_set = target_cache_dir / "hierarchy_target_cache_manifest.json"
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_hash(manifest_path),
        "manifest_hash": manifest_sha,
        "hlt_cache_dir": str(hlt_cache_dir.resolve()),
        "offline_cache_dir": str(offline_cache_dir.resolve()),
        "target_cache_dir": str(target_cache_dir.resolve()),
        "target_cache_set_sha256": _file_hash(target_set),
        "splits": rows,
    }


def write_approved_profile(
    *,
    candidate_profile_path: str | Path,
    c5_ten_epoch_certification: str | Path,
    c6_ten_epoch_certification: str | Path,
    c5_fp32_reference: str | Path,
    c6_fp32_reference: str | Path,
    c5_tagger_sanity: str | Path,
    c6_tagger_sanity: str | Path,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
    target_cache_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Close the passed 10/30-epoch and tagger evidence into one immutable approval."""

    candidate = validate_runtime_profile(
        candidate_profile_path,
        expected_status="accelerated_candidate_v1",
        require_current_environment=True,
    )
    candidate_profile = candidate["profile"]
    candidate_hash = candidate_profile["candidate_profile_hash"]
    environment_hash = candidate_profile["code_environment_hash"]
    evidence_paths = {
        "c5_ten_epoch_certification": Path(c5_ten_epoch_certification),
        "c6_ten_epoch_certification": Path(c6_ten_epoch_certification),
        "c5_fp32_reference": Path(c5_fp32_reference),
        "c6_fp32_reference": Path(c6_fp32_reference),
        "c5_tagger_sanity": Path(c5_tagger_sanity),
        "c6_tagger_sanity": Path(c6_tagger_sanity),
    }
    artifacts: dict[str, Any] = {}
    for name, path in evidence_paths.items():
        payload = _read_json(path)
        _validate_approval_evidence(
            name=name,
            payload=payload,
            evidence_path=path,
            candidate_hash=candidate_hash,
            environment_hash=environment_hash,
            candidate_profile=candidate_profile,
            candidate_profile_path=Path(candidate["path"]),
            candidate_profile_file_sha256=str(candidate["file_sha256"]),
        )
        artifacts[name] = {"path": str(path.resolve()), "sha256": _file_hash(path)}
    pilot_inputs = _input_provenance(
        manifest_path=Path(manifest_path),
        hlt_cache_dir=Path(hlt_cache_dir),
        offline_cache_dir=Path(offline_cache_dir),
        target_cache_dir=Path(target_cache_dir),
    )
    payload: dict[str, Any] = {
        "contract": RUNTIME_APPROVED_CONTRACT,
        "status": "accelerated_approved_v1",
        "queue_rights": {"full_pilot": True, "high_data": True, "final_claims": True},
        "runtime_profiles_by_variant": candidate_profile["runtime_profiles_by_variant"],
        "execution_by_variant": candidate_profile["execution_by_variant"],
        "shared_optimizer": candidate_profile["shared_optimizer"],
        "code_environment": candidate_profile["code_environment"],
        "code_environment_hash": environment_hash,
        "closure": {
            "candidate": {
                "path": candidate["path"],
                "sha256": candidate["file_sha256"],
                "candidate_profile_hash": candidate_hash,
            },
            "artifacts": artifacts,
            "pilot_inputs": pilot_inputs,
        },
    }
    payload["approved_profile_hash"] = _canonical_hash(payload)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable approved profile: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "RUNTIME_APPROVED_CONTRACT",
    "execution_shell_exports",
    "execution_variant_for_run",
    "resolve_execution",
    "validate_runtime_profile",
    "write_approved_profile",
]
