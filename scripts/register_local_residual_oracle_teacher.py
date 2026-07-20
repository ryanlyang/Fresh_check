#!/usr/bin/env python3
"""Register a provenance-compatible local residual-field oracle teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.hlt_cache import HLT_METADATA_FILENAME  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_split_manifest, manifest_hash  # noqa: E402
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES  # noqa: E402
from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_CONFIG_CONTRACT,
    LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
    normalize_residual_field_source,
    resolve_local_residual_field_indices,
)
from teacher_logit_reco.local_particle_residual_field.oracle_teacher import (  # noqa: E402
    ORACLE_TEACHER_TRAIN_SPLITS,
    build_oracle_teacher_reuse_contract,
    validate_oracle_teacher_reuse_contract,
)


REGISTRATION_CONTRACT = "local_residual_field_oracle_teacher_registration_v2"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return dict(payload)


def _float_matches(actual: Any, expected: float | None, *, name: str, tolerance: float = 1.0e-8) -> None:
    if expected is None:
        return
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is missing or not numeric in source run") from exc
    if abs(value - float(expected)) > float(tolerance):
        raise ValueError(f"{name} mismatch: source has {value}, expected {float(expected)}")


def _config_field(payload: Mapping[str, Any], name: str) -> Any:
    if name in payload:
        return payload.get(name)
    train_config = payload.get("train_config") if isinstance(payload.get("train_config"), Mapping) else {}
    if name in train_config:
        return train_config.get(name)
    model_config = payload.get("model_config") if isinstance(payload.get("model_config"), Mapping) else {}
    return model_config.get(name)


def _source_compatible(actual: str, expected: str) -> bool:
    actual_norm = normalize_residual_field_source(actual)
    expected_norm = normalize_residual_field_source(expected)
    return actual_norm == expected_norm or {
        actual_norm,
        expected_norm,
    } <= {"oracle", "oracle_scaled"}


def _source_teacher_payload(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_report = _load_json(source_dir / "run_report.json")
    teacher_config_path = source_dir / "teacher_config.json"
    if teacher_config_path.exists():
        teacher_config = _load_json(teacher_config_path)
    else:
        teacher_config = {
            "contract": LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_CONFIG_CONTRACT,
            "teacher_id": source_dir.name,
            "field_source": run_report.get("field_source"),
            "field_subset": run_report.get("field_subset") or (),
            "selected_field_indices": run_report.get("selected_field_indices") or (),
            "selected_field_names": run_report.get("selected_field_names") or (),
            "selected_field_groups": run_report.get("selected_field_groups") or {},
            "model_config": run_report.get("model_config") or {},
            "train_config": run_report.get("config") or {},
            "dataset_metadata": run_report.get("dataset_metadata") or {},
        }
    return run_report, teacher_config


def _validate_recipe(
    teacher_config: Mapping[str, Any],
    *,
    expected_field_source: str | None,
    expected_alpha: float | None,
    expected_noise_std: float | None,
    expected_field_dropout: float | None,
    expected_group_dropout: float | None,
) -> None:
    source_field = str(_config_field(teacher_config, "field_source") or "")
    if expected_field_source and not _source_compatible(source_field, expected_field_source):
        raise ValueError(f"field_source mismatch: source has {source_field}, expected {expected_field_source}")
    _float_matches(_config_field(teacher_config, "oracle_field_alpha"), expected_alpha, name="oracle_field_alpha")
    _float_matches(
        _config_field(teacher_config, "oracle_field_noise_std"),
        expected_noise_std,
        name="oracle_field_noise_std",
    )
    _float_matches(
        _config_field(teacher_config, "oracle_field_dropout"),
        expected_field_dropout,
        name="oracle_field_dropout",
    )
    _float_matches(
        _config_field(teacher_config, "oracle_field_group_dropout"),
        expected_group_dropout,
        name="oracle_field_group_dropout",
    )


def _remap_groups(groups: Mapping[str, Sequence[int]], selected_indices: Sequence[int]) -> dict[str, list[int]]:
    old_to_new = {int(old): int(new) for new, old in enumerate(selected_indices)}
    output: dict[str, list[int]] = {}
    for group, values in groups.items():
        remapped = [old_to_new[int(index)] for index in values if int(index) in old_to_new]
        if remapped:
            output[str(group)] = remapped
    return output or {"all": list(range(len(tuple(selected_indices))))}


def _active_expected_payload(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_split_manifest(args.manifest_path)
    active_manifest_hash = manifest_hash(manifest)
    dataset_metadata: dict[str, Any] = {}
    reference_names: tuple[str, ...] | None = None
    reference_groups: dict[str, list[int]] | None = None
    target_label_names: tuple[str, ...] | None = None
    for split in args.required_train_splits:
        hlt_path = Path(args.hlt_cache_dir) / HLT_METADATA_FILENAME.format(split=split)
        target_path = Path(args.target_cache_dir) / f"{split}_local_particle_residual_fields_metadata.json"
        if not hlt_path.exists():
            raise FileNotFoundError(f"active HLT cache metadata is missing: {hlt_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"active target cache metadata is missing: {target_path}")
        hlt = _load_json(hlt_path)
        target = _load_json(target_path)
        names = tuple(str(name) for name in target.get("field_names") or ())
        groups = {
            str(group): [int(index) for index in values]
            for group, values in dict(target.get("field_groups") or {}).items()
        }
        labels = tuple(str(name) for name in target.get("label_names") or ())
        if reference_names is None:
            reference_names, reference_groups, target_label_names = names, groups, labels
        elif names != reference_names or groups != reference_groups:
            raise ValueError(f"active target field schema differs on split {split}")
        if labels and target_label_names and labels != target_label_names:
            raise ValueError(f"active target label ordering differs on split {split}")
        for source_name, payload in (("HLT", hlt), ("target", target)):
            declared = payload.get("source_manifest_hash")
            if declared not in (None, active_manifest_hash):
                raise ValueError(
                    f"active {source_name} metadata for {split} has source_manifest_hash={declared}, "
                    f"expected {active_manifest_hash}"
                )
        hlt_identity = hlt.get("jet_identity_hash")
        target_identity = target.get("hlt_jet_identity_hash") or target.get("jet_identity_hash")
        if hlt_identity and target_identity and hlt_identity != target_identity:
            raise ValueError(f"active HLT/target jet_identity_hash mismatch on split {split}")
        dataset_metadata[str(split)] = {
            "alignment_report": {
                "source_manifest_hash": active_manifest_hash,
                "hlt_content_hash": hlt.get("hlt_content_hash"),
                "offline_content_hash": target.get("offline_content_hash"),
                "target_content_hash": target.get("target_content_hash"),
                "jet_identity_hash": hlt_identity or target_identity,
            },
            "hlt_metadata": hlt,
            "target_metadata": target,
        }
    assert reference_names is not None and reference_groups is not None
    selected_indices = resolve_local_residual_field_indices(
        field_names=reference_names,
        field_groups=reference_groups,
        subset=tuple(args.expected_field_subset),
    )
    selected_names = [reference_names[index] for index in selected_indices]
    selected_groups = _remap_groups(reference_groups, selected_indices)
    expected_labels = tuple(str(name) for name in args.expected_label_names)
    if target_label_names and tuple(target_label_names) != expected_labels:
        raise ValueError(
            f"active target label ordering {list(target_label_names)} != expected {list(expected_labels)}"
        )
    field_dim = len(selected_names)
    model_config = {
        "contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
        "num_classes": int(args.expected_num_classes),
        "field_dim": field_dim,
        "base_feature_dim": int(args.expected_base_feature_dim),
        "augmented_feature_dim": int(args.expected_base_feature_dim) + field_dim,
        "model_size": str(args.expected_model_size),
        "field_source": str(args.expected_field_source),
        "field_names": selected_names,
        "field_groups": selected_groups,
    }
    return {
        "field_source": str(args.expected_field_source),
        "oracle_field_alpha": float(args.expected_alpha),
        "oracle_field_noise_std": float(args.expected_noise_std),
        "oracle_field_dropout": float(args.expected_field_dropout),
        "oracle_field_group_dropout": float(args.expected_group_dropout),
        "field_subset": list(args.expected_field_subset),
        "selected_field_indices": list(selected_indices),
        "selected_field_names": selected_names,
        "selected_field_groups": selected_groups,
        "model_config": model_config,
        "train_config": {
            "num_classes": int(args.expected_num_classes),
            "label_names": list(expected_labels),
            "model_size": str(args.expected_model_size),
            "field_subset": list(args.expected_field_subset),
        },
        "dataset_metadata": dataset_metadata,
    }


def _materialize_file(source: Path, target: Path, *, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if mode == "symlink":
        try:
            os.symlink(source, target)
            return "symlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy_fallback"
    raise ValueError(f"unknown link mode {mode!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument(
        "--candidate-run-dir",
        action="append",
        default=[],
        help="Additional reuse candidate; candidates are audited in the supplied order.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--expected-field-source", required=True)
    parser.add_argument("--expected-alpha", type=float, required=True)
    parser.add_argument("--expected-noise-std", type=float, required=True)
    parser.add_argument("--expected-field-dropout", type=float, required=True)
    parser.add_argument("--expected-group-dropout", type=float, required=True)
    parser.add_argument("--expected-field-subset", nargs="*", default=[])
    parser.add_argument("--expected-label-names", nargs="+", default=list(LABEL_NAMES))
    parser.add_argument("--expected-num-classes", type=int, default=len(LABEL_NAMES))
    parser.add_argument("--expected-base-feature-dim", type=int, default=len(PF_FEATURE_NAMES))
    parser.add_argument("--expected-model-size", default="base")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--hlt-cache-dir", default="")
    parser.add_argument("--target-cache-dir", default="")
    parser.add_argument("--required-train-splits", nargs="+", default=list(ORACLE_TEACHER_TRAIN_SPLITS))
    parser.add_argument("--expected-teacher-config", default="")
    parser.add_argument("--allow-unverified-registration", action="store_true")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    allowed_runner_artifacts = {"slurm_run_config.json"}
    existing_entries = (
        [path for path in output_dir.iterdir() if path.name not in allowed_runner_artifacts]
        if output_dir.exists()
        else []
    )
    if existing_entries and not bool(args.overwrite):
        raise FileExistsError(
            f"output directory contains registration artifacts; pass --overwrite to replace them: {output_dir}"
        )
    expected_contract: dict[str, Any] | None = None
    if args.expected_teacher_config:
        expected_payload = _load_json(Path(args.expected_teacher_config))
        expected_contract = dict(
            expected_payload.get("reuse_contract")
            or build_oracle_teacher_reuse_contract(
                expected_payload,
                required_splits=tuple(args.required_train_splits),
            )
        )
    elif args.manifest_path and args.hlt_cache_dir and args.target_cache_dir:
        expected_contract = build_oracle_teacher_reuse_contract(
            _active_expected_payload(args),
            required_splits=tuple(args.required_train_splits),
        )
    elif not bool(args.allow_unverified_registration):
        raise ValueError(
            "provenance verification is required: provide --expected-teacher-config or all of "
            "--manifest-path/--hlt-cache-dir/--target-cache-dir; use --allow-unverified-registration only "
            "for an explicitly acknowledged legacy diagnostic"
        )

    candidate_dirs = [Path(args.source_run_dir), *(Path(value) for value in args.candidate_run_dir)]
    candidate_audit: list[dict[str, Any]] = []
    selected: tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for candidate_dir in candidate_dirs:
        try:
            for required in ("best_model_val.pt", "run_report.json", "training_curves.json"):
                path = candidate_dir / required
                if not path.exists():
                    raise FileNotFoundError(f"source oracle teacher is missing required artifact: {path}")
            candidate_report, candidate_config = _source_teacher_payload(candidate_dir)
            if not bool(candidate_report.get("ok", True)):
                raise ValueError("source run_report.json is not ok")
            _validate_recipe(
                candidate_config,
                expected_field_source=args.expected_field_source,
                expected_alpha=args.expected_alpha,
                expected_noise_std=args.expected_noise_std,
                expected_field_dropout=args.expected_field_dropout,
                expected_group_dropout=args.expected_group_dropout,
            )
            candidate_contract = build_oracle_teacher_reuse_contract(
                candidate_config,
                required_splits=tuple(args.required_train_splits),
            )
            candidate_validation = (
                validate_oracle_teacher_reuse_contract(candidate_contract, expected_contract)
                if expected_contract is not None
                else {
                    "ok": True,
                    "contract": candidate_contract["contract"],
                    "unverified_legacy_override": True,
                    "mismatches": [],
                }
            )
            candidate_audit.append(
                {
                    "source_run_dir": str(candidate_dir),
                    "ok": bool(candidate_validation.get("ok")),
                    "reuse_contract_hash": candidate_contract.get("reuse_contract_hash"),
                    "validation": candidate_validation,
                }
            )
            if bool(candidate_validation.get("ok")):
                selected = (
                    candidate_dir,
                    candidate_report,
                    candidate_config,
                    candidate_contract,
                    candidate_validation,
                )
                break
        except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
            candidate_audit.append(
                {
                    "source_run_dir": str(candidate_dir),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if selected is None:
        details = "; ".join(
            f"{item['source_run_dir']}: {item.get('error') or item.get('validation', {}).get('mismatches')}"
            for item in candidate_audit
        )
        raise ValueError(f"oracle teacher reuse provenance mismatch: no compatible candidate; {details}")
    source_dir, run_report, teacher_config, actual_contract, validation = selected

    output_dir.mkdir(parents=True, exist_ok=True)
    materialization = {
        "best_model_val.pt": _materialize_file(
            source_dir / "best_model_val.pt", output_dir / "best_model_val.pt", mode=args.link_mode
        ),
        "training_curves.json": _materialize_file(
            source_dir / "training_curves.json", output_dir / "training_curves.json", mode=args.link_mode
        ),
    }
    registered_teacher_config = dict(teacher_config)
    registered_teacher_config.update(
        {
            "contract": LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_CONFIG_CONTRACT,
            "teacher_id": str(args.teacher_id),
            "registered_from": str(source_dir),
            "checkpoint": str(output_dir / "best_model_val.pt"),
            "reuse_contract": actual_contract,
            "reuse_validation": validation,
            "reuse_candidate_audit": candidate_audit,
        }
    )
    registered_report = dict(run_report)
    registered_report.update(
        {
            "ok": bool(run_report.get("ok", True)),
            "teacher_id": str(args.teacher_id),
            "registered_from": str(source_dir),
            "checkpoint": str(output_dir / "best_model_val.pt"),
            "teacher_config": str(output_dir / "teacher_config.json"),
            "teacher_reuse_contract": actual_contract,
            "teacher_reuse_contract_hash": actual_contract["reuse_contract_hash"],
            "reuse_validation": validation,
            "reuse_candidate_audit": candidate_audit,
        }
    )
    registration_report = {
        "ok": True,
        "contract": REGISTRATION_CONTRACT,
        "teacher_id": str(args.teacher_id),
        "source_run_dir": str(source_dir),
        "output_dir": str(output_dir),
        "link_mode": str(args.link_mode),
        "materialization": materialization,
        "expected_field_source": args.expected_field_source,
        "expected_alpha": args.expected_alpha,
        "expected_noise_std": args.expected_noise_std,
        "expected_field_dropout": args.expected_field_dropout,
        "expected_group_dropout": args.expected_group_dropout,
        "actual_reuse_contract": actual_contract,
        "expected_reuse_contract": expected_contract,
        "reuse_validation": validation,
        "unverified_legacy_override": bool(args.allow_unverified_registration and expected_contract is None),
        "candidate_audit": candidate_audit,
    }
    save_json(output_dir / "teacher_config.json", registered_teacher_config)
    save_json(output_dir / "run_report.json", registered_report)
    save_json(output_dir / "registration_report.json", registration_report)
    print(json.dumps(registration_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
