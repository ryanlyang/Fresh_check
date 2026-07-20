#!/usr/bin/env python3
"""Validate Step 2 oracle-teacher logit cache artifacts and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import load_prediction_block  # noqa: E402
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.local_particle_residual_field import ORACLE_RESIDUAL_FIELD_SOURCES  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.oracle_teacher import (  # noqa: E402
    ORACLE_TEACHER_LOGIT_SPLITS,
)


VALIDATION_CONTRACT = "local_residual_field_oracle_teacher_logits_validation_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--splits", nargs="+", default=list(ORACLE_TEACHER_LOGIT_SPLITS))
    parser.add_argument("--allow-partial-splits", action="store_true")
    parser.add_argument("--output-path", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested = tuple(str(split) for split in args.splits)
    if len(set(requested)) != len(requested):
        raise ValueError("oracle teacher logit splits must be unique")
    unsupported = sorted(set(requested) - set(ORACLE_TEACHER_LOGIT_SPLITS))
    if unsupported:
        raise ValueError(f"unsupported oracle teacher logit splits: {unsupported}")
    missing_required = sorted(set(ORACLE_TEACHER_LOGIT_SPLITS) - set(requested))
    if missing_required and not bool(args.allow_partial_splits):
        raise ValueError(f"missing required oracle teacher logit splits: {missing_required}")

    teacher_dir = Path(args.teacher_dir)
    checkpoint = teacher_dir / "best_model_val.pt"
    teacher_config_path = teacher_dir / "teacher_config.json"
    run_report_path = teacher_dir / "run_report.json"
    for path in (checkpoint, teacher_config_path, run_report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    teacher_config = _load_json(teacher_config_path)
    run_report = _load_json(run_report_path)
    teacher_id = str(args.teacher_id)
    if str(teacher_config.get("teacher_id")) != teacher_id:
        raise ValueError(
            f"teacher_config teacher_id={teacher_config.get('teacher_id')!r} does not match {teacher_id!r}"
        )
    if not bool(run_report.get("ok", True)):
        raise ValueError("oracle teacher run report is not ok")
    reuse_contract = teacher_config.get("reuse_contract")
    if not isinstance(reuse_contract, Mapping) or not reuse_contract.get("reuse_contract_hash"):
        raise ValueError("teacher_config.json lacks the verified oracle teacher reuse contract")
    checkpoint_hash = _sha256_file(checkpoint)
    teacher_config_hash = _sha256_file(teacher_config_path)

    prediction_root = Path(args.prediction_dir)
    teacher_prediction_dir = prediction_root / teacher_id
    forbidden = sorted(str(path) for path in teacher_prediction_dir.glob("final_test*"))
    if forbidden:
        raise ValueError(f"final-test oracle teacher logit artifacts are forbidden: {forbidden}")
    manifest_path = teacher_prediction_dir / "prediction_manifest.json"
    manifest = _load_json(manifest_path)
    if str(manifest.get("model_name")) != teacher_id:
        raise ValueError("prediction manifest model_name does not match teacher_id")
    if tuple(str(split) for split in manifest.get("splits") or ()) != requested:
        raise ValueError("prediction manifest splits do not match the requested Step 2 splits")

    split_reports: dict[str, Any] = {}
    for split in requested:
        metadata_path = teacher_prediction_dir / f"{split}_predictions_metadata.json"
        canonical_metadata_path = teacher_prediction_dir / f"{split}_metadata.json"
        if not metadata_path.exists() or not canonical_metadata_path.exists():
            raise FileNotFoundError(f"missing oracle teacher metadata for split {split}")
        metadata = _load_json(metadata_path)
        canonical_metadata = _load_json(canonical_metadata_path)
        if metadata != canonical_metadata:
            raise ValueError(f"canonical metadata alias differs for split {split}")
        if str(metadata.get("teacher_id")) != teacher_id:
            raise ValueError(f"prediction teacher_id mismatch on split {split}")
        if str(metadata.get("checkpoint_hash")) != checkpoint_hash:
            raise ValueError(f"prediction checkpoint hash mismatch on split {split}")
        if str(metadata.get("teacher_config_hash")) != teacher_config_hash:
            raise ValueError(f"prediction teacher_config hash mismatch on split {split}")
        if str(metadata.get("teacher_reuse_contract_hash")) != str(reuse_contract["reuse_contract_hash"]):
            raise ValueError(f"prediction teacher reuse contract mismatch on split {split}")
        field_source = str(metadata.get("field_source"))
        if field_source not in set(ORACLE_RESIDUAL_FIELD_SOURCES) | {"zero"}:
            raise ValueError(f"prediction field source is not oracle-backed on split {split}")
        expected_runtime_inputs = (
            "HLT_plus_zero_residual_fields" if field_source == "zero" else "HLT_plus_true_residual_fields"
        )
        if metadata.get("runtime_inputs") != expected_runtime_inputs:
            raise ValueError(f"oracle diagnostic runtime_inputs mismatch on split {split}")
        expected_uses_true_fields = field_source != "zero"
        if bool(metadata.get("uses_true_fields")) != expected_uses_true_fields or metadata.get("deployable") is not False:
            raise ValueError(f"oracle diagnostic deployment flags mismatch on split {split}")
        block = load_prediction_block(prediction_root, teacher_id, split, verify_hash=True)
        split_reports[split] = {
            "n_jets": int(block.logits.shape[0]),
            "num_classes": int(block.logits.shape[1]),
            "prediction_content_hash": metadata.get("prediction_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
            "metrics": metadata.get("metrics"),
        }

    report = {
        "ok": True,
        "contract": VALIDATION_CONTRACT,
        "teacher_id": teacher_id,
        "teacher_dir": str(teacher_dir),
        "prediction_dir": str(prediction_root),
        "checkpoint_hash": checkpoint_hash,
        "teacher_config_hash": teacher_config_hash,
        "teacher_reuse_contract_hash": reuse_contract["reuse_contract_hash"],
        "splits": list(requested),
        "partial_splits": bool(missing_required),
        "split_reports": split_reports,
        "final_test_oracle_logits_present": False,
    }
    output_path = Path(args.output_path) if args.output_path else teacher_prediction_dir / "cache_validation_report.json"
    save_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
