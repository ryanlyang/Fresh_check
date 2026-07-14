#!/usr/bin/env python3
"""Fail-closed completion checks for constrained coarse-to-fine artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

HIERARCHY_TARGET_CACHE_CONTRACT = "constrained_coarse_to_fine_hierarchy_target_cache_v1"
HIERARCHY_TARGET_CACHE_SET_CONTRACT = "constrained_coarse_to_fine_hierarchy_target_cache_set_v1"
HIERARCHY_TARGET_EXPECTED_HLT_PROFILE = "fixed_hlt_v2_realistic"
HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION = "v1"
HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH = 2.5

_RECONSTRUCTOR_REPORT_VARIANTS = {
    "B0": "B0_global_only",
    "B1": "B1_global_8",
    "B2": "B2_global_8_32",
    "B3": "B3_global_8_32_128",
    "B4": "B4_no_moments",
    "B5": "B5_no_composition",
    "B6": "B6_no_counts",
    "B7": "B7_direct_child_totals",
    "C0": "C0_deterministic_k8",
    "C1": "C1_deterministic_k16",
    "C2": "C2_no_dust",
    "C3": "C3_sinkhorn",
    "C4": "C4_hungarian",
    "C5": "C5_uncertainty",
    "C6": "C6_multiview",
    "C5-B1": "C5-B1",
    "C5-B2": "C5-B2",
    "C5-B3": "C5-B3",
    "C5-no-slot": "C5_uncertainty",
    "C5-unconstrained": "C5_uncertainty",
    "Cdirect-unconstrained": "C5_uncertainty",
}


def _expected_training_variant(run_id: str, *, tagger: bool) -> str:
    if not tagger:
        return _RECONSTRUCTOR_REPORT_VARIANTS.get(run_id, run_id)
    if "-seed" in run_id:
        base, suffix = run_id.rsplit("-seed", 1)
        if base.startswith("D") and suffix.isdigit():
            return base
    return run_id


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_hash(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else path.open
    if path.suffix == ".gz":
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with opener("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty required file: {path}")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_hlt_contract(row: Mapping[str, Any], manifest_sha: str, label: str) -> None:
    _require_equal(row.get("source_manifest_hash"), manifest_sha, f"{label} source_manifest_hash")
    _require_equal(row.get("hlt_profile"), HIERARCHY_TARGET_EXPECTED_HLT_PROFILE, f"{label} hlt_profile")
    _require_equal(
        str(row.get("hlt_profile_version") or ""),
        HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION,
        f"{label} hlt_profile_version",
    )
    try:
        strength = float(row.get("hlt_degradation_strength"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} lacks numeric hlt_degradation_strength") from exc
    if abs(strength - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
        raise ValueError(
            f"{label} HLT strength mismatch: {strength} != {HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH}"
        )


def _require_provenance(report: Mapping[str, Any], manifest_sha: str, label: str) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} lacks provenance")
    for split in ("model_train", "model_val"):
        row = provenance.get(split)
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} lacks {split} provenance")
        _require_hlt_contract(row, manifest_sha, f"{label}/{split}")
        for field in ("hlt_content_hash", "jet_identity_hash"):
            if row.get(field) in (None, ""):
                raise ValueError(f"{label}/{split} lacks {field}")


def _validate_hlt_cache(root: Path, splits: tuple[str, ...], manifest_sha: str) -> None:
    for split in splits:
        _require_file(root / f"{split}_fixed_hlt.npz")
        metadata = _read_json(root / f"{split}_fixed_hlt_metadata.json")
        _require_hlt_contract(metadata, manifest_sha, f"HLT cache/{split}")
        for field in ("hlt_content_hash", "jet_identity_hash"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"HLT cache/{split} lacks {field}")


def _validate_offline_cache(root: Path, splits: tuple[str, ...], manifest_sha: str) -> None:
    for split in splits:
        _require_file(root / f"{split}_offline.npz")
        metadata = _read_json(root / f"{split}_offline_metadata.json")
        _require_equal(metadata.get("source_manifest_hash"), manifest_sha, f"offline cache/{split} manifest")
        for field in ("offline_content_hash", "jet_identity_hash"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"offline cache/{split} lacks {field}")


def _validate_target_cache(root: Path, splits: tuple[str, ...], manifest_sha: str) -> None:
    cache_set = _read_json(root / "hierarchy_target_cache_manifest.json")
    _require_equal(cache_set.get("cache_set_contract"), HIERARCHY_TARGET_CACHE_SET_CONTRACT, "target cache set contract")
    _require_equal(cache_set.get("source_manifest_hash"), manifest_sha, "target cache set manifest")
    _require_hlt_contract(cache_set, manifest_sha, "target cache set")
    declared = set(str(value) for value in cache_set.get("splits", ()))
    missing = sorted(set(splits) - declared)
    if missing:
        raise ValueError(f"target cache set is missing splits: {missing}")
    for split in splits:
        metadata = _read_json(root / f"{split}_hierarchy_targets_metadata.json")
        _require_equal(metadata.get("cache_contract"), HIERARCHY_TARGET_CACHE_CONTRACT, f"target/{split} contract")
        _require_hlt_contract(metadata, manifest_sha, f"target/{split}")
        for field in ("target_content_hash", "hlt_content_hash", "offline_content_hash", "jet_identity_hash"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"target/{split} lacks {field}")
        shards = metadata.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"target/{split} has no declared shards")
        shard_root = root / f"{split}_hierarchy_targets"
        for row in shards:
            if not isinstance(row, Mapping) or not row.get("filename"):
                raise ValueError(f"target/{split} contains malformed shard metadata")
            _require_file(shard_root / str(row["filename"]))


def _validate_training(root: Path, run_id: str, manifest_sha: str, *, tagger: bool) -> None:
    report = _read_json(root / "run_report.json")
    if report.get("ok") is not True:
        raise ValueError(f"{run_id} run_report is not successful")
    observed = report.get("variant") if tagger else (report.get("variant") or report.get("family"))
    expected = _expected_training_variant(run_id, tagger=tagger)
    if observed not in (expected, None):
        raise ValueError(f"{run_id} report identifies a different run: {observed}")
    declared_run_id = report.get("run_id")
    if declared_run_id not in (None, run_id):
        raise ValueError(f"{run_id} report declares a different campaign run_id: {declared_run_id}")
    if not tagger and run_id in {"C5-unconstrained", "Cdirect-unconstrained"}:
        model = report.get("model")
        slot = model.get("slot_config") if isinstance(model, Mapping) else None
        if not isinstance(slot, Mapping) or slot.get("direct_particle_decoding") is not True:
            raise ValueError(f"{run_id} is not attested as a direct particle decoder")
        if slot.get("constrain_accounting") is not False:
            raise ValueError(f"{run_id} unexpectedly retains hierarchical slot accounting constraints")
    checkpoint = root / "best_model_val.pt"
    _require_file(checkpoint)
    if report.get("checkpoint_sha256") in (None, ""):
        raise ValueError(f"{run_id} run_report lacks checkpoint_sha256")
    if tagger and not isinstance(report.get("provenance"), Mapping):
        source_metadata = _read_json(root / "source_metadata.json")
        report = {**report, "provenance": source_metadata.get("provenance")}
    _require_provenance(report, manifest_sha, run_id)


def _validate_predictions(root: Path, run_id: str, splits: tuple[str, ...], manifest_sha: str) -> None:
    report = _read_json(root / "prediction_run_report.json")
    if report.get("ok") is not True:
        raise ValueError(f"{run_id} prediction_run_report is not successful")
    rows = report.get("splits")
    if not isinstance(rows, Mapping):
        raise ValueError(f"{run_id} prediction report lacks split rows")
    for split in splits:
        _require_file(root / f"{split}_predictions.npz")
        _require_file(root / f"{split}_representations.npz")
        metadata = _read_json(root / f"{split}_predictions_metadata.json")
        _require_equal(metadata.get("run_id"), run_id, f"prediction/{run_id}/{split} run_id")
        _require_hlt_contract(metadata, manifest_sha, f"prediction/{run_id}/{split}")
        if metadata.get("deployable_hlt_only") is not True:
            raise ValueError(f"prediction/{run_id}/{split} is not deployable_hlt_only")
        for field in (
            "prediction_content_hash",
            "checkpoint_sha256",
            "hlt_content_hash",
            "jet_identity_hash",
            "fusion_representation_sha256",
        ):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"prediction/{run_id}/{split} lacks {field}")
        if split == "final_test":
            if metadata.get("final_test_confirmed") is not True:
                raise ValueError(f"prediction/{run_id}/final_test is not confirmed")
            receipt = _read_json(root / "final_test_claim_receipt.json")
            if receipt.get("immutable_final_claim") is not True:
                raise ValueError(f"prediction/{run_id}/final_test lacks immutable claim receipt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("hlt-cache", "offline-cache", "target-cache", "reconstructor", "tagger", "prediction"))
    parser.add_argument("--path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--splits", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path)
    manifest_sha = _manifest_hash(Path(args.manifest))
    splits = tuple(str(value) for value in args.splits)
    if args.kind.endswith("cache") and not splits:
        raise ValueError(f"{args.kind} validation requires --splits")
    if args.kind in ("reconstructor", "tagger", "prediction") and not args.run_id:
        raise ValueError(f"{args.kind} validation requires --run-id")
    if args.kind == "hlt-cache":
        _validate_hlt_cache(root, splits, manifest_sha)
    elif args.kind == "offline-cache":
        _validate_offline_cache(root, splits, manifest_sha)
    elif args.kind == "target-cache":
        _validate_target_cache(root, splits, manifest_sha)
    elif args.kind == "reconstructor":
        _validate_training(root, str(args.run_id), manifest_sha, tagger=False)
    elif args.kind == "tagger":
        _validate_training(root, str(args.run_id), manifest_sha, tagger=True)
    else:
        if not splits:
            raise ValueError("prediction validation requires --splits")
        _validate_predictions(root, str(args.run_id), splits, manifest_sha)
    print(f"valid {args.kind}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
