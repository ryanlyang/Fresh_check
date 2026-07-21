#!/usr/bin/env python3
"""Validate metadata contracts before reusing LPRF campaign inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.jetclass_data import load_split_manifest, manifest_hash  # noqa: E402


CONTRACT = "local_residual_field_curriculum_reused_inputs_audit_v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata is not a JSON object: {path}")
    return payload


def _find(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    for value in payload.values():
        if isinstance(value, Mapping):
            found = _find(value, key)
            if found not in (None, ""):
                return found
    return None


def _audit_metadata_dir(
    root: str,
    *,
    patterns: tuple[str, ...],
    expected_manifest_hash: str,
    required_content_keys: tuple[str, ...],
    expected_profile: str | None = None,
    expected_strength: float | None = None,
) -> list[dict[str, Any]]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"reused cache directory does not exist: {directory}")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(directory.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"no metadata files found under reused cache: {directory}")
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = _load(path)
        source_hash = _find(payload, "source_manifest_hash")
        if str(source_hash or "") != str(expected_manifest_hash):
            raise ValueError(
                f"reused metadata manifest hash mismatch at {path}: {source_hash!r} != {expected_manifest_hash!r}"
            )
        content = {key: _find(payload, key) for key in required_content_keys}
        missing = [key for key, value in content.items() if value in (None, "")]
        if missing:
            raise ValueError(f"reused metadata {path} is missing content hashes: {missing}")
        if expected_profile is not None:
            profile = _find(payload, "hlt_profile")
            if str(profile or "") != str(expected_profile):
                raise ValueError(f"reused HLT profile mismatch at {path}: {profile!r} != {expected_profile!r}")
        if expected_strength is not None:
            strength = _find(payload, "hlt_degradation_strength")
            try:
                matches = math.isclose(float(strength), float(expected_strength), rel_tol=0.0, abs_tol=1.0e-12)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(
                    f"reused HLT degradation strength mismatch at {path}: {strength!r} != {expected_strength!r}"
                )
        rows.append({"path": str(path), "content_hashes": content})
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--hlt-cache-dir", default="")
    parser.add_argument("--offline-cache-dir", default="")
    parser.add_argument("--target-cache-dir", default="")
    parser.add_argument("--offline-teacher-logits-dir", default="")
    parser.add_argument("--expected-hlt-profile", required=True)
    parser.add_argument("--expected-hlt-degradation-strength", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_split_manifest(args.manifest_path)
    source_hash = manifest_hash(manifest)
    audits: dict[str, Any] = {}
    if args.hlt_cache_dir:
        audits["hlt_cache"] = _audit_metadata_dir(
            args.hlt_cache_dir,
            patterns=("*_fixed_hlt_metadata.json",),
            expected_manifest_hash=source_hash,
            required_content_keys=("hlt_content_hash",),
            expected_profile=args.expected_hlt_profile,
            expected_strength=args.expected_hlt_degradation_strength,
        )
    if args.offline_cache_dir:
        audits["offline_cache"] = _audit_metadata_dir(
            args.offline_cache_dir,
            patterns=("*_offline_metadata.json",),
            expected_manifest_hash=source_hash,
            required_content_keys=("offline_content_hash",),
        )
    if args.target_cache_dir:
        audits["target_cache"] = _audit_metadata_dir(
            args.target_cache_dir,
            patterns=("*_local_particle_residual_fields_metadata.json",),
            expected_manifest_hash=source_hash,
            required_content_keys=("hlt_content_hash", "offline_content_hash", "target_content_hash"),
        )
    if args.offline_teacher_logits_dir:
        audits["offline_teacher_logits"] = _audit_metadata_dir(
            args.offline_teacher_logits_dir,
            patterns=("*_predictions_metadata.json", "*_metadata.json"),
            expected_manifest_hash=source_hash,
            required_content_keys=("prediction_content_hash",),
        )
    if not audits:
        raise ValueError("at least one reused cache directory must be provided")
    report = {
        "ok": True,
        "contract": CONTRACT,
        "manifest_path": str(args.manifest_path),
        "source_manifest_hash": source_hash,
        "expected_hlt_profile": args.expected_hlt_profile,
        "expected_hlt_degradation_strength": args.expected_hlt_degradation_strength,
        "audits": audits,
    }
    save_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
