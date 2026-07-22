#!/usr/bin/env python3
"""Build and audit the exact from-scratch A0_seed1 training recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.fusion_seed_control import (  # noqa: E402
    build_a0_seed1_recipe,
    extract_a0_train_config,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-source-metadata", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recipe-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--a0-run-config", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--hlt-cache-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


def _current_source_provenance() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1") or ""
    return {
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_status_hash": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_dirty": bool(status),
    }


_TRAINING_SOURCE_FILES = (
    "teacher_logit_reco/local_particle_residual_field/tagger_train.py",
    "teacher_logit_reco/local_particle_residual_field/tagger.py",
    "teacher_logit_reco/local_particle_residual_field/data.py",
    "jetclass_fresh/hlt_baseline.py",
    "jetclass_fresh/part_inputs.py",
)


def _git_blob_sha256(commit: str, relative_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _training_source_audit(a0_commit: str | None) -> dict[str, Any]:
    rows = []
    for relative_path in _TRAINING_SOURCE_FILES:
        current_path = REPO_ROOT / relative_path
        current_hash = sha256_file(current_path) if current_path.is_file() else None
        a0_hash = _git_blob_sha256(str(a0_commit), relative_path) if a0_commit else None
        rows.append(
            {
                "path": relative_path,
                "a0_source_hash": a0_hash,
                "current_source_hash": current_hash,
                "matches": bool(a0_hash is not None and current_hash == a0_hash),
            }
        )
    return {
        "a0_source_commit": a0_commit,
        "files": rows,
        "training_source_match": bool(a0_commit and all(row["matches"] for row in rows)),
    }


def _optional_file_hash(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else None


def _write_new_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    save_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_metadata_path = Path(args.a0_source_metadata)
    checkpoint_path = Path(args.a0_checkpoint)
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"A0 source metadata does not exist: {source_metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"A0 checkpoint does not exist: {checkpoint_path}")

    source_payload = _load_json(source_metadata_path)
    a0_config = extract_a0_train_config(source_payload)
    a0_run_config = _load_json(args.a0_run_config)
    source_audit = _training_source_audit(a0_run_config.get("source_commit"))
    training_module = REPO_ROOT / "teacher_logit_reco" / "local_particle_residual_field" / "tagger_train.py"
    provenance = {
        "a0_source_metadata": str(source_metadata_path),
        "a0_source_metadata_hash": sha256_file(source_metadata_path),
        "a0_checkpoint": str(checkpoint_path),
        "a0_checkpoint_hash": sha256_file(checkpoint_path),
        "a0_source_commit": a0_run_config.get("source_commit"),
        "a0_source_status_hash": a0_run_config.get("source_status_hash"),
        "training_source_match": source_audit["training_source_match"],
        "training_source_audit": source_audit,
        "candidate_source": _current_source_provenance(),
        "training_module": str(training_module),
        "training_module_hash": sha256_file(training_module),
        "manifest_path": args.manifest_path or a0_config.get("manifest_path"),
        "manifest_hash": _optional_file_hash(args.manifest_path or a0_config.get("manifest_path")),
        "hlt_cache_dir": args.hlt_cache_dir or a0_config.get("hlt_cache_dir"),
        "deterministic_backend_settings": {
            "training_seed": 20522,
            "dataloader_seed_offsets": [0, 1, 2],
            "torch_deterministic_algorithms_forced": False,
        },
    }
    recipe, audit = build_a0_seed1_recipe(a0_config, output_dir=args.output_dir, provenance=provenance)

    recipe_path = Path(args.recipe_output)
    audit_path = Path(args.audit_output)
    _write_new_json(audit_path, audit, overwrite=bool(args.overwrite))
    _write_new_json(recipe_path, recipe.to_dict(), overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "ok": True,
                "recipe": str(recipe_path),
                "audit": str(audit_path),
                "config_hash": recipe.config_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
