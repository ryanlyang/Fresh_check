#!/usr/bin/env python3
"""Execute one task-local Stage-C cache/fusion wave and discard its banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.step5 import (  # noqa: E402
    validate_stage_c_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_abc import (  # noqa: E402
    STREAMED_ABC_FUSION_RECEIPT_CONTRACT,
    validate_streamed_abc_execution_profile,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: Sequence[str]) -> None:
    completed = subprocess.run(
        [str(value) for value in argv], cwd=REPO_ROOT, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"streamed fusion child failed ({completed.returncode}): "
            + " ".join(str(value) for value in argv)
        )


def _base_tmp() -> Path:
    candidates = (
        os.environ.get("RETB_STREAM_ROOT"),
        os.environ.get("SLURM_TMPDIR"),
        "/dev/shm",
    )
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir() and os.access(path, os.W_OK):
            return path
    raise RuntimeError(
        "no writable RETB task-local root; set RETB_STREAM_ROOT or SLURM_TMPDIR"
    )


def _fusion_rows(
    registry: Mapping[str, Any], *, shape_id: str, seed: int
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for section in ("canonical_fusion_rows", "uniform_control_rows"):
        for row in registry[section]:
            config = row["configuration"]
            run_id = str(row["run_id"])
            if (
                str(config["shape_id"]) == shape_id
                and int(row["seed"]) == seed
                and run_id not in seen
            ):
                seen.add(run_id)
                rows.append(row)
    if not rows:
        raise ValueError("streamed fusion coordinate has no registered runs")
    return rows


def _expert_parents(
    root: Path,
    registry: Mapping[str, Any],
    *,
    shape_id: str,
    seed: int,
) -> list[tuple[str, Path, Path]]:
    by_expert = {}
    for row in registry["expert_confirmation_rows"]:
        config = row["configuration"]
        if str(config["shape_id"]) == shape_id and int(row["seed"]) == seed:
            by_expert[str(config["expert_id"])] = str(row["run_id"])
    if set(by_expert) != set(registry["expert_order"]):
        raise ValueError("streamed fusion expert parent coverage differs")
    output = []
    for expert in registry["expert_order"]:
        run_id = by_expert[str(expert)]
        parent = (
            root
            / "runs"
            / "stage_c"
            / "offline_experts"
            / run_id
            / f"seed_{seed}"
        )
        registration = parent / "checkpoint_registration.json"
        checkpoint = parent / "best_model_val.pt"
        if not registration.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"streamed fusion parent is absent: {parent}")
        output.append((str(expert), registration, checkpoint))
    return output


def _expected_run_outputs(root: Path, row: Mapping[str, Any]) -> list[Path]:
    run_root = root / "runs" / "stage_c" / str(row["run_id"])
    variant = str(row["configuration"]["fusion_variant"])
    if variant in {"F_BEST_SINGLE", "F_UNIFORM_LOGIT_MEAN"}:
        paths = [
            run_root / "val_stop_parameter_free_evaluation.json",
            run_root / "val_design_parameter_free_evaluation.json",
        ]
        if variant == "F_BEST_SINGLE":
            paths.append(run_root / "best_single_selection.json")
        return paths
    return [
        run_root / "fusion_registration.json",
        run_root / "best_model_val.pt",
        run_root / "val_design_inference.json",
        run_root / "val_design_predictions.npz",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    profile = load_hashed_json(
        root / "registry" / "retb_streamed_abc_execution_profile.json"
    )
    validate_streamed_abc_execution_profile(profile)
    if profile["campaign_id"] != campaign["campaign_id"]:
        raise ValueError("streamed execution profile belongs to another campaign")
    registry = load_hashed_json(root / "registry" / "retb_stage_c_runs.json")
    validate_stage_c_run_registry(registry)
    if args.shape_id not in registry["uniform_shape_order"]:
        raise ValueError("streamed fusion shape is not registered")
    if args.pipeline_seed not in registry["pipeline_seeds"]:
        raise ValueError("streamed fusion seed is not registered")
    rows = _fusion_rows(
        registry, shape_id=args.shape_id, seed=args.pipeline_seed
    )
    result = {
        "dry_run": bool(args.dry_run),
        "shape_id": args.shape_id,
        "pipeline_seed": args.pipeline_seed,
        "run_ids": [str(row["run_id"]) for row in rows],
        "task_local_root": str(_base_tmp()),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parents = _expert_parents(
        root,
        registry,
        shape_id=args.shape_id,
        seed=args.pipeline_seed,
    )
    wave_root = Path(
        tempfile.mkdtemp(
            prefix=f"retb-{args.shape_id}-{args.pipeline_seed}-",
            dir=_base_tmp(),
        )
    )
    cache_records: dict[str, dict[str, Any]] = {}
    try:
        cache_manifests: dict[str, Path] = {}
        for split in ("model_train", "val_stop", "val_design"):
            output_dir = wave_root / split
            command = [
                sys.executable,
                "scripts/build_retb_frozen_token_cache.py",
                "--campaign-root",
                str(root),
                "--split",
                split,
                "--pipeline-seed",
                str(args.pipeline_seed),
                "--shape-id",
                args.shape_id,
                "--input-npz",
                str(root / "inputs" / "offline" / split / "offline_inputs.npz"),
                "--input-manifest",
                str(
                    root
                    / "inputs"
                    / "offline"
                    / split
                    / "offline_input_manifest.json"
                ),
                "--relation-normalization",
                str(root / "inputs" / "normalization" / "offline_500k" / "relation.json"),
                "--region-normalization",
                str(root / "inputs" / "normalization" / "offline_500k" / "region.json"),
                "--region-tree-root",
                str(root / "inputs" / "region_tree" / "offline"),
                "--output-dir",
                str(output_dir),
            ]
            for expert, registration, checkpoint in parents:
                command.extend(
                    [
                        "--expert-registration",
                        f"{expert}={registration}",
                        "--expert-checkpoint",
                        f"{expert}={checkpoint}",
                    ]
                )
            _run(command)
            manifest = output_dir / f"{split}_frozen_tokens.json"
            cache_manifests[split] = manifest
            metadata = load_hashed_json(manifest)
            npz = output_dir / str(metadata["npz_filename"])
            cache_records[split] = {
                "manifest_sha256": metadata["content_hash"],
                "npz_sha256": _sha256(npz),
                "npz_bytes": npz.stat().st_size,
            }

        for row in rows:
            run_id = str(row["run_id"])
            variant = str(row["configuration"]["fusion_variant"])
            output_dir = root / "runs" / "stage_c" / run_id
            if variant in {"F_BEST_SINGLE", "F_UNIFORM_LOGIT_MEAN"}:
                command = [
                    sys.executable,
                    "scripts/evaluate_retb_offline_fusion_control.py",
                    "--campaign-root",
                    str(root),
                    "--run-id",
                    run_id,
                    "--cache",
                    str(cache_manifests["val_stop"]),
                    "--val-stop-cache",
                    str(cache_manifests["val_stop"]),
                    "--val-design-cache",
                    str(cache_manifests["val_design"]),
                    "--output-dir",
                    str(output_dir),
                ]
            else:
                command = [
                    sys.executable,
                    "scripts/train_retb_offline_fusion.py",
                    "--campaign-root",
                    str(root),
                    "--run-id",
                    run_id,
                    "--model-train-cache",
                    str(cache_manifests["model_train"]),
                    "--val-stop-cache",
                    str(cache_manifests["val_stop"]),
                    "--val-design-cache",
                    str(cache_manifests["val_design"]),
                    "--output-dir",
                    str(output_dir),
                ]
            _run(command)
            resume = output_dir / "resume_state.pt"
            if resume.is_file():
                resume.unlink()

        output_hashes = {}
        for row in rows:
            for path in _expected_run_outputs(root, row):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"streamed fusion output is absent: {path}"
                    )
                output_hashes[str(path.relative_to(root))] = _sha256(path)
        shutil.rmtree(wave_root)
        if wave_root.exists():
            raise RuntimeError("task-local frozen-token cache cleanup failed")
        receipt = with_content_hash(
            {
                "contract": STREAMED_ABC_FUSION_RECEIPT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign["content_hash"],
                "execution_profile_sha256": profile["content_hash"],
                "stage_c_registry_sha256": registry["content_hash"],
                "shape_id": args.shape_id,
                "pipeline_seed": args.pipeline_seed,
                "run_ids": [str(row["run_id"]) for row in rows],
                "ephemeral_cache_records": cache_records,
                "persistent_output_hashes": output_hashes,
                "ephemeral_cache_deleted_before_worker_exit": True,
                "source": campaign["source"],
            }
        )
        publication = write_immutable_json(args.receipt, receipt)
        result.update(
            {
                "receipt_sha256": receipt["content_hash"],
                "publication": publication,
                "persistent_output_count": len(output_hashes),
            }
        )
    finally:
        shutil.rmtree(wave_root, ignore_errors=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
