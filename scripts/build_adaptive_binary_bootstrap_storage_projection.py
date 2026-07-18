#!/usr/bin/env python3
"""Build a conservative 30 GB projection from a prepared ABPH pilot root."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.config import canonical_hash  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (  # noqa: E402
    ABPH_BASELINE_VARIANTS,
    ABPH_LOGIT_PREDICTION_MEMBERS,
    ABPH_NEURAL_TAGGER_VARIANTS,
    ABPH_RECONSTRUCTOR_VARIANTS,
    ABPH_RENDERER_VARIANTS,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (  # noqa: E402
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    StorageProjectionRow,
    build_storage_projection,
    write_storage_projection,
)


MIB = 1024**2
CHECKPOINT_FLOOR_BYTES = 256 * MIB
REPORT_AND_AUDIT_BYTES = 512 * MIB


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--campaign-mode", choices=("pilot", "highdata"), default="pilot")
    parser.add_argument("--output", required=True)
    return parser


def _tree_measurement(path: Path) -> tuple[int, int, int]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    sizes = [item.stat().st_size for item in path.rglob("*") if item.is_file()]
    if not sizes:
        raise ValueError(f"measurement tree contains no files: {path}")
    return sum(sizes), max(sizes), len(sizes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measurement_identity(root: Path, measurements: dict) -> str:
    evidence = []
    required = (
        root / "audits" / "actual_target_feasibility.json",
        root / "runs" / "A0_hlt_part" / "run_report.json",
        root / "runs" / "A0_hlt_part" / "best_model_val.pt",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    feasibility = json.loads(required[0].read_text(encoding="utf-8"))
    if feasibility.get("ok") is not True:
        raise ValueError("prepared-root actual-target feasibility did not pass")
    return canonical_hash(
        {
            "contract": "adaptive_binary_bootstrap_projection_measurement_v1",
            "prepared_root": str(root),
            "measurements": measurements,
            "evidence": evidence,
        }
    )


def _with_margin(value: int, fraction: float) -> int:
    return int(math.ceil(int(value) * float(fraction)))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = Path(args.prepared_root).resolve()
    campaign = Path(args.campaign_root).resolve()
    inputs_bytes, inputs_largest, input_files = _tree_measurement(prepared / "inputs")
    targets_bytes, targets_largest, target_files = _tree_measurement(prepared / "targets")

    checkpoint_paths = tuple((prepared / "runs").glob("*/best_model_val.pt"))
    if not checkpoint_paths:
        raise ValueError("prepared root contains no selected checkpoint measurement")
    measured_checkpoint_max = max(path.stat().st_size for path in checkpoint_paths)
    checkpoint_unit = max(_with_margin(measured_checkpoint_max, 1.35), CHECKPOINT_FLOOR_BYTES)

    baseline_count = len(ABPH_BASELINE_VARIANTS)
    reconstructor_count = len(ABPH_RECONSTRUCTOR_VARIANTS) + len(ABPH_RENDERER_VARIANTS)
    tagger_count = len(ABPH_NEURAL_TAGGER_VARIANTS) + 2  # F0 seed2 and seed3
    prediction_count = len(ABPH_LOGIT_PREDICTION_MEMBERS)
    prediction_unit = 32 * MIB

    measurements = {
        "inputs": {"bytes": inputs_bytes, "largest_file_bytes": inputs_largest, "files": input_files},
        "targets": {"bytes": targets_bytes, "largest_file_bytes": targets_largest, "files": target_files},
        "selected_checkpoint_max_bytes": measured_checkpoint_max,
        "checkpoint_projection_unit_bytes": checkpoint_unit,
        "variant_counts": {
            "baselines": baseline_count,
            "reconstructors_renderers": reconstructor_count,
            "neural_taggers_and_extra_seeds": tagger_count,
            "prediction_members": prediction_count,
        },
    }
    provenance_hash = _measurement_identity(prepared, measurements)
    rows = (
        StorageProjectionRow(
            artifact_family="campaign_contracts_reports_and_audits",
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            expected_bytes=REPORT_AND_AUDIT_BYTES,
            active_from_wave=0,
            active_through_wave=6,
            retained=True,
            atomic_write_overhead_bytes=64 * MIB,
            measurement_source="prepared_root_plus_fixed_metadata_reserve_v1",
        ),
        StorageProjectionRow(
            artifact_family="transient_hlt_offline_inputs",
            artifact_class=StorageArtifactClass.SHARED_TRANSIENT,
            expected_bytes=_with_margin(inputs_bytes, 1.15),
            active_from_wave=1,
            active_through_wave=5,
            retained=False,
            atomic_write_overhead_bytes=inputs_largest,
            measurement_source="prepared_root_exact_tree_bytes_v1",
        ),
        StorageProjectionRow(
            artifact_family="baseline_selected_checkpoints",
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            expected_bytes=checkpoint_unit * baseline_count,
            active_from_wave=1,
            active_through_wave=6,
            retained=True,
            atomic_write_overhead_bytes=checkpoint_unit,
            measurement_source="prepared_root_max_checkpoint_with_35pct_floor_v1",
        ),
        StorageProjectionRow(
            artifact_family="shared_transient_compact_targets",
            artifact_class=StorageArtifactClass.SHARED_TRANSIENT,
            expected_bytes=_with_margin(targets_bytes, 1.10),
            active_from_wave=2,
            active_through_wave=3,
            retained=False,
            atomic_write_overhead_bytes=targets_largest,
            measurement_source="prepared_root_exact_tree_bytes_with_10pct_margin_v1",
        ),
        StorageProjectionRow(
            artifact_family="reconstructor_renderer_selected_checkpoints",
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            expected_bytes=checkpoint_unit * reconstructor_count,
            active_from_wave=3,
            active_through_wave=6,
            retained=True,
            atomic_write_overhead_bytes=checkpoint_unit,
            measurement_source="prepared_root_max_checkpoint_with_35pct_floor_v1",
        ),
        StorageProjectionRow(
            artifact_family="tagger_selected_checkpoints",
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            expected_bytes=checkpoint_unit * tagger_count,
            active_from_wave=4,
            active_through_wave=6,
            retained=True,
            atomic_write_overhead_bytes=checkpoint_unit,
            measurement_source="prepared_root_max_checkpoint_with_35pct_floor_v1",
        ),
        StorageProjectionRow(
            artifact_family="bundled_logit_predictions_and_fusion",
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            expected_bytes=prediction_unit * prediction_count,
            active_from_wave=5,
            active_through_wave=6,
            retained=True,
            atomic_write_overhead_bytes=prediction_unit,
            measurement_source="pilot_split_logit_upper_bound_v1",
        ),
    )
    projection = build_storage_projection(
        campaign_root=campaign,
        campaign_mode=args.campaign_mode,
        profile=ABPH_STREAMING_STORAGE_PROFILE,
        rows=rows,
        measurement_contract="prepared_pilot_real_artifact_bootstrap_v1",
        sample_provenance_hash=provenance_hash,
    )
    write_storage_projection(args.output, projection)
    print(json.dumps(projection, indent=2, sort_keys=True))
    return 0 if projection["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
