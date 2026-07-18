#!/usr/bin/env python3
"""Measure real ABPH targets and freeze the shared-vs-rank-local mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline.ram_workspace import (  # noqa: E402
    RankLocalWorkspace,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.accounting_preflight import (  # noqa: E402
    ABPH_STEP4_PREFLIGHT_CONTRACT,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (  # noqa: E402
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    require_storage_projection,
    write_quota_managed_json,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.target_mode import (  # noqa: E402
    campaign_and_hlt_bytes,
    measure_real_target_sample,
    select_target_mode,
    write_target_mode_selection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--campaign-mode", choices=("pilot", "highdata"), required=True)
    parser.add_argument("--storage-profile", required=True)
    parser.add_argument("--storage-projection", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--jets-per-class", type=int, default=64)
    parser.add_argument("--target-chunk-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--measurement-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.campaign_root).resolve()
    workspace = RankLocalWorkspace.from_environment()
    projection = require_storage_projection(
        args.storage_projection,
        campaign_root=root,
        campaign_mode=args.campaign_mode,
        profile=args.storage_profile,
    )
    measurement = measure_real_target_sample(
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        data_dir=args.data_dir,
        jets_per_class=args.jets_per_class,
        workspace=workspace,
    )
    manifest = load_split_manifest(args.manifest)
    source_provenance_by_split = {}
    for split in ("model_train", "model_val"):
        hlt_metadata = json.loads(
            (Path(args.hlt_cache_dir) / f"{split}_fixed_hlt_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        offline_metadata = json.loads(
            (Path(args.offline_cache_dir) / f"{split}_offline_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if hlt_metadata.get("source_manifest_hash") != measurement["manifest_hash"]:
            raise ValueError(f"{split} HLT cache is not bound to the active manifest")
        if offline_metadata.get("source_manifest_hash") != measurement["manifest_hash"]:
            raise ValueError(f"{split} offline cache is not bound to the active manifest")
        if hlt_metadata.get("jet_identity_hash") != offline_metadata.get(
            "jet_identity_hash"
        ):
            raise ValueError(f"{split} HLT/offline identities differ")
        source_provenance_by_split[split] = {
            "source_manifest_hash": hlt_metadata.get("source_manifest_hash"),
            "hlt_content_hash": hlt_metadata.get("hlt_content_hash"),
            "offline_content_hash": offline_metadata.get("offline_content_hash"),
            "jet_identity_hash": hlt_metadata.get("jet_identity_hash"),
        }
    current_bytes, hlt_bytes = campaign_and_hlt_bytes(root, args.hlt_cache_dir)
    selection = select_target_mode(
        campaign_root=root,
        campaign_mode=args.campaign_mode,
        split_sizes={
            "model_train": len(manifest.splits["model_train"]),
            "model_val": len(manifest.splits["model_val"]),
        },
        measurement=measurement,
        storage_projection=projection,
        workspace_capacity=workspace.probe.to_dict(),
        hlt_cache_bytes=hlt_bytes,
        current_campaign_bytes=current_bytes,
        target_chunk_size=args.target_chunk_size,
        source_provenance_by_split=source_provenance_by_split,
    )
    if args.storage_profile == ABPH_STREAMING_STORAGE_PROFILE:
        write_quota_managed_json(
            root,
            args.output,
            selection,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role="target_mode_selection",
            source_provenance_hash=selection["content_hash"],
            run_id="target_mode_preflight",
            profile=args.storage_profile,
        )
    else:
        write_target_mode_selection(args.output, selection)
    if args.measurement_output:
        measurement_path = Path(args.measurement_output)
        measurement_path.parent.mkdir(parents=True, exist_ok=True)
        write_quota_managed_json(
            root,
            measurement_path,
            measurement,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role="target_sample_measurement",
            source_provenance_hash=measurement["content_hash"],
            run_id="target_mode_preflight",
            profile=args.storage_profile,
        )
    feasibility_rows = {}
    feasibility_problems: list[str] = []
    for name, row in measurement["feasibility_reports"].items():
        problems = list(row.get("problems", ()))
        feasibility_problems.extend(f"{name}: {problem}" for problem in problems)
        feasibility_rows[name] = {
            **dict(row),
            "ok": not problems and int(row.get("compiler_failure_count", -1)) == 0,
        }
    synthetic = measurement["synthetic_edge_cases"]
    if not synthetic.get("ok"):
        feasibility_problems.append("synthetic edge-case matrix failed")
    feasibility = {
        "contract": ABPH_STEP4_PREFLIGHT_CONTRACT,
        "ok": not feasibility_problems,
        "target_mode": selection["selected_mode"],
        "target_mode_selection_hash": selection["content_hash"],
        "max_jets_per_class": int(args.jets_per_class),
        "reports": feasibility_rows,
        "synthetic_edge_cases": synthetic,
        "problems": feasibility_problems,
    }
    feasibility_path = root / "audits" / "actual_target_feasibility.json"
    write_quota_managed_json(
        root,
        feasibility_path,
        feasibility,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="actual_target_feasibility",
        source_provenance_hash=selection["content_hash"],
        run_id="target_mode_preflight",
        profile=args.storage_profile,
    )
    if feasibility_problems:
        raise RuntimeError("real target-mode feasibility preflight failed")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
