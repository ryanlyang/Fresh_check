#!/usr/bin/env python3
"""Initialize the prespecified offline RPT transfer campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part.contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.offline_transfer import (  # noqa: E402
    build_offline_transfer_campaign,
)
from teacher_logit_reco.relational_part.provenance import source_snapshot  # noqa: E402


PARENT_FILES = (
    "registry/global_determinism.json",
    "registry/relation_family_registry.json",
    "registry/screening_registry.json",
    "inputs/raw_input_schema.json",
    "inputs/normalization_contract.json",
    "inputs/angular_tree_resource_contract.json",
    "backend/backend_manifest.json",
)


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"required parent artifact is absent or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"campaign artifact differs: {destination}")
        return
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--parent-campaign-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    parent_root = args.parent_campaign_root.resolve()
    parent = load_hashed_json(parent_root / "campaign_spec.json")
    root.mkdir(parents=True, exist_ok=True)
    for relative in PARENT_FILES:
        _copy_file(parent_root / relative, root / relative)
    backend = load_hashed_json(root / "backend" / "backend_manifest.json")
    binary = str(backend["binary_filename"])
    _copy_file(parent_root / "backend" / binary, root / "backend" / binary)
    _copy_file(args.split_manifest.resolve(), root / "inputs" / "split_manifest.json.gz")
    campaign = build_offline_transfer_campaign(
        campaign_id=args.campaign_id,
        parent_campaign=parent,
        parent_campaign_path=parent_root / "campaign_spec.json",
        split_manifest_path=root / "inputs" / "split_manifest.json.gz",
        source=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(root / "campaign_spec.json", campaign)
    for directory in (
        "inputs/offline_cache",
        "inputs/relation_tree_cache",
        "registry/model_contracts",
        "runs",
        "selection",
        "final_test",
        "reports",
        "job_ledgers/slurm",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
