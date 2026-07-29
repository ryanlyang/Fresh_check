#!/usr/bin/env python3
"""Build HLT-v3 from an authenticated RETB offline-input cache."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import load_hashed_json  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--offline-input-manifest", required=True, type=Path)
    parser.add_argument("--logical-role", required=True)
    parser.add_argument("--replica-id", required=True, type=int)
    parser.add_argument(
        "--realization-policy",
        choices=("R_FIXED", "R_MULTI", "R_RANDOM"),
        required=True,
    )
    parser.add_argument("--profile-id", default="D_NOMINAL")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    metadata = load_hashed_json(
        args.offline_input_manifest,
        expected_contract="retb_offline_input_cache_v1",
    )
    if (
        metadata["campaign_spec_sha256"] != campaign["content_hash"]
        or metadata["logical_role"] != args.logical_role
    ):
        raise ValueError("offline-input cache lineage differs")
    npz_path = args.offline_input_manifest.parent / metadata["npz_filename"]
    if not npz_path.is_file() or _sha256(npz_path) != metadata["npz_sha256"]:
        raise ValueError("offline-input cache bytes differ")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_retb_hlt_v3_cache.py"),
        "--campaign-root",
        str(args.campaign_root),
        "--offline-npz",
        str(npz_path),
        "--logical-role",
        args.logical_role,
        "--replica-id",
        str(args.replica_id),
        "--realization-policy",
        args.realization_policy,
        "--profile-id",
        args.profile_id,
        "--identity-manifest-sha256",
        metadata["identity_manifest_sha256"],
        "--raw-input-sha256",
        metadata["npz_sha256"],
        "--output-dir",
        str(args.output_dir),
    ]
    return int(subprocess.run(command, cwd=REPO_ROOT, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
