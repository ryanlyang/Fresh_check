#!/usr/bin/env python3
"""Build one authenticated val-design HLT-v3 degradation cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    INPUT_VIEW_MANIFEST_CONTRACT,
    PARENT_STATUS_CONTRACT,
    load_and_validate_campaign,
    load_hashed_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.metrics import (  # noqa: E402
    DEGRADATION_PROFILES,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    build_hlt_v3_cache,
    build_hlt_v3_cache_metadata,
    publish_hlt_v3_cache,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _try_finalize(root: Path, campaign) -> dict | None:
    hashes = {}
    for profile in DEGRADATION_PROFILES:
        for replica in range(4):
            path = (
                root
                / "robustness"
                / "caches"
                / profile
                / f"replica_{replica}"
                / "hlt_v3_metadata.json"
            )
            if not path.is_file():
                return None
            artifact = load_hashed_json(
                path, expected_contract="retb_hlt_v3_cache_v1"
            )
            if (
                artifact.get("degradation_profile_id") != profile
                or int(artifact.get("replica_id", -1)) != replica
            ):
                raise ValueError("robustness cache coordinate differs")
            hashes[f"{profile}::replica_{replica}"] = artifact[
                "content_hash"
            ]
    completion = with_content_hash(
        {
            "contract": "hosd_robustness_cache_wave_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "artifact_hashes": dict(sorted(hashes.items())),
            "profiles": list(DEGRADATION_PROFILES),
            "replicas": [0, 1, 2, 3],
            "artifact_count": len(hashes),
            "label_blind": True,
            "all_profiles_complete": True,
        }
    )
    write_immutable_json(
        root / "robustness" / "cache_completion.json", completion
    )
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--profile", required=True, choices=DEGRADATION_PROFILES
    )
    parser.add_argument("--replica", required=True, type=int)
    parser.add_argument("--offline-input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.replica not in range(4):
        raise ValueError("robustness replica lies outside 0..3")
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    with np.load(args.offline_input, allow_pickle=False) as payload:
        identity_key = (
            "identity" if "identity" in payload.files else "identities"
        )
        if (
            {"label", "labels", "class", "classes", "y"}
            & set(payload.files)
            or not {identity_key, "raw_tokens", "mask"}.issubset(
                payload.files
            )
        ):
            raise ValueError("robustness offline input is not label blind")
        identities = tuple(
            str(value) for value in payload[identity_key].tolist()
        )
        tokens = np.asarray(payload["raw_tokens"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
    arrays, diagnostics = build_hlt_v3_cache(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="val_design",
        replica_id=args.replica,
        realization_policy="R_FIXED",
        profile_id=args.profile,
    )
    profile_path = root / "inputs" / "hlt_v3_profile.json"
    if not profile_path.is_file():
        parent_lock = load_hashed_json(
            root / "inputs" / "resolved_inherited_parent_lock.json",
            expected_contract=PARENT_STATUS_CONTRACT,
        )
        matches = [
            row
            for row in parent_lock["requirements"]
            if row["parent_id"] == "hlt_v3_profile"
        ]
        if (
            len(matches) != 1
            or not matches[0].get("reusable")
            or not matches[0].get("resolved_path")
        ):
            raise ValueError("resolved HLT-v3 profile parent is unavailable")
        profile_path = Path(matches[0]["resolved_path"])
    profile = load_hashed_json(
        profile_path,
        expected_contract="retb_hlt_v3_profile_v1",
    )
    offline_manifest = load_hashed_json(
        args.offline_input.with_suffix(args.offline_input.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    metadata = build_hlt_v3_cache_metadata(
        arrays=arrays,
        diagnostics=diagnostics,
        logical_role="val_design",
        replica_id=args.replica,
        realization_policy="R_FIXED",
        degradation_profile_id=args.profile,
        profile_contract=profile,
        split_manifest_sha256=offline_manifest["parent_hashes"][
            "split_manifest"
        ],
        identity_manifest_sha256=offline_manifest[
            "identity_order_sha256"
        ],
        raw_input_sha256=_sha(args.offline_input),
    )
    # RETB cache contracts are source-independent numerical contracts; bind
    # the active campaign through the wave completion and input manifest.
    output = args.output_dir or (
        root
        / "robustness"
        / "caches"
        / args.profile
        / f"replica_{args.replica}"
    )
    publication = publish_hlt_v3_cache(
        output, arrays=arrays, metadata=metadata
    )
    completion = _try_finalize(root, campaign)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "replica": args.replica,
                "publication": publication,
                "wave_complete": completion is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
