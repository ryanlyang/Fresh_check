#!/usr/bin/env python3
"""Build or publish immutable RETB Step-4 offline-expert contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step3 import (  # noqa: E402
    validate_step3_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (  # noqa: E402
    build_step4_bundle,
    publish_step4_bundle,
    validate_step4_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_step3(root: Path) -> dict[str, dict[str, object]]:
    paths = {
        "particle_tap": root / "registry" / "retb_particle_state_tap.json",
        "measurement_embedding": (
            root / "registry" / "retb_measurement_state_embedding.json"
        ),
        "layerwise_pair_bias": (
            root / "registry" / "retb_layerwise_pair_bias.json"
        ),
        "summary_tokenizer": (
            root / "registry" / "retb_summary_tokenizer.json"
        ),
        "token_only_head": (
            root / "registry" / "retb_token_only_expert_head.json"
        ),
        "token_shapes": root / "registry" / "retb_token_shapes.json",
        "expert_architecture": (
            root / "registry" / "retb_expert_architecture.json"
        ),
        "candidate_registry": root / "registry" / "retb_step3_candidates.json",
        "step3_bundle": (
            root / "registry" / "retb_step3_architecture_bundle.json"
        ),
        "step3_report": root / "reports" / "retb_step3_report.json",
    }
    return {name: load_hashed_json(path) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step3 = _load_step3(args.campaign_root)
    step3_sha = validate_step3_bundle(step3)
    bundle = build_step4_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        step3_bundle_sha256=step3_sha,
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    step4_sha = validate_step4_bundle(bundle)
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "campaign_root": str(args.campaign_root.resolve()),
        "step3_bundle_sha256": step3_sha,
        "step4_bundle_sha256": step4_sha,
        "primary_shape_screen_rows": bundle["step4_bundle"][
            "primary_shape_screen_rows"
        ],
        "artifact_hashes": bundle["step4_bundle"]["artifact_hashes"],
    }
    if not args.dry_run:
        result["publication"] = publish_step4_bundle(
            campaign_root=args.campaign_root,
            bundle=bundle,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
