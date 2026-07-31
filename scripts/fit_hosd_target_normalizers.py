#!/usr/bin/env python3
"""Fit immutable HOSD train-only target and HET normalization artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_heteroscedastic_metadata,
    fit_target_normalizer,
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_CACHE_SPEC_CONTRACT,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fitting-population",
        choices=("target_500k", "target_scale"),
        required=True,
    )
    parser.add_argument(
        "--normalization-role", choices=("target", "residual"), default="target"
    )
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    spec = load_hashed_json(
        args.cache_dir / "cache_spec.json",
        expected_contract=TARGET_CACHE_SPEC_CONTRACT,
    )
    cache = load_target_cache(args.cache_dir, cache_spec=spec)
    if cache.manifest.get("source") != campaign["source"]:
        raise ValueError("target cache source differs from active campaign")
    registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    registry_rows = {row["target_id"]: row for row in registry["targets"]}
    component_kinds = {
        target_id: tuple(
            component["component_kind"]
            for component in registry_rows[target_id]["components"]
        )
        for target_id in cache.manifest["persisted_target_ids"]
        if target_id in registry_rows
    }
    normalizer = fit_target_normalizer(
        cache,
        fitting_population=args.fitting_population,
        source=campaign["source"],
        component_kinds=component_kinds,
        normalization_role=args.normalization_role,
    )
    hetero = build_heteroscedastic_metadata(normalizer, source=campaign["source"])
    write_immutable_json(args.output, normalizer)
    hetero_path = args.output.with_name("heteroscedastic_metadata.json")
    write_immutable_json(hetero_path, hetero)
    print(
        json.dumps(
            {
                "normalizer_sha256": normalizer["content_hash"],
                "heteroscedastic_metadata_sha256": hetero["content_hash"],
                "fit_split": normalizer["fit_split"],
                "identity_values_stored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
