#!/usr/bin/env python3
"""Publish exact-replica residual or immutable target-control HOSD caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_control_batches,
    build_target_cache_spec,
    build_target_control_manifest,
    fit_target_normalizer,
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
    publish_target_cache,
    residual_batches,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_CACHE_SPEC_CONTRACT,
    TARGET_NORMALIZER_CONTRACT,
    TARGET_SHUFFLE_PLAN_CONTRACT,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_b_runtime import (  # noqa: E402
    try_finalize_stage_b_wave,
)


def _load_cache(path: Path):
    spec = load_hashed_json(
        path / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
    )
    return spec, load_target_cache(path, cache_spec=spec)


def _slice_batches(batches, indices: np.ndarray):
    output = {}
    for target_id, batch in batches.items():
        output[target_id] = type(batch)(
            target_id=batch.target_id,
            component_names=batch.component_names,
            availability_groups=batch.availability_groups,
            values=batch.values[indices],
            loss_mask=batch.loss_mask[indices],
            diagnostics=batch.diagnostics,
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--canonical-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("residual", "target_mean", "global_shuffle", "within_class_shuffle"),
    )
    parser.add_argument("--hlt-cache", type=Path)
    parser.add_argument("--target-pairs", type=Path)
    parser.add_argument("--auto-target-pairs", action="store_true")
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--shuffle-plan-dir", type=Path)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument("--shard-size", type=int, default=2048)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    target_registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    if target_registry.get("source") != campaign["source"]:
        raise ValueError("target registry source differs from active campaign")
    _, canonical = _load_cache(args.canonical_cache)
    if canonical.manifest.get("source") != campaign["source"]:
        raise ValueError("canonical target cache source differs from active campaign")
    if args.kind == "residual":
        if (
            args.hlt_cache is None
            or (args.target_pairs is None) == (not args.auto_target_pairs)
        ):
            raise ValueError(
                "residual requires HLT cache and exactly one target-pair source"
            )
        _, hlt = _load_cache(args.hlt_cache)
        if hlt.manifest.get("source") != campaign["source"]:
            raise ValueError("HLT target cache source differs from active campaign")
        target_pairs = (
            {
                target_id: target_id
                for target_id in sorted(
                    set(canonical.values) & set(hlt.values)
                )
            }
            if args.auto_target_pairs
            else json.loads(args.target_pairs.read_text(encoding="utf-8"))
        )
        if not isinstance(target_pairs, dict):
            raise ValueError("target-pairs must be a JSON object")
        if not target_pairs:
            raise ValueError("residual target-pair intersection is empty")
        output_ids = {
            offline_id: f"{offline_id}__RES__{hlt.manifest['hlt_replica_id']}"
            for offline_id in target_pairs
        }
        batches = residual_batches(
            canonical,
            hlt,
            target_pairs=target_pairs,
            output_target_ids=output_ids,
        )
        parent_hashes = {
            "campaign_spec": campaign["content_hash"],
            "offline_cache": canonical.manifest["content_hash"],
            "hlt_cache": hlt.manifest["content_hash"],
            "target_registry": target_registry["content_hash"],
        }
        artifact_kind = "residual"
        replica_id = hlt.manifest["hlt_replica_id"]
        normalizer_hash = None
        plan_hashes = {}
    else:
        if args.normalizer is None:
            raise ValueError("control cache requires --normalizer")
        normalizer = load_hashed_json(
            args.normalizer, expected_contract=TARGET_NORMALIZER_CONTRACT
        )
        if normalizer.get("source") != campaign["source"]:
            raise ValueError("target normalizer source differs from active campaign")
        plans = {}
        if args.kind != "target_mean":
            if args.shuffle_plan_dir is None:
                raise ValueError("shuffle control requires --shuffle-plan-dir")
            for target_id in canonical.manifest["persisted_target_ids"]:
                plans[target_id] = load_hashed_json(
                    args.shuffle_plan_dir / f"{target_id}.json",
                    expected_contract=TARGET_SHUFFLE_PLAN_CONTRACT,
                )
                if plans[target_id]["shuffle_kind"] != args.kind.removesuffix("_shuffle"):
                    raise ValueError("shuffle plan kind differs from requested control")
                if plans[target_id]["canonical_cache_manifest_sha256"] != canonical.manifest[
                    "content_hash"
                ]:
                    raise ValueError("shuffle plan is bound to a different canonical cache")
        batches = build_control_batches(
            canonical,
            normalizer=normalizer,
            control_kind=args.kind,
            shuffle_plans=plans,
        )
        parent_hashes = {
            "campaign_spec": campaign["content_hash"],
            "canonical_cache": canonical.manifest["content_hash"],
            "target_normalizer": normalizer["content_hash"],
            "target_registry": target_registry["content_hash"],
            **{
                f"shuffle_plan_{target_id}": plan["content_hash"]
                for target_id, plan in sorted(plans.items())
            },
        }
        artifact_kind = "control"
        replica_id = None
        normalizer_hash = normalizer["content_hash"]
        plan_hashes = {
            target_id: plan["content_hash"] for target_id, plan in plans.items()
        }
    components = {
        target_id: tuple(batch.component_names)
        for target_id, batch in batches.items()
    }
    spec = build_target_cache_spec(
        cache_id=args.cache_id,
        split=canonical.manifest["split"],
        artifact_kind=artifact_kind,
        identities=canonical.identities,
        target_components=components,
        parent_hashes=parent_hashes,
        source=campaign["source"],
        shard_size=args.shard_size,
        hlt_replica_id=replica_id,
    )
    manifest = publish_target_cache(
        args.output_dir,
        cache_spec=spec,
        identities=canonical.identities,
        generator=lambda indices: _slice_batches(batches, indices),
    )
    if args.kind != "residual":
        control = build_target_control_manifest(
            control_kind=args.kind,
            canonical_cache_manifest_sha256=canonical.manifest["content_hash"],
            control_cache_manifest_sha256=manifest["content_hash"],
            target_normalizer_sha256=normalizer_hash,
            shuffle_plan_hashes=plan_hashes,
            source=campaign["source"],
        )
        write_immutable_json(args.output_dir / "control_manifest.json", control)
    wave = (
        try_finalize_stage_b_wave(
            campaign_root=args.campaign_root,
            wave_kind="residual",
            target_registry=target_registry,
            source=campaign["source"],
        )
        if args.kind == "residual"
        else None
    )
    print(
        json.dumps(
            {
                "kind": args.kind,
                "cache_manifest_sha256": manifest["content_hash"],
                "canonical_cache_mutated": False,
                "wave_completion_sha256": (
                    None if wave is None else wave["content_hash"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
