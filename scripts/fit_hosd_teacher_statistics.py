#!/usr/bin/env python3
"""Fit HOSD train-only latent whitening or conditional-residual statistics."""

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
    fit_conditional_residual,
    fit_latent_whitening,
    build_hlt_conditional_context,
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_CACHE_SPEC_CONTRACT,
    TEACHER_LOCK_CONTRACT,
    write_immutable_json,
)


def _cache(path: Path):
    spec = load_hashed_json(
        path / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
    )
    return load_target_cache(path, cache_spec=spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("whitening", "conditional_residual"))
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-lock", type=Path)
    parser.add_argument("--hlt-context-npz", type=Path)
    parser.add_argument("--hlt-input-npz", type=Path)
    parser.add_argument("--relation-normalizer", type=Path)
    parser.add_argument(
        "--fitting-population", choices=("target_500k", "target_scale"), required=True
    )
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    cache = _cache(args.cache_dir)
    if cache.manifest.get("source") != campaign["source"]:
        raise ValueError("statistics cache source differs from active campaign")
    expected_split = (
        "model_train" if args.fitting_population == "target_500k" else "scale_train"
    )
    if cache.manifest["split"] != expected_split:
        raise ValueError("statistics cache is not the declared train population")
    if args.target_id not in cache.values:
        raise ValueError("target ID is absent from cache")
    if args.mode == "whitening":
        if args.teacher_lock is None:
            raise ValueError("whitening requires --teacher-lock")
        lock = load_hashed_json(
            args.teacher_lock, expected_contract=TEACHER_LOCK_CONTRACT
        )
        artifact = fit_latent_whitening(
            cache.values[args.target_id],
            teacher_lock_sha256=lock["content_hash"],
            fitting_population=args.fitting_population,
            source=campaign["source"],
        )
    else:
        if (args.hlt_context_npz is None) == (args.hlt_input_npz is None):
            raise ValueError(
                "conditional residual requires exactly one of --hlt-context-npz "
                "or --hlt-input-npz"
            )
        context_parent = None
        if args.hlt_context_npz is not None:
            with np.load(args.hlt_context_npz, allow_pickle=False) as archive:
                if set(archive.files) != {"identity", "context"}:
                    raise ValueError("HLT context NPZ must contain only identity and context")
                identities = tuple(str(value) for value in archive["identity"].tolist())
                context = np.asarray(archive["context"], dtype=np.float64)
            context_parent = args.hlt_context_npz
        else:
            if args.relation_normalizer is None:
                raise ValueError("raw HLT context requires --relation-normalizer")
            relation = load_hashed_json(args.relation_normalizer)
            if relation.get("source") != campaign["source"]:
                raise ValueError("relation normalizer source differs from campaign")
            with np.load(args.hlt_input_npz, allow_pickle=False) as archive:
                forbidden = {"label", "labels", "class", "classes", "y"} & set(
                    archive.files
                )
                if forbidden or not {"identity", "raw_tokens", "mask"}.issubset(
                    archive.files
                ):
                    raise ValueError("HLT context input is not label-blind reader-shaped data")
                identities = tuple(str(value) for value in archive["identity"].tolist())
                context = build_hlt_conditional_context(
                    archive["raw_tokens"],
                    archive["mask"],
                    d0_uncertainty_floor=float(
                        relation["track_uncertainty_floors"]["d0"]["floor"]
                    ),
                    dz_uncertainty_floor=float(
                        relation["track_uncertainty_floors"]["dz"]["floor"]
                    ),
                    sentinel_policy=relation["track_sentinel_policy"],
                )
            context_parent = args.hlt_input_npz
        if identities != cache.identities:
            raise ValueError("HLT context identities differ from residual cache")
        artifact = fit_conditional_residual(
            cache.values[args.target_id],
            cache.masks[args.target_id],
            context,
            target_id=args.target_id,
            train_cache_hashes={
                "residual_cache": cache.manifest["content_hash"],
                "hlt_context": __import__("hashlib").sha256(
                    context_parent.read_bytes()
                ).hexdigest(),
            },
            source=campaign["source"],
            fitting_population=args.fitting_population,
        )
    publication = write_immutable_json(args.output, artifact)
    print(json.dumps({**publication, "statistics_sha256": artifact["content_hash"], "mode": args.mode}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
