#!/usr/bin/env python3
"""Profile Step-5 predictor architectures on the locked 128-particle fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    PARTICLE_VIEW_PREDICTOR_ARCHITECTURES,
    PVA3_CANONICAL_ARCHITECTURE,
    HierarchicalParticleViewPredictor,
    build_predictor_architecture_config,
    profile_predictor_resources,
    validate_predictor_resource_profile,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture-id",
        action="append",
        choices=PARTICLE_VIEW_PREDICTOR_ARCHITECTURES,
        help="May be repeated; defaults to the canonical PVA3.",
    )
    parser.add_argument("--view-dim", type=int, required=True)
    parser.add_argument("--input-dim", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--campaign-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101, choices=(101, 202, 303))
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    return result


def _shared_stem(input_dim: int, width: int) -> nn.Module:
    # The shared object is passed by identity, allowing exported-bundle
    # parameter accounting to deduplicate it when the consumer is present.
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.LayerNorm(width),
        nn.GELU(),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = _device(args.device)
    architectures = args.architecture_id or [PVA3_CANONICAL_ARCHITECTURE]
    output_dir = Path(args.output_dir)
    for architecture_id in architectures:
        torch.manual_seed(args.seed)
        config = build_predictor_architecture_config(
            architecture_id,
            view_dim=args.view_dim,
            input_dim=args.input_dim,
        )
        shared = (
            _shared_stem(config.input_dim, config.width)
            if config.shared_consumer_stem
            else None
        )
        model = HierarchicalParticleViewPredictor(
            config, shared_particle_embedding=shared
        ).to(device)
        profile = profile_predictor_resources(
            model,
            warmup=args.warmup,
            repetitions=args.repetitions,
            campaign_batch_size=args.campaign_batch_size,
        )
        validate_predictor_resource_profile(
            profile, expected_config_sha256=config.content_hash
        )
        write_immutable_json(
            output_dir / f"{architecture_id}.resource.json", profile
        )
        print(
            f"{architecture_id}: params={profile['total_parameters']} "
            f"flops={profile['forward_flops']['exact_integer_total']} "
            f"output={output_dir / f'{architecture_id}.resource.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

