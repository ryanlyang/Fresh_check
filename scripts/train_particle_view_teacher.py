#!/usr/bin/env python3
"""Train one locked Step-2 A0_view or Toff_view teacher."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import (  # noqa: E402
    ParticleViewTorchDataset,
    make_data_loader,
    require_torch,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from jetclass_fresh.jetclass_data import JetView, load_split_manifest  # noqa: E402
from teacher_logit_reco.architecture_view_part import (  # noqa: E402
    load_cached_offline_view,
)
from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
    ParticleViewTeacherTrainConfig,
    audit_unified_split_manifest,
    build_teacher_recipe,
    canonical_sha256,
    load_hashed_json,
    logical_split_identities,
    train_particle_view_teacher,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--role", choices=("A0_view", "Toff_view"), required=True)
    result.add_argument("--architecture", choices=("base", "large"), default="base")
    result.add_argument("--seed", choices=(101, 202, 303), type=int, required=True)
    result.add_argument("--unified-manifest", required=True)
    result.add_argument("--parent-manifest", required=True)
    result.add_argument("--hlt-cache-dir")
    result.add_argument("--offline-cache-dir")
    result.add_argument("--preprocessing-sha256", required=True)
    result.add_argument("--source-sha256", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="auto")
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--max-train-batches", type=int)
    result.add_argument("--max-val-batches", type=int)
    result.add_argument("--no-amp", action="store_true")
    return result


def subset(view: JetView, indices: list[int], name: str) -> JetView:
    return JetView(
        tokens=np.ascontiguousarray(view.tokens[indices], dtype=np.float32),
        mask=np.ascontiguousarray(view.mask[indices], dtype=bool),
        labels=np.ascontiguousarray(view.labels[indices], dtype=np.int64),
        jet_ids=[view.jet_ids[index] for index in indices],
        split=name,
        metadata={**view.metadata, "logical_split": name},
    )


def require_identities(view: JetView, expected, name: str) -> None:
    if [row.to_dict() for row in view.jet_ids] != [
        row.to_dict() for row in expected
    ]:
        raise ValueError(f"{name} cache identities/order differ from manifest")


def versions_hash() -> str:
    torch = require_torch()
    try:
        weaver = importlib.metadata.version("weaver-core")
    except importlib.metadata.PackageNotFoundError:
        weaver = "unknown"
    return canonical_sha256(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "weaver_core": weaver,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.role == "A0_view":
        if not args.hlt_cache_dir or args.offline_cache_dir:
            raise ValueError("A0_view requires only --hlt-cache-dir")
        if args.architecture != "base":
            raise ValueError("A0_view architecture is locked to base")
        load = lambda split: load_cached_hlt_view(args.hlt_cache_dir, split)
        source_view = "fixed_hlt"
    else:
        if not args.offline_cache_dir or args.hlt_cache_dir:
            raise ValueError("Toff_view requires only --offline-cache-dir")
        load = lambda split: load_cached_offline_view(
            args.offline_cache_dir, split, verify_hash=True
        )
        source_view = "offline"
    unified = load_hashed_json(args.unified_manifest)
    parent = load_split_manifest(args.parent_manifest)
    audit_unified_split_manifest(
        unified,
        parent=parent,
        config=LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
    )
    train = load("model_train")
    model_val = load("model_val")
    expected_train = logical_split_identities(
        unified,
        parent=parent,
        split_name="train",
        config=LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
    )
    expected_stop = logical_split_identities(
        unified,
        parent=parent,
        split_name="model_val_stop",
        config=LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
    )
    require_identities(train, expected_train, "train")
    index = {identity.key(): row for row, identity in enumerate(model_val.jet_ids)}
    try:
        stop_indices = [index[identity.key()] for identity in expected_stop]
    except KeyError as exc:
        raise ValueError("model_val cache omits stop-split identities") from exc
    stop = subset(model_val, stop_indices, "model_val_stop")
    require_identities(stop, expected_stop, "model_val_stop")
    recipe = build_teacher_recipe(
        role=args.role,
        architecture=args.architecture,
        seed=args.seed,
        unified_split_manifest=unified,
        preprocessing_sha256=args.preprocessing_sha256,
        source_sha256=args.source_sha256,
        initialization_implementation_sha256=canonical_sha256(
            {
                "builder": "build_particle_transformer_classifier",
                "constructor": "weaver.nn.model.ParticleTransformer",
            }
        ),
        library_versions_sha256=versions_hash(),
    )
    physical_batch = recipe.to_payload()["physical_batch_size"]
    train_loader = make_data_loader(
        ParticleViewTorchDataset(train, expected_view=source_view),
        batch_size=physical_batch,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        source_view=source_view,
    )
    stop_loader = make_data_loader(
        ParticleViewTorchDataset(stop, expected_view=source_view),
        batch_size=physical_batch,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        source_view=source_view,
    )
    report = train_particle_view_teacher(
        recipe=recipe,
        train_loader=train_loader,
        model_val_stop_loader=stop_loader,
        config=ParticleViewTeacherTrainConfig(
            output_dir=args.output_dir,
            device=args.device,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            amp=not args.no_amp,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
