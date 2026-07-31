#!/usr/bin/env python3
"""Retrain the locked O_BASE/O_FULLREL teachers on scale_train."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_offline_capacity_controls import (  # noqa: E402
    BASE_CONFIG,
    FAMILIES,
    _lineage,
    _train,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    OfflineClassifierAdapter,
    analytical_particle_transformer_flops,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.capacity import pair_encoder_flops  # noqa: E402
from teacher_logit_reco.relational_part.model import (  # noqa: E402
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
)

import torch  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.pipeline_seed not in {101, 202, 303}:
        raise ValueError("scale teacher seed differs")
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    for role, worker in (
        ("scale_train", "scale_training_worker"),
        ("val_stop", "scale_training_worker"),
        ("val_design", "design_worker"),
    ):
        authorize_dataset_access(worker_role=worker, requested_resource=role)
    source = source_snapshot(REPO_ROOT)
    execution = load_hashed_json(
        root / "registry" / "retb_offline_capacity_execution.json"
    )
    relation = load_hashed_json(
        root / "inputs" / "normalization" / "offline_scale" / "relation.json"
    )
    region = load_hashed_json(
        root / "inputs" / "normalization" / "offline_scale" / "region.json"
    )
    lineage = _lineage(
        root,
        campaign,
        training_role="scale_train",
        normalization_population="scale",
    )
    lineage["scale_train_manifest"] = campaign["parent_artifact_hashes"][
        "scale_train_manifest"
    ]
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    models = {
        "O_BASE": OfflineClassifierAdapter(
            RelationalParticleTransformer(weaver_module=weaver)
        ),
        "O_FULLREL": OfflineClassifierAdapter(
            RelationalFamilyParticleTransformer(
                FAMILIES,
                normalization_artifact=relation,
                region_normalization_artifact=region,
                weaver_module=weaver,
            )
        ),
    }
    base = analytical_particle_transformer_flops(configuration=BASE_CONFIG)
    flops = {
        "O_BASE": base + pair_encoder_flops(4, (64, 64, 64)),
        "O_FULLREL": base + pair_encoder_flops(62, (64, 64, 64)),
    }
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    rows = {}
    for teacher_id, model in models.items():
        training, profile = _train(
            root=root,
            campaign=campaign,
            control_id=teacher_id,
            model=model,
            flops=flops[teacher_id],
            execution_sha=execution["content_hash"],
            lineage=lineage,
            source=source,
            batch_size=args.batch_size,
            device=device,
            seed=args.pipeline_seed,
            training_role="scale_train",
            output_root=args.output_dir,
        )
        rows[teacher_id] = {
            "training_registration_sha256": training["content_hash"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "checkpoint_path": str(
                (
                    args.output_dir
                    / teacher_id
                    / "training"
                    / "best_model_val.pt"
                ).resolve()
            ),
            "capacity_profile_sha256": profile["content_hash"],
        }
    bundle = bind_source(
        with_content_hash(
            {
                "contract": "retb_scale_teacher_bundle_v1",
                "schema_version": 1,
                "pipeline_seed": args.pipeline_seed,
                "training_population": "scale_train",
                "scale_train_manifest_sha256": campaign[
                    "parent_artifact_hashes"
                ]["scale_train_manifest"],
                "teachers": rows,
                "architecture_reselection_performed": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source,
    )
    publication = write_immutable_json(
        args.output_dir / "scale_teacher_bundle.json", bundle
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
