#!/usr/bin/env python3
"""Evaluate all Step-12 native/reconstructed bypass controls on val_design."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    evaluate_final_consumer_bypass_controls,
    load_final_consumer_dataset,
    load_selected_final_consumer_checkpoint,
    load_final_consumer_template,
    make_final_consumer_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    FINAL_CONSUMER_RUN_CONTRACT,
    validate_materialized_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--val-design-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=FINAL_CONSUMER_RUN_CONTRACT
    )
    validate_materialized_final_consumer_run(run)
    if (
        run.get("source") != campaign.get("source")
        or run["consumer_kind"]
        not in {"HF_ADAPTER", "HF_UNRESTRICTED"}
    ):
        raise ValueError("bypass-control run lineage differs")
    _, model, heads, fusion, refiner = load_final_consumer_template(
        args.template,
        expected_run_record_sha256=run["content_hash"],
        expected_source=campaign["source"],
    )
    registration = load_selected_final_consumer_checkpoint(
        model=model,
        registration_path=args.registration,
        checkpoint_path=args.checkpoint,
        expected_run_record_sha256=run["content_hash"],
        expected_source=campaign["source"],
    )
    cache, dataset = load_final_consumer_dataset(
        args.val_design_cache,
        expected_split="val_design",
        expected_source=campaign["source"],
    )
    expected_cache_parents = {
        "identity_manifest": run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        "HLT_view_cache": run["parent_hashes"][
            "val_design_fixed_view_cache"
        ],
        "joint_prediction_cache": run["parent_hashes"][
            "joint_prediction_checkpoint"
        ],
        "native_HLT_cache": run["parent_hashes"][
            "native_HLT_checkpoint_bundle"
        ],
        "offline_target_cache": run["parent_hashes"][
            "offline_target_cache"
        ],
        "target_normalizer_set": run["parent_hashes"][
            "target_normalizer_set"
        ],
        "uncertainty_calibration": run["parent_hashes"][
            "uncertainty_calibration"
        ],
    }
    if cache["parent_hashes"] != expected_cache_parents:
        raise ValueError("bypass-control cache lineage differs")
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_design"
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model.to(device)
    result = evaluate_final_consumer_bypass_controls(
        model=model,
        consumer_kind=run["consumer_kind"],
        loader=make_final_consumer_loader(
            dataset,
            batch_size=args.batch_size,
            seed=int(run["pipeline_seed"]),
            training=False,
        ),
        frozen_expert_heads=heads,
        frozen_offline_fusion=fusion,
        device=device,
        refiner=refiner,
    )
    arrays = {
        "identities": np.asarray(result["identities"], dtype=np.str_)
    }
    for control, row in result["controls"].items():
        arrays[f"logits__{control}"] = row.pop("logits")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    publication = write_immutable_bytes(
        args.output_dir / "bypass_control_logits.npz",
        stream.getvalue(),
    )
    artifact = bind_source(
        with_content_hash(
            {
                "contract": "retb_final_consumer_bypass_controls_v1",
                "schema_version": 1,
                "run_sha256": run["content_hash"],
                "registration_sha256": registration["content_hash"],
                "split": "val_design",
                "controls": result["controls"],
                "npz_filename": "bypass_control_logits.npz",
                "npz_sha256": publication["file_sha256"],
                "identity_count": len(result["identities"]),
                "input_cache_sha256": cache["content_hash"],
                "used_for_checkpoint_selection": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output = write_immutable_json(
        args.output_dir / "bypass_controls.json", artifact
    )
    print(
        json.dumps(
            {
                "bypass_controls_sha256": artifact["content_hash"],
                "publication": output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
