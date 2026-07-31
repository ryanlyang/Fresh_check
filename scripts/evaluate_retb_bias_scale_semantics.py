#!/usr/bin/env python3
"""Evaluate fixed-versus-learned relation-bias scale controls on val_design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_retb_offline_optimization_wave import _model, _sha256  # noqa: E402
from scripts.train_retb_offline_expert import _load_trees  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source, load_hashed_json, with_content_hash, write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    OfflineExpertDataset, collect_expert_diagnostics, make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.step4 import validate_stage_b_run_registry  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access, load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    authorize_dataset_access(worker_role="design_worker", requested_resource="val_design")
    registry = load_hashed_json(root / "registry" / "retb_stage_b_runs.json")
    validate_stage_b_run_registry(registry)
    relation = load_hashed_json(
        root / "inputs" / "normalization" / "offline_500k" / "relation.json"
    )
    region = load_hashed_json(
        root / "inputs" / "normalization" / "offline_500k" / "region.json"
    )
    manifest = load_hashed_json(
        root / "inputs" / "offline" / "val_design" / "offline_input_manifest.json"
    )
    input_path = root / "inputs" / "offline" / "val_design" / "offline_inputs.npz"
    if manifest.get("npz_sha256") != _sha256(input_path):
        raise ValueError("bias-scale val_design input bytes differ")
    with np.load(input_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    identities = [str(value) for value in arrays["identities"].tolist()]
    dataset = OfflineExpertDataset(
        tokens=arrays["tokens"], mask=arrays["mask"], labels=arrays["labels"],
        identities=identities,
        region_trees=_load_trees(
            root / "inputs" / "region_tree" / "offline",
            split="val_design",
            identities=identities,
        ),
    )
    loader = make_offline_expert_loader(
        dataset, seed=0, training=False, batch_size=args.batch_size
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    rows = []
    for member in registry["dual_topology_controls"]:
        topology = str(member["configuration"]["topology"])
        if topology not in {"B_DUAL_FIXED", "B_DUAL_GATED"}:
            raise ValueError("bias-scale topology registry differs")
        run_id = str(member["run_id"])
        run_root = root / "runs" / "stage_b" / run_id / f"seed_{member['seed']}"
        registration = load_hashed_json(run_root / "checkpoint_registration.json")
        checkpoint_path = run_root / "best_model_val.pt"
        if registration.get("checkpoint_sha256") != _sha256(checkpoint_path):
            raise ValueError("bias-scale checkpoint lineage differs")
        model = _model(
            configuration=member["configuration"],
            relation_normalization=relation,
            region_normalization=(
                region
                if member["configuration"]["expert_id"] == "REGION"
                else None
            ),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        _, prediction = collect_expert_diagnostics(model, loader, device=device)
        metrics = evaluate_classification(
            prediction["logits"], prediction["labels"], split="val_design"
        )
        logits_sha = hashlib.sha256(
            np.asarray(prediction["logits"], dtype=np.float32).tobytes(order="C")
        ).hexdigest()
        rows.append({
            "run_id": run_id,
            "pipeline_seed": int(member["seed"]),
            "expert_id": str(member["configuration"]["expert_id"]),
            "topology": topology,
            "checkpoint_registration_sha256": registration["content_hash"],
            "logits_float32_sha256": logits_sha,
            "metrics": metrics,
        })
    expected = {
        (expert, topology)
        for expert in ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION", "BASE4")
        for topology in ("B_DUAL_FIXED", "B_DUAL_GATED")
    }
    if {(row["expert_id"], row["topology"]) for row in rows} != expected:
        raise ValueError("bias-scale fixed/gated coverage differs")
    artifact = bind_source(
        with_content_hash({
            "contract": "retb_bias_scale_semantic_evaluation_v1",
            "schema_version": 1,
            "split": "val_design",
            "stage_b_registry_sha256": registry["content_hash"],
            "val_design_input_manifest_sha256": manifest["content_hash"],
            "rows": sorted(rows, key=lambda row: row["run_id"]),
            "fixed_and_learned_scale_topologies_complete": True,
            "no_control_used_for_selection": True,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.output, artifact)
    print(json.dumps({"publication": publication, "row_count": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
