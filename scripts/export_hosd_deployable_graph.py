#!/usr/bin/env python3
"""Build and parity-check one self-contained HLT-only HOSD export."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_auxiliary_model,
    build_baseline_model,
    build_combination_model,
    build_feedback_model,
    build_label_free_hlt_loader,
    export_deployable_graph,
    load_and_validate_campaign,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--descriptor-json", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--identities-npz", required=True, type=Path)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--lineage", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=("FP32", "BF16"), default="FP32")
    parser.add_argument(
        "--analytical-inference-flops-batch1-n128",
        required=True,
        type=int,
    )
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    descriptor = json.loads(args.descriptor_json.read_text(encoding="utf-8"))
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    seed = int(descriptor.get("seed", 101))
    kind = descriptor["graph_kind"]
    if kind == "BASELINE":
        model = build_baseline_model(descriptor["baseline_id"], weaver_module=module)
    elif kind == "AUXILIARY":
        model, _ = build_auxiliary_model(descriptor["row"], weaver_module=module)
    elif kind == "FEEDBACK":
        model = build_feedback_model(descriptor["row"], weaver_module=module)
    elif kind == "COMBINATION":
        model = build_combination_model(descriptor["graph"], seed=seed, weaver_module=module)
    else:
        raise ValueError("unknown export graph kind")
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_graph = descriptor.get(
        "graph_id",
        descriptor.get("row", {}).get(
            "row_id", descriptor.get("graph", {}).get("graph_id")
        ),
    )
    observed_graph = checkpoint.get(
        "graph_id",
        checkpoint.get("row_id", checkpoint.get("baseline_id")),
    )
    if (
        checkpoint.get("source") != campaign["source"]
        or checkpoint.get("campaign_spec_sha256") != campaign["content_hash"]
        or observed_graph != expected_graph
    ):
        raise ValueError("deployable export checkpoint lineage differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with np.load(args.identities_npz, allow_pickle=False) as payload:
        if set(payload.files) != {"identities"}:
            raise ValueError("export parity identities NPZ differs")
        identities = [str(value) for value in payload["identities"].tolist()]
    caches = {}
    for value in args.cache:
        replica, separator, path = value.partition("=")
        if not separator or int(replica) in caches:
            raise ValueError("cache arguments must be unique REPLICA=PATH")
        caches[int(replica)] = Path(path)
    loader = build_label_free_hlt_loader(cache_paths=caches, identities=identities, logical_role="val_stop", realization_policy="R_FIXED", batch_size=min(8, len(identities)))
    representative = next(iter(loader))
    lineage = dict(value.split("=", 1) for value in args.lineage)
    manifest = export_deployable_graph(
        descriptor=descriptor,
        research_model=model,
        representative_batch=representative,
        output_path=args.output,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        lineage_hashes=lineage,
        source=campaign["source"],
        weaver_module=module,
        precision=args.precision,
        analytical_inference_flops_batch1_n128=(
            args.analytical_inference_flops_batch1_n128
        ),
    )
    print(json.dumps({"content_hash": manifest["content_hash"], "export_sha256": manifest["export_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
