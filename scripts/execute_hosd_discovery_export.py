#!/usr/bin/env python3
"""Export one locked discovery graph and attest complete discovery coverage."""

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
    monolithic_flop_ledger,
    particle_net_flop_ledger,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    GRAPH_REGISTRY_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def _definition(registry, graph_id: str):
    rows = [
        row
        for row in registry["definitions_by_role"].values()
        if row["graph_id"] == graph_id
    ]
    if len(rows) != 1 and not rows:
        raise ValueError("discovery export graph is absent")
    if not rows:
        raise AssertionError("unreachable")
    first = rows[0]
    if any(row != first for row in rows[1:]):
        raise ValueError("discovery graph aliases differ")
    return dict(first)


def _paths(root: Path, definition):
    graph_id, kind = definition["graph_id"], definition["graph_kind"]
    if kind == "BASELINE":
        run = root / "baselines" / definition["baseline_id"] / "seed_101"
        completion = run / "baseline_completion.json"
        result = None
    elif kind == "AUXILIARY":
        run = root / "auxiliary" / graph_id / "seed_101"
        completion = run / "auxiliary_completion.json"
        result = run / "design_select_result.json"
    elif kind == "FEEDBACK":
        run = root / "feedback" / graph_id / "seed_101"
        completion = run / "feedback_completion.json"
        result = run / "design_select_result.json"
    elif kind == "COMBINATION":
        run = root / "combinations" / graph_id / "seed_101"
        completion = run / "combination_completion.json"
        result = run / "design_select_result.json"
    else:
        raise ValueError("discovery export graph kind differs")
    return run / "best_model_val.pt", completion, result


def _model(definition, module):
    kind = definition["graph_kind"]
    if kind == "BASELINE":
        return build_baseline_model(
            definition["baseline_id"], weaver_module=module
        )
    if kind == "AUXILIARY":
        return build_auxiliary_model(
            definition["row"], weaver_module=module
        )[0]
    if kind == "FEEDBACK":
        return build_feedback_model(definition["row"], weaver_module=module)
    if kind == "COMBINATION":
        return build_combination_model(
            definition["graph"], seed=101, weaver_module=module
        )
    raise ValueError("discovery export graph kind differs")


def _flops(definition, result):
    if result is not None:
        return int(float(result["deployed_analytical_flops"]))
    if definition["baseline_id"] == "H_PARTICLENET":
        return particle_net_flop_ledger(1.0)["total_flops"]
    return monolithic_flop_ledger(
        {
            "embed_dim": 128,
            "particle_blocks": 8,
            "class_blocks": 2,
            "attention_heads": 8,
        }
    )["total_flops"]


def _completion(root: Path, registry, source):
    definitions = {}
    for row in registry["definitions_by_role"].values():
        definitions.setdefault(str(row["graph_id"]), row)
    manifests = {}
    for graph_id in sorted(definitions):
        path = root / "confirmation_500k" / "discovery_exports" / f"{graph_id}.pt.json"
        if not path.is_file():
            return None
        manifest = load_hashed_json(path)
        if (
            manifest.get("source") != source
            or manifest.get("descriptor") != definitions[graph_id]
            or not manifest.get("hlt_only")
        ):
            raise ValueError("discovery export coverage semantics differ")
        manifests[graph_id] = manifest["content_hash"]
    artifact = with_content_hash(
        {
            "contract": "hosd_discovery_export_completion_v1",
            "schema_version": 1,
            "source": dict(source),
            "graph_registry_sha256": registry["content_hash"],
            "export_manifest_hashes": manifests,
            "graph_count": len(manifests),
            "coverage_exact": True,
            "hlt_only": True,
        }
    )
    write_immutable_json(
        root
        / "confirmation_500k"
        / "discovery_export_completion.json",
        artifact,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--identities-npz", required=True, type=Path)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    registry = load_hashed_json(
        root / "registry" / "locked_graph_registry.json",
        expected_contract=GRAPH_REGISTRY_CONTRACT,
    )
    definition = _definition(registry, args.graph_id)
    checkpoint_path, completion_path, result_path = _paths(root, definition)
    completion = load_hashed_json(completion_path)
    result = None if result_path is None else load_hashed_json(result_path)
    if completion.get("source") != campaign["source"]:
        raise ValueError("discovery training completion source differs")
    if args.dry_run:
        print(
            json.dumps(
                {"graph_id": args.graph_id, "executed": False},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    import torch

    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model = _model(definition, module)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("source") != campaign["source"]
        or checkpoint.get("campaign_spec_sha256") != campaign["content_hash"]
    ):
        raise ValueError("discovery checkpoint lineage differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with np.load(args.identities_npz, allow_pickle=False) as payload:
        if set(payload.files) != {"identities"}:
            raise ValueError("discovery parity identities differ")
        identities = tuple(str(value) for value in payload["identities"].tolist())
    caches = {}
    for value in args.cache:
        key, separator, path = value.partition("=")
        if not separator or int(key) in caches:
            raise ValueError("discovery caches must be unique REPLICA=PATH")
        caches[int(key)] = Path(path)
    if set(caches) != {0}:
        raise ValueError("discovery export requires fixed HLT replica zero")
    loader = build_label_free_hlt_loader(
        cache_paths=caches,
        identities=identities,
        logical_role="val_stop",
        realization_policy="R_FIXED",
        batch_size=min(8, len(identities)),
    )
    output = (
        root / "confirmation_500k" / "discovery_exports" / f"{args.graph_id}.pt"
    )
    manifest = export_deployable_graph(
        descriptor=definition,
        research_model=model,
        representative_batch=next(iter(loader)),
        output_path=output,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        lineage_hashes={
            "graph_registry": registry["content_hash"],
            "training_completion": completion["content_hash"],
        },
        source=campaign["source"],
        weaver_module=module,
        analytical_inference_flops_batch1_n128=_flops(definition, result),
    )
    coverage = _completion(root, registry, campaign["source"])
    print(
        json.dumps(
            {
                "graph_id": args.graph_id,
                "export_manifest_sha256": manifest["content_hash"],
                "coverage_complete": coverage is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
