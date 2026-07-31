#!/usr/bin/env python3
"""Compile or finalize the immutable Stage-H robustness matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_robustness_plan,
    build_robustness_summary,
    build_robustness_result,
    build_label_free_hlt_loader,
    infer_deployable_graph,
    load_deployable_graph,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    MECHANISM_SUMMARY_CONTRACT,
    ROBUSTNESS_PLAN_CONTRACT,
    ROBUSTNESS_RESULT_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("compile", "execute", "finalize"), required=True
    )
    parser.add_argument("--graph-id", action="append", default=[])
    parser.add_argument("--result", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--row-id")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--covariates", type=Path)
    parser.add_argument("--subgroup-edges", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    if args.mode == "compile":
        mechanism = load_hashed_json(
            args.campaign_root
            / "mechanism_controls"
            / "design_confirm_summary.json",
            expected_contract=MECHANISM_SUMMARY_CONTRACT,
        )
        artifact = build_robustness_plan(
            graph_ids=args.graph_id,
            mechanism_summary_sha256=mechanism["content_hash"],
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "robustness" / "evaluation_plan.json"
        )
    elif args.mode == "execute":
        if any(
            value is None
            for value in (
                args.row_id,
                args.export,
                args.labels,
                args.covariates,
                args.subgroup_edges,
            )
        ) or not args.cache:
            parser.error(
                "execute requires row/export/cache/labels/covariates/subgroup-edges"
            )
        plan = load_hashed_json(
            args.campaign_root / "robustness" / "evaluation_plan.json",
            expected_contract=ROBUSTNESS_PLAN_CONTRACT,
        )
        rows = {row["row_id"]: row for row in plan["rows"]}
        if args.row_id not in rows:
            raise ValueError("robustness row is absent")
        row = rows[args.row_id]
        caches = {}
        for value in args.cache:
            replica, separator, path = value.partition("=")
            if not separator or int(replica) in caches:
                raise ValueError("caches must be unique REPLICA=PATH")
            caches[int(replica)] = Path(path)
        from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
            load_hlt_v3_cache,
        )

        for path in caches.values():
            _, metadata = load_hlt_v3_cache(path)
            if (
                metadata.get("degradation_profile_id")
                != row["degradation_profile"]
            ):
                raise ValueError("robustness cache degradation profile differs")
        with np.load(args.labels, allow_pickle=False) as payload:
            if not {"identities", "labels"}.issubset(payload.files):
                raise ValueError("robustness label file differs")
            identities = tuple(str(value) for value in payload["identities"].tolist())
            labels = np.asarray(payload["labels"], dtype=np.int64)
        module = importlib.import_module("weaver.nn.model.ParticleTransformer")
        model, export_payload = load_deployable_graph(
            args.export, weaver_module=module, source=campaign["source"]
        )
        if export_payload["descriptor"].get("graph_id") != row["graph_id"]:
            raise ValueError("robustness export graph differs")
        loader = build_label_free_hlt_loader(
            cache_paths=caches,
            identities=identities,
            logical_role="design_confirm",
            realization_policy=row["replica_policy"],
            batch_size=args.batch_size,
        )
        import torch

        device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        observed, logits = infer_deployable_graph(model, loader, device=device)
        if observed != identities:
            raise ValueError("robustness inference identity order differs")
        with np.load(args.covariates, allow_pickle=False) as payload:
            if "identities" not in payload.files:
                raise ValueError("robustness covariates lack identities")
            covariate_ids = tuple(
                str(value) for value in payload["identities"].tolist()
            )
            if covariate_ids != identities:
                raise ValueError("robustness covariate identities differ")
            subgroup_values = {
                name: np.asarray(payload[name], dtype=np.float64)
                for name in payload.files
                if name != "identities"
            }
        edge_payload = load_hashed_json(args.subgroup_edges)
        if edge_payload.get("source") != campaign["source"]:
            raise ValueError("robustness subgroup edges source differs")
        output = args.output or (
            args.campaign_root / "robustness" / "rows" / f"{args.row_id}.json"
        )
        prediction_path = output.with_suffix(".predictions.npz")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{prediction_path.name}.", dir=prediction_path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    identities=np.asarray(identities, dtype="U"),
                    logits=logits.astype(np.float32),
                )
            os.replace(temporary, prediction_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        artifact = build_robustness_result(
            plan=plan,
            row_id=args.row_id,
            identities=identities,
            labels=labels,
            logits=logits,
            subgroup_values=subgroup_values,
            subgroup_edges=edge_payload["edges"],
            prediction_sha256=hashlib.sha256(
                prediction_path.read_bytes()
            ).hexdigest(),
            export_sha256=hashlib.sha256(args.export.read_bytes()).hexdigest(),
            source=campaign["source"],
        )
    else:
        plan = load_hashed_json(
            args.campaign_root / "robustness" / "evaluation_plan.json",
            expected_contract=ROBUSTNESS_PLAN_CONTRACT,
        )
        result_paths = args.result or [
            args.campaign_root
            / "robustness"
            / "rows"
            / f"{row['row_id']}.json"
            for row in plan["rows"]
        ]
        artifact = build_robustness_summary(
            plan=plan,
            results=[
                load_hashed_json(
                    path, expected_contract=ROBUSTNESS_RESULT_CONTRACT
                )
                for path in result_paths
            ],
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "robustness" / "summary.json"
        )
    publication = write_immutable_json(output, artifact)
    wave = None
    if args.mode == "execute":
        from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (
            try_finalize_row_wave,
        )

        wave = try_finalize_row_wave(
            wave_id="stage_h_robustness",
            expected_paths={
                row["row_id"]: args.campaign_root
                / "robustness"
                / "rows"
                / f"{row['row_id']}.json"
                for row in plan["rows"]
            },
            expected_rows={
                row["row_id"]: {
                    "row_id": row["row_id"],
                    "graph_id": row["graph_id"],
                    "degradation_profile": row["degradation_profile"],
                    "replica_policy": row["replica_policy"],
                }
                for row in plan["rows"]
            },
            expected_contract=ROBUSTNESS_RESULT_CONTRACT,
            parent_hashes={"robustness_plan": plan["content_hash"]},
            source=campaign["source"],
            output=args.campaign_root
            / "robustness"
            / "evaluation_completion.json",
        )
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
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
