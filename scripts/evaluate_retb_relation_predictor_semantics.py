#!/usr/bin/env python3
"""Evaluate Section-28 relation and predictor controls on val_design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_retb_joint_training_row import _raw_view  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    SEMANTIC_CONTROL_POLICY, bind_source, load_hashed_json, with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    DEPLOYABLE_EXPORT_CONTRACT, load_deployable_retb_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    load_final_consumer_dataset,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access, load_and_validate_campaign_source,
)

RELATION_MODES = (
    "active", "within_jet_cyclic",
    "wrong_event_matched_multiplicity", "directional_endpoint_swap",
)
PREDICTOR_MODES = (
    "active", "zero_hlt_evidence", "shuffle_hlt_evidence_between_events",
    "remove_native_particle_context", "remove_noncorresponding_expert_banks",
)
CONTRACT = "retb_relation_predictor_semantic_controls_v3"
RELATION_EXPERTS = tuple(expert for expert in EXPERT_ORDER if expert != "BASE4")


def _batch(raw: Mapping[str, Any], indices: np.ndarray, device: torch.device) -> dict[str, Any]:
    return {
        "identities": [str(raw["identities"][int(i)]) for i in indices],
        "replica_ids": torch.zeros(len(indices), dtype=torch.int64, device=device),
        "degraded_view_hashes": [
            str(raw["degraded_view_hashes_by_replica"][0][int(i)])
            for i in indices
        ],
        "features": torch.from_numpy(raw["features"][0][indices]).to(device),
        "vectors": torch.from_numpy(raw["vectors"][0][indices]).to(device),
        "mask": torch.from_numpy(raw["mask"][0][indices]).to(device),
        "raw_tokens": torch.from_numpy(raw["raw_tokens"][0][indices]).to(device),
        "region_trees_by_expert": {
            expert: [raw["region_trees_by_expert"][expert][0][int(i)] for i in indices]
            for expert in EXPERT_ORDER
        },
    }


def _groups(
    mask: np.ndarray, *, matched: bool, batch_size: int
) -> list[np.ndarray]:
    if int(batch_size) < 2:
        raise ValueError("semantic-control batches require at least two events")

    def nonsingleton_chunks(indices: np.ndarray) -> list[np.ndarray]:
        chunks = []
        start = 0
        while start < len(indices):
            remaining = len(indices) - start
            take = remaining if remaining <= int(batch_size) + 1 else int(batch_size)
            if take < 2:
                raise ValueError("semantic event shuffle would contain a singleton")
            chunks.append(indices[start : start + take])
            start += take
        return chunks

    if not matched:
        return nonsingleton_chunks(np.arange(len(mask), dtype=np.int64))
    counts = mask[:, 0].astype(bool).sum(axis=1)
    output = []
    for count in sorted(set(int(value) for value in counts)):
        indices = np.flatnonzero(counts == count)
        # An exact, non-self, matched-multiplicity wrong event is
        # mathematically impossible for a singleton stratum.  The globally
        # frozen policy excludes only those strata from this causal contrast;
        # their identities and the resulting eligible fraction are attested.
        if len(indices) < 2:
            continue
        output.extend(nonsingleton_chunks(indices))
    if not output:
        raise ValueError(
            "wrong-event relation has no derangeable matched-multiplicity stratum"
        )
    return output


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(metrics[name])
        for name in ("accuracy", "cross_entropy", "macro_per_class_accuracy")
    }


def _metric_delta(
    metrics: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    return {
        f"{name}_control_minus_reference": float(metrics[name])
        - float(reference[name])
        for name in ("accuracy", "cross_entropy", "macro_per_class_accuracy")
    }


def _evaluate(
    graph: Any, raw: Mapping[str, Any], labels: np.ndarray, *,
    relation_mode: str, predictor_mode: str, batch_size: int,
    device: torch.device, relation_expert: str | None = None,
    groups: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    joint = graph.frontend.joint_graph
    joint.set_semantic_relation_transform(
        relation_mode, expert_id=relation_expert
    )
    joint.set_semantic_predictor_control(predictor_mode)
    resolved_groups = groups or _groups(
        raw["mask"][0], matched=False, batch_size=batch_size
    )
    logits, truth, identities = [], [], []
    graph.to(device).eval()
    with torch.no_grad():
        for indices in resolved_groups:
            result = graph(hlt_inputs=_batch(raw, indices, device))
            logits.append(result["logits"].float().cpu().numpy())
            truth.append(labels[indices])
            identities.extend(str(raw["identities"][int(i)]) for i in indices)
    values = np.concatenate(logits).astype(np.float32)
    targets = np.concatenate(truth).astype(np.int64)
    selected = np.concatenate(resolved_groups)
    excluded = np.setdiff1d(
        np.arange(len(labels), dtype=np.int64), selected, assume_unique=False
    )
    if len(targets) != len(selected):
        raise RuntimeError("semantic control population accounting differs")
    return {
        "metrics": evaluate_classification(values, targets, split="val_design"),
        "event_count": len(targets),
        "identity_order_sha256": __import__("hashlib").sha256(
            "\n".join(identities).encode("utf-8")
        ).hexdigest(),
        "relation_expert": relation_expert,
        "matched_multiplicity_exact": relation_mode
        == "wrong_event_matched_multiplicity",
        "all_val_design_events_evaluated": len(excluded) == 0,
        "eligible_population": {
            "total_event_count": int(len(labels)),
            "evaluated_event_count": int(len(selected)),
            "evaluated_fraction": float(len(selected) / len(labels)),
            "excluded_singleton_stratum_event_count": int(len(excluded)),
            "excluded_identity_order_sha256": __import__("hashlib").sha256(
                "\n".join(
                    str(raw["identities"][int(index)]) for index in excluded
                ).encode("utf-8")
            ).hexdigest(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--deployable-export", required=True, type=Path)
    parser.add_argument("--val-design-cache", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    if campaign.get("semantic_control_policy") != SEMANTIC_CONTROL_POLICY:
        raise ValueError("campaign semantic-control policy differs")
    authorize_dataset_access(worker_role="design_worker", requested_resource="val_design")
    export = load_hashed_json(
        args.deployable_export, expected_contract=DEPLOYABLE_EXPORT_CONTRACT
    )
    cache, dataset = load_final_consumer_dataset(
        args.val_design_cache,
        expected_split="val_design", expected_source=campaign["source"],
    )
    if export.get("source") != campaign.get("source"):
        raise ValueError("semantic deployable source differs")
    raw, raw_sha = _raw_view(
        root, split="val_design", identities=dataset.identities
    )
    raw = {
        "identities": dataset.identities,
        "degraded_view_hashes_by_replica": dataset.degraded_view_hashes,
        **raw,
    }
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    graph = load_deployable_retb_graph(
        args.deployable_export, expected_source=campaign["source"]
    )
    controls: dict[str, dict[str, Any]] = {}
    active = _evaluate(
        graph, raw, dataset.labels, relation_mode="active",
        predictor_mode="active", batch_size=args.batch_size, device=device,
    )
    controls["RELATION__active"] = active
    for expert in RELATION_EXPERTS:
        row = _evaluate(
            graph, raw, dataset.labels, relation_mode="zero",
            predictor_mode="active", batch_size=args.batch_size, device=device,
            relation_expert=expert,
        )
        row["reference_metrics"] = _metric_summary(active["metrics"])
        row["metric_deltas"] = _metric_delta(row["metrics"], active["metrics"])
        controls[f"RELATION__zero__{expert}"] = row
    for mode in RELATION_MODES:
        if mode == "active":
            continue
        matched_groups = (
            _groups(raw["mask"][0], matched=True, batch_size=args.batch_size)
            if mode == "wrong_event_matched_multiplicity"
            else None
        )
        reference = (
            _evaluate(
                graph, raw, dataset.labels, relation_mode="active",
                predictor_mode="active", batch_size=args.batch_size,
                device=device, groups=matched_groups,
            )
            if matched_groups is not None
            else active
        )
        controls[f"RELATION__{mode}"] = _evaluate(
            graph, raw, dataset.labels, relation_mode=mode,
            predictor_mode="active", batch_size=args.batch_size, device=device,
            groups=matched_groups,
        )
        controls[f"RELATION__{mode}"]["reference_metrics"] = _metric_summary(
            reference["metrics"]
        )
        controls[f"RELATION__{mode}"]["metric_deltas"] = _metric_delta(
            controls[f"RELATION__{mode}"]["metrics"], reference["metrics"]
        )
    for mode in PREDICTOR_MODES:
        row = _evaluate(
            graph, raw, dataset.labels, relation_mode="active",
            predictor_mode=mode, batch_size=args.batch_size, device=device,
        )
        row["reference_metrics"] = _metric_summary(active["metrics"])
        row["metric_deltas"] = _metric_delta(row["metrics"], active["metrics"])
        controls[f"PREDICTOR__{mode}"] = row
    # Restore ordinary inference state before serializing the attestation.
    graph.frontend.joint_graph.set_semantic_relation_transform("active")
    graph.frontend.joint_graph.set_semantic_predictor_control("active")
    artifact = bind_source(
        with_content_hash({
            "contract": CONTRACT,
            "schema_version": 3,
            "graph_id": args.graph_id,
            "pipeline_seed": int(args.pipeline_seed),
            "split": "val_design",
            "deployable_export_sha256": export["content_hash"],
            "val_design_cache_sha256": cache["content_hash"],
            "raw_hlt_view_sha256": raw_sha,
            "controls": controls,
            "per_expert_relation_zero_coverage": list(RELATION_EXPERTS),
            "wrong_event_singleton_policy": (
                "exclude_exact_singleton_multiplicity_strata_from_both_the_"
                "perturbed_row_and_its_paired_active_reference"
            ),
            "semantic_control_policy": dict(SEMANTIC_CONTROL_POLICY),
            "wrong_event_relation_is_multiplicity_matched_and_never_self": True,
            "no_control_used_for_selection": True,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }), source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.output, artifact)
    print(json.dumps({"publication": publication, "control_count": len(controls)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
