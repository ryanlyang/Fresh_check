#!/usr/bin/env python3
"""Consume the exactly-once claim and publish complete final-row results."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_final_evaluation,
    build_final_row_result,
    build_label_free_hlt_loader,
    consume_final_execution_claim,
    infer_deployable_graph,
    load_deployable_graph,
    load_final_execution_claim,
    load_authorized_identity_labels,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    FINAL_INPUT_PREPARATION_CONTRACT,
    FINAL_EXECUTION_LOCK_CONTRACT,
    FINAL_ROW_RESULT_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--rows-json", type=Path)
    parser.add_argument("--identities-labels-npz", required=True, type=Path)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    authorize_access(worker_role="final_inference", requested_resource="final_test_execution_lock")
    lock = load_hashed_json(args.campaign_root / "selection" / "final_test_execution_lock.json", expected_contract=FINAL_EXECUTION_LOCK_CONTRACT)
    prepared = load_hashed_json(
        args.campaign_root / "final_test" / "prepared_inputs.json",
        expected_contract=FINAL_INPUT_PREPARATION_CONTRACT,
    )
    if prepared["content_hash"] != lock["prepared_inputs_sha256"]:
        raise ValueError("final prepared-input lineage differs")
    output = args.output or args.campaign_root / "final_test" / "final_evaluation.json"
    if output.exists():
        artifact = load_hashed_json(output, expected_contract="hosd_final_evaluation_v1")
        if (
            artifact.get("execution_lock_sha256") != lock["content_hash"]
            or artifact.get("source") != campaign["source"]
        ):
            raise ValueError("reusable final evaluation lock differs")
        print(json.dumps({"content_hash": artifact["content_hash"], "publication": "reused"}, indent=2, sort_keys=True))
        return 0
    import torch

    identities, labels = load_authorized_identity_labels(
        args.identities_labels_npz,
        worker_role="final_inference",
        requested_resource="postlock_final_test_identity_labels",
    )
    caches = {}
    for value in args.cache:
        replica, separator, path = value.partition("=")
        if not separator or int(replica) in caches:
            raise ValueError("cache arguments must be unique REPLICA=PATH")
        caches[int(replica)] = Path(path)
    if set(caches) != {0}:
        raise ValueError("final-test inference requires fixed replica 0")
    consumed_input_hashes = {
        hashlib.sha256(args.identities_labels_npz.read_bytes()).hexdigest(),
        *{
            hashlib.sha256(
                (path / "hlt_v3_metadata.json").read_bytes()
            ).hexdigest()
            for path in caches.values()
        },
    }
    if not consumed_input_hashes.issubset(
        set(prepared["input_hashes"].values())
    ):
        raise ValueError("final runtime inputs were not prepared and locked")
    loader = build_label_free_hlt_loader(
        cache_paths=caches,
        identities=identities,
        logical_role="final_test",
        realization_policy="R_FIXED",
        batch_size=args.batch_size,
    )
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    rows = (
        json.loads(args.rows_json.read_text(encoding="utf-8"))
        if args.rows_json is not None
        else [
            {
                **row,
                "export_path": str(
                    row.get(
                        "export_path",
                        (
                            args.campaign_root
                            / "scale_up"
                            / "exports"
                            / f"{row['graph_id']}__seed_{row['seed']}.pt"
                        ).resolve(),
                    )
                ),
            }
            for row in lock["final_rows"]
        ]
    )
    if (
        not isinstance(rows, list)
        or {
            (
                str(row["row_id"]),
                str(row["graph_id"]),
                int(row["seed"]),
                str(row["export_sha256"]),
                str(row["checkpoint_sha256"]),
            )
            for row in rows
        }
        != {
            (
                str(row["row_id"]),
                str(row["graph_id"]),
                int(row["seed"]),
                str(row["export_sha256"]),
                str(row["checkpoint_sha256"]),
            )
            for row in lock["final_rows"]
        }
        or len(rows) != len(lock["final_rows"])
    ):
        raise ValueError("final rows JSON differs from execution lock")
    prepared_models = []
    for row in rows:
        export_path = Path(row["export_path"])
        model, export = load_deployable_graph(
            export_path,
            weaver_module=module,
            source=campaign["source"],
        )
        export_sha256 = export["content_hash"]
        if export_sha256 != row["export_sha256"]:
            raise ValueError("final row export manifest differs")
        if export["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("final row checkpoint lineage differs")
        prepared_models.append((row, model, export, export_sha256))
    claim_path = (
        args.campaign_root / "final_test" / "execution_claim_consumed.json"
    )
    claim = (
        load_final_execution_claim(
            execution_lock=lock,
            claim_path=claim_path,
            source=campaign["source"],
        )
        if claim_path.exists()
        else consume_final_execution_claim(
            execution_lock=lock,
            claim_path=claim_path,
            source=campaign["source"],
        )
    )
    results = []
    from teacher_logit_reco.relation_expert_token_bridge.evaluation import evaluate_classification
    for row, model, export, export_sha256 in prepared_models:
        result_path = (
            args.campaign_root
            / "final_test"
            / "rows"
            / f"{row['row_id']}.json"
        )
        if result_path.exists():
            result = load_hashed_json(
                result_path, expected_contract=FINAL_ROW_RESULT_CONTRACT
            )
            if (
                result.get("execution_lock_sha256") != lock["content_hash"]
                or result.get("execution_claim_sha256")
                != claim["content_hash"]
                or result.get("export_sha256") != export_sha256
                or result.get("checkpoint_sha256")
                != export["checkpoint_sha256"]
                or result.get("source") != campaign["source"]
            ):
                raise ValueError("reusable final row lineage differs")
            results.append(result)
            continue
        observed_ids, logits = infer_deployable_graph(
            model,
            loader,
            device=(
                "cuda"
                if args.device == "auto" and torch.cuda.is_available()
                else "cpu"
                if args.device == "auto"
                else args.device
            ),
        )
        if observed_ids != identities:
            raise ValueError("final inference identity order changed")
        metrics = evaluate_classification(logits, labels, split="final_test")
        result = build_final_row_result(
            execution_lock=lock,
            consumed_claim=claim,
            row_id=row["row_id"],
            graph_id=row["graph_id"],
            seed=int(row["seed"]),
            export_sha256=export_sha256,
            checkpoint_sha256=export["checkpoint_sha256"],
            classification_metrics=metrics,
            source=campaign["source"],
        )
        write_immutable_json(result_path, result)
        results.append(result)
    artifact = build_final_evaluation(execution_lock=lock, consumed_claim=claim, row_results=results, source=campaign["source"])
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
