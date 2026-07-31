#!/usr/bin/env python3
"""Select complete Stage-D evidence modes and freeze confirmation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_d_selection import (  # noqa: E402
    select_stage_d_evidence_modes,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    materialize_stage_d_confirmation_rows,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(path: Path) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as payload:
        labels = np.asarray(payload["labels"], dtype=np.int64)
        logits = np.asarray(payload["logits"], dtype=np.float64)
    if logits.shape != (len(labels), 10) or not np.isfinite(logits).all():
        raise ValueError("Stage-D evidence prediction arrays differ")
    shifted = logits - logits.max(axis=1, keepdims=True)
    accuracy = float((logits.argmax(axis=1) == labels).mean())
    cross_entropy = float(
        (
            np.log(np.exp(shifted).sum(axis=1))
            - shifted[np.arange(len(labels)), labels]
        ).mean()
    )
    return accuracy, cross_entropy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirmation-output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    results = []
    seen = set()
    for row in registry["encoder_screen_rows"]:
        config = row["configuration"]
        if config["shape_id"] not in {"SHAPE_COMPACT", "SHAPE_HIGH"}:
            continue
        run_id = str(row["run_id"])
        if run_id in seen:
            continue
        seen.add(run_id)
        root = (
            args.campaign_root
            / "runs"
            / "stage_d"
            / "hlt_experts"
            / run_id
            / "seed_101"
        )
        manifest = load_hashed_json(root / "native_output_manifest.json")
        record = manifest["files"]["val_design_replica_0"]
        prediction = root / record["relative_path"]
        if _sha256(prediction) != record["file_sha256"]:
            raise ValueError("Stage-D evidence prediction bytes differ")
        accuracy, cross_entropy = _metrics(prediction)
        results.append(
            {
                "run_id": run_id,
                "val_design_accuracy": accuracy,
                "val_design_cross_entropy": cross_entropy,
                "result_sha256": manifest["content_hash"],
            }
        )
    selection = bind_source(
        select_stage_d_evidence_modes(registry=registry, results=results),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    if selection["source"] != campaign["source"]:
        raise ValueError("Stage-D evidence selection source differs")
    write_immutable_json(args.output, selection)
    selected_ids = [
        row["selected_screen_run_id"] for row in selection["selected_rows"]
    ] + list(selection["selected_native_fusion_run_ids"])
    confirmation = bind_source(
        materialize_stage_d_confirmation_rows(
            registry, selected_run_ids=selected_ids
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.confirmation_output, confirmation)
    print(
        json.dumps(
            {
                "selection_sha256": selection["content_hash"],
                "confirmation_sha256": confirmation["content_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
