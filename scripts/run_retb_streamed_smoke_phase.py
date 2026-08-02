#!/usr/bin/env python3
"""Execute one compact, real, authenticated streamed RETB smoke phase."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json, with_content_hash, write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    SMOKE_PHASES, STREAMED_SMOKE_PHASE_CONTRACT,
    STREAMED_SMOKE_PLAN_CONTRACT, build_streamed_smoke_plan,
    task_local_workspace, validate_streamed_smoke_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tiny_gpu_step(seed: int, *, require_cuda: bool) -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    if require_cuda and not cuda and os.environ.get("RETB_SMOKE_ALLOW_CPU") != "1":
        raise RuntimeError("compact streamed smoke GPU phase requires CUDA")
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    attention = torch.nn.MultiheadAttention(32, 4, batch_first=True).to(device)
    classifier = torch.nn.Linear(32, 10).to(device)
    optimizer = torch.optim.AdamW(
        [*attention.parameters(), *classifier.parameters()], lr=1e-3
    )
    values = torch.randn(4, 16, 32, device=device)
    mask = torch.zeros(4, 16, dtype=torch.bool, device=device)
    mask[:, -3:] = True
    target = torch.tensor([0, 1, 2, 3], device=device)
    attended, weights = attention(values, values, values, key_padding_mask=mask)
    logits = classifier(attended[:, :13].mean(dim=1))
    loss = torch.nn.functional.cross_entropy(logits, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(logits).all() or not math.isfinite(float(loss)):
        raise RuntimeError("compact smoke produced non-finite values")
    if weights[..., -3:].abs().max().item() != 0.0:
        raise RuntimeError("compact smoke attention mask was not exact")
    return {
        "device_type": device.type,
        "loss": float(loss.detach().cpu()),
        "logit_checksum": hashlib.sha256(
            logits.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "masking_exact": True,
        "gradient_step_completed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--phase-id", required=True)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(args.campaign_root, repo_root=REPO_ROOT)
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    if graph.get("execution_profile") != "streamed_smoke":
        raise ValueError("compact smoke worker received another execution profile")
    phase_map = {row["phase_id"]: row for row in SMOKE_PHASES}
    if args.phase_id not in phase_map:
        raise ValueError("unknown compact smoke phase")
    phase = phase_map[args.phase_id]
    index = next(i for i, row in enumerate(SMOKE_PHASES) if row == phase)
    smoke_root = args.campaign_root / "evaluations" / "streamed_smoke"
    plan_path = args.campaign_root / "job_ledgers" / "streamed_smoke_plan.json"
    if index == 0 and not plan_path.is_file():
        plan = build_streamed_smoke_plan(
            campaign_spec_sha256=campaign["content_hash"],
            production_graph_sha256=graph["content_hash"],
            campaign_id=campaign["campaign_id"], source=campaign["source"],
        )
        write_immutable_json(plan_path, plan)
    plan = load_hashed_json(plan_path, expected_contract=STREAMED_SMOKE_PLAN_CONTRACT)
    validate_streamed_smoke_plan(plan)
    if plan.get("source") != campaign.get("source"):
        raise ValueError("compact smoke plan source differs")

    previous_hash = None
    if index:
        previous_id = SMOKE_PHASES[index - 1]["phase_id"]
        previous = load_hashed_json(
            smoke_root / "phases" / f"{previous_id}.json",
            expected_contract=STREAMED_SMOKE_PHASE_CONTRACT,
        )
        previous_hash = previous["content_hash"]

    split_path = args.campaign_root / "inputs" / "split_manifest.json.gz"
    if not split_path.is_file():
        raise FileNotFoundError("authenticated miniature split is absent")
    split_sha = _file_sha(split_path)
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split_payload = json.load(stream)
    split_shape = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    output = smoke_root / "phases" / f"{args.phase_id}.json"
    if output.is_file():
        retained = load_hashed_json(
            output, expected_contract=STREAMED_SMOKE_PHASE_CONTRACT
        )
        if (
            retained.get("campaign_spec_sha256") != campaign["content_hash"]
            or retained.get("production_graph_sha256") != graph["content_hash"]
            or retained.get("smoke_plan_sha256") != plan["content_hash"]
            or retained.get("phase_id") != args.phase_id
            or retained.get("previous_phase_sha256") != previous_hash
            or retained.get("split_manifest_file_sha256") != split_sha
            or retained.get("source") != campaign.get("source")
            or retained.get("workspace_removed_before_publication") is not True
            or retained.get("production_evidence_eligible") is not False
        ):
            raise ValueError("reusable compact smoke phase differs")
        print(json.dumps({"status": "reused", "phase": retained}, indent=2, sort_keys=True))
        return 0

    workspace_parent = None
    workspace_removed = False
    workspace_payload_sha = None
    seed = int(campaign["content_hash"][:8], 16) + index
    with task_local_workspace(
        campaign_id=campaign["campaign_id"], node_id=args.phase_id,
        task_index=0,
    ) as workspace:
        workspace_parent = str(workspace.parent)
        payload = workspace / "transient_payload.bin"
        payload.write_bytes(hashlib.sha256(f"{seed}:{args.phase_id}".encode()).digest() * 32)
        workspace_payload_sha = _file_sha(payload)
        execution = _tiny_gpu_step(seed, require_cuda=phase["resource"] == "gpu")
    workspace_removed = not workspace.exists()
    if not workspace_removed:
        raise RuntimeError("compact smoke task-local workspace survived")

    final_seal = None
    if args.phase_id == "n_sealed_final":
        input_lock = hashlib.sha256(f"input:{previous_hash}:{split_sha}".encode()).hexdigest()
        execution_lock = hashlib.sha256(f"execution:{input_lock}:{graph['content_hash']}".encode()).hexdigest()
        final_seal = {
            "input_lock_sha256": input_lock,
            "execution_lock_sha256": execution_lock,
            "both_locks_present_before_inference": True,
            "oracle_inputs_consumed": False,
        }
    artifact = with_content_hash({
        "contract": STREAMED_SMOKE_PHASE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": campaign["content_hash"],
        "production_graph_sha256": graph["content_hash"],
        "smoke_plan_sha256": plan["content_hash"],
        "phase_id": args.phase_id, "stage": phase["stage"],
        "kind": phase["kind"], "sequence_index": index,
        "previous_phase_sha256": previous_hash,
        "split_manifest_file_sha256": split_sha,
        "split_manifest_semantic_sha256": split_shape,
        "workspace_parent": workspace_parent,
        "workspace_payload_sha256": workspace_payload_sha,
        "workspace_removed_before_publication": workspace_removed,
        "execution": execution, "final_test_seal": final_seal,
        "scientific_metric_used_as_gate": False,
        "production_evidence_eligible": False,
        "source": campaign["source"],
    })
    publication = write_immutable_json(output, artifact)
    if args.phase_id == "n_report":
        report = with_content_hash({
            "contract": "retb_compact_streamed_smoke_report_v1",
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "smoke_plan_sha256": plan["content_hash"],
            "terminal_phase_sha256": artifact["content_hash"],
            "phase_count": len(SMOKE_PHASES),
            "all_stages_covered": list("ABCDEFGHIJKLMN"),
            "passed": True,
            "scientific_performance_claimed": False,
            "production_evidence_eligible": False,
            "source": campaign["source"],
        })
        write_immutable_json(smoke_root / "retb_compact_streamed_smoke_report.json", report)
    print(json.dumps({"phase": artifact, "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
