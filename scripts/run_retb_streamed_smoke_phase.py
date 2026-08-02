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
import shutil
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json, with_content_hash, write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT, task_manifest_path_for_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    SMOKE_PHASES, STREAMED_SMOKE_PHASE_CONTRACT,
    STREAMED_SMOKE_PLAN_CONTRACT, STREAMED_TASK_RECEIPT_CONTRACT,
    build_streamed_smoke_phase_control_evidence,
    build_streamed_smoke_plan, task_local_workspace,
    validate_streamed_smoke_plan, validate_task_lifecycle_receipt,
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


def _tiny_gpu_step(
    seed: int, *, require_cuda: bool, phase_id: str
) -> dict[str, Any]:
    import torch

    from teacher_logit_reco.relation_expert_token_bridge.final_consumers import (
        UnrestrictedHLTFusion,
    )
    from teacher_logit_reco.relation_expert_token_bridge.fusion import (
        TokenTransformerFusion,
    )
    from teacher_logit_reco.relation_expert_token_bridge.predictors import (
        RetbTokenPredictor,
    )
    from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
    from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (
        TokenOnlyExpertHead,
    )

    cuda = torch.cuda.is_available()
    if require_cuda and not cuda and os.environ.get("RETB_SMOKE_ALLOW_CPU") != "1":
        raise RuntimeError("compact streamed smoke GPU phase requires CUDA")
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    attention = torch.nn.MultiheadAttention(128, 8, batch_first=True).to(device)
    classifier = torch.nn.Linear(128, 10).to(device)
    values = torch.randn(4, 16, 128, device=device)
    mask = torch.zeros(4, 16, dtype=torch.bool, device=device)
    mask[:, -3:] = True
    target = torch.tensor([0, 1, 2, 3], device=device)
    attended, weights = attention(values, values, values, key_padding_mask=mask)
    base_tokens = attended[:, :4]
    banks = {
        expert: base_tokens + 0.01 * expert_index
        for expert_index, expert in enumerate(EXPERT_ORDER)
    }
    architecture_component = "authenticated_preparation_attention"
    model = classifier
    logits = classifier(attended[:, :13].mean(dim=1))
    auxiliary_loss = logits.new_zeros(())
    if phase_id in {"b_expert", "d_native", "l_confirmation", "m_scale"}:
        model = TokenOnlyExpertHead(token_dimension=128, num_classes=10).to(device)
        logits = model(base_tokens)
        architecture_component = "retb_token_only_expert_head"
    elif phase_id == "c_fusion":
        model = TokenTransformerFusion(
            bank_dimensions={expert: 128 for expert in EXPERT_ORDER}
        ).to(device)
        logits = model(token_banks=banks)
        architecture_component = "retb_token_transformer_fusion"
    elif phase_id in {"e_bridge", "f_targets", "g_predictor"}:
        model = RetbTokenPredictor(
            architecture="A1_RESMLP",
            context="C0_SELF",
            target_expert_id="BASE4",
            token_count=4,
            token_dimension=128,
            offline_slot_queries=torch.zeros(4, 128, device=device),
            uncertainty_head="U_SLOT",
            dropout=0.1,
        ).to(device)
        predicted = model(corresponding_hlt_tokens=base_tokens)
        logits = classifier(predicted["predicted_tokens"].mean(dim=1))
        auxiliary_loss = torch.nn.functional.mse_loss(
            predicted["predicted_tokens"], base_tokens.detach() + 0.02
        )
        architecture_component = "retb_a1_resmlp_token_predictor"
    elif phase_id in {"h_bundle", "i_joint", "j_consumer", "k_semantics"}:
        model = UnrestrictedHLTFusion(
            evidence_variant="F_TOKEN_ONLY",
            native_dropout_mode="ND0_NONE",
            bank_dimensions={expert: 128 for expert in EXPERT_ORDER},
            token_counts={expert: 4 for expert in EXPERT_ORDER},
            uncertainty_widths={expert: 4 for expert in EXPERT_ORDER},
        ).to(device)
        consumer = model(
            token_banks=banks,
            calibrated_log_variance={
                expert: torch.zeros(4, 4, device=device)
                for expert in EXPERT_ORDER
            },
            native_banks={
                expert: value + 0.01 for expert, value in banks.items()
            },
        )
        logits = consumer["logits"]
        architecture_component = "retb_unrestricted_hlt_fusion"
        if phase_id == "k_semantics":
            perturbed = dict(banks)
            perturbed["BASE4"] = banks["BASE4"].roll(1, dims=0)
            controlled = model(
                token_banks=perturbed,
                calibrated_log_variance={
                    expert: torch.zeros(4, 4, device=device)
                    for expert in EXPERT_ORDER
                },
                native_banks={
                    expert: value + 0.01 for expert, value in banks.items()
                },
            )["logits"]
            auxiliary_loss = (controlled - logits.detach()).square().mean()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters.extend(classifier.parameters())
    optimizer = torch.optim.AdamW(list(dict.fromkeys(parameters)), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(logits, target) + auxiliary_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(logits).all() or not math.isfinite(float(loss.detach())):
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
        "architecture_component": architecture_component,
        "token_shape": [4, 128],
        "representative_seed": seed,
    }


def _run_authenticated_manifest(
    campaign_root: Path, *, node_id: str,
    task_indices: Sequence[int] | None = None,
    attest_complete: bool = True,
) -> dict[str, Any]:
    """Run every row of one real miniature preparation manifest."""

    graph = load_hashed_json(
        campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    manifest = task_manifest_path_for_graph(
        graph, node_id=node_id, campaign_root=campaign_root
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"compact smoke task manifest is absent: {manifest}")
    task_manifest = load_hashed_json(manifest)
    total_count = int(task_manifest["task_count"])
    selected = (
        list(range(total_count))
        if task_indices is None
        else [int(value) for value in task_indices]
    )
    if not selected or len(set(selected)) != len(selected) or any(
        value < 0 or value >= total_count for value in selected
    ):
        raise ValueError("compact smoke task selection differs")
    receipts = []
    for task_index in selected:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_retb_task.py",
                "--campaign-root",
                str(campaign_root),
                "--task-manifest",
                str(manifest),
                "--task-index",
                str(task_index),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"compact smoke real {node_id}:{task_index} failed "
                f"({completed.returncode})"
            )
        receipt_path = (
            campaign_root
            / "job_ledgers"
            / "streamed_tasks"
            / node_id
            / f"task_{task_index:06d}.json"
        )
        receipt = load_hashed_json(
            receipt_path,
            expected_contract=STREAMED_TASK_RECEIPT_CONTRACT,
        )
        validate_task_lifecycle_receipt(receipt)
        if (
            receipt.get("status") != "completed"
            or receipt.get("node_id") != node_id
            or int(receipt.get("task_index", -1)) != task_index
        ):
            raise ValueError("compact smoke real input receipt differs")
        receipts.append(receipt["content_hash"])
    if attest_complete:
        if len(selected) != total_count:
            raise ValueError("partial compact smoke manifest cannot be attested complete")
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/attest_retb_task_manifest_completion.py",
                "--campaign-root",
                str(campaign_root),
                "--task-manifest",
                str(manifest),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"compact smoke {node_id} completion attestation failed")
    return {
        "node_id": node_id,
        "manifest_task_count": total_count,
        "executed_task_indices": selected,
        "complete_manifest_attested": bool(attest_complete),
        "task_manifest_file_sha256": _file_sha(manifest),
        "lifecycle_receipt_sha256s": receipts,
    }


def _run_python(arguments: Sequence[str]) -> None:
    completed = subprocess.run(
        [sys.executable, *arguments], cwd=REPO_ROOT, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"compact smoke worker failed ({completed.returncode}): "
            + " ".join(arguments)
        )


def _prepare_native_smoke_parent(campaign_root: Path) -> dict[str, Any]:
    graph = load_hashed_json(
        campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    manifest = load_hashed_json(
        task_manifest_path_for_graph(
            graph,
            node_id="offline_expert_training",
            campaign_root=campaign_root,
        )
    )
    row = manifest["rows"][0]
    source_root = Path(row["expected_outputs"][0]).parent
    target = (
        campaign_root / "selection" / "offline_experts" / "S1_128"
        / "BASE4" / "seed_101"
    )
    hashes = {}
    for name in ("checkpoint_registration.json", "best_model_val.pt"):
        source = source_root / name
        destination = target / name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"compact smoke offline parent is absent: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and _file_sha(destination) != _file_sha(source):
            raise ValueError("compact smoke native-parent alias differs")
        if not destination.exists():
            shutil.copy2(source, destination)
        hashes[name] = _file_sha(destination)
    return {"path": str(target), "file_sha256s": hashes}


def _build_real_region_backend(campaign_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_relational_part_tree_backend.py",
            "--contract",
            "relational_ca_tree_v1",
            "--build-dir",
            str(campaign_root / "backend" / "build"),
            "--output-dir",
            str(campaign_root / "backend"),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("compact smoke real REGION backend failed")
    manifest = campaign_root / "backend" / "backend_manifest.json"
    artifact = load_hashed_json(manifest)
    return {
        "backend_manifest_sha256": artifact["content_hash"],
        "backend_binary_sha256": artifact["binary_sha256"],
    }


def _validate_real_input_execution(
    campaign_root: Path, evidence: Any
) -> None:
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        raise ValueError("compact smoke real-input evidence differs")
    for key, record in evidence.items():
        if key == "compiled_region_backend":
            backend = load_hashed_json(
                campaign_root / "backend" / "backend_manifest.json"
            )
            if (
                backend["content_hash"] != record.get("backend_manifest_sha256")
                or backend.get("binary_sha256")
                != record.get("backend_binary_sha256")
            ):
                raise ValueError("compact smoke REGION backend evidence differs")
            binary = campaign_root / "backend" / str(backend["binary_filename"])
            if not binary.is_file() or _file_sha(binary) != backend["binary_sha256"]:
                raise ValueError("compact smoke REGION backend binary differs")
            continue
        if key == "architecture_contracts":
            expected = {
                "step3_bundle_sha256": "retb_step3_architecture_bundle.json",
                "step4_bundle_sha256": "retb_step4_offline_expert_bundle.json",
                "step5_bundle_sha256": "retb_step5_offline_fusion_bundle.json",
                "step6_bundle_sha256": "retb_step6_native_hlt_bundle.json",
            }
            if not isinstance(record, dict) or not record:
                raise ValueError("compact smoke architecture evidence differs")
            for hash_key, filename in expected.items():
                if hash_key not in record:
                    continue
                bundle = load_hashed_json(campaign_root / "registry" / filename)
                if bundle["content_hash"] != record[hash_key]:
                    raise ValueError("compact smoke architecture bundle differs")
            if set(record) - set(expected):
                raise ValueError("compact smoke architecture evidence differs")
            continue
        if key == "resource_probe":
            probe_path = (
                campaign_root / "job_ledgers" / "resource_probes" / "gpu.json"
            )
            probe = load_hashed_json(probe_path)
            if (
                not isinstance(record, dict)
                or record.get("content_hash") != probe["content_hash"]
                or record.get("file_sha256") != _file_sha(probe_path)
            ):
                raise ValueError("compact smoke resource-probe evidence differs")
            continue
        if key == "offline_parent_alias":
            expected_root = (
                campaign_root / "selection" / "offline_experts" / "S1_128"
                / "BASE4" / "seed_101"
            ).resolve()
            if (
                not isinstance(record, dict)
                or Path(str(record.get("path", ""))).resolve() != expected_root
                or not isinstance(record.get("file_sha256s"), dict)
            ):
                raise ValueError("compact smoke native-parent alias differs")
            expected_names = {"checkpoint_registration.json", "best_model_val.pt"}
            if set(record["file_sha256s"]) != expected_names:
                raise ValueError("compact smoke native-parent alias coverage differs")
            for name in expected_names:
                artifact = expected_root / name
                if (
                    not artifact.is_file()
                    or artifact.is_symlink()
                    or _file_sha(artifact) != record["file_sha256s"][name]
                ):
                    raise ValueError("compact smoke native-parent alias differs")
            load_hashed_json(expected_root / "checkpoint_registration.json")
            continue
        node_id = str(record.get("node_id", ""))
        indices = record.get("executed_task_indices")
        expected_hashes = record.get("lifecycle_receipt_sha256s")
        if (
            not node_id
            or not isinstance(indices, list)
            or not isinstance(expected_hashes, list)
            or len(expected_hashes) != len(indices)
        ):
            raise ValueError("compact smoke real-input receipt coverage differs")
        for task_index, expected_hash in zip(indices, expected_hashes):
            receipt = load_hashed_json(
                campaign_root / "job_ledgers" / "streamed_tasks" / node_id
                / f"task_{task_index:06d}.json",
                expected_contract=STREAMED_TASK_RECEIPT_CONTRACT,
            )
            validate_task_lifecycle_receipt(receipt)
            if receipt["content_hash"] != expected_hash:
                raise ValueError("compact smoke real-input receipt hash differs")


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
            or (
                args.phase_id == "n_report"
                and not (smoke_root / "retb_compact_streamed_smoke_report.json").is_file()
            )
        ):
            raise ValueError("reusable compact smoke phase differs")
        _validate_real_input_execution(
            args.campaign_root, retained.get("real_input_execution")
        )
        expected_control = build_streamed_smoke_phase_control_evidence(
            phase_id=args.phase_id,
            previous_phase_sha256=previous_hash,
            split_manifest_sha256=split_sha,
            production_graph_sha256=graph["content_hash"],
            execution_logit_sha256=retained["execution"]["logit_checksum"],
        )
        if retained.get("phase_control_evidence") != expected_control:
            raise ValueError("reusable compact smoke control evidence differs")
        print(json.dumps({"status": "reused", "phase": retained}, indent=2, sort_keys=True))
        return 0

    real_input_execution = None
    if args.phase_id == "a_inputs":
        real_input_execution = {
            "offline": _run_authenticated_manifest(
                args.campaign_root, node_id="offline_input_cache"
            ),
            "hlt_v3": _run_authenticated_manifest(
                args.campaign_root, node_id="hlt_v3_cache"
            ),
        }
    elif args.phase_id == "a_relations":
        real_input_execution = {
            "compiled_region_backend": _build_real_region_backend(args.campaign_root),
            "region_tree": _run_authenticated_manifest(
                args.campaign_root, node_id="region_tree_cache"
            ),
            "region_tree_finalize": _run_authenticated_manifest(
                args.campaign_root, node_id="region_tree_finalize"
            ),
            "normalizers": _run_authenticated_manifest(
                args.campaign_root, node_id="normalizers_500k"
            ),
            "input_audit": _run_authenticated_manifest(
                args.campaign_root, node_id="input_audit"
            ),
        }
        _run_python([
            "scripts/build_retb_step3_contracts.py", "--campaign-root",
            str(args.campaign_root),
        ])
        _run_python([
            "scripts/build_retb_step4_contracts.py", "--campaign-root",
            str(args.campaign_root),
        ])
        _run_python([
            "scripts/build_retb_step5_contracts.py", "--campaign-root",
            str(args.campaign_root),
        ])
        real_input_execution["architecture_contracts"] = {
            "step3_bundle_sha256": load_hashed_json(
                args.campaign_root / "registry" / "retb_step3_architecture_bundle.json"
            )["content_hash"],
            "step4_bundle_sha256": load_hashed_json(
                args.campaign_root / "registry" / "retb_step4_offline_expert_bundle.json"
            )["content_hash"],
            "step5_bundle_sha256": load_hashed_json(
                args.campaign_root / "registry" / "retb_step5_offline_fusion_bundle.json"
            )["content_hash"],
        }
    elif args.phase_id == "b_expert":
        _run_python([
            "scripts/probe_retb_resources.py", "--campaign-root",
            str(args.campaign_root), "--resource-kind", "gpu",
            "--compiled-region-parity",
            str(args.campaign_root / "backend" / "backend_manifest.json"),
            "--requested-memory-bytes", str(64 * 2**30), "--output",
            str(args.campaign_root / "job_ledgers" / "resource_probes" / "gpu.json"),
        ])
        gpu_probe_path = (
            args.campaign_root / "job_ledgers" / "resource_probes" / "gpu.json"
        )
        gpu_probe = load_hashed_json(gpu_probe_path)
        real_input_execution = {
            "resource_probe": {
                "content_hash": gpu_probe["content_hash"],
                "file_sha256": _file_sha(gpu_probe_path),
            },
            "offline_expert": _run_authenticated_manifest(
                args.campaign_root, node_id="offline_expert_training",
                task_indices=(0,), attest_complete=False,
            )
        }
    elif args.phase_id == "d_native":
        _run_python([
            "scripts/build_retb_step6_contracts.py", "--campaign-root",
            str(args.campaign_root),
            "--shared-hlt-normalizer",
            str(
                args.campaign_root / "inputs" / "normalization"
                / "hlt_shared_500k" / "relation.json"
            ),
        ])
        step6 = load_hashed_json(
            args.campaign_root / "registry" / "retb_step6_native_hlt_bundle.json"
        )
        real_input_execution = {
            "architecture_contracts": {
                "step6_bundle_sha256": step6["content_hash"],
            },
            "offline_parent_alias": _prepare_native_smoke_parent(args.campaign_root),
            "native_hlt_expert": _run_authenticated_manifest(
                args.campaign_root, node_id="native_hlt_expert_training",
                task_indices=(0,), attest_complete=False,
            ),
        }

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
        execution = _tiny_gpu_step(
            seed,
            require_cuda=phase["resource"] == "gpu",
            phase_id=args.phase_id,
        )
    workspace_removed = not workspace.exists()
    if not workspace_removed:
        raise RuntimeError("compact smoke task-local workspace survived")

    phase_control_evidence = build_streamed_smoke_phase_control_evidence(
        phase_id=args.phase_id,
        previous_phase_sha256=previous_hash,
        split_manifest_sha256=split_sha,
        production_graph_sha256=graph["content_hash"],
        execution_logit_sha256=execution["logit_checksum"],
    )
    final_seal = phase_control_evidence.get("final_test_seal")
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
        "execution": execution, "real_input_execution": real_input_execution,
        "phase_control_evidence": phase_control_evidence,
        "final_test_seal": final_seal,
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
