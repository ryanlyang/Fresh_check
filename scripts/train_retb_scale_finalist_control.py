#!/usr/bin/env python3
"""Train one post-lock, scale-population RETB HLT capacity control."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.train_retb_native_hlt_expert import _labels  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.capacity import (  # noqa: E402
    select_monolithic_capacity_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_capacity_controls import (  # noqa: E402
    build_hlt_capacity_control_model,
    build_hlt_capacity_control_row,
    publish_hlt_capacity_control_export,
    validate_hlt_capacity_control_row,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    NativeHLTExpertDataset,
    make_native_hlt_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    MonolithicBase4ParticleTransformer,
    analytical_particle_transformer_flops,
    build_monolithic_grid,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (  # noqa: E402
    OfflineCapacityTrainingConfig,
    build_capacity_profile,
    train_offline_capacity_model,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    VARIANT_LOSS_WEIGHTS,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_losses import (  # noqa: E402
    FIXED_WEIGHTS,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.capacity import (  # noqa: E402
    pair_encoder_flops,
)


CONTROL_KINDS = ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG")
BASE_CONFIGURATION = (128, 4, 8, 8, 2)
LABEL_LEDGER_CONTRACT = "retb_hlt_long_exposure_ledger_v2"


def _hlt_cache(root: Path, split: str, replica: int) -> Path:
    return (
        root
        / "inputs"
        / "hlt_v3"
        / split
        / f"replica_{replica}"
        / ("R_MULTI" if split == "scale_train" else "R_FIXED")
        / "D_NOMINAL"
    )


def _dataset(root: Path, split: str) -> NativeHLTExpertDataset:
    replicas = range(4) if split == "scale_train" else (0,)
    arrays, metadata = {}, {}
    for replica in replicas:
        arrays[replica], metadata[replica] = load_hlt_v3_cache(
            _hlt_cache(root, split, replica)
        )
    labels, identities = _labels(
        root / "inputs" / "offline" / split / "offline_inputs.npz"
    )
    return NativeHLTExpertDataset(
        replica_arrays=arrays,
        replica_metadata=metadata,
        labels=labels,
        identities=identities,
        logical_role=split,
        realization_policy=(
            "R_MULTI" if split == "scale_train" else "R_FIXED"
        ),
    )


def _capacity_selection(
    *, target: Mapping[str, Any], weaver: Any
) -> dict[str, Any]:
    candidates = []
    for configuration in build_monolithic_grid():
        model = MonolithicBase4ParticleTransformer(
            configuration, weaver_module=weaver
        )
        flops = (
            analytical_particle_transformer_flops(
                configuration=configuration
            )
            + pair_encoder_flops(4, (64, 64, 64))
        )
        candidates.append(
            {
                "configuration": list(configuration),
                "parameter_count": sum(
                    int(value.numel()) for value in model.parameters()
                ),
                "inference_flops_batch1": int(flops),
                "inference_flops_batch128": 128 * int(flops),
            }
        )
        del model
    return select_monolithic_capacity_controls(
        target_parameters=int(target["parameter_count"]),
        target_flops_batch1=float(target["analytical_flops_batch1"]),
        target_flops_batch128=float(target["analytical_flops_batch128"]),
        candidates=candidates,
        domain="hlt",
        target_complete_graph_sha256=target[
            "complete_graph_capacity_sha256"
        ],
    )


def _updates_and_presentations(path: Path) -> tuple[int, int] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_content_hash(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    counts = payload.get(
        "optimizer_update_counts",
        payload.get("planned_update_counts", {}),
    )
    rows = payload.get("rows", [])
    row_updates = [
        int(
            row.get(
                "optimizer_updates_completed",
                row.get("optimizer_update_ordinal", 0),
            )
        )
        for row in rows
        if isinstance(row, Mapping)
    ]
    updates = int(
        payload.get(
            "optimizer_updates_completed",
            counts.get(
                "total_optimizer_updates",
                max(row_updates, default=0),
            ),
        )
    )
    presentations = int(payload.get("labeled_example_presentations", 0))
    if presentations <= 0 and isinstance(rows, list):
        presentations = sum(
            int(row.get("training_examples_presented", 0))
            for row in rows
            if isinstance(row, Mapping)
        )
    if (
        updates <= 0
        or payload.get("fixed_budget_completed") is False
        or payload.get("fixed_epoch_budget_completed") is False
    ):
        return None
    return updates, presentations


def _long_exposure_ledger(
    *,
    root: Path,
    owner_graph_id: str,
    seed: int,
    scale_train_events: int,
) -> dict[str, Any]:
    finalist = (
        root / "runs" / "scale" / "graphs" / owner_graph_id / f"seed_{seed}"
    )
    seed_root = root / "runs" / "scale" / "refits" / f"seed_{seed}"
    excluded = {
        "offline_experts",
        "offline_fusion",
        "target_cache",
        "target_normalizers",
        "teachers",
        "coordinates",
        "uncertainty",
        "uncertainty_calibrations",
        "calibration",
    }
    included_markers = {
        "native_hlt",
        "native_fusion",
        "predictors",
        "joint",
        "token_refiner",
        "selected_refiner",
        "adapter",
        "final_consumer",
    }
    phase_payloads: dict[Path, list[tuple[Path, Mapping[str, Any]]]] = {}
    for parent in (seed_root, finalist):
        for path in sorted(parent.rglob("*.json")):
            lowered = {part.lower() for part in path.parts}
            if lowered & excluded or not lowered & included_markers:
                continue
            try:
                payload = load_hashed_json(path)
            except (OSError, ValueError, TypeError):
                continue
            phase_payloads.setdefault(path.parent.resolve(), []).append(
                (path, payload)
            )

    candidates = {}
    for phase_root, payload_rows in phase_payloads.items():
        for path, payload in payload_rows:
            parsed = _updates_and_presentations(path)
            if parsed is None:
                continue
            updates, presentations = parsed
            priority = 2 if isinstance(payload.get("rows"), list) else 1
            previous = candidates.get(phase_root)
            if previous is not None and previous[0] >= priority:
                continue
            candidates[phase_root] = (
                priority,
                path,
                payload,
                updates,
                presentations,
            )

    def ce_evidence(
        path: Path, payloads: list[tuple[Path, Mapping[str, Any]]]
    ) -> dict[str, Any]:
        parts = {part.lower() for part in path.parts}
        if parts & {"native_hlt", "native_fusion", "final_consumer", "token_refiner", "adapter"}:
            return {
                "eligible": True,
                "basis": "phase_contract_has_nonzero_ground_truth_CE",
            }
        if "predictors" in parts:
            objectives = {
                str(payload["objective_id"])
                for _, payload in payloads
                if payload.get("objective_id") is not None
            }
            if len(objectives) != 1:
                raise ValueError(
                    f"predictor CE eligibility is ambiguous: {path}"
                )
            objective = next(iter(objectives))
            weights = (
                FIXED_WEIGHTS["W_CANONICAL"]
                if objective == "W_GRADNORM"
                else FIXED_WEIGHTS.get(objective)
            )
            if weights is None:
                raise ValueError("predictor objective is unregistered")
            return {
                "eligible": float(weights[-1]) > 0.0,
                "basis": "serialized_predictor_objective_CE_weight",
                "objective_id": objective,
                "ground_truth_CE_weight": float(weights[-1]),
            }
        if "joint" in parts:
            variants = {
                str(payload["variant"])
                for _, payload in payloads
                if payload.get("variant") is not None
            }
            if len(variants) != 1:
                raise ValueError(
                    f"joint CE eligibility is ambiguous: {path}"
                )
            variant = next(iter(variants))
            weights = VARIANT_LOSS_WEIGHTS.get(variant)
            if weights is None:
                raise ValueError("joint variant is unregistered")
            direct = float(weights["fused_CE"]) + float(
                weights["native_HLT_CE"]
            )
            return {
                "eligible": direct > 0.0,
                "basis": "serialized_joint_variant_direct_CE_weights",
                "variant": variant,
                "ground_truth_CE_weight": direct,
            }
        raise ValueError(f"HLT exposure phase is unclassified: {path}")

    rows, excluded_rows = [], []
    for _, path, payload, updates, presentations in sorted(
        candidates.values(), key=lambda item: str(item[1])
    ):
        # Some trainers record update counts but omit a presentation total.
        # Infer it from the fixed effective batch and authenticated population.
        if presentations <= 0:
            effective_batch = int(
                payload.get("config", {}).get(
                    "effective_batch_size", 128
                )
            )
            updates_per_epoch = math.ceil(
                int(scale_train_events) / effective_batch
            )
            complete_epochs, partial_updates = divmod(
                updates, updates_per_epoch
            )
            presentations = (
                complete_epochs * int(scale_train_events)
                + min(
                    partial_updates * effective_batch,
                    int(scale_train_events),
                )
            )
        evidence = ce_evidence(
            path.parent.resolve(), phase_payloads[path.parent.resolve()]
        )
        row = {
            "component_path": str(path.resolve()),
            "component_sha256": payload["content_hash"],
            "optimizer_updates": updates,
            "labeled_example_presentations": presentations,
            "ground_truth_CE_evidence": evidence,
        }
        (rows if evidence["eligible"] else excluded_rows).append(row)
    if not rows:
        raise ValueError("H_BASE_LONG found no HLT-side training phases")
    total = sum(row["labeled_example_presentations"] for row in rows)
    return with_content_hash(
        {
            "contract": LABEL_LEDGER_CONTRACT,
            "schema_version": 2,
            "owner_finalist_graph_id": owner_graph_id,
            "pipeline_seed": int(seed),
            "component_rows": rows,
            "excluded_zero_CE_component_rows": excluded_rows,
            "total_labeled_example_presentations": total,
            "effective_batch_size": 128,
            "optimizer_update_budget": math.ceil(total / 128),
            "rounding": "ceil",
            "pure_offline_KD_and_calibration_phases_excluded": True,
            "every_included_phase_has_proven_nonzero_ground_truth_CE": True,
            "performance_used_to_set_budget": False,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-finalists", required=True, type=Path)
    parser.add_argument("--owner-finalist-graph-id", required=True)
    parser.add_argument("--control-kind", required=True, choices=CONTROL_KINDS)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    output = args.output_dir.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    if (
        finalists.get("source") != campaign.get("source")
        or args.owner_finalist_graph_id
        not in finalists["finalist_graph_ids"]
        or args.pipeline_seed not in {101, 202, 303}
    ):
        raise ValueError("scale finalist-control lineage differs")
    completed_path = output / "control_row.json"
    if completed_path.is_file():
        completed = load_hashed_json(completed_path)
        validate_hlt_capacity_control_row(completed)
        export = load_hashed_json(output / "deployable_control.json")
        if (
            completed.get("source") != campaign["source"]
            or export.get("source") != campaign["source"]
            or completed["owner_finalist_graph_id"]
            != args.owner_finalist_graph_id
            or completed["control_kind"] != args.control_kind
            or int(completed["pipeline_seed"]) != args.pipeline_seed
            or completed["deployable_export_sha256"]
            != export["content_hash"]
        ):
            raise ValueError("reusable scale finalist-control differs")
        print(json.dumps(completed, indent=2, sort_keys=True))
        return 0
    for resource in ("scale_train", "val_stop", "val_design"):
        authorize_dataset_access(
            worker_role=(
                "design_worker"
                if resource == "val_design"
                else "scale_training_worker"
            ),
            requested_resource=resource,
        )
    completion = load_hashed_json(root / "selection" / "scale_completion.json")
    scale_row = next(
        row
        for row in completion["runs"]
        if row["graph_id"] == args.owner_finalist_graph_id
        and int(row["pipeline_seed"]) == args.pipeline_seed
    )
    capacity_path = (
        root
        / "runs"
        / "scale"
        / "graphs"
        / args.owner_finalist_graph_id
        / f"seed_{args.pipeline_seed}"
        / "export"
        / "complete_graph_capacity.json"
    )
    capacity = load_hashed_json(capacity_path)
    target = {
        **scale_row["capacity"],
        "complete_graph_capacity_sha256": capacity["content_hash"],
    }
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    scale_train = _dataset(root, "scale_train")
    val_stop = _dataset(root, "val_stop")
    val_design = _dataset(root, "val_design")
    miniature = campaign["campaign_profile"] == "miniature_test"

    if args.control_kind == "H_BASE_LONG":
        ledger = bind_source(
            _long_exposure_ledger(
                root=root,
                owner_graph_id=args.owner_finalist_graph_id,
                seed=args.pipeline_seed,
                scale_train_events=len(scale_train),
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        selection = ledger
        configuration = None
        exact_updates = (
            min(4, int(ledger["optimizer_update_budget"]))
            if miniature
            else int(ledger["optimizer_update_budget"])
        )
        write_immutable_json(output / "label_exposure_ledger.json", ledger)
        flops = (
            analytical_particle_transformer_flops(
                configuration=BASE_CONFIGURATION
            )
            + pair_encoder_flops(4, (64, 64, 64))
        )
    else:
        selection = bind_source(
            _capacity_selection(target=target, weaver=weaver),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        configuration = selection[args.control_kind]["configuration"]
        exact_updates = None
        flops = int(selection[args.control_kind]["inference_flops_batch1"])
        write_immutable_json(
            output / "capacity_control_selection.json", selection
        )
    model = build_hlt_capacity_control_model(
        control_kind=args.control_kind,
        configuration=configuration,
        weaver_module=weaver,
    )
    profile = build_capacity_profile(
        control_id=args.control_kind,
        model=model,
        analytical_flops_batch1=int(flops),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "complete_graph_profile.json", profile)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    registration = train_offline_capacity_model(
        model=model,
        train_loader=make_native_hlt_expert_loader(
            scale_train,
            seed=args.pipeline_seed,
            training=True,
            batch_size=args.microbatch_size,
        ),
        val_stop_loader=make_native_hlt_expert_loader(
            val_stop, seed=0, training=False, batch_size=args.microbatch_size
        ),
        val_design_loader=make_native_hlt_expert_loader(
            val_design,
            seed=0,
            training=False,
            batch_size=args.microbatch_size,
        ),
        output_dir=output / "training",
        config=OfflineCapacityTrainingConfig(
            control_id=args.control_kind,
            seed=args.pipeline_seed,
            maximum_epochs=2 if miniature else 40,
            microbatch_size=args.microbatch_size,
            gradient_accumulation_steps=2,
            campaign_profile=(
                "miniature_test" if miniature else "production"
            ),
            exact_optimizer_update_budget=exact_updates,
        ),
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        execution_registry_sha256=selection["content_hash"],
        lineage_hashes={
            "campaign_spec": campaign["content_hash"],
            "locked_scale_finalists": finalists["content_hash"],
            "scale_completion": completion["content_hash"],
            "complete_graph_capacity": capacity["content_hash"],
            "control_selection_or_ledger": selection["content_hash"],
        },
        profile=profile,
        source_snapshot=source_snapshot(REPO_ROOT),
        device=device,
    )
    checkpoint = output / "training" / "best_model_val.pt"
    export = publish_hlt_capacity_control_export(
        output=output / "deployable_control.json",
        owner_finalist_graph_id=args.owner_finalist_graph_id,
        control_kind=args.control_kind,
        pipeline_seed=args.pipeline_seed,
        configuration=configuration,
        checkpoint_path=checkpoint,
        checkpoint_sha256=registration["checkpoint_sha256"],
        training_registration_sha256=registration["content_hash"],
        capacity_selection_sha256=selection["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    row = bind_source(
        build_hlt_capacity_control_row(
            owner_finalist_graph_id=args.owner_finalist_graph_id,
            control_kind=args.control_kind,
            pipeline_seed=args.pipeline_seed,
            checkpoint_sha256=registration["checkpoint_sha256"],
            deployable_export_sha256=export["content_hash"],
            training_registration_sha256=registration["content_hash"],
            optimizer_updates_completed=registration[
                "optimizer_updates_completed"
            ],
            labeled_example_presentations=registration[
                "labeled_example_presentations"
            ],
            capacity_selection_sha256=selection["content_hash"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(output / "control_row.json", row)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
