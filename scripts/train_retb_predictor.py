#!/usr/bin/env python3
"""Train one fixed-budget RETB Stage-F/G/H token predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    validate_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (  # noqa: E402
    FUSION_CHECKPOINT_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_training import (  # noqa: E402
    PredictorDataset,
    PredictorTrainingConfig,
    make_predictor_loader,
    evaluate_predictor,
    train_predictor,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (  # noqa: E402
    calibrate_predictor_inference_cache,
    publish_predictor_inference_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.predictors import (  # noqa: E402
    RetbTokenPredictor,
    build_predictor_capacity_report,
    predictor_analytical_flops,
    profile_predictor,
    uncertainty_width,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step9 import (  # noqa: E402
    PREDICTOR_RUN_CONTRACT,
    validate_materialized_predictor_run,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    TARGET_NORMALIZER_CONTRACT,
    load_frozen_token_head,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)

import torch  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return arrays


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate_normalizer(
    normalizer: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    validate_content_hash(
        normalizer, expected_contract=TARGET_NORMALIZER_CONTRACT
    )
    mean = np.asarray(normalizer.get("mean"), dtype=np.float32)
    standard_deviation = np.asarray(
        normalizer.get("standard_deviation"), dtype=np.float32
    )
    if (
        normalizer.get("content_hash")
        != run["parent_hashes"]["target_normalizer"]
        or normalizer.get("source") != campaign.get("source")
        or normalizer.get("expert_id") != run["expert_id"]
        or normalizer.get("target_mode") != run["target_mode"]
        or int(normalizer.get("pipeline_seed", -1))
        != int(run["pipeline_seed"])
        or normalizer.get("shape_id") != run["shape_id"]
        or normalizer.get("fit_split") != "model_train"
        or float(normalizer.get("standard_deviation_floor", -1.0)) != 1.0e-4
        or mean.shape
        != (int(run["token_count"]), int(run["token_dimension"]))
        or standard_deviation.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(standard_deviation).all()
        or bool((standard_deviation < 0).any())
    ):
        raise ValueError("predictor target normalizer semantics differ")
    return mean, standard_deviation


def _validate_fusion_checkpoint(
    payload: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
) -> Mapping[str, Any]:
    allocation = payload.get("allocation")
    if (
        payload.get("contract") != FUSION_CHECKPOINT_CONTRACT
        or int(payload.get("schema_version", -1)) != 1
        or not isinstance(payload.get("model_state_dict"), Mapping)
        or not isinstance(allocation, Mapping)
        or set(allocation) != set(EXPERT_ORDER)
        or payload.get("shape_id") != run["shape_id"]
        or list(allocation.get(run["expert_id"], ()))
        != [int(run["token_count"]), int(run["token_dimension"])]
    ):
        raise ValueError("predictor frozen-fusion checkpoint semantics differ")
    for expert in EXPERT_ORDER:
        shape = list(allocation[expert])
        if (
            len(shape) != 2
            or int(shape[0]) not in {1, 2, 4, 8, 16}
            or int(shape[1]) not in {64, 128}
        ):
            raise ValueError("predictor frozen-fusion allocation differs")
    return allocation


def _dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    split: str,
    run: Mapping[str, Any],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    lineage_hashes: Mapping[str, str],
) -> PredictorDataset:
    required = {
        "identities",
        "labels",
        "unbiased_particle_states",
        "particle_mask",
        "target_tokens",
        "target_expert_logits",
        "target_hybrid_logits",
        "offline_slot_queries",
    } | {
        f"hlt_tokens_{expert}" for expert in EXPERT_ORDER
    } | {
        f"oracle_tokens_{expert}" for expert in EXPERT_ORDER
    }
    relation_fields = {
        f"{kind}_{expert}"
        for expert in ("PT", "TRACK", "REGION")
        for kind in ("relation_particle_states", "relation_particle_mask")
    }
    actual_fields = set(arrays)
    if actual_fields != required and actual_fields != required | relation_fields:
        raise ValueError("prepared predictor NPZ fields differ")
    relation_present = relation_fields.issubset(arrays)
    expert = run["expert_id"]
    return PredictorDataset(
        identities=[str(value) for value in arrays["identities"].tolist()],
        labels=arrays["labels"],
        hlt_token_banks={
            name: arrays[f"hlt_tokens_{name}"] for name in EXPERT_ORDER
        },
        unbiased_particle_states=arrays["unbiased_particle_states"],
        particle_mask=arrays["particle_mask"],
        relation_particle_states=(
            None
            if not relation_present
            else {
                name: arrays[f"relation_particle_states_{name}"]
                for name in ("PT", "TRACK", "REGION")
            }
        ),
        relation_particle_masks=(
            None
            if not relation_present
            else {
                name: arrays[f"relation_particle_mask_{name}"]
                for name in ("PT", "TRACK", "REGION")
            }
        ),
        target_tokens=arrays["target_tokens"],
        target_expert_logits=arrays["target_expert_logits"],
        target_hybrid_logits=arrays["target_hybrid_logits"],
        other_oracle_banks={
            name: arrays[f"oracle_tokens_{name}"]
            for name in EXPERT_ORDER
            if name != expert
        },
        target_expert_id=expert,
        token_mean=mean,
        token_standard_deviation=standard_deviation,
        normalization_mode=run["normalization_mode"],
        split=split,
        lineage_hashes=lineage_hashes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--model-train", required=True, type=Path)
    parser.add_argument("--val-stop", required=True, type=Path)
    parser.add_argument("--val-design", type=Path)
    parser.add_argument("--target-normalizer", required=True, type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-variant", default="F_TOKEN_TRANSFORMER")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-design-output", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--microbatch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=PREDICTOR_RUN_CONTRACT
    )
    validate_materialized_predictor_run(run)
    if run.get("source") != campaign.get("source"):
        raise ValueError("predictor run source lineage differs")
    if (
        _sha256(args.target_checkpoint)
        != run["parent_hashes"]["offline_target_checkpoint"]
        or _sha256(args.fusion_checkpoint)
        != run["parent_hashes"]["offline_fusion"]
    ):
        raise ValueError("predictor frozen-consumer checkpoint hashes differ")
    normalizer = load_hashed_json(args.target_normalizer)
    mean, standard_deviation = _validate_normalizer(
        normalizer, run=run, campaign=campaign
    )
    train_arrays = _load_arrays(args.model_train)
    val_arrays = _load_arrays(args.val_stop)
    design_flags = (
        args.val_design is not None,
        args.val_design_output is not None,
        args.calibration_output is not None,
    )
    if any(design_flags) and not all(design_flags):
        raise ValueError(
            "val-design data, output, and calibration output are all required together"
        )
    queries = np.asarray(train_arrays["offline_slot_queries"], dtype=np.float32)
    if not np.array_equal(queries, val_arrays["offline_slot_queries"]):
        raise ValueError("predictor slot queries drifted across splits")
    if queries.shape != (run["token_count"], run["token_dimension"]):
        raise ValueError("predictor slot-query shape differs")
    if _array_sha256(queries) != run["parent_hashes"]["slot_queries"]:
        raise ValueError("predictor slot-query content hash differs")
    resolved = {
        "run_id": run["run_id"],
        "run_sha256": run["content_hash"],
        "model_train_sha256": _sha256(args.model_train),
        "val_stop_sha256": _sha256(args.val_stop),
        "output_dir": str(args.output_dir.resolve()),
        "dry_run": args.dry_run,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    lineage = {
        **run["parent_hashes"],
        "prepared_model_train": resolved["model_train_sha256"],
        "prepared_val_stop": resolved["val_stop_sha256"],
    }
    train_dataset = _dataset(
        train_arrays,
        split="model_train",
        run=run,
        mean=mean,
        standard_deviation=standard_deviation,
        lineage_hashes=lineage,
    )
    val_dataset = _dataset(
        val_arrays,
        split="val_stop",
        run=run,
        mean=mean,
        standard_deviation=standard_deviation,
        lineage_hashes=lineage,
    )
    effective_batch = (
        args.microbatch_size * args.gradient_accumulation_steps
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    config = PredictorTrainingConfig(
        seed=int(run["pipeline_seed"]),
        architecture=run["architecture"],
        context=run["context"],
        objective_id=run["objective_id"],
        uncertainty_head=run["uncertainty_head"],
        normalization_mode=run["normalization_mode"],
        learning_rate=float(run["learning_rate"]),
        dropout=float(run["dropout"]),
        maximum_epochs=2 if miniature else 40,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=effective_batch,
        campaign_profile="miniature_test" if miniature else "production",
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = RetbTokenPredictor(
        architecture=run["architecture"],
        context=run["context"],
        target_expert_id=run["expert_id"],
        token_count=int(run["token_count"]),
        token_dimension=int(run["token_dimension"]),
        offline_slot_queries=torch.from_numpy(queries),
        uncertainty_head=run["uncertainty_head"],
        dropout=float(run["dropout"]),
    )
    head = load_frozen_token_head(
        checkpoint_path=args.target_checkpoint,
        expected_checkpoint_sha256=run["parent_hashes"][
            "offline_target_checkpoint"
        ],
        target_mode=run["target_mode"],
        token_dimension=int(run["token_dimension"]),
    )
    fusion_payload = torch.load(
        args.fusion_checkpoint, map_location="cpu", weights_only=False
    )
    allocation = _validate_fusion_checkpoint(fusion_payload, run=run)
    fusion = build_fusion_model(
        args.fusion_variant,
        bank_dimensions={
            expert: int(allocation[expert][1]) for expert in EXPERT_ORDER
        },
    )
    fusion.load_state_dict(fusion_payload["model_state_dict"], strict=True)
    registration = train_predictor(
        model=model,
        train_loader=make_predictor_loader(
            train_dataset,
            batch_size=args.microbatch_size,
            seed=config.seed,
            training=True,
        ),
        val_stop_loader=make_predictor_loader(
            val_dataset,
            batch_size=args.microbatch_size,
            seed=config.seed,
            training=False,
        ),
        frozen_expert_head=head,
        frozen_fusion=fusion,
        token_mean=mean,
        token_standard_deviation=standard_deviation,
        output_dir=args.output_dir,
        run_record=run,
        step9_bundle_sha256=run["parent_hashes"]["step9_bundle"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        lineage_hashes=lineage,
        config=config,
        device=(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        ),
    )
    write_immutable_json(args.output_dir / "worker_registration.json", registration)
    selected = torch.load(
        args.output_dir / "best_model_val.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(selected["model_state_dict"], strict=True)
    resolved_device = next(model.parameters()).device
    raw_profile_batch = next(
        iter(
            make_predictor_loader(
                val_dataset,
                batch_size=min(args.microbatch_size, len(val_dataset)),
                seed=config.seed,
                training=False,
            )
        )
    )
    profile_kwargs = {
        "corresponding_hlt_tokens": raw_profile_batch["hlt_token_banks"][
            run["expert_id"]
        ].to(resolved_device),
        "hlt_token_banks": {
            name: value.to(resolved_device)
            for name, value in raw_profile_batch["hlt_token_banks"].items()
        },
        "unbiased_particle_states": raw_profile_batch[
            "unbiased_particle_states"
        ].to(resolved_device),
        "particle_mask": raw_profile_batch["particle_mask"].to(
            resolved_device
        ),
        "relation_particle_states": (
            None
            if raw_profile_batch["relation_particle_states"] is None
            else {
                name: value.to(resolved_device)
                for name, value in raw_profile_batch[
                    "relation_particle_states"
                ].items()
            }
        ),
        "relation_particle_masks": (
            None
            if raw_profile_batch["relation_particle_masks"] is None
            else {
                name: value.to(resolved_device)
                for name, value in raw_profile_batch[
                    "relation_particle_masks"
                ].items()
            }
        ),
    }
    evidence_count = (
        0
        if run["context"] == "C3_ALL_PARTICLE"
        else int(run["token_count"])
    )
    evidence_projection_flops = 0
    profile_batch_size = len(raw_profile_batch["identities"])
    target_dimension = int(run["token_dimension"])
    if run["context"] != "C3_ALL_PARTICLE":
        corresponding = raw_profile_batch["hlt_token_banks"][run["expert_id"]]
        evidence_projection_flops += (
            2
            * profile_batch_size
            * int(corresponding.shape[1])
            * int(corresponding.shape[2])
            * target_dimension
        )
    if run["context"] in {"C2_ALL", "C3_ALL_PARTICLE"}:
        evidence_count += sum(
            int(value.shape[1])
            for value in raw_profile_batch["hlt_token_banks"].values()
        )
        evidence_projection_flops += sum(
            2
            * profile_batch_size
            * int(value.shape[1])
            * int(value.shape[2])
            * target_dimension
            for value in raw_profile_batch["hlt_token_banks"].values()
        )
    if run["context"] in {"C1_NATIVE", "C2_ALL", "C3_ALL_PARTICLE"}:
        evidence_count += int(
            raw_profile_batch["unbiased_particle_states"].shape[1]
        )
        base_states = raw_profile_batch["unbiased_particle_states"]
        evidence_projection_flops += (
            2
            * profile_batch_size
            * int(base_states.shape[1])
            * int(base_states.shape[2])
            * target_dimension
        )
    if run["context"] == "C3_ALL_PARTICLE":
        evidence_count += sum(
            int(value.shape[1])
            for value in raw_profile_batch["relation_particle_states"].values()
        )
        evidence_projection_flops += sum(
            2
            * profile_batch_size
            * int(value.shape[1])
            * int(value.shape[2])
            * target_dimension
            for value in raw_profile_batch[
                "relation_particle_states"
            ].values()
        )
    analytical = predictor_analytical_flops(
        architecture=run["architecture"],
        batch_size=len(raw_profile_batch["identities"]),
        token_count=int(run["token_count"]),
        token_dimension=int(run["token_dimension"]),
        evidence_token_count=evidence_count,
        uncertainty_width_value=uncertainty_width(
            run["uncertainty_head"], int(run["token_dimension"])
        ),
        residual_hidden_width=model.residual_hidden_width,
        evidence_projection_flops=evidence_projection_flops,
    )
    selected_profile = profile_predictor(
        model, forward_kwargs=profile_kwargs, analytical_flops=analytical
    )
    affine_baseline = RetbTokenPredictor(
        architecture="A0_AFFINE",
        context="C0_SELF",
        target_expert_id=run["expert_id"],
        token_count=int(run["token_count"]),
        token_dimension=int(run["token_dimension"]),
        offline_slot_queries=torch.from_numpy(queries),
        uncertainty_head=run["uncertainty_head"],
        dropout=0.0,
    ).to(resolved_device)
    affine_baseline(
        corresponding_hlt_tokens=profile_kwargs[
            "corresponding_hlt_tokens"
        ]
    )
    affine_baseline_parameter_count = sum(
        parameter.numel() for parameter in affine_baseline.parameters()
    )
    zero_profile = None
    if run["architecture"] in {
        "A3_SLOT_DECODER_DIRECT",
        "A4_SLOT_DECODER_GATED",
    }:
        model.zero_evidence_control = True
        zero_profile = profile_predictor(
            model, forward_kwargs=profile_kwargs, analytical_flops=analytical
        )
        model.zero_evidence_control = False
    capacity = bind_source(
        build_predictor_capacity_report(
            run_id=run["run_id"],
            architecture=run["architecture"],
            token_dimension=int(run["token_dimension"]),
            selected_profile=selected_profile,
            affine_baseline_parameter_count=affine_baseline_parameter_count,
            zero_evidence_profile=zero_profile,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output_dir / "capacity_report.json", capacity)
    if args.val_design is not None:
        design_arrays = _load_arrays(args.val_design)
        if not np.array_equal(
            queries, design_arrays["offline_slot_queries"]
        ):
            raise ValueError("val-design slot queries differ")
        design_lineage = {
            **lineage,
            "prepared_val_design": _sha256(args.val_design),
        }
        design_dataset = _dataset(
            design_arrays,
            split="val_design",
            run=run,
            mean=mean,
            standard_deviation=standard_deviation,
            lineage_hashes=design_lineage,
        )
        inference = evaluate_predictor(
            model=model,
            loader=make_predictor_loader(
                design_dataset,
                batch_size=args.microbatch_size,
                seed=config.seed,
                training=False,
            ),
            frozen_expert_head=head,
            frozen_fusion=fusion,
            token_mean=mean,
            token_standard_deviation=standard_deviation,
            objective_id=config.objective_id,
            normalization_mode=config.normalization_mode,
            device=next(model.parameters()).device,
            gradnorm_weights=(
                None
                if selected.get("gradnorm_state") is None
                else selected["gradnorm_state"]["current"]
            ),
        )
        inference_manifest = publish_predictor_inference_cache(
            output_dir=args.val_design_output,
            split="val_design",
            pipeline_seed=config.seed,
            expert_id=run["expert_id"],
            uncertainty_head=run["uncertainty_head"],
            identities=inference["identities"],
            predicted_tokens=inference["predicted_original_tokens"],
            normalized_predicted_tokens=inference["predicted_tokens"],
            log_variance=inference["log_variance"],
            expert_logits=inference["expert_logits"],
            hybrid_logits=inference["hybrid_logits"],
            predictor_registration_sha256=registration["content_hash"],
            predictor_checkpoint_sha256=registration["checkpoint_sha256"],
            target_cache_manifest_sha256=run["parent_hashes"][
                "val_design_target_cache"
            ],
            target_normalizer_sha256=run["parent_hashes"][
                "target_normalizer"
            ],
            identity_manifest_sha256=run["parent_hashes"][
                "val_design_identity_manifest"
            ],
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        calibration = calibrate_predictor_inference_cache(
            manifest_path=(
                args.val_design_output / "predictor_outputs_manifest.json"
            ),
            expected_pipeline_seed=config.seed,
            expected_registration_sha256=registration["content_hash"],
            target_tokens=design_dataset.target_tokens,
            identity_order_sha256=inference_manifest[
                "identity_order_sha256"
            ],
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        write_immutable_json(args.calibration_output, calibration)
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
