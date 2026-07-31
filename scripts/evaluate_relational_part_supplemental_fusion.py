#!/usr/bin/env python3
"""Compare fixed control and relational logit fusions on RPT val_select."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CONTRACT = "relational_part_posthoc_fusion_comparison_v1"
CONTROL_RUN_IDS = (
    "RPT_BASE",
    "RPT_BASE_WIDE_MAX",
    "RPT_FULL_ZERO_REL",
)
RELATION_RUN_IDS = (
    "RPT_TRACK",
    "RPT_CHARGE",
    "RPT_PT",
)
PAIR_FUSIONS = {
    "RPT_TRACK_CHARGE_LOGIT_FUSION": ("RPT_TRACK", "RPT_CHARGE"),
    "RPT_TRACK_PT_LOGIT_FUSION": ("RPT_TRACK", "RPT_PT"),
    "RPT_CHARGE_PT_LOGIT_FUSION": ("RPT_CHARGE", "RPT_PT"),
}


def _center_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must be a two-dimensional matrix")
    if not np.isfinite(values).all():
        raise FloatingPointError("logits contain nonfinite values")
    return values - values.mean(axis=1, keepdims=True, dtype=np.float64)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(truth):
        raise ValueError("logits and labels have incompatible shapes")
    maximum = values.max(axis=1, keepdims=True)
    log_norm = np.log(
        np.exp(values - maximum).sum(axis=1, dtype=np.float64)
    ) + maximum[:, 0]
    return float(
        (log_norm - values[np.arange(len(truth)), truth]).mean(
            dtype=np.float64
        )
    )


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    minimum_inverse_temperature: float = 1e-3,
    maximum_inverse_temperature: float = 100.0,
    iterations: int = 100,
) -> dict[str, Any]:
    """Fit one convex inverse-temperature parameter by fixed bisection."""

    values = _center_logits(logits)
    truth = np.asarray(labels, dtype=np.int64)
    if values.shape[0] != len(truth) or not len(truth):
        raise ValueError("temperature fitting requires aligned nonempty arrays")
    if bool(((truth < 0) | (truth >= values.shape[1])).any()):
        raise ValueError("temperature labels are outside the logit class range")
    lower = float(minimum_inverse_temperature)
    upper = float(maximum_inverse_temperature)
    if not 0.0 < lower < upper or int(iterations) <= 0:
        raise ValueError("invalid inverse-temperature fit bounds")

    true_logits = values[np.arange(len(truth)), truth]

    def derivative(inverse_temperature: float) -> float:
        scaled = values * np.float64(inverse_temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights /= weights.sum(axis=1, keepdims=True, dtype=np.float64)
        expected = (weights * values).sum(axis=1, dtype=np.float64)
        return float((expected - true_logits).mean(dtype=np.float64))

    lower_derivative = derivative(lower)
    upper_derivative = derivative(upper)
    boundary = None
    if lower_derivative >= 0.0:
        fitted = lower
        boundary = "minimum_inverse_temperature"
    elif upper_derivative <= 0.0:
        fitted = upper
        boundary = "maximum_inverse_temperature"
    else:
        for _ in range(int(iterations)):
            midpoint = (lower + upper) / 2.0
            if derivative(midpoint) <= 0.0:
                lower = midpoint
            else:
                upper = midpoint
        fitted = (lower + upper) / 2.0
    temperature = 1.0 / fitted
    prefit = _cross_entropy(values, truth)
    postfit = _cross_entropy(values * fitted, truth)
    return {
        "method": "convex_inverse_temperature_fixed_bisection",
        "inverse_temperature": float(fitted),
        "temperature": float(temperature),
        "minimum_inverse_temperature": float(minimum_inverse_temperature),
        "maximum_inverse_temperature": float(maximum_inverse_temperature),
        "iterations": int(iterations),
        "boundary_solution": boundary,
        "val_stop_cross_entropy_before": prefit,
        "val_stop_cross_entropy_after": postfit,
        "fit_split": "val_stop",
    }


def fuse_equal_weight_logits(
    logits_by_run: Mapping[str, np.ndarray],
    run_ids: Sequence[str],
    *,
    inverse_temperatures: Mapping[str, float] | None = None,
) -> np.ndarray:
    if not run_ids:
        raise ValueError("fusion requires at least one member")
    rows = []
    expected_shape = None
    for run_id in run_ids:
        if run_id not in logits_by_run:
            raise KeyError(f"fusion logits are absent for {run_id}")
        values = _center_logits(logits_by_run[run_id])
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError("fusion member logits have different shapes")
        scale = (
            1.0
            if inverse_temperatures is None
            else float(inverse_temperatures[run_id])
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("fusion inverse temperatures must be positive")
        rows.append(values * np.float64(scale))
    return np.stack(rows, axis=0).mean(axis=0, dtype=np.float64)


def pairwise_diversity(
    predictions_by_run: Mapping[str, np.ndarray],
    labels: np.ndarray,
    run_ids: Sequence[str],
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    correct_rows = []
    pairs = []
    for run_id in run_ids:
        prediction = np.asarray(predictions_by_run[run_id], dtype=np.int64)
        if prediction.shape != truth.shape:
            raise ValueError("diversity predictions are not label-aligned")
        correct_rows.append(prediction == truth)
    for left_index, left_id in enumerate(run_ids):
        left = np.asarray(predictions_by_run[left_id], dtype=np.int64)
        left_wrong = left != truth
        for right_id in run_ids[left_index + 1 :]:
            right = np.asarray(predictions_by_run[right_id], dtype=np.int64)
            right_wrong = right != truth
            left_std = float(left_wrong.std(dtype=np.float64))
            right_std = float(right_wrong.std(dtype=np.float64))
            correlation = (
                None
                if left_std == 0.0 or right_std == 0.0
                else float(np.corrcoef(left_wrong, right_wrong)[0, 1])
            )
            pairs.append(
                {
                    "left_run_id": left_id,
                    "right_run_id": right_id,
                    "prediction_disagreement": float(
                        (left != right).mean(dtype=np.float64)
                    ),
                    "both_wrong_fraction": float(
                        (left_wrong & right_wrong).mean(dtype=np.float64)
                    ),
                    "exactly_one_correct_fraction": float(
                        (left_wrong ^ right_wrong).mean(dtype=np.float64)
                    ),
                    "error_indicator_pearson_correlation": correlation,
                }
            )
    correct = np.stack(correct_rows, axis=0)
    return {
        "member_run_ids": list(run_ids),
        "pairwise": pairs,
        "oracle_any_member_correct_accuracy": float(
            correct.any(axis=0).mean(dtype=np.float64)
        ),
        "all_members_correct_fraction": float(
            correct.all(axis=0).mean(dtype=np.float64)
        ),
        "all_members_wrong_fraction": float(
            (~correct).all(axis=0).mean(dtype=np.float64)
        ),
    }


def _move_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _collect_logits(
    model: Any,
    loader: Any,
    *,
    device: torch.device,
    model_forward: Any,
) -> dict[str, Any]:
    logits = []
    labels = []
    identities: list[str] = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            output = model_forward(model, batch)
            logits.append(output.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().long().cpu().numpy())
            identities.extend(str(value) for value in raw["event_identities"])
    if not logits:
        raise ValueError("fusion evaluation loader is empty")
    return {
        "logits": np.concatenate(logits, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "event_identities": np.asarray(identities, dtype=np.str_),
    }


def _qcd_rejection_at_75(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
    qcd_signal_rejection: Any,
) -> dict[str, Any]:
    return {
        class_names[index]: qcd_signal_rejection(
            logits,
            labels,
            signal_index=index,
            target_efficiency=0.75,
        )
        for index in range(1, len(class_names))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-source-status-drift",
        action="store_true",
        help=(
            "For this post-hoc diagnostic only, require the campaign commit "
            "but permit working-tree status drift such as Python caches."
        ),
    )
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    campaign_root = args.campaign_root.resolve()
    if not (source_root / "teacher_logit_reco" / "relational_part").is_dir():
        raise FileNotFoundError(f"relational-part source is absent: {source_root}")
    sys.path.insert(0, str(source_root))

    from jetclass_fresh.hlt_cache import load_cached_hlt_view
    from teacher_logit_reco.relational_part import (
        CLASS_NAMES,
        GLOBAL_DETERMINISM_CONTRACT,
        RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
        REGION_NORMALIZATION_CONTRACT,
        SCREENING_REGISTRY_CONTRACT,
        RelationalJetDataset,
        build_runtime_model,
        canonical_sha256,
        evaluate_logits,
        load_hashed_json,
        load_region_tree_split,
        make_relational_loader,
        paired_prediction_statistics_many,
        qcd_signal_rejection,
        sha256_file,
        validate_campaign_source,
        validate_content_hash,
        with_content_hash,
        write_immutable_json,
    )
    from teacher_logit_reco.relational_part.evaluation import model_forward
    from teacher_logit_reco.relational_part.provenance import source_snapshot

    campaign = load_hashed_json(campaign_root / "campaign_spec.json")
    if args.allow_source_status_drift:
        observed_source = source_snapshot(source_root)
        if observed_source["source_commit"] != campaign["source"]["commit"]:
            raise ValueError(
                "supplemental fusion source commit differs from campaign"
            )
        source_validation_mode = "commit_exact_status_drift_allowed"
    else:
        observed_source = validate_campaign_source(
            campaign, repo_root=source_root
        )
        source_validation_mode = "commit_and_status_exact"
    determinism = load_hashed_json(
        campaign_root / "registry" / "global_determinism.json",
        expected_contract=GLOBAL_DETERMINISM_CONTRACT,
    )
    bootstrap = determinism["paired_bootstrap"]
    if (
        int(bootstrap["replicates"]) != 10_000
        or int(bootstrap["seed"]) != 917_301
    ):
        raise ValueError("campaign paired-bootstrap policy drifted")
    screening = load_hashed_json(
        campaign_root / "registry" / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    normalization = load_hashed_json(
        campaign_root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region_path = campaign_root / "inputs" / "region_normalization.json"
    region = (
        load_hashed_json(
            region_path, expected_contract=REGION_NORMALIZATION_CONTRACT
        )
        if region_path.is_file()
        else None
    )

    run_ids = (*CONTROL_RUN_IDS, *RELATION_RUN_IDS)
    registrations = {}
    model_contracts = {}
    official_metrics = {}
    checkpoint_paths = {}
    for run_id in run_ids:
        run_root = campaign_root / "runs" / run_id / f"seed_{args.seed}"
        registration = load_hashed_json(
            run_root / "checkpoint_registration.json",
            expected_contract="relational_part_checkpoint_registration_v2",
        )
        model_contract = load_hashed_json(
            campaign_root
            / "registry"
            / "model_contracts"
            / f"{run_id}.json"
        )
        metrics = load_hashed_json(
            run_root / "val_select_metrics.json",
            expected_contract="relational_part_evaluation_v1",
        )
        checkpoint_path = run_root / str(registration["checkpoint_file"])
        if registration.get("run_id") != run_id or int(
            registration.get("seed", -1)
        ) != int(args.seed):
            raise ValueError(f"checkpoint registration mismatch for {run_id}")
        if registration.get("model_contract_sha256") != model_contract[
            "content_hash"
        ]:
            raise ValueError(f"model-contract lineage mismatch for {run_id}")
        if registration.get("val_select_metrics_sha256") != metrics[
            "content_hash"
        ]:
            raise ValueError(f"val_select lineage mismatch for {run_id}")
        if sha256_file(checkpoint_path) != registration["checkpoint_sha256"]:
            raise ValueError(f"checkpoint file hash mismatch for {run_id}")
        registrations[run_id] = registration
        model_contracts[run_id] = model_contract
        official_metrics[run_id] = metrics
        checkpoint_paths[run_id] = checkpoint_path

    views = {
        split: load_cached_hlt_view(
            campaign_root / "inputs" / "hlt_cache", split, verify_hash=True
        )
        for split in ("model_val", "stack_val")
    }
    loaders = {
        split: make_relational_loader(
            RelationalJetDataset(
                view,
                region_trees=load_region_tree_split(
                    campaign_root / "inputs" / "relation_tree_cache",
                    split=split,
                    expected_identities=view.jet_ids,
                ),
            ),
            seed=args.seed,
            training=False,
        )
        for split, view in views.items()
    }
    expected_hlt = {
        "model_val": registrations[run_ids[0]]["lineage_hashes"][
            "hlt_model_val"
        ],
        "stack_val": registrations[run_ids[0]]["lineage_hashes"][
            "hlt_stack_val"
        ],
    }
    for run_id in run_ids:
        for split, lineage_name in (
            ("model_val", "hlt_model_val"),
            ("stack_val", "hlt_stack_val"),
        ):
            if registrations[run_id]["lineage_hashes"][lineage_name] != expected_hlt[
                split
            ]:
                raise ValueError("fusion checkpoints use different HLT caches")
            if str(views[split].metadata["hlt_content_hash"]) != expected_hlt[split]:
                raise ValueError(f"opened {split} HLT cache hash mismatch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    predictions = {"val_stop": {}, "val_select": {}}
    reference = {"val_stop": None, "val_select": None}
    for run_id in run_ids:
        model = build_runtime_model(
            run_id,
            screening_registry=screening,
            normalization_artifact=normalization,
            region_normalization_artifact=region,
        )
        checkpoint = torch.load(
            checkpoint_paths[run_id], map_location="cpu", weights_only=False
        )
        if checkpoint.get("model_contract_sha256") != model_contracts[run_id][
            "content_hash"
        ]:
            raise ValueError(f"checkpoint payload mismatch for {run_id}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        for source_split, report_split in (
            ("model_val", "val_stop"),
            ("stack_val", "val_select"),
        ):
            collected = _collect_logits(
                model,
                loaders[source_split],
                device=device,
                model_forward=model_forward,
            )
            if reference[report_split] is None:
                reference[report_split] = {
                    "labels": collected["labels"],
                    "event_identities": collected["event_identities"],
                }
            elif not np.array_equal(
                collected["labels"], reference[report_split]["labels"]
            ) or not np.array_equal(
                collected["event_identities"],
                reference[report_split]["event_identities"],
            ):
                raise ValueError("fusion inference outputs are not event-aligned")
            predictions[report_split][run_id] = collected["logits"]
        replay = evaluate_logits(
            predictions["val_select"][run_id],
            reference["val_select"]["labels"],
            split="val_select",
        )
        if replay["content_hash"] != official_metrics[run_id]["content_hash"]:
            raise ValueError(f"val_select replay differs for {run_id}")
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_stop_labels = reference["val_stop"]["labels"]
    val_select_labels = reference["val_select"]["labels"]
    val_select_ids = reference["val_select"]["event_identities"]
    temperatures = {
        run_id: fit_temperature(
            predictions["val_stop"][run_id], val_stop_labels
        )
        for run_id in run_ids
    }
    inverse_temperatures = {
        run_id: row["inverse_temperature"]
        for run_id, row in temperatures.items()
    }

    group_definitions = {
        "CONTROL_LOGIT_FUSION": CONTROL_RUN_IDS,
        "RELATION_LOGIT_FUSION": RELATION_RUN_IDS,
        **PAIR_FUSIONS,
    }
    fusion_rows = {}
    fusion_predictions = {}
    for fusion_id, members in group_definitions.items():
        calibrated = fuse_equal_weight_logits(
            predictions["val_select"],
            members,
            inverse_temperatures=inverse_temperatures,
        )
        uncalibrated = fuse_equal_weight_logits(
            predictions["val_select"], members
        )
        calibrated_metrics = evaluate_logits(
            calibrated, val_select_labels, split="val_select"
        )
        uncalibrated_metrics = evaluate_logits(
            uncalibrated, val_select_labels, split="val_select"
        )
        fusion_rows[fusion_id] = {
            "member_run_ids": list(members),
            "primary_method": (
                "val_stop_temperature_scaled_equal_weight_centered_logit_mean"
            ),
            "primary_metrics": calibrated_metrics,
            "primary_qcd_signal_rejection_at_0p75": _qcd_rejection_at_75(
                calibrated,
                val_select_labels,
                class_names=CLASS_NAMES,
                qcd_signal_rejection=qcd_signal_rejection,
            ),
            "uncalibrated_sensitivity_metrics": uncalibrated_metrics,
            "uncalibrated_qcd_signal_rejection_at_0p75": _qcd_rejection_at_75(
                uncalibrated,
                val_select_labels,
                class_names=CLASS_NAMES,
                qcd_signal_rejection=qcd_signal_rejection,
            ),
            "member_diversity": pairwise_diversity(
                {
                    run_id: predictions["val_select"][run_id].argmax(axis=1)
                    for run_id in members
                },
                val_select_labels,
                members,
            ),
        }
        fusion_predictions[fusion_id] = {
            "logits": calibrated,
            "labels": val_select_labels,
            "predictions": calibrated.argmax(axis=1),
            "event_identities": val_select_ids,
            "metadata": {
                "run_id": fusion_id,
                "seed": int(args.seed),
                "checkpoint_sha256": canonical_sha256(
                    [registrations[run_id]["checkpoint_sha256"] for run_id in members]
                ),
            },
        }

    member_predictions = {
        run_id: {
            "logits": predictions["val_select"][run_id],
            "labels": val_select_labels,
            "predictions": predictions["val_select"][run_id].argmax(axis=1),
            "event_identities": val_select_ids,
            "metadata": {
                "run_id": run_id,
                "seed": int(args.seed),
                "checkpoint_sha256": registrations[run_id]["checkpoint_sha256"],
            },
        }
        for run_id in run_ids
    }
    control_baseline_comparisons = paired_prediction_statistics_many(
        {
            "RELATION_LOGIT_FUSION": fusion_predictions[
                "RELATION_LOGIT_FUSION"
            ],
            **{
                run_id: member_predictions[run_id]
                for run_id in CONTROL_RUN_IDS
            },
        },
        fusion_predictions["CONTROL_LOGIT_FUSION"],
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    relation_baseline_comparisons = paired_prediction_statistics_many(
        {
            **{
                run_id: member_predictions[run_id]
                for run_id in RELATION_RUN_IDS
            },
            **{
                fusion_id: fusion_predictions[fusion_id]
                for fusion_id in PAIR_FUSIONS
            },
        },
        fusion_predictions["RELATION_LOGIT_FUSION"],
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    primary_paired = control_baseline_comparisons.pop(
        "RELATION_LOGIT_FUSION"
    )

    artifact = with_content_hash(
        {
            "contract": CONTRACT,
            "schema_version": 1,
            "scientific_status": "supplemental_post_hoc_diagnostic_only",
            "official_campaign_metric": False,
            "eligible_for_model_selection": False,
            "changes_campaign_dependencies": False,
            "fusion_groups_fixed_in_artifact": True,
            "seed": int(args.seed),
            "calibration_split": "val_stop",
            "evaluation_split": "val_select",
            "final_test_opened": False,
            "class_order": list(CLASS_NAMES),
            "control_member_run_ids": list(CONTROL_RUN_IDS),
            "relation_member_run_ids": list(RELATION_RUN_IDS),
            "temperature_calibration": temperatures,
            "fusion_results": fusion_rows,
            "primary_relation_minus_control_paired_statistics": primary_paired,
            "control_members_minus_control_fusion_paired_statistics": (
                control_baseline_comparisons
            ),
            "relation_members_and_pair_fusions_minus_three_way_relation_fusion_paired_statistics": (
                relation_baseline_comparisons
            ),
            "campaign_spec_sha256": campaign["content_hash"],
            "global_determinism_sha256": determinism["content_hash"],
            "hlt_cache_hashes": expected_hlt,
            "event_identity_hashes": {
                split: canonical_sha256(
                    [str(value) for value in reference[split]["event_identities"]]
                )
                for split in ("val_stop", "val_select")
            },
            "checkpoint_registration_hashes": {
                run_id: registrations[run_id]["content_hash"] for run_id in run_ids
            },
            "checkpoint_hashes": {
                run_id: registrations[run_id]["checkpoint_sha256"]
                for run_id in run_ids
            },
            "official_val_select_metrics_hashes": {
                run_id: official_metrics[run_id]["content_hash"]
                for run_id in run_ids
            },
            "campaign_source": model_contracts[run_ids[0]].get("source"),
            "source_validation": {
                "mode": source_validation_mode,
                "status_drift_allowed_for_supplemental_diagnostic": bool(
                    args.allow_source_status_drift
                ),
                "expected_commit": campaign["source"]["commit"],
                "observed_commit": observed_source["source_commit"],
                "observed_status_sha256": observed_source[
                    "source_status_sha256"
                ],
                "observed_dirty": bool(observed_source["source_dirty"]),
                "official_campaign_workers_affected": False,
            },
            "diagnostic_script_sha256": sha256_file(Path(__file__)),
            "calculation_dtype": "float64",
        }
    )
    output = args.output or (
        campaign_root
        / "supplemental_diagnostics"
        / "fusion"
        / "control_vs_track_charge_pt.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_content_hash(existing, expected_contract=CONTRACT)
        if existing != artifact:
            raise FileExistsError(
                f"supplemental fusion output differs from existing artifact: {output}"
            )
    else:
        write_immutable_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
