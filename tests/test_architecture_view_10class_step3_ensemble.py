from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity
from scripts import run_architecture_view_10class_fusion as fusion_cli

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT,
    ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT,
    ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    ArchitectureView10ClassFusionConfig,
    av10_metrics_from_logits,
    load_architecture_view_10class_blocks_for_split,
    run_architecture_view_10class_fusion,
    save_architecture_view_10class_prediction_cache,
)


SPLITS = ("stack_train", "stack_val", "final_test")
MODEL_A = ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART
MODEL_B = ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART
MODEL_C = ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART


def _labels(n_jets: int = 40) -> np.ndarray:
    base = np.arange(n_jets, dtype=np.int64) % 10
    base[:8] = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return base


def _jet_ids(split: str, labels: np.ndarray) -> list[JetIdentity]:
    return [
        JetIdentity(file=f"{split}.root", entry=int(index), label=int(label))
        for index, label in enumerate(labels)
    ]


def _logits(labels: np.ndarray, *, strength: float = 3.0, bad: bool = False, scale: float = 1.0) -> np.ndarray:
    logits = np.full((labels.shape[0], 10), -0.4, dtype=np.float32)
    target = (labels + 1) % 10 if bad else labels
    logits[np.arange(labels.shape[0]), target] = float(strength)
    logits[:, 0] += 0.05
    logits[:, 1] -= 0.03
    return (logits * float(scale)).astype(np.float32)


def _write_cache(
    prediction_dir: Path,
    *,
    variant: str,
    split: str,
    labels: np.ndarray,
    logits: np.ndarray,
) -> None:
    save_architecture_view_10class_prediction_cache(
        prediction_dir=prediction_dir,
        variant=variant,
        split=split,
        logits=logits,
        labels=labels,
        jet_ids=_jet_ids(split, labels),
        metadata={
            "checkpoint_path": f"/fake/{variant}.pt",
            "checkpoint_hash": f"hash-{variant}",
            "hlt_content_hash": f"hlt-{split}",
        },
    )


def _write_prediction_suite(prediction_dir: Path, variants: tuple[str, ...] = (MODEL_A, MODEL_B)) -> np.ndarray:
    labels = _labels()
    for split in SPLITS:
        for variant in variants:
            if variant == MODEL_A:
                logits = _logits(labels, strength=3.0)
            elif variant == MODEL_B:
                logits = _logits(labels, strength=2.4)
            else:
                logits = _logits(labels, strength=1.6, bad=(split == "stack_train"))
            _write_cache(prediction_dir, variant=variant, split=split, labels=labels, logits=logits)
    return labels


def test_uniform_logit_mean_matches_manual_average(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    labels = _write_prediction_suite(prediction_dir)
    config = ArchitectureView10ClassFusionConfig(
        prediction_dir=str(prediction_dir),
        output_dir=str(tmp_path / "fusion"),
        model_names=(MODEL_A, MODEL_B),
        groups={"toy": (MODEL_A, MODEL_B)},
        fusion_modes=("uniform_logit_mean",),
        confirm_final_test=True,
    )

    report = run_architecture_view_10class_fusion(config)
    blocks = load_architecture_view_10class_blocks_for_split(prediction_dir, (MODEL_A, MODEL_B), "final_test")
    manual_logits = np.mean(np.stack([block.logits for block in blocks], axis=0), axis=0)
    manual_metrics = av10_metrics_from_logits(manual_logits, labels)
    reported = report["groups"]["toy"]["fusion_modes"]["uniform_logit_mean"]["metrics"]["final_test"]

    assert report["contract"] == ARCHITECTURE_VIEW_10CLASS_FUSION_CONTRACT
    assert reported["accuracy"] == manual_metrics["accuracy"]
    assert np.isclose(reported["cross_entropy"], manual_metrics["cross_entropy"])


def test_temperature_scaling_preserves_or_improves_stack_train_nll(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    labels = _labels()
    for split in SPLITS:
        _write_cache(
            prediction_dir,
            variant=MODEL_A,
            split=split,
            labels=labels,
            logits=_logits(labels, strength=2.5, bad=True, scale=3.0),
        )
    report = run_architecture_view_10class_fusion(
        ArchitectureView10ClassFusionConfig(
            prediction_dir=str(prediction_dir),
            output_dir=str(tmp_path / "fusion"),
            model_names=(MODEL_A,),
            groups={"single": (MODEL_A,)},
            fusion_modes=("temperature_scaled_logit_mean",),
            temperature_grid=(0.5, 1.0, 2.0, 4.0),
            confirm_final_test=True,
        )
    )
    per_model = report["groups"]["single"]["fusion_modes"]["temperature_scaled_logit_mean"]["fit"]["per_model"][0]
    selected_ce = min(row["metrics"]["cross_entropy"] for row in per_model["candidates"])
    unit_ce = next(row["metrics"]["cross_entropy"] for row in per_model["candidates"] if row["temperature"] == 1.0)

    assert selected_ce <= unit_ce + 1.0e-12


def test_weighted_fusion_records_stack_only_fit_and_final_eval_separately(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    _write_prediction_suite(prediction_dir, variants=(MODEL_A, MODEL_B, MODEL_C))
    report = run_architecture_view_10class_fusion(
        ArchitectureView10ClassFusionConfig(
            prediction_dir=str(prediction_dir),
            output_dir=str(tmp_path / "fusion"),
            model_names=(MODEL_A, MODEL_B, MODEL_C),
            groups={"weighted": (MODEL_A, MODEL_B, MODEL_C)},
            fusion_modes=("scalar_weighted_logit_mean", "ridge_logit_stacker"),
            scalar_weight_trials=8,
            c_grid=(0.1, 1.0),
            confirm_final_test=True,
        )
    )

    assert report["leakage_rules"]["fit_split"] == ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT
    assert report["leakage_rules"]["selection_split"] == ARCHITECTURE_VIEW_10CLASS_FUSION_SELECTION_SPLIT
    for mode in ("scalar_weighted_logit_mean", "ridge_logit_stacker"):
        mode_report = report["groups"]["weighted"]["fusion_modes"][mode]
        fit_payload = json.dumps(mode_report["fit"])
        assert "final_test" not in fit_payload
        assert "final_test" in mode_report["metrics"]


def test_binary_projection_weighted_reports_fpr50(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    labels = _write_prediction_suite(prediction_dir, variants=(MODEL_A, MODEL_B))
    report = run_architecture_view_10class_fusion(
        ArchitectureView10ClassFusionConfig(
            prediction_dir=str(prediction_dir),
            output_dir=str(tmp_path / "fusion"),
            model_names=(MODEL_A, MODEL_B),
            groups={"binary": (MODEL_A, MODEL_B)},
            fusion_modes=("binary_projection_weighted",),
            binary_weight_trials=8,
            confirm_final_test=True,
        )
    )
    pair = report["groups"]["binary"]["fusion_modes"]["binary_projection_weighted"]["binary_projection_results"]["QCD_vs_Hgg"]

    assert pair["available"] is True
    assert pair["fit_split"] == ARCHITECTURE_VIEW_10CLASS_FUSION_FIT_SPLIT
    assert "fpr_at_signal_eff_0p50" in pair["metrics"]["final_test"]
    qcd_index = ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES.index("QCD")
    hgg_index = ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES.index("Hgg")
    expected_binary_jets = int(((labels == qcd_index) | (labels == hgg_index)).sum())
    assert pair["metrics"]["final_test"]["n_jets"] == expected_binary_jets


def test_fusion_cli_accepts_ablation_variants(monkeypatch, tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    output_dir = tmp_path / "fusion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_architecture_view_10class_fusion.py",
            "--prediction-dir",
            str(prediction_dir),
            "--output-dir",
            str(output_dir),
            "--model-names",
            "av10_hlt_baseline_recheck",
            "av10_lc_mlp_delta_features",
            "--group",
            "av10_input_delta:av10_hlt_baseline_recheck,av10_lc_mlp_delta_features",
        ],
    )

    args = fusion_cli.parse_args()

    assert tuple(args.model_names) == ("av10_hlt_baseline_recheck", "av10_lc_mlp_delta_features")
    assert args.group == [
        ("av10_input_delta", ("av10_hlt_baseline_recheck", "av10_lc_mlp_delta_features"))
    ]
