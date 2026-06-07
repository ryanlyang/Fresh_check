"""Step 11 five-seed HLT-only ensemble control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

from .fusion import (
    DEFAULT_C_GRID,
    STACK_SPLITS,
    FusionModelSpec,
    collect_frozen_predictions,
    evaluate_fusion_methods,
    save_linear_stacker,
)
from .hlt_baseline import save_json


HLT5_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_HLT5_CHECKPOINT_ROOT = "checkpoints/jetclass_fresh_hlt_baselines/hlt5_seed_control"
FUSION_COMPARISON_METHODS = [
    "uniform_probability_average",
    "weighted_probability_average",
    "weighted_logit_average",
    "stacked_logistic_regression",
]


@dataclass
class HLT5FusionRunConfig:
    """Configuration for the Step 11 HLT-only seed ensemble fusion."""

    output_dir: str
    hlt_cache_dir: str
    hlt_checkpoint_root: str = DEFAULT_HLT5_CHECKPOINT_ROOT
    seeds: list[int] = field(default_factory=lambda: list(HLT5_SEEDS))
    splits: list[str] = field(default_factory=lambda: list(STACK_SPLITS))
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_jets_per_split: int | None = None
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    C_grid: list[float] = field(default_factory=lambda: list(DEFAULT_C_GRID))
    feature_mode: str = "logits_probs"
    max_iter: int = 500


def hlt_seed_model_name(seed: int) -> str:
    return f"hlt_seed{int(seed)}"


def hlt_seed_checkpoint_path(checkpoint_root: str | Path, seed: int) -> Path:
    return Path(checkpoint_root) / f"seed{int(seed)}" / "best_model_val.pt"


def default_hlt5_specs(
    *,
    checkpoint_root: str | Path = DEFAULT_HLT5_CHECKPOINT_ROOT,
    seeds: Sequence[int] = HLT5_SEEDS,
) -> list[FusionModelSpec]:
    """Return the five HLT-only frozen model specs for Step 11."""

    return [
        FusionModelSpec(
            name=hlt_seed_model_name(seed),
            kind="hlt",
            checkpoint=str(hlt_seed_checkpoint_path(checkpoint_root, seed)),
        )
        for seed in seeds
    ]


def run_hlt5_fusion(config: HLT5FusionRunConfig) -> Dict[str, Any]:
    """Collect frozen HLT5 predictions and run the Step 10 stacker procedure."""

    if "final_test" in config.splits and not config.confirm_final_test:
        raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")

    output_dir = Path(config.output_dir)
    report_path = output_dir / "fusion_report.json"
    if report_path.exists():
        raise FileExistsError(f"Fusion report already exists; refusing to overwrite locked result: {report_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    specs = default_hlt5_specs(checkpoint_root=config.hlt_checkpoint_root, seeds=config.seeds)
    save_json(
        output_dir / "fusion_config.json",
        {
            "experiment_step": "step11_five_seed_hlt_control",
            "config": asdict(config),
            "model_specs": [spec.to_dict() for spec in specs],
            "leakage_rule": (
                "All five HLT models consume the same cached fixed_hlt views; "
                "only neural-network training seed and checkpoint differ."
            ),
        },
    )

    prediction_report = collect_frozen_predictions(
        specs,
        hlt_cache_dir=config.hlt_cache_dir,
        prediction_dir=prediction_dir,
        splits=config.splits,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=config.device,
        max_jets_per_split=config.max_jets_per_split,
        overwrite=config.overwrite_predictions,
        skip_existing=config.skip_existing_predictions,
    )
    if list(config.splits) != STACK_SPLITS:
        return {
            "experiment_step": "step11_five_seed_hlt_control",
            "prediction_report": prediction_report,
            "fusion_report": None,
        }

    fusion = evaluate_fusion_methods(
        prediction_dir,
        [spec.name for spec in specs],
        C_grid=config.C_grid,
        feature_mode=config.feature_mode,
        max_iter=config.max_iter,
    )
    save_linear_stacker(fusion["stacker"], output_dir / "stacked_logistic_regression.npz")
    fusion_report = dict(fusion["report"])
    fusion_report.update(
        {
            "experiment_step": "step11_five_seed_hlt_control",
            "seed_control_seeds": [int(seed) for seed in config.seeds],
            "final_test_evaluated": True,
            "locked_final_test_note": (
                "This Step 11 HLT5 report evaluates final_test after stack choices are fixed "
                "by stack_train/stack_val."
            ),
        }
    )
    final_report = {"prediction_report": prediction_report, **fusion_report}
    save_json(report_path, final_report)
    return final_report


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _method_metrics(report: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    section = report.get(method, {})
    if method == "uniform_probability_average":
        return section if isinstance(section, Mapping) else {}
    if isinstance(section, Mapping):
        metrics = section.get("metrics", {})
        return metrics if isinstance(metrics, Mapping) else {}
    return {}


def _ordered_common_splits(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    common = set(left).intersection(right)
    ordered = [split for split in STACK_SPLITS if split in common]
    ordered.extend(sorted(common.difference(ordered)))
    return ordered


def compare_hlt5_to_reco7_reports(
    hlt5_report: Mapping[str, Any],
    reco7_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare Step 11 HLT5 fusion metrics against Step 10 reco7+HLT metrics."""

    comparison: Dict[str, Any] = {
        "experiment_step": "step11_hlt5_vs_reco7_comparison",
        "hlt5_model_names": list(hlt5_report.get("model_names", [])),
        "reco7_plus_hlt_model_names": list(reco7_report.get("model_names", [])),
        "methods": {},
    }
    for method in FUSION_COMPARISON_METHODS:
        hlt_metrics_by_split = _method_metrics(hlt5_report, method)
        reco_metrics_by_split = _method_metrics(reco7_report, method)
        method_rows: Dict[str, Any] = {}
        for split in _ordered_common_splits(hlt_metrics_by_split, reco_metrics_by_split):
            hlt_metrics = dict(hlt_metrics_by_split[split])
            reco_metrics = dict(reco_metrics_by_split[split])
            row = {
                "hlt5": hlt_metrics,
                "reco7_plus_hlt": reco_metrics,
            }
            for metric_name in ("accuracy", "cross_entropy"):
                if metric_name in hlt_metrics and metric_name in reco_metrics:
                    row[f"{metric_name}_delta_reco7_minus_hlt5"] = float(
                        reco_metrics[metric_name] - hlt_metrics[metric_name]
                    )
            method_rows[split] = row
        if method_rows:
            comparison["methods"][method] = method_rows

    stack_final = comparison["methods"].get("stacked_logistic_regression", {}).get("final_test")
    if stack_final:
        comparison["summary"] = {
            "stacked_logistic_final_test_accuracy_delta_reco7_minus_hlt5": stack_final.get(
                "accuracy_delta_reco7_minus_hlt5"
            ),
            "stacked_logistic_final_test_cross_entropy_delta_reco7_minus_hlt5": stack_final.get(
                "cross_entropy_delta_reco7_minus_hlt5"
            ),
        }
    return comparison


def compare_hlt5_to_reco7(
    hlt5_report_path: str | Path,
    reco7_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load two locked fusion reports, compare them, and optionally save JSON."""

    comparison = compare_hlt5_to_reco7_reports(
        _load_json(hlt5_report_path),
        _load_json(reco7_report_path),
    )
    comparison.update(
        {
            "hlt5_report_path": str(hlt5_report_path),
            "reco7_report_path": str(reco7_report_path),
        }
    )
    if output_path is not None:
        save_json(Path(output_path), comparison)
    return comparison
