from __future__ import annotations

import json
from pathlib import Path

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_CONTRACT,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_STEP,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CORE_FUSION_GROUP,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    ARCHITECTURE_VIEW_10CLASS_SCALAR_FUSION_MODE,
    ArchitectureView10ClassAblationReportConfig,
    build_architecture_view_10class_ablation_report,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metrics(*, acc: float, ce: float = 0.6) -> dict:
    return {
        "accuracy": float(acc),
        "cross_entropy": float(ce),
        "loss": float(ce),
        "macro_per_class_accuracy": float(acc) - 0.01,
        "n_jets": 100,
        "diagnostics": {
            "delta_h_norm_mean": 0.01,
            "gate_mean": 0.02,
            "embed_injection.delta_to_embedding_norm_ratio": 0.03,
        },
    }


def _run_report(variant: str, *, final_acc: float, input_source: str = "hlt", total_params: int = 1000) -> dict:
    return {
        "variant": variant,
        "input_source": input_source,
        "inference_input_source": input_source,
        "inference_consumes_hlt_only": input_source == "hlt",
        "best_epoch": 4,
        "epochs_completed": 6,
        "checkpoint": f"/fake/{variant}/best_model_val.pt",
        "best_model_val_metrics": _metrics(acc=final_acc - 0.002),
        "stack_val_metrics": _metrics(acc=final_acc - 0.001),
        "final_test_metrics": _metrics(acc=final_acc),
        "parameter_accounting": {
            "total_params": int(total_params),
            "trainable_params": int(total_params),
            "part_params": 900,
            "trainable_part_params": 900,
            "adapter_params": max(int(total_params) - 900, 0),
            "trainable_adapter_params": max(int(total_params) - 900, 0),
        },
        "variant_behavior": {"variant": variant, "input_source": input_source},
    }


def _write_run_reports(root: Path, reports: dict[str, dict]) -> None:
    for variant, payload in reports.items():
        _write_json(root / variant / "run_report.json", payload)
        _write_json(
            root / variant / "training_curves.json",
            {
                "epochs": [
                    {
                        "epoch": 1,
                        "train": {
                            "diagnostics": {
                                "grad_norm.active_adapter": 0.12,
                                "grad_norm.part": 0.34,
                            }
                        },
                        "model_val": {
                            "diagnostics": {
                                "delta_h_norm_mean": 0.02,
                            }
                        },
                    }
                ]
            },
        )


def _fusion_report(path: Path) -> None:
    _write_json(
        path,
        {
            "contract": "architecture_view_10class_fusion_v1",
            "groups": {
                ARCHITECTURE_VIEW_10CLASS_CORE_FUSION_GROUP: {
                    "model_names": ["av10_pn_context_to_part", "av10_pfn_context_to_part"],
                    "fusion_modes": {
                        ARCHITECTURE_VIEW_10CLASS_SCALAR_FUSION_MODE: {
                            "fit": {"weights": [0.4, 0.6]},
                            "metrics": {
                                "stack_val": {"accuracy": 0.760, "cross_entropy": 0.5, "n_jets": 100},
                                "final_test": {"accuracy": 0.762, "cross_entropy": 0.49, "n_jets": 100},
                            },
                        },
                        "uniform_logit_mean": {
                            "metrics": {
                                "stack_val": {"accuracy": 0.758, "cross_entropy": 0.52, "n_jets": 100},
                                "final_test": {"accuracy": 0.759, "cross_entropy": 0.51, "n_jets": 100},
                            }
                        },
                    },
                }
            },
        },
    )


def test_step5_ablation_report_merges_hlt_fusion_offline_and_answers_decisions(tmp_path: Path) -> None:
    hlt_root = tmp_path / "taggers"
    offline_root = tmp_path / "offline_taggers"
    hlt_variants = (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    )
    _write_run_reports(
        hlt_root,
        {
            ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
                final_acc=0.750,
                total_params=1000,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
                final_acc=0.752,
                total_params=1500,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
                final_acc=0.751,
                total_params=1120,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
                final_acc=0.754,
                total_params=1080,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
                final_acc=0.760,
                total_params=1100,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
                final_acc=0.758,
                total_params=1080,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
                final_acc=0.753,
                total_params=1100,
            ),
            ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
                final_acc=0.751,
                total_params=1100,
            ),
        },
    )
    _write_run_reports(
        offline_root,
        {
            ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE: _run_report(
                ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
                final_acc=0.800,
                input_source="offline",
                total_params=1000,
            ),
            ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER: _run_report(
                ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
                final_acc=0.806,
                input_source="offline",
                total_params=1100,
            ),
        },
    )
    fusion_path = tmp_path / "fusion" / "fusion_report.json"
    _fusion_report(fusion_path)

    report = build_architecture_view_10class_ablation_report(
        ArchitectureView10ClassAblationReportConfig(
            output_dir=str(tmp_path / "final_report"),
            hlt_tagger_root=str(hlt_root),
            hlt_variants=hlt_variants,
            fusion_report=str(fusion_path),
            offline_tagger_root=str(offline_root),
            offline_variants=(
                ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
                ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
            ),
            require_fusion=True,
            require_offline_transfer=True,
            confirm_final_test=True,
        )
    )

    assert report["ok"], report["problems"]
    assert report["contract"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_CONTRACT
    assert report["step"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_STEP
    assert report["summary"]["best_hlt_variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
    assert report["summary"]["best_fusion_accuracy"] == 0.762
    feature_row = next(
        row for row in report["hlt_ablation_rows"] if row["variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
    )
    assert feature_row["delta_vs_baseline"] == 0.010000000000000009
    offline_feature_row = next(
        row
        for row in report["offline_transfer_rows"]
        if row["variant"] == ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER
    )
    assert offline_feature_row["delta_vs_baseline"] == 0.006000000000000005
    assert report["parameter_accounting_rows"][0]["ratio_vs_hlt_baseline_total_params"] == 1.0
    decisions = report["interpretation_summary"]["decisions"]
    assert decisions["did_larger_part_close_gap"]["satisfied"] is False
    assert decisions["did_extra_part_block_close_gap"]["satisfied"] is False
    assert decisions["did_frozen_adapter_improve"]["satisfied"] is True
    assert decisions["did_shuffled_control_fail"]["satisfied"] is True
    assert decisions["did_lc_mlp_delta_input_repair_work"]["satisfied"] is True
    assert decisions["did_input_repair_match_embedding_repair"]["satisfied"] is True
    assert decisions["did_offline_transfer_work"]["satisfied"] is True
    assert report["interpretation_summary"]["evidence_values"]["lc_mlp_delta_gain"] == 0.008000000000000007
    part_only_parameter_row = next(
        row
        for row in report["parameter_accounting_rows"]
        if row["variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER
    )
    assert part_only_parameter_row["parameter_match_reference_variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
    assert part_only_parameter_row["adapter_param_ratio_vs_match_reference"] == 0.9
    assert part_only_parameter_row["adapter_param_match_within_20pct"] is True
    train_grad_row = next(
        row
        for row in report["diagnostic_rows"]
        if row["variant"] == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
        and row.get("split") == "train"
        and row.get("diagnostic") == "grad_norm.active_adapter"
    )
    assert train_grad_row["epoch"] == 1
    assert train_grad_row["value"] == 0.12
    assert (tmp_path / "final_report" / "architecture_view_10class_ablation_report.json").exists()
    assert (tmp_path / "final_report" / "decision_summary.txt").exists()
    assert (tmp_path / "final_report" / "parameter_accounting.csv").exists()


def test_step5_report_records_missing_required_inputs(tmp_path: Path) -> None:
    hlt_root = tmp_path / "taggers"
    _write_run_reports(
        hlt_root,
        {
            ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK: _run_report(
                ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
                final_acc=0.750,
            )
        },
    )
    report = build_architecture_view_10class_ablation_report(
        ArchitectureView10ClassAblationReportConfig(
            output_dir=str(tmp_path / "final_report"),
            hlt_tagger_root=str(hlt_root),
            hlt_variants=(ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,),
            require_fusion=True,
            require_offline_transfer=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("require_fusion=True" in problem for problem in report["problems"])
    assert any("require_offline_transfer=True" in problem for problem in report["problems"])
