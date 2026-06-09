#!/usr/bin/env python3
"""Demonstrate fresh-check model loading, inference, and feature preparation without fusion.

This script intentionally does not train a stacker and does not combine models.
It shows how to:

1. Construct fresh-check model specs for HLT + dual-view reco variants.
2. Load the frozen checkpoints through jetclass_fresh.fusion helpers.
3. Run inference on requested cached fixed-HLT splits.
4. Save per-model logits/probs/labels/jet identities.
5. Reload prediction blocks and verify row alignment.
6. Build candidate fusion feature matrices without fitting any fusion model.

Use this as a transparent handoff/demo for writing an independent fusion script.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import (  # noqa: E402
    FusionModelSpec,
    STACK_SPLITS,
    classification_metrics_from_logits,
    collect_frozen_predictions,
    default_reco7_plus_hlt_specs,
    load_blocks_for_split,
    stack_feature_matrix,
    validate_prediction_alignment,
)
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from jetclass_fresh.reconstructor import RECONSTRUCTOR_VARIANT_NAMES  # noqa: E402


DEFAULT_GROUPS = {
    "hlt_only": ["hlt_baseline"],
    "reco7_only": list(RECONSTRUCTOR_VARIANT_NAMES),
    "hlt_plus_reco7": ["hlt_baseline", *list(RECONSTRUCTOR_VARIANT_NAMES)],
}


# These are external PracticeTagging model directories from the fixed-HLT filename run.
# The fresh-check library cannot necessarily instantiate those old architectures.
# They are included here so the independent implementation has explicit paths.
PRACTICETAGGING_FIXEDHLT_SOURCES = [
    ("hlt_baseline", "baseline_hlt", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core01_base"),
    ("offline_teacher", "offline_teacher", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core01_base"),
    ("m2_base", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core01_base"),
    ("m2_consstrong", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core02_consstrong"),
    ("m2_budgetlite", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core03_budgetlite"),
    ("m2_genlow", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core04_genlow"),
    ("m2_genhigh", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core05_genhigh"),
    ("m2_splitstrong", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core06_splitstrong"),
    ("m2_splitlight", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core07_splitlight"),
    ("m2_physstrong", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core08_physstrong"),
    ("m2_offdropmid", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core09_offdropmid"),
    ("m2_offdrophigh", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core10_offdrophigh"),
    ("m2_topk60ish", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core11_topk60ish"),
    ("m2_antioverlap", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core12_antioverlap"),
]


def parse_group(text: str) -> tuple[str, List[str]]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Groups must be formatted name:model1,model2,...")
    name, values = text.split(":", 1)
    models = [item.strip() for item in values.split(",") if item.strip()]
    if not name.strip() or not models:
        raise argparse.ArgumentTypeError("Groups must include a nonempty name and at least one model")
    return name.strip(), models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument(
        "--hlt-checkpoint",
        default="checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/best_model_val.pt",
    )
    parser.add_argument("--reco-root", default="checkpoints/jetclass_fresh_reco7")
    parser.add_argument("--output-dir", default="checkpoints/jetclass_fresh_model_loading_demo")
    parser.add_argument("--variants", nargs="+", choices=RECONSTRUCTOR_VARIANT_NAMES, default=list(RECONSTRUCTOR_VARIANT_NAMES))
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=["stack_train", "stack_val"])
    parser.add_argument("--confirm-final-test", action="store_true", help="Required if --splits includes final_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-jets-per-split", type=int, default=2000)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--feature-mode", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        default=[],
        help="Optional feature-matrix group formatted name:model1,model2. May be repeated.",
    )
    parser.add_argument(
        "--validate-practicetagging-paths",
        action="store_true",
        help="Only checks that external PracticeTagging run directories/args/checkpoints exist; does not load them.",
    )
    return parser.parse_args()


def validate_splits(splits: Sequence[str], confirm_final_test: bool) -> None:
    if "final_test" in splits and not bool(confirm_final_test):
        raise SystemExit("Refusing to touch final_test without --confirm-final-test")


def fresh_specs(args: argparse.Namespace) -> List[FusionModelSpec]:
    return default_reco7_plus_hlt_specs(
        hlt_checkpoint=args.hlt_checkpoint,
        reco_root=args.reco_root,
        variants=list(args.variants),
    )


def validate_practicetagging_paths() -> Dict[str, object]:
    candidates_by_kind = {
        "baseline_hlt": ("baseline_hlt_best.pt", "baseline_best.pt", "baseline.pt"),
        "offline_teacher": ("teacher_offline_best.pt", "teacher.pt"),
        "stage2_reco": ("offline_reconstructor_stage2.pt", "offline_reconstructor.pt"),
        "stage2_dual": ("dual_joint_stage2.pt", "dual_joint.pt"),
    }
    rows = []
    for name, kind, run_dir_str in PRACTICETAGGING_FIXEDHLT_SOURCES:
        run_dir = Path(run_dir_str)
        row = {
            "name": name,
            "kind": kind,
            "run_dir": str(run_dir),
            "run_dir_exists": run_dir.exists(),
            "args_json_exists": (run_dir / "args.json").exists(),
            "found_checkpoints": {},
        }
        if kind == "baseline_hlt":
            keys = ["baseline_hlt"]
        elif kind == "offline_teacher":
            keys = ["offline_teacher"]
        else:
            keys = ["stage2_reco", "stage2_dual"]
        for key in keys:
            found = [str(run_dir / filename) for filename in candidates_by_kind[key] if (run_dir / filename).exists()]
            row["found_checkpoints"][key] = found
        rows.append(row)
    return {"sources": rows}


def summarize_prediction_blocks(prediction_dir: Path, specs: Sequence[FusionModelSpec], splits: Sequence[str]) -> Dict[str, object]:
    model_names = [spec.name for spec in specs]
    out: Dict[str, object] = {"splits": {}, "models": model_names, "label_names": list(LABEL_NAMES)}
    for split in splits:
        split_rows = []
        blocks = []
        for name in model_names:
            block = load_blocks_for_split(prediction_dir, [name], split)[0]
            blocks.append(block)
            metrics = classification_metrics_from_logits(block.logits, block.labels)
            split_rows.append(
                {
                    "model": name,
                    "split": split,
                    "logits_shape": list(block.logits.shape),
                    "probs_shape": list(block.probs.shape),
                    "labels_shape": list(block.labels.shape),
                    "jet_identity_hash": block.metadata.get("jet_identity_hash"),
                    "metrics": metrics,
                    "metadata": block.metadata,
                }
            )
        validate_prediction_alignment(blocks)
        out["splits"][split] = {
            "aligned": True,
            "n_models": len(blocks),
            "n_jets": int(len(blocks[0].labels)) if blocks else 0,
            "rows": split_rows,
        }
    return out


def summarize_feature_groups(prediction_dir: Path, groups: Dict[str, List[str]], splits: Sequence[str], feature_mode: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for group_name, model_names in groups.items():
        group_report = {"models": list(model_names), "feature_mode": feature_mode, "splits": {}}
        for split in splits:
            blocks = load_blocks_for_split(prediction_dir, model_names, split)
            validate_prediction_alignment(blocks)
            features = stack_feature_matrix(blocks, feature_mode=feature_mode)
            labels = blocks[0].labels
            group_report["splits"][split] = {
                "feature_shape": list(features.shape),
                "labels_shape": list(labels.shape),
                "finite_features": bool(np.isfinite(features).all()),
                "first_five_labels": labels[:5].astype(int).tolist(),
            }
        out[group_name] = group_report
    return out


def main() -> int:
    args = parse_args()
    validate_splits(args.splits, args.confirm_final_test)

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    specs = fresh_specs(args)
    groups = dict(DEFAULT_GROUPS)
    if args.group:
        groups = {name: models for name, models in args.group}

    available = {spec.name for spec in specs}
    for group_name, model_names in groups.items():
        missing = sorted(set(model_names) - available)
        if missing:
            raise ValueError(f"Group {group_name!r} references missing models: {missing}; available={sorted(available)}")

    print("Collecting raw frozen predictions. No fusion or stacker fitting will be performed.")
    print(f"Output dir: {output_dir}")
    print(f"Prediction dir: {prediction_dir}")
    print(f"Splits: {args.splits}")
    print(f"Models: {[spec.name for spec in specs]}")

    collection_report = collect_frozen_predictions(
        specs,
        hlt_cache_dir=args.hlt_cache_dir,
        prediction_dir=prediction_dir,
        splits=list(args.splits),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=args.device,
        max_jets_per_split=args.max_jets_per_split,
        overwrite=bool(args.overwrite_predictions),
        skip_existing=not bool(args.no_skip_existing_predictions),
    )

    raw_report = summarize_prediction_blocks(prediction_dir, specs, args.splits)
    feature_report = summarize_feature_groups(prediction_dir, groups, args.splits, args.feature_mode)
    external_report = validate_practicetagging_paths() if bool(args.validate_practicetagging_paths) else None

    report = {
        "purpose": "model_loading_and_raw_prediction_demo_no_fusion",
        "explicitly_not_done": [
            "no logistic regression stacker fitted",
            "no weighted averaging fitted",
            "no model selection by stack_val or final_test",
        ],
        "config": {
            "hlt_cache_dir": str(args.hlt_cache_dir),
            "hlt_checkpoint": str(args.hlt_checkpoint),
            "reco_root": str(args.reco_root),
            "splits": list(args.splits),
            "variants": list(args.variants),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
            "max_jets_per_split": None if args.max_jets_per_split is None else int(args.max_jets_per_split),
            "feature_mode": str(args.feature_mode),
        },
        "model_specs": [asdict(spec) for spec in specs],
        "collection_report": collection_report,
        "raw_prediction_report": raw_report,
        "candidate_feature_matrices": feature_report,
        "practicetagging_fixedhlt_path_report": external_report,
    }
    save_json(output_dir / "model_loading_demo_report.json", report)

    print("\nRaw model metrics")
    for split, split_report in raw_report["splits"].items():
        print(f"  {split}:")
        for row in split_report["rows"]:
            acc = row["metrics"]["accuracy"]
            ce = row["metrics"]["cross_entropy"]
            print(f"    {row['model']:<18s} acc={acc:.6f} ce={ce:.6f} logits={row['logits_shape']}")

    print("\nCandidate feature matrices prepared, but not fused")
    for group_name, group_report in feature_report.items():
        for split, split_report in group_report["splits"].items():
            print(f"  {group_name:<16s} {split:<11s} X={split_report['feature_shape']} y={split_report['labels_shape']}")

    print(f"\nSaved report: {output_dir / 'model_loading_demo_report.json'}")
    print(f"Saved prediction blocks under: {prediction_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
