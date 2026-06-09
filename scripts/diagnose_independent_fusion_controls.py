#!/usr/bin/env python3
"""Diagnose row-shuffle warnings from independent fusion reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID, STACK_SPLITS  # noqa: E402
from jetclass_fresh.independent_fusion import (  # noqa: E402
    FEATURE_MODES,
    block_features,
    default_groups_for_models,
    discover_prediction_models,
    fit_stacker_selecting_c_on_val,
    metrics_from_probs,
    validate_fusion_groups,
)
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402


def parse_group(text: str) -> tuple[str, List[str]]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Groups must be formatted name:model1,model2,...")
    name, raw_models = text.split(":", 1)
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    if not name.strip() or not models:
        raise argparse.ArgumentTypeError("Groups must include a nonempty name and at least one model")
    return name.strip(), models


def parse_target(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Targets must be formatted group:feature_mode")
    group, mode = text.split(":", 1)
    group = group.strip()
    mode = mode.strip()
    if not group or mode not in FEATURE_MODES:
        raise argparse.ArgumentTypeError(f"Target must use one of feature modes {FEATURE_MODES}")
    return group, mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument(
        "--fusion-dir",
        default=None,
        help="Optional fusion output directory containing fusion_report.json and controls.json.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--model-names", nargs="+", default=None)
    parser.add_argument("--group", action="append", type=parse_group, default=[])
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        default=[],
        help="Control target formatted group:feature_mode. Defaults to suspicious row-shuffle flags.",
    )
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def column_shuffle(features: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    shuffled = np.asarray(features).copy()
    for column in range(shuffled.shape[1]):
        shuffled[:, column] = shuffled[rng.permutation(shuffled.shape[0]), column]
    return shuffled


def row_permute(features: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    return np.asarray(features)[rng.permutation(features.shape[0])].copy()


def class_distribution(labels: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = int(labels.max()) + 1 if labels.size else 0
    counts = np.bincount(labels, minlength=n_classes).astype(np.int64)
    total = int(labels.size)
    return {
        "n": total,
        "counts": counts.astype(int).tolist(),
        "fractions": (counts / float(max(total, 1))).astype(float).tolist(),
        "majority_class": int(np.argmax(counts)) if total else None,
        "majority_accuracy": float(np.max(counts) / float(total)) if total else 0.0,
    }


def prediction_distribution(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    preds = np.argmax(probs, axis=1).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    pred_max = int(preds.max()) if preds.size else 0
    label_max = int(labels.max()) if labels.size else 0
    n_classes = max(pred_max, label_max) + 1
    counts = np.bincount(preds, minlength=n_classes).astype(np.int64)
    return {
        "predicted_counts": counts.astype(int).tolist(),
        "predicted_fractions": (counts / float(max(len(preds), 1))).astype(float).tolist(),
        "unique_predicted_classes": int(np.sum(counts > 0)),
    }


def fit_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    c_grid: Sequence[float],
    max_iter: int,
    feature_mode: str,
    model_names: Sequence[str],
    num_classes: int,
) -> Dict[str, object]:
    stacker, selection = fit_stacker_selecting_c_on_val(
        x_train,
        y_train,
        x_val,
        y_val,
        c_grid=c_grid,
        max_iter=max_iter,
        feature_mode=feature_mode,
        model_names=model_names,
        num_classes=num_classes,
    )
    val_probs = stacker.predict_probs(x_val)
    test_probs = stacker.predict_probs(x_test)
    return {
        "selection": selection,
        "metrics": {
            "stack_val": metrics_from_probs(val_probs, y_val),
            "final_test": metrics_from_probs(test_probs, y_test),
        },
        "prediction_distribution": {
            "stack_val": prediction_distribution(val_probs, y_val),
            "final_test": prediction_distribution(test_probs, y_test),
        },
        "weight_norm": float(np.linalg.norm(stacker.coef)),
        "intercept": stacker.intercept.astype(float).tolist(),
    }


def infer_targets(args: argparse.Namespace, groups: Dict[str, List[str]]) -> List[tuple[str, str]]:
    if args.target:
        return list(args.target)
    if args.fusion_dir:
        report_path = Path(args.fusion_dir) / "fusion_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            targets = []
            for flag in report.get("suspicious_flags", []):
                if flag.get("name") == "row_shuffled_features_did_not_collapse":
                    group = str(flag.get("group"))
                    mode = str(flag.get("feature_mode"))
                    if group in groups and mode in FEATURE_MODES:
                        targets.append((group, mode))
            if targets:
                return targets
    return [(group_name, mode) for group_name in groups for mode in FEATURE_MODES]


def main() -> int:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    model_names = list(args.model_names or discover_prediction_models(prediction_dir))
    groups = {name: models for name, models in args.group} if args.group else default_groups_for_models(model_names)
    validate_fusion_groups(groups, model_names)
    targets = infer_targets(args, groups)

    rng = np.random.RandomState(int(args.seed))
    labels_by_split = {}
    for split in STACK_SPLITS:
        _, labels, _ = block_features(prediction_dir, [model_names[0]], split=split, feature_mode="logits")
        labels_by_split[split] = labels

    output: Dict[str, object] = {
        "prediction_dir": str(prediction_dir),
        "fusion_dir": args.fusion_dir,
        "model_names": model_names,
        "targets": [{"group": group, "feature_mode": mode} for group, mode in targets],
        "class_distribution": {split: class_distribution(labels) for split, labels in labels_by_split.items()},
        "diagnostics": {},
        "interpretation": [],
    }

    print("Class distributions")
    for split in STACK_SPLITS:
        dist = output["class_distribution"][split]
        print(
            f"  {split:<11s} n={dist['n']} "
            f"majority_class={dist['majority_class']} majority_acc={dist['majority_accuracy']:.6f}"
        )

    for group_name, mode in targets:
        if group_name not in groups:
            raise SystemExit(f"Unknown target group {group_name!r}; available={sorted(groups)}")
        model_group = groups[group_name]
        x_train, y_train, train_blocks = block_features(prediction_dir, model_group, split="stack_train", feature_mode=mode)
        x_val, y_val, _ = block_features(prediction_dir, model_group, split="stack_val", feature_mode=mode)
        x_test, y_test, _ = block_features(prediction_dir, model_group, split="final_test", feature_mode=mode)
        num_classes = int(train_blocks[0].logits.shape[1])

        train_only = fit_and_score(
            column_shuffle(x_train, rng),
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            c_grid=args.c_grid,
            max_iter=args.max_iter,
            feature_mode=mode,
            model_names=model_group,
            num_classes=num_classes,
        )
        strict_column = fit_and_score(
            column_shuffle(x_train, rng),
            y_train,
            column_shuffle(x_val, rng),
            y_val,
            column_shuffle(x_test, rng),
            y_test,
            c_grid=args.c_grid,
            max_iter=args.max_iter,
            feature_mode=mode,
            model_names=model_group,
            num_classes=num_classes,
        )
        row_permuted = fit_and_score(
            row_permute(x_train, rng),
            y_train,
            row_permute(x_val, rng),
            y_val,
            row_permute(x_test, rng),
            y_test,
            c_grid=args.c_grid,
            max_iter=args.max_iter,
            feature_mode=mode,
            model_names=model_group,
            num_classes=num_classes,
        )

        key = f"{group_name}/{mode}"
        output["diagnostics"][key] = {
            "models": model_group,
            "train_only_column_shuffle": train_only,
            "all_splits_column_shuffle": strict_column,
            "all_splits_row_permutation": row_permuted,
        }

        train_only_acc = train_only["metrics"]["final_test"]["accuracy"]
        strict_acc = strict_column["metrics"]["final_test"]["accuracy"]
        row_acc = row_permuted["metrics"]["final_test"]["accuracy"]
        majority = output["class_distribution"]["final_test"]["majority_accuracy"]
        print(f"\n{key}")
        print(f"  train-only column shuffle final_acc={train_only_acc:.6f}")
        print(f"  all-splits column shuffle final_acc={strict_acc:.6f}")
        print(f"  all-splits row permutation final_acc={row_acc:.6f}")
        print(f"  final-test majority baseline acc={majority:.6f}")
        for control_name, control in [
            ("train_only", train_only),
            ("strict_column", strict_column),
            ("row_permuted", row_permuted),
        ]:
            pred = control["prediction_distribution"]["final_test"]
            print(
                f"  {control_name:<14s} unique_pred_classes={pred['unique_predicted_classes']} "
                f"pred_fracs={np.round(pred['predicted_fractions'], 4).tolist()}"
            )
        if train_only_acc > 0.20 and strict_acc < 0.15 and row_acc < 0.15:
            output["interpretation"].append(
                {
                    "target": key,
                    "status": "train_only_control_artifact_likely",
                    "reason": "The original train-only row-shuffle warning is high, but stricter all-split shuffles collapse.",
                }
            )
        elif strict_acc > 0.20 or row_acc > 0.20:
            output["interpretation"].append(
                {
                    "target": key,
                    "status": "inspect_further",
                    "reason": "A strict all-split shuffle still has high accuracy.",
                }
            )
        else:
            output["interpretation"].append(
                {
                    "target": key,
                    "status": "collapsed",
                    "reason": "Shuffle controls are near majority/chance baselines.",
                }
            )

    print("\nInterpretation")
    for item in output["interpretation"]:
        print(f"  {item['target']}: {item['status']} - {item['reason']}")

    output_json = args.output_json
    if output_json is None and args.fusion_dir:
        output_json = str(Path(args.fusion_dir) / "row_shuffle_diagnostics.json")
    if output_json:
        save_json(output_json, output)
        print(f"\nSaved diagnostics: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
