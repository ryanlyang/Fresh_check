#!/usr/bin/env python3
"""Run frozen-score fusion across local-graph and multi-scale subjet models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed  # noqa: E402
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from run_local_graph_score_fusion import (  # noqa: E402
    _add_control_rows,
    _add_delta_columns,
    _add_equal_average_rows,
    _add_logistic_rows,
    _add_raw_model_rows,
    _add_weighted_grid_rows,
    _select_best_row,
    _write_csv,
    _write_markdown,
    save_json,
)
from teacher_logit_reco.local_graph_part.fusion import (  # noqa: E402
    LOCAL_GRAPH_SCORE_FUSION_CONTRACT,
    LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID,
    LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT,
    LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT,
    LOCAL_GRAPH_SCORE_FUSION_STEP,
    LocalGraphPredictionBlock,
    load_prediction_block,
    save_prediction_block,
    write_prediction_manifest,
)
from teacher_logit_reco.local_graph_part.protocol import (  # noqa: E402
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    local_graph_part_protocol_manifest,
)
from teacher_logit_reco.local_graph_part.train import load_local_graph_tagger_checkpoint  # noqa: E402
from teacher_logit_reco.multiscale_subjet_part.train import load_multiscale_subjet_checkpoint  # noqa: E402
from teacher_logit_reco.set_matching.train import source_metadata  # noqa: E402
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader  # noqa: E402


CROSS_FAMILY_SCORE_FUSION_STEP = "local_graph_multiscale_score_fusion_step1"
CROSS_FAMILY_SCORE_FUSION_CONTRACT = "local_graph_multiscale_score_fusion_v1"
DEFAULT_LOCAL_GRAPH_VARIANTS = (
    "hlt_part_baseline",
    "local_edgeconv_adapter",
    "local_point_attention_adapter",
    "local_point_attention_adapter_warmstart",
)
DEFAULT_MULTISCALE_VARIANTS = (
    "no_scale_bias",
    "one_scale_medium",
    "few_subjets",
)


@dataclass(frozen=True)
class FusionModelSpec:
    """One checkpoint that can emit binary logits for score fusion."""

    name: str
    family: str
    checkpoint_path: Path
    source_variant: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_dataset(hlt_cache_dir: Path, split: str, *, max_jets: int | None, verify_hash: bool) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=bool(verify_hash))
    return SubtokenHLTJetDataset(
        view,
        label_filter=(0, 1),
        label_names=LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
        max_jets=max_jets,
    )


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _load_model_for_spec(spec: FusionModelSpec, *, device):
    if spec.family == "local_graph":
        return load_local_graph_tagger_checkpoint(spec.checkpoint_path, device=device)
    if spec.family == "multiscale_subjet":
        return load_multiscale_subjet_checkpoint(spec.checkpoint_path, device=device)
    raise ValueError(f"unknown model family for {spec.name}: {spec.family}")


def _evaluate_spec(
    *,
    spec: FusionModelSpec,
    split: str,
    hlt_cache_dir: Path,
    max_jets: int | None,
    batch_size: int,
    num_workers: int,
    device,
    verify_hlt_hash: bool,
) -> LocalGraphPredictionBlock:
    torch = require_torch()
    model, payload = _load_model_for_spec(spec, device=device)
    dataset = _load_dataset(hlt_cache_dir, split, max_jets=max_jets, verify_hash=bool(verify_hlt_hash))
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        seed=7717,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    indices_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = _extract_logits(model(tokens, mask))
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            indices_rows.append(batch["indices"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 2), dtype=np.float32)
    labels_np = np.concatenate(labels_rows, axis=0) if labels_rows else np.zeros((0,), dtype=np.int64)
    indices_np = np.concatenate(indices_rows, axis=0) if indices_rows else np.zeros((0,), dtype=np.int64)
    return LocalGraphPredictionBlock(
        variant=spec.name,
        split=str(split),
        logits=logits_np,
        labels=labels_np,
        indices=indices_np,
        metadata={
            "family": spec.family,
            "source_variant": spec.source_variant,
            "checkpoint": str(spec.checkpoint_path),
            "run_report": str(spec.checkpoint_path.parent / "run_report.json"),
            "checkpoint_variant": payload.get("variant"),
            "checkpoint_epoch": payload.get("epoch"),
            "dataset": dict(dataset.metadata),
        },
    )


def _ensure_prediction_block(
    *,
    prediction_dir: Path,
    spec: FusionModelSpec,
    split: str,
    hlt_cache_dir: Path,
    max_jets: int | None,
    batch_size: int,
    num_workers: int,
    device,
    overwrite: bool,
    verify_hlt_hash: bool,
) -> LocalGraphPredictionBlock:
    npz_path = prediction_dir / spec.name / f"{split}_predictions.npz"
    if npz_path.exists() and not bool(overwrite):
        return load_prediction_block(prediction_dir, spec.name, split)
    block = _evaluate_spec(
        spec=spec,
        split=split,
        hlt_cache_dir=hlt_cache_dir,
        max_jets=max_jets,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device=device,
        verify_hlt_hash=bool(verify_hlt_hash),
    )
    save_prediction_block(block, prediction_dir, overwrite=bool(overwrite))
    return block


def _available_specs(
    *,
    local_graph_tagger_root: Path,
    multiscale_tagger_root: Path,
    local_graph_variants: Sequence[str],
    multiscale_variants: Sequence[str],
    multiscale_prefix: str,
    require_all: bool,
) -> tuple[list[FusionModelSpec], list[str]]:
    specs: list[FusionModelSpec] = []
    missing: list[str] = []
    for variant in local_graph_variants:
        checkpoint = local_graph_tagger_root / str(variant) / "best_model_val.pt"
        if checkpoint.exists():
            specs.append(
                FusionModelSpec(
                    name=str(variant),
                    family="local_graph",
                    checkpoint_path=checkpoint,
                    source_variant=str(variant),
                )
            )
        else:
            missing.append(f"local_graph:{variant}")
    for variant in multiscale_variants:
        checkpoint = multiscale_tagger_root / str(variant) / "best_model_val.pt"
        name = f"{multiscale_prefix}{variant}"
        if checkpoint.exists():
            specs.append(
                FusionModelSpec(
                    name=name,
                    family="multiscale_subjet",
                    checkpoint_path=checkpoint,
                    source_variant=str(variant),
                )
            )
        else:
            missing.append(f"multiscale_subjet:{variant}")
    if missing and bool(require_all):
        raise FileNotFoundError(f"missing checkpoints for requested fusion variants: {missing}")
    return specs, missing


def _default_model_sets(variants: Sequence[str], baseline_variant: str, *, multiscale_prefix: str) -> list[tuple[str, ...]]:
    available = set(variants)
    curated = [
        (baseline_variant, "local_edgeconv_adapter"),
        (baseline_variant, "local_point_attention_adapter"),
        (baseline_variant, "local_point_attention_adapter_warmstart"),
        (baseline_variant, "local_edgeconv_adapter", "local_point_attention_adapter"),
        (baseline_variant, f"{multiscale_prefix}no_scale_bias"),
        (baseline_variant, f"{multiscale_prefix}one_scale_medium"),
        (baseline_variant, f"{multiscale_prefix}few_subjets"),
        (baseline_variant, "local_edgeconv_adapter", "local_point_attention_adapter", f"{multiscale_prefix}no_scale_bias"),
        (baseline_variant, "local_edgeconv_adapter", "local_point_attention_adapter", f"{multiscale_prefix}one_scale_medium"),
        (baseline_variant, "local_edgeconv_adapter", "local_point_attention_adapter", f"{multiscale_prefix}few_subjets"),
        (
            baseline_variant,
            "local_edgeconv_adapter",
            "local_point_attention_adapter",
            f"{multiscale_prefix}no_scale_bias",
            f"{multiscale_prefix}one_scale_medium",
            f"{multiscale_prefix}few_subjets",
        ),
    ]
    model_sets: list[tuple[str, ...]] = []
    for candidate in curated:
        if all(item in available for item in candidate):
            model_sets.append(tuple(candidate))
    return _dedupe_model_sets(model_sets)


def _all_subset_model_sets(variants: Sequence[str], baseline_variant: str, *, max_size: int) -> list[tuple[str, ...]]:
    if baseline_variant not in variants:
        return []
    others = [variant for variant in variants if variant != baseline_variant]
    model_sets: list[tuple[str, ...]] = []
    for size in range(1, max(1, int(max_size))):
        for subset in combinations(others, size):
            model_sets.append((baseline_variant, *subset))
    return _dedupe_model_sets(model_sets)


def _dedupe_model_sets(model_sets: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    deduped: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for model_set in model_sets:
        key = tuple(str(item) for item in model_set)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _model_sets_from_args(args: argparse.Namespace, variants: Sequence[str]) -> list[tuple[str, ...]]:
    if args.model_set:
        requested = []
        for text in args.model_set:
            requested.append(tuple(item.strip() for item in str(text).replace(",", " ").split() if item.strip()))
        available = set(variants)
        missing = sorted({item for model_set in requested for item in model_set if item not in available})
        if missing:
            raise ValueError(f"requested model sets include unavailable variants: {missing}")
        return _dedupe_model_sets(requested)
    if bool(args.all_model_subsets):
        return _all_subset_model_sets(variants, args.baseline_variant, max_size=int(args.max_model_set_size))
    return _default_model_sets(variants, args.baseline_variant, multiscale_prefix=args.multiscale_prefix)


def run_cross_family_fusion(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    set_training_seed(int(args.seed))
    device = resolve_device(args.device)
    local_graph_root = Path(args.local_graph_root)
    multiscale_root = Path(args.multiscale_root)
    local_graph_tagger_root = Path(args.local_graph_tagger_root or local_graph_root / "taggers")
    multiscale_tagger_root = Path(args.multiscale_tagger_root or multiscale_root / "taggers")
    hlt_cache_dir = Path(args.hlt_cache_dir)
    output_dir = Path(args.output_dir or local_graph_root / "score_fusion_with_multiscale")
    prediction_dir = Path(args.prediction_dir or output_dir / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs, missing_specs = _available_specs(
        local_graph_tagger_root=local_graph_tagger_root,
        multiscale_tagger_root=multiscale_tagger_root,
        local_graph_variants=args.local_graph_variants,
        multiscale_variants=args.multiscale_variants,
        multiscale_prefix=str(args.multiscale_prefix),
        require_all=bool(args.require_all_variants),
    )
    variant_names = [spec.name for spec in specs]
    if args.baseline_variant not in variant_names:
        raise FileNotFoundError(f"baseline variant is not available: {args.baseline_variant}")
    blocks_by_split: dict[str, dict[str, LocalGraphPredictionBlock]] = {
        LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT: {},
        LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT: {},
    }
    all_blocks: list[LocalGraphPredictionBlock] = []
    for split, max_jets in (
        (LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT, args.max_stack_jets),
        (LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT, args.max_final_test_jets),
    ):
        if split == LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT and not bool(args.confirm_final_test):
            raise ValueError("Refusing to evaluate final_test fusion without --confirm-final-test")
        for spec in specs:
            block = _ensure_prediction_block(
                prediction_dir=prediction_dir,
                spec=spec,
                split=split,
                hlt_cache_dir=hlt_cache_dir,
                max_jets=max_jets,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
                overwrite=bool(args.overwrite_predictions),
                verify_hlt_hash=not bool(args.skip_hlt_hash_check),
            )
            blocks_by_split[split][spec.name] = block
            all_blocks.append(block)

    model_sets = _model_sets_from_args(args, variant_names)
    if not model_sets:
        raise ValueError("No fusion model sets could be built from the available variants")

    primary_metric = str(args.primary_metric)
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    _add_raw_model_rows(rows, blocks_by_split=blocks_by_split, variants=variant_names, primary_metric=primary_metric)
    _add_equal_average_rows(rows, blocks_by_split=blocks_by_split, model_sets=model_sets, primary_metric=primary_metric)
    _add_weighted_grid_rows(
        rows,
        weight_rows,
        blocks_by_split=blocks_by_split,
        model_sets=model_sets,
        primary_metric=primary_metric,
        weight_grid_step=float(args.weight_grid_step),
    )
    _add_logistic_rows(
        rows,
        weight_rows,
        blocks_by_split=blocks_by_split,
        model_sets=model_sets,
        primary_metric=primary_metric,
        c_grid=tuple(float(value) for value in args.c_grid),
        max_iter=int(args.max_iter),
        prefer_sklearn=not bool(args.no_sklearn),
    )
    if bool(args.run_controls):
        _add_control_rows(
            rows,
            blocks_by_split=blocks_by_split,
            model_sets=model_sets,
            baseline_variant=args.baseline_variant,
            primary_metric=primary_metric,
            c_grid=tuple(float(value) for value in args.c_grid),
            max_iter=int(args.max_iter),
            prefer_sklearn=not bool(args.no_sklearn),
            seed=int(args.control_seed),
        )
    _add_delta_columns(rows, baseline_variant=args.baseline_variant)

    valid_rows = [row for row in rows if not row.get("negative_control")]
    control_rows = [row for row in rows if row.get("negative_control")]
    best_valid = _select_best_row(valid_rows)
    best_control = _select_best_row(control_rows)
    raw_baseline = next((row for row in rows if row["method"] == f"raw_{args.baseline_variant}"), None)
    calibrated_baseline = next((row for row in rows if row["method"] == "control_baseline_only_logistic_margin"), None)
    raw_metric = raw_baseline.get("final_test_primary_metric_value") if raw_baseline else None
    calibrated_metric = calibrated_baseline.get("final_test_primary_metric_value") if calibrated_baseline else None

    problems = []
    if best_valid is None:
        problems.append("no valid fusion rows were produced")
    if raw_baseline is None:
        problems.append("raw HLT baseline row was not produced")
    if calibrated_baseline is None and bool(args.run_controls):
        problems.append("calibrated HLT-only baseline row was not produced")

    output_paths = {
        "fusion_report_json": str(output_dir / "fusion_report.json"),
        "fusion_report_md": str(output_dir / "fusion_report.md"),
        "fusion_metric_table_csv": str(output_dir / "fusion_metric_table.csv"),
        "fusion_weights_csv": str(output_dir / "fusion_weights.csv"),
        "fusion_controls_csv": str(output_dir / "fusion_controls.csv"),
        "fusion_prediction_manifest_json": str(output_dir / "fusion_prediction_manifest.json"),
        "run_report_json": str(output_dir / "run_report.json"),
    }
    manifest = write_prediction_manifest(
        output_paths["fusion_prediction_manifest_json"],
        blocks=all_blocks,
        extra={
            "local_graph_root": str(local_graph_root),
            "local_graph_tagger_root": str(local_graph_tagger_root),
            "multiscale_root": str(multiscale_root),
            "multiscale_tagger_root": str(multiscale_tagger_root),
            "hlt_cache_dir": str(hlt_cache_dir),
            "available_variants": variant_names,
            "missing_specs": missing_specs,
            "model_sets": [" ".join(model_set) for model_set in model_sets],
            "stack_split": LOCAL_GRAPH_SCORE_FUSION_STACK_SPLIT,
            "final_test_split": LOCAL_GRAPH_SCORE_FUSION_FINAL_SPLIT,
            "max_stack_jets": args.max_stack_jets,
            "max_final_test_jets": args.max_final_test_jets,
        },
    )
    report = {
        "step": CROSS_FAMILY_SCORE_FUSION_STEP,
        "contract": CROSS_FAMILY_SCORE_FUSION_CONTRACT,
        "inherits_fusion_contract": LOCAL_GRAPH_SCORE_FUSION_CONTRACT,
        "inherits_fusion_step": LOCAL_GRAPH_SCORE_FUSION_STEP,
        "ok": len(problems) == 0,
        "problems": problems,
        "source": source_metadata(),
        "protocol": local_graph_part_protocol_manifest(),
        "config": _jsonable(vars(args)),
        "local_graph_root": str(local_graph_root),
        "multiscale_root": str(multiscale_root),
        "hlt_cache_dir": str(hlt_cache_dir),
        "prediction_manifest": manifest,
        "summary": {
            "primary_metric": primary_metric,
            "baseline_variant": args.baseline_variant,
            "available_variants": variant_names,
            "missing_specs": missing_specs,
            "model_sets": [" ".join(model_set) for model_set in model_sets],
            "raw_baseline_final_test_primary_metric": raw_metric,
            "calibrated_baseline_final_test_primary_metric": calibrated_metric,
            "best_valid_method": best_valid.get("method") if best_valid else None,
            "best_valid_model_set": best_valid.get("model_set") if best_valid else None,
            "best_valid_final_test_primary_metric": best_valid.get("final_test_primary_metric_value")
            if best_valid
            else None,
            "best_valid_delta_vs_raw_baseline": best_valid.get("delta_vs_raw_hlt_baseline") if best_valid else None,
            "best_valid_delta_vs_calibrated_baseline": best_valid.get("delta_vs_calibrated_hlt_baseline")
            if best_valid
            else None,
            "best_control_method": best_control.get("method") if best_control else None,
            "best_control_final_test_primary_metric": best_control.get("final_test_primary_metric_value")
            if best_control
            else None,
            "interpretation_rule": (
                "A cross-family fusion is interesting only if it improves final_test FPR@50 versus both raw and calibrated HLT baselines."
            ),
        },
        "fusion_metric_table": rows,
        "fusion_weights": weight_rows,
        "fusion_controls": control_rows,
        "outputs": output_paths,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - start),
            "device": str(device),
        },
    }
    _write_csv(Path(output_paths["fusion_metric_table_csv"]), rows)
    _write_csv(Path(output_paths["fusion_weights_csv"]), weight_rows)
    _write_csv(Path(output_paths["fusion_controls_csv"]), control_rows)
    save_json(Path(output_paths["fusion_report_json"]), report)
    save_json(Path(output_paths["run_report_json"]), report)
    _write_markdown(Path(output_paths["fusion_report_md"]), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-graph-root", required=True)
    parser.add_argument("--multiscale-root", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--local-graph-tagger-root", default=None)
    parser.add_argument("--multiscale-tagger-root", default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-graph-variants", nargs="*", default=list(DEFAULT_LOCAL_GRAPH_VARIANTS))
    parser.add_argument("--multiscale-variants", nargs="*", default=list(DEFAULT_MULTISCALE_VARIANTS))
    parser.add_argument("--multiscale-prefix", default="ms_")
    parser.add_argument("--baseline-variant", default="hlt_part_baseline")
    parser.add_argument("--model-set", action="append", default=None, help="Explicit model set; use comma or spaces between variant names.")
    parser.add_argument("--all-model-subsets", action="store_true")
    parser.add_argument("--max-model-set-size", type=int, default=5)
    parser.add_argument("--primary-metric", default=LOCAL_GRAPH_PART_PRIMARY_METRIC)
    parser.add_argument("--max-stack-jets", type=int, default=1000000)
    parser.add_argument("--max-final-test-jets", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7717)
    parser.add_argument("--c-grid", nargs="*", type=float, default=list(LOCAL_GRAPH_SCORE_FUSION_DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    parser.add_argument("--control-seed", type=int, default=17717)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--require-all-variants", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--no-sklearn", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    args.run_controls = not bool(args.skip_controls)
    return args


def main() -> int:
    report = run_cross_family_fusion(parse_args())
    summary = report["summary"]
    print("local_graph_multiscale_score_fusion_complete:")
    print(f"  ok: {report['ok']}")
    print(f"  output_dir: {Path(report['outputs']['fusion_report_json']).parent}")
    print(f"  primary_metric: {summary['primary_metric']}")
    print(f"  raw_baseline: {summary['raw_baseline_final_test_primary_metric']}")
    print(f"  calibrated_baseline: {summary['calibrated_baseline_final_test_primary_metric']}")
    print(f"  best_valid_method: {summary['best_valid_method']}")
    print(f"  best_valid_model_set: {summary['best_valid_model_set']}")
    print(f"  best_valid_final_metric: {summary['best_valid_final_test_primary_metric']}")
    print(f"  fusion_report: {report['outputs']['fusion_report_json']}")
    print(f"  metric_table: {report['outputs']['fusion_metric_table_csv']}")
    if report["problems"]:
        print("  problems:")
        for problem in report["problems"]:
            print(f"    - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
