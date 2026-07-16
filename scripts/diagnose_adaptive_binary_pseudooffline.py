#!/usr/bin/env python3
"""Evaluate non-selection ABPH pseudo-view and fusion-use diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_PSEUDO_INPUT_ABLATIONS,
    ABPH_TAGGER_DIAGNOSTIC_CONTRACT,
    FrozenPseudoBatchSource,
    PseudoViewInputs,
    TaggerDiagnosticOverride,
    ablate_pseudo_inputs,
    build_variant_hierarchy_aware_tagger,
    resolve_variant_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--split", default="model_val", choices=("model_val",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("ABPH_DIAGNOSTIC_BATCH_SIZE", "128")),
    )
    parser.add_argument(
        "--maximum-batches",
        type=int,
        default=int(os.environ.get("ABPH_DIAGNOSTIC_MAX_BATCHES", "0")),
    )
    parser.add_argument(
        "--smoke", action="store_true", default=os.environ.get("ABPH_SMOKE", "0") == "1"
    )
    return parser


def _torch_load(path: Path, device: Any) -> dict[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - old research PyTorch
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint {path} is not a mapping")
    return payload


def _model_state(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("selected tagger checkpoint has no model state")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _source_for_variant(
    root: Path,
    variant: str,
    *,
    split: str,
    batch_size: int,
    maximum_batches: int | None,
) -> FrozenPseudoBatchSource:
    resolved = resolve_variant_config(variant)
    run_id = str(resolved["variant"]["run_id"])
    dual = bool(resolved["model"]["fusion"].get("dual_hierarchy"))
    if dual and run_id != "E11":
        cache_dirs = (root / "pseudo_predictions" / "E7_shared_root_dual" / split,)
        independent = False
    elif run_id == "E11":
        cache_dirs = (
            root / "pseudo_predictions" / "D1_kt32_mh4_particles" / split,
            root / "pseudo_predictions" / "D2_ca32_mh4_particles" / split,
        )
        independent = True
    elif resolved["model"]["hierarchy"].get("grouping") == "cambridge_aachen":
        cache_dirs = (root / "pseudo_predictions" / "D2_ca32_mh4_particles" / split,)
        independent = False
    else:
        cache_dirs = (root / "pseudo_predictions" / "D1_kt32_mh4_particles" / split,)
        independent = False
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"diagnostic pseudo cache is missing: {cache_dir}")
    return FrozenPseudoBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        cache_dirs=cache_dirs,
        split=split,
        batch_size=max(2, int(batch_size)),
        independent_roots=independent,
        maximum_batches=maximum_batches,
    )


def _evaluate(
    model: Any,
    source: FrozenPseudoBatchSource,
    *,
    device: Any,
    pseudo_transform: Callable[[PseudoViewInputs], PseudoViewInputs] | None = None,
    override: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    correct = 0
    count = 0
    loss_sum = 0.0
    batches = 0
    context = override() if override is not None else None
    if context is not None:
        context.__enter__()
    try:
        with torch.no_grad():
            for batch in source.iter_batches(shuffle=False, seed=24731):
                hlt_tokens = torch.as_tensor(batch.hlt_tokens, device=device).float()
                hlt_mask = torch.as_tensor(batch.hlt_mask, device=device).bool()
                labels = torch.as_tensor(batch.labels, device=device).long()
                pseudo = PseudoViewInputs.from_deployable_batch(
                    batch.pseudo, device=device, dtype=hlt_tokens.dtype
                )
                if pseudo_transform is not None:
                    pseudo = pseudo_transform(pseudo)
                roots = None
                if batch.independent_roots is not None:
                    roots = {
                        name: torch.as_tensor(value, device=device, dtype=hlt_tokens.dtype)
                        for name, value in batch.independent_roots.items()
                    }
                logits = model(
                    hlt_tokens,
                    hlt_mask,
                    pseudo,
                    independent_root_ledgers=roots,
                ).logits
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                loss_sum += float(loss.detach().cpu())
                correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
                count += int(labels.numel())
                batches += 1
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    if count == 0:
        raise RuntimeError("diagnostic evaluation produced zero jets")
    return {
        "available": True,
        "accuracy": correct / count,
        "loss": loss_sum / count,
        "n_jets": count,
        "n_batches": batches,
    }


def _diagnose_variant(args: argparse.Namespace, variant: str, device: Any) -> list[dict[str, Any]]:
    root = Path(args.campaign_root)
    checkpoint = root / "runs" / variant / "best_model_val.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected diagnostic checkpoint is missing: {checkpoint}")
    model = build_variant_hierarchy_aware_tagger(variant, smoke=bool(args.smoke)).to(device)
    model.load_state_dict(_model_state(_torch_load(checkpoint, device)), strict=True)
    model.eval()
    maximum_batches = 1 if args.smoke else (int(args.maximum_batches) or None)

    def source() -> FrozenPseudoBatchSource:
        return _source_for_variant(
            root,
            variant,
            split=args.split,
            batch_size=args.batch_size,
            maximum_batches=maximum_batches,
        )

    baseline = _evaluate(model, source(), device=device)
    rows = [{"variant": variant, "diagnostic": "unaltered", **baseline}]
    for ablation in ABPH_PSEUDO_INPUT_ABLATIONS:
        if ablation == "remove_hierarchy_depth":
            transforms = tuple(
                (
                    f"{ablation}_{depth}",
                    lambda pseudo, depth=depth: ablate_pseudo_inputs(
                        pseudo, ablation, hierarchy_depth=depth
                    ),
                )
                for depth in range(1, 6)
            )
        else:
            transforms = ((ablation, lambda pseudo, mode=ablation: ablate_pseudo_inputs(pseudo, mode)),)
        for label, transform in transforms:
            metrics = _evaluate(model, source(), device=device, pseudo_transform=transform)
            rows.append(
                {
                    "variant": variant,
                    "diagnostic": label,
                    **metrics,
                    "accuracy_delta_vs_unaltered": metrics["accuracy"] - baseline["accuracy"],
                    "loss_delta_vs_unaltered": metrics["loss"] - baseline["loss"],
                }
            )
    for index in range(len(model.fusion_stacks)):
        metrics = _evaluate(
            model,
            source(),
            device=device,
            override=lambda index=index: TaggerDiagnosticOverride(
                model, fusion_location_index=index
            ),
        )
        rows.append(
            {
                "variant": variant,
                "diagnostic": f"remove_fusion_location_{index}",
                **metrics,
                "accuracy_delta_vs_unaltered": metrics["accuracy"] - baseline["accuracy"],
                "loss_delta_vs_unaltered": metrics["loss"] - baseline["loss"],
            }
        )
    for trust in (0.0, 1.0):
        metrics = _evaluate(
            model,
            source(),
            device=device,
            override=lambda trust=trust: TaggerDiagnosticOverride(model, forced_trust=trust),
        )
        rows.append(
            {
                "variant": variant,
                "diagnostic": f"forced_trust_{int(trust)}",
                **metrics,
                "accuracy_delta_vs_unaltered": metrics["accuracy"] - baseline["accuracy"],
                "loss_delta_vs_unaltered": metrics["loss"] - baseline["loss"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        rows.extend(_diagnose_variant(args, str(variant), device))
    output_dir = Path(args.campaign_root) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tagger_use_metrics.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "contract": ABPH_TAGGER_DIAGNOSTIC_CONTRACT,
        "ok": True,
        "selection_eligible": False,
        "split": args.split,
        "variants": list(args.variants),
        "rows": rows,
        "metrics_csv": str(csv_path),
        "final_test_loaded": False,
        "offline_inputs_loaded": False,
        "teacher_logits_loaded": False,
    }
    _atomic_json(output_dir / "tagger_use_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
