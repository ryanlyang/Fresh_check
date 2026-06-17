#!/usr/bin/env python3
"""Run a compact V2-reconstructor teacher-logit + dual-view debug probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.dual_view import DualViewTaggerTrainConfig, train_dual_view_tagger  # noqa: E402
from jetclass_fresh.fusion import (  # noqa: E402
    classification_metrics_from_logits,
    evaluate_dual_view_model,
    load_dual_view_model_from_checkpoint,
)
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed  # noqa: E402
from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_split_manifest, manifest_hash  # noqa: E402
from jetclass_fresh.reconstructor import (  # noqa: E402
    StageAReconstructorTrainConfig,
    build_reconstructor,
    get_reconstructor_variant_config,
    reconstructor_checkpoint_payload,
)
from teacher_logit_reco.losses import TeacherLogitRecoLossConfig, compute_teacher_logit_reco_loss  # noqa: E402
from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES, assert_teacher_frozen, load_frozen_teacher  # noqa: E402
from teacher_logit_reco.train_global_transformer import (  # noqa: E402
    PairedTeacherLogitDataset,
    make_teacher_logit_loader,
    source_metadata,
)
from teacher_logit_reco.views import (  # noqa: E402
    SoftReconstructedView,
    load_paired_jet_views,
    slice_jet_view,
    summarize_paired_jet_views,
)


EXPERIMENT_STEP = "v2_teacher_logit_dualview_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default="part")
    parser.add_argument("--variant", default="m2_base")
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--test-split", default="final_test")
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1407)
    parser.add_argument("--reco-epochs", type=int, default=8)
    parser.add_argument("--dual-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dual-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--dual-lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--reco-early-stop-patience", type=int, default=3)
    parser.add_argument("--dual-early-stop-patience", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--model-size", choices=("tiny", "base"), default="base")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--teacher-weight-threshold", type=float, default=0.0)
    parser.add_argument("--reco-weight-threshold", type=float, default=0.0)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--correction-budget-weight", type=float, default=0.02)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()


def _average_metric_rows(rows: list[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"n_jets": 0, "n_batches": 0, "total_loss": float("nan")}
    keys = sorted({key for row in rows for key in row if key not in ("n_jets", "n_batches")})
    weights = np.asarray([max(1, int(row.get("n_jets", 0))) for row in rows], dtype=np.float64)
    output: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        output[key] = float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan")
    output["n_jets"] = int(sum(int(row.get("n_jets", 0)) for row in rows))
    output["n_batches"] = int(sum(int(row.get("n_batches", 0)) for row in rows))
    return output


def _v2_output_to_soft_view(output, batch: Mapping[str, Any], *, split: str, jet_ids) -> SoftReconstructedView:
    torch = require_torch()
    split_tokens = getattr(output, "split_tokens", None)
    split_weights = getattr(output, "split_weights", None)
    generated_tokens = getattr(output, "generated_tokens", None)
    generated_weights = getattr(output, "generated_weights", None)
    if split_tokens is None:
        split_tokens = output.tokens.new_zeros((output.tokens.shape[0], 0, output.tokens.shape[-1]))
    if split_weights is None:
        split_weights = output.weights.new_zeros((output.weights.shape[0], 0))
    if generated_tokens is None:
        generated_tokens = output.tokens.new_zeros((output.tokens.shape[0], 0, output.tokens.shape[-1]))
    if generated_weights is None:
        generated_weights = output.weights.new_zeros((output.weights.shape[0], 0))

    extra_tokens = torch.cat([split_tokens, generated_tokens], dim=1)
    extra_weights = torch.cat([split_weights, generated_weights], dim=1)
    parent_tokens = getattr(output, "corrected_parent_tokens", None)
    parent_weights = getattr(output, "corrected_parent_weights", None)
    if parent_tokens is None:
        parent_tokens = getattr(output, "edited_tokens", batch["hlt_tokens"])
    if parent_weights is None:
        parent_weights = getattr(output, "edited_weights", batch["hlt_mask"].float())

    aux = {
        "parent_delta": parent_tokens - batch["hlt_tokens"],
        "sanitized_hlt_tokens": batch["hlt_tokens"],
        "sanitized_hlt_mask": batch["hlt_mask"],
        "parent_weights": parent_weights,
        "extra_tokens": extra_tokens,
        "extra_weights": extra_weights,
    }
    return SoftReconstructedView(
        tokens=output.tokens,
        mask=output.candidate_mask,
        weights=output.weights,
        labels=batch["labels"],
        jet_ids=list(jet_ids),
        split=split,
        metadata={
            "view": "v2_m2_full_candidate_soft_reco",
            "construction": "v2_original_mechanism_candidates",
            "parent_aligned_downstream_view": True,
            "teacher_loss_uses_full_candidate_view": True,
        },
        aux=aux,
    )


def _batch_to_device(batch: Mapping[str, Any], device) -> Dict[str, Any]:
    return {
        "hlt_tokens": batch["hlt_tokens"].to(device=device, non_blocking=True),
        "hlt_mask": batch["hlt_mask"].to(device=device, non_blocking=True),
        "offline_tokens": batch["offline_tokens"].to(device=device, non_blocking=True),
        "offline_mask": batch["offline_mask"].to(device=device, non_blocking=True),
        "labels": batch["labels"].to(device=device, non_blocking=True),
        "jet_ids": batch["jet_ids"],
    }


def run_v2_teacher_logit_epoch(
    reconstructor,
    teacher,
    loader,
    *,
    device,
    loss_config: TeacherLogitRecoLossConfig,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float = 1.0,
    max_batches: int | None = None,
) -> Dict[str, float]:
    torch = require_torch()
    is_train = optimizer is not None
    reconstructor.train(bool(is_train))
    teacher.model.eval()
    assert_teacher_frozen(teacher)
    rows: list[Dict[str, float]] = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    autocast_enabled = bool(amp and getattr(device, "type", None) == "cuda")

    with context:
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _batch_to_device(raw_batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = reconstructor(batch["hlt_tokens"], batch["hlt_mask"])
                reco_view = _v2_output_to_soft_view(
                    output,
                    batch,
                    split=loader.dataset.split,
                    jet_ids=batch["jet_ids"],
                )
                offline_logits = teacher.forward_view_no_grad(batch["offline_tokens"], batch["offline_mask"])
                reco_logits = teacher.forward_soft_view(reco_view)
                loss = compute_teacher_logit_reco_loss(
                    offline_logits=offline_logits,
                    reco_logits=reco_logits,
                    labels=batch["labels"],
                    reco_view=reco_view,
                    offline_tokens=batch["offline_tokens"],
                    offline_mask=batch["offline_mask"],
                    config=loss_config,
                )
            if not bool(torch.isfinite(loss.total_loss)):
                raise FloatingPointError(f"Non-finite V2 teacher-logit loss in batch {batch_index}")
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss.total_loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.total_loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), float(grad_clip_norm))
                    optimizer.step()
            row = loss.detached_float_dict()
            row["n_jets"] = int(batch["labels"].numel())
            row["n_batches"] = 1
            rows.append(row)
    return _average_metric_rows(rows)


def train_v2_teacher_logit_reconstructor(args: argparse.Namespace, *, output_dir: Path, device):
    torch = require_torch()
    stage_a_dir = output_dir / "stage_a"
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    variant_config = get_reconstructor_variant_config(args.variant)
    stage_config = StageAReconstructorTrainConfig(
        output_dir=str(stage_a_dir),
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        variant=args.variant,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.reco_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.reco_early_stop_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        read_chunk_size=args.read_chunk_size,
    )
    loss_config = TeacherLogitRecoLossConfig(
        teacher_kl_weight=args.teacher_kl_weight,
        ce_weight=args.ce_weight,
        correction_budget_weight=args.correction_budget_weight,
        jet_summary_weight=args.jet_summary_weight,
        temperature=args.temperature,
    )
    train_pair = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.train_split,
        data_dir=args.data_dir,
        max_jets=args.train_size,
        read_chunk_size=args.read_chunk_size,
    )
    val_pair = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.val_split,
        data_dir=args.data_dir,
        max_jets=args.val_size,
        read_chunk_size=args.read_chunk_size,
    )
    train_dataset = PairedTeacherLogitDataset(train_pair)
    val_dataset = PairedTeacherLogitDataset(val_pair)
    train_loader = make_teacher_logit_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = make_teacher_logit_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )

    reconstructor = build_reconstructor(variant_config).to(device)
    teacher = load_frozen_teacher(
        args.teacher_checkpoint,
        architecture=args.teacher_architecture,
        device=str(device),
        max_constits=args.max_constits,
        weight_threshold=args.teacher_weight_threshold,
    )
    assert_teacher_frozen(teacher)
    optimizer = torch.optim.AdamW(
        reconstructor.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(not args.no_amp and device.type == "cuda"))
    source = source_metadata()
    manifest_sha = manifest_hash(load_split_manifest(args.manifest_path))
    run_metadata = {
        "experiment_step": f"{EXPERIMENT_STEP}_stage_a",
        "training_objective": "teacher_logit_loss_on_v2_full_candidate_view",
        "downstream_corrected_view": "parent_aligned_v2_corrected_view",
        "config": asdict(stage_config),
        "variant_config": asdict(variant_config),
        "loss_config": loss_config.to_dict(),
        "teacher": dict(teacher.metadata),
        "source": source,
        "manifest_hash": manifest_sha,
        "train_pair": summarize_paired_jet_views(train_pair),
        "val_pair": summarize_paired_jet_views(val_pair),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "leakage_rule": (
            "This debug run trains the reconstructor on model_train and selects on model_val. "
            "Offline constituents and offline-teacher logits are supervision only. "
            "The trained reconstructor and dual-view tagger consume fixed-HLT tokens at inference."
        ),
        "no_stack_partitions_loaded": True,
    }
    save_json(stage_a_dir / "config.json", run_metadata)

    curves: list[Dict[str, Any]] = []
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, int(args.reco_epochs) + 1):
        train_metrics = run_v2_teacher_logit_epoch(
            reconstructor,
            teacher,
            train_loader,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            amp=not bool(args.no_amp),
            grad_clip_norm=args.grad_clip_norm,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_v2_teacher_logit_epoch(
            reconstructor,
            teacher,
            val_loader,
            device=device,
            loss_config=loss_config,
            amp=False,
            max_batches=args.max_val_batches,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(stage_a_dir / "training_curves.json", {"epochs": curves})

        val_loss = float(val_metrics.get("total_loss", float("nan")))
        payload = reconstructor_checkpoint_payload(
            reconstructor,
            optimizer,
            epoch=epoch,
            config=stage_config,
            variant_config=variant_config,
            metrics=row,
        )
        payload.update(
            {
                "experiment_step": f"{EXPERIMENT_STEP}_stage_a",
                "training_objective": "teacher_logit_loss_on_v2_full_candidate_view",
                "loss_config": loss_config.to_dict(),
                "teacher_metadata": dict(teacher.metadata),
                "source": source,
            }
        )
        torch.save(payload, stage_a_dir / "last.pt")
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_epoch = int(epoch)
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(payload, stage_a_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1
        if (
            int(args.reco_early_stop_patience) >= 0
            and epochs_without_improvement >= int(args.reco_early_stop_patience)
        ):
            break

    if best_epoch < 0 or not (stage_a_dir / "best_model_val.pt").exists():
        raise FloatingPointError("Stage A did not produce a finite teacher-logit validation loss")
    report = {
        "experiment_step": f"{EXPERIMENT_STEP}_stage_a",
        "variant": args.variant,
        "best_epoch": int(best_epoch),
        "best_model_val_total_loss": float(best_val_loss),
        "best_model_val_reco_argmax_accuracy": float(
            curves[best_epoch - 1]["model_val"].get("metric_reco_argmax_accuracy", 0.0)
        ),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(stage_a_dir / "best_model_val.pt"),
        "last_checkpoint": str(stage_a_dir / "last.pt"),
        "teacher": dict(teacher.metadata),
        "not_a_classifier": True,
        "inference_consumes_hlt_only": True,
    }
    save_json(stage_a_dir / "model_val_reconstruction_report.json", report)
    save_json(stage_a_dir / "run_report.json", report)
    return report, teacher


def evaluate_teacher_paths(args: argparse.Namespace, *, output_dir: Path, reconstructor, teacher, device):
    torch = require_torch()
    test_pair = load_paired_jet_views(
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        split=args.test_split,
        data_dir=args.data_dir,
        max_jets=args.test_size,
        read_chunk_size=args.read_chunk_size,
    )
    dataset = PairedTeacherLogitDataset(test_pair)
    loader = make_teacher_logit_loader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 200,
    )
    reconstructor.eval()
    teacher.model.eval()
    labels_rows: list[np.ndarray] = []
    logits_by_name: Dict[str, list[np.ndarray]] = {
        "offline_teacher_on_offline": [],
        "offline_teacher_on_hlt": [],
        "offline_teacher_on_reconstructed": [],
    }
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_index >= int(args.max_eval_batches):
                break
            batch = _batch_to_device(raw_batch, device)
            output = reconstructor(batch["hlt_tokens"], batch["hlt_mask"])
            reco_view = _v2_output_to_soft_view(
                output,
                batch,
                split=loader.dataset.split,
                jet_ids=batch["jet_ids"],
            )
            logits_by_name["offline_teacher_on_offline"].append(
                teacher.forward_view_no_grad(batch["offline_tokens"], batch["offline_mask"]).detach().cpu().numpy()
            )
            logits_by_name["offline_teacher_on_hlt"].append(
                teacher.forward_view_no_grad(batch["hlt_tokens"], batch["hlt_mask"]).detach().cpu().numpy()
            )
            logits_by_name["offline_teacher_on_reconstructed"].append(
                teacher.forward_soft_view_no_grad(reco_view).detach().cpu().numpy()
            )
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))

    labels = np.concatenate(labels_rows, axis=0)
    metrics = {}
    for name, rows in logits_by_name.items():
        logits = np.concatenate(rows, axis=0).astype(np.float32)
        metrics[name] = classification_metrics_from_logits(logits, labels)
    np.savez_compressed(
        output_dir / "teacher_path_test_logits.npz",
        labels=labels.astype(np.int64),
        **{name: np.concatenate(rows, axis=0).astype(np.float32) for name, rows in logits_by_name.items()},
    )
    return metrics, test_pair


def main() -> int:
    args = parse_args()
    if args.train_split != "model_train" or args.val_split != "model_val":
        raise ValueError("This debug runner may train only on model_train and select only on model_val")
    for name in ("train_size", "val_size", "test_size", "reco_epochs", "dual_epochs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")

    torch = require_torch()
    set_training_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "experiment_config.json",
        {
            "experiment_step": EXPERIMENT_STEP,
            "args": vars(args),
            "label_names": list(LABEL_NAMES),
            "source": source_metadata(),
        },
    )

    stage_a_report, teacher = train_v2_teacher_logit_reconstructor(args, output_dir=output_dir, device=device)

    stage2_dir = output_dir / "stage2_dual_view"
    train_hlt_view = slice_jet_view(load_cached_hlt_view(args.hlt_cache_dir, args.train_split), args.train_size)
    val_hlt_view = slice_jet_view(load_cached_hlt_view(args.hlt_cache_dir, args.val_split), args.val_size)
    dual_config = DualViewTaggerTrainConfig(
        output_dir=str(stage2_dir),
        hlt_cache_dir=args.hlt_cache_dir,
        reconstructor_checkpoint=str(output_dir / "stage_a" / "best_model_val.pt"),
        variant=args.variant,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed + 1,
        batch_size=args.dual_batch_size,
        epochs=args.dual_epochs,
        lr=args.dual_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.no_amp),
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.dual_early_stop_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        model_size=args.model_size,
        max_constits=args.max_constits,
        reco_weight_threshold=args.reco_weight_threshold,
    )
    dual_report = train_dual_view_tagger(
        dual_config,
        train_view=train_hlt_view,
        val_view=val_hlt_view,
    )
    save_json(stage2_dir / "run_report.json", dual_report)

    reconstructor = build_reconstructor(get_reconstructor_variant_config(args.variant)).to(device)
    checkpoint = torch.load(output_dir / "stage_a" / "best_model_val.pt", map_location=device)
    reconstructor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reconstructor.eval()

    teacher_metrics, test_pair = evaluate_teacher_paths(
        args,
        output_dir=output_dir,
        reconstructor=reconstructor,
        teacher=teacher,
        device=device,
    )
    tagger, frozen_reconstructor, dual_payload, reco_payload = load_dual_view_model_from_checkpoint(
        stage2_dir / "best_model_val.pt",
        device=device,
    )
    dual_block = evaluate_dual_view_model(
        "v2_teacher_logit_fresh_dual_view",
        tagger,
        frozen_reconstructor,
        test_pair.hlt,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
        max_constits=args.max_constits,
    )
    dual_metrics = classification_metrics_from_logits(dual_block.logits, dual_block.labels)
    evaluation_report = {
        "experiment_step": f"{EXPERIMENT_STEP}_evaluation",
        "test_split": args.test_split,
        "test_n_jets": int(len(dual_block.labels)),
        "test_pair": summarize_paired_jet_views(test_pair),
        "teacher_paths": teacher_metrics,
        "fresh_dual_view_tagger": {
            "metrics": dual_metrics,
            "checkpoint": str(stage2_dir / "best_model_val.pt"),
            "checkpoint_epoch": dual_payload.get("epoch"),
            "reconstructor_epoch": reco_payload.get("epoch"),
        },
        "interpretation": {
            "offline_teacher_on_offline": "upper reference using offline constituents",
            "offline_teacher_on_hlt": "same frozen teacher run directly on fixed-HLT",
            "offline_teacher_on_reconstructed": "fixed-HLT -> V2 teacher-logit reconstructor -> frozen teacher",
            "fresh_dual_view_tagger": "fixed-HLT + parent-aligned corrected view -> newly trained dual-view tagger",
        },
    }
    save_json(output_dir / "evaluation_report.json", evaluation_report)

    report = {
        "ok": True,
        "experiment_step": EXPERIMENT_STEP,
        "output_dir": str(output_dir),
        "stage_a": stage_a_report,
        "stage2_dual_view": dual_report,
        "evaluation": evaluation_report,
        "reduced_splits": {
            "train_split": args.train_split,
            "train_size": int(args.train_size),
            "val_split": args.val_split,
            "val_size": int(args.val_size),
            "test_split": args.test_split,
            "test_size": int(args.test_size),
        },
        "source": source_metadata(),
    }
    save_json(output_dir / "run_report.json", report)
    print("v2_teacher_logit_dualview_debug_complete:")
    print(f"  output_dir: {output_dir}")
    print(f"  stage_a_best_val_loss: {stage_a_report['best_model_val_total_loss']:.6f}")
    print(f"  dual_view_val_accuracy: {dual_report['best_model_val_accuracy']:.6f}")
    print("  test_accuracy:")
    for name, row in teacher_metrics.items():
        print(f"    {name}: {row['accuracy']:.6f}")
    print(f"    fresh_dual_view_tagger: {dual_metrics['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
