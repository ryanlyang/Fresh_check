"""Training loop for the teacher-logit Global Transformer reconstructor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import LABEL_NAMES, manifest_hash

from .global_transformer import GlobalTransformerReconstructor, GlobalTransformerReconstructorConfig
from .losses import TeacherLogitRecoLossConfig, compute_teacher_logit_reco_loss
from .teachers import assert_teacher_frozen, load_frozen_teacher
from .views import PairedJetViews, load_paired_jet_views, summarize_paired_jet_views


EXPERIMENT_STEP = "teacher_logit_reco_step5_global_transformer_train"
RECONSTRUCTOR_ARCHITECTURE = "global_transformer"


@dataclass
class TeacherLogitGlobalTransformerTrainConfig:
    """Configuration for Step 5 Global Transformer reconstructor training."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    teacher_checkpoint: str
    data_dir: str | None = None
    teacher_architecture: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 1205
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    verify_hlt_hash: bool = True
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    compile_model: bool = False
    max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_extra_candidates: int = 32
    dropout: float = 0.05
    max_delta_logpt: float = 0.50
    max_delta_eta: float = 0.25
    max_delta_phi: float = 0.25
    max_delta_loge: float = 0.50
    parent_weight_bias: float = 4.0
    extra_weight_bias: float = -3.0
    max_total_extra_pt_fraction: float = 0.20
    max_extra_delta_eta: float = 1.25
    max_extra_delta_phi: float = 1.25
    teacher_kl_weight: float = 1.0
    ce_weight: float = 0.3
    correction_budget_weight: float = 0.01
    jet_summary_weight: float = 0.05
    temperature: float = 2.0

    def __post_init__(self) -> None:
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 5 may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        for name in ("max_train_batches", "max_val_batches", "max_train_jets", "max_val_jets"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative when provided")

    def model_config(self) -> GlobalTransformerReconstructorConfig:
        return GlobalTransformerReconstructorConfig(
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            num_heads=int(self.num_heads),
            num_extra_candidates=int(self.num_extra_candidates),
            dropout=float(self.dropout),
            max_delta_logpt=float(self.max_delta_logpt),
            max_delta_eta=float(self.max_delta_eta),
            max_delta_phi=float(self.max_delta_phi),
            max_delta_loge=float(self.max_delta_loge),
            parent_weight_bias=float(self.parent_weight_bias),
            extra_weight_bias=float(self.extra_weight_bias),
            max_total_extra_pt_fraction=float(self.max_total_extra_pt_fraction),
            max_extra_delta_eta=float(self.max_extra_delta_eta),
            max_extra_delta_phi=float(self.max_extra_delta_phi),
        )

    def loss_config(self) -> TeacherLogitRecoLossConfig:
        return TeacherLogitRecoLossConfig(
            teacher_kl_weight=float(self.teacher_kl_weight),
            ce_weight=float(self.ce_weight),
            correction_budget_weight=float(self.correction_budget_weight),
            jet_summary_weight=float(self.jet_summary_weight),
            temperature=float(self.temperature),
        )


def source_metadata(repo_root: str | Path | None = None) -> Dict[str, Any]:
    """Record source commit/status if git is available."""

    repo = Path(repo_root or Path(__file__).resolve().parents[1])
    metadata = {"source_commit": "unknown", "source_status_hash": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["source_commit"] = commit
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        metadata["source_status_hash"] = hashlib.sha256(status.encode("utf-8")).hexdigest()
    except Exception:
        pass
    return metadata


class PairedTeacherLogitDataset:
    """Torch dataset over aligned fixed-HLT and offline views."""

    def __init__(self, pair: PairedJetViews, *, max_jets: int | None = None) -> None:
        require_torch()
        pair = pair.slice(max_jets)
        self.hlt_tokens = np.asarray(pair.hlt.tokens, dtype=np.float32)
        self.hlt_mask = np.asarray(pair.hlt.mask, dtype=bool)
        self.offline_tokens = np.asarray(pair.offline.tokens, dtype=np.float32)
        self.offline_mask = np.asarray(pair.offline.mask, dtype=bool)
        self.labels = np.asarray(pair.labels, dtype=np.int64)
        self.jet_ids = list(pair.jet_ids)
        self.split = pair.split
        self.metadata = dict(pair.metadata)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return (
            self.hlt_tokens[index],
            self.hlt_mask[index],
            self.offline_tokens[index],
            self.offline_mask[index],
            self.labels[index],
            self.jet_ids[index],
        )


def collate_paired_teacher_logit(samples):
    torch = require_torch()
    return {
        "hlt_tokens": torch.from_numpy(np.stack([row[0] for row in samples], axis=0)).float(),
        "hlt_mask": torch.from_numpy(np.stack([row[1] for row in samples], axis=0)).bool(),
        "offline_tokens": torch.from_numpy(np.stack([row[2] for row in samples], axis=0)).float(),
        "offline_mask": torch.from_numpy(np.stack([row[3] for row in samples], axis=0)).bool(),
        "labels": torch.as_tensor([row[4] for row in samples], dtype=torch.long),
        "jet_ids": [row[5] for row in samples],
    }


def make_teacher_logit_loader(
    dataset: PairedTeacherLogitDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_paired_teacher_logit,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def load_train_val_pairs(config: TeacherLogitGlobalTransformerTrainConfig) -> tuple[PairedJetViews, PairedJetViews]:
    train_pair = load_paired_jet_views(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=config.train_split,
        data_dir=config.data_dir,
        max_jets=config.max_train_jets,
        verify_hlt_hash=config.verify_hlt_hash,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
    )
    val_pair = load_paired_jet_views(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=config.val_split,
        data_dir=config.data_dir,
        max_jets=config.max_val_jets,
        verify_hlt_hash=config.verify_hlt_hash,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
    )
    return train_pair, val_pair


def _to_device_batch(batch: Mapping[str, Any], device) -> Dict[str, Any]:
    return {
        "hlt_tokens": batch["hlt_tokens"].to(device=device, non_blocking=True),
        "hlt_mask": batch["hlt_mask"].to(device=device, non_blocking=True),
        "offline_tokens": batch["offline_tokens"].to(device=device, non_blocking=True),
        "offline_mask": batch["offline_mask"].to(device=device, non_blocking=True),
        "labels": batch["labels"].to(device=device, non_blocking=True),
        "jet_ids": batch["jet_ids"],
    }


def _accumulate(metrics: Dict[str, float], row: Mapping[str, float], *, weight: int) -> None:
    for key, value in row.items():
        if np.isfinite(float(value)):
            metrics[key] = metrics.get(key, 0.0) + float(value) * int(weight)


def run_teacher_logit_reco_epoch(
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
) -> Dict[str, Any]:
    """Run one train or validation epoch."""

    torch = require_torch()
    is_train = optimizer is not None
    reconstructor.train(bool(is_train))
    teacher.model.eval()
    assert_teacher_frozen(teacher)
    autocast_enabled = bool(amp and getattr(device, "type", None) == "cuda")
    metric_sums: Dict[str, float] = {}
    total_seen = 0
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _to_device_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                reco_view = reconstructor(
                    batch["hlt_tokens"],
                    batch["hlt_mask"],
                    labels=batch["labels"],
                    jet_ids=batch["jet_ids"],
                    split=loader.dataset.split,
                )
                with torch.no_grad():
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
                raise FloatingPointError(f"Non-finite total loss on batch {batch_index}")

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

            batch_size = int(batch["labels"].numel())
            total_seen += batch_size
            n_batches += 1
            _accumulate(metric_sums, loss.detached_float_dict(), weight=batch_size)

    if total_seen == 0:
        return {"n_jets": 0, "n_batches": int(n_batches), "total_loss": float("nan")}
    output = {key: value / float(total_seen) for key, value in sorted(metric_sums.items())}
    output["n_jets"] = int(total_seen)
    output["n_batches"] = int(n_batches)
    return output


def teacher_logit_reco_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: TeacherLogitGlobalTransformerTrainConfig,
    model_config: GlobalTransformerReconstructorConfig,
    loss_config: TeacherLogitRecoLossConfig,
    teacher_metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
):
    return {
        "epoch": int(epoch),
        "reconstructor_architecture": RECONSTRUCTOR_ARCHITECTURE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "teacher_metadata": dict(teacher_metadata),
        "metrics": dict(metrics),
        "label_names": list(LABEL_NAMES),
        "experiment_step": EXPERIMENT_STEP,
        "source": dict(source),
    }


def train_teacher_logit_global_transformer_reco(
    config: TeacherLogitGlobalTransformerTrainConfig,
    *,
    model=None,
    teacher=None,
    train_pair: PairedJetViews | None = None,
    val_pair: PairedJetViews | None = None,
) -> Dict[str, Any]:
    """Train the Step 5 teacher-logit Global Transformer reconstructor."""

    # The dataclass normally enforces this, but keep the check here for callers
    # that may construct config-like objects in tests.
    if config.train_split != "model_train" or config.val_split != "model_val":
        raise ValueError("Step 5 may train only on model_train and select only on model_val")

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_pair is None or val_pair is None:
        train_pair, val_pair = load_train_val_pairs(config)

    train_dataset = PairedTeacherLogitDataset(train_pair, max_jets=config.max_train_jets)
    val_dataset = PairedTeacherLogitDataset(val_pair, max_jets=config.max_val_jets)
    train_loader = make_teacher_logit_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_teacher_logit_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    model_config = config.model_config()
    loss_config = config.loss_config()
    reconstructor = model or GlobalTransformerReconstructor(model_config)
    reconstructor = reconstructor.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        reconstructor = torch.compile(reconstructor)

    teacher = teacher or load_frozen_teacher(
        config.teacher_checkpoint,
        architecture=config.teacher_architecture,
        device=str(device),
        max_constits=int(config.max_constits),
        weight_threshold=float(config.teacher_weight_threshold),
    )
    assert_teacher_frozen(teacher)

    optimizer = torch.optim.AdamW(
        reconstructor.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    manifest_sha = train_pair.metadata.get("source_manifest_hash")
    if manifest_sha is None:
        try:
            from jetclass_fresh.jetclass_data import load_split_manifest

            manifest_sha = manifest_hash(load_split_manifest(config.manifest_path))
        except Exception:
            manifest_sha = None

    run_metadata = {
        "experiment_step": EXPERIMENT_STEP,
        "reconstructor_architecture": RECONSTRUCTOR_ARCHITECTURE,
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "teacher": dict(teacher.metadata),
        "source": source,
        "manifest_hash": manifest_sha,
        "train_pair": summarize_paired_jet_views(train_pair),
        "val_pair": summarize_paired_jet_views(val_pair),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "leakage_rule": (
            "Step 5 trains the reconstructor only on model_train and selects only on model_val. "
            "Offline constituents and offline teacher logits are used only as train/validation supervision; "
            "the reconstructor inference path consumes fixed-HLT tokens only."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_teacher_logit_reco_epoch(
            reconstructor,
            teacher,
            train_loader,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_teacher_logit_reco_epoch(
            reconstructor,
            teacher,
            val_loader,
            device=device,
            loss_config=loss_config,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "model_val": val_metrics,
        }
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        val_loss = float(val_metrics.get("total_loss", float("nan")))
        improved = np.isfinite(val_loss) and val_loss < best_val_loss
        payload = teacher_logit_reco_checkpoint_payload(
            reconstructor,
            optimizer,
            epoch=epoch,
            config=config,
            model_config=model_config,
            loss_config=loss_config,
            teacher_metadata=teacher.metadata,
            metrics=row,
            source=source,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1

        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError(
            "Teacher-logit Global Transformer did not produce a finite model_val total_loss, "
            "so no best_model_val.pt was written"
        )

    report = {
        "experiment_step": EXPERIMENT_STEP,
        "reconstructor_architecture": RECONSTRUCTOR_ARCHITECTURE,
        "best_epoch": int(best_epoch),
        "best_model_val_total_loss": float(best_val_loss),
        "best_model_val_reco_argmax_accuracy": float(
            curves[best_epoch - 1]["model_val"].get("metric_reco_argmax_accuracy", 0.0)
        ),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "teacher": dict(teacher.metadata),
        "source": source,
        "no_final_test_evaluation": True,
        "not_a_classifier": True,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report
