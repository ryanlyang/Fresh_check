"""Training loop for the teacher-logit PFN-style reconstructor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import LABEL_NAMES, manifest_hash

from .losses import TeacherLogitRecoLossConfig
from .particle_flow_reconstructor import ParticleFlowReconstructor, ParticleFlowReconstructorConfig
from .teachers import assert_teacher_frozen, load_frozen_teacher
from .train_global_transformer import (
    PairedTeacherLogitDataset,
    load_train_val_pairs,
    make_teacher_logit_loader,
    run_teacher_logit_reco_epoch,
    source_metadata,
)
from .views import PairedJetViews, summarize_paired_jet_views


EXPERIMENT_STEP = "teacher_logit_reco_step5_particle_flow_train"
RECONSTRUCTOR_ARCHITECTURE = "particle_flow"


@dataclass
class TeacherLogitParticleFlowTrainConfig:
    """Configuration for Step 5 PFN-style reconstructor training."""

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
    phi_dims: tuple[int, ...] = (128, 128, 128)
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)
    decoder_dims: tuple[int, ...] = (256, 128)
    slot_dim: int | None = None
    num_extra_candidates: int = 32
    dropout: float = 0.05
    max_delta_logpt: float = 1.0
    max_delta_eta: float = 0.35
    max_delta_phi: float = 0.35
    max_delta_loge: float = 1.0
    parent_weight_bias: float = 2.0
    extra_weight_bias: float = -3.0
    max_total_extra_pt_fraction: float = 0.20
    max_extra_delta_eta: float = 1.25
    max_extra_delta_phi: float = 1.25
    teacher_kl_weight: float = 1.0
    ce_weight: float = 0.25
    correction_budget_weight: float = 0.05
    jet_summary_weight: float = 0.05
    temperature: float = 2.0

    def __post_init__(self) -> None:
        self.phi_dims = tuple(int(dim) for dim in self.phi_dims)
        self.context_mlp_dims = tuple(int(dim) for dim in self.context_mlp_dims)
        self.decoder_dims = tuple(int(dim) for dim in self.decoder_dims)
        if self.slot_dim is not None:
            self.slot_dim = int(self.slot_dim)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 5 may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if not self.phi_dims:
            raise ValueError("phi_dims must contain at least one dimension")
        if any(dim <= 0 for dim in self.phi_dims):
            raise ValueError("phi_dims must all be positive")
        if int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if not self.context_mlp_dims:
            raise ValueError("context_mlp_dims must contain at least one dimension")
        if any(dim <= 0 for dim in self.context_mlp_dims):
            raise ValueError("context_mlp_dims must all be positive")
        if not self.decoder_dims:
            raise ValueError("decoder_dims must contain at least one dimension")
        if any(dim <= 0 for dim in self.decoder_dims):
            raise ValueError("decoder_dims must all be positive")
        for name in ("max_train_batches", "max_val_batches", "max_train_jets", "max_val_jets"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative when provided")

    def model_config(self) -> ParticleFlowReconstructorConfig:
        return ParticleFlowReconstructorConfig(
            phi_dims=self.phi_dims,
            context_dim=int(self.context_dim),
            context_mlp_dims=self.context_mlp_dims,
            decoder_dims=self.decoder_dims,
            slot_dim=self.slot_dim,
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


def teacher_logit_particle_flow_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: TeacherLogitParticleFlowTrainConfig,
    model_config: ParticleFlowReconstructorConfig,
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


def train_teacher_logit_particle_flow_reco(
    config: TeacherLogitParticleFlowTrainConfig,
    *,
    model=None,
    teacher=None,
    train_pair: PairedJetViews | None = None,
    val_pair: PairedJetViews | None = None,
) -> Dict[str, Any]:
    """Train the Step 5 teacher-logit PFN-style reconstructor."""

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
    reconstructor = model or ParticleFlowReconstructor(model_config)
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
            "Step 5 trains the PFN-style reconstructor only on model_train and selects only on model_val. "
            "Offline constituents and offline teacher logits are used only as train/validation supervision; "
            "the reconstructor inference path consumes fixed-HLT tokens only."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[Dict[str, Any]] = []
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
        payload = teacher_logit_particle_flow_checkpoint_payload(
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
            "Teacher-logit PFN reconstructor did not produce a finite model_val total_loss, "
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
