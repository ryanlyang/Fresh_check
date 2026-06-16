"""Training loop for aggressive teacher-logit reconstructors.

Step 8 intentionally uses one shared trainer for all aggressive encoder
families.  The encoder changes, but the paired fixed-HLT/offline data, frozen
teacher path, aggressive soft-view contract, losses, and checkpoint format stay
the same.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import LABEL_NAMES, manifest_hash

from .aggressive_global_transformer import (
    AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    AggressiveGlobalTransformerReconstructorConfig,
)
from .aggressive_particle_reconstructors import (
    AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
    AggressiveParticleCnnReconstructorConfig,
    AggressiveParticleFlowReconstructorConfig,
    AggressiveParticleNetReconstructorConfig,
)
from .aggressive_soft_view import AGGRESSION_LEVEL
from .losses import TeacherLogitRecoLossConfig
from .particle_cnn_reconstructor import PARTICLE_CNN_ORDERING_ASSUMPTION
from .reconstructor_builders import build_teacher_logit_reconstructor, normalize_reconstructor_architecture
from .teachers import assert_teacher_frozen, load_frozen_teacher
from .train_global_transformer import (
    PairedTeacherLogitDataset,
    load_train_val_pairs,
    make_teacher_logit_loader,
    run_teacher_logit_reco_epoch,
    source_metadata,
)
from .views import PairedJetViews, summarize_paired_jet_views


EXPERIMENT_STEP = "teacher_logit_reco_step8_aggressive_train"
AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES = (
    AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR,
    AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR,
)


def _positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        output = (int(value),)
    else:
        output = tuple(int(item) for item in value)
    if not output:
        raise ValueError(f"{field_name} must contain at least one value")
    if any(item <= 0 for item in output):
        raise ValueError(f"{field_name} must contain only positive values")
    return output


@dataclass
class TeacherLogitAggressiveReconstructorTrainConfig:
    """Configuration for aggressive teacher-logit reconstructor training."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    teacher_checkpoint: str
    reco_architecture: str = "aggressive_global_transformer"
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

    # Global Transformer encoder.
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4

    # ParticleNet encoder.
    edgeconv_dims: tuple[int, ...] = (64, 128, 128)
    k: int = 16

    # PFN encoder.
    phi_dims: tuple[int, ...] = (128, 128, 128)

    # PFN/P-CNN shared context encoder settings.
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)

    # P-CNN encoder.
    hidden_channels: int = 128
    num_blocks: int = 6
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)

    # Shared aggressive soft-view head.
    embedding_dim: int = 128
    dropout: float = 0.05
    num_extra_candidates: int = 64
    max_delta_logpt: float = 1.0
    max_delta_eta: float = 0.20
    max_delta_phi: float = 0.20
    max_delta_loge: float = 1.0
    parent_weight_bias: float = 2.0
    extra_weight_bias: float = -2.0
    max_total_extra_pt_fraction: float = 0.50
    max_extra_delta_eta: float = 1.50
    max_extra_delta_phi: float = 1.50
    max_global_logpt_scale: float = 0.35
    max_global_loge_scale: float = 0.35
    max_global_eta_shift: float = 0.05
    max_global_phi_shift: float = 0.05
    extra_usage_weight_threshold: float = 0.05
    eta_limit: float = 5.0
    min_pt: float = 1.0e-4

    # Shared aggressive loss.
    teacher_kl_weight: float = 1.0
    ce_weight: float = 0.5
    correction_budget_weight: float = 0.02
    jet_summary_weight: float = 0.05
    temperature: float = 2.0
    aggressive_extra_budget_weight: float = 0.05
    aggressive_parent_weight_budget_weight: float = 0.02
    aggressive_global_calibration_budget_weight: float = 0.02
    extra_count_budget_weight: float = 0.05
    min_parent_weight_fraction: float = 0.25
    parent_prune_budget_weight: float = 0.05

    def __post_init__(self) -> None:
        self.edgeconv_dims = _positive_int_tuple(self.edgeconv_dims, field_name="edgeconv_dims")
        self.phi_dims = _positive_int_tuple(self.phi_dims, field_name="phi_dims")
        self.context_mlp_dims = _positive_int_tuple(self.context_mlp_dims, field_name="context_mlp_dims")
        self.kernel_sizes = _positive_int_tuple(self.kernel_sizes, field_name="kernel_sizes")
        self.dilations = _positive_int_tuple(self.dilations, field_name="dilations")

        if self.normalized_reco_architecture() not in AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES:
            expected = ", ".join(AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES)
            raise ValueError(
                f"Step 8 aggressive trainer requires an aggressive architecture; "
                f"got {self.reco_architecture!r}, expected one of: {expected}"
            )
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 8 may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if int(self.k) <= 0:
            raise ValueError("k must be positive")
        if int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if int(self.hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if int(self.num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if len(self.kernel_sizes) != int(self.num_blocks):
            raise ValueError("kernel_sizes length must match num_blocks")
        if len(self.dilations) != int(self.num_blocks):
            raise ValueError("dilations length must match num_blocks")
        if any(kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain odd values")
        if int(self.embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")
        if int(self.num_extra_candidates) < 0:
            raise ValueError("num_extra_candidates must be non-negative")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("max_train_batches", "max_val_batches", "max_train_jets", "max_val_jets"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative when provided")

    def normalized_reco_architecture(self) -> str:
        return normalize_reconstructor_architecture(self.reco_architecture)

    def aggressive_head_config(self) -> Dict[str, Any]:
        return {
            "num_extra_candidates": int(self.num_extra_candidates),
            "dropout": float(self.dropout),
            "max_delta_logpt": float(self.max_delta_logpt),
            "max_delta_eta": float(self.max_delta_eta),
            "max_delta_phi": float(self.max_delta_phi),
            "max_delta_loge": float(self.max_delta_loge),
            "parent_weight_bias": float(self.parent_weight_bias),
            "extra_weight_bias": float(self.extra_weight_bias),
            "max_total_extra_pt_fraction": float(self.max_total_extra_pt_fraction),
            "max_extra_delta_eta": float(self.max_extra_delta_eta),
            "max_extra_delta_phi": float(self.max_extra_delta_phi),
            "max_global_logpt_scale": float(self.max_global_logpt_scale),
            "max_global_loge_scale": float(self.max_global_loge_scale),
            "max_global_eta_shift": float(self.max_global_eta_shift),
            "max_global_phi_shift": float(self.max_global_phi_shift),
            "extra_usage_weight_threshold": float(self.extra_usage_weight_threshold),
            "eta_limit": float(self.eta_limit),
            "min_pt": float(self.min_pt),
            "aggression_level": AGGRESSION_LEVEL,
        }

    def model_config(self):
        arch = self.normalized_reco_architecture()
        head_config = self.aggressive_head_config()
        if arch == AGGRESSIVE_GLOBAL_TRANSFORMER_RECONSTRUCTOR:
            return AggressiveGlobalTransformerReconstructorConfig(
                hidden_dim=int(self.hidden_dim),
                num_layers=int(self.num_layers),
                num_heads=int(self.num_heads),
                dropout=float(self.dropout),
                aggressive_head_config=head_config,
            )
        if arch == AGGRESSIVE_PARTICLE_NET_RECONSTRUCTOR:
            return AggressiveParticleNetReconstructorConfig(
                edgeconv_dims=self.edgeconv_dims,
                k=int(self.k),
                dropout=float(self.dropout),
                embedding_dim=int(self.embedding_dim),
                aggressive_head_config=head_config,
            )
        if arch == AGGRESSIVE_PARTICLE_FLOW_RECONSTRUCTOR:
            return AggressiveParticleFlowReconstructorConfig(
                phi_dims=self.phi_dims,
                context_dim=int(self.context_dim),
                context_mlp_dims=self.context_mlp_dims,
                dropout=float(self.dropout),
                embedding_dim=int(self.embedding_dim),
                aggressive_head_config=head_config,
            )
        if arch == AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR:
            return AggressiveParticleCnnReconstructorConfig(
                hidden_channels=int(self.hidden_channels),
                num_blocks=int(self.num_blocks),
                kernel_sizes=self.kernel_sizes,
                dilations=self.dilations,
                context_dim=int(self.context_dim),
                context_mlp_dims=self.context_mlp_dims,
                dropout=float(self.dropout),
                embedding_dim=int(self.embedding_dim),
                aggressive_head_config=head_config,
            )
        raise AssertionError(f"Unhandled aggressive reconstructor architecture: {arch}")

    def loss_config(self) -> TeacherLogitRecoLossConfig:
        return TeacherLogitRecoLossConfig.aggressive_defaults(
            teacher_kl_weight=float(self.teacher_kl_weight),
            ce_weight=float(self.ce_weight),
            correction_budget_weight=float(self.correction_budget_weight),
            jet_summary_weight=float(self.jet_summary_weight),
            temperature=float(self.temperature),
            aggressive_extra_budget_weight=float(self.aggressive_extra_budget_weight),
            aggressive_parent_weight_budget_weight=float(self.aggressive_parent_weight_budget_weight),
            aggressive_global_calibration_budget_weight=float(self.aggressive_global_calibration_budget_weight),
            max_total_extra_pt_fraction=float(self.max_total_extra_pt_fraction),
            extra_count_budget_weight=float(self.extra_count_budget_weight),
            min_parent_weight_fraction=float(self.min_parent_weight_fraction),
            parent_prune_budget_weight=float(self.parent_prune_budget_weight),
        )


def teacher_logit_aggressive_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    architecture: str,
    config: TeacherLogitAggressiveReconstructorTrainConfig,
    model_config,
    loss_config: TeacherLogitRecoLossConfig,
    teacher_metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
):
    payload = {
        "epoch": int(epoch),
        "reconstructor_architecture": str(architecture),
        "aggression_level": AGGRESSION_LEVEL,
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
    if architecture == AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR:
        payload["ordering_assumption"] = PARTICLE_CNN_ORDERING_ASSUMPTION
    return payload


def train_teacher_logit_aggressive_reco(
    config: TeacherLogitAggressiveReconstructorTrainConfig,
    *,
    model=None,
    teacher=None,
    train_pair: PairedJetViews | None = None,
    val_pair: PairedJetViews | None = None,
) -> Dict[str, Any]:
    """Train an aggressive teacher-logit reconstructor on model_train/model_val."""

    if config.train_split != "model_train" or config.val_split != "model_val":
        raise ValueError("Step 8 may train only on model_train and select only on model_val")

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

    architecture = config.normalized_reco_architecture()
    model_config = config.model_config()
    loss_config = config.loss_config()
    reconstructor = model or build_teacher_logit_reconstructor(architecture, model_config)
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
        "reconstructor_architecture": architecture,
        "aggression_level": AGGRESSION_LEVEL,
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
            "Step 8 trains aggressive reconstructors only on model_train and selects only on model_val. "
            "Offline constituents and offline teacher logits are used only as train/validation supervision; "
            "the reconstructor inference path consumes fixed-HLT tokens only."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    if architecture == AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR:
        run_metadata["ordering_assumption"] = PARTICLE_CNN_ORDERING_ASSUMPTION
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
        payload = teacher_logit_aggressive_checkpoint_payload(
            reconstructor,
            optimizer,
            epoch=epoch,
            architecture=architecture,
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
            f"Teacher-logit aggressive reconstructor {architecture} did not produce a finite model_val total_loss, "
            "so no best_model_val.pt was written"
        )

    report = {
        "experiment_step": EXPERIMENT_STEP,
        "reconstructor_architecture": architecture,
        "aggression_level": AGGRESSION_LEVEL,
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
    if architecture == AGGRESSIVE_PARTICLE_CNN_RECONSTRUCTOR:
        report["ordering_assumption"] = PARTICLE_CNN_ORDERING_ASSUMPTION
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES",
    "EXPERIMENT_STEP",
    "TeacherLogitAggressiveReconstructorTrainConfig",
    "teacher_logit_aggressive_checkpoint_payload",
    "train_teacher_logit_aggressive_reco",
]
