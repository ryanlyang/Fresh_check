"""Deployable HLT + HLT2 particle dual-view fusion model."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import default_part_config, require_torch
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
)

from .config import (
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_EXPERIMENT_NAME,
)

try:  # Keep imports lightweight on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


HLT_SDV_STEP4_EXPERIMENT_STEP = "hlt_sdv_step4_dual_hlt_fusion_model"
HLT_SDV_MODEL_CONTRACT = "hlt_self_dualview_fusion_model_v1"
HLT_SDV_MODEL_ARCHITECTURE = "deployable_hlt_hlt2_part_concat_abs_product"
HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM = 512
HLT_SDV_DEFAULT_DROPOUT = 0.05
HLT_SDV_DEFAULT_MODEL_SIZE = "base"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hlt_sdv_embedding_branch_config(
    *,
    model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a classifier-free Particle Transformer branch config."""

    cfg = default_part_config(num_classes=PD10_NUM_CLASSES, model_size=model_size)
    if overrides:
        cfg.update(dict(overrides))
    cfg["num_classes"] = None
    cfg["fc_params"] = None
    return cfg


def hlt_sdv_branch_dim_from_config(branch_config: Mapping[str, Any]) -> int:
    embed_dims = list(branch_config.get("embed_dims") or [])
    if not embed_dims:
        raise ValueError("branch_config must include non-empty embed_dims")
    return int(embed_dims[-1])


class HLTSDVParticleTransformerEmbeddingBranch(_ModuleBase):
    """Classifier-free Particle Transformer branch returning jet embeddings."""

    def __init__(self, **kwargs) -> None:
        torch = require_torch()
        super().__init__()
        try:
            from weaver.nn.model.ParticleTransformer import ParticleTransformer
        except ImportError as exc:  # pragma: no cover - depends on research env
            raise ImportError(
                "HLT self-dualview fusion requires weaver-core on the research compute."
            ) from exc

        self.config = dict(kwargs)
        self.mod = ParticleTransformer(**kwargs)
        self.branch_dim = hlt_sdv_branch_dim_from_config(self.config)

    def forward(self, inputs: Mapping[str, Any]):
        _ = inputs.get("points")
        return self.mod(inputs["features"], v=inputs["lorentz_vectors"], mask=inputs["mask"])


def _infer_branch_dim(branch: Any) -> int | None:
    for attr in ("branch_dim", "embedding_dim", "embed_dim", "output_dim"):
        value = getattr(branch, attr, None)
        if value is not None:
            return int(value)
    config = getattr(branch, "config", None)
    if isinstance(config, Mapping):
        for key in ("branch_dim", "embedding_dim", "embed_dim", "output_dim"):
            if config.get(key) is not None:
                return int(config[key])
        embed_dims = config.get("embed_dims")
        if embed_dims:
            return int(list(embed_dims)[-1])
    return None


class HLTSelfDualViewFusionModel(_ModuleBase):
    """Two-branch deployable HLT/HLT2 Particle Transformer fusion model."""

    def __init__(
        self,
        *,
        num_classes: int = PD10_NUM_CLASSES,
        model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE,
        branch_config: Mapping[str, Any] | None = None,
        hlt_branch: Any | None = None,
        hlt2_branch: Any | None = None,
        branch_dim: int | None = None,
        fusion_hidden_dim: int = HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM,
        representation_dim: int = PD10_REPRESENTATION_DIM,
        dropout: float = HLT_SDV_DEFAULT_DROPOUT,
    ) -> None:
        torch = require_torch()
        nn = torch.nn
        super().__init__()
        if int(num_classes) != PD10_NUM_CLASSES:
            raise ValueError(f"HLT-SDV is a 10-class model; got num_classes={num_classes}")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(fusion_hidden_dim) <= 0 or int(representation_dim) <= 0:
            raise ValueError("fusion_hidden_dim and representation_dim must be positive")

        cfg = hlt_sdv_embedding_branch_config(model_size=model_size, overrides=branch_config)
        if hlt_branch is None:
            hlt_branch = HLTSDVParticleTransformerEmbeddingBranch(**cfg)
        if hlt2_branch is None:
            hlt2_branch = HLTSDVParticleTransformerEmbeddingBranch(**cfg)
        inferred_dim = (
            int(branch_dim)
            if branch_dim is not None
            else (_infer_branch_dim(hlt_branch) or _infer_branch_dim(hlt2_branch))
        )
        if inferred_dim is None:
            inferred_dim = hlt_sdv_branch_dim_from_config(cfg)
        if int(inferred_dim) <= 0:
            raise ValueError("branch_dim must be positive")

        self.hlt_branch = hlt_branch
        self.hlt2_branch = hlt2_branch
        self.branch_dim = int(inferred_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.branch_dim * 4),
            nn.Linear(self.branch_dim * 4, int(fusion_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fusion_hidden_dim), int(representation_dim)),
            nn.GELU(),
            nn.LayerNorm(int(representation_dim)),
        )
        self.classifier = nn.Linear(int(representation_dim), int(num_classes))
        self.config = {
            "contract": HLT_SDV_MODEL_CONTRACT,
            "experiment_name": HLT_SDV_EXPERIMENT_NAME,
            "experiment_step": HLT_SDV_STEP4_EXPERIMENT_STEP,
            "architecture": HLT_SDV_MODEL_ARCHITECTURE,
            "num_classes": int(num_classes),
            "model_size": str(model_size),
            "branch_config": dict(cfg),
            "branch_dim": int(self.branch_dim),
            "fusion_input_dim": int(self.branch_dim * 4),
            "fusion_hidden_dim": int(fusion_hidden_dim),
            "representation_dim": int(representation_dim),
            "dropout": float(dropout),
            "hlt_feature_names": list(PF_FEATURE_NAMES),
            "hlt2_feature_names": list(PF_FEATURE_NAMES),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
            "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
            "requires_deterministic_hlt2_transform": True,
            "returns_offline_particles": False,
            "inference_export_requires_teacher_features": False,
        }

    def no_weight_decay(self) -> set[str]:
        return {"hlt_branch.mod.cls_token", "hlt2_branch.mod.cls_token"}

    def branch_parameters(self):
        yield from self.hlt_branch.parameters()
        yield from self.hlt2_branch.parameters()

    def head_parameters(self):
        yield from self.fusion.parameters()
        yield from self.classifier.parameters()

    def set_branches_trainable(self, trainable: bool) -> None:
        for parameter in self.branch_parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(
        self,
        hlt_inputs: Mapping[str, Any],
        hlt2_inputs: Mapping[str, Any],
        *,
        return_representation: bool = False,
    ):
        torch = require_torch()
        hlt_embedding = self.hlt_branch(hlt_inputs)
        hlt2_embedding = self.hlt2_branch(hlt2_inputs)
        if int(hlt_embedding.shape[-1]) != self.branch_dim or int(hlt2_embedding.shape[-1]) != self.branch_dim:
            raise ValueError(
                f"branch embeddings must both have dim {self.branch_dim}; "
                f"got {tuple(hlt_embedding.shape)} and {tuple(hlt2_embedding.shape)}"
            )
        fusion_input = torch.cat(
            [
                hlt_embedding,
                hlt2_embedding,
                torch.abs(hlt2_embedding - hlt_embedding),
                hlt_embedding * hlt2_embedding,
            ],
            dim=1,
        )
        representation = self.fusion(fusion_input)
        logits = self.classifier(representation)
        if return_representation:
            return logits, torch.nn.functional.normalize(representation, dim=1)
        return logits

    def forward_batch(self, batch: Mapping[str, Any], *, return_representation: bool = False):
        return self(
            batch["hlt_inputs"],
            batch["hlt2_inputs"],
            return_representation=return_representation,
        )


def build_hlt_sdv_fusion_model(
    *,
    num_classes: int = PD10_NUM_CLASSES,
    model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE,
    branch_config: Mapping[str, Any] | None = None,
    fusion_hidden_dim: int = HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM,
    representation_dim: int = PD10_REPRESENTATION_DIM,
    dropout: float = HLT_SDV_DEFAULT_DROPOUT,
) -> HLTSelfDualViewFusionModel:
    return HLTSelfDualViewFusionModel(
        num_classes=num_classes,
        model_size=model_size,
        branch_config=branch_config,
        fusion_hidden_dim=fusion_hidden_dim,
        representation_dim=representation_dim,
        dropout=dropout,
    )


def strip_compile_prefix_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove common wrapping prefixes from checkpoint state dicts."""

    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key.removeprefix(prefix)
                    changed = True
        stripped[new_key] = value
    return stripped


def extract_model_state_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if state is None and isinstance(payload, Mapping):
        state = payload.get("state_dict")
    if state is None:
        raise ValueError("checkpoint payload does not contain model_state_dict or state_dict")
    return strip_compile_prefix_from_state_dict(dict(state))


def load_matching_branch_weights(
    branch,
    checkpoint: str | Path,
    *,
    device,
    branch_label: str,
    min_match_fraction: float = 0.50,
) -> dict[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(checkpoint, map_location=device)
    source_state = extract_model_state_dict(payload)
    target_state = branch.state_dict()
    matched: dict[str, Any] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append(key)
            continue
        matched[key] = value
    min_matched = max(1, int(float(min_match_fraction) * float(len(target_state))))
    if len(matched) < min_matched:
        raise RuntimeError(
            f"{branch_label} initialization from {checkpoint} matched only {len(matched)} tensors; "
            f"expected at least {min_matched} of {len(target_state)} target tensors"
        )
    branch.load_state_dict(matched, strict=False)
    return {
        "branch": str(branch_label),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_experiment_step": payload.get("experiment_step"),
        "matched_tensors": int(len(matched)),
        "target_tensors": int(len(target_state)),
        "min_required_matched_tensors": int(min_matched),
        "skipped_missing_count": int(len(skipped_missing)),
        "skipped_shape_count": int(len(skipped_shape)),
        "skipped_shape_keys": skipped_shape[:20],
    }


def initialize_hlt_sdv_branches_from_checkpoints(
    model: HLTSelfDualViewFusionModel,
    *,
    hlt_checkpoint: str | Path,
    hlt2_checkpoint: str | Path | None = None,
    device,
    min_match_fraction: float = 0.50,
) -> dict[str, Any]:
    """Initialize deployable branches from HLT and optional HLT2-specialist checkpoints."""

    branch2_checkpoint = hlt_checkpoint if hlt2_checkpoint is None else hlt2_checkpoint
    return {
        "hlt_branch": load_matching_branch_weights(
            model.hlt_branch,
            hlt_checkpoint,
            device=device,
            branch_label="hlt_branch",
            min_match_fraction=min_match_fraction,
        ),
        "hlt2_branch": load_matching_branch_weights(
            model.hlt2_branch,
            branch2_checkpoint,
            device=device,
            branch_label="hlt2_branch",
            min_match_fraction=min_match_fraction,
        ),
        "source": "same_hlt_part_teacher_checkpoint"
        if hlt2_checkpoint is None
        else "hlt_part_teacher_plus_hlt2_only_checkpoint",
        "both_branches_initialized_from_same_checkpoint": hlt2_checkpoint is None,
        "hlt_checkpoint": str(hlt_checkpoint),
        "hlt2_checkpoint": str(branch2_checkpoint),
    }


def initialize_hlt_sdv_branches_from_hlt_checkpoint(
    model: HLTSelfDualViewFusionModel,
    *,
    hlt_checkpoint: str | Path,
    device,
    min_match_fraction: float = 0.50,
) -> dict[str, Any]:
    """Initialize both deployable branches from the same HLT ParT teacher."""

    return initialize_hlt_sdv_branches_from_checkpoints(
        model,
        hlt_checkpoint=hlt_checkpoint,
        hlt2_checkpoint=None,
        device=device,
        min_match_fraction=min_match_fraction,
    )


def forward_hlt_sdv_batch(
    model: HLTSelfDualViewFusionModel,
    batch: Mapping[str, Any],
    *,
    return_representation: bool = False,
):
    return model.forward_batch(batch, return_representation=return_representation)


__all__ = [
    "HLTSDVParticleTransformerEmbeddingBranch",
    "HLT_SDV_DEFAULT_DROPOUT",
    "HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM",
    "HLT_SDV_DEFAULT_MODEL_SIZE",
    "HLT_SDV_MODEL_ARCHITECTURE",
    "HLT_SDV_MODEL_CONTRACT",
    "HLT_SDV_STEP4_EXPERIMENT_STEP",
    "HLTSelfDualViewFusionModel",
    "build_hlt_sdv_fusion_model",
    "extract_model_state_dict",
    "forward_hlt_sdv_batch",
    "hlt_sdv_branch_dim_from_config",
    "hlt_sdv_embedding_branch_config",
    "initialize_hlt_sdv_branches_from_checkpoints",
    "initialize_hlt_sdv_branches_from_hlt_checkpoint",
    "load_matching_branch_weights",
    "sha256_file",
    "strip_compile_prefix_from_state_dict",
]
