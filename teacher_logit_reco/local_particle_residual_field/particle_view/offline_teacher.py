"""Step-2 matched HLT/offline Particle Transformer teacher contracts.

The canonical HLT and offline teachers are structurally identical.  They
differ only in registered source/preprocessing hashes and the checkpoint
learned from that source.  Contextual taps capture particle-block outputs
after both residual additions and before class pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jetclass_fresh.hlt_baseline import (
    build_particle_transformer_classifier,
    default_part_config,
    require_torch,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES, MAX_CONSTITUENTS
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .splits import (
    PARTICLE_VIEW_TRAINING_TOPOLOGY,
    PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    logical_split_binding,
)

try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

if _torch is None:  # pragma: no cover
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


PARTICLE_VIEW_TEACHER_RECIPE_CONTRACT = "particle_view_teacher_recipe_v1"
PARTICLE_VIEW_TEACHER_REGISTRATION_CONTRACT = (
    "particle_view_teacher_registration_v1"
)
PARTICLE_VIEW_TOKEN_TAP_CONTRACT = "particle_view_contextual_token_tap_v1"
PARTICLE_VIEW_TOKEN_TAP_REGISTRATION_CONTRACT = (
    "particle_view_contextual_token_tap_registration_v1"
)
PARTICLE_VIEW_EXISTING_TEACHER_SOURCE_CONTRACT = (
    "particle_view_existing_teacher_source_v1"
)
PARTICLE_VIEW_DIRECT_CONTROL_GRID_CONTRACT = (
    "particle_view_direct_control_grid_v1"
)
PARTICLE_VIEW_TEACHER_CHECKPOINT_CONTRACT = "particle_view_teacher_checkpoint_v1"

TEACHER_ROLES = ("A0_view", "Toff_view")
TEACHER_ARCHITECTURES = ("base", "large")
TOKEN_TAP_CHOICES = ("raw_embed", "middle", "penultimate", "final", "mix_last3")
CANONICAL_TOKEN_TAP = "penultimate"
CANONICAL_TAP_LOCATION = "post_attention_and_ffn_residual_pre_pool"
TEACHER_SELECTION_ACCURACY_TOLERANCE = 1.0e-4


def _architecture_payload(architecture: str) -> dict[str, Any]:
    if architecture not in TEACHER_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {TEACHER_ARCHITECTURES}")
    config = default_part_config(
        num_classes=len(LABEL_NAMES), model_size=architecture
    )
    return {
        "implementation": "weaver.nn.model.ParticleTransformer",
        "input_dim": config["input_dim"],
        "num_classes": config["num_classes"],
        "pair_input_dim": config["pair_input_dim"],
        "use_pre_activation_pair": config["use_pre_activation_pair"],
        "embed_dims": list(config["embed_dims"]),
        "pair_embed_dims": list(config["pair_embed_dims"]),
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "num_cls_layers": config["num_cls_layers"],
        "block_params": config["block_params"],
        "cls_block_params": dict(config["cls_block_params"]),
        "fc_params": list(config["fc_params"]),
        "activation": config["activation"],
        "trim": config["trim"],
        "for_inference": config["for_inference"],
        "pooling": "class_token",
        "final_normalization": "LayerNorm",
        "classifier": "single_linear_10_class",
        "classifier_dropout": 0.0,
    }


@dataclass(frozen=True)
class ParticleViewTeacherRecipe:
    role: str
    particle_source: str
    architecture: str
    seed: int
    unified_split_manifest_sha256: str
    train_identity_sha256: str
    train_split_sha256: str
    model_val_stop_split_sha256: str
    preprocessing_sha256: str
    source_sha256: str
    initialization_implementation_sha256: str
    library_versions_sha256: str

    def to_payload(self) -> dict[str, Any]:
        if self.role not in TEACHER_ROLES:
            raise ValueError(f"role must be one of {TEACHER_ROLES}")
        source = "fixed_hlt" if self.role == "A0_view" else "offline"
        if self.particle_source != source:
            raise ValueError(f"{self.role} must use {source}")
        if self.role == "A0_view" and self.architecture != "base":
            raise ValueError("A0_view uses the locked base architecture")
        architecture = _architecture_payload(self.architecture)
        if self.seed not in {101, 202, 303}:
            raise ValueError("teacher seed must be 101, 202, or 303")
        hashes = {
            name: require_sha256(name, getattr(self, name))
            for name in (
                "unified_split_manifest_sha256",
                "train_identity_sha256",
                "train_split_sha256",
                "model_val_stop_split_sha256",
                "preprocessing_sha256",
                "source_sha256",
                "initialization_implementation_sha256",
                "library_versions_sha256",
            )
        }
        large = self.architecture == "large"
        return {
            "contract": PARTICLE_VIEW_TEACHER_RECIPE_CONTRACT,
            "role": self.role,
            "particle_source": self.particle_source,
            "architecture_name": self.architecture,
            "architecture": architecture,
            "class_names": list(LABEL_NAMES),
            "feature_names": list(PF_FEATURE_NAMES),
            "max_particles": MAX_CONSTITUENTS,
            "mask_contract": "bool_valid_particle_mask_v1",
            "pair_feature_contract": "weaver_four_vector_pair_features_dim4_v1",
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 3.0e-4,
                "weight_decay": 1.0e-4,
                "betas": [0.9, 0.999],
                "gradient_norm_clip": 1.0,
            },
            "schedule": {
                "name": "linear_warmup_cosine_v1",
                "warmup_updates": 2_000,
                "minimum_learning_rate": 3.0e-6,
                "maximum_epochs": 40,
                "early_stop_patience": 8,
            },
            "checkpoint_selection": {
                "selection_split": "model_val_stop",
                "ranking_split": "model_val_select",
                "accuracy_tolerance": 1.0e-4,
                "order": [
                    "highest_accuracy_within_tolerance",
                    "lowest_cross_entropy",
                    "lowest_ece",
                    "earliest_epoch",
                ],
            },
            "effective_batch_size": 128,
            "physical_batch_size": 64 if large else 128,
            "gradient_accumulation_steps": 2 if large else 1,
            "amp": True,
            "from_scratch": True,
            "pretrained_weights_allowed": False,
            "seed": self.seed,
            "rngs_seeded": ["python", "numpy", "torch_cpu", "torch_cuda_all"],
            "training_topology": PARTICLE_VIEW_TRAINING_TOPOLOGY,
            "train_split": "train",
            **hashes,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def build_teacher_recipe(
    *,
    role: str,
    architecture: str,
    seed: int,
    unified_split_manifest: Mapping[str, Any],
    preprocessing_sha256: str,
    source_sha256: str,
    initialization_implementation_sha256: str,
    library_versions_sha256: str,
) -> ParticleViewTeacherRecipe:
    validate_content_hash(
        unified_split_manifest,
        expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    )
    _, train_split_hash, train_identity_hash = logical_split_binding(
        unified_split_manifest, "train"
    )
    _, stop_hash, _ = logical_split_binding(
        unified_split_manifest, "model_val_stop"
    )
    return ParticleViewTeacherRecipe(
        role=role,
        particle_source="fixed_hlt" if role == "A0_view" else "offline",
        architecture=architecture,
        seed=seed,
        unified_split_manifest_sha256=unified_split_manifest["content_hash"],
        train_identity_sha256=train_identity_hash,
        train_split_sha256=train_split_hash,
        model_val_stop_split_sha256=stop_hash,
        preprocessing_sha256=preprocessing_sha256,
        source_sha256=source_sha256,
        initialization_implementation_sha256=initialization_implementation_sha256,
        library_versions_sha256=library_versions_sha256,
    )


def validate_matched_teacher_recipes(
    hlt_recipe: ParticleViewTeacherRecipe,
    offline_recipe: ParticleViewTeacherRecipe,
) -> str:
    hlt, offline = hlt_recipe.to_payload(), offline_recipe.to_payload()
    if hlt["role"] != "A0_view" or offline["role"] != "Toff_view":
        raise ValueError("matched recipes must be A0_view and Toff_view")
    if offline["architecture_name"] != "base":
        raise ValueError("matched Toff_view must use base architecture")
    allowed = {"role", "particle_source", "preprocessing_sha256", "source_sha256"}
    left = {key: value for key, value in hlt.items() if key not in allowed}
    right = {key: value for key, value in offline.items() if key not in allowed}
    if left != right:
        raise ValueError("matched teacher recipes drifted structurally")
    return canonical_sha256(left)


def build_particle_view_teacher_model(
    recipe: ParticleViewTeacherRecipe | Mapping[str, Any],
):
    payload = (
        recipe.to_payload()
        if isinstance(recipe, ParticleViewTeacherRecipe)
        else dict(recipe)
    )
    architecture = str(payload.get("architecture_name"))
    if payload.get("architecture") != _architecture_payload(architecture):
        raise ValueError("serialized teacher architecture drifted")
    set_training_seed(int(payload["seed"]))
    return build_particle_transformer_classifier(
        num_classes=len(LABEL_NAMES), model_size=architecture
    )


def token_tap_block_indices(num_layers: int, tap_choice: str) -> tuple[int, ...]:
    if num_layers < 3:
        raise ValueError("contextual taps require at least three particle blocks")
    choices = {
        "raw_embed": (),
        "middle": (num_layers // 2 - 1,),
        "penultimate": (num_layers - 2,),
        "final": (num_layers - 1,),
        "mix_last3": tuple(range(num_layers - 3, num_layers)),
    }
    if tap_choice not in choices:
        raise ValueError(f"tap_choice must be one of {TOKEN_TAP_CHOICES}")
    return choices[tap_choice]


@dataclass(frozen=True)
class ParticleTokenTapSpec:
    particle_source: str
    architecture: str
    tap_choice: str = CANONICAL_TOKEN_TAP

    def to_payload(self) -> dict[str, Any]:
        if self.particle_source not in {"fixed_hlt", "offline"}:
            raise ValueError("invalid token-tap particle source")
        architecture = _architecture_payload(self.architecture)
        indices = token_tap_block_indices(
            int(architecture["num_layers"]), self.tap_choice
        )
        return {
            "contract": PARTICLE_VIEW_TOKEN_TAP_CONTRACT,
            "particle_source": self.particle_source,
            "architecture_name": self.architecture,
            "tap_choice": self.tap_choice,
            "module_source": (
                "embedding" if self.tap_choice == "raw_embed" else "particle_blocks"
            ),
            "particle_block_indices": list(indices),
            "tensor_location": (
                "post_embedding_pre_particle_blocks"
                if self.tap_choice == "raw_embed"
                else CANONICAL_TAP_LOCATION
            ),
            "dropout_disabled": True,
            "includes_class_token": False,
            "includes_pooled_embedding": False,
            "includes_logits": False,
            "includes_labels": False,
            "output_layout": "batch_layer_particle_channel",
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def build_token_tap_registration(
    *,
    teacher_registration: Mapping[str, Any],
    tap_spec: ParticleTokenTapSpec,
    input_normalization_sha256: str,
) -> dict[str, Any]:
    """Bind one exact tap to one frozen checkpoint and input normalization."""

    validate_teacher_registration(teacher_registration)
    normalization = require_sha256(
        "input_normalization_sha256", input_normalization_sha256
    )
    payload = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TOKEN_TAP_REGISTRATION_CONTRACT,
            "teacher_registration_sha256": teacher_registration["content_hash"],
            "teacher_checkpoint_sha256": teacher_registration[
                "checkpoint_sha256"
            ],
            "teacher_recipe_sha256": teacher_registration["recipe_sha256"],
            "train_identity_sha256": teacher_registration[
                "train_identity_sha256"
            ],
            "input_normalization_sha256": normalization,
            "tap_spec": tap_spec.to_payload(),
            "tap_spec_sha256": tap_spec.content_hash,
            "teacher_frozen": True,
        }
    )
    validate_token_tap_registration(payload)
    return payload


def validate_token_tap_registration(payload: Mapping[str, Any]) -> str:
    validate_content_hash(
        payload,
        expected_contract=PARTICLE_VIEW_TOKEN_TAP_REGISTRATION_CONTRACT,
    )
    for name in (
        "teacher_registration_sha256",
        "teacher_checkpoint_sha256",
        "teacher_recipe_sha256",
        "train_identity_sha256",
        "input_normalization_sha256",
        "tap_spec_sha256",
    ):
        require_sha256(name, payload.get(name))
    tap = payload.get("tap_spec")
    if not isinstance(tap, Mapping) or canonical_sha256(tap) != payload[
        "tap_spec_sha256"
    ]:
        raise ValueError("token-tap specification hash mismatch")
    if (
        tap.get("tensor_location")
        not in {
            CANONICAL_TAP_LOCATION,
            "post_embedding_pre_particle_blocks",
        }
        or tap.get("dropout_disabled") is not True
        or tap.get("includes_class_token") is not False
        or tap.get("includes_logits") is not False
    ):
        raise ValueError("token tap exposes a forbidden tensor/location")
    if payload.get("teacher_frozen") is not True:
        raise ValueError("token tap requires a frozen teacher")
    return str(payload["content_hash"])


def build_existing_teacher_source_registration(
    *,
    checkpoint_path: str | Path,
    canonical_train_identity_sha256: str,
    observed_train_identity_sha256: str | None,
    serialized_recipe: Mapping[str, Any] | None,
    recipe_reproduced_exactly: bool,
    provenance_metadata_sha256: str,
    description: str,
) -> dict[str, Any]:
    """Classify a pre-existing offline checkpoint without trusting it silently."""

    canonical_train = require_sha256(
        "canonical_train_identity_sha256", canonical_train_identity_sha256
    )
    observed_train = (
        None
        if observed_train_identity_sha256 is None
        else require_sha256(
            "observed_train_identity_sha256",
            observed_train_identity_sha256,
        )
    )
    require_sha256("provenance_metadata_sha256", provenance_metadata_sha256)
    recipe_payload = None if serialized_recipe is None else dict(serialized_recipe)
    exact_recipe = bool(recipe_reproduced_exactly)
    if exact_recipe:
        if (
            recipe_payload is None
            or recipe_payload.get("contract")
            != PARTICLE_VIEW_TEACHER_RECIPE_CONTRACT
            or recipe_payload.get("role") != "Toff_view"
        ):
            raise ValueError(
                "exact existing-teacher evidence requires a serialized Toff recipe"
            )
        architecture = str(recipe_payload.get("architecture_name"))
        if recipe_payload.get("architecture") != _architecture_payload(
            architecture
        ):
            raise ValueError("existing teacher serialized architecture drifted")
    selectable = bool(
        exact_recipe
        and observed_train is not None
        and observed_train == canonical_train
    )
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_EXISTING_TEACHER_SOURCE_CONTRACT,
            "description": str(description),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "canonical_train_identity_sha256": canonical_train,
            "observed_train_identity_sha256": observed_train,
            "provenance_metadata_sha256": provenance_metadata_sha256,
            "serialized_recipe": recipe_payload,
            "serialized_recipe_sha256": (
                None
                if recipe_payload is None
                else canonical_sha256(recipe_payload)
            ),
            "recipe_reproduced_exactly": exact_recipe,
            "selection_status": (
                "selectable_provenance_compatible"
                if selectable
                else "diagnostic_nonselectable"
            ),
            "selectable": selectable,
        }
    )
    return artifact


@dataclass(frozen=True)
class ContextualParticleTokens:
    logits: Any
    particle_tokens: Any
    particle_mask: Any
    tap_spec_sha256: str
    block_indices: tuple[int, ...]

    @property
    def single_layer_tokens(self):
        if self.particle_tokens.shape[1] != 1:
            raise ValueError("tap contains multiple layers; apply a mixture")
        return self.particle_tokens[:, 0]


def _batch_particle_layout(
    value,
    *,
    batch_size: int,
    particles: int,
    particle_mask=None,
):
    """Normalize Weaver token layouts and restore its trimmed padding suffix.

    Weaver trims each batch to the largest active particle count before its
    transformer blocks.  The external particle-view contract remains fixed
    width, so a legitimately trimmed inactive suffix is restored as exact
    zeros.  Trimming any active particle remains a fail-closed error.
    """

    torch = require_torch()
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("token-tap module returned no tensor")
        value = value[0]
    if not hasattr(value, "ndim") or value.ndim != 3:
        raise ValueError("particle token tap must be rank 3")
    if value.shape[0] == batch_size and value.shape[1] <= particles:
        normalized = value
    elif value.shape[1] == batch_size and value.shape[0] <= particles:
        normalized = value.permute(1, 0, 2).contiguous()
    else:
        raise ValueError(
            f"token shape {tuple(value.shape)} disagrees with "
            f"batch={batch_size}, particles={particles}"
        )

    observed_particles = int(normalized.shape[1])
    if observed_particles == particles:
        return normalized
    if particle_mask is None or tuple(particle_mask.shape) != (
        batch_size,
        particles,
    ):
        raise ValueError("trimmed particle tokens require the full particle mask")
    if particle_mask[:, observed_particles:].any():
        raise ValueError("Weaver trimmed one or more active particles")
    padding = normalized.new_zeros(
        (batch_size, particles - observed_particles, normalized.shape[2])
    )
    return torch.cat((normalized, padding), dim=1)


class FrozenContextualParticleTeacher(_ModuleBase):
    """Frozen ParT with a deterministic pre-pooling particle-token tap."""

    def __init__(self, teacher, tap_spec: ParticleTokenTapSpec) -> None:
        require_torch()
        super().__init__()
        if not hasattr(teacher, "mod"):
            raise ValueError("teacher must expose repository ParT as .mod")
        if not hasattr(teacher.mod, "embed") or not hasattr(teacher.mod, "blocks"):
            raise ValueError("teacher ParT does not expose embed/blocks")
        self.teacher = teacher
        self.tap_spec = tap_spec
        self.tap_payload = tap_spec.to_payload()
        freeze_particle_teacher(self.teacher)

    def train(self, mode: bool = True):
        super().train(False)
        self.teacher.eval()
        return self

    def forward(self, points, features, lorentz_vectors, mask):
        torch = require_torch()
        if features.ndim != 3:
            raise ValueError("features must be [batch, channels, particles]")
        if mask.ndim != 3 or mask.shape[1] != 1 or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [batch, 1, particles]")
        batch_size, _, particles = features.shape
        if tuple(mask.shape) != (batch_size, 1, particles):
            raise ValueError("feature/mask shapes disagree")
        captured: dict[int, Any] = {}
        handles = []

        def capture(index):
            def hook(_module, _inputs, output):
                captured[index] = output
            return hook

        indices = tuple(self.tap_payload["particle_block_indices"])
        if self.tap_spec.tap_choice == "raw_embed":
            expected = (-1,)
            handles.append(
                self.teacher.mod.embed.register_forward_hook(capture(-1))
            )
        else:
            expected = indices
            for index in indices:
                handles.append(
                    self.teacher.mod.blocks[index].register_forward_hook(
                        capture(index)
                    )
                )
        try:
            self.teacher.eval()
            with torch.no_grad():
                logits = self.teacher(
                    points, features, lorentz_vectors, mask
                )
        finally:
            for handle in handles:
                handle.remove()
        if tuple(captured) != expected:
            raise RuntimeError(
                f"tap modules did not execute: expected={expected}, "
                f"observed={tuple(captured)}"
            )
        valid = mask[:, 0]
        layers = []
        for index in expected:
            tokens = _batch_particle_layout(
                captured[index],
                batch_size=batch_size,
                particles=particles,
                particle_mask=valid,
            ).detach()
            if not torch.isfinite(tokens).all():
                raise FloatingPointError("teacher particle tokens are non-finite")
            layers.append(tokens.masked_fill(~valid.unsqueeze(-1), 0.0))
        particle_tokens = torch.stack(layers, dim=1)
        if (
            logits.ndim != 2
            or logits.shape[0] != batch_size
            or logits.shape[1] != len(LABEL_NAMES)
        ):
            raise ValueError("teacher logits shape changed")
        return ContextualParticleTokens(
            logits=logits.detach(),
            particle_tokens=particle_tokens,
            particle_mask=valid,
            tap_spec_sha256=self.tap_spec.content_hash,
            block_indices=expected,
        )


class FrozenTokenLayerMixture(_ModuleBase):
    """Learned softmax mixture over detached last-three teacher layers."""

    def __init__(self) -> None:
        torch = require_torch()
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3))

    def forward(self, particle_tokens, particle_mask):
        torch = require_torch()
        if particle_tokens.ndim != 4 or particle_tokens.shape[1] != 3:
            raise ValueError("mixture expects [batch, 3, particles, channels]")
        if tuple(particle_mask.shape) != (
            particle_tokens.shape[0],
            particle_tokens.shape[2],
        ):
            raise ValueError("mixture mask shape mismatch")
        weights = torch.softmax(self.logits, dim=0).view(1, 3, 1, 1)
        mixed = (particle_tokens.detach() * weights).sum(dim=1)
        return mixed.masked_fill(~particle_mask.unsqueeze(-1), 0.0)


def freeze_particle_teacher(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    return model


def audit_frozen_teacher(model) -> dict[str, Any]:
    parameters = list(model.parameters())
    if model.training or any(parameter.requires_grad for parameter in parameters):
        raise ValueError("teacher is not fully frozen in evaluation mode")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("frozen teacher retains gradients")
    return {
        "frozen": True,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": 0,
        "parameters_with_gradients": 0,
    }


def teacher_learning_rate(
    *,
    update_index: int,
    total_updates: int,
    peak_learning_rate: float = 3.0e-4,
    minimum_learning_rate: float = 3.0e-6,
    warmup_updates: int = 2_000,
) -> float:
    if total_updates <= 0 or not 0 <= update_index < total_updates:
        raise ValueError("invalid teacher schedule position")
    warmup = min(warmup_updates, total_updates)
    if update_index < warmup:
        return peak_learning_rate * (update_index + 1) / warmup
    denominator = max(total_updates - warmup - 1, 1)
    progress = min(max((update_index - warmup) / denominator, 0.0), 1.0)
    return minimum_learning_rate + (
        peak_learning_rate - minimum_learning_rate
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))


def select_teacher_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("teacher selection requires candidates")
    rows = []
    for candidate in candidates:
        row = {
            **dict(candidate),
            "epoch": int(candidate["epoch"]),
            "accuracy": float(candidate["accuracy"]),
            "cross_entropy": float(candidate["cross_entropy"]),
            "ece": float(candidate["ece"]),
        }
        if row["epoch"] <= 0 or not all(
            math.isfinite(row[name])
            for name in ("accuracy", "cross_entropy", "ece")
        ):
            raise ValueError("teacher candidate is invalid/non-finite")
        rows.append(row)
    best_accuracy = max(row["accuracy"] for row in rows)
    pool = [
        row
        for row in rows
        if best_accuracy - row["accuracy"]
        <= TEACHER_SELECTION_ACCURACY_TOLERANCE
    ]
    return dict(
        min(
            pool,
            key=lambda row: (
                row["cross_entropy"],
                row["ece"],
                row["epoch"],
            ),
        )
    )


def build_teacher_checkpoint_payload(
    *,
    recipe: ParticleViewTeacherRecipe,
    model,
    epoch: int,
    optimizer_updates: int,
    model_val_stop_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if epoch <= 0 or optimizer_updates <= 0:
        raise ValueError("checkpoint epoch/update counts must be positive")
    metrics = {
        "accuracy": float(model_val_stop_metrics["accuracy"]),
        "cross_entropy": float(model_val_stop_metrics["cross_entropy"]),
        "ece": float(model_val_stop_metrics["ece"]),
    }
    select_teacher_checkpoint([{"epoch": epoch, **metrics}])
    return {
        "contract": PARTICLE_VIEW_TEACHER_CHECKPOINT_CONTRACT,
        "recipe": recipe.to_payload(),
        "recipe_sha256": recipe.content_hash,
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "optimizer_updates": int(optimizer_updates),
        "model_val_stop": metrics,
        "class_names": list(LABEL_NAMES),
    }


def _torch_load(path: str | Path):
    torch = require_torch()
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(Path(path), map_location="cpu")


def validate_teacher_checkpoint(
    checkpoint_path: str | Path,
    *,
    recipe: ParticleViewTeacherRecipe,
) -> Mapping[str, Any]:
    payload = _torch_load(checkpoint_path)
    expected = {
        "contract",
        "recipe",
        "recipe_sha256",
        "model_state_dict",
        "epoch",
        "optimizer_updates",
        "model_val_stop",
        "class_names",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("teacher checkpoint field inventory mismatch")
    if payload["contract"] != PARTICLE_VIEW_TEACHER_CHECKPOINT_CONTRACT:
        raise ValueError("teacher checkpoint contract mismatch")
    if (
        payload["recipe"] != recipe.to_payload()
        or payload["recipe_sha256"] != recipe.content_hash
    ):
        raise ValueError("teacher checkpoint recipe drift")
    if payload["class_names"] != list(LABEL_NAMES):
        raise ValueError("teacher checkpoint class order drift")
    select_teacher_checkpoint(
        [{"epoch": payload["epoch"], **payload["model_val_stop"]}]
    )
    return payload


def build_teacher_registration(
    *,
    recipe: ParticleViewTeacherRecipe,
    checkpoint_path: str | Path,
    selected_checkpoint_kind: str = "trained_canonical",
    selectable: bool = True,
    provenance_reason: str = "exact_locked_recipe",
) -> dict[str, Any]:
    checkpoint = validate_teacher_checkpoint(
        checkpoint_path, recipe=recipe
    )
    kinds = {
        "trained_canonical",
        "existing_provenance_compatible",
        "existing_diagnostic",
    }
    if selected_checkpoint_kind not in kinds:
        raise ValueError("invalid selected_checkpoint_kind")
    if selected_checkpoint_kind == "existing_diagnostic" and selectable:
        raise ValueError("existing diagnostics cannot be selectable")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TEACHER_REGISTRATION_CONTRACT,
            "role": recipe.role,
            "particle_source": recipe.particle_source,
            "architecture_name": recipe.architecture,
            "seed": recipe.seed,
            "recipe": recipe.to_payload(),
            "recipe_sha256": recipe.content_hash,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "unified_split_manifest_sha256": recipe.unified_split_manifest_sha256,
            "train_identity_sha256": recipe.train_identity_sha256,
            "train_split_sha256": recipe.train_split_sha256,
            "model_val_stop_split_sha256": recipe.model_val_stop_split_sha256,
            "preprocessing_sha256": recipe.preprocessing_sha256,
            "source_sha256": recipe.source_sha256,
            "selected_epoch": int(checkpoint["epoch"]),
            "optimizer_updates": int(checkpoint["optimizer_updates"]),
            "model_val_stop": dict(checkpoint["model_val_stop"]),
            "selected_checkpoint_kind": selected_checkpoint_kind,
            "selectable": bool(selectable),
            "provenance_reason": str(provenance_reason),
            "frozen_after_registration": True,
            "refit_after_selection": False,
        }
    )
    validate_teacher_registration(artifact)
    return artifact


def validate_teacher_registration(payload: Mapping[str, Any]) -> str:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_TEACHER_REGISTRATION_CONTRACT
    )
    if canonical_sha256(payload.get("recipe")) != payload.get("recipe_sha256"):
        raise ValueError("teacher registration recipe hash mismatch")
    for name in (
        "checkpoint_sha256",
        "recipe_sha256",
        "unified_split_manifest_sha256",
        "train_identity_sha256",
        "train_split_sha256",
        "model_val_stop_split_sha256",
        "preprocessing_sha256",
        "source_sha256",
    ):
        require_sha256(name, payload.get(name))
    if (
        payload.get("frozen_after_registration") is not True
        or payload.get("refit_after_selection") is not False
    ):
        raise ValueError("teacher freeze/no-refit contract changed")
    if (
        payload.get("selected_checkpoint_kind") == "existing_diagnostic"
        and payload.get("selectable") is not False
    ):
        raise ValueError("existing diagnostic cannot be selectable")
    return str(payload["content_hash"])


def reload_registered_teacher(
    *,
    registration: Mapping[str, Any],
    checkpoint_path: str | Path,
    model_factory: Callable[[Mapping[str, Any]], Any] | None = None,
):
    validate_teacher_registration(registration)
    if sha256_file(checkpoint_path) != registration["checkpoint_sha256"]:
        raise ValueError("teacher checkpoint hash differs from registration")
    checkpoint = _torch_load(checkpoint_path)
    if checkpoint.get("recipe") != registration["recipe"]:
        raise ValueError("checkpoint recipe differs from registration")
    factory = model_factory or build_particle_view_teacher_model
    model = factory(registration["recipe"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    freeze_particle_teacher(model)
    audit_frozen_teacher(model)
    return model


def write_teacher_registration(
    path: str | Path, registration: Mapping[str, Any]
) -> dict[str, Any]:
    validate_teacher_registration(registration)
    return write_immutable_json(path, registration)


def build_predeclared_direct_control_grid() -> dict[str, Any]:
    candidates = []
    for width in (96, 128, 160, 192):
        for layers in (6, 8, 10, 12):
            candidates.append(
                {
                    "config_id": f"w{width}_l{layers}_c2",
                    "input_dim": len(PF_FEATURE_NAMES),
                    "embed_dims": [width, 4 * width, width],
                    "pair_embed_dims": [max(width // 2, 32)] * 3,
                    "num_heads": 8,
                    "num_layers": layers,
                    "num_cls_layers": 2,
                    "pair_input_dim": 4,
                    "hlt_only": True,
                }
            )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_DIRECT_CONTROL_GRID_CONTRACT,
            "matching_quantities": [
                "learned_parameter_count",
                "forward_flops",
            ],
            "parameter_relative_tolerance": 0.05,
            "forward_flop_relative_tolerance": 0.10,
            "tie_break_order": [
                "requested_quantity_relative_error",
                "other_quantity_relative_error",
                "smaller_parameter_count",
                "lexicographic_config_id",
            ],
            "candidates": candidates,
        }
    )


__all__ = [
    "CANONICAL_TAP_LOCATION",
    "CANONICAL_TOKEN_TAP",
    "ContextualParticleTokens",
    "FrozenContextualParticleTeacher",
    "FrozenTokenLayerMixture",
    "PARTICLE_VIEW_DIRECT_CONTROL_GRID_CONTRACT",
    "PARTICLE_VIEW_EXISTING_TEACHER_SOURCE_CONTRACT",
    "PARTICLE_VIEW_TEACHER_CHECKPOINT_CONTRACT",
    "PARTICLE_VIEW_TEACHER_RECIPE_CONTRACT",
    "PARTICLE_VIEW_TEACHER_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_TOKEN_TAP_CONTRACT",
    "PARTICLE_VIEW_TOKEN_TAP_REGISTRATION_CONTRACT",
    "ParticleTokenTapSpec",
    "ParticleViewTeacherRecipe",
    "audit_frozen_teacher",
    "build_particle_view_teacher_model",
    "build_existing_teacher_source_registration",
    "build_predeclared_direct_control_grid",
    "build_teacher_checkpoint_payload",
    "build_teacher_recipe",
    "build_teacher_registration",
    "build_token_tap_registration",
    "freeze_particle_teacher",
    "reload_registered_teacher",
    "select_teacher_checkpoint",
    "teacher_learning_rate",
    "token_tap_block_indices",
    "validate_matched_teacher_recipes",
    "validate_teacher_checkpoint",
    "validate_teacher_registration",
    "validate_token_tap_registration",
    "write_teacher_registration",
]
