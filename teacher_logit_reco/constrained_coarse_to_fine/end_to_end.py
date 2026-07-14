"""Step 8 staged end-to-end training for constrained pseudo-offline taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jetclass_fresh.hlt_baseline import (
    amp_autocast_context,
    amp_grad_scaler,
    resolve_device,
    set_training_seed,
)

from .fusion import (
    ARCH_HLT_CAPACITY_CONTROL,
    ARCH_HLT_EXTRA_ATTENTION,
    ARCH_HLT_ONLY,
    ARCH_PSEUDO_ONLY,
    A2_OFFLINE_REFERENCE,
    D5_END_TO_END,
    D8_MULTIDEPTH,
    ConstrainedDualStreamTagger,
    FusionTaggerOutput,
    ParticleStreamInput,
    build_dual_stream_fusion_tagger,
    fusion_variant_spec,
    grid_view_from_hierarchy_output,
    normalize_fusion_variant,
    particle_stream_from_tokens,
    pseudo_particle_views_from_rendered,
)
from .losses import HierarchyReconstructionLossConfig, compute_hierarchy_reconstruction_loss
from .pseudo import (
    PseudoParticleRenderOutput,
    load_coarse_to_fine_reconstructor_checkpoint,
    render_pseudo_particle_batch,
)
from .slot_loss import ParticleSlotLossConfig, compute_particle_slot_loss, prepare_cell_slot_targets
from .slots import CTierParticleReconstructor, CTierReconstructorOutput
from .train import (
    CoarseToFineTrainConfig,
    _iter_split_loaders,
    _load_split_source,
)


END_TO_END_TRAIN_CONTRACT = "constrained_coarse_to_fine_end_to_end_training_v1"
RECONSTRUCTOR_SOURCE_CONTRACT = "constrained_coarse_to_fine_reconstructor_sources_v1"

PHASE_FUSION_WARMUP = "fusion_head_warmup"
PHASE_FROZEN_RECONSTRUCTOR = "frozen_reconstructor_tagger"
PHASE_TERMINAL_DECODER = "terminal_decoder_finetune"
PHASE_UPPER_HIERARCHY = "upper_hierarchy_finetune"
PHASE_FULL_GENTLE = "full_gentle_finetune"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_state() -> dict[str, Any]:
    commit = os.environ.get("SOURCE_COMMIT") or os.environ.get("GIT_COMMIT")
    status_hash = os.environ.get("SOURCE_STATUS_HASH")
    try:
        if not commit:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        if not status_hash:
            status = subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            status_hash = hashlib.sha256(status.encode("utf-8")).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"source_commit": commit, "source_status_hash": status_hash}


@dataclass(frozen=True)
class ReconstructorSourceSpec:
    """One independently trained reconstructor and the views consumed from it."""

    name: str
    checkpoint_path: str
    view_names: tuple[str, ...]
    view_indices: tuple[int, ...] = ()
    expected_variant: str | None = None
    alias_of: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        names = tuple(str(row).strip() for row in self.view_names)
        indices = tuple(range(len(names))) if not self.view_indices else tuple(int(row) for row in self.view_indices)
        if not name or not names or any(not row for row in names):
            raise ValueError("reconstructor source and view names must be non-empty")
        if len(names) != len(set(names)) or len(names) != len(indices):
            raise ValueError("source view names must be unique and align with view_indices")
        if any(index < 0 for index in indices):
            raise ValueError("view_indices must be nonnegative")
        if self.alias_of is not None and str(self.alias_of) == name:
            raise ValueError("a reconstructor source cannot alias itself")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "view_names", names)
        object.__setattr__(self, "view_indices", indices)
        object.__setattr__(self, "alias_of", None if self.alias_of is None else str(self.alias_of))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "view_names": list(self.view_names), "view_indices": list(self.view_indices)}


@dataclass(frozen=True)
class ReconstructorViewBinding:
    view_name: str
    module_key: str
    source_name: str
    view_index: int
    source_variant: str


@dataclass
class ResolvedReconstructorSources:
    modules: nn.ModuleDict
    bindings: tuple[ReconstructorViewBinding, ...]
    source_metadata: Mapping[str, Mapping[str, Any]]
    aliases: Mapping[str, str]

    @property
    def view_names(self) -> tuple[str, ...]:
        return tuple(row.view_name for row in self.bindings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": RECONSTRUCTOR_SOURCE_CONTRACT,
            "active_view_names": list(self.view_names),
            "bindings": [asdict(row) for row in self.bindings],
            "sources": _jsonable(self.source_metadata),
            "aliases": dict(self.aliases),
            "unique_reconstructor_count": len(self.modules),
        }


def resolve_reconstructor_sources(
    specs: Sequence[ReconstructorSourceSpec],
    *,
    device: torch.device | str = "cpu",
) -> ResolvedReconstructorSources:
    """Load sources once, enforce declared aliases, and bind unique views.

    An alias is accepted only when both checkpoint bytes and resolved model
    configuration match its target. Alias views are deliberately omitted from
    the active view list so D8 never trains twice on one identical pseudo view.
    """

    rows = tuple(specs)
    by_name = {row.name: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("reconstructor source names must be unique")
    all_view_names = [name for row in rows for name in row.view_names]
    if len(all_view_names) != len(set(all_view_names)):
        raise ValueError("view names must be unique across reconstructor sources")
    loaded: dict[str, tuple[CTierParticleReconstructor, Mapping[str, Any], str, str]] = {}
    loaded_paths: dict[str, tuple[CTierParticleReconstructor, Mapping[str, Any], str, str]] = {}
    for row in rows:
        path = Path(row.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"missing reconstructor checkpoint for {row.name}: {path}")
        path_key = str(path.resolve())
        cached = loaded_paths.get(path_key)
        if cached is None:
            model, payload = load_coarse_to_fine_reconstructor_checkpoint(path, device=device)
            checkpoint_hash = _file_sha256(path)
            config_hash = _canonical_hash(payload.get("model"))
            cached = (model, payload, checkpoint_hash, config_hash)
            loaded_paths[path_key] = cached
        model, payload, checkpoint_hash, config_hash = cached
        variant = str(model.slot_decoder.config.variant)
        if row.expected_variant is not None and variant != str(row.expected_variant):
            raise ValueError(f"source {row.name} resolved to {variant}, expected {row.expected_variant}")
        max_views = int(model.slot_decoder.config.slot_spec.num_views)
        if any(index >= max_views for index in row.view_indices):
            raise ValueError(f"source {row.name} exposes {max_views} views but requested {row.view_indices}")
        loaded[row.name] = (model, payload, checkpoint_hash, config_hash)
    aliases: dict[str, str] = {}
    for row in rows:
        if row.alias_of is None:
            continue
        if row.alias_of not in by_name:
            raise ValueError(f"source {row.name} aliases unknown source {row.alias_of}")
        _, _, source_hash, source_config = loaded[row.name]
        _, _, target_hash, target_config = loaded[row.alias_of]
        if source_hash != target_hash or source_config != target_config:
            raise ValueError(
                f"declared alias {row.name}->{row.alias_of} does not reuse the same checkpoint and configuration"
            )
        aliases[row.name] = row.alias_of
    modules = nn.ModuleDict()
    module_for_identity: dict[tuple[str, str], str] = {}
    source_metadata: dict[str, Mapping[str, Any]] = {}
    bindings: list[ReconstructorViewBinding] = []
    for row in rows:
        model, payload, checkpoint_hash, config_hash = loaded[row.name]
        source_metadata[row.name] = {
            "checkpoint_path": str(Path(row.checkpoint_path)),
            "checkpoint_sha256": checkpoint_hash,
            "model_config_hash": config_hash,
            "checkpoint_role": payload.get("checkpoint_role"),
            "variant": str(model.slot_decoder.config.variant),
            "view_names": list(row.view_names),
            "view_indices": list(row.view_indices),
            "alias_of": row.alias_of,
            "provenance": payload.get("provenance"),
            # Keep the constructor payload inside the downstream checkpoint so
            # selected D-tier models remain deployable after source cleanup.
            "model": payload.get("model"),
        }
        if row.alias_of is not None:
            continue
        identity = (checkpoint_hash, config_hash)
        module_key = module_for_identity.get(identity)
        if module_key is None:
            module_key = f"reconstructor_{len(module_for_identity)}"
            module_for_identity[identity] = module_key
            modules[module_key] = model
        for view_name, view_index in zip(row.view_names, row.view_indices):
            bindings.append(
                ReconstructorViewBinding(
                    view_name=view_name,
                    module_key=module_key,
                    source_name=row.name,
                    view_index=int(view_index),
                    source_variant=str(model.slot_decoder.config.variant),
                )
            )
    if not bindings:
        raise ValueError("reconstructor resolution produced no active pseudo views")
    return ResolvedReconstructorSources(modules, tuple(bindings), source_metadata, aliases)


@dataclass(frozen=True)
class EndToEndForwardOutput:
    tagger: FusionTaggerOutput
    reconstructors: Mapping[str, CTierReconstructorOutput]
    renders: Mapping[str, PseudoParticleRenderOutput]
    pseudo_views: tuple[Any, ...]

    @property
    def logits(self) -> torch.Tensor:
        return self.tagger.logits


class EndToEndCoarseToFineTagger(nn.Module):
    """Live HLT -> reconstructor -> pseudo view -> fusion graph."""

    def __init__(
        self,
        tagger: ConstrainedDualStreamTagger,
        sources: ResolvedReconstructorSources,
        *,
        min_particle_pt: float = 0.0,
        dust_reliability: float = 0.10,
    ) -> None:
        super().__init__()
        if tuple(tagger.view_names) != sources.view_names:
            raise ValueError(
                f"tagger views {tuple(tagger.view_names)} do not match resolved source views {sources.view_names}"
            )
        self.tagger = tagger
        self.reconstructors = sources.modules
        self.bindings = sources.bindings
        self.source_metadata = sources.source_metadata
        self.aliases = sources.aliases
        self.min_particle_pt = float(min_particle_pt)
        self.dust_reliability = float(dust_reliability)

    def _source_bindings(self, module_key: str) -> tuple[ReconstructorViewBinding, ...]:
        return tuple(row for row in self.bindings if row.module_key == module_key)

    @staticmethod
    def _latent(
        model: CTierParticleReconstructor,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int | None,
        key: str,
    ) -> torch.Tensor | None:
        spec = model.slot_decoder.config.slot_spec
        if int(spec.num_views) <= 1:
            return None
        if seed is None:
            return None
        key_offset = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + key_offset)
        return torch.randn(
            batch,
            int(spec.num_views),
            int(spec.stochastic_latent_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def forward_detailed(
        self,
        hlt: ParticleStreamInput,
        *,
        reference_eta: torch.Tensor,
        reference_phi: torch.Tensor,
        stochastic_seed: int | None = None,
        view_availability_override: torch.Tensor | None = None,
    ) -> EndToEndForwardOutput:
        reconstructor_outputs: dict[str, CTierReconstructorOutput] = {}
        renders: dict[str, PseudoParticleRenderOutput] = {}
        pseudo_by_name: dict[str, Any] = {}
        for module_key, reconstructor in self.reconstructors.items():
            latent = self._latent(
                reconstructor,
                batch=hlt.batch_size,
                device=hlt.features.device,
                dtype=hlt.features.dtype,
                seed=stochastic_seed,
                key=module_key,
            )
            output = reconstructor(
                hlt.points,
                hlt.features,
                hlt.lorentz_vectors,
                hlt.mask,
                stochastic_latent=latent,
            )
            source_bindings = self._source_bindings(module_key)
            if self.tagger.spec.requires_grid_tokens:
                if len(source_bindings) != 1:
                    raise ValueError("D7 requires exactly one live hierarchy source")
                binding = source_bindings[0]
                pseudo_by_name[binding.view_name] = grid_view_from_hierarchy_output(
                    output.hierarchy,
                    reference_eta=reference_eta,
                    reference_phi=reference_phi,
                    name=binding.view_name,
                    radial_boundary=float(reconstructor.hierarchy.layout.radial_boundary),
                    coordinate_extent=float(reconstructor.hierarchy.layout.coordinate_extent),
                    source_variant=binding.source_variant,
                )
                reconstructor_outputs[module_key] = output
                continue
            rendered = render_pseudo_particle_batch(
                output,
                reference_eta=reference_eta,
                reference_phi=reference_phi,
                model=reconstructor,
                min_particle_pt=self.min_particle_pt,
                dust_reliability=self.dust_reliability,
            )
            views = pseudo_particle_views_from_rendered(
                rendered,
                view_names=tuple(row.view_name for row in source_bindings),
                view_indices=tuple(row.view_index for row in source_bindings),
                source_variant=source_bindings[0].source_variant,
            )
            reconstructor_outputs[module_key] = output
            renders[module_key] = rendered
            pseudo_by_name.update({view.name: view for view in views})
        pseudo_views = tuple(pseudo_by_name[name] for name in self.tagger.view_names)
        return EndToEndForwardOutput(
            tagger=self.tagger.forward_detailed(
                hlt,
                pseudo_views,
                view_availability_override=view_availability_override,
            ),
            reconstructors=reconstructor_outputs,
            renders=renders,
            pseudo_views=pseudo_views,
        )

    def forward(
        self,
        hlt: ParticleStreamInput,
        *,
        reference_eta: torch.Tensor,
        reference_phi: torch.Tensor,
        stochastic_seed: int | None = None,
    ) -> torch.Tensor:
        return self.forward_detailed(
            hlt,
            reference_eta=reference_eta,
            reference_phi=reference_phi,
            stochastic_seed=stochastic_seed,
        ).logits


def build_end_to_end_tagger(
    variant: str,
    sources: Sequence[ReconstructorSourceSpec],
    *,
    fusion_overrides: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[EndToEndCoarseToFineTagger, ResolvedReconstructorSources]:
    normalized = normalize_fusion_variant(variant)
    spec = fusion_variant_spec(normalized)
    if spec.architecture in {ARCH_HLT_ONLY, ARCH_HLT_CAPACITY_CONTROL, ARCH_HLT_EXTRA_ATTENTION}:
        if sources:
            raise ValueError(f"{normalized} is HLT-only and cannot declare reconstructor sources")
        resolved = ResolvedReconstructorSources(nn.ModuleDict(), (), {}, {})
    else:
        resolved = resolve_reconstructor_sources(sources, device=device)
    overrides = dict(fusion_overrides or {})
    overrides["view_names"] = resolved.view_names
    tagger = build_dual_stream_fusion_tagger(normalized, overrides=overrides)
    model = EndToEndCoarseToFineTagger(tagger, resolved)
    model.to(device)
    return model, resolved


def _filtered_dataclass_payload(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {name: value for name, value in payload.items() if name in allowed}


def _embedded_reconstructor(
    metadata: Mapping[str, Any],
) -> CTierParticleReconstructor:
    model_payload = metadata.get("model")
    provenance = metadata.get("provenance")
    train_provenance = provenance.get("model_train") if isinstance(provenance, Mapping) else None
    layout_payload = train_provenance.get("layout") if isinstance(train_provenance, Mapping) else None
    if not isinstance(model_payload, Mapping) or model_payload.get("family") != "C":
        raise ValueError("end-to-end checkpoint lacks an embedded C-tier model constructor")
    if not isinstance(layout_payload, Mapping):
        raise ValueError("end-to-end checkpoint lacks embedded model_train hierarchy layout")
    hierarchy_payload = model_payload.get("hierarchy_config")
    slot_payload = model_payload.get("slot_config")
    if not isinstance(hierarchy_payload, Mapping) or not isinstance(slot_payload, Mapping):
        raise ValueError("embedded reconstructor lacks hierarchy or slot configuration")
    from .layout import default_hierarchy_target_layout
    from .model import CoarseToFineReconstructorConfig
    from .slots import ParticleSlotDecoderConfig, build_c_tier_reconstructor

    layout = default_hierarchy_target_layout(
        radial_boundary=float(layout_payload["radial_boundary"]),
        coordinate_extent=float(layout_payload["coordinate_extent"]),
    )
    hierarchy_overrides = _filtered_dataclass_payload(CoarseToFineReconstructorConfig, hierarchy_payload)
    hierarchy_overrides.pop("variant", None)
    slot_overrides = _filtered_dataclass_payload(ParticleSlotDecoderConfig, slot_payload)
    slot_overrides.pop("variant", None)
    return build_c_tier_reconstructor(
        str(model_payload["variant"]),
        hierarchy_overrides=hierarchy_overrides,
        slot_overrides=slot_overrides,
        layout=layout,
    )


def load_end_to_end_tagger_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
    require_model_val_checkpoint: bool = True,
) -> tuple[EndToEndCoarseToFineTagger, Mapping[str, Any], ResolvedReconstructorSources]:
    """Rebuild a selected Step 8 tagger using only its own checkpoint bytes."""

    resolved_device = torch.device(device)
    try:
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=resolved_device)
    if payload.get("checkpoint_contract") != END_TO_END_TRAIN_CONTRACT:
        raise ValueError("end-to-end checkpoint contract mismatch")
    if require_model_val_checkpoint and payload.get("checkpoint_role") != "best_model_val":
        raise ValueError("prediction requires a best_model_val end-to-end checkpoint")
    reconstructor_payload = payload.get("reconstructors")
    if not isinstance(reconstructor_payload, Mapping):
        raise ValueError("end-to-end checkpoint lacks reconstructor source metadata")
    source_rows = reconstructor_payload.get("sources")
    binding_rows = reconstructor_payload.get("bindings")
    if not isinstance(source_rows, Mapping) or not isinstance(binding_rows, Sequence):
        raise ValueError("end-to-end reconstructor metadata is malformed")
    modules = nn.ModuleDict()
    bindings: list[ReconstructorViewBinding] = []
    source_for_module: dict[str, str] = {}
    for raw in binding_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("end-to-end checkpoint has a malformed view binding")
        binding = ReconstructorViewBinding(
            view_name=str(raw["view_name"]),
            module_key=str(raw["module_key"]),
            source_name=str(raw["source_name"]),
            view_index=int(raw["view_index"]),
            source_variant=str(raw["source_variant"]),
        )
        bindings.append(binding)
        source_for_module.setdefault(binding.module_key, binding.source_name)
    variant = normalize_fusion_variant(str(payload.get("variant")))
    if not bindings and fusion_variant_spec(variant).architecture not in {
        ARCH_HLT_ONLY,
        ARCH_HLT_CAPACITY_CONTROL,
        ARCH_HLT_EXTRA_ATTENTION,
    }:
        raise ValueError("end-to-end checkpoint exposes no active pseudo views")
    for module_key, source_name in source_for_module.items():
        metadata = source_rows.get(source_name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"missing embedded metadata for reconstructor source {source_name}")
        modules[module_key] = _embedded_reconstructor(metadata)
    aliases = reconstructor_payload.get("aliases")
    resolved = ResolvedReconstructorSources(
        modules=modules,
        bindings=tuple(bindings),
        source_metadata={str(name): row for name, row in source_rows.items() if isinstance(row, Mapping)},
        aliases={} if not isinstance(aliases, Mapping) else {str(name): str(value) for name, value in aliases.items()},
    )
    fusion_payload = payload.get("fusion_config")
    if not isinstance(fusion_payload, Mapping):
        raise ValueError("end-to-end checkpoint lacks fusion model configuration")
    from .fusion import FusionTaggerConfig

    config = FusionTaggerConfig(**_filtered_dataclass_payload(FusionTaggerConfig, fusion_payload))
    if tuple(config.resolved_view_names) != resolved.view_names:
        raise ValueError("checkpoint fusion views do not match embedded reconstructor bindings")
    model = EndToEndCoarseToFineTagger(
        ConstrainedDualStreamTagger(config),
        resolved,
        min_particle_pt=float(payload.get("min_particle_pt", 0.0)),
        dust_reliability=float(payload.get("dust_reliability", 0.10)),
    )
    result = model.load_state_dict(payload["model_state_dict"], strict=bool(strict))
    if strict and (result.missing_keys or result.unexpected_keys):
        raise ValueError(
            f"end-to-end checkpoint state mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.to(resolved_device)
    model.eval()
    return model, payload, resolved


@dataclass(frozen=True)
class EndToEndScheduleConfig:
    fusion_only_warmup_epochs: int = 1
    frozen_reconstructor_epochs: int = 4
    terminal_decoder_epochs: int = 4
    upper_hierarchy_epochs: int = 2
    unfreeze_reconstructor_hlt_encoder: bool = False
    tagger_learning_rate: float = 2.0e-4
    terminal_decoder_lr_scale: float = 0.075
    upper_hierarchy_lr_scale: float = 0.035
    reconstructor_hlt_encoder_lr_scale: float = 0.03
    reconstruction_weight: float = 0.10
    gate_entropy_weight: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "fusion_only_warmup_epochs",
            "frozen_reconstructor_epochs",
            "terminal_decoder_epochs",
            "upper_hierarchy_epochs",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if int(self.fusion_only_warmup_epochs) > int(self.frozen_reconstructor_epochs):
            raise ValueError("fusion-only warmup cannot exceed the frozen-reconstructor stage")
        for name in (
            "tagger_learning_rate",
            "terminal_decoder_lr_scale",
            "upper_hierarchy_lr_scale",
            "reconstructor_hlt_encoder_lr_scale",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("reconstruction_weight", "gate_entropy_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EndToEndTrainPhase:
    name: str
    fusion_head_only: bool
    terminal_decoder_trainable: bool
    upper_hierarchy_trainable: bool
    reconstructor_hlt_encoder_trainable: bool
    reconstruction_weight: float
    gate_entropy_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def end_to_end_phase(
    epoch: int,
    variant: str,
    config: EndToEndScheduleConfig,
) -> EndToEndTrainPhase:
    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    variant = normalize_fusion_variant(variant)
    if not fusion_variant_spec(variant).requires_end_to_end_schedule:
        return EndToEndTrainPhase(
            PHASE_FROZEN_RECONSTRUCTOR,
            False,
            False,
            False,
            False,
            0.0,
            float(config.gate_entropy_weight),
        )
    fusion_warmup = int(config.fusion_only_warmup_epochs) if variant == D8_MULTIDEPTH else 0
    if int(epoch) < fusion_warmup:
        return EndToEndTrainPhase(
            PHASE_FUSION_WARMUP, True, False, False, False, 0.0, float(config.gate_entropy_weight)
        )
    if int(epoch) < int(config.frozen_reconstructor_epochs):
        return EndToEndTrainPhase(
            PHASE_FROZEN_RECONSTRUCTOR,
            False,
            False,
            False,
            False,
            0.0,
            float(config.gate_entropy_weight),
        )
    terminal_stop = int(config.frozen_reconstructor_epochs) + int(config.terminal_decoder_epochs)
    if int(epoch) < terminal_stop:
        return EndToEndTrainPhase(
            PHASE_TERMINAL_DECODER,
            False,
            True,
            False,
            False,
            float(config.reconstruction_weight),
            0.0,
        )
    upper_stop = terminal_stop + int(config.upper_hierarchy_epochs)
    if int(epoch) < upper_stop or not bool(config.unfreeze_reconstructor_hlt_encoder):
        return EndToEndTrainPhase(
            PHASE_UPPER_HIERARCHY,
            False,
            True,
            True,
            False,
            float(config.reconstruction_weight),
            0.0,
        )
    return EndToEndTrainPhase(
        PHASE_FULL_GENTLE,
        False,
        True,
        True,
        True,
        float(config.reconstruction_weight),
        0.0,
    )


def _set_trainable(module: nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(bool(trainable))


def _set_phase_module_modes(
    model: EndToEndCoarseToFineTagger,
    phase: EndToEndTrainPhase,
    *,
    training: bool,
) -> None:
    model.train(bool(training))
    if not training:
        return
    if phase.fusion_head_only:
        if model.tagger.hlt_encoder is not None:
            model.tagger.hlt_encoder.eval()
        model.tagger.pseudo_encoders.eval()
    for reconstructor in model.reconstructors.values():
        if not phase.terminal_decoder_trainable:
            reconstructor.eval()
            continue
        reconstructor.train()
        if not phase.upper_hierarchy_trainable:
            reconstructor.hierarchy.eval()
        elif not phase.reconstructor_hlt_encoder_trainable:
            reconstructor.hierarchy.hlt_encoder.eval()


def apply_end_to_end_phase(
    model: EndToEndCoarseToFineTagger,
    phase: EndToEndTrainPhase,
) -> dict[str, Any]:
    """Apply an explicit trainability contract with no dormant trainable modules."""

    _set_trainable(model.tagger, not phase.fusion_head_only)
    if phase.fusion_head_only:
        for module in (
            model.tagger.cross_layers,
            model.tagger.pooled_gate,
            model.tagger.hlt_head,
            model.tagger.pseudo_head,
            model.tagger.classifier,
            model.tagger.capacity_residual,
        ):
            _set_trainable(module, True)
        _set_trainable(model.tagger.hlt_encoder, False)
        _set_trainable(model.tagger.shadow_hlt_encoder, False)
        _set_trainable(model.tagger.pseudo_encoders, False)
    for reconstructor in model.reconstructors.values():
        _set_trainable(reconstructor, False)
        if phase.terminal_decoder_trainable:
            _set_trainable(reconstructor.slot_decoder, True)
        if phase.upper_hierarchy_trainable:
            _set_trainable(reconstructor.hierarchy.global_predictor, True)
            _set_trainable(reconstructor.hierarchy.level_decoders, True)
        if phase.reconstructor_hlt_encoder_trainable:
            _set_trainable(reconstructor.hierarchy.hlt_encoder, True)
    groups = trainable_parameter_groups(model)
    return {
        "phase": phase.to_dict(),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "group_parameter_counts": {name: sum(parameter.numel() for parameter in rows) for name, rows in groups.items()},
        "reconstructors_frozen": not phase.terminal_decoder_trainable,
    }


def trainable_parameter_groups(model: EndToEndCoarseToFineTagger) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {}
    claimed: set[int] = set()

    def take(name: str, module: nn.Module | None) -> None:
        if module is None:
            return
        rows = [parameter for parameter in module.parameters() if parameter.requires_grad and id(parameter) not in claimed]
        if rows:
            groups.setdefault(name, []).extend(rows)
            claimed.update(id(parameter) for parameter in rows)

    take("tagger.hlt_encoder", model.tagger.hlt_encoder)
    take("tagger.shadow_hlt_encoder", model.tagger.shadow_hlt_encoder)
    take("tagger.pseudo_encoders", model.tagger.pseudo_encoders)
    tagger_remaining = [
        parameter
        for parameter in model.tagger.parameters()
        if parameter.requires_grad and id(parameter) not in claimed
    ]
    if tagger_remaining:
        groups["tagger.fusion_and_head"] = tagger_remaining
        claimed.update(id(parameter) for parameter in tagger_remaining)
    for key, reconstructor in model.reconstructors.items():
        take(f"reconstructor.{key}.slot_decoder", reconstructor.slot_decoder)
        take(f"reconstructor.{key}.hierarchy_decoders", reconstructor.hierarchy.global_predictor)
        take(f"reconstructor.{key}.hierarchy_decoders", reconstructor.hierarchy.level_decoders)
        take(f"reconstructor.{key}.hlt_encoder", reconstructor.hierarchy.hlt_encoder)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if claimed != expected:
        raise AssertionError(f"optimizer grouping missed {len(expected - claimed)} trainable parameters")
    flattened = [id(parameter) for rows in groups.values() for parameter in rows]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("a trainable parameter appears in multiple optimizer groups")
    if set(flattened) != expected:
        raise AssertionError("optimizer groups do not exactly cover trainable parameters")
    return groups


def build_end_to_end_optimizer(
    model: EndToEndCoarseToFineTagger,
    schedule: EndToEndScheduleConfig,
    *,
    weight_decay: float = 1.0e-4,
) -> torch.optim.Optimizer:
    groups = trainable_parameter_groups(model)
    payload: list[dict[str, Any]] = []
    for name, parameters in groups.items():
        scale = 1.0
        if name.endswith("slot_decoder"):
            scale = float(schedule.terminal_decoder_lr_scale)
        elif name.endswith("hierarchy_decoders"):
            scale = float(schedule.upper_hierarchy_lr_scale)
        elif name.endswith("hlt_encoder") and name.startswith("reconstructor."):
            scale = float(schedule.reconstructor_hlt_encoder_lr_scale)
        payload.append(
            {
                "params": parameters,
                "lr": float(schedule.tagger_learning_rate) * scale,
                "group_name": name,
                "lr_scale": scale,
            }
        )
    if not payload:
        raise ValueError("end-to-end phase has no trainable parameters")
    return torch.optim.AdamW(payload, weight_decay=float(weight_decay))


@dataclass(frozen=True)
class EndToEndLossConfig:
    reconstruction_slot_weight: float = 1.0
    kd_loss_weight: float = 0.0
    kd_temperature: float = 2.0

    def __post_init__(self) -> None:
        if float(self.reconstruction_slot_weight) < 0.0 or float(self.kd_loss_weight) < 0.0:
            raise ValueError("loss weights must be nonnegative")
        if float(self.kd_temperature) <= 0.0:
            raise ValueError("kd_temperature must be positive")


@dataclass(frozen=True)
class EndToEndLossOutput:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


def compute_end_to_end_loss(
    output: EndToEndForwardOutput,
    batch: Mapping[str, torch.Tensor],
    phase: EndToEndTrainPhase,
    config: EndToEndLossConfig | None = None,
) -> EndToEndLossOutput:
    """Combine tagging, optional KD, and unique-source reconstruction losses."""

    config = config or EndToEndLossConfig()
    labels = batch["labels"].long()
    ce = F.cross_entropy(output.logits, labels)
    total = ce + float(phase.gate_entropy_weight) * output.tagger.gate_entropy_regularizer
    metrics: dict[str, torch.Tensor] = {
        "loss.total_ce": ce,
        "metric.accuracy": (output.logits.argmax(dim=-1) == labels).float().mean(),
        "loss.gate_entropy_regularizer": output.tagger.gate_entropy_regularizer,
    }
    if float(config.kd_loss_weight) > 0.0:
        if "teacher_logits" not in batch:
            raise ValueError("KD is enabled but the batch has no teacher_logits")
        teacher = batch["teacher_logits"].to(device=output.logits.device, dtype=output.logits.dtype)
        if teacher.shape != output.logits.shape:
            raise ValueError("teacher and student logits do not have matching shapes")
        temperature = float(config.kd_temperature)
        kd = F.kl_div(
            F.log_softmax(output.logits / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
        total = total + float(config.kd_loss_weight) * kd
        metrics["loss.kd"] = kd
    reconstruction_rows: list[torch.Tensor] = []
    if float(phase.reconstruction_weight) > 0.0:
        required = {
            "global_accounting",
            "level1_accounting",
            "level2_accounting",
            "level3_accounting",
            "offline_tokens",
            "offline_mask",
            "final_cell_indices",
            "reference_eta",
            "reference_phi",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"end-to-end reconstruction supervision is missing {missing}")
        hierarchy_targets = {name: batch[name] for name in required if name.endswith("_accounting")}
        slot_target_cache: dict[int, Any] = {}
        for source_key, reconstructor_output in output.reconstructors.items():
            hierarchy_loss = compute_hierarchy_reconstruction_loss(
                reconstructor_output.hierarchy,
                hierarchy_targets,
                HierarchyReconstructionLossConfig(),
            )
            terminal_level = int(reconstructor_output.slots.terminal_level)
            if terminal_level not in slot_target_cache:
                slot_target_cache[terminal_level] = prepare_cell_slot_targets(
                    batch["offline_tokens"],
                    batch["offline_mask"],
                    batch["final_cell_indices"],
                    batch["reference_eta"],
                    batch["reference_phi"],
                    terminal_level=terminal_level,
                )
            slot_loss = compute_particle_slot_loss(
                reconstructor_output.slots,
                slot_target_cache[terminal_level],
                ParticleSlotLossConfig(
                    matching_mode=str(reconstructor_output.slots.diagnostics["matching_mode"])
                ),
            )
            reconstruction = hierarchy_loss.loss + float(config.reconstruction_slot_weight) * slot_loss.loss
            reconstruction_rows.append(reconstruction)
            metrics[f"reconstruction.{source_key}.hierarchy"] = hierarchy_loss.loss
            metrics[f"reconstruction.{source_key}.slot"] = slot_loss.loss
            metrics[f"reconstruction.{source_key}.terminal_closure"] = reconstructor_output.slots.diagnostics[
                "category_pt_closure_abs_max"
            ]
        reconstruction_mean = torch.stack(reconstruction_rows).mean()
        total = total + float(phase.reconstruction_weight) * reconstruction_mean
        metrics["loss.reconstruction_unique_source_mean"] = reconstruction_mean
    metrics["loss.total"] = total
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("end-to-end loss is non-finite")
    return EndToEndLossOutput(total, metrics)


def load_hlt_stream_warm_start(
    tagger: ConstrainedDualStreamTagger,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Strictly initialize the trusted HLT stream from a compatible checkpoint."""

    if tagger.hlt_encoder is None:
        raise ValueError("the selected tagger has no HLT stream to warm-start")
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("HLT warm-start checkpoint has no model state")
    target_keys = tuple(tagger.hlt_encoder.state_dict())
    prefixes = (
        "tagger.hlt_encoder.",
        "hlt_encoder.",
        "hierarchy.hlt_encoder.",
        "reconstructors.reconstructor_0.hierarchy.hlt_encoder.",
    )
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = []
    for prefix in prefixes:
        cleaned = {str(key)[len(prefix) :]: value for key, value in state.items() if str(key).startswith(prefix)}
        if cleaned:
            candidates.append((prefix, cleaned))
    for prefix, cleaned in candidates:
        if set(cleaned) != set(target_keys):
            continue
        tagger.hlt_encoder.load_state_dict(cleaned, strict=True)
        return {
            "checkpoint_path": str(path),
            "checkpoint_sha256": _file_sha256(path),
            "source_prefix": prefix,
            "loaded_tensor_count": len(cleaned),
            "strict": True,
        }
    raise ValueError(
        "HLT warm-start checkpoint is not exactly compatible with the trusted HLT encoder; "
        f"target has {len(target_keys)} tensors"
    )


@dataclass(frozen=True)
class EndToEndTrainConfig:
    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    offline_cache_dir: str
    target_cache_dir: str
    reconstructor_sources: tuple[ReconstructorSourceSpec, ...]
    variant: str = D5_END_TO_END
    hlt_warm_start_checkpoint: str | None = None
    allow_random_hlt_start: bool = False
    teacher_logits_train_path: str | None = None
    teacher_logits_val_path: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 28031
    epochs: int = 12
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 0
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    max_nonfinite_batches: int = 8
    device: str = "auto"
    amp: bool = True
    verify_hash: bool = True
    pin_memory: bool = True
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    save_last_checkpoint: bool = True
    schedule: EndToEndScheduleConfig = EndToEndScheduleConfig()
    loss: EndToEndLossConfig = EndToEndLossConfig()
    fusion_overrides: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        variant = normalize_fusion_variant(self.variant)
        spec = fusion_variant_spec(variant)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "reconstructor_sources", tuple(self.reconstructor_sources))
        if str(self.train_split) != "model_train" or str(self.val_split) != "model_val":
            raise ValueError("Step 8 trains on model_train and selects only on model_val")
        for name in ("epochs", "batch_size", "eval_batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_workers) < 0 or int(self.max_nonfinite_batches) < 0:
            raise ValueError("num_workers and max_nonfinite_batches must be nonnegative")
        if float(self.weight_decay) < 0.0 or float(self.grad_clip_norm) <= 0.0:
            raise ValueError("weight_decay must be nonnegative and grad_clip_norm positive")
        if (
            spec.architecture not in {ARCH_PSEUDO_ONLY, ARCH_HLT_ONLY}
            and not self.hlt_warm_start_checkpoint
            and not self.allow_random_hlt_start
        ):
            raise ValueError(
                f"{variant} requires a trusted HLT warm start unless allow_random_hlt_start is explicit"
            )
        if float(self.loss.kd_loss_weight) > 0.0:
            missing = [
                split
                for split, path in (
                    (self.train_split, self.teacher_logits_train_path),
                    (self.val_split, self.teacher_logits_val_path),
                )
                if not path or not Path(path).exists()
            ]
            if missing:
                raise FileNotFoundError(f"KD is enabled but teacher logits are missing for {missing}")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reconstructor_sources": [row.to_dict() for row in self.reconstructor_sources],
            "schedule": self.schedule.to_dict(),
            "loss": asdict(self.loss),
            "fusion_overrides": dict(self.fusion_overrides or {}),
            "contract": END_TO_END_TRAIN_CONTRACT,
        }


def _data_config(config: EndToEndTrainConfig) -> CoarseToFineTrainConfig:
    return CoarseToFineTrainConfig(
        output_dir=config.output_dir,
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        offline_cache_dir=config.offline_cache_dir,
        target_cache_dir=config.target_cache_dir,
        variant="C5",
        epochs=1,
        batch_size=int(config.batch_size),
        eval_batch_size=int(config.eval_batch_size),
        num_workers=int(config.num_workers),
        device=config.device,
        amp=bool(config.amp),
        verify_hash=bool(config.verify_hash),
        pin_memory=bool(config.pin_memory),
        max_train_jets=config.max_train_jets,
        max_val_jets=config.max_val_jets,
        save_last_checkpoint=False,
    )


def _teacher_logits(path: str | None, source: Any, num_classes: int) -> torch.Tensor | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        key = "logits" if "logits" in payload.files else "teacher_logits" if "teacher_logits" in payload.files else None
        if key is None or "labels" not in payload.files:
            raise ValueError(f"teacher logit cache {path} must contain logits and labels")
        logits = np.asarray(payload[key], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if logits.shape != (len(source.hlt_view.labels), int(num_classes)):
        raise ValueError(f"teacher logits {path} do not align with {source.split}")
    if not np.array_equal(labels, source.hlt_view.labels):
        raise ValueError(f"teacher logit labels differ from {source.split}")
    if not np.isfinite(logits).all():
        raise ValueError(f"teacher logits contain non-finite values: {path}")
    npz_path = Path(path)
    candidates = (
        npz_path.with_name(npz_path.stem + "_metadata.json"),
        npz_path.with_name(npz_path.stem.replace("_logits", "") + "_metadata.json"),
        npz_path.with_suffix(".json"),
    )
    metadata_path = next((row for row in candidates if row.exists()), None)
    if metadata_path is None:
        raise FileNotFoundError(f"teacher logits require provenance metadata beside {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if metadata.get("split") not in (None, source.split):
        problems.append("split mismatch")
    if int(metadata.get("n_jets", -1)) != len(source.hlt_view.labels):
        problems.append("n_jets mismatch")
    if metadata.get("jet_identity_hash") != source.provenance.get("jet_identity_hash"):
        problems.append("jet identity hash mismatch")
    source_manifest = metadata.get("source_manifest_hash")
    if source_manifest is not None and source_manifest != source.provenance.get("source_manifest_hash"):
        problems.append("manifest hash mismatch")
    if problems:
        raise ValueError(f"teacher logits provenance is invalid for {source.split}: " + "; ".join(problems))
    return torch.from_numpy(logits)


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device, non_blocking=True) for name, value in batch.items()}


def _metric_means(rows: list[tuple[int, Mapping[str, Any]]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    weights: dict[str, int] = {}
    for count, metrics in rows:
        for name, value in metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                scalar = float(value.detach().float().cpu().item())
            elif isinstance(value, (int, float)):
                scalar = float(value)
            else:
                continue
            if math.isfinite(scalar):
                totals[name] = totals.get(name, 0.0) + count * scalar
                weights[name] = weights.get(name, 0) + count
    return {name: totals[name] / max(1, weights[name]) for name in sorted(totals)}


def _run_end_to_end_epoch(
    model: EndToEndCoarseToFineTagger,
    source: Any,
    data_config: CoarseToFineTrainConfig,
    train_config: EndToEndTrainConfig,
    phase: EndToEndTrainPhase,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    teacher_logits: torch.Tensor | None,
    epoch: int,
    max_jets: int | None,
    amp_enabled: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    phase_metadata = apply_end_to_end_phase(model, phase)
    _set_phase_module_modes(model, phase, training=training)
    metric_rows: list[tuple[int, Mapping[str, Any]]] = []
    skipped = 0
    batch_number = 0
    for loader in _iter_split_loaders(
        source,
        data_config,
        train=training,
        require_offline_particles=True,
        max_jets=max_jets,
        epoch=epoch,
    ):
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            if teacher_logits is not None:
                batch["teacher_logits"] = teacher_logits.index_select(0, raw_batch["row_index"].long()).to(device)
            if train_config.variant == A2_OFFLINE_REFERENCE:
                if "offline_tokens" not in batch or "offline_mask" not in batch:
                    raise ValueError("A2 requires aligned offline particle inputs")
                hlt = particle_stream_from_tokens(batch["offline_tokens"], batch["offline_mask"])
            else:
                hlt = ParticleStreamInput(batch["points"], batch["features"], batch["vectors"], batch["mask"])
            if training:
                optimizer.zero_grad(set_to_none=True)
            try:
                with amp_autocast_context(bool(amp_enabled)):
                    output = model.forward_detailed(
                        hlt,
                        reference_eta=batch["reference_eta"],
                        reference_phi=batch["reference_phi"],
                        stochastic_seed=None if training else int(train_config.seed) + batch_number,
                    )
                    loss_output = compute_end_to_end_loss(output, batch, phase, train_config.loss)
                if training:
                    assert scaler is not None
                    scaler.scale(loss_output.loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        float(train_config.grad_clip_norm),
                        error_if_nonfinite=False,
                    )
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("end-to-end gradients are non-finite")
                    scaler.step(optimizer)
                    scaler.update()
                metric_rows.append((int(batch["labels"].shape[0]), loss_output.metrics))
            except FloatingPointError:
                skipped += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                if skipped > int(train_config.max_nonfinite_batches):
                    raise
            batch_number += 1
    metrics = _metric_means(metric_rows)
    metrics["nonfinite_batches_skipped"] = float(skipped)
    metrics["n_jets"] = float(sum(count for count, _ in metric_rows))
    metrics["phase"] = phase.name
    metrics["phase_metadata"] = phase_metadata
    return metrics


def _write_curves_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for row in curves:
        flat: dict[str, Any] = {"epoch": row["epoch"], "phase": row["phase"]}
        for split in ("train", "model_val"):
            for name, value in row[split].items():
                if isinstance(value, (int, float, str)):
                    flat[f"{split}.{name}"] = value
        rows.append(flat)
    fields = sorted({name for row in rows for name in row}, key=lambda name: (name not in {"epoch", "phase"}, name))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_source_provenance(
    resolved: ResolvedReconstructorSources,
    sources: Mapping[str, Any],
) -> None:
    problems: list[str] = []
    for name, metadata in resolved.source_metadata.items():
        provenance = metadata.get("provenance")
        for split, source in sources.items():
            checkpoint_split = provenance.get(split) if isinstance(provenance, Mapping) else None
            if not isinstance(checkpoint_split, Mapping):
                problems.append(f"{name} lacks {split} checkpoint provenance")
                continue
            if checkpoint_split.get("source_manifest_hash") != source.provenance["source_manifest_hash"]:
                problems.append(f"{name} {split} manifest hash mismatch")
            if checkpoint_split.get("hlt_content_hash") != source.provenance["hlt_content_hash"]:
                problems.append(f"{name} {split} HLT cache hash mismatch")
    if problems:
        raise ValueError("invalid Step 8 reconstructor provenance: " + "; ".join(problems))


def _optimizer_with_preserved_state(
    model: EndToEndCoarseToFineTagger,
    schedule: EndToEndScheduleConfig,
    *,
    weight_decay: float,
    previous: torch.optim.Optimizer | None,
) -> torch.optim.Optimizer:
    optimizer = build_end_to_end_optimizer(model, schedule, weight_decay=weight_decay)
    if previous is None:
        return optimizer
    active = {parameter for group in optimizer.param_groups for parameter in group["params"]}
    for parameter, state in previous.state.items():
        if parameter in active:
            optimizer.state[parameter] = state
    return optimizer


def train_end_to_end_tagger(config: EndToEndTrainConfig) -> dict[str, Any]:
    """Train frozen controls or staged D5/D6/D8 with model-val-only selection."""

    set_training_seed(int(config.seed))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    data_config = _data_config(config)
    train_source = _load_split_source(data_config, config.train_split, require_offline_particles=True)
    train_layout = train_source.layout.to_dict()
    train_provenance = dict(train_source.provenance)
    model, resolved = build_end_to_end_tagger(
        config.variant,
        config.reconstructor_sources,
        fusion_overrides=config.fusion_overrides,
        device=device,
    )
    _validate_source_provenance(
        resolved,
        {config.train_split: train_source},
    )
    del train_source
    gc.collect()
    val_source = _load_split_source(data_config, config.val_split, require_offline_particles=True)
    if train_layout != val_source.layout.to_dict():
        raise ValueError("model_train/model_val hierarchy layouts differ")
    val_provenance = dict(val_source.provenance)
    _validate_source_provenance(resolved, {config.val_split: val_source})
    del val_source
    gc.collect()
    warm_start = None
    if config.hlt_warm_start_checkpoint and model.tagger.hlt_encoder is not None:
        warm_start = load_hlt_stream_warm_start(model.tagger, config.hlt_warm_start_checkpoint)
    elif config.hlt_warm_start_checkpoint:
        warm_start = {
            "checkpoint_path": str(config.hlt_warm_start_checkpoint),
            "not_loaded": True,
            "reason": "selected pseudo-only tagger has no HLT stream",
        }
    source_metadata = {
        "contract": END_TO_END_TRAIN_CONTRACT,
        "config": config.to_dict(),
        "reconstructors": resolved.to_dict(),
        "trusted_hlt_warm_start": warm_start,
        "provenance": {"model_train": train_provenance, "model_val": val_provenance},
        "split_loading": "sequential_reload_per_epoch",
        "simultaneously_resident_source_splits": 1,
        "selection_split": "model_val",
        "final_test_loaded": False,
        "input_view": "offline" if config.variant == A2_OFFLINE_REFERENCE else "fixed_hlt_v2_realistic",
        "offline_inputs_used_by_deployable_forward": config.variant == A2_OFFLINE_REFERENCE,
        "source_state": _source_state(),
    }
    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "source_metadata.json", source_metadata)
    configuration_hash = _canonical_hash(
        {"fusion": model.tagger.config.to_dict(), "reconstructors": resolved.to_dict()}
    )
    curves: list[dict[str, Any]] = []
    best_epoch = -1
    best_ce = float("inf")
    best_metrics: Mapping[str, Any] = {}
    optimizer: torch.optim.Optimizer | None = None
    scaler = amp_grad_scaler(amp_enabled)
    last_phase_name = None
    for epoch in range(int(config.epochs)):
        phase = end_to_end_phase(epoch, config.variant, config.schedule)
        model.train(True)
        apply_end_to_end_phase(model, phase)
        if phase.name != last_phase_name:
            optimizer = _optimizer_with_preserved_state(
                model,
                config.schedule,
                weight_decay=float(config.weight_decay),
                previous=optimizer,
            )
            last_phase_name = phase.name
        assert optimizer is not None
        train_source = _load_split_source(data_config, config.train_split, require_offline_particles=True)
        if dict(train_source.provenance) != train_provenance:
            raise ValueError("model_train provenance changed during sequential epoch loading")
        train_teacher = _teacher_logits(
            config.teacher_logits_train_path,
            train_source,
            model.tagger.config.num_classes,
        )
        train_metrics = _run_end_to_end_epoch(
            model,
            train_source,
            data_config,
            config,
            phase,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            teacher_logits=train_teacher,
            epoch=epoch,
            max_jets=config.max_train_jets,
            amp_enabled=amp_enabled,
        )
        del train_teacher
        del train_source
        gc.collect()
        val_source = _load_split_source(data_config, config.val_split, require_offline_particles=True)
        if dict(val_source.provenance) != val_provenance:
            raise ValueError("model_val provenance changed during sequential epoch loading")
        val_teacher = _teacher_logits(
            config.teacher_logits_val_path,
            val_source,
            model.tagger.config.num_classes,
        )
        with torch.no_grad():
            val_metrics = _run_end_to_end_epoch(
                model,
                val_source,
                data_config,
                config,
                phase,
                device=device,
                optimizer=None,
                scaler=None,
                teacher_logits=val_teacher,
                epoch=epoch,
                max_jets=config.max_val_jets,
                amp_enabled=amp_enabled,
            )
        del val_teacher
        del val_source
        gc.collect()
        row = {"epoch": epoch, "phase": phase.name, "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        val_ce = float(val_metrics.get("loss.total_ce", float("nan")))
        if math.isfinite(val_ce) and val_ce < best_ce:
            best_epoch = epoch
            best_ce = val_ce
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "checkpoint_contract": END_TO_END_TRAIN_CONTRACT,
                    "checkpoint_role": "best_model_val",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "variant": config.variant,
                    "fusion_config": model.tagger.config.to_dict(),
                    "reconstructors": resolved.to_dict(),
                    "configuration_hash": configuration_hash,
                    "min_particle_pt": model.min_particle_pt,
                    "dust_reliability": model.dust_reliability,
                    "metrics": val_metrics,
                    "provenance": source_metadata["provenance"],
                    "source_state": source_metadata["source_state"],
                },
                output_dir / "best_model_val.pt",
            )
        if config.save_last_checkpoint:
            torch.save(
                {
                    "checkpoint_contract": END_TO_END_TRAIN_CONTRACT,
                    "checkpoint_role": "last",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "variant": config.variant,
                    "metrics": val_metrics,
                },
                output_dir / "last.pt",
            )
        _write_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_curves_csv(output_dir / "diagnostics" / "epoch_metrics.csv", curves)
    checkpoint = output_dir / "best_model_val.pt"
    if best_epoch < 0 or not checkpoint.exists():
        raise RuntimeError("Step 8 did not produce a finite model_val checkpoint")
    report = {
        "ok": True,
        "contract": END_TO_END_TRAIN_CONTRACT,
        "variant": config.variant,
        "best_epoch": best_epoch,
        "best_model_val_cross_entropy": best_ce,
        "best_model_val_metrics": best_metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "configuration_hash": configuration_hash,
        "reconstructors": resolved.to_dict(),
        "provenance": source_metadata["provenance"],
        "trusted_hlt_warm_start": warm_start,
        "phase_history": [row["phase"] for row in curves],
        "selection_split": "model_val",
        "final_test_evaluated": False,
        "source_state": source_metadata["source_state"],
        "input_view": source_metadata["input_view"],
        "deployable_hlt_only": config.variant != A2_OFFLINE_REFERENCE,
    }
    _write_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "END_TO_END_TRAIN_CONTRACT",
    "RECONSTRUCTOR_SOURCE_CONTRACT",
    "PHASE_FUSION_WARMUP",
    "PHASE_FROZEN_RECONSTRUCTOR",
    "PHASE_TERMINAL_DECODER",
    "PHASE_UPPER_HIERARCHY",
    "PHASE_FULL_GENTLE",
    "ReconstructorSourceSpec",
    "ReconstructorViewBinding",
    "ResolvedReconstructorSources",
    "EndToEndForwardOutput",
    "EndToEndCoarseToFineTagger",
    "EndToEndScheduleConfig",
    "EndToEndTrainPhase",
    "EndToEndLossConfig",
    "EndToEndLossOutput",
    "EndToEndTrainConfig",
    "resolve_reconstructor_sources",
    "build_end_to_end_tagger",
    "load_end_to_end_tagger_checkpoint",
    "end_to_end_phase",
    "apply_end_to_end_phase",
    "trainable_parameter_groups",
    "build_end_to_end_optimizer",
    "compute_end_to_end_loss",
    "load_hlt_stream_warm_start",
    "train_end_to_end_tagger",
]
