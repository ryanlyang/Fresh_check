"""Single-reco sanity taggers for DETR/free-slot reconstructed views."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.experiment import (
    RECONSTRUCTED_VIEW_NAMES,
    view_name_for_reconstructor,
)
from teacher_logit_reco.set_matching.five_view_train import FiveViewTaggerTrainConfig, train_five_view_tagger

from .experiment import DETR_SLOT_ENCODER_ARCHITECTURES, normalize_detr_slot_encoder_architecture


DETR_SLOT_SINGLE_RECO_TAGGER_STEP = "detr_free_slot_step13_single_reco_sanity_taggers"
DETR_SLOT_SINGLE_RECO_VARIANTS: tuple[str, ...] = tuple(
    f"hlt_plus_{architecture}" for architecture in DETR_SLOT_ENCODER_ARCHITECTURES
)


def detr_slot_single_reco_variant_name(architecture: str) -> str:
    """Return the canonical sanity-tagger variant name for one DETR view."""

    return f"hlt_plus_{normalize_detr_slot_encoder_architecture(architecture)}"


def detr_slot_architecture_from_single_reco_variant(variant: str) -> str:
    """Parse ``hlt_plus_<arch>`` into a normalized DETR encoder architecture."""

    text = str(variant).strip()
    if not text.startswith("hlt_plus_"):
        raise ValueError(f"DETR single-reco variant must start with 'hlt_plus_', got {variant!r}")
    return normalize_detr_slot_encoder_architecture(text.removeprefix("hlt_plus_"))


def detr_slot_single_reco_drop_views(architecture: str) -> tuple[str, ...]:
    """Drop every reconstructed view except the requested architecture."""

    architecture = normalize_detr_slot_encoder_architecture(architecture)
    kept = view_name_for_reconstructor(architecture)
    return tuple(view for view in RECONSTRUCTED_VIEW_NAMES if view != kept)


def detr_slot_single_reco_output_dir(output_root: str | Path, architecture: str) -> Path:
    """Return ``<output_root>/hlt_plus_<arch>``."""

    return Path(output_root) / detr_slot_single_reco_variant_name(architecture)


def detr_slot_single_reco_expected_cache_paths(
    reconstructed_view_dir: str | Path,
    *,
    splits: Sequence[str] = ("stack_train", "stack_val", "final_test"),
) -> dict[str, dict[str, str]]:
    """Return the cache files the existing five-view loader will expect."""

    root = Path(reconstructed_view_dir)
    return {
        architecture: {
            str(split): str(root / architecture / f"{split}_reconstructed_view.npz")
            for split in splits
        }
        for architecture in DETR_SLOT_ENCODER_ARCHITECTURES
    }


def build_detr_slot_single_reco_tagger_config(
    base_config: FiveViewTaggerTrainConfig,
    *,
    architecture: str,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> FiveViewTaggerTrainConfig:
    """Return a five-view config with the correct single-reco drop mask."""

    architecture = normalize_detr_slot_encoder_architecture(architecture)
    payload = asdict(base_config)
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    elif output_root is not None:
        payload["output_dir"] = str(detr_slot_single_reco_output_dir(output_root, architecture))
    payload["drop_views"] = detr_slot_single_reco_drop_views(architecture)
    return FiveViewTaggerTrainConfig(**payload)


def train_detr_slot_single_reco_tagger(
    base_config: FiveViewTaggerTrainConfig,
    *,
    architecture: str,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one ``HLT + one DETR reco view`` sanity tagger."""

    architecture = normalize_detr_slot_encoder_architecture(architecture)
    config = build_detr_slot_single_reco_tagger_config(
        base_config,
        architecture=architecture,
        output_root=output_root,
        output_dir=output_dir,
    )
    report = train_five_view_tagger(config)
    sanity_report = {
        "experiment_step": DETR_SLOT_SINGLE_RECO_TAGGER_STEP,
        "architecture": architecture,
        "variant": detr_slot_single_reco_variant_name(architecture),
        "kept_reconstructed_view": view_name_for_reconstructor(architecture),
        "drop_views": list(config.drop_views),
        "five_view_report": report,
        "config": asdict(config),
    }
    save_json(Path(config.output_dir) / "detr_single_reco_sanity_report.json", sanity_report)
    output = dict(report)
    output["detr_single_reco_sanity"] = sanity_report
    return output


def train_detr_slot_single_reco_suite(
    base_config: FiveViewTaggerTrainConfig,
    *,
    architectures: Sequence[str] = DETR_SLOT_ENCODER_ARCHITECTURES,
    output_root: str | Path,
) -> dict[str, Any]:
    """Train one sanity tagger per requested DETR architecture."""

    normalized = tuple(normalize_detr_slot_encoder_architecture(architecture) for architecture in architectures)
    if not normalized:
        raise ValueError("at least one DETR architecture is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate DETR architectures are not allowed: {normalized}")
    reports = {
        architecture: train_detr_slot_single_reco_tagger(
            base_config,
            architecture=architecture,
            output_root=output_root,
        )
        for architecture in normalized
    }
    summary = {
        "experiment_step": DETR_SLOT_SINGLE_RECO_TAGGER_STEP,
        "architectures": list(normalized),
        "output_root": str(output_root),
        "variants": {architecture: detr_slot_single_reco_variant_name(architecture) for architecture in normalized},
        "reports": reports,
    }
    save_json(Path(output_root) / "single_reco_sanity_suite_report.json", summary)
    return summary


__all__ = [
    "DETR_SLOT_SINGLE_RECO_TAGGER_STEP",
    "DETR_SLOT_SINGLE_RECO_VARIANTS",
    "build_detr_slot_single_reco_tagger_config",
    "detr_slot_architecture_from_single_reco_variant",
    "detr_slot_single_reco_drop_views",
    "detr_slot_single_reco_expected_cache_paths",
    "detr_slot_single_reco_output_dir",
    "detr_slot_single_reco_variant_name",
    "train_detr_slot_single_reco_suite",
    "train_detr_slot_single_reco_tagger",
]
