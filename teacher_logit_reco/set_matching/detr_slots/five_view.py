"""Five-view tagger integration for DETR/free-slot reconstructed views."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.experiment import (
    HLT_VIEW_NAME,
    RECONSTRUCTED_VIEW_NAMES,
    view_name_for_reconstructor,
)
from teacher_logit_reco.set_matching.five_view_train import FiveViewTaggerTrainConfig, train_five_view_tagger

from .experiment import (
    DETR_SLOT_ENCODER_ARCHITECTURES,
    DETR_SLOT_HLT_VIEW_NAME,
    detr_slot_view_name_for_encoder,
)


DETR_SLOT_FIVE_VIEW_TAGGER_STEP = "detr_free_slot_step14_five_view_tagger_integration"
DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS: tuple[str, ...] = (
    "hlt_only",
    "hlt_plus_gt",
    "hlt_plus_pn",
    "hlt_plus_pfn",
    "hlt_plus_pcnn",
    "five_view_plain",
    "five_view_geometry",
    "five_view_no_confidence",
    "view_label_shuffle_control",
)


def detr_slot_five_view_name_mapping() -> dict[str, str]:
    """Map semantic DETR view names to the existing five-view loader names."""

    mapping = {DETR_SLOT_HLT_VIEW_NAME: HLT_VIEW_NAME}
    for architecture in DETR_SLOT_ENCODER_ARCHITECTURES:
        mapping[detr_slot_view_name_for_encoder(architecture)] = view_name_for_reconstructor(architecture)
    return mapping


def detr_slot_semantic_view_names_for_variant(variant: str) -> tuple[str, ...]:
    """Return semantic DETR view names left active by a tagger variant."""

    dropped_loader_views = set(detr_slot_five_view_variant_drop_views(variant))
    inverse = {loader_name: semantic for semantic, loader_name in detr_slot_five_view_name_mapping().items()}
    return tuple(
        inverse[loader_name]
        for loader_name in (HLT_VIEW_NAME,) + RECONSTRUCTED_VIEW_NAMES
        if loader_name not in dropped_loader_views
    )


def detr_slot_five_view_variant_drop_views(variant: str) -> tuple[str, ...]:
    """Return existing-loader view names to drop for a DETR five-view variant."""

    variant = str(variant).strip()
    all_reco = tuple(RECONSTRUCTED_VIEW_NAMES)
    if variant == "hlt_only":
        return all_reco
    for architecture in DETR_SLOT_ENCODER_ARCHITECTURES:
        if variant == f"hlt_plus_{architecture}":
            kept = view_name_for_reconstructor(architecture)
            return tuple(view for view in all_reco if view != kept)
    if variant in {"five_view_plain", "five_view_geometry", "five_view_no_confidence", "view_label_shuffle_control"}:
        return ()
    raise ValueError(f"Unknown DETR five-view tagger variant {variant!r}; expected one of {DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS}")


def detr_slot_five_view_output_dir(output_root: str | Path, variant: str) -> Path:
    """Return ``<output_root>/<variant>`` for a canonical DETR tagger variant."""

    if str(variant).strip() not in DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS:
        raise ValueError(f"Unknown DETR five-view tagger variant {variant!r}")
    return Path(output_root) / str(variant).strip()


def detr_slot_five_view_expected_cache_paths(
    reconstructed_view_dir: str | Path,
    *,
    splits: Sequence[str] = ("stack_train", "stack_val", "final_test"),
) -> dict[str, dict[str, str]]:
    """Return Step 12 cache paths consumed by the existing five-view loader."""

    root = Path(reconstructed_view_dir)
    return {
        architecture: {
            str(split): str(root / architecture / f"{split}_reconstructed_view.npz")
            for split in splits
        }
        for architecture in DETR_SLOT_ENCODER_ARCHITECTURES
    }


def build_detr_slot_five_view_tagger_config(
    base_config: FiveViewTaggerTrainConfig,
    *,
    variant: str,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> FiveViewTaggerTrainConfig:
    """Return a five-view tagger config specialized to one DETR variant."""

    variant = str(variant).strip()
    if variant not in DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS:
        raise ValueError(f"Unknown DETR five-view tagger variant {variant!r}")
    payload = asdict(base_config)
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    elif output_root is not None:
        payload["output_dir"] = str(detr_slot_five_view_output_dir(output_root, variant))
    payload["drop_views"] = detr_slot_five_view_variant_drop_views(variant)
    payload["use_geometry_attention"] = variant == "five_view_geometry"
    payload["use_confidence"] = False if variant == "five_view_no_confidence" else bool(base_config.use_confidence)
    payload["selection_mode"] = "all_slots" if variant == "five_view_no_confidence" else str(base_config.selection_mode)
    payload["shuffle_view_labels"] = variant == "view_label_shuffle_control"
    return FiveViewTaggerTrainConfig(**payload)


def train_detr_slot_five_view_tagger(
    base_config: FiveViewTaggerTrainConfig,
    *,
    variant: str,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one DETR-backed five-view tagger or ablation variant."""

    config = build_detr_slot_five_view_tagger_config(
        base_config,
        variant=variant,
        output_root=output_root,
        output_dir=output_dir,
    )
    report = train_five_view_tagger(config)
    integration_report = {
        "experiment_step": DETR_SLOT_FIVE_VIEW_TAGGER_STEP,
        "variant": str(variant),
        "semantic_view_names": list(detr_slot_semantic_view_names_for_variant(variant)),
        "loader_view_names": list((HLT_VIEW_NAME,) + RECONSTRUCTED_VIEW_NAMES),
        "semantic_to_loader_view_name": detr_slot_five_view_name_mapping(),
        "drop_views": list(config.drop_views),
        "five_view_report": report,
        "config": asdict(config),
    }
    save_json(Path(config.output_dir) / "detr_five_view_tagger_report.json", integration_report)
    output = dict(report)
    output["detr_five_view_integration"] = integration_report
    return output


def train_detr_slot_five_view_suite(
    base_config: FiveViewTaggerTrainConfig,
    *,
    variants: Sequence[str] = DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS,
    output_root: str | Path,
) -> dict[str, Any]:
    """Train the configured DETR-backed five-view tagger variant suite."""

    normalized = tuple(str(variant).strip() for variant in variants)
    if not normalized:
        raise ValueError("at least one DETR five-view variant is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate DETR five-view variants are not allowed: {normalized}")
    unknown = [variant for variant in normalized if variant not in DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown DETR five-view variants: {unknown}")
    reports = {
        variant: train_detr_slot_five_view_tagger(
            base_config,
            variant=variant,
            output_root=output_root,
        )
        for variant in normalized
    }
    summary = {
        "experiment_step": DETR_SLOT_FIVE_VIEW_TAGGER_STEP,
        "variants": list(normalized),
        "output_root": str(output_root),
        "semantic_to_loader_view_name": detr_slot_five_view_name_mapping(),
        "expected_cache_paths": detr_slot_five_view_expected_cache_paths(base_config.reconstructed_view_dir or ""),
        "reports": reports,
    }
    save_json(Path(output_root) / "five_view_tagger_suite_report.json", summary)
    return summary


__all__ = [
    "DETR_SLOT_FIVE_VIEW_TAGGER_STEP",
    "DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS",
    "build_detr_slot_five_view_tagger_config",
    "detr_slot_five_view_expected_cache_paths",
    "detr_slot_five_view_name_mapping",
    "detr_slot_five_view_output_dir",
    "detr_slot_five_view_variant_drop_views",
    "detr_slot_semantic_view_names_for_variant",
    "train_detr_slot_five_view_suite",
    "train_detr_slot_five_view_tagger",
]
