"""Production cache/model assembly shared by Step-6 command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_cache import load_cached_hlt_view

from .ca_tree import unpack_tree_shard
from .data import RelationalJetDataset, make_relational_loader
from .model import (
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
    build_confirmation_architecture_model,
    build_registered_screening_model,
    build_registered_wide_model,
)


def load_region_tree_split(
    tree_root: str | Path,
    *,
    split: str,
    expected_identities: Sequence[Any],
) -> list[Mapping[str, Any]]:
    root = Path(tree_root) / f"{split}_exclusive_ca_v1"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"REGION tree manifest is absent: {manifest}")
    shards = sorted((root / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"REGION tree shards are absent under {root}")
    identities: list[str] = []
    trees: list[Mapping[str, Any]] = []
    for shard in shards:
        shard_identities, shard_trees = unpack_tree_shard(shard)
        identities.extend(shard_identities)
        trees.extend(shard_trees)
    expected = [
        identity.key() if hasattr(identity, "key") else str(identity)
        for identity in expected_identities
    ]
    if identities != expected:
        raise ValueError(f"{split} REGION identities differ from the HLT cache")
    return trees


def build_cached_loaders(
    *,
    cache_dir: str | Path,
    seed: int,
    families: Sequence[str],
    tree_root: str | Path | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    views = {
        split: load_cached_hlt_view(cache_dir, split, verify_hash=True)
        for split in ("model_train", "model_val", "stack_val")
    }
    uses_region = "REGION" in families
    if uses_region and tree_root is None:
        raise ValueError("REGION run requires --tree-root")
    datasets = {}
    for split, view in views.items():
        trees = (
            load_region_tree_split(
                tree_root,
                split=split,
                expected_identities=view.jet_ids,
            )
            if uses_region
            else None
        )
        datasets[split] = RelationalJetDataset(view, region_trees=trees)
    loaders = (
        make_relational_loader(
            datasets["model_train"], seed=seed, training=True
        ),
        make_relational_loader(
            datasets["model_val"], seed=seed, training=False
        ),
        make_relational_loader(
            datasets["stack_val"], seed=seed, training=False
        ),
    )
    hashes = {
        split: str(view.metadata["hlt_content_hash"])
        for split, view in views.items()
    }
    return *loaders, hashes


def build_final_test_loader(
    *,
    cache_dir: str | Path,
    seed: int,
    families: Sequence[str],
    tree_root: str | Path | None = None,
) -> tuple[Any, str]:
    """Open only the sealed final-test HLT view and its required tree sidecar."""

    view = load_cached_hlt_view(
        cache_dir, "final_test", verify_hash=True
    )
    uses_region = "REGION" in families
    if uses_region and tree_root is None:
        raise ValueError("REGION final evaluation requires --tree-root")
    trees = (
        load_region_tree_split(
            tree_root,
            split="final_test",
            expected_identities=view.jet_ids,
        )
        if uses_region
        else None
    )
    dataset = RelationalJetDataset(view, region_trees=trees)
    loader = make_relational_loader(
        dataset, seed=int(seed), training=False
    )
    return loader, str(view.metadata["hlt_content_hash"])


def build_val_select_loader(
    *,
    cache_dir: str | Path,
    seed: int,
    families: Sequence[str],
    tree_root: str | Path | None = None,
) -> tuple[Any, str]:
    """Open only stack_val for post-confirmation semantic diagnostics."""

    view = load_cached_hlt_view(cache_dir, "stack_val", verify_hash=True)
    uses_region = "REGION" in families
    if uses_region and tree_root is None:
        raise ValueError("REGION semantic evaluation requires --tree-root")
    trees = (
        load_region_tree_split(
            tree_root,
            split="stack_val",
            expected_identities=view.jet_ids,
        )
        if uses_region
        else None
    )
    dataset = RelationalJetDataset(view, region_trees=trees)
    return (
        make_relational_loader(dataset, seed=int(seed), training=False),
        str(view.metadata["hlt_content_hash"]),
    )


def build_runtime_model(
    run_id: str,
    *,
    screening_registry: Mapping[str, Any],
    normalization_artifact: Mapping[str, Any],
    region_normalization_artifact: Mapping[str, Any] | None = None,
    selected_families: Sequence[str] = (),
    unary_registry: Mapping[str, Any] | None = None,
    weaver_module: Any | None = None,
) -> Any:
    if run_id == "RPT_SELECTED_UNARY":
        if unary_registry is None:
            raise ValueError("RPT_SELECTED_UNARY requires its immutable registry")
        from .semantic_controls import UnaryEndpointParticleTransformer

        return UnaryEndpointParticleTransformer(
            unary_registry=unary_registry,
            normalization_artifact=normalization_artifact,
            region_normalization_artifact=region_normalization_artifact,
            weaver_module=weaver_module,
        )
    if run_id == "RPT_BASE":
        return RelationalParticleTransformer(weaver_module=weaver_module)
    if run_id == "RPT_BASE_WIDE_MAX":
        return build_registered_wide_model(
            run_id,
            screening_registry=screening_registry,
            weaver_module=weaver_module,
        )
    if run_id in {
        "RPT_BASE_LAYERWISE",
        "RPT_BASE_EDGEVALUE",
        "RPT_SELECTED_LAYERWISE",
        "RPT_SELECTED_EDGEVALUE",
    }:
        return build_confirmation_architecture_model(
            run_id,
            selected_families=tuple(selected_families),
            normalization_artifact=normalization_artifact,
            region_normalization_artifact=region_normalization_artifact,
            weaver_module=weaver_module,
        )
    if run_id == "RPT_SELECTED_UNION":
        model = RelationalFamilyParticleTransformer(
            families=selected_families,
            normalization_artifact=normalization_artifact,
            region_normalization_artifact=region_normalization_artifact,
            weaver_module=weaver_module,
        )
        model.run_id = run_id
        return model
    return build_registered_screening_model(
        run_id,
        normalization_artifact=normalization_artifact,
        screening_registry=screening_registry,
        region_normalization_artifact=region_normalization_artifact,
        weaver_module=weaver_module,
    )


__all__ = [
    "build_cached_loaders",
    "build_final_test_loader",
    "build_val_select_loader",
    "build_runtime_model",
    "load_region_tree_split",
]
