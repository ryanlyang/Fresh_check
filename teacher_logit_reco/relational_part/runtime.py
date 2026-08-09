"""Production cache/model assembly shared by Step-6 command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_cache import load_cached_hlt_view
from teacher_logit_reco.architecture_view_part.train import (
    load_cached_offline_view,
)

from .ca_tree import unpack_tree_shard
from .ca_tree import (
    ANGULAR_TREE_SHARD_CONTRACT,
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    VIEW_TREE_SHARD_CONTRACT,
    VIEW_TREE_SPLIT_MANIFEST_CONTRACT,
)
from .contracts import load_hashed_json, sha256_file
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
    input_view: str = "fixed_hlt",
    input_content_sha256: str | None = None,
) -> list[Mapping[str, Any]]:
    root = Path(tree_root) / f"{split}_exclusive_ca_v1"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"REGION tree manifest is absent: {manifest}")
    expected_manifest_contract = (
        ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT
        if input_view == "fixed_hlt"
        else VIEW_TREE_SPLIT_MANIFEST_CONTRACT
    )
    manifest_artifact = load_hashed_json(
        manifest, expected_contract=expected_manifest_contract
    )
    expected_content = (
        None
        if input_content_sha256 is None
        else str(input_content_sha256)
    )
    parent_key = (
        "hlt_content_sha256"
        if input_view == "fixed_hlt"
        else "input_content_sha256"
    )
    if expected_content is not None and manifest_artifact.get(
        "parents", {}
    ).get(parent_key) != expected_content:
        raise ValueError(f"{split} REGION manifest belongs to another input cache")
    if input_view == "offline" and manifest_artifact.get("parents", {}).get(
        "input_view"
    ) != "offline":
        raise ValueError("offline REGION manifest view differs")
    shards = sorted((root / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"REGION tree shards are absent under {root}")
    identities: list[str] = []
    trees: list[Mapping[str, Any]] = []
    if len(shards) != len(manifest_artifact.get("shards", ())):
        raise ValueError(f"{split} REGION shard count differs from manifest")
    metadata_contract = (
        ANGULAR_TREE_SHARD_CONTRACT
        if input_view == "fixed_hlt"
        else VIEW_TREE_SHARD_CONTRACT
    )
    for shard_index, shard in enumerate(shards):
        metadata = load_hashed_json(
            shard.with_suffix(".metadata.json"),
            expected_contract=metadata_contract,
        )
        row = manifest_artifact["shards"][shard_index]
        if (
            int(row["shard_index"]) != shard_index
            or row["metadata_sha256"] != metadata["content_hash"]
            or row["npz_sha256"] != sha256_file(shard)
            or metadata["npz_sha256"] != row["npz_sha256"]
        ):
            raise ValueError(f"{split} REGION shard authentication differs")
        shard_identities, shard_trees = unpack_tree_shard(shard)
        identities.extend(shard_identities)
        trees.extend(shard_trees)
    expected = [
        identity.key() if hasattr(identity, "key") else str(identity)
        for identity in expected_identities
    ]
    if identities != expected:
        raise ValueError(f"{split} REGION identities differ from the input cache")
    return trees


def load_cached_input_view(
    cache_dir: str | Path,
    split: str,
    *,
    input_view: str,
):
    """Load one authenticated raw-token view without relabeling its semantics."""

    if input_view == "fixed_hlt":
        return load_cached_hlt_view(cache_dir, split, verify_hash=True)
    if input_view == "offline":
        return load_cached_offline_view(cache_dir, split, verify_hash=True)
    raise ValueError("input_view must be fixed_hlt or offline")


def cached_view_content_hash(view: Any, *, input_view: str) -> str:
    key = "hlt_content_hash" if input_view == "fixed_hlt" else "offline_content_hash"
    value = view.metadata.get(key)
    if not isinstance(value, str):
        raise ValueError(f"cached {input_view} view lacks {key}")
    return value


def build_cached_loaders(
    *,
    cache_dir: str | Path,
    seed: int,
    families: Sequence[str],
    tree_root: str | Path | None = None,
    input_view: str = "fixed_hlt",
) -> tuple[Any, Any, Any, dict[str, Any]]:
    views = {
        split: load_cached_input_view(cache_dir, split, input_view=input_view)
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
                input_view=input_view,
                input_content_sha256=cached_view_content_hash(
                    view, input_view=input_view
                ),
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
        split: cached_view_content_hash(view, input_view=input_view)
        for split, view in views.items()
    }
    return *loaders, hashes


def build_final_test_loader(
    *,
    cache_dir: str | Path,
    seed: int,
    families: Sequence[str],
    tree_root: str | Path | None = None,
    input_view: str = "fixed_hlt",
) -> tuple[Any, str]:
    """Open only the sealed final-test HLT view and its required tree sidecar."""

    view = load_cached_input_view(cache_dir, "final_test", input_view=input_view)
    uses_region = "REGION" in families
    if uses_region and tree_root is None:
        raise ValueError("REGION final evaluation requires --tree-root")
    trees = (
        load_region_tree_split(
            tree_root,
            split="final_test",
            expected_identities=view.jet_ids,
            input_view=input_view,
            input_content_sha256=cached_view_content_hash(
                view, input_view=input_view
            ),
        )
        if uses_region
        else None
    )
    dataset = RelationalJetDataset(view, region_trees=trees)
    loader = make_relational_loader(
        dataset, seed=int(seed), training=False
    )
    return loader, cached_view_content_hash(view, input_view=input_view)


def build_val_select_loader(
    *,
    cache_dir: str | Path,
    seed: int,
    families: Sequence[str],
    tree_root: str | Path | None = None,
    input_view: str = "fixed_hlt",
) -> tuple[Any, str]:
    """Open only stack_val for post-confirmation semantic diagnostics."""

    view = load_cached_input_view(cache_dir, "stack_val", input_view=input_view)
    uses_region = "REGION" in families
    if uses_region and tree_root is None:
        raise ValueError("REGION semantic evaluation requires --tree-root")
    trees = (
        load_region_tree_split(
            tree_root,
            split="stack_val",
            expected_identities=view.jet_ids,
            input_view=input_view,
            input_content_sha256=cached_view_content_hash(
                view, input_view=input_view
            ),
        )
        if uses_region
        else None
    )
    dataset = RelationalJetDataset(view, region_trees=trees)
    return (
        make_relational_loader(dataset, seed=int(seed), training=False),
        cached_view_content_hash(view, input_view=input_view),
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
    if run_id == "OFF_RPT_BASE":
        model = RelationalParticleTransformer(weaver_module=weaver_module)
        model.run_id = run_id
        return model
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
        "OFF_RPT_BASE_EDGEVALUE",
        "OFF_RPT_SELECTED_LAYERWISE",
        "OFF_RPT_SELECTED_EDGEVALUE",
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
    "cached_view_content_hash",
    "load_cached_input_view",
    "load_region_tree_split",
]
