#!/usr/bin/env python3
"""Fit every frozen 500k target statistic and publish exact wave coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_heteroscedastic_metadata,
    build_hlt_conditional_context,
    fit_conditional_residual,
    fit_latent_whitening,
    fit_target_normalizer,
    load_and_validate_campaign,
    load_hashed_json,
    load_target_cache,
    materialize_native_relation_target,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    TARGET_CACHE_SPEC_CONTRACT,
    TARGET_NORMALIZATION_WAVE_CONTRACT,
    TEACHER_LOCK_CONTRACT,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.scale_runtime import (  # noqa: E402
    fit_pair_normalizer_from_views,
    merge_target_normalizers,
)


def _mapping(values: list[str], *, name: str, paths: bool = False):
    result = {}
    for value in values:
        key, separator, payload = value.partition("=")
        if not separator or int(key) in result:
            raise ValueError(f"{name} requires unique REPLICA=PATH")
        result[int(key)] = Path(payload) if paths else payload
    if set(result) != {0, 1, 2, 3}:
        raise ValueError(f"{name} requires replicas 0,1,2,3")
    return result


def _cache(path: Path):
    spec = load_hashed_json(
        path / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
    )
    return load_target_cache(path, cache_spec=spec)


def _input(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        identity_key = (
            "identity" if "identity" in payload.files else "identities"
        )
        if (
            {"label", "labels", "class", "classes", "y"}
            & set(payload.files)
            or not {identity_key, "raw_tokens", "mask"}.issubset(payload.files)
        ):
            raise ValueError("normalization HLT input is not label blind")
        identities = tuple(
            str(value) for value in payload[identity_key].tolist()
        )
        raw = np.asarray(payload["raw_tokens"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
    return identities, raw, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--model-train-hlt", action="append", default=[], required=True
    )
    parser.add_argument(
        "--model-train-tree", action="append", default=[], required=True
    )
    parser.add_argument("--relation-normalizer", required=True, type=Path)
    parser.add_argument("--teacher-lock", type=Path)
    args = parser.parse_args(argv)

    root = args.campaign_root
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    source = campaign["source"]
    relation = load_hashed_json(args.relation_normalizer)
    if relation.get("source") != source:
        raise ValueError("normalization relation parent source differs")
    views = _mapping(args.model_train_hlt, name="--model-train-hlt", paths=True)
    trees = _mapping(
        args.model_train_tree, name="--model-train-tree", paths=True
    )
    canonical = _cache(root / "targets" / "canonical" / "model_train")
    if canonical.manifest.get("source") != source:
        raise ValueError("canonical normalization cache source differs")
    registry = load_hashed_json(
        root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    registry_rows = {row["target_id"]: row for row in registry["targets"]}
    physical_kinds = {
        target_id: tuple(
            component["component_kind"]
            for component in registry_rows[target_id]["components"]
        )
        for target_id in canonical.manifest["persisted_target_ids"]
    }
    physical = fit_target_normalizer(
        canonical,
        fitting_population="target_500k",
        source=source,
        component_kinds=physical_kinds,
        normalization_role="target",
    )
    pair_normalizers = [
        fit_pair_normalizer_from_views(
            target_id=target_id,
            view_paths_by_replica=views,
            relation_normalizer=relation,
            tree_paths_by_replica=(
                trees if target_id == "T_HLT_REGION_PAIR_8" else None
            ),
            fitting_population="target_500k",
            split="model_train",
            source=source,
        )
        for target_id in ("T_HLT_TRACK_PAIR_13", "T_HLT_REGION_PAIR_8")
    ]
    teacher_base = _cache(
        root / "teachers" / "outputs" / "model_train" / "O_BASE"
    )
    teacher_full = _cache(
        root / "teachers" / "outputs" / "model_train" / "O_FULLREL"
    )
    teacher_normalizers = [
        fit_target_normalizer(
            cache,
            fitting_population="target_500k",
            source=source,
            normalization_role="target",
        )
        for cache in (teacher_base, teacher_full)
    ]
    target_normalizer = merge_target_normalizers(
        [physical, *pair_normalizers, *teacher_normalizers],
        fitting_population="target_500k",
        normalization_role="target",
        source=source,
        parent_hashes={
            "canonical_cache": canonical.manifest["content_hash"],
            "teacher_o_base_cache": teacher_base.manifest["content_hash"],
            "teacher_o_fullrel_cache": teacher_full.manifest["content_hash"],
            "relation_normalizer": relation["content_hash"],
        },
    )
    target_dir = root / "normalization" / "target_500k"
    write_immutable_json(
        target_dir / "normalizer_manifest.json", target_normalizer
    )
    hetero = build_heteroscedastic_metadata(
        target_normalizer, source=source
    )
    write_immutable_json(
        target_dir / "heteroscedastic_metadata.json", hetero
    )

    lock_path = args.teacher_lock or root / "teachers" / "teacher_lock.json"
    lock = load_hashed_json(lock_path, expected_contract=TEACHER_LOCK_CONTRACT)
    latent = teacher_base.values["T_OFFLINE_POOLED_LATENT"]
    whitening = fit_latent_whitening(
        latent,
        teacher_lock_sha256=lock["content_hash"],
        fitting_population="target_500k",
        source=source,
    )
    write_immutable_json(target_dir / "latent_whitening.json", whitening)

    residual_normalizers = []
    residual_caches = {}
    conditional_hashes = {}
    residual_dir = root / "normalization" / "residual_500k"
    floors = relation["track_uncertainty_floors"]
    for replica in range(4):
        cache = _cache(
            root
            / "targets"
            / "residuals"
            / "model_train"
            / f"replica_{replica}"
        )
        residual_caches[replica] = cache
        residual_normalizers.append(
            fit_target_normalizer(
                cache,
                fitting_population="target_500k",
                source=source,
                normalization_role="residual",
            )
        )
        identities, raw, mask = _input(views[replica])
        if identities != cache.identities:
            raise ValueError(
                "conditional-residual HLT/cache identity order differs"
            )
        context = build_hlt_conditional_context(
            raw,
            mask,
            d0_uncertainty_floor=float(floors["d0"]["floor"]),
            dz_uncertainty_floor=float(floors["dz"]["floor"]),
            sentinel_policy=relation["track_sentinel_policy"],
        )
        for target_id in sorted(cache.values):
            artifact = fit_conditional_residual(
                cache.values[target_id],
                cache.masks[target_id],
                context,
                target_id=target_id,
                train_cache_hashes={
                    "residual_cache": cache.manifest["content_hash"],
                    "hlt_view": hashlib.sha256(
                        views[replica].read_bytes()
                    ).hexdigest(),
                },
                source=source,
                fitting_population="target_500k",
            )
            write_immutable_json(
                residual_dir
                / "conditional"
                / f"replica_{replica}"
                / f"{target_id}.json",
                artifact,
            )
            conditional_hashes[
                f"replica_{replica}::{target_id}"
            ] = artifact["content_hash"]
    residual_normalizer = merge_target_normalizers(
        residual_normalizers,
        fitting_population="target_500k",
        normalization_role="residual",
        source=source,
        parent_hashes={
            f"residual_cache_replica_{replica}": cache.manifest[
                "content_hash"
            ]
            for replica, cache in sorted(residual_caches.items())
        },
    )
    write_immutable_json(
        residual_dir / "normalizer_manifest.json", residual_normalizer
    )
    conditional = with_content_hash(
        {
            "contract": "hosd_conditional_residual_wave_v1",
            "schema_version": 1,
            "source": dict(source),
            "fitting_population": "target_500k",
            "fit_split": "model_train",
            "artifact_hashes": dict(sorted(conditional_hashes.items())),
            "artifact_count": len(conditional_hashes),
            "replicas": [0, 1, 2, 3],
            "identity_values_stored": False,
            "performance_results_read": False,
        }
    )
    write_immutable_json(
        residual_dir / "conditional_completion.json", conditional
    )
    native_hashes = {}
    for split in ("model_train", "val_stop", "val_design"):
        replicas = range(4) if split == "model_train" else (0,)
        for replica in replicas:
            artifact = materialize_native_relation_target(
                target_cache_root=(
                    root
                    / "targets"
                    / "hlt_analogues"
                    / split
                    / f"replica_{replica}"
                ),
                output_path=(
                    root
                    / "targets"
                    / "native_relations"
                    / split
                    / f"replica_{replica}.npz"
                ),
                campaign_spec_sha256=campaign["content_hash"],
                source=source,
            )
            native_hashes[f"{split}::replica_{replica}"] = artifact[
                "content_hash"
            ]
    native_completion = with_content_hash(
        {
            "contract": "hosd_native_relation_target_wave_v1",
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": campaign["content_hash"],
            "artifact_hashes": dict(sorted(native_hashes.items())),
            "splits": ["model_train", "val_stop", "val_design"],
            "replica_policy": {
                "model_train": [0, 1, 2, 3],
                "val_stop": [0],
                "val_design": [0],
            },
            "artifact_count": len(native_hashes),
            "same_view_replica_binding": True,
            "offline_information_consumed": False,
        }
    )
    write_immutable_json(
        root / "targets" / "native_relations" / "completion.json",
        native_completion,
    )
    completion = with_content_hash(
        {
            "contract": TARGET_NORMALIZATION_WAVE_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": campaign["content_hash"],
            "target_normalizer_sha256": target_normalizer["content_hash"],
            "residual_normalizer_sha256": residual_normalizer["content_hash"],
            "heteroscedastic_metadata_sha256": hetero["content_hash"],
            "latent_whitening_sha256": whitening["content_hash"],
            "conditional_completion_sha256": conditional["content_hash"],
            "native_relation_completion_sha256": native_completion[
                "content_hash"
            ],
            "pair_normalizer_hashes": {
                row["targets"][0]["target_id"]: row["content_hash"]
                for row in pair_normalizers
            },
            "fit_split": "model_train",
            "all_train_derived_statistics_complete": True,
            "performance_results_read": False,
        }
    )
    write_immutable_json(
        root / "normalization" / "target_500k" / "completion.json",
        completion,
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
