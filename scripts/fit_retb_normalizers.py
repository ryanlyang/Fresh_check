#!/usr/bin/env python3
"""Fit numerical offline/shared-HLT relation and REGION normalizers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (  # noqa: E402
    bind_fitted_normalizer,
    bind_fitted_region_normalizer,
    build_stage_a_normalizer_bundle,
    load_authenticated_tree_selection,
    stage_a_normalizer_population_registry_path,
    validate_stage_a_contract_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    fit_region_normalization,
    fit_relation_normalization,
    select_normalization_jet_indices,
)
from teacher_logit_reco.relational_part.normalization import (  # noqa: E402
    NORMALIZATION_JET_LIMIT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_a_contracts(root: Path) -> dict[str, dict[str, Any]]:
    names = {
        "stage_a_contract_bundle": (
            root / "registry" / "retb_stage_a_contract_bundle.json"
        ),
        "relation_family_registry": (
            root / "registry" / "inherited_relation_family_registry.json"
        ),
        "normalization_contract": (
            root / "inputs" / "inherited_normalization_contract.json"
        ),
        "angular_tree_resource": (
            root / "inputs" / "inherited_angular_tree_resource.json"
        ),
        "inherited_raw_input_schema": (
            root / "inputs" / "inherited_relational_raw_input_schema.json"
        ),
        "hlt_v3_profile": root / "inputs" / "hlt_v3_profile.json",
        "normalizer_population_registry": (
            stage_a_normalizer_population_registry_path(root)
        ),
    }
    return {name: load_hashed_json(path) for name, path in names.items()}


def _offline_selected(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    logical_role: str,
):
    cache = root / "inputs" / "offline" / logical_role
    metadata = load_hashed_json(
        cache / "offline_input_manifest.json",
        expected_contract="retb_offline_input_cache_v1",
    )
    if (
        metadata["campaign_spec_sha256"] != campaign["content_hash"]
        or metadata["identity_manifest_sha256"]
        != campaign["parent_artifact_hashes"]["split_manifest"]
        or metadata["raw_input_schema_sha256"]
        != campaign["parent_artifact_hashes"]["raw_input_schema"]
    ):
        raise ValueError("offline normalizer source lineage differs")
    npz_path = cache / metadata["npz_filename"]
    if _sha256(npz_path) != metadata["npz_sha256"]:
        raise ValueError("offline normalizer source bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        identities = [str(value) for value in payload["identities"].tolist()]
        selected = select_normalization_jet_indices(identities)
        tokens = np.asarray(payload["tokens"][selected], dtype=np.float32)
        mask = np.asarray(payload["mask"][selected], dtype=bool)
    selected_ids = [identities[int(index)] for index in selected]
    return tokens, mask, selected_ids, metadata


def _hlt_selected(
    root: Path,
    *,
    campaign: Mapping[str, Any],
    profile_sha256: str,
    logical_role: str,
):
    cache_roots = [
        (
            root
            / "inputs"
            / "hlt_v3"
            / logical_role
            / f"replica_{replica}"
            / "R_MULTI"
            / "D_NOMINAL"
        )
        for replica in range(4)
    ]
    arrays_by_replica = []
    metadata_by_replica = []
    base_identities: list[str] | None = None
    selected: np.ndarray | None = None
    group_limit = max(1, NORMALIZATION_JET_LIMIT // 4)
    for replica, cache in enumerate(cache_roots):
        arrays, metadata = load_hlt_v3_cache(
            cache,
            expected_profile_contract_sha256=profile_sha256,
            expected_logical_role=logical_role,
            expected_replica_id=replica,
            expected_realization_policy="R_MULTI",
        )
        identities = [
            str(value) for value in arrays["identities"].tolist()
        ]
        if (
            metadata["identity_manifest_sha256"]
            != campaign["parent_artifact_hashes"]["split_manifest"]
        ):
            raise ValueError("shared-HLT normalizer identity lineage differs")
        if base_identities is None:
            base_identities = identities
            selected = select_normalization_jet_indices(
                identities, limit=min(group_limit, len(identities))
            )
        elif identities != base_identities:
            raise ValueError("shared-HLT normalizer replica identities differ")
        arrays_by_replica.append(
            (
                np.asarray(arrays["tokens"][selected], dtype=np.float32),
                np.asarray(arrays["mask"][selected], dtype=bool),
            )
        )
        metadata_by_replica.append(metadata)
    if base_identities is None or selected is None:
        raise AssertionError("shared-HLT normalizer has no replicas")
    selected_ids = [base_identities[int(index)] for index in selected]
    # [identity, replica, particle, field] makes replica weighting explicit.
    token_stack = np.stack([row[0] for row in arrays_by_replica], axis=1)
    mask_stack = np.stack([row[1] for row in arrays_by_replica], axis=1)
    tokens = token_stack.reshape(
        len(selected_ids) * 4, *token_stack.shape[2:]
    )
    mask = mask_stack.reshape(len(selected_ids) * 4, *mask_stack.shape[2:])
    identities = [
        f"{identity}@retb_replica_{replica}"
        for identity in selected_ids
        for replica in range(4)
    ]
    return tokens, mask, identities, selected_ids, metadata_by_replica


def _selected_trees(
    tree_root: Path,
    identities: Sequence[str],
) -> tuple[list[Mapping[str, Any]], str]:
    trees, manifest = load_authenticated_tree_selection(
        tree_root, identities
    )
    return trees, manifest["content_hash"]


def _fit_relation(
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: Sequence[str],
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    identity_manifest_sha: str,
    view_hashes: Sequence[str],
    logical_domain: str,
    campaign: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    recipe = contracts["normalizer_population_registry"]["recipes"][
        logical_domain
    ]
    inherited = fit_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=contracts["normalization_contract"],
        relation_registry=contracts["relation_family_registry"],
        raw_input_schema=contracts["inherited_raw_input_schema"],
        hlt_binding_sha256=contracts["stage_a_contract_bundle"][
            "content_hash"
        ],
        source_manifest_sha256=identity_manifest_sha,
        hlt_model_train_content_sha256=canonical_sha256(
            {"view_content_sha256s": list(view_hashes)}
        ),
    )
    return bind_fitted_normalizer(
        inherited,
        logical_domain=logical_domain,
        population_recipe=recipe,
        identity_manifest_sha256=identity_manifest_sha,
        view_content_sha256s=view_hashes,
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )


def _publish_or_validate(path: Path, artifact: Mapping[str, Any]) -> dict:
    return write_immutable_json(path, artifact)


def _progress(domain: str):
    def emit(row: Mapping[str, Any]) -> None:
        print(
            json.dumps(
                {"normalizer_domain": domain, **dict(row)},
                sort_keys=True,
            ),
            flush=True,
        )

    return emit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--inherited-estimator-contract-sha256",
        help="Compatibility assertion; must equal the published inherited contract.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional immutable normalization root. Stage M uses a "
            "seed-scoped root so concurrent seed rows cannot race."
        ),
    )
    parser.add_argument(
        "--population",
        choices=("500k", "scale"),
        default="500k",
        help=(
            "500k fits the Stage-A model_train statistics; scale fits the "
            "locked Stage-M scale_train statistics."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    result: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "fit_domains": (
            ["offline_500k", "shared_hlt_500k"]
            if args.population == "500k"
            else ["offline_scale", "shared_hlt_scale"]
        ),
        "fit_execution": "inherited_deterministic_numeric_estimator",
    }
    if args.dry_run and not (
        args.campaign_root / "registry" / "retb_stage_a_contract_bundle.json"
    ).is_file():
        result["requires_stage_a_contracts"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    contracts = _stage_a_contracts(args.campaign_root)
    validate_stage_a_contract_bundle(contracts, campaign_spec=campaign)
    if (
        args.inherited_estimator_contract_sha256 is not None
        and args.inherited_estimator_contract_sha256
        != contracts["normalization_contract"]["content_hash"]
    ):
        raise ValueError("inherited estimator compatibility assertion differs")
    if args.dry_run:
        result["stage_a_contract_bundle_sha256"] = contracts[
            "stage_a_contract_bundle"
        ]["content_hash"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    snapshot = source_snapshot(REPO_ROOT)
    training_role = (
        "model_train" if args.population == "500k" else "scale_train"
    )
    offline_domain = (
        "offline_500k" if args.population == "500k" else "offline_scale"
    )
    hlt_domain = (
        "shared_hlt_500k"
        if args.population == "500k"
        else "shared_hlt_scale"
    )
    offline_tokens, offline_mask, offline_ids, offline_meta = (
        _offline_selected(
            args.campaign_root,
            campaign,
            logical_role=training_role,
        )
    )
    offline_relation = _fit_relation(
        offline_tokens,
        offline_mask,
        offline_ids,
        contracts=contracts,
        identity_manifest_sha=contracts["normalizer_population_registry"][
            "recipes"
        ][offline_domain]["identity_manifest_sha256"],
        view_hashes=[offline_meta["npz_sha256"]],
        logical_domain=offline_domain,
        campaign=campaign,
        snapshot=snapshot,
    )
    offline_trees, offline_tree_manifest = _selected_trees(
        args.campaign_root
        / "inputs"
        / "region_tree"
        / "offline"
        / f"{training_role}_exclusive_ca_v1",
        offline_ids,
    )
    offline_region_raw = fit_region_normalization(
        offline_tokens,
        offline_mask,
        offline_ids,
        offline_trees,
        relation_normalization_artifact=offline_relation,
        angular_tree_resource_sha256=contracts["angular_tree_resource"][
            "content_hash"
        ],
        progress_callback=_progress("offline_500k"),
    )
    offline_region = bind_fitted_region_normalizer(
        offline_region_raw,
        relation_normalizer=offline_relation,
        logical_domain=offline_domain,
        population_recipe=contracts["normalizer_population_registry"][
            "recipes"
        ][offline_domain],
        tree_manifest_sha256s=[offline_tree_manifest],
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )

    (
        hlt_tokens,
        hlt_mask,
        hlt_ids,
        selected_base_ids,
        hlt_metadata,
    ) = _hlt_selected(
        args.campaign_root,
        campaign=campaign,
        profile_sha256=contracts["hlt_v3_profile"]["content_hash"],
        logical_role=training_role,
    )
    hlt_view_hashes = [
        row["array_content_sha256"] for row in hlt_metadata
    ]
    hlt_relation = _fit_relation(
        hlt_tokens,
        hlt_mask,
        hlt_ids,
        contracts=contracts,
        identity_manifest_sha=contracts["normalizer_population_registry"][
            "recipes"
        ][hlt_domain]["identity_manifest_sha256"],
        view_hashes=hlt_view_hashes,
        logical_domain=hlt_domain,
        campaign=campaign,
        snapshot=snapshot,
    )
    trees_by_replica = []
    tree_manifest_hashes = []
    for replica in range(4):
        trees, manifest_sha = _selected_trees(
            args.campaign_root
            / "inputs"
            / "region_tree"
            / "hlt"
            / f"{training_role}_r{replica}_exclusive_ca_v1",
            selected_base_ids,
        )
        trees_by_replica.append(trees)
        tree_manifest_hashes.append(manifest_sha)
    hlt_trees = [
        trees_by_replica[replica][identity_index]
        for identity_index in range(len(selected_base_ids))
        for replica in range(4)
    ]
    hlt_region_raw = fit_region_normalization(
        hlt_tokens,
        hlt_mask,
        hlt_ids,
        hlt_trees,
        relation_normalization_artifact=hlt_relation,
        angular_tree_resource_sha256=contracts["angular_tree_resource"][
            "content_hash"
        ],
        progress_callback=_progress("shared_hlt_500k"),
    )
    hlt_region = bind_fitted_region_normalizer(
        hlt_region_raw,
        relation_normalizer=hlt_relation,
        logical_domain=hlt_domain,
        population_recipe=contracts["normalizer_population_registry"][
            "recipes"
        ][hlt_domain],
        tree_manifest_sha256s=tree_manifest_hashes,
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )
    output_root = (
        args.campaign_root / "inputs" / "normalization"
        if args.output_root is None
        else args.output_root
    )
    artifacts = {
        f"{offline_domain}_relation": offline_relation,
        f"{offline_domain}_region": offline_region,
        f"{hlt_domain}_relation": hlt_relation,
        f"{hlt_domain}_region": hlt_region,
    }
    offline_directory = (
        "offline_500k" if args.population == "500k" else "offline_scale"
    )
    hlt_directory = (
        "hlt_shared_500k"
        if args.population == "500k"
        else "hlt_shared_scale"
    )
    paths = {
        f"{offline_domain}_relation": (
            output_root / offline_directory / "relation.json"
        ),
        f"{offline_domain}_region": (
            output_root / offline_directory / "region.json"
        ),
        f"{hlt_domain}_relation": (
            output_root / hlt_directory / "relation.json"
        ),
        f"{hlt_domain}_region": (
            output_root / hlt_directory / "region.json"
        ),
    }
    publications = {
        name: _publish_or_validate(paths[name], artifact)
        for name, artifact in artifacts.items()
    }
    if args.population == "500k":
        bundle = build_stage_a_normalizer_bundle(
            campaign_spec_sha256=campaign["content_hash"],
            stage_a_contract_bundle_sha256=contracts[
                "stage_a_contract_bundle"
            ]["content_hash"],
            population_registry=contracts[
                "normalizer_population_registry"
            ],
            offline_relation=offline_relation,
            offline_region=offline_region,
            shared_hlt_relation=hlt_relation,
            shared_hlt_region=hlt_region,
            source_snapshot=snapshot,
        )
        bundle_path = args.output or (
            output_root / "stage_a_normalizer_bundle.json"
        )
    else:
        bundle = bind_source(
            with_content_hash(
                {
                    "contract": "retb_scale_normalizer_bundle_v1",
                    "schema_version": 1,
                    "training_population": "scale_train",
                    "campaign_spec_sha256": campaign["content_hash"],
                    "stage_a_contract_bundle_sha256": contracts[
                        "stage_a_contract_bundle"
                    ]["content_hash"],
                    "population_registry_sha256": contracts[
                        "normalizer_population_registry"
                    ]["content_hash"],
                    "scale_train_manifest_sha256": campaign[
                        "parent_artifact_hashes"
                    ]["scale_train_manifest"],
                    "offline_cache_identity_parent_sha256": offline_meta[
                        "identity_manifest_sha256"
                    ],
                    "artifact_hashes": {
                        "offline_relation": offline_relation[
                            "content_hash"
                        ],
                        "offline_region": offline_region["content_hash"],
                        "shared_hlt_relation": hlt_relation[
                            "content_hash"
                        ],
                        "shared_hlt_region": hlt_region["content_hash"],
                    },
                    "replica_ids": [0, 1, 2, 3],
                    "labels_consumed": False,
                    "model_train_statistics_reused": False,
                }
            ),
            source_snapshot=snapshot,
        )
        bundle_path = args.output or (
            output_root / "scale_normalizer_bundle.json"
        )
    publications["normalizer_bundle"] = write_immutable_json(
        bundle_path, bundle
    )
    result.update(
        {
            "normalizer_bundle_sha256": bundle["content_hash"],
            "artifact_hashes": bundle["artifact_hashes"],
            "publications": publications,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
