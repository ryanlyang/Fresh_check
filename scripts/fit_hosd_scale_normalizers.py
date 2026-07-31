#!/usr/bin/env python3
"""Refit offline and shared-HLT relation/REGION statistics on scale_train."""

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

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    INPUT_VIEW_MANIFEST_CONTRACT,
    SCALE_INPUT_COMPLETION_CONTRACT,
    SCALE_NORMALIZER_COMPLETION_CONTRACT,
    SCALE_TREE_WAVE_COMPLETION_CONTRACT,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.normalizer_lineage import (  # noqa: E402
    build_normalizer_population_recipe,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (  # noqa: E402
    bind_fitted_normalizer,
    bind_fitted_region_normalizer,
    load_authenticated_tree_selection,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    fit_region_normalization,
    fit_relation_normalization,
    select_normalization_jet_indices,
)
from teacher_logit_reco.relational_part.normalization import (  # noqa: E402
    NORMALIZATION_JET_LIMIT,
)


def _load_view(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    manifest = load_hashed_json(
        path.with_suffix(path.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    with np.load(path, allow_pickle=False) as payload:
        if not {"identity", "raw_tokens", "mask"}.issubset(payload.files):
            raise ValueError("scale-normalizer view fields differ")
        identities = [str(value) for value in payload["identity"].tolist()]
        tokens = np.asarray(payload["raw_tokens"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
    if (
        len(identities) != int(manifest["identity_count"])
        or tokens.shape != (len(identities), 128, 14)
        or mask.shape != (len(identities), 128)
    ):
        raise ValueError("scale-normalizer view population differs")
    return tokens, mask, identities, manifest


def _fit_relation(
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: Sequence[str],
    *,
    normalization_contract: Mapping[str, Any],
    relation_registry: Mapping[str, Any],
    raw_input_schema: Mapping[str, Any],
    identity_manifest_sha256: str,
    population_identity_count: int,
    view_content_sha256s: Sequence[str],
    logical_domain: str,
    campaign: Mapping[str, Any],
    hlt_profile: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> tuple[dict, dict]:
    recipe = build_normalizer_population_recipe(
        logical_domain=logical_domain,
        identity_manifest_sha256=identity_manifest_sha256,
        identity_count=int(population_identity_count),
        raw_input_schema_sha256=raw_input_schema["content_hash"],
        hlt_v3_profile_sha256=(
            None
            if logical_domain == "offline_scale"
            else hlt_profile["content_hash"]
        ),
        inherited_estimator_contract_sha256=normalization_contract[
            "content_hash"
        ],
    )
    inherited = fit_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=normalization_contract,
        relation_registry=relation_registry,
        raw_input_schema=raw_input_schema,
        hlt_binding_sha256=campaign["content_hash"],
        source_manifest_sha256=identity_manifest_sha256,
        hlt_model_train_content_sha256=canonical_sha256(
            {"view_content_sha256s": list(view_content_sha256s)}
        ),
    )
    artifact = bind_fitted_normalizer(
        inherited,
        logical_domain=logical_domain,
        population_recipe=recipe,
        identity_manifest_sha256=identity_manifest_sha256,
        view_content_sha256s=view_content_sha256s,
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )
    return artifact, recipe


def _progress(domain: str):
    def emit(payload: Mapping[str, Any]) -> None:
        print(
            json.dumps(
                {"normalizer_domain": domain, **dict(payload)},
                sort_keys=True,
            ),
            flush=True,
        )

    return emit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--normalization-contract", required=True, type=Path)
    parser.add_argument("--relation-registry", required=True, type=Path)
    parser.add_argument("--raw-input-schema", required=True, type=Path)
    parser.add_argument("--hlt-profile", required=True, type=Path)
    parser.add_argument("--tree-resource", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    input_completion = load_hashed_json(
        root / "scale_up" / "inputs" / "completion.json",
        expected_contract=SCALE_INPUT_COMPLETION_CONTRACT,
    )
    tree_completion = load_hashed_json(
        root / "scale_up" / "trees" / "completion.json",
        expected_contract=SCALE_TREE_WAVE_COMPLETION_CONTRACT,
    )
    normalization_contract = load_hashed_json(args.normalization_contract)
    relation_registry = load_hashed_json(args.relation_registry)
    raw_input_schema = load_hashed_json(args.raw_input_schema)
    hlt_profile = load_hashed_json(args.hlt_profile)
    tree_resource = load_hashed_json(args.tree_resource)
    for name, artifact in (
        ("input completion", input_completion),
        ("tree completion", tree_completion),
        ("normalization contract", normalization_contract),
        ("relation registry", relation_registry),
        ("raw input schema", raw_input_schema),
        ("HLT profile", hlt_profile),
        ("tree resource", tree_resource),
    ):
        if (
            artifact.get("source") is not None
            and artifact.get("source") != campaign["source"]
        ):
            raise ValueError(f"Stage-J {name} source lineage differs")
    if (
        tree_completion.get("scale_input_completion_sha256")
        != input_completion["content_hash"]
    ):
        raise ValueError("Stage-J tree/input completion lineage differs")
    scale_manifest_sha = require_sha256(
        input_completion["scale_train_manifest_sha256"],
        name="scale_train_manifest_sha256",
    )
    scale_identity_count = int(input_completion["rows"][0].get("identity_count", 0))
    if scale_identity_count <= 0:
        # Input completion v1 records counts in each authenticated view
        # manifest rather than duplicating them in the row.
        scale_identity_count = int(
            load_hashed_json(
                Path(input_completion["rows"][0]["npz_path"]).with_suffix(
                    ".npz.json"
                ),
                expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
            )["identity_count"]
        )
    if any(
        int(row.get("identity_count", scale_identity_count))
        != scale_identity_count
        for row in input_completion["rows"]
    ):
        raise ValueError("Stage-J scale view identity counts differ")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "scale_up" / "normalization"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "fit_split": "scale_train",
                    "logical_domains": [
                        "offline_scale",
                        "shared_hlt_scale",
                    ],
                    "output_root": str(output_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    snapshot = source_snapshot(REPO_ROOT)
    offline_path = (
        root / "scale_up" / "inputs" / "offline" / "scale_train.npz"
    )
    offline_tokens, offline_mask, offline_ids, offline_view = _load_view(
        offline_path
    )
    selected = select_normalization_jet_indices(offline_ids)
    offline_tokens = offline_tokens[selected]
    offline_mask = offline_mask[selected]
    offline_selected_ids = [offline_ids[int(index)] for index in selected]
    offline_relation, offline_recipe = _fit_relation(
        offline_tokens,
        offline_mask,
        offline_selected_ids,
        normalization_contract=normalization_contract,
        relation_registry=relation_registry,
        raw_input_schema=raw_input_schema,
        identity_manifest_sha256=scale_manifest_sha,
        population_identity_count=scale_identity_count,
        view_content_sha256s=[offline_view["npz_sha256"]],
        logical_domain="offline_scale",
        campaign=campaign,
        hlt_profile=hlt_profile,
        snapshot=snapshot,
    )
    offline_trees, offline_tree_manifest = load_authenticated_tree_selection(
        root
        / "scale_up"
        / "trees"
        / "offline"
        / "scale_train_exclusive_ca_v1",
        offline_selected_ids,
    )
    offline_region_raw = fit_region_normalization(
        offline_tokens,
        offline_mask,
        offline_selected_ids,
        offline_trees,
        relation_normalization_artifact=offline_relation,
        angular_tree_resource_sha256=tree_resource["content_hash"],
        progress_callback=_progress("offline_scale"),
    )
    offline_region = bind_fitted_region_normalizer(
        offline_region_raw,
        relation_normalizer=offline_relation,
        logical_domain="offline_scale",
        population_recipe=offline_recipe,
        tree_manifest_sha256s=[offline_tree_manifest["content_hash"]],
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )

    group_limit = max(1, NORMALIZATION_JET_LIMIT // 4)
    base_ids: list[str] | None = None
    hlt_token_rows = []
    hlt_mask_rows = []
    hlt_view_hashes = []
    hlt_trees_by_replica = []
    hlt_tree_hashes = []
    selected_base_ids: list[str] | None = None
    for replica in range(4):
        path = (
            root / "scale_up" / "inputs" / "hlt" / f"replica_{replica}.npz"
        )
        tokens, mask, identities, view = _load_view(path)
        if base_ids is None:
            base_ids = identities
            selected_indices = select_normalization_jet_indices(
                identities, limit=min(group_limit, len(identities))
            )
            selected_base_ids = [
                identities[int(index)] for index in selected_indices
            ]
        elif identities != base_ids:
            raise ValueError("Stage-J HLT replica identities differ")
        hlt_token_rows.append(tokens[selected_indices])
        hlt_mask_rows.append(mask[selected_indices])
        hlt_view_hashes.append(view["npz_sha256"])
        trees, tree_manifest = load_authenticated_tree_selection(
            root / "scale_up" / "trees" / "hlt" / f"replica_{replica}",
            selected_base_ids,
        )
        hlt_trees_by_replica.append(trees)
        hlt_tree_hashes.append(tree_manifest["content_hash"])
    if base_ids is None or selected_base_ids is None:
        raise AssertionError("Stage-J shared-HLT population is empty")
    token_stack = np.stack(hlt_token_rows, axis=1)
    mask_stack = np.stack(hlt_mask_rows, axis=1)
    hlt_tokens = token_stack.reshape(
        len(selected_base_ids) * 4, *token_stack.shape[2:]
    )
    hlt_mask = mask_stack.reshape(
        len(selected_base_ids) * 4, *mask_stack.shape[2:]
    )
    hlt_ids = [
        f"{identity}@hosd_replica_{replica}"
        for identity in selected_base_ids
        for replica in range(4)
    ]
    hlt_trees = [
        hlt_trees_by_replica[replica][identity_index]
        for identity_index in range(len(selected_base_ids))
        for replica in range(4)
    ]
    hlt_relation, hlt_recipe = _fit_relation(
        hlt_tokens,
        hlt_mask,
        hlt_ids,
        normalization_contract=normalization_contract,
        relation_registry=relation_registry,
        raw_input_schema=raw_input_schema,
        identity_manifest_sha256=scale_manifest_sha,
        population_identity_count=scale_identity_count,
        view_content_sha256s=hlt_view_hashes,
        logical_domain="shared_hlt_scale",
        campaign=campaign,
        hlt_profile=hlt_profile,
        snapshot=snapshot,
    )
    hlt_region_raw = fit_region_normalization(
        hlt_tokens,
        hlt_mask,
        hlt_ids,
        hlt_trees,
        relation_normalization_artifact=hlt_relation,
        angular_tree_resource_sha256=tree_resource["content_hash"],
        progress_callback=_progress("shared_hlt_scale"),
    )
    hlt_region = bind_fitted_region_normalizer(
        hlt_region_raw,
        relation_normalizer=hlt_relation,
        logical_domain="shared_hlt_scale",
        population_recipe=hlt_recipe,
        tree_manifest_sha256s=hlt_tree_hashes,
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=snapshot,
    )

    artifacts = {
        "offline_relation": offline_relation,
        "offline_region": offline_region,
        "shared_hlt_relation": hlt_relation,
        "shared_hlt_region": hlt_region,
    }
    paths = {
        "offline_relation": output_root / "offline_scale" / "relation.json",
        "offline_region": output_root / "offline_scale" / "region.json",
        "shared_hlt_relation": (
            output_root / "shared_hlt_scale" / "relation.json"
        ),
        "shared_hlt_region": (
            output_root / "shared_hlt_scale" / "region.json"
        ),
    }
    publications = {
        name: write_immutable_json(paths[name], artifacts[name])
        for name in sorted(artifacts)
    }
    normalizer_hashes = {
        "O_BASE": {
            "relation_normalization": offline_relation["content_hash"],
        },
        "O_FULLREL": {
            "relation_normalization": offline_relation["content_hash"],
            "region_normalization": offline_region["content_hash"],
        },
    }
    normalizer_hashes_path = output_root / "teacher_normalizer_hashes.json"
    normalizer_hashes_bytes = (
        json.dumps(normalizer_hashes, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    normalizer_hashes_publication = write_immutable_bytes(
        normalizer_hashes_path, normalizer_hashes_bytes
    )
    normalizer_hashes_file_sha256 = hashlib.sha256(
        normalizer_hashes_bytes
    ).hexdigest()
    completion = with_content_hash(
        {
            "contract": SCALE_NORMALIZER_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "scale_input_completion_sha256": input_completion["content_hash"],
            "scale_tree_completion_sha256": tree_completion["content_hash"],
            "scale_train_manifest_sha256": scale_manifest_sha,
            "artifact_paths": {
                name: str(path.resolve()) for name, path in sorted(paths.items())
            },
            "artifact_hashes": {
                name: artifact["content_hash"]
                for name, artifact in sorted(artifacts.items())
            },
            "teacher_normalizer_hashes_path": str(
                normalizer_hashes_path.resolve()
            ),
            "teacher_normalizer_hashes_sha256": normalizer_hashes_file_sha256,
            "fit_split": "scale_train",
            "validation_or_test_statistics_used": False,
            "model_train_statistics_reused": False,
            "offline_and_hlt_statistics_separate": True,
        }
    )
    publication = write_immutable_json(
        output_root / "completion.json", completion
    )
    print(
        json.dumps(
            {
                "completion_sha256": completion["content_hash"],
                "publications": publications,
                "teacher_normalizer_hashes_publication": (
                    normalizer_hashes_publication
                ),
                "completion_publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
