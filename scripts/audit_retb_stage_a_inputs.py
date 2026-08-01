#!/usr/bin/env python3
"""Audit complete Stage-A RETB input, tree, and normalizer lineage."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_audit import (  # noqa: E402
    assert_layout_determinism,
    assert_train_scale_shared_identity,
    audit_strength_monotonicity,
    build_hlt_v3_degradation_audit,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (  # noqa: E402
    build_hlt_v3_view,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (  # noqa: E402
    STAGE_A_HLT_TREE_VIEWS,
    STAGE_A_OFFLINE_TREE_ROLES,
    build_stage_a_input_audit,
    identity_newline_sha256,
    load_authenticated_tree_selection,
    padding_is_exact_zero,
    validate_stage_a_normalizer_bundle,
    validate_stage_a_tree_index,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_abc import (  # noqa: E402
    STREAMED_HLT_VIEWS,
    STREAMED_HLT_NORMALIZER_JETS_PER_REPLICA,
    STREAMED_OFFLINE_ROLES,
    build_streamed_input_audit,
    validate_streamed_abc_execution_profile,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    RelationalPairBuilder,
    build_standard_four_pair_features,
    select_normalization_jet_indices,
    validate_region_normalization,
    validate_relation_normalization_artifact,
)
from jetclass_fresh.jetclass_data import (  # noqa: E402
    JetIdentity,
    load_split_manifest,
)
from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)


OFFLINE_ROLES = (
    "model_train",
    "val_stop",
    "val_design",
    "stack_val",
    "final_test",
    "scale_train",
)
HLT_VIEWS = (
    *(("model_train", replica, "R_MULTI") for replica in range(4)),
    *(("scale_train", replica, "R_MULTI") for replica in range(4)),
    *((
        role,
        0,
        "R_FIXED",
    ) for role in ("val_stop", "val_design", "stack_val", "final_test")),
)
RELATION_AUDIT_JET_LIMIT = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_parent_for_role(campaign: dict, role: str) -> str:
    parents = campaign["parent_artifact_hashes"]
    if role in {"model_train", "stack_val", "final_test"}:
        return str(parents["split_manifest"])
    if role in {"val_stop", "val_design"}:
        return str(parents["validation_partition_manifest"])
    return str(parents["scale_train_manifest"])


def _audit_arrays(
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: list[str],
) -> dict:
    values = np.asarray(tokens)
    valid = np.asarray(mask)
    if (
        values.ndim != 3
        or tuple(values.shape[1:]) != (128, 14)
        or valid.shape != values.shape[:2]
    ):
        raise ValueError("Stage-A raw cache tensor shape differs")
    return {
        "event_count": len(identities),
        "particle_capacity": int(values.shape[1]),
        "raw_particle_field_count": int(values.shape[2]),
        "tokens_dtype": str(values.dtype),
        "mask_dtype": str(valid.dtype),
        "finite_valid_tokens": bool(np.isfinite(values[valid]).all()),
        "padding_zero_exact": padding_is_exact_zero(values, valid),
        "identities_unique": len(identities) == len(set(identities)),
        "identity_order_sha256": identity_newline_sha256(identities),
        "valid_particle_count": int(valid.sum()),
        "all_empty_jet_count": int(np.sum(~valid.any(axis=1))),
    }


def _normalizer_artifacts(root: Path) -> dict:
    paths = {
        "offline_500k_relation": (
            root / "inputs" / "normalization" / "offline_500k" / "relation.json"
        ),
        "offline_500k_region": (
            root / "inputs" / "normalization" / "offline_500k" / "region.json"
        ),
        "shared_hlt_500k_relation": (
            root / "inputs" / "normalization" / "hlt_shared_500k" / "relation.json"
        ),
        "shared_hlt_500k_region": (
            root / "inputs" / "normalization" / "hlt_shared_500k" / "region.json"
        ),
    }
    artifacts = {name: load_hashed_json(path) for name, path in paths.items()}
    validate_relation_normalization_artifact(
        artifacts["offline_500k_relation"]
    )
    validate_region_normalization(
        artifacts["offline_500k_region"],
        relation_normalization_sha256=artifacts[
            "offline_500k_relation"
        ]["content_hash"],
    )
    validate_relation_normalization_artifact(
        artifacts["shared_hlt_500k_relation"]
    )
    validate_region_normalization(
        artifacts["shared_hlt_500k_region"],
        relation_normalization_sha256=artifacts[
            "shared_hlt_500k_relation"
        ]["content_hash"],
    )
    return artifacts


def _selected_trees(
    tree_root: Path, identities: list[str]
) -> list[dict]:
    trees, _manifest = load_authenticated_tree_selection(
        tree_root, identities
    )
    return [dict(tree) for tree in trees]


def _relation_owner_views(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    trees: list[dict],
    relation_normalizer: dict,
    region_normalizer: dict,
    source_view: str,
) -> dict[str, np.ndarray]:
    """Rebuild deterministic raw relation tensors through their owners."""

    inputs = build_particle_transformer_inputs_from_tokens(
        tokens, mask, source_view=source_view
    )
    features = torch.from_numpy(inputs.pf_features)
    vectors = torch.from_numpy(inputs.pf_vectors)
    valid = torch.from_numpy(np.asarray(mask, dtype=bool)[:, None, :])
    raw = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    torch.manual_seed(0)
    builder = RelationalPairBuilder(
        ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"),
        normalization_artifact=relation_normalizer,
        region_normalization_artifact=region_normalizer,
        weaver_module=weaver,
    )
    builder.eval()
    with torch.no_grad():
        base = build_standard_four_pair_features(
            vectors, mask=valid, module=weaver
        )
        encoders = builder.encoders
        pt = encoders["PT"](vectors, valid, return_details=True)["raw"]
        track = encoders["TRACK"](
            raw, valid, return_details=True
        )["compatibility"]
        pid = encoders["PID"](
            features[:, 6:11], valid, return_details=True
        )["pair_indices"]
        charge = encoders["CHARGE"](
            features[:, 5], valid, return_details=True
        )["raw"]
        density = encoders["DENSITY"](
            raw, valid, return_details=True
        )["descriptor"]
        region = encoders["REGION"](
            raw, valid, trees, return_details=True
        )["raw"]
    return {
        "standard_four": base.cpu().numpy(),
        "PT": pt.cpu().numpy(),
        "TRACK": track.cpu().numpy(),
        "PID": pid.cpu().numpy(),
        "CHARGE": charge.cpu().numpy(),
        "DENSITY": density.cpu().numpy(),
        "REGION": region.cpu().numpy(),
    }


def _build_degradation_audit(
    *,
    root: Path,
    campaign: dict,
    normalizers: dict[str, dict],
    streamed_abc: bool = False,
) -> dict:
    offline_dir = root / "inputs" / "offline" / "model_train"
    offline_meta = load_hashed_json(
        offline_dir / "offline_input_manifest.json",
        expected_contract="retb_offline_input_cache_v1",
    )
    with np.load(
        offline_dir / offline_meta["npz_filename"], allow_pickle=False
    ) as payload:
        all_identities = [
            str(value) for value in payload["identities"].tolist()
        ]
        selected = select_normalization_jet_indices(
            all_identities,
            limit=min(RELATION_AUDIT_JET_LIMIT, len(all_identities)),
        )
        offline_tokens = np.asarray(
            payload["tokens"][selected], dtype=np.float32
        )
        offline_mask = np.asarray(payload["mask"][selected], dtype=bool)
    identities = [all_identities[int(index)] for index in selected]
    hlt_dir = (
        root
        / "inputs"
        / (
            "hlt_v3_streamed_normalizer_sample"
            if streamed_abc
            else "hlt_v3"
        )
        / "model_train"
        / "replica_0"
        / "R_MULTI"
        / "D_NOMINAL"
    )
    profile = load_hashed_json(root / "inputs" / "hlt_v3_profile.json")
    hlt_arrays, hlt_meta = load_hlt_v3_cache(
        hlt_dir,
        expected_profile_contract_sha256=profile["content_hash"],
        expected_logical_role="model_train",
        expected_replica_id=0,
        expected_realization_policy="R_MULTI",
    )
    hlt_identities = [
        str(value) for value in hlt_arrays["identities"].tolist()
    ]
    if streamed_abc:
        hlt_lookup = {
            identity: index for index, identity in enumerate(hlt_identities)
        }
        if not all(identity in hlt_lookup for identity in identities):
            raise ValueError("streamed relation-audit identities are absent")
        hlt_selected = np.asarray(
            [hlt_lookup[identity] for identity in identities], dtype=np.int64
        )
    else:
        if hlt_identities != all_identities:
            raise ValueError("relation-audit offline/HLT identities differ")
        hlt_selected = selected
    hlt_tokens = np.asarray(
        hlt_arrays["tokens"][hlt_selected], dtype=np.float32
    )
    hlt_mask = np.asarray(hlt_arrays["mask"][hlt_selected], dtype=bool)
    hlt_states = np.asarray(
        hlt_arrays["measurement_states"][hlt_selected], dtype=np.int8
    )
    del hlt_arrays
    offline_trees = _selected_trees(
        root
        / "inputs"
        / "region_tree"
        / "offline"
        / "model_train_exclusive_ca_v1",
        identities,
    )
    hlt_trees = _selected_trees(
        root
        / "inputs"
        / "region_tree"
        / (
            "hlt_streamed_normalizer_sample" if streamed_abc else "hlt"
        )
        / "model_train_r0_exclusive_ca_v1",
        identities,
    )
    offline_relations = _relation_owner_views(
        tokens=offline_tokens,
        mask=offline_mask,
        trees=offline_trees,
        relation_normalizer=normalizers["offline_500k_relation"],
        region_normalizer=normalizers["offline_500k_region"],
        source_view="retb_stage_a_offline_audit",
    )
    hlt_relations = _relation_owner_views(
        tokens=hlt_tokens,
        mask=hlt_mask,
        trees=hlt_trees,
        relation_normalizer=normalizers["shared_hlt_500k_relation"],
        region_normalizer=normalizers["shared_hlt_500k_region"],
        source_view="retb_stage_a_hlt_audit",
    )
    generated = build_hlt_v3_view(
        offline_tokens,
        offline_mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_MULTI",
        profile_id="D_NOMINAL",
    )
    if not all(
        np.array_equal(expected, actual)
        for expected, actual in zip(
            (hlt_tokens, hlt_mask, hlt_states), generated[:3]
        )
    ):
        raise ValueError("relation-audit regenerated HLT view differs")
    audit = build_hlt_v3_degradation_audit(
        offline_tokens=offline_tokens,
        offline_mask=offline_mask,
        hlt_tokens=hlt_tokens,
        hlt_mask=hlt_mask,
        measurement_states=hlt_states,
        diagnostics=generated[3],
        relation_views={
            family: (offline_relations[family], hlt_relations[family])
            for family in offline_relations
        },
        profile_contract_sha256=profile["content_hash"],
        cache_metadata_sha256=hlt_meta["content_hash"],
        split_manifest_sha256=hlt_meta["split_manifest_sha256"],
        identity_manifest_sha256=hlt_meta["identity_manifest_sha256"],
        monotonicity=audit_strength_monotonicity(
            offline_tokens,
            offline_mask,
            canonical_identities=identities,
            logical_role="model_train",
            replica_id=0,
        ),
        layout_determinism=assert_layout_determinism(
            offline_tokens,
            offline_mask,
            canonical_identities=identities,
            logical_role="model_train",
            replica_id=0,
            realization_policy="R_MULTI",
            profile_id="D_NOMINAL",
            shard_boundaries=[len(identities) // 2],
        ),
        train_scale_equality=assert_train_scale_shared_identity(
            offline_tokens[0],
            offline_mask[0],
            canonical_identity=identities[0],
            replica_id=0,
            realization_policy="R_MULTI",
            profile_id="D_NOMINAL",
        ),
    )
    audit = bind_source(audit, source_snapshot=source_snapshot(REPO_ROOT))
    if audit["source"] != campaign["source"]:
        raise ValueError("HLT-v3 degradation audit source differs")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--streamed-abc", action="store_true")
    args = parser.parse_args()
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "offline_roles": list(
                        STREAMED_OFFLINE_ROLES
                        if args.streamed_abc
                        else OFFLINE_ROLES
                    ),
                    "hlt_view_count": len(
                        STREAMED_HLT_VIEWS
                        if args.streamed_abc
                        else HLT_VIEWS
                    ),
                    "tree_view_count": (
                        len(
                            STREAMED_OFFLINE_ROLES
                            if args.streamed_abc
                            else STAGE_A_OFFLINE_TREE_ROLES
                        )
                        + len(
                            STREAMED_HLT_VIEWS
                            if args.streamed_abc
                            else STAGE_A_HLT_TREE_VIEWS
                        )
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    split_manifest = load_split_manifest(
        args.campaign_root / "bootstrap" / "split_manifest.json.gz"
    )
    validation_manifest = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "validation_partition_manifest.json.gz"
    )
    scale_manifest = load_hashed_json(
        args.campaign_root / "inputs" / "scale_train_manifest.json.gz"
    )
    expected_identity_rows = {
        "model_train": list(split_manifest.splits["model_train"]),
        "val_stop": [
            JetIdentity.from_dict(row)
            for row in validation_manifest["roles"]["val_stop"]
        ],
        "val_design": [
            JetIdentity.from_dict(row)
            for row in validation_manifest["roles"]["val_design"]
        ],
        "stack_val": list(split_manifest.splits["stack_val"]),
        "final_test": list(split_manifest.splits["final_test"]),
        "scale_train": [
            JetIdentity.from_dict(row)
            for row in scale_manifest["identities"]
        ],
    }
    offline_rows = []
    offline_identities = {}
    offline_metadata = {}
    offline_roles = STREAMED_OFFLINE_ROLES if args.streamed_abc else OFFLINE_ROLES
    hlt_views = STREAMED_HLT_VIEWS if args.streamed_abc else HLT_VIEWS
    for role in offline_roles:
        cache = args.campaign_root / "inputs" / "offline" / role
        metadata = load_hashed_json(
            cache / "offline_input_manifest.json",
            expected_contract="retb_offline_input_cache_v1",
        )
        npz_path = cache / metadata["npz_filename"]
        if _sha256(npz_path) != metadata["npz_sha256"]:
            raise ValueError("audited offline cache bytes differ")
        with np.load(npz_path, allow_pickle=False) as payload:
            tokens = np.asarray(payload["tokens"])
            mask = np.asarray(payload["mask"])
            labels = np.asarray(payload["labels"])
            identities = [
                str(value) for value in payload["identities"].tolist()
            ]
        expected_rows = expected_identity_rows[role]
        if (
            identities != [row.key() for row in expected_rows]
            or labels.dtype != np.int64
            or not np.array_equal(
                labels,
                np.asarray([row.label for row in expected_rows], dtype=np.int64),
            )
        ):
            raise ValueError("offline cache identity/label coverage differs")
        if (
            metadata["campaign_spec_sha256"] != campaign["content_hash"]
            or metadata["logical_role"] != role
            or metadata["identity_manifest_sha256"]
            != _identity_parent_for_role(campaign, role)
            or metadata["raw_input_schema_sha256"]
            != campaign["parent_artifact_hashes"]["raw_input_schema"]
        ):
            raise ValueError("offline cache lineage differs")
        row = {
            "view_id": f"offline:{role}",
            "logical_role": role,
            "metadata_sha256": metadata["content_hash"],
            "identity_manifest_sha256": metadata[
                "identity_manifest_sha256"
            ],
            "labels_dtype": str(labels.dtype),
            "label_order_sha256": hashlib.sha256(
                labels.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "class_counts": [
                int(value)
                for value in np.bincount(labels, minlength=10)
            ],
            **_audit_arrays(tokens, mask, identities),
        }
        if row["event_count"] != int(metadata["event_count"]):
            raise ValueError("offline audit event count differs")
        offline_rows.append(row)
        offline_identities[role] = identities
        offline_metadata[role] = metadata

    hlt_rows = []
    for role, replica, policy in hlt_views:
        cache = (
            args.campaign_root
            / "inputs"
            / (
                "hlt_v3_streamed_normalizer_sample"
                if args.streamed_abc
                else "hlt_v3"
            )
            / role
            / f"replica_{replica}"
            / policy
            / "D_NOMINAL"
        )
        arrays, metadata = load_hlt_v3_cache(
            cache,
            expected_logical_role=role,
            expected_replica_id=replica,
            expected_realization_policy=policy,
        )
        identities = [
            str(value) for value in arrays["identities"].tolist()
        ]
        expected_hlt_identities = offline_identities[role]
        if args.streamed_abc:
            selected = select_normalization_jet_indices(
                expected_hlt_identities,
                limit=min(
                    STREAMED_HLT_NORMALIZER_JETS_PER_REPLICA,
                    len(expected_hlt_identities),
                ),
            )
            expected_hlt_identities = [
                expected_hlt_identities[int(index)] for index in selected
            ]
        if identities != expected_hlt_identities:
            raise ValueError("offline/HLT identity order differs")
        if (
            metadata["identity_manifest_sha256"]
            != _identity_parent_for_role(campaign, role)
            or metadata["raw_input_sha256"]
            != offline_metadata[role]["npz_sha256"]
        ):
            raise ValueError("HLT cache source lineage differs")
        hlt_rows.append(
            {
                "view_id": f"hlt:{role}:r{replica}:{policy}",
                "logical_role": role,
                "replica_id": replica,
                "realization_policy": policy,
                "metadata_sha256": metadata["content_hash"],
                "identity_manifest_sha256": metadata[
                    "identity_manifest_sha256"
                ],
                **_audit_arrays(
                    arrays["tokens"], arrays["mask"], identities
                ),
            }
        )
    tree_index = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "region_tree"
        / (
            "tree_cache_index_streamed_abc.json"
            if args.streamed_abc
            else "tree_cache_index.json"
        )
    )
    if not args.streamed_abc:
        validate_stage_a_tree_index(tree_index)
    expected_tree_ids = {
        *(
            f"offline:{role}"
            for role in (
                STREAMED_OFFLINE_ROLES
                if args.streamed_abc
                else STAGE_A_OFFLINE_TREE_ROLES
            )
        ),
        *(
            f"hlt:{role}:r{replica}:{policy}"
            for role, replica, policy in (
                STREAMED_HLT_VIEWS
                if args.streamed_abc
                else STAGE_A_HLT_TREE_VIEWS
            )
        ),
    }
    if {row["view_id"] for row in tree_index["views"]} != expected_tree_ids:
        raise ValueError("Stage-A tree index view set differs")
    normalizer_artifacts = _normalizer_artifacts(args.campaign_root)
    normalizer_bundle = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "normalization"
        / "stage_a_normalizer_bundle.json"
    )
    validate_stage_a_normalizer_bundle(
        normalizer_bundle, artifacts=normalizer_artifacts
    )
    if (
        tree_index["campaign_spec_sha256"] != campaign["content_hash"]
        or normalizer_bundle["campaign_spec_sha256"]
        != campaign["content_hash"]
        or tree_index.get("source") != campaign.get("source")
        or normalizer_bundle.get("source") != campaign.get("source")
    ):
        raise ValueError("Stage-A audit source lineage differs")
    degradation_audit = _build_degradation_audit(
        root=args.campaign_root,
        campaign=campaign,
        normalizers=normalizer_artifacts,
        streamed_abc=bool(args.streamed_abc),
    )
    degradation_path = (
        args.campaign_root / "inputs" / "hlt_v3_degradation_audit.json"
    )
    degradation_publication = write_immutable_json(
        degradation_path, degradation_audit
    )
    if args.streamed_abc:
        profile = load_hashed_json(
            args.campaign_root
            / "registry"
            / "retb_streamed_abc_execution_profile.json"
        )
        validate_streamed_abc_execution_profile(profile)
        audit = build_streamed_input_audit(
            campaign_spec_sha256=campaign["content_hash"],
            execution_profile_sha256=profile["content_hash"],
            offline_views=offline_rows,
            hlt_views=hlt_rows,
            tree_index_sha256=tree_index["content_hash"],
            normalizer_bundle_sha256=normalizer_bundle["content_hash"],
            hlt_v3_degradation_audit_sha256=degradation_audit[
                "content_hash"
            ],
            source=campaign["source"],
        )
    else:
        audit = build_stage_a_input_audit(
            campaign_spec_sha256=campaign["content_hash"],
            offline_views=offline_rows,
            hlt_views=hlt_rows,
            tree_index_sha256=tree_index["content_hash"],
            normalizer_bundle_sha256=normalizer_bundle["content_hash"],
            hlt_v3_degradation_audit_sha256=degradation_audit["content_hash"],
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    output = args.output or (
        args.campaign_root / "inputs" / "input_audit.json"
    )
    result = {
        "input_audit_sha256": audit["content_hash"],
        "offline_view_count": len(offline_rows),
        "hlt_view_count": len(hlt_rows),
        "hlt_v3_degradation_audit_sha256": degradation_audit[
            "content_hash"
        ],
        "hlt_v3_degradation_audit_publication": degradation_publication,
        "publication": write_immutable_json(output, audit),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
