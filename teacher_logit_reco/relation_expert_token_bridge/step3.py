"""Build, validate, and publish the immutable RETB Step-3 architecture bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .expert_model import (
    EXPERT_ARCHITECTURE_CONTRACT,
    build_expert_architecture_contract,
    validate_expert_architecture_contract,
)
from .layerwise_pair_bias import (
    LAYERWISE_PAIR_BIAS_CONTRACT,
    build_layerwise_pair_bias_contract,
    validate_layerwise_pair_bias_contract,
)
from .particle_tap import (
    MEASUREMENT_EMBED_CONTRACT,
    PARTICLE_TAP_CONTRACT,
    build_measurement_embedding_contract,
    build_particle_tap_contract,
    validate_measurement_embedding_contract,
    validate_particle_tap_contract,
)
from .summary_tokens import (
    SUMMARY_TOKENIZER_CONTRACT,
    TOKEN_ONLY_HEAD_CONTRACT,
    build_summary_tokenizer_contract,
    build_token_only_head_contract,
    validate_summary_tokenizer_contract,
    validate_token_only_head_contract,
)
from .token_shape_registry import (
    TOKEN_SHAPE_CONTRACT,
    build_token_shape_contract,
    validate_token_shape_contract,
)


STEP3_CANDIDATE_REGISTRY_CONTRACT = "retb_step3_candidate_registry_v1"
STEP3_BUNDLE_CONTRACT = "retb_step3_architecture_bundle_v1"
STEP3_REPORT_CONTRACT = "retb_step3_report_v1"


def _bind(payload: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return bind_source(payload, source_snapshot=source)


def build_step3_bundle(
    *,
    campaign_spec_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_sha = require_sha256(
        campaign_spec_sha256, name="campaign_spec_sha256"
    )
    particle_tap = _bind(build_particle_tap_contract(), source_snapshot)
    measurement = _bind(
        build_measurement_embedding_contract(), source_snapshot
    )
    layerwise = _bind(build_layerwise_pair_bias_contract(), source_snapshot)
    tokenizer = _bind(build_summary_tokenizer_contract(), source_snapshot)
    token_head = _bind(build_token_only_head_contract(), source_snapshot)
    shapes = _bind(build_token_shape_contract(), source_snapshot)
    expert = _bind(
        build_expert_architecture_contract(
            particle_tap_sha256=particle_tap["content_hash"],
            layerwise_pair_bias_sha256=layerwise["content_hash"],
            measurement_embedding_sha256=measurement["content_hash"],
            summary_tokenizer_sha256=tokenizer["content_hash"],
            token_only_head_sha256=token_head["content_hash"],
            token_shape_registry_sha256=shapes["content_hash"],
        ),
        source_snapshot,
    )
    candidates = _bind(
        with_content_hash(
            {
                "contract": STEP3_CANDIDATE_REGISTRY_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "ordinary_particle_views": {
                    "O_BASE": {
                        "measurement_embedding": False,
                        "ordinary_weaver_head": True,
                    },
                    "H_BASE": {
                        "measurement_embedding": False,
                        "ordinary_weaver_head": True,
                    },
                },
                "registered_particle_view_candidate": {
                    "id": "V_MEASUREMENT_EMBED",
                    "measurement_embedding": True,
                    "dimension": 128,
                    "injection": (
                        "after_particle_embedding_before_particle_block_1"
                    ),
                },
                "pair_topologies": [
                    "B_CONCAT",
                    "B_DUAL_FIXED",
                    "B_DUAL_GATED",
                ],
                "dual_controls": [
                    "BASE4_SECOND_BASE4_PATH",
                    "ZERO_RELATION_LOGITS",
                ],
                "tokenizer_modes": [
                    "TOK_CANONICAL",
                    "TOK_MASKED_MEAN",
                    "TOK_ONE_QUERY_NO_SELF",
                    "TOK_K_QUERY_NO_SELF",
                    "TOK_MULTI_DEPTH",
                ],
                "classification_bypass_allowed": False,
                "scientific_selection_performed": False,
            }
        ),
        source_snapshot,
    )
    artifacts = {
        "particle_tap": particle_tap,
        "measurement_embedding": measurement,
        "layerwise_pair_bias": layerwise,
        "summary_tokenizer": tokenizer,
        "token_only_head": token_head,
        "token_shapes": shapes,
        "expert_architecture": expert,
        "candidate_registry": candidates,
    }
    manifest = _bind(
        with_content_hash(
            {
                "contract": STEP3_BUNDLE_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "artifact_hashes": {
                    name: artifact["content_hash"]
                    for name, artifact in sorted(artifacts.items())
                },
                "state_dictionary_contract": (
                    "retb_expert_state_dictionary_v1"
                ),
                "old_single_bias_load_allowed": False,
                "materialize_B_L_H_N_N": False,
            }
        ),
        source_snapshot,
    )
    report = _bind(
        with_content_hash(
            {
                "contract": STEP3_REPORT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_sha,
                "step3_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "explicit_particle_state_tap": True,
                    "separate_base_relation_stems": True,
                    "per_layer_projection_streamed": True,
                    "dual_gate_initialized_to_one": True,
                    "measurement_state_candidate_registered": True,
                    "uniform_shapes_frozen": True,
                    "heterogeneous_shape_rules_frozen": True,
                    "token_only_classification": True,
                    "multi_depth_blocks_4_and_8": True,
                    "state_dictionary_versioned": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot,
    )
    return {**artifacts, "step3_bundle": manifest, "step3_report": report}


def validate_step3_bundle(bundle: Mapping[str, Any]) -> str:
    expected_names = {
        "particle_tap",
        "measurement_embedding",
        "layerwise_pair_bias",
        "summary_tokenizer",
        "token_only_head",
        "token_shapes",
        "expert_architecture",
        "candidate_registry",
        "step3_bundle",
        "step3_report",
    }
    if set(bundle) != expected_names:
        raise ValueError("Step-3 bundle members differ from the locked layout")
    validators = {
        "particle_tap": validate_particle_tap_contract,
        "measurement_embedding": validate_measurement_embedding_contract,
        "layerwise_pair_bias": validate_layerwise_pair_bias_contract,
        "summary_tokenizer": validate_summary_tokenizer_contract,
        "token_only_head": validate_token_only_head_contract,
        "token_shapes": validate_token_shape_contract,
        "expert_architecture": validate_expert_architecture_contract,
    }
    hashes = {}
    for name, validator in validators.items():
        hashes[name] = validator(bundle[name])
    hashes["candidate_registry"] = validate_content_hash(
        bundle["candidate_registry"],
        expected_contract=STEP3_CANDIDATE_REGISTRY_CONTRACT,
    )
    manifest_sha = validate_content_hash(
        bundle["step3_bundle"], expected_contract=STEP3_BUNDLE_CONTRACT
    )
    validate_content_hash(
        bundle["step3_report"], expected_contract=STEP3_REPORT_CONTRACT
    )
    expected_hashes = {
        name: digest for name, digest in sorted(hashes.items())
    }
    if bundle["step3_bundle"]["artifact_hashes"] != expected_hashes:
        raise ValueError("Step-3 bundle artifact hashes differ")
    if bundle["step3_report"]["step3_bundle_sha256"] != manifest_sha:
        raise ValueError("Step-3 report belongs to another architecture bundle")
    source = bundle["step3_bundle"].get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Step-3 bundle lacks source provenance")
    expected = build_step3_bundle(
        campaign_spec_sha256=bundle["step3_bundle"].get(
            "campaign_spec_sha256"
        ),
        source_snapshot={
            "source_commit": source.get("commit"),
            "source_status_sha256": source.get("status_sha256"),
            "source_dirty": source.get("dirty"),
        },
    )
    for name in sorted(expected_names):
        if dict(bundle[name]) != expected[name]:
            raise ValueError(
                f"Step-3 artifact {name!r} differs from the locked definition"
            )
    return manifest_sha


def publish_step3_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_sha = validate_step3_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "particle_tap": root / "registry" / "retb_particle_state_tap.json",
        "measurement_embedding": (
            root / "registry" / "retb_measurement_state_embedding.json"
        ),
        "layerwise_pair_bias": (
            root / "registry" / "retb_layerwise_pair_bias.json"
        ),
        "summary_tokenizer": (
            root / "registry" / "retb_summary_tokenizer.json"
        ),
        "token_only_head": (
            root / "registry" / "retb_token_only_expert_head.json"
        ),
        "token_shapes": root / "registry" / "retb_token_shapes.json",
        "expert_architecture": (
            root / "registry" / "retb_expert_architecture.json"
        ),
        "candidate_registry": (
            root / "registry" / "retb_step3_candidates.json"
        ),
        "step3_bundle": (
            root / "registry" / "retb_step3_architecture_bundle.json"
        ),
        "step3_report": root / "reports" / "retb_step3_report.json",
    }
    publications = {
        name: write_immutable_json(path, bundle[name])
        for name, path in paths.items()
    }
    return {
        "campaign_root": str(root.resolve()),
        "step3_bundle_sha256": manifest_sha,
        "publications": publications,
    }


__all__ = [
    "STEP3_BUNDLE_CONTRACT",
    "STEP3_CANDIDATE_REGISTRY_CONTRACT",
    "STEP3_REPORT_CONTRACT",
    "build_step3_bundle",
    "publish_step3_bundle",
    "validate_step3_bundle",
]
