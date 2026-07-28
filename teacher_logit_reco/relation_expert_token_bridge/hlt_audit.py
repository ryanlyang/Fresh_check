"""Fail-closed raw, transformed-input, relation, and determinism audits for HLT-v3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.part_inputs import (
    PF_FEATURE_NAMES,
    build_particle_transformer_inputs_from_tokens,
)

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .hlt_cache import cache_array_content_hash
from .hlt_v3 import (
    PID_NAMES,
    apply_hlt_v3_single_jet,
    build_hlt_v3_view,
    measurement_validity_states,
)


HLT_V3_AUDIT_CONTRACT = "retb_hlt_v3_degradation_audit_v1"
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
RAW_FIELD_NAMES = (
    "pt",
    "eta",
    "phi",
    "energy",
    "charge",
    "is_charged_hadron",
    "is_neutral_hadron",
    "is_photon",
    "is_electron",
    "is_muon",
    "d0",
    "d0err",
    "dz",
    "dzerr",
)
REQUIRED_RELATION_FAMILIES = (
    "standard_four",
    "PT",
    "TRACK",
    "PID",
    "CHARGE",
    "DENSITY",
    "REGION",
)


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return {
            "count": 0,
            "quantile_method": "linear",
            "quantiles": {f"{value:g}": None for value in QUANTILES},
        }
    quantiles = np.quantile(selected, QUANTILES, method="linear")
    return {
        "count": int(len(selected)),
        "quantile_method": "linear",
        "quantiles": {
            f"{key:g}": float(value)
            for key, value in zip(QUANTILES, quantiles)
        },
    }


def _pid_categories(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flags = np.rint(np.asarray(tokens)[:, :, 5:10]).astype(np.int8)
    count = flags.sum(axis=-1)
    categories = np.full(mask.shape, 5, dtype=np.int8)
    one = count == 1
    categories[one] = np.argmax(flags[one], axis=-1)
    categories[~mask] = 5
    return categories


def _validate_hlt_domains(
    tokens: np.ndarray,
    mask: np.ndarray,
    states: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(tokens)
    valid = np.asarray(mask, dtype=bool)
    state = np.asarray(states, dtype=np.int8)
    if values.ndim != 3 or values.shape[-1] != len(RAW_FIELD_NAMES):
        raise ValueError("HLT audit tokens must have shape [B,N,14]")
    if valid.shape != values.shape[:2] or state.shape != valid.shape:
        raise ValueError("HLT audit mask/state shape mismatch")
    if not bool(np.isfinite(values).all()):
        raise FloatingPointError("HLT audit values contain NaN or infinity")
    if bool(np.any(values[~valid] != 0.0)):
        raise ValueError("HLT-v3 padding is not exactly zero")
    selected = values[valid]
    if bool(np.any(selected[:, 0] < 0.0) or np.any(selected[:, 3] < 0.0)):
        raise ValueError("HLT-v3 contains negative pT or energy")
    charge_distance = np.min(
        np.abs(selected[:, 4, None] - np.array([-1.0, 0.0, 1.0])),
        axis=1,
    )
    if bool(np.any(charge_distance > 1.0e-6)):
        raise ValueError("HLT-v3 charge lies outside {-1,0,+1}")
    pid = selected[:, 5:10]
    if bool(np.any(np.abs(pid - np.rint(pid)) > 1.0e-6)):
        raise ValueError("HLT-v3 PID flags are not binary")
    if bool(np.any(np.rint(pid).sum(axis=1) > 1)):
        raise ValueError("HLT-v3 contains multi-hot PID")
    expected_state = measurement_validity_states(values, valid)
    if not np.array_equal(state, expected_state):
        raise ValueError("HLT-v3 measurement states differ from raw fields")
    if bool(np.any((state < 0) | (state > 2))):
        raise ValueError("HLT-v3 measurement state lies outside 0..2")
    return {
        "finite_all_values": True,
        "padding_exact_zero": True,
        "charge_domain_exact": True,
        "pid_binary_and_at_most_one_hot": True,
        "measurement_states_exact": True,
        "valid_particle_count": int(np.sum(valid)),
        "valid_track_count": int(np.sum(state == 1)),
        "missing_track_count": int(np.sum(state == 2)),
    }


def _aligned_response(
    offline_tokens: np.ndarray,
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    for jet_index, row in enumerate(diagnostics):
        source_indices = row.get("canonical_output_indices")
        count = int(np.sum(hlt_mask[jet_index]))
        if source_indices is None:
            if row.get("identity_short_circuit") is not True:
                raise ValueError("HLT diagnostic lacks canonical output indices")
            source_indices = list(range(count))
        if len(source_indices) != count:
            raise ValueError("HLT diagnostic output-index count differs")
        before.extend(
            np.asarray(offline_tokens[jet_index, source_indices], dtype=np.float64)
        )
        after.extend(np.asarray(hlt_tokens[jet_index, :count], dtype=np.float64))
    return (
        np.asarray(before, dtype=np.float64).reshape(-1, len(RAW_FIELD_NAMES)),
        np.asarray(after, dtype=np.float64).reshape(-1, len(RAW_FIELD_NAMES)),
    )


def _raw_response_audit(
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _aligned_response(
        offline_tokens, hlt_tokens, hlt_mask, diagnostics
    )
    delta = after - before
    changed = delta != 0.0
    field_rows = {}
    for index, name in enumerate(RAW_FIELD_NAMES):
        field_rows[name] = {
            "aligned_count": int(len(delta)),
            "changed_count": int(np.sum(changed[:, index])),
            "changed_fraction": (
                float(np.mean(changed[:, index])) if len(delta) else 0.0
            ),
            "before": _quantile_summary(before[:, index]),
            "after": _quantile_summary(after[:, index]),
            "residual": _quantile_summary(delta[:, index]),
        }
    offline_counts = np.sum(offline_mask, axis=1)
    hlt_counts = np.sum(hlt_mask, axis=1)
    offline_states = measurement_validity_states(offline_tokens, offline_mask)
    hlt_states = measurement_validity_states(hlt_tokens, hlt_mask)
    return {
        "fields": field_rows,
        "constituent_count_before": _quantile_summary(offline_counts),
        "constituent_count_after": _quantile_summary(hlt_counts),
        "valid_track_count_before": _quantile_summary(
            np.sum(offline_states == 1, axis=1)
        ),
        "valid_track_count_after": _quantile_summary(
            np.sum(hlt_states == 1, axis=1)
        ),
        "aligned_d0_dz_residual_correlation": (
            float(np.corrcoef(delta[:, 10], delta[:, 12])[0, 1])
            if len(delta) > 1
            and np.std(delta[:, 10]) > 0.0
            and np.std(delta[:, 12]) > 0.0
            else 0.0
        ),
    }


def _transformed_input_audit(
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
) -> dict[str, Any]:
    offline = build_particle_transformer_inputs_from_tokens(
        offline_tokens,
        offline_mask,
        source_view="offline_audit",
    )
    hlt = build_particle_transformer_inputs_from_tokens(
        hlt_tokens,
        hlt_mask,
        source_view="hlt_v3_audit",
    )
    before_valid = np.broadcast_to(
        offline.pf_mask, offline.pf_features.shape
    )
    after_valid = np.broadcast_to(hlt.pf_mask, hlt.pf_features.shape)
    channels = {}
    for index, name in enumerate(PF_FEATURE_NAMES):
        before = offline.pf_features[:, index, :][before_valid[:, index, :]]
        after = hlt.pf_features[:, index, :][after_valid[:, index, :]]
        channels[name] = {
            "before": _quantile_summary(before),
            "after": _quantile_summary(after),
            "before_clip_min_fraction": float(np.mean(before == -5.0))
            if len(before)
            else 0.0,
            "before_clip_max_fraction": float(np.mean(before == 5.0))
            if len(before)
            else 0.0,
            "after_clip_min_fraction": float(np.mean(after == -5.0))
            if len(after)
            else 0.0,
            "after_clip_max_fraction": float(np.mean(after == 5.0))
            if len(after)
            else 0.0,
        }
    erasure = {}
    for raw_index, transformed_name in ((11, "part_d0err"), (13, "part_dzerr")):
        raw_before = np.asarray(offline_tokens[:, :, raw_index])
        raw_after = np.asarray(hlt_tokens[:, :, raw_index])
        overlap = offline_mask & hlt_mask
        raw_changed = overlap & (raw_before != raw_after)
        transformed_index = PF_FEATURE_NAMES.index(transformed_name)
        transformed_before = offline.pf_features[:, transformed_index, :]
        transformed_after = hlt.pf_features[:, transformed_index, :]
        erased = raw_changed & (transformed_before == transformed_after)
        saturated = raw_changed & (
            (transformed_after == 0.0) | (transformed_after == 1.0)
        )
        denominator = int(np.sum(raw_changed))
        erasure[transformed_name] = {
            "raw_changed_count": denominator,
            "erased_by_transform_count": int(np.sum(erased)),
            "erased_by_transform_fraction": (
                float(np.sum(erased) / denominator) if denominator else 0.0
            ),
            "after_saturated_count": int(np.sum(saturated)),
            "after_saturated_fraction": (
                float(np.sum(saturated) / denominator) if denominator else 0.0
            ),
        }
    return {
        "derived_from_supplied_views_only": True,
        "feature_order": list(PF_FEATURE_NAMES),
        "channels": channels,
        "uncertainty_change_erasure_or_saturation": erasure,
        "all_17_channels_covered": len(channels) == 17,
    }


def summarize_relation_views(
    relation_views: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Summarize inherited relation-family tensors rebuilt by their owners."""

    if set(relation_views) != set(REQUIRED_RELATION_FAMILIES):
        missing = sorted(set(REQUIRED_RELATION_FAMILIES) - set(relation_views))
        extra = sorted(set(relation_views) - set(REQUIRED_RELATION_FAMILIES))
        raise ValueError(
            f"relation audit coverage differs: missing={missing}, extra={extra}"
        )
    summaries = {}
    for family in REQUIRED_RELATION_FAMILIES:
        before, after = relation_views[family]
        before_array = np.asarray(before)
        after_array = np.asarray(after)
        if not bool(
            np.isfinite(before_array).all() and np.isfinite(after_array).all()
        ):
            raise FloatingPointError(f"{family} relation audit is nonfinite")
        summaries[family] = {
            "before_shape": list(before_array.shape),
            "after_shape": list(after_array.shape),
            "before": _quantile_summary(before_array),
            "after": _quantile_summary(after_array),
        }
    return {
        "canonical_family_order": list(REQUIRED_RELATION_FAMILIES),
        "all_families_covered": True,
        "families": summaries,
        "track_validity_states_explicit": True,
    }


def assert_layout_determinism(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identities: Sequence[str],
    logical_role: str,
    replica_id: int,
    realization_policy: str,
    profile_id: str,
    shard_boundaries: Sequence[int],
) -> dict[str, Any]:
    full = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=canonical_identities,
        logical_role=logical_role,
        replica_id=replica_id,
        realization_policy=realization_policy,
        profile_id=profile_id,
    )
    parts = [[], [], []]
    start = 0
    for stop in (*[int(value) for value in shard_boundaries], len(tokens)):
        if stop < start or stop > len(tokens):
            raise ValueError("invalid shard boundary")
        view = build_hlt_v3_view(
            tokens[start:stop],
            mask[start:stop],
            canonical_identities=canonical_identities[start:stop],
            logical_role=logical_role,
            replica_id=replica_id,
            realization_policy=realization_policy,
            profile_id=profile_id,
        )
        for index in range(3):
            parts[index].append(view[index])
        start = stop
    combined = tuple(np.concatenate(rows, axis=0) for rows in parts)
    for expected, actual in zip(full[:3], combined):
        if not np.array_equal(expected, actual):
            raise AssertionError("HLT-v3 output depends on shard boundaries")

    permutation = np.arange(len(tokens))[::-1]
    permuted = build_hlt_v3_view(
        tokens[permutation],
        mask[permutation],
        canonical_identities=[canonical_identities[index] for index in permutation],
        logical_role=logical_role,
        replica_id=replica_id,
        realization_policy=realization_policy,
        profile_id=profile_id,
    )
    inverse = np.argsort(permutation)
    for expected, actual in zip(full[:3], permuted[:3]):
        if not np.array_equal(expected, actual[inverse]):
            raise AssertionError("HLT-v3 output depends on batch traversal order")
    return {
        "shard_boundaries": [int(value) for value in shard_boundaries],
        "shard_layout_exact": True,
        "batch_order_exact": True,
    }


def assert_train_scale_shared_identity(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identity: str,
    replica_id: int,
    realization_policy: str = "R_MULTI",
    profile_id: str = "D_NOMINAL",
) -> dict[str, Any]:
    outputs = []
    for role in ("model_train", "scale_train"):
        outputs.append(
            apply_hlt_v3_single_jet(
                tokens,
                mask,
                canonical_identity=canonical_identity,
                logical_role=role,
                replica_id=replica_id,
                realization_policy=realization_policy,
                profile_id=profile_id,
            )
        )
    for left, right in zip(outputs[0][:3], outputs[1][:3]):
        if not np.array_equal(left, right):
            raise AssertionError("shared train/scale identity differs")
    return {
        "canonical_identity": canonical_identity,
        "replica_id": int(replica_id),
        "byte_identical": True,
    }


def audit_strength_monotonicity(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identities: Sequence[str],
    logical_role: str,
    replica_id: int,
) -> dict[str, Any]:
    profiles = (
        ("D_OFFLINE_IDENTITY", 0.0),
        ("D_MILD", 0.5),
        ("D_NOMINAL", 1.0),
        ("D_SEVERE", 1.5),
    )
    rows = []
    for profile_id, strength in profiles:
        output, output_mask, states, diagnostics = build_hlt_v3_view(
            tokens,
            mask,
            canonical_identities=canonical_identities,
            logical_role=logical_role,
            replica_id=replica_id,
            realization_policy="R_MULTI",
            profile_id=profile_id,
        )
        mechanism_total = sum(
            sum(int(value) for value in row.get("mechanism_counts", {}).values())
            for row in diagnostics
        )
        expected_probability_mass = sum(
            sum(float(value) for value in row.get("probability_sums", {}).values())
            for row in diagnostics
        )
        rows.append(
            {
                "profile_id": profile_id,
                "strength": strength,
                "valid_particles": int(np.sum(output_mask)),
                "available_tracks": int(np.sum(states == 1)),
                "mechanism_count": mechanism_total,
                "expected_probability_mass": expected_probability_mass,
                "array_content_sha256": cache_array_content_hash(
                    tokens=output,
                    mask=output_mask,
                    measurement_states=states,
                    identities=canonical_identities,
                ),
            }
        )
    probability = [row["expected_probability_mass"] for row in rows]
    if any(right + 1.0e-12 < left for left, right in zip(probability, probability[1:])):
        raise AssertionError("HLT-v3 expected corruption mass is not monotonic")
    if not np.array_equal(
        build_hlt_v3_view(
            tokens,
            mask,
            canonical_identities=canonical_identities,
            logical_role=logical_role,
            replica_id=replica_id,
            profile_id="D_OFFLINE_IDENTITY",
        )[0],
        tokens,
    ):
        raise AssertionError("strength-zero HLT-v3 tokens are not bitwise exact")
    return {
        "rows": rows,
        "expected_probability_mass_nondecreasing": True,
        "strength_zero_bitwise_identity": True,
    }


def build_hlt_v3_degradation_audit(
    *,
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    measurement_states: np.ndarray,
    diagnostics: Sequence[Mapping[str, Any]],
    relation_views: Mapping[str, tuple[np.ndarray, np.ndarray]],
    profile_contract_sha256: str,
    cache_metadata_sha256: str,
    split_manifest_sha256: str,
    identity_manifest_sha256: str,
    monotonicity: Mapping[str, Any],
    layout_determinism: Mapping[str, Any],
    train_scale_equality: Mapping[str, Any],
) -> dict[str, Any]:
    if len(diagnostics) != len(hlt_tokens):
        raise ValueError("one HLT diagnostic is required per jet")
    domain = _validate_hlt_domains(hlt_tokens, hlt_mask, measurement_states)
    if monotonicity.get("expected_probability_mass_nondecreasing") is not True:
        raise ValueError("monotonicity attestation is absent")
    if layout_determinism.get("shard_layout_exact") is not True:
        raise ValueError("shard-layout determinism attestation is absent")
    if train_scale_equality.get("byte_identical") is not True:
        raise ValueError("train/scale equality attestation is absent")
    return with_content_hash(
        {
            "contract": HLT_V3_AUDIT_CONTRACT,
            "schema_version": 1,
            "profile_contract_sha256": require_sha256(
                profile_contract_sha256, name="profile_contract_sha256"
            ),
            "cache_metadata_sha256": require_sha256(
                cache_metadata_sha256, name="cache_metadata_sha256"
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256, name="identity_manifest_sha256"
            ),
            "domain_audit": domain,
            "raw_response_audit": _raw_response_audit(
                offline_tokens,
                offline_mask,
                hlt_tokens,
                hlt_mask,
                diagnostics,
            ),
            "transformed_input_audit": _transformed_input_audit(
                offline_tokens,
                offline_mask,
                hlt_tokens,
                hlt_mask,
            ),
            "relation_input_audit": summarize_relation_views(relation_views),
            "monotonicity": dict(monotonicity),
            "layout_determinism": dict(layout_determinism),
            "train_scale_shared_identity": dict(train_scale_equality),
            "class_labels_supplied_to_generator": False,
            "derived_inputs_rebuilt_from_degraded_view_only": True,
            "fake_duplicate_split_constituents_enabled": False,
            "proxy_claim": "HLT_like_controlled_proxy_not_real_HLT",
            "ok": True,
        }
    )


def validate_hlt_v3_degradation_audit(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=HLT_V3_AUDIT_CONTRACT
    )
    if payload.get("ok") is not True:
        raise ValueError("HLT-v3 degradation audit did not pass")
    transformed = payload.get("transformed_input_audit", {})
    if transformed.get("all_17_channels_covered") is not True:
        raise ValueError("HLT-v3 audit does not cover all transformed channels")
    relations = payload.get("relation_input_audit", {})
    if relations.get("canonical_family_order") != list(REQUIRED_RELATION_FAMILIES):
        raise ValueError("HLT-v3 audit relation-family coverage differs")
    return digest


__all__ = [
    "HLT_V3_AUDIT_CONTRACT",
    "RAW_FIELD_NAMES",
    "REQUIRED_RELATION_FAMILIES",
    "assert_layout_determinism",
    "assert_train_scale_shared_identity",
    "audit_strength_monotonicity",
    "build_hlt_v3_degradation_audit",
    "summarize_relation_views",
    "validate_hlt_v3_degradation_audit",
]
