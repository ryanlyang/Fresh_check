from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    bind_source,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_audit import (
    REQUIRED_RELATION_FAMILIES,
    assert_layout_determinism,
    assert_train_scale_shared_identity,
    audit_strength_monotonicity,
    build_hlt_v3_degradation_audit,
    validate_hlt_v3_degradation_audit,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    build_hlt_v3_cache,
    build_hlt_v3_cache_metadata,
    load_hlt_v3_cache,
    publish_hlt_v3_cache,
    validate_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_v3 import (
    DEGRADATION_PROFILES,
    HLT_V3_PROFILE_CONTRACT,
    HltV3Parameters,
    apply_hlt_v3_single_jet,
    build_hlt_v3_profile_contract,
    build_hlt_v3_view,
    charge_flip_probability,
    measurement_validity_states,
    merge_equal_neutral_tokens,
    scale_mechanism_terms,
    track_loss_probability,
    track_tail_probability,
    validate_hlt_v3_profile_contract,
)
from teacher_logit_reco.relation_expert_token_bridge.normalizer_lineage import (
    build_normalizer_population_registry,
    normalizer_population_rows,
    require_normalizer_parent,
    validate_normalizer_population_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    RAW_INPUT_SCHEMA_CONTRACT,
    build_raw_input_schema,
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.replicas import (
    build_hlt_replica_manifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _sample(batch: int = 8, length: int = 24) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(1207)
    tokens = np.zeros((batch, length, 14), dtype=np.float32)
    mask = np.zeros((batch, length), dtype=bool)
    for jet in range(batch):
        count = length - 3 + (jet % 3)
        for particle in range(count):
            mask[jet, particle] = True
            pt = float(rng.uniform(0.05, 80.0))
            eta = float(rng.uniform(-2.4, 2.4))
            phi = float(rng.uniform(-np.pi, np.pi))
            mass = float(rng.uniform(0.0, 1.0))
            tokens[jet, particle, :4] = (
                pt,
                eta,
                phi,
                np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2),
            )
            category = particle % 6
            if category < 5:
                tokens[jet, particle, 5 + category] = 1.0
            if category in (0, 3, 4):
                tokens[jet, particle, 4] = -1.0 if particle % 2 else 1.0
                tokens[jet, particle, 10:14] = (
                    rng.normal(0.0, 0.1),
                    rng.uniform(0.005, 0.05),
                    rng.normal(0.0, 0.2),
                    rng.uniform(0.01, 0.08),
                )
    return tokens, mask, [f"file.root#{index}" for index in range(batch)]


def _profile_contract() -> dict[str, object]:
    return build_hlt_v3_profile_contract(
        raw_input_schema_sha256=SHA_A,
        hlt_replica_manifest_sha256=SHA_B,
    )


def test_profile_and_raw_schema_are_explicitly_versioned() -> None:
    schema = build_raw_input_schema()
    assert schema["contract"] == RAW_INPUT_SCHEMA_CONTRACT
    assert schema["schema_version"] == 2
    assert schema["invalid_track_measurement_sentinel"] == {
        "d0": 0.0,
        "d0err": 0.0,
        "dz": 0.0,
        "dzerr": 0.0,
        "inferred_from_observed_zeros": False,
    }
    profile = _profile_contract()
    assert profile["contract"] == HLT_V3_PROFILE_CONTRACT
    assert profile["profile_name"] == "fixed_hlt_v3_track_dominant_proxy"
    assert profile["proxy_claim"] == "HLT_like_controlled_proxy_not_real_HLT"
    assert profile["fake_duplicate_split_constituents"] is False
    assert validate_hlt_v3_profile_contract(profile) == profile["content_hash"]


def test_strength_zero_is_bitwise_and_constructs_no_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens, mask, identities = _sample(batch=1)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("identity path constructed an RNG")

    monkeypatch.setattr(
        "teacher_logit_reco.relation_expert_token_bridge.hlt_v3._rng",
        forbidden,
    )
    output, output_mask, states, diagnostic = apply_hlt_v3_single_jet(
        tokens[0],
        mask[0],
        canonical_identity=identities[0],
        logical_role="model_train",
        replica_id=0,
        profile_id="D_OFFLINE_IDENTITY",
    )
    assert np.array_equal(output, tokens[0])
    assert np.array_equal(output_mask, mask[0])
    assert np.array_equal(states, measurement_validity_states(tokens[0], mask[0]))
    assert diagnostic["rng_constructed"] is False


def test_hand_calculated_type_multiplier_and_clipping_order() -> None:
    base = {
        "loss_probability": np.array([0.8]),
        "sigma_p": np.array([0.2]),
        "tail_probability": np.array([0.7]),
        "sigma_eta": np.array([0.1]),
        "sigma_phi": np.array([0.12]),
        "reassignment_probability": np.array([0.9]),
    }
    scaled = scale_mechanism_terms(
        base,
        pid_category=1,  # neutral hadron: 1.35, 1.30, 1.25, 1.50
        strength=1.5,
        replica_multipliers=(1.2, 0.8, 0.9, 1.25),
    )
    assert scaled["loss_probability"][0] == pytest.approx(1.0)
    assert scaled["sigma_p"][0] == pytest.approx(0.25)
    assert scaled["kinematic_tail_probability"][0] == pytest.approx(1.0)
    assert scaled["kinematic_tail_delta_scale"] == pytest.approx(1.3 * 1.5 * 1.2)
    assert scaled["sigma_eta"][0] == pytest.approx(0.1 * 1.25 * 1.5 * 1.2)
    assert scaled["sigma_phi"][0] == pytest.approx(0.25)
    assert scaled["reassignment_probability"][0] == pytest.approx(1.0)
    assert scaled["reassignment_delta_scale"] == pytest.approx(1.5 * 1.5 * 1.2)
    pt = np.array([0.8, 100.0])
    eta = np.array([0.0, 1.6])
    density = np.array([0.0, 8.0])
    loss = track_loss_probability(
        pt=pt,
        eta=eta,
        density=density,
        strength=1.5,
        replica_multiplier=1.2,
    )
    assert loss[0] == pytest.approx((0.03 + 0.08 * 0.5) * 1.5 * 1.2)
    assert loss[1] == pytest.approx((0.03 + 0.03 + 0.02) * 1.5 * 1.2)
    tail = track_tail_probability(
        eta=eta,
        density=density,
        strength=0.5,
        replica_multiplier=1.25,
    )
    assert tail[0] == pytest.approx(0.01 * 0.5 * 1.25)
    assert tail[1] == pytest.approx((0.01 + 0.005 + 0.01) * 0.5 * 1.25)
    flip = charge_flip_probability(pt=pt, eta=eta, strength=1.5)
    assert flip[0] == pytest.approx((0.002 + 0.001 * 0.008) * 1.5)
    assert flip[1] == pytest.approx((0.002 + 0.002 + 0.001) * 1.5)


def test_true_four_vector_neutral_merge_and_mass() -> None:
    first = np.zeros(14, dtype=np.float32)
    second = np.zeros(14, dtype=np.float32)
    first[:4] = [10.0, 0.3, 0.2, 10.0 * np.cosh(0.3) + 0.5]
    second[:4] = [7.0, 0.31, 0.21, 7.0 * np.cosh(0.31) + 0.3]
    first[6] = second[6] = 1.0
    merged, mass = merge_equal_neutral_tokens(first, second, category=1)
    vectors = []
    for row in (first, second):
        vectors.append(
            np.array(
                [
                    row[0] * np.cos(row[2]),
                    row[0] * np.sin(row[2]),
                    row[0] * np.sinh(row[1]),
                    row[3],
                ],
                dtype=np.float64,
            )
        )
    expected = vectors[0] + vectors[1]
    actual = np.array(
        [
            merged[0] * np.cos(merged[2]),
            merged[0] * np.sin(merged[2]),
            merged[0] * np.sinh(merged[1]),
            merged[3],
        ]
    )
    assert np.allclose(actual, expected, rtol=0.0, atol=3e-7)
    assert mass == pytest.approx(
        np.sqrt(max(expected[3] ** 2 - np.sum(expected[:3] ** 2), 0.0))
    )
    assert merged[4] == 0.0
    assert np.array_equal(merged[5:10], [0, 1, 0, 0, 0])
    assert np.array_equal(merged[10:14], np.zeros(4))
    with pytest.raises(ValueError, match="only neutral"):
        merge_equal_neutral_tokens(first, second, category=0)


def test_only_equal_neutral_categories_merge_and_charged_never_merge() -> None:
    tokens = np.zeros((6, 14), dtype=np.float32)
    mask = np.ones(6, dtype=bool)
    for index in range(6):
        tokens[index, :4] = [10 - index, 0.0001 * index, 0.0, 10 - index]
    tokens[0:2, 6] = 1.0
    tokens[2:4, 7] = 1.0
    tokens[4:6, 5] = 1.0
    tokens[4:6, 4] = 1.0
    tokens[4:6, 10:14] = [0.1, 0.02, 0.2, 0.03]
    parameters = HltV3Parameters(
        hlt_pt_threshold=0.0,
        merge_radius=1.0,
        merge_probability=1.0,
        eff_plateau_barrel=1.0,
        eff_plateau_endcap=1.0,
        eff_turnon_pt_barrel=-100.0,
        eff_turnon_pt_endcap=-100.0,
        eff_width_pt_barrel=0.0,
        eff_width_pt_endcap=0.0,
        density_loss_scale=0.0,
        jet_quality_sigma=0.0,
    )
    output, output_mask, _states, diagnostic = apply_hlt_v3_single_jet(
        tokens,
        mask,
        canonical_identity="merge-fixture",
        logical_role="model_train",
        replica_id=0,
        profile_id="D_MISSING_ONLY",
        parameters=parameters,
    )
    assert diagnostic["mechanism_counts"]["merge"] == 2
    assert int(np.sum(output_mask)) == 4
    categories = np.argmax(output[output_mask, 5:10], axis=1)
    assert np.sum(categories == 0) == 2
    assert np.sum(categories == 1) == 1
    assert np.sum(categories == 2) == 1


def test_nonmerged_mass_is_preserved_after_response() -> None:
    token = np.zeros((1, 14), dtype=np.float32)
    mask = np.ones(1, dtype=bool)
    pt, eta, phi, mass = 15.0, 0.8, 1.2, 2.0
    token[0, :4] = [pt, eta, phi, np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)]
    token[0, 5] = 1.0
    token[0, 4] = 1.0
    token[0, 10:14] = [0.1, 0.02, 0.3, 0.04]
    output, output_mask, _states, _diagnostic = apply_hlt_v3_single_jet(
        token,
        mask,
        canonical_identity="mass-fixture",
        logical_role="model_train",
        replica_id=0,
        profile_id="D_KIN_ONLY",
    )
    row = output[output_mask][0].astype(np.float64)
    reconstructed = np.sqrt(
        max(row[3] ** 2 - (row[0] * np.cosh(row[1])) ** 2, 0.0)
    )
    assert reconstructed == pytest.approx(mass, abs=2e-4)


def test_exact_pt_ties_keep_pre_sort_canonical_order() -> None:
    tokens = np.zeros((3, 14), dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    for index, eta in enumerate((0.3, -0.2, 0.1)):
        tokens[index, :4] = [10.0, eta, 0.0, 10.0 * np.cosh(eta)]
        tokens[index, 5] = 1.0
        tokens[index, 4] = 1.0
        tokens[index, 10:14] = [0.1 + index, 0.02, 0.2, 0.03]
    output, output_mask, _states, diagnostic = apply_hlt_v3_single_jet(
        tokens,
        mask,
        canonical_identity="tie-fixture",
        logical_role="model_train",
        replica_id=0,
        profile_id="D_TRACK_ONLY",
    )
    assert diagnostic["canonical_output_indices"] == [0, 1, 2]
    assert np.array_equal(output[output_mask, 1], tokens[:, 1])


def test_measurement_states_pid_charge_and_field_profiles() -> None:
    tokens, mask, identities = _sample(batch=2)
    for profile_id in (
        "D_KIN_ONLY",
        "D_TRACK_ONLY",
        "D_MISSING_ONLY",
        "D_NOMINAL",
        "D_MILD",
        "D_SEVERE",
    ):
        output, output_mask, states, _ = build_hlt_v3_view(
            tokens,
            mask,
            canonical_identities=identities,
            logical_role="val_stop",
            replica_id=0,
            profile_id=profile_id,
        )
        assert np.isfinite(output).all()
        assert np.array_equal(states, measurement_validity_states(output, output_mask))
        selected = output[output_mask]
        assert np.all(np.isin(selected[:, 4], [-1.0, 0.0, 1.0]))
        assert np.all(np.rint(selected[:, 5:10]).sum(axis=1) <= 1)
    assert DEGRADATION_PROFILES["D_TRACK_ONLY"].kinematic_response is False
    assert DEGRADATION_PROFILES["D_KIN_ONLY"].track_loss is False
    assert DEGRADATION_PROFILES["D_MISSING_ONLY"].track_response is False
    with pytest.raises(ValueError, match="comparison-only"):
        build_hlt_v3_view(
            tokens,
            mask,
            canonical_identities=identities,
            logical_role="val_stop",
            replica_id=0,
            profile_id="D_LEGACY_V2",
        )


def test_replica_shard_batch_and_train_scale_determinism() -> None:
    tokens, mask, identities = _sample()
    layout = assert_layout_determinism(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=2,
        realization_policy="R_RANDOM",
        profile_id="D_NOMINAL",
        shard_boundaries=[1, 5],
    )
    assert layout["shard_layout_exact"] is True
    for replica_id in range(4):
        result = assert_train_scale_shared_identity(
            tokens[0],
            mask[0],
            canonical_identity=identities[0],
            replica_id=replica_id,
            realization_policy="R_RANDOM",
        )
        assert result["byte_identical"] is True
    first = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_RANDOM",
    )[0]
    second = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=1,
        realization_policy="R_RANDOM",
    )[0]
    assert not np.array_equal(first, second)


def test_cache_metadata_is_noninterchangeable_and_immutable(tmp_path: Path) -> None:
    tokens, mask, identities = _sample(batch=3)
    profile = _profile_contract()
    arrays, diagnostics = build_hlt_v3_cache(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_MULTI",
        profile_id="D_NOMINAL",
    )
    metadata = build_hlt_v3_cache_metadata(
        arrays=arrays,
        diagnostics=diagnostics,
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_MULTI",
        degradation_profile_id="D_NOMINAL",
        profile_contract=profile,
        split_manifest_sha256=SHA_A,
        identity_manifest_sha256=SHA_B,
        raw_input_sha256=SHA_C,
    )
    validate_hlt_v3_cache(
        arrays,
        metadata,
        expected_profile_contract_sha256=profile["content_hash"],
        expected_logical_role="model_train",
        expected_replica_id=0,
    )
    publication = publish_hlt_v3_cache(
        tmp_path, arrays=arrays, metadata=metadata
    )
    assert publication["array_status"] == "published"
    loaded_arrays, loaded_metadata = load_hlt_v3_cache(tmp_path)
    assert np.array_equal(loaded_arrays["tokens"], arrays["tokens"])
    assert loaded_metadata == metadata
    repeated = publish_hlt_v3_cache(
        tmp_path, arrays=arrays, metadata=metadata
    )
    assert repeated["array_status"] == "already_present"

    old = dict(metadata)
    old.pop("content_hash")
    old["contract"] = "retb_hlt_v2_cache_v1"
    old = with_content_hash(old)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_hlt_v3_cache(arrays, old)
    with pytest.raises(ValueError, match="profile parent"):
        validate_hlt_v3_cache(
            arrays,
            metadata,
            expected_profile_contract_sha256=SHA_D,
        )


def test_monotonicity_and_complete_transformed_relation_audit() -> None:
    tokens, mask, identities = _sample(batch=6)
    hlt, hlt_mask, states, diagnostics = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
        profile_id="D_NOMINAL",
    )
    monotonicity = audit_strength_monotonicity(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
    )
    assert monotonicity["expected_probability_mass_nondecreasing"] is True
    relation_views = {
        family: (
            np.arange(12, dtype=np.float32).reshape(1, 3, 4),
            np.arange(12, dtype=np.float32).reshape(1, 3, 4) + 0.5,
        )
        for family in REQUIRED_RELATION_FAMILIES
    }
    audit = build_hlt_v3_degradation_audit(
        offline_tokens=tokens,
        offline_mask=mask,
        hlt_tokens=hlt,
        hlt_mask=hlt_mask,
        measurement_states=states,
        diagnostics=diagnostics,
        relation_views=relation_views,
        profile_contract_sha256=SHA_A,
        cache_metadata_sha256=SHA_B,
        split_manifest_sha256=SHA_C,
        identity_manifest_sha256=SHA_D,
        monotonicity=monotonicity,
        layout_determinism={
            "shard_layout_exact": True,
            "batch_order_exact": True,
        },
        train_scale_equality={"byte_identical": True},
    )
    assert audit["transformed_input_audit"]["all_17_channels_covered"] is True
    assert set(audit["transformed_input_audit"]["channels"]) == {
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_d0",
        "part_d0err",
        "part_dz",
        "part_dzerr",
        "part_deta",
        "part_dphi",
    }
    assert audit["relation_input_audit"]["all_families_covered"] is True
    assert validate_hlt_v3_degradation_audit(audit) == audit["content_hash"]


def test_hand_calculated_normalizer_populations_and_parent_rejection() -> None:
    identities = ["jet-c", "jet-a", "jet-b"]
    assert normalizer_population_rows(
        identities, logical_domain="offline_500k"
    ) == (("jet-a", None), ("jet-b", None), ("jet-c", None))
    assert normalizer_population_rows(
        identities, logical_domain="shared_hlt_500k"
    )[:5] == (
        ("jet-a", 0),
        ("jet-a", 1),
        ("jet-a", 2),
        ("jet-a", 3),
        ("jet-b", 0),
    )
    registry = build_normalizer_population_registry(
        model_train_manifest_sha256=SHA_A,
        model_train_identity_count=500_000,
        scale_train_manifest_sha256=SHA_B,
        scale_train_identity_count=3_000_000,
        raw_input_schema_sha256=SHA_C,
        hlt_v3_profile_sha256=SHA_D,
        inherited_estimator_contract_sha256="e" * 64,
    )
    validate_normalizer_population_registry(registry)
    hlt_recipe = registry["recipes"]["shared_hlt_500k"]
    assert hlt_recipe["replica_weighting"]["population_entry_count"] == 2_000_000
    assert hlt_recipe["replica_ids"] == [0, 1, 2, 3]
    assert hlt_recipe["shared_by_realization_policies"] == [
        "R_FIXED",
        "R_MULTI",
        "R_RANDOM",
    ]
    require_normalizer_parent(
        {
            "normalizer_population_recipe_sha256": hlt_recipe["content_hash"]
        },
        expected_recipe_sha256=hlt_recipe["content_hash"],
    )
    with pytest.raises(ValueError, match="another normalizer"):
        require_normalizer_parent(
            {
                "normalizer_population_recipe_sha256": hlt_recipe[
                    "content_hash"
                ]
            },
            expected_recipe_sha256="f" * 64,
        )


def test_v2_base_term_refactor_preserves_replica_contract_and_source_hash() -> None:
    replica = build_hlt_replica_manifest(
        split_manifest_sha256=SHA_A,
        validation_partition_sha256=SHA_B,
        scale_train_manifest_sha256=SHA_C,
    )
    profile = build_hlt_v3_profile_contract(
        raw_input_schema_sha256=SHA_D,
        hlt_replica_manifest_sha256=replica["content_hash"],
    )
    validate_hlt_v3_profile_contract(profile)
    drifted = dict(profile)
    drifted.pop("content_hash")
    drifted["v2_base_term_helpers"] = json.loads(
        json.dumps(profile["v2_base_term_helpers"])
    )
    drifted["v2_base_term_helpers"]["efficiency"]["source_sha256"] = "0" * 64
    drifted = with_content_hash(drifted)
    with pytest.raises(ValueError, match="source drifted"):
        validate_hlt_v3_profile_contract(drifted)


def test_fixed_substreams_are_independent() -> None:
    from teacher_logit_reco.relation_expert_token_bridge import hlt_v3

    base_seed = 123456789
    merge_rng = hlt_v3._rng(base_seed, "merge")
    track_rng = hlt_v3._rng(base_seed, "track_core")
    merge_first = merge_rng.random(5)
    _ = merge_rng.random(10_000)
    track_after_unrelated_draws = track_rng.random(5)
    assert not np.array_equal(merge_first, track_after_unrelated_draws)
    assert np.array_equal(
        track_after_unrelated_draws,
        hlt_v3._rng(base_seed, "track_core").random(5),
    )
    assert len(set(hlt_v3.SUBSTREAM_IDS.values())) == len(hlt_v3.SUBSTREAM_IDS)


def test_all_step2_clis_resolve_dry_run_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.audit_retb_hlt_v3 import main as audit_main
    from scripts.build_retb_hlt_v3_cache import main as cache_main
    from scripts.fit_retb_normalizers import main as normalizer_main

    repo_root = Path(__file__).resolve().parents[1]
    snapshot = source_snapshot(repo_root)
    parents = {
        name: "0123456789abcdef"[index] * 64
        for index, name in enumerate(
            (
                "artifact_layout",
                "final_select_label_manifest",
                "global_determinism",
                "hlt_replica_manifest",
                "raw_input_schema",
                "scale_train_manifest",
                "split_audit",
                "split_manifest",
                "storage_measurements",
                "validation_partition_manifest",
            )
        )
    }
    campaign = build_campaign_spec(
        campaign_id="step2-cli-test",
        campaign_profile="miniature_test",
        source_snapshot=snapshot,
        parent_artifact_hashes=parents,
        run_registry_hashes={"runs": "f" * 64},
    )
    write_immutable_json(tmp_path / "campaign_spec.json", campaign)
    assert (
        cache_main(
            [
                "--campaign-root",
                str(tmp_path),
                "--logical-role",
                "model_train",
                "--replica-id",
                "0",
                "--identity-manifest-sha256",
                SHA_A,
                "--raw-input-sha256",
                SHA_B,
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (tmp_path / "inputs" / "hlt_v3_profile.json").exists()
    profile = bind_source(
        build_hlt_v3_profile_contract(
            raw_input_schema_sha256=parents["raw_input_schema"],
            hlt_replica_manifest_sha256=parents["hlt_replica_manifest"],
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(tmp_path / "inputs" / "hlt_v3_profile.json", profile)
    assert audit_main(["--campaign-root", str(tmp_path), "--dry-run"]) == 0
    assert (
        normalizer_main(
            [
                "--campaign-root",
                str(tmp_path),
                "--inherited-estimator-contract-sha256",
                SHA_C,
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (
        tmp_path / "inputs" / "normalizer_population_registry.json"
    ).exists()
    assert "dry_run" in capsys.readouterr().out
