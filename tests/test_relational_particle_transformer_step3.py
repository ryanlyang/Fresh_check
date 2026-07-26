from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part import (
    CHARGE_RAW_FEATURE_NAMES,
    NORMALIZATION_CONTRACT,
    PIDEncoder,
    PT_RAW_FEATURE_NAMES,
    ChargeEncoder,
    PTEncoder,
    RELATION_FAMILY_REGISTRY_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    RelationalFamilyParticleTransformer,
    audit_pid_flags,
    average_tied_descending_rank,
    build_charge_raw_features,
    build_global_determinism_contract,
    build_normalization_contract,
    build_pid_charge_relation_contract,
    build_raw_input_schema_contract,
    build_pt_raw_features,
    build_pt_relation_contract,
    build_registered_step3_model,
    build_relation_family_registry,
    build_screening_registry,
    build_step3_model_contract,
    fit_step3_relation_normalization,
    pid_categories,
    quantize_charge,
    validate_content_hash,
    validate_relation_normalization_artifact,
)


def _tokens(
    *,
    jets: int = 6,
    particles: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[JetIdentity]]:
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.zeros((jets, particles), dtype=bool)
    identities: list[JetIdentity] = []
    for jet in range(jets):
        count = particles - (jet % 3)
        mask[jet, :count] = True
        pt = np.asarray(
            [12.0 + jet, 12.0 + jet, 7.0, 3.0, 1.5, 0.8],
            dtype=np.float32,
        )[:count]
        eta = np.linspace(-0.4, 0.5, count, dtype=np.float32)
        tokens[jet, :count, 0] = pt
        tokens[jet, :count, 1] = eta
        tokens[jet, :count, 2] = np.linspace(-2.8, 2.7, count)
        tokens[jet, :count, 3] = pt * np.cosh(eta)
        tokens[jet, :count, 4] = np.resize(
            np.asarray([-1.0, 0.0, 1.0], dtype=np.float32), count
        )
        for particle in range(count):
            category = (jet + particle) % 6
            if category < 5:
                tokens[jet, particle, 5 + category] = 1.0
        tokens[jet, :count, 10] = np.linspace(-0.02, 0.03, count)
        tokens[jet, :count, 11] = 0.005 + 0.001 * jet
        tokens[jet, :count, 12] = np.linspace(-0.04, 0.05, count)
        tokens[jet, :count, 13] = 0.01 + 0.001 * jet
        identities.append(
            JetIdentity(file=f"class_{jet % 2}/file_{jet // 2}.root", entry=jet, label=jet % 10)
        )
    return tokens, mask, identities


def _normalizer_fixture():
    tokens, mask, identities = _tokens()
    normalization_contract = build_normalization_contract(
        split_binding_sha256="1" * 64
    )
    relation_registry = build_relation_family_registry()
    raw_input_schema = build_raw_input_schema_contract()
    artifact = fit_step3_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=normalization_contract,
        relation_registry=relation_registry,
        raw_input_schema=raw_input_schema,
        hlt_binding_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        hlt_model_train_content_sha256="4" * 64,
    )
    return tokens, mask, identities, normalization_contract, relation_registry, artifact


def _torch_inputs(tokens: np.ndarray, mask: np.ndarray):
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        source_view="fixed_hlt",
    )
    return (
        torch.from_numpy(inputs.pf_points),
        torch.from_numpy(inputs.pf_features),
        torch.from_numpy(inputs.pf_vectors),
        torch.from_numpy(inputs.pf_mask),
    )


def _real_weaver():
    try:
        return importlib.import_module("weaver.nn.model.ParticleTransformer")
    except ImportError:
        pytest.skip("real weaver-core is unavailable in the local environment")


def test_step3_normalizer_is_train_only_hashed_and_order_invariant() -> None:
    (
        tokens,
        mask,
        identities,
        normalization_contract,
        relation_registry,
        artifact,
    ) = _normalizer_fixture()
    assert normalization_contract["contract"] == NORMALIZATION_CONTRACT
    assert normalization_contract["schema_version"] == 3
    assert relation_registry["contract"] == RELATION_FAMILY_REGISTRY_CONTRACT
    assert relation_registry["schema_version"] == 4
    assert artifact["contract"] == RELATION_NORMALIZATION_ARTIFACT_CONTRACT
    assert artifact["fit_split"] == "model_train"
    assert artifact["selected_jet_count"] == len(identities)
    assert artifact["selected_directed_pair_count"] == sum(
        min(int(count) ** 2, 64) for count in mask.sum(axis=1)
    )
    assert validate_relation_normalization_artifact(artifact) == artifact[
        "content_hash"
    ]
    assert validate_content_hash(artifact) == artifact["content_hash"]
    assert {
        (record["family_id"], record["feature_name"])
        for record in artifact["records"]
    }.issuperset({
        ("PT", "query_log_pt_fraction"),
        ("PT", "context_log_pt_fraction"),
        ("PT", "context_minus_query_log_pt_fraction"),
        ("PT", "log_pair_scalar_pt_fraction"),
        ("CHARGE", "query_charge"),
        ("CHARGE", "context_charge"),
        ("CHARGE", "charge_product"),
        ("CHARGE", "half_absolute_charge_difference"),
    })
    for record in artifact["records"]:
        assert record["applicable_count"] == artifact["sample_sets"][
            record["applicability_rule_id"]
        ]["applicable_count"]
        assert record["robust_scale"] >= 1.0e-6
        assert 0.0 <= record["applicable_zero_fraction"] <= 1.0
        assert 0.0 <= record["post_normalization_clip_fraction"] <= 1.0
    query_log_samples = []
    for row, count in enumerate(mask.sum(axis=1)):
        pt = tokens[row, :count, 0].astype(np.float64)
        values = np.log(
            (pt + 1.0e-6) / (pt.sum(dtype=np.float64) + 1.0e-6)
        )
        query_log_samples.extend(np.repeat(values, int(count)))
    expected_q25, expected_median, expected_q75 = np.quantile(
        np.asarray(query_log_samples),
        [0.25, 0.5, 0.75],
        method="linear",
    )
    query_record = next(
        record
        for record in artifact["records"]
        if record["family_id"] == "PT"
        and record["feature_name"] == "query_log_pt_fraction"
    )
    assert query_record["q25"] == pytest.approx(expected_q25)
    assert query_record["median"] == pytest.approx(expected_median)
    assert query_record["q75"] == pytest.approx(expected_q75)

    permutation = np.asarray([4, 1, 5, 0, 3, 2])
    reordered = fit_step3_relation_normalization(
        tokens[permutation],
        mask[permutation],
        [identities[index] for index in permutation],
        normalization_contract=normalization_contract,
        relation_registry=relation_registry,
        raw_input_schema=build_raw_input_schema_contract(),
        hlt_binding_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        hlt_model_train_content_sha256="4" * 64,
    )
    assert reordered == artifact

    stale = dict(normalization_contract)
    stale.pop("content_hash")
    stale["fit_split"] = "stack_val"
    from teacher_logit_reco.relational_part import with_content_hash

    with pytest.raises(ValueError, match="model_train"):
        fit_step3_relation_normalization(
            tokens,
            mask,
            identities,
            normalization_contract=with_content_hash(stale),
            relation_registry=relation_registry,
            raw_input_schema=build_raw_input_schema_contract(),
            hlt_binding_sha256="2" * 64,
            source_manifest_sha256="3" * 64,
            hlt_model_train_content_sha256="4" * 64,
        )
    bad_pid = tokens.copy()
    bad_pid[0, 0, 6] = 1.0
    with pytest.raises(ValueError, match="multi-hot"):
        fit_step3_relation_normalization(
            bad_pid,
            mask,
            identities,
            normalization_contract=normalization_contract,
            relation_registry=relation_registry,
            raw_input_schema=build_raw_input_schema_contract(),
            hlt_binding_sha256="2" * 64,
            source_manifest_sha256="3" * 64,
            hlt_model_train_content_sha256="4" * 64,
        )
    bad_charge = tokens.copy()
    bad_charge[0, 0, 4] = 0.2
    with pytest.raises(ValueError, match="charge"):
        fit_step3_relation_normalization(
            bad_charge,
            mask,
            identities,
            normalization_contract=normalization_contract,
            relation_registry=relation_registry,
            raw_input_schema=build_raw_input_schema_contract(),
            hlt_binding_sha256="2" * 64,
            source_manifest_sha256="3" * 64,
            hlt_model_train_content_sha256="4" * 64,
        )


def test_pt_direction_tied_rank_mask_permutation_and_gradients() -> None:
    _, _, _, _, relation_registry, artifact = _normalizer_fixture()
    pt = torch.tensor([[10.0, 10.0, 5.0, 0.0]])
    phi = torch.tensor([[0.2, -0.5, 2.9, 0.0]])
    vectors = torch.stack(
        (
            pt * torch.cos(phi),
            pt * torch.sin(phi),
            torch.zeros_like(pt),
            pt,
        ),
        dim=1,
    )
    mask = torch.tensor([[[True, True, True, False]]])
    rank = average_tied_descending_rank(pt, mask)
    torch.testing.assert_close(rank, torch.tensor([[0.25, 0.25, 1.0, 0.0]]))
    raw = build_pt_raw_features(vectors, mask)
    assert raw.shape == (1, len(PT_RAW_FEATURE_NAMES), 4, 4)
    torch.testing.assert_close(raw[:, 4], -raw[:, 4].transpose(-1, -2))
    torch.testing.assert_close(raw[:, 6], -raw[:, 6].transpose(-1, -2))
    torch.testing.assert_close(raw[:, 9], -raw[:, 9].transpose(-1, -2))
    assert raw[:, :, 3].count_nonzero() == 0
    assert raw[:, :, :, 3].count_nonzero() == 0

    encoder = PTEncoder(artifact)
    details = encoder(vectors, mask, return_details=True)
    assert details["encoded"].shape == (1, 8, 4, 4)
    assert details["encoded"][:, :, 3].count_nonzero() == 0
    assert details["encoded"][:, :, :, 3].count_nonzero() == 0
    fixed_indices = [0, 1, 6, 7, 8, 9]
    torch.testing.assert_close(
        details["normalized"][:, fixed_indices],
        details["raw"][:, fixed_indices],
    )

    permutation = torch.tensor([2, 0, 1, 3])
    inverse = torch.argsort(permutation)
    permuted = encoder(
        vectors[:, :, permutation],
        mask[:, :, permutation],
    )
    recovered = permuted[:, :, inverse][:, :, :, inverse]
    torch.testing.assert_close(recovered, details["encoded"], atol=1e-6, rtol=1e-6)

    details["encoded"].sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    contract = build_pt_relation_contract(
        relation_registry_sha256=relation_registry["content_hash"],
        relation_normalization_sha256=artifact["content_hash"],
    )
    assert contract["fixed_scale_feature_names"] == [
        name
        for name in PT_RAW_FEATURE_NAMES
        if name not in contract["robust_feature_names"]
    ]
    assert validate_content_hash(contract) == contract["content_hash"]


def test_pid_zero_hot_directionality_strictness_mask_and_rare_gradients() -> None:
    flags = torch.zeros(1, 5, 7)
    for category in range(5):
        flags[0, category, category] = 1.0
    mask = torch.tensor([[[True, True, True, True, True, True, False]]])
    categories = pid_categories(flags, mask)
    assert categories.tolist() == [[0, 1, 2, 3, 4, 5, 5]]
    audit = audit_pid_flags(flags, mask)
    assert audit["zero_hot_count"] == 1
    assert audit["multi_hot_count"] == 0

    encoder = PIDEncoder()
    details = encoder(flags, mask, return_details=True)
    assert details["pair_indices"][0, 0, 2].item() == 2
    assert details["pair_indices"][0, 2, 0].item() == 12
    assert details["encoded"].shape == (1, 8, 7, 7)
    assert details["encoded"][:, :, 6].count_nonzero() == 0
    assert details["encoded"][:, :, :, 6].count_nonzero() == 0
    details["encoded"].sum().backward()
    for embedding in (
        encoder.query_embedding,
        encoder.context_embedding,
        encoder.pair_embedding,
    ):
        assert embedding.weight.grad is not None
        assert torch.all(embedding.weight.grad.norm(dim=1) > 0)

    multi = flags.clone()
    multi[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="multi-hot"):
        encoder(multi, mask)
    nonbinary = flags.clone()
    nonbinary[0, 0, 0] = 0.9
    with pytest.raises(ValueError, match="within 1e-6"):
        encoder(nonbinary, mask)


def test_charge_nine_states_twelve_channel_input_and_strict_tolerance() -> None:
    _, _, _, _, relation_registry, artifact = _normalizer_fixture()
    charge = torch.tensor([[-1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[[True, True, True, False]]])
    quantized, state = quantize_charge(charge, mask)
    assert quantized.tolist() == [[-1.0, 0.0, 1.0, 0.0]]
    assert state.tolist() == [[0, 1, 2, 1]]
    raw, _ = build_charge_raw_features(charge, mask)
    assert raw.shape == (1, len(CHARGE_RAW_FEATURE_NAMES), 4, 4)
    # Query -1, context +1.
    torch.testing.assert_close(
        raw[0, :, 0, 2],
        torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    encoder = ChargeEncoder(artifact)
    details = encoder(charge, mask, return_details=True)
    assert details["family_encoder_input"].shape == (1, 12, 4, 4)
    assert details["encoded"].shape == (1, 6, 4, 4)
    assert details["pair_indices"][0, 0, 2].item() == 2
    assert details["pair_indices"][0, 2, 0].item() == 6
    torch.testing.assert_close(
        details["normalized"][:, 4:],
        details["raw"][:, 4:],
    )
    assert details["encoded"][:, :, 3].count_nonzero() == 0
    details["encoded"].sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    invalid = charge.clone()
    invalid[0, 0] = -0.99
    with pytest.raises(ValueError, match="within 1e-6"):
        encoder(invalid, mask)

    contract = build_pid_charge_relation_contract(
        relation_registry_sha256=relation_registry["content_hash"],
        relation_normalization_sha256=artifact["content_hash"],
    )
    assert contract["CHARGE"]["family_encoder_input_dimension"] == 12
    assert contract["PID"]["directed_pair_index"] == (
        "query_category*6+context_category"
    )
    assert validate_content_hash(contract) == contract["content_hash"]


def test_step3_registered_model_contracts() -> None:
    tokens, mask_np, _, _, relation_registry, artifact = _normalizer_fixture()
    screening = build_screening_registry(
        relation_registry_sha256=relation_registry["content_hash"]
    )
    determinism = build_global_determinism_contract()
    pt_contract = build_pt_relation_contract(
        relation_registry_sha256=relation_registry["content_hash"],
        relation_normalization_sha256=artifact["content_hash"],
    )
    contract = build_step3_model_contract(
        "RPT_PT",
        normalization_artifact=artifact,
        screening_registry=screening,
        relation_registry_sha256=relation_registry["content_hash"],
        pair_base_sha256="6" * 64,
        family_contract=pt_contract,
        weaver_runtime_sha256="5" * 64,
        global_determinism_sha256=determinism["content_hash"],
    )
    assert contract["enabled_relations"] == ["base4", "PT"]
    assert contract["pair_path"]["combined_dimension"] == 12
    assert contract["pair_path"]["shared_pair_stem"] is True
    assert validate_content_hash(contract) == contract["content_hash"]
    pid_charge_contract = build_pid_charge_relation_contract(
        relation_registry_sha256=relation_registry["content_hash"],
        relation_normalization_sha256=artifact["content_hash"],
    )
    for run_id, expected_dimension in (
        ("RPT_PID", 12),
        ("RPT_CHARGE", 10),
    ):
        categorical_contract = build_step3_model_contract(
            run_id,
            normalization_artifact=artifact,
            screening_registry=screening,
            relation_registry_sha256=relation_registry["content_hash"],
            pair_base_sha256="6" * 64,
            family_contract=pid_charge_contract,
            weaver_runtime_sha256="5" * 64,
            global_determinism_sha256=determinism["content_hash"],
        )
        assert categorical_contract["pair_path"][
            "combined_dimension"
        ] == expected_dimension
        assert validate_content_hash(categorical_contract) == (
            categorical_contract["content_hash"]
        )


def test_real_weaver_registered_singles_miniature_training() -> None:
    module = _real_weaver()
    tokens, mask_np, _, _, relation_registry, artifact = _normalizer_fixture()
    screening = build_screening_registry(
        relation_registry_sha256=relation_registry["content_hash"]
    )
    points, features, vectors, mask = _torch_inputs(tokens[:2], mask_np[:2])
    labels = torch.tensor([0, 1])
    permutation = torch.tensor([2, 0, 4, 1, 5, 3])
    single_tokens = np.zeros((1, 4, 14), dtype=np.float32)
    single_mask = np.zeros((1, 4), dtype=bool)
    single_mask[0, 0] = True
    single_tokens[0, 0, 0] = 2.0
    single_tokens[0, 0, 3] = 2.0
    single_tokens[0, 0, 4] = 1.0
    single_tokens[0, 0, 5] = 1.0
    single_inputs = _torch_inputs(single_tokens, single_mask)
    forced_inputs = _torch_inputs(
        np.zeros((1, 4, 14), dtype=np.float32),
        np.zeros((1, 4), dtype=bool),
    )
    for run_id, family, expected_dimension in (
        ("RPT_PT", "PT", 12),
        ("RPT_PID", "PID", 12),
        ("RPT_CHARGE", "CHARGE", 10),
    ):
        torch.manual_seed(1701)
        model = build_registered_step3_model(
            run_id,
            normalization_artifact=artifact,
            screening_registry=screening,
            weaver_module=module,
        )
        assert isinstance(model, RelationalFamilyParticleTransformer)
        assert model.run_id == run_id
        assert model.families == (family,)
        details = model.pair_features(
            features, vectors, mask, return_details=True
        )
        assert details["combined"].shape == (
            2,
            expected_dimension,
            tokens.shape[1],
            tokens.shape[1],
        )
        invalid_pairs = ~details["pair_mask"]
        assert details["encoded"][family].masked_select(
            invalid_pairs.expand_as(details["encoded"][family])
        ).count_nonzero() == 0

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        logits = model(points, features, vectors, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        family_parameters = list(model.pair_builder.encoders[family].parameters())
        assert family_parameters
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in family_parameters
        )
        stem_parameters = list(model.mod.pair_embed.fts_embed.parameters())
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in stem_parameters
        )
        before = family_parameters[0].detach().clone()
        optimizer.step()
        assert not torch.equal(before, family_parameters[0].detach())

        model.eval()
        with torch.no_grad():
            nominal = model(points, features, vectors, mask)
            permuted = model(
                points[:, :, permutation],
                features[:, :, permutation],
                vectors[:, :, permutation],
                mask[:, :, permutation],
            )
            garbage_features = features.clone()
            garbage_vectors = vectors.clone()
            garbage_features.masked_fill_(~mask, 100.0)
            garbage_vectors.masked_fill_(~mask, -50.0)
            garbage = model(points, garbage_features, garbage_vectors, mask)
        diagnostics = model.diagnostics(features, vectors, mask)
        assert diagnostics["families"] == [family]
        assert diagnostics["valid_directed_pair_count"] == int(
            (mask.sum(dim=-1).square()).sum()
        )
        assert diagnostics["pair_bias_finite"] is True
        assert family in diagnostics
        with torch.no_grad():
            one_particle = model(*single_inputs)
            forced_nonempty = model(*forced_inputs)
        assert torch.isfinite(one_particle).all()
        assert torch.isfinite(forced_nonempty).all()
        with pytest.raises(ValueError, match="all-empty rows"):
            model(
                torch.zeros(1, 2, 4),
                torch.zeros(1, 17, 4),
                torch.zeros(1, 4, 4),
                torch.zeros(1, 1, 4, dtype=torch.bool),
            )
        torch.testing.assert_close(permuted, nominal, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(garbage, nominal, atol=1e-6, rtol=1e-6)
        del model, optimizer
