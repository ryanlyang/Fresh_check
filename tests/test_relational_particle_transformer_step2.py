from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part import (
    PAIR_BASE_CONTRACT,
    RPT_BASE_CONFIG,
    RPT_BASE_EFFECTIVE_WEAVER_DEFAULTS,
    RPT_BASE_MODEL_CONTRACT,
    STANDARD_FOUR_FEATURE_NAMES,
    RelationalParticleTransformer,
    build_global_determinism_contract,
    build_pair_base_contract,
    build_relation_family_registry,
    build_rpt_base_model_contract,
    build_standard_four_pair_features,
    exact_rpt_base_config,
    inspect_weaver_runtime,
    resolve_weaver_pairwise_helper,
    validate_content_hash,
)


def _real_weaver():
    try:
        return importlib.import_module("weaver.nn.model.ParticleTransformer")
    except ImportError:
        pytest.skip("real weaver-core is unavailable in the local environment")


def _batch(
    *,
    batch: int = 2,
    length: int = 6,
    valid_counts: tuple[int, ...] = (6, 3),
    seed: int = 9001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(valid_counts) != batch:
        raise ValueError("valid_counts length must match batch")
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(batch, 2, length, generator=generator)
    features = torch.randn(batch, 17, length, generator=generator)
    pt = torch.rand(batch, length, generator=generator) * 40.0 + 0.5
    eta = torch.randn(batch, length, generator=generator) * 0.8
    phi = (torch.rand(batch, length, generator=generator) * 2.0 - 1.0) * np.pi
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px.square() + py.square() + pz.square() + 0.25)
    vectors = torch.stack((px, py, pz, energy), dim=1)
    mask = torch.zeros(batch, 1, length, dtype=torch.bool)
    for row, count in enumerate(valid_counts):
        mask[row, 0, :count] = True
    points = points.masked_fill(~mask, 0.0)
    features = features.masked_fill(~mask, 0.0)
    vectors = vectors.masked_fill(~mask, 0.0)
    return points.float(), features.float(), vectors.float(), mask


def _models(module):
    torch.manual_seed(71)
    reference = ParticleTransformerHLTClassifier(**exact_rpt_base_config())
    torch.manual_seed(71)
    explicit = RelationalParticleTransformer(weaver_module=module)
    reference.eval()
    explicit.eval()
    return reference, explicit


def _reference_pair_bias(reference, vectors, mask):
    parameters = inspect.signature(reference.mod.pair_embed.forward).parameters
    if "mask" in parameters:
        return reference.mod.pair_embed(vectors, mask=mask)
    return reference.mod.pair_embed(vectors)


def test_step2_contracts_bind_exact_base_and_parity_policy() -> None:
    relation_registry = build_relation_family_registry()
    determinism = build_global_determinism_contract()
    pair_contract = build_pair_base_contract(
        relation_registry_sha256=relation_registry["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
    )
    model_contract = build_rpt_base_model_contract(
        pair_base_sha256=pair_contract["content_hash"],
        weaver_runtime_sha256="a" * 64,
        global_determinism_sha256=determinism["content_hash"],
    )

    assert pair_contract["contract"] == PAIR_BASE_CONTRACT
    assert pair_contract["standard_four"]["feature_names"] == list(
        STANDARD_FOUR_FEATURE_NAMES
    )
    assert pair_contract["standard_four"]["layout"] == (
        "[batch,channels,query,context]"
    )
    assert pair_contract["explicit_path"]["argument"] == "uu"
    assert pair_contract["explicit_path"][
        "state_dictionary_keys_shapes_dtypes_preserved"
    ] is True
    assert model_contract["contract"] == RPT_BASE_MODEL_CONTRACT
    assert model_contract["run_id"] == "RPT_BASE"
    assert model_contract["config"] == RPT_BASE_CONFIG == exact_rpt_base_config()
    assert (
        model_contract["required_effective_weaver_defaults"]
        == RPT_BASE_EFFECTIVE_WEAVER_DEFAULTS
        == {
            "pair_extra_dim": 0,
            "remove_self_pair": False,
            "use_amp": False,
        }
    )
    assert model_contract["pair_path"]["new_relation_channels"] == 0
    assert validate_content_hash(pair_contract) == pair_contract["content_hash"]
    assert validate_content_hash(model_contract) == model_contract["content_hash"]


def test_standard_four_builder_is_dense_finite_and_fail_closed() -> None:
    def helper(xi, xj, num_outputs=4):
        base = xi[:, :1] + 2.0 * xj[:, :1]
        return torch.cat([base + index for index in range(num_outputs)], dim=1)

    module = SimpleNamespace(pairwise_lv_fts=helper)
    vectors = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor(
        [[[True, True, False]], [[True, False, False]]],
        dtype=torch.bool,
    )
    result = build_standard_four_pair_features(
        vectors, mask=mask, module=module
    )
    assert result.shape == (2, 4, 3, 3)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    # Padded raw features intentionally remain Weaver-derived for parity.
    assert result[0, :, 2, 0].abs().sum() > 0

    with pytest.raises(ValueError, match="shape"):
        build_standard_four_pair_features(
            torch.zeros(2, 3, 4), module=module
        )
    invalid = vectors.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="lorentz_vectors"):
        build_standard_four_pair_features(invalid, module=module)

    bad = SimpleNamespace(pairwise_lv_fts=lambda only_one: only_one)
    with pytest.raises(RuntimeError, match="unsupported signature"):
        resolve_weaver_pairwise_helper(bad)


def test_real_weaver_standard_four_pair_bias_logits_gradients_and_state_parity() -> None:
    module = _real_weaver()
    runtime = inspect_weaver_runtime(module)
    assert runtime["helper_name"] in {"pairwise_lv_fts", "pairwise_lv_fts_pp"}
    assert runtime["signatures"]["ParticleTransformer.forward"].find("uu") >= 0
    assert validate_content_hash(runtime) == runtime["content_hash"]
    reference, explicit = _models(module)
    points, features, vectors, mask = _batch()
    tolerance = build_global_determinism_contract()["parity"][
        "authoritative_weaver_explicit_uu"
    ]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])

    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    for name in reference_state:
        assert reference_state[name].shape == explicit_state[name].shape
        assert reference_state[name].dtype == explicit_state[name].dtype
        torch.testing.assert_close(
            reference_state[name], explicit_state[name], atol=0.0, rtol=0.0
        )

    helper_name, helper = resolve_weaver_pairwise_helper(module)
    assert helper_name in {"pairwise_lv_fts", "pairwise_lv_fts_pp"}
    expected_pairs = helper(
        vectors.unsqueeze(-1),
        vectors.unsqueeze(-2),
        num_outputs=4,
    )
    explicit_pairs = explicit.explicit_standard_four(vectors, mask)
    torch.testing.assert_close(
        explicit_pairs, expected_pairs, atol=atol, rtol=rtol
    )

    with torch.no_grad():
        reference_bias = _reference_pair_bias(reference, vectors, mask)
        explicit_bias = explicit.mod.pair_embed(
            vectors,
            uu=explicit_pairs,
            mask=mask,
        )
        reference_logits = reference(points, features, vectors, mask)
        explicit_logits = explicit(points, features, vectors, mask)
    torch.testing.assert_close(
        explicit_bias, reference_bias, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        explicit_logits, reference_logits, atol=atol, rtol=rtol
    )

    reference.zero_grad(set_to_none=True)
    explicit.zero_grad(set_to_none=True)
    reference_features = features.detach().clone().requires_grad_(True)
    explicit_features = features.detach().clone().requires_grad_(True)
    weights = torch.linspace(-0.7, 0.9, 10, dtype=torch.float32)
    reference_loss = (
        reference(points, reference_features, vectors, mask) * weights
    ).sum()
    explicit_loss = (
        explicit(points, explicit_features, vectors, mask) * weights
    ).sum()
    reference_loss.backward()
    explicit_loss.backward()
    torch.testing.assert_close(
        explicit_features.grad,
        reference_features.grad,
        atol=atol,
        rtol=rtol,
    )

    reference_parameters = dict(reference.named_parameters())
    explicit_parameters = dict(explicit.named_parameters())
    assert list(reference_parameters) == list(explicit_parameters)
    for name in reference_parameters:
        reference_gradient = reference_parameters[name].grad
        explicit_gradient = explicit_parameters[name].grad
        assert (reference_gradient is None) == (explicit_gradient is None), name
        if reference_gradient is not None:
            torch.testing.assert_close(
                explicit_gradient,
                reference_gradient,
                atol=atol,
                rtol=rtol,
                msg=lambda message, name=name: f"{name}: {message}",
            )


def test_real_weaver_training_mode_first_forward_parity_without_amp() -> None:
    module = _real_weaver()
    reference, explicit = _models(module)
    reference.train()
    explicit.train()
    points, features, vectors, mask = _batch(
        batch=2,
        length=5,
        valid_counts=(5, 3),
        seed=33,
    )
    tolerance = build_global_determinism_contract()["parity"][
        "authoritative_weaver_explicit_uu"
    ]
    rng_state = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_state.clone())
    reference_logits = reference(points, features, vectors, mask)
    torch.random.set_rng_state(rng_state.clone())
    explicit_logits = explicit(points, features, vectors, mask)
    torch.testing.assert_close(
        explicit_logits,
        reference_logits,
        atol=float(tolerance["atol"]),
        rtol=float(tolerance["rtol"]),
    )
    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    assert list(reference_state) == list(explicit_state)
    for name in reference_state:
        torch.testing.assert_close(
            explicit_state[name],
            reference_state[name],
            atol=float(tolerance["atol"]),
            rtol=float(tolerance["rtol"]),
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_real_weaver_padding_one_particle_and_forced_nonempty_safety() -> None:
    module = _real_weaver()
    reference, explicit = _models(module)
    tolerance = build_global_determinism_contract()["parity"][
        "authoritative_weaver_explicit_uu"
    ]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])

    points, features, vectors, mask = _batch(
        batch=1, length=7, valid_counts=(3,), seed=104
    )
    garbage_points = points.clone()
    garbage_features = features.clone()
    garbage_vectors = vectors.clone()
    garbage_points[:, :, 3:] = 50.0
    garbage_features[:, :, 3:] = -70.0
    garbage_vectors[:, :, 3:] = 30.0
    garbage_vectors[:, 3, 3:] = 100.0
    with torch.no_grad():
        reference_clean = reference(points, features, vectors, mask)
        reference_garbage = reference(
            garbage_points, garbage_features, garbage_vectors, mask
        )
        explicit_clean = explicit(points, features, vectors, mask)
        explicit_garbage = explicit(
            garbage_points, garbage_features, garbage_vectors, mask
        )
    torch.testing.assert_close(
        reference_clean, reference_garbage, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        explicit_clean, explicit_garbage, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        explicit_clean, reference_clean, atol=atol, rtol=rtol
    )

    one_points, one_features, one_vectors, one_mask = _batch(
        batch=1, length=5, valid_counts=(1,), seed=105
    )
    with torch.no_grad():
        one_reference = reference(
            one_points, one_features, one_vectors, one_mask
        )
        one_explicit = explicit(
            one_points, one_features, one_vectors, one_mask
        )
    assert torch.isfinite(one_reference).all()
    assert torch.isfinite(one_explicit).all()
    torch.testing.assert_close(
        one_explicit, one_reference, atol=atol, rtol=rtol
    )

    raw_tokens = np.zeros((1, 5, 14), dtype=np.float32)
    raw_mask = np.zeros((1, 5), dtype=bool)
    canonical = build_particle_transformer_inputs_from_tokens(
        raw_tokens,
        raw_mask,
        source_view="fixed_hlt",
    )
    assert canonical.metadata["forced_nonempty_particle_transformer_rows"] == 1
    forced_points = torch.from_numpy(canonical.pf_points)
    forced_features = torch.from_numpy(canonical.pf_features)
    forced_vectors = torch.from_numpy(canonical.pf_vectors)
    forced_mask = torch.from_numpy(canonical.pf_mask)
    with torch.no_grad():
        forced_reference = reference(
            forced_points, forced_features, forced_vectors, forced_mask
        )
        forced_explicit = explicit(
            forced_points, forced_features, forced_vectors, forced_mask
        )
    assert torch.isfinite(forced_reference).all()
    assert torch.isfinite(forced_explicit).all()
    torch.testing.assert_close(
        forced_explicit, forced_reference, atol=atol, rtol=rtol
    )

    with pytest.raises(ValueError, match="all-empty rows"):
        explicit(
            torch.zeros(1, 2, 5),
            torch.zeros(1, 17, 5),
            torch.zeros(1, 4, 5),
            torch.zeros(1, 1, 5, dtype=torch.bool),
        )
