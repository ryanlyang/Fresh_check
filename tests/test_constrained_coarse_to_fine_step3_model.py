from __future__ import annotations

import unittest

import torch

from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    B0_GLOBAL_ONLY,
    B1_GLOBAL_8,
    B2_GLOBAL_8_32,
    B3_FULL_HIERARCHY,
    B4_NO_MOMENTS,
    B5_NO_COMPOSITION,
    B6_NO_COUNTS,
    B7_DIRECT_CHILD_TOTALS,
    B_TIER_VARIANTS,
    COARSE_TO_FINE_RECONSTRUCTOR_CONTRACT,
    DERIVED_DIAGNOSTIC_FIELD_NAMES,
    LEVEL_CELL_COUNTS,
    MOMENT_FIELD_NAMES,
    PID_COUNT_INDICES,
    PID_PT_INDICES,
    CoarseToFineReconstructorConfig,
    build_coarse_to_fine_reconstructor,
    primitive_accounting,
)


def _toy_part_inputs(*, requires_grad: bool = False):
    torch.manual_seed(3103)
    batch, particles = 2, 7
    features = torch.randn(batch, 17, particles)
    points = 0.2 * torch.randn(batch, 2, particles)
    mask = torch.tensor(
        [
            [[True, True, True, True, False, False, False]],
            [[True, True, True, True, True, False, False]],
        ]
    )
    pid_index = torch.tensor([[0, 1, 2, 0, 4, 3, 1], [2, 2, 0, 1, 4, 3, 0]])
    features[:, 6:11, :] = 0.0
    for b in range(batch):
        for p in range(particles):
            features[b, 6 + int(pid_index[b, p]), p] = 1.0
    pt = torch.tensor(
        [[32.0, 18.0, 9.0, 4.0, 2.0, 1.0, 0.5], [28.0, 14.0, 10.0, 7.0, 3.0, 2.0, 1.0]]
    )
    eta = points[:, 0, :]
    phi = points[:, 1, :]
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px.square() + py.square() + pz.square() + 0.14**2)
    vectors = torch.stack((px, py, pz, energy), dim=1)
    valid = mask.to(dtype=features.dtype)
    features = features * valid
    points = points * valid
    vectors = vectors * valid
    if requires_grad:
        features.requires_grad_()
    return points, features, vectors, mask


def _small_config(variant: str) -> CoarseToFineReconstructorConfig:
    return CoarseToFineReconstructorConfig(
        variant=variant,
        d_model=32,
        num_heads=4,
        encoder_layers=1,
        pool_layers=1,
        decoder_layers_per_level=1,
        pair_hidden_dim=16,
        ffn_multiplier=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    )


class ConstrainedCoarseToFineStep3ModelTests(unittest.TestCase):
    def test_variant_registry_and_output_depths(self):
        self.assertEqual(len(B_TIER_VARIANTS), 8)
        expected_depth = {
            B0_GLOBAL_ONLY: 0,
            B1_GLOBAL_8: 1,
            B2_GLOBAL_8_32: 2,
            B3_FULL_HIERARCHY: 3,
            B4_NO_MOMENTS: 3,
            B5_NO_COMPOSITION: 3,
            B6_NO_COUNTS: 3,
            B7_DIRECT_CHILD_TOTALS: 3,
        }
        inputs = _toy_part_inputs()
        for variant, depth in expected_depth.items():
            with self.subTest(variant=variant):
                model = build_coarse_to_fine_reconstructor(_small_config(variant)).eval()
                with torch.no_grad():
                    output = model(*inputs)
                self.assertEqual(output.variant, variant)
                self.assertEqual(len(output.levels), depth)
                self.assertEqual(output.global_accounting.shape, (2, len(ACCOUNTING_FIELD_NAMES)))
                self.assertEqual(output.global_log_sigma.shape, output.global_accounting.shape)
                self.assertEqual(output.global_auxiliary.shape, (2, len(DERIVED_DIAGNOSTIC_FIELD_NAMES)))
                self.assertEqual(output.global_auxiliary_names, DERIVED_DIAGNOSTIC_FIELD_NAMES)
                self.assertTrue(torch.isfinite(output.global_accounting).all())
                self.assertTrue(torch.isfinite(output.global_log_sigma).all())
                self.assertTrue(torch.isfinite(output.global_auxiliary).all())
                for level, level_output in enumerate(output.levels, start=1):
                    self.assertEqual(
                        level_output.accounting.shape,
                        (2, LEVEL_CELL_COUNTS[level], len(ACCOUNTING_FIELD_NAMES)),
                    )
                    self.assertEqual(level_output.cell_tokens.shape, (2, LEVEL_CELL_COUNTS[level], 32))
                    self.assertEqual(level_output.log_sigma.shape, level_output.accounting.shape)
                    self.assertTrue(torch.isfinite(level_output.accounting).all())
                self.assertEqual(output.summary()["contract"], COARSE_TO_FINE_RECONSTRUCTOR_CONTRACT)

    def test_constrained_variants_close_every_parent_primitive(self):
        inputs = _toy_part_inputs()
        for variant in B_TIER_VARIANTS[:-1]:
            with self.subTest(variant=variant):
                model = build_coarse_to_fine_reconstructor(_small_config(variant)).eval()
                with torch.no_grad():
                    output = model(*inputs)
                parent = output.global_accounting.unsqueeze(1)
                for level in output.levels:
                    error = level.parent_closure_error(parent)
                    self.assertLess(float(error.max()), 2.0e-5)
                    self.assertTrue(level.hard_allocation)
                    self.assertIsNotNone(level.primitive_fractions)
                    parent = level.accounting

    def test_b7_is_capacity_matched_but_does_not_enforce_closure(self):
        model = build_coarse_to_fine_reconstructor(_small_config(B7_DIRECT_CHILD_TOTALS)).eval()
        with torch.no_grad():
            output = model(*_toy_part_inputs())
        parent = output.global_accounting.unsqueeze(1)
        errors = []
        for level in output.levels:
            errors.append(level.parent_closure_error(parent).max())
            self.assertFalse(level.hard_allocation)
            self.assertIsNone(level.primitive_fractions)
            parent = level.accounting
        self.assertGreater(float(torch.stack(errors).max()), 1.0e-3)
        self.assertFalse(output.diagnostics["hard_allocation"])

    def test_channel_ablation_semantics_are_explicit_and_exact(self):
        inputs = _toy_part_inputs()
        field_index = {name: index for index, name in enumerate(ACCOUNTING_FIELD_NAMES)}

        no_moments = build_coarse_to_fine_reconstructor(_small_config(B4_NO_MOMENTS)).eval()(*inputs)
        moment_indices = [field_index[name] for name in MOMENT_FIELD_NAMES]
        self.assertTrue(torch.equal(no_moments.global_accounting[:, moment_indices], torch.zeros_like(no_moments.global_accounting[:, moment_indices])))
        self.assertFalse(no_moments.supervised_field_mask[moment_indices].any())
        for level in no_moments.levels:
            self.assertTrue(torch.equal(level.accounting[..., moment_indices], torch.zeros_like(level.accounting[..., moment_indices])))

        no_counts = build_coarse_to_fine_reconstructor(_small_config(B6_NO_COUNTS)).eval()(*inputs)
        count_indices = [field_index["expected_constituent_count"], *PID_COUNT_INDICES]
        self.assertTrue(torch.equal(no_counts.global_accounting[:, count_indices], torch.zeros_like(no_counts.global_accounting[:, count_indices])))
        self.assertFalse(no_counts.supervised_field_mask[count_indices].any())
        for level in no_counts.levels:
            self.assertTrue(torch.equal(level.accounting[..., count_indices], torch.zeros_like(level.accounting[..., count_indices])))

    def test_b5_uses_hlt_composition_priors_and_shared_spatial_allocations(self):
        points, features, vectors, mask = _toy_part_inputs()
        model = build_coarse_to_fine_reconstructor(_small_config(B5_NO_COMPOSITION)).eval()
        with torch.no_grad():
            output = model(points, features, vectors, mask)
        category_pt = output.global_accounting[:, list(PID_PT_INDICES)]
        predicted_fraction = category_pt / category_pt.sum(dim=-1, keepdim=True)

        feature_rows = features.transpose(1, 2)
        vector_rows = vectors.transpose(1, 2)
        valid = mask[:, 0, :].float().unsqueeze(-1)
        pid = feature_rows[..., 6:11]
        pt = torch.sqrt(vector_rows[..., 0].square() + vector_rows[..., 1].square() + 1.0e-8).unsqueeze(-1)
        expected = (pid * pt * valid).sum(dim=1)
        expected = expected / expected.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(predicted_fraction, expected, atol=2.0e-6, rtol=2.0e-6))
        self.assertFalse(output.supervised_field_mask[list(PID_PT_INDICES)].any())
        first_logits = output.level(1).allocation_logits
        self.assertTrue(
            torch.allclose(
                first_logits[..., list(range(1, 6))],
                first_logits[..., 1:2].expand_as(first_logits[..., 1:6]),
                atol=1.0e-7,
            )
        )

    def test_padding_is_invariant_and_masked_embeddings_are_zero(self):
        points, features, vectors, mask = _toy_part_inputs()
        model = build_coarse_to_fine_reconstructor(_small_config(B2_GLOBAL_8_32)).eval()
        with torch.no_grad():
            reference = model(points, features, vectors, mask)
            altered_features = features.clone()
            altered_points = points.clone()
            altered_vectors = vectors.clone()
            padded = ~mask[:, 0, :]
            altered_features.transpose(1, 2)[padded] = 1.0e4
            altered_points.transpose(1, 2)[padded] = -1.0e4
            altered_vectors.transpose(1, 2)[padded] = 5.0e3
            altered = model(altered_points, altered_features, altered_vectors, mask)
        self.assertTrue(torch.allclose(reference.global_accounting, altered.global_accounting, atol=2.0e-5, rtol=2.0e-5))
        for left, right in zip(reference.levels, altered.levels):
            self.assertTrue(torch.allclose(left.accounting, right.accounting, atol=2.0e-5, rtol=2.0e-5))
        self.assertTrue(torch.equal(reference.hlt.particle_embeddings[~reference.hlt.particle_mask], torch.zeros_like(reference.hlt.particle_embeddings[~reference.hlt.particle_mask])))

    def test_uncertainties_are_bounded_and_used_path_backpropagates(self):
        points, features, vectors, mask = _toy_part_inputs(requires_grad=True)
        model = build_coarse_to_fine_reconstructor(_small_config(B3_FULL_HIERARCHY))
        output = model(points, features, vectors, mask)
        tensors = [output.global_log_sigma, *(level.log_sigma for level in output.levels)]
        for tensor in tensors:
            self.assertTrue(torch.all(tensor >= -8.0))
            self.assertTrue(torch.all(tensor <= 8.0))
        loss = output.global_accounting.log1p().mean()
        loss = loss + sum(level.accounting.log1p().mean() + 0.01 * level.log_sigma.square().mean() for level in output.levels)
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        used_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(used_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in used_gradients))

    def test_warm_start_accepts_wrapped_encoder_state_and_rejects_unrelated_state(self):
        source = build_coarse_to_fine_reconstructor(_small_config(B0_GLOBAL_ONLY))
        target = build_coarse_to_fine_reconstructor(_small_config(B0_GLOBAL_ONLY))
        wrapped = {f"module.hlt_encoder.{key}": value.clone() for key, value in source.hlt_encoder.state_dict().items()}
        report = target.load_hlt_encoder_warm_start(wrapped, strict=True)
        self.assertTrue(report["loaded_keys"])
        for key, value in source.hlt_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, target.hlt_encoder.state_dict()[key]))
        with self.assertRaisesRegex(ValueError, "did not match"):
            target.load_hlt_encoder_warm_start({"unrelated.weight": torch.ones(1)}, strict=False)


if __name__ == "__main__":
    unittest.main()
