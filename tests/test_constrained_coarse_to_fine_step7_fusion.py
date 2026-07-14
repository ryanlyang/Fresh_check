from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.constrained_coarse_to_fine import (
    C5_B1,
    C6_MULTIVIEW,
    D0_PSEUDO_ONLY,
    D1_LATE_LOGIT_FUSION,
    D2_REPRESENTATION_FUSION,
    D3_CROSS_ATTENTION,
    D4_UNCERTAINTY_GATED,
    D5_B1,
    D5_B2,
    D5_B3,
    D5_END_TO_END,
    D6_MULTIVIEW,
    D7_GRID_ONLY,
    D8_MULTIDEPTH,
    D_TIER_VARIANTS,
    DUAL_STREAM_FUSION_CONTRACT,
    E0_SHUFFLED_CELLS,
    E1_RANDOM_COORDINATES,
    E2_SHUFFLED_COMPOSITION,
    E3_NO_UNCERTAINTY,
    E4_UNCONSTRAINED_SOURCE,
    E5_NO_SLOT_LOSS_SOURCE,
    E6_CAPACITY_MATCHED_HLT,
    E_TIER_VARIANTS,
    FusionTaggerConfig,
    apply_fusion_control,
    build_c_tier_reconstructor,
    build_dual_stream_fusion_tagger,
    default_hierarchy_target_layout,
    fusion_variant_spec,
    grid_view_from_arrays,
    hlt_reference_axis,
    particle_stream_from_tokens,
    pseudo_particle_views_from_arrays,
    render_pseudo_particle_batch,
)
from teacher_logit_reco.constrained_coarse_to_fine.fusion import (
    CONTROL_NO_UNCERTAINTY,
    CONTROL_RANDOM_COORDINATES,
    CONTROL_SHUFFLED_CELLS,
    CONTROL_SHUFFLED_COMPOSITION,
)
from teacher_logit_reco.constrained_coarse_to_fine.pseudo import _generate_rendered, _render_arrays


def _particle(pt: float, eta: float, phi: float, pid: int) -> np.ndarray:
    token = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    token[0] = pt
    token[1] = eta
    token[2] = phi
    token[3] = pt * np.cosh(eta)
    token[4] = (-1.0, 0.0, 1.0)[pid % 3]
    token[5 + pid] = 1.0
    return token


def _raw_batch(batch: int = 2):
    tokens = np.zeros((batch, 8, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((batch, 8), dtype=bool)
    for jet in range(batch):
        mask[jet, :6] = True
        for particle in range(6):
            tokens[jet, particle] = _particle(
                25.0 - 2.0 * particle + jet,
                -0.20 + 0.075 * particle,
                -0.15 + 0.06 * particle,
                particle % 5,
            )
    return tokens, mask


def _small_reconstructor(variant: str):
    return build_c_tier_reconstructor(
        variant,
        hierarchy_overrides={
            "d_model": 32,
            "num_heads": 4,
            "encoder_layers": 1,
            "pool_layers": 1,
            "decoder_layers_per_level": 1,
            "ffn_multiplier": 2.0,
            "pair_hidden_dim": 8,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        slot_overrides={"ffn_multiplier": 2.0, "dropout": 0.0, "attention_dropout": 0.0},
        layout=default_hierarchy_target_layout(radial_boundary=0.16),
    ).eval()


def _pseudo_arrays(variant: str = C5_B1):
    raw, mask = _raw_batch()
    part = build_particle_transformer_inputs_from_tokens(raw, mask)
    inputs = (
        torch.from_numpy(part.pf_points),
        torch.from_numpy(part.pf_features),
        torch.from_numpy(part.pf_vectors),
        torch.from_numpy(part.pf_mask),
    )
    reference_eta, reference_phi, _ = hlt_reference_axis(raw, mask)
    model = _small_reconstructor(variant)
    with torch.no_grad():
        if variant == C6_MULTIVIEW:
            rendered = _generate_rendered(
                model,
                points=inputs[0],
                features=inputs[1],
                vectors=inputs[2],
                mask=inputs[3],
                reference_eta=torch.from_numpy(reference_eta),
                reference_phi=torch.from_numpy(reference_phi),
                num_views=4,
                generator=torch.Generator().manual_seed(303),
                min_particle_pt=0.0,
                dust_reliability=0.1,
            )
        else:
            output = model(*inputs)
            rendered = render_pseudo_particle_batch(
                output,
                reference_eta=torch.from_numpy(reference_eta),
                reference_phi=torch.from_numpy(reference_phi),
                model=model,
            )
    arrays = _render_arrays(rendered, np.dtype("float32"))
    arrays["reference_eta"] = reference_eta
    arrays["reference_phi"] = reference_phi
    metadata = {
        "variant": variant,
        "source_checkpoint_model": {
            "hierarchy_config": model.hierarchy.config.to_dict(),
            "slot_config": model.slot_decoder.config.to_dict(),
        },
    }
    return raw, mask, arrays, metadata


def _small_tagger(variant: str, *, view_names=None):
    overrides = {
        "d_model": 32,
        "num_heads": 4,
        "hlt_encoder_layers": 1,
        "hlt_pool_layers": 1,
        "pseudo_local_layers": 1,
        "pseudo_global_layers": 1,
        "fusion_layers": 1,
        "ffn_multiplier": 2.0,
        "pair_hidden_dim": 8,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "pseudo_view_dropout": 0.0,
    }
    if view_names is not None:
        overrides["view_names"] = tuple(view_names)
    return build_dual_stream_fusion_tagger(variant, overrides=overrides)


class ConstrainedCoarseToFineStep7FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, mask, arrays, metadata = _pseudo_arrays()
        cls.raw = torch.from_numpy(raw)
        cls.mask = torch.from_numpy(mask)
        cls.hlt = particle_stream_from_tokens(cls.raw, cls.mask)
        cls.arrays = arrays
        cls.metadata = metadata
        cls.canonical = pseudo_particle_views_from_arrays(
            arrays,
            metadata,
            view_name_prefix="canonical",
        )[0]

    def test_registry_contains_complete_d_and_e_tiers_with_schedule_semantics(self):
        self.assertEqual(
            set(D_TIER_VARIANTS),
            {
                D0_PSEUDO_ONLY,
                D1_LATE_LOGIT_FUSION,
                D2_REPRESENTATION_FUSION,
                D3_CROSS_ATTENTION,
                D4_UNCERTAINTY_GATED,
                D5_END_TO_END,
                D5_B1,
                D5_B2,
                D5_B3,
                D6_MULTIVIEW,
                D7_GRID_ONLY,
                D8_MULTIDEPTH,
            },
        )
        self.assertEqual(len(E_TIER_VARIANTS), 7)
        for variant in (D5_END_TO_END, D5_B1, D5_B2, D5_B3, D6_MULTIVIEW, D8_MULTIDEPTH):
            self.assertTrue(fusion_variant_spec(variant).requires_end_to_end_schedule)
        self.assertEqual(fusion_variant_spec(E4_UNCONSTRAINED_SOURCE).source_recipe, "unconstrained_particle_reconstructor")
        self.assertEqual(fusion_variant_spec(E5_NO_SLOT_LOSS_SOURCE).source_recipe, "no_slot_loss_reconstructor")

    def test_pseudo_only_late_representation_and_cross_attention_paths(self):
        for variant in (D0_PSEUDO_ONLY, D1_LATE_LOGIT_FUSION, D2_REPRESENTATION_FUSION, D3_CROSS_ATTENTION):
            with self.subTest(variant=variant):
                model = _small_tagger(variant).eval()
                output = model.forward_detailed(self.hlt, (self.canonical,))
                self.assertEqual(output.logits.shape, (2, 10))
                self.assertTrue(torch.isfinite(output.logits).all())
                self.assertEqual(output.diagnostics["contract"], DUAL_STREAM_FUSION_CONTRACT)
                if variant == D1_LATE_LOGIT_FUSION:
                    torch.testing.assert_close(output.logits, 0.5 * (output.hlt_logits + output.pseudo_logits))

    def test_uncertainty_gates_are_bounded_and_hlt_identity_path_is_ungated(self):
        model = _small_tagger(D4_UNCERTAINTY_GATED)
        output = model.forward_detailed(self.hlt, (self.canonical,))
        self.assertEqual(output.token_gates.shape, (2, 8, 1))
        self.assertEqual(output.pooled_gates.shape, (2, 1))
        self.assertTrue(torch.all(output.token_gates >= 0.0))
        self.assertTrue(torch.all(output.token_gates <= 1.0))
        self.assertTrue(torch.all(output.pooled_gates < 1.0))
        self.assertTrue(output.diagnostics["hlt_skip_ungated"])
        loss = torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1]))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_c6_multiview_and_d8_multidepth_keep_streams_distinct(self):
        _, _, c6_arrays, c6_metadata = _pseudo_arrays(C6_MULTIVIEW)
        c6_views = pseudo_particle_views_from_arrays(c6_arrays, c6_metadata, view_name_prefix="stochastic")
        d6 = _small_tagger(D6_MULTIVIEW).eval().forward_detailed(self.hlt, c6_views)
        self.assertEqual(len(d6.pseudo_representations), 4)
        self.assertEqual(d6.pooled_gates.shape, (2, 4))
        self.assertLessEqual(float(d6.pooled_gates.sum(dim=1).max().detach()), 1.0)

        names = ("best_c", "c5_b1", "c5_b2", "c5_b3")
        depth_views = tuple(replace(self.canonical, name=name, terminal_level=index + 1) for index, name in enumerate(names))
        model = _small_tagger(D8_MULTIDEPTH).eval()
        d8 = model.forward_detailed(self.hlt, depth_views)
        self.assertEqual(tuple(d8.pseudo_representations), names)
        self.assertEqual(len({id(encoder) for encoder in model.pseudo_encoders.values()}), 4)
        self.assertEqual(d8.pooled_gates.shape, (2, 4))

    def test_grid_only_and_capacity_control_are_hlt_deployable(self):
        grid = grid_view_from_arrays(self.arrays, self.metadata, name="grid")
        d7 = _small_tagger(D7_GRID_ONLY).eval().forward_detailed(self.hlt, (grid,))
        self.assertEqual(d7.logits.shape, (2, 10))
        self.assertEqual(grid.num_particles, 8)
        e6_model = _small_tagger(E6_CAPACITY_MATCHED_HLT).eval()
        d4_model = _small_tagger(D4_UNCERTAINTY_GATED).eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in e6_model.parameters()),
            sum(parameter.numel() for parameter in d4_model.parameters()),
        )
        e6 = e6_model.forward_detailed(self.hlt)
        self.assertEqual(e6.logits.shape, (2, 10))
        self.assertTrue(e6.diagnostics["hlt_skip_ungated"])
        self.assertEqual(e6.diagnostics["parameter_match_reference"], D4_UNCERTAINTY_GATED)
        with self.assertRaisesRegex(ValueError, "must not receive"):
            _small_tagger(E6_CAPACITY_MATCHED_HLT).eval().forward_detailed(self.hlt, (self.canonical,))

    def test_e_tier_interventions_are_channel_specific(self):
        view = self.canonical
        shuffled_cells = apply_fusion_control(view, CONTROL_SHUFFLED_CELLS, seed=1)
        self.assertTrue(torch.equal(shuffled_cells.raw_tokens, view.raw_tokens))
        self.assertFalse(torch.equal(shuffled_cells.cell_indices, view.cell_indices))

        coordinates = apply_fusion_control(view, CONTROL_RANDOM_COORDINATES, seed=2)
        torch.testing.assert_close(coordinates.raw_tokens[..., 0], view.raw_tokens[..., 0])
        torch.testing.assert_close(coordinates.raw_tokens[..., 3:10], view.raw_tokens[..., 3:10])
        self.assertGreater(float((coordinates.raw_tokens[..., 1:3] - view.raw_tokens[..., 1:3]).abs().sum()), 0.0)

        composition = apply_fusion_control(view, CONTROL_SHUFFLED_COMPOSITION, seed=3)
        torch.testing.assert_close(composition.raw_tokens[..., :4], view.raw_tokens[..., :4])
        torch.testing.assert_close(
            composition.raw_tokens[..., 5:10].sum(dim=1),
            view.raw_tokens[..., 5:10].sum(dim=1),
        )
        self.assertGreater(float((composition.raw_tokens[..., 5:10] - view.raw_tokens[..., 5:10]).abs().sum()), 0.0)

        no_uncertainty = apply_fusion_control(view, CONTROL_NO_UNCERTAINTY, seed=4)
        self.assertTrue(torch.all(no_uncertainty.reliability == 1.0))
        self.assertTrue(torch.all(no_uncertainty.slot_log_sigma == 0.0))
        self.assertFalse(no_uncertainty.uncertainty_mask.any())

        for variant in (E0_SHUFFLED_CELLS, E1_RANDOM_COORDINATES, E2_SHUFFLED_COMPOSITION, E3_NO_UNCERTAINTY):
            output = _small_tagger(variant).eval().forward_detailed(self.hlt, (view,))
            self.assertTrue(torch.isfinite(output.logits).all())

    def test_view_requirements_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            FusionTaggerConfig(variant=D6_MULTIVIEW, view_names=("only",))
        with self.assertRaisesRegex(ValueError, "at least two"):
            FusionTaggerConfig(variant=D8_MULTIDEPTH, view_names=("only",))
        with self.assertRaisesRegex(ValueError, "grid-token"):
            _small_tagger(D7_GRID_ONLY).eval().forward_detailed(self.hlt, (replace(self.canonical, name="grid"),))

    def test_remaining_d5_depth_and_external_source_recipes_have_runnable_architectures(self):
        for variant in (D5_END_TO_END, E4_UNCONSTRAINED_SOURCE, E5_NO_SLOT_LOSS_SOURCE):
            with self.subTest(variant=variant):
                output = _small_tagger(variant).eval().forward_detailed(self.hlt, (self.canonical,))
                self.assertEqual(output.logits.shape, (2, 10))
        for variant, name in ((D5_B1, "c5_b1"), (D5_B2, "c5_b2"), (D5_B3, "c5_b3")):
            with self.subTest(variant=variant):
                view = replace(self.canonical, name=name)
                output = _small_tagger(variant).eval().forward_detailed(self.hlt, (view,))
                self.assertEqual(output.logits.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
