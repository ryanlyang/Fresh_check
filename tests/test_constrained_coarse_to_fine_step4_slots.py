from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    C0_DETERMINISTIC_K8,
    C2_NO_DUST,
    C3_SINKHORN,
    C4_HUNGARIAN,
    C5_B1,
    C5_B2,
    C5_B3,
    C5_UNCERTAINTY,
    C6_MULTIVIEW,
    C_TIER_VARIANTS,
    PARTICLE_SLOT_DECODER_CONTRACT,
    PARTICLE_SLOT_LOSS_CONTRACT,
    PID_CATEGORY_NAMES,
    ParticleSlotLossConfig,
    build_c_tier_reconstructor,
    build_hierarchy_targets,
    c_tier_variant_spec,
    compute_particle_slot_loss,
    default_hierarchy_target_layout,
    hungarian_slot_diagnostic,
    prepare_cell_slot_targets,
)


def _particle(pt: float, eta: float, phi: float, pid: int, charge: float = 0.0) -> np.ndarray:
    token = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    token[0] = pt
    token[1] = eta
    token[2] = phi
    token[3] = pt * np.cosh(eta)
    token[4] = charge
    token[5 + pid] = 1.0
    return token


def _toy_raw_views():
    hlt = np.zeros((2, 8, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((2, 8), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    hlt[0, 0] = _particle(30.0, 0.02, 0.03, 0, 1.0)
    hlt[0, 1] = _particle(12.0, -0.10, 0.12, 1)
    hlt[0, 2] = _particle(7.0, 0.18, -0.10, 2)
    hlt_mask[0, :3] = True
    hlt[1, 0] = _particle(26.0, -0.03, -0.04, 0, -1.0)
    hlt[1, 1] = _particle(11.0, 0.14, 0.16, 2)
    hlt[1, 2] = _particle(5.0, -0.20, -0.12, 1)
    hlt_mask[1, :3] = True
    offline[0, 0] = _particle(31.0, 0.02, 0.02, 0, 1.0)
    offline[0, 1] = _particle(10.0, -0.09, 0.11, 1)
    offline[0, 2] = _particle(8.0, 0.17, -0.11, 2)
    offline[0, 3] = _particle(3.0, 0.05, 0.18, 4, -1.0)
    offline_mask[0, :4] = True
    offline[1, 0] = _particle(27.0, -0.02, -0.03, 0, -1.0)
    offline[1, 1] = _particle(10.0, 0.13, 0.15, 2)
    offline[1, 2] = _particle(6.0, -0.19, -0.13, 1)
    offline[1, 3] = _particle(2.0, -0.06, 0.08, 3, 1.0)
    offline_mask[1, :4] = True
    return hlt, hlt_mask, offline, offline_mask


def _torch_inputs():
    hlt, hlt_mask, offline, offline_mask = _toy_raw_views()
    part = build_particle_transformer_inputs_from_tokens(hlt, hlt_mask)
    inputs = (
        torch.from_numpy(part.pf_points),
        torch.from_numpy(part.pf_features),
        torch.from_numpy(part.pf_vectors),
        torch.from_numpy(part.pf_mask),
    )
    return inputs, hlt, hlt_mask, offline, offline_mask


def _small_overrides():
    hierarchy = {
        "d_model": 32,
        "num_heads": 4,
        "encoder_layers": 1,
        "pool_layers": 1,
        "decoder_layers_per_level": 1,
        "pair_hidden_dim": 8,
        "ffn_multiplier": 2.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    slots = {"ffn_multiplier": 2.0, "dropout": 0.0, "attention_dropout": 0.0}
    return hierarchy, slots


def _build(variant: str):
    hierarchy, slots = _small_overrides()
    return build_c_tier_reconstructor(
        variant,
        hierarchy_overrides=hierarchy,
        slot_overrides=slots,
        layout=default_hierarchy_target_layout(radial_boundary=0.16),
    )


def _targets_for(output, offline, offline_mask, hlt, hlt_mask):
    hierarchy_targets = build_hierarchy_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        layout=default_hierarchy_target_layout(radial_boundary=0.16),
    )
    return prepare_cell_slot_targets(
        torch.from_numpy(offline),
        torch.from_numpy(offline_mask),
        torch.from_numpy(hierarchy_targets.final_cell_indices),
        torch.from_numpy(hierarchy_targets.reference_eta),
        torch.from_numpy(hierarchy_targets.reference_phi),
        terminal_level=output.slots.terminal_level,
    )


class ConstrainedCoarseToFineStep4SlotTests(unittest.TestCase):
    def test_c_variant_registry_and_shared_depth_control_spec(self):
        self.assertEqual(len(C_TIER_VARIANTS), 10)
        depth_specs = [c_tier_variant_spec(name) for name in (C5_B1, C5_B2, C5_B3)]
        self.assertEqual([spec.hierarchy_depth for spec in depth_specs], [1, 2, 3])
        self.assertEqual([spec.terminal_cell_count for spec in depth_specs], [8, 32, 128])
        self.assertIs(depth_specs[0].slot_spec, depth_specs[1].slot_spec)
        self.assertIs(depth_specs[1].slot_spec, depth_specs[2].slot_spec)
        self.assertEqual(depth_specs[0].slot_spec.to_dict(), depth_specs[2].slot_spec.to_dict())

    def test_renderer_shapes_and_exact_accounting_constraints(self):
        inputs, *_ = _torch_inputs()
        model = _build(C5_B1).eval()
        with torch.no_grad():
            output = model(*inputs)
        slots = output.slots
        self.assertEqual(slots.summary()["contract"], PARTICLE_SLOT_DECODER_CONTRACT)
        self.assertEqual(slots.total_pt.shape, (2, 1, 8, 16))
        self.assertEqual(slots.category_pt.shape, (2, 1, 8, 16, len(PID_CATEGORY_NAMES)))
        self.assertEqual(slots.local_coordinates.shape, (2, 1, 8, 16, 2))
        self.assertEqual(slots.log_sigma.shape, (2, 1, 8, 16, 5))
        self.assertIsNotNone(slots.dust_total_pt)
        accounting = slots.terminal_accounting[:, None]
        category_indices = [ACCOUNTING_FIELD_NAMES.index(f"{name}_pT") for name in PID_CATEGORY_NAMES]
        count_indices = [ACCOUNTING_FIELD_NAMES.index(f"{name}_count") for name in PID_CATEGORY_NAMES]
        self.assertTrue(
            torch.allclose(
                slots.category_pt.sum(dim=-2) + slots.dust_category_pt,
                accounting[..., category_indices],
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(slots.category_count.sum(dim=-2), accounting[..., count_indices], atol=2.0e-6)
        )
        self.assertTrue(
            torch.allclose(
                slots.total_energy.sum(dim=-1) + slots.dust_total_energy,
                accounting[..., ACCOUNTING_FIELD_NAMES.index("total_energy")],
                atol=2.0e-6,
            )
        )
        conserved = [
            ACCOUNTING_FIELD_NAMES.index("total_pT"),
            ACCOUNTING_FIELD_NAMES.index("total_energy"),
            ACCOUNTING_FIELD_NAMES.index("expected_constituent_count"),
            *category_indices,
            *count_indices,
        ]
        self.assertTrue(
            torch.allclose(slots.rendered_accounting[..., conserved], accounting[..., conserved], atol=3.0e-6)
        )

    def test_coordinates_are_bounded_to_terminal_cells(self):
        inputs, *_ = _torch_inputs()
        model = _build(C5_B2).eval()
        with torch.no_grad():
            output = model(*inputs)
            slots = output.slots
        geometry = model.hierarchy.layout.cell_geometry(2)
        bounds = torch.tensor(
            [[row["eta_min"], row["eta_max"], row["phi_min"], row["phi_max"]] for row in geometry]
        )
        coordinates = slots.local_coordinates
        self.assertTrue(torch.all(coordinates[..., 0] >= bounds[None, None, :, None, 0]))
        self.assertTrue(torch.all(coordinates[..., 0] <= bounds[None, None, :, None, 1]))
        self.assertTrue(torch.all(coordinates[..., 1] >= bounds[None, None, :, None, 2]))
        self.assertTrue(torch.all(coordinates[..., 1] <= bounds[None, None, :, None, 3]))
        radial = torch.tensor([[row["radial_min"], row["radial_max"]] for row in geometry])
        radius = torch.linalg.vector_norm(coordinates, dim=-1)
        active = slots.total_pt > 1.0e-8
        self.assertTrue(torch.all(radius[active] >= radial[None, None, :, None, 0].expand_as(radius)[active] - 2.0e-5))
        self.assertTrue(torch.all(radius[active] <= radial[None, None, :, None, 1].expand_as(radius)[active] + 2.0e-5))

        terminal = output.hierarchy.levels[-1].accounting
        feasible = []
        for row in geometry:
            nearest_eta = 0.0 if row["eta_min"] <= 0.0 <= row["eta_max"] else min(abs(row["eta_min"]), abs(row["eta_max"]))
            nearest_phi = 0.0 if row["phi_min"] <= 0.0 <= row["phi_max"] else min(abs(row["phi_min"]), abs(row["phi_max"]))
            minimum = (nearest_eta**2 + nearest_phi**2) ** 0.5
            maximum = max(
                (eta**2 + phi**2) ** 0.5
                for eta in (row["eta_min"], row["eta_max"])
                for phi in (row["phi_min"], row["phi_max"])
            )
            feasible.append(maximum + 1.0e-12 >= row["radial_min"] and minimum <= row["radial_max"] + 1.0e-12)
        invalid = ~torch.tensor(feasible, dtype=torch.bool)
        self.assertTrue(torch.equal(terminal[:, invalid], torch.zeros_like(terminal[:, invalid])))

    def test_no_dust_and_k8_variants_have_expected_physical_shapes(self):
        inputs, *_ = _torch_inputs()
        with torch.no_grad():
            c0 = _build(C0_DETERMINISTIC_K8).eval()(*inputs).slots
            c2 = _build(C2_NO_DUST).eval()(*inputs).slots
        self.assertEqual(c0.num_real_slots, 8)
        self.assertIsNotNone(c0.dust_total_pt)
        self.assertEqual(c2.num_real_slots, 16)
        self.assertIsNone(c2.dust_total_pt)
        target_pt = c2.terminal_accounting[:, None, :, ACCOUNTING_FIELD_NAMES.index("total_pT")]
        self.assertTrue(torch.allclose(c2.total_pt.sum(dim=-1), target_pt, atol=2.0e-6))

    def test_direct_unconstrained_decoder_does_not_scale_slots_from_hierarchy_accounting(self):
        inputs, *_ = _torch_inputs()
        hierarchy, slots = _small_overrides()
        slots.update({"constrain_accounting": False, "direct_particle_decoding": True})
        model = build_c_tier_reconstructor(
            C5_UNCERTAINTY,
            hierarchy_overrides=hierarchy,
            slot_overrides=slots,
            layout=default_hierarchy_target_layout(radial_boundary=0.16),
        ).eval()
        with torch.no_grad():
            output = model(*inputs).slots
        self.assertTrue(output.diagnostics["direct_particle_decoding"])
        self.assertFalse(output.diagnostics["accounting_constraints_enabled"])
        hierarchy_pt = output.terminal_accounting[:, None, :, ACCOUNTING_FIELD_NAMES.index("total_pT")]
        rendered_pt = output.total_pt.sum(dim=-1) + output.dust_total_pt
        self.assertFalse(torch.allclose(rendered_pt, hierarchy_pt, atol=1.0e-4, rtol=1.0e-4))
        geometry = model.hierarchy.layout.cell_geometry(3)
        feasible = []
        for row in geometry:
            nearest_eta = 0.0 if row["eta_min"] <= 0.0 <= row["eta_max"] else min(abs(row["eta_min"]), abs(row["eta_max"]))
            nearest_phi = 0.0 if row["phi_min"] <= 0.0 <= row["phi_max"] else min(abs(row["phi_min"]), abs(row["phi_max"]))
            minimum = (nearest_eta**2 + nearest_phi**2) ** 0.5
            maximum = max(
                (eta**2 + phi**2) ** 0.5
                for eta in (row["eta_min"], row["eta_max"])
                for phi in (row["phi_min"], row["phi_max"])
            )
            feasible.append(maximum >= row["radial_min"] and minimum <= row["radial_max"])
        invalid = ~torch.tensor(feasible, dtype=torch.bool)
        self.assertTrue(torch.equal(rendered_pt[:, :, invalid], torch.zeros_like(rendered_pt[:, :, invalid])))

    def test_multiview_uses_explicit_latents_and_preserves_constraints_per_view(self):
        inputs, *_ = _torch_inputs()
        model = _build(C6_MULTIVIEW).eval()
        latent_dim = model.slot_decoder.spec.stochastic_latent_dim
        latent = torch.zeros(2, 4, latent_dim)
        latent[:, 1] = 1.0
        latent[:, 2] = -1.0
        latent[:, 3] = 0.5
        with torch.no_grad():
            slots = model(*inputs, stochastic_latent=latent).slots
        self.assertEqual(slots.num_views, 4)
        self.assertTrue(torch.equal(slots.stochastic_latent, latent))
        self.assertGreater(float((slots.local_coordinates[:, 0] - slots.local_coordinates[:, 1]).abs().mean()), 1.0e-6)
        target = slots.terminal_accounting[:, None, :, ACCOUNTING_FIELD_NAMES.index("total_pT")]
        rendered = slots.rendered_accounting[..., ACCOUNTING_FIELD_NAMES.index("total_pT")]
        self.assertTrue(torch.allclose(rendered, target.expand_as(rendered), atol=3.0e-6))

    def test_target_grouping_maps_fine_cells_to_requested_depth(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        with torch.no_grad():
            output = _build(C5_B1).eval()(*inputs)
        targets = _targets_for(output, offline, offline_mask, hlt, hlt_mask)
        self.assertEqual(targets.terminal_level, 1)
        self.assertEqual(targets.num_cells, 8)
        self.assertEqual(int(targets.mask.sum()), int(offline_mask.sum()))
        self.assertTrue(torch.all(targets.terminal_cell_indices[torch.from_numpy(offline_mask)] < 8))

    def test_sinkhorn_loss_is_finite_permutation_invariant_and_backpropagates(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        model = _build(C3_SINKHORN)
        output = model(*inputs)
        targets = _targets_for(output, offline, offline_mask, hlt, hlt_mask)
        config = ParticleSlotLossConfig(matching_mode="sinkhorn", sinkhorn_iterations=15)
        loss = compute_particle_slot_loss(output.slots, targets, config)
        self.assertEqual(loss.detached_summary()["contract"], PARTICLE_SLOT_LOSS_CONTRACT)
        self.assertTrue(torch.isfinite(loss.loss))
        self.assertGreater(float(loss.loss.detach()), 0.0)
        loss.loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

        permutation = np.array([2, 0, 3, 1, 4, 5, 6, 7])
        permuted_offline = offline[:, permutation]
        permuted_mask = offline_mask[:, permutation]
        hierarchy_targets = build_hierarchy_targets(
            hlt,
            hlt_mask,
            permuted_offline,
            permuted_mask,
            layout=default_hierarchy_target_layout(radial_boundary=0.16),
        )
        permuted_targets = prepare_cell_slot_targets(
            torch.from_numpy(permuted_offline),
            torch.from_numpy(permuted_mask),
            torch.from_numpy(hierarchy_targets.final_cell_indices),
            torch.from_numpy(hierarchy_targets.reference_eta),
            torch.from_numpy(hierarchy_targets.reference_phi),
            terminal_level=output.slots.terminal_level,
        )
        repeated = compute_particle_slot_loss(output.slots, permuted_targets, config)
        self.assertTrue(torch.allclose(loss.loss.detach(), repeated.loss.detach(), atol=2.0e-5, rtol=2.0e-5))

    def test_hungarian_evaluation_diagnostic_is_independent_of_train_mode(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        with torch.no_grad():
            output = _build(C4_HUNGARIAN).eval()(*inputs)
        targets = _targets_for(output, offline, offline_mask, hlt, hlt_mask)
        diagnostic = hungarian_slot_diagnostic(
            output.slots,
            targets,
            ParticleSlotLossConfig(matching_mode="ordered", brute_force_limit=8),
        )
        self.assertTrue(torch.isfinite(diagnostic.loss))
        self.assertTrue(diagnostic.assignments)
        self.assertTrue(all(assignment.method.startswith("hungarian") for assignment in diagnostic.assignments))

    def test_uncertainty_only_appears_on_declared_variants(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        with torch.no_grad():
            uncertain_output = _build(C5_UNCERTAINTY).eval()(*inputs)
            plain = _build(C3_SINKHORN).eval()(*inputs).slots
        uncertain = uncertain_output.slots
        self.assertIsNotNone(uncertain.log_sigma)
        self.assertTrue(torch.all(uncertain.log_sigma >= -8.0))
        self.assertTrue(torch.all(uncertain.log_sigma <= 8.0))
        self.assertIsNone(plain.log_sigma)
        targets = _targets_for(uncertain_output, offline, offline_mask, hlt, hlt_mask)
        loss = compute_particle_slot_loss(
            uncertain,
            targets,
            ParticleSlotLossConfig(matching_mode="sinkhorn", sinkhorn_iterations=10),
        )
        self.assertTrue(torch.isfinite(loss.components["uncertainty_nll"]))


if __name__ == "__main__":
    unittest.main()
