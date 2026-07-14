from __future__ import annotations

import unittest

import torch

from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    CONSTRAINED_ACCOUNTING_LAYER_CONTRACT,
    PID_COUNT_INDICES,
    PID_PT_INDICES,
    PRIMITIVE_ACCOUNTING_FIELD_NAMES,
    SIGNED_MOMENT_NAMES,
    TOTAL_COUNT_INDEX,
    TOTAL_PT_INDEX,
    CategoryCountSlotAllocator,
    CellBoundCoordinateTransform,
    ConstrainedSlotNormalizer,
    PositiveAccountingParameterization,
    PositiveNegativeMomentReconstructor,
    SoftmaxAccountingAllocator,
    canonicalize_accounting,
    default_hierarchy_target_layout,
)


def _assert_coupled_totals(test: unittest.TestCase, accounting: torch.Tensor, *, atol: float = 1.0e-6) -> None:
    test.assertTrue(
        torch.allclose(
            accounting[..., TOTAL_PT_INDEX],
            accounting[..., list(PID_PT_INDICES)].sum(dim=-1),
            atol=atol,
            rtol=1.0e-6,
        )
    )
    test.assertTrue(
        torch.allclose(
            accounting[..., TOTAL_COUNT_INDEX],
            accounting[..., list(PID_COUNT_INDICES)].sum(dim=-1),
            atol=atol,
            rtol=1.0e-6,
        )
    )


class ConstrainedCoarseToFineStep2Tests(unittest.TestCase):
    def test_positive_accounting_derives_coupled_totals_and_backpropagates(self):
        torch.manual_seed(7)
        layer = PositiveAccountingParameterization()
        raw = torch.randn(2, 3, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES), requires_grad=True)
        accounting = layer(raw)

        self.assertEqual(CONSTRAINED_ACCOUNTING_LAYER_CONTRACT, "constrained_coarse_to_fine_accounting_layers_v1")
        self.assertEqual(accounting.shape, (2, 3, len(ACCOUNTING_FIELD_NAMES)))
        self.assertTrue(torch.all(accounting > 0.0))
        _assert_coupled_totals(self, accounting)

        accounting.square().mean().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)

    def test_softmax_allocator_conserves_every_parent_channel_at_two_levels(self):
        torch.manual_seed(11)
        positive = PositiveAccountingParameterization()
        parent_raw = torch.randn(2, 3, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES), requires_grad=True)
        parent = positive(parent_raw)
        first_logits = (50.0 * torch.randn(2, 3, 4, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES))).requires_grad_()
        first = SoftmaxAccountingAllocator(4)(parent, first_logits)

        self.assertEqual(first.children.shape, (2, 3, 4, len(ACCOUNTING_FIELD_NAMES)))
        self.assertTrue(torch.isfinite(first.children).all())
        self.assertTrue(torch.isfinite(first.primitive_fractions).all())
        self.assertTrue(torch.allclose(first.children.sum(dim=-2), parent, atol=2.0e-5, rtol=2.0e-6))
        _assert_coupled_totals(self, first.children, atol=2.0e-5)

        second_logits = torch.randn(2, 3, 4, 4, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES), requires_grad=True)
        second = SoftmaxAccountingAllocator(4)(first.children, second_logits)
        self.assertTrue(torch.allclose(second.children.sum(dim=-2), first.children, atol=2.0e-5, rtol=2.0e-6))
        _assert_coupled_totals(self, second.children, atol=2.0e-5)

        loss = second.children.square().mean() + first.primitive_fractions.square().mean()
        loss.backward()
        for gradient in (parent_raw.grad, first_logits.grad, second_logits.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_slot_normalizer_conserves_category_pt_and_expected_count(self):
        torch.manual_seed(17)
        cell = PositiveAccountingParameterization()(
            torch.randn(2, 5, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES))
        )
        category_logits = torch.randn(2, 5, 16, len(PID_PT_INDICES), requires_grad=True)
        count_logits = torch.randn(2, 5, 16, requires_grad=True)
        output = ConstrainedSlotNormalizer()(cell, category_logits, count_logits)

        category_totals = cell[..., list(PID_PT_INDICES)]
        self.assertTrue(
            torch.allclose(output.category_pt_per_slot.sum(dim=-2), category_totals, atol=2.0e-6, rtol=2.0e-6)
        )
        self.assertTrue(
            torch.allclose(output.total_pt_per_slot.sum(dim=-1), cell[..., TOTAL_PT_INDEX], atol=2.0e-6, rtol=2.0e-6)
        )
        self.assertTrue(
            torch.allclose(
                output.expected_count_per_slot.sum(dim=-1),
                cell[..., TOTAL_COUNT_INDEX],
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )
        self.assertTrue(torch.allclose(output.pid_probabilities.sum(dim=-1), torch.ones_like(output.total_pt_per_slot)))
        self.assertTrue(torch.isfinite(output.pid_probabilities).all())

        (output.total_pt_per_slot.square().mean() + output.expected_count_per_slot.square().mean()).backward()
        self.assertTrue(torch.isfinite(category_logits.grad).all())
        self.assertTrue(torch.isfinite(count_logits.grad).all())

    def test_category_count_allocator_conserves_each_category_and_handles_zero_cells(self):
        cell = torch.zeros(2, len(ACCOUNTING_FIELD_NAMES), dtype=torch.float32)
        logits = torch.tensor(
            [
                [[1000.0] * len(PID_COUNT_INDICES), [-1000.0] * len(PID_COUNT_INDICES)],
                [[-3.0] * len(PID_COUNT_INDICES), [2.0] * len(PID_COUNT_INDICES)],
            ],
            dtype=torch.float32,
        )
        output = CategoryCountSlotAllocator()(cell, logits)
        self.assertTrue(torch.equal(output.category_per_slot.sum(dim=-2), torch.zeros_like(cell[..., list(PID_COUNT_INDICES)])))
        self.assertTrue(torch.equal(output.total_per_slot, torch.zeros_like(output.total_per_slot)))
        self.assertTrue(torch.isfinite(output.category_probabilities).all())
        self.assertTrue(torch.allclose(output.category_probabilities.sum(dim=-1), torch.ones_like(output.total_per_slot)))

    def test_positive_negative_moments_reconstruct_axes_and_widths(self):
        accounting = torch.zeros(1, len(ACCOUNTING_FIELD_NAMES), dtype=torch.float64)
        index = {name: ACCOUNTING_FIELD_NAMES.index(name) for name in ACCOUNTING_FIELD_NAMES}
        accounting[0, index["charged_pT"]] = 10.0
        accounting[0, index["charged_count"]] = 2.0
        accounting[0, index["sum_pT_abs_deta_pos"]] = 3.0
        accounting[0, index["sum_pT_abs_deta_neg"]] = 1.0
        accounting[0, index["sum_pT_abs_dphi_pos"]] = 1.5
        accounting[0, index["sum_pT_abs_dphi_neg"]] = 2.5
        accounting[0, index["sum_pT_deta2"]] = 1.0
        accounting[0, index["sum_pT_dphi2"]] = 2.0
        accounting[0, index["sum_pT_r"]] = 2.5
        accounting[0, index["sum_pT_r2"]] = 2.5
        accounting = canonicalize_accounting(accounting)

        output = PositiveNegativeMomentReconstructor()(accounting)
        self.assertEqual(output.names, SIGNED_MOMENT_NAMES)
        self.assertAlmostEqual(float(output.field("sum_pT_deta")), 2.0)
        self.assertAlmostEqual(float(output.field("sum_pT_dphi")), -1.0)
        self.assertAlmostEqual(float(output.field("axis_deta")), 0.2)
        self.assertAlmostEqual(float(output.field("axis_dphi")), -0.1)
        self.assertAlmostEqual(float(output.field("width_eta")), 0.06)
        self.assertAlmostEqual(float(output.field("width_phi")), 0.19)
        self.assertAlmostEqual(float(output.field("mean_r")), 0.25)
        self.assertAlmostEqual(float(output.field("r_rms")), 0.5)

    def test_cell_bound_coordinates_broadcast_and_remain_differentiable(self):
        raw = torch.tensor(
            [
                [[[0.0, 0.0], [4.0, -4.0]], [[-2.0, 2.0], [1.0, -1.0]]],
                [[[1.0, 2.0], [-3.0, 3.0]], [[5.0, -5.0], [0.5, -0.5]]],
            ],
            requires_grad=True,
        )
        bounds = torch.tensor(
            [[-0.8, 0.0, -0.4, 0.0], [0.0, 0.8, 0.0, 0.4]],
            dtype=torch.float32,
        )
        coordinates = CellBoundCoordinateTransform()(raw, bounds)
        expanded_bounds = bounds.unsqueeze(-2).expand(2, 2, 2, 4)
        self.assertEqual(coordinates.shape, (2, 2, 2, 2))
        self.assertTrue(torch.all(coordinates[..., 0] >= expanded_bounds[..., 0]))
        self.assertTrue(torch.all(coordinates[..., 0] <= expanded_bounds[..., 1]))
        self.assertTrue(torch.all(coordinates[..., 1] >= expanded_bounds[..., 2]))
        self.assertTrue(torch.all(coordinates[..., 1] <= expanded_bounds[..., 3]))
        coordinates.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)

        bad_bounds = torch.tensor([0.5, -0.5, -1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "max >= min"):
            CellBoundCoordinateTransform()(torch.zeros(2), bad_bounds)

    def test_radial_projection_is_exact_for_configurable_level3_layout(self):
        torch.manual_seed(31)
        layout = default_hierarchy_target_layout(radial_boundary=0.2, coordinate_extent=0.8)
        geometry = layout.cell_geometry(3)
        bounds = torch.tensor(
            [[row["eta_min"], row["eta_max"], row["phi_min"], row["phi_max"]] for row in geometry],
            dtype=torch.float32,
        )
        radial = torch.tensor(
            [[row["radial_min"], row["radial_max"]] for row in geometry], dtype=torch.float32
        )
        raw = torch.randn(4, len(geometry), 7, 2, requires_grad=True)
        coordinates = CellBoundCoordinateTransform()(
            raw,
            bounds[None, :, None, :],
            radial[None, :, None, :],
        )
        radius = torch.linalg.vector_norm(coordinates, dim=-1)
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
        active = torch.tensor(feasible, dtype=torch.bool)[None, :, None].expand_as(radius)
        radial_min = radial[None, :, None, 0].expand_as(radius)
        radial_max = radial[None, :, None, 1].expand_as(radius)
        self.assertTrue(torch.all(radius[active] >= radial_min[active] - 2.0e-6))
        self.assertTrue(torch.all(radius[active] <= radial_max[active] + 2.0e-6))
        expanded_bounds = bounds[None, :, None, :].expand(4, -1, 7, -1)
        self.assertTrue(torch.all(coordinates[..., 0] >= expanded_bounds[..., 0] - 2.0e-6))
        self.assertTrue(torch.all(coordinates[..., 0] <= expanded_bounds[..., 1] + 2.0e-6))
        self.assertTrue(torch.all(coordinates[..., 1] >= expanded_bounds[..., 2] - 2.0e-6))
        self.assertTrue(torch.all(coordinates[..., 1] <= expanded_bounds[..., 3] + 2.0e-6))
        coordinates.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        half_coordinates = CellBoundCoordinateTransform()(
            raw.detach().half(),
            bounds[None, :, None, :],
            radial[None, :, None, :],
        )
        half_radius = torch.linalg.vector_norm(half_coordinates, dim=-1)
        self.assertEqual(half_coordinates.dtype, torch.float32)
        self.assertTrue(torch.all(half_radius[active] >= radial_min[active] - 2.0e-5))
        self.assertTrue(torch.all(half_radius[active] <= radial_max[active] + 2.0e-5))

    def test_malformed_allocation_shapes_fail_closed(self):
        parent = PositiveAccountingParameterization()(torch.zeros(2, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES)))
        with self.assertRaisesRegex(ValueError, "allocation_logits shape"):
            SoftmaxAccountingAllocator(4)(
                parent,
                torch.zeros(2, 3, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES)),
            )

    def test_half_precision_allocations_stay_finite_and_conservative(self):
        torch.manual_seed(29)
        parent = PositiveAccountingParameterization(minimum=1.0e-4)(
            torch.randn(2, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES), dtype=torch.float16)
        )
        logits = 20.0 * torch.randn(
            2,
            4,
            len(PRIMITIVE_ACCOUNTING_FIELD_NAMES),
            dtype=torch.float16,
        )
        output = SoftmaxAccountingAllocator(4)(parent, logits)
        self.assertTrue(torch.isfinite(output.children).all())
        self.assertTrue(torch.isfinite(output.primitive_fractions).all())
        self.assertTrue(torch.allclose(output.children.sum(dim=-2), parent, atol=2.0e-2, rtol=2.0e-3))
        _assert_coupled_totals(self, output.children, atol=2.0e-2)


if __name__ == "__main__":
    unittest.main()
