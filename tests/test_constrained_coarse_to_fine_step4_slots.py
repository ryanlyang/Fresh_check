from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

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
from teacher_logit_reco.constrained_coarse_to_fine.slots import _select_local_hlt_memory
from teacher_logit_reco.constrained_coarse_to_fine.slot_loss import (
    _pairwise_components,
    _sinkhorn_square,
    _weighted_mean,
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


def _nonpacked_targets_with_empty_cells(targets, *, minimum_targets_in_one_cell: int = 2):
    """Create a deliberately ragged target cache for batched/scalar parity."""

    batch, cells, width = targets.mask.shape
    if int(minimum_targets_in_one_cell) <= 0:
        raise ValueError("minimum_targets_in_one_cell must be positive")
    padded_width = max(4, int(width) + 1, int(minimum_targets_in_one_cell) + 1)
    coordinates = targets.local_coordinates.new_zeros(batch, cells, padded_width, 2)
    pt = targets.pt.new_zeros(batch, cells, padded_width)
    energy = targets.energy.new_zeros(batch, cells, padded_width)
    pid_index = targets.pid_index.new_zeros(batch, cells, padded_width)
    charge_index = targets.charge_index.new_zeros(batch, cells, padded_width)
    mask = torch.zeros(batch, cells, padded_width, dtype=torch.bool)
    duplicated = False
    for batch_index in range(batch):
        for cell_index in range(cells):
            sources = torch.nonzero(targets.mask[batch_index, cell_index], as_tuple=False).flatten().tolist()
            if sources and not duplicated:
                # Guarantee at least one multi-target cell and leave an interior
                # padding gap, so the parity test cannot accidentally rely on
                # compacted target masks.
                while len(sources) < int(minimum_targets_in_one_cell):
                    sources.append(sources[0])
                duplicated = True
            if not sources:
                continue
            destinations = list(range(max(0, len(sources) - 1))) + [padded_width - 1]
            for source, destination in zip(sources, destinations, strict=True):
                coordinates[batch_index, cell_index, destination] = targets.local_coordinates[
                    batch_index, cell_index, source
                ]
                pt[batch_index, cell_index, destination] = targets.pt[batch_index, cell_index, source]
                energy[batch_index, cell_index, destination] = targets.energy[batch_index, cell_index, source]
                pid_index[batch_index, cell_index, destination] = targets.pid_index[
                    batch_index, cell_index, source
                ]
                charge_index[batch_index, cell_index, destination] = targets.charge_index[
                    batch_index, cell_index, source
                ]
                mask[batch_index, cell_index, destination] = True
    assert duplicated
    return replace(
        targets,
        local_coordinates=coordinates,
        pt=pt,
        energy=energy,
        pid_index=pid_index,
        charge_index=charge_index,
        mask=mask,
    )


def _scalar_reference_slot_loss(output, targets, config):
    """The pre-batching slot-loss calculation retained for regression parity."""

    batch, views, cells, slots = output.total_pt.shape
    zero = output.total_pt.sum() * 0.0
    component_rows = {name: [] for name in ("log_pt", "coordinate", "pid", "charge", "log_energy", "uncertainty")}
    existence_rows = []
    count_rows = []
    missing_rows = []
    reliability_rows = []
    for batch_index in range(batch):
        for cell_index in range(cells):
            target_indices = torch.nonzero(targets.mask[batch_index, cell_index], as_tuple=False).flatten()
            target_count = int(target_indices.numel())
            for view_index in range(views):
                existence_target = torch.zeros(slots, device=output.total_pt.device, dtype=output.total_pt.dtype)
                if target_count:
                    pair = _pairwise_components(
                        output, targets, batch_index, view_index, cell_index, target_indices, config
                    )
                    if config.matching_mode == "sinkhorn":
                        size = max(slots, target_count)
                        square = pair["total"].new_zeros(size, size)
                        square[:slots, :target_count] = pair["total"]
                        if target_count < size:
                            square[:slots, target_count:] = F.softplus(
                                output.existence_logits[batch_index, view_index, cell_index]
                            )[:, None]
                        if slots < size:
                            square[slots:, :target_count] = float(config.missing_target_weight)
                        transport = _sinkhorn_square(square, config) * float(size)
                        real_transport = transport[:slots, :target_count]
                        existence_target = real_transport.sum(dim=-1).clamp(0.0, 1.0).detach()
                        for name in ("log_pt", "coordinate", "pid", "charge", "log_energy"):
                            component_rows[name].append(
                                _weighted_mean(pair[name], real_transport, config.epsilon)
                            )
                        missing_rows.append(transport[slots:, :target_count].sum() / max(1, target_count))
                        if output.log_sigma is not None:
                            sigma = output.log_sigma[batch_index, view_index, cell_index]
                            errors = torch.stack(
                                (pair["log_pt"], pair["deta"], pair["dphi"], pair["log_energy"], pair["pid"]),
                                dim=-1,
                            )
                            log_sigma = sigma[:, None, :].expand_as(errors)
                            nll = 0.5 * errors.square() * torch.exp(-2.0 * log_sigma) + log_sigma
                            component_rows["uncertainty"].append(
                                _weighted_mean(nll.mean(dim=-1), real_transport, config.epsilon)
                            )
                    else:
                        matched = min(slots, target_count)
                        indices = torch.arange(matched, device=output.total_pt.device)
                        existence_target[indices] = 1.0
                        for name in ("log_pt", "coordinate", "pid", "charge", "log_energy"):
                            component_rows[name].append(pair[name][indices, indices].mean())
                        missing_rows.append(pair["total"].new_tensor(float(max(0, target_count - slots))))
                logits = output.existence_logits[batch_index, view_index, cell_index]
                existence_rows.append(F.binary_cross_entropy_with_logits(logits, existence_target))
                count_rows.append((torch.sigmoid(logits).sum() - float(target_count)).abs())
                realized_error = (torch.sigmoid(logits).sum() - float(target_count)).abs().detach()
                realized_error = realized_error / max(1.0, float(target_count))
                reliability_rows.append(
                    F.smooth_l1_loss(
                        output.reliability[batch_index, view_index, cell_index].mean(),
                        1.0 - realized_error.clamp(0.0, 1.0),
                    )
                )

    def mean(rows):
        return torch.stack(rows).mean() if rows else zero

    pid_raw = F.log_softmax(output.raw_pid_logits, dim=-1)
    constrained_pid = output.pid_probabilities.detach().clamp_min(config.epsilon)
    pid_consistency = F.kl_div(pid_raw, constrained_pid, reduction="none").sum(dim=-1).mean()
    accounting_scale = torch.log1p(output.terminal_accounting.clamp_min(0.0))[:, None, :, :]
    rendered_scale = torch.log1p(output.rendered_accounting.clamp_min(0.0))
    accounting_consistency = F.smooth_l1_loss(
        rendered_scale,
        accounting_scale.expand_as(rendered_scale),
        beta=float(config.huber_beta),
    )
    if output.dust_total_pt is None:
        dust = zero
    else:
        cell_pt = output.terminal_accounting[..., ACCOUNTING_FIELD_NAMES.index("total_pT")]
        dust = (output.dust_total_pt / cell_pt[:, None, :].clamp_min(config.epsilon)).mean()
    components = {
        "matched_log_pt": mean(component_rows["log_pt"]),
        "matched_coordinate": mean(component_rows["coordinate"]),
        "matched_pid": mean(component_rows["pid"]),
        "matched_charge": mean(component_rows["charge"]),
        "matched_log_energy": mean(component_rows["log_energy"]),
        "existence": mean(existence_rows),
        "count": mean(count_rows),
        "pid_consistency": pid_consistency,
        "accounting_consistency": accounting_consistency,
        "dust": dust,
        "missing_target": mean(missing_rows),
        "uncertainty_nll": mean(component_rows["uncertainty"]),
        "reliability": mean(reliability_rows),
    }
    loss = (
        config.log_pt_weight * components["matched_log_pt"]
        + config.coordinate_weight * components["matched_coordinate"]
        + config.pid_weight * components["matched_pid"]
        + config.charge_weight * components["matched_charge"]
        + config.log_energy_weight * components["matched_log_energy"]
        + config.existence_weight * components["existence"]
        + config.count_weight * components["count"]
        + config.pid_consistency_weight * components["pid_consistency"]
        + config.accounting_consistency_weight * components["accounting_consistency"]
        + config.dust_weight * components["dust"]
        + config.missing_target_weight * components["missing_target"]
        + config.uncertainty_weight * components["uncertainty_nll"]
        + config.reliability_weight * components["reliability"]
    )
    return loss, components


class ConstrainedCoarseToFineStep4SlotTests(unittest.TestCase):
    def test_cell_local_memory_masks_cross_cell_particles_and_uses_only_zero_dummy(self):
        embedding = torch.tensor([[[3.0, -2.0]]])
        hierarchy = SimpleNamespace(
            hlt=SimpleNamespace(
                particle_embeddings=embedding,
                particle_mask=torch.tensor([[True]]),
            )
        )
        vectors = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
        empty_cell = torch.tensor([[0.20, 0.30, 0.20, 0.30]])
        radial_bounds = torch.tensor([[0.0, 1.0]])
        selected, selected_mask = _select_local_hlt_memory(
            hierarchy,
            vectors,
            empty_cell,
            radial_bounds,
            1,
            coordinate_extent=0.8,
        )
        self.assertTrue(bool(selected_mask[0, 0, 0]))
        self.assertTrue(torch.equal(selected[0, 0, 0], torch.zeros_like(selected[0, 0, 0])))

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

    def test_batched_training_sinkhorn_omits_assignment_bookkeeping_without_changing_loss(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        output = _build(C3_SINKHORN).eval()(*inputs)
        targets = _targets_for(output, offline, offline_mask, hlt, hlt_mask)
        config = ParticleSlotLossConfig(matching_mode="sinkhorn", sinkhorn_iterations=8)
        diagnostic = compute_particle_slot_loss(output.slots, targets, config, return_assignments=True)
        training = compute_particle_slot_loss(output.slots, targets, config, return_assignments=False)
        self.assertTrue(diagnostic.assignments)
        self.assertEqual(training.assignments, ())
        self.assertTrue(torch.allclose(training.loss, diagnostic.loss, atol=2.0e-6, rtol=2.0e-6))
        for name in diagnostic.components:
            self.assertTrue(
                torch.allclose(training.components[name], diagnostic.components[name], atol=2.0e-6, rtol=2.0e-6),
                name,
            )

    def test_batched_matching_matches_scalar_reference_components_and_gradients(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        for matching_mode in ("ordered", "sinkhorn"):
            with self.subTest(matching_mode=matching_mode):
                model = _build(C5_B1)
                output = model(*inputs)
                targets = _nonpacked_targets_with_empty_cells(
                    _targets_for(output, offline, offline_mask, hlt, hlt_mask)
                )
                self.assertTrue(torch.any(~targets.mask[:, :, :-1]))
                self.assertTrue(torch.any(targets.mask.sum(dim=-1) > 1))
                config = ParticleSlotLossConfig(
                    matching_mode=matching_mode,
                    sinkhorn_iterations=8,
                )
                batched = compute_particle_slot_loss(
                    output.slots,
                    targets,
                    config,
                    return_assignments=False,
                )
                reference_loss, reference_components = _scalar_reference_slot_loss(
                    output.slots,
                    targets,
                    config,
                )
                self.assertTrue(torch.allclose(batched.loss, reference_loss, atol=3.0e-6, rtol=3.0e-6))
                for name, reference in reference_components.items():
                    self.assertTrue(
                        torch.allclose(batched.components[name], reference, atol=3.0e-6, rtol=3.0e-6),
                        name,
                    )
                model.zero_grad(set_to_none=True)
                batched.loss.backward(retain_graph=True)
                batched_gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                model.zero_grad(set_to_none=True)
                reference_loss.backward()
                reference_gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                self.assertEqual(set(batched_gradients), set(reference_gradients))
                for name in batched_gradients:
                    self.assertTrue(
                        torch.allclose(
                            batched_gradients[name], reference_gradients[name], atol=5.0e-6, rtol=5.0e-6
                        ),
                        name,
                    )

    def test_batched_sinkhorn_matches_scalar_reference_for_c6_and_oversubscribed_cell(self):
        inputs, hlt, hlt_mask, offline, offline_mask = _torch_inputs()
        model = _build(C6_MULTIVIEW)
        output = model(*inputs)
        slots = output.slots
        self.assertEqual(slots.num_views, 4)
        targets = _nonpacked_targets_with_empty_cells(
            _targets_for(output, offline, offline_mask, hlt, hlt_mask),
            minimum_targets_in_one_cell=slots.num_real_slots + 1,
        )
        self.assertGreater(int(targets.mask.sum(dim=-1).max()), slots.num_real_slots)
        config = ParticleSlotLossConfig(matching_mode="sinkhorn", sinkhorn_iterations=6)
        batched = compute_particle_slot_loss(
            slots,
            targets,
            config,
            return_assignments=False,
        )
        reference_loss, reference_components = _scalar_reference_slot_loss(slots, targets, config)
        self.assertTrue(torch.allclose(batched.loss, reference_loss, atol=4.0e-6, rtol=4.0e-6))
        for name, reference in reference_components.items():
            self.assertTrue(
                torch.allclose(batched.components[name], reference, atol=4.0e-6, rtol=4.0e-6),
                name,
            )
        model.zero_grad(set_to_none=True)
        batched.loss.backward(retain_graph=True)
        batched_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        model.zero_grad(set_to_none=True)
        reference_loss.backward()
        reference_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        self.assertEqual(set(batched_gradients), set(reference_gradients))
        for name in batched_gradients:
            self.assertTrue(
                torch.allclose(
                    batched_gradients[name], reference_gradients[name], atol=7.0e-6, rtol=7.0e-6
                ),
                name,
            )

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
