from __future__ import annotations

import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_FIELD_NAMES,
    CANONICAL_STATE_PHI_BUILDER_VERSION,
    build_canonical_jet_state_phi,
    build_canonical_jet_state_phi_from_view,
    default_canonical_jet_state_layout,
)


def _field(name: str) -> int:
    return CANONICAL_STATE_FIELD_NAMES.index(name)


def _tokens(n_jets: int = 1, max_particles: int = 6) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((n_jets, max_particles, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, max_particles), dtype=bool)
    return tokens, mask


def _set_particle(
    tokens: np.ndarray,
    mask: np.ndarray,
    jet: int,
    part: int,
    *,
    pt: float,
    eta: float,
    phi: float,
    energy: float | None = None,
    charge: float = 0.0,
    pid: int = 6,
) -> None:
    tokens[jet, part, 0] = float(pt)
    tokens[jet, part, 1] = float(eta)
    tokens[jet, part, 2] = float(phi)
    tokens[jet, part, 3] = float(energy if energy is not None else pt * np.cosh(eta))
    tokens[jet, part, 4] = float(charge)
    tokens[jet, part, pid] = 1.0
    mask[jet, part] = True


class CanonicalStateStep3PhiTests(unittest.TestCase):
    def test_phi_shapes_are_finite_and_metadata_matches_layout(self):
        tokens, mask = _tokens(n_jets=2)
        _set_particle(tokens, mask, 0, 0, pt=2.0, eta=0.01, phi=0.00, charge=1.0, pid=5)
        _set_particle(tokens, mask, 0, 1, pt=1.0, eta=0.04, phi=0.00, pid=7)
        _set_particle(tokens, mask, 1, 0, pt=3.0, eta=0.20, phi=0.10, pid=6)

        out = build_canonical_jet_state_phi(tokens, mask, source_metadata={"source_view": "hlt"})
        layout = default_canonical_jet_state_layout()

        self.assertEqual(out.phi_tokens.shape, (2, layout.k_state, layout.d_phi))
        self.assertEqual(out.state_mask.shape, (2, layout.k_state))
        self.assertTrue(np.isfinite(out.phi_tokens).all())
        self.assertTrue(out.diagnostics["all_finite"])
        self.assertEqual(out.diagnostics["builder_version"], CANONICAL_STATE_PHI_BUILDER_VERSION)
        self.assertEqual(out.layout_metadata["token_names"], list(layout.token_names))
        self.assertEqual(out.source_metadata["source_view"], "hlt")

    def test_simple_hand_computed_global_and_radial_fields(self):
        tokens, mask = _tokens()
        _set_particle(tokens, mask, 0, 0, pt=1.0, eta=0.01, phi=0.00, energy=1.0, charge=1.0, pid=5)
        _set_particle(tokens, mask, 0, 1, pt=3.0, eta=0.05, phi=0.00, energy=3.0, pid=7)

        out = build_canonical_jet_state_phi(tokens, mask)
        phi = out.phi_tokens[0]
        state_mask = out.state_mask[0]

        global_energy = phi[0]
        self.assertTrue(state_mask[0])
        self.assertAlmostEqual(global_energy[_field("sum_pt_frac")], 1.0, places=6)
        self.assertAlmostEqual(global_energy[_field("sum_energy_frac")], 1.0, places=6)
        self.assertAlmostEqual(global_energy[_field("log1p_count")], np.log1p(2.0) / 5.0, places=6)
        self.assertAlmostEqual(global_energy[_field("mean_pt_frac")], 0.5, places=6)
        self.assertAlmostEqual(global_energy[_field("max_pt_frac")], 0.75, places=6)
        self.assertAlmostEqual(global_energy[_field("charged_pt_frac")], 0.25, places=6)
        self.assertAlmostEqual(global_energy[_field("photon_pt_frac")], 0.75, places=6)
        self.assertAlmostEqual(global_energy[_field("hadron_pt_frac")], 0.25, places=6)

        ring0 = phi[4]
        ring1 = phi[5]
        self.assertTrue(state_mask[4])
        self.assertTrue(state_mask[5])
        self.assertAlmostEqual(ring0[_field("sum_pt_frac")], 0.25, places=6)
        self.assertAlmostEqual(ring1[_field("sum_pt_frac")], 0.75, places=6)
        self.assertAlmostEqual(ring0[_field("charged_pt_frac")], 1.0, places=6)
        self.assertAlmostEqual(ring1[_field("photon_pt_frac")], 1.0, places=6)

    def test_masked_padding_is_ignored_even_if_values_are_nonzero(self):
        tokens, mask = _tokens()
        _set_particle(tokens, mask, 0, 0, pt=2.0, eta=0.01, phi=0.0, energy=2.0, pid=6)
        tokens[0, 5, 0] = 1000.0
        tokens[0, 5, 3] = 1000.0

        out = build_canonical_jet_state_phi(tokens, mask)
        global_energy = out.phi_tokens[0, 0]

        self.assertAlmostEqual(global_energy[_field("sum_pt_frac")], 1.0, places=6)
        self.assertAlmostEqual(global_energy[_field("max_pt_frac")], 1.0, places=6)
        self.assertEqual(out.diagnostics["masked_particle_counts"], [1])
        self.assertEqual(out.diagnostics["valid_particle_counts"], [1])

    def test_empty_jet_returns_zero_phi_and_false_state_mask(self):
        tokens, mask = _tokens()

        out = build_canonical_jet_state_phi(tokens, mask)

        self.assertFalse(out.state_mask.any())
        self.assertTrue(np.all(out.phi_tokens == 0.0))
        self.assertEqual(out.diagnostics["valid_particle_counts"], [0])
        self.assertEqual(out.diagnostics["state_valid_counts"], [0])

    def test_nonfinite_particles_are_sanitized_and_reported(self):
        tokens, mask = _tokens()
        _set_particle(tokens, mask, 0, 0, pt=2.0, eta=0.01, phi=0.0, energy=2.0, pid=6)
        _set_particle(tokens, mask, 0, 1, pt=1.0, eta=np.nan, phi=0.0, energy=1.0, pid=7)

        out = build_canonical_jet_state_phi(tokens, mask)

        self.assertTrue(np.isfinite(out.phi_tokens).all())
        self.assertEqual(out.diagnostics["valid_particle_counts"], [1])
        self.assertEqual(out.diagnostics["masked_particle_counts"], [2])
        self.assertLess(out.diagnostics["finite_particle_fraction"][0], 1.0)
        self.assertGreater(out.phi_tokens[0, 0, _field("quality_or_missingness_proxy")], 0.0)

    def test_hlt_and_offline_views_use_same_builder_path(self):
        tokens, mask = _tokens()
        _set_particle(tokens, mask, 0, 0, pt=2.0, eta=0.02, phi=0.01, energy=2.0, pid=6)
        jet_ids = [JetIdentity(file="toy.root", entry=0, label=0)]
        hlt_view = JetView(tokens=tokens, mask=mask, labels=np.asarray([0]), jet_ids=jet_ids, split="model_val", metadata={"source_view": "hlt"})
        off_view = JetView(tokens=tokens.copy(), mask=mask.copy(), labels=np.asarray([0]), jet_ids=jet_ids, split="model_val", metadata={"source_view": "offline"})

        hlt_out = build_canonical_jet_state_phi_from_view(hlt_view)
        off_out = build_canonical_jet_state_phi_from_view(off_view)

        np.testing.assert_allclose(hlt_out.phi_tokens, off_out.phi_tokens)
        np.testing.assert_array_equal(hlt_out.state_mask, off_out.state_mask)
        self.assertEqual(hlt_out.source_metadata["source_view"], "hlt")
        self.assertEqual(off_out.source_metadata["source_view"], "offline")

    def test_phi_is_deterministic_and_single_jet_input_supported(self):
        tokens, mask = _tokens()
        _set_particle(tokens, mask, 0, 0, pt=2.0, eta=0.02, phi=0.01, energy=2.0, pid=6)
        _set_particle(tokens, mask, 0, 1, pt=1.0, eta=0.11, phi=-0.03, energy=1.0, pid=8)

        first = build_canonical_jet_state_phi(tokens, mask)
        second = build_canonical_jet_state_phi(tokens, mask)
        single = build_canonical_jet_state_phi(tokens[0], mask[0])

        np.testing.assert_array_equal(first.phi_tokens, second.phi_tokens)
        np.testing.assert_array_equal(first.state_mask, second.state_mask)
        np.testing.assert_allclose(first.phi_tokens[0], single.phi_tokens)
        np.testing.assert_array_equal(first.state_mask[0], single.state_mask)
        self.assertTrue(single.diagnostics["squeezed_input"])

    def test_anchor_tokens_use_fixed_soft_assignment_not_top_pt_slots(self):
        tokens, mask = _tokens(max_particles=4)
        _set_particle(tokens, mask, 0, 0, pt=1.0, eta=0.00, phi=0.00, energy=1.0, pid=6)
        _set_particle(tokens, mask, 0, 1, pt=5.0, eta=0.02, phi=0.01, energy=5.0, pid=7)

        out = build_canonical_jet_state_phi(tokens, mask)
        coarse0 = out.phi_tokens[0, 20]

        self.assertTrue(out.state_mask[0, 20])
        self.assertTrue(out.state_mask[0, 22])
        self.assertGreater(coarse0[_field("sum_pt_frac")], 0.0)
        self.assertLess(coarse0[_field("sum_pt_frac")], 1.0)


if __name__ == "__main__":
    unittest.main()
