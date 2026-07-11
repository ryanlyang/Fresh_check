from __future__ import annotations

import json
import unittest

from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_ANCHOR_COUNTS,
    CANONICAL_STATE_ANGULAR_SECTORS,
    CANONICAL_STATE_FIELD_NAMES,
    CANONICAL_STATE_GLOBAL_TOKENS,
    CANONICAL_STATE_LAYOUT_VERSION,
    CANONICAL_STATE_RADIAL_BIN_EDGES,
    CANONICAL_STATE_RESIDUAL_SCALES,
    CANONICAL_STATE_TOKEN_FAMILIES,
    CANONICAL_STATE_TOKEN_TYPE_IDS,
    CanonicalJetStateConfig,
    CanonicalJetStateLayout,
    build_token_specs,
    canonical_jet_state_layout_manifest,
    default_canonical_jet_state_config,
    default_canonical_jet_state_layout,
)


class CanonicalStateStep2LayoutTests(unittest.TestCase):
    def test_field_order_is_stable_and_expected(self):
        cfg = default_canonical_jet_state_config()
        self.assertEqual(cfg.layout_version, CANONICAL_STATE_LAYOUT_VERSION)
        self.assertEqual(cfg.field_names, CANONICAL_STATE_FIELD_NAMES)
        self.assertEqual(
            cfg.field_names,
            (
                "sum_pt_frac",
                "sum_energy_frac",
                "log1p_count",
                "mean_pt_frac",
                "max_pt_frac",
                "pt_weighted_mean_deta",
                "pt_weighted_mean_dphi",
                "pt_weighted_var_deta",
                "pt_weighted_var_dphi",
                "mass_proxy",
                "width_proxy",
                "charged_pt_frac",
                "neutral_pt_frac",
                "photon_pt_frac",
                "electron_pt_frac",
                "muon_pt_frac",
                "hadron_pt_frac",
                "quality_or_missingness_proxy",
            ),
        )
        self.assertEqual(cfg.d_phi, 18)

    def test_config_rejects_field_or_layout_drift(self):
        bad_configs = [
            {"layout_version": "future"},
            {"field_names": tuple(reversed(CANONICAL_STATE_FIELD_NAMES))},
            {"global_tokens": tuple(reversed(CANONICAL_STATE_GLOBAL_TOKENS))},
            {"radial_bin_edges": (0.0, 0.1, None)},
            {"radial_bin_edges": (0.0, 0.03, 0.02, 0.10, 0.15, 0.22, 0.30, 0.40, None)},
            {"angular_sectors": 12},
            {"anchor_counts": {**CANONICAL_STATE_ANCHOR_COUNTS, "anchor_fine": 15}},
            {"residual_scales": {**CANONICAL_STATE_RESIDUAL_SCALES, "sum_pt_frac": 0.0}},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CanonicalJetStateConfig(**kwargs)

    def test_token_order_and_counts_are_stable(self):
        layout = default_canonical_jet_state_layout()
        specs = layout.token_specs

        self.assertEqual(layout.k_state, 48)
        self.assertEqual(layout.d_phi, 18)
        self.assertEqual(layout.family_slices()["global"], (0, 4))
        self.assertEqual(layout.family_slices()["radial"], (4, 12))
        self.assertEqual(layout.family_slices()["angular"], (12, 20))
        self.assertEqual(layout.family_slices()["anchor_coarse"], (20, 24))
        self.assertEqual(layout.family_slices()["anchor_medium"], (24, 32))
        self.assertEqual(layout.family_slices()["anchor_fine"], (32, 48))
        self.assertEqual(layout.token_names[:4], CANONICAL_STATE_GLOBAL_TOKENS)
        self.assertEqual(layout.token_names[4], "radial_ring_00")
        self.assertEqual(layout.token_names[11], "radial_ring_07")
        self.assertEqual(layout.token_names[12], "angular_sector_00")
        self.assertEqual(layout.token_names[19], "angular_sector_07")
        self.assertEqual(layout.token_names[20], "anchor_coarse_slot_00")
        self.assertEqual(layout.token_names[47], "anchor_fine_slot_15")

        for index, spec in enumerate(specs):
            self.assertEqual(spec.index, index)
            self.assertEqual(spec.token_type_id, CANONICAL_STATE_TOKEN_TYPE_IDS[spec.family])
            self.assertEqual(spec.scale_id, CANONICAL_STATE_TOKEN_TYPE_IDS[spec.family])

    def test_token_metadata_contains_geometry(self):
        layout = default_canonical_jet_state_layout()
        radial = layout.token_specs[4:12]
        angular = layout.token_specs[12:20]
        anchors = layout.token_specs[20:]

        self.assertEqual(radial[0].radius_inner, 0.0)
        self.assertEqual(radial[0].radius_outer, 0.03)
        self.assertEqual(radial[-1].radius_inner, 0.40)
        self.assertIsNone(radial[-1].radius_outer)
        self.assertEqual(tuple(spec.ring_id for spec in radial), tuple(range(8)))
        self.assertEqual(tuple(spec.sector_id for spec in angular), tuple(range(CANONICAL_STATE_ANGULAR_SECTORS)))
        for spec in angular:
            self.assertIsNotNone(spec.angular_center)
            self.assertIsNotNone(spec.angular_width)
        self.assertEqual(anchors[0].anchor_radius, 0.30)
        self.assertEqual(anchors[4].anchor_radius, 0.18)
        self.assertEqual(anchors[12].anchor_radius, 0.10)
        for spec in anchors:
            self.assertIsNotNone(spec.anchor_deta)
            self.assertIsNotNone(spec.anchor_dphi)
        self.assertAlmostEqual(anchors[4].anchor_deta, 0.0)
        self.assertAlmostEqual(anchors[4].anchor_dphi, 0.0)
        self.assertEqual(len({(spec.anchor_deta, spec.anchor_dphi) for spec in anchors[-16:]}), 16)

    def test_metadata_roundtrip_is_lossless(self):
        layout = default_canonical_jet_state_layout()
        payload = layout.to_dict()
        encoded = json.dumps(payload, sort_keys=True)
        decoded = json.loads(encoded)
        roundtrip = CanonicalJetStateLayout.from_dict(decoded)

        self.assertEqual(roundtrip.to_dict(), payload)
        self.assertEqual(roundtrip.token_names, layout.token_names)
        self.assertEqual(roundtrip.feature_scale_vector(), layout.feature_scale_vector())
        self.assertEqual(roundtrip.residual_scale_vector(), layout.residual_scale_vector())
        self.assertEqual(canonical_jet_state_layout_manifest(), payload)

    def test_residual_scales_are_positive_bounded_and_field_aligned(self):
        layout = default_canonical_jet_state_layout()
        residual = layout.residual_scale_vector()
        feature = layout.feature_scale_vector()

        self.assertEqual(len(residual), layout.d_phi)
        self.assertEqual(len(feature), layout.d_phi)
        self.assertTrue(all(0.0 < value <= 1.0 for value in residual))
        self.assertTrue(all(value > 0.0 for value in feature))
        self.assertLessEqual(layout.config.residual_scales["pt_weighted_var_deta"], 0.05)
        self.assertGreaterEqual(layout.config.residual_scales["log1p_count"], 0.5)
        self.assertLess(
            layout.config.residual_scales["electron_pt_frac"],
            layout.config.residual_scales["charged_pt_frac"],
        )

    def test_build_token_specs_matches_layout(self):
        cfg = default_canonical_jet_state_config()
        specs = build_token_specs(cfg)
        layout = CanonicalJetStateLayout(cfg, specs)

        self.assertEqual(len(specs), 48)
        self.assertEqual(layout.token_specs, specs)
        self.assertEqual(tuple(spec.family for spec in specs[:4]), ("global",) * 4)
        self.assertEqual(tuple(spec.family for spec in specs[-16:]), ("anchor_fine",) * 16)
        self.assertEqual(set(CANONICAL_STATE_TOKEN_FAMILIES), {spec.family for spec in specs})


if __name__ == "__main__":
    unittest.main()
