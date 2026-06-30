import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_MODALITIES,
    LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM,
    LOCAL_COMPRESSION_MODALITY_GEOMETRY,
    LOCAL_COMPRESSION_MODALITY_IDENTITY,
    LOCAL_COMPRESSION_MODALITY_QUALITY_CONSISTENCY,
    LOCAL_COMPRESSION_MODALITY_TRACKING_ERROR,
    LocalCompressionModalities,
    LocalCompressionFeatureConfig,
    build_local_compression_canonical_inputs,
    build_local_compression_modalities,
    build_local_compression_modalities_from_tokens,
    default_local_compression_modality_specs,
)


torch = require_torch()


def make_modality_tokens():
    tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((2, 5), dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :3] = True

    # Batch 0:
    # 0: charged hadron with track info, consistent.
    tokens[0, 0, 0] = 40.0
    tokens[0, 0, 1] = 0.1
    tokens[0, 0, 2] = math.pi - 0.05
    tokens[0, 0, 3] = 40.0 * math.cosh(0.1) + 0.2
    tokens[0, 0, 4] = 1.0
    tokens[0, 0, 5] = 1.0
    tokens[0, 0, 10] = 0.03
    tokens[0, 0, 11] = 0.10
    tokens[0, 0, 12] = -0.02
    tokens[0, 0, 13] = 0.20

    # 1: neutral photon with no track signal.
    tokens[0, 1, 0] = 30.0
    tokens[0, 1, 1] = -0.2
    tokens[0, 1, 2] = -math.pi + 0.1
    tokens[0, 1, 3] = 30.0 * math.cosh(-0.2) + 0.2
    tokens[0, 1, 7] = 1.0

    # 2: photon with track-like signal.
    tokens[0, 2, 0] = 20.0
    tokens[0, 2, 1] = 0.3
    tokens[0, 2, 2] = 0.4
    tokens[0, 2, 3] = 20.0 * math.cosh(0.3) + 0.2
    tokens[0, 2, 7] = 1.0
    tokens[0, 2, 10] = 0.05

    # 3: charged PID with zero charge, intentionally inconsistent.
    tokens[0, 3, 0] = 10.0
    tokens[0, 3, 1] = -0.1
    tokens[0, 3, 2] = 0.8
    tokens[0, 3, 3] = 10.0 * math.cosh(-0.1) + 0.2
    tokens[0, 3, 5] = 1.0

    # Batch 1: simple valid charged particles.
    for idx in range(3):
        pt = 12.0 + 4.0 * idx
        eta = 0.05 * idx
        phi = -0.5 + 0.2 * idx
        tokens[1, idx, 0] = pt
        tokens[1, idx, 1] = eta
        tokens[1, idx, 2] = phi
        tokens[1, idx, 3] = pt * math.cosh(eta) + 0.2
        tokens[1, idx, 4] = 1.0
        tokens[1, idx, 5] = 1.0
        tokens[1, idx, 11] = 0.05
        tokens[1, idx, 13] = 0.06
    return tokens, mask


class LocalCompressionStep3ModalityTests(unittest.TestCase):
    def test_builds_all_default_modalities_with_finite_masked_values(self):
        tokens, mask = make_modality_tokens()
        canonical = build_local_compression_canonical_inputs(tokens, mask, max_constits=5)
        modalities = build_local_compression_modalities(canonical)

        self.assertIsInstance(modalities, LocalCompressionModalities)
        self.assertEqual(modalities.modality_names, LOCAL_COMPRESSION_MODALITIES)
        self.assertEqual(tuple(modalities.modality_mask.shape), (2, 5, len(LOCAL_COMPRESSION_MODALITIES)))
        self.assertTrue(torch.equal(modalities.particle_mask, canonical.particle_mask))
        self.assertEqual(
            modalities.summary()["active_modality_count"],
            int(mask.sum().item()) * len(LOCAL_COMPRESSION_MODALITIES),
        )
        for name in modalities.modality_names:
            values = modalities.values_by_modality[name]
            self.assertTrue(torch.isfinite(values).all())
            self.assertEqual(float(values[0, 4].abs().sum().item()), 0.0)
            self.assertEqual(float(values[1, 3:].abs().sum().item()), 0.0)

    def test_geometry_contains_phi_wrap_features_and_pt_rank(self):
        tokens, mask = make_modality_tokens()
        _canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=5)
        geometry = modalities.values_by_modality[LOCAL_COMPRESSION_MODALITY_GEOMETRY]
        fields = modalities.feature_names_by_modality[LOCAL_COMPRESSION_MODALITY_GEOMETRY]
        sin_idx = fields.index("sin_phi")
        cos_idx = fields.index("cos_phi")
        rank_idx = fields.index("pt_rank")
        log_rank_idx = fields.index("log_pt_rank")

        self.assertAlmostEqual(float(geometry[0, 0, sin_idx].item()), math.sin(math.pi - 0.05), places=5)
        self.assertAlmostEqual(float(geometry[0, 1, cos_idx].item()), math.cos(-math.pi + 0.1), places=5)
        self.assertAlmostEqual(float(geometry[0, 0, rank_idx].item()), 0.0, places=6)
        self.assertAlmostEqual(float(geometry[0, 3, rank_idx].item()), 1.0, places=6)
        self.assertGreater(float(geometry[0, 3, log_rank_idx].item()), float(geometry[0, 1, log_rank_idx].item()))

    def test_quality_consistency_features_behave_as_expected(self):
        tokens, mask = make_modality_tokens()
        _canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=5)
        quality = modalities.values_by_modality[LOCAL_COMPRESSION_MODALITY_QUALITY_CONSISTENCY]
        fields = modalities.feature_names_by_modality[LOCAL_COMPRESSION_MODALITY_QUALITY_CONSISTENCY]
        charged_consistency = fields.index("charged_pid_consistency")
        neutral_applicability = fields.index("neutral_track_applicability")
        track_error = fields.index("track_error_summary")

        self.assertEqual(float(quality[0, 0, charged_consistency].item()), 1.0)
        self.assertEqual(float(quality[0, 1, charged_consistency].item()), 1.0)
        self.assertEqual(float(quality[0, 3, charged_consistency].item()), 0.0)
        self.assertEqual(float(quality[0, 1, neutral_applicability].item()), 1.0)
        self.assertEqual(float(quality[0, 2, neutral_applicability].item()), 0.0)
        self.assertAlmostEqual(float(quality[0, 0, track_error].item()), 0.15, places=6)

    def test_modality_sources_cover_raw_pf_and_derived_fields(self):
        tokens, mask = make_modality_tokens()
        _canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=5)
        self.assertIn("raw:pt", modalities.source_names_by_modality[LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM])
        self.assertIn("pf:part_pt_log", modalities.source_names_by_modality[LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM])
        self.assertIn("derived:part_px", modalities.source_names_by_modality[LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM])
        self.assertIn("pf:part_d0", modalities.source_names_by_modality[LOCAL_COMPRESSION_MODALITY_TRACKING_ERROR])
        self.assertIn("raw:isPhoton", modalities.source_names_by_modality[LOCAL_COMPRESSION_MODALITY_IDENTITY])
        stacked = modalities.stacked_values()
        self.assertEqual(tuple(stacked.shape[:3]), (2, 5, len(LOCAL_COMPRESSION_MODALITIES)))

    def test_feature_config_rejects_reordered_modalities(self):
        specs = list(default_local_compression_modality_specs())
        specs[0], specs[1] = specs[1], specs[0]

        with self.assertRaisesRegex(ValueError, "modality order"):
            LocalCompressionFeatureConfig(modalities=tuple(specs))


if __name__ == "__main__":
    unittest.main()
