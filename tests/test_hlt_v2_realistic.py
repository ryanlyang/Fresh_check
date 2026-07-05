import hashlib
import json
import unittest
from dataclasses import asdict

import numpy as np

from jetclass_fixed_hlt import (
    HLT_PROFILE_V2_REALISTIC,
    FixedHLTParams,
    apply_hlt_single_jet_v2_realistic,
    build_fixed_hlt_v2_realistic_view,
    scaled_fixed_hlt_v2_realistic_params,
    summarize_hlt_diagnostics,
)


def make_synthetic_tokens(n_jets=24, max_constits=16):
    tokens = np.zeros((n_jets, max_constits, 14), dtype=np.float32)
    mask = np.ones((n_jets, max_constits), dtype=bool)
    for jet in range(n_jets):
        for idx in range(max_constits):
            # Include very soft constituents so threshold severity is monotonic
            # and stable even with stochastic efficiency/merging.
            pt = 0.04 + 0.035 * idx + 0.003 * (jet % 5)
            eta = -1.2 + 0.035 * idx + 0.015 * (jet % 3)
            phi = -0.8 + 0.030 * idx + 0.010 * (jet % 4)
            tokens[jet, idx, 0] = pt
            tokens[jet, idx, 1] = eta
            tokens[jet, idx, 2] = phi
            tokens[jet, idx, 3] = pt * np.cosh(eta)
            tokens[jet, idx, 4] = 1.0 if idx % 2 == 0 else -1.0
            tokens[jet, idx, 5 + (idx % 5)] = 1.0
            tokens[jet, idx, 10:14] = np.array([0.1, 0.01, 0.2, 0.02], dtype=np.float32)
    return tokens, mask


def params_hash(params) -> str:
    payload = json.dumps(asdict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HLTv2RealisticProfileTests(unittest.TestCase):
    def test_strength_zero_is_exact_offline_identity(self):
        tokens, mask = make_synthetic_tokens(n_jets=5, max_constits=9)
        params = scaled_fixed_hlt_v2_realistic_params(0.0)

        hlt_tokens, hlt_mask, diagnostics = build_fixed_hlt_v2_realistic_view(tokens, mask, seed=1057, params=params)
        summary = summarize_hlt_diagnostics(diagnostics)

        self.assertTrue(np.array_equal(hlt_tokens, tokens))
        self.assertTrue(np.array_equal(hlt_mask, mask))
        self.assertEqual(summary["drop_total_fraction"], 0.0)
        self.assertEqual(summary["drop_eff_fraction"], 0.0)
        self.assertEqual(summary["drop_merge_fraction"], 0.0)
        self.assertEqual(summary["drop_threshold_fraction"], 0.0)

    def test_single_jet_strength_zero_preserves_nonpacked_mask_and_padding(self):
        tokens, mask = make_synthetic_tokens(n_jets=1, max_constits=6)
        tokens = tokens[0]
        mask = mask[0]
        mask[:] = np.array([True, False, True, False, True, False])
        tokens[~mask, 0] = np.array([9.0, 8.0, 7.0], dtype=np.float32)
        params = scaled_fixed_hlt_v2_realistic_params(0.0)

        out_tokens, out_mask, diagnostics = apply_hlt_single_jet_v2_realistic(
            tokens,
            mask,
            params,
            np.random.RandomState(123),
            max_constits=tokens.shape[0],
        )

        self.assertTrue(np.array_equal(out_tokens, tokens))
        self.assertTrue(np.array_equal(out_mask, mask))
        self.assertEqual(diagnostics["drop_total"], 0.0)

    def test_strength_increases_drop_severity_on_synthetic_batch(self):
        tokens, mask = make_synthetic_tokens()
        low_params = scaled_fixed_hlt_v2_realistic_params(0.5)
        mid_params = scaled_fixed_hlt_v2_realistic_params(1.0)
        high_params = scaled_fixed_hlt_v2_realistic_params(1.5)

        _, _, low_diag = build_fixed_hlt_v2_realistic_view(tokens, mask, seed=1234, params=low_params)
        _, _, mid_diag = build_fixed_hlt_v2_realistic_view(tokens, mask, seed=1234, params=mid_params)
        _, _, high_diag = build_fixed_hlt_v2_realistic_view(tokens, mask, seed=1234, params=high_params)

        low_summary = summarize_hlt_diagnostics(low_diag)
        mid_summary = summarize_hlt_diagnostics(mid_diag)
        high_summary = summarize_hlt_diagnostics(high_diag)

        self.assertLessEqual(low_summary["drop_total_fraction"], mid_summary["drop_total_fraction"])
        self.assertLessEqual(mid_summary["drop_total_fraction"], high_summary["drop_total_fraction"])
        self.assertGreater(high_summary["drop_total_fraction"], low_summary["drop_total_fraction"])
        self.assertGreaterEqual(low_summary["mean_hlt_constits"], mid_summary["mean_hlt_constits"])
        self.assertGreaterEqual(mid_summary["mean_hlt_constits"], high_summary["mean_hlt_constits"])

    def test_v1_and_v2_parameter_metadata_are_distinct(self):
        v1_params = FixedHLTParams()
        v2_params = scaled_fixed_hlt_v2_realistic_params(1.0)

        self.assertEqual(v2_params.profile_name, HLT_PROFILE_V2_REALISTIC)
        self.assertNotEqual(params_hash(v1_params), params_hash(v2_params))
        self.assertNotEqual(set(asdict(v1_params)), set(asdict(v2_params)))


if __name__ == "__main__":
    unittest.main()
