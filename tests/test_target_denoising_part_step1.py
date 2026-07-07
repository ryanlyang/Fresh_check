import math
import unittest

import numpy as np

from jetclass_fixed_hlt import HLT_PROFILE_V2_REALISTIC, HLT_PROFILE_V2_REALISTIC_VERSION
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_profile
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.target_denoising_part import (
    DENOISING_TARGET_NAMES,
    TARGET_STATUS_DIRECT,
    TARGET_STATUS_NO_TARGET,
    TargetDenoisingDatasetConfig,
    TargetDenoisingPairedDataset,
    build_rank_aligned_residual_targets,
    collate_target_denoising_batch,
    wrap_delta_phi_np,
)


def _tokens(n_jets=3, n_particles=5):
    tokens = np.zeros((n_jets, n_particles, 14), dtype=np.float32)
    for row in range(n_jets):
        for col in range(n_particles):
            tokens[row, col, 0] = 5.0 + row + col
            tokens[row, col, 1] = 0.1 * row - 0.02 * col
            tokens[row, col, 2] = -2.5 + 0.4 * col
            tokens[row, col, 3] = 6.0 + row + col
            tokens[row, col, 4] = 1.0 if col % 2 == 0 else -1.0
            tokens[row, col, 5 + (col % 5)] = 1.0
    return tokens


def _view(tokens, mask, *, split="model_train", view="offline", manifest_hash="manifest-a", labels=None):
    labels = np.arange(tokens.shape[0], dtype=np.int64) % 10 if labels is None else np.asarray(labels, dtype=np.int64)
    jet_ids = [
        JetIdentity(file=f"class_{int(label)}.root", entry=1000 + index, label=int(label))
        for index, label in enumerate(labels)
    ]
    metadata = {
        "view": view,
        "source_manifest_hash": manifest_hash,
    }
    if view == "fixed_hlt":
        params = fixed_hlt_params_from_profile(HLT_PROFILE_V2_REALISTIC, 1.0)
        metadata.update(
            {
                "hlt_profile": HLT_PROFILE_V2_REALISTIC,
                "hlt_profile_version": HLT_PROFILE_V2_REALISTIC_VERSION,
                "hlt_degradation_strength": 1.0,
                "hlt_params": fixed_hlt_params_dict(params),
                "hlt_content_hash": "fake-content-hash",
            }
        )
    return JetView(
        tokens=tokens.astype(np.float32),
        mask=mask.astype(bool),
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata=metadata,
    )


class TargetDenoisingPartStep1Tests(unittest.TestCase):
    def make_views(self, *, split="model_train"):
        offline = _tokens()
        hlt = offline.copy()
        hlt[:, :, 0] *= 0.9
        hlt[:, :, 1] += 0.01
        hlt[:, :, 2] = ((hlt[:, :, 2] + 0.03 + math.pi) % (2.0 * math.pi)) - math.pi
        hlt[:, :, 3] *= 0.95
        mask = np.ones(offline.shape[:2], dtype=bool)
        mask[:, -1] = False
        return (
            _view(hlt, mask, split=split, view="fixed_hlt"),
            _view(offline, mask, split=split, view="offline"),
        )

    def make_config(self, **kwargs):
        values = {
            "manifest_path": "unused.json.gz",
            "hlt_cache_dir": "unused_cache",
            "expected_hlt_profile": HLT_PROFILE_V2_REALISTIC,
            "expected_hlt_profile_version": HLT_PROFILE_V2_REALISTIC_VERSION,
            "expected_hlt_degradation_strength": 1.0,
        }
        values.update(kwargs)
        return TargetDenoisingDatasetConfig(**values)

    def test_delta_phi_wraps_to_short_direction(self):
        delta = wrap_delta_phi_np(np.asarray([2.0 * math.pi - 0.1, -2.0 * math.pi + 0.2], dtype=np.float32))
        self.assertAlmostEqual(float(delta[0]), -0.1, places=5)
        self.assertAlmostEqual(float(delta[1]), 0.2, places=5)

    def test_identity_targets_are_zero_with_direct_status(self):
        tokens = _tokens(n_jets=2, n_particles=4)
        mask = np.ones(tokens.shape[:2], dtype=bool)
        mask[:, -1] = False
        targets = build_rank_aligned_residual_targets(tokens, mask, tokens.copy(), mask.copy())
        self.assertEqual(list(targets.target_names), list(DENOISING_TARGET_NAMES))
        self.assertTrue(np.allclose(targets.residuals, 0.0))
        self.assertTrue(np.array_equal(targets.target_mask, mask))
        self.assertTrue(np.all(targets.target_status[mask] == TARGET_STATUS_DIRECT))
        self.assertTrue(np.all(targets.target_status[~mask] == TARGET_STATUS_NO_TARGET))

    def test_dataset_enforces_hlt_contract_and_alignment(self):
        hlt_view, offline_view = self.make_views()
        dataset = TargetDenoisingPairedDataset(
            hlt_view,
            offline_view,
            config=self.make_config(),
            expected_manifest_hash="manifest-a",
        )
        self.assertEqual(len(dataset), 3)
        metadata = dataset.to_metadata()
        self.assertEqual(metadata["contract"], "target_conditioned_pairwise_denoising_dataset_v1")
        self.assertEqual(metadata["hlt_profile"], HLT_PROFILE_V2_REALISTIC)
        self.assertTrue(metadata["target_summary"]["summary_is_preview"])
        self.assertGreater(metadata["target_summary"]["target_count"], 0)

    def test_dataset_rejects_wrong_hlt_strength(self):
        hlt_view, offline_view = self.make_views()
        hlt_view.metadata["hlt_degradation_strength"] = 0.5
        with self.assertRaisesRegex(ValueError, "HLT strength mismatch"):
            TargetDenoisingPairedDataset(
                hlt_view,
                offline_view,
                config=self.make_config(),
                expected_manifest_hash="manifest-a",
            )

    def test_aligned_direct_rejects_mask_changed_hlt_without_provenance(self):
        hlt_view, offline_view = self.make_views()
        hlt_view.mask[0, 1] = False
        offline_view.mask[0, 1] = True
        with self.assertRaisesRegex(ValueError, "aligned_direct target denoising needs per-particle HLT provenance"):
            TargetDenoisingPairedDataset(
                hlt_view,
                offline_view,
                config=self.make_config(),
                expected_manifest_hash="manifest-a",
            )

    def test_dataset_rejects_final_test_targets_without_guard(self):
        hlt_view, offline_view = self.make_views(split="final_test")
        with self.assertRaisesRegex(ValueError, "final_test offline denoising targets are guarded"):
            TargetDenoisingPairedDataset(
                hlt_view,
                offline_view,
                config=self.make_config(split="final_test"),
                expected_manifest_hash="manifest-a",
            )

    def test_collate_builds_canonical_inputs_and_targets(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        hlt_view, offline_view = self.make_views()
        dataset = TargetDenoisingPairedDataset(
            hlt_view,
            offline_view,
            config=self.make_config(),
            expected_manifest_hash="manifest-a",
        )
        batch = collate_target_denoising_batch([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["features"].shape), (2, len(PF_FEATURE_NAMES), 5))
        self.assertEqual(tuple(batch["hlt_feature_rows"].shape), (2, 5, len(PF_FEATURE_NAMES)))
        self.assertEqual(tuple(batch["target_residuals"].shape), (2, 5, len(DENOISING_TARGET_NAMES)))
        self.assertEqual(tuple(batch["target_mask"].shape), (2, 5))
        self.assertTrue(bool(batch["target_mask"][:, :-1].all()))
        self.assertFalse(bool(batch["target_mask"][:, -1].any()))


if __name__ == "__main__":
    unittest.main()
