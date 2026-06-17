import unittest

import numpy as np

from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    FileRecord,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SplitManifest,
)
from teacher_logit_reco.set_matching.data import (
    SetMatchingJetDataset,
    audit_set_matching_pair,
    build_set_matching_feature_normalization,
    collate_set_matching_samples,
    compute_feature_normalization_stats,
)
from teacher_logit_reco.views import PairedJetViews


def make_tokens(n_jets=3, n_parts=5, *, offset=0.0):
    tokens = np.zeros((n_jets, n_parts, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, n_parts), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 10
    for jet_index in range(n_jets):
        valid = min(n_parts, 2 + jet_index)
        mask[jet_index, :valid] = True
        for part_index in range(valid):
            pt = 10.0 + offset + jet_index + part_index
            eta = 0.05 * (jet_index - part_index)
            phi = 0.2 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.array([0.1, 0.01, -0.2, 0.02], dtype=np.float32)
    return tokens, mask, labels


def make_view(*, view_name, n_jets=3, n_parts=5, offset=0.0, split="model_train"):
    tokens, mask, labels = make_tokens(n_jets=n_jets, n_parts=n_parts, offset=offset)
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[
            JetIdentity(file=f"ZJetsToNuNu_{index % 2}.root", entry=index, label=int(label))
            for index, label in enumerate(labels)
        ],
        split=split,
        metadata={"view": view_name},
    )


def make_pair(n_jets=3, n_parts=5):
    hlt = make_view(view_name="fixed_hlt", n_jets=n_jets, n_parts=n_parts, offset=0.0)
    offline = make_view(view_name="offline", n_jets=n_jets, n_parts=n_parts, offset=1.0)
    return PairedJetViews(hlt=hlt, offline=offline)


def make_manifest(pair):
    splits = {
        "model_train": list(pair.jet_ids),
        "model_val": [],
        "stack_train": [],
        "stack_val": [],
        "final_test": [],
    }
    return SplitManifest(
        data_dir="/tmp/jetclass",
        max_constits=pair.hlt.tokens.shape[1],
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: len(rows) for split, rows in splits.items()},
        split_seeds={
            "model_train": 153,
            "model_val": 254,
            "stack_train": 356,
            "stack_val": 457,
            "final_test": 558,
        },
        file_records=[FileRecord(path="ZJetsToNuNu_0.root", label=0, num_entries=20)],
        splits=splits,
    )


class SetMatchingDatasetTests(unittest.TestCase):
    def test_dataset_returns_aligned_hlt_offline_samples(self):
        pair = make_pair(n_jets=3, n_parts=5)
        dataset = SetMatchingJetDataset(pair)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.feature_dim, RAW_TOKEN_DIM)
        self.assertEqual(dataset.metadata["dataset"], "set_matching_paired_hlt_offline")

        sample = dataset[1]
        self.assertEqual(sample.hlt_tokens.shape, (5, RAW_TOKEN_DIM))
        self.assertEqual(sample.offline_tokens.shape, (5, RAW_TOKEN_DIM))
        self.assertEqual(sample.label, 1)
        self.assertEqual(sample.jet_id, pair.jet_ids[1])
        self.assertEqual(int(np.sum(sample.hlt_mask)), 3)
        self.assertEqual(int(np.sum(sample.offline_mask)), 3)

    def test_trimmed_dataset_and_collate_pad_variable_particle_counts(self):
        pair = make_pair(n_jets=3, n_parts=6)
        dataset = SetMatchingJetDataset(pair, trim_to_valid=True)
        samples = [dataset[0], dataset[2]]
        self.assertEqual(samples[0].hlt_tokens.shape[0], 2)
        self.assertEqual(samples[1].hlt_tokens.shape[0], 4)

        batch = collate_set_matching_samples(samples, as_torch=False)
        self.assertEqual(batch["hlt_tokens"].shape, (2, 4, RAW_TOKEN_DIM))
        self.assertEqual(batch["offline_tokens"].shape, (2, 4, RAW_TOKEN_DIM))
        self.assertEqual(batch["hlt_mask"].tolist(), [[True, True, False, False], [True, True, True, True]])
        self.assertEqual(batch["offline_mask"].tolist(), [[True, True, False, False], [True, True, True, True]])
        self.assertEqual(batch["labels"].tolist(), [0, 2])
        self.assertEqual(batch["split"], "model_train")

    def test_collate_rejects_cross_split_batch(self):
        pair = make_pair(n_jets=2, n_parts=4)
        dataset = SetMatchingJetDataset(pair, trim_to_valid=True)
        left = dataset[0]
        right = dataset[1]
        object.__setattr__(right, "split", "model_val")
        with self.assertRaises(ValueError):
            collate_set_matching_samples([left, right], as_torch=False)


class SetMatchingFeatureStatsAndAuditTests(unittest.TestCase):
    def test_feature_normalization_uses_only_masked_particles(self):
        pair = make_pair(n_jets=2, n_parts=4)
        stats = compute_feature_normalization_stats(pair.hlt.tokens, pair.hlt.mask, source_view="fixed_hlt")
        expected_values = pair.hlt.tokens[pair.hlt.mask]
        self.assertEqual(stats.count, expected_values.shape[0])
        self.assertTrue(np.allclose(stats.mean, expected_values.mean(axis=0)))
        self.assertTrue(np.all(stats.std > 0.0))
        transformed = stats.transform(pair.hlt.tokens)
        self.assertEqual(transformed.shape, pair.hlt.tokens.shape)
        self.assertEqual(stats.to_dict()["source_view"], "fixed_hlt")

    def test_builds_hlt_offline_and_combined_stats(self):
        pair = make_pair(n_jets=2, n_parts=4)
        dataset = SetMatchingJetDataset(pair)
        stats = build_set_matching_feature_normalization(dataset)
        self.assertEqual(set(stats), {"hlt", "offline", "combined"})
        self.assertEqual(stats["hlt"].metadata["split"], "model_train")
        self.assertEqual(stats["offline"].source_view, "offline")
        self.assertEqual(
            stats["combined"].count,
            int(np.sum(pair.hlt.mask) + np.sum(pair.offline.mask)),
        )

    def test_identity_audit_reports_manifest_alignment(self):
        pair = make_pair(n_jets=3, n_parts=5)
        dataset = SetMatchingJetDataset(pair)
        report = audit_set_matching_pair(dataset, manifest=make_manifest(pair), expected_split="model_train")
        self.assertTrue(report["ok"])
        self.assertEqual(report["duplicate_identity_count"], 0)
        self.assertEqual(report["label_mismatch_count"], 0)
        self.assertTrue(report["matches_manifest_split_prefix"])
        self.assertEqual(report["hlt_shape"], [3, 5, RAW_TOKEN_DIM])

    def test_identity_audit_detects_duplicate_or_bad_labels(self):
        pair = make_pair(n_jets=3, n_parts=5)
        pair.hlt.jet_ids[2] = pair.hlt.jet_ids[0]
        pair.offline.jet_ids[2] = pair.offline.jet_ids[0]
        pair.hlt.labels[1] = 9
        pair.offline.labels[1] = 9
        dataset = SetMatchingJetDataset(pair)
        report = audit_set_matching_pair(dataset)
        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_identity_count"], 1)
        self.assertEqual(report["label_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
