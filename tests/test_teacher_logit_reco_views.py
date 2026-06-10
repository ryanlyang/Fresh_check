import unittest

import numpy as np

from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    FileRecord,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    SplitManifest,
    RAW_TOKEN_DIM,
)
from jetclass_fresh.part_inputs import PF_VECTOR_NAMES
from teacher_logit_reco.views import (
    PairedJetViews,
    SoftReconstructedView,
    limit_split_manifest,
    make_identity_soft_view,
    make_soft_view_from_parents_and_extras,
    soft_view_to_particle_transformer_inputs,
    summarize_paired_jet_views,
    summarize_soft_view,
)


def make_tokens(n_jets=2, n_parts=4):
    tokens = np.zeros((n_jets, n_parts, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, n_parts), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 10
    for jet_index in range(n_jets):
        mask[jet_index, :3] = True
        for part_index in range(3):
            pt = 10.0 + jet_index + part_index
            eta = 0.1 * part_index
            phi = 0.2 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta) + 0.1
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.array([0.1, 0.01, -0.2, 0.02], dtype=np.float32)
    return tokens, mask, labels


def make_view(*, view_name="fixed_hlt", n_jets=2, n_parts=4):
    tokens, mask, labels = make_tokens(n_jets=n_jets, n_parts=n_parts)
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[
            JetIdentity(file=f"file_{index % 2}.root", entry=index, label=int(label))
            for index, label in enumerate(labels)
        ],
        split="model_train",
        metadata={"view": view_name},
    )


def make_manifest(n_rows=5):
    identities = [
        JetIdentity(file="ZJetsToNuNu_000.root", entry=index, label=0)
        for index in range(n_rows)
    ]
    splits = {
        "model_train": list(identities),
        "model_val": [],
        "stack_train": [],
        "stack_val": [],
        "final_test": [],
    }
    return SplitManifest(
        data_dir="/tmp/jetclass",
        max_constits=8,
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
        file_records=[FileRecord(path="ZJetsToNuNu_000.root", label=0, num_entries=20)],
        splits=splits,
    )


class TeacherLogitRecoViewStep1Tests(unittest.TestCase):
    def test_paired_views_validate_alignment_and_summary(self):
        hlt = make_view(view_name="fixed_hlt")
        offline = make_view(view_name="offline")
        pair = PairedJetViews(hlt=hlt, offline=offline)
        summary = summarize_paired_jet_views(pair)
        self.assertEqual(pair.split, "model_train")
        self.assertEqual(summary["n_jets"], 2)
        self.assertEqual(summary["hlt_shape"], [2, 4, RAW_TOKEN_DIM])

        bad_offline = make_view(view_name="offline")
        bad_offline.jet_ids[0] = JetIdentity(file="different.root", entry=99, label=0)
        with self.assertRaises(ValueError):
            PairedJetViews(hlt=hlt, offline=bad_offline)

    def test_limit_split_manifest_only_truncates_requested_split(self):
        manifest = make_manifest(n_rows=5)
        limited = limit_split_manifest(manifest, "model_train", 3)
        self.assertEqual(len(limited.splits["model_train"]), 3)
        self.assertEqual(limited.split_sizes["model_train"], 3)
        self.assertEqual(manifest.split_sizes["model_train"], 5)
        self.assertTrue(limited.metadata["limited_for_teacher_logit_reco_smoke"])

    def test_identity_soft_view_converts_to_part_inputs_with_weights(self):
        hlt = make_view(view_name="fixed_hlt", n_jets=1, n_parts=4)
        soft = make_identity_soft_view(hlt)
        inputs = soft_view_to_particle_transformer_inputs(soft)
        self.assertEqual(inputs.pf_features.shape, (1, 17, 4))
        self.assertEqual(inputs.pf_vectors.shape, (1, 4, 4))
        self.assertTrue(inputs.metadata["candidate_weights_folded"])
        self.assertEqual(summarize_soft_view(soft)["valid_candidate_count"]["mean"], 3.0)

    def test_parent_and_extra_soft_view_folds_candidate_weights(self):
        parent = make_view(view_name="fixed_hlt", n_jets=1, n_parts=4)
        parent_weights = parent.mask.astype(np.float32)
        parent_weights[0, 1] = 0.5

        extra_tokens = np.zeros((1, 2, RAW_TOKEN_DIM), dtype=np.float32)
        extra_tokens[0, 0, 0] = 3.0
        extra_tokens[0, 0, 1] = 0.1
        extra_tokens[0, 0, 2] = 0.3
        extra_tokens[0, 0, 3] = 3.1
        extra_tokens[0, 1, 0] = 7.0
        extra_tokens[0, 1, 3] = 7.0
        extra_weights = np.array([[0.25, 0.0]], dtype=np.float32)

        soft = make_soft_view_from_parents_and_extras(
            parent_tokens=parent.tokens,
            parent_mask=parent.mask,
            parent_weights=parent_weights,
            extra_tokens=extra_tokens,
            extra_weights=extra_weights,
            labels=parent.labels,
            jet_ids=parent.jet_ids,
            split=parent.split,
        )
        self.assertEqual(soft.tokens.shape, (1, 6, RAW_TOKEN_DIM))
        self.assertEqual(soft.metadata["n_parent_candidates"], 4)
        self.assertEqual(soft.metadata["n_extra_candidates"], 2)

        inputs = soft.to_particle_transformer_inputs(weight_threshold=0.0)
        px_index = PF_VECTOR_NAMES.index("part_px")
        original_parent_px = parent.tokens[0, 1, 0] * np.cos(parent.tokens[0, 1, 2])
        self.assertAlmostEqual(float(inputs.pf_vectors[0, px_index, 1]), float(0.5 * original_parent_px), places=5)
        self.assertTrue(bool(inputs.pf_mask[0, 0, 4]))
        self.assertFalse(bool(inputs.pf_mask[0, 0, 5]))

    def test_soft_view_rejects_negative_weights(self):
        hlt = make_view(view_name="fixed_hlt", n_jets=1, n_parts=4)
        weights = hlt.mask.astype(np.float32)
        weights[0, 0] = -0.1
        with self.assertRaises(ValueError):
            SoftReconstructedView(
                tokens=hlt.tokens,
                mask=hlt.mask,
                weights=weights,
                labels=hlt.labels,
                jet_ids=hlt.jet_ids,
                split=hlt.split,
            )


if __name__ == "__main__":
    unittest.main()
