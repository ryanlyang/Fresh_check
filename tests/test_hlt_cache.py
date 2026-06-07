from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    audit_hlt_cache,
    fixed_hlt_params_dict,
    generate_and_cache_hlt_view,
    load_cached_hlt_view,
    load_hlt_metadata,
)
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    FileRecord,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    SplitManifest,
    manifest_hash,
)


def make_small_offline_view(source_manifest_hash=None):
    tokens = np.zeros((3, 6, 14), dtype=np.float32)
    mask = np.zeros((3, 6), dtype=bool)
    labels = np.array([0, 1, 9], dtype=np.int64)
    jet_ids = [
        JetIdentity(file="ZJetsToNuNu_000.root", entry=2, label=0),
        JetIdentity(file="HToBB_000.root", entry=4, label=1),
        JetIdentity(file="TTBarLep_000.root", entry=6, label=9),
    ]

    for jet_index in range(3):
        n_constits = 4
        mask[jet_index, :n_constits] = True
        for part_index in range(n_constits):
            pt = 1.8 + 0.4 * part_index + 0.2 * jet_index
            eta = -0.3 + 0.15 * part_index
            phi = -0.2 + 0.18 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            tokens[jet_index, part_index, 4] = 1.0 if part_index % 2 == 0 else 0.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.array([0.1, 0.01, 0.2, 0.02], dtype=np.float32)

    metadata = {"view": "offline"}
    if source_manifest_hash is not None:
        metadata["source_manifest_hash"] = source_manifest_hash
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split="model_train",
        metadata=metadata,
    )


def make_small_manifest(jet_ids):
    splits = {
        "model_train": list(jet_ids),
        "model_val": [],
        "stack_train": [],
        "stack_val": [],
        "final_test": [],
    }
    split_sizes = {split: len(rows) for split, rows in splits.items()}
    return SplitManifest(
        data_dir="/tmp/jetclass",
        max_constits=6,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes=split_sizes,
        split_seeds={
            "model_train": 153,
            "model_val": 254,
            "stack_train": 356,
            "stack_val": 457,
            "final_test": 558,
        },
        file_records=[
            FileRecord(path="ZJetsToNuNu_000.root", label=0, num_entries=10),
            FileRecord(path="HToBB_000.root", label=1, num_entries=10),
            FileRecord(path="TTBarLep_000.root", label=9, num_entries=10),
        ],
        splits=splits,
    )


class HLTCacheStep3Tests(unittest.TestCase):
    def test_fixed_hlt_cache_roundtrip(self):
        base_view = make_small_offline_view()
        manifest = make_small_manifest(base_view.jet_ids)
        offline_view = make_small_offline_view(source_manifest_hash=manifest_hash(manifest))

        with tempfile.TemporaryDirectory() as tmp:
            metadata = generate_and_cache_hlt_view(
                offline_view,
                tmp,
                seed=DEFAULT_HLT_SEEDS["model_train"],
            )
            loaded = load_cached_hlt_view(tmp, "model_train")
            audit = audit_hlt_cache(manifest, tmp, splits=["model_train"])

        self.assertEqual(loaded.tokens.shape, offline_view.tokens.shape)
        self.assertEqual(loaded.mask.shape, offline_view.mask.shape)
        self.assertEqual(loaded.jet_ids, offline_view.jet_ids)
        self.assertEqual(metadata["hlt_params"], fixed_hlt_params_dict())
        self.assertEqual(metadata["seed"], DEFAULT_HLT_SEEDS["model_train"])
        self.assertIn("drop_total_fraction", metadata["hlt_diagnostics_summary"])
        self.assertTrue(audit["ok"])

    def test_fixed_hlt_cache_is_deterministic_for_same_seed(self):
        offline_view = make_small_offline_view()
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            meta_left = generate_and_cache_hlt_view(
                offline_view,
                left,
                seed=DEFAULT_HLT_SEEDS["model_train"],
            )
            meta_right = generate_and_cache_hlt_view(
                offline_view,
                right,
                seed=DEFAULT_HLT_SEEDS["model_train"],
            )
            view_left = load_cached_hlt_view(left, "model_train")
            view_right = load_cached_hlt_view(right, "model_train")

        self.assertEqual(meta_left["hlt_content_hash"], meta_right["hlt_content_hash"])
        self.assertTrue(np.array_equal(view_left.tokens, view_right.tokens))
        self.assertTrue(np.array_equal(view_left.mask, view_right.mask))

    def test_metadata_loader_reads_json_sidecar(self):
        offline_view = make_small_offline_view()
        with tempfile.TemporaryDirectory() as tmp:
            generate_and_cache_hlt_view(
                offline_view,
                tmp,
                seed=DEFAULT_HLT_SEEDS["model_train"],
            )
            metadata = load_hlt_metadata(Path(tmp), "model_train")
        self.assertEqual(metadata["view"], "fixed_hlt")
        self.assertEqual(metadata["n_jets"], 3)


if __name__ == "__main__":
    unittest.main()
