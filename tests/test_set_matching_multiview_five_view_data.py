import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_cache import save_hlt_cache
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.set_matching.five_view_data import (
    SET_MATCHING_FIVE_VIEW_DATA_STEP,
    FiveViewCacheArrays,
    FiveViewDatasetConfig,
    FiveViewJetDataset,
    audit_five_view_alignment,
    build_five_view_dataset_from_arrays,
    collate_five_view_samples,
    load_reconstructed_view_cache,
)
from teacher_logit_reco.set_matching.experiment import VIEW_NAMES, view_name_for_reconstructor


def make_identity_rows(n_jets=4):
    labels = np.arange(n_jets, dtype=np.int64) % 10
    jet_ids = [
        JetIdentity(file=f"ZJetsToNuNu_{index % 2}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    return labels, jet_ids


def make_tokens(n_jets=4, n_parts=6, *, offset=0.0):
    labels, _ = make_identity_rows(n_jets)
    tokens = np.zeros((n_jets, n_parts, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, n_parts), dtype=bool)
    for jet_index in range(n_jets):
        valid = min(n_parts, 3 + jet_index)
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


def make_view_array(name, *, source_type, offset=0.0, confidence_values=None):
    tokens, mask, labels = make_tokens(offset=offset)
    _, jet_ids = make_identity_rows(tokens.shape[0])
    if confidence_values is None:
        confidence = np.where(mask, 1.0, 0.0).astype(np.float32)
    else:
        confidence = np.zeros(mask.shape, dtype=np.float32)
        values = np.asarray(confidence_values, dtype=np.float32)
        confidence[:, : values.shape[0]] = values[None, :]
        confidence = np.where(mask, confidence, 0.0).astype(np.float32)
    return FiveViewCacheArrays(
        name=name,
        tokens=tokens,
        mask=mask,
        confidence=confidence,
        labels=labels,
        jet_ids=jet_ids,
        source_type=source_type,
        metadata={"view": name},
    )


def make_five_views():
    confidence = np.array([0.95, 0.04, 0.80, 0.02, 0.60, 0.01], dtype=np.float32)
    views = [
        make_view_array("hlt", source_type="original_hlt", offset=0.0),
    ]
    for index, arch in enumerate(("gt", "pn", "pfn", "pcnn"), start=1):
        views.append(
            make_view_array(
                view_name_for_reconstructor(arch),
                source_type="reconstructed",
                offset=float(index),
                confidence_values=confidence,
            )
        )
    return views


class FiveViewArrayTests(unittest.TestCase):
    def test_alignment_audit_accepts_matching_rows_and_rejects_bad_labels(self):
        views = make_five_views()
        report = audit_five_view_alignment(views)
        self.assertTrue(report["ok"])
        self.assertEqual(report["view_names"], list(VIEW_NAMES))

        bad = list(views)
        tokens, mask, labels = make_tokens(offset=5.0)
        labels = labels.copy()
        labels[1] = 9
        _, jet_ids = make_identity_rows(tokens.shape[0])
        bad[2] = FiveViewCacheArrays(
            name="pn_reco",
            tokens=tokens,
            mask=mask,
            confidence=np.where(mask, 1.0, 0.0).astype(np.float32),
            labels=labels,
            jet_ids=jet_ids,
            source_type="reconstructed",
            metadata={"view": "pn_reco"},
        )
        bad_report = audit_five_view_alignment(bad)
        self.assertFalse(bad_report["ok"])
        self.assertTrue(any("labels" in problem for problem in bad_report["problems"]))

    def test_build_dataset_filters_drops_and_shuffles_view_labels(self):
        views = make_five_views()
        config = FiveViewDatasetConfig(
            output_dir="out",
            hlt_cache_dir="hlt_cache",
            split="stack_val",
            max_tokens_per_view=3,
            min_tokens_per_view=2,
            confidence_threshold=0.50,
            drop_views=("pfn_reco",),
            shuffle_view_labels=True,
            view_label_shuffle_seed=12,
        )
        dataset = build_five_view_dataset_from_arrays(views, config=config)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.view_features.shape, (4, 5, 3, RAW_TOKEN_DIM))
        self.assertEqual(dataset.view_names, VIEW_NAMES)
        self.assertEqual(dataset.metadata["experiment_step"], SET_MATCHING_FIVE_VIEW_DATA_STEP)
        self.assertEqual(dataset.metadata["selection_mode"], "topk_or_threshold")
        self.assertFalse(dataset.view_masks[:, 3].any())
        self.assertTrue(np.all(dataset.view_confidence[:, 3] == 0.0))
        self.assertEqual(dataset.view_ids[0], 0)
        self.assertEqual(sorted(dataset.view_ids[1:].tolist()), [1, 2, 3, 4])
        self.assertFalse(np.array_equal(dataset.view_ids, np.arange(5)))
        self.assertTrue(dataset.view_masks[:, 0].sum(axis=1).min() >= 3)
        self.assertTrue(dataset.view_masks[:, 1].sum(axis=1).min() >= 2)

        sample = dataset[0]
        self.assertEqual(sample.view_features.shape, (5, 3, RAW_TOKEN_DIM))
        self.assertEqual(sample.source_indices.shape, (5, 3))
        self.assertEqual(sample.label, 0)

    def test_collate_returns_tagger_ready_shapes(self):
        config = FiveViewDatasetConfig(
            output_dir="out",
            hlt_cache_dir="hlt_cache",
            split="stack_val",
            max_tokens_per_view=4,
            confidence_threshold=0.05,
        )
        dataset = build_five_view_dataset_from_arrays(make_five_views(), config=config)
        batch = collate_five_view_samples([dataset[0], dataset[2]], as_torch=False)

        self.assertEqual(batch["view_features"].shape, (2, 5, 4, RAW_TOKEN_DIM))
        self.assertEqual(batch["view_masks"].shape, (2, 5, 4))
        self.assertEqual(batch["view_confidence"].shape, (2, 5, 4))
        self.assertEqual(batch["labels"].tolist(), [0, 2])
        self.assertEqual(batch["indices"].tolist(), [0, 2])
        self.assertEqual(batch["view_ids"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(batch["source_type_ids"].tolist(), [0, 1, 1, 1, 1])


class FiveViewCacheLoadingTests(unittest.TestCase):
    def test_load_reconstructed_view_cache_and_dataset_from_files(self):
        labels, jet_ids = make_identity_rows(3)
        tokens, mask, _ = make_tokens(n_jets=3, n_parts=5, offset=0.0)
        hlt_view = JetView(
            tokens=tokens,
            mask=mask,
            labels=labels,
            jet_ids=jet_ids,
            split="stack_val",
            metadata={"view": "fixed_hlt"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            jet_files = ["ZJetsToNuNu_0.root", "ZJetsToNuNu_1.root"]
            save_hlt_cache(hlt_view, {}, {"jet_files": jet_files}, tmp / "hlt_cache", overwrite=True)
            output_dir = tmp / "set_matching_multiview_test"
            file_indices = np.asarray([jet_files.index(jet.file) for jet in jet_ids], dtype=np.int32)
            entries = np.asarray([jet.entry for jet in jet_ids], dtype=np.int64)
            for arch_index, arch in enumerate(("gt", "pn", "pfn", "pcnn"), start=1):
                path = output_dir / "reconstructed_views" / arch / "stack_val_reconstructed_view.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                confidence = np.where(mask, 0.75, 0.0).astype(np.float32)
                np.savez_compressed(
                    path,
                    tokens=(tokens + float(arch_index)).astype(np.float32),
                    mask=mask,
                    confidence=confidence,
                    labels=labels,
                    jet_file_indices=file_indices,
                    jet_entries=entries,
                )
                metadata = {
                    "architecture": arch,
                    "split": "stack_val",
                    "jet_files": jet_files,
                }
                path.with_name(f"{path.stem}_metadata.json").write_text(json.dumps(metadata))

            loaded = load_reconstructed_view_cache(
                output_dir / "reconstructed_views" / "gt" / "stack_val_reconstructed_view.npz",
                expected_architecture="gt",
                expected_split="stack_val",
            )
            self.assertEqual(loaded.name, "gt_reco")
            self.assertEqual(loaded.tokens.shape, (3, 5, RAW_TOKEN_DIM))

            config = FiveViewDatasetConfig(
                output_dir=str(output_dir),
                hlt_cache_dir=str(tmp / "hlt_cache"),
                split="stack_val",
                max_tokens_per_view=4,
                verify_hlt_hash=False,
            )
            dataset = FiveViewJetDataset.from_caches(config)
            self.assertEqual(dataset.view_features.shape, (3, 5, 4, RAW_TOKEN_DIM))
            self.assertTrue(dataset.metadata["alignment_audit"]["ok"])


if __name__ == "__main__":
    unittest.main()
