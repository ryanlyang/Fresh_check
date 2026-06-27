import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength, save_hlt_cache
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.dualview_part import (
    DUALVIEW_PART_DATA_CONTRACT,
    DUALVIEW_PART_HLT_DEGRADATION_STRENGTH,
    DUALVIEW_PART_STEP3,
    DualViewPartDatasetConfig,
    DualViewPartJetDataset,
    collate_dualview_part_samples,
    make_dualview_part_loader,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def make_binary_identities(n_jets=4):
    labels = np.arange(n_jets, dtype=np.int64) % 2
    jet_ids = [
        JetIdentity(
            file="ZJetsToNuNu_0.root" if int(label) == 0 else "HToGG_0.root",
            entry=100 + index,
            label=int(label),
        )
        for index, label in enumerate(labels)
    ]
    return labels, jet_ids


def unique_files(jet_ids):
    return list(dict.fromkeys(identity.file for identity in jet_ids))


def identity_arrays(jet_ids):
    files = unique_files(jet_ids)
    file_to_index = {file: index for index, file in enumerate(files)}
    return (
        files,
        np.asarray([file_to_index[identity.file] for identity in jet_ids], dtype=np.int32),
        np.asarray([identity.entry for identity in jet_ids], dtype=np.int64),
    )


def make_tokens(n_jets=4, n_parts=6, *, offset=0.0):
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
    return tokens, mask


def write_hlt_cache(root: Path, *, split="stack_train", n_jets=4):
    labels, jet_ids = make_binary_identities(n_jets)
    tokens, mask = make_tokens(n_jets=n_jets, offset=0.0)
    hlt_view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": "fixed_hlt"},
    )
    save_hlt_cache(
        hlt_view,
        {},
        {
            "jet_files": unique_files(jet_ids),
            "view": "fixed_hlt",
            "split": split,
            "source_label_names": ["QCD", "Hgg"],
            "hlt_degradation_strength": DUALVIEW_PART_HLT_DEGRADATION_STRENGTH,
            "hlt_params": fixed_hlt_params_dict(
                fixed_hlt_params_from_strength(DUALVIEW_PART_HLT_DEGRADATION_STRENGTH)
            ),
        },
        root / "binary_inputs" / "hlt_cache",
        overwrite=True,
    )
    return labels, jet_ids, tokens, mask


def write_pn_reco_cache(root: Path, jet_ids, labels, *, split="stack_train", offset=1.0):
    tokens, mask = make_tokens(n_jets=len(jet_ids), offset=offset)
    confidence_template = np.asarray([0.95, 0.04, 0.80, 0.02, 0.60, 0.01], dtype=np.float32)
    confidence = np.zeros(mask.shape, dtype=np.float32)
    confidence[:, : confidence_template.shape[0]] = confidence_template[None, :]
    confidence = np.where(mask, confidence, 0.0).astype(np.float32)
    files, file_indices, entries = identity_arrays(jet_ids)
    path = root / "reconstructed_views" / "pn" / f"{split}_reconstructed_view.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tokens=tokens,
        mask=mask,
        confidence=confidence,
        labels=labels,
        jet_file_indices=file_indices,
        jet_entries=entries,
    )
    path.with_name(f"{path.stem}_metadata.json").write_text(
        json.dumps(
            {
                "architecture": "pn",
                "split": split,
                "jet_files": files,
            }
        ),
        encoding="utf-8",
    )
    return path


class DualViewPartStep3DataTests(unittest.TestCase):
    def make_cached_dataset(self, tmpdir, *, split="stack_train", n_jets=4):
        root = Path(tmpdir) / "dualview_part_qcd_hgg_binary_hlt0p6_true500k"
        labels, jet_ids, _, _ = write_hlt_cache(root, split=split, n_jets=n_jets)
        write_pn_reco_cache(root, jet_ids, labels, split=split)
        config = DualViewPartDatasetConfig(
            output_dir=str(root),
            split=split,
            max_pn_tokens=3,
            min_pn_tokens=2,
            confidence_threshold=0.50,
            verify_hlt_hash=False,
        )
        return DualViewPartJetDataset.from_caches(config)

    def test_loads_hlt_and_pn_reco_caches_with_alignment_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self.make_cached_dataset(tmpdir)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.hlt_tokens.shape, (4, 6, RAW_TOKEN_DIM))
        self.assertEqual(dataset.pn_reco_tokens.shape, (4, 3, RAW_TOKEN_DIM))
        self.assertEqual(dataset.metadata["experiment_step"], DUALVIEW_PART_STEP3)
        self.assertEqual(dataset.metadata["output_contract"], DUALVIEW_PART_DATA_CONTRACT)
        self.assertTrue(dataset.metadata["alignment_audit"]["ok"])
        self.assertTrue(dataset.metadata["contract_audit"]["enforced"])
        self.assertTrue(dataset.metadata["contract_audit"]["hlt_cache"]["hlt_params_match"])
        self.assertEqual(dataset.metadata["view_names"], ["hlt", "pn_reco"])
        np.testing.assert_array_equal(dataset.pn_source_indices[-1], np.asarray([0, 2, 4], dtype=np.int32))

        sample = dataset[1]
        self.assertEqual(sample.label, 1)
        self.assertEqual(sample.split, "stack_train")
        self.assertEqual(sample.pn_reco_confidence.shape, (3,))
        self.assertTrue(sample.pn_reco_mask.any())

    def test_label_filter_keeps_compact_binary_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dualview_part_qcd_hgg_binary_hlt0p6_true500k"
            labels, jet_ids, _, _ = write_hlt_cache(root, split="stack_val", n_jets=6)
            write_pn_reco_cache(root, jet_ids, labels, split="stack_val")
            dataset = DualViewPartJetDataset.from_caches(
                DualViewPartDatasetConfig(
                    output_dir=str(root),
                    split="stack_val",
                    label_filter=(1,),
                    max_pn_tokens=4,
                    min_pn_tokens=2,
                    verify_hlt_hash=False,
                )
            )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.labels.tolist(), [1, 1, 1])
        self.assertTrue(dataset.metadata["label_filter_report"]["applied"])

    def test_default_label_filter_rejects_noncompact_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dualview_part_qcd_hgg_binary_hlt0p6_true500k"
            labels, jet_ids, _, _ = write_hlt_cache(root, split="stack_train", n_jets=4)
            bad_labels = labels.copy()
            bad_labels[-1] = 3
            bad_jet_ids = [
                JetIdentity(file=identity.file, entry=identity.entry, label=int(label))
                for identity, label in zip(jet_ids, bad_labels)
            ]
            tokens, mask = make_tokens(n_jets=4, offset=0.0)
            hlt_view = JetView(
                tokens=tokens,
                mask=mask,
                labels=bad_labels,
                jet_ids=bad_jet_ids,
                split="stack_train",
                metadata={"view": "fixed_hlt"},
            )
            save_hlt_cache(
                hlt_view,
                {},
                {
                    "jet_files": unique_files(bad_jet_ids),
                    "view": "fixed_hlt",
                    "split": "stack_train",
                    "source_label_names": ["QCD", "Hgg"],
                    "hlt_params": fixed_hlt_params_dict(
                        fixed_hlt_params_from_strength(DUALVIEW_PART_HLT_DEGRADATION_STRENGTH)
                    ),
                },
                root / "binary_inputs" / "hlt_cache",
                overwrite=True,
            )
            write_pn_reco_cache(root, bad_jet_ids, bad_labels, split="stack_train")

            with self.assertRaises(ValueError):
                DualViewPartJetDataset.from_caches(
                    DualViewPartDatasetConfig(
                        output_dir=str(root),
                        split="stack_train",
                        verify_hlt_hash=False,
                    )
                )

    def test_wrong_hlt_degradation_metadata_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dualview_part_qcd_hgg_binary_hlt0p6_true500k"
            labels, jet_ids = make_binary_identities(4)
            tokens, mask = make_tokens(n_jets=4, offset=0.0)
            hlt_view = JetView(
                tokens=tokens,
                mask=mask,
                labels=labels,
                jet_ids=jet_ids,
                split="stack_train",
                metadata={"view": "fixed_hlt"},
            )
            save_hlt_cache(
                hlt_view,
                {},
                {
                    "jet_files": unique_files(jet_ids),
                    "view": "fixed_hlt",
                    "split": "stack_train",
                    "source_label_names": ["QCD", "Hgg"],
                    "hlt_params": fixed_hlt_params_dict(fixed_hlt_params_from_strength(1.0)),
                },
                root / "binary_inputs" / "hlt_cache",
                overwrite=True,
            )
            write_pn_reco_cache(root, jet_ids, labels, split="stack_train")

            with self.assertRaises(ValueError):
                DualViewPartJetDataset.from_caches(
                    DualViewPartDatasetConfig(
                        output_dir=str(root),
                        split="stack_train",
                        verify_hlt_hash=False,
                    )
                )

    def test_alignment_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dualview_part_qcd_hgg_binary_hlt0p6_true500k"
            labels, jet_ids, _, _ = write_hlt_cache(root, split="stack_train", n_jets=4)
            bad_jet_ids = list(jet_ids)
            bad_jet_ids[2] = JetIdentity(file=bad_jet_ids[2].file, entry=999, label=bad_jet_ids[2].label)
            write_pn_reco_cache(root, bad_jet_ids, labels, split="stack_train")

            with self.assertRaises(ValueError):
                DualViewPartJetDataset.from_caches(
                    DualViewPartDatasetConfig(
                        output_dir=str(root),
                        split="stack_train",
                        verify_hlt_hash=False,
                    )
                )

    def test_collate_returns_raw_and_anchor_ready_numpy_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self.make_cached_dataset(tmpdir)
            batch = collate_dualview_part_samples([dataset[0], dataset[2]], as_torch=False)

        self.assertEqual(batch["hlt_tokens"].shape, (2, 6, RAW_TOKEN_DIM))
        self.assertEqual(batch["hlt_inputs"]["features"].shape, (2, 17, 6))
        self.assertEqual(batch["hlt_inputs"]["points"].shape, (2, 2, 6))
        self.assertEqual(batch["pn_reco_tokens"].shape, (2, 3, RAW_TOKEN_DIM))
        self.assertEqual(batch["pn_reco_confidence"].shape, (2, 3))
        self.assertEqual(batch["labels"].tolist(), [0, 0])
        self.assertEqual(batch["indices"].tolist(), [0, 2])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_loader_collates_torch_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self.make_cached_dataset(tmpdir)
            loader = make_dualview_part_loader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                max_hlt_constits=4,
            )
            batch = next(iter(loader))

        self.assertIsInstance(batch["hlt_inputs"]["features"], torch.Tensor)
        self.assertEqual(tuple(batch["hlt_inputs"]["features"].shape), (2, 17, 4))
        self.assertEqual(tuple(batch["pn_reco_tokens"].shape), (2, 3, RAW_TOKEN_DIM))
        self.assertEqual(tuple(batch["labels"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
