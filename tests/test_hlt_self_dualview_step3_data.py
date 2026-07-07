import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view, save_hlt_cache
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.hlt_self_dualview import (
    HLT_SDV_BRANCH2_HLT2,
    HLT_SDV_BRANCH2_SAME_HLT,
    HLT_SDV_DATASET_CONTRACT,
    HLT_SDV_VARIANT_SAME_VIEW,
    HLTSelfDualViewDataset,
    align_hlt_sdv_views,
    assert_hlt_sdv_batch_deployable,
    collate_hlt_sdv_batch,
    hlt_sdv_branch2_mode_from_variant,
    hlt_sdv_dual_hlt2_variant_name,
    load_hlt_sdv_dataset,
)


def _identity_arrays(jet_ids):
    unique_files = []
    file_to_index = {}
    file_indices = np.zeros((len(jet_ids),), dtype=np.int32)
    entries = np.zeros((len(jet_ids),), dtype=np.int64)
    for index, identity in enumerate(jet_ids):
        if identity.file not in file_to_index:
            file_to_index[identity.file] = len(unique_files)
            unique_files.append(identity.file)
        file_indices[index] = file_to_index[identity.file]
        entries[index] = int(identity.entry)
    return unique_files, file_indices, entries


def _content_hash(view):
    _, file_indices, entries = _identity_arrays(view.jet_ids)
    return hash_arrays(
        {
            "tokens": view.tokens,
            "mask": view.mask,
            "labels": view.labels,
            "jet_file_indices": file_indices,
            "jet_entries": entries,
        }
    )


def make_view(*, split="model_val", view_name="fixed_hlt", reverse=False, offset=0.0):
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    jet_ids = [
        JetIdentity(file=f"class{int(label)}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    tokens = np.zeros((3, 4, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.ones((3, 4), dtype=bool)
    for jet in range(3):
        for particle in range(4):
            pt = 10.0 + jet + particle + offset
            eta = 0.1 * particle
            phi = -0.2 * particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt * np.cosh(eta) + 0.1
            tokens[jet, particle, 4] = (-1.0, 0.0, 1.0, 0.0)[particle]
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 10] = 0.01 * particle
            tokens[jet, particle, 11] = 0.02
            tokens[jet, particle, 12] = 0.03 * particle
            tokens[jet, particle, 13] = 0.04
    if reverse:
        order = np.arange(3)[::-1]
        tokens = tokens[order]
        mask = mask[order]
        labels = labels[order]
        jet_ids = [jet_ids[int(index)] for index in order]
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": view_name},
    )
    view.metadata.update(
        {
            "jet_files": _identity_arrays(view.jet_ids)[0],
            "jet_identity_hash": jet_identity_hash(view.jet_ids),
            "hlt_content_hash": _content_hash(view),
            "hlt2_content_hash": _content_hash(view) if view_name == "hlt2" else None,
            "allowed_inputs": "HLT_only" if view_name == "hlt2" else None,
            "uses_offline_particles": False,
        }
    )
    return view


def save_view_as_cache(view: JetView, cache_dir: Path):
    counts = np.sum(view.mask, axis=1).astype(np.float32)
    diagnostics = {
        "n_offline": counts.copy(),
        "n_after_eff": counts.copy(),
        "n_after_threshold": counts.copy(),
        "n_after_merge": counts.copy(),
        "drop_eff": np.zeros_like(counts),
        "drop_threshold": np.zeros_like(counts),
        "drop_merge": np.zeros_like(counts),
        "drop_total": np.zeros_like(counts),
        "merge_count": np.zeros_like(counts),
    }
    save_hlt_cache(view, diagnostics, view.metadata, cache_dir, overwrite=True)


class HLTSDVStep3DataTest(unittest.TestCase):
    def test_align_hlt2_view_to_parent_hlt_order(self):
        hlt = make_view(view_name="fixed_hlt")
        hlt2_reversed = make_view(view_name="hlt2", reverse=True, offset=0.5)

        aligned_hlt, aligned_hlt2 = align_hlt_sdv_views(hlt, hlt2_reversed)

        self.assertEqual(aligned_hlt.jet_ids, aligned_hlt2.jet_ids)
        self.assertTrue(np.array_equal(aligned_hlt.labels, aligned_hlt2.labels))
        self.assertTrue(np.all(aligned_hlt2.tokens[:, :, 0] > aligned_hlt.tokens[:, :, 0]))

    def test_same_view_dataset_and_collate_are_deployable(self):
        torch = require_torch()
        hlt = make_view(view_name="fixed_hlt")
        dataset = HLTSelfDualViewDataset(hlt, branch2_mode=HLT_SDV_BRANCH2_SAME_HLT)
        metadata = dataset.to_metadata()

        self.assertEqual(metadata["contract"], HLT_SDV_DATASET_CONTRACT)
        self.assertTrue(metadata["branch2_uses_parent_hlt_cache"])
        self.assertEqual(len(dataset), 3)
        self.assertTrue(np.array_equal(dataset.hlt_tokens, dataset.hlt2_tokens))

        batch = collate_hlt_sdv_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["labels"].shape, torch.Size([2]))
        self.assertEqual(batch["hlt_inputs"]["features"].shape[0], 2)
        self.assertEqual(batch["hlt2_inputs"]["features"].shape[0], 2)
        self.assertIs(batch["hlt"], batch["hlt_inputs"])
        self.assertIs(batch["hlt2"], batch["hlt2_inputs"])
        self.assertEqual(batch["branch2_mode"], HLT_SDV_BRANCH2_SAME_HLT)
        assert_hlt_sdv_batch_deployable(batch)

    def test_hlt2_dataset_keeps_branches_separate(self):
        hlt = make_view(view_name="fixed_hlt")
        hlt2 = make_view(view_name="hlt2", offset=0.5)
        dataset = HLTSelfDualViewDataset(hlt, hlt2, branch2_mode=HLT_SDV_BRANCH2_HLT2, max_jets=2)

        self.assertEqual(len(dataset), 2)
        self.assertFalse(np.array_equal(dataset.hlt_tokens, dataset.hlt2_tokens))
        self.assertEqual(dataset.to_metadata()["branch2_mode"], HLT_SDV_BRANCH2_HLT2)
        batch = collate_hlt_sdv_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["branch2_mode"], HLT_SDV_BRANCH2_HLT2)
        self.assertFalse(batch["returns_offline_particles"])

    def test_load_dataset_from_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hlt = make_view(view_name="fixed_hlt")
            hlt2 = make_view(view_name="hlt2", offset=0.5)
            hlt_dir = root / "hlt_cache"
            hlt2_dir = root / "hlt2_cache"
            save_view_as_cache(hlt, hlt_dir)
            save_view_as_cache(hlt2, hlt2_dir)

            same = load_hlt_sdv_dataset(
                hlt_dir,
                "model_val",
                branch2_mode=HLT_SDV_BRANCH2_SAME_HLT,
                verify_hlt_hash=True,
            )
            self.assertTrue(np.array_equal(same.hlt_tokens, same.hlt2_tokens))

            hlt2_loaded = load_hlt_sdv_dataset(
                hlt_dir,
                "model_val",
                hlt2_cache_dir=hlt2_dir,
                branch2_mode=HLT_SDV_BRANCH2_HLT2,
                verify_hlt_hash=True,
                verify_hlt2_hash=True,
            )
            self.assertFalse(np.array_equal(hlt2_loaded.hlt_tokens, hlt2_loaded.hlt2_tokens))
            self.assertEqual(load_cached_hlt_view(hlt2_dir, "model_val").metadata["view"], "hlt2")

    def test_variant_to_branch2_mode(self):
        self.assertEqual(hlt_sdv_branch2_mode_from_variant(HLT_SDV_VARIANT_SAME_VIEW), HLT_SDV_BRANCH2_SAME_HLT)
        self.assertEqual(
            hlt_sdv_branch2_mode_from_variant(hlt_sdv_dual_hlt2_variant_name(0.20)),
            HLT_SDV_BRANCH2_HLT2,
        )
        with self.assertRaises(ValueError):
            hlt_sdv_branch2_mode_from_variant("hlt2_only_part_s0p20")


if __name__ == "__main__":
    unittest.main()
