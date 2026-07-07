import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view, save_hlt_cache
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    FileRecord,
    JetIdentity,
    JetView,
    SplitManifest,
    manifest_hash,
)
from teacher_logit_reco.hlt_self_dualview import (
    HLT2_CACHE_CONTRACT,
    audit_hlt2_cache,
    generate_and_cache_hlt2_view,
    hlt2_params_from_strength,
    hlt2_seed_for_identity,
    hlt_sdv_strength_tag,
    write_hlt2_audit_reports,
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


class HLTSDVStep2HLT2CacheTest(unittest.TestCase):
    def make_manifest(self):
        splits = {split: [] for split in SPLIT_ORDER}
        splits["model_train"] = [
            JetIdentity(file="QCD.root", entry=0, label=0),
            JetIdentity(file="Hbb.root", entry=0, label=1),
            JetIdentity(file="Hgg.root", entry=0, label=3),
        ]
        return SplitManifest(
            data_dir="data",
            max_constits=4,
            class_names=list(LABEL_NAMES),
            file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
            split_sizes={split: len(splits[split]) for split in SPLIT_ORDER},
            split_seeds={split: 100 + idx for idx, split in enumerate(SPLIT_ORDER)},
            file_records=[
                FileRecord(path="QCD.root", label=0, num_entries=10),
                FileRecord(path="Hbb.root", label=1, num_entries=10),
                FileRecord(path="Hgg.root", label=3, num_entries=10),
            ],
            splits=splits,
            metadata={"test": True},
        )

    def make_parent_hlt_cache(self, cache_dir: Path, manifest: SplitManifest):
        jet_ids = manifest.splits["model_train"]
        labels = np.asarray([identity.label for identity in jet_ids], dtype=np.int64)
        tokens = np.zeros((len(jet_ids), 4, 14), dtype=np.float32)
        mask = np.zeros((len(jet_ids), 4), dtype=bool)
        for jet_index in range(len(jet_ids)):
            for part_index in range(4):
                tokens[jet_index, part_index, 0] = 2.0 + jet_index + 0.2 * part_index
                tokens[jet_index, part_index, 1] = 0.02 * (jet_index + part_index)
                tokens[jet_index, part_index, 2] = 0.03 * (jet_index - part_index)
                tokens[jet_index, part_index, 3] = tokens[jet_index, part_index, 0] * np.cosh(
                    tokens[jet_index, part_index, 1]
                )
                tokens[jet_index, part_index, 4] = 1.0
                tokens[jet_index, part_index, 5] = 1.0
                mask[jet_index, part_index] = True
        view = JetView(tokens=tokens, mask=mask, labels=labels, jet_ids=jet_ids, split="model_train")
        jet_files, _, _ = _identity_arrays(jet_ids)
        metadata = {
            "version": 1,
            "view": "fixed_hlt",
            "split": "model_train",
            "seed": 11,
            "source_manifest_hash": manifest_hash(manifest),
            "n_jets": len(jet_ids),
            "jet_files": jet_files,
            "jet_identity_hash": jet_identity_hash(jet_ids),
            "hlt_content_hash": _content_hash(view),
        }
        diagnostics = {
            "n_offline": np.sum(mask, axis=1).astype(np.float32),
            "n_after_eff": np.sum(mask, axis=1).astype(np.float32),
            "n_after_threshold": np.sum(mask, axis=1).astype(np.float32),
            "n_after_merge": np.sum(mask, axis=1).astype(np.float32),
            "drop_eff": np.zeros((len(jet_ids),), dtype=np.float32),
            "drop_threshold": np.zeros((len(jet_ids),), dtype=np.float32),
            "drop_merge": np.zeros((len(jet_ids),), dtype=np.float32),
            "drop_total": np.zeros((len(jet_ids),), dtype=np.float32),
            "merge_count": np.zeros((len(jet_ids),), dtype=np.float32),
        }
        save_hlt_cache(view, diagnostics, metadata, cache_dir, overwrite=True)
        return view

    def test_strength_zero_is_exact_identity_and_audit_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_manifest()
            parent_dir = root / "hlt_cache"
            parent = self.make_parent_hlt_cache(parent_dir, manifest)
            hlt2_dir = root / "hlt_self_dualview" / "hlt2_cache" / "s0p00"

            generate_and_cache_hlt2_view(
                manifest,
                parent_dir,
                hlt2_dir,
                strength=0.0,
                splits=("model_train",),
                overwrite=True,
            )
            hlt2 = load_cached_hlt_view(hlt2_dir, "model_train", verify_hash=True)
            self.assertTrue(np.array_equal(parent.tokens, hlt2.tokens))
            self.assertTrue(np.array_equal(parent.mask, hlt2.mask))
            self.assertEqual(hlt2.metadata["contract"], HLT2_CACHE_CONTRACT)
            self.assertFalse(hlt2.metadata["uses_offline_particles"])
            self.assertEqual(hlt2.metadata["source_view"], "HLT")
            self.assertEqual(hlt2.metadata["derived_view"], "HLT2")
            self.assertEqual(hlt2.metadata["hlt2_strength_tag"], "s0p00")

            audit = audit_hlt2_cache(manifest, parent_dir, hlt2_dir, strength=0.0, splits=("model_train",))
            self.assertTrue(audit["ok"], audit.get("problems"))

    def test_positive_strength_is_deterministic_and_changes_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_manifest()
            parent_dir = root / "hlt_cache"
            parent = self.make_parent_hlt_cache(parent_dir, manifest)
            hlt2_dir = root / "hlt_self_dualview" / "hlt2_cache" / "s0p20"

            first = generate_and_cache_hlt2_view(
                manifest,
                parent_dir,
                hlt2_dir,
                strength=0.20,
                splits=("model_train",),
                overwrite=True,
            )
            first_view = load_cached_hlt_view(hlt2_dir, "model_train", verify_hash=True)
            second = generate_and_cache_hlt2_view(
                manifest,
                parent_dir,
                hlt2_dir,
                strength=0.20,
                splits=("model_train",),
                overwrite=True,
            )
            second_view = load_cached_hlt_view(hlt2_dir, "model_train", verify_hash=True)

            self.assertEqual(hlt_sdv_strength_tag(0.20), "s0p20")
            self.assertEqual(first["reports"]["model_train"]["hlt2_content_hash"], second["reports"]["model_train"]["hlt2_content_hash"])
            self.assertTrue(np.array_equal(first_view.tokens, second_view.tokens))
            self.assertNotEqual(first_view.metadata["hlt2_content_hash"], parent.metadata.get("hlt_content_hash"))
            audit = audit_hlt2_cache(manifest, parent_dir, hlt2_dir, strength=0.20, splits=("model_train",))
            self.assertTrue(audit["ok"], audit.get("problems"))

    def test_seed_and_params_are_stable(self):
        identity = JetIdentity(file="QCD.root", entry=123, label=0)
        seed_a = hlt2_seed_for_identity(identity, split="model_train", strength=0.20)
        seed_b = hlt2_seed_for_identity(identity, split="model_train", strength="s0p20")
        seed_c = hlt2_seed_for_identity(identity, split="model_val", strength=0.20)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)
        self.assertEqual(hlt2_params_from_strength(0.0).hlt_pt_threshold, 0.0)
        self.assertAlmostEqual(hlt2_params_from_strength(0.20).hlt_pt_threshold, 0.016)

    def test_write_audit_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_manifest()
            parent_dir = root / "hlt_cache"
            self.make_parent_hlt_cache(parent_dir, manifest)
            hlt2_dir = root / "hlt_self_dualview" / "hlt2_cache" / "s0p00"
            generate_and_cache_hlt2_view(
                manifest,
                parent_dir,
                hlt2_dir,
                strength=0.0,
                splits=("model_train",),
                overwrite=True,
            )
            result = write_hlt2_audit_reports(
                manifest,
                parent_dir,
                hlt2_dir,
                root / "audits",
                strength=0.0,
                splits=("model_train",),
            )
            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["audit_report"]).exists())
            self.assertTrue(Path(result["summary"]).exists())


if __name__ == "__main__":
    unittest.main()
