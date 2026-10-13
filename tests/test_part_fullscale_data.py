"""Small real-ROOT tests for official ParT preprocessing and replayable streams."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import awkward as ak
import numpy as np
import torch
import uproot

from part_ev_fullscale import data


LABELS = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")


def write_root(path, events=8):
    """Vary lengths, use deliberately non-reconstructed stored jet quantities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [2, 3, 130] + [2 + index % 5 for index in range(max(0, events - 3))]
    lengths = lengths[:events]
    columns = {}
    particle_fields = ("px", "py", "pz", "energy", "deta", "dphi", "charge",
                       "isChargedHadron", "isNeutralHadron", "isPhoton", "isElectron", "isMuon",
                       "d0val", "d0err", "dzval", "dzerr")
    for field in particle_fields:
        rows = []
        for event, length in enumerate(lengths):
            index = np.arange(length, dtype=np.float32)
            values = {
                "px": 1 + index * .1 + event * .02, "py": .2 + index * .03,
                "pz": -1 + index * .2, "energy": 4 + index * .3 + event * .01,
                "deta": -.3 + index * .001, "dphi": .4 - index * .002,
                "charge": (-1.) ** index, "isChargedHadron": np.ones(length),
                "isNeutralHadron": np.zeros(length), "isPhoton": np.zeros(length),
                "isElectron": np.zeros(length), "isMuon": np.zeros(length),
                "d0val": .02 * index, "d0err": .01 + .03 * index,
                "dzval": -.03 * index, "dzerr": .02 + .05 * index,
            }[field]
            rows.append(np.asarray(values, dtype=np.float32))
        columns["part_" + field] = ak.Array(rows)
    for name in ("pt", "eta", "phi", "energy", "nparticles", "sdmass", "tau1", "tau2", "tau3", "tau4"):
        columns["jet_" + name] = np.full(events, 7, dtype=np.float32)
    columns["jet_pt"] = np.full(events, 500, dtype=np.float32)
    columns["jet_energy"] = np.full(events, 900, dtype=np.float32)
    columns["jet_nparticles"] = np.asarray(lengths, dtype=np.float32)
    for index, label in enumerate(LABELS):
        columns["label_" + label] = (np.arange(events) % len(LABELS) == index).astype(np.int32)
    types = {name: "var * float32" if name.startswith("part_") else
             ("int32" if name.startswith("label_") else "float32") for name in columns}
    with uproot.recreate(path) as output:
        output.mktree("tree", types)
        output["tree"].extend(columns)
    return columns


def fixture_inventory(split, paths):
    records = []
    for path in paths:
        metadata = path.stat()
        records.append({"name": path.name, "path": str(path), "size_bytes": metadata.st_size,
                        "mtime_ns": metadata.st_mtime_ns})
    return {"splits": {split: {"files": records}}}


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="part-fullscale-data-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_inventory_files(self, include_test=False):
        for split in ("train", "val", "test") if include_test else ("train", "val"):
            directory_name, indices = data.SPLITS[split]
            directory = self.root / "Pythia" / directory_name
            directory.mkdir(parents=True)
            for label in data.CLASSES:
                for index in indices:
                    (directory / f"{label}_{index:03d}.root").write_bytes(b"header-not-inspected")

    def test_exact_official_inventory_and_global_rank_shards(self):
        self.make_inventory_files(include_test=True)
        inventory = data.inspect_dataset(self.root, include_test=True)
        self.assertEqual([inventory["splits"][key]["file_count"] for key in ("train", "val", "test")],
                         [1000, 50, 200])
        self.assertEqual(inventory["splits"]["train"]["expected_event_count"], 100_000_000)
        self.assertIsNone(inventory["splits"]["train"]["event_count"])
        self.assertEqual(inventory, data.inspect_dataset(self.root, include_test=True))
        partitions = [data.rank_file_dict("train", inventory, rank, 4) for rank in range(4)]
        all_paths = set()
        for partition in partitions:
            self.assertEqual(set(partition), set(data.CLASSES))
            self.assertEqual({len(paths) for paths in partition.values()}, {25})
            paths = set(sum(partition.values(), []))
            self.assertFalse(paths & all_paths)
            all_paths.update(paths)
        self.assertEqual(len(all_paths), 1000)
        # Local rank zero on node two is global rank two, not rank zero again.
        self.assertNotEqual(partitions[0], partitions[2])

    def test_inventory_rejects_missing_extra_and_changed_files(self):
        self.make_inventory_files()
        inventory = data.inspect_dataset(self.root)
        target = self.root / "Pythia/train_100M/HToBB_000.root"
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since inventory"):
            data.rank_file_dict("train", inventory)
        target.unlink()
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            data.inspect_dataset(self.root)
        target.write_bytes(b"replacement")
        (target.parent / "HToBB_100.root").write_bytes(b"wrong split")
        with self.assertRaisesRegex(ValueError, "unexpected=.*HToBB_100"):
            data.inspect_dataset(self.root)

    def test_optional_header_counts_are_observed_and_enforced(self):
        for folder, index in (("train_100M", 0), ("val_5M", 120)):
            write_root(self.root / "Pythia" / folder / f"HToBB_{index:03d}.root", events=3)
        with patch.object(data, "CLASSES", ("HToBB",)), patch.object(
                data, "SPLITS", {"train": ("train_100M", range(1)), "val": ("val_5M", range(120, 121))}):
            with self.assertRaisesRegex(ValueError, "Wrong event count"):
                data.inspect_dataset(self.root, check_entries=True)
            with patch.object(data, "EVENTS_PER_FILE", 3):
                inventory = data.inspect_dataset(self.root, check_entries=True)
                self.assertEqual(inventory["splits"]["train"]["event_count"], 3)
                self.assertEqual(inventory["splits"]["val"]["files"][0]["num_entries"], 3)

    def test_standalone_test_inspection_never_requires_training_files(self):
        directory = self.root / "separate-test-copy"
        directory.mkdir()
        for label in data.CLASSES:
            for index in range(100, 120):
                (directory / f"{label}_{index:03d}.root").write_bytes(b"header-not-inspected")
        inventory = data.inspect_test_dataset(directory, check_entries=False)
        self.assertEqual(set(inventory["splits"]), {"test"})
        self.assertEqual(inventory["splits"]["test"]["file_count"], 200)
        self.assertEqual(inventory["splits"]["test"]["expected_event_count"], 20_000_000)
        canonical = self.root / "Pythia/test_20M"
        canonical.parent.mkdir()
        directory.rename(canonical)
        by_root = data.inspect_dataset(self.root, splits=("test",))
        self.assertEqual(set(by_root["splits"]), {"test"})


@unittest.skipUnless(importlib.util.find_spec("weaver"), "weaver-core is not installed")
class OfficialLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="part-fullscale-loader-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_real_root_exact_official_preprocessing_wrap_and_raw_fourvectors(self):
        from weaver.utils.data.config import DataConfig
        from weaver.utils.data.fileio import _read_files
        from weaver.utils.dataset import _preprocess
        path = self.root / "HToBB_100.root"
        original = write_root(path, events=3)
        inventory = fixture_inventory("test", [path])
        loader, config = data.build_loader("test", inventory, batch_size=2, workers=0)
        batches = list(loader)
        self.assertEqual([batch[0]["pf_features"].shape[0] for batch in batches], [2, 1])
        expected_config = DataConfig.load(str(data.DATA_CONFIG), load_reweight_info=False)
        raw = _read_files([str(path)], expected_config.test_load_branches)
        expected, order = _preprocess(raw, expected_config,
                                      {"training": False, "reweight": False, "shuffle": False})
        np.testing.assert_array_equal(order, np.arange(3))
        for name in config.input_names:
            actual = torch.cat([batch[0][name] for batch in batches]).numpy()
            np.testing.assert_array_equal(actual, expected["_" + name])
        vectors = batches[0][0]["pf_vectors"][0].numpy()
        np.testing.assert_array_equal(vectors[:, :2], np.stack(
            [ak.to_numpy(original["part_" + component][0]) for component in ("px", "py", "pz", "energy")]))
        # Pinned Weaver wrap padding tiles the flattened read chunk (not just
        # the current jet). Preserve its behavior exactly; only masks zero-pad.
        np.testing.assert_array_equal(vectors[:, 2:4], np.stack(
            [ak.to_numpy(original["part_" + component][1])[:2]
             for component in ("px", "py", "pz", "energy")]))
        np.testing.assert_array_equal(batches[0][0]["pf_mask"][0, 0, :5], [1, 1, 0, 0, 0])
        features = batches[0][0]["pf_features"][0].numpy()
        self.assertTrue(np.any(features[:, 2:4] != 0))
        pt = np.hypot(original["part_px"][0, 0], original["part_py"][0, 0])
        self.assertAlmostEqual(float(features[2, 0]), float(np.clip((np.log(pt / 500) + 4.7) * .7, -5, 5)), places=6)
        self.assertEqual(tuple(features.shape), (17, 128))
        self.assertEqual(tuple(batches[1][0]["pf_vectors"].shape), (1, 4, 128))

    def test_train_validation_options_and_rng_isolation(self):
        train_paths = []
        for label in data.CLASSES:
            path = self.root / f"{label}_000.root"
            write_root(path, events=128)
            train_paths.append(path)
        train = fixture_inventory("train", train_paths)
        torch.manual_seed(93)
        np.random.seed(29)
        torch_before, numpy_before = torch.get_rng_state().clone(), np.random.get_state()
        loader, config = data.build_loader("train", train, 4, workers=0, seed=17)
        iterator = iter(loader)
        first = next(iterator)
        torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        underlying = loader.dataset.weaver_dataset
        self.assertTrue(underlying._infinity_mode)
        self.assertFalse(underlying._async_load)
        self.assertEqual(underlying._fetch_step, .01)
        self.assertTrue(underlying._sampler_options["training"])
        self.assertEqual(config.label_names, ("_label_",))
        second = next(iterator)
        rebuilt, _ = data.build_loader("train", train, 4, workers=0, seed=17)
        replayed = data.replay_batches(iter(rebuilt), 1)
        np.random.rand(27)  # Model/analysis NumPy must not perturb stream order.
        actual = next(replayed)
        for name in first[0]:
            torch.testing.assert_close(actual[0][name], second[0][name], rtol=0, atol=0)
        val = fixture_inventory("val", [train_paths[0]])
        val_loader, _ = data.build_loader("val", val, 4, workers=0)
        self.assertTrue(val_loader.dataset.weaver_dataset._sampler_options["training"])
        with self.assertRaisesRegex(ValueError, "global rank zero"):
            data.build_loader("val", val, 4, rank=1, world_size=2, workers=0)

    def test_worker_stream_replay_and_shutdown(self):
        paths = [self.root / f"HToBB_{index}.root" for index in (120, 121)]
        for path in paths:
            write_root(path, events=256)
        inventory = fixture_inventory("val", paths)
        original, _ = data.build_loader("val", inventory, 4, workers=2, seed=41)
        rebuilt, _ = data.build_loader("val", inventory, 4, workers=2, seed=41)
        try:
            iterator = iter(original)
            data.replay_batches(iterator, 3)
            expected = next(iterator)
            replay = data.replay_batches(iter(rebuilt), 3)
            actual = next(replay)
            for name in expected[0]:
                torch.testing.assert_close(actual[0][name], expected[0][name], rtol=0, atol=0)
            torch.testing.assert_close(actual[1]["_label_"], expected[1]["_label_"], rtol=0, atol=0)
            self.assertTrue(original.persistent_workers)
        finally:
            data.shutdown_loader(original)
            data.shutdown_loader(rebuilt)
        self.assertIsNone(original._iterator)

    def test_modified_auto_config_wrong_version_and_unsafe_rank_are_rejected(self):
        path = self.root / "HToBB_120.root"
        write_root(path)
        inventory = fixture_inventory("val", [path])
        copied = self.root / "JetClass_full.yaml"
        content = data.DATA_CONFIG.read_bytes()
        copied.write_bytes(content + b"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "unchanged official"):
            data.build_loader("val", inventory, 2, workers=0, data_config_path=copied)
        copied.write_bytes(content)
        auto = Path(str(copied).replace(".yaml", f".{hashlib.md5(content).hexdigest()}.auto.yaml"))
        auto.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "auto YAML"):
            data.build_loader("val", inventory, 2, workers=0, data_config_path=copied)
        with patch.object(data.importlib.metadata, "version", return_value="0.5.0"):
            with self.assertRaisesRegex(RuntimeError, "0.4.17"):
                data.build_loader("val", inventory, 2, workers=0)
        with self.assertRaisesRegex(ValueError, "global rank"):
            data.rank_file_dict("val", inventory, rank=2, world_size=2)
        with self.assertRaisesRegex(ValueError, "one file per"):
            data.build_loader("val", inventory, 2, workers=2)
        duplicate = copy.deepcopy(inventory)
        duplicate["splits"]["val"]["files"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            data.rank_file_dict("val", duplicate)

    def test_weaver_silent_partial_fetch_is_fatal(self):
        path = self.root / "HToBB_120.root"
        write_root(path, events=256)
        inventory = fixture_inventory("val", [path])
        loader, _ = data.build_loader("val", inventory, 2, workers=0)
        # The first 1% interval contains two entries. Simulate Weaver swallowing
        # a file read error but returning a nonempty partial table.
        with patch("weaver.utils.data.fileio._read_files", return_value=ak.Array([{"x": 1}])):
            with self.assertRaisesRegex(RuntimeError, "Incomplete ROOT fetch"):
                next(iter(loader))


if __name__ == "__main__":
    unittest.main()
