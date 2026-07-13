from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    HLT_PROFILE_V2_REALISTIC,
    fixed_hlt_params_from_profile,
    generate_and_cache_hlt_view,
)
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    FileRecord,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from teacher_logit_reco.architecture_view_part import save_cached_offline_view
from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    DERIVED_DIAGNOSTIC_FIELD_NAMES,
    HIERARCHY_TARGET_CACHE_CONTRACT,
    LEVEL_CELL_COUNTS,
    PID_CATEGORY_NAMES,
    assign_hierarchy_cells,
    audit_hierarchy_target_cache,
    build_hierarchy_target_cache,
    build_hierarchy_target_caches,
    build_hierarchy_targets,
    default_hierarchy_target_layout,
    derive_accounting_diagnostics,
    fit_radial_boundary_from_hlt,
    hierarchy_target_cache_paths,
    load_hierarchy_target_cache,
    load_hierarchy_target_shard,
    require_hierarchy_consistency,
)


def _particle(
    pt: float,
    eta: float,
    phi: float,
    pid: int,
    *,
    energy_scale: float = 1.1,
) -> np.ndarray:
    token = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    token[0] = float(pt)
    token[1] = float(eta)
    token[2] = float(phi)
    token[3] = float(pt) * float(energy_scale)
    token[4] = 1.0 if int(pid) in (0, 3, 4) else 0.0
    token[5 + int(pid)] = 1.0
    return token


def _direct_toy_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hlt = np.zeros((2, 8, RAW_TOKEN_DIM), dtype=np.float32)
    hlt_mask = np.zeros((2, 8), dtype=bool)
    offline = np.zeros_like(hlt)
    offline_mask = np.zeros_like(hlt_mask)

    hlt[0, 0] = _particle(30.0, 0.02, np.pi - 0.02, 0)
    hlt[0, 1] = _particle(20.0, -0.03, -np.pi + 0.03, 1)
    hlt_mask[0, :2] = True
    hlt[1, 0] = _particle(25.0, 0.10, 0.15, 2)
    hlt[1, 1] = _particle(15.0, -0.10, -0.15, 0)
    hlt_mask[1, :2] = True

    rows0 = (
        _particle(11.0, 0.10, np.pi - 0.05, 0),
        _particle(7.0, -0.12, -np.pi + 0.06, 1),
        _particle(5.0, 0.32, np.pi - 0.28, 2),
        _particle(3.0, -0.35, -np.pi + 0.30, 3),
        _particle(2.0, 0.95, np.pi - 0.50, 4),
    )
    for index, row in enumerate(rows0):
        offline[0, index] = row
    offline_mask[0, : len(rows0)] = True

    rows1 = (
        _particle(13.0, 0.08, 0.12, 0),
        _particle(9.0, -0.20, -0.18, 1),
        _particle(6.0, 0.30, 0.35, 2),
    )
    for index, row in enumerate(rows1):
        offline[1, index] = row
    offline_mask[1, : len(rows1)] = True
    # Exercise the deterministic unknown-PID fallback without losing pT/count.
    offline[1, 2, 5:10] = 0.0
    return hlt, hlt_mask, offline, offline_mask


def _toy_tokens(split_index: int, n_jets: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[JetIdentity]]:
    tokens = np.zeros((n_jets, 8, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 8), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % len(LABEL_NAMES)
    identities: list[JetIdentity] = []
    prefixes = list(FILE_PREFIX_TO_LABEL.keys())
    for jet_index, label in enumerate(labels):
        file_name = f"{prefixes[int(label)]}_{split_index:03d}.root"
        identities.append(JetIdentity(file=file_name, entry=jet_index, label=int(label)))
        valid = 5 + jet_index % 2
        mask[jet_index, :valid] = True
        for particle_index in range(valid):
            tokens[jet_index, particle_index] = _particle(
                5.0 + 0.4 * particle_index + 0.1 * jet_index,
                -0.35 + 0.12 * particle_index,
                -0.28 + 0.10 * particle_index,
                particle_index % len(PID_CATEGORY_NAMES),
            )
    return tokens, mask, labels, identities


def _toy_manifest() -> SplitManifest:
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        _, _, _, identities = _toy_tokens(split_index)
        splits[split] = identities
    file_records = [
        FileRecord(path=f"{prefix}_{split_index:03d}.root", label=label, num_entries=20)
        for prefix, label in FILE_PREFIX_TO_LABEL.items()
        for split_index in range(len(SPLIT_ORDER))
    ]
    return SplitManifest(
        data_dir="toy",
        max_constits=8,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: len(rows) for split, rows in splits.items()},
        split_seeds={split: 100 + index for index, split in enumerate(SPLIT_ORDER)},
        file_records=file_records,
        splits=splits,
        metadata={"test_manifest": True},
    )


def _write_source_caches(root: Path, *, strength: float = 2.5) -> tuple[Path, Path, Path]:
    manifest = _toy_manifest()
    manifest_path = root / "split_manifest.json.gz"
    save_split_manifest(manifest, manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_cache_dir = root / "hlt_cache"
    offline_cache_dir = root / "offline_cache"
    params = fixed_hlt_params_from_profile(HLT_PROFILE_V2_REALISTIC, strength)
    for split in ("model_train", "model_val"):
        split_index = SPLIT_ORDER.index(split)
        tokens, mask, labels, identities = _toy_tokens(split_index)
        view = JetView(
            tokens=tokens,
            mask=mask,
            labels=labels,
            jet_ids=identities,
            split=split,
            metadata={"source_manifest_hash": manifest_sha, "view": "offline", "input_source": "offline"},
        )
        generate_and_cache_hlt_view(
            view,
            hlt_cache_dir,
            seed=DEFAULT_HLT_SEEDS[split],
            params=params,
            hlt_degradation_strength=float(strength),
        )
        save_cached_offline_view(view, offline_cache_dir)
    return manifest_path, hlt_cache_dir, offline_cache_dir


class ConstrainedCoarseToFineStep1TargetTests(unittest.TestCase):
    def test_layout_and_assignments_are_nested_and_stable(self):
        layout = default_hierarchy_target_layout(radial_boundary=0.2, coordinate_extent=0.8)
        self.assertEqual(layout.level_cell_counts, LEVEL_CELL_COUNTS)
        self.assertEqual(len(layout.cell_geometry(1)), 8)
        self.assertEqual(len(layout.cell_geometry(2)), 32)
        self.assertEqual(len(layout.cell_geometry(3)), 128)
        np.testing.assert_array_equal(layout.parent_indices(2), np.arange(32, dtype=np.int16) // 4)
        np.testing.assert_array_equal(layout.parent_indices(3), np.arange(128, dtype=np.int16) // 4)

        deta = np.asarray([[-0.7, -0.1, 0.1, 0.7, 0.0]], dtype=np.float32)
        dphi = np.asarray([[-0.6, 0.1, -0.1, 0.6, 0.0]], dtype=np.float32)
        mask = np.asarray([[True, True, True, True, False]])
        level1, level2, level3, _, _ = assign_hierarchy_cells(deta, dphi, mask, layout=layout)
        np.testing.assert_array_equal(level2[:, :4] // 4, level1[:, :4])
        np.testing.assert_array_equal(level3[:, :4] // 4, level2[:, :4])
        self.assertEqual(int(level3[0, 4]), -1)

    def test_accounting_closes_and_derives_physics_quantities(self):
        hlt, hlt_mask, offline, offline_mask = _direct_toy_batch()
        layout = default_hierarchy_target_layout(radial_boundary=0.22, coordinate_extent=0.8)
        output = build_hierarchy_targets(
            hlt,
            hlt_mask,
            offline,
            offline_mask,
            layout=layout,
        )
        require_hierarchy_consistency(output)
        self.assertEqual(output.global_accounting.shape, (2, len(ACCOUNTING_FIELD_NAMES)))
        self.assertEqual(output.level1_accounting.shape, (2, 8, len(ACCOUNTING_FIELD_NAMES)))
        self.assertEqual(output.level2_accounting.shape, (2, 32, len(ACCOUNTING_FIELD_NAMES)))
        self.assertEqual(output.level3_accounting.shape, (2, 128, len(ACCOUNTING_FIELD_NAMES)))
        self.assertEqual(output.final_cell_indices.shape, offline_mask.shape)

        index = {name: ACCOUNTING_FIELD_NAMES.index(name) for name in ACCOUNTING_FIELD_NAMES}
        expected_pt = np.sum(np.where(offline_mask, offline[:, :, 0], 0.0), axis=1)
        expected_energy = np.sum(np.where(offline_mask, offline[:, :, 3], 0.0), axis=1)
        expected_count = offline_mask.sum(axis=1)
        np.testing.assert_allclose(output.global_accounting[:, index["total_pT"]], expected_pt, atol=1.0e-5)
        np.testing.assert_allclose(output.global_accounting[:, index["total_energy"]], expected_energy, atol=1.0e-5)
        np.testing.assert_allclose(
            output.global_accounting[:, index["expected_constituent_count"]], expected_count, atol=0.0
        )
        self.assertEqual(int(output.unknown_pid_counts[1]), 1)
        self.assertGreaterEqual(int(output.clipped_particle_counts[0]), 1)

        derived = derive_accounting_diagnostics(output.global_accounting)
        self.assertEqual(derived.shape, (2, len(DERIVED_DIAGNOSTIC_FIELD_NAMES)))
        self.assertTrue(np.isfinite(derived).all())
        fraction_indices = [DERIVED_DIAGNOSTIC_FIELD_NAMES.index(f"{name}_pT_fraction") for name in PID_CATEGORY_NAMES]
        np.testing.assert_allclose(derived[:, fraction_indices].sum(axis=1), 1.0, atol=1.0e-6)

    def test_radial_fit_is_deterministic_and_in_range(self):
        hlt, hlt_mask, _, _ = _direct_toy_batch()
        first, first_report = fit_radial_boundary_from_hlt(
            hlt,
            hlt_mask,
            coordinate_extent=0.8,
            histogram_bins=128,
            chunk_size=1,
        )
        second, second_report = fit_radial_boundary_from_hlt(
            hlt,
            hlt_mask,
            coordinate_extent=0.8,
            histogram_bins=128,
            chunk_size=2,
        )
        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)
        self.assertLess(first, np.sqrt(2.0) * 0.8)
        self.assertEqual(first_report["method"], second_report["method"])

    def test_sharded_cache_round_trip_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, offline_cache_dir = _write_source_caches(root)
            output_dir = root / "targets"
            reports = build_hierarchy_target_caches(
                manifest_path=manifest_path,
                hlt_cache_dir=hlt_cache_dir,
                offline_cache_dir=offline_cache_dir,
                output_cache_dir=output_dir,
                splits=("model_train", "model_val"),
                radial_boundary=0.2,
                chunk_size=3,
            )
            shard = load_hierarchy_target_shard(output_dir, "model_train", 0)
            loaded = load_hierarchy_target_cache(output_dir, "model_train")
            audit = audit_hierarchy_target_cache(
                output_dir,
                manifest_path=manifest_path,
                splits=("model_train", "model_val"),
                expected_split_sizes={"model_train": 10, "model_val": 10},
            )

        self.assertEqual(reports["reports"]["model_train"]["cache_contract"], HIERARCHY_TARGET_CACHE_CONTRACT)
        self.assertEqual(reports["reports"]["model_train"]["n_shards"], 4)
        self.assertEqual(shard.start, 0)
        self.assertEqual(shard.stop, 3)
        self.assertEqual(loaded.output.global_accounting.shape[0], 10)
        self.assertEqual(tuple(loaded.jet_ids), tuple(_toy_manifest().splits["model_train"]))
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertEqual(reports["cache_set"]["hlt_degradation_strength"], 2.5)

    def test_shard_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, offline_cache_dir = _write_source_caches(root)
            output_dir = root / "targets"
            build_hierarchy_target_caches(
                manifest_path=manifest_path,
                hlt_cache_dir=hlt_cache_dir,
                offline_cache_dir=offline_cache_dir,
                output_cache_dir=output_dir,
                splits=("model_train",),
                radial_boundary=0.2,
                chunk_size=4,
            )
            shard_dir, metadata_path = hierarchy_target_cache_paths(output_dir, "model_train")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            shard_path = shard_dir / metadata["shards"][0]["filename"]
            with np.load(shard_path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            arrays["global_accounting"] = arrays["global_accounting"].copy()
            arrays["global_accounting"][0, 0] += 1.0
            np.savez_compressed(shard_path, **arrays)
            with self.assertRaisesRegex(ValueError, "shard hash mismatch"):
                load_hierarchy_target_shard(output_dir, "model_train", 0)

    def test_wrong_hlt_strength_and_final_test_targets_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, offline_cache_dir = _write_source_caches(root, strength=1.0)
            layout = default_hierarchy_target_layout(radial_boundary=0.2)
            with self.assertRaisesRegex(ValueError, "degradation strength mismatch"):
                build_hierarchy_target_cache(
                    manifest_path=manifest_path,
                    hlt_cache_dir=hlt_cache_dir,
                    offline_cache_dir=offline_cache_dir,
                    output_cache_dir=root / "targets",
                    split="model_train",
                    layout=layout,
                )
            with self.assertRaisesRegex(ValueError, "final_test"):
                build_hierarchy_target_cache(
                    manifest_path=manifest_path,
                    hlt_cache_dir=hlt_cache_dir,
                    offline_cache_dir=offline_cache_dir,
                    output_cache_dir=root / "targets",
                    split="final_test",
                    layout=layout,
                )


if __name__ == "__main__":
    unittest.main()

