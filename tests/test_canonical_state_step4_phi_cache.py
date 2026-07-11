from __future__ import annotations

import importlib.util
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
from teacher_logit_reco.architecture_view_part import load_cached_offline_view, save_cached_offline_view
from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_PHI_CACHE_CONTRACT,
    CANONICAL_STATE_PHI_HLT_SOURCE,
    CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
    CANONICAL_STATE_PHI_OFFLINE_SOURCE,
    CANONICAL_STATE_SPLIT_ORDER,
    audit_canonical_phi_cache,
    audit_canonical_phi_pair_alignment,
    build_phi_cache_from_hlt_cache,
    build_phi_cache_from_offline_cache,
    load_canonical_phi_cache,
    phi_cache_paths,
    save_canonical_phi_cache,
)
from teacher_logit_reco.canonical_state.run_variants import CanonicalStateCachedDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_SCRIPT = REPO_ROOT / "scripts" / "cache_canonical_state_phi.py"


def _load_cache_script():
    spec = importlib.util.spec_from_file_location("cache_canonical_state_phi", CACHE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_tokens(split_index: int, n_jets: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[JetIdentity]]:
    tokens = np.zeros((n_jets, 8, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 8), dtype=bool)
    labels = np.asarray([index % len(LABEL_NAMES) for index in range(n_jets)], dtype=np.int64)
    jet_ids: list[JetIdentity] = []
    prefixes = list(FILE_PREFIX_TO_LABEL.keys())
    for jet_index, label in enumerate(labels):
        file_name = f"{prefixes[int(label)]}_{split_index:03d}.root"
        jet_ids.append(JetIdentity(file=file_name, entry=jet_index, label=int(label)))
        valid = 5 + (jet_index % 2)
        mask[jet_index, :valid] = True
        for part_index in range(valid):
            pt = 5.0 + 0.2 * jet_index + 0.4 * part_index + 0.1 * split_index
            eta = -0.5 + 0.08 * part_index
            phi = -0.3 + 0.11 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta) + 0.1
            tokens[jet_index, part_index, 4] = 1.0 if part_index % 2 == 0 else 0.0
            tokens[jet_index, part_index, 5 + (part_index % 5)] = 1.0
            tokens[jet_index, part_index, 10:14] = np.asarray([0.1, 0.01, 0.2, 0.02], dtype=np.float32)
    return tokens, mask, labels, jet_ids


def _toy_view(split: str, *, split_index: int, source_manifest_hash: str) -> JetView:
    tokens, mask, labels, jet_ids = _toy_tokens(split_index)
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "offline",
            "input_source": "offline",
            "source_manifest_hash": source_manifest_hash,
        },
    )


def _toy_manifest() -> SplitManifest:
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        _, _, _, jet_ids = _toy_tokens(split_index)
        splits[split] = jet_ids
    split_sizes = {split: len(rows) for split, rows in splits.items()}
    file_records = []
    for prefix, label in FILE_PREFIX_TO_LABEL.items():
        for split_index in range(len(SPLIT_ORDER)):
            file_records.append(FileRecord(path=f"{prefix}_{split_index:03d}.root", label=label, num_entries=20))
    return SplitManifest(
        data_dir="toy",
        max_constits=8,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes=split_sizes,
        split_seeds={split: index + 10 for index, split in enumerate(SPLIT_ORDER)},
        file_records=file_records,
        splits=splits,
        metadata={"test_manifest": True},
    )


def _write_raw_caches(root: Path) -> tuple[Path, Path, Path, SplitManifest]:
    manifest = _toy_manifest()
    manifest_path = root / "split_manifest.json.gz"
    save_split_manifest(manifest, manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_cache_dir = root / "hlt_cache"
    offline_cache_dir = root / "offline_cache"
    params = fixed_hlt_params_from_profile(HLT_PROFILE_V2_REALISTIC, 2.5)
    for split in CANONICAL_STATE_SPLIT_ORDER:
        split_index = SPLIT_ORDER.index(split)
        view = _toy_view(split, split_index=split_index, source_manifest_hash=manifest_sha)
        generate_and_cache_hlt_view(
            view,
            hlt_cache_dir,
            seed=DEFAULT_HLT_SEEDS[split],
            params=params,
            hlt_degradation_strength=2.5,
        )
        save_cached_offline_view(view, offline_cache_dir)
    return manifest_path, hlt_cache_dir, offline_cache_dir, manifest


class CanonicalStateStep4PhiCacheTests(unittest.TestCase):
    def test_save_load_hlt_phi_cache_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, _, _ = _write_raw_caches(root)
            phi_hlt_dir = root / "phi_hlt"

            reports = build_phi_cache_from_hlt_cache(
                hlt_cache_dir,
                phi_hlt_dir,
                splits=CANONICAL_STATE_SPLIT_ORDER,
            )
            loaded = load_canonical_phi_cache(
                phi_hlt_dir,
                "model_train",
                source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
            )
            audit = audit_canonical_phi_cache(
                phi_hlt_dir,
                source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
                manifest_path=manifest_path,
                splits=CANONICAL_STATE_SPLIT_ORDER,
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertEqual(reports["model_train"]["cache_contract"], CANONICAL_STATE_PHI_CACHE_CONTRACT)
        self.assertEqual(loaded.phi_tokens.shape, (10, 48, 18))
        self.assertEqual(loaded.state_mask.shape, (10, 48))
        self.assertEqual(loaded.metadata["source_cache_hash_name"], "hlt_content_hash")
        self.assertTrue(loaded.metadata["source_cache_hash"])
        self.assertEqual(loaded.metadata["layout_version"], "canonical_jet_state_layout_v1")
        self.assertEqual(loaded.metadata["phi_builder_version"], "canonical_phi_builder_v1")
        self.assertTrue(audit["ok"], audit["problems"])

    def test_save_load_offline_phi_cache_and_pair_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, offline_cache_dir, _ = _write_raw_caches(root)
            phi_hlt_dir = root / "phi_hlt"
            phi_offline_dir = root / "phi_offline"

            build_phi_cache_from_hlt_cache(
                hlt_cache_dir,
                phi_hlt_dir,
                splits=CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
            )
            build_phi_cache_from_offline_cache(
                offline_cache_dir,
                phi_offline_dir,
                splits=CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
            )
            loaded = load_canonical_phi_cache(
                phi_offline_dir,
                "model_val",
                source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
            )
            pair = audit_canonical_phi_pair_alignment(
                phi_hlt_dir,
                phi_offline_dir,
                manifest_path=manifest_path,
                splits=CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertEqual(loaded.metadata["source_cache_hash_name"], "offline_content_hash")
        self.assertTrue(loaded.metadata["allowed_for_primary_training"])
        self.assertTrue(pair["ok"], pair["problems"])
        for split in CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS:
            self.assertTrue(pair["split_reports"][split]["ok"], pair["split_reports"][split]["problems"])

    def test_layout_and_field_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, hlt_cache_dir, _, _ = _write_raw_caches(root)
            phi_hlt_dir = root / "phi_hlt"
            build_phi_cache_from_hlt_cache(hlt_cache_dir, phi_hlt_dir, splits=("model_train",))
            _, metadata_path = phi_cache_paths(phi_hlt_dir, "model_train", CANONICAL_STATE_PHI_HLT_SOURCE)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["field_names"] = list(reversed(metadata["field_names"]))
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "field_names"):
                load_canonical_phi_cache(
                    phi_hlt_dir,
                    "model_train",
                    source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
                )

    def test_manifest_mismatch_and_missing_source_hash_fail_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, _, _ = _write_raw_caches(root)
            phi_hlt_dir = root / "phi_hlt"
            build_phi_cache_from_hlt_cache(hlt_cache_dir, phi_hlt_dir, splits=("model_train",))
            _, metadata_path = phi_cache_paths(phi_hlt_dir, "model_train", CANONICAL_STATE_PHI_HLT_SOURCE)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_manifest_hash"] = "stale"
            metadata["source_cache_hash"] = None
            metadata["hlt_content_hash"] = None
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = audit_canonical_phi_cache(
                phi_hlt_dir,
                source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
                manifest_path=manifest_path,
                splits=("model_train",),
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertFalse(audit["ok"])
        self.assertTrue(any("source_manifest_hash" in problem for problem in audit["problems"]))
        self.assertTrue(any("missing source cache hash" in problem for problem in audit["problems"]))

    def test_variant_dataset_rejects_phi_hlt_from_different_hlt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, hlt_cache_dir, _, manifest = _write_raw_caches(root)
            phi_hlt_dir = root / "phi_hlt"
            build_phi_cache_from_hlt_cache(hlt_cache_dir, phi_hlt_dir, splits=("model_train",))
            _, metadata_path = phi_cache_paths(phi_hlt_dir, "model_train", CANONICAL_STATE_PHI_HLT_SOURCE)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_cache_hash"] = "stale-hlt-content"
            metadata["hlt_content_hash"] = "stale-hlt-content"
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Phi_hlt source_cache_hash"):
                CanonicalStateCachedDataset(
                    hlt_cache_dir=hlt_cache_dir,
                    phi_hlt_cache_dir=phi_hlt_dir,
                    phi_offline_cache_dir=None,
                    split="model_train",
                    max_jets=None,
                    include_phi_off=False,
                    expected_manifest_hash=manifest_hash(manifest),
                )

    def test_hlt_offline_alignment_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_cache_dir, offline_cache_dir, _ = _write_raw_caches(root)
            phi_hlt_dir = root / "phi_hlt"
            phi_offline_dir = root / "phi_offline"
            build_phi_cache_from_hlt_cache(hlt_cache_dir, phi_hlt_dir, splits=("model_train",))

            offline_view = load_cached_offline_view(offline_cache_dir, "model_train")
            shifted_ids = tuple(
                JetIdentity(file=identity.file, entry=int(identity.entry) + 100, label=identity.label)
                for identity in offline_view.jet_ids
            )
            shifted_view = JetView(
                tokens=offline_view.tokens,
                mask=offline_view.mask,
                labels=offline_view.labels,
                jet_ids=shifted_ids,
                split=offline_view.split,
                metadata={**offline_view.metadata, "jet_identity_hash": None},
            )
            save_canonical_phi_cache(shifted_view, phi_offline_dir, source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE)
            pair = audit_canonical_phi_pair_alignment(
                phi_hlt_dir,
                phi_offline_dir,
                manifest_path=manifest_path,
                splits=("model_train",),
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertFalse(pair["ok"])
        self.assertTrue(any("jet identities do not match split manifest" in problem for problem in pair["problems"]))

    def test_offline_final_test_phi_is_oracle_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _, offline_cache_dir, _ = _write_raw_caches(root)
            phi_offline_dir = root / "phi_offline"
            view = load_cached_offline_view(offline_cache_dir, "final_test")

            with self.assertRaisesRegex(ValueError, "oracle-only"):
                save_canonical_phi_cache(view, phi_offline_dir, source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE)
            metadata = save_canonical_phi_cache(
                view,
                phi_offline_dir,
                source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
                allow_final_test_offline_oracle=True,
            )
            with self.assertRaisesRegex(ValueError, "oracle-only"):
                load_canonical_phi_cache(
                    phi_offline_dir,
                    "final_test",
                    source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
                )
            loaded = load_canonical_phi_cache(
                phi_offline_dir,
                "final_test",
                source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
                allow_oracle_final_test=True,
            )
            audit_primary = audit_canonical_phi_cache(
                phi_offline_dir,
                source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
                manifest_path=manifest_path,
                splits=("final_test",),
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )
            audit_oracle = audit_canonical_phi_cache(
                phi_offline_dir,
                source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
                manifest_path=manifest_path,
                splits=("final_test",),
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
                allow_oracle_final_test=True,
            )

        self.assertTrue(metadata["oracle_only"])
        self.assertFalse(metadata["allowed_for_primary_training"])
        self.assertTrue(loaded.metadata["oracle_only"])
        self.assertFalse(audit_primary["ok"])
        self.assertTrue(audit_oracle["ok"], audit_oracle["problems"])

    def test_cache_script_parses_defaults(self):
        module = _load_cache_script()
        args = module.parse_args(
            [
                "--source-view",
                "hlt",
                "--input-cache-dir",
                "input",
                "--output-cache-dir",
                "output",
            ]
        )
        self.assertEqual(args.source_view, "hlt")
        self.assertIsNone(args.splits)
        self.assertFalse(args.allow_final_test_offline_oracle)


if __name__ == "__main__":
    unittest.main()
