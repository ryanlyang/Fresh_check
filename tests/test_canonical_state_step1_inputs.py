from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    HLT_PROFILE_V1,
    HLT_PROFILE_V2_REALISTIC,
    HLT_PROFILE_V2_REALISTIC_VERSION,
    fixed_hlt_params_dict,
    fixed_hlt_params_from_profile,
    fixed_hlt_params_from_strength,
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
from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_CONTRACT,
    CANONICAL_STATE_EXPERIMENT_NAME,
    CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES,
    CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
    CANONICAL_STATE_HLT_PROFILE,
    CANONICAL_STATE_HLT_PROFILE_VERSION,
    CANONICAL_STATE_INPUTS_CONTRACT,
    CANONICAL_STATE_LABEL_FILTER,
    CANONICAL_STATE_LABEL_NAMES,
    CANONICAL_STATE_SPLIT_ORDER,
    CanonicalStateInputContractConfig,
    build_canonical_state_step1_input_audit_report,
    canonical_state_hlt_params_dict,
    canonical_state_split_sizes,
    default_canonical_state_experiment_layout,
    default_canonical_state_input_contract_config,
    require_canonical_state_step1_input_contract,
    write_canonical_state_step1_input_audit_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_canonical_state_step1_inputs.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_canonical_state_step1_inputs", AUDIT_PATH)
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


def _toy_manifest(*, include_all_splits: bool = True) -> SplitManifest:
    split_names = SPLIT_ORDER if include_all_splits else SPLIT_ORDER[:-1]
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        if split in split_names:
            _, _, _, jet_ids = _toy_tokens(split_index)
        else:
            jet_ids = []
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


def _write_hlt_cache(
    manifest: SplitManifest,
    manifest_path: Path,
    cache_dir: Path,
    *,
    profile: str = HLT_PROFILE_V2_REALISTIC,
    strength: float = 2.5,
    splits: tuple[str, ...] = CANONICAL_STATE_SPLIT_ORDER,
) -> None:
    manifest_sha = manifest_hash(manifest)
    for split in splits:
        split_index = SPLIT_ORDER.index(split)
        view = _toy_view(split, split_index=split_index, source_manifest_hash=manifest_sha)
        if profile == HLT_PROFILE_V1:
            params = fixed_hlt_params_from_strength(strength)
        else:
            params = fixed_hlt_params_from_profile(profile, strength)
        generate_and_cache_hlt_view(
            view,
            cache_dir,
            seed=DEFAULT_HLT_SEEDS[split],
            params=params,
            hlt_degradation_strength=strength,
        )
    save_split_manifest(manifest, manifest_path)


class CanonicalStateStep1InputTests(unittest.TestCase):
    def test_config_locks_hlt_v2_s2p5_high_data_contract(self):
        cfg = default_canonical_state_input_contract_config()
        payload = cfg.to_dict()

        self.assertEqual(cfg.label_names, tuple(LABEL_NAMES))
        self.assertEqual(cfg.label_names, CANONICAL_STATE_LABEL_NAMES)
        self.assertEqual(cfg.label_filter, CANONICAL_STATE_LABEL_FILTER)
        self.assertEqual(cfg.split_sizes, CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES)
        self.assertEqual(cfg.hlt_profile, HLT_PROFILE_V2_REALISTIC)
        self.assertEqual(CANONICAL_STATE_HLT_PROFILE, HLT_PROFILE_V2_REALISTIC)
        self.assertEqual(cfg.hlt_profile_version, HLT_PROFILE_V2_REALISTIC_VERSION)
        self.assertEqual(CANONICAL_STATE_HLT_PROFILE_VERSION, HLT_PROFILE_V2_REALISTIC_VERSION)
        self.assertEqual(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH, 2.5)
        self.assertEqual(
            canonical_state_hlt_params_dict(),
            fixed_hlt_params_dict(fixed_hlt_params_from_profile(HLT_PROFILE_V2_REALISTIC, 2.5)),
        )
        self.assertEqual(payload["contract"], CANONICAL_STATE_CONTRACT)
        self.assertEqual(payload["experiment_name"], CANONICAL_STATE_EXPERIMENT_NAME)
        self.assertTrue(payload["confirm_final_test"])

    def test_config_rejects_contract_drift(self):
        bad_configs = [
            {"label_names": tuple(reversed(CANONICAL_STATE_LABEL_NAMES))},
            {"label_filter": tuple(reversed(CANONICAL_STATE_LABEL_FILTER))},
            {"num_classes": 2},
            {"split_sizes": {**CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES, "stack_train": 2_999_999}},
            {"hlt_profile": HLT_PROFILE_V1},
            {"hlt_profile_version": "v0"},
            {"hlt_degradation_strength": 1.0},
            {"raw_token_dim": 13},
            {"confirm_final_test": False},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CanonicalStateInputContractConfig(**kwargs)

    def test_size_helper_allows_smoke_overrides(self):
        smoke = {
            "model_train": 10,
            "model_val": 10,
            "stack_train": 10,
            "stack_val": 10,
            "final_test": 10,
        }
        self.assertEqual(canonical_state_split_sizes(smoke), smoke)

    def test_step1_audit_validates_v2_cache_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            hlt_cache_dir = root / "hlt_cache"
            _write_hlt_cache(manifest, manifest_path, hlt_cache_dir)

            expected = canonical_state_split_sizes(
                {split: 10 for split in CANONICAL_STATE_SPLIT_ORDER}
            )
            report = build_canonical_state_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                expected_split_sizes=expected,
            )
            result = write_canonical_state_step1_input_audit_reports(
                manifest_path,
                hlt_cache_dir,
                root / "audit",
                expected_split_sizes=expected,
            )
            audit_report_exists = Path(result["audit_report"]).exists()
            summary_exists = Path(result["summary"]).exists()

        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["contract"], CANONICAL_STATE_INPUTS_CONTRACT)
        self.assertEqual(report["hlt_profile"], HLT_PROFILE_V2_REALISTIC)
        self.assertEqual(report["hlt_profile_version"], HLT_PROFILE_V2_REALISTIC_VERSION)
        self.assertEqual(report["hlt_degradation_strength"], 2.5)
        for split in CANONICAL_STATE_SPLIT_ORDER:
            item = report["audits"]["hlt_cache"]["split_reports"][split]
            self.assertTrue(item["ok"], item["problems"])
            self.assertEqual(item["n_jets"], 10)
            self.assertEqual(item["hlt_profile"], HLT_PROFILE_V2_REALISTIC)
            self.assertEqual(item["hlt_profile_version"], HLT_PROFILE_V2_REALISTIC_VERSION)
            self.assertEqual(item["hlt_params"], canonical_state_hlt_params_dict())
            self.assertIn("metadata_sha256", item)
        self.assertTrue(result["ok"])
        self.assertTrue(audit_report_exists)
        self.assertTrue(summary_exists)

    def test_require_step1_contract_raises_on_bad_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            hlt_cache_dir = root / "hlt_cache"
            _write_hlt_cache(manifest, manifest_path, hlt_cache_dir, strength=1.0)

            with self.assertRaisesRegex(ValueError, "Step 1 input contract failed"):
                require_canonical_state_step1_input_contract(
                    manifest_path,
                    hlt_cache_dir,
                    expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
                )

    def test_step1_audit_rejects_wrong_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            hlt_cache_dir = root / "hlt_cache"
            _write_hlt_cache(manifest, manifest_path, hlt_cache_dir, profile=HLT_PROFILE_V1, strength=2.5)

            report = build_canonical_state_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("HLT profile" in item for item in report["problems"]))

    def test_step1_audit_rejects_wrong_strength(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            hlt_cache_dir = root / "hlt_cache"
            _write_hlt_cache(manifest, manifest_path, hlt_cache_dir, strength=1.0)

            report = build_canonical_state_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("HLT degradation strength" in item for item in report["problems"]))

    def test_step1_audit_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            hlt_cache_dir = root / "hlt_cache"
            _write_hlt_cache(manifest, manifest_path, hlt_cache_dir)

            metadata_path = hlt_cache_dir / "model_val_fixed_hlt_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_manifest_hash"] = "stale"
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            report = build_canonical_state_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("source_manifest_hash" in item for item in report["problems"]))

    def test_step1_audit_rejects_missing_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            hlt_cache_dir = root / "hlt_cache"
            _write_hlt_cache(
                manifest,
                manifest_path,
                hlt_cache_dir,
                splits=tuple(split for split in CANONICAL_STATE_SPLIT_ORDER if split != "final_test"),
            )

            report = build_canonical_state_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                expected_split_sizes={split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("final_test" in item for item in report["problems"]))

    def test_audit_script_defaults_to_canonical_state_layout_and_accepts_smoke_sizes(self):
        module = _load_audit_module()
        args = module.parse_args([])
        layout = default_canonical_state_experiment_layout(output_root="checkpoints")
        self.assertEqual(args.manifest, str(layout.split_manifest_path))
        self.assertEqual(args.hlt_cache_dir, str(layout.hlt_cache_dir))
        self.assertEqual(args.output_dir, str(layout.step1_audit_dir))

        smoke = module.parse_args(
            [
                "--expected-model-train",
                "10",
                "--expected-model-val",
                "10",
                "--expected-stack-train",
                "10",
                "--expected-stack-val",
                "10",
                "--expected-final-test",
                "10",
            ]
        )
        self.assertEqual(
            module.expected_size_overrides(smoke),
            {split: 10 for split in CANONICAL_STATE_SPLIT_ORDER},
        )
