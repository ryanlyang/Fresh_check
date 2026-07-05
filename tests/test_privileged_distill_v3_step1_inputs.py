from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    HLT_PROFILE_V1,
    fixed_hlt_params_dict,
    fixed_hlt_params_from_profile,
    fixed_hlt_params_from_strength,
)
from jetclass_fresh.hlt_cache import generate_and_cache_hlt_view
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
from teacher_logit_reco.privileged_distill_v3 import (
    PDV3_CONTRACT,
    PDV3_EXPERIMENT_NAME,
    PDV3_HLT_DEGRADATION_STRENGTH,
    PDV3_HLT_PROFILE,
    PDV3_INPUTS_CONTRACT,
    PDV3_LABEL_FILTER,
    PDV3_LABEL_NAMES,
    PDV3_MANIFEST_SPLIT_ORDER,
    PDV3_MANIFEST_SPLIT_SIZES,
    PDV3_MODEL_SPLIT_ORDER,
    PDV3_MODEL_SPLIT_SIZES,
    PDV3_STACK_PLACEHOLDER_SPLIT_SIZES,
    PDV3InputContractConfig,
    build_pdv3_step1_input_audit_report,
    default_pdv3_experiment_layout,
    default_pdv3_input_contract_config,
    pdv3_hlt_params_dict,
    pdv3_manifest_split_sizes,
    pdv3_model_split_sizes,
    pdv3_stack_placeholder_split_sizes,
    write_pdv3_step1_input_audit_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_pdv3_step1_inputs.py"
SBATCH_DIR = REPO_ROOT / "sbatch"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_pdv3_step1_inputs", AUDIT_PATH)
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
        if split in {"stack_train", "stack_val"}:
            jet_ids = jet_ids[:2]
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


class PDV3Step1InputTests(unittest.TestCase):
    def test_config_locks_default_hlt_profile_full_10class_contract(self):
        cfg = default_pdv3_input_contract_config()
        payload = cfg.to_dict()

        self.assertEqual(cfg.label_names, tuple(LABEL_NAMES))
        self.assertEqual(cfg.label_names, PDV3_LABEL_NAMES)
        self.assertEqual(cfg.label_filter, PDV3_LABEL_FILTER)
        self.assertEqual(cfg.model_split_sizes, PDV3_MODEL_SPLIT_SIZES)
        self.assertEqual(cfg.stack_placeholder_split_sizes, PDV3_STACK_PLACEHOLDER_SPLIT_SIZES)
        self.assertEqual(cfg.manifest_split_sizes, PDV3_MANIFEST_SPLIT_SIZES)
        self.assertEqual(PDV3_MODEL_SPLIT_ORDER, ("model_train", "model_val", "final_test"))
        self.assertEqual(PDV3_MANIFEST_SPLIT_ORDER, ("model_train", "model_val", "stack_train", "stack_val", "final_test"))
        self.assertEqual(PDV3_HLT_PROFILE, HLT_PROFILE_V1)
        self.assertEqual(cfg.hlt_profile, PDV3_HLT_PROFILE)
        self.assertEqual(PDV3_HLT_DEGRADATION_STRENGTH, 0.2)
        self.assertEqual(
            pdv3_hlt_params_dict(),
            fixed_hlt_params_dict(fixed_hlt_params_from_profile(PDV3_HLT_PROFILE, 0.2)),
        )
        self.assertEqual(payload["contract"], PDV3_CONTRACT)
        self.assertEqual(payload["experiment_name"], PDV3_EXPERIMENT_NAME)
        self.assertTrue(payload["require_offline_cache"])

    def test_config_rejects_contract_drift(self):
        bad_configs = [
            {"label_names": tuple(reversed(PDV3_LABEL_NAMES))},
            {"label_filter": tuple(reversed(PDV3_LABEL_FILTER))},
            {"num_classes": 2},
            {"model_split_sizes": {**PDV3_MODEL_SPLIT_SIZES, "model_train": 4_999_999}},
            {"stack_placeholder_split_sizes": {"stack_train": 10, "stack_val": 11}},
            {"hlt_degradation_strength": 0.6},
            {"raw_token_dim": 13},
            {"require_offline_cache": False},
            {"confirm_final_test": False},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PDV3InputContractConfig(**kwargs)

    def test_size_helpers_allow_smoke_overrides(self):
        smoke_model = {"model_train": 10, "model_val": 10, "final_test": 10}
        smoke_stack = {"stack_train": 2, "stack_val": 2}
        self.assertEqual(pdv3_model_split_sizes(smoke_model), smoke_model)
        self.assertEqual(pdv3_stack_placeholder_split_sizes(smoke_stack), smoke_stack)
        self.assertEqual(
            pdv3_manifest_split_sizes(smoke_model, smoke_stack),
            {
                "model_train": 10,
                "model_val": 10,
                "stack_train": 2,
                "stack_val": 2,
                "final_test": 10,
            },
        )

    def test_step1_audit_validates_hlt_and_offline_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            save_split_manifest(manifest, manifest_path)
            manifest_sha = manifest_hash(manifest)
            hlt_cache_dir = root / "hlt_cache"
            offline_cache_dir = root / "offline_cache"
            for split in PDV3_MODEL_SPLIT_ORDER:
                split_index = SPLIT_ORDER.index(split)
                view = _toy_view(split, split_index=split_index, source_manifest_hash=manifest_sha)
                generate_and_cache_hlt_view(
                    view,
                    hlt_cache_dir,
                    seed=DEFAULT_HLT_SEEDS[split],
                    params=fixed_hlt_params_from_strength(0.2),
                    hlt_degradation_strength=0.2,
                )
                save_cached_offline_view(view, offline_cache_dir)

            report = build_pdv3_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                offline_cache_dir,
                expected_split_sizes={"model_train": 10, "model_val": 10, "final_test": 10},
                expected_placeholder_sizes={"stack_train": 2, "stack_val": 2},
            )
            result = write_pdv3_step1_input_audit_reports(
                manifest_path,
                hlt_cache_dir,
                offline_cache_dir,
                root / "audit",
                expected_split_sizes={"model_train": 10, "model_val": 10, "final_test": 10},
                expected_placeholder_sizes={"stack_train": 2, "stack_val": 2},
            )
            audit_report_exists = Path(result["audit_report"]).exists()
            summary_exists = Path(result["summary"]).exists()

        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["contract"], PDV3_INPUTS_CONTRACT)
        self.assertEqual(report["hlt_profile"], PDV3_HLT_PROFILE)
        self.assertEqual(report["hlt_degradation_strength"], 0.2)
        for split in PDV3_MODEL_SPLIT_ORDER:
            hlt_item = report["audits"]["hlt_cache"]["split_reports"][split]
            offline_item = report["audits"]["offline_cache"]["split_reports"][split]
            self.assertEqual(hlt_item["jet_identity_hash"], offline_item["jet_identity_hash"])
            self.assertEqual(hlt_item["hlt_profile"], PDV3_HLT_PROFILE)
            self.assertEqual(hlt_item["hlt_params"], pdv3_hlt_params_dict())
        self.assertTrue(result["ok"])
        self.assertTrue(audit_report_exists)
        self.assertTrue(summary_exists)

    def test_step1_audit_detects_wrong_hlt_strength(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _toy_manifest()
            manifest_path = root / "split_manifest.json.gz"
            save_split_manifest(manifest, manifest_path)
            manifest_sha = manifest_hash(manifest)
            hlt_cache_dir = root / "hlt_cache"
            offline_cache_dir = root / "offline_cache"
            for split in PDV3_MODEL_SPLIT_ORDER:
                split_index = SPLIT_ORDER.index(split)
                view = _toy_view(split, split_index=split_index, source_manifest_hash=manifest_sha)
                generate_and_cache_hlt_view(
                    view,
                    hlt_cache_dir,
                    seed=DEFAULT_HLT_SEEDS[split],
                    params=fixed_hlt_params_from_strength(0.6),
                    hlt_degradation_strength=0.6,
                )
                save_cached_offline_view(view, offline_cache_dir)

            report = build_pdv3_step1_input_audit_report(
                manifest_path,
                hlt_cache_dir,
                offline_cache_dir,
                expected_split_sizes={"model_train": 10, "model_val": 10, "final_test": 10},
                expected_placeholder_sizes={"stack_train": 2, "stack_val": 2},
            )

        self.assertFalse(report["ok"])
        self.assertTrue(any("HLT params do not match PDV3 HLT profile" in item for item in report["problems"]))

    def test_audit_script_defaults_to_pdv3_layout_and_accepts_smoke_sizes(self):
        module = _load_audit_module()
        args = module.parse_args([])
        layout = default_pdv3_experiment_layout(output_root="checkpoints")
        self.assertEqual(args.manifest, str(layout.split_manifest_path))
        self.assertEqual(args.hlt_cache_dir, str(layout.hlt_cache_dir))
        self.assertEqual(args.offline_cache_dir, str(layout.offline_cache_dir))
        self.assertEqual(args.output_dir, str(layout.step1_audit_dir))

        smoke = module.parse_args(
            [
                "--expected-model-train",
                "10",
                "--expected-model-val",
                "10",
                "--expected-final-test",
                "10",
                "--expected-stack-train",
                "2",
                "--expected-stack-val",
                "2",
            ]
        )
        expected_model, expected_stack = module.expected_size_overrides(smoke)
        self.assertEqual(expected_model, {"model_train": 10, "model_val": 10, "final_test": 10})
        self.assertEqual(expected_stack, {"stack_train": 2, "stack_val": 2})

    def test_sbatch_step1_wiring_uses_hlt0p2_and_paired_caches(self):
        common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")
        build_splits = (SBATCH_DIR / "run_pdv3_build_splits.sh").read_text(encoding="utf-8")
        build_hlt = (SBATCH_DIR / "run_pdv3_build_hlt_cache.sh").read_text(encoding="utf-8")
        offline = (SBATCH_DIR / "run_pdv3_cache_offline_inputs.sh").read_text(encoding="utf-8")
        audit = (SBATCH_DIR / "run_pdv3_audit_inputs.sh").read_text(encoding="utf-8")
        submit = (SBATCH_DIR / "submit_pdv3_step1_inputs.sh").read_text(encoding="utf-8")

        self.assertIn("PDV3_ROOT:=${OUTPUT_ROOT}/privileged_distill_v3_av10_adapter_hlt0p2_5m", common)
        self.assertIn("PDV3_HLT_DEGRADATION_STRENGTH:=0.2", common)
        self.assertIn("PDV3_MODEL_TRAIN_SIZE:=5000000", common)
        self.assertIn("PDV3_MODEL_VAL_SIZE:=1000000", common)
        self.assertIn("PDV3_FINAL_TEST_SIZE:=1000000", common)
        self.assertIn("--out \"${PDV3_MANIFEST_PATH}\"", build_splits)
        self.assertIn("--hlt-degradation-strength \"${PDV3_HLT_DEGRADATION_STRENGTH}\"", build_hlt)
        self.assertIn("scripts/cache_architecture_view_offline_inputs.py", offline)
        self.assertIn("--offline-cache-dir \"${PDV3_OFFLINE_CACHE_DIR}\"", audit)
        self.assertIn("scripts/audit_pdv3_step1_inputs.py", audit)
        self.assertIn("run_pdv3_build_splits.sh", submit)
        self.assertIn("run_pdv3_build_hlt_cache.sh", submit)
        self.assertIn("run_pdv3_cache_offline_inputs.sh", submit)
        self.assertIn("run_pdv3_audit_inputs.sh", submit)


if __name__ == "__main__":
    unittest.main()
