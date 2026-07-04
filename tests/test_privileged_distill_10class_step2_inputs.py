import importlib.util
from pathlib import Path
import unittest

from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength
from jetclass_fresh.jetclass_data import LABEL_NAMES

from teacher_logit_reco.privileged_distill_10class import (
    PD10_HLT_DEGRADATION_STRENGTH,
    PD10_MANIFEST_SPLIT_ORDER,
    PD10_MANIFEST_SPLIT_SIZES,
    PD10_MANIFEST_STACK_SPLIT_SIZES,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    default_pd10_experiment_layout,
    pd10_expected_split_sizes,
    pd10_hlt_params_dict,
    pd10_manifest_split_sizes,
    pd10_stack_placeholder_split_sizes,
)
from teacher_logit_reco.privileged_distill_10class.inputs import (
    class_balance_problems,
    hlt_cache_split_problems,
    placeholder_split_size_problems,
    split_size_problems,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_pd10_step2_splits_hlt_cache.py"
SBATCH_DIR = REPO_ROOT / "sbatch"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_pd10_step2_splits_hlt_cache", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PD10Step2InputTests(unittest.TestCase):
    def test_manifest_split_sizes_include_5m_1m_1m_and_stack_placeholders(self):
        self.assertEqual(PD10_SPLIT_ORDER, ("model_train", "model_val", "final_test"))
        self.assertEqual(PD10_SPLIT_SIZES["model_train"], 5_000_000)
        self.assertEqual(PD10_SPLIT_SIZES["model_val"], 1_000_000)
        self.assertEqual(PD10_SPLIT_SIZES["final_test"], 1_000_000)
        self.assertEqual(PD10_MANIFEST_SPLIT_ORDER, ("model_train", "model_val", "stack_train", "stack_val", "final_test"))
        self.assertEqual(PD10_MANIFEST_STACK_SPLIT_SIZES, {"stack_train": 10, "stack_val": 10})
        self.assertEqual(
            pd10_manifest_split_sizes(),
            {
                "model_train": 5_000_000,
                "model_val": 1_000_000,
                "stack_train": 10,
                "stack_val": 10,
                "final_test": 1_000_000,
            },
        )
        self.assertEqual(pd10_manifest_split_sizes(), PD10_MANIFEST_SPLIT_SIZES)

    def test_expected_split_sizes_are_model_facing_pd10_sizes(self):
        self.assertEqual(pd10_expected_split_sizes(), {split: PD10_SPLIT_SIZES[split] for split in PD10_SPLIT_ORDER})
        self.assertNotIn("stack_train", pd10_expected_split_sizes())
        self.assertEqual(pd10_stack_placeholder_split_sizes(), {"stack_train": 10, "stack_val": 10})
        smoke_model = {"model_train": 20_000, "model_val": 5_000, "final_test": 10_000}
        smoke_stack = {"stack_train": 2, "stack_val": 3}
        self.assertEqual(pd10_expected_split_sizes(smoke_model), smoke_model)
        self.assertEqual(pd10_stack_placeholder_split_sizes(smoke_stack), smoke_stack)
        self.assertEqual(
            pd10_manifest_split_sizes(smoke_model, smoke_stack),
            {
                "model_train": 20_000,
                "model_val": 5_000,
                "stack_train": 2,
                "stack_val": 3,
                "final_test": 10_000,
            },
        )

    def test_pd10_hlt_profile_defaults_to_hlt0p6(self):
        self.assertEqual(PD10_HLT_DEGRADATION_STRENGTH, 0.6)
        self.assertEqual(pd10_hlt_params_dict(), fixed_hlt_params_dict(fixed_hlt_params_from_strength(0.6)))
        self.assertNotEqual(pd10_hlt_params_dict(), fixed_hlt_params_dict(fixed_hlt_params_from_strength(1.0)))

    def test_split_size_problems_check_pd10_splits_and_placeholders_separately(self):
        declared = pd10_manifest_split_sizes()
        actual = pd10_manifest_split_sizes()
        self.assertEqual(split_size_problems(declared, actual), [])
        self.assertEqual(placeholder_split_size_problems(declared, actual), [])

        bad_actual = dict(actual)
        bad_actual["model_train"] = 4_999_990
        problems = split_size_problems(declared, bad_actual)
        self.assertIn("model_train actual count is 4999990, expected 5000000", problems)

        bad_placeholder = dict(actual)
        bad_placeholder["stack_val"] = 20
        problems = placeholder_split_size_problems(declared, bad_placeholder)
        self.assertIn("stack_val placeholder actual count is 20, expected 10", problems)

    def test_class_balance_problems_require_balanced_ten_class_pd10_splits(self):
        balanced = {
            split: {label: PD10_SPLIT_SIZES[split] // 10 for label in LABEL_NAMES}
            for split in PD10_SPLIT_ORDER
        }
        self.assertEqual(class_balance_problems(balanced), [])

        unbalanced = {split: dict(counts) for split, counts in balanced.items()}
        unbalanced["final_test"]["QCD"] -= 1
        problems = class_balance_problems(unbalanced)
        self.assertIn("final_test/QCD count is 99999, expected 100000", problems)

    def test_hlt_cache_split_problems_detect_pd10_metadata_mismatches(self):
        item = {
            "n_jets": 5_000_000,
            "seed": 1053,
            "hlt_params": pd10_hlt_params_dict(),
            "expected_hlt_params": pd10_hlt_params_dict(),
            "source_manifest_hash": "abc",
            "content_hash_matches_metadata": True,
        }
        self.assertEqual(
            hlt_cache_split_problems(
                item,
                base_problems=[],
                expected_size=5_000_000,
                expected_seed=1053,
                manifest_sha="abc",
            ),
            [],
        )

        bad = dict(item)
        bad.update(
            {
                "n_jets": 4_999_999,
                "seed": 999,
                "hlt_params": fixed_hlt_params_dict(fixed_hlt_params_from_strength(1.0)),
                "content_hash_matches_metadata": False,
            }
        )
        problems = hlt_cache_split_problems(
            bad,
            base_problems=["base problem"],
            expected_size=5_000_000,
            expected_seed=1053,
            manifest_sha="abc",
        )
        self.assertIn("base problem", problems)
        self.assertIn("n_jets is 4999999, expected 5000000", problems)
        self.assertIn("seed is 999, expected 1053", problems)
        self.assertIn("HLT params do not match configured PD10 fixed-HLT profile (strength=0.6)", problems)
        self.assertIn("recomputed HLT content hash does not match metadata", problems)

    def test_audit_script_imports_and_uses_default_layout(self):
        module = load_audit_module()
        args = module.parse_args([])

        layout = default_pd10_experiment_layout(output_root="checkpoints")
        self.assertEqual(args.manifest, str(layout.split_manifest_path))
        self.assertEqual(args.hlt_cache_dir, str(layout.hlt_cache_dir))
        self.assertEqual(args.output_dir, str(layout.step2_audit_dir))
        self.assertIsNone(args.expected_model_train)

        smoke = module.parse_args(
            [
                "--expected-model-train",
                "20000",
                "--expected-model-val",
                "5000",
                "--expected-final-test",
                "10000",
                "--expected-stack-train",
                "2",
                "--expected-stack-val",
                "3",
            ]
        )
        expected_model, expected_stack = module.expected_size_overrides(smoke)
        self.assertEqual(expected_model, {"model_train": 20_000, "model_val": 5_000, "final_test": 10_000})
        self.assertEqual(expected_stack, {"stack_train": 2, "stack_val": 3})

    def test_pd10_sbatch_runners_are_wired_to_canonical_step2_inputs(self):
        common = (SBATCH_DIR / "common.sh").read_text(encoding="utf-8")
        split = (SBATCH_DIR / "run_pd10_build_splits.sh").read_text(encoding="utf-8")
        cache = (SBATCH_DIR / "run_pd10_build_hlt_cache.sh").read_text(encoding="utf-8")
        audit = (SBATCH_DIR / "run_pd10_audit_splits_hlt_cache.sh").read_text(encoding="utf-8")

        self.assertIn("PD10_ROOT:=${OUTPUT_ROOT}/privileged_distill_10class_5m", common)
        self.assertIn("PD10_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data", common)
        self.assertIn("PD10_MODEL_TRAIN_SIZE:=5000000", common)
        self.assertIn("PD10_MODEL_VAL_SIZE:=1000000", common)
        self.assertIn("PD10_STACK_TRAIN_SIZE:=10", common)
        self.assertIn("PD10_STACK_VAL_SIZE:=10", common)
        self.assertIn("PD10_FINAL_TEST_SIZE:=1000000", common)
        self.assertIn("PD10_HLT_SPLITS:=model_train model_val final_test", common)
        self.assertIn("*hlt0p4*|*hlt0P4*) PD10_HLT_DEGRADATION_STRENGTH=0.4", common)
        self.assertIn("*hlt0p6*|*hlt0P6*) PD10_HLT_DEGRADATION_STRENGTH=0.6", common)
        self.assertIn("*) PD10_HLT_DEGRADATION_STRENGTH=0.6", common)
        self.assertIn("export PD10_HLT_DEGRADATION_STRENGTH", common)

        self.assertIn("scripts/build_jetclass_splits.py", split)
        self.assertIn("--data-dir \"${PD10_DATA_DIR}\"", split)
        self.assertIn("--out \"${PD10_MANIFEST_PATH}\"", split)
        self.assertIn("--model-train \"${PD10_MODEL_TRAIN_SIZE}\"", split)
        self.assertIn("--stack-train \"${PD10_STACK_TRAIN_SIZE}\"", split)

        self.assertIn("scripts/build_fixed_hlt_cache.py", cache)
        self.assertIn("--manifest \"${PD10_MANIFEST_PATH}\"", cache)
        self.assertIn("--cache-dir \"${PD10_HLT_CACHE_DIR}\"", cache)
        self.assertIn("--hlt-degradation-strength \"${PD10_HLT_DEGRADATION_STRENGTH}\"", cache)
        self.assertIn('fresh_split_words split_args "${PD10_HLT_SPLITS}"', cache)

        self.assertIn("scripts/audit_pd10_step2_splits_hlt_cache.py", audit)
        self.assertIn("--output-dir \"${PD10_STEP2_AUDIT_DIR}\"", audit)
        self.assertIn("--expected-model-train \"${PD10_MODEL_TRAIN_SIZE}\"", audit)
        self.assertIn("--expected-stack-val \"${PD10_STACK_VAL_SIZE}\"", audit)
        self.assertIn('fresh_assert_json_ok "${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json"', audit)


if __name__ == "__main__":
    unittest.main()
