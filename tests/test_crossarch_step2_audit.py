import importlib.util
from pathlib import Path
import unittest

from teacher_logit_reco.crossarch_experiment import SPLIT_ORDER, SPLIT_SIZES


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_crossarch_step2_splits_hlt_cache.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_crossarch_step2_splits_hlt_cache", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CrossArchStep2AuditHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_audit_module()

    def test_expected_split_sizes_are_crossarch_sizes(self):
        self.assertEqual(self.audit.expected_split_sizes(), {split: SPLIT_SIZES[split] for split in SPLIT_ORDER})
        self.assertEqual(self.audit.expected_split_sizes()["stack_train"], 500_000)
        self.assertEqual(self.audit.expected_split_sizes()["stack_val"], 150_000)

    def test_split_size_problems_require_declared_and_actual_counts(self):
        declared = self.audit.expected_split_sizes()
        actual = self.audit.expected_split_sizes()
        self.assertEqual(self.audit.split_size_problems(declared, actual), [])

        bad_actual = dict(actual)
        bad_actual["stack_train"] = 250_000
        problems = self.audit.split_size_problems(declared, bad_actual)
        self.assertIn("stack_train actual count is 250000, expected 500000", problems)

        bad_declared = dict(declared)
        bad_declared["stack_val"] = 50_000
        problems = self.audit.split_size_problems(bad_declared, actual)
        self.assertIn("stack_val declared size is 50000, expected 150000", problems)

    def test_class_balance_problems_require_balanced_ten_class_splits(self):
        balanced = {
            split: {label: SPLIT_SIZES[split] // 10 for label in self.audit.LABEL_NAMES}
            for split in SPLIT_ORDER
        }
        self.assertEqual(self.audit.class_balance_problems(balanced), [])

        unbalanced = {split: dict(counts) for split, counts in balanced.items()}
        unbalanced["model_val"]["QCD"] -= 1
        problems = self.audit.class_balance_problems(unbalanced)
        self.assertIn("model_val/QCD count is 14999, expected 15000", problems)

    def test_hlt_cache_split_problems_detect_metadata_mismatches(self):
        item = {
            "n_jets": 500_000,
            "seed": 1053,
            "hlt_params": {"a": 1.0},
            "expected_hlt_params": {"a": 1.0},
            "source_manifest_hash": "abc",
            "content_hash_matches_metadata": True,
        }
        self.assertEqual(
            self.audit.hlt_cache_split_problems(
                item,
                base_problems=[],
                expected_size=500_000,
                expected_seed=1053,
                manifest_sha="abc",
            ),
            [],
        )

        bad = dict(item)
        bad.update({"seed": 999, "content_hash_matches_metadata": False})
        problems = self.audit.hlt_cache_split_problems(
            bad,
            base_problems=["base problem"],
            expected_size=500_000,
            expected_seed=1053,
            manifest_sha="abc",
        )
        self.assertIn("base problem", problems)
        self.assertIn("seed is 999, expected 1053", problems)
        self.assertIn("recomputed HLT content hash does not match metadata", problems)


if __name__ == "__main__":
    unittest.main()
