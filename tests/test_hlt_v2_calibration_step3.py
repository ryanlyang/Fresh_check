import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_calibration_module():
    spec = importlib.util.spec_from_file_location(
        "calibrate_hlt_v2_profile_test_module",
        REPO_ROOT / "scripts" / "calibrate_hlt_v2_profile.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_view():
    n_jets = 6
    max_constits = 8
    tokens = np.zeros((n_jets, max_constits, 14), dtype=np.float32)
    mask = np.ones((n_jets, max_constits), dtype=bool)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    jet_ids = []
    for jet in range(n_jets):
        jet_ids.append(JetIdentity(file=f"class{int(labels[jet])}.root", entry=jet, label=int(labels[jet])))
        for idx in range(max_constits):
            pt = 0.04 + 0.035 * idx + 0.002 * jet
            eta = -0.5 + 0.04 * idx
            phi = -0.3 + 0.05 * idx
            tokens[jet, idx, 0] = pt
            tokens[jet, idx, 1] = eta
            tokens[jet, idx, 2] = phi
            tokens[jet, idx, 3] = pt * np.cosh(eta)
            tokens[jet, idx, 5 + (idx % 5)] = 1.0
    return JetView(tokens=tokens, mask=mask, labels=labels, jet_ids=jet_ids, split="model_val")


class HLTv2CalibrationStep3Tests(unittest.TestCase):
    def test_select_balanced_identities_limits_per_class(self):
        module = load_calibration_module()
        view = make_view()

        selected = module.select_balanced_identities(
            view.jet_ids,
            max_jets=None,
            max_jets_per_class=1,
            seed=123,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(sorted(identity.label for identity in selected), [0, 1, 2])

    def test_evaluate_strength_zero_is_identity_row(self):
        module = load_calibration_module()
        view = make_view()

        rows = module.evaluate_strengths(view, strengths=[0.0, 1.5], hlt_seed=1054)

        self.assertEqual(rows[0]["strength"], 0.0)
        self.assertEqual(rows[0]["drop_total_fraction"], 0.0)
        self.assertEqual(rows[0]["jet_pt_frac_shift_mean"], 0.0)
        self.assertEqual(rows[0]["jet_pt_abs_frac_shift_p90"], 0.0)
        self.assertGreaterEqual(rows[1]["drop_total_fraction"], rows[0]["drop_total_fraction"])
        self.assertIn("class_QCD_drop_total_fraction", rows[0])

    def test_report_writers_create_csv_markdown_and_json(self):
        module = load_calibration_module()
        view = make_view()
        rows = module.evaluate_strengths(view, strengths=[0.0, 1.0], hlt_seed=1054)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "hlt_v2_calibration_summary.csv"
            md_path = root / "hlt_v2_calibration_summary.md"
            json_path = root / "hlt_v2_calibration_summary.json"
            module.write_csv(rows, csv_path)
            module.write_markdown(
                rows,
                md_path,
                split="model_val",
                subset_size=len(view.jet_ids),
                source_manifest_hash="source-hash",
                subset_manifest_hash="subset-hash",
            )
            module.write_json_report({"ok": True, "rows": rows}, json_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("drop_total_fraction", csv_path.read_text(encoding="utf-8"))
            self.assertIn("HLT V2 Calibration Summary", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
