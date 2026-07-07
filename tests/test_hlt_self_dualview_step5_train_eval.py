import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, save_hlt_cache
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.hlt_self_dualview import (
    HLTSDVEvalConfig,
    HLTSDVTrainConfig,
    HLT_SDV_BRANCH2_HLT2,
    HLT_SDV_BRANCH2_SAME_HLT,
    HLT_SDV_PREDICTION_CONTRACT,
    HLT_SDV_VARIANT_SAME_VIEW,
    evaluate_hlt_sdv_model,
    hlt_sdv_dual_hlt2_variant_name,
    train_hlt_sdv_model,
)


torch = require_torch()
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_pd10_hlt_self_dualview.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_pd10_hlt_self_dualview.py"


class DummySDVModel(torch.nn.Module):
    def __init__(self, in_dim: int = 17, hidden: int = 8, num_classes: int = 10) -> None:
        super().__init__()
        self.hlt_branch = torch.nn.Linear(int(in_dim), int(hidden))
        self.hlt2_branch = torch.nn.Linear(int(in_dim), int(hidden))
        self.classifier = torch.nn.Linear(int(hidden) * 4, int(num_classes))
        self.config = {
            "num_classes": int(num_classes),
            "dummy_sdv_model": True,
            "branch_dim": int(hidden),
            "fusion_hidden_dim": int(hidden) * 4,
            "representation_dim": int(hidden) * 4,
            "dropout": 0.0,
        }

    def _embed(self, branch, inputs):
        return branch(inputs["features"].mean(dim=-1))

    def branch_parameters(self):
        yield from self.hlt_branch.parameters()
        yield from self.hlt2_branch.parameters()

    def head_parameters(self):
        yield from self.classifier.parameters()

    def set_branches_trainable(self, trainable: bool) -> None:
        for parameter in self.branch_parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(self, hlt_inputs, hlt2_inputs):
        h1 = self._embed(self.hlt_branch, hlt_inputs)
        h2 = self._embed(self.hlt2_branch, hlt2_inputs)
        return self.classifier(torch.cat([h1, h2, torch.abs(h2 - h1), h1 * h2], dim=1))


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


def make_view(split: str, *, view_name: str, offset: float = 0.0, n_jets: int = 6) -> JetView:
    labels = np.asarray([index % 3 for index in range(n_jets)], dtype=np.int64)
    jet_ids = [
        JetIdentity(file=f"{split}_class{int(label)}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    tokens = np.zeros((n_jets, 4, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.ones((n_jets, 4), dtype=bool)
    for jet in range(n_jets):
        for particle in range(4):
            pt = 5.0 + float(labels[jet]) + 0.1 * particle + offset
            eta = 0.05 * particle
            phi = -0.03 * particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt * np.cosh(eta) + 0.1
            tokens[jet, particle, 4] = 1.0
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": view_name, "uses_offline_particles": False},
    )
    jet_files, _, _ = _identity_arrays(jet_ids)
    content_hash = _content_hash(view)
    view.metadata.update(
        {
            "jet_files": jet_files,
            "jet_identity_hash": jet_identity_hash(jet_ids),
            "hlt_content_hash": content_hash,
            "hlt2_content_hash": content_hash if view_name == "hlt2" else None,
            "allowed_inputs": "HLT_only" if view_name == "hlt2" else None,
        }
    )
    return view


def save_cache(view: JetView, cache_dir: Path) -> None:
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


def write_synthetic_caches(root: Path) -> tuple[Path, Path]:
    hlt_dir = root / "hlt_cache"
    hlt2_dir = root / "hlt2_cache"
    for split in ("model_train", "model_val", "final_test"):
        save_cache(make_view(split, view_name="fixed_hlt", offset=0.0), hlt_dir)
        save_cache(make_view(split, view_name="hlt2", offset=0.3), hlt2_dir)
    return hlt_dir, hlt2_dir


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HLTSDVStep5TrainEvalTest(unittest.TestCase):
    def test_config_guardrails(self):
        with self.assertRaisesRegex(ValueError, "hlt2_cache_dir is required"):
            HLTSDVTrainConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                hlt_teacher_checkpoint="hlt.pt",
                variant_name=hlt_sdv_dual_hlt2_variant_name(0.20),
                evaluate_final_test=False,
            )
        with self.assertRaisesRegex(ValueError, "must not provide hlt2_cache_dir"):
            HLTSDVTrainConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                hlt2_cache_dir="hlt2",
                hlt_teacher_checkpoint="hlt.pt",
                variant_name=HLT_SDV_VARIANT_SAME_VIEW,
                evaluate_final_test=False,
            )
        with self.assertRaisesRegex(ValueError, "final-test evaluation requires"):
            HLTSDVTrainConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                hlt2_cache_dir="hlt2",
                hlt_teacher_checkpoint="hlt.pt",
                variant_name=hlt_sdv_dual_hlt2_variant_name(0.20),
                evaluate_final_test=True,
                confirm_final_test=False,
            )

    def test_train_writes_checkpoint_model_val_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hlt_dir, hlt2_dir = write_synthetic_caches(root)
            checkpoint = root / "hlt_teacher.pt"
            checkpoint.write_bytes(b"unused in unit test")
            config = HLTSDVTrainConfig(
                output_dir=str(root / "model"),
                hlt_cache_dir=str(hlt_dir),
                hlt2_cache_dir=str(hlt2_dir),
                hlt_teacher_checkpoint=str(checkpoint),
                variant_name=hlt_sdv_dual_hlt2_variant_name(0.20),
                branch2_mode=HLT_SDV_BRANCH2_HLT2,
                epochs=2,
                head_warmup_epochs=1,
                batch_size=3,
                eval_batch_size=3,
                max_train_jets=6,
                max_val_jets=6,
                max_final_test_jets=6,
                device="cpu",
                amp=False,
                initialize_branches=False,
                evaluate_final_test=False,
                overwrite=True,
            )

            report = train_hlt_sdv_model(config, model=DummySDVModel())

            self.assertTrue(report["ok"])
            self.assertEqual(report["branch2_mode"], HLT_SDV_BRANCH2_HLT2)
            self.assertFalse(report["requires_offline_inputs"])
            self.assertFalse(report["requires_teacher_features"])
            self.assertTrue(config.checkpoint_path.exists())
            self.assertTrue((Path(config.output_dir) / "training_curves.json").exists())
            metadata = report["model_val_prediction_metadata"]
            self.assertEqual(metadata["contract"], HLT_SDV_PREDICTION_CONTRACT)
            self.assertEqual(metadata["model_name"], config.variant_name)
            self.assertEqual(metadata["rich_metrics"]["n_jets"], 6)
            self.assertIn("confusion_matrix", metadata["rich_metrics"])

    def test_evaluate_uses_hlt_and_hlt2_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hlt_dir, hlt2_dir = write_synthetic_caches(root)
            checkpoint = root / "selected.pt"
            torch.save({"model_state_dict": DummySDVModel().state_dict(), "epoch": 1, "model_config": {}}, checkpoint)
            config = HLTSDVEvalConfig(
                checkpoint=str(checkpoint),
                output_dir=str(root / "eval"),
                hlt_cache_dir=str(hlt_dir),
                hlt2_cache_dir=str(hlt2_dir),
                variant_name=hlt_sdv_dual_hlt2_variant_name(0.20),
                branch2_mode=HLT_SDV_BRANCH2_HLT2,
                split="model_val",
                batch_size=3,
                device="cpu",
                overwrite=True,
            )

            report = evaluate_hlt_sdv_model(config, model=DummySDVModel())

            self.assertTrue(report["ok"])
            self.assertEqual(report["metrics"]["n_jets"], 6)
            self.assertFalse(report["requires_offline_inputs"])
            self.assertFalse(report["requires_teacher_features"])
            self.assertEqual(report["prediction_metadata"]["contract"], HLT_SDV_PREDICTION_CONTRACT)

    def test_cli_build_config_defaults(self):
        train_module = load_script(TRAIN_SCRIPT, "train_pd10_hlt_self_dualview")
        eval_module = load_script(EVAL_SCRIPT, "evaluate_pd10_hlt_self_dualview")
        with tempfile.TemporaryDirectory() as tmp:
            pd10_root = Path(tmp) / "pd10"
            args = train_module.parse_args(
                [
                    "--pd10-root",
                    str(pd10_root),
                    "--variant",
                    HLT_SDV_VARIANT_SAME_VIEW,
                    "--skip-final-test",
                ]
            )
            train_config = train_module.build_config(args)
            self.assertEqual(train_config.branch2_mode, HLT_SDV_BRANCH2_SAME_HLT)
            self.assertIsNone(train_config.hlt2_cache_dir)

            eval_args = eval_module.parse_args(
                [
                    "--pd10-root",
                    str(pd10_root),
                    "--variant",
                    hlt_sdv_dual_hlt2_variant_name(0.20),
                    "--split",
                    "model_val",
                ]
            )
            eval_config = eval_module.build_config(eval_args)
            self.assertEqual(eval_config.branch2_mode, HLT_SDV_BRANCH2_HLT2)
            self.assertIn("s0p20", str(eval_config.hlt2_cache_dir))


if __name__ == "__main__":
    unittest.main()
