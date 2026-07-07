import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, save_hlt_cache
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.hlt_self_dualview import (
    HLT2OnlyTrainConfig,
    HLTTTAControlConfig,
    HLT_SDV_HLT2_ONLY_CONTRACT,
    HLT_SDV_TTA_CONTRACT,
    hlt_sdv_dual_hlt2_variant_name,
    load_hlt_sdv_dataset,
    make_hlt_sdv_data_loader,
    move_hlt_sdv_batch_to_device,
    run_hlt_tta_control,
    train_hlt2_only_control,
)


torch = require_torch()
REPO_ROOT = Path(__file__).resolve().parents[1]
HLT2_SCRIPT = REPO_ROOT / "scripts" / "train_pd10_hlt2_only_control.py"
TTA_SCRIPT = REPO_ROOT / "scripts" / "evaluate_pd10_hlt_tta_control.py"


class DummyPartModel(torch.nn.Module):
    def __init__(self, in_dim: int = 17, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(int(in_dim), int(num_classes))
        self.config = {"dummy_part_model": True, "input_dim": int(in_dim), "num_classes": int(num_classes)}
        with torch.no_grad():
            self.fc.weight.fill_(0.02)
            self.fc.bias.copy_(torch.linspace(-0.2, 0.2, int(num_classes)))

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors, mask
        return self.fc(features.mean(dim=-1))


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
    labels = np.asarray([index % 4 for index in range(n_jets)], dtype=np.int64)
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
        save_cache(make_view(split, view_name="hlt2", offset=0.4), hlt2_dir)
    return hlt_dir, hlt2_dir


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HLTSDVStep6ControlsTest(unittest.TestCase):
    def test_final_test_guardrails(self):
        with self.assertRaisesRegex(ValueError, "final-test evaluation requires"):
            HLT2OnlyTrainConfig(
                output_dir="out",
                hlt2_cache_dir="hlt2",
                hlt_teacher_checkpoint="hlt.pt",
                evaluate_final_test=True,
                confirm_final_test=False,
            )
        with self.assertRaisesRegex(ValueError, "final-test evaluation requires"):
            HLTTTAControlConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                hlt2_cache_dir="hlt2",
                hlt_teacher_checkpoint="hlt.pt",
                evaluate_final_test=True,
                confirm_final_test=False,
            )

    def test_hlt2_only_training_writes_deployable_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, hlt2_dir = write_synthetic_caches(root)
            checkpoint = root / "hlt_teacher.pt"
            checkpoint.write_bytes(b"unused unit checkpoint")
            config = HLT2OnlyTrainConfig(
                output_dir=str(root / "hlt2_only"),
                hlt2_cache_dir=str(hlt2_dir),
                hlt_teacher_checkpoint=str(checkpoint),
                epochs=2,
                batch_size=3,
                eval_batch_size=3,
                max_train_jets=6,
                max_val_jets=6,
                max_final_test_jets=6,
                device="cpu",
                amp=False,
                initialize_from_hlt_checkpoint=False,
                evaluate_final_test=False,
                overwrite=True,
            )

            report = train_hlt2_only_control(config, model=DummyPartModel())

            self.assertTrue(report["ok"])
            self.assertEqual(report["contract"], HLT_SDV_HLT2_ONLY_CONTRACT)
            self.assertFalse(report["requires_offline_inputs"])
            self.assertFalse(report["requires_teacher_features"])
            self.assertTrue(config.checkpoint_path.exists())
            metadata = report["model_val_prediction_metadata"]
            self.assertEqual(metadata["contract"], HLT_SDV_HLT2_ONLY_CONTRACT)
            self.assertEqual(metadata["source_view"], "hlt2")
            self.assertEqual(metadata["rich_metrics"]["n_jets"], 6)

    def test_tta_control_is_literal_hlt_hlt2_logit_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hlt_dir, hlt2_dir = write_synthetic_caches(root)
            checkpoint = root / "hlt_teacher.pt"
            checkpoint.write_bytes(b"unused unit checkpoint")
            model = DummyPartModel()
            config = HLTTTAControlConfig(
                output_dir=str(root / "tta"),
                hlt_cache_dir=str(hlt_dir),
                hlt2_cache_dir=str(hlt2_dir),
                hlt_teacher_checkpoint=str(checkpoint),
                batch_size=3,
                max_val_jets=6,
                max_final_test_jets=6,
                device="cpu",
                evaluate_final_test=False,
                overwrite=True,
            )

            report = run_hlt_tta_control(config, model=model)
            block = load_prediction_block(config.prediction_dir, config.model_name, "model_val")

            dataset = load_hlt_sdv_dataset(hlt_dir, "model_val", hlt2_cache_dir=hlt2_dir, max_jets=6)
            loader = make_hlt_sdv_data_loader(dataset, batch_size=3, shuffle=False, num_workers=0, seed=123)
            expected_chunks = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = move_hlt_sdv_batch_to_device(batch, torch.device("cpu"))
                    hlt_logits = model(
                        batch["hlt_inputs"]["points"],
                        batch["hlt_inputs"]["features"],
                        batch["hlt_inputs"]["lorentz_vectors"],
                        batch["hlt_inputs"]["mask"],
                    )
                    hlt2_logits = model(
                        batch["hlt2_inputs"]["points"],
                        batch["hlt2_inputs"]["features"],
                        batch["hlt2_inputs"]["lorentz_vectors"],
                        batch["hlt2_inputs"]["mask"],
                    )
                    expected_chunks.append((0.5 * (hlt_logits + hlt2_logits)).numpy())
            expected = np.concatenate(expected_chunks, axis=0).astype(np.float32)

            self.assertTrue(report["ok"])
            self.assertEqual(report["contract"], HLT_SDV_TTA_CONTRACT)
            np.testing.assert_allclose(block.logits, expected, rtol=1.0e-6, atol=1.0e-6)
            self.assertFalse(block.metadata["requires_offline_inputs"])
            self.assertFalse(block.metadata["requires_teacher_features"])
            self.assertEqual(block.metadata["logit_combination"], "0.5 * HLT_logits + 0.5 * HLT2_logits")

    def test_cli_build_config_defaults(self):
        hlt2_module = load_script(HLT2_SCRIPT, "train_pd10_hlt2_only_control")
        tta_module = load_script(TTA_SCRIPT, "evaluate_pd10_hlt_tta_control")
        with tempfile.TemporaryDirectory() as tmp:
            pd10_root = Path(tmp) / "pd10"
            hlt2_args = hlt2_module.parse_args(["--pd10-root", str(pd10_root), "--skip-final-test"])
            hlt2_config = hlt2_module.build_config(hlt2_args)
            self.assertEqual(hlt2_config.variant_name, "hlt2_only_part_s0p20")
            self.assertIn("s0p20", str(hlt2_config.hlt2_cache_dir))

            tta_args = tta_module.parse_args(["--pd10-root", str(pd10_root), "--skip-final-test"])
            tta_config = tta_module.build_config(tta_args)
            self.assertEqual(tta_config.variant_name, "tta_hlt_part_hlt_plus_hlt2_s0p20")
            self.assertIn("s0p20", str(tta_config.hlt2_cache_dir))


if __name__ == "__main__":
    unittest.main()
