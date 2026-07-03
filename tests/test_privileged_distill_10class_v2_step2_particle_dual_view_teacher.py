import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import save_prediction_block
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.privileged_distill_10class import (
    PD10_NUM_CLASSES,
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS,
    PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT,
    PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS,
    PD10_PARTICLE_DUAL_VIEW_MODEL_NAME,
    PD10_REPRESENTATION_DIM,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    PD10_V2_STEP2_CACHE_EXPERIMENT_STEP,
    PD10ParticleDualViewTeacherCacheConfig,
    PD10ParticleDualViewTeacherTrainConfig,
    PD10PairedParticleViewDataset,
    align_pd10_hlt_offline_views,
    build_pd10_particle_dual_view_logit_block,
    collate_pd10_particle_dual_view_batch,
    default_pd10_experiment_layout,
    load_pd10_particle_dual_view_logit_block,
    pd10_particle_dual_view_cache_selection_seed,
    pd10_particle_dual_view_logit_cache_dir,
    pd10_particle_dual_view_representation_cache_dir,
    pd10_particle_dual_view_teacher_checkpoint,
    pd10_particle_dual_view_teacher_dir,
    validate_pd10_particle_dual_view_logit_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_pd10_particle_dual_view_teacher.py"
CACHE_SCRIPT_PATH = REPO_ROOT / "scripts" / "cache_pd10_particle_dual_view_teacher.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_ids(labels, *, prefix="jet"):
    return [
        JetIdentity(file=f"{prefix}_class{int(label)}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]


def make_view(labels=(0, 1, 2), *, split="model_val", view_name="fixed_hlt", reverse=False) -> JetView:
    labels_np = np.asarray(labels, dtype=np.int64)
    n_jets = int(labels_np.shape[0])
    tokens = np.zeros((n_jets, 4, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.ones((n_jets, 4), dtype=bool)
    for jet in range(n_jets):
        for particle in range(4):
            pt = 10.0 + jet + particle
            eta = 0.1 * particle
            phi = -0.2 * particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt * np.cosh(eta) + 0.1
            tokens[jet, particle, 4] = (-1.0, 0.0, 1.0, 0.0)[particle]
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 10] = 0.01 * particle
            tokens[jet, particle, 11] = 0.02
            tokens[jet, particle, 12] = 0.03 * particle
            tokens[jet, particle, 13] = 0.04
    jet_ids = make_ids(labels_np, prefix="paired")
    if reverse:
        order = np.arange(n_jets)[::-1]
        tokens = tokens[order]
        mask = mask[order]
        labels_np = labels_np[order]
        jet_ids = [jet_ids[int(index)] for index in order]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels_np,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": view_name, "hlt_params": {} if view_name == "fixed_hlt" else None},
    )


def make_cache_config(root: Path, *, confirm_final_test=True) -> PD10ParticleDualViewTeacherCacheConfig:
    checkpoint = root / "particle_dual_view.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    return PD10ParticleDualViewTeacherCacheConfig(
        checkpoint=str(checkpoint),
        manifest_path=str(root / "split_manifest.json.gz"),
        hlt_cache_dir=str(root / "hlt_cache"),
        logit_output_dir=str(root / "teacher_logits"),
        representation_output_dir=str(root / "teacher_representations"),
        splits=("model_val", "final_test") if confirm_final_test else ("model_val",),
        confirm_final_test=confirm_final_test,
    )


class PD10V2Step2ParticleDualViewTeacherTests(unittest.TestCase):
    def test_train_and_cache_configs_paths_and_guardrails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hlt_ckpt = root / "hlt.pt"
            off_ckpt = root / "offline.pt"
            hlt_ckpt.write_bytes(b"hlt")
            off_ckpt.write_bytes(b"offline")
            train_cfg = PD10ParticleDualViewTeacherTrainConfig(
                output_dir=str(root / "teacher"),
                manifest_path=str(root / "split_manifest.json.gz"),
                hlt_cache_dir=str(root / "hlt_cache"),
                hlt_teacher_checkpoint=str(hlt_ckpt),
                offline_teacher_checkpoint=str(off_ckpt),
                epochs=2,
                model_size="tiny",
            )
            self.assertEqual(train_cfg.teacher_target, PD10_TEACHER_PARTICLE_DUAL_VIEW)
            self.assertEqual(train_cfg.model_name, PD10_PARTICLE_DUAL_VIEW_MODEL_NAME)
            self.assertEqual(train_cfg.checkpoint_path, root / "teacher" / "best_model_val.pt")
            self.assertEqual(train_cfg.allowed_inputs, "HLT_plus_offline_train_time_privileged")

            with self.assertRaises(ValueError):
                PD10ParticleDualViewTeacherTrainConfig(
                    output_dir="out",
                    manifest_path="manifest",
                    hlt_cache_dir="hlt_cache",
                    hlt_teacher_checkpoint="hlt.pt",
                    offline_teacher_checkpoint="offline.pt",
                    epochs=0,
                )
            with self.assertRaises(ValueError):
                PD10ParticleDualViewTeacherTrainConfig(
                    output_dir="out",
                    manifest_path="manifest",
                    hlt_cache_dir="hlt_cache",
                    hlt_teacher_checkpoint="hlt.pt",
                    offline_teacher_checkpoint="offline.pt",
                    max_train_jets=PD10_SPLIT_SIZES["model_train"] + 1,
                )

            with self.assertRaises(ValueError):
                make_cache_config(root, confirm_final_test=False).__class__(
                    checkpoint=str(root / "particle_dual_view.pt"),
                    manifest_path=str(root / "split_manifest.json.gz"),
                    hlt_cache_dir=str(root / "hlt_cache"),
                    logit_output_dir=str(root / "teacher_logits"),
                    representation_output_dir=str(root / "teacher_representations"),
                    splits=("final_test",),
                )
            cache_cfg = make_cache_config(root, confirm_final_test=True)
            self.assertEqual(cache_cfg.model_name, PD10_PARTICLE_DUAL_VIEW_MODEL_NAME)
            self.assertEqual(cache_cfg.splits, ("model_val", "final_test"))
            self.assertEqual(cache_cfg.logit_dir, root / "teacher_logits" / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME)
            self.assertEqual(
                pd10_particle_dual_view_cache_selection_seed(cache_cfg, "model_val"),
                cache_cfg.control_seed + 1009 * 2,
            )

            layout_root = root / "checkpoints"
            self.assertEqual(
                pd10_particle_dual_view_teacher_dir(output_root=layout_root),
                layout_root / "privileged_distill_10class_5m" / "teachers" / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME,
            )
            self.assertEqual(
                pd10_particle_dual_view_teacher_checkpoint(output_root=layout_root),
                pd10_particle_dual_view_teacher_dir(output_root=layout_root) / "best_model_val.pt",
            )
            self.assertEqual(
                pd10_particle_dual_view_logit_cache_dir(output_root=layout_root),
                layout_root / "privileged_distill_10class_5m" / "teacher_logits" / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME,
            )
            self.assertEqual(
                pd10_particle_dual_view_representation_cache_dir(output_root=layout_root),
                layout_root
                / "privileged_distill_10class_5m"
                / "teacher_representations"
                / PD10_PARTICLE_DUAL_VIEW_MODEL_NAME,
            )

    def test_paired_alignment_and_collate_keep_views_separate(self):
        torch = require_torch()
        hlt_view = make_view(labels=(0, 1, 2), view_name="fixed_hlt")
        offline_reversed = make_view(labels=(0, 1, 2), view_name="offline", reverse=True)

        aligned_hlt, aligned_offline = align_pd10_hlt_offline_views(hlt_view, offline_reversed)
        self.assertEqual([jet.key() for jet in aligned_hlt.jet_ids], [jet.key() for jet in aligned_offline.jet_ids])
        self.assertTrue(np.array_equal(aligned_hlt.labels, aligned_offline.labels))

        dataset = PD10PairedParticleViewDataset(aligned_hlt, aligned_offline)
        batch = collate_pd10_particle_dual_view_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["labels"].shape, torch.Size([2]))
        self.assertEqual(batch["hlt"]["features"].shape[0], 2)
        self.assertEqual(batch["offline"]["features"].shape[0], 2)
        self.assertEqual(batch["hlt"]["features"].shape[1], batch["offline"]["features"].shape[1])
        self.assertEqual(batch["hlt"]["mask"].dtype, torch.bool)
        self.assertEqual(batch["offline"]["mask"].dtype, torch.bool)

        bad_offline = make_view(labels=(0, 9, 2), view_name="offline")
        bad_offline.jet_ids = list(hlt_view.jet_ids)
        with self.assertRaises(ValueError):
            align_pd10_hlt_offline_views(hlt_view, bad_offline)

    def test_build_save_load_particle_dual_view_logit_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_cache_config(root, confirm_final_test=True)
            labels = np.asarray([0, 1, 2], dtype=np.int64)
            jet_ids = make_ids(labels)
            logits = np.full((3, PD10_NUM_CLASSES), -2.0, dtype=np.float32)
            logits[np.arange(3), labels] = 2.0

            block = build_pd10_particle_dual_view_logit_block(
                cfg,
                "model_val",
                logits=logits,
                labels=labels,
                jet_ids=jet_ids,
                source_metadata={
                    "hlt_content_hash": "hlt-hash",
                    "source_manifest_hash": "manifest-hash",
                    "subset_selection": {"selected": 3},
                },
                checkpoint_payload={"epoch": 3, "experiment_step": "unit_test"},
            )
            metadata = save_prediction_block(block, cfg.logit_output_dir)
            validate_pd10_particle_dual_view_logit_metadata(metadata, split="model_val")

            loaded = load_pd10_particle_dual_view_logit_block(cfg.logit_output_dir, "model_val")
            self.assertEqual(loaded.logits.shape, (3, PD10_NUM_CLASSES))
            self.assertEqual(loaded.metadata["contract"], PD10_PARTICLE_DUAL_VIEW_LOGIT_CACHE_CONTRACT)
            self.assertEqual(loaded.metadata["experiment_step"], PD10_V2_STEP2_CACHE_EXPERIMENT_STEP)
            self.assertEqual(loaded.metadata["teacher_target"], "particle_dual_view")
            self.assertEqual(loaded.metadata["allowed_inputs"], "HLT_plus_offline_train_time_privileged")
            self.assertEqual(loaded.metadata["student_deployment_inputs"], "HLT_only")
            self.assertTrue(loaded.metadata["teacher_logits_train_time_only"])
            self.assertTrue(loaded.metadata["uses_raw_offline_particles"])
            self.assertTrue(loaded.metadata["teacher_inference_requires_offline_inputs"])
            self.assertFalse(loaded.metadata["inference_export_requires_teacher_features"])

            bad = dict(metadata)
            bad["allowed_inputs"] = "HLT_only"
            with self.assertRaises(ValueError):
                validate_pd10_particle_dual_view_logit_metadata(bad, split="model_val")

            bad = dict(metadata)
            bad["uses_raw_offline_particles"] = False
            with self.assertRaises(ValueError):
                validate_pd10_particle_dual_view_logit_metadata(bad, split="model_val")

            bad = dict(metadata)
            bad["teacher_inference_requires_offline_inputs"] = False
            with self.assertRaises(ValueError):
                validate_pd10_particle_dual_view_logit_metadata(bad, split="model_val")

            bad = dict(metadata)
            bad["inference_export_requires_teacher_features"] = True
            with self.assertRaises(ValueError):
                validate_pd10_particle_dual_view_logit_metadata(bad, split="model_val")

    def test_scripts_defaults_use_canonical_layouts(self):
        train_module = load_script(TRAIN_SCRIPT_PATH, "train_pd10_particle_dual_view_teacher")
        cache_module = load_script(CACHE_SCRIPT_PATH, "cache_pd10_particle_dual_view_teacher")
        layout = default_pd10_experiment_layout(output_root="checkpoints")

        train_args = train_module.parse_args([])
        self.assertEqual(train_args.output_dir, str(pd10_particle_dual_view_teacher_dir(output_root="checkpoints")))
        self.assertEqual(train_args.hlt_teacher_checkpoint, str(layout.teacher_checkpoint("hlt")))
        self.assertEqual(train_args.offline_teacher_checkpoint, str(layout.teacher_checkpoint("offline")))
        self.assertEqual(train_args.epochs, PD10_PARTICLE_DUAL_VIEW_DEFAULT_EPOCHS)

        cache_args = cache_module.parse_args(["--confirm-final-test"])
        self.assertEqual(cache_args.checkpoint, str(pd10_particle_dual_view_teacher_checkpoint(output_root="checkpoints")))
        self.assertEqual(cache_args.logit_output_dir, str(layout.teacher_logits_dir))
        self.assertEqual(cache_args.representation_output_dir, str(layout.root / "teacher_representations"))
        self.assertEqual(cache_args.splits, list(PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS))
        self.assertTrue(cache_args.confirm_final_test)


if __name__ == "__main__":
    unittest.main()
