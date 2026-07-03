import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.privileged_distill_10class import (
    PD10_DEFAULT_REPRESENTATION_BETA,
    PD10_EXTENDED_STUDENT_TARGET_MODES,
    PD10_EXTENDED_TEACHER_TARGETS,
    PD10_REPRESENTATION_DIM,
    PD10_REPRESENTATION_MODE_COSINE,
    PD10_TARGET_CONFIDENCE_WEIGHTED_PLUS_REP,
    PD10_TARGET_FULL_LOGITS_PLUS_REP,
    PD10_TARGET_REP_ONLY,
    PD10_TARGET_TOP3_PLUS_REP,
    PD10_TEACHER_NONE,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT,
    PD10_TEACHER_REPRESENTATION_SPLITS,
    PD10_V2_STEP1_EXPERIMENT_STEP,
    PD10TeacherRepresentationCacheConfig,
    build_pd10_teacher_representation_block,
    load_pd10_teacher_representation_block,
    normalize_pd10_extended_student_target_mode,
    normalize_pd10_extended_teacher_target,
    normalize_pd10_representation_mode,
    normalize_pd10_representation_teacher_target,
    normalize_pd10_teacher_target,
    pd10_extended_teacher_model_name,
    pd10_teacher_representation_cache_dir,
    pd10_teacher_representation_paths,
    save_pd10_teacher_representation_block,
    validate_pd10_teacher_representation_metadata,
    write_pd10_teacher_representation_manifest,
)


def make_jet_ids(labels=(0, 1, 2)):
    return [
        JetIdentity(file=f"sample_class{int(label)}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]


class PD10V2Step1RepresentationTests(unittest.TestCase):
    def test_v2_config_constants_are_additive_to_pd10_v04(self):
        self.assertEqual(PD10_TEACHER_PARTICLE_DUAL_VIEW, "particle_dual_view")
        self.assertIn(PD10_TEACHER_PARTICLE_DUAL_VIEW, PD10_EXTENDED_TEACHER_TARGETS)
        self.assertEqual(
            pd10_extended_teacher_model_name("particle_dual"),
            "particle_dual_view_teacher_10class",
        )
        self.assertEqual(normalize_pd10_extended_teacher_target("particle-level-dual-view"), "particle_dual_view")
        self.assertEqual(normalize_pd10_representation_teacher_target("pdv"), "particle_dual_view")
        with self.assertRaises(ValueError):
            normalize_pd10_teacher_target("particle_dual_view")
        with self.assertRaises(ValueError):
            normalize_pd10_representation_teacher_target(PD10_TEACHER_NONE)

        self.assertEqual(PD10_REPRESENTATION_DIM, 256)
        self.assertEqual(PD10_DEFAULT_REPRESENTATION_BETA, 0.10)
        self.assertEqual(normalize_pd10_representation_mode("cos"), PD10_REPRESENTATION_MODE_COSINE)
        self.assertEqual(normalize_pd10_extended_student_target_mode("rep"), PD10_TARGET_REP_ONLY)
        self.assertEqual(
            normalize_pd10_extended_student_target_mode("logits-plus-rep"),
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
        )
        self.assertEqual(normalize_pd10_extended_student_target_mode("top3_rep"), PD10_TARGET_TOP3_PLUS_REP)
        self.assertEqual(
            normalize_pd10_extended_student_target_mode("confidence_rep"),
            PD10_TARGET_CONFIDENCE_WEIGHTED_PLUS_REP,
        )
        self.assertTrue(
            {
                PD10_TARGET_REP_ONLY,
                PD10_TARGET_FULL_LOGITS_PLUS_REP,
                PD10_TARGET_TOP3_PLUS_REP,
                PD10_TARGET_CONFIDENCE_WEIGHTED_PLUS_REP,
            }.issubset(set(PD10_EXTENDED_STUDENT_TARGET_MODES))
        )

    def test_representation_cache_config_paths_and_final_test_guardrails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                PD10TeacherRepresentationCacheConfig(
                    teacher_target="particle_dual_view",
                    output_dir=str(root / "teacher_representations"),
                )

            cfg = PD10TeacherRepresentationCacheConfig(
                teacher_target="particle_dual_view",
                output_dir=str(root / "teacher_representations"),
                confirm_final_test=True,
            )
            self.assertEqual(cfg.teacher_target, PD10_TEACHER_PARTICLE_DUAL_VIEW)
            self.assertEqual(cfg.splits, PD10_TEACHER_REPRESENTATION_SPLITS)
            self.assertEqual(cfg.representation_dim, PD10_REPRESENTATION_DIM)
            self.assertEqual(cfg.model_name, "particle_dual_view_teacher_10class")
            self.assertEqual(cfg.teacher_output_dir, Path(cfg.output_dir) / cfg.model_name)
            self.assertEqual(
                pd10_teacher_representation_cache_dir("pdv", output_root=root),
                root / "privileged_distill_10class_5m" / "teacher_representations" / cfg.model_name,
            )
            npz_path, metadata_path = pd10_teacher_representation_paths(
                cfg.output_dir,
                "particle_dual_view",
                "model_val",
            )
            self.assertTrue(npz_path.as_posix().endswith("model_val_representations.npz"))
            self.assertTrue(metadata_path.as_posix().endswith("model_val_representations_metadata.json"))

    def test_build_save_load_representation_block_preserves_alignment_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PD10TeacherRepresentationCacheConfig(
                teacher_target="particle_dual_view",
                output_dir=str(root / "teacher_representations"),
                splits=("model_val",),
                confirm_final_test=False,
            )
            labels = np.asarray([0, 1, 2], dtype=np.int64)
            reps = np.arange(labels.shape[0] * PD10_REPRESENTATION_DIM, dtype=np.float32).reshape(
                labels.shape[0],
                PD10_REPRESENTATION_DIM,
            )
            jet_ids = make_jet_ids(labels)
            block = build_pd10_teacher_representation_block(
                cfg,
                "model_val",
                representations=reps,
                labels=labels,
                jet_ids=jet_ids,
                source_metadata={
                    "source_view": "paired_hlt_offline_particles",
                    "hlt_content_hash": "hlt-hash",
                    "offline_source_manifest_hash": "offline-manifest-hash",
                    "offline_privileged_inputs_loaded": True,
                },
                extra_metadata={"checkpoint_sha256": "abc123"},
            )
            metadata = save_pd10_teacher_representation_block(block, cfg.output_dir)
            validate_pd10_teacher_representation_metadata(
                metadata,
                teacher_target="particle_dual_view",
                split="model_val",
                representation_dim=PD10_REPRESENTATION_DIM,
            )

            loaded = load_pd10_teacher_representation_block(cfg.output_dir, "particle_dual_view", "model_val")
            self.assertEqual(loaded.representations.shape, (3, PD10_REPRESENTATION_DIM))
            self.assertTrue(np.array_equal(loaded.labels, labels))
            self.assertEqual([jet.key() for jet in loaded.jet_ids], [jet.key() for jet in jet_ids])
            self.assertEqual(loaded.metadata["contract"], PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT)
            self.assertEqual(loaded.metadata["experiment_step"], PD10_V2_STEP1_EXPERIMENT_STEP)
            self.assertEqual(loaded.metadata["allowed_inputs"], "HLT_plus_offline_train_time_privileged")
            self.assertEqual(loaded.metadata["student_deployment_inputs"], "HLT_only")
            self.assertTrue(loaded.metadata["teacher_representations_train_time_only"])
            self.assertFalse(loaded.metadata["inference_export_requires_teacher_features"])
            self.assertEqual(loaded.metadata["representation_dim"], PD10_REPRESENTATION_DIM)

            manifest = write_pd10_teacher_representation_manifest(cfg, [loaded.metadata])
            self.assertTrue(manifest["ok"])
            self.assertEqual(manifest["teacher_target"], "particle_dual_view")
            self.assertEqual(manifest["representation_dim"], PD10_REPRESENTATION_DIM)
            self.assertEqual(len(manifest["prediction_rows"]), 1)

    def test_representation_block_refuses_bad_shapes_labels_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PD10TeacherRepresentationCacheConfig(
                teacher_target="particle_dual_view",
                output_dir=str(root / "teacher_representations"),
                splits=("model_val",),
            )
            labels = np.asarray([0, 1, 2], dtype=np.int64)
            jet_ids = make_jet_ids(labels)
            reps = np.zeros((3, PD10_REPRESENTATION_DIM), dtype=np.float32)

            with self.assertRaises(ValueError):
                build_pd10_teacher_representation_block(
                    cfg,
                    "model_val",
                    representations=np.zeros((3, PD10_REPRESENTATION_DIM - 1), dtype=np.float32),
                    labels=labels,
                    jet_ids=jet_ids,
                )
            with self.assertRaises(ValueError):
                build_pd10_teacher_representation_block(
                    cfg,
                    "model_val",
                    representations=reps,
                    labels=np.asarray([0, 9, 2], dtype=np.int64),
                    jet_ids=jet_ids,
                )
            bad_reps = reps.copy()
            bad_reps[0, 0] = np.nan
            with self.assertRaises(FloatingPointError):
                build_pd10_teacher_representation_block(
                    cfg,
                    "model_val",
                    representations=bad_reps,
                    labels=labels,
                    jet_ids=jet_ids,
                )

            valid_block = build_pd10_teacher_representation_block(
                cfg,
                "model_val",
                representations=reps,
                labels=labels,
                jet_ids=jet_ids,
            )
            metadata = save_pd10_teacher_representation_block(valid_block, cfg.output_dir)

            bad_allowed = dict(metadata)
            bad_allowed["allowed_inputs"] = "HLT_only"
            with self.assertRaises(ValueError):
                validate_pd10_teacher_representation_metadata(
                    bad_allowed,
                    teacher_target="particle_dual_view",
                    split="model_val",
                )

            bad_train_time = dict(metadata)
            bad_train_time["teacher_representations_train_time_only"] = False
            with self.assertRaises(ValueError):
                validate_pd10_teacher_representation_metadata(
                    bad_train_time,
                    teacher_target="particle_dual_view",
                    split="model_val",
                )

            bad_inference = dict(metadata)
            bad_inference["inference_export_requires_teacher_features"] = True
            with self.assertRaises(ValueError):
                validate_pd10_teacher_representation_metadata(
                    bad_inference,
                    teacher_target="particle_dual_view",
                    split="model_val",
                )


if __name__ == "__main__":
    unittest.main()

