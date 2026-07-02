import unittest
from pathlib import Path

from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM

from teacher_logit_reco.privileged_distill_10class import (
    PD10_CONTRACT,
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_EXPERIMENT_NAME,
    PD10_LABEL_FILTER,
    PD10_LABEL_NAMES,
    PD10_NUM_CLASSES,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_STUDENT_INIT_SCRATCH,
    PD10_STUDENT_INIT_WARM_START,
    PD10_TARGET_CONFIDENCE_WEIGHTED,
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_TOP3,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    PD10_TEACHER_OFFLINE,
    PD10ExperimentConfig,
    PD10ExperimentLayout,
    PD10StudentVariantSpec,
    PD10TeacherSpec,
    build_pd10_core_student_variants,
    build_pd10_priority_student_variants,
    build_pd10_teacher_specs,
    default_pd10_experiment_config,
    default_pd10_experiment_layout,
    normalize_pd10_split_name,
    normalize_pd10_student_init_mode,
    normalize_pd10_student_target_mode,
    normalize_pd10_teacher_target,
    pd10_config_manifest,
    pd10_student_variant_name,
    pd10_teacher_model_name,
)


class PD10Step1ConfigTests(unittest.TestCase):
    def test_default_config_declares_full_10class_5m_contract(self):
        cfg = default_pd10_experiment_config()
        payload = cfg.to_dict()

        self.assertEqual(cfg.label_names, tuple(LABEL_NAMES))
        self.assertEqual(cfg.label_names, PD10_LABEL_NAMES)
        self.assertEqual(cfg.label_filter, PD10_LABEL_FILTER)
        self.assertEqual(cfg.num_classes, PD10_NUM_CLASSES)
        self.assertEqual(tuple(cfg.split_sizes.keys()), PD10_SPLIT_ORDER)
        self.assertEqual(cfg.split_sizes, PD10_SPLIT_SIZES)
        self.assertEqual(cfg.split_sizes["model_train"], 5_000_000)
        self.assertEqual(cfg.split_sizes["model_val"], 1_000_000)
        self.assertEqual(cfg.split_sizes["final_test"], 1_000_000)
        self.assertEqual(cfg.default_temperature, PD10_DEFAULT_TEMPERATURE)
        self.assertEqual(cfg.default_alpha, PD10_DEFAULT_ALPHA)
        self.assertEqual(cfg.raw_token_dim, RAW_TOKEN_DIM)
        self.assertTrue(cfg.confirm_final_test)
        self.assertEqual(payload["contract"], PD10_CONTRACT)
        self.assertEqual(payload["experiment_name"], PD10_EXPERIMENT_NAME)
        self.assertEqual(len(payload["core_student_variants"]), 8)
        self.assertEqual(len(payload["priority_student_variants"]), 4)

    def test_normalizers_cover_expected_aliases(self):
        self.assertEqual(normalize_pd10_split_name("model_train"), "model_train")
        self.assertEqual(normalize_pd10_teacher_target("ce_only"), PD10_TEACHER_NONE)
        self.assertEqual(normalize_pd10_teacher_target("self-distill"), PD10_TEACHER_HLT)
        self.assertEqual(normalize_pd10_teacher_target("offline_part"), PD10_TEACHER_OFFLINE)
        self.assertEqual(normalize_pd10_teacher_target("hlt+offline"), PD10_TEACHER_DUAL_VIEW)
        self.assertEqual(normalize_pd10_student_init_mode("from_scratch"), PD10_STUDENT_INIT_SCRATCH)
        self.assertEqual(normalize_pd10_student_init_mode("warmstart"), PD10_STUDENT_INIT_WARM_START)
        self.assertEqual(normalize_pd10_student_target_mode("full"), PD10_TARGET_FULL_LOGITS)
        self.assertEqual(normalize_pd10_student_target_mode("top-k"), PD10_TARGET_TOP3)
        self.assertEqual(normalize_pd10_student_target_mode("confidence"), PD10_TARGET_CONFIDENCE_WEIGHTED)
        with self.assertRaises(ValueError):
            normalize_pd10_teacher_target("mystery_teacher")
        with self.assertRaises(ValueError):
            normalize_pd10_split_name("stack_train")

    def test_teacher_specs_have_strict_allowed_inputs_and_names(self):
        specs = build_pd10_teacher_specs()
        self.assertEqual([spec.teacher_target for spec in specs], ["hlt", "offline", "dual_view"])
        self.assertEqual(pd10_teacher_model_name("hlt"), "hlt_part_teacher_10class")
        self.assertEqual(pd10_teacher_model_name("offline"), "offline_part_teacher_10class")
        self.assertEqual(pd10_teacher_model_name("dual"), "dual_view_logit_teacher_10class")

        hlt, offline, dual_view = specs
        self.assertTrue(hlt.uses_hlt)
        self.assertFalse(hlt.uses_offline)
        self.assertEqual(hlt.allowed_inputs, "HLT_only")
        self.assertFalse(offline.uses_hlt)
        self.assertTrue(offline.uses_offline)
        self.assertEqual(offline.allowed_inputs, "offline_only_train_time_privileged")
        self.assertTrue(dual_view.uses_hlt)
        self.assertTrue(dual_view.uses_offline)
        self.assertEqual(dual_view.allowed_inputs, "HLT_plus_offline_train_time_privileged")

        with self.assertRaises(ValueError):
            PD10TeacherSpec(PD10_TEACHER_NONE)
        with self.assertRaises(ValueError):
            PD10TeacherSpec(PD10_TEACHER_HLT, allowed_inputs="offline_only_train_time_privileged")
        with self.assertRaises(ValueError):
            PD10TeacherSpec(PD10_TEACHER_OFFLINE, uses_hlt=True)

    def test_student_variant_names_and_core_matrix_are_unambiguous(self):
        self.assertEqual(
            pd10_student_variant_name(PD10_STUDENT_INIT_SCRATCH, PD10_TEACHER_NONE),
            "pd10_student_scratch_ce_only",
        )
        self.assertEqual(
            pd10_student_variant_name("warm", "dual", "full", temperature=2.0, kd_alpha=0.5),
            "pd10_student_warm_start_dual_view_full_logits_t2_a0p5",
        )
        self.assertEqual(
            pd10_student_variant_name("warm", "dual", "top3", temperature=2.0, kd_alpha=0.5),
            "pd10_student_warm_start_dual_view_top3_t2_a0p5",
        )

        core = build_pd10_core_student_variants()
        self.assertEqual(len(core), 8)
        self.assertEqual(len({spec.name for spec in core}), 8)
        self.assertEqual(core[0].name, "pd10_student_scratch_ce_only")
        self.assertEqual(core[1].teacher_model_name, "hlt_part_teacher_10class")
        self.assertEqual(core[3].teacher_model_name, "dual_view_logit_teacher_10class")
        self.assertEqual(core[4].name, "pd10_student_warm_start_ce_only")
        self.assertFalse(core[0].requires_teacher)
        self.assertTrue(core[1].requires_teacher)
        self.assertEqual(core[0].kd_alpha, 0.0)

        priority = build_pd10_priority_student_variants()
        self.assertEqual(len(priority), 4)
        self.assertTrue(all(spec.init_mode == PD10_STUDENT_INIT_WARM_START for spec in priority))
        self.assertTrue(all(spec.teacher_target == PD10_TEACHER_DUAL_VIEW for spec in priority))
        self.assertIn(PD10_TARGET_TOP3, {spec.target_mode for spec in priority})
        self.assertIn(PD10_TARGET_CONFIDENCE_WEIGHTED, {spec.target_mode for spec in priority})

    def test_student_teacher_compatibility_rejects_bad_combinations(self):
        with self.assertRaises(ValueError):
            PD10StudentVariantSpec(
                init_mode=PD10_STUDENT_INIT_SCRATCH,
                teacher_target=PD10_TEACHER_NONE,
                target_mode=PD10_TARGET_TOP3,
            )
        with self.assertRaises(ValueError):
            PD10StudentVariantSpec(
                init_mode=PD10_STUDENT_INIT_SCRATCH,
                teacher_target=PD10_TEACHER_HLT,
                kd_alpha=0.0,
            )
        with self.assertRaises(ValueError):
            PD10StudentVariantSpec(
                init_mode=PD10_STUDENT_INIT_SCRATCH,
                teacher_target=PD10_TEACHER_HLT,
                temperature=0.0,
            )

    def test_config_rejects_setup_drift(self):
        bad_configs = [
            {"label_names": tuple(reversed(PD10_LABEL_NAMES))},
            {"label_filter": tuple(reversed(PD10_LABEL_FILTER))},
            {"num_classes": 2},
            {"split_sizes": {"model_train": 5_000_000, "model_val": 1_000_000}},
            {"split_sizes": {**PD10_SPLIT_SIZES, "model_train": 4_999_999}},
            {"teacher_targets": ("none", "hlt", "offline")},
            {"student_init_modes": ("scratch",)},
            {"target_modes": ("full_logits", "top3")},
            {"default_temperature": 0.0},
            {"default_alpha": 0.0},
            {"top_k": 5},
            {"raw_token_dim": 13},
            {"confirm_final_test": False},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PD10ExperimentConfig(**kwargs)

    def test_from_mapping_rejects_unknown_keys_and_round_trips(self):
        cfg = PD10ExperimentConfig.from_mapping(default_pd10_experiment_config().to_dict())

        self.assertEqual(cfg.label_names, PD10_LABEL_NAMES)
        with self.assertRaises(ValueError):
            PD10ExperimentConfig.from_mapping({"not_a_real_key": 1})

    def test_layout_paths_use_fresh_pd10_namespace(self):
        layout = default_pd10_experiment_layout(output_root="/tmp/checkpoints")

        self.assertEqual(layout.root, Path("/tmp/checkpoints") / PD10_EXPERIMENT_NAME)
        self.assertEqual(layout.split_manifest_path, layout.root / "split_manifest" / "split_manifest.json.gz")
        self.assertEqual(layout.hlt_cache_dir, layout.root / "hlt_cache")
        self.assertEqual(
            layout.teacher_checkpoint(PD10_TEACHER_HLT),
            layout.root / "teachers" / "hlt_part_teacher_10class" / "best_model_val.pt",
        )
        self.assertEqual(
            layout.teacher_logit_cache_dir(PD10_TEACHER_DUAL_VIEW),
            layout.root / "teacher_logits" / "dual_view_logit_teacher_10class",
        )
        variant = PD10StudentVariantSpec(
            init_mode=PD10_STUDENT_INIT_WARM_START,
            teacher_target=PD10_TEACHER_DUAL_VIEW,
        )
        self.assertEqual(layout.student_dir(variant), layout.root / "students" / variant.name)
        self.assertIn("final_report", layout.to_dict()["final_report_dir"])
        self.assertNotIn("teacher_logit_reco_crossarch_500k", layout.root.as_posix())

    def test_manifest_combines_config_and_layout(self):
        manifest = pd10_config_manifest()

        self.assertEqual(manifest["config"]["contract"], PD10_CONTRACT)
        self.assertTrue(manifest["layout"]["root"].endswith(PD10_EXPERIMENT_NAME))


if __name__ == "__main__":
    unittest.main()
