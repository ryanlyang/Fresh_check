import unittest
from pathlib import Path

from teacher_logit_reco.crossarch_experiment import (
    DEFAULT_FUSERS,
    EXPERIMENT_NAME,
    MIXED4_RECO_TEACHER_PAIRS,
    RECONSTRUCTOR_ARCHITECTURES,
    SAME_FAMILY_RECO_TEACHER_PAIRS,
    SPLIT_ORDER,
    SPLIT_SIZES,
    TEACHER_ARCHITECTURES,
    CrossArchExperimentConfig,
    CrossArchExperimentLayout,
    CrossArchSourceSpec,
    build_fusion_groups,
    build_hlt_source_specs,
    build_reco_source_specs,
    default_crossarch_experiment_config,
    hlt_model_name,
    normalize_reconstructor_architecture,
    normalize_teacher_architecture,
    reco_model_name,
    validate_fusion_groups,
)


class CrossArchExperimentNamingTests(unittest.TestCase):
    def test_architecture_normalization_and_model_names(self):
        self.assertEqual(normalize_reconstructor_architecture("global_transformer"), "gt")
        self.assertEqual(normalize_reconstructor_architecture("PFC"), "pfn")
        self.assertEqual(normalize_teacher_architecture("ParticleTransformer"), "part")
        self.assertEqual(normalize_teacher_architecture("particle_flow"), "pfn")
        self.assertEqual(reco_model_name("gt", "part"), "gt_reco_to_part_teacher")
        self.assertEqual(reco_model_name("particle_net", "PFN"), "pn_reco_to_pfn_teacher")
        self.assertEqual(hlt_model_name("ParticleTransformer"), "hlt_part")
        self.assertEqual(hlt_model_name("pcnn"), "hlt_pcnn")
        with self.assertRaises(ValueError):
            normalize_reconstructor_architecture("not_a_reco")
        with self.assertRaises(ValueError):
            normalize_teacher_architecture("not_a_teacher")

    def test_source_specs_validate_and_serialize(self):
        reco = CrossArchSourceSpec(
            name="pcnn_reco_to_part_teacher",
            source_kind="teacher_logit_reco",
            reco_architecture="particle_cnn",
            teacher_architecture="ParticleTransformer",
        )
        self.assertEqual(reco.reco_architecture, "pcnn")
        self.assertEqual(reco.teacher_architecture, "part")
        self.assertEqual(reco.reconstructor_implementation, "particle_cnn")
        self.assertEqual(reco.to_dict()["source_kind"], "teacher_logit_reco")

        hlt = CrossArchSourceSpec(name="hlt_pfn", source_kind="direct_hlt", hlt_architecture="PFC")
        self.assertEqual(hlt.hlt_architecture, "pfn")
        self.assertIsNone(hlt.reco_architecture)
        with self.assertRaises(ValueError):
            CrossArchSourceSpec(
                name="wrong_name",
                source_kind="teacher_logit_reco",
                reco_architecture="gt",
                teacher_architecture="part",
            )
        with self.assertRaises(ValueError):
            CrossArchSourceSpec(name="hlt_part", source_kind="bad_kind", hlt_architecture="part")


class CrossArchExperimentGridTests(unittest.TestCase):
    def test_default_split_sizes_and_architecture_grid(self):
        self.assertEqual(tuple(SPLIT_SIZES.keys()), SPLIT_ORDER)
        self.assertEqual(SPLIT_SIZES["model_train"], 500_000)
        self.assertEqual(SPLIT_SIZES["model_val"], 150_000)
        self.assertEqual(SPLIT_SIZES["stack_train"], 500_000)
        self.assertEqual(SPLIT_SIZES["stack_val"], 150_000)
        self.assertEqual(SPLIT_SIZES["final_test"], 500_000)
        self.assertEqual(RECONSTRUCTOR_ARCHITECTURES, ("gt", "pn", "pfn", "pcnn"))
        self.assertEqual(TEACHER_ARCHITECTURES, ("part", "pn", "pfn", "pcnn"))

    def test_builds_exactly_sixteen_reco_sources_and_four_hlt_sources(self):
        reco_sources = build_reco_source_specs()
        hlt_sources = build_hlt_source_specs()
        self.assertEqual(len(reco_sources), 16)
        self.assertEqual(len({source.name for source in reco_sources}), 16)
        self.assertEqual(len(hlt_sources), 4)
        self.assertEqual(len({source.name for source in hlt_sources}), 4)
        self.assertEqual(reco_sources[0].name, "gt_reco_to_part_teacher")
        self.assertEqual(reco_sources[-1].name, "pcnn_reco_to_pcnn_teacher")
        self.assertEqual([source.name for source in hlt_sources], ["hlt_part", "hlt_pn", "hlt_pfn", "hlt_pcnn"])

    def test_fusion_groups_have_expected_counts_and_members(self):
        groups = build_fusion_groups()
        validate_fusion_groups(groups)
        self.assertEqual(set(groups), {"all16", "cross12", "part_teacher4", "mixed4", "hlt4"})
        self.assertEqual(len(groups["all16"].model_names), 16)
        self.assertEqual(len(groups["cross12"].model_names), 12)
        self.assertEqual(len(groups["part_teacher4"].model_names), 4)
        self.assertEqual(len(groups["mixed4"].model_names), 4)
        self.assertEqual(len(groups["hlt4"].model_names), 4)

        same_family_names = {reco_model_name(reco, teacher) for reco, teacher in SAME_FAMILY_RECO_TEACHER_PAIRS}
        self.assertTrue(same_family_names.isdisjoint(groups["cross12"].model_names))
        self.assertEqual(
            set(groups["part_teacher4"].model_names),
            {reco_model_name(reco, "part") for reco in RECONSTRUCTOR_ARCHITECTURES},
        )
        self.assertEqual(
            groups["mixed4"].model_names,
            tuple(reco_model_name(reco, teacher) for reco, teacher in MIXED4_RECO_TEACHER_PAIRS),
        )
        self.assertEqual(groups["hlt4"].model_names, ("hlt_part", "hlt_pn", "hlt_pfn", "hlt_pcnn"))

    def test_optional_fusion_groups_are_available_but_not_default(self):
        groups = build_fusion_groups(include_optional=True)
        self.assertIn("all16_plus_hlt4", groups)
        self.assertEqual(len(groups["all16_plus_hlt4"].model_names), 20)
        self.assertIn("hlt_part", groups["part_teacher4_plus_hlt_part"].model_names)
        self.assertEqual(len(groups["cross12_plus_hlt4"].model_names), 16)


class CrossArchExperimentConfigTests(unittest.TestCase):
    def test_default_config_serializes_grid_groups_fusers_and_layout(self):
        cfg = default_crossarch_experiment_config()
        payload = cfg.to_dict()
        self.assertEqual(cfg.experiment_name, EXPERIMENT_NAME)
        self.assertEqual(len(cfg.reco_sources), 16)
        self.assertEqual(len(cfg.hlt_sources), 4)
        self.assertEqual(len(cfg.all_sources), 20)
        self.assertEqual(len(cfg.fusion_groups["all16"].model_names), 16)
        self.assertIn("uncertainty_logistic_logits_probs", cfg.fusers)
        self.assertIn("entropy_bin_gated_logistic", cfg.fusers)
        self.assertEqual(tuple(payload["split_sizes"].keys()), SPLIT_ORDER)
        self.assertEqual(len(payload["reco_sources"]), 16)
        self.assertEqual(payload["fusion_groups"]["mixed4"]["n_models"], 4)
        self.assertTrue(payload["layout"]["root"].endswith(EXPERIMENT_NAME))
        self.assertEqual(tuple(DEFAULT_FUSERS), cfg.fusers)

    def test_config_rejects_wrong_split_sizes_or_partial_architecture_sets(self):
        with self.assertRaises(ValueError):
            CrossArchExperimentConfig(split_sizes={"model_train": 1})
        bad_sizes = dict(SPLIT_SIZES)
        bad_sizes["final_test"] = 0
        with self.assertRaises(ValueError):
            CrossArchExperimentConfig(split_sizes=bad_sizes)
        with self.assertRaises(ValueError):
            CrossArchExperimentConfig(reconstructors=("gt", "pn"))
        with self.assertRaises(ValueError):
            CrossArchExperimentConfig(teachers=("part", "pn"))
        with self.assertRaises(ValueError):
            CrossArchExperimentConfig(fusers=())

    def test_layout_paths_are_fresh_crossarch_namespace(self):
        layout = CrossArchExperimentLayout(output_root="/tmp/checkpoints")
        self.assertEqual(layout.root, Path("/tmp/checkpoints") / EXPERIMENT_NAME)
        self.assertEqual(layout.split_manifest_path, layout.root / "split_manifest" / "split_manifest.json.gz")
        self.assertEqual(layout.hlt_cache_dir, layout.root / "hlt_cache")
        self.assertEqual(layout.offline_teacher_checkpoint("part"), layout.root / "offline_teachers" / "part" / "best_model_val.pt")
        self.assertEqual(layout.hlt_baseline_checkpoint("pcnn"), layout.root / "hlt_baselines" / "pcnn" / "best_model_val.pt")
        self.assertEqual(layout.reco_model_checkpoint("PFC", "PCNN"), layout.root / "reco_models" / "pfn" / "pcnn" / "best_model_val.pt")
        self.assertEqual(layout.prediction_source_dir("pcnn_reco_to_part_teacher"), layout.root / "predictions" / "pcnn_reco_to_part_teacher")
        self.assertEqual(layout.fusion_group_dir("all16"), layout.root / "fusion" / "all16")
        self.assertNotIn("teacher_logit_reco_pfn", str(layout.root))
        self.assertNotIn("jetclass_fresh_fusion", str(layout.root))


if __name__ == "__main__":
    unittest.main()
