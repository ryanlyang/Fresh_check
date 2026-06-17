import unittest
from pathlib import Path

from teacher_logit_reco.set_matching.experiment import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_SLOTS,
    DEFAULT_MAX_TOKENS_PER_VIEW,
    EXPERIMENT_NAME,
    EXPERIMENT_STEP,
    FIVE_VIEW_GROUP_NAME,
    SET_RECONSTRUCTOR_ARCHITECTURES,
    SET_RECONSTRUCTOR_IMPLEMENTATIONS,
    SPLIT_ORDER,
    SPLIT_SIZES,
    VIEW_KIND_ORIGINAL_HLT,
    VIEW_KIND_SET_RECONSTRUCTION,
    VIEW_NAMES,
    FiveViewGroupSpec,
    SetMatchingMultiViewConfig,
    SetMatchingMultiViewLayout,
    SetMatchingViewSpec,
    build_five_view_group,
    build_reconstructed_view_specs,
    build_view_specs,
    default_set_matching_multiview_config,
    normalize_set_reconstructor_architecture,
    normalize_split_name,
    normalize_view_name,
    set_reconstructor_model_name,
    view_name_for_reconstructor,
)


class SetMatchingMultiViewNamingTests(unittest.TestCase):
    def test_architecture_and_view_normalization(self):
        self.assertEqual(normalize_set_reconstructor_architecture("part"), "gt")
        self.assertEqual(normalize_set_reconstructor_architecture("ParticleNet"), "pn")
        self.assertEqual(normalize_set_reconstructor_architecture("PFC"), "pfn")
        self.assertEqual(normalize_set_reconstructor_architecture("P-CNN"), "pcnn")
        self.assertEqual(view_name_for_reconstructor("ParticleNet"), "pn_reco")
        self.assertEqual(set_reconstructor_model_name("pfn"), "setmatch_pfn_reco")
        self.assertEqual(normalize_view_name("fixed_hlt"), "hlt")
        self.assertEqual(normalize_view_name("ParticleNet"), "pn_reco")
        self.assertEqual(normalize_view_name("pcnn_reco"), "pcnn_reco")
        self.assertEqual(normalize_split_name("final_test"), "final_test")
        with self.assertRaises(ValueError):
            normalize_set_reconstructor_architecture("not_a_reco")
        with self.assertRaises(ValueError):
            normalize_view_name("not_a_view")
        with self.assertRaises(ValueError):
            normalize_split_name("not_a_split")

    def test_view_specs_validate_and_serialize(self):
        hlt = SetMatchingViewSpec(name="hlt", view_kind=VIEW_KIND_ORIGINAL_HLT)
        self.assertEqual(hlt.name, "hlt")
        self.assertEqual(hlt.source_type, "original_hlt")
        self.assertIsNone(hlt.reconstructor_architecture)
        self.assertIsNone(hlt.implementation)

        reco = SetMatchingViewSpec(
            name="pn_reco",
            view_kind=VIEW_KIND_SET_RECONSTRUCTION,
            reconstructor_architecture="ParticleNet",
        )
        self.assertEqual(reco.reconstructor_architecture, "pn")
        self.assertEqual(reco.implementation, SET_RECONSTRUCTOR_IMPLEMENTATIONS["pn"])
        self.assertEqual(reco.to_dict()["source_type"], "reconstructed")

        with self.assertRaises(ValueError):
            SetMatchingViewSpec(name="pn_reco", view_kind=VIEW_KIND_ORIGINAL_HLT)
        with self.assertRaises(ValueError):
            SetMatchingViewSpec(name="pn_reco", view_kind=VIEW_KIND_SET_RECONSTRUCTION)
        with self.assertRaises(ValueError):
            SetMatchingViewSpec(name="gt_reco", view_kind=VIEW_KIND_SET_RECONSTRUCTION, reconstructor_architecture="pn")


class SetMatchingMultiViewConfigTests(unittest.TestCase):
    def test_default_constants_and_config(self):
        self.assertEqual(tuple(SPLIT_SIZES.keys()), SPLIT_ORDER)
        self.assertEqual(SPLIT_SIZES["model_train"], 500_000)
        self.assertEqual(SPLIT_SIZES["model_val"], 150_000)
        self.assertEqual(SPLIT_SIZES["stack_train"], 500_000)
        self.assertEqual(SPLIT_SIZES["stack_val"], 150_000)
        self.assertEqual(SPLIT_SIZES["final_test"], 500_000)
        self.assertEqual(SET_RECONSTRUCTOR_ARCHITECTURES, ("gt", "pn", "pfn", "pcnn"))
        self.assertEqual(VIEW_NAMES, ("hlt", "gt_reco", "pn_reco", "pfn_reco", "pcnn_reco"))

        cfg = default_set_matching_multiview_config()
        self.assertEqual(cfg.experiment_name, EXPERIMENT_NAME)
        self.assertEqual(cfg.max_slots, DEFAULT_MAX_SLOTS)
        self.assertEqual(cfg.max_tokens_per_view, DEFAULT_MAX_TOKENS_PER_VIEW)
        self.assertEqual(cfg.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertEqual(cfg.reconstructors, SET_RECONSTRUCTOR_ARCHITECTURES)
        self.assertEqual(cfg.views, VIEW_NAMES)

        payload = cfg.to_dict()
        self.assertEqual(payload["experiment_step"], EXPERIMENT_STEP)
        self.assertEqual(tuple(payload["split_sizes"].keys()), SPLIT_ORDER)
        self.assertEqual(payload["five_view_group"]["n_views"], 5)
        self.assertEqual(payload["five_view_group"]["name"], FIVE_VIEW_GROUP_NAME)
        self.assertTrue(payload["layout"]["root"].endswith(EXPERIMENT_NAME))

    def test_builds_primary_view_specs_and_group(self):
        specs = build_view_specs()
        self.assertEqual([spec.name for spec in specs], list(VIEW_NAMES))
        self.assertEqual(specs[0].source_type, "original_hlt")
        self.assertEqual([spec.reconstructor_architecture for spec in specs[1:]], ["gt", "pn", "pfn", "pcnn"])

        reco_specs = build_reconstructed_view_specs(["part", "particle_net", "pfc", "p-cnn"])
        self.assertEqual([spec.name for spec in reco_specs], ["gt_reco", "pn_reco", "pfn_reco", "pcnn_reco"])

        group = build_five_view_group()
        self.assertEqual(group.name, FIVE_VIEW_GROUP_NAME)
        self.assertEqual(group.view_names, VIEW_NAMES)
        self.assertEqual(group.to_dict()["n_views"], 5)
        with self.assertRaises(ValueError):
            build_five_view_group(("hlt", "pn_reco"))
        with self.assertRaises(ValueError):
            FiveViewGroupSpec(name="bad", view_names=("hlt", "hlt"))

    def test_config_rejects_partial_or_invalid_values(self):
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(split_sizes={"model_train": 1})
        bad_sizes = dict(SPLIT_SIZES)
        bad_sizes["final_test"] = 0
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(split_sizes=bad_sizes)
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(reconstructors=("gt", "pn"))
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(views=("hlt", "pn_reco"))
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(max_slots=0)
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            SetMatchingMultiViewConfig(max_tokens_per_view=16, min_tokens_per_view=32)

    def test_layout_paths_are_isolated_from_existing_roots(self):
        layout = SetMatchingMultiViewLayout(output_root="/tmp/checkpoints")
        self.assertEqual(layout.root, Path("/tmp/checkpoints") / EXPERIMENT_NAME)
        self.assertEqual(layout.split_manifest_path, layout.root / "split_manifest" / "split_manifest.json.gz")
        self.assertEqual(layout.hlt_cache_dir, layout.root / "hlt_cache")
        self.assertEqual(layout.normalization_path, layout.root / "normalization" / "feature_normalization.json")
        self.assertEqual(layout.reconstructor_checkpoint("ParticleNet"), layout.root / "reconstructors" / "pn" / "best_model_val.pt")
        self.assertEqual(
            layout.reconstructed_view_cache_path("p-cnn", "final_test"),
            layout.root / "reconstructed_views" / "pcnn" / "final_test_reconstructed_view.npz",
        )
        self.assertEqual(
            layout.five_view_cache_path("stack_train"),
            layout.root / "five_view_cache" / "stack_train_five_view.npz",
        )
        self.assertEqual(layout.tagger_checkpoint(), layout.root / "taggers" / "five_view_tagger" / "best_model_val.pt")
        self.assertEqual(layout.ablation_dir("h2_pn_reco"), layout.root / "ablations" / "h2_pn_reco")
        self.assertNotIn("teacher_logit_reco_crossarch", str(layout.root))
        self.assertNotIn("jetclass_fresh_fusion", str(layout.root))


if __name__ == "__main__":
    unittest.main()
