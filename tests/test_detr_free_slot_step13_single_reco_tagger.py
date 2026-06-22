import unittest

from teacher_logit_reco.set_matching.detr_slots.sanity_taggers import (
    DETR_SLOT_SINGLE_RECO_TAGGER_STEP,
    DETR_SLOT_SINGLE_RECO_VARIANTS,
    build_detr_slot_single_reco_tagger_config,
    detr_slot_architecture_from_single_reco_variant,
    detr_slot_single_reco_drop_views,
    detr_slot_single_reco_expected_cache_paths,
    detr_slot_single_reco_output_dir,
    detr_slot_single_reco_variant_name,
)
from teacher_logit_reco.set_matching.five_view_train import FiveViewTaggerTrainConfig


def tiny_five_view_config():
    return FiveViewTaggerTrainConfig(
        output_dir="unused/taggers/hlt_plus_gt",
        experiment_dir="unused",
        hlt_cache_dir="unused/hlt_cache",
        reconstructed_view_dir="unused/detr_slot_reconstructed_views",
        confirm_split_settings=True,
        confirm_final_test=True,
        batch_size=2,
        epochs=1,
        num_workers=0,
        max_tokens_per_view=8,
        min_tokens_per_view=0,
        embed_dim=16,
        stage1_layers=1,
        stage1_heads=4,
        stage2_layers=1,
        stage2_heads=4,
        label_names=("QCD", "Tbqq"),
        num_classes=2,
        label_filter=(0, 9),
    )


class DetrFreeSlotStep13SingleRecoTaggerTests(unittest.TestCase):
    def test_variant_names_and_reverse_mapping_are_stable(self):
        self.assertEqual(DETR_SLOT_SINGLE_RECO_TAGGER_STEP, "detr_free_slot_step13_single_reco_sanity_taggers")
        self.assertEqual(DETR_SLOT_SINGLE_RECO_VARIANTS, ("hlt_plus_gt", "hlt_plus_pn", "hlt_plus_pfn", "hlt_plus_pcnn"))
        self.assertEqual(detr_slot_single_reco_variant_name("part"), "hlt_plus_gt")
        self.assertEqual(detr_slot_architecture_from_single_reco_variant("hlt_plus_pcnn"), "pcnn")

    def test_single_reco_drop_views_keep_only_requested_architecture(self):
        self.assertEqual(detr_slot_single_reco_drop_views("gt"), ("pn_reco", "pfn_reco", "pcnn_reco"))
        self.assertEqual(detr_slot_single_reco_drop_views("pn"), ("gt_reco", "pfn_reco", "pcnn_reco"))
        self.assertEqual(detr_slot_single_reco_drop_views("pfn"), ("gt_reco", "pn_reco", "pcnn_reco"))
        self.assertEqual(detr_slot_single_reco_drop_views("pcnn"), ("gt_reco", "pn_reco", "pfn_reco"))

    def test_build_config_applies_output_and_drop_views(self):
        config = build_detr_slot_single_reco_tagger_config(
            tiny_five_view_config(),
            architecture="pn",
            output_root="outputs",
        )

        self.assertEqual(config.output_dir, "outputs/hlt_plus_pn")
        self.assertEqual(config.drop_views, ("gt_reco", "pfn_reco", "pcnn_reco"))
        self.assertEqual(config.reconstructed_view_dir, "unused/detr_slot_reconstructed_views")

    def test_expected_cache_paths_match_step12_file_contract(self):
        paths = detr_slot_single_reco_expected_cache_paths("cache_root", splits=("stack_val",))

        self.assertEqual(paths["gt"]["stack_val"], "cache_root/gt/stack_val_reconstructed_view.npz")
        self.assertEqual(paths["pcnn"]["stack_val"], "cache_root/pcnn/stack_val_reconstructed_view.npz")
        self.assertEqual(detr_slot_single_reco_output_dir("outputs", "pfn").as_posix(), "outputs/hlt_plus_pfn")


if __name__ == "__main__":
    unittest.main()
