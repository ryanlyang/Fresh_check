import unittest
from pathlib import Path

from teacher_logit_reco.set_matching.detr_slots.five_view import (
    DETR_SLOT_FIVE_VIEW_TAGGER_STEP,
    DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS,
    build_detr_slot_five_view_tagger_config,
    detr_slot_five_view_expected_cache_paths,
    detr_slot_five_view_name_mapping,
    detr_slot_five_view_output_dir,
    detr_slot_five_view_variant_drop_views,
    detr_slot_semantic_view_names_for_variant,
)
from teacher_logit_reco.set_matching.five_view_train import FiveViewTaggerTrainConfig


def tiny_five_view_config():
    return FiveViewTaggerTrainConfig(
        output_dir="unused/taggers/five_view_plain",
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


class DetrFreeSlotStep14FiveViewTaggerTests(unittest.TestCase):
    def test_variant_names_and_view_mapping_are_stable(self):
        self.assertEqual(DETR_SLOT_FIVE_VIEW_TAGGER_STEP, "detr_free_slot_step14_five_view_tagger_integration")
        self.assertEqual(
            DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS,
            (
                "hlt_only",
                "hlt_plus_gt",
                "hlt_plus_pn",
                "hlt_plus_pfn",
                "hlt_plus_pcnn",
                "five_view_plain",
                "five_view_geometry",
                "five_view_no_confidence",
                "view_label_shuffle_control",
            ),
        )
        self.assertEqual(
            detr_slot_five_view_name_mapping(),
            {
                "hlt": "hlt",
                "detr_gt": "gt_reco",
                "detr_pn": "pn_reco",
                "detr_pfn": "pfn_reco",
                "detr_pcnn": "pcnn_reco",
            },
        )

    def test_variant_drop_views_and_semantic_active_views(self):
        self.assertEqual(detr_slot_five_view_variant_drop_views("hlt_only"), ("gt_reco", "pn_reco", "pfn_reco", "pcnn_reco"))
        self.assertEqual(detr_slot_semantic_view_names_for_variant("hlt_only"), ("hlt",))

        self.assertEqual(detr_slot_five_view_variant_drop_views("hlt_plus_pcnn"), ("gt_reco", "pn_reco", "pfn_reco"))
        self.assertEqual(detr_slot_semantic_view_names_for_variant("hlt_plus_pcnn"), ("hlt", "detr_pcnn"))

        self.assertEqual(detr_slot_five_view_variant_drop_views("five_view_plain"), ())
        self.assertEqual(
            detr_slot_semantic_view_names_for_variant("five_view_plain"),
            ("hlt", "detr_gt", "detr_pn", "detr_pfn", "detr_pcnn"),
        )

    def test_build_config_applies_variant_semantics(self):
        base = tiny_five_view_config()

        no_confidence = build_detr_slot_five_view_tagger_config(
            base,
            variant="five_view_no_confidence",
            output_root="outputs",
        )
        self.assertEqual(Path(no_confidence.output_dir).as_posix(), "outputs/five_view_no_confidence")
        self.assertFalse(no_confidence.use_confidence)
        self.assertEqual(no_confidence.selection_mode, "all_slots")
        self.assertEqual(no_confidence.drop_views, ())

        geometry = build_detr_slot_five_view_tagger_config(
            base,
            variant="five_view_geometry",
            output_root="outputs",
        )
        self.assertTrue(geometry.use_geometry_attention)
        self.assertTrue(geometry.use_confidence)
        self.assertEqual(geometry.selection_mode, base.selection_mode)

        shuffled = build_detr_slot_five_view_tagger_config(
            base,
            variant="view_label_shuffle_control",
            output_root="outputs",
        )
        self.assertTrue(shuffled.shuffle_view_labels)
        self.assertEqual(shuffled.drop_views, ())

        hlt_plus_pn = build_detr_slot_five_view_tagger_config(
            base,
            variant="hlt_plus_pn",
            output_dir="custom/pn",
        )
        self.assertEqual(hlt_plus_pn.output_dir, "custom/pn")
        self.assertEqual(hlt_plus_pn.drop_views, ("gt_reco", "pfn_reco", "pcnn_reco"))

    def test_expected_cache_paths_match_step12_file_contract(self):
        paths = detr_slot_five_view_expected_cache_paths("cache_root", splits=("stack_val",))

        self.assertEqual(Path(paths["gt"]["stack_val"]).as_posix(), "cache_root/gt/stack_val_reconstructed_view.npz")
        self.assertEqual(Path(paths["pn"]["stack_val"]).as_posix(), "cache_root/pn/stack_val_reconstructed_view.npz")
        self.assertEqual(Path(paths["pfn"]["stack_val"]).as_posix(), "cache_root/pfn/stack_val_reconstructed_view.npz")
        self.assertEqual(Path(paths["pcnn"]["stack_val"]).as_posix(), "cache_root/pcnn/stack_val_reconstructed_view.npz")
        self.assertEqual(detr_slot_five_view_output_dir("outputs", "five_view_plain").as_posix(), "outputs/five_view_plain")


if __name__ == "__main__":
    unittest.main()
