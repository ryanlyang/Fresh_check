import unittest

from teacher_logit_reco.crossarch_reco_domain_taggers import (
    CrossArchRecoDomainTaggerPredictionConfig,
    CrossArchRecoDomainTaggerTrainConfig,
    reco_domain_tagger_model_name,
    split_size_for_reco_domain_prediction,
)


class CrossArchRecoDomainTaggerTests(unittest.TestCase):
    def test_model_name_is_distinct_from_frozen_teacher_reco_source(self):
        self.assertEqual(
            reco_domain_tagger_model_name("pn", "pfn"),
            "pn_reco_to_pfn_adapted_tagger",
        )

    def test_train_config_normalizes_aliases_and_guards_split_scope(self):
        config = CrossArchRecoDomainTaggerTrainConfig(
            reco_architecture="particle_net",
            teacher_architecture="particle_flow",
            reconstructor_checkpoint="reco.pt",
            output_dir="out",
            cache_dir="cache",
        )
        self.assertEqual(config.reco_architecture, "pn")
        self.assertEqual(config.teacher_architecture, "pfn")
        self.assertEqual(config.model_name, "pn_reco_to_pfn_adapted_tagger")
        with self.assertRaises(ValueError):
            CrossArchRecoDomainTaggerTrainConfig(
                reco_architecture="pn",
                teacher_architecture="pfn",
                reconstructor_checkpoint="reco.pt",
                output_dir="out",
                cache_dir="cache",
                train_split="stack_train",
            )

    def test_prediction_config_requires_final_test_confirmation(self):
        with self.assertRaises(ValueError):
            CrossArchRecoDomainTaggerPredictionConfig(
                reco_architecture="gt",
                teacher_architecture="part",
                reconstructor_checkpoint="reco.pt",
                tagger_checkpoint="tagger.pt",
                cache_dir="cache",
                prediction_dir="pred",
                output_dir="out",
            )

    def test_prediction_split_sizes_follow_crossarch_defaults(self):
        config = CrossArchRecoDomainTaggerPredictionConfig(
            reco_architecture="gt",
            teacher_architecture="part",
            reconstructor_checkpoint="reco.pt",
            tagger_checkpoint="tagger.pt",
            cache_dir="cache",
            prediction_dir="pred",
            output_dir="out",
            confirm_final_test=True,
        )
        self.assertEqual(split_size_for_reco_domain_prediction(config, "stack_train"), 500_000)
        self.assertEqual(split_size_for_reco_domain_prediction(config, "stack_val"), 150_000)
        self.assertEqual(split_size_for_reco_domain_prediction(config, "final_test"), 500_000)
        config.max_jets_per_split = 123
        self.assertEqual(split_size_for_reco_domain_prediction(config, "final_test"), 123)


if __name__ == "__main__":
    unittest.main()
