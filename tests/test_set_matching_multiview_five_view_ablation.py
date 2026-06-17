import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.set_matching.experiment import (
    RECONSTRUCTED_VIEW_NAMES,
    SOURCE_TYPE_ORIGINAL_HLT,
    SOURCE_TYPE_RECONSTRUCTED,
    VIEW_NAMES,
)
from teacher_logit_reco.set_matching.five_view_ablation import (
    CANONICAL_ABLATION_NAMES,
    FiveViewAblationEvalConfig,
    FiveViewAblationSpec,
    canonical_five_view_ablation_specs,
    discover_five_view_ablation_specs,
    evaluate_five_view_ablation_suite,
    parse_ablation_checkpoint_spec,
)
from teacher_logit_reco.set_matching.five_view_data import FiveViewJetDataset
from teacher_logit_reco.set_matching.five_view_model import FiveViewParticleTransformerConfig, build_five_view_tagger, save_five_view_tagger_checkpoint


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def make_five_view_dataset(*, split="stack_val", n_jets=4, n_tokens=4):
    rng = np.random.default_rng(101 if split == "stack_val" else 202)
    features = rng.normal(size=(n_jets, 5, n_tokens, RAW_TOKEN_DIM)).astype(np.float32)
    masks = np.ones((n_jets, 5, n_tokens), dtype=bool)
    masks[:, :, -1] = False
    confidence = np.where(masks, 0.75, 0.0).astype(np.float32)
    confidence[:, 0] = np.where(masks[:, 0], 1.0, 0.0)
    labels = (np.arange(n_jets, dtype=np.int64) % 3).astype(np.int64)
    jet_ids = [
        JetIdentity(file=f"synthetic_{index % 2}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    return FiveViewJetDataset(
        view_features=features,
        view_masks=masks,
        view_confidence=confidence,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        view_names=VIEW_NAMES,
        source_types=[SOURCE_TYPE_ORIGINAL_HLT] + [SOURCE_TYPE_RECONSTRUCTED] * 4,
        view_ids=np.arange(5, dtype=np.int64),
        source_type_ids=np.asarray([0, 1, 1, 1, 1], dtype=np.int64),
        metadata={"split": split, "synthetic": True},
    )


class FiveViewAblationSpecTests(unittest.TestCase):
    def test_canonical_specs_cover_required_plan_comparisons(self):
        specs = canonical_five_view_ablation_specs("taggers")
        self.assertEqual(tuple(spec.name for spec in specs), CANONICAL_ABLATION_NAMES)
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["hlt_only"].drop_views, RECONSTRUCTED_VIEW_NAMES)
        self.assertEqual(by_name["hlt_plus_pn"].drop_views, ("gt_reco", "pfn_reco", "pcnn_reco"))
        self.assertFalse(by_name["five_view_plain"].drop_views)
        self.assertTrue(by_name["view_label_shuffle_control"].shuffle_view_labels)

    def test_parse_checkpoint_specs(self):
        spec = parse_ablation_checkpoint_spec("custom=/tmp/model.pt")
        self.assertEqual(spec.name, "custom")
        self.assertEqual(spec.checkpoint, "/tmp/model.pt")

        path_only = parse_ablation_checkpoint_spec("taggers/five_view_plain/best_model_val.pt")
        self.assertEqual(path_only.name, "five_view_plain")

    def test_discovery_skips_missing_canonical_and_keeps_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit = Path(tmpdir) / "explicit.pt"
            explicit.touch()
            config = FiveViewAblationEvalConfig(
                output_dir=str(Path(tmpdir) / "out"),
                experiment_dir=str(Path(tmpdir) / "experiment"),
                hlt_cache_dir="hlt_cache",
                checkpoint_specs=(FiveViewAblationSpec(name="explicit", checkpoint=str(explicit)),),
            )
            specs, skipped = discover_five_view_ablation_specs(config)
            self.assertIn("explicit", {spec.name for spec in specs})
            self.assertTrue(any(row.get("reason") == "missing_checkpoint" for row in skipped))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class FiveViewAblationEvaluationTests(unittest.TestCase):
    def tiny_model(self):
        config = FiveViewParticleTransformerConfig(
            particle_feature_dim=RAW_TOKEN_DIM,
            num_classes=10,
            embed_dim=32,
            stage1_layers=1,
            stage1_heads=4,
            stage2_layers=1,
            stage2_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        return build_five_view_tagger(config)

    def test_evaluate_explicit_ablation_checkpoints_writes_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full = tmp / "taggers" / "five_view_plain" / "best_model_val.pt"
            hlt = tmp / "taggers" / "hlt_only" / "best_model_val.pt"
            save_five_view_tagger_checkpoint(full, self.tiny_model())
            save_five_view_tagger_checkpoint(hlt, self.tiny_model())
            output_dir = tmp / "ablations"
            config = FiveViewAblationEvalConfig(
                output_dir=str(output_dir),
                experiment_dir=str(tmp / "experiment"),
                hlt_cache_dir="unused_hlt_cache",
                checkpoint_specs=(
                    FiveViewAblationSpec(name="five_view_plain", checkpoint=str(full), use_checkpoint_dataset_config=False),
                    FiveViewAblationSpec(
                        name="hlt_only",
                        checkpoint=str(hlt),
                        drop_views=RECONSTRUCTED_VIEW_NAMES,
                        use_checkpoint_dataset_config=False,
                    ),
                ),
                include_canonical=False,
                batch_size=2,
                device="cpu",
                max_tokens_per_view=4,
                min_tokens_per_view=0,
            )
            report = evaluate_five_view_ablation_suite(
                config,
                datasets_by_split={"stack_val": make_five_view_dataset()},
            )

            self.assertEqual(set(report["evaluated_ablations"]), {"five_view_plain", "hlt_only"})
            self.assertEqual(len(report["summary_rows"]), 2)
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "per_class_metrics.csv").exists())
            self.assertFalse(report["final_test_evaluated"])


if __name__ == "__main__":
    unittest.main()
