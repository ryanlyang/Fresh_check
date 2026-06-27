import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT,
    LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
    LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
    HLTPartBaselineRawTokenConfig,
    build_hlt_part_baseline_raw_token_classifier,
    build_local_graph_comparison_classifier,
)
from teacher_logit_reco.local_graph_part.train import (
    LOCAL_GRAPH_PART_TRAIN_STEP,
    LocalGraphTaggerTrainConfig,
    train_local_graph_tagger,
    warm_start_local_graph_part_model,
)
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset

_TORCH = require_torch()


class TinyPartBackbone(_TORCH.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        torch = require_torch()
        self.config = {"architecture": "tiny_test_part_backbone", "num_classes": int(num_classes)}
        self.proj = torch.nn.Linear(17 + 4 + 1, int(num_classes))

    def __call__(self, points, features, lorentz_vectors, mask):
        torch = require_torch()
        del points
        particle_mask = mask.float()
        packed = torch.cat([features, lorentz_vectors, particle_mask], dim=1).transpose(1, 2)
        denom = torch.clamp(particle_mask.squeeze(1).sum(dim=1, keepdim=True), min=1.0)
        pooled = (packed * particle_mask.transpose(1, 2)).sum(dim=1) / denom
        return self.proj(pooled)

    def no_weight_decay(self):
        return set()


class TinyModPartBackbone(_TORCH.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        torch = require_torch()
        self.config = {"architecture": "tiny_mod_test_part_backbone", "num_classes": int(num_classes)}
        self.mod = torch.nn.Linear(17 + 4 + 1, int(num_classes))

    def __call__(self, points, features, lorentz_vectors, mask):
        torch = require_torch()
        del points
        particle_mask = mask.float()
        packed = torch.cat([features, lorentz_vectors, particle_mask], dim=1).transpose(1, 2)
        denom = torch.clamp(particle_mask.squeeze(1).sum(dim=1, keepdim=True), min=1.0)
        pooled = (packed * particle_mask.transpose(1, 2)).sum(dim=1) / denom
        return self.mod(pooled)

    def no_weight_decay(self):
        return set()


def make_toy_view(split: str, *, n_jets: int = 8) -> JetView:
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    labels = np.asarray([index % 2 for index in range(n_jets)], dtype=np.int64)
    for jet in range(n_jets):
        valid = 4 + (jet % 2)
        mask[jet, :valid] = True
        for particle in range(valid):
            tokens[jet, particle, 0] = 10.0 + 0.5 * jet + 1.5 * particle + 2.0 * labels[jet]
            tokens[jet, particle, 1] = -0.4 + 0.12 * particle
            tokens[jet, particle, 2] = -0.2 + 0.17 * particle
            tokens[jet, particle, 3] = tokens[jet, particle, 0] * np.cosh(tokens[jet, particle, 1]) + 0.2
            tokens[jet, particle, 4] = -1.0 + (particle % 3)
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 10] = 0.03 * particle
            tokens[jet, particle, 11] = 0.04
            tokens[jet, particle, 12] = -0.07 * particle
            tokens[jet, particle, 13] = 0.08
    jet_ids = [JetIdentity(file="toy.root", entry=index, label=int(labels[index])) for index in range(n_jets)]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"toy-{split}", "hlt_params": {"strength": 0.6}},
    )


class LocalGraphPartStep6TrainTests(unittest.TestCase):
    def dataset(self, split: str) -> SubtokenHLTJetDataset:
        return SubtokenHLTJetDataset(
            make_toy_view(split),
            label_filter=(0, 1),
            label_names=("QCD", "Hgg"),
        )

    def base_config(self, output_dir: str, *, variant: str) -> LocalGraphTaggerTrainConfig:
        return LocalGraphTaggerTrainConfig(
            output_dir=output_dir,
            hlt_cache_dir="unused",
            variant=variant,
            confirm_split_settings=True,
            confirm_final_test=True,
            seed=123,
            batch_size=4,
            eval_batch_size=4,
            epochs=1,
            num_workers=0,
            device="cpu",
            amp=False,
            early_stop_patience=-1,
            max_train_batches=1,
            max_val_batches=1,
            max_stack_val_batches=1,
            max_final_test_batches=1,
            model_size="tiny",
            k=3,
            local_embed_dim=16,
            local_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
        )

    def test_baseline_raw_token_wrapper_uses_baseline_contract(self):
        torch = require_torch()
        model = build_hlt_part_baseline_raw_token_classifier(
            HLTPartBaselineRawTokenConfig(num_classes=2, model_size="tiny"),
            part_model=TinyPartBackbone(num_classes=2),
        )
        view = make_toy_view("model_val", n_jets=2)
        tokens = torch.from_numpy(view.tokens).float()
        mask = torch.from_numpy(view.mask).bool()
        output = model(tokens, mask, return_outputs=True)

        self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT)
        self.assertEqual(output.summary()["variant"], LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE)
        self.assertEqual(tuple(output.logits.shape), (2, 2))

    def test_comparison_builder_distinguishes_baseline_and_adapter(self):
        baseline = build_local_graph_comparison_classifier(
            LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
            model_size="tiny",
            part_model=TinyPartBackbone(num_classes=2),
        )
        adapter = build_local_graph_comparison_classifier(
            LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
            model_size="tiny",
            local_embed_dim=16,
            part_model=TinyPartBackbone(num_classes=2),
        )

        self.assertEqual(baseline.output_contract, LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT)
        self.assertNotEqual(adapter.output_contract, baseline.output_contract)

    def test_train_local_graph_tagger_trains_baseline_with_shared_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.base_config(tmp, variant=LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE)
            model = build_hlt_part_baseline_raw_token_classifier(
                HLTPartBaselineRawTokenConfig(num_classes=2, model_size="tiny"),
                part_model=TinyPartBackbone(num_classes=2),
            )
            report = train_local_graph_tagger(
                config,
                model=model,
                train_dataset=self.dataset("model_train"),
                val_dataset=self.dataset("model_val"),
                stack_val_dataset=self.dataset("stack_val"),
                final_test_dataset=self.dataset("final_test"),
            )

            self.assertEqual(report["experiment_step"], LOCAL_GRAPH_PART_TRAIN_STEP)
            self.assertEqual(report["variant"], LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE)
            self.assertEqual(report["selection_metric"], "fpr_at_signal_eff_0p50")
            self.assertTrue(report["final_test_evaluated"])
            self.assertIn("binary_metrics", report["final_test_metrics"])
            self.assertTrue((Path(tmp) / "run_report.json").exists())

    def test_train_local_graph_tagger_trains_adapter_with_same_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.base_config(tmp, variant=LOCAL_GRAPH_MODEL_VARIANT_EDGECONV)
            model = build_local_graph_comparison_classifier(
                LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
                num_classes=2,
                model_size="tiny",
                k=3,
                local_embed_dim=16,
                local_heads=2,
                dropout=0.0,
                attention_dropout=0.0,
                part_model=TinyPartBackbone(num_classes=2),
            )
            report = train_local_graph_tagger(
                config,
                model=model,
                train_dataset=self.dataset("model_train"),
                val_dataset=self.dataset("model_val"),
                stack_val_dataset=self.dataset("stack_val"),
                final_test_dataset=self.dataset("final_test"),
            )

            self.assertEqual(report["variant"], LOCAL_GRAPH_MODEL_VARIANT_EDGECONV)
            self.assertEqual(report["selection_metric_direction"], "minimize")
            self.assertIn("diagnostics", report["stack_val_metrics"])

    def test_warm_start_loads_baseline_part_model_weights_into_adapter(self):
        torch = require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "baseline.pt"
            baseline = build_hlt_part_baseline_raw_token_classifier(
                HLTPartBaselineRawTokenConfig(num_classes=2, model_size="tiny"),
                part_model=TinyPartBackbone(num_classes=2),
            )
            adapter = build_local_graph_comparison_classifier(
                LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
                num_classes=2,
                model_size="tiny",
                k=3,
                local_embed_dim=16,
                part_model=TinyPartBackbone(num_classes=2),
            )
            with torch.no_grad():
                baseline.part_model.proj.weight.fill_(0.25)
                baseline.part_model.proj.bias.fill_(-0.125)
                adapter.part_model.proj.weight.fill_(1.5)
                adapter.part_model.proj.bias.fill_(0.75)
            torch.save({"model_state_dict": baseline.state_dict()}, checkpoint)

            report = warm_start_local_graph_part_model(adapter, checkpoint, require=True)

            self.assertGreaterEqual(report["loaded_key_count"], 2)
            self.assertTrue(bool(torch.allclose(adapter.part_model.proj.weight, baseline.part_model.proj.weight)))
            self.assertTrue(bool(torch.allclose(adapter.part_model.proj.bias, baseline.part_model.proj.bias)))

    def test_warm_start_maps_older_direct_hlt_baseline_mod_keys(self):
        torch = require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "old_baseline.pt"
            adapter = build_local_graph_comparison_classifier(
                LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
                num_classes=2,
                model_size="tiny",
                k=3,
                local_embed_dim=16,
                part_model=TinyModPartBackbone(num_classes=2),
            )
            source_weight = torch.full_like(adapter.part_model.mod.weight, 0.33)
            source_bias = torch.full_like(adapter.part_model.mod.bias, -0.44)
            torch.save({"model_state_dict": {"mod.weight": source_weight, "mod.bias": source_bias}}, checkpoint)

            report = warm_start_local_graph_part_model(adapter, checkpoint, require=True)

            self.assertEqual(report["loaded_key_count"], 2)
            self.assertTrue(bool(torch.allclose(adapter.part_model.mod.weight, source_weight)))
            self.assertTrue(bool(torch.allclose(adapter.part_model.mod.bias, source_bias)))

    def test_train_local_graph_tagger_warm_starts_and_freezes_part_backbone(self):
        torch = require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "baseline.pt"
            baseline = build_hlt_part_baseline_raw_token_classifier(
                HLTPartBaselineRawTokenConfig(num_classes=2, model_size="tiny"),
                part_model=TinyPartBackbone(num_classes=2),
            )
            with torch.no_grad():
                baseline.part_model.proj.weight.fill_(0.11)
                baseline.part_model.proj.bias.fill_(0.07)
            torch.save({"model_state_dict": baseline.state_dict()}, checkpoint)

            config = self.base_config(str(tmp_path / "adapter"), variant=LOCAL_GRAPH_MODEL_VARIANT_EDGECONV)
            config.epochs = 2
            config.warm_start_checkpoint = str(checkpoint)
            config.require_warm_start = True
            config.freeze_part_epochs = 1
            model = build_local_graph_comparison_classifier(
                LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
                num_classes=2,
                model_size="tiny",
                k=3,
                local_embed_dim=16,
                local_heads=2,
                dropout=0.0,
                attention_dropout=0.0,
                part_model=TinyPartBackbone(num_classes=2),
            )
            report = train_local_graph_tagger(
                config,
                model=model,
                train_dataset=self.dataset("model_train"),
                val_dataset=self.dataset("model_val"),
                stack_val_dataset=self.dataset("stack_val"),
                final_test_dataset=self.dataset("final_test"),
            )

            self.assertEqual(report["warm_start"]["loaded_key_count"], 2)
            self.assertEqual(report["freeze_schedule"]["adapter_only_epochs"], 1)
            self.assertEqual(report["final_epoch"]["phase"], "full_finetune")
            self.assertGreaterEqual(len(report["freeze_events"]), 2)
            self.assertTrue((tmp_path / "adapter" / "diagnostics" / "warm_start_report.json").exists())


if __name__ == "__main__":
    unittest.main()
