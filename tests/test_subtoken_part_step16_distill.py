import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_DISTILL_STEP,
    SUBTOKEN_PART_VERSION_B,
    SubtokenDistillJetDataset,
    SubtokenDistillTrainConfig,
    align_teacher_logits_to_student,
    compute_subtoken_distillation_kl_loss,
    load_subtoken_tagger_checkpoint,
    train_subtoken_distilled_tagger,
)


class FakeOfflineTeacher:
    def __init__(self):
        self.calls = 0
        self.device = require_torch().device("cpu")
        self.metadata = {
            "architecture": "fake_part",
            "checkpoint_path": "fake_offline_teacher.pt",
            "frozen": True,
        }

    def forward_view_no_grad(self, tokens, mask, *, weights=None):
        torch = require_torch()
        self.calls += 1
        score = torch.where(mask, tokens[:, :, 0], torch.zeros_like(tokens[:, :, 0])).sum(dim=1) / 100.0
        return torch.stack([-score, score], dim=1)


class SubtokenPartStep16DistillationTests(unittest.TestCase):
    def make_view(self, *, split="model_train", n_jets=8, view="fixed_hlt", offset=0.0):
        torch = require_torch()
        tokens = torch.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((n_jets, 6), dtype=torch.bool)
        labels = torch.tensor([0, 3] * ((n_jets + 1) // 2), dtype=torch.long)[:n_jets]
        for index in range(n_jets):
            label = int(labels[index].item())
            base = 30.0 + 8.0 * label + float(index) + float(offset)
            tokens[index, :, 0] = base - torch.arange(6, dtype=torch.float32)
            tokens[index, :, 1] = torch.linspace(-0.5, 0.5, 6) + 0.1 * label
            tokens[index, :, 2] = torch.linspace(-2.0, 2.0, 6)
            tokens[index, :, 3] = tokens[index, :, 0] + 20.0
            tokens[index, :, 4] = 1.0 if label == 3 else -1.0
            tokens[index, :, 5 + (label % 5)] = 1.0
            tokens[index, :, 10] = 0.1 * (label + 1)
            tokens[index, :, 11] = 0.2
            tokens[index, :, 12] = -0.1 * (label + 1)
            tokens[index, :, 13] = 0.3
        mask[1, 4:] = False
        mask[3, 5:] = False
        tokens = tokens * mask[:, :, None]
        jet_ids = [
            JetIdentity(file=f"{split}_{index}.root", entry=index, label=int(labels[index].item()))
            for index in range(n_jets)
        ]
        metadata = {
            "view": view,
            "hlt_content_hash": f"{split}_hlt_hash",
            "jet_identity_hash": f"{split}_ids",
            "source_manifest_hash": "manifest_hash",
        }
        if view == "offline":
            metadata = {"view": "offline", "source_manifest_hash": "manifest_hash"}
        return JetView(
            tokens=tokens.numpy(),
            mask=mask.numpy(),
            labels=labels.numpy(),
            jet_ids=jet_ids,
            split=split,
            metadata=metadata,
        )

    def make_paired_dataset(self, split="model_train", n_jets=8):
        return SubtokenDistillJetDataset(
            self.make_view(split=split, n_jets=n_jets, view="fixed_hlt", offset=0.0),
            self.make_view(split=split, n_jets=n_jets, view="offline", offset=1.0),
            label_filter=(0, 3),
            label_names=("QCD", "Hgg"),
        )

    def make_hlt_dataset(self, split="model_val", n_jets=8):
        from teacher_logit_reco.subtoken_part import SubtokenHLTJetDataset

        return SubtokenHLTJetDataset(
            self.make_view(split=split, n_jets=n_jets, view="fixed_hlt", offset=0.0),
            label_filter=(0, 3),
            label_names=("QCD", "Hgg"),
        )

    def make_config(self, output_dir):
        return SubtokenDistillTrainConfig(
            output_dir=str(output_dir),
            hlt_cache_dir="unused_hlt_cache",
            manifest_path="unused_manifest.json.gz",
            offline_teacher_checkpoint="unused_teacher.pt",
            confirm_split_settings=True,
            confirm_final_test=False,
            seed=23,
            batch_size=4,
            eval_batch_size=4,
            epochs=1,
            lr=1.0e-3,
            weight_decay=0.0,
            num_workers=0,
            device="cpu",
            amp=False,
            max_train_batches=1,
            max_val_batches=1,
            max_stack_val_batches=1,
            selection_metric="accuracy",
            num_classes=2,
            label_names=("QCD", "Hgg"),
            label_filter=(0, 3),
            embed_dim=8,
            local_layers=1,
            local_heads=2,
            context_layers=1,
            context_heads=2,
            global_layers=1,
            global_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            distillation_temperature=2.0,
            distillation_weight=0.25,
            classification_weight=1.0,
        )

    def test_distillation_loss_is_finite(self):
        torch = require_torch()
        student = torch.tensor([[1.0, -0.5], [-0.25, 0.75]], dtype=torch.float32)
        teacher = torch.tensor([[0.5, -0.25], [-1.0, 2.0]], dtype=torch.float32)

        loss = compute_subtoken_distillation_kl_loss(student, teacher, temperature=2.0)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_full_teacher_logits_are_sliced_to_student_labels(self):
        torch = require_torch()
        student = torch.zeros((3, 2), dtype=torch.float32)
        teacher = torch.arange(30, dtype=torch.float32).reshape(3, 10)

        aligned = align_teacher_logits_to_student(student, teacher, label_filter=(0, 3))

        self.assertEqual(tuple(aligned.shape), (3, 2))
        self.assertTrue(bool(torch.equal(aligned[:, 0], teacher[:, 0])))
        self.assertTrue(bool(torch.equal(aligned[:, 1], teacher[:, 3])))

    def test_paired_dataset_carries_hlt_and_offline_views(self):
        dataset = self.make_paired_dataset(n_jets=6)
        sample = dataset[0]

        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.metadata["teacher_source_view"], "offline")
        self.assertEqual(tuple(sample["tokens"].shape), (6, RAW_TOKEN_DIM))
        self.assertEqual(tuple(sample["offline_tokens"].shape), (6, RAW_TOKEN_DIM))
        self.assertEqual(set(dataset.labels.tolist()), {0, 1})

    def test_training_uses_teacher_only_for_train_and_saves_hlt_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "distill"
            config = self.make_config(output_dir)
            teacher = FakeOfflineTeacher()

            report = train_subtoken_distilled_tagger(
                config,
                teacher=teacher,
                train_dataset=self.make_paired_dataset(split="model_train", n_jets=8),
                val_dataset=self.make_hlt_dataset(split="model_val", n_jets=8),
                stack_val_dataset=self.make_hlt_dataset(split="stack_val", n_jets=8),
            )

            self.assertEqual(report["experiment_step"], SUBTOKEN_PART_DISTILL_STEP)
            self.assertEqual(report["model_config"]["version"], SUBTOKEN_PART_VERSION_B)
            self.assertTrue(report["offline_teacher_used_for_training_only"])
            self.assertFalse(report["inference_requires_offline"])
            self.assertEqual(teacher.calls, 1)
            model, payload = load_subtoken_tagger_checkpoint(output_dir / "best_model_val.pt")
            self.assertEqual(model.config.version, SUBTOKEN_PART_VERSION_B)
            self.assertNotIn("offline_teacher_state_dict", payload)
            self.assertNotIn("offline_teacher_model_state_dict", payload)
            self.assertTrue(payload["inference_consumes_hlt_only"])


if __name__ == "__main__":
    unittest.main()
