import importlib.util
import unittest
from unittest.mock import patch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots.outputs import detr_slot_output_from_tensors
    from teacher_logit_reco.set_matching.detr_slots.training import (
        DETR_SLOT_TRAIN_STEP,
        DetrSlotReconstructorTrainConfig,
        build_detr_slot_reconstructor,
        run_detr_slot_reco_epoch,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def tiny_config(architecture="gt"):
    return DetrSlotReconstructorTrainConfig(
        output_dir="unused",
        manifest_path="unused_manifest.json.gz",
        hlt_cache_dir="unused_hlt_cache",
        architecture=architecture,
        confirm_split_settings=True,
        batch_size=2,
        epochs=1,
        num_slots=4,
        export_max_tokens=4,
        memory_dim=16,
        context_dim=16,
        embed_dim=16,
        decoder_layers=1,
        decoder_heads=4,
        head_hidden_dim=32,
        gt_layers=1,
        gt_heads=4,
        edgeconv_dims=(16,),
        k=2,
        phi_dims=(16,),
        context_mlp_dims=(16,),
        hidden_channels=16,
        kernel_sizes=(3,),
        dilations=(1,),
        dropout=0.0,
        allow_bruteforce_fallback=True,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep11TrainingTests(unittest.TestCase):
    def test_config_requires_explicit_split_confirmation(self):
        with self.assertRaises(ValueError):
            DetrSlotReconstructorTrainConfig(
                output_dir="unused",
                manifest_path="unused",
                hlt_cache_dir="unused",
            )

    def test_tiny_models_forward_for_all_architectures(self):
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        tokens[:, :, 0] = 5.0
        tokens[:, :, 3] = 6.0
        mask = torch.ones((2, 5), dtype=torch.bool)
        for architecture in ("gt", "pn", "pfn", "pcnn"):
            with self.subTest(architecture=architecture):
                model = build_detr_slot_reconstructor(tiny_config(architecture))
                output = model(tokens, mask)

                self.assertEqual(tuple(output.loss_features.shape), (2, 4, RAW_TOKEN_DIM))
                self.assertEqual(tuple(output.existence_logits.shape), (2, 4))
                self.assertEqual(tuple(output.aux_outputs.shape), (2, 4, RAW_TOKEN_DIM - 4))
                self.assertEqual(output.aux["train_step"], DETR_SLOT_TRAIN_STEP)
                self.assertEqual(output.aux["encoder_architecture"], architecture)

    def test_epoch_forwards_aux_logits_to_hungarian_loss(self):
        class FakeOutput:
            def __init__(self):
                self.diagnostics = {"aux_metric": 1.0}

            def to_loss_kwargs(self, **kwargs):
                if not kwargs.get("include_aux_logits", False):
                    raise AssertionError("DETR training must call include_aux_logits=True")
                return {"sentinel": torch.tensor(1.0)}

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.forward_grad_enabled = None

            def forward(self, *_args, **_kwargs):
                self.forward_grad_enabled = torch.is_grad_enabled()
                return FakeOutput()

        class FakeLoss:
            def __init__(self):
                self.total_loss = torch.tensor(0.0)

            def detached_float_dict(self, prefix=""):
                return {f"{prefix}total": 0.0}

        batch = {
            "hlt_tokens": torch.zeros((2, 3, RAW_TOKEN_DIM), dtype=torch.float32),
            "hlt_mask": torch.ones((2, 3), dtype=torch.bool),
            "offline_tokens": torch.zeros((2, 3, RAW_TOKEN_DIM), dtype=torch.float32),
            "offline_mask": torch.ones((2, 3), dtype=torch.bool),
            "labels": torch.zeros((2,), dtype=torch.long),
            "indices": torch.arange(2, dtype=torch.long),
            "jet_ids": ["a", "b"],
            "split": "model_val",
        }
        model = FakeModel()
        with patch("teacher_logit_reco.set_matching.detr_slots.training.compute_detr_slot_hungarian_loss") as mocked:
            mocked.return_value = FakeLoss()
            metrics = run_detr_slot_reco_epoch(
                model,
                [batch],
                device=torch.device("cpu"),
                loss_config=tiny_config().loss_config(),
                amp=False,
            )

        self.assertEqual(metrics["num_batches"], 1.0)
        self.assertEqual(metrics["num_jets"], 2.0)
        self.assertFalse(model.forward_grad_enabled)
        mocked.assert_called_once()

    def test_checkpoint_output_contract_names_are_stable(self):
        model = build_detr_slot_reconstructor(tiny_config("gt"))
        config = model.to_config_dict()

        self.assertEqual(config["contract"], "hlt_to_free_slot_offline_set_v1")
        self.assertEqual(config["train_step"], DETR_SLOT_TRAIN_STEP)
        self.assertEqual(config["architecture"], "gt")


if __name__ == "__main__":
    unittest.main()
