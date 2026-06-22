import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.set_matching.detr_slots.cache import (
        DEFAULT_DETR_SLOT_CACHE_SPLITS,
        DETR_SLOT_CACHE_STEP,
        DetrSlotRecoViewCacheConfig,
        _select_top_detr_slots,
    )
    from teacher_logit_reco.set_matching.detr_slots.outputs import detr_slot_output_from_tensors
else:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DetrFreeSlotStep12CacheTests(unittest.TestCase):
    def test_cache_config_final_test_requires_confirmation(self):
        with self.assertRaises(ValueError):
            DetrSlotRecoViewCacheConfig(
                output_dir="unused",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt_cache",
                reconstructor_checkpoint="best_model_val.pt",
                splits=("final_test",),
            )

        config = DetrSlotRecoViewCacheConfig(
            output_dir="cache_root",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt_cache",
            reconstructor_checkpoint="best_model_val.pt",
            architecture="part",
            splits=DEFAULT_DETR_SLOT_CACHE_SPLITS,
            confirm_final_test=True,
        )
        self.assertEqual(config.architecture, "gt")
        self.assertEqual(config.cache_path("gt", "stack_val").as_posix(), "cache_root/gt/stack_val_reconstructed_view.npz")
        self.assertEqual(config.metadata_path("gt", "stack_val").name, "stack_val_reconstructed_view_metadata.json")

    def test_select_top_detr_slots_orders_by_existence_confidence(self):
        tokens = torch.zeros((1, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        for slot in range(4):
            tokens[0, slot, 0] = float(slot + 1)
            tokens[0, slot, 3] = float(slot + 2)
        output = detr_slot_output_from_tensors(
            tokens=tokens,
            existence_logits=torch.tensor([[0.0, 3.0, -3.0, 1.0]], dtype=torch.float32),
            slot_mask=torch.tensor([[True, True, True, False]]),
        )

        selected_tokens, mask, confidence, logits, source_indices = _select_top_detr_slots(
            output,
            export_max_tokens=3,
            confidence_threshold=0.70,
            min_tokens_per_view=1,
        )

        self.assertEqual(selected_tokens.shape, (1, 3, RAW_TOKEN_DIM))
        self.assertEqual(source_indices.tolist(), [[1, 0, 2]])
        self.assertEqual(logits.tolist(), [[3.0, 0.0, -3.0]])
        self.assertEqual(mask.tolist(), [[True, False, False]])
        self.assertGreater(float(confidence[0, 0]), float(confidence[0, 1]))

    def test_select_top_detr_slots_respects_min_tokens(self):
        tokens = torch.zeros((1, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        tokens[..., 0] = 1.0
        tokens[..., 3] = 2.0
        output = detr_slot_output_from_tensors(
            tokens=tokens,
            existence_logits=torch.tensor([[-4.0, -5.0, -6.0]], dtype=torch.float32),
            slot_mask=torch.ones((1, 3), dtype=torch.bool),
        )

        _, mask, _, _, source_indices = _select_top_detr_slots(
            output,
            export_max_tokens=3,
            confidence_threshold=0.90,
            min_tokens_per_view=2,
        )

        self.assertEqual(source_indices.tolist(), [[0, 1, 2]])
        self.assertEqual(mask.tolist(), [[True, True, False]])

    def test_step_constant_is_stable(self):
        self.assertEqual(DETR_SLOT_CACHE_STEP, "detr_free_slot_step12_cache_reconstructed_views")


if __name__ == "__main__":
    unittest.main()
