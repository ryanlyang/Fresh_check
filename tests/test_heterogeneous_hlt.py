import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.heterogeneous_hlt import (
        build_heterogeneous_hlt_classifier,
        default_model_name_for_architecture,
        evaluate_heterogeneous_hlt_model,
        normalize_architecture_name,
    )
    from jetclass_fresh.hlt_baseline import HLTBaselineTrainConfig
else:  # pragma: no cover - environment dependent
    torch = None


def make_fixed_hlt_view(n_jets=5):
    tokens = np.zeros((n_jets, 8, 14), dtype=np.float32)
    mask = np.zeros((n_jets, 8), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 3
    for jet_index in range(n_jets):
        mask[jet_index, :4] = True
        for part_index in range(4):
            pt = 2.0 + 0.1 * jet_index + part_index
            eta = 0.03 * part_index
            phi = 0.11 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5] = 1.0
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[JetIdentity(file="synthetic.root", entry=i, label=int(label)) for i, label in enumerate(labels)],
        split="stack_val",
        metadata={"view": "fixed_hlt", "hlt_content_hash": "synthetic"},
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class HeterogeneousHLTTests(unittest.TestCase):
    def test_architecture_aliases_and_model_names(self):
        self.assertEqual(normalize_architecture_name("ParticleNet"), "pn")
        self.assertEqual(default_model_name_for_architecture("pcnn"), "hlt_pcnn")

    def test_pfn_and_pcnn_forward_on_hlt_batch(self):
        view = make_fixed_hlt_view(6)
        for architecture in ("pfn", "pcnn"):
            model = build_heterogeneous_hlt_classifier(architecture, num_classes=10, model_size="tiny")
            model.eval()
            block = evaluate_heterogeneous_hlt_model(
                model,
                view,
                model_name=default_model_name_for_architecture(architecture),
                architecture=architecture,
                batch_size=3,
                num_workers=0,
                device=torch.device("cpu"),
            )
            self.assertEqual(block.logits.shape, (6, 10))
            self.assertEqual(block.labels.tolist(), view.labels.tolist())
            self.assertEqual(block.metadata["hlt_architecture"], architecture)


if __name__ == "__main__":
    unittest.main()
