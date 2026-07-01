from __future__ import annotations

from types import SimpleNamespace

import torch

from teacher_logit_reco.arch_residual_part.model import (
    ARCH_RESIDUAL_ARCHITECTURES,
    ArchResidualExpertConfig,
    ArchResidualPartModel,
    FrozenHLTPartOutput,
)


class _DummyCanonical:
    def __init__(self, features: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> None:
        self._features = features
        self.selected_tokens = tokens
        self.particle_mask = mask

    def feature_rows(self) -> torch.Tensor:
        return self._features


class _DummyBaseline(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)
        self.part_model = SimpleNamespace()

    def forward_outputs(self, tokens: torch.Tensor, mask: torch.Tensor, *, max_constits: int | None = None):
        del max_constits
        features = torch.randn(tokens.shape[0], tokens.shape[1], 19, device=tokens.device)
        logits = self.fixed_logits[: tokens.shape[0]].to(tokens.device)
        return FrozenHLTPartOutput(
            logits=logits,
            canonical_inputs=_DummyCanonical(features, tokens, mask),
        )


def test_arch_residual_zero_init_preserves_baseline_logits():
    tokens = torch.randn(3, 8, 14)
    mask = torch.ones(3, 8, dtype=torch.bool)
    baseline_logits = torch.tensor([[0.0, 1.0], [1.5, -0.5], [-0.25, 0.75]])

    for architecture in ARCH_RESIDUAL_ARCHITECTURES:
        model = ArchResidualPartModel(
            ArchResidualExpertConfig(
                architecture=architecture,
                hidden_dim=32,
                particle_layers=2,
                global_layers=1,
                edge_k=4,
                dropout=0.0,
            ),
            baseline=_DummyBaseline(baseline_logits),
        )
        output = model(tokens, mask, return_outputs=True)
        assert torch.allclose(output.logits, baseline_logits, atol=1e-7)
        assert torch.allclose(output.correction, torch.zeros_like(output.correction), atol=1e-7)
        assert output.residual_only_logits.shape == baseline_logits.shape
