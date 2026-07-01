from __future__ import annotations

import math
from pathlib import Path
import tempfile

import pytest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH,
    ARCHITECTURE_VIEW_MODEL_CONTRACT,
    ARCHITECTURE_VIEW_PRIMARY_METRIC,
    ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ArchitectureViewConfig,
    ArchitectureViewResidualParT,
    compute_architecture_view_init_logit_diff_vs_baseline,
    load_architecture_view_hlt_part_checkpoint,
    warm_start_architecture_view_part_model,
)
from teacher_logit_reco.local_compression_part import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES


torch = require_torch()


class _SeqFirstEmbed(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, rows):
        return self.linear(rows).transpose(0, 1).contiguous()


class _FakeParTMod(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = _SeqFirstEmbed(in_dim, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, features, v=None, mask=None):
        del v
        rows = features.transpose(1, 2).contiguous()
        embedded = self.embed(rows)
        if embedded.shape[0] == rows.shape[1] and embedded.shape[1] == rows.shape[0]:
            embedded_rows = embedded.transpose(0, 1).contiguous()
        else:
            embedded_rows = embedded
        particle_mask = mask.squeeze(1).to(dtype=embedded_rows.dtype)
        pooled = (embedded_rows * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(self.norm(pooled))


class DummyEmbeddingPart(ParticleTransformerHLTClassifier):
    """Tiny local stand-in with the same ``mod.embed`` injection seam."""

    def __init__(self, embed_dim: int = 16, num_classes: int = 2) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_embedding_part": True, "embed_dim": int(embed_dim), "num_classes": int(num_classes)}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), int(num_classes))

    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def make_tokens(batch: int = 2, particles: int = 7):
    tokens = torch.zeros((batch, particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((batch, particles), dtype=torch.bool)
    mask[0, :6] = True
    mask[1, :5] = True
    for batch_index in range(batch):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 18.0 + 2.0 * particle_index + 0.7 * batch_index
            eta = -0.3 + 0.09 * particle_index
            phi = -math.pi + 0.2 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.4
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5:10] = 0.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.01 * particle_index
            tokens[batch_index, particle_index, 11] = 0.04 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.06 + 0.01 * batch_index
    return tokens, mask


def small_config(**overrides):
    payload = {
        "view_dim": 10,
        "hidden_dim": 20,
        "pn_k": 3,
        "pn_layers": 1,
        "pfn_hidden_dim": 20,
        "pcnn_channels": 20,
        "pcnn_layers": 1,
        "fusion_hidden_dim": 24,
        "part_embed_dim": 16,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(overrides)
    return ArchitectureViewConfig(**payload)


def write_checkpoint(path: Path, model: torch.nn.Module, *, hlt_strength: float = 0.6) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selection_metric": ARCHITECTURE_VIEW_PRIMARY_METRIC,
            "hlt_degradation_strength": float(hlt_strength),
            "split_manifest_hash": "split-hash-av",
            "label_names": ["QCD", "Hgg"],
            "label_filter": [0, 1],
            "num_classes": 2,
            "model_config": dict(getattr(model, "config", {})),
        },
        path,
    )


def test_zero_injection_logits_match_direct_hlt_part_baseline():
    tokens, mask = make_tokens()
    part_model = DummyEmbeddingPart(embed_dim=16)
    model = ArchitectureViewResidualParT(
        small_config(),
        variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
        part_model=part_model,
    )
    model.eval()
    canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])
    with torch.no_grad():
        baseline_logits = part_model(
            canonical.points,
            canonical.features,
            canonical.lorentz_vectors,
            canonical.mask,
        )
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    assert output.output_contract == ARCHITECTURE_VIEW_MODEL_CONTRACT
    assert output.injection_summary["injection_applied"] is True
    assert output.injection_summary["embed_output_shape"][0] == tokens.shape[1]
    assert torch.allclose(output.view_output.delta_h, torch.zeros_like(output.view_output.delta_h))
    assert torch.allclose(output.logits, baseline_logits, atol=1.0e-6, rtol=1.0e-6)


def test_nonzero_embedding_delta_changes_logits():
    tokens, mask = make_tokens()
    model = ArchitectureViewResidualParT(
        small_config(),
        variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
        part_model=DummyEmbeddingPart(embed_dim=16),
    )
    model.eval()
    with torch.no_grad():
        zero_output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
        model.view_module.fusion.delta_projection.bias.fill_(2.0)
        changed_output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    assert changed_output.view_output.delta_h.abs().sum().item() > 0.0
    assert (changed_output.logits - zero_output.logits).abs().max().item() > 0.0


def test_baseline_recheck_variant_uses_no_views_and_still_matches_baseline():
    tokens, mask = make_tokens()
    part_model = DummyEmbeddingPart(embed_dim=16)
    model = ArchitectureViewResidualParT(
        small_config(),
        variant=ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
        part_model=part_model,
    )
    output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
    assert output.config.enabled_views == ()
    assert output.view_output.combined_view.shape[-1] == 0
    assert torch.allclose(output.view_output.delta_h, torch.zeros_like(output.view_output.delta_h))


def test_loads_exact_baseline_checkpoint_with_zero_missing_keys_and_metadata_checks():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_model_val.pt"
        source = DummyEmbeddingPart(embed_dim=16)
        target = ArchitectureViewResidualParT(
            small_config(),
            variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
            part_model=DummyEmbeddingPart(embed_dim=16),
        )
        with torch.no_grad():
            target.part_model.mod.head.bias.fill_(9.0)
        write_checkpoint(path, source)

        report = warm_start_architecture_view_part_model(
            target,
            path,
            expected_split_manifest_hash="split-hash-av",
            require_metadata=True,
        )

        assert report.missing_key_count == 0
        assert report.loaded_key_count == len(source.state_dict())
        assert target.baseline_checkpoint_report["baseline_checkpoint_split_manifest_hash"] == "split-hash-av"
        for key, value in source.state_dict().items():
            assert torch.allclose(target.part_model.state_dict()[key], value)


def test_strict_hlt_metadata_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_model_val.pt"
        write_checkpoint(path, DummyEmbeddingPart(embed_dim=16), hlt_strength=1.0)
        model = ArchitectureViewResidualParT(
            small_config(),
            variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
            part_model=DummyEmbeddingPart(embed_dim=16),
        )
        with pytest.raises(ValueError, match="hlt_degradation_strength mismatch"):
            load_architecture_view_hlt_part_checkpoint(path, model, require_metadata=True)


def test_init_logit_diff_records_exact_baseline_recovery():
    tokens, mask = make_tokens()
    model = ArchitectureViewResidualParT(
        small_config(),
        variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
        part_model=DummyEmbeddingPart(embed_dim=16),
    )
    diff = compute_architecture_view_init_logit_diff_vs_baseline(
        model,
        tokens,
        mask,
        max_constits=tokens.shape[1],
    )
    assert diff["allclose_atol_1e_6"]
    assert diff["max_abs_logit_diff"] <= 1.0e-6
    assert model.init_logit_diff_vs_baseline["allclose_atol_1e_6"]
    assert diff["embed_injection"]["injection_applied"] is True
