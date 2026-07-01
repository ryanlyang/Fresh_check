from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_BRANCH_PFN,
    ArchitectureViewConfig,
    ArchitectureViewResidualParT,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_binary_projection_metrics,
    architecture_view_binary_projection_scores,
    architecture_view_effective_variant,
    architecture_view_variant_num_classes,
    enabled_views_for_variant,
    load_architecture_view_hlt_part_checkpoint,
    normalize_architecture_view_variant,
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
        embedded = self.embed(rows).transpose(0, 1).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=embedded.dtype)
        pooled = (embedded * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(self.norm(pooled))


class DummyTenClassPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 16, num_classes: int = 10) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_ten_class_part": True, "embed_dim": int(embed_dim), "num_classes": int(num_classes)}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), int(num_classes))

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def _tokens(batch: int = 3, particles: int = 8):
    values = torch.zeros((batch, particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.ones((batch, particles), dtype=torch.bool)
    mask[0, -1] = False
    for b in range(batch):
        for p in range(particles):
            pt = 20.0 + 0.5 * b + p
            eta = -0.4 + 0.1 * p
            values[b, p, 0] = pt
            values[b, p, 1] = eta
            values[b, p, 2] = -2.0 + 0.2 * p
            values[b, p, 3] = pt * torch.cosh(torch.tensor(eta)).item() + 0.2
            values[b, p, 4] = 1.0 if p % 2 == 0 else -1.0
            values[b, p, 5 + (p % 5)] = 1.0
            values[b, p, 11] = 0.04
            values[b, p, 13] = 0.06
    values[~mask] = 0.0
    return values, mask


def _config(**overrides):
    payload = {
        "num_classes": 10,
        "view_dim": 8,
        "hidden_dim": 16,
        "pn_k": 3,
        "pn_layers": 1,
        "pfn_hidden_dim": 16,
        "pcnn_channels": 16,
        "pcnn_layers": 1,
        "fusion_hidden_dim": 20,
        "part_embed_dim": 16,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(overrides)
    return ArchitectureViewConfig(**payload)


def _write_checkpoint(path: Path, model: torch.nn.Module, *, label_names=None, label_filter=None, num_classes=10) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selection_metric": ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
            "hlt_degradation_strength": 0.6,
            "split_manifest_hash": "av10-split",
            "label_names": list(label_names or ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
            "label_filter": list(label_filter or ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
            "num_classes": int(num_classes),
            "model_config": dict(getattr(model, "config", {})),
        },
        path,
    )


def test_10class_variant_registry_and_behavior_mapping():
    assert normalize_architecture_view_variant("av10_pfn") == ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART
    assert architecture_view_variant_num_classes(ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART) == 10
    assert enabled_views_for_variant(ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART) == (
        ARCHITECTURE_VIEW_BRANCH_PFN,
    )
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK) == "av_baseline_recheck"
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL) == (
        "av_context_mlp_control"
    )
    assert enabled_views_for_variant(ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART)


def test_10class_model_logits_shape_and_zero_injection_recovery():
    tokens, mask = _tokens()
    part = DummyTenClassPart(embed_dim=16, num_classes=10)
    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
        part_model=part,
    )
    canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])
    with torch.no_grad():
        baseline_logits = part(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
    assert output.logits.shape == (tokens.shape[0], 10)
    assert output.config.num_classes == 10
    assert output.variant == ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART
    assert output.diagnostics()["variant"] == ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART
    assert torch.allclose(output.logits, baseline_logits, atol=1.0e-6, rtol=1.0e-6)


def test_10class_training_config_requires_full_labels_and_uses_accuracy_default():
    config = ArchitectureViewTaggerTrainConfig(
        output_dir="out",
        manifest_path="manifest.json.gz",
        hlt_cache_dir="hlt",
        baseline_checkpoint="baseline.pt",
        confirm_split_settings=True,
        confirm_final_test=True,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
        variant=ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    )
    assert config.resolved_num_classes == 10
    assert config.selection_metric == ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC
    assert config.model_config().num_classes == 10

    with pytest.raises(ValueError, match="binary-only"):
        ArchitectureViewTaggerTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            baseline_checkpoint="baseline.pt",
            confirm_split_settings=True,
            confirm_final_test=True,
            label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
            label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
            selection_metric="fpr_at_signal_eff_0p50",
            variant=ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
        )

    with pytest.raises(ValueError, match="expects num_classes=10"):
        ArchitectureViewTaggerTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            baseline_checkpoint="baseline.pt",
            confirm_split_settings=True,
            confirm_final_test=True,
            variant=ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
        )


def test_10class_binary_projection_score_extraction():
    logits = np.zeros((4, 10), dtype=np.float32)
    qcd = ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES.index("QCD")
    hgg = ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES.index("Hgg")
    logits[:, hgg] = np.asarray([3.0, 2.0, -1.0, -2.0], dtype=np.float32)
    logits[:, qcd] = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    scores = architecture_view_binary_projection_scores(
        logits,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        positive_label="Hgg",
        negative_label="QCD",
    )
    assert np.allclose(scores, logits[:, hgg] - logits[:, qcd])

    labels = np.asarray([hgg, hgg, qcd, qcd], dtype=np.int64)
    metrics = architecture_view_binary_projection_metrics(
        logits,
        labels,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        pairs=(("QCD", "Hgg"),),
    )
    assert metrics["QCD_vs_Hgg"]["available"]
    assert metrics["QCD_vs_Hgg"]["n_jets"] == 4
    assert metrics["QCD_vs_Hgg"]["positive_class_name"] == "Hgg"


def test_10class_checkpoint_metadata_is_strict():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.pt"
        source = DummyTenClassPart(embed_dim=16, num_classes=10)
        target = ArchitectureViewResidualParT(
            _config(),
            variant=ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
            part_model=DummyTenClassPart(embed_dim=16, num_classes=10),
        )
        _write_checkpoint(path, source)
        report = load_architecture_view_hlt_part_checkpoint(
            path,
            target,
            expected_selection_metric=ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
            expected_split_manifest_hash="av10-split",
            expected_label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
            expected_label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
            expected_num_classes=10,
            require_metadata=True,
        )
        assert report.baseline_checkpoint_num_classes == 10

        wrong_path = Path(tmp) / "wrong.pt"
        _write_checkpoint(wrong_path, source, label_names=("QCD", "Hgg"), label_filter=(0, 1), num_classes=2)
        with pytest.raises(ValueError, match="label_names|label_filter|num_classes"):
            load_architecture_view_hlt_part_checkpoint(
                wrong_path,
                target,
                expected_selection_metric=ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
                expected_split_manifest_hash="av10-split",
                expected_label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
                expected_label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
                expected_num_classes=10,
                require_metadata=True,
            )
