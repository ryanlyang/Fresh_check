from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    JetIdentity,
    JetView,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_DEEPSETS_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_FINETUNE_ONLY_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_NOISE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_DEEPSETS_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_SELF_ATTENTION_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_ONLY_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_WITHIN_JET_SHUFFLED_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
    ArchitectureViewConfig,
    ArchitectureViewResidualParT,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_variant_is_runnable,
    train_architecture_view_tagger,
)
from teacher_logit_reco.local_compression_part import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset


torch = require_torch()


class _BatchFirstEmbed(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, rows):
        return self.linear(rows)


class _FakeParTMod(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = _BatchFirstEmbed(in_dim, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, features, v=None, mask=None):
        del v
        rows = features.transpose(1, 2).contiguous()
        embedded = self.embed(rows)
        particle_mask = mask.squeeze(1).to(dtype=embedded.dtype)
        pooled = (embedded * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(self.norm(pooled))


class DummyTenClassPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 16) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_embedding_part": True, "embed_dim": int(embed_dim), "num_classes": 10}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), 10)

    def no_weight_decay(self):
        return set()

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def _config() -> ArchitectureViewConfig:
    return ArchitectureViewConfig(
        view_dim=8,
        hidden_dim=16,
        pn_k=3,
        pn_layers=1,
        pfn_hidden_dim=16,
        pcnn_channels=16,
        pcnn_layers=1,
        fusion_hidden_dim=20,
        part_embed_dim=16,
        num_classes=10,
        dropout=0.0,
        attention_dropout=0.0,
    )


def _tokens(batch: int = 2, particles: int = 6):
    tokens = torch.zeros((batch, particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((batch, particles), dtype=torch.bool)
    mask[:, :5] = True
    for batch_index in range(batch):
        for particle_index in range(5):
            label_hint = batch_index + particle_index
            pt = 18.0 + 0.8 * label_hint
            eta = -0.4 + 0.08 * particle_index
            phi = -math.pi + 0.25 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.3
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.01 * particle_index
            tokens[batch_index, particle_index, 11] = 0.03
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.04
    return tokens, mask


def _baseline_logits(model: ArchitectureViewResidualParT, tokens, mask):
    canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])
    return model.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)


def _toy_view(split: str, *, n_jets: int = 20) -> JetView:
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    labels = np.asarray([index % 10 for index in range(n_jets)], dtype=np.int64)
    for jet in range(n_jets):
        valid = 4 + (jet % 2)
        mask[jet, :valid] = True
        for particle in range(valid):
            pt = 12.0 + 0.4 * jet + 0.7 * particle + 0.2 * labels[jet]
            eta = -0.25 + 0.08 * particle
            phi = -math.pi + 0.22 * particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt * math.cosh(eta) + 0.25
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
    jet_ids = [JetIdentity(file="toy.root", entry=index, label=int(labels[index])) for index in range(n_jets)]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"toy-{split}",
            "jet_identity_hash": f"toy-identity-{split}",
            "hlt_params": {"strength": 0.6},
        },
    )


def _dataset(split: str) -> SubtokenHLTJetDataset:
    return SubtokenHLTJetDataset(
        _toy_view(split),
        label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    )


def _write_manifest(path: Path) -> str:
    manifest = SplitManifest(
        data_dir="toy",
        max_constits=6,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label={str(name): index for index, name in enumerate(LABEL_NAMES)},
        split_sizes={split: 0 for split in SPLIT_ORDER},
        split_seeds={split: index + 1 for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits={split: [] for split in SPLIT_ORDER},
        metadata={"test_manifest": True},
    )
    save_split_manifest(manifest, path)
    return manifest_hash(manifest)


def _write_checkpoint(path: Path, model: torch.nn.Module, *, split_manifest_hash: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selection_metric": ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
            "hlt_degradation_strength": 0.6,
            "split_manifest_hash": split_manifest_hash,
            "label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
            "label_filter": list(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
            "num_classes": 10,
            "model_config": dict(getattr(model, "config", {})),
        },
        path,
    )


def test_step3_wide_feature_mlp_has_more_adapter_capacity_and_zero_recovery():
    tokens, mask = _tokens()
    base = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        part_model=DummyTenClassPart(),
    )
    wide = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
        part_model=DummyTenClassPart(),
    )
    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE)
    assert wide.parameter_accounting()["adapter_params"] > base.parameter_accounting()["adapter_params"]

    wide.eval()
    with torch.no_grad():
        baseline = _baseline_logits(wide, tokens, mask)
        output = wide(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    diagnostics = output.diagnostics()
    assert diagnostics["variant_behavior"]["uses_wide_feature_mlp_adapter"]
    assert diagnostics["view_fusion"]["wide_feature_mlp_adapter"]
    assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)


def test_step3_shuffled_feature_adapter_is_feature_mlp_control_with_broken_context():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
        part_model=DummyTenClassPart(),
    )
    model.eval()
    with torch.no_grad():
        baseline = _baseline_logits(model, tokens, mask)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    diagnostics = output.diagnostics()
    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER)
    assert diagnostics["variant_behavior"]["uses_shuffled_feature_adapter"]
    assert diagnostics["variant_behavior"]["feature_shuffle_policy"] == "cross_particle_roll_plus_feature_permutation"
    assert diagnostics["view_fusion"]["shuffled_feature_adapter"]
    assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)


def test_step3_feature_mlp_adapter_reports_real_adapter_output_norm():
    tokens, mask = _tokens()
    model = ArchitectureViewResidualParT(
        _config(),
        variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
        part_model=DummyTenClassPart(),
    )
    with torch.no_grad():
        model.context_control[-1].bias.fill_(0.5)
        output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

    view_fusion = output.diagnostics()["view_fusion"]
    assert view_fusion["adapter_output_norm_mean"] > 0.0
    assert view_fusion["adapter_output_norm_p90"] > 0.0
    assert output.diagnostics()["part_jet_representation_shape"] == [2, 16]


def test_multi_adapter_step2_contextual_feature_adapters_zero_recover_baseline():
    tokens, mask = _tokens()
    variants = (
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_DEEPSETS_ADAPTER,
            "uses_feature_deepsets_context_adapter",
            "feature_deepsets_context_adapter",
        ),
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER,
            "uses_feature_self_attention_context_adapter",
            "feature_self_attention_context_adapter",
        ),
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_WITHIN_JET_SHUFFLED_ADAPTER,
            "uses_within_jet_shuffled_context_adapter",
            "within_jet_shuffled_context_adapter",
        ),
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_NOISE_ADAPTER,
            "uses_noise_context_adapter",
            "uses_noise_context_adapter",
        ),
    )
    for variant, behavior_flag, diagnostic_flag in variants:
        model = ArchitectureViewResidualParT(
            _config(),
            variant=variant,
            part_model=DummyTenClassPart(),
        )
        model.eval()
        with torch.no_grad():
            baseline = _baseline_logits(model, tokens, mask)
            output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

        diagnostics = output.diagnostics()
        assert architecture_view_variant_is_runnable(variant)
        assert diagnostics["variant_behavior"][behavior_flag]
        assert diagnostics["view_fusion"][diagnostic_flag]
        if variant == ARCHITECTURE_VIEW_10CLASS_CONTEXT_WITHIN_JET_SHUFFLED_ADAPTER:
            assert diagnostics["view_fusion"]["feature_shuffle_policy"] == (
                "valid_particles_only_within_jet_feature_permutation_and_roll"
            )
            assert diagnostics["view_fusion"]["padding_rows_used_as_shuffle_sources"] is False
        assert diagnostics["view_fusion"]["baseline_recovery_zero_projection"]
        assert output.parameter_accounting["adapter_params"] > 0
        assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)


def test_multi_adapter_step2_part_embedding_adapters_zero_recover_baseline_through_hook():
    tokens, mask = _tokens()
    variants = (
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_ONLY_MLP_ADAPTER,
            "uses_part_only_adapter",
            "part_embedding_mlp",
        ),
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_DEEPSETS_ADAPTER,
            "uses_part_embedding_deepsets_context_adapter",
            "part_embedding_deepsets_context_adapter",
        ),
        (
            ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_SELF_ATTENTION_ADAPTER,
            "uses_part_embedding_self_attention_context_adapter",
            "part_embedding_self_attention_context_adapter",
        ),
    )
    for variant, behavior_flag, adapter_kind in variants:
        model = ArchitectureViewResidualParT(
            _config(),
            variant=variant,
            part_model=DummyTenClassPart(),
        )
        model.eval()
        with torch.no_grad():
            baseline = _baseline_logits(model, tokens, mask)
            output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

        diagnostics = output.diagnostics()
        assert architecture_view_variant_is_runnable(variant)
        assert diagnostics["variant_behavior"][behavior_flag]
        assert diagnostics["embed_injection"]["adapter_kind"] == adapter_kind
        assert diagnostics["embed_injection"]["injection_applied"]
        assert output.parameter_accounting["adapter_params"] > 0
        assert torch.allclose(output.logits, baseline, atol=1.0e-6, rtol=1.0e-6)


def test_multi_adapter_step3_finetune_only_control_trains_part_without_adapters():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = tmp_path / "split_manifest.json"
        manifest_sha = _write_manifest(manifest)
        checkpoint = tmp_path / "baseline.pt"
        _write_checkpoint(checkpoint, DummyTenClassPart(), split_manifest_hash=manifest_sha)
        output_dir = tmp_path / "out"
        config = ArchitectureViewTaggerTrainConfig(
            output_dir=str(output_dir),
            manifest_path=str(manifest),
            hlt_cache_dir="unused",
            baseline_checkpoint=str(checkpoint),
            confirm_split_settings=True,
            confirm_final_test=True,
            seed=31,
            batch_size=5,
            eval_batch_size=5,
            epochs=1,
            adapter_lr=1.0e-3,
            part_lr=1.0e-4,
            num_workers=0,
            device="cpu",
            amp=False,
            early_stop_patience=-1,
            max_train_batches=1,
            max_val_batches=1,
            max_stack_val_batches=1,
            max_final_test_batches=1,
            label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
            label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
            variant=ARCHITECTURE_VIEW_10CLASS_CONTEXT_FINETUNE_ONLY_CONTROL,
            view_dim=8,
            hidden_dim=16,
            pn_k=3,
            pn_layers=1,
            pfn_hidden_dim=16,
            pcnn_channels=16,
            pcnn_layers=1,
            fusion_hidden_dim=20,
            part_embed_dim=16,
            dropout=0.0,
            attention_dropout=0.0,
            freeze_part_epochs=2,
        )
        model = ArchitectureViewResidualParT(
            _config(),
            variant=ARCHITECTURE_VIEW_10CLASS_CONTEXT_FINETUNE_ONLY_CONTROL,
            part_model=DummyTenClassPart(),
        )

        report = train_architecture_view_tagger(
            config,
            model=model,
            train_dataset=_dataset("model_train"),
            val_dataset=_dataset("model_val"),
            stack_val_dataset=_dataset("stack_val"),
            final_test_dataset=_dataset("final_test"),
        )

    assert report["best_epoch"] == 1
    assert report["epochs_completed"] == 1
    assert report["effective_freeze_part_epochs"] == 2
    assert report["freeze_part_for_all_epochs"] is False
    assert report["freeze_warmup_has_trainable_params"] is False
    assert report["freeze_warmup_note"] == (
        "schedule_matched_no_adapter_control_has_no_trainable_parameters_until_part_unfreeze"
    )
    assert report["variant_behavior"]["uses_finetune_only_control"]
    assert report["variant_behavior"]["trainable_no_adapter_control"]
    assert report["parameter_accounting"]["adapter_params"] == 0
    assert report["parameter_accounting"]["trainable_part_params"] == 0
    assert report["baseline_load_report"]["weight_warm_start_skipped"] is False
    assert report["training_curves"]


def test_step3_frozen_part_feature_adapter_keeps_part_frozen_during_training():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = tmp_path / "split_manifest.json"
        manifest_sha = _write_manifest(manifest)
        checkpoint = tmp_path / "baseline.pt"
        _write_checkpoint(checkpoint, DummyTenClassPart(), split_manifest_hash=manifest_sha)
        output_dir = tmp_path / "out"
        config = ArchitectureViewTaggerTrainConfig(
            output_dir=str(output_dir),
            manifest_path=str(manifest),
            hlt_cache_dir="unused",
            baseline_checkpoint=str(checkpoint),
            confirm_split_settings=True,
            confirm_final_test=True,
            seed=23,
            batch_size=5,
            eval_batch_size=5,
            epochs=1,
            adapter_lr=1.0e-3,
            part_lr=1.0e-4,
            num_workers=0,
            device="cpu",
            amp=False,
            early_stop_patience=-1,
            max_train_batches=1,
            max_val_batches=1,
            max_stack_val_batches=1,
            max_final_test_batches=1,
            label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
            label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
            variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
            view_dim=8,
            hidden_dim=16,
            pn_k=3,
            pn_layers=1,
            pfn_hidden_dim=16,
            pcnn_channels=16,
            pcnn_layers=1,
            fusion_hidden_dim=20,
            part_embed_dim=16,
            dropout=0.0,
            attention_dropout=0.0,
            freeze_part_epochs=0,
        )
        model = ArchitectureViewResidualParT(
            _config(),
            variant=ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
            part_model=DummyTenClassPart(),
        )

        report = train_architecture_view_tagger(
            config,
            model=model,
            train_dataset=_dataset("model_train"),
            val_dataset=_dataset("model_val"),
            stack_val_dataset=_dataset("stack_val"),
            final_test_dataset=_dataset("final_test"),
        )

    assert report["freeze_part_for_all_epochs"] is True
    assert report["parameter_accounting"]["trainable_part_params"] == 0
    assert report["variant_behavior"]["uses_frozen_part_feature_adapter"]
