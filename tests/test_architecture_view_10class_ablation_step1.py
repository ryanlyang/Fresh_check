from __future__ import annotations

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_DEFAULT_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_CONTEXT_ADAPTER_VARIANTS,
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
    ARCHITECTURE_VIEW_BRANCH_PCNN,
    ArchitectureViewConfig,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    architecture_view_config_manifest,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_effective_variant,
    architecture_view_runnable_variants,
    architecture_view_variant_is_runnable,
    architecture_view_variant_num_classes,
    architecture_view_variant_spec,
    enabled_views_for_variant,
    normalize_architecture_view_variant,
)


def _train_config(variant: str) -> ArchitectureViewTaggerTrainConfig:
    return ArchitectureViewTaggerTrainConfig(
        output_dir="out",
        manifest_path="manifest.json.gz",
        hlt_cache_dir="hlt",
        baseline_checkpoint="baseline.pt",
        confirm_split_settings=True,
        confirm_final_test=True,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
        variant=variant,
    )


def test_av10_ablation_registry_names_and_metadata_are_stable():
    assert ARCHITECTURE_VIEW_10CLASS_ABLATION_DEFAULT_VARIANTS[0] == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK
    )
    assert ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER in ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS
    assert normalize_architecture_view_variant("av10_feature_mlp") == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER
    )
    assert normalize_architecture_view_variant("av10_input_delta") == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES
    )
    assert normalize_architecture_view_variant("av10_big_part") == ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART
    assert architecture_view_variant_num_classes(ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART) == 10

    spec = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER)
    assert spec.suite == "av10_ablation"
    assert spec.input_source == "hlt"
    assert spec.adapter_type == "feature_mlp_context"
    assert spec.parameter_target == "current_context_mlp_adapter"
    assert spec.is_candidate
    assert spec.is_runnable


def test_av10_ablation_registry_tracks_controls_and_future_variants():
    larger = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART)
    assert larger.is_control
    assert not larger.is_candidate
    assert larger.is_runnable
    assert larger.part_size == "larger"
    assert larger.implementation_step == "implemented_step2_larger_part_control"

    part_only = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER)
    assert part_only.is_control
    assert part_only.adapter_type == "part_embedding_mlp"
    assert part_only.is_runnable

    shuffled = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER)
    assert shuffled.is_control
    assert shuffled.shuffle_policy == "shuffled_features"
    assert shuffled.is_runnable

    wide = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE)
    assert wide.adapter_type == "feature_mlp_context_wide"
    assert wide.is_runnable

    input_delta = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES)
    assert input_delta.adapter_type == "feature_mlp_delta_F"
    assert input_delta.parameter_target == "input_feature_repair_adapter"
    assert input_delta.implementation_step == "implemented_step6_lc_mlp_delta_features"
    assert input_delta.is_runnable

    frozen = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER)
    assert frozen.freeze_policy == "frozen_part_adapter_only"
    assert frozen.is_runnable


def test_av10_ablation_behavior_mapping_for_currently_runnable_variants():
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK) == (
        ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
    )
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER) == (
        ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL
    )
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES) == (
        ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
    )
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER) == (
        ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL
    )
    assert enabled_views_for_variant(ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT) == (
        ARCHITECTURE_VIEW_BRANCH_PCNN,
    )
    assert architecture_view_variant_is_runnable(ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT)
    assert ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART in architecture_view_runnable_variants()


def test_training_config_accepts_step3_ablation_variants():
    config = _train_config(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER)
    assert config.resolved_num_classes == 10
    assert config.selection_metric == ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC
    assert config.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER

    larger_config = _train_config(ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART)
    assert larger_config.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART
    assert _train_config(ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE).variant == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE
    )
    assert _train_config(ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES).variant == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES
    )
    assert _train_config(ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER).variant == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER
    )
    assert _train_config(ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER).variant == (
        ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER
    )


def test_contextual_adapter_step2_variants_are_registered_and_module_variants_are_runnable():
    expected = (
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_FINETUNE_ONLY_CONTROL,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_ONLY_MLP_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_DEEPSETS_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_DEEPSETS_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_SELF_ATTENTION_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_WITHIN_JET_SHUFFLED_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_NOISE_ADAPTER,
    )
    assert ARCHITECTURE_VIEW_10CLASS_CONTEXT_ADAPTER_VARIANTS == expected
    for variant in expected:
        assert variant in ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS
        spec = architecture_view_variant_spec(variant)
        assert spec.suite == "av10_contextual_adapter"
        assert spec.input_source == "hlt"
        assert architecture_view_variant_num_classes(variant) == 10
        assert spec.is_runnable
        assert variant in architecture_view_runnable_variants()

    assert normalize_architecture_view_variant("av10_feature_deepsets") == (
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_DEEPSETS_ADAPTER
    )
    assert normalize_architecture_view_variant("av10_feature_attention_context") == (
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER
    )
    assert normalize_architecture_view_variant("av10_embedding_attention") == (
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_SELF_ATTENTION_ADAPTER
    )
    assert normalize_architecture_view_variant("av10_within_jet_shuffle") == (
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_WITHIN_JET_SHUFFLED_ADAPTER
    )
    assert normalize_architecture_view_variant("av10_noise_context") == (
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_NOISE_ADAPTER
    )


def test_contextual_adapter_step1_specs_capture_planned_controls_and_adapter_types():
    finetune = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_CONTEXT_FINETUNE_ONLY_CONTROL)
    assert finetune.is_control
    assert not finetune.is_candidate
    assert finetune.adapter_type == "none"
    assert finetune.parameter_target == "baseline_part_finetune_schedule"
    assert finetune.implementation_step == "implemented_multi_adapter_step3_finetune_only_control"

    part_only = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_ONLY_MLP_ADAPTER)
    assert part_only.is_control
    assert part_only.adapter_type == "part_embedding_mlp"

    feature_deepsets = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_DEEPSETS_ADAPTER)
    assert feature_deepsets.adapter_type == "feature_deepsets_context"
    assert feature_deepsets.is_candidate

    feature_attention = architecture_view_variant_spec(
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER
    )
    assert feature_attention.adapter_type == "feature_self_attention_context"

    embedding_deepsets = architecture_view_variant_spec(
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_DEEPSETS_ADAPTER
    )
    assert embedding_deepsets.adapter_type == "part_embedding_deepsets_context"

    embedding_attention = architecture_view_variant_spec(
        ARCHITECTURE_VIEW_10CLASS_CONTEXT_PART_EMBEDDING_SELF_ATTENTION_ADAPTER
    )
    assert embedding_attention.adapter_type == "part_embedding_self_attention_context"

    shuffled = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_CONTEXT_WITHIN_JET_SHUFFLED_ADAPTER)
    assert shuffled.is_control
    assert shuffled.shuffle_policy == "within_jet_shuffled_context"

    noise = architecture_view_variant_spec(ARCHITECTURE_VIEW_10CLASS_CONTEXT_NOISE_ADAPTER)
    assert noise.is_control
    assert noise.adapter_type == "noise_context"
    assert noise.shuffle_policy == "deterministic_noise_context"


def test_contextual_adapter_step1_effective_variants_and_manifest_entries():
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_CONTEXT_FINETUNE_ONLY_CONTROL) == (
        ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
    )
    assert architecture_view_effective_variant(ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_DEEPSETS_ADAPTER) == (
        ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL
    )
    assert enabled_views_for_variant(ARCHITECTURE_VIEW_10CLASS_CONTEXT_FEATURE_SELF_ATTENTION_ADAPTER) == ()
    manifest = architecture_view_config_manifest()
    assert manifest["ten_class_context_adapter_variants"] == list(ARCHITECTURE_VIEW_10CLASS_CONTEXT_ADAPTER_VARIANTS)
    for variant in ARCHITECTURE_VIEW_10CLASS_CONTEXT_ADAPTER_VARIANTS:
        assert variant in manifest["ten_class_all_variants"]
        assert manifest["variants"][variant]["suite"] == "av10_contextual_adapter"


def test_contextual_adapter_step1_config_fields_validate_and_round_trip():
    config = ArchitectureViewConfig(
        context_adapter_dim=128,
        context_adapter_heads=8,
        context_adapter_layers=2,
        context_adapter_mlp_ratio=3.0,
        context_adapter_noise_seed=17,
    )
    payload = config.to_dict()
    assert payload["context_adapter_dim"] == 128
    assert payload["context_adapter_heads"] == 8
    assert payload["context_adapter_layers"] == 2
    assert payload["context_adapter_mlp_ratio"] == 3.0
    assert payload["context_adapter_noise_seed"] == 17
    restored = ArchitectureViewConfig.from_dict(payload)
    assert restored.context_adapter_dim == 128
    assert restored.context_adapter_heads == 8
    assert restored.context_adapter_layers == 2
    assert restored.context_adapter_mlp_ratio == 3.0
    assert restored.context_adapter_noise_seed == 17
