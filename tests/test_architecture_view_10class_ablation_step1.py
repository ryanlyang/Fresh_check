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
    ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
    ARCHITECTURE_VIEW_BRANCH_PCNN,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
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
