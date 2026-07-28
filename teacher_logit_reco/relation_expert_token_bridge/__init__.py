"""Relation-Expert Token Bridge campaign infrastructure."""

from .campaign import build_step1_bundle, publish_step1_bundle, validate_step1_bundle
from .contracts import (
    CAMPAIGN_SPEC_CONTRACT,
    STEP1_REPORT_CONTRACT,
    canonical_sha256,
    load_hashed_json,
    validate_content_hash,
)
from .determinism import (
    GLOBAL_DETERMINISM_CONTRACT,
    build_global_determinism,
    optimizer_update_counts,
    scheduled_learning_rate,
)
from .evaluation import (
    CLASSIFICATION_METRICS_CONTRACT,
    evaluate_classification,
    expected_calibration_error,
    qcd_signal_rejection,
    stable_probabilities,
)
from .expert_training import (
    EXPERT_LOSS_CANDIDATES,
    OfflineExpertDataset,
    OfflineExpertTrainingConfig,
    apply_attachment_trainability,
    build_attachment_pretraining_record,
    build_expert_loss_registry,
    build_teacher_logits_manifest,
    collect_expert_diagnostics,
    copy_obase_particle_backbone,
    make_offline_expert_loader,
    offline_expert_objective,
    preferred_expert_epoch,
    train_offline_expert,
    validate_teacher_logits_manifest,
)
from .provenance import source_snapshot
from .hlt_audit import (
    audit_strength_monotonicity,
    build_hlt_v3_degradation_audit,
    validate_hlt_v3_degradation_audit,
)
from .hlt_cache import (
    build_hlt_v3_cache,
    load_hlt_v3_cache,
    publish_hlt_v3_cache,
)
from .hlt_v3 import (
    HLT_V3_PROFILE_NAME,
    HLT_V3_PROFILE_VERSION,
    apply_hlt_v3_single_jet,
    build_hlt_v3_profile_contract,
    build_hlt_v3_view,
    measurement_validity_states,
)
from .normalizer_lineage import (
    build_normalizer_population_registry,
    normalizer_population_rows,
    validate_normalizer_population_registry,
)
from .expert_model import (
    RetbExpertModel,
    RetbParticleEncoder,
    build_expert_architecture_contract,
)
from .layerwise_pair_bias import (
    LayerwisePairBiasProvider,
    build_layerwise_pair_bias_contract,
)
from .particle_tap import (
    MeasurementStateEmbedding,
    ReferenceParticleStateTap,
    derive_measurement_states_torch,
)
from .step3 import build_step3_bundle, publish_step3_bundle, validate_step3_bundle
from .step4 import (
    aggregate_optimization_candidate_metrics,
    build_full_optimization_rows,
    build_locked_optimization_selection,
    build_optimization_candidate_metrics,
    build_primary_shape_screen_rows,
    build_stage_b_run_registry,
    build_step4_bundle,
    execute_miniature_stage_b,
    materialize_optimization_winner_followups,
    publish_step4_bundle,
    resolve_stage_b_run,
    select_optimization_candidate,
    validate_optimization_candidate_metrics,
    validate_locked_optimization_selection,
    validate_stage_b_run_registry,
    validate_step4_bundle,
)
from .summary_tokens import (
    CanonicalSummaryTokenizer,
    MultiDepthSummaryTokenizer,
    TokenOnlyExpertHead,
)
from .token_shape_registry import (
    build_token_shape_contract,
    resolve_expert_shapes,
    resolve_uniform_shape,
)
from .registry import build_registries, resolve_run_id
from .replicas import (
    DOMAIN_SEEDS,
    REALIZATION_POLICIES,
    event_rng_seed,
    replica_for,
)
from .splits import RetbSplitConfig
from .storage import (
    build_storage_measurements,
    miniature_storage_measurements,
)
from .workflow import authorize_dataset_access, validate_campaign_source

__all__ = [
    "CAMPAIGN_SPEC_CONTRACT",
    "CLASSIFICATION_METRICS_CONTRACT",
    "DOMAIN_SEEDS",
    "EXPERT_LOSS_CANDIDATES",
    "GLOBAL_DETERMINISM_CONTRACT",
    "HLT_V3_PROFILE_NAME",
    "HLT_V3_PROFILE_VERSION",
    "LayerwisePairBiasProvider",
    "MeasurementStateEmbedding",
    "OfflineExpertDataset",
    "OfflineExpertTrainingConfig",
    "ReferenceParticleStateTap",
    "RetbExpertModel",
    "RetbParticleEncoder",
    "REALIZATION_POLICIES",
    "RetbSplitConfig",
    "STEP1_REPORT_CONTRACT",
    "build_global_determinism",
    "aggregate_optimization_candidate_metrics",
    "build_attachment_pretraining_record",
    "build_expert_loss_registry",
    "build_teacher_logits_manifest",
    "build_hlt_v3_cache",
    "build_hlt_v3_degradation_audit",
    "build_hlt_v3_profile_contract",
    "build_hlt_v3_view",
    "build_expert_architecture_contract",
    "build_layerwise_pair_bias_contract",
    "build_normalizer_population_registry",
    "build_step3_bundle",
    "build_step4_bundle",
    "build_stage_b_run_registry",
    "build_primary_shape_screen_rows",
    "build_full_optimization_rows",
    "build_locked_optimization_selection",
    "build_optimization_candidate_metrics",
    "build_token_shape_contract",
    "build_registries",
    "build_step1_bundle",
    "build_storage_measurements",
    "authorize_dataset_access",
    "canonical_sha256",
    "event_rng_seed",
    "evaluate_classification",
    "expected_calibration_error",
    "derive_measurement_states_torch",
    "apply_hlt_v3_single_jet",
    "audit_strength_monotonicity",
    "load_hashed_json",
    "miniature_storage_measurements",
    "measurement_validity_states",
    "normalizer_population_rows",
    "offline_expert_objective",
    "optimizer_update_counts",
    "scheduled_learning_rate",
    "publish_step1_bundle",
    "replica_for",
    "resolve_run_id",
    "source_snapshot",
    "load_hlt_v3_cache",
    "publish_hlt_v3_cache",
    "publish_step3_bundle",
    "publish_step4_bundle",
    "resolve_expert_shapes",
    "resolve_uniform_shape",
    "resolve_stage_b_run",
    "select_optimization_candidate",
    "stable_probabilities",
    "qcd_signal_rejection",
    "validate_content_hash",
    "validate_campaign_source",
    "validate_hlt_v3_degradation_audit",
    "validate_normalizer_population_registry",
    "validate_optimization_candidate_metrics",
    "validate_locked_optimization_selection",
    "validate_step3_bundle",
    "validate_step4_bundle",
    "validate_stage_b_run_registry",
    "validate_step1_bundle",
    "CanonicalSummaryTokenizer",
    "MultiDepthSummaryTokenizer",
    "TokenOnlyExpertHead",
    "apply_attachment_trainability",
    "collect_expert_diagnostics",
    "copy_obase_particle_backbone",
    "execute_miniature_stage_b",
    "make_offline_expert_loader",
    "materialize_optimization_winner_followups",
    "preferred_expert_epoch",
    "train_offline_expert",
    "validate_teacher_logits_manifest",
]
