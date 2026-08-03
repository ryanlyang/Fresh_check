"""Machine-readable A0-G5 registry for the ABPH campaign."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .config import (
    ABPH_DEPLOYABLE_EVAL_SPLITS,
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_HLT_PROFILE_VERSION,
    ABPH_RESOLVED_CONFIG_CONTRACT,
    canonical_hash,
)
from .schemas import ABPH_MAX_PARTICLES, schema_manifest


ABPH_VARIANT_REGISTRY_CONTRACT = "adaptive_binary_pseudooffline_variant_registry_v1"

ALL_SPLITS: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
MODEL_VAL_ONLY: tuple[str, ...] = ("model_val",)
STACK_EVAL_SPLITS: tuple[str, ...] = ("stack_train", "stack_val", "final_test")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(row) for key, row in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(row) for row in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(row) for key, row in value.items()}
    if isinstance(value, tuple):
        return [_thaw(row) for row in value]
    return value


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {str(key): _thaw(value) for key, value in base.items()}
    for key, value in update.items():
        key = str(key)
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _thaw(value)
    return result


def _base_config() -> dict[str, Any]:
    schemas = schema_manifest()
    return {
        "contract": ABPH_RESOLVED_CONFIG_CONTRACT,
        "data": {
            "hlt_profile": ABPH_HLT_PROFILE,
            "hlt_profile_version": ABPH_HLT_PROFILE_VERSION,
            "hlt_degradation_strength": ABPH_HLT_DEGRADATION_STRENGTH,
            "max_particles": ABPH_MAX_PARTICLES,
            "input_view": "hlt",
            "requires_offline_targets": False,
            "requires_teacher_logits": False,
            "offline_target_splits": ["model_train", "model_val"],
            "final_test_teacher_free": True,
        },
        "schemas": {
            "manifest_hash": schemas["manifest_hash"],
            "schema_hashes": {
                name: row["schema_hash"] for name, row in schemas["schemas"].items()
            },
        },
        "model": {
            "hlt_part": {
                "enabled": True,
                "embed_dims": [192, 768, 192],
                "pair_embed_dims": [96, 96, 96],
                "num_heads": 8,
                "num_layers": 12,
                "num_cls_layers": 3,
            },
            "root_predictor": {
                "enabled": True,
                "kind": "semantic_query_probabilistic",
                "query_blocks": 4,
                "shared_across_hierarchies": True,
            },
            "hierarchy": {
                "enabled": True,
                "grouping": "exclusive_kt",
                "grouping_radius": 0.8,
                "grouping_power": 1,
                "capacities": [2, 4, 8, 16, 32],
                "constrained": True,
                "level_specific_weights": True,
                "decoder_blocks_per_level": 4,
            },
            "distribution": {
                "enabled": True,
                "mean_views": 1,
                "stochastic_views": 4,
                "latent_dim": 64,
                "shared_compiled_root": True,
            },
            "renderer": {
                "enabled": True,
                "kind": "local_set_nbody",
                "exact_nbody_projection": True,
                "local_matching": True,
            },
            "pseudo_part": {
                "enabled": True,
                "warm_start": "offline_part_large",
                "share_weights_across_hypotheses": True,
            },
            "fusion": {
                "enabled": True,
                "kind": "bidirectional_dualcross",
                "locations": [4, 8, "preclassification"],
                "blocks_per_location": 2,
                "hierarchy_memory": True,
                "uncertainty_gates": True,
                "rezero_init": 1.0e-3,
                "dual_hierarchy": False,
            },
            "hierarchy_modules": {
                "model_dim": 256,
                "num_heads": 8,
                "ff_dim": 1024,
                "dropout": 0.10,
                "view_aggregator_blocks": 4,
            },
        },
        "training": {
            "schedule": "performance_first_curriculum",
            "initialization": "warm_started",
            "seed_count": 3,
            "optimizer": {
                "name": "adamw",
                "betas": [0.9, 0.95],
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "warmup_fraction": 0.05,
                "schedule": "cosine_to_5pct",
                "ema_decay": 0.9999,
            },
            "objective": {
                "label_ce": 1.0,
                "label_smoothing": 0.02,
                "hlt_anchor_ce": 0.20,
                "joint_reconstruction": 0.10,
                "pseudo_aux_ce": 0.0,
                "hierarchy_aux_ce": 0.0,
                "offline_logit_kd": 0.0,
                "kd_temperature": 2.0,
            },
            "reconstruction_weights": {
                "root": 1.0,
                "group_2": 1.0,
                "group_4": 1.0,
                "group_8": 0.75,
                "group_16": 0.50,
                "group_32": 0.50,
                "topology": 0.50,
                "frontier": 1.0,
                "particle": 1.0,
                "particle_feature": 0.50,
                "distribution": 0.25,
                "calibration": 0.10,
                "auxiliary": 0.25,
            },
        },
        "evaluation": {
            "selection_split": "model_val",
            "fusion_fit_split": "stack_train",
            "fusion_selection_split": "stack_val",
            "allowed_splits": list(ALL_SPLITS),
            "deployable": True,
            "diagnostic_only": False,
            "oracle": False,
            "final_test_eligible": True,
        },
    }


@dataclass(frozen=True)
class AdaptiveBinaryVariantSpec:
    """Declarative patch over the complete locked campaign configuration."""

    run_id: str
    name: str
    tier: str
    title: str
    overrides: Mapping[str, Any]
    dependencies: tuple[str, ...] = ()
    primary: bool = False

    def __post_init__(self) -> None:
        run_id = str(self.run_id).strip()
        name = str(self.name).strip()
        tier = str(self.tier).strip().upper()
        if not run_id or not name or tier not in tuple("ABCDEFG"):
            raise ValueError("variant run_id/name must be non-empty and tier must be A-G")
        if not run_id.upper().startswith(tier):
            raise ValueError(f"variant {run_id} does not belong to tier {tier}")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "overrides", _freeze(dict(self.overrides)))
        object.__setattr__(self, "dependencies", tuple(str(row) for row in self.dependencies))
        object.__setattr__(self, "primary", bool(self.primary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": ABPH_VARIANT_REGISTRY_CONTRACT,
            "run_id": self.run_id,
            "name": self.name,
            "tier": self.tier,
            "title": self.title,
            "dependencies": list(self.dependencies),
            "primary": self.primary,
            "overrides": _thaw(self.overrides),
        }


def _spec(
    run_id: str,
    name: str,
    title: str,
    overrides: Mapping[str, Any],
    *,
    dependencies: tuple[str, ...] = (),
    primary: bool = False,
) -> AdaptiveBinaryVariantSpec:
    return AdaptiveBinaryVariantSpec(
        run_id=run_id,
        name=name,
        tier=run_id[0],
        title=title,
        overrides=_deep_merge(_tier_defaults(run_id[0]), overrides),
        dependencies=dependencies,
        primary=primary,
    )


def _disabled_reconstruction() -> dict[str, Any]:
    return {
        "root_predictor": {"enabled": False},
        "hierarchy": {"enabled": False},
        "distribution": {"enabled": False},
        "renderer": {"enabled": False},
        "pseudo_part": {"enabled": False},
        "fusion": {"enabled": False},
    }


def _tier_defaults(tier: str) -> dict[str, Any]:
    if tier == "B":
        return {
            "data": {"requires_offline_targets": True},
            "model": {
                "hierarchy": {"enabled": False},
                "distribution": {"enabled": False},
                "renderer": {"enabled": False},
                "pseudo_part": {"enabled": False},
                "fusion": {"enabled": False},
            },
        }
    if tier == "C":
        return {
            "data": {"requires_offline_targets": True},
            "model": {
                "distribution": {"enabled": False},
                "renderer": {"enabled": False},
                "pseudo_part": {"enabled": False},
                "fusion": {"enabled": False},
            },
        }
    if tier == "D":
        return {
            "data": {"requires_offline_targets": True},
            "model": {
                "pseudo_part": {"enabled": False},
                "fusion": {"enabled": False},
            },
        }
    if tier in {"E", "F"}:
        return {"data": {"requires_offline_targets": True}}
    return {}


def _registry_rows() -> tuple[AdaptiveBinaryVariantSpec, ...]:
    hlt_only = {
        "data": {"requires_offline_targets": False},
        "model": _disabled_reconstruction(),
        "training": {"objective": {"hlt_anchor_ce": 0.0, "joint_reconstruction": 0.0}},
    }
    oracle_eval = {
        "evaluation": {
            "allowed_splits": list(MODEL_VAL_ONLY),
            "deployable": False,
            "diagnostic_only": True,
            "oracle": True,
            "final_test_eligible": False,
        }
    }
    rows = [
        _spec("A0", "A0_hlt_part", "clean large HLT ParT", hlt_only, primary=True),
        _spec(
            "A0b", "A0b_hlt_part_canonical_base", "canonical base HLT ParT",
            _deep_merge(hlt_only, {"model": {"hlt_part": {"embed_dims": [128, 512, 128], "num_layers": 8, "num_cls_layers": 2}}}),
        ),
        _spec(
            "A1", "A1_hlt_schedule_control", "schedule-matched HLT-only control",
            _deep_merge(hlt_only, {"training": {"schedule": "fused_update_budget_without_pseudo", "initialization": "A0_warm_start"}}),
            dependencies=("A0_hlt_part",),
        ),
        _spec(
            "A2", "A2_hlt_capacity_control", "predeclared deep HLT-only capacity proxy",
            _deep_merge(hlt_only, {"model": {"hlt_part": {
                "embed_dims": [256, 1024, 256],
                "pair_embed_dims": [128, 128, 128],
                "num_heads": 8,
                "num_layers": 20,
                "num_cls_layers": 5,
                "capacity_control_kind": "predeclared_deep_single_stream_proxy",
            }}}),
        ),
        _spec(
            "A3", "A3_hlt_from_scratch", "large HLT ParT from scratch",
            _deep_merge(hlt_only, {"training": {"initialization": "from_scratch"}}),
        ),
        _spec(
            "A4", "A4_offline_part_ceiling", "offline ParT ceiling",
            _deep_merge(hlt_only, {
                "data": {"input_view": "offline", "requires_offline_targets": True},
                "evaluation": {"deployable": False, "final_test_eligible": False},
            }),
        ),
        _spec(
            "A5", "A5_hlt_part_xl", "predeclared XL HLT-only control",
            _deep_merge(hlt_only, {"model": {"hlt_part": {
                "embed_dims": [256, 1024, 256], "pair_embed_dims": [128, 128, 128],
                "num_heads": 8, "num_layers": 16, "num_cls_layers": 4,
            }}}),
        ),
        _spec("B0", "B0_pooled_mlp_root", "pooled MLP root predictor", {
            "model": {"root_predictor": {"kind": "pooled_mlp", "query_blocks": 0}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("A0_hlt_part",)),
        _spec("B1", "B1_semantic_query_root", "deterministic semantic-query root", {
            "model": {"root_predictor": {"kind": "semantic_query_deterministic"}, "distribution": {"enabled": False}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("A0_hlt_part",)),
        _spec("B2", "B2_semantic_query_probabilistic", "probabilistic semantic-query root", {
            "model": {"root_predictor": {"kind": "semantic_query_probabilistic"}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("A0_hlt_part",), primary=True),
        _spec("B3", "B3_root_sampled_ablation", "sampled-root calibration diagnostic", {
            "model": {"distribution": {"enabled": False, "sample_root": True, "shared_compiled_root": False}},
            "evaluation": {
                "diagnostic_only": True,
                "final_test_eligible": False,
                "sampled_root_downstream_rollout": False,
            },
        }, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("B4", "B4_oracle_root_diagnostic", "oracle root diagnostic", _deep_merge({
            "data": {"input_view": "hlt_plus_oracle_root", "requires_offline_targets": True},
            "model": {"root_predictor": {"kind": "oracle_root"}},
        }, oracle_eval)),
        _spec("C0", "C0_direct_8_group_set", "direct eight-group set", {
            "model": {"hierarchy": {"kind": "direct_set", "capacities": [8]}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C1", "C1_kt_2", "exclusive-kT hierarchy capacity 2", {"model": {"hierarchy": {"capacities": [2]}}, "evaluation": {"final_test_eligible": False}}, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C2", "C2_kt_4", "exclusive-kT hierarchy capacity 4", {"model": {"hierarchy": {"capacities": [2, 4]}}, "evaluation": {"final_test_eligible": False}}, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C3", "C3_kt_8", "exclusive-kT hierarchy capacity 8", {"model": {"hierarchy": {"capacities": [2, 4, 8]}}, "evaluation": {"final_test_eligible": False}}, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C4", "C4_kt_16", "exclusive-kT hierarchy capacity 16", {"model": {"hierarchy": {"capacities": [2, 4, 8, 16]}}, "evaluation": {"final_test_eligible": False}}, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C5", "C5_kt_32", "primary exclusive-kT hierarchy capacity 32", {"evaluation": {"final_test_eligible": False}}, dependencies=("B2_semantic_query_probabilistic",), primary=True),
        _spec("C6", "C6_ca_32", "C/A hierarchy capacity 32", {
            "model": {"hierarchy": {"grouping": "cambridge_aachen", "grouping_power": 0}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C7", "C7_shared_level_weights", "shared hierarchy-level weights", {
            "model": {"hierarchy": {"level_specific_weights": False}},
            "evaluation": {"diagnostic_only": True, "final_test_eligible": False},
        }, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C8", "C8_unconstrained_split_control", "unconstrained child-head loss diagnostic", {
            "model": {"hierarchy": {
                "constrained": False,
                "unconstrained_auxiliary_heads": True,
                "deployment_rollout_remains_constrained": True,
            }},
            "evaluation": {"deployable": False, "diagnostic_only": True, "final_test_eligible": False},
        }, dependencies=("B2_semantic_query_probabilistic",)),
        _spec("C9", "C9_oracle_parent_rollout", "oracle-parent rollout diagnostic", _deep_merge({
            "data": {"input_view": "hlt_plus_oracle_parents", "requires_offline_targets": True},
            "model": {"hierarchy": {"oracle_parent_rollout": True}},
        }, oracle_eval), dependencies=("B2_semantic_query_probabilistic",)),
        _spec("D0", "D0_kt32_mean_particles", "deterministic kT32 particle renderer", {
            "model": {"distribution": {"enabled": False, "mean_views": 1, "stochastic_views": 0}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("C5_kt_32",)),
        _spec("D1", "D1_kt32_mh4_particles", "primary kT32 multi-hypothesis renderer", {
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("C5_kt_32",), primary=True),
        _spec("D2", "D2_ca32_mh4_particles", "C/A multi-hypothesis renderer", {
            "model": {"hierarchy": {"grouping": "cambridge_aachen", "grouping_power": 0}},
            "evaluation": {"final_test_eligible": False},
        }, dependencies=("C6_ca_32",)),
        _spec("D3", "D3_global_particle_set", "global particle set decoder", {
            "model": {"renderer": {"kind": "global_set", "local_matching": False}},
            "evaluation": {"diagnostic_only": True, "final_test_eligible": False},
        }, dependencies=("C5_kt_32",)),
        _spec("D4", "D4_no_nbody_projection", "renderer without N-body projection", {
            "model": {"renderer": {"exact_nbody_projection": False}},
            "evaluation": {"deployable": False, "diagnostic_only": True, "final_test_eligible": False},
        }, dependencies=("C5_kt_32",)),
        _spec("D5", "D5_oracle_groups_particles", "oracle L5 group renderer", _deep_merge({
            "data": {"input_view": "hlt_plus_oracle_l5", "requires_offline_targets": True},
            "model": {
                "renderer": {"oracle_groups": True},
                "distribution": {"enabled": False, "stochastic_views": 0},
            },
        }, oracle_eval), dependencies=("C5_kt_32",)),
        _spec("D6", "D6_true_offline_particles", "oracle offline particle-feature injection diagnostic", _deep_merge({
            "data": {"input_view": "hlt_plus_offline_particles", "requires_offline_targets": True},
            "model": {"renderer": {"kind": "oracle_offline_particles"}},
        }, oracle_eval), dependencies=("A4_offline_part_ceiling", "D1_kt32_mh4_particles")),
        _spec("D7", "D7_kt8_mh4_particles_screen", "supplemental kT8 multi-hypothesis renderer", {
            "model": {"hierarchy": {"capacities": [2, 4, 8]}},
            "evaluation": {
                "final_test_eligible": False,
                "supplemental_screen": True,
            },
        }, dependencies=("C3_kt_8",)),
        _spec("E0", "E0_pseudo_only", "pseudo-particle-only ParT", {
            "model": {"hlt_part": {"enabled": False}, "fusion": {"enabled": False}},
        }, dependencies=("D1_kt32_mh4_particles",)),
        _spec("E1", "E1_hlt_pseudo_logit_mean", "untrained HLT/pseudo logit mean", {
            "model": {"fusion": {"kind": "untrained_logit_mean", "locations": []}},
        }, dependencies=("A0_hlt_part", "E0_pseudo_only")),
        _spec("E2", "E2_late_representation_fusion", "late pooled-representation fusion", {
            "model": {"fusion": {"kind": "late_representation", "locations": ["preclassification"]}},
        }),
        _spec("E3", "E3_single_cross_attention", "single cross-attention fusion", {
            "model": {"fusion": {"kind": "single_cross_attention", "locations": [8], "blocks_per_location": 1, "hierarchy_memory": False}},
        }),
        _spec("E4", "E4_hierarchy_memory_fusion", "single cross-attention plus hierarchy memory", {
            "model": {"fusion": {"kind": "single_cross_attention", "locations": [8], "blocks_per_location": 1, "hierarchy_memory": True}},
        }),
        _spec("E5", "E5_kt32_mh4_dualcross", "primary kT32 dual-cross tagger", {}, dependencies=("A0_hlt_part", "D1_kt32_mh4_particles"), primary=True),
        _spec("E6", "E6_ca32_mh4_dualcross", "C/A dual-cross tagger", {
            "model": {"hierarchy": {"grouping": "cambridge_aachen", "grouping_power": 0}},
        }, dependencies=("A0_hlt_part", "D2_ca32_mh4_particles")),
        _spec("E7", "E7_dual_hierarchy_dualcross", "shared-root kT plus C/A dual-hierarchy tagger", {
            "model": {
                "fusion": {"dual_hierarchy": True, "view_types": ["exclusive_kt", "cambridge_aachen"]},
                "root_predictor": {"shared_across_hierarchies": True},
            },
        }, dependencies=("E5_kt32_mh4_dualcross", "E6_ca32_mh4_dualcross"), primary=True),
        _spec("E8", "E8_no_hierarchy_tokens", "tagger without hierarchy tokens", {
            "model": {"fusion": {"hierarchy_memory": False}},
            "evaluation": {"diagnostic_only": True},
        }),
        _spec("E9", "E9_no_uncertainty_gates", "tagger without uncertainty gates", {
            "model": {"fusion": {"uncertainty_gates": False}},
            "evaluation": {"diagnostic_only": True},
        }),
        _spec("E10", "E10_no_baseline_residual_init", "tagger without ReZero baseline initialization", {
            "model": {"fusion": {"rezero_init": None}},
            "evaluation": {"diagnostic_only": True},
        }),
        _spec("E11", "E11_independent_root_dual_hierarchy_diagnostic", "independent-root dual-hierarchy diagnostic", {
            "model": {
                "fusion": {"dual_hierarchy": True, "view_types": ["exclusive_kt", "cambridge_aachen"]},
                "root_predictor": {"shared_across_hierarchies": False},
                "distribution": {"shared_compiled_root": False},
            },
            "evaluation": {"deployable": False, "diagnostic_only": True, "final_test_eligible": False},
        }),
        _spec("E12", "E12_kt8_mh4_dualcross_screen", "supplemental HLT plus kT8 dual-cross tagger", {
            "data": {"pseudo_sources": ["D7_kt8_mh4_particles_screen"]},
            "model": {"hierarchy": {"capacities": [2, 4, 8]}},
            "evaluation": {
                "final_test_eligible": False,
                "supplemental_screen": True,
            },
        }, dependencies=("A0_hlt_part", "D7_kt8_mh4_particles_screen")),
        _spec("F0", "F0_ce_reco_primary", "primary dual-hierarchy CE plus maintained reconstruction", {
            "model": {
                "fusion": {
                    "dual_hierarchy": True,
                    "view_types": ["exclusive_kt", "cambridge_aachen"],
                },
                "root_predictor": {"shared_across_hierarchies": True},
            },
        }, dependencies=("E7_dual_hierarchy_dualcross",), primary=True),
        _spec("F1", "F1_ce_only_frozen_reconstructor", "CE-only frozen reconstructor", {
            "training": {"schedule": "frozen_reconstructor", "objective": {"joint_reconstruction": 0.0}},
        }),
        _spec("F2", "F2_ce_only_joint", "CE-only joint drift diagnostic", {
            "training": {"objective": {"joint_reconstruction": 0.0}},
            "evaluation": {"diagnostic_only": True},
        }),
        _spec("F3", "F3_ce_reco_branch_aux", "primary objective plus branch auxiliary CE", {
            "training": {"objective": {"pseudo_aux_ce": 0.10, "hierarchy_aux_ce": 0.05}},
        }),
        _spec("F4", "F4_ce_logit_kd", "CE plus offline logit KD", {
            "data": {"requires_teacher_logits": True},
            "training": {"objective": {"joint_reconstruction": 0.0, "offline_logit_kd": 0.25}},
        }),
        _spec("F5", "F5_ce_reco_logit_kd", "primary objective plus offline logit KD", {
            "data": {"requires_teacher_logits": True},
            "training": {"objective": {"offline_logit_kd": 0.25}},
        }),
        _spec("F6", "F6_hlt_from_scratch", "HLT branch from scratch", {"training": {"initialization": "hlt_from_scratch"}}),
        _spec("F7", "F7_pseudo_from_scratch", "pseudo branch from scratch", {"training": {"initialization": "pseudo_from_scratch"}}),
        _spec("F8", "F8_all_from_scratch", "both tagger branches from scratch", {"training": {"initialization": "all_tagger_branches_from_scratch"}}),
        _spec("F9", "F9_joint_from_start", "reconstructor/tagger joint from start", {"training": {"schedule": "joint_from_start", "initialization": "joint_initial_states"}}),
        _spec("G0", "G0_kt_hypothesis_aggregator", "learned kT hypothesis aggregator", {
            "model": {"fusion": {"kind": "hypothesis_aggregator"}},
        }, dependencies=("E5_kt32_mh4_dualcross",)),
        _spec("G1", "G1_kt_ca_early_fusion", "kT/C-A early particle and hierarchy fusion", {
            "model": {"fusion": {"dual_hierarchy": True}},
        }, dependencies=("E7_dual_hierarchy_dualcross",)),
        _spec("G2", "G2_kt_ca_logit_fusion", "stack-fitted kT/C-A logit fusion", {
            "model": {"fusion": {"kind": "stack_fitted_logit", "locations": []}},
            "evaluation": {"allowed_splits": list(STACK_EVAL_SPLITS)},
        }, dependencies=("E5_kt32_mh4_dualcross", "E6_ca32_mh4_dualcross")),
        _spec("G3", "G3_particle_and_logit_fusion", "early particle plus complementary logit fusion", {
            "model": {"fusion": {"kind": "particle_and_stack_logit", "dual_hierarchy": True}},
            "evaluation": {"allowed_splits": list(STACK_EVAL_SPLITS)},
        }, dependencies=("E7_dual_hierarchy_dualcross",)),
        _spec("G4", "G4_seed_ensemble_primary", "three-seed primary ensemble", {
            "training": {"seed_count": 3},
            "model": {"fusion": {"kind": "seed_ensemble", "locations": []}},
            "evaluation": {"allowed_splits": list(STACK_EVAL_SPLITS)},
        }, dependencies=("F0_ce_reco_primary",), primary=True),
        _spec("G5", "G5_best_complementary_ensemble", "frozen complementary ensemble", {
            "model": {"fusion": {"kind": "frozen_complementary_ensemble", "locations": []}},
            "evaluation": {"allowed_splits": list(STACK_EVAL_SPLITS)},
        }, dependencies=("E5_kt32_mh4_dualcross", "E6_ca32_mh4_dualcross", "E7_dual_hierarchy_dualcross"), primary=True),
    ]
    return tuple(rows)


ABPH_VARIANT_REGISTRY: Mapping[str, AdaptiveBinaryVariantSpec] = MappingProxyType(
    {row.name: row for row in _registry_rows()}
)
ABPH_EXPECTED_VARIANT_NAMES: tuple[str, ...] = tuple(ABPH_VARIANT_REGISTRY)
ABPH_NON_GATING_VARIANT_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "D5_oracle_groups_particles": (
            "real JetClass L5 oracle ledgers can have fixed PID minimum-mass "
            "budgets above their measured invariant masses, so no exact N-body "
            "renderer can satisfy the diagnostic's simultaneous oracle constraints"
        ),
        "D7_kt8_mh4_particles_screen": (
            "supplemental early-signal renderer attached to the completed C3 kT8 "
            "checkpoint; it is not part of the canonical A0-G5 claim matrix"
        ),
        "E12_kt8_mh4_dualcross_screen": (
            "supplemental early-signal tagger for the C3-derived kT8 renderer; "
            "it is not part of the canonical A0-G5 claim matrix"
        ),
    }
)
ABPH_REQUIRED_CAMPAIGN_VARIANT_NAMES: tuple[str, ...] = tuple(
    name
    for name in ABPH_EXPECTED_VARIANT_NAMES
    if name not in ABPH_NON_GATING_VARIANT_REASONS
)


def _validate_registry_specs() -> None:
    rows = tuple(ABPH_VARIANT_REGISTRY.values())
    run_ids = tuple(row.run_id for row in rows)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("ABPH registry contains duplicate run IDs")
    known = set(ABPH_VARIANT_REGISTRY)
    for row in rows:
        missing = [name for name in row.dependencies if name not in known]
        if missing:
            raise ValueError(f"variant {row.name} has unknown dependencies {missing}")


_validate_registry_specs()

ABPH_VARIANT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        **{row.run_id.lower(): row.name for row in ABPH_VARIANT_REGISTRY.values()},
        **{row.name.lower(): row.name for row in ABPH_VARIANT_REGISTRY.values()},
        **{row.name.lower().replace("_", "-"): row.name for row in ABPH_VARIANT_REGISTRY.values()},
    }
)


def normalize_variant_name(value: str) -> str:
    key = str(value).strip().lower()
    normalized = ABPH_VARIANT_ALIASES.get(key) or ABPH_VARIANT_ALIASES.get(key.replace("_", "-"))
    if normalized is None:
        raise ValueError(
            f"unknown ABPH variant {value!r}; expected a short ID or one of {ABPH_EXPECTED_VARIANT_NAMES}"
        )
    return normalized


def variant_spec(value: str) -> AdaptiveBinaryVariantSpec:
    return ABPH_VARIANT_REGISTRY[normalize_variant_name(value)]


def _validate_resolved_config(payload: Mapping[str, Any]) -> None:
    required_sections = ("contract", "data", "schemas", "model", "training", "evaluation", "variant")
    missing = [key for key in required_sections if key not in payload]
    if missing:
        raise ValueError(f"resolved ABPH configuration lacks sections {missing}")
    model = payload["model"]
    required_modules = (
        "hlt_part", "root_predictor", "hierarchy", "distribution", "renderer",
        "pseudo_part", "fusion", "hierarchy_modules",
    )
    missing_modules = [key for key in required_modules if key not in model]
    if missing_modules:
        raise ValueError(f"resolved ABPH model configuration lacks modules {missing_modules}")
    evaluation = payload["evaluation"]
    if bool(evaluation.get("oracle")) and bool(evaluation.get("final_test_eligible")):
        raise ValueError("oracle variants cannot be final-test eligible")
    if payload["data"].get("requires_teacher_logits") and not payload["data"].get("final_test_teacher_free"):
        raise ValueError("KD variants must remain teacher-free at final-test prediction")


def resolve_variant_config(value: str) -> dict[str, Any]:
    """Resolve a short/full variant name to a complete, deterministic config."""

    spec = variant_spec(value)
    payload = _deep_merge(_base_config(), spec.overrides)
    if spec.tier in {"E", "G"}:
        # E-tier and neural G0/G1 runs consume a selected frozen
        # reconstructor. Maintained reconstruction is introduced only by the
        # explicit F-tier joint-training recipes.
        payload["training"]["objective"]["joint_reconstruction"] = 0.0
    if spec.run_id == "E0":
        payload["training"]["objective"]["hlt_anchor_ce"] = 0.0
    payload["variant"] = {
        "contract": ABPH_VARIANT_REGISTRY_CONTRACT,
        "run_id": spec.run_id,
        "name": spec.name,
        "tier": spec.tier,
        "title": spec.title,
        "dependencies": list(spec.dependencies),
        "primary": spec.primary,
    }
    _validate_resolved_config(payload)
    payload["resolved_config_hash"] = canonical_hash(payload)
    return payload


def registry_manifest() -> dict[str, Any]:
    variants = {name: resolve_variant_config(name) for name in ABPH_EXPECTED_VARIANT_NAMES}
    payload = {
        "contract": ABPH_VARIANT_REGISTRY_CONTRACT,
        "variant_names": list(ABPH_EXPECTED_VARIANT_NAMES),
        "variants": variants,
    }
    payload["registry_hash"] = canonical_hash(payload)
    return payload


__all__ = [
    "ABPH_EXPECTED_VARIANT_NAMES",
    "ABPH_NON_GATING_VARIANT_REASONS",
    "ABPH_REQUIRED_CAMPAIGN_VARIANT_NAMES",
    "ABPH_VARIANT_ALIASES",
    "ABPH_VARIANT_REGISTRY",
    "ABPH_VARIANT_REGISTRY_CONTRACT",
    "AdaptiveBinaryVariantSpec",
    "normalize_variant_name",
    "registry_manifest",
    "resolve_variant_config",
    "variant_spec",
]
