"""Exact graph-specific HOSD capacity controls and analytical FLOP ledgers."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Sequence

from jetclass_fresh.heterogeneous_hlt import ParticleNetHLTClassifier
from teacher_logit_reco.relational_part.model import (
    RelationalParticleTransformer,
    exact_rpt_base_config,
)

from .baselines import particle_net_config
from .contracts import (
    CAPACITY_CONTROL_COMPILATION_CONTRACT,
    CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    CAPACITY_CONTROL_RESULT_CONTRACT,
    CONFIRMATION_PLAN_CONTRACT,
    CAPACITY_GRID_CONTRACT,
    CAPACITY_PROFILE_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


MONOLITHIC_WIDTHS = (96, 112, 128, 144, 160, 192)
MONOLITHIC_PARTICLE_BLOCKS = (6, 8, 10, 12)
MONOLITHIC_CLASS_BLOCKS = (1, 2, 3)
MONOLITHIC_HEADS = (4, 8)
PARTICLENET_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


def _round_multiple_of_eight(value: float) -> int:
    # Widths never land at an exact half-multiple for the registered source
    # grid, but define half-up explicitly so platform rounding cannot matter.
    return max(8, 8 * int(value / 8.0 + 0.5))


def monolithic_grid() -> list[dict[str, int]]:
    return [
        {
            "embed_dim": width,
            "particle_blocks": particle_blocks,
            "class_blocks": class_blocks,
            "attention_heads": heads,
        }
        for width in MONOLITHIC_WIDTHS
        for particle_blocks in MONOLITHIC_PARTICLE_BLOCKS
        for class_blocks in MONOLITHIC_CLASS_BLOCKS
        for heads in MONOLITHIC_HEADS
        if width % heads == 0
    ]


def monolithic_config(row: Mapping[str, Any]) -> dict[str, Any]:
    width = int(row["embed_dim"])
    heads = int(row["attention_heads"])
    particle_blocks = int(row["particle_blocks"])
    class_blocks = int(row["class_blocks"])
    expected = {
        (
            item["embed_dim"],
            item["particle_blocks"],
            item["class_blocks"],
            item["attention_heads"],
        )
        for item in monolithic_grid()
    }
    if (width, particle_blocks, class_blocks, heads) not in expected:
        raise ValueError("monolithic capacity configuration is outside the grid")
    config = exact_rpt_base_config()
    config.update(
        {
            "embed_dims": [width, 4 * width, width],
            "pair_embed_dims": [width // 2, width // 2, width // 2],
            "num_heads": heads,
            "num_layers": particle_blocks,
            "num_cls_layers": class_blocks,
        }
    )
    return config


class HOSDMonolithicParticleTransformer(RelationalParticleTransformer):
    """Standard-four HLT ParT differing only by the registered capacity grid."""

    def __init__(
        self, row: Mapping[str, Any], *, weaver_module: Any | None = None
    ) -> None:
        self.capacity_configuration = dict(row)
        super().__init__(
            config=monolithic_config(row),
            weaver_module=weaver_module,
            allow_registered_capacity_config=True,
        )


def particle_net_scaled_config(multiplier: float) -> dict[str, Any]:
    if float(multiplier) not in PARTICLENET_MULTIPLIERS:
        raise ValueError("ParticleNet multiplier is outside the registered grid")
    base = particle_net_config()
    base.pop("architecture")
    result = copy.deepcopy(base)
    result["conv_params"] = [
        (
            int(neighbors),
            tuple(
                _round_multiple_of_eight(float(width) * float(multiplier))
                for width in widths
            ),
        )
        for neighbors, widths in base["conv_params"]
    ]
    result["fc_params"] = [
        (_round_multiple_of_eight(float(width) * float(multiplier)), dropout)
        for width, dropout in base["fc_params"]
    ]
    return result


def build_capacity_model(
    kind: str,
    configuration: Mapping[str, Any],
    *,
    weaver_module: Any | None = None,
) -> Any:
    if kind == "MONOLITHIC":
        return HOSDMonolithicParticleTransformer(
            configuration, weaver_module=weaver_module
        )
    if kind == "PARTICLENET":
        return ParticleNetHLTClassifier(
            **particle_net_scaled_config(float(configuration["multiplier"]))
        )
    raise ValueError("unknown HOSD capacity model kind")


def exact_trainable_parameter_count(model: Any) -> int:
    count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if bool(parameter.requires_grad)
    )
    if count <= 0:
        raise ValueError("capacity model has no trainable parameters")
    return count


def global_target_head_flops(
    head: Any, *, particles: int = 128
) -> int:
    """Analytical batch-one FLOPs for the registered four-query global head."""

    queries = 4
    width = 128
    n = int(particles)
    if n <= 0:
        raise ValueError("global-head particle count must be positive")
    input_projection = getattr(head, "input_projection", None)
    input_projection_flops = (
        0
        if input_projection is None
        or input_projection.__class__.__name__ == "Identity"
        else 2
        * n
        * int(input_projection.in_features)
        * int(input_projection.out_features)
    )
    output = head.output
    availability = head.availability
    return int(
        input_projection_flops
        # Q, K, V, and output projections of multi-head attention.
        + 2 * width * width * (queries + 2 * n + queries)
        # Query-key scores and weighted-value accumulation.
        + 4 * queries * n * width
        + 2 * 512 * 256
        + 2 * 256 * int(output.out_features)
        + 2 * 256 * int(availability.out_features)
    )


def combination_model_flop_ledger(
    model: Any,
    *,
    particles: int = 128,
    classes: int = 10,
) -> dict[str, Any]:
    """Exact analytical ledger for deployed H_BASE plus training-only heads."""

    base = monolithic_flop_ledger(
        {
            "embed_dim": 128,
            "attention_heads": 8,
            "particle_blocks": 8,
            "class_blocks": 2,
        },
        particles=particles,
        classes=classes,
    )
    head_terms = {
        str(key): global_target_head_flops(head, particles=particles)
        for key, head in sorted(model.heads.items())
    }
    if getattr(model, "native_relation_head", None) is not None:
        head_terms["native_relation"] = global_target_head_flops(
            model.native_relation_head, particles=particles
        )
    return {
        "contract": "hosd_analytical_combination_flops_v1",
        "batch_size": 1,
        "valid_particles": int(particles),
        "classes": int(classes),
        "multiply_add_flops": 2,
        "deployed_hlt_only_flops": int(base["total_flops"]),
        "training_only_head_terms": head_terms,
        "training_total_flops": int(
            base["total_flops"] + sum(head_terms.values())
        ),
        "deployed_total_flops": int(base["total_flops"]),
        "heads_discarded_at_deployment": True,
    }


def feedback_model_flop_ledger(
    model: Any,
    *,
    particles: int = 128,
    classes: int = 10,
) -> dict[str, Any]:
    """Analytical batch-one ledger for a deployed Stage-E feedback graph."""

    n = int(particles)
    if n <= 0:
        raise ValueError("feedback FLOP ledger requires positive particles")
    base = monolithic_flop_ledger(
        {
            "embed_dim": 128,
            "attention_heads": 8,
            "particle_blocks": 8,
            "class_blocks": 2,
        },
        particles=n,
        classes=classes,
    )
    interface = str(model.interface)
    dimension = int(
        model.consumer.bias_network[0].in_features
        if interface == "FB_PAIR"
        else model.global_predictor.target_dimension
        if hasattr(model.global_predictor, "target_dimension")
        else 512
    )
    terms: dict[str, int] = {}
    if interface == "FB_PAIR":
        if model.consumer.predictor is not None:
            pair_input_factor = 3 if model.consumer.symmetric else 4
            terms["pair_predictor"] = int(
                2 * n * n * (pair_input_factor * 128) * 128
                + 2 * n * n * 128 * dimension
            )
        terms["pair_bias_network"] = int(
            2 * n * n * dimension * 128
            + 2 * n * n * 128 * 8
        )
    else:
        if hasattr(model.global_predictor, "target_dimension"):
            terms["global_predictor"] = global_target_head_flops(
                model.global_predictor, particles=n
            )
            source_dimension = int(
                model.consumer.structure_projection.in_features
                if hasattr(model.consumer, "structure_projection")
                else model.consumer.projection.in_features
            )
        else:
            # DirectFourTokenHead: four-query attention plus 128->128 output.
            terms["unrestricted_predictor"] = int(
                2 * 128 * 128 * (4 + 2 * n + 4)
                + 4 * 4 * n * 128
                + 2 * 4 * 128 * 128
            )
            source_dimension = 512
        if interface == "FB_TOKEN":
            if hasattr(model.consumer, "structure_projection"):
                terms["structure_projection"] = int(
                    2 * source_dimension * 4 * 128
                )
            terms["token_cross_attention"] = int(
                2 * 128 * 128 * (n + 2 * 4 + n)
                + 4 * n * 4 * 128
            )
        elif interface == "FB_FILM":
            film_source_dimension = (
                4 * 128
                if not hasattr(model.global_predictor, "target_dimension")
                else source_dimension
            )
            terms["film_projection"] = int(
                2 * film_source_dimension * 2 * 128
            )
            terms["film_application_blocks_5_to_8"] = int(4 * 2 * n * 128)
        else:
            raise ValueError("unknown feedback interface in FLOP ledger")
    builder_profile = None
    exact_builder = getattr(model, "exact_pair_builder", None)
    if exact_builder is not None:
        directed_pairs = n * max(0, n - 1)
        unordered_pairs = n * max(0, n - 1) // 2
        target_id = str(exact_builder.target_id)
        if target_id == "T_HLT_TRACK_PAIR_13":
            operations = {
                "validity_and_sentinel_tests": 10 * n,
                "directed_pair_coordinate_evaluations": 13 * directed_pairs,
                "uncertainty_floor_applications": 4 * n,
                "normalization_component_applications": dimension * directed_pairs,
            }
            tree_policy = "not_applicable"
        elif target_id == "T_HLT_REGION_PAIR_8":
            operations = {
                "ca_candidate_distance_evaluations_upper_bound": (
                    n * (n + 1) * max(0, n - 1) // 6
                ),
                "ca_merge_operations": max(0, n - 1),
                "lca_pair_queries": unordered_pairs,
                "pair_coordinate_evaluations": dimension * unordered_pairs,
                "normalization_component_applications": dimension * unordered_pairs,
            }
            tree_policy = (
                "reuse_authenticated_same_event_tree_when_bound;"
                "otherwise_reconstruct_deterministic_ca_tree_once_per_event"
            )
        else:  # pragma: no cover - constructor already restricts this branch
            raise ValueError("unknown exact-HLT builder target")
        builder_profile = with_content_hash(
            {
                "contract": "hosd_exact_hlt_builder_operation_ledger_v1",
                "schema_version": 1,
                "target_id": target_id,
                "valid_particles_upper_bound": n,
                "pair_domain": (
                    "directed" if target_id == "T_HLT_TRACK_PAIR_13" else "unordered"
                ),
                "operation_counts": operations,
                "tree_reuse_policy": tree_policy,
                "normalization_cost_included": True,
                "builder_operations_excluded_from_multiply_add_flops": True,
                "measured_timing_evidence_contract": (
                    "hosd_exact_hlt_builder_timing_v1"
                ),
                "measured_timing_evidence_required_before_production": True,
            }
        )
    return {
        "contract": "hosd_analytical_feedback_flops_v2",
        "batch_size": 1,
        "valid_particles": n,
        "classes": int(classes),
        "multiply_add_flops": 2,
        "base_hlt_part": int(base["total_flops"]),
        "feedback_terms": terms,
        "deployed_total_flops": int(base["total_flops"] + sum(terms.values())),
        "training_only_terms": {},
        "all_feedback_modules_retained_at_deployment": True,
        "exact_hlt_builder_profile": builder_profile,
    }


def auxiliary_model_flop_ledger(
    model: Any,
    *,
    particles: int = 128,
    classes: int = 10,
    sampled_pairs_per_event: int = 1_024,
) -> dict[str, Any]:
    """Analytical training/evaluation forward FLOPs for one auxiliary graph."""

    n = int(particles)
    base = monolithic_flop_ledger(
        {
            "embed_dim": 128,
            "attention_heads": 8,
            "particle_blocks": 8,
            "class_blocks": 2,
        },
        particles=n,
        classes=classes,
    )["total_flops"]
    head = getattr(model, "target_head", None)
    if head is None:
        training_head = evaluation_head = 0
        head_kind = "none"
    elif isinstance(head, GlobalTargetHead):
        training_head = evaluation_head = global_target_head_flops(
            head, particles=n
        )
        head_kind = "global"
    elif isinstance(head, PairTargetHead):
        factor = 3 if bool(head.symmetric) else 4
        target_dimension = int(head.network[-1].out_features)
        per_pair = (
            2 * factor * 128 * 128
            + 2 * 128 * target_dimension
        )
        training_head = per_pair * min(
            int(sampled_pairs_per_event), n * max(0, n - 1)
        )
        evaluation_head = per_pair * n * n
        head_kind = "pair"
    else:
        raise TypeError("auxiliary FLOP ledger encountered an unknown head")
    return {
        "contract": "hosd_analytical_auxiliary_training_flops_v1",
        "batch_size": 1,
        "valid_particles": n,
        "classes": int(classes),
        "head_kind": head_kind,
        "sampled_pairs_per_training_event": (
            min(int(sampled_pairs_per_event), n * max(0, n - 1))
            if head_kind == "pair"
            else None
        ),
        "deployed_forward_flops": int(base),
        "training_forward_flops": int(base + training_head),
        "evaluation_forward_flops": int(base + evaluation_head),
        "training_head_flops": int(training_head),
        "evaluation_head_flops": int(evaluation_head),
    }


def monolithic_flop_ledger(
    row: Mapping[str, Any], *, particles: int = 128, classes: int = 10
) -> dict[str, Any]:
    """Count multiply-adds as two FLOPs for batch one and N valid particles."""

    config = monolithic_config(row)
    width = int(row["embed_dim"])
    heads = int(row["attention_heads"])
    particle_blocks = int(row["particle_blocks"])
    class_blocks = int(row["class_blocks"])
    n = int(particles)
    pair_width = width // 2
    symmetric_pairs = n * (n + 1) // 2
    terms = {
        "input_encoder": 2 * n * (17 * width + width * 4 * width + 4 * width * width),
        "pair_encoder": 2
        * symmetric_pairs
        * (
            4 * pair_width
            + pair_width * pair_width
            + pair_width * pair_width
            + pair_width * heads
        ),
        "particle_attention_projections": particle_blocks
        * 8
        * n
        * width
        * width,
        "particle_attention_scores_and_values": particle_blocks
        * 4
        * n
        * n
        * width,
        "particle_ffn": particle_blocks * 16 * n * width * width,
        "class_attention_projections": class_blocks
        * 8
        * (n + 1)
        * width
        * width,
        "class_attention_scores_and_values": class_blocks * 4 * n * width,
        "class_ffn": class_blocks * 16 * width * width,
        "classifier": 2 * width * int(classes),
    }
    total = sum(terms.values())
    return {
        "contract": "hosd_analytical_part_flops_v1",
        "batch_size": 1,
        "valid_particles": n,
        "classes": int(classes),
        "multiply_add_flops": 2,
        "configuration": dict(row),
        "resolved_model_config": config,
        "terms": terms,
        "excluded_nonmultiply_operations": [
            "normalization",
            "activation",
            "softmax",
            "masking",
            "trimming",
            "relation_or_tree_construction",
        ],
        "total_flops": int(total),
    }


def particle_net_flop_ledger(
    multiplier: float, *, particles: int = 128, classes: int = 10
) -> dict[str, Any]:
    config = particle_net_scaled_config(multiplier)
    n = int(particles)
    terms: dict[str, int] = {}
    input_width = int(config["input_dims"])
    for stage, (neighbors, widths) in enumerate(config["conv_params"]):
        edge_count = n * int(neighbors)
        stage_terms = 0
        prior = 2 * input_width
        for width in widths:
            stage_terms += 2 * edge_count * prior * int(width)
            prior = int(width)
        output_width = int(widths[-1])
        if input_width != output_width:
            stage_terms += 2 * n * input_width * output_width
        terms[f"edgeconv_stage_{stage}"] = stage_terms
        input_width = output_width
    fc_width = int(config["fc_params"][0][0])
    terms["global_mean"] = 0
    terms["fc"] = 2 * input_width * fc_width
    terms["classifier"] = 2 * fc_width * int(classes)
    return {
        "contract": "hosd_analytical_particlenet_flops_v1",
        "batch_size": 1,
        "valid_particles": n,
        "classes": int(classes),
        "multiply_add_flops": 2,
        "configuration": {"multiplier": float(multiplier)},
        "resolved_model_config": config,
        "terms": terms,
        "excluded_nonmultiply_operations": [
            "knn_search_and_distance",
            "normalization",
            "activation",
            "pooling",
            "masking",
        ],
        "total_flops": int(sum(terms.values())),
    }


def build_capacity_grid_artifact(
    *,
    source: Mapping[str, Any],
    model_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Instantiate every registered candidate and bind exact active counts."""

    factory = build_capacity_model if model_factory is None else model_factory
    rows = []
    for configuration in monolithic_grid():
        model = factory("MONOLITHIC", configuration)
        ledger = monolithic_flop_ledger(configuration)
        rows.append(
            {
                "kind": "MONOLITHIC",
                "configuration": configuration,
                "config_hash": canonical_sha256(configuration),
                "trainable_parameter_count": exact_trainable_parameter_count(model),
                "analytical_flops_batch1_n128": ledger["total_flops"],
                "flop_ledger": ledger,
            }
        )
    for multiplier in PARTICLENET_MULTIPLIERS:
        configuration = {"multiplier": multiplier}
        model = factory("PARTICLENET", configuration)
        ledger = particle_net_flop_ledger(multiplier)
        rows.append(
            {
                "kind": "PARTICLENET",
                "configuration": configuration,
                "config_hash": canonical_sha256(configuration),
                "trainable_parameter_count": exact_trainable_parameter_count(model),
                "analytical_flops_batch1_n128": ledger["total_flops"],
                "flop_ledger": ledger,
            }
        )
    return with_content_hash(
        {
            "contract": CAPACITY_GRID_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "monolithic_grid": {
                "embed_dim": list(MONOLITHIC_WIDTHS),
                "particle_blocks": list(MONOLITHIC_PARTICLE_BLOCKS),
                "class_blocks": list(MONOLITHIC_CLASS_BLOCKS),
                "attention_heads": list(MONOLITHIC_HEADS),
                "candidate_count": len(monolithic_grid()),
            },
            "particle_net_multipliers": list(PARTICLENET_MULTIPLIERS),
            "rows": rows,
            "row_count": len(rows),
            "parameter_count_method": "instantiated_requires_grad_numel",
            "flop_domain": "batch1_128_valid_particles_10_classes_mac_is_2",
            "performance_read": False,
        }
    )


def build_graph_capacity_profile(
    *,
    graph_id: str,
    deployed_parameter_count: int,
    deployed_analytical_flops: int,
    export_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if int(deployed_parameter_count) <= 0 or int(deployed_analytical_flops) <= 0:
        raise ValueError("graph capacity values must be positive")
    return with_content_hash(
        {
            "contract": CAPACITY_PROFILE_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "graph_id": str(graph_id),
            "deployable_export_sha256": require_sha256(
                export_sha256, name="export_sha256"
            ),
            "deployed_trainable_parameter_count": int(deployed_parameter_count),
            "deployed_analytical_flops_batch1_n128": int(
                deployed_analytical_flops
            ),
            "auxiliary_heads_removed_before_measurement": True,
        }
    )


def compile_graph_capacity_controls(
    *,
    graph_profile: Mapping[str, Any],
    grid: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(graph_profile, expected_contract=CAPACITY_PROFILE_CONTRACT)
    validate_content_hash(grid, expected_contract=CAPACITY_GRID_CONTRACT)
    if graph_profile.get("source") != dict(source) or grid.get("source") != dict(source):
        raise ValueError("capacity inputs are bound to a different source")
    target_parameters = int(graph_profile["deployed_trainable_parameter_count"])
    target_flops = int(graph_profile["deployed_analytical_flops_batch1_n128"])
    mono = [row for row in grid["rows"] if row["kind"] == "MONOLITHIC"]
    particle_net = [row for row in grid["rows"] if row["kind"] == "PARTICLENET"]

    def mono_sum(row: Mapping[str, Any]) -> int:
        cfg = row["configuration"]
        return (
            int(cfg["embed_dim"])
            + int(cfg["particle_blocks"])
            + int(cfg["class_blocks"])
            + int(cfg["attention_heads"])
        )

    param = min(
        mono,
        key=lambda row: (
            abs(int(row["trainable_parameter_count"]) - target_parameters),
            abs(int(row["analytical_flops_batch1_n128"]) - target_flops),
            mono_sum(row),
            (
                int(row["configuration"]["embed_dim"]),
                int(row["configuration"]["particle_blocks"]),
                int(row["configuration"]["class_blocks"]),
                int(row["configuration"]["attention_heads"]),
            ),
        ),
    )
    flop = min(
        mono,
        key=lambda row: (
            abs(int(row["analytical_flops_batch1_n128"]) - target_flops),
            abs(int(row["trainable_parameter_count"]) - target_parameters),
            mono_sum(row),
            (
                int(row["configuration"]["embed_dim"]),
                int(row["configuration"]["particle_blocks"]),
                int(row["configuration"]["class_blocks"]),
                int(row["configuration"]["attention_heads"]),
            ),
        ),
    )
    pn = min(
        particle_net,
        key=lambda row: (
            abs(int(row["trainable_parameter_count"]) - target_parameters),
            abs(int(row["analytical_flops_batch1_n128"]) - target_flops),
            float(row["configuration"]["multiplier"]),
            row["config_hash"],
        ),
    )
    graph_hash = canonical_sha256(str(graph_profile["graph_id"]))[:16]
    selected = {
        f"H_MONO_PARAM_{graph_hash}": param,
        f"H_MONO_FLOP_{graph_hash}": flop,
        f"H_PARTICLENET_PARAM_{graph_hash}": pn,
    }
    return with_content_hash(
        {
            "contract": CAPACITY_CONTROL_COMPILATION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "graph_id": graph_profile["graph_id"],
            "graph_profile_sha256": graph_profile["content_hash"],
            "capacity_grid_sha256": grid["content_hash"],
            "selected_controls": selected,
            "selection_trace": {
                "H_MONO_PARAM": [
                    "absolute_parameter_mismatch",
                    "absolute_flop_mismatch",
                    "smaller_embed_plus_particle_plus_class_plus_heads",
                    "lexicographically_smaller_embed_particle_class_heads",
                ],
                "H_MONO_FLOP": [
                    "absolute_flop_mismatch",
                    "absolute_parameter_mismatch",
                    "smaller_embed_plus_particle_plus_class_plus_heads",
                    "lexicographically_smaller_embed_particle_class_heads",
                ],
                "H_PARTICLENET_PARAM": [
                    "absolute_parameter_mismatch",
                    "absolute_flop_mismatch",
                    "smaller_multiplier",
                    "lexicographically_smaller_config_hash",
                ],
            },
            "performance_read": False,
        }
    )


def build_capacity_control_execution_plan(
    *,
    confirmation_plan: Mapping[str, Any],
    compilations: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        confirmation_plan, expected_contract=CONFIRMATION_PLAN_CONTRACT
    )
    by_graph = {}
    for compilation in compilations:
        validate_content_hash(
            compilation, expected_contract=CAPACITY_CONTROL_COMPILATION_CONTRACT
        )
        if compilation.get("source") != dict(source):
            raise ValueError("capacity compilation source differs")
        graph_id = str(compilation["graph_id"])
        if graph_id in by_graph:
            raise ValueError("capacity compilation graph is duplicated")
        by_graph[graph_id] = compilation
    expected_graphs = {
        str(row["parent_graph_id"])
        for row in confirmation_plan["capacity_control_rows"]
    }
    if set(by_graph) != expected_graphs:
        raise ValueError("capacity compilation graph coverage differs")
    prefix = {
        "H_MONO_PARAM": "H_MONO_PARAM_",
        "H_MONO_FLOP": "H_MONO_FLOP_",
        "H_PARTICLENET_PARAM": "H_PARTICLENET_PARAM_",
    }
    rows = []
    for row in confirmation_plan["capacity_control_rows"]:
        compilation = by_graph[str(row["parent_graph_id"])]
        matches = [
            (control_id, definition)
            for control_id, definition in compilation[
                "selected_controls"
            ].items()
            if control_id.startswith(prefix[str(row["control_kind"])])
        ]
        if len(matches) != 1:
            raise ValueError("capacity selected-control kind coverage differs")
        control_id, definition = matches[0]
        rows.append(
            {
                **dict(row),
                "control_graph_id": control_id,
                "control_definition": dict(definition),
                "capacity_compilation_sha256": compilation["content_hash"],
                "fixed_budget": True,
                "performance_can_cancel": False,
            }
        )
    return with_content_hash(
        {
            "contract": CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "confirmation_plan_sha256": confirmation_plan["content_hash"],
            "compilation_hashes": {
                key: by_graph[key]["content_hash"] for key in sorted(by_graph)
            },
            "rows": rows,
            "row_count": len(rows),
            "performance_read_during_compilation": False,
            "all_controls_executable": True,
        }
    )


def build_capacity_control_result(
    *,
    execution_plan: Mapping[str, Any],
    row_id: str,
    classification_metrics: Mapping[str, Any],
    checkpoint_sha256: str,
    prediction_sha256: str,
    deployable_export_sha256: str,
    deployable_export_file: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_plan,
        expected_contract=CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    )
    rows = {row["row_id"]: row for row in execution_plan["rows"]}
    if row_id not in rows:
        raise ValueError("capacity result row is absent")
    if not str(deployable_export_file).strip():
        raise ValueError("capacity result lacks its deployable export file")
    return with_content_hash(
        {
            "contract": CAPACITY_CONTROL_RESULT_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "capacity_execution_plan_sha256": execution_plan["content_hash"],
            **rows[row_id],
            "classification_metrics": dict(classification_metrics),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "prediction_sha256": require_sha256(
                prediction_sha256, name="prediction_sha256"
            ),
            "deployable_export_sha256": require_sha256(
                deployable_export_sha256,
                name="deployable_export_sha256",
            ),
            "deployable_export_file": str(deployable_export_file),
            "completed": True,
        }
    )


__all__ = [
    "auxiliary_model_flop_ledger",
    "HOSDMonolithicParticleTransformer",
    "MONOLITHIC_WIDTHS",
    "PARTICLENET_MULTIPLIERS",
    "build_capacity_grid_artifact",
    "build_capacity_model",
    "build_graph_capacity_profile",
    "compile_graph_capacity_controls",
    "build_capacity_control_execution_plan",
    "build_capacity_control_result",
    "exact_trainable_parameter_count",
    "feedback_model_flop_ledger",
    "monolithic_config",
    "monolithic_flop_ledger",
    "monolithic_grid",
    "particle_net_flop_ledger",
    "particle_net_scaled_config",
]
