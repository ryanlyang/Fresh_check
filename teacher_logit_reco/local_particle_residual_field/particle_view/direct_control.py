"""Predeclared Stage-A capacity/FLOP-matched direct HLT controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import (
    build_particle_transformer_classifier,
    default_part_config,
    require_torch,
    resolve_device,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .consumer import ParticleViewConsumer, ParticleViewConsumerConfig
from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .controls import DirectControlCandidate, select_direct_resource_control
from .offline_teacher import (
    build_predeclared_direct_control_grid,
    select_teacher_checkpoint,
    teacher_learning_rate,
)
from .predictor import (
    PVA3_CANONICAL_ARCHITECTURE,
    build_canonical_particle_view_predictor,
    count_unique_parameters,
    flop_fixture_sha256,
    predictor_semantic_flops,
)
from .splits import (
    PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    logical_split_binding,
)
from .teacher_train import evaluate_particle_view_teacher


PARTICLE_VIEW_STAGE_A_RESOURCE_PLAN_CONTRACT = (
    "particle_view_stage_a_direct_resource_plan_v1"
)
PARTICLE_VIEW_DIRECT_CONTROL_RECIPE_CONTRACT = (
    "particle_view_direct_control_recipe_v1"
)
PARTICLE_VIEW_DIRECT_CONTROL_CHECKPOINT_CONTRACT = (
    "particle_view_direct_control_checkpoint_v1"
)
PARTICLE_VIEW_DIRECT_CONTROL_REGISTRATION_CONTRACT = (
    "particle_view_direct_control_registration_v1"
)
PARTICLE_VIEW_DIRECT_CONTROL_CURVES_CONTRACT = (
    "particle_view_direct_control_curves_v1"
)

STAGE_A_DIRECT_CONTROL_RUNS = {
    "STAGE_A_PARAMETER_MATCH": "parameters",
    "STAGE_A_FLOP_MATCH": "flops",
}


def _linear_flops(rows: int, input_dim: int, output_dim: int) -> int:
    return 2 * rows * input_dim * output_dim + rows * output_dim


def _layer_norm_flops(rows: int, width: int) -> int:
    return rows * (7 * width + 2)


def particle_transformer_semantic_flops(
    config: Mapping[str, Any],
    *,
    particles: int = 128,
) -> dict[str, int]:
    """Versioned unfused semantic count for a direct Weaver ParT.

    Reshape, transpose, masking memory movement, and disk preprocessing are
    zero-cost. Learned projections, attention products, softmax, residuals,
    layer normalization, pair arithmetic, and classification are included.
    """

    if particles != 128:
        raise ValueError("direct-control FLOPs use exactly 128 particles")
    width = int(config["embed_dims"][-1])
    heads = int(config["num_heads"])
    layers = int(config["num_layers"])
    cls_layers = int(config["num_cls_layers"])
    if width % heads:
        raise ValueError("direct-control width must divide num_heads")
    breakdown: dict[str, int] = {}

    def add(name: str, value: int) -> None:
        breakdown[name] = breakdown.get(name, 0) + int(value)

    previous = int(config["input_dim"])
    for output in config["embed_dims"]:
        output = int(output)
        add("particle_embedding_linear", _linear_flops(particles, previous, output))
        add("particle_embedding_layer_norm", _layer_norm_flops(particles, output))
        add("particle_embedding_gelu", 5 * particles * output)
        previous = output
    pairs = particles * particles
    previous = int(config["pair_input_dim"])
    for output in config["pair_embed_dims"]:
        output = int(output)
        add("pair_embedding_linear", _linear_flops(pairs, previous, output))
        add("pair_embedding_layer_norm", _layer_norm_flops(pairs, output))
        add("pair_embedding_gelu", 5 * pairs * output)
        previous = output
    # Registered four-vector pair features: directed angular/pt/mass schema.
    add("four_vector_pair_features", 36 * particles + 43 * pairs)

    depth = width // heads
    for _ in range(layers):
        add("particle_qkv", 3 * _linear_flops(particles, width, width))
        add("particle_attention_qk", 2 * heads * particles * particles * depth)
        add("particle_attention_scale_pair_add", 2 * heads * pairs)
        add("particle_attention_softmax", heads * particles * (3 * particles - 1))
        add("particle_attention_value", 2 * heads * particles * particles * depth)
        add("particle_attention_output", _linear_flops(particles, width, width))
        add("particle_attention_residual_norm", particles * width + _layer_norm_flops(particles, width))
        add("particle_ffn_in", _linear_flops(particles, width, 4 * width))
        add("particle_ffn_gelu", 5 * particles * 4 * width)
        add("particle_ffn_out", _linear_flops(particles, 4 * width, width))
        add("particle_ffn_residual_norm", particles * width + _layer_norm_flops(particles, width))

    memory = particles + 1
    for _ in range(cls_layers):
        add("class_qkv", _linear_flops(1, width, width) + 2 * _linear_flops(memory, width, width))
        add("class_attention_qk", 2 * heads * memory * depth)
        add("class_attention_softmax", heads * (3 * memory - 1))
        add("class_attention_value", 2 * heads * memory * depth)
        add("class_attention_output", _linear_flops(1, width, width))
        add("class_attention_residual_norm", width + _layer_norm_flops(1, width))
        add("class_ffn_in", _linear_flops(1, width, 4 * width))
        add("class_ffn_gelu", 5 * 4 * width)
        add("class_ffn_out", _linear_flops(1, 4 * width, width))
        add("class_ffn_residual_norm", width + _layer_norm_flops(1, width))
    add("classifier", _linear_flops(1, width, len(LABEL_NAMES)))
    return dict(sorted(breakdown.items()))


def particle_view_consumer_semantic_flops(
    config: ParticleViewConsumerConfig,
    *,
    particles: int = 128,
) -> dict[str, int]:
    """Count only the view adapters added around the base A0 ParT."""

    dim, hidden, heads = config.view_dim, config.hidden_dim, config.num_heads
    pairs = particles * particles
    result = {
        "consumer_view_adapter": (
            _linear_flops(particles, dim, hidden)
            + 5 * particles * hidden
            + _linear_flops(particles, hidden, hidden)
        ),
        "consumer_gate": (
            _linear_flops(particles, hidden + dim, hidden)
            + 5 * particles * hidden
            + _linear_flops(particles, hidden, 1)
            + particles
        ),
        "consumer_pair_features": pairs * 4 * dim,
        "consumer_pair_adapter": (
            _linear_flops(pairs, 4 * dim, hidden)
            + 5 * pairs * hidden
            + _linear_flops(pairs, hidden, heads)
        ),
        "consumer_pair_trust_and_add": pairs * (heads + 3),
        "consumer_token_trust_scale_and_add": particles * hidden * 3,
        "consumer_trust_regularizer": 2 * particles - 1,
    }
    return dict(sorted(result.items()))


def particle_transformer_parameter_count(config: Mapping[str, Any]) -> int:
    """Registered Weaver ParticleTransformer learned-scalar accounting.

    This mirrors the repository's locked Weaver configuration without
    importing Weaver. ``build_direct_control_model`` verifies the resulting
    total against the instantiated backend before training.
    """

    width = int(config["embed_dims"][-1])
    heads = int(config["num_heads"])
    count = 0
    # Input normalization and each normalized linear embedding stage.
    previous = int(config["input_dim"])
    count += 2 * previous
    for output in config["embed_dims"]:
        output = int(output)
        count += previous * output + output
        count += 2 * output
        previous = output
    # Pair input normalization, hidden 1x1 projections, and head projection.
    previous = int(config["pair_input_dim"])
    count += 2 * previous
    for output in config["pair_embed_dims"]:
        output = int(output)
        count += previous * output + output
        count += 2 * output
        previous = output
    count += previous * heads + heads
    # Particle attention/FFN blocks.
    for _ in range(int(config["num_layers"])):
        count += 2 * width  # pre-attention norm
        count += 3 * width * width + 3 * width  # qkv
        count += width * width + width  # attention output
        count += 2 * width  # post-attention norm
        count += width * (4 * width) + 4 * width
        count += (4 * width) * width + width
    # Learned class token and class-attention blocks.
    count += width
    for _ in range(int(config["num_cls_layers"])):
        count += 2 * width
        count += 3 * width * width + 3 * width
        count += width * width + width
        count += 2 * width
        count += width * (4 * width) + 4 * width
        count += (4 * width) * width + width
    count += 2 * width  # final class normalization
    count += width * len(LABEL_NAMES) + len(LABEL_NAMES)
    return int(count)


def particle_view_consumer_parameter_count(
    config: ParticleViewConsumerConfig,
) -> int:
    dim, hidden, heads = config.view_dim, config.hidden_dim, config.num_heads
    view_adapter = dim * hidden + hidden + hidden * hidden + hidden
    gate = (hidden + dim) * hidden + hidden + hidden + 1
    pair_adapter = 4 * dim * hidden + hidden + hidden * heads + heads
    return int(view_adapter + gate + pair_adapter + 2)


def _direct_model(candidate: Mapping[str, Any]):
    overrides = {
        key: candidate[key]
        for key in (
            "input_dim",
            "embed_dims",
            "pair_embed_dims",
            "num_heads",
            "num_layers",
            "num_cls_layers",
            "pair_input_dim",
        )
    }
    return build_particle_transformer_classifier(
        num_classes=len(LABEL_NAMES),
        model_size="base",
        overrides=overrides,
    )


def _flop_counter_sha256() -> str:
    return canonical_sha256(
        {
            "counter": "particle_view_flops_v1",
            "direct_part_extension": "direct_hlt_part_semantic_v1",
            "consumer_extension": "particle_view_consumer_semantic_v1",
            "multiply_add_cost": 2,
            "reshape_transpose_mask_memory_cost": 0,
            "particles": 128,
            "batch_size": 1,
            "precision": "float32",
        }
    )


def build_stage_a_direct_resource_plan() -> dict[str, Any]:
    """Profile the canonical deployable architecture and all 16 direct models."""

    grid = build_predeclared_direct_control_grid()
    a0_config = default_part_config(
        num_classes=len(LABEL_NAMES), model_size="base"
    )
    consumer_config = ParticleViewConsumerConfig(view_dim=4)
    predictor = build_canonical_particle_view_predictor(view_dim=4)
    predictor_parameters = count_unique_parameters(predictor)
    parameter_method = "registered_formula_plus_native_predictor_v1"
    try:
        a0 = build_particle_transformer_classifier(
            num_classes=len(LABEL_NAMES), model_size="base"
        )
        consumer = ParticleViewConsumer(a0, consumer_config)
        target_parameters = count_unique_parameters((consumer, predictor))
        a0_parameters = count_unique_parameters(a0)
        consumer_parameters = target_parameters - a0_parameters - predictor_parameters
        parameter_method = "native_weaver_unique_storage_v1"
        del consumer, a0
    except ImportError:
        a0_parameters = particle_transformer_parameter_count(a0_config)
        consumer_parameters = particle_view_consumer_parameter_count(
            consumer_config
        )
        target_parameters = (
            a0_parameters + consumer_parameters + predictor_parameters
        )
    part_breakdown = particle_transformer_semantic_flops(a0_config)
    consumer_breakdown = particle_view_consumer_semantic_flops(consumer_config)
    predictor_flops = predictor_semantic_flops(predictor, particles=128)
    target_breakdown = {
        **{f"a0.{key}": value for key, value in part_breakdown.items()},
        **{
            f"consumer.{key}": value
            for key, value in consumer_breakdown.items()
        },
        **{
            f"predictor.{key}": value
            for key, value in predictor_flops["per_operator"].items()
        },
    }
    target_flops = sum(target_breakdown.values())
    target_profile = with_content_hash(
        {
            "contract": "particle_view_stage_a_canonical_resource_target_v1",
            "view_dim": 4,
            "predictor_architecture_id": PVA3_CANONICAL_ARCHITECTURE,
            "predictor_config_sha256": predictor.config.content_hash,
            "consumer_config": consumer_config.to_payload(),
            "consumer_config_sha256": consumer_config.content_hash,
            "a0_config": a0_config,
            "a0_config_sha256": canonical_sha256(a0_config),
            "deployed_parameters": target_parameters,
            "parameter_breakdown": {
                "a0": a0_parameters,
                "consumer_adapters": consumer_parameters,
                "predictor": predictor_parameters,
            },
            "parameter_count_method": parameter_method,
            "forward_flops": target_flops,
            "forward_flop_breakdown": dict(sorted(target_breakdown.items())),
            "particles": 128,
            "batch_size": 1,
            "precision": "float32",
            "predictor_consumer_weights_shared": False,
        }
    )
    del predictor

    profiles = []
    candidates = []
    for candidate in grid["candidates"]:
        candidate_parameter_method = "registered_weaver_formula_v1"
        try:
            model = _direct_model(candidate)
            parameters = count_unique_parameters(model)
            candidate_parameter_method = "native_weaver_unique_storage_v1"
            del model
        except ImportError:
            parameters = particle_transformer_parameter_count(candidate)
        breakdown = particle_transformer_semantic_flops(candidate)
        flops = sum(breakdown.values())
        config_hash = canonical_sha256(candidate)
        profile = with_content_hash(
            {
                "contract": "particle_view_direct_candidate_resource_v1",
                "config": dict(candidate),
                "config_sha256": config_hash,
                "deployed_parameters": parameters,
                "parameter_count_method": candidate_parameter_method,
                "forward_flops": flops,
                "forward_flop_breakdown": breakdown,
                "particles": 128,
                "batch_size": 1,
                "precision": "float32",
            }
        )
        profiles.append(profile)
        candidates.append(
            DirectControlCandidate(
                config_id=candidate["config_id"],
                deployed_parameters=parameters,
                forward_flops=flops,
                config_sha256=config_hash,
            )
        )
    fixture_hash = flop_fixture_sha256(input_dim=17, particles=128)
    counter_hash = _flop_counter_sha256()
    selections = {
        quantity: select_direct_resource_control(
            candidates=candidates,
            target_parameters=target_parameters,
            target_flops=target_flops,
            requested_quantity=quantity,
            selected_bundle_sha256=target_profile["content_hash"],
            flop_fixture_sha256=fixture_hash,
            flop_counter_sha256=counter_hash,
        )
        for quantity in ("parameters", "flops")
    }
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STAGE_A_RESOURCE_PLAN_CONTRACT,
            "canonical_target": target_profile,
            "candidate_grid_sha256": grid["content_hash"],
            "candidate_profiles": profiles,
            "selections": selections,
            "flop_fixture_sha256": fixture_hash,
            "flop_counter_sha256": counter_hash,
            "quality_warnings_non_gating": True,
            "training_backend_parameter_verification_required": True,
        }
    )
    validate_stage_a_direct_resource_plan(artifact)
    return artifact


def validate_stage_a_direct_resource_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_STAGE_A_RESOURCE_PLAN_CONTRACT
    )
    if set(payload) != {
        "contract",
        "canonical_target",
        "candidate_grid_sha256",
        "candidate_profiles",
        "selections",
        "flop_fixture_sha256",
        "flop_counter_sha256",
        "quality_warnings_non_gating",
        "training_backend_parameter_verification_required",
        "content_hash",
    }:
        raise ValueError("Stage-A resource plan field inventory mismatch")
    require_sha256("candidate_grid_sha256", payload["candidate_grid_sha256"])
    require_sha256("flop_fixture_sha256", payload["flop_fixture_sha256"])
    require_sha256("flop_counter_sha256", payload["flop_counter_sha256"])
    if (
        payload["quality_warnings_non_gating"] is not True
        or payload["training_backend_parameter_verification_required"] is not True
    ):
        raise ValueError("direct resource warnings became gating")
    target = payload["canonical_target"]
    validate_content_hash(
        target,
        expected_contract="particle_view_stage_a_canonical_resource_target_v1",
    )
    if (
        target["view_dim"] != 4
        or target["predictor_architecture_id"]
        != PVA3_CANONICAL_ARCHITECTURE
        or target["particles"] != 128
        or target["batch_size"] != 1
    ):
        raise ValueError("Stage-A canonical resource target changed")
    if (
        sum(target["parameter_breakdown"].values())
        != target["deployed_parameters"]
        or sum(target["forward_flop_breakdown"].values())
        != target["forward_flops"]
        or target["deployed_parameters"] <= 0
        or target["forward_flops"] <= 0
    ):
        raise ValueError("Stage-A canonical resource totals are inconsistent")
    profiles = payload["candidate_profiles"]
    grid = build_predeclared_direct_control_grid()
    if len(profiles) != len(grid["candidates"]) or payload[
        "candidate_grid_sha256"
    ] != grid["content_hash"]:
        raise ValueError("direct candidate grid/profile inventory mismatch")
    by_id = {}
    for profile, candidate in zip(profiles, grid["candidates"]):
        validate_content_hash(
            profile,
            expected_contract="particle_view_direct_candidate_resource_v1",
        )
        if profile["config"] != candidate:
            raise ValueError("direct candidate profile order/config changed")
        if (
            profile["config_sha256"] != canonical_sha256(candidate)
            or profile["deployed_parameters"] <= 0
            or profile["forward_flops"] <= 0
            or sum(profile["forward_flop_breakdown"].values())
            != profile["forward_flops"]
        ):
            raise ValueError("direct candidate resource profile is invalid")
        by_id[candidate["config_id"]] = profile
    if set(payload["selections"]) != {"parameters", "flops"}:
        raise ValueError("direct resource selection inventory mismatch")
    normalized_candidates = [
        DirectControlCandidate(
            config_id=profile["config"]["config_id"],
            deployed_parameters=int(profile["deployed_parameters"]),
            forward_flops=int(profile["forward_flops"]),
            config_sha256=profile["config_sha256"],
        )
        for profile in profiles
    ]
    for quantity, selection in payload["selections"].items():
        validate_content_hash(selection, expected_contract="particle_view_direct_match_v1")
        selected = selection["selected"]
        profile = by_id.get(selected["config_id"])
        if profile is None or (
            selected["deployed_parameters"] != profile["deployed_parameters"]
            or selected["forward_flops"] != profile["forward_flops"]
            or selected["config_sha256"] != profile["config_sha256"]
        ):
            raise ValueError("direct resource selection/profile mismatch")
        if selection["requested_quantity"] != quantity:
            raise ValueError("direct resource requested quantity changed")
        expected_selection = select_direct_resource_control(
            candidates=normalized_candidates,
            target_parameters=int(target["deployed_parameters"]),
            target_flops=int(target["forward_flops"]),
            requested_quantity=quantity,
            selected_bundle_sha256=target["content_hash"],
            flop_fixture_sha256=payload["flop_fixture_sha256"],
            flop_counter_sha256=payload["flop_counter_sha256"],
        )
        if dict(selection) != expected_selection:
            raise ValueError("direct resource selection is not deterministic")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "candidate_count": len(profiles),
        "parameter_config_id": payload["selections"]["parameters"]["selected"][
            "config_id"
        ],
        "flop_config_id": payload["selections"]["flops"]["selected"][
            "config_id"
        ],
    }


@dataclass(frozen=True)
class DirectControlRecipe:
    run_id: str
    seed: int
    selection: Mapping[str, Any]
    resource_plan_sha256: str
    unified_split_manifest_sha256: str
    train_identity_sha256: str
    train_split_sha256: str
    model_val_stop_split_sha256: str
    preprocessing_sha256: str
    source_sha256: str
    library_versions_sha256: str

    def to_payload(self) -> dict[str, Any]:
        if self.run_id not in STAGE_A_DIRECT_CONTROL_RUNS:
            raise ValueError("unknown Stage-A direct-control run")
        if self.seed not in {101, 202, 303}:
            raise ValueError("direct-control seed is not registered")
        validate_content_hash(
            self.selection, expected_contract="particle_view_direct_match_v1"
        )
        quantity = STAGE_A_DIRECT_CONTROL_RUNS[self.run_id]
        if self.selection["requested_quantity"] != quantity:
            raise ValueError("direct-control run/selection quantity mismatch")
        for name in (
            "resource_plan_sha256",
            "unified_split_manifest_sha256",
            "train_identity_sha256",
            "train_split_sha256",
            "model_val_stop_split_sha256",
            "preprocessing_sha256",
            "source_sha256",
            "library_versions_sha256",
        ):
            require_sha256(name, getattr(self, name))
        return {
            "contract": PARTICLE_VIEW_DIRECT_CONTROL_RECIPE_CONTRACT,
            "run_id": self.run_id,
            "seed": self.seed,
            "requested_quantity": quantity,
            "resource_selection": dict(self.selection),
            "resource_selection_sha256": self.selection["content_hash"],
            "resource_plan_sha256": self.resource_plan_sha256,
            "model_config": dict(self.selection["selected"]),
            "unified_split_manifest_sha256": self.unified_split_manifest_sha256,
            "train_identity_sha256": self.train_identity_sha256,
            "train_split_sha256": self.train_split_sha256,
            "model_val_stop_split_sha256": self.model_val_stop_split_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "source_sha256": self.source_sha256,
            "library_versions_sha256": self.library_versions_sha256,
            "train_split": "train",
            "particle_source": "fixed_hlt",
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 3.0e-4,
                "weight_decay": 1.0e-4,
                "betas": [0.9, 0.999],
                "gradient_norm_clip": 1.0,
            },
            "schedule": {
                "name": "linear_warmup_cosine_v1",
                "warmup_updates": 2_000,
                "minimum_learning_rate": 3.0e-6,
                "maximum_epochs": 40,
                "early_stop_patience": 8,
            },
            "physical_batch_size": 128,
            "amp": True,
            "from_scratch": True,
            "quality_warning": self.selection["quality_warning"],
            "warning_is_non_gating": True,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def build_direct_control_recipe(
    *,
    run_id: str,
    seed: int,
    resource_plan: Mapping[str, Any],
    unified_split_manifest: Mapping[str, Any],
    preprocessing_sha256: str,
    source_sha256: str,
    library_versions_sha256: str,
) -> DirectControlRecipe:
    validate_stage_a_direct_resource_plan(resource_plan)
    validate_content_hash(
        unified_split_manifest,
        expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    )
    _, train_hash, identity_hash = logical_split_binding(
        unified_split_manifest, "train"
    )
    _, stop_hash, _ = logical_split_binding(
        unified_split_manifest, "model_val_stop"
    )
    quantity = STAGE_A_DIRECT_CONTROL_RUNS[run_id]
    return DirectControlRecipe(
        run_id=run_id,
        seed=int(seed),
        selection=resource_plan["selections"][quantity],
        resource_plan_sha256=resource_plan["content_hash"],
        unified_split_manifest_sha256=unified_split_manifest["content_hash"],
        train_identity_sha256=identity_hash,
        train_split_sha256=train_hash,
        model_val_stop_split_sha256=stop_hash,
        preprocessing_sha256=preprocessing_sha256,
        source_sha256=source_sha256,
        library_versions_sha256=library_versions_sha256,
    )


def build_direct_control_model(
    recipe: DirectControlRecipe | Mapping[str, Any],
):
    payload = recipe.to_payload() if isinstance(recipe, DirectControlRecipe) else dict(recipe)
    selected = payload["resource_selection"]["selected"]
    grid = build_predeclared_direct_control_grid()
    candidate = next(
        row for row in grid["candidates"] if row["config_id"] == selected["config_id"]
    )
    if canonical_sha256(candidate) != selected["config_sha256"]:
        raise ValueError("direct-control selected configuration changed")
    torch = require_torch()
    torch.manual_seed(int(payload["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(payload["seed"]))
    model = _direct_model(candidate)
    actual = count_unique_parameters(model)
    declared = int(selected["deployed_parameters"])
    if actual != declared:
        raise ValueError(
            "instantiated Weaver direct-control parameter count differs from "
            f"registered formula: actual={actual}, declared={declared}"
        )
    return model


@dataclass(frozen=True)
class DirectControlTrainConfig:
    output_dir: str
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    amp: bool = True


def _autocast(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _scaler(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move(batch: Mapping[str, Any], device):
    expected = {"points", "features", "lorentz_vectors", "mask", "labels"}
    if set(batch) != expected:
        raise ValueError("direct-control batch inventory mismatch")
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def train_direct_hlt_control(
    *,
    recipe: DirectControlRecipe,
    train_loader,
    model_val_stop_loader,
    config: DirectControlTrainConfig,
    model=None,
) -> dict[str, Any]:
    """Train one selected direct ParT using only HLT particles and labels."""

    torch = require_torch()
    payload = recipe.to_payload()
    if getattr(train_loader, "batch_size", None) != 128:
        raise ValueError("direct-control loader batch size must be 128")
    device = resolve_device(config.device)
    model = (model or build_direct_control_model(recipe)).to(device)
    batches = len(train_loader)
    if config.max_train_batches is not None:
        batches = min(batches, int(config.max_train_batches))
    if batches <= 0:
        raise ValueError("direct-control training loader is empty")
    epochs = payload["schedule"]["maximum_epochs"]
    total_updates = batches * epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=payload["optimizer"]["learning_rate"],
        weight_decay=payload["optimizer"]["weight_decay"],
        betas=tuple(payload["optimizer"]["betas"]),
    )
    amp = bool(config.amp and device.type == "cuda")
    scaler = _scaler(torch, amp)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_model_val_stop.pt"
    curves = []
    updates = 0
    current_epoch = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for index, raw in enumerate(train_loader):
            if index >= batches:
                break
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, amp):
                logits = model(
                    batch["points"],
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["labels"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("direct-control loss is nonfinite")
            scaler.scale(loss).backward()
            lr = teacher_learning_rate(
                update_index=updates, total_updates=total_updates
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), payload["optimizer"]["gradient_norm_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            count = int(batch["labels"].numel())
            total_loss += float(loss.detach().item()) * count
            examples += count
            updates += 1
        metrics = evaluate_particle_view_teacher(
            model,
            model_val_stop_loader,
            device=device,
            max_batches=config.max_val_batches,
        )
        curves.append(
            {
                "epoch": epoch,
                "optimizer_updates": updates,
                "train_cross_entropy": total_loss / max(examples, 1),
                **metrics,
            }
        )
        selected = select_teacher_checkpoint(curves)
        selected_epoch = int(selected["epoch"])
        if selected_epoch != current_epoch:
            current_epoch = selected_epoch
            stale = 0
            if selected_epoch == epoch:
                torch.save(
                    {
                        "contract": PARTICLE_VIEW_DIRECT_CONTROL_CHECKPOINT_CONTRACT,
                        "recipe": payload,
                        "recipe_sha256": recipe.content_hash,
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "optimizer_updates": updates,
                        "model_val_stop": {
                            key: metrics[key]
                            for key in ("accuracy", "cross_entropy", "ece")
                        },
                        "class_names": list(LABEL_NAMES),
                    },
                    checkpoint_path,
                )
        else:
            stale += 1
        if stale >= payload["schedule"]["early_stop_patience"]:
            break
    checkpoint_hash = sha256_file(checkpoint_path)
    selected = select_teacher_checkpoint(curves)
    curves_artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DIRECT_CONTROL_CURVES_CONTRACT,
            "recipe_sha256": recipe.content_hash,
            "epochs": curves,
        }
    )
    write_immutable_json(output / "training_curves.json", curves_artifact)
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DIRECT_CONTROL_REGISTRATION_CONTRACT,
            "run_id": recipe.run_id,
            "seed": recipe.seed,
            "recipe": payload,
            "recipe_sha256": recipe.content_hash,
            "resource_plan_sha256": recipe.resource_plan_sha256,
            "resource_selection_sha256": recipe.selection["content_hash"],
            "checkpoint_sha256": checkpoint_hash,
            "selected_epoch": int(selected["epoch"]),
            "model_val_stop": {
                key: selected[key]
                for key in ("accuracy", "cross_entropy", "ece")
            },
            "optimizer_updates": int(
                next(
                    row["optimizer_updates"]
                    for row in curves
                    if row["epoch"] == selected["epoch"]
                )
            ),
            "hlt_only_training": True,
            "hlt_only_inference": True,
            "privileged_inputs": False,
            "quality_warning": payload["quality_warning"],
            "warning_is_non_gating": True,
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(output / "direct_control_registration.json", registration)
    report = with_content_hash(
        {
            "contract": "particle_view_direct_control_report_v1",
            "status": "COMPLETE",
            "run_id": recipe.run_id,
            "seed": recipe.seed,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "registration_sha256": registration["content_hash"],
            "selected_epoch": registration["selected_epoch"],
            "model_val_stop": registration["model_val_stop"],
            "quality_warning": registration["quality_warning"],
            "warning_is_non_gating": True,
        }
    )
    write_immutable_json(output / "direct_control_report.json", report)
    return report


__all__ = [
    "DirectControlRecipe",
    "DirectControlTrainConfig",
    "PARTICLE_VIEW_DIRECT_CONTROL_CHECKPOINT_CONTRACT",
    "PARTICLE_VIEW_DIRECT_CONTROL_CURVES_CONTRACT",
    "PARTICLE_VIEW_DIRECT_CONTROL_RECIPE_CONTRACT",
    "PARTICLE_VIEW_DIRECT_CONTROL_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_STAGE_A_RESOURCE_PLAN_CONTRACT",
    "STAGE_A_DIRECT_CONTROL_RUNS",
    "build_direct_control_model",
    "build_direct_control_recipe",
    "build_stage_a_direct_resource_plan",
    "particle_transformer_semantic_flops",
    "particle_view_consumer_semantic_flops",
    "train_direct_hlt_control",
    "validate_stage_a_direct_resource_plan",
]
