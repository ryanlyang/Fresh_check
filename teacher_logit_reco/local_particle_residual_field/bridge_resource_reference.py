"""Measured representative and confirmed-runtime resource references."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from jetclass_fresh.hlt_baseline import resolve_device

from .bridge import (
    BRIDGE_CHANNEL_PHYSICAL45,
    PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
    validate_bridge_recipe,
)
from .bridge_consumer import T10_CLEAN, build_consumer_tensor_batch
from .bridge_contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    write_immutable_json,
)
from .hierarchical_global_reconstructor import (
    ARCH_A3_HLG_PRIMARY,
    BundleResourceReference,
    DeployedBundleResourceReference,
    RepresentativeArchitectureResourceReference,
    build_step7_hlg_correction_model,
    measure_module_forward_resources,
    measure_step7_resources,
    resource_reference_from_artifact,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_bridge_recipe_execution_binding,
    validate_prediction_anchored_execution_spec,
)
from .bridge_splits import PREDICTION_ANCHORED_SPLIT_CONTRACT


def _profile_arrays(
    particle_width: int, valid_particles: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < int(valid_particles) <= int(particle_width):
        raise ValueError("profile valid-particle count must be within particle width")
    tokens = np.zeros((1, int(particle_width), 14), dtype=np.float32)
    mask = np.zeros((1, int(particle_width)), dtype=bool)
    mask[:, : int(valid_particles)] = True
    rank = np.arange(int(valid_particles), dtype=np.float32)
    tokens[0, : int(valid_particles), 0] = 1.0 + rank / max(int(valid_particles), 1)
    tokens[0, : int(valid_particles), 1] = 0.002 * rank
    tokens[0, : int(valid_particles), 2] = 0.003 * rank
    tokens[0, : int(valid_particles), 3] = 2.0 + rank / max(int(valid_particles), 1)
    fields = np.zeros((1, int(particle_width), 50), dtype=np.float32)
    return tokens, mask, fields


def _r0_unhooked_flops(model: torch.nn.Module, valid_particles: int) -> int:
    """Count custom attention/geometry operations not represented by module hooks."""

    config = getattr(model, "config", None)
    layers = int(getattr(config, "num_layers", 0))
    heads = int(getattr(config, "num_heads", 0))
    width = int(getattr(config, "d_model", 0))
    particles = int(valid_particles)
    if min(layers, heads, width, particles) <= 0:
        return 0
    attention = layers * (
        4 * particles * particles * width
        + 5 * heads * particles * particles
    )
    geometry = particles * particles * 20
    return int(attention + geometry)


def measure_bundle_component_profiles(
    *,
    r0_model: torch.nn.Module,
    t10_model: torch.nn.Module,
    a3_model: torch.nn.Module,
    particle_width: int,
    valid_particles: int,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Execute and profile R0, canonical A3, and T10 with one convention."""

    tokens_np, mask_np, fields_np = _profile_arrays(particle_width, valid_particles)
    torch_device = resolve_device(str(device))
    r0_model = r0_model.to(torch_device)
    t10_model = t10_model.to(torch_device)
    a3_model = a3_model.to(torch_device)
    tokens = torch.as_tensor(tokens_np, device=torch_device)
    mask = torch.as_tensor(mask_np, device=torch_device)
    f0 = torch.as_tensor(fields_np, device=torch_device)
    h0 = torch.zeros(
        (1, int(particle_width), 160), dtype=torch.float32, device=torch_device
    )
    consumer_batch = build_consumer_tensor_batch(
        tokens=tokens_np,
        mask=mask_np,
        labels=np.zeros((1,), dtype=np.int64),
        f0=fields_np,
        f_true=fields_np,
        run_id=T10_CLEAN,
        device=torch_device,
    )

    r0_profile = measure_module_forward_resources(
        r0_model,
        forward_call=lambda: r0_model(tokens, mask),
        architecture_id="R0_C0",
        scope="frozen_r0_field_predictor",
        particle_width=particle_width,
        valid_particles=valid_particles,
        explicit_unhooked_flops=_r0_unhooked_flops(r0_model, valid_particles),
    )
    a3_profile = measure_step7_resources(
        a3_model,
        particle_width=particle_width,
        valid_particles=valid_particles,
    )
    t10_profile = measure_module_forward_resources(
        t10_model,
        forward_call=lambda: t10_model(
            consumer_batch["points"],
            consumer_batch["features"],
            consumer_batch["lorentz_vectors"],
            consumer_batch["mask"],
            tokens=consumer_batch["tokens"],
            raw_mask=consumer_batch["raw_mask"],
            oracle_fields=consumer_batch["oracle_fields"],
        ),
        architecture_id="T10",
        scope="frozen_selected_bridge_consumer",
        particle_width=particle_width,
        valid_particles=valid_particles,
    )
    return {"r0": r0_profile, "a3": a3_profile, "t10": t10_profile}


def build_representative_architecture_resource_reference(
    *,
    r0_model: torch.nn.Module,
    t10_model: torch.nn.Module,
    a3_model: torch.nn.Module,
    particle_width: int,
    valid_particles: int,
    source_manifest_sha256: str,
    device: Any = "cpu",
) -> RepresentativeArchitectureResourceReference:
    profiles = measure_bundle_component_profiles(
        r0_model=r0_model,
        t10_model=t10_model,
        a3_model=a3_model,
        particle_width=particle_width,
        valid_particles=valid_particles,
        device=device,
    )
    return RepresentativeArchitectureResourceReference(
        particle_width=int(particle_width),
        valid_particles=int(valid_particles),
        r0_parameters=profiles["r0"].total_parameters,
        r0_forward_flops=profiles["r0"].forward_flops,
        a3_parameters=profiles["a3"].total_parameters,
        a3_forward_flops=profiles["a3"].forward_flops,
        t10_parameters=profiles["t10"].total_parameters,
        t10_forward_flops=profiles["t10"].forward_flops,
        r0_config_sha256=canonical_sha256(r0_model.config.to_dict()),
        a3_config_sha256=a3_model.config.to_artifact()["content_hash"],
        t10_config_sha256=canonical_sha256(t10_model.config.to_dict()),
        source_manifest_sha256=str(source_manifest_sha256),
    )


def build_confirmed_runtime_resource_reference(
    *,
    representative_artifact: Mapping[str, Any],
    measured_runtime: RepresentativeArchitectureResourceReference,
    r0_checkpoint_sha256: str,
    t10_checkpoint_sha256: str,
    physical45_scaler_sha256: str,
    r0_registration_sha256: str,
    execution_spec_sha256: str,
    child_manifest_sha256: str,
    selected_consumer_sha256: str,
    physical45_recipe_sha256: str,
) -> DeployedBundleResourceReference:
    """Require exact architecture/resource identity before adding checkpoint hashes."""

    representative = resource_reference_from_artifact(representative_artifact)
    if not isinstance(representative, RepresentativeArchitectureResourceReference):
        raise ValueError("preflight resource reference is not representative-only")
    metric_names = (
        "particle_width", "valid_particles", "r0_parameters", "r0_forward_flops",
        "a3_parameters", "a3_forward_flops", "t10_parameters", "t10_forward_flops",
        "r0_config_sha256", "a3_config_sha256", "t10_config_sha256",
        "source_manifest_sha256",
    )
    changed = [
        name for name in metric_names
        if getattr(measured_runtime, name) != getattr(representative, name)
    ]
    if changed:
        raise ValueError(
            "runtime architectures/resources differ from representative preflight: "
            + ", ".join(changed)
        )
    return DeployedBundleResourceReference(
        particle_width=representative.particle_width,
        valid_particles=representative.valid_particles,
        r0_parameters=representative.r0_parameters,
        r0_forward_flops=representative.r0_forward_flops,
        a3_parameters=representative.a3_parameters,
        a3_forward_flops=representative.a3_forward_flops,
        t10_parameters=representative.t10_parameters,
        t10_forward_flops=representative.t10_forward_flops,
        r0_checkpoint_sha256=str(r0_checkpoint_sha256),
        a3_config_sha256=representative.a3_config_sha256,
        t10_checkpoint_sha256=str(t10_checkpoint_sha256),
        physical45_scaler_sha256=str(physical45_scaler_sha256),
        r0_registration_sha256=str(r0_registration_sha256),
        execution_spec_sha256=str(execution_spec_sha256),
        child_manifest_sha256=str(child_manifest_sha256),
        selected_consumer_sha256=str(selected_consumer_sha256),
        physical45_recipe_sha256=str(physical45_recipe_sha256),
        source_manifest_sha256=representative.source_manifest_sha256,
        representative_reference_sha256=str(representative_artifact["content_hash"]),
    )


def publish_confirmed_runtime_resource_reference(
    *,
    representative_reference_path: str | Path,
    execution_spec_path: str | Path,
    r0_checkpoint_path: str | Path,
    r0_registration_path: str | Path,
    selected_consumer_path: str | Path,
    physical45_recipe_path: str | Path,
    physical45_scaler_path: str | Path,
    output_path: str | Path,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Bind identical measured resources to the actual confirmed checkpoints."""

    from .bridge_ram import (
        PREDICTION_ANCHORED_R0_REGISTRATION_CONTRACT,
        FrozenR0Runner,
    )
    from .fusion import load_local_residual_field_tagger_from_checkpoint

    representative_artifact = load_hashed_json(representative_reference_path)
    representative = resource_reference_from_artifact(representative_artifact)
    if not isinstance(representative, RepresentativeArchitectureResourceReference):
        raise ValueError("preflight resource reference is not representative-only")
    execution_spec = load_hashed_json(
        execution_spec_path,
        expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    )
    validate_prediction_anchored_execution_spec(
        execution_spec, verify_file_hashes=False
    )
    child = load_hashed_json(
        execution_spec["child_manifest"]["path"],
        expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT,
    )
    if child["content_hash"] != execution_spec["child_manifest"]["content_hash"]:
        raise ValueError("execution spec and child manifest differ")
    if (
        representative.source_manifest_sha256
        != execution_spec["parent_manifest"]["sha256"]
    ):
        raise ValueError(
            "representative resource reference belongs to a different execution"
        )
    selected = load_hashed_json(
        selected_consumer_path, expected_contract="selected_bridge_consumer_v2"
    )
    if selected.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("runtime resource publication requires confirmed consumer")
    try:
        selected_rho_endpoint = float(selected.get("selected_rho_endpoint"))
    except (TypeError, ValueError) as exc:
        raise ValueError("selected consumer has no valid rho endpoint") from exc
    if selected_rho_endpoint != 0.10:
        raise ValueError("selected consumer changed the locked rho endpoint")
    if selected.get("bridge_channel_policy") != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("selected consumer changed the physical45 channel policy")
    r0_path = Path(r0_checkpoint_path)
    r0_sha256 = sha256_file(r0_path)
    if selected.get("f0_checkpoint_sha256") != r0_sha256:
        raise ValueError("selected consumer was confirmed against a different R0 checkpoint")
    t10_path = Path(str(selected.get("checkpoint_path", "")))
    if sha256_file(t10_path) != selected.get("checkpoint_sha256"):
        raise ValueError("selected T10 checkpoint hash changed before resource publication")
    recipe = load_hashed_json(physical45_recipe_path)
    validate_bridge_recipe(recipe)
    if recipe.get("content_hash") != selected.get("bridge_recipe_sha256"):
        raise ValueError("selected consumer and physical45 recipe differ")
    if recipe.get("channel_policy") != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("runtime resource publication requires the physical45 recipe")
    validate_bridge_recipe_execution_binding(
        recipe,
        execution_spec=execution_spec,
        child_manifest=child,
        r0_checkpoint_sha256=r0_sha256,
    )
    recipe_parents = recipe.get("parent_hashes", {})
    registration = load_hashed_json(
        r0_registration_path,
        expected_contract=PREDICTION_ANCHORED_R0_REGISTRATION_CONTRACT,
    )
    if (
        registration.get("checkpoint_sha256") != r0_sha256
        or registration.get("split_manifest") != child["content_hash"]
        or registration.get("preprocessing")
        != execution_spec["preprocessing_sha256"]
        or registration.get("target_schema")
        != execution_spec["target_schema_sha256"]
    ):
        raise ValueError(
            "R0 registration/checkpoint/child-manifest binding changed"
        )
    scaler = load_hashed_json(
        physical45_scaler_path,
        expected_contract=PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
    )
    if scaler.get("channel_policy") != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("runtime resource publication requires the physical45 scaler")
    if (
        scaler.get("rho_decimal") != recipe.get("rho_decimal")
        or scaler.get("fit_split") != "stack_train_distill"
    ):
        raise ValueError("physical45 scaler recipe endpoint/fit split changed")
    scaler_parents = scaler.get("parent_hashes", {})
    expected_scaler_parents = {
        "r0_checkpoint_sha256": r0_sha256,
        "source_manifest_sha256": recipe_parents.get("split_manifest_sha256"),
        "target_schema_sha256": recipe_parents.get("target_schema_sha256"),
        "mask_sha256": recipe_parents.get("event_order_sha256"),
    }
    changed_scaler_parents = [
        name
        for name, expected in expected_scaler_parents.items()
        if scaler_parents.get(name) != expected
    ]
    if changed_scaler_parents:
        raise ValueError(
            "physical45 scaler provenance differs from selected recipe/R0: "
            + ", ".join(changed_scaler_parents)
        )
    torch_device = resolve_device(str(device))
    r0_runner = FrozenR0Runner(r0_path, device=torch_device)
    t10_model, _payload = load_local_residual_field_tagger_from_checkpoint(
        t10_path, device=torch_device
    )
    a3_model = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler, dropout=0.05
    )
    measured = build_representative_architecture_resource_reference(
        r0_model=r0_runner.model,
        t10_model=t10_model,
        a3_model=a3_model,
        particle_width=representative.particle_width,
        valid_particles=representative.valid_particles,
        source_manifest_sha256=representative.source_manifest_sha256,
        device=torch_device,
    )
    runtime = build_confirmed_runtime_resource_reference(
        representative_artifact=representative_artifact,
        measured_runtime=measured,
        r0_checkpoint_sha256=r0_sha256,
        t10_checkpoint_sha256=sha256_file(t10_path),
        physical45_scaler_sha256=scaler["content_hash"],
        r0_registration_sha256=registration["content_hash"],
        execution_spec_sha256=execution_spec["content_hash"],
        child_manifest_sha256=child["content_hash"],
        selected_consumer_sha256=selected["content_hash"],
        physical45_recipe_sha256=recipe["content_hash"],
    )
    artifact = runtime.to_artifact()
    write_immutable_json(output_path, artifact)
    return artifact


__all__ = [
    "measure_bundle_component_profiles",
    "build_representative_architecture_resource_reference",
    "build_confirmed_runtime_resource_reference",
    "publish_confirmed_runtime_resource_reference",
]
