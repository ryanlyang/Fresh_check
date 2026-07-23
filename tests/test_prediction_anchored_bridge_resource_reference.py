from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from tests.test_prediction_anchored_bridge_execution import (
    _fixture as _execution_fixture,
)
from tests.test_prediction_anchored_bridge_step6 import _fixture as _scaler_fixture
from teacher_logit_reco.local_particle_residual_field import (
    ARCH_A3_HLG_PRIMARY,
    LocalResidualFieldReconstructorConfig,
    build_bridge_recipe,
    build_confirmed_runtime_resource_reference,
    build_local_residual_field_reconstructor,
    build_representative_architecture_resource_reference,
    build_step7_hlg_correction_model,
    measure_bundle_component_profiles,
    publish_confirmed_runtime_resource_reference,
    resource_reference_from_artifact,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    sha256_file,
    with_content_hash,
    write_immutable_json,
)


class _ToyConsumerConfig:
    def to_dict(self):
        return {
            "contract": "toy_consumer_config_v1",
            "model_size": "tiny",
            "field_dim": 50,
            "field_source": "oracle",
        }


class _ToyConsumer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _ToyConsumerConfig()
        self.projection = torch.nn.Linear(1, 10)

    def forward(
        self,
        points,
        features,
        lorentz_vectors,
        mask,
        **_kwargs,
    ):
        del points, lorentz_vectors, mask
        pooled = features.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return self.projection(pooled)


def _models():
    _batch, scaler = _scaler_fixture(n=2, p=4)
    r0 = build_local_residual_field_reconstructor(
        LocalResidualFieldReconstructorConfig(
            d_model=20,
            num_heads=5,
            num_layers=1,
            context_layers=1,
            field_groups={"all": tuple(range(50))},
        )
    )
    a3 = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler
    )
    return r0, _ToyConsumer(), a3


def test_representative_reference_profiles_real_forwards_and_has_no_checkpoint_hashes():
    r0, t10, a3 = _models()
    profiles = measure_bundle_component_profiles(
        r0_model=r0,
        t10_model=t10,
        a3_model=a3,
        particle_width=4,
        valid_particles=4,
    )
    for name in ("r0", "a3", "t10"):
        assert profiles[name].forward_flops > 0
        assert profiles[name].method == "executed_bundle_forward_resource_hooks_v2"
    assert profiles["r0"].forward_flops != profiles["r0"].total_parameters * 4
    assert profiles["t10"].forward_flops != profiles["t10"].total_parameters * 4

    reference = build_representative_architecture_resource_reference(
        r0_model=r0,
        t10_model=t10,
        a3_model=a3,
        particle_width=4,
        valid_particles=4,
        source_manifest_sha256="a" * 64,
    )
    artifact = reference.to_artifact()
    assert artifact["contract"] == (
        "prediction_anchored_representative_architecture_resource_reference_v1"
    )
    assert artifact["checkpoint_hashes_present"] is False
    assert "r0_checkpoint_sha256" not in artifact
    assert "t10_checkpoint_sha256" not in artifact
    assert resource_reference_from_artifact(artifact) == reference


def test_runtime_reference_requires_exact_resources_then_adds_actual_checkpoint_hashes():
    r0, t10, a3 = _models()
    representative = build_representative_architecture_resource_reference(
        r0_model=r0,
        t10_model=t10,
        a3_model=a3,
        particle_width=4,
        valid_particles=4,
        source_manifest_sha256="a" * 64,
    )
    representative_artifact = representative.to_artifact()
    runtime = build_confirmed_runtime_resource_reference(
        representative_artifact=representative_artifact,
        measured_runtime=representative,
        r0_checkpoint_sha256="b" * 64,
        t10_checkpoint_sha256="c" * 64,
        physical45_scaler_sha256="d" * 64,
        r0_registration_sha256="e" * 64,
        execution_spec_sha256="f" * 64,
        child_manifest_sha256="1" * 64,
        selected_consumer_sha256="2" * 64,
        physical45_recipe_sha256="3" * 64,
    )
    artifact = runtime.to_artifact()
    assert artifact["r0_checkpoint_sha256"] == "b" * 64
    assert artifact["t10_checkpoint_sha256"] == "c" * 64
    assert artifact["physical45_scaler_sha256"] == "d" * 64
    assert artifact["representative_reference_sha256"] == representative_artifact[
        "content_hash"
    ]
    assert artifact["resource_values_identical_to_representative"] is True
    assert resource_reference_from_artifact(artifact, require_runtime=True) == runtime

    with pytest.raises(ValueError, match="r0_forward_flops"):
        build_confirmed_runtime_resource_reference(
            representative_artifact=representative_artifact,
            measured_runtime=replace(
                representative,
                r0_forward_flops=representative.r0_forward_flops + 1,
            ),
            r0_checkpoint_sha256="b" * 64,
            t10_checkpoint_sha256="c" * 64,
            physical45_scaler_sha256="d" * 64,
            r0_registration_sha256="e" * 64,
            execution_spec_sha256="f" * 64,
            child_manifest_sha256="1" * 64,
            selected_consumer_sha256="2" * 64,
            physical45_recipe_sha256="3" * 64,
        )


def _filesystem_publisher_fixture(tmp_path, monkeypatch):
    import teacher_logit_reco.local_particle_residual_field.bridge_ram as ram_module
    import teacher_logit_reco.local_particle_residual_field.bridge_resource_reference as resource_module
    import teacher_logit_reco.local_particle_residual_field.fusion as fusion_module

    spec_root = tmp_path / "bound_execution"
    spec_root.mkdir()
    spec, spec_path = _execution_fixture(spec_root)
    child = json.loads(
        (spec_root / "child.json").read_text(encoding="utf-8")
    )
    r0, t10, representative_a3 = _models()
    reference = build_representative_architecture_resource_reference(
        r0_model=r0,
        t10_model=t10,
        a3_model=representative_a3,
        particle_width=4,
        valid_particles=4,
        source_manifest_sha256=spec["parent_manifest"]["sha256"],
    )
    reference_path = tmp_path / "representative.json"
    write_immutable_json(reference_path, reference.to_artifact())

    r0_path = tmp_path / "r0.pt"
    t10_path = tmp_path / "t10.pt"
    r0_path.write_bytes(b"actual-r0-weights")
    t10_path.write_bytes(b"actual-t10-weights")
    r0_sha256 = sha256_file(r0_path)
    t10_sha256 = sha256_file(t10_path)
    recipe = build_bridge_recipe(
        rho="0.100",
        channel_policy="physical45",
        r0_checkpoint_sha256=r0_sha256,
        hlt_source_sha256=spec["sources"]["stack_train"]["hlt_npz"]["sha256"],
        offline_source_sha256=spec["sources"]["stack_train"]["offline_npz"][
            "sha256"
        ],
        split_manifest_sha256=child["children"]["stack_train_distill"][
            "content_hash"
        ],
        target_schema_sha256=spec["target_schema_sha256"],
        preprocessing_sha256=spec["preprocessing_sha256"],
        event_order_sha256="7" * 64,
    )
    recipe_path = tmp_path / "recipe.json"
    write_immutable_json(recipe_path, recipe)

    _batch, raw_scaler = _scaler_fixture(n=2, p=4)
    runtime_scaler = with_content_hash(
        {
            **{
                key: value
                for key, value in raw_scaler.items()
                if key not in {"content_hash", "parent_hashes"}
            },
            "parent_hashes": {
                "source_manifest_sha256": child["children"][
                    "stack_train_distill"
                ]["content_hash"],
                "r0_checkpoint_sha256": r0_sha256,
                "target_schema_sha256": spec["target_schema_sha256"],
                "mask_sha256": "7" * 64,
                "fit_code_sha256": "8" * 64,
            },
        }
    )
    assert runtime_scaler["content_hash"] != representative_a3.scaler_sha256
    scaler_path = tmp_path / "scaler.json"
    write_immutable_json(scaler_path, runtime_scaler)
    selected = with_content_hash(
        {
            "contract": "selected_bridge_consumer_v2",
            "status": "CONFIRMED_LOCKED",
            "checkpoint_path": str(t10_path),
            "checkpoint_sha256": t10_sha256,
            "f0_checkpoint_sha256": r0_sha256,
            "bridge_recipe_sha256": recipe["content_hash"],
            "selected_rho_endpoint": 0.10,
            "bridge_channel_policy": "physical45",
        }
    )
    selected_path = tmp_path / "selected.json"
    write_immutable_json(selected_path, selected)
    registration = with_content_hash(
        {
            "contract": "prediction_anchored_frozen_r0_registration_v1",
            "checkpoint_sha256": r0_sha256,
            "split_manifest": child["content_hash"],
            "preprocessing": spec["preprocessing_sha256"],
            "target_schema": spec["target_schema_sha256"],
        }
    )
    registration_path = tmp_path / "r0_registration.json"
    write_immutable_json(registration_path, registration)

    devices = []

    class _FakeR0Runner:
        def __init__(self, _path, *, device):
            devices.append(str(device))
            self.model = r0

    def _load_t10(_path, *, device):
        devices.append(str(device))
        return t10, {}

    def _resolve_device(device):
        devices.append(f"resolve:{device}")
        return torch.device("cpu")

    monkeypatch.setattr(resource_module, "resolve_device", _resolve_device)
    monkeypatch.setattr(ram_module, "FrozenR0Runner", _FakeR0Runner)
    monkeypatch.setattr(
        fusion_module, "load_local_residual_field_tagger_from_checkpoint", _load_t10
    )
    return {
        "representative_reference_path": reference_path,
        "execution_spec_path": spec_path,
        "r0_checkpoint_path": r0_path,
        "r0_registration_path": registration_path,
        "selected_consumer_path": selected_path,
        "physical45_recipe_path": recipe_path,
        "physical45_scaler_path": scaler_path,
        "output_path": tmp_path / "runtime.json",
        "runtime_scaler": runtime_scaler,
        "r0_sha256": r0_sha256,
        "recipe": recipe,
        "selected": selected,
        "spec": spec,
        "devices": devices,
    }


def test_filesystem_runtime_publisher_resolves_auto_and_accepts_different_bound_scaler(
    tmp_path, monkeypatch
):
    fixture = _filesystem_publisher_fixture(tmp_path, monkeypatch)
    artifact = publish_confirmed_runtime_resource_reference(
        **{
            key: value
            for key, value in fixture.items()
            if key
            not in {
                "runtime_scaler",
                "r0_sha256",
                "recipe",
                "selected",
                "spec",
                "devices",
            }
        },
        device="auto",
    )
    assert "resolve:auto" in fixture["devices"]
    assert artifact["physical45_scaler_sha256"] == fixture["runtime_scaler"][
        "content_hash"
    ]
    assert artifact["r0_checkpoint_sha256"] == fixture["r0_sha256"]
    assert artifact["execution_spec_sha256"] == fixture["spec"]["content_hash"]
    assert artifact["selected_consumer_sha256"] == fixture["selected"]["content_hash"]
    assert artifact["physical45_recipe_sha256"] == fixture["recipe"]["content_hash"]


def test_filesystem_runtime_publisher_rejects_selection_from_another_r0(
    tmp_path, monkeypatch
):
    fixture = _filesystem_publisher_fixture(tmp_path, monkeypatch)
    selected = with_content_hash(
        {
            "contract": "selected_bridge_consumer_v2",
            "status": "CONFIRMED_LOCKED",
            "checkpoint_path": str(tmp_path / "t10.pt"),
            "checkpoint_sha256": sha256_file(tmp_path / "t10.pt"),
            "f0_checkpoint_sha256": "f" * 64,
            "bridge_recipe_sha256": fixture["recipe"]["content_hash"],
            "selected_rho_endpoint": 0.10,
            "bridge_channel_policy": "physical45",
        }
    )
    stale_path = tmp_path / "stale-selected.json"
    write_immutable_json(stale_path, selected)
    fixture["selected_consumer_path"] = stale_path
    with pytest.raises(ValueError, match="different R0 checkpoint"):
        publish_confirmed_runtime_resource_reference(
            **{
                key: value
                for key, value in fixture.items()
                if key
                not in {
                    "runtime_scaler",
                    "r0_sha256",
                    "recipe",
                    "selected",
                    "spec",
                    "devices",
                }
            },
            device="auto",
        )


def test_filesystem_runtime_publisher_rejects_stale_scaler_lineage(
    tmp_path, monkeypatch
):
    fixture = _filesystem_publisher_fixture(tmp_path, monkeypatch)
    scaler = fixture["runtime_scaler"]
    bad_scaler = with_content_hash(
        {
            **{key: value for key, value in scaler.items() if key != "content_hash"},
            "parent_hashes": {
                **scaler["parent_hashes"],
                "source_manifest_sha256": "f" * 64,
            },
        }
    )
    stale_path = tmp_path / "stale-scaler.json"
    write_immutable_json(stale_path, bad_scaler)
    fixture["physical45_scaler_path"] = stale_path
    with pytest.raises(ValueError, match="scaler provenance"):
        publish_confirmed_runtime_resource_reference(
            **{
                key: value
                for key, value in fixture.items()
                if key
                not in {
                    "runtime_scaler",
                    "r0_sha256",
                    "recipe",
                    "selected",
                    "spec",
                    "devices",
                }
            },
            device="auto",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("selected_rho_endpoint", 0.20, "rho endpoint"),
        ("bridge_channel_policy", "all50", "physical45 channel policy"),
    ),
)
def test_filesystem_runtime_publisher_rejects_selected_semantic_drift(
    tmp_path, monkeypatch, field, value, message
):
    fixture = _filesystem_publisher_fixture(tmp_path, monkeypatch)
    selected = with_content_hash(
        {
            **{
                key: item
                for key, item in fixture["selected"].items()
                if key != "content_hash"
            },
            field: value,
        }
    )
    stale_path = tmp_path / f"stale-{field}.json"
    write_immutable_json(stale_path, selected)
    fixture["selected_consumer_path"] = stale_path
    with pytest.raises(ValueError, match=message):
        publish_confirmed_runtime_resource_reference(
            **{
                key: item
                for key, item in fixture.items()
                if key
                not in {
                    "runtime_scaler",
                    "r0_sha256",
                    "recipe",
                    "selected",
                    "spec",
                    "devices",
                }
            },
            device="auto",
        )


def test_filesystem_runtime_publisher_rejects_recipe_from_another_execution(
    tmp_path, monkeypatch
):
    fixture = _filesystem_publisher_fixture(tmp_path, monkeypatch)
    recipe = fixture["recipe"]
    stale_recipe = with_content_hash(
        {
            **{key: value for key, value in recipe.items() if key != "content_hash"},
            "parent_hashes": {
                **recipe["parent_hashes"],
                "hlt_source_sha256": "f" * 64,
            },
        }
    )
    stale_recipe_path = tmp_path / "stale-recipe.json"
    write_immutable_json(stale_recipe_path, stale_recipe)
    selected = with_content_hash(
        {
            **{
                key: value
                for key, value in fixture["selected"].items()
                if key != "content_hash"
            },
            "bridge_recipe_sha256": stale_recipe["content_hash"],
        }
    )
    stale_selected_path = tmp_path / "stale-recipe-selected.json"
    write_immutable_json(stale_selected_path, selected)
    fixture["physical45_recipe_path"] = stale_recipe_path
    fixture["selected_consumer_path"] = stale_selected_path
    with pytest.raises(ValueError, match="different execution.*hlt_source"):
        publish_confirmed_runtime_resource_reference(
            **{
                key: value
                for key, value in fixture.items()
                if key
                not in {
                    "runtime_scaler",
                    "r0_sha256",
                    "recipe",
                    "selected",
                    "spec",
                    "devices",
                }
            },
            device="auto",
        )
