from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from scripts.train_constrained_coarse_to_fine import build_parser
from teacher_logit_reco.constrained_coarse_to_fine import (
    CTierReconstructorOutput,
    CoarseToFineReconstructorOutput,
    CoarseToFineTrainConfig,
    HLTEncoderOutput,
    HierarchyLevelOutput,
    ParticleSlotDecoderOutput,
    runtime,
)
from teacher_logit_reco.constrained_coarse_to_fine import train as c2f_train


@dataclass
class _Node:
    dsl_name: str
    active: bool = True


class _FakeRegistry:
    def __init__(self) -> None:
        self._graphs = {
            ("bmm", "AutogradCUDA"): [_Node("triton"), _Node("other")],
        }
        self.disabled: list[str] = []

    def get_dsl_operations(self, name: str) -> list[str]:
        return ["bmm"] if name == "triton" else []

    def deregister_op_overrides(self, *, disable_dsl_names: str) -> None:
        self.disabled.append(disable_dsl_names)
        for nodes in self._graphs.values():
            for node in nodes:
                if node.dsl_name == disable_dsl_names:
                    node.active = False


def test_native_triton_fallback_skips_cpu() -> None:
    report = runtime.configure_torch_native_triton_fallback(torch.device("cpu"), mode="disable", probe=False)

    assert report["fallback_selected"] is False
    assert report["reason"] == "non_cuda_device"


def test_native_triton_fallback_disables_registered_overrides(monkeypatch) -> None:
    registry = _FakeRegistry()
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: registry)

    report = runtime.configure_torch_native_triton_fallback("cuda", mode="disable", probe=False)

    assert registry.disabled == ["triton"]
    assert report["fallback_selected"] is True
    assert report["registered_triton_operations"] == ["bmm"]
    assert report["active_triton_overrides_after_disable"] == 0
    assert report["probe_passed"] is False


def test_native_triton_auto_mode_is_arm_only(monkeypatch) -> None:
    registry = _FakeRegistry()
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: registry)
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")

    report = runtime.configure_torch_native_triton_fallback("cuda", mode="auto", probe=False)

    assert registry.disabled == []
    assert report["fallback_selected"] is False
    assert report["reason"] == "policy_keeps_native_overrides"


def test_runtime_profile_is_hash_bound_and_records_precision_contract() -> None:
    kwargs = {
        "profile": "fp32_reference",
        "precision_mode": "fp32",
        "batch_size": 16,
        "eval_batch_size": 32,
        "num_workers": 4,
        "prefetch_factor": 2,
        "learning_rate": 2.0e-4,
        "hlt_encoder_lr_scale": 0.05,
        "weight_decay": 1.0e-4,
        "grad_clip_norm": 1.0,
        "lr_schedule": "constant",
        "warmup_fraction": 0.10,
        "min_lr_ratio": 0.05,
        "min_epochs": 0,
        "early_stop_patience": 6,
        "fixed_horizon": False,
        "max_epochs": 30,
        "hungarian_workers": 1,
        "hungarian_executor": "serial",
    }

    first = runtime.build_runtime_profile(**kwargs)
    second = runtime.build_runtime_profile(**kwargs)

    assert first == second
    assert first["runtime_profile_hash"]
    assert first["precision"] == {
        "precision_mode": "fp32",
        "autocast_enabled": False,
        "autocast_dtype": None,
        "grad_scaler_enabled": False,
    }


def test_runtime_profile_rejects_incompatible_precision_and_prefetch() -> None:
    kwargs = {
        "profile": "fp32_reference",
        "precision_mode": "fp32",
        "batch_size": 16,
        "eval_batch_size": 32,
        "num_workers": 0,
        "prefetch_factor": None,
        "learning_rate": 2.0e-4,
        "hlt_encoder_lr_scale": 0.05,
        "weight_decay": 1.0e-4,
        "grad_clip_norm": 1.0,
        "lr_schedule": "constant",
        "warmup_fraction": 0.10,
        "min_lr_ratio": 0.05,
        "min_epochs": 0,
        "early_stop_patience": 6,
        "fixed_horizon": False,
        "max_epochs": 30,
        "hungarian_workers": 1,
        "hungarian_executor": "serial",
    }

    with pytest.raises(ValueError, match="requires precision mode"):
        runtime.build_runtime_profile(**{**kwargs, "profile": "bf16_calibration"})
    with pytest.raises(ValueError, match="prefetch_factor requires"):
        runtime.build_runtime_profile(**{**kwargs, "prefetch_factor": 2})


def test_train_config_persists_named_runtime_profile() -> None:
    config = CoarseToFineTrainConfig(
        output_dir="out",
        manifest_path="manifest.json.gz",
        hlt_cache_dir="hlt",
        offline_cache_dir="offline",
        target_cache_dir="targets",
        variant="C5",
        batch_size=16,
        eval_batch_size=32,
        num_workers=4,
        prefetch_factor=2,
    )

    payload = config.to_dict()

    assert payload["runtime_profile_name"] == "fp32_reference"
    assert payload["runtime_profile_hash"] == payload["runtime_profile"]["runtime_profile_hash"]
    assert payload["runtime_profile"]["input_pipeline"] == {"num_workers": 4, "prefetch_factor": 2}


def test_step_scheduler_warmup_cosine_state_is_resume_stable() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1.0e-3}])
    scheduler = c2f_train._StepLearningRateScheduler(
        optimizer,
        name="warmup_cosine",
        total_steps=20,
        steps_per_epoch=8,
        warmup_fraction=0.5,
        min_lr_ratio=0.05,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    for _ in range(5):
        scheduler.step()
    state = scheduler.state_dict()
    resumed_parameter = torch.nn.Parameter(torch.zeros(()))
    resumed_optimizer = torch.optim.AdamW([{"params": [resumed_parameter], "lr": 1.0e-3}])
    resumed = c2f_train._StepLearningRateScheduler(
        resumed_optimizer,
        name="warmup_cosine",
        total_steps=20,
        steps_per_epoch=8,
        warmup_fraction=0.5,
        min_lr_ratio=0.05,
    )
    resumed.load_state_dict(state)

    assert resumed.global_step == 5
    assert resumed.current_lrs() == pytest.approx(scheduler.current_lrs())
    scheduler.step()
    resumed.step()
    assert resumed.current_lrs() == pytest.approx(scheduler.current_lrs())


def test_candidate_and_approved_profiles_require_last_checkpoint() -> None:
    with pytest.raises(ValueError, match="require save_last_checkpoint"):
        CoarseToFineTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
            runtime_profile="accelerated_candidate_v1",
            precision_mode="bf16_forward_fp32_loss",
            save_last_checkpoint=False,
        )


def test_exploratory_pilot_profile_is_accepted_by_training_cli() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir", "out", "--manifest", "manifest.json.gz",
            "--hlt-cache-dir", "hlt", "--offline-cache-dir", "offline",
            "--target-cache-dir", "targets", "--runtime-profile", "bf16_exploratory_pilot_v1",
        ]
    )
    assert args.runtime_profile == "bf16_exploratory_pilot_v1"


def test_candidate_and_approved_profiles_forbid_skipped_nonfinite_batches() -> None:
    for profile in ("accelerated_candidate_v1", "accelerated_approved_v1"):
        with pytest.raises(ValueError, match="max_nonfinite_batches=0"):
            CoarseToFineTrainConfig(
                output_dir="out", manifest_path="manifest.json.gz", hlt_cache_dir="hlt",
                offline_cache_dir="offline", target_cache_dir="targets", runtime_profile=profile,
                precision_mode="bf16_forward_fp32_loss", max_nonfinite_batches=1,
            )


def test_resume_validation_rejects_runtime_or_provenance_changes() -> None:
    config = CoarseToFineTrainConfig(
        output_dir="out",
        manifest_path="manifest.json.gz",
        hlt_cache_dir="hlt",
        offline_cache_dir="offline",
        target_cache_dir="targets",
        epochs=4,
        batch_size=2,
    )
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": config.learning_rate}])
    scheduler = c2f_train._StepLearningRateScheduler(
        optimizer,
        name=config.lr_schedule,
        total_steps=8,
        steps_per_epoch=2,
        warmup_fraction=config.warmup_fraction,
        min_lr_ratio=config.min_lr_ratio,
    )
    provenance = {"model_train": {"hlt_content_hash": "hlt"}, "model_val": {"hlt_content_hash": "hlt"}}
    precision = runtime.precision_execution_state(config.precision_mode, "cpu")
    source_state = {"code_environment": {"code_environment_hash": "environment"}}
    payload = {
        "checkpoint_contract": c2f_train.COARSE_TO_FINE_TRAIN_CONTRACT,
        "checkpoint_role": "last",
        "epoch": 0,
        "family": "C",
        "variant": "C5",
        "config": config.to_dict(),
        "provenance": provenance,
        "precision_execution": precision,
        "code_environment": source_state["code_environment"],
        "scheduler_state": scheduler.state_dict(),
        "training_state": {
            "completed_epoch": 0,
            "global_optimizer_step": 0,
            "curves": [{"epoch": 0}],
            "best_epoch": 0,
            "best_loss": 1.0,
            "best_model_val": {},
            "epochs_without_improvement": 0,
        },
        "rng_state": {"python": object(), "numpy": object(), "torch_cpu": object()},
    }

    state = c2f_train._validate_resume_checkpoint(
        payload,
        config=config,
        family="C",
        variant="C5",
        provenance=provenance,
        precision_execution=precision,
        scheduler=scheduler,
        source_state=source_state,
    )
    assert state["completed_epoch"] == 0

    payload["provenance"] = {"model_train": {"hlt_content_hash": "stale"}}
    with pytest.raises(ValueError, match="provenance"):
        c2f_train._validate_resume_checkpoint(
            payload,
            config=config,
            family="C",
            variant="C5",
            provenance=provenance,
            precision_execution=precision,
            scheduler=scheduler,
            source_state=source_state,
        )


def test_code_environment_fingerprint_is_hash_bound(monkeypatch, tmp_path) -> None:
    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command, **_kwargs):
        if "rev-parse" in command:
            return _Result("abc123\n")
        if "status" in command:
            return _Result("")
        raise AssertionError(command)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    environment = runtime.collect_code_environment(tmp_path)

    assert environment["source_commit"] == "abc123"
    assert environment["source_tree_clean"] is True
    assert environment["code_environment_hash"]


def test_precision_execution_keeps_bf16_unscaled_and_fp16_scaler_backed(monkeypatch) -> None:
    bf16_cpu = runtime.precision_execution_state("bf16_forward_fp32_loss", "cpu")
    assert bf16_cpu["autocast_enabled"] is False
    assert bf16_cpu["grad_scaler_enabled"] is False
    assert runtime.precision_grad_scaler("bf16_forward_fp32_loss", "cpu") is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    bf16_cuda = runtime.precision_execution_state("bf16_forward_fp32_loss", "cuda")
    fp16_cuda = runtime.precision_execution_state("fp16_forward_fp32_loss", "cuda")

    assert bf16_cuda["autocast_enabled"] is True
    assert bf16_cuda["autocast_dtype"] == "bfloat16"
    assert bf16_cuda["grad_scaler_enabled"] is False
    assert fp16_cuda["autocast_enabled"] is True
    assert fp16_cuda["autocast_dtype"] == "float16"
    assert fp16_cuda["grad_scaler_enabled"] is True


def test_precision_autocast_uses_explicit_bf16_and_disables_for_loss(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _Context:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    def fake_autocast(device_type, *, dtype=None, enabled=True):
        calls.append({"device_type": device_type, "dtype": dtype, "enabled": enabled})
        return _Context()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)

    with runtime.precision_autocast_context("bf16_forward_fp32_loss", "cuda"):
        pass
    with runtime.precision_autocast_disabled_context("cuda"):
        pass

    assert calls == [
        {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True},
        {"device_type": "cuda", "dtype": None, "enabled": False},
    ]


def test_fp32_loss_view_casts_typed_outputs_without_detaching_gradients() -> None:
    def half(shape):
        return torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

    hierarchy = CoarseToFineReconstructorOutput(
        variant="B3",
        global_accounting=half((1, 1, 4)),
        global_log_sigma=half((1, 1, 4)),
        global_auxiliary=half((1, 1, 3)),
        global_auxiliary_names=("a", "b", "c"),
        global_token=half((1, 4)),
        levels=(
            HierarchyLevelOutput(
                name="level1",
                level=1,
                accounting=half((1, 2, 4)),
                cell_tokens=half((1, 2, 4)),
                log_sigma=half((1, 2, 4)),
                parent_indices=torch.zeros(2, dtype=torch.long),
                allocation_logits=half((1, 2, 1)),
                primitive_fractions=None,
                hard_allocation=True,
            ),
        ),
        hlt=HLTEncoderOutput(
            particle_embeddings=half((1, 2, 4)),
            jet_embedding=half((1, 4)),
            particle_mask=torch.ones((1, 2), dtype=torch.bool),
            pool_attention=half((1, 2)),
            pair_bias=half((1, 1, 2, 2)),
        ),
        supervised_field_mask=torch.ones(4, dtype=torch.bool),
        diagnostics={},
    )
    slots = ParticleSlotDecoderOutput(
        variant="C5",
        terminal_level=1,
        terminal_accounting=half((1, 2, 4)),
        terminal_cell_tokens=half((1, 2, 4)),
        real_slot_embeddings=half((1, 1, 2, 2, 4)),
        local_coordinates=half((1, 1, 2, 2, 2)),
        total_pt=half((1, 1, 2, 2)),
        category_pt=half((1, 1, 2, 2, 4)),
        total_energy=half((1, 1, 2, 2)),
        expected_count=half((1, 1, 2)),
        category_count=half((1, 1, 2, 4)),
        pid_probabilities=half((1, 1, 2, 2, 4)),
        raw_pid_logits=half((1, 1, 2, 2, 4)),
        charge_logits=half((1, 1, 2, 2, 3)),
        existence_logits=half((1, 1, 2, 2)),
        log_sigma=half((1, 1, 2, 2, 3)),
        reliability=half((1, 1, 2, 2)),
        dust_total_pt=half((1, 1, 2)),
        dust_category_pt=half((1, 1, 2, 4)),
        dust_total_energy=half((1, 1, 2)),
        rendered_accounting=half((1, 1, 2, 4)),
        stochastic_latent=None,
        diagnostics={},
    )
    output = CTierReconstructorOutput(hierarchy=hierarchy, slots=slots)

    fp32_output = c2f_train._fp32_loss_view(output)
    loss = fp32_output.hierarchy.global_accounting.sum() + fp32_output.slots.total_pt.sum()
    loss.backward()

    assert fp32_output.hierarchy.global_accounting.dtype == torch.float32
    assert fp32_output.hierarchy.levels[0].accounting.dtype == torch.float32
    assert fp32_output.slots.total_pt.dtype == torch.float32
    assert fp32_output.slots.raw_pid_logits.dtype == torch.float32
    assert hierarchy.global_accounting.grad is not None
    assert slots.total_pt.grad is not None
