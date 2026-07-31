"""Common from-scratch trainer for all relational Particle Transformer rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_model, model_forward

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


TRAINING_CONTRACT = "relational_part_training_v1"
TRAINING_CURVES_CONTRACT = "relational_part_training_curves_v1"
CHECKPOINT_CONTRACT = "relational_part_checkpoint_v1"
CHECKPOINT_REGISTRATION_CONTRACT = "relational_part_checkpoint_registration_v2"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    maximum_epochs: int = 40
    minimum_epochs: int = 12
    early_stop_patience: int = 8
    learning_rate: float = 1.0e-3
    minimum_learning_rate: float = 1.0e-5
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 2
    accuracy_window: float = 0.0001
    num_workers: int = 0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.seed not in (101, 202, 303) and self.campaign_profile in {
            "production",
            "hosd_teacher",
        }:
            raise ValueError("production seed must be 101, 202, or 303")
        if self.maximum_epochs <= 0:
            raise ValueError("maximum_epochs must be positive")
        if not 1 <= self.minimum_epochs <= self.maximum_epochs:
            raise ValueError("minimum_epochs lies outside the epoch budget")
        if self.early_stop_patience <= 0:
            raise ValueError("early-stop patience must be positive")
        if self.campaign_profile not in {
            "production",
            "miniature_test",
            "hosd_teacher",
        }:
            raise ValueError("unknown training campaign profile")
        locked = {
            "maximum_epochs": 40,
            "minimum_epochs": 12,
            "early_stop_patience": 8,
            "learning_rate": 1.0e-3,
            "minimum_learning_rate": 1.0e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 1.0e-4,
            "gradient_clip": 1.0,
            "microbatch_size": 64,
            "gradient_accumulation_steps": 2,
            "accuracy_window": 0.0001,
            "num_workers": 0,
        }
        if self.campaign_profile == "production":
            drift = {
                name: (getattr(self, name), expected)
                for name, expected in locked.items()
                if getattr(self, name) != expected
            }
            if drift:
                raise ValueError(f"production training protocol drifted: {drift}")
        if self.campaign_profile == "hosd_teacher":
            hosd_locked = {
                **locked,
                "minimum_epochs": 40,
                "accuracy_window": 0.0,
            }
            drift = {
                name: (getattr(self, name), expected)
                for name, expected in hosd_locked.items()
                if getattr(self, name) != expected
            }
            if drift:
                raise ValueError(f"HOSD teacher training protocol drifted: {drift}")
        if self.microbatch_size != 64 or self.gradient_accumulation_steps != 2:
            raise ValueError(
                "the globally frozen update schedule requires microbatch 64, "
                "accumulation 2"
            )

    def artifact(self, *, global_determinism_sha256: str) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": TRAINING_CONTRACT,
                "schema_version": 1,
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256, name="global_determinism_sha256"
                ),
                "config": asdict(self),
                "loss": "unweighted_10_class_cross_entropy",
                "optimizer": "AdamW",
                "schedule": "one_based_linear_warmup_then_cosine",
                "precision_policy": "cuda_bf16_else_cuda_fp16_scaler_else_cpu_fp32",
                "data_order": {
                    "source_order": "split_manifest_event_identity_order",
                    "epoch_is_one_based": True,
                    "generator": "torch.Generator",
                    "generator_seed": "seed_times_1000003_plus_epoch",
                    "permutation": "torch.randperm(dataset_length)",
                    "same_for_every_configuration_at_seed": True,
                },
                "checkpoint_selector": (
                    "exact_max_balanced_accuracy_then_min_CE_then_earliest"
                    if self.accuracy_window == 0
                    else "global_max_accuracy_0p0001_window_then_min_CE_then_earliest"
                ),
                "val_select_checkpoint_selection_allowed": False,
                "resume_granularity": "completed_epoch_boundary",
                "checkpoint_retention": [
                    "best_model_val.pt",
                    "last.pt_only_while_resumable",
                ],
            }
        )


class DeterministicEpochSampler(
    torch.utils.data.Sampler if torch is not None else object
):
    """Same seed/epoch identity permutation for every configuration."""

    def __init__(self, data_source: Sequence[Any], *, seed: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for deterministic sampling")
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) <= 0:
            raise ValueError("epoch is one-based")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        return iter(
            torch.randperm(len(self.data_source), generator=generator).tolist()
        )

    def __len__(self) -> int:
        return len(self.data_source)


def preferred_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    accuracy_window: float = 0.0001,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("checkpoint selection requires completed epochs")
    checked = []
    for row in rows:
        epoch = int(row["epoch"])
        metrics = row["val_stop"]
        accuracy = float(metrics["accuracy"])
        cross_entropy = float(metrics["cross_entropy"])
        if not (math.isfinite(accuracy) and math.isfinite(cross_entropy)):
            raise FloatingPointError(f"epoch {epoch} has nonfinite selection metrics")
        checked.append((row, epoch, accuracy, cross_entropy))
    maximum = max(item[2] for item in checked)
    eligible = [
        item for item in checked if maximum - item[2] <= float(accuracy_window)
    ]
    return min(eligible, key=lambda item: (item[3], item[1]))[0]


def update_patience(
    rows: Sequence[Mapping[str, Any]],
    *,
    previous_count: int,
    accuracy_window: float = 0.0001,
) -> tuple[int, bool]:
    selected = preferred_checkpoint(rows, accuracy_window=accuracy_window)
    current_is_preferred = int(selected["epoch"]) == int(rows[-1]["epoch"])
    return (0 if current_is_preferred else int(previous_count) + 1), current_is_preferred


def resolve_precision(device: Any) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for training")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return {
            "mode": "fp32",
            "autocast": False,
            "dtype": None,
            "gradient_scaler": False,
        }
    if torch.cuda.is_bf16_supported():
        return {
            "mode": "bf16",
            "autocast": True,
            "dtype": torch.bfloat16,
            "gradient_scaler": False,
        }
    return {
        "mode": "fp16",
        "autocast": True,
        "dtype": torch.float16,
        "gradient_scaler": True,
    }


def model_state_sha256(model: Any) -> str:
    return _state_sha256(model.state_dict())


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gradient_norms(model: Any) -> dict[str, float | None]:
    groups: dict[str, list[Any]] = {}
    for name, parameter in model.named_parameters():
        if "pair_builder.encoders." not in name:
            continue
        family = name.split("pair_builder.encoders.", 1)[1].split(".", 1)[0]
        if parameter.grad is not None:
            groups.setdefault(family, []).append(parameter.grad.detach())
    return {
        family: (
            float(
                torch.sqrt(
                    sum(value.float().square().sum() for value in values)
                ).cpu()
            )
            if values
            else None
        )
        for family, values in sorted(groups.items())
    }


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _capture_diagnostics(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    device: Any,
) -> dict[str, Any] | None:
    diagnostic = getattr(model, "diagnostics", None)
    if not callable(diagnostic):
        return None
    parameters = inspect.signature(diagnostic).parameters
    was_training = bool(model.training)
    model.eval()
    try:
        if not parameters:
            return diagnostic()
        aliases = {
            "lorentz_vectors": ("lorentz_vectors", "vectors"),
            "raw_tokens": ("raw_tokens", "tokens"),
        }
        results = []
        weights = []
        for raw in loader:
            batch = _move(raw, device)
            kwargs = {}
            for name in parameters:
                for candidate in aliases.get(name, (name,)):
                    if candidate in batch:
                        kwargs[name] = batch[candidate]
                        break
            results.append(diagnostic(**kwargs))
            weights.append(int(batch["labels"].shape[0]))
        if not results:
            raise ValueError("val_select diagnostics received an empty loader")

        sum_scalar_fields = {
            "valid_directed_pair_count",
            "valid_particle_count",
            "track_valid_count",
            "masked_query_count",
            "zero_hot_count",
            "multi_hot_count",
            "invalid_binary_value_count",
            "electron_query_pairs",
            "electron_context_pairs",
            "muon_query_pairs",
            "muon_context_pairs",
            "finite_entry_count",
            "count",
        }
        sum_sequence_fields = {
            "category_counts",
            "charge_state_counts",
            "directed_pair_counts",
            "validity_state_counts",
            "bin_counts",
            "event_counts",
            "correct_counts",
        }
        event_distribution_fields = {
            "node_counts",
            "maximum_leaf_depths",
        }
        immutable_fields = {
            "requested_cluster_counts",
            "pair_embedding_norms",
            "uncertainty_floor_audit",
        }

        def add_values(left: Any, right: Any) -> Any:
            if not isinstance(left, (list, tuple)) and isinstance(
                right, (list, tuple)
            ):
                if float(left) != 0.0:
                    raise ValueError(
                        "cannot add a scalar diagnostic statistic to a sequence"
                    )
                return [add_values(0.0, value) for value in right]
            if isinstance(left, (list, tuple)) and isinstance(
                right, (list, tuple)
            ):
                if len(left) != len(right):
                    raise ValueError(
                        "population-statistic numerator shape drifted by batch"
                    )
                return [
                    add_values(left[index], right[index])
                    for index in range(len(left))
                ]
            return left + right

        def divide_values(numerator: Any, denominator: Any) -> Any:
            if isinstance(numerator, (list, tuple)):
                if isinstance(denominator, (list, tuple)):
                    if len(numerator) != len(denominator):
                        raise ValueError(
                            "population-statistic denominator shape drifted"
                        )
                    return [
                        divide_values(numerator[index], denominator[index])
                        for index in range(len(numerator))
                    ]
                return [
                    divide_values(value, denominator) for value in numerator
                ]
            value = float(denominator)
            return None if value == 0.0 else float(numerator) / value

        def aggregate_population_stat(
            specifications: Sequence[Mapping[str, Any]],
        ) -> Any:
            kinds = {str(specification.get("kind")) for specification in specifications}
            if len(kinds) != 1:
                raise ValueError("diagnostic aggregation kind drifted by batch")
            kind = kinds.pop()
            if kind == "sum":
                total: Any = 0
                for specification in specifications:
                    total = add_values(total, specification["value"])
                return total
            if kind == "ratio":
                numerator = specifications[0]["numerator"]
                denominator = specifications[0]["denominator"]
                for specification in specifications[1:]:
                    numerator = add_values(
                        numerator, specification["numerator"]
                    )
                    denominator = add_values(
                        denominator, specification["denominator"]
                    )
                return divide_values(numerator, denominator)
            if kind == "root_mean_square":
                square_sum = specifications[0]["square_sum"]
                denominator = specifications[0]["denominator"]
                for specification in specifications[1:]:
                    square_sum = add_values(
                        square_sum, specification["square_sum"]
                    )
                    denominator = add_values(
                        denominator, specification["denominator"]
                    )
                mean_square = divide_values(square_sum, denominator)
                if isinstance(mean_square, list):
                    return [
                        None if value is None else math.sqrt(value)
                        for value in mean_square
                    ]
                return (
                    None if mean_square is None else math.sqrt(mean_square)
                )
            if kind == "concatenate":
                return [
                    item
                    for specification in specifications
                    for item in specification["values"]
                ]
            if kind == "immutable":
                value = specifications[0]["value"]
                if any(
                    specification["value"] != value
                    for specification in specifications[1:]
                ):
                    raise ValueError("immutable diagnostic value drifted by batch")
                return value
            raise ValueError(f"unknown diagnostic aggregation kind {kind!r}")

        def aggregate(values: Sequence[Any], key: str = "") -> Any:
            present = [value for value in values if value is not None]
            if not present:
                return None
            if all(isinstance(value, Mapping) for value in present):
                statistic_maps = [
                    value.get("_population_statistics", {}) for value in present
                ]
                keys = set(present[0]) - {"_population_statistics"}
                if any(
                    set(value) - {"_population_statistics"} != keys
                    for value in present[1:]
                ):
                    raise ValueError("diagnostic mapping structure drifted by batch")
                statistic_keys = set(statistic_maps[0])
                if any(set(value) != statistic_keys for value in statistic_maps[1:]):
                    raise ValueError(
                        "population-statistic schema drifted by batch"
                    )
                if not statistic_keys.issubset(keys):
                    missing = sorted(statistic_keys - keys)
                    raise ValueError(
                        "population statistics target absent fields: "
                        f"{missing}"
                    )
                output = {}
                for name in sorted(keys):
                    if name in statistic_keys:
                        output[name] = aggregate_population_stat(
                            [value[name] for value in statistic_maps]
                        )
                    else:
                        output[name] = aggregate(
                            [value[name] for value in present], str(name)
                        )
                for name in statistic_keys:
                    if name not in output:
                        raise ValueError(
                            f"population statistic targets absent field {name!r}"
                        )
                return output
            if all(
                isinstance(value, (list, tuple)) for value in present
            ):
                if key == "pair_bias_shape":
                    tail = tuple(present[0][1:])
                    if any(tuple(value[1:]) != tail for value in present[1:]):
                        raise ValueError(
                            "pair-bias non-batch shape drifted by batch"
                        )
                    return [
                        sum(int(value[0]) for value in present),
                        *tail,
                    ]
                if key in event_distribution_fields:
                    return [
                        item for value in present for item in value
                    ]
                if key in immutable_fields:
                    if any(value != present[0] for value in present[1:]):
                        raise ValueError(
                            f"immutable diagnostic {key} drifted by batch"
                        )
                    return list(present[0])
                length = len(present[0])
                if any(len(value) != length for value in present[1:]):
                    raise ValueError(
                        f"fixed diagnostic sequence {key!r} drifted by batch"
                    )
                if key in sum_sequence_fields:
                    return [
                        add_values(
                            0.0, sum(value[index] for value in present)
                        )
                        for index in range(length)
                    ]
                return [
                    aggregate([value[index] for value in present], key)
                    for index in range(length)
                ]
            if all(isinstance(value, bool) for value in present):
                return all(present)
            if all(
                isinstance(value, (int, float, np.integer, np.floating))
                for value in present
            ):
                if key in sum_scalar_fields:
                    return int(sum(int(value) for value in present))
                weighted_values = [
                    (value, weights[index])
                    for index, value in enumerate(values)
                    if value is not None
                ]
                denominator = float(
                    sum(weight for _, weight in weighted_values)
                )
                return float(
                    sum(
                        float(value) * weight
                        for value, weight in weighted_values
                    )
                    / denominator
                )
            if all(value == present[0] for value in present[1:]):
                return present[0]
            return sorted({str(value) for value in present})

        result = aggregate(results)
        return {
            "scope": "complete_val_select_population",
            "aggregation": (
                "schema_aware_v2: declared ratios use exact sufficient "
                "statistics; declared counts/histograms sum; event "
                "distributions concatenate; immutable values must agree"
            ),
            "batch_count": len(results),
            "event_count": sum(weights),
            "batch_diagnostics_sha256": hashlib.sha256(
                canonical_json_bytes(results)
            ).hexdigest(),
            "values": result,
        }
    finally:
        if was_training:
            model.train()


def train_relational_model(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    val_select_loader: Any,
    output_dir: str | Path,
    run_id: str,
    model_contract_sha256: str,
    run_registry_sha256: str,
    relation_registry_sha256: str,
    global_determinism_sha256: str,
    lineage_hashes: Mapping[str, str],
    config: TrainingConfig,
    device: str | Any = "cpu",
    resource_profile: Mapping[str, Any] | None = None,
    resume: bool = True,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_model,
    inference_input_role: str = "hlt_only",
) -> dict[str, Any]:
    """Train, globally select on val_stop, then evaluate val_select exactly once."""

    if torch is None:
        raise RuntimeError("PyTorch is required for training")
    config.validate()
    if inference_input_role not in {"hlt_only", "offline_teacher"}:
        raise ValueError("unknown relational training inference_input_role")
    if resource_profile is None and config.campaign_profile == "production":
        raise ValueError("production training requires a parameter/FLOP profile")
    resource_profile_sha256 = (
        None
        if resource_profile is None
        else validate_content_hash(resource_profile)
    )
    contract = config.artifact(
        global_determinism_sha256=global_determinism_sha256
    )
    model_contract_sha256 = require_sha256(
        model_contract_sha256, name="model_contract_sha256"
    )
    run_registry_sha256 = require_sha256(
        run_registry_sha256, name="run_registry_sha256"
    )
    relation_registry_sha256 = require_sha256(
        relation_registry_sha256, name="relation_registry_sha256"
    )
    lineage = {
        str(name): require_sha256(value, name=f"lineage_hashes.{name}")
        for name, value in sorted(lineage_hashes.items())
    }
    root = Path(output_dir)
    if root.exists() and root.is_symlink():
        raise ValueError("training output directory cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best_model_val.pt"
    last_path = root / "last.pt"
    registration_path = root / "checkpoint_registration.json"
    if registration_path.exists():
        raise FileExistsError("completed run registration already exists")
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError("non-resume training refuses existing checkpoints")

    resolved_device = torch.device(device)
    precision = resolve_precision(resolved_device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    try:
        training_event_count = len(train_loader.dataset)
    except (AttributeError, TypeError):
        training_event_count = len(train_loader) * config.microbatch_size
    counts = optimizer_update_counts(
        training_event_count=training_event_count,
        maximum_epochs=config.maximum_epochs,
    )
    expected_batches = counts["microbatches_per_epoch"]
    if len(train_loader) != expected_batches:
        raise ValueError(
            "train loader length disagrees with the locked non-dropping "
            f"microbatch schedule: {len(train_loader)} != {expected_batches}"
        )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(
                "cuda", enabled=precision["gradient_scaler"]
            )
        except TypeError:  # pragma: no cover - older supported torch
            scaler = torch.amp.GradScaler(
                enabled=precision["gradient_scaler"]
            )
    else:  # pragma: no cover - older supported torch
        scaler = torch.cuda.amp.GradScaler(
            enabled=precision["gradient_scaler"]
        )
    rows: list[dict[str, Any]] = []
    candidate_states: dict[int, dict[str, Any]] = {}
    patience_count = 0
    update_ordinal = 0
    start_epoch = 1
    initial_model_state_sha256 = model_state_sha256(model)

    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("contract") != CHECKPOINT_CONTRACT:
            raise ValueError("resume checkpoint contract mismatch")
        expected = {
            "run_id": run_id,
            "training_contract_sha256": contract["content_hash"],
            "model_contract_sha256": model_contract_sha256,
            "run_registry_sha256": run_registry_sha256,
            "relation_registry_sha256": relation_registry_sha256,
        }
        drift = {
            name: (state.get(name), value)
            for name, value in expected.items()
            if state.get(name) != value
        }
        if drift:
            raise ValueError(f"resume checkpoint lineage drifted: {drift}")
        model.load_state_dict(state["model_state_dict"], strict=True)
        initial_model_state_sha256 = require_sha256(
            state["initial_model_state_sha256"],
            name="resume.initial_model_state_sha256",
        )
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        rows = list(state["rows"])
        candidate_states = {
            int(epoch): values
            for epoch, values in state["candidate_states"].items()
        }
        patience_count = int(state["patience_count"])
        update_ordinal = int(state["optimizer_update_ordinal"])
        start_epoch = int(state["epoch_completed"]) + 1
        _restore_rng_state(state["rng_state"])
        if not best_path.is_file():
            raise FileNotFoundError("resumable run lacks selected checkpoint")
    else:
        _set_seed(config.seed)

    stopped_early = False
    for epoch in range(start_epoch, config.maximum_epochs + 1):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_events = 0
        accumulation_events = 0
        gradient_records: list[dict[str, float | None]] = []
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved_device)
            if "labels" not in batch:
                raise ValueError("training batch lacks labels")
            event_count = int(batch["labels"].numel())
            if event_count <= 0:
                raise ValueError("empty training microbatch")
            autocast = torch.autocast(
                device_type=resolved_device.type,
                dtype=precision["dtype"],
                enabled=precision["autocast"],
            )
            with autocast:
                logits = model_forward(model, batch)
                loss_sum = torch.nn.functional.cross_entropy(
                    logits, batch["labels"].long(), reduction="sum"
                )
            if not bool(torch.isfinite(loss_sum)):
                raise FloatingPointError("training loss is nonfinite")
            scaler.scale(loss_sum).backward()
            epoch_loss_sum += float(loss_sum.detach().float().cpu())
            epoch_events += event_count
            accumulation_events += event_count
            step_now = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not step_now:
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulation_events)
            gradient_records.append(_gradient_norms(model))
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("training gradient is nonfinite")
            update_ordinal += 1
            learning_rate = scheduled_learning_rate(
                update_ordinal=update_ordinal,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_lr=config.learning_rate,
                minimum_lr=config.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulation_events = 0
        expected_updates = epoch * counts["optimizer_updates_per_epoch"]
        if update_ordinal != expected_updates:
            raise RuntimeError(
                f"optimizer update ordinal drifted: {update_ordinal} != "
                f"{expected_updates}"
            )
        val_stop = dict(
            evaluator(
                model,
                val_stop_loader,
                split="val_stop",
                device=resolved_device,
            )
        )
        row = {
            "epoch": epoch,
            "optimizer_update_ordinal": update_ordinal,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_cross_entropy": epoch_loss_sum / epoch_events,
            "relation_encoder_gradient_norms": gradient_records,
            "val_stop": val_stop,
        }
        rows.append(row)
        candidate_states[epoch] = _cpu_state(model)
        maximum_accuracy = max(float(item["val_stop"]["accuracy"]) for item in rows)
        candidate_states = {
            candidate_epoch: candidate_state
            for candidate_epoch, candidate_state in candidate_states.items()
            if maximum_accuracy
            - float(rows[candidate_epoch - 1]["val_stop"]["accuracy"])
            <= config.accuracy_window
        }
        patience_count, current_selected = update_patience(
            rows,
            previous_count=patience_count,
            accuracy_window=config.accuracy_window,
        )
        selected = preferred_checkpoint(
            rows, accuracy_window=config.accuracy_window
        )
        selected_epoch = int(selected["epoch"])
        if selected_epoch not in candidate_states:
            raise RuntimeError("global selector chose an unretained frontier state")
        retained_best_epoch = None
        if best_path.exists():
            retained_best_epoch = int(
                torch.load(
                    best_path, map_location="cpu", weights_only=False
                )["epoch"]
            )
        if retained_best_epoch != selected_epoch:
            selected_state = candidate_states[selected_epoch]
            _atomic_torch_save(
                {
                    "contract": CHECKPOINT_CONTRACT,
                    "schema_version": 1,
                    "kind": "selected_inference",
                    "run_id": run_id,
                    "seed": config.seed,
                    "epoch": selected_epoch,
                    "model_contract_sha256": model_contract_sha256,
                    "training_contract_sha256": contract["content_hash"],
                    "model_state_dict": selected_state,
                    "model_state_sha256": _state_sha256(selected_state),
                    "selection_metrics": {
                        "accuracy": float(selected["val_stop"]["accuracy"]),
                        "cross_entropy": float(
                            selected["val_stop"]["cross_entropy"]
                        ),
                    },
                },
                best_path,
            )
        _atomic_torch_save(
            {
                "contract": CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "resumable_last",
                "run_id": run_id,
                "seed": config.seed,
                "epoch_completed": epoch,
                "model_contract_sha256": model_contract_sha256,
                "training_contract_sha256": contract["content_hash"],
                "run_registry_sha256": run_registry_sha256,
                "relation_registry_sha256": relation_registry_sha256,
                "model_state_dict": _cpu_state(model),
                "initial_model_state_sha256": initial_model_state_sha256,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": (
                    scaler.state_dict() if precision["gradient_scaler"] else None
                ),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "candidate_states": candidate_states,
                "patience_count": patience_count,
                "globally_preferred_epoch": int(selected["epoch"]),
                "rng_state": _rng_state(),
            },
            last_path,
        )
        if (
            epoch >= config.minimum_epochs
            and patience_count >= config.early_stop_patience
        ):
            stopped_early = True
            break

    selected = preferred_checkpoint(rows, accuracy_window=config.accuracy_window)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if int(best["epoch"]) != int(selected["epoch"]):
        raise RuntimeError("retained best checkpoint differs from global selector")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.to(resolved_device)
    val_select = dict(
        evaluator(
            model,
            val_select_loader,
            split="val_select",
            device=resolved_device,
        )
    )
    diagnostics = _capture_diagnostics(model, val_select_loader, resolved_device)
    curves = with_content_hash(
        {
            "contract": TRAINING_CURVES_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "seed": config.seed,
            "training_contract_sha256": contract["content_hash"],
            "rows": rows,
            "selected_epoch": int(selected["epoch"]),
            "stopped_early": stopped_early,
            "precision": {
                name: value
                for name, value in precision.items()
                if name != "dtype"
            },
            "planned_update_counts": counts,
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    write_immutable_json(root / "val_stop_metrics.json", selected["val_stop"])
    write_immutable_json(root / "val_select_metrics.json", val_select)
    checkpoint_sha = sha256_file(best_path)
    registration = with_content_hash(
        {
            "contract": CHECKPOINT_REGISTRATION_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "configuration_role": (
                "reference_baseline"
                if run_id == "RPT_BASE"
                else "semantic_control"
                if run_id == "RPT_SELECTED_UNARY"
                else "capacity_control"
                if run_id in {"RPT_BASE_WIDE_MAX", "RPT_FULL_ZERO_REL"}
                else "architecture_control"
                if run_id in {"RPT_BASE_LAYERWISE", "RPT_BASE_EDGEVALUE"}
                else "scientific_finalist"
            ),
            "relational_selection_eligible": run_id not in {
                "RPT_BASE",
                "RPT_SELECTED_UNARY",
                "RPT_BASE_WIDE_MAX",
                "RPT_FULL_ZERO_REL",
                "RPT_BASE_LAYERWISE",
                "RPT_BASE_EDGEVALUE",
            },
            "seed": config.seed,
            "model_contract_sha256": model_contract_sha256,
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": run_registry_sha256,
            "relation_registry_sha256": relation_registry_sha256,
            "lineage_hashes": lineage,
            "checkpoint_file": best_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": best["model_state_sha256"],
            "initial_model_state_sha256": initial_model_state_sha256,
            "selected_epoch": int(selected["epoch"]),
            "selected_val_stop": {
                "accuracy": float(selected["val_stop"]["accuracy"]),
                "cross_entropy": float(selected["val_stop"]["cross_entropy"]),
            },
            "epochs_completed": len(rows),
            "optimizer_updates_completed": update_ordinal,
            "patience_count_at_completion": patience_count,
            "stopped_early": stopped_early,
            "precision_mode": precision["mode"],
            "val_select_metrics_sha256": val_select["content_hash"],
            "val_select_evaluation_count": 1,
            "val_select_used_for_checkpoint_selection": False,
            "parameter_and_flop_profile": (
                None if resource_profile is None else dict(resource_profile)
            ),
            "resource_profile_sha256": resource_profile_sha256,
            "relation_diagnostics": diagnostics,
            "retained_checkpoints": ["best_model_val.pt"],
            "optimizer_state_retained": False,
            "inference_input_role": inference_input_role,
            "hlt_only_inference": inference_input_role == "hlt_only",
            "offline_or_teacher_required": inference_input_role != "hlt_only",
        }
    )
    write_immutable_json(registration_path, registration)
    if last_path.exists():
        last_path.unlink()
    return registration


__all__ = [
    "CHECKPOINT_CONTRACT",
    "CHECKPOINT_REGISTRATION_CONTRACT",
    "DeterministicEpochSampler",
    "TRAINING_CONTRACT",
    "TRAINING_CURVES_CONTRACT",
    "TrainingConfig",
    "model_state_sha256",
    "preferred_checkpoint",
    "resolve_precision",
    "train_relational_model",
    "update_patience",
]
