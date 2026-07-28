"""Fixed-budget PILOT_T0 training over identity-bound cached evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .bridge_targets import pilot_t0_objective
from .contracts import (
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .expert_training import DeterministicExpertSampler, preferred_expert_epoch
from .evaluation import evaluate_classification
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PILOT_TRAINING_CONTRACT = "retb_pilot_t0_training_v1"
PILOT_REGISTRATION_CONTRACT = "retb_pilot_t0_registration_v1"
PILOT_CURVES_CONTRACT = "retb_pilot_t0_curves_v1"
BRIDGE_CANDIDATE_TRAINING_CONTRACT = "retb_bridge_candidate_training_v1"
BRIDGE_CANDIDATE_REGISTRATION_CONTRACT = "retb_bridge_candidate_registration_v1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for PILOT_T0 training")
    return torch


class BridgePilotDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    def __init__(
        self,
        *,
        identities: Sequence[str],
        labels: np.ndarray,
        hlt_token_banks: Mapping[str, np.ndarray],
        unbiased_particle_states: np.ndarray,
        particle_mask: np.ndarray,
        target_tokens: np.ndarray,
        token_mean: np.ndarray,
        token_standard_deviation: np.ndarray,
        target_expert_logits: np.ndarray,
        target_hybrid_logits: np.ndarray,
        other_t0_banks: Mapping[str, np.ndarray],
        target_expert_id: str,
        split: str,
        lineage_hashes: Mapping[str, str],
    ) -> None:
        _require_torch()
        ids = tuple(str(value) for value in identities)
        truth = np.asarray(labels, dtype=np.int64)
        if (
            not ids
            or len(ids) != len(set(ids))
            or truth.shape != (len(ids),)
            or bool(((truth < 0) | (truth >= 10)).any())
            or target_expert_id not in EXPERT_ORDER
            or split not in {"model_train", "val_stop"}
        ):
            raise ValueError("PILOT_T0 dataset population differs")
        if set(hlt_token_banks) != set(EXPERT_ORDER):
            raise ValueError("PILOT_T0 HLT expert coverage differs")
        self.identities, self.labels = ids, truth
        self.hlt_token_banks = {
            name: np.asarray(hlt_token_banks[name], dtype=np.float32)
            for name in EXPERT_ORDER
        }
        self.unbiased_particle_states = np.asarray(
            unbiased_particle_states, dtype=np.float32
        )
        self.particle_mask = np.asarray(particle_mask, dtype=bool)
        original_target_tokens = np.asarray(target_tokens, dtype=np.float32)
        self.token_mean = np.asarray(token_mean, dtype=np.float32)
        self.token_standard_deviation = np.asarray(
            token_standard_deviation, dtype=np.float32
        )
        if (
            self.token_mean.shape != original_target_tokens.shape[1:]
            or self.token_standard_deviation.shape
            != original_target_tokens.shape[1:]
            or not np.isfinite(self.token_mean).all()
            or not np.isfinite(self.token_standard_deviation).all()
            or bool((self.token_standard_deviation < 0).any())
        ):
            raise ValueError("PILOT_T0 token normalizer differs")
        self.target_tokens_original = original_target_tokens
        self.target_tokens = (
            (original_target_tokens - self.token_mean)
            / np.maximum(self.token_standard_deviation, 1.0e-4)
        ).astype(np.float32)
        self.target_expert_logits = np.asarray(
            target_expert_logits, dtype=np.float32
        )
        self.target_hybrid_logits = np.asarray(
            target_hybrid_logits, dtype=np.float32
        )
        self.other_t0_banks = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in other_t0_banks.items()
        }
        if set(self.other_t0_banks) != set(EXPERT_ORDER) - {target_expert_id}:
            raise ValueError("PILOT_T0 other-bank coverage differs")
        self.target_expert_id = target_expert_id
        self.split = split
        self.lineage_hashes = {
            name: require_sha256(value, name=f"lineage_hashes.{name}")
            for name, value in sorted(lineage_hashes.items())
        }
        n = len(ids)
        arrays = [
            *self.hlt_token_banks.values(),
            self.unbiased_particle_states,
            self.particle_mask,
            self.target_tokens,
            self.target_tokens_original,
            self.target_expert_logits,
            self.target_hybrid_logits,
            *self.other_t0_banks.values(),
        ]
        if (
            any(len(value) != n for value in arrays)
            or self.target_expert_logits.shape != (n, 10)
            or self.target_hybrid_logits.shape != (n, 10)
            or self.particle_mask.shape
            != self.unbiased_particle_states.shape[:2]
            or not all(
                np.isfinite(value).all()
                for value in arrays
                if value.dtype != np.bool_
            )
        ):
            raise ValueError("PILOT_T0 cached arrays differ")

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "identity": self.identities[index],
            "label": self.labels[index],
            "hlt_token_banks": {
                name: self.hlt_token_banks[name][index]
                for name in EXPERT_ORDER
            },
            "unbiased_particle_states": self.unbiased_particle_states[index],
            "particle_mask": self.particle_mask[index],
            "target_tokens": self.target_tokens[index],
            "target_tokens_original": self.target_tokens_original[index],
            "token_mean": self.token_mean,
            "token_standard_deviation": self.token_standard_deviation,
            "target_expert_logits": self.target_expert_logits[index],
            "target_hybrid_logits": self.target_hybrid_logits[index],
            "other_t0_banks": {
                name: value[index]
                for name, value in self.other_t0_banks.items()
            },
        }


def _collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    module = _require_torch()
    return {
        "identities": [row["identity"] for row in rows],
        "labels": module.as_tensor([row["label"] for row in rows]).long(),
        "hlt_token_banks": {
            name: module.from_numpy(
                np.stack([row["hlt_token_banks"][name] for row in rows])
            ).float()
            for name in EXPERT_ORDER
        },
        "unbiased_particle_states": module.from_numpy(
            np.stack([row["unbiased_particle_states"] for row in rows])
        ).float(),
        "particle_mask": module.from_numpy(
            np.stack([row["particle_mask"] for row in rows])
        ).bool(),
        "target_tokens": module.from_numpy(
            np.stack([row["target_tokens"] for row in rows])
        ).float(),
        "target_tokens_original": module.from_numpy(
            np.stack([row["target_tokens_original"] for row in rows])
        ).float(),
        "token_mean": module.from_numpy(
            np.stack([row["token_mean"] for row in rows])
        ).float(),
        "token_standard_deviation": module.from_numpy(
            np.stack([row["token_standard_deviation"] for row in rows])
        ).float(),
        "target_expert_logits": module.from_numpy(
            np.stack([row["target_expert_logits"] for row in rows])
        ).float(),
        "target_hybrid_logits": module.from_numpy(
            np.stack([row["target_hybrid_logits"] for row in rows])
        ).float(),
        "other_t0_banks": {
            name: module.from_numpy(
                np.stack([row["other_t0_banks"][name] for row in rows])
            ).float()
            for name in rows[0]["other_t0_banks"]
        },
    }


def make_bridge_pilot_loader(
    dataset: BridgePilotDataset,
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    module = _require_torch()
    sampler = (
        DeterministicExpertSampler(dataset, seed=seed)
        if training
        else module.utils.data.SequentialSampler(dataset)
    )
    return module.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=0,
        collate_fn=_collate,
    )


@dataclass(frozen=True)
class PilotTrainingConfig:
    seed: int
    maximum_epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    minimum_learning_rate: float = 1.0e-5
    dropout: float = 0.0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("PILOT_T0 profile is unknown")
        if self.dropout != 0.0 or self.learning_rate != 5.0e-4:
            raise ValueError("PILOT_T0 optimizer/dropout is globally fixed")
        if (
            self.maximum_epochs <= 0
            or self.batch_size <= 0
            or self.minimum_learning_rate != 1.0e-5
        ):
            raise ValueError("PILOT_T0 fixed training protocol drifted")
        if self.campaign_profile == "production" and (
            self.seed not in {101, 202, 303}
            or self.maximum_epochs != 40
            or self.batch_size != 256
        ):
            raise ValueError("PILOT_T0 production protocol drifted")

    def artifact(
        self,
        *,
        pilot_architecture_sha256: str,
        global_determinism_sha256: str,
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": PILOT_TRAINING_CONTRACT,
                "schema_version": 1,
                "config": asdict(self),
                "pilot_architecture_sha256": require_sha256(
                    pilot_architecture_sha256,
                    name="pilot_architecture_sha256",
                ),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "objective": "W_TOKEN_HEAVY",
                "context": "C2_ALL",
                "uncertainty": "U_SLOT",
                "normalization": "N_UNCLIPPED",
                "fixed_epoch_budget": True,
                "performance_based_termination": False,
            }
        )


@dataclass(frozen=True)
class BridgeCandidateTrainingConfig:
    seed: int
    target_mode: str
    maximum_epochs: int = 40
    effective_batch_size: int = 256
    predictor_learning_rate: float = 5.0e-4
    target_learning_rate: float = 5.0e-4
    final_two_block_learning_rate_multiplier: float = 0.1
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.target_mode not in {
            "T1_ANCHORED_BRIDGE",
            "T1_TASK_BRIDGE",
            "T2_PROJECT",
            "T3_LOGIT",
        }:
            raise ValueError("bridge candidate training mode is unknown")
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("bridge candidate profile is unknown")
        if self.effective_batch_size < 2:
            raise ValueError("bridge candidate covariance batch must be >=2")
        if (
            self.maximum_epochs <= 0
            or self.predictor_learning_rate != 5.0e-4
            or self.target_learning_rate != 5.0e-4
            or self.final_two_block_learning_rate_multiplier != 0.1
        ):
            raise ValueError("bridge candidate fixed optimizer protocol drifted")
        if self.campaign_profile == "production" and (
            self.seed not in {101, 202, 303}
            or self.maximum_epochs != 40
            or self.effective_batch_size != 256
        ):
            raise ValueError("bridge candidate production protocol drifted")

    def artifact(
        self,
        *,
        materialized_run_sha256: str,
        global_determinism_sha256: str,
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": BRIDGE_CANDIDATE_TRAINING_CONTRACT,
                "schema_version": 1,
                "config": asdict(self),
                "materialized_run_sha256": require_sha256(
                    materialized_run_sha256,
                    name="materialized_run_sha256",
                ),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "alternation": [
                    "predictor_with_offline_target_detached",
                    "offline_target_with_predictor_detached",
                ],
                "covariance_population": (
                    "one_complete_effective_batch_no_microbatch_averaging"
                ),
                "fixed_epoch_budget": True,
                "performance_based_termination": False,
            }
        )


def _move(value: Any, device: Any) -> Any:
    module = _require_torch()
    if isinstance(value, module.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_pilot_t0(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    expert_head: Any,
    hybrid_fusion: Any,
    target_expert_id: str,
    output_dir: str | Path,
    materialized_run: Mapping[str, Any],
    pilot_architecture_sha256: str,
    global_determinism_sha256: str,
    config: PilotTrainingConfig,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    run_sha = validate_content_hash(
        materialized_run, expected_contract="retb_stage_e_materialized_run_v1"
    )
    if (
        materialized_run["component"] != "BRIDGE_PILOT"
        or materialized_run["pipeline_seed"] != config.seed
        or target_expert_id not in EXPERT_ORDER
        or train_loader.dataset.target_expert_id != target_expert_id
        or val_stop_loader.dataset.target_expert_id != target_expert_id
        or train_loader.dataset.split != "model_train"
        or val_stop_loader.dataset.split != "val_stop"
        or train_loader.dataset.lineage_hashes
        != val_stop_loader.dataset.lineage_hashes
        or not np.array_equal(
            train_loader.dataset.token_mean,
            val_stop_loader.dataset.token_mean,
        )
        or not np.array_equal(
            train_loader.dataset.token_standard_deviation,
            val_stop_loader.dataset.token_standard_deviation,
        )
    ):
        raise ValueError("PILOT_T0 run/dataset lineage differs")
    expected_parents = materialized_run["configuration"]["parents"]
    world_size = (
        module.distributed.get_world_size()
        if module.distributed.is_available()
        and module.distributed.is_initialized()
        else 1
    )
    required_lineage = {
        "T0_checkpoint",
        "HLT_encoder_checkpoint",
        "unbiased_HLT_particle_encoder_checkpoint",
        "target_normalizer",
        "T0_fusion",
    }
    if (
        set(train_loader.dataset.lineage_hashes) != required_lineage
        or train_loader.dataset.lineage_hashes["T0_checkpoint"]
        != expected_parents["T0_checkpoint"]
        or train_loader.dataset.lineage_hashes["HLT_encoder_checkpoint"]
        != expected_parents["HLT_encoder_checkpoint"]
        or train_loader.dataset.lineage_hashes[
            "unbiased_HLT_particle_encoder_checkpoint"
        ]
        != expected_parents["unbiased_HLT_particle_encoder_checkpoint"]
        or int(getattr(train_loader, "batch_size", -1)) * world_size
        != config.batch_size
        or int(getattr(val_stop_loader, "batch_size", -1)) * world_size
        != config.batch_size
    ):
        raise ValueError("PILOT_T0 parent or batch lineage differs")
    contract = config.artifact(
        pilot_architecture_sha256=pilot_architecture_sha256,
        global_determinism_sha256=global_determinism_sha256,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "checkpoint_registration.json"
    if registration_path.exists():
        registration = load_hashed_json(
            registration_path, expected_contract=PILOT_REGISTRATION_CONTRACT
        )
        checkpoint = root / "best_model_val.pt"
        if (
            registration["materialized_run_sha256"] != run_sha
            or registration["training_contract_sha256"] != contract["content_hash"]
            or not checkpoint.is_file()
            or _sha256(checkpoint) != registration["checkpoint_sha256"]
        ):
            raise ValueError("reusable PILOT_T0 registration differs")
        return registration
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError("production PILOT_T0 requires GH200 BF16")
    use_bf16 = resolved.type == "cuda"
    for frozen in (expert_head, hybrid_fusion):
        frozen.to(resolved).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    model.to(resolved)
    optimizer = module.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.batch_size,
        gradient_accumulation_steps=1,
    )

    def objective(batch: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        output = model(
            hlt_token_banks=batch["hlt_token_banks"],
            unbiased_particle_states=batch["unbiased_particle_states"],
            particle_mask=batch["particle_mask"],
        )
        predicted = output["predicted_tokens"]
        predicted_original = (
            predicted
            * batch["token_standard_deviation"].clamp_min(1.0e-4)
            + batch["token_mean"]
        )
        expert_logits = expert_head(predicted_original)
        hybrid_banks = dict(batch["other_t0_banks"])
        hybrid_banks[target_expert_id] = predicted_original
        predicted_hybrid = hybrid_fusion(token_banks=hybrid_banks)
        return pilot_t0_objective(
            predicted_tokens=predicted,
            target_tokens=batch["target_tokens"],
            log_variance=output["log_variance"],
            predicted_expert_logits=expert_logits,
            target_expert_logits=batch["target_expert_logits"],
            predicted_hybrid_logits=predicted_hybrid,
            target_hybrid_logits=batch["target_hybrid_logits"],
            labels=batch["labels"],
        )

    def evaluate() -> dict[str, float]:
        model.eval()
        logits, labels = [], []
        with module.no_grad():
            for raw in val_stop_loader:
                batch = _move(raw, resolved)
                with module.autocast(
                    device_type=resolved.type,
                    dtype=module.bfloat16 if use_bf16 else None,
                    enabled=use_bf16,
                ):
                    output = model(
                        hlt_token_banks=batch["hlt_token_banks"],
                        unbiased_particle_states=batch[
                            "unbiased_particle_states"
                        ],
                        particle_mask=batch["particle_mask"],
                    )
                    hybrid_banks = dict(batch["other_t0_banks"])
                    predicted_original = (
                        output["predicted_tokens"]
                        * batch["token_standard_deviation"].clamp_min(1.0e-4)
                        + batch["token_mean"]
                    )
                    hybrid_banks[target_expert_id] = predicted_original
                    hybrid_logits = hybrid_fusion(
                        token_banks=hybrid_banks
                    )
                logits.append(hybrid_logits.float().cpu().numpy())
                labels.append(batch["labels"].cpu().numpy())
        metrics = evaluate_classification(
            np.concatenate(logits),
            np.concatenate(labels),
            split="val_stop",
        )
        return {
            "accuracy": metrics["accuracy"],
            "cross_entropy": metrics["cross_entropy"],
        }

    rows, best_state, update = [], None, 0
    for epoch in range(1, config.maximum_epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        for raw in train_loader:
            batch = _move(raw, resolved)
            optimizer.zero_grad(set_to_none=True)
            with module.autocast(
                device_type=resolved.type,
                dtype=module.bfloat16 if use_bf16 else None,
                enabled=use_bf16,
            ):
                loss, _ = objective(batch)
            loss.backward()
            norm = module.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(module.isfinite(norm)):
                raise FloatingPointError("PILOT_T0 gradient is nonfinite")
            update += 1
            lr = scheduled_learning_rate(
                update_ordinal=update,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=config.learning_rate,
                minimum_learning_rate=config.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
        metrics = evaluate()
        rows.append({"epoch": epoch, "val_stop": metrics})
        if int(preferred_expert_epoch(rows)["epoch"]) == epoch:
            best_state = {
                name: value.detach().cpu().clone()
                if isinstance(value, module.Tensor)
                else value
                for name, value in model.state_dict().items()
            }
    if best_state is None or update != counts["total_optimizer_updates"]:
        raise RuntimeError("PILOT_T0 fixed budget drifted")
    selected = preferred_expert_epoch(rows)
    checkpoint = root / "best_model_val.pt"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".best_model_val.", suffix=".tmp", dir=root
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        module.save(
            {
                "run_id": materialized_run["run_id"],
                "selected_epoch": int(selected["epoch"]),
                "model_state_dict": best_state,
                "training_contract_sha256": contract["content_hash"],
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    curves = with_content_hash(
        {
            "contract": PILOT_CURVES_CONTRACT,
            "schema_version": 1,
            "run_id": materialized_run["run_id"],
            "rows": rows,
            "optimizer_update_counts": counts,
            "fixed_budget_completed": True,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    registration = with_content_hash(
        {
            "contract": PILOT_REGISTRATION_CONTRACT,
            "schema_version": 1,
            "run_id": materialized_run["run_id"],
            "pipeline_seed": config.seed,
            "expert_id": target_expert_id,
            "shape_id": materialized_run["configuration"]["shape_id"],
            "materialized_run_sha256": run_sha,
            "training_contract_sha256": contract["content_hash"],
            "dataset_lineage_hashes": train_loader.dataset.lineage_hashes,
            "checkpoint_sha256": _sha256(checkpoint),
            "training_curves_sha256": curves["content_hash"],
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": True,
            "performance_based_termination": False,
            "precision_mode": "bf16" if use_bf16 else "fp32",
        }
    )
    write_immutable_json(registration_path, registration)
    return registration


def train_bridge_candidate(
    *,
    predictor: Any,
    offline_target: Any,
    train_loader: Any,
    val_stop_evaluator: Any,
    phase_loss_builder: Any,
    output_dir: str | Path,
    materialized_run: Mapping[str, Any],
    pilot_checkpoint: Mapping[str, Any],
    global_determinism_sha256: str,
    config: BridgeCandidateTrainingConfig,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Train a candidate with exact two-phase alternation and joint checkpoints.

    `phase_loss_builder(batch, predictor, offline_target, phase)` must detach
    the inactive side. Gradient-leak checks below make that contract executable.
    """
    from .bridge_targets import alternating_bridge_update

    module = _require_torch()
    config.validate()
    run_sha = validate_content_hash(
        materialized_run, expected_contract="retb_stage_e_materialized_run_v1"
    )
    if (
        materialized_run["component"] != "BRIDGE_TARGET"
        or materialized_run["pipeline_seed"] != config.seed
        or materialized_run["configuration"]["target_mode"]
        != config.target_mode
    ):
        raise ValueError("bridge candidate materialized run differs")
    pilot_sha = require_sha256(
        pilot_checkpoint.get("checkpoint_sha256"),
        name="pilot_checkpoint.checkpoint_sha256",
    )
    world_size = (
        module.distributed.get_world_size()
        if module.distributed.is_available()
        and module.distributed.is_initialized()
        else 1
    )
    validate_content_hash(
        pilot_checkpoint, expected_contract=PILOT_REGISTRATION_CONTRACT
    )
    if (
        materialized_run["configuration"]["parents"][
            "initial_PILOT_T0_checkpoint"
        ]
        != pilot_sha
        or int(pilot_checkpoint["pipeline_seed"]) != config.seed
        or pilot_checkpoint["expert_id"]
        != materialized_run["configuration"]["expert_id"]
        or pilot_checkpoint.get("shape_id")
        != materialized_run["configuration"]["shape_id"]
        or pilot_checkpoint.get("dataset_lineage_hashes", {}).get(
            "T0_checkpoint"
        )
        != materialized_run["configuration"]["parents"]["T0_checkpoint"]
        or pilot_checkpoint.get("dataset_lineage_hashes", {}).get(
            "HLT_encoder_checkpoint"
        )
        != materialized_run["configuration"]["parents"][
            "HLT_encoder_checkpoint"
        ]
        or pilot_checkpoint.get("dataset_lineage_hashes", {}).get(
            "unbiased_HLT_particle_encoder_checkpoint"
        )
        != materialized_run["configuration"]["parents"][
            "unbiased_HLT_particle_encoder_checkpoint"
        ]
        or int(getattr(train_loader, "batch_size", -1)) * world_size
        != config.effective_batch_size
    ):
        raise ValueError("bridge candidate initial PILOT_T0 parent differs")
    contract = config.artifact(
        materialized_run_sha256=run_sha,
        global_determinism_sha256=global_determinism_sha256,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "checkpoint_registration.json"
    if registration_path.exists():
        registration = load_hashed_json(
            registration_path,
            expected_contract=BRIDGE_CANDIDATE_REGISTRATION_CONTRACT,
        )
        checkpoint = root / "best_model_val.pt"
        if (
            registration["materialized_run_sha256"] != run_sha
            or registration["training_contract_sha256"] != contract["content_hash"]
            or not checkpoint.is_file()
            or _sha256(checkpoint) != registration["checkpoint_sha256"]
        ):
            raise ValueError("reusable bridge candidate differs")
        return registration
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError("production bridge candidate requires GH200 BF16")
    use_bf16 = resolved.type == "cuda"
    predictor_only = config.target_mode == "T3_LOGIT"
    if predictor_only and offline_target is not None:
        raise ValueError("T3_LOGIT cannot train an offline token target")
    if not predictor_only and offline_target is None:
        raise ValueError("T1/T2 require an offline token target")
    predictor.to(resolved)
    trainable_target_names: tuple[str, ...] = ()
    if offline_target is not None:
        offline_target.to(resolved)
        configure = getattr(
            offline_target, "configure_bridge_trainability", None
        )
        if configure is None:
            if config.campaign_profile == "production":
                raise TypeError(
                    "production bridge target lacks registered trainability"
                )
            trainable_target_names = tuple(
                name
                for name, parameter in offline_target.named_parameters()
                if parameter.requires_grad
            )
        else:
            trainable_target_names = tuple(
                configure(
                    unfreeze_final_two_blocks=bool(
                        materialized_run["configuration"]["template"][
                            "unfreeze_final_two_blocks"
                        ]
                    )
                )
            )
    predictor_optimizer = module.optim.AdamW(
        predictor.parameters(),
        lr=config.predictor_learning_rate,
        weight_decay=1.0e-4,
    )
    target_groups = []
    final_two, ordinary = [], []
    for name, parameter in (
        () if predictor_only else offline_target.named_parameters()
    ):
        if not parameter.requires_grad:
            continue
        (
            final_two
            if ".blocks.6." in name or ".blocks.7." in name
            else ordinary
        ).append(parameter)
    if ordinary:
        target_groups.append(
            {"params": ordinary, "lr": config.target_learning_rate}
        )
    if final_two:
        target_groups.append(
            {
                "params": final_two,
                "lr": (
                    config.target_learning_rate
                    * config.final_two_block_learning_rate_multiplier
                ),
            }
        )
    if not target_groups and not predictor_only:
        raise ValueError("bridge candidate offline target has no parameters")
    target_optimizer = (
        None
        if predictor_only
        else module.optim.AdamW(target_groups, weight_decay=1.0e-4)
    )
    total_batches = config.maximum_epochs * len(train_loader)
    if total_batches <= 0:
        raise ValueError("bridge candidate loader is empty")
    rows, best_state = [], None
    predictor_update = target_update = 0
    predictor_base_lrs = [
        float(group["lr"]) for group in predictor_optimizer.param_groups
    ]
    target_base_lrs = (
        []
        if predictor_only
        else [float(group["lr"]) for group in target_optimizer.param_groups]
    )
    for epoch in range(1, config.maximum_epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        predictor.train()
        if offline_target is not None:
            offline_target.train()
        for raw in train_loader:
            batch = _move(raw, resolved)
            schedules = [
                (
                    predictor_optimizer,
                    predictor_base_lrs,
                    predictor_update + 1,
                )
            ]
            if not predictor_only:
                schedules.append(
                    (target_optimizer, target_base_lrs, target_update + 1)
                )
            for optimizer, bases, ordinal in schedules:
                multiplier = scheduled_learning_rate(
                    update_ordinal=ordinal,
                    total_optimizer_updates=total_batches,
                    warmup_updates=max(1, int(np.ceil(0.05 * total_batches))),
                    base_learning_rate=1.0,
                    minimum_learning_rate=0.02,
                )
                for group, base in zip(
                    optimizer.param_groups, bases, strict=True
                ):
                    group["lr"] = base * multiplier
            with module.autocast(
                device_type=resolved.type,
                dtype=module.bfloat16 if use_bf16 else None,
                enabled=use_bf16,
            ):
                predictor_loss = phase_loss_builder(
                    batch, predictor, offline_target, "predictor"
                )
            if predictor_only:
                predictor_optimizer.zero_grad(set_to_none=True)
                predictor_loss.backward()
                norm = module.nn.utils.clip_grad_norm_(
                    predictor.parameters(), 1.0
                )
                if not bool(module.isfinite(norm)):
                    raise FloatingPointError("T3_LOGIT gradient is nonfinite")
                predictor_optimizer.step()
            else:
                with module.autocast(
                    device_type=resolved.type,
                    dtype=module.bfloat16 if use_bf16 else None,
                    enabled=use_bf16,
                ):
                    target_loss = phase_loss_builder(
                        batch, predictor, offline_target, "offline_target"
                    )
                alternating_bridge_update(
                    phase="predictor",
                    predictor_loss=predictor_loss,
                    target_loss=target_loss,
                    predictor_optimizer=predictor_optimizer,
                    target_optimizer=target_optimizer,
                )
            predictor_update += 1
            if not predictor_only:
                alternating_bridge_update(
                    phase="offline_target",
                    predictor_loss=predictor_loss,
                    target_loss=target_loss,
                    predictor_optimizer=predictor_optimizer,
                    target_optimizer=target_optimizer,
                )
                target_update += 1
        metrics = dict(val_stop_evaluator(predictor, offline_target, resolved))
        rows.append(
            {
                "epoch": epoch,
                "val_stop": {
                    "accuracy": float(metrics["accuracy"]),
                    "cross_entropy": float(metrics["cross_entropy"]),
                },
            }
        )
        if int(preferred_expert_epoch(rows)["epoch"]) == epoch:
            best_state = {
                "predictor": copy.deepcopy(
                    {
                        name: value.detach().cpu().clone()
                        if isinstance(value, module.Tensor)
                        else value
                        for name, value in predictor.state_dict().items()
                    }
                ),
                "offline_target": (
                    None
                    if predictor_only
                    else copy.deepcopy(
                        {
                            name: value.detach().cpu().clone()
                            if isinstance(value, module.Tensor)
                            else value
                            for name, value in offline_target.state_dict().items()
                        }
                    )
                ),
            }
    if (
        best_state is None
        or predictor_update != total_batches
        or target_update != (0 if predictor_only else total_batches)
    ):
        raise RuntimeError("bridge candidate alternating budget drifted")
    selected = preferred_expert_epoch(rows)
    checkpoint = root / "best_model_val.pt"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".best_model_val.", suffix=".tmp", dir=root
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        module.save(
            {
                "run_id": materialized_run["run_id"],
                "selected_epoch": int(selected["epoch"]),
                "predictor_state_dict": best_state["predictor"],
                "offline_target_state_dict": best_state["offline_target"],
                "training_contract_sha256": contract["content_hash"],
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    curves = with_content_hash(
        {
            "contract": "retb_bridge_candidate_curves_v1",
            "schema_version": 1,
            "run_id": materialized_run["run_id"],
            "rows": rows,
            "predictor_updates": predictor_update,
            "offline_target_updates": target_update,
            "fixed_budget_completed": True,
            "performance_based_termination": False,
            "precision_mode": "bf16" if use_bf16 else "fp32",
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    registration = with_content_hash(
        {
            "contract": BRIDGE_CANDIDATE_REGISTRATION_CONTRACT,
            "schema_version": 1,
            "run_id": materialized_run["run_id"],
            "target_mode": config.target_mode,
            "pipeline_seed": config.seed,
            "expert_id": materialized_run["configuration"]["expert_id"],
            "shape_id": materialized_run["configuration"]["shape_id"],
            "materialized_run_sha256": run_sha,
            "initial_pilot_checkpoint_sha256": pilot_sha,
            "parent_checkpoint_hashes": dict(
                materialized_run["configuration"]["parents"]
            ),
            "training_contract_sha256": contract["content_hash"],
            "checkpoint_sha256": _sha256(checkpoint),
            "training_curves_sha256": curves["content_hash"],
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": True,
            "alternating_detach_verified_every_update": not predictor_only,
            "token_target_trained": not predictor_only,
            "offline_target_trainable_parameter_names": list(
                trainable_target_names
            ),
            "token_fidelity_claim": False if predictor_only else None,
            "performance_based_termination": False,
            "precision_mode": "bf16" if use_bf16 else "fp32",
        }
    )
    write_immutable_json(registration_path, registration)
    return registration


__all__ = [
    "BridgePilotDataset",
    "BridgeCandidateTrainingConfig",
    "PilotTrainingConfig",
    "make_bridge_pilot_loader",
    "train_pilot_t0",
    "train_bridge_candidate",
]
