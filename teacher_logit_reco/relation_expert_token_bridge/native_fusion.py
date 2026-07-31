"""Native HLT expert-bank fusion models, replica caches, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_classification
from .expert_training import DeterministicExpertSampler, preferred_expert_epoch
from .fusion import (
    TokenTransformerFusion,
    TrainedLogitLinear,
    UniformLogitMean,
)
from .registry import EXPERT_ORDER
from .replicas import REALIZATION_POLICIES, replica_for

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


NATIVE_FUSION_CONTRACT = "retb_native_hlt_fusion_architecture_v1"
NATIVE_FUSION_CACHE_CONTRACT = "retb_native_hlt_fusion_cache_v2"
NATIVE_FUSION_TRAINING_CONTRACT = "retb_native_hlt_fusion_training_v1"
NATIVE_FUSION_REGISTRATION_CONTRACT = "retb_native_hlt_fusion_registration_v1"
NATIVE_FUSION_CURVES_CONTRACT = "retb_native_hlt_fusion_curves_v1"
NATIVE_FUSION_VARIANTS = (
    "HF_NATIVE",
    "HF_LOGIT_MEAN",
    "HF_TRAINED_LOGIT",
    "HF_7X_UNBIASED_LOGIT_MEAN",
    "HF_7X_UNBIASED_TOKEN_FUSION",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for native HLT fusion")
    return torch


def _precision(device: Any) -> dict[str, Any]:
    module = _require_torch()
    resolved = module.device(device)
    enabled = resolved.type == "cuda"
    if enabled and not module.cuda.is_bf16_supported():
        raise RuntimeError("native HLT fusion CUDA execution requires BF16")
    return {
        "mode": "bf16" if enabled else "fp32",
        "enabled": enabled,
        "dtype": module.bfloat16 if enabled else None,
    }


def build_native_fusion_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": NATIVE_FUSION_CONTRACT,
            "schema_version": 1,
            "variants": list(NATIVE_FUSION_VARIANTS),
            "HF_NATIVE": {
                "architecture": "same_3_layer_8_head_width128_token_transformer",
                "inputs": "native_HLT_expert_tokens_only",
                "offline_reconstruction": False,
                "experts_frozen": True,
            },
            "HF_LOGIT_MEAN": {"parameter_updates": 0},
            "HF_TRAINED_LOGIT": {"input_dimension": 70, "output_dimension": 10},
            "unbiased_controls": {
                "expert_count": 7,
                "relation_family_for_every_expert": "BASE4",
                "same_aggregate_expert_and_fusion_topology": True,
            },
            "training_replica_policy": "identity_epoch_bound",
            "joint_deployment_policy": "R_MULTI",
            "evaluation_replica_id": 0,
            "offline_targets_permitted": False,
            "fixed_epochs": 40,
            "performance_based_termination": False,
        }
    )


def build_native_fusion_model(
    variant: str,
    *,
    bank_dimensions: Mapping[str, int],
) -> Any:
    if variant in {"HF_NATIVE", "HF_7X_UNBIASED_TOKEN_FUSION"}:
        return TokenTransformerFusion(bank_dimensions=bank_dimensions)
    if variant in {"HF_LOGIT_MEAN", "HF_7X_UNBIASED_LOGIT_MEAN"}:
        return UniformLogitMean()
    if variant == "HF_TRAINED_LOGIT":
        return TrainedLogitLinear()
    raise ValueError("native HLT fusion variant is not registered")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_native_fusion_cache(
    *,
    output_dir: str | Path,
    split: str,
    pipeline_seed: int,
    shape_id: str,
    realization_policy: str,
    identities: Sequence[str],
    labels: np.ndarray,
    token_banks_by_replica: Mapping[int, Mapping[str, np.ndarray]],
    expert_logits_by_replica: Mapping[int, Mapping[str, np.ndarray]],
    expert_registration_hashes: Mapping[str, str],
    hlt_cache_hashes_by_replica: Mapping[int, str],
    identity_manifest_sha256: str,
    label_manifest_sha256: str,
    source_snapshot: Mapping[str, Any] | None = None,
    unbiased_particle_states_by_replica: Mapping[int, np.ndarray] | None = None,
    particle_masks_by_replica: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    if realization_policy not in REALIZATION_POLICIES:
        raise ValueError("native fusion realization policy is unknown")
    if set(expert_registration_hashes) != set(EXPERT_ORDER):
        raise ValueError("native fusion expert registration coverage differs")
    expected = (
        {0}
        if split not in {"model_train", "scale_train"}
        or realization_policy == "R_FIXED"
        else {0, 1, 2, 3}
    )
    if (
        unbiased_particle_states_by_replica is None
        or particle_masks_by_replica is None
    ):
        if (
            source_snapshot is not None
            or unbiased_particle_states_by_replica is not None
            or particle_masks_by_replica is not None
        ):
            raise ValueError(
                "authenticated native fusion cache requires unbiased "
                "particle states and masks"
            )
        # Backward-compatible miniature fixture path. Production publications
        # are source-bound and therefore cannot enter this branch.
        synthesized_states: dict[int, np.ndarray] = {}
        synthesized_masks: dict[int, np.ndarray] = {}
        for replica in expected:
            base = np.asarray(
                token_banks_by_replica[replica]["BASE4"],
                dtype=np.float32,
            )
            synthesized_states[replica] = np.pad(
                base, ((0, 0), (0, 0), (0, max(0, 128 - base.shape[-1])))
            )[..., :128]
            synthesized_masks[replica] = np.ones(
                base.shape[:2], dtype=bool
            )
        unbiased_particle_states_by_replica = synthesized_states
        particle_masks_by_replica = synthesized_masks
    if (
        set(token_banks_by_replica) != expected
        or set(expert_logits_by_replica) != expected
        or set(unbiased_particle_states_by_replica) != expected
        or set(particle_masks_by_replica) != expected
        or set(hlt_cache_hashes_by_replica) != expected
    ):
        raise ValueError("native fusion cache replica coverage differs")
    ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        not ids
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or bool(((truth < 0) | (truth >= 10)).any())
    ):
        raise ValueError("native fusion cache population differs")
    allocation = None
    arrays: dict[str, np.ndarray] = {
        "identities": np.asarray(ids),
        "labels": truth,
    }
    for replica in sorted(expected):
        if (
            set(token_banks_by_replica[replica]) != set(EXPERT_ORDER)
            or set(expert_logits_by_replica[replica]) != set(EXPERT_ORDER)
        ):
            raise ValueError("native fusion cache expert coverage differs")
        current = {}
        for expert in EXPERT_ORDER:
            tokens = np.asarray(
                token_banks_by_replica[replica][expert], dtype=np.float32
            )
            logits = np.asarray(
                expert_logits_by_replica[replica][expert], dtype=np.float32
            )
            if (
                tokens.ndim != 3
                or tokens.shape[0] != len(ids)
                or tokens.shape[1] not in {1, 2, 4, 8, 16}
                or tokens.shape[2] not in {64, 128}
                or logits.shape != (len(ids), 10)
                or not np.isfinite(tokens).all()
                or not np.isfinite(logits).all()
            ):
                raise ValueError("native fusion cache arrays differ")
            current[expert] = [int(tokens.shape[1]), int(tokens.shape[2])]
            arrays[f"tokens_r{replica}_{expert}"] = tokens
            arrays[f"logits_r{replica}_{expert}"] = logits
        states = np.asarray(
            unbiased_particle_states_by_replica[replica],
            dtype=np.float32,
        )
        particle_mask = np.asarray(
            particle_masks_by_replica[replica], dtype=bool
        )
        if (
            states.ndim != 3
            or states.shape[0] != len(ids)
            or states.shape[2] != 128
            or particle_mask.shape != states.shape[:2]
            or not np.isfinite(states).all()
        ):
            raise ValueError("native unbiased particle evidence differs")
        arrays[f"unbiased_particle_states_r{replica}"] = states
        arrays[f"particle_mask_r{replica}"] = particle_mask
        if allocation is None:
            allocation = current
        elif allocation != current:
            raise ValueError("native fusion allocation differs across replicas")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{split}_native_hlt_tokens.npz"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{npz_path.name}.", suffix=".tmp.npz", dir=root
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        if npz_path.exists():
            if npz_path.read_bytes() != temporary.read_bytes():
                raise FileExistsError("refusing to overwrite native fusion cache")
        else:
            os.link(temporary, npz_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    payload: dict[str, Any] = {
        "contract": NATIVE_FUSION_CACHE_CONTRACT,
        "schema_version": 2,
        "split": split,
        "pipeline_seed": int(pipeline_seed),
        "shape_id": str(shape_id),
        "realization_policy": realization_policy,
        "replica_ids": sorted(expected),
        "expert_order": list(EXPERT_ORDER),
        "allocation": allocation,
        "event_count": len(ids),
        "identity_manifest_sha256": require_sha256(
            identity_manifest_sha256, name="identity_manifest_sha256"
        ),
        "label_manifest_sha256": require_sha256(
            label_manifest_sha256, name="label_manifest_sha256"
        ),
        "expert_registration_hashes": {
            name: require_sha256(
                expert_registration_hashes[name],
                name=f"expert_registration_hashes.{name}",
            )
            for name in EXPERT_ORDER
        },
        "hlt_cache_hashes_by_replica": {
            str(replica): require_sha256(
                hlt_cache_hashes_by_replica[replica],
                name=f"hlt_cache_hashes_by_replica.{replica}",
            )
            for replica in sorted(expected)
        },
        "npz_filename": npz_path.name,
        "npz_sha256": _file_sha256(npz_path),
        "contains_offline_targets": False,
        "contains_unbiased_base4_particle_states": True,
        "identity_epoch_replica_selection": True,
    }
    manifest = with_content_hash(payload)
    if source_snapshot is not None:
        manifest = bind_source(manifest, source_snapshot=source_snapshot)
    write_immutable_json(root / f"{split}_native_hlt_tokens.json", manifest)
    return manifest


def load_native_fusion_cache(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_hashed_json(
        path, expected_contract=NATIVE_FUSION_CACHE_CONTRACT
    )
    for name in (
        "identity_manifest_sha256",
        "label_manifest_sha256",
        "npz_sha256",
    ):
        require_sha256(manifest.get(name), name=name)
    if manifest.get("expert_order") != list(EXPERT_ORDER):
        raise ValueError("native fusion expert order differs")
    if set(manifest.get("expert_registration_hashes", {})) != set(EXPERT_ORDER):
        raise ValueError("native fusion registration coverage differs")
    for expert, digest in manifest["expert_registration_hashes"].items():
        require_sha256(digest, name=f"expert_registration_hashes.{expert}")
    npz_path = path.parent / manifest["npz_filename"]
    if (
        not npz_path.is_file()
        or npz_path.is_symlink()
        or _file_sha256(npz_path) != manifest["npz_sha256"]
    ):
        raise ValueError("native fusion cache bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    expected_fields = {"identities", "labels"} | {
        f"{kind}_r{replica}_{expert}"
        for replica in manifest["replica_ids"]
        for expert in EXPERT_ORDER
        for kind in ("tokens", "logits")
    } | {
        f"{kind}_r{replica}"
        for replica in manifest["replica_ids"]
        for kind in ("unbiased_particle_states", "particle_mask")
    }
    if set(arrays) != expected_fields:
        raise ValueError("native fusion cache fields differ")
    identities = tuple(str(value) for value in arrays["identities"].tolist())
    if len(identities) != manifest["event_count"] or len(identities) != len(
        set(identities)
    ):
        raise ValueError("native fusion cache identities differ")
    labels = np.asarray(arrays["labels"])
    if (
        labels.shape != (manifest["event_count"],)
        or not np.issubdtype(labels.dtype, np.integer)
        or bool(((labels < 0) | (labels >= 10)).any())
    ):
        raise ValueError("native fusion cache labels differ")
    for replica in manifest["replica_ids"]:
        states = arrays[f"unbiased_particle_states_r{replica}"]
        particle_mask = arrays[f"particle_mask_r{replica}"]
        if (
            states.shape
            != (
                manifest["event_count"],
                particle_mask.shape[1],
                128,
            )
            or particle_mask.shape != states.shape[:2]
            or not np.isfinite(states).all()
        ):
            raise ValueError("native particle evidence arrays differ")
        for expert in EXPERT_ORDER:
            tokens = arrays[f"tokens_r{replica}_{expert}"]
            logits = arrays[f"logits_r{replica}_{expert}"]
            if (
                list(tokens.shape[1:]) != manifest["allocation"][expert]
                or tokens.shape[0] != manifest["event_count"]
                or logits.shape != (manifest["event_count"], 10)
                or not np.isfinite(tokens).all()
                or not np.isfinite(logits).all()
            ):
                raise ValueError("native fusion cache semantic arrays differ")
    return manifest, arrays


class NativeFusionDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    def __init__(self, manifest: Mapping[str, Any], arrays: Mapping[str, Any]):
        _require_torch()
        self.manifest = dict(manifest)
        self.arrays = arrays
        self.identities = tuple(str(value) for value in arrays["identities"])
        self.labels = np.asarray(arrays["labels"], dtype=np.int64)
        self.zero_based_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) <= 0:
            raise ValueError("native fusion epoch is one-based")
        self.zero_based_epoch = int(epoch) - 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        replica = replica_for(
            policy=self.manifest["realization_policy"],
            logical_role=self.manifest["split"],
            epoch=self.zero_based_epoch,
            canonical_identity=self.identities[index],
        )
        return {
            "identity": self.identities[index],
            "label": self.labels[index],
            "replica_id": replica,
            "token_banks": {
                name: self.arrays[f"tokens_r{replica}_{name}"][index]
                for name in EXPERT_ORDER
            },
            "expert_logits": {
                name: self.arrays[f"logits_r{replica}_{name}"][index]
                for name in EXPERT_ORDER
            },
        }


def _collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    module = _require_torch()
    return {
        "identities": [row["identity"] for row in rows],
        "labels": module.as_tensor([row["label"] for row in rows]).long(),
        "replica_ids": module.as_tensor(
            [row["replica_id"] for row in rows]
        ).long(),
        "token_banks": {
            name: module.from_numpy(
                np.stack([row["token_banks"][name] for row in rows])
            ).float()
            for name in EXPERT_ORDER
        },
        "expert_logits": {
            name: module.from_numpy(
                np.stack([row["expert_logits"][name] for row in rows])
            ).float()
            for name in EXPERT_ORDER
        },
    }


def make_native_fusion_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    module = _require_torch()
    manifest, arrays = load_native_fusion_cache(manifest_path)
    dataset = NativeFusionDataset(manifest, arrays)
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
class NativeFusionTrainingConfig:
    seed: int
    variant: str = "HF_NATIVE"
    realization_policy: str = "R_MULTI"
    maximum_epochs: int = 40
    batch_size: int = 512
    learning_rate: float = 5.0e-4
    minimum_learning_rate: float = 1.0e-5
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.variant not in {
            "HF_NATIVE",
            "HF_TRAINED_LOGIT",
            "HF_7X_UNBIASED_TOKEN_FUSION",
        }:
            raise ValueError("native fusion trainer received untrainable variant")
        if self.realization_policy not in REALIZATION_POLICIES:
            raise ValueError("native fusion realization policy is unknown")
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("native fusion campaign profile is unknown")
        if min(self.maximum_epochs, self.batch_size) <= 0:
            raise ValueError("native fusion training integers must be positive")
        if self.campaign_profile == "production" and (
            self.seed not in {101, 202, 303}
            or self.maximum_epochs != 40
            or self.batch_size != 512
        ):
            raise ValueError("native fusion production protocol drifted")

    def artifact(
        self, *, native_fusion_contract_sha256: str, global_determinism_sha256: str
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": NATIVE_FUSION_TRAINING_CONTRACT,
                "schema_version": 1,
                "config": asdict(self),
                "native_fusion_contract_sha256": require_sha256(
                    native_fusion_contract_sha256,
                    name="native_fusion_contract_sha256",
                ),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "experts_frozen": True,
                "offline_targets_permitted": False,
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


def _evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    module = _require_torch()
    precision = _precision(device)
    model.eval()
    logits, labels = [], []
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with module.autocast(
                device_type=module.device(device).type,
                dtype=precision["dtype"],
                enabled=precision["enabled"],
            ):
                output = model(
                    token_banks=batch["token_banks"],
                    expert_logits=batch["expert_logits"],
                )
            if not bool(module.isfinite(output).all()):
                raise FloatingPointError("native fusion validation is nonfinite")
            logits.append(output.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return evaluate_classification(
        np.concatenate(logits), np.concatenate(labels), split="val_stop"
    )


def evaluate_native_hlt_fusion(
    *,
    model: Any,
    manifest_path: str | Path,
    batch_size: int = 512,
    device: str | Any = "cpu",
    split: str = "val_stop",
) -> dict[str, Any]:
    """Evaluate trainable and parameter-free native fusion controls alike."""
    module = _require_torch()
    manifest, _ = load_native_fusion_cache(manifest_path)
    if manifest["split"] != split:
        raise ValueError("native fusion evaluation split differs")
    loader = make_native_fusion_loader(
        manifest_path, batch_size=batch_size, seed=0, training=False
    )
    resolved = module.device(device)
    model.to(resolved)
    precision = _precision(resolved)
    model.eval()
    logits, labels = [], []
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, resolved)
            with module.autocast(
                device_type=resolved.type,
                dtype=precision["dtype"],
                enabled=precision["enabled"],
            ):
                output = model(
                    token_banks=batch["token_banks"],
                    expert_logits=batch["expert_logits"],
                )
            if not bool(module.isfinite(output).all()):
                raise FloatingPointError("native fusion evaluation is nonfinite")
            logits.append(output.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return evaluate_classification(
        np.concatenate(logits), np.concatenate(labels), split=split
    )


def train_native_hlt_fusion(
    *,
    model: Any,
    model_train_manifest: str | Path,
    val_stop_manifest: str | Path,
    output_dir: str | Path,
    run_id: str,
    run_registry_sha256: str,
    native_fusion_contract_sha256: str,
    global_determinism_sha256: str,
    config: NativeFusionTrainingConfig,
    training_split: str = "model_train",
    device: str | Any = "cpu",
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    registration_path = Path(output_dir) / "fusion_registration.json"
    train_meta, _ = load_native_fusion_cache(model_train_manifest)
    val_meta, _ = load_native_fusion_cache(val_stop_manifest)
    if (
        training_split not in {"model_train", "scale_train"}
        or train_meta["split"] != training_split
        or val_meta["split"] != "val_stop"
        or train_meta["pipeline_seed"] != config.seed
        or val_meta["pipeline_seed"] != config.seed
    ):
        raise ValueError("native fusion split/seed lineage differs")
    if (
        train_meta["realization_policy"] != config.realization_policy
        or val_meta["realization_policy"] != config.realization_policy
    ):
        raise ValueError("native fusion realization policy differs")
    for key in ("shape_id", "allocation", "expert_registration_hashes"):
        if train_meta[key] != val_meta[key]:
            raise ValueError("native fusion cache lineage differs")
    contract = config.artifact(
        native_fusion_contract_sha256=native_fusion_contract_sha256,
        global_determinism_sha256=global_determinism_sha256,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registry_sha = require_sha256(
        run_registry_sha256, name="run_registry_sha256"
    )
    if registration_path.exists():
        registration = load_hashed_json(
            registration_path,
            expected_contract=NATIVE_FUSION_REGISTRATION_CONTRACT,
        )
        expected = {
            "run_id": run_id,
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": registry_sha,
            "model_train_cache_sha256": train_meta["content_hash"],
            "val_stop_cache_sha256": val_meta["content_hash"],
        }
        if any(registration.get(key) != value for key, value in expected.items()):
            raise ValueError("reusable native fusion registration lineage differs")
        checkpoint = root / "best_model_val.pt"
        if (
            not checkpoint.is_file()
            or _file_sha256(checkpoint) != registration["checkpoint_sha256"]
        ):
            raise ValueError("reusable native fusion checkpoint bytes differ")
        curves = load_hashed_json(
            root / "training_curves.json",
            expected_contract=NATIVE_FUSION_CURVES_CONTRACT,
        )
        if curves["content_hash"] != registration["training_curves_sha256"]:
            raise ValueError("reusable native fusion curves differ")
        return registration
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError("production native HLT fusion requires GH200 BF16")
    model.to(resolved)
    precision = _precision(resolved)
    train_loader = make_native_fusion_loader(
        model_train_manifest,
        batch_size=config.batch_size,
        seed=config.seed,
        training=True,
    )
    val_loader = make_native_fusion_loader(
        val_stop_manifest,
        batch_size=config.batch_size,
        seed=0,
        training=False,
    )
    optimizer = module.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.batch_size,
        gradient_accumulation_steps=1,
    )
    rows, best_state, update = [], None, 0
    for epoch in range(1, config.maximum_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for raw in train_loader:
            batch = _move(raw, resolved)
            optimizer.zero_grad(set_to_none=True)
            with module.autocast(
                device_type=resolved.type,
                dtype=precision["dtype"],
                enabled=precision["enabled"],
            ):
                output = model(
                    token_banks=batch["token_banks"],
                    expert_logits=batch["expert_logits"],
                )
                loss = module.nn.functional.cross_entropy(
                    output, batch["labels"]
                )
            if not bool(module.isfinite(loss)):
                raise FloatingPointError("native fusion loss is nonfinite")
            loss.backward()
            norm = module.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(module.isfinite(norm)):
                raise FloatingPointError("native fusion gradient is nonfinite")
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
        metrics = _evaluate(model, val_loader, resolved)
        rows.append(
            {
                "epoch": epoch,
                "val_stop": {
                    "accuracy": metrics["accuracy"],
                    "cross_entropy": metrics["cross_entropy"],
                },
            }
        )
        selected = preferred_expert_epoch(rows)
        if selected["epoch"] == epoch:
            best_state = {
                name: value.detach().cpu().clone()
                if isinstance(value, module.Tensor)
                else value
                for name, value in model.state_dict().items()
            }
    if update != counts["total_optimizer_updates"]:
        raise RuntimeError("native fusion optimizer update count drifted")
    selected = preferred_expert_epoch(rows)
    if best_state is None:
        raise RuntimeError("native fusion training retained no checkpoint")
    model.load_state_dict(best_state, strict=True)
    metrics = _evaluate(model, val_loader, resolved)
    curves = with_content_hash(
        {
            "contract": NATIVE_FUSION_CURVES_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "rows": rows,
            "selected_epoch": int(selected["epoch"]),
            "optimizer_update_counts": counts,
            "fixed_budget_completed": len(rows) == config.maximum_epochs,
            "performance_based_termination": False,
            "precision_mode": precision["mode"],
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    checkpoint = root / "best_model_val.pt"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".best_model_val.", suffix=".tmp", dir=root
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        module.save(
            {
                "run_id": run_id,
                "selected_epoch": selected["epoch"],
                "model_state_dict": best_state,
                "training_contract_sha256": contract["content_hash"],
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    registration = with_content_hash(
        {
            "contract": NATIVE_FUSION_REGISTRATION_CONTRACT,
            "schema_version": 1,
            "run_id": run_id,
            "variant": config.variant,
            "pipeline_seed": config.seed,
            "realization_policy": config.realization_policy,
            "shape_id": train_meta["shape_id"],
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": registry_sha,
            "model_train_cache_sha256": train_meta["content_hash"],
            "val_stop_cache_sha256": val_meta["content_hash"],
            "checkpoint_sha256": _file_sha256(checkpoint),
            "training_curves_sha256": curves["content_hash"],
            "val_stop_metrics": metrics,
            "selected_epoch": selected["epoch"],
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": True,
            "offline_targets_consumed": False,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(root / "fusion_registration.json", registration)
    return registration


__all__ = [
    "NATIVE_FUSION_CACHE_CONTRACT",
    "NATIVE_FUSION_CONTRACT",
    "NATIVE_FUSION_VARIANTS",
    "NativeFusionDataset",
    "NativeFusionTrainingConfig",
    "build_native_fusion_contract",
    "build_native_fusion_model",
    "evaluate_native_hlt_fusion",
    "load_native_fusion_cache",
    "make_native_fusion_loader",
    "publish_native_fusion_cache",
    "train_native_hlt_fusion",
]
