"""Training and evaluation runner for Canonical Multi-Scale Jet State variants.

This module is intentionally pragmatic: it gives Step 10 a real backend that
can train/evaluate the A0-G3 graph, save checkpoints, emit prediction caches for
fusion jobs, and write provenance-rich reports.  A few plan-level variants are
operationally approximated here rather than silently faked; those limitations
are recorded in each run report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import (
    amp_autocast_context,
    amp_grad_scaler,
    require_torch,
    resolve_device,
    set_training_seed,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.independent_fusion import metrics_from_logits
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_split_manifest, manifest_hash

from .cache import (
    CANONICAL_STATE_PHI_HLT_SOURCE,
    CANONICAL_STATE_PHI_OFFLINE_SOURCE,
    load_canonical_phi_cache,
)
from .config import (
    CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
    CANONICAL_STATE_HLT_PROFILE,
    CANONICAL_STATE_HLT_PROFILE_VERSION,
    CANONICAL_STATE_LABEL_FILTER,
)
from .losses import CanonicalStateLossWeights, compute_canonical_state_losses
from .predictor import build_canonical_state_residual_predictor
from .tagger import (
    STATE_CONTEXT_DELTA_PHI,
    STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
    STATE_CONTEXT_ORACLE_PHI_OFF,
    STATE_CONTEXT_PARTICLES_ONLY,
    CanonicalStateConditionedParT,
    CanonicalStateTaggerConfig,
)
from .training import (
    apply_canonical_state_train_phase,
    build_canonical_state_training_schedule,
    canonical_state_optimizer_group_specs,
    canonical_state_should_skip_batch,
)
from .variants import (
    CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
    FINAL_TEST_POLICY_MODEL_VAL_ONLY,
    FINAL_TEST_POLICY_REPORT_ONLY,
    FINAL_TEST_POLICY_STACK_ONLY,
    MODEL_KIND_ANALYSIS_REPORT,
    MODEL_KIND_AV10_FEATURE_MLP,
    MODEL_KIND_LOGIT_FUSION,
    MODEL_KIND_ORACLE_DIAGNOSTIC,
    MODEL_KIND_PART_BASELINE,
    MODEL_KIND_PARTICLE_VIEW_FUSION,
    MODEL_KIND_SEED_ENSEMBLE,
    MODEL_KIND_STATE_CONDITIONED_PART,
    MODEL_KIND_STATE_ONLY_TAGGER,
    MODEL_KIND_STATE_PREDICTOR_ONLY,
    MODEL_KIND_STATE_TOKEN_FUSION,
    WARM_START_NONE,
    WARM_START_PRETRAINED_PREDICTOR,
    CanonicalStateVariantSpec,
    canonical_state_variant_spec,
)


CANONICAL_STATE_VARIANT_RUNNER_CONTRACT = "canonical_state_variant_real_runner_v1"
PREDICTION_CACHE_CONTRACT = "canonical_state_prediction_cache_v1"


@dataclass(frozen=True)
class CanonicalStateVariantRunConfig:
    """Runtime configuration for one canonical-state variant job."""

    run_id: str
    output_dir: str | Path
    manifest: str | Path
    hlt_cache_dir: str | Path
    phi_hlt_cache_dir: str | Path
    phi_offline_cache_dir: str | Path | None = None
    baseline_checkpoint: str | Path | None = None
    variant_root: str | Path | None = None
    confirm_final_test: bool = False
    seed: int = 10101
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    warmup_epochs: int = 2
    adapter_warmup_epochs: int = 2
    part_lr: float = 3.0e-5
    adapter_lr: float = 3.0e-4
    predictor_lr: float = 3.0e-4
    head_lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 4
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_train_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    model_size: str = "base"

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def variant_root_path(self) -> Path:
        if self.variant_root:
            return Path(self.variant_root)
        return self.output_path().parent

    def max_jets_for_split(self, split: str) -> int | None:
        return {
            "model_train": self.max_train_jets,
            "model_val": self.max_val_jets,
            "stack_train": self.max_stack_train_jets,
            "stack_val": self.max_stack_val_jets,
            "final_test": self.max_final_test_jets,
        }.get(str(split))


class CanonicalStateCachedDataset:
    """One split with HLT particles plus Phi_hlt and optional Phi_offline."""

    def __init__(
        self,
        *,
        hlt_cache_dir: str | Path,
        phi_hlt_cache_dir: str | Path,
        phi_offline_cache_dir: str | Path | None,
        split: str,
        max_jets: int | None,
        include_phi_off: bool,
        allow_oracle_final_test: bool = False,
        expected_manifest_hash: str | None = None,
    ) -> None:
        self.split = str(split)
        self.hlt_view = load_cached_hlt_view(hlt_cache_dir, self.split, verify_hash=True)
        self.phi_hlt = load_canonical_phi_cache(
            phi_hlt_cache_dir,
            self.split,
            source_view=CANONICAL_STATE_PHI_HLT_SOURCE,
            verify_hash=True,
        )
        if np.asarray(self.hlt_view.labels).shape[0] != np.asarray(self.phi_hlt.labels).shape[0]:
            raise ValueError(f"{self.split}: HLT and Phi_hlt row counts differ")
        if not np.array_equal(np.asarray(self.hlt_view.labels), np.asarray(self.phi_hlt.labels)):
            raise ValueError(f"{self.split}: HLT and Phi_hlt labels differ")
        if tuple(self.hlt_view.jet_ids) != tuple(self.phi_hlt.jet_ids):
            raise ValueError(f"{self.split}: HLT and Phi_hlt jet identities differ")
        self._validate_hlt_contract(expected_manifest_hash)

        self.phi_off = None
        if include_phi_off:
            if not phi_offline_cache_dir:
                raise ValueError(f"{self.split}: phi_offline_cache_dir is required")
            self.phi_off = load_canonical_phi_cache(
                phi_offline_cache_dir,
                self.split,
                source_view=CANONICAL_STATE_PHI_OFFLINE_SOURCE,
                verify_hash=True,
                allow_oracle_final_test=bool(allow_oracle_final_test),
            )
            if not np.array_equal(np.asarray(self.hlt_view.labels), np.asarray(self.phi_off.labels)):
                raise ValueError(f"{self.split}: HLT and Phi_offline labels differ")
            if tuple(self.hlt_view.jet_ids) != tuple(self.phi_off.jet_ids):
                raise ValueError(f"{self.split}: HLT and Phi_offline jet identities differ")

        n_rows = int(len(self.hlt_view.labels))
        self.limit = n_rows if max_jets is None else min(int(max_jets), n_rows)
        self.tokens = np.asarray(self.hlt_view.tokens[: self.limit], dtype=np.float32)
        self.mask = np.asarray(self.hlt_view.mask[: self.limit], dtype=bool)
        self.labels = np.asarray(self.hlt_view.labels[: self.limit], dtype=np.int64)
        self.phi_hlt_tokens = np.asarray(self.phi_hlt.phi_tokens[: self.limit], dtype=np.float32)
        self.phi_hlt_state_mask = np.asarray(self.phi_hlt.state_mask[: self.limit], dtype=bool)
        self.phi_off_tokens = (
            np.asarray(self.phi_off.phi_tokens[: self.limit], dtype=np.float32)
            if self.phi_off is not None
            else np.zeros_like(self.phi_hlt_tokens, dtype=np.float32)
        )
        self.phi_off_state_mask = (
            np.asarray(self.phi_off.state_mask[: self.limit], dtype=bool)
            if self.phi_off is not None
            else np.zeros_like(self.phi_hlt_state_mask, dtype=bool)
        )
        self.has_phi_off = self.phi_off is not None

    def _validate_hlt_contract(self, expected_manifest_hash: str | None) -> None:
        hlt_meta = dict(self.hlt_view.metadata)
        phi_meta = dict(self.phi_hlt.metadata)
        source_manifest = hlt_meta.get("source_manifest_hash") or phi_meta.get("source_manifest_hash")
        if expected_manifest_hash and source_manifest != expected_manifest_hash:
            raise ValueError(
                f"{self.split}: source_manifest_hash {source_manifest!r} does not match active manifest {expected_manifest_hash!r}"
            )
        if hlt_meta.get("hlt_profile") != CANONICAL_STATE_HLT_PROFILE:
            raise ValueError(f"{self.split}: HLT profile {hlt_meta.get('hlt_profile')!r} is not {CANONICAL_STATE_HLT_PROFILE!r}")
        if hlt_meta.get("hlt_profile_version") != CANONICAL_STATE_HLT_PROFILE_VERSION:
            raise ValueError(
                f"{self.split}: HLT profile version {hlt_meta.get('hlt_profile_version')!r} is not "
                f"{CANONICAL_STATE_HLT_PROFILE_VERSION!r}"
            )
        strength = hlt_meta.get("hlt_degradation_strength")
        if strength is None or abs(float(strength) - float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH)) > 1.0e-12:
            raise ValueError(
                f"{self.split}: HLT strength {strength!r} is not {CANONICAL_STATE_HLT_DEGRADATION_STRENGTH:g}"
            )
        hlt_content_hash = hlt_meta.get("hlt_content_hash")
        phi_source_hash = phi_meta.get("source_cache_hash") or phi_meta.get("hlt_content_hash")
        if not hlt_content_hash:
            raise ValueError(f"{self.split}: HLT cache metadata is missing hlt_content_hash")
        if not phi_source_hash:
            raise ValueError(f"{self.split}: Phi_hlt metadata is missing source_cache_hash")
        if str(phi_meta.get("source_cache_hash_name", "hlt_content_hash")) != "hlt_content_hash":
            raise ValueError(
                f"{self.split}: Phi_hlt source_cache_hash_name is "
                f"{phi_meta.get('source_cache_hash_name')!r}, expected 'hlt_content_hash'"
            )
        if str(phi_source_hash) != str(hlt_content_hash):
            raise ValueError(
                f"{self.split}: Phi_hlt source_cache_hash does not match loaded HLT hlt_content_hash: "
                f"{phi_source_hash} != {hlt_content_hash}"
            )

    def __len__(self) -> int:
        return int(self.limit)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": self.labels[index],
            "phi_hlt": self.phi_hlt_tokens[index],
            "phi_hlt_state_mask": self.phi_hlt_state_mask[index],
            "phi_off": self.phi_off_tokens[index],
            "phi_off_state_mask": self.phi_off_state_mask[index],
            "has_phi_off": np.bool_(self.has_phi_off),
        }

    def metadata_report(self) -> dict[str, Any]:
        hlt_meta = dict(self.hlt_view.metadata)
        phi_hlt_meta = dict(self.phi_hlt.metadata)
        phi_off_meta = dict(self.phi_off.metadata) if self.phi_off is not None else {}
        return {
            "split": self.split,
            "n_jets": int(self.limit),
            "source_manifest_hash": hlt_meta.get("source_manifest_hash") or phi_hlt_meta.get("source_manifest_hash"),
            "hlt_content_hash": hlt_meta.get("hlt_content_hash"),
            "jet_identity_hash": hlt_meta.get("jet_identity_hash") or phi_hlt_meta.get("jet_identity_hash"),
            "hlt_profile": hlt_meta.get("hlt_profile"),
            "hlt_profile_version": hlt_meta.get("hlt_profile_version"),
            "hlt_degradation_strength": hlt_meta.get("hlt_degradation_strength"),
            "phi_hlt_content_hash": phi_hlt_meta.get("phi_content_hash"),
            "phi_hlt_source_cache_hash": phi_hlt_meta.get("source_cache_hash"),
            "phi_offline_content_hash": phi_off_meta.get("phi_content_hash"),
            "phi_offline_source_cache_hash": phi_off_meta.get("source_cache_hash"),
            "has_phi_offline": bool(self.has_phi_off),
        }


def _collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "tokens": torch.from_numpy(np.stack([sample["tokens"] for sample in samples], axis=0)).float(),
        "mask": torch.from_numpy(np.stack([sample["mask"] for sample in samples], axis=0)).bool(),
        "labels": torch.from_numpy(np.asarray([sample["labels"] for sample in samples], dtype=np.int64)).long(),
        "phi_hlt": torch.from_numpy(np.stack([sample["phi_hlt"] for sample in samples], axis=0)).float(),
        "phi_hlt_state_mask": torch.from_numpy(np.stack([sample["phi_hlt_state_mask"] for sample in samples], axis=0)).bool(),
        "phi_off": torch.from_numpy(np.stack([sample["phi_off"] for sample in samples], axis=0)).float(),
        "phi_off_state_mask": torch.from_numpy(np.stack([sample["phi_off_state_mask"] for sample in samples], axis=0)).bool(),
        "has_phi_off": torch.from_numpy(np.asarray([sample["has_phi_off"] for sample in samples], dtype=bool)).bool(),
    }


def _loader(dataset: CanonicalStateCachedDataset, *, batch_size: int, num_workers: int, shuffle: bool) -> Any:
    torch = require_torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(torch.cuda.is_available()),
        collate_fn=_collate,
    )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _loss_weights_without_unavailable_terms(
    weights: CanonicalStateLossWeights,
    *,
    teacher_logits_available: bool,
    phi_off_available: bool,
    delta_available: bool,
    disabled: set[str],
) -> CanonicalStateLossWeights:
    data = weights.to_dict()
    data.pop("contract", None)
    data.pop("active_terms", None)
    if float(data.get("logit_kd", 0.0)) > 0.0 and not bool(teacher_logits_available):
        data["logit_kd"] = 0.0
        disabled.add("logit_kd_no_teacher_logits_cache")
    state_target_terms = ("state_huber", "state_l1", "uncertainty_state")
    if not bool(phi_off_available):
        for name in state_target_terms:
            if float(data.get(name, 0.0)) > 0.0:
                data[name] = 0.0
                disabled.add(f"{name}_no_phi_offline_on_split")
    if not bool(delta_available):
        for name in ("delta_norm", "smoothness", *state_target_terms):
            if float(data.get(name, 0.0)) > 0.0:
                data[name] = 0.0
                disabled.add(f"{name}_no_delta_prediction")
    return CanonicalStateLossWeights(**data)


def _state_mask_for_delta(delta_phi: Any | None, batch: Mapping[str, Any]) -> Any | None:
    if delta_phi is None:
        return None
    torch = require_torch()
    mask = batch["phi_hlt_state_mask"].to(device=delta_phi.device, dtype=torch.bool)
    if bool(batch["has_phi_off"].all().detach().cpu().item()):
        mask = mask & batch["phi_off_state_mask"].to(device=delta_phi.device, dtype=torch.bool)
    return mask


def _oracle_delta_for_batch(spec: CanonicalStateVariantSpec, batch: Mapping[str, Any], *, split: str) -> Any | None:
    if not bool(spec.oracle_inputs_allowed) or spec.tagger_mode != STATE_CONTEXT_DELTA_PHI:
        return None
    if str(split) == "final_test":
        raise ValueError(f"{spec.run_id}: oracle delta_phi_true is not allowed on final_test")
    if not bool(batch["has_phi_off"].all().detach().cpu().item()):
        raise ValueError(f"{spec.run_id}/{split}: oracle delta_phi_true requires Phi_offline")
    return batch["phi_off"] - batch["phi_hlt"]


def _state_context_mask_kwargs(spec: CanonicalStateVariantSpec, batch: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    base_mask = batch["phi_hlt_state_mask"]
    kwargs: dict[str, Any] = {"state_mask": base_mask}
    if not bool(spec.oracle_inputs_allowed) or str(split) == "final_test":
        return kwargs
    if not bool(batch["has_phi_off"].all().detach().cpu().item()):
        return kwargs
    offline_mask = batch["phi_off_state_mask"]
    if spec.tagger_mode == STATE_CONTEXT_ORACLE_PHI_OFF:
        kwargs["state_mask"] = offline_mask
    elif spec.tagger_mode == STATE_CONTEXT_DELTA_PHI:
        kwargs["delta_state_mask"] = offline_mask
    return kwargs


def _save_predictions(output_dir: Path, split: str, logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    path = output_dir / f"{split}_predictions.npz"
    np.savez_compressed(
        path,
        logits=np.asarray(logits, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )
    return {
        "prediction_cache_contract": PREDICTION_CACHE_CONTRACT,
        "split": str(split),
        "path": str(path),
        "n_jets": int(len(labels)),
        "has_logits": True,
    }


def _load_predictions(run_root: Path, run_id: str, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = run_root / run_id / f"{split}_predictions.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return data["logits"].astype(np.float32, copy=False), data["labels"].astype(np.int64, copy=False)


def _metrics_from_arrays(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    metrics = metrics_from_logits(np.asarray(logits, dtype=np.float32), np.asarray(labels, dtype=np.int64))
    # Canonical report looks for macro_auc; independent_fusion calls it macro_ovr_auc.
    if "macro_auc" not in metrics and "macro_ovr_auc" in metrics:
        metrics["macro_auc"] = metrics.get("macro_ovr_auc")
    return metrics


def _load_manifest_info(config: CanonicalStateVariantRunConfig) -> dict[str, Any]:
    manifest = load_split_manifest(config.manifest)
    return {
        "manifest_path": str(config.manifest),
        "manifest_hash": manifest_hash(manifest),
        "split_names": sorted(str(key) for key in manifest.splits),
    }


def _load_dataset(
    config: CanonicalStateVariantRunConfig,
    split: str,
    *,
    include_phi_off: bool,
    allow_oracle_final_test: bool = False,
) -> CanonicalStateCachedDataset:
    return CanonicalStateCachedDataset(
        hlt_cache_dir=config.hlt_cache_dir,
        phi_hlt_cache_dir=config.phi_hlt_cache_dir,
        phi_offline_cache_dir=config.phi_offline_cache_dir,
        split=split,
        max_jets=config.max_jets_for_split(split),
        include_phi_off=bool(include_phi_off),
        allow_oracle_final_test=bool(allow_oracle_final_test),
        expected_manifest_hash=_load_manifest_info(config)["manifest_hash"],
    )


def _common_provenance(
    *,
    config: CanonicalStateVariantRunConfig,
    spec: CanonicalStateVariantSpec,
    datasets: Mapping[str, CanonicalStateCachedDataset],
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    manifest = _load_manifest_info(config)
    model_val_dataset = datasets.get("model_val")
    model_val_meta = model_val_dataset.metadata_report() if model_val_dataset is not None else {}
    hlt_contract = {
        "hlt_profile": model_val_meta.get("hlt_profile"),
        "hlt_profile_version": model_val_meta.get("hlt_profile_version"),
        "hlt_degradation_strength": model_val_meta.get("hlt_degradation_strength"),
    }
    return {
        "runner_contract": CANONICAL_STATE_VARIANT_RUNNER_CONTRACT,
        "variant_contract": CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
        "ok": True,
        "run_id": spec.run_id,
        "title": spec.title,
        "tier": spec.tier,
        "model_kind": spec.model_kind,
        "spec": spec.to_dict(),
        "manifest": manifest,
        "hlt_input_contract": hlt_contract,
        "label_names": list(LABEL_NAMES),
        "label_filter": list(CANONICAL_STATE_LABEL_FILTER),
        "config": asdict(config),
        "implementation_limitations": list(limitations),
        "dataset_summaries": {split: dataset.metadata_report() for split, dataset in datasets.items()},
        "model_train_dataset": datasets["model_train"].metadata_report() if "model_train" in datasets else None,
        "model_val_dataset": datasets["model_val"].metadata_report() if "model_val" in datasets else None,
        "stack_train_dataset": datasets["stack_train"].metadata_report() if "stack_train" in datasets else None,
        "stack_val_dataset": datasets["stack_val"].metadata_report() if "stack_val" in datasets else None,
        "final_test_dataset": datasets["final_test"].metadata_report() if "final_test" in datasets else None,
        "model_val_hlt_content_hash": model_val_meta.get("hlt_content_hash"),
        "model_val_jet_identity_hash": model_val_meta.get("jet_identity_hash"),
    }


def _build_tagger_model(spec: CanonicalStateVariantSpec, config: CanonicalStateVariantRunConfig) -> CanonicalStateConditionedParT:
    mode = spec.tagger_mode or STATE_CONTEXT_PARTICLES_ONLY
    limitations: list[str] = []
    if spec.model_kind == MODEL_KIND_AV10_FEATURE_MLP:
        mode = STATE_CONTEXT_FEATURE_MLP_PLUS_STATE
        limitations.append("A2 uses canonical-state feature_mlp_plus_state adapter path, not the separate AV10 trainer.")
    tagger_config = CanonicalStateTaggerConfig(
        mode=mode,
        num_classes=len(LABEL_NAMES),
        part_model_size=str(config.model_size),
        predictor_config={"variant": spec.predictor_variant} if spec.predictor_variant else None,
    )
    model = CanonicalStateConditionedParT(tagger_config)
    model._canonical_runner_limitations = limitations  # type: ignore[attr-defined]
    return model


def _load_checkpoint_if_available(model: Any, checkpoint: str | Path | None) -> dict[str, Any]:
    if not checkpoint:
        return {"loaded": False, "reason": "no_checkpoint"}
    path = Path(checkpoint)
    if not path.exists():
        return {"loaded": False, "reason": f"missing_checkpoint:{path}"}
    torch = require_torch()
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if state is None and isinstance(payload, Mapping):
        state = payload.get("state_dict") or payload
    if not isinstance(state, Mapping):
        return {"loaded": False, "reason": f"checkpoint_has_no_state_dict:{path}"}
    keys = tuple(str(key) for key in state.keys())
    if any(key.startswith("part_model.") for key in keys):
        result = model.load_state_dict(state, strict=False)
        return {
            "loaded": True,
            "path": str(path),
            "target": "canonical_state_wrapper",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    part_model = getattr(model, "part_model", None)
    if part_model is not None:
        result = part_model.load_state_dict(state, strict=False)
        return {
            "loaded": True,
            "path": str(path),
            "target": "part_model",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    result = model.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "target": "model",
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _load_predictor_checkpoint_if_available(model: Any, checkpoint: str | Path | None) -> dict[str, Any]:
    if not checkpoint:
        return {"loaded": False, "reason": "no_predictor_checkpoint"}
    path = Path(checkpoint)
    if not path.exists():
        return {"loaded": False, "reason": f"missing_predictor_checkpoint:{path}"}
    predictor = getattr(model, "state_predictor", None)
    if predictor is None:
        return {"loaded": False, "reason": "model_has_no_state_predictor"}
    torch = require_torch()
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if state is None and isinstance(payload, Mapping):
        state = payload.get("state_dict") or payload
    if not isinstance(state, Mapping):
        return {"loaded": False, "reason": f"checkpoint_has_no_state_dict:{path}"}
    result = predictor.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "target": "state_predictor",
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _stable_seed_offset(run_id: str) -> int:
    return int(sum((index + 1) * ord(char) for index, char in enumerate(str(run_id))) % 10000)


def _set_trainable(module: Any | None, trainable: bool) -> None:
    if module is None or not hasattr(module, "parameters"):
        return
    for parameter in module.parameters():
        parameter.requires_grad_(bool(trainable))
    if hasattr(module, "train"):
        module.train(bool(trainable))


def _apply_variant_trainability(model: Any, spec: CanonicalStateVariantSpec, phase: Any | None = None) -> None:
    """Remove dormant module groups from optimizer accounting for controls."""

    if spec.tagger_mode == STATE_CONTEXT_PARTICLES_ONLY or spec.model_kind == MODEL_KIND_PART_BASELINE:
        _set_trainable(getattr(model, "state_predictor", None), False)
        _set_trainable(getattr(model, "state_encoder", None), False)
        _set_trainable(getattr(model, "state_adapter", None), False)
        _set_trainable(getattr(model, "feature_adapter", None), False)
        _set_trainable(getattr(model, "state_only_head", None), False)
    if spec.model_kind == MODEL_KIND_STATE_ONLY_TAGGER:
        _set_trainable(getattr(model, "part_model", None), False)
        _set_trainable(getattr(model, "state_predictor", None), False)
        _set_trainable(getattr(model, "state_adapter", None), False)
        _set_trainable(getattr(model, "feature_adapter", None), False)
        _set_trainable(getattr(model, "state_encoder", None), True)
        _set_trainable(getattr(model, "state_only_head", None), True)
    if spec.tagger_mode == STATE_CONTEXT_FEATURE_MLP_PLUS_STATE or spec.model_kind in {
        MODEL_KIND_AV10_FEATURE_MLP,
        MODEL_KIND_PARTICLE_VIEW_FUSION,
    }:
        train_feature = True if phase is None else bool(getattr(phase, "state_adapter_trainable", False) or getattr(phase, "feature_adapter_trainable", False))
        _set_trainable(getattr(model, "feature_adapter", None), train_feature)


def _loss_for_tagger_batch(
    *,
    output: Any,
    batch: Mapping[str, Any],
    split: str,
    weights: CanonicalStateLossWeights,
    disabled_terms: set[str],
) -> Any:
    predictor_output = getattr(output, "predictor_output", None)
    delta_phi = getattr(predictor_output, "delta_phi", None) if predictor_output is not None else None
    log_sigma = getattr(predictor_output, "log_sigma", None) if predictor_output is not None else None
    effective_weights = _loss_weights_without_unavailable_terms(
        weights,
        teacher_logits_available=False,
        phi_off_available=bool(batch["has_phi_off"].all().detach().cpu().item()),
        delta_available=delta_phi is not None,
        disabled=disabled_terms,
    )
    return compute_canonical_state_losses(
        logits=output.logits,
        labels=batch["labels"],
        weights=effective_weights,
        split=split,
        phi_hlt=batch["phi_hlt"],
        phi_off=batch["phi_off"] if bool(batch["has_phi_off"].all().detach().cpu().item()) else None,
        delta_phi_pred=delta_phi,
        log_sigma=log_sigma,
        teacher_logits=None,
        state_mask=_state_mask_for_delta(delta_phi, batch),
    )


def _evaluate_tagger(
    model: Any,
    dataset: CanonicalStateCachedDataset,
    *,
    config: CanonicalStateVariantRunConfig,
    spec: CanonicalStateVariantSpec,
    split: str,
    device: Any,
    save_predictions: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    model.eval()
    loader = _loader(dataset, batch_size=config.eval_batch_size, num_workers=config.num_workers, shuffle=False)
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    diagnostics: dict[str, Any] = {}
    disabled_terms: set[str] = set()
    loss_values: list[float] = []
    torch = require_torch()
    with torch.no_grad():
        for batch_cpu in loader:
            batch = _move_batch(batch_cpu, device)
            allow_oracle = bool(spec.oracle_inputs_allowed and split != "final_test")
            output = model(
                batch["tokens"],
                batch["mask"],
                phi_hlt=batch["phi_hlt"],
                delta_phi=_oracle_delta_for_batch(spec, batch, split=split),
                **_state_context_mask_kwargs(spec, batch, split=split),
                phi_off=batch["phi_off"] if bool(batch["has_phi_off"].all().detach().cpu().item()) else None,
                split=split,
                allow_oracle_context=allow_oracle,
            )
            loss = _loss_for_tagger_batch(
                output=output,
                batch=batch,
                split=split,
                weights=spec.loss_weights,
                disabled_terms=disabled_terms,
            )
            loss_values.append(float(loss.total.detach().cpu().item()))
            logits_chunks.append(output.logits.detach().cpu().float().numpy())
            labels_chunks.append(batch["labels"].detach().cpu().long().numpy())
            diagnostics = dict(output.diagnostics)
    logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)
    labels = np.concatenate(labels_chunks, axis=0) if labels_chunks else np.zeros((0,), dtype=np.int64)
    metrics = _metrics_from_arrays(logits, labels)
    metrics["loss"] = float(np.mean(loss_values)) if loss_values else None
    metrics["disabled_loss_terms"] = sorted(disabled_terms)
    pred_meta = _save_predictions(config.output_path(), split, logits, labels) if save_predictions else None
    return metrics, pred_meta, diagnostics


def _train_tagger(config: CanonicalStateVariantRunConfig, spec: CanonicalStateVariantSpec) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(int(config.seed) + _stable_seed_offset(spec.run_id))
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    needs_phi_off_train = bool(spec.oracle_inputs_allowed) or any(
        float(getattr(spec.loss_weights, name)) > 0.0
        for name in ("state_huber", "state_l1", "uncertainty_state")
    )
    datasets = {
        "model_train": _load_dataset(config, "model_train", include_phi_off=needs_phi_off_train),
        "model_val": _load_dataset(config, "model_val", include_phi_off=needs_phi_off_train or spec.oracle_inputs_allowed),
        "stack_val": _load_dataset(config, "stack_val", include_phi_off=bool(spec.oracle_inputs_allowed)),
    }
    if spec.allows_primary_final_test() and bool(config.confirm_final_test):
        datasets["final_test"] = _load_dataset(config, "final_test", include_phi_off=False)
    if spec.final_test_policy == FINAL_TEST_POLICY_MODEL_VAL_ONLY:
        datasets["final_test"] = datasets["model_val"]

    model = _build_tagger_model(spec, config).to(device)
    limitations = list(getattr(model, "_canonical_runner_limitations", []))
    warm_start_report = {"loaded": False, "reason": "warm_start_disabled"}
    predictor_warm_start_report = {"loaded": False, "reason": "not_requested"}
    if spec.warm_start_policy != WARM_START_NONE and config.baseline_checkpoint:
        warm_start_report = _load_checkpoint_if_available(model, config.baseline_checkpoint)
        if not bool(warm_start_report.get("loaded")):
            limitations.append(f"warm_start_requested_but_not_loaded:{warm_start_report.get('reason')}")
    if spec.warm_start_policy == WARM_START_PRETRAINED_PREDICTOR:
        c0_path = config.variant_root_path() / "C0" / "best_model_val.pt"
        predictor_warm_start_report = _load_predictor_checkpoint_if_available(model, c0_path)
        if not bool(predictor_warm_start_report.get("loaded")):
            limitations.append(
                f"pretrained_predictor_requested_but_not_loaded:{predictor_warm_start_report.get('reason')}"
            )

    schedule = build_canonical_state_training_schedule(
        spec.schedule,
        total_epochs=int(config.epochs),
        warmup_epochs=int(config.warmup_epochs),
        adapter_warmup_epochs=int(config.adapter_warmup_epochs),
        adapter_lr=float(config.adapter_lr),
        part_lr=float(config.part_lr),
        head_lr=float(config.head_lr),
        predictor_lr=float(config.predictor_lr),
        loss_weights=spec.loss_weights,
    )
    train_loader = _loader(
        datasets["model_train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    scaler = amp_grad_scaler(bool(config.amp and device.type == "cuda"))
    best_metric = -float("inf")
    best_epoch = 0
    best_path = output_dir / "best_model_val.pt"
    epoch_rows: list[dict[str, Any]] = []
    nonfinite_batches = 0
    patience_left = int(config.early_stop_patience)
    disabled_loss_terms: set[str] = set()

    for epoch in range(1, int(config.epochs) + 1):
        phase = schedule.phase_for_epoch(epoch)
        model.train()
        phase_report = apply_canonical_state_train_phase(model, phase)
        _apply_variant_trainability(model, spec, phase)
        opt_groups = canonical_state_optimizer_group_specs(model, phase)
        optimizer = None
        if opt_groups:
            optimizer = torch.optim.AdamW(
                [{key: value for key, value in group.items() if key != "name"} for group in opt_groups],
                weight_decay=float(config.weight_decay),
            )
        train_logits: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        train_losses: list[float] = []
        for batch_cpu in train_loader:
            if optimizer is None:
                break
            batch = _move_batch(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast_context(bool(config.amp and device.type == "cuda")):
                output = model(
                    batch["tokens"],
                    batch["mask"],
                    phi_hlt=batch["phi_hlt"],
                    delta_phi=_oracle_delta_for_batch(spec, batch, split="model_train"),
                    **_state_context_mask_kwargs(spec, batch, split="model_train"),
                    phi_off=batch["phi_off"] if bool(batch["has_phi_off"].all().detach().cpu().item()) else None,
                    split="model_train",
                )
                loss = _loss_for_tagger_batch(
                    output=output,
                    batch=batch,
                    split="model_train",
                    weights=phase.loss_weights,
                    disabled_terms=disabled_loss_terms,
                )
            if canonical_state_should_skip_batch(loss=loss.total, logits=output.logits):
                nonfinite_batches += 1
                continue
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.total.detach().cpu().item()))
            train_logits.append(output.logits.detach().cpu().float().numpy())
            train_labels.append(batch["labels"].detach().cpu().long().numpy())
        val_metrics, _, val_diag = _evaluate_tagger(
            model,
            datasets["model_val"],
            config=config,
            spec=spec,
            split="model_val",
            device=device,
            save_predictions=False,
        )
        train_metrics = {}
        if train_logits:
            train_metrics = _metrics_from_arrays(np.concatenate(train_logits, axis=0), np.concatenate(train_labels, axis=0))
            train_metrics["loss"] = float(np.mean(train_losses)) if train_losses else None
        score = float(val_metrics.get("accuracy") or 0.0)
        epoch_rows.append(
            {
                "epoch": epoch,
                "phase": phase_report,
                "train_metrics": train_metrics,
                "model_val_metrics": val_metrics,
                "model_val_diagnostics": val_diag,
                "nonfinite_batches_total": int(nonfinite_batches),
            }
        )
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            patience_left = int(config.early_stop_patience)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "run_id": spec.run_id,
                    "epoch": epoch,
                    "model_val_metrics": val_metrics,
                    "config": asdict(config),
                    "spec": spec.to_dict(),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_id": spec.run_id,
            "epoch": int(epoch_rows[-1]["epoch"] if epoch_rows else 0),
            "config": asdict(config),
            "spec": spec.to_dict(),
        },
        output_dir / "last.pt",
    )

    eval_splits = ["model_val", "stack_val"]
    if "final_test" in datasets and spec.allows_primary_final_test():
        eval_splits.append("final_test")
    metrics_by_split: dict[str, Any] = {}
    prediction_caches: dict[str, Any] = {}
    diagnostics_by_split: dict[str, Any] = {}
    for split in eval_splits:
        dataset = datasets[split]
        metrics, pred_meta, diagnostics = _evaluate_tagger(
            model,
            dataset,
            config=config,
            spec=spec,
            split=split,
            device=device,
            save_predictions=True,
        )
        metrics_by_split[split] = metrics
        diagnostics_by_split[split] = diagnostics
        if pred_meta is not None:
            prediction_caches[split] = pred_meta

    report = _common_provenance(config=config, spec=spec, datasets=datasets, limitations=limitations)
    report.update(
        {
            "warm_start": warm_start_report,
            "predictor_warm_start": predictor_warm_start_report,
            "training_schedule": schedule.to_dict(),
            "epoch_metrics": epoch_rows,
            "best_epoch": int(best_epoch),
            "best_model_val_metrics": metrics_by_split.get("model_val"),
            "model_val_metrics": metrics_by_split.get("model_val"),
            "stack_val_metrics": metrics_by_split.get("stack_val"),
            "final_test_metrics": metrics_by_split.get("final_test"),
            "prediction_caches": prediction_caches,
            "diagnostics": diagnostics_by_split,
            "disabled_loss_terms": sorted(disabled_loss_terms),
            "checkpoint": str(best_path),
            "last_checkpoint": str(output_dir / "last.pt"),
            "nonfinite_batches": int(nonfinite_batches),
            "final_test_evaluated": "final_test" in metrics_by_split,
        }
    )
    _write_json(output_dir / "training_curves.json", {"epochs": epoch_rows})
    _write_json(output_dir / "run_report.json", report)
    return report


def _masked_metric_mean(value: Any, state_mask: Any) -> float:
    torch = require_torch()
    mask = state_mask.to(device=value.device, dtype=value.dtype)
    while int(mask.ndim) < int(value.ndim):
        mask = mask.unsqueeze(-1)
    expanded_mask = torch.ones_like(value) * mask
    denom = expanded_mask.sum().clamp_min(1.0)
    return float((value * expanded_mask).sum().detach().cpu().item() / float(denom.detach().cpu().item()))


def _state_prediction_metrics(delta: Any, phi_hlt: Any, phi_off: Any, state_mask: Any) -> dict[str, Any]:
    torch = require_torch()
    state_mask = state_mask.to(device=delta.device, dtype=torch.bool)
    target = phi_off - phi_hlt
    err = delta - target
    return {
        "mse": _masked_metric_mean(err.square(), state_mask),
        "mae": _masked_metric_mean(err.abs(), state_mask),
        "delta_l2_mean": _masked_metric_mean(torch.linalg.vector_norm(delta, dim=-1), state_mask),
        "target_l2_mean": _masked_metric_mean(torch.linalg.vector_norm(target, dim=-1), state_mask),
        "valid_state_tokens_mean": float(state_mask.sum(dim=1).float().mean().detach().cpu().item()),
    }


def _evaluate_predictor(
    predictor: Any,
    dataset: CanonicalStateCachedDataset,
    *,
    config: CanonicalStateVariantRunConfig,
    split: str,
    device: Any,
) -> dict[str, Any]:
    predictor.eval()
    loader = _loader(dataset, batch_size=config.eval_batch_size, num_workers=config.num_workers, shuffle=False)
    torch = require_torch()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_cpu in loader:
            batch = _move_batch(batch_cpu, device)
            output = predictor(batch["tokens"], batch["mask"], batch["phi_hlt"], state_mask=batch["phi_hlt_state_mask"])
            rows.append(
                _state_prediction_metrics(
                    output.delta_phi,
                    batch["phi_hlt"],
                    batch["phi_off"],
                    _state_mask_for_delta(output.delta_phi, batch),
                )
            )
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([row[key] for row in rows if row.get(key) is not None])) if rows else None
        for key in keys
    } | {"n_jets": int(len(dataset)), "split": str(split)}


def _train_predictor(config: CanonicalStateVariantRunConfig, spec: CanonicalStateVariantSpec) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(int(config.seed) + _stable_seed_offset(spec.run_id))
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    datasets = {
        "model_train": _load_dataset(config, "model_train", include_phi_off=True),
        "model_val": _load_dataset(config, "model_val", include_phi_off=True),
        "stack_val": _load_dataset(config, "stack_val", include_phi_off=True),
    }
    predictor = build_canonical_state_residual_predictor(spec.predictor_variant or "P0").to(device)
    train_loader = _loader(datasets["model_train"], batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(config.predictor_lr), weight_decay=float(config.weight_decay))
    scaler = amp_grad_scaler(bool(config.amp and device.type == "cuda"))
    best_metric = float("inf")
    best_epoch = 0
    best_path = output_dir / "best_model_val.pt"
    epoch_rows: list[dict[str, Any]] = []
    nonfinite_batches = 0
    patience_left = int(config.early_stop_patience)
    disabled_loss_terms: set[str] = set()
    zero_logits = None

    for epoch in range(1, int(config.epochs) + 1):
        predictor.train()
        train_losses: list[float] = []
        for batch_cpu in train_loader:
            batch = _move_batch(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast_context(bool(config.amp and device.type == "cuda")):
                output = predictor(batch["tokens"], batch["mask"], batch["phi_hlt"], state_mask=batch["phi_hlt_state_mask"])
                if zero_logits is None or int(zero_logits.shape[0]) != int(batch["labels"].shape[0]):
                    zero_logits = torch.zeros((int(batch["labels"].shape[0]), len(LABEL_NAMES)), device=device)
                loss = compute_canonical_state_losses(
                    logits=zero_logits,
                    labels=batch["labels"],
                    weights=_loss_weights_without_unavailable_terms(
                        spec.loss_weights,
                        teacher_logits_available=False,
                        phi_off_available=True,
                        delta_available=True,
                        disabled=disabled_loss_terms,
                    ),
                    split="model_train",
                    phi_hlt=batch["phi_hlt"],
                    phi_off=batch["phi_off"],
                    delta_phi_pred=output.delta_phi,
                    log_sigma=output.log_sigma,
                    state_mask=_state_mask_for_delta(output.delta_phi, batch),
                )
            if canonical_state_should_skip_batch(loss=loss.total):
                nonfinite_batches += 1
                continue
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), float(config.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.total.detach().cpu().item()))
        val_metrics = _evaluate_predictor(predictor, datasets["model_val"], config=config, split="model_val", device=device)
        score = float(val_metrics.get("mse") if val_metrics.get("mse") is not None else float("inf"))
        epoch_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "model_val_state_prediction_metrics": val_metrics,
                "nonfinite_batches_total": int(nonfinite_batches),
            }
        )
        if score < best_metric:
            best_metric = score
            best_epoch = epoch
            patience_left = int(config.early_stop_patience)
            torch.save(
                {
                    "model_state_dict": predictor.state_dict(),
                    "run_id": spec.run_id,
                    "epoch": epoch,
                    "model_val_state_prediction_metrics": val_metrics,
                    "config": asdict(config),
                    "spec": spec.to_dict(),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_path.exists():
        predictor.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"], strict=True)
    state_metrics = {
        "model_val": _evaluate_predictor(predictor, datasets["model_val"], config=config, split="model_val", device=device),
        "stack_val": _evaluate_predictor(predictor, datasets["stack_val"], config=config, split="stack_val", device=device),
    }
    torch.save(
        {"model_state_dict": predictor.state_dict(), "run_id": spec.run_id, "config": asdict(config), "spec": spec.to_dict()},
        output_dir / "last.pt",
    )
    report = _common_provenance(config=config, spec=spec, datasets=datasets)
    report.update(
        {
            "best_epoch": int(best_epoch),
            "best_model_val_state_prediction_metrics": state_metrics["model_val"],
            "state_prediction_metrics": state_metrics,
            "model_val_state_prediction_metrics": state_metrics["model_val"],
            "stack_val_state_prediction_metrics": state_metrics["stack_val"],
            "epoch_metrics": epoch_rows,
            "disabled_loss_terms": sorted(disabled_loss_terms),
            "checkpoint": str(best_path),
            "last_checkpoint": str(output_dir / "last.pt"),
            "nonfinite_batches": int(nonfinite_batches),
            "final_test_evaluated": False,
        }
    )
    _write_json(output_dir / "training_curves.json", {"epochs": epoch_rows})
    _write_json(output_dir / "run_report.json", report)
    return report


def _read_report(run_root: Path, run_id: str) -> Mapping[str, Any] | None:
    path = run_root / run_id / "run_report.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fuse_predictions(config: CanonicalStateVariantRunConfig, spec: CanonicalStateVariantSpec) -> dict[str, Any]:
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = config.variant_root_path()
    reports = {run_id: _read_report(run_root, run_id) for run_id in spec.fusion_inputs}
    valid_reports = [report for report in reports.values() if isinstance(report, Mapping)]
    limitations: list[str] = []
    if spec.model_kind in {MODEL_KIND_STATE_TOKEN_FUSION, MODEL_KIND_PARTICLE_VIEW_FUSION}:
        limitations.append(f"{spec.model_kind} is implemented as logit fusion in this runner.")
    splits = ["model_val", "stack_val"]
    if spec.allows_primary_final_test() and bool(config.confirm_final_test):
        splits.append("final_test")
    if spec.final_test_policy == FINAL_TEST_POLICY_STACK_ONLY:
        splits = ["model_val", "stack_val"]
    metrics_by_split: dict[str, Any] = {}
    prediction_caches: dict[str, Any] = {}
    missing_inputs: dict[str, list[str]] = {}
    for split in splits:
        logits_list: list[np.ndarray] = []
        labels_ref: np.ndarray | None = None
        missing: list[str] = []
        for run_id in spec.fusion_inputs:
            loaded = _load_predictions(run_root, run_id, split)
            if loaded is None:
                missing.append(run_id)
                continue
            logits, labels = loaded
            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                raise ValueError(f"{spec.run_id}/{split}: labels differ for fusion input {run_id}")
            logits_list.append(logits)
        missing_inputs[split] = missing
        if missing or not logits_list or labels_ref is None:
            metrics_by_split[split] = {
                "available": False,
                "missing_inputs": missing,
                "required_inputs": list(spec.fusion_inputs),
                "n_available_inputs": int(len(logits_list)),
            }
            continue
        fused = np.mean(np.stack(logits_list, axis=0), axis=0)
        if spec.run_id == "Fshuffle" and split != "final_test":
            rng = np.random.default_rng(int(config.seed) + 911)
            fused = 0.5 * (fused + fused[rng.permutation(fused.shape[0])])
            limitations.append("Fshuffle blends one shuffled copy of the fused logits on non-final splits.")
        metrics_by_split[split] = _metrics_from_arrays(fused, labels_ref)
        prediction_caches[split] = _save_predictions(output_dir, split, fused, labels_ref)
    base_report = valid_reports[0] if valid_reports else {}
    report = {
        "runner_contract": CANONICAL_STATE_VARIANT_RUNNER_CONTRACT,
        "variant_contract": CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
        "ok": not any(missing_inputs.get(split) for split in splits),
        "run_id": spec.run_id,
        "title": spec.title,
        "tier": spec.tier,
        "model_kind": spec.model_kind,
        "spec": spec.to_dict(),
        "config": asdict(config),
        "manifest": base_report.get("manifest") or _load_manifest_info(config),
        "hlt_input_contract": base_report.get("hlt_input_contract") or {},
        "label_names": list(LABEL_NAMES),
        "label_filter": list(CANONICAL_STATE_LABEL_FILTER),
        "model_val_dataset": base_report.get("model_val_dataset"),
        "model_val_hlt_content_hash": base_report.get("model_val_hlt_content_hash"),
        "model_val_jet_identity_hash": base_report.get("model_val_jet_identity_hash"),
        "fusion_inputs": list(spec.fusion_inputs),
        "fusion_input_reports": {run_id: bool(report) for run_id, report in reports.items()},
        "missing_prediction_inputs": missing_inputs,
        "implementation_limitations": limitations,
        "best_model_val_metrics": metrics_by_split.get("model_val"),
        "model_val_metrics": metrics_by_split.get("model_val"),
        "stack_val_metrics": metrics_by_split.get("stack_val"),
        "final_test_metrics": metrics_by_split.get("final_test"),
        "prediction_caches": prediction_caches,
        "final_test_evaluated": "final_test" in metrics_by_split and bool(metrics_by_split["final_test"].get("available", True)),
    }
    _write_json(output_dir / "run_report.json", report)
    return report


def _train_seed_ensemble(config: CanonicalStateVariantRunConfig, spec: CanonicalStateVariantSpec) -> dict[str, Any]:
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    member_reports: list[Mapping[str, Any]] = []
    member_ids: list[str] = []
    for member_index in range(int(spec.seed_count)):
        member_id = f"seed_{member_index:02d}"
        member_ids.append(member_id)
        member_config = replace(
            config,
            output_dir=output_dir / member_id,
            seed=int(config.seed) + 1009 * (member_index + 1),
            baseline_checkpoint=None,
        )
        member_reports.append(_train_tagger(member_config, spec))
    splits = ["model_val", "stack_val"]
    if spec.allows_primary_final_test() and bool(config.confirm_final_test):
        splits.append("final_test")
    metrics_by_split: dict[str, Any] = {}
    prediction_caches: dict[str, Any] = {}
    missing_inputs: dict[str, list[str]] = {}
    for split in splits:
        logits_list: list[np.ndarray] = []
        labels_ref: np.ndarray | None = None
        missing: list[str] = []
        for member_id in member_ids:
            path = output_dir / member_id / f"{split}_predictions.npz"
            if not path.exists():
                missing.append(member_id)
                continue
            with np.load(path, allow_pickle=False) as data:
                logits = data["logits"].astype(np.float32, copy=False)
                labels = data["labels"].astype(np.int64, copy=False)
            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                raise ValueError(f"{spec.run_id}/{split}: seed member labels differ")
            logits_list.append(logits)
        missing_inputs[split] = missing
        if missing or not logits_list or labels_ref is None:
            metrics_by_split[split] = {
                "available": False,
                "missing_inputs": missing,
                "required_inputs": list(member_ids),
                "n_available_inputs": int(len(logits_list)),
            }
            continue
        fused = np.mean(np.stack(logits_list, axis=0), axis=0)
        metrics_by_split[split] = _metrics_from_arrays(fused, labels_ref)
        prediction_caches[split] = _save_predictions(output_dir, split, fused, labels_ref)
    base_report = dict(member_reports[0]) if member_reports else {}
    report = dict(base_report)
    report.update(
        {
            "runner_contract": CANONICAL_STATE_VARIANT_RUNNER_CONTRACT,
            "variant_contract": CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
            "ok": not any(missing_inputs.get(split) for split in splits),
            "run_id": spec.run_id,
            "title": spec.title,
            "tier": spec.tier,
            "model_kind": spec.model_kind,
            "spec": spec.to_dict(),
            "seed_ensemble_member_ids": member_ids,
            "seed_ensemble_member_reports": [str(output_dir / member_id / "run_report.json") for member_id in member_ids],
            "missing_prediction_inputs": missing_inputs,
            "best_model_val_metrics": metrics_by_split.get("model_val"),
            "model_val_metrics": metrics_by_split.get("model_val"),
            "stack_val_metrics": metrics_by_split.get("stack_val"),
            "final_test_metrics": metrics_by_split.get("final_test"),
            "prediction_caches": prediction_caches,
            "final_test_evaluated": "final_test" in metrics_by_split and bool(metrics_by_split["final_test"].get("available", True)),
        }
    )
    _write_json(output_dir / "run_report.json", report)
    return report


def _analysis_report(config: CanonicalStateVariantRunConfig, spec: CanonicalStateVariantSpec) -> dict[str, Any]:
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = config.variant_root_path()
    dependency_reports = {
        run_id: _read_report(run_root, run_id)
        for run_id in tuple(dict.fromkeys((*spec.dependencies, *spec.fusion_inputs)))
    }
    datasets = {"model_val": _load_dataset(config, "model_val", include_phi_off=False)}
    report = _common_provenance(
        config=config,
        spec=spec,
        datasets=datasets,
        limitations=("analysis/oracle run: report-only summary, no primary final-test claim",),
    )
    report.update(
        {
            "dependency_reports_available": {run_id: isinstance(payload, Mapping) for run_id, payload in dependency_reports.items()},
            "dependency_metrics": {
                run_id: {
                    "model_val_metrics": payload.get("model_val_metrics") if isinstance(payload, Mapping) else None,
                    "state_prediction_metrics": payload.get("state_prediction_metrics") if isinstance(payload, Mapping) else None,
                }
                for run_id, payload in dependency_reports.items()
            },
            "final_test_evaluated": False,
        }
    )
    _write_json(output_dir / "run_report.json", report)
    return report


def run_canonical_state_variant(config: CanonicalStateVariantRunConfig) -> dict[str, Any]:
    """Train/evaluate one canonical-state variant and write ``run_report.json``."""

    spec = canonical_state_variant_spec(config.run_id)
    if spec.model_kind == MODEL_KIND_STATE_PREDICTOR_ONLY:
        return _train_predictor(config, spec)
    if spec.model_kind == MODEL_KIND_SEED_ENSEMBLE:
        return _train_seed_ensemble(config, spec)
    if spec.model_kind == MODEL_KIND_LOGIT_FUSION:
        return _fuse_predictions(config, spec)
    if spec.model_kind == MODEL_KIND_ORACLE_DIAGNOSTIC:
        return _train_tagger(config, spec)
    if spec.model_kind == MODEL_KIND_ANALYSIS_REPORT:
        return _analysis_report(config, spec)
    if spec.model_kind in {
        MODEL_KIND_PART_BASELINE,
        MODEL_KIND_AV10_FEATURE_MLP,
        MODEL_KIND_STATE_CONDITIONED_PART,
        MODEL_KIND_STATE_ONLY_TAGGER,
        MODEL_KIND_STATE_TOKEN_FUSION,
        MODEL_KIND_PARTICLE_VIEW_FUSION,
    }:
        return _train_tagger(config, spec)
    raise ValueError(f"unsupported canonical-state model kind {spec.model_kind!r}")


__all__ = [
    "CANONICAL_STATE_VARIANT_RUNNER_CONTRACT",
    "PREDICTION_CACHE_CONTRACT",
    "CanonicalStateVariantRunConfig",
    "CanonicalStateCachedDataset",
    "run_canonical_state_variant",
]
