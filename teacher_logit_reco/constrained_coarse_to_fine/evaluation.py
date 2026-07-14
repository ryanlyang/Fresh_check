"""HLT-only prediction caches and mechanism diagnostics for Step 9."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.hlt_baseline import amp_autocast_context, resolve_device
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view, normalize_hlt_profile
from jetclass_fresh.independent_fusion import metrics_from_logits
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_split_manifest, manifest_hash
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions

from .cache import (
    HIERARCHY_TARGET_EXPECTED_HLT_PROFILE,
    HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION,
    HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH,
)
from .end_to_end import (
    END_TO_END_TRAIN_CONTRACT,
    EndToEndCoarseToFineTagger,
    load_end_to_end_tagger_checkpoint,
)
from .fusion import D8_MULTIDEPTH, particle_stream_from_tokens
from .targets import hlt_reference_axis


COARSE_TO_FINE_PREDICTION_CONTRACT = "constrained_coarse_to_fine_prediction_cache_v1"
DEFAULT_PREDICTION_SPLITS = ("model_val", "stack_train", "stack_val")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _jsonable(row) for name, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(row) for row in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object at {path}")
    return value


@dataclass(frozen=True)
class EndToEndPredictionConfig:
    prediction_dir: str
    model_name: str
    manifest_path: str
    hlt_cache_dir: str
    checkpoint_path: str
    splits: tuple[str, ...] = DEFAULT_PREDICTION_SPLITS
    batch_size: int = 128
    device: str = "auto"
    amp: bool = True
    verify_hash: bool = True
    overwrite: bool = False
    confirm_final_test: bool = False
    max_jets_per_split: int | None = None
    d8_view_ablation_max_jets: int | None = 50_000
    seed: int = 29117

    def __post_init__(self) -> None:
        splits = tuple(str(row) for row in self.splits)
        if not splits or len(splits) != len(set(splits)):
            raise ValueError("prediction splits must be non-empty and unique")
        if "model_train" in splits:
            raise ValueError("Step 9 does not cache deployable model_train predictions")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("final_test prediction requires confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_jets_per_split is not None and int(self.max_jets_per_split) <= 0:
            raise ValueError("max_jets_per_split must be positive when provided")
        object.__setattr__(self, "splits", splits)


def _validate_hlt_split(config: EndToEndPredictionConfig, split: str) -> tuple[Any, str]:
    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    if split not in manifest.splits:
        raise ValueError(f"split {split!r} is absent from the active manifest")
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hash))
    expected_ids = tuple(manifest.splits[split])
    if tuple(view.jet_ids) != expected_ids:
        raise ValueError(f"HLT jet identities do not match the active manifest for {split}")
    if view.metadata.get("source_manifest_hash") != manifest_sha:
        raise ValueError(f"HLT source_manifest_hash mismatch for {split}")
    if normalize_hlt_profile(view.metadata.get("hlt_profile")) != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE:
        raise ValueError(f"HLT profile mismatch for {split}")
    if str(view.metadata.get("hlt_profile_version") or "") != HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION:
        raise ValueError(f"HLT profile version mismatch for {split}")
    strength = view.metadata.get("hlt_degradation_strength")
    if strength is None or abs(float(strength) - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
        raise ValueError(f"HLT degradation strength mismatch for {split}: {strength}")
    if not view.metadata.get("hlt_content_hash"):
        raise ValueError(f"HLT cache for {split} lacks hlt_content_hash")
    return view, manifest_sha


def _require_checkpoint_provenance(
    payload: Mapping[str, Any],
    *,
    split: str,
    manifest_sha: str,
    hlt_hash: str,
) -> None:
    provenance = payload.get("provenance")
    row = provenance.get(split) if isinstance(provenance, Mapping) else None
    # Training/selection splits must be bound to the exact data used. Stack and
    # final splits were intentionally never opened during training.
    if split in {"model_train", "model_val"}:
        if not isinstance(row, Mapping):
            raise ValueError(f"checkpoint lacks required {split} provenance")
        if row.get("source_manifest_hash") != manifest_sha:
            raise ValueError(f"checkpoint {split} manifest hash mismatch")
        if row.get("hlt_content_hash") != hlt_hash:
            raise ValueError(f"checkpoint {split} HLT content hash mismatch")


def _detailed_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    base = metrics_from_logits(logits, labels)
    preds = np.argmax(logits, axis=1).astype(np.int64)
    detailed = classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        logits=logits,
        label_names=tuple(LABEL_NAMES),
    )
    detailed["cross_entropy"] = base.get("cross_entropy")
    detailed["macro_ovr_auc"] = base.get("macro_ovr_auc")
    detailed["fpr_at_50pct_signal_efficiency"] = base.get("fpr_at_50pct_signal_efficiency")
    return detailed


def _confidence_bins(logits: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    probs = softmax_np(logits)
    confidence = probs.max(axis=1)
    correct = np.argmax(probs, axis=1) == labels
    result = []
    for index in range(int(bins)):
        low, high = index / bins, (index + 1) / bins
        selected = (confidence >= low) & (confidence <= high if index == bins - 1 else confidence < high)
        result.append(
            {
                "low": low,
                "high": high,
                "n_jets": int(selected.sum()),
                "accuracy": None if not np.any(selected) else float(correct[selected].mean()),
                "mean_confidence": None if not np.any(selected) else float(confidence[selected].mean()),
            }
        )
    return result


def _view_masks(view_names: Sequence[str]) -> dict[str, np.ndarray]:
    names = tuple(str(row) for row in view_names)
    result = {"hlt_only": np.zeros(len(names), dtype=bool)}
    for count in (1, 2):
        for indices in itertools.combinations(range(len(names)), count):
            mask = np.zeros(len(names), dtype=bool)
            mask[list(indices)] = True
            result["hlt_plus_" + "+".join(names[index] for index in indices)] = mask
    result["all_views"] = np.ones(len(names), dtype=bool)
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or right.size != left.size or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _mechanism_diagnostics(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    hlt_logits: np.ndarray | None,
    per_view_logits: np.ndarray | None,
    gates: np.ndarray | None,
    uncertainties: np.ndarray | None,
    representation_cosines: np.ndarray | None,
    view_names: Sequence[str],
) -> dict[str, Any]:
    final_preds = np.argmax(logits, axis=1)
    report: dict[str, Any] = {"confidence_bins": _confidence_bins(logits, labels)}
    if hlt_logits is not None:
        hlt_preds = np.argmax(hlt_logits, axis=1)
        disagreement = hlt_preds != final_preds
        report["hlt_fused_disagreement_fraction"] = float(disagreement.mean())
        report["high_disagreement_accuracy"] = (
            None if not np.any(disagreement) else float((final_preds[disagreement] == labels[disagreement]).mean())
        )
    if gates is not None:
        report["pooled_gate"] = {
            "view_names": list(view_names),
            "mean": gates.mean(axis=0).tolist(),
            "p10": np.quantile(gates, 0.10, axis=0).tolist(),
            "p90": np.quantile(gates, 0.90, axis=0).tolist(),
            "dominant_view_fraction": [float(np.mean(np.argmax(gates, axis=1) == index)) for index in range(gates.shape[1])],
            "one_view_over_0p8_fraction": float(np.mean(np.max(gates, axis=1) > 0.8)),
            "per_class_mean": {
                str(index): (None if not np.any(labels == index) else gates[labels == index].mean(axis=0).tolist())
                for index in range(len(LABEL_NAMES))
            },
        }
        if uncertainties is not None:
            report["uncertainty_gate_correlation"] = [
                _pearson(uncertainties[:, index], gates[:, index]) for index in range(gates.shape[1])
            ]
    if per_view_logits is not None:
        view_predictions = np.argmax(per_view_logits, axis=-1)
        unique_counts = np.asarray([len(set(row.tolist())) for row in view_predictions], dtype=np.int64)
        report["coarse_fine_agreement_bins"] = [
            {
                "unique_view_predictions": int(count),
                "n_jets": int(np.sum(unique_counts == count)),
                "fraction": float(np.mean(unique_counts == count)),
                "fused_accuracy": (
                    None
                    if not np.any(unique_counts == count)
                    else float(np.mean(final_preds[unique_counts == count] == labels[unique_counts == count]))
                ),
            }
            for count in sorted(set(unique_counts.tolist()))
        ]
        pairwise = []
        for left, right in itertools.combinations(range(per_view_logits.shape[1]), 2):
            pairwise.append(
                {
                    "left": str(view_names[left]),
                    "right": str(view_names[right]),
                    "prediction_disagreement_fraction": float(
                        np.mean(np.argmax(per_view_logits[:, left], axis=1) != np.argmax(per_view_logits[:, right], axis=1))
                    ),
                    "representation_cosine_mean": (
                        None if representation_cosines is None else float(representation_cosines[:, left, right].mean())
                    ),
                }
            )
        report["pseudo_view_pairs"] = pairwise
    return report


def cache_end_to_end_prediction_split(
    config: EndToEndPredictionConfig,
    split: str,
    *,
    model_bundle: tuple[EndToEndCoarseToFineTagger, Mapping[str, Any], Any] | None = None,
) -> dict[str, Any]:
    """Cache one split while refusing all offline/target inputs."""

    if split == "final_test" and not config.confirm_final_test:
        raise ValueError("final_test prediction requires explicit confirmation")
    model_dir = Path(config.prediction_dir) / config.model_name
    final_receipt = model_dir / "final_test_claim_receipt.json"
    final_prediction = model_dir / "final_test_predictions.npz"
    if split == "final_test" and (final_receipt.exists() or final_prediction.exists()):
        raise FileExistsError(
            "final_test claim already exists; overwrite is forbidden for immutable final claims"
        )
    view, manifest_sha = _validate_hlt_split(config, split)
    device = resolve_device(config.device)
    model, payload, resolved = model_bundle or load_end_to_end_tagger_checkpoint(
        config.checkpoint_path, device=device
    )
    if payload.get("checkpoint_contract") != END_TO_END_TRAIN_CONTRACT:
        raise ValueError("prediction checkpoint is not a Step 8 end-to-end model")
    _require_checkpoint_provenance(
        payload,
        split=split,
        manifest_sha=manifest_sha,
        hlt_hash=str(view.metadata["hlt_content_hash"]),
    )
    model.eval()
    limit = len(view.labels) if config.max_jets_per_split is None else min(len(view.labels), int(config.max_jets_per_split))
    labels = np.asarray(view.labels[:limit], dtype=np.int64)
    axes_eta, axes_phi, _ = hlt_reference_axis(view.tokens[:limit], view.mask[:limit])
    logits_rows: list[np.ndarray] = []
    hlt_rows: list[np.ndarray] = []
    per_view_rows: list[np.ndarray] = []
    gate_rows: list[np.ndarray] = []
    uncertainty_rows: list[np.ndarray] = []
    cosine_rows: list[np.ndarray] = []
    representation_rows: list[np.ndarray] = []
    ablation_logits: dict[str, list[np.ndarray]] = {}
    masks = _view_masks(model.tagger.view_names) if str(payload.get("variant")) == D8_MULTIDEPTH else {}
    ablation_limit = limit if config.d8_view_ablation_max_jets is None else min(limit, int(config.d8_view_ablation_max_jets))
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    with torch.no_grad():
        for start in range(0, limit, int(config.batch_size)):
            stop = min(limit, start + int(config.batch_size))
            tokens = torch.from_numpy(np.asarray(view.tokens[start:stop], dtype=np.float32)).to(device)
            valid = torch.from_numpy(np.asarray(view.mask[start:stop], dtype=bool)).to(device)
            hlt = particle_stream_from_tokens(tokens, valid)
            ref_eta = torch.from_numpy(axes_eta[start:stop]).to(device)
            ref_phi = torch.from_numpy(axes_phi[start:stop]).to(device)
            with amp_autocast_context(amp_enabled):
                output = model.forward_detailed(
                    hlt,
                    reference_eta=ref_eta,
                    reference_phi=ref_phi,
                    stochastic_seed=int(config.seed) + start,
                )
            tagger = output.tagger
            logits_rows.append(tagger.logits.float().cpu().numpy())
            representation_rows.append(tagger.fusion_representation.float().cpu().numpy())
            if tagger.hlt_logits is not None:
                hlt_rows.append(tagger.hlt_logits.float().cpu().numpy())
            if model.tagger.pseudo_head is not None and tagger.pseudo_representations:
                by_view = torch.stack(
                    [model.tagger.pseudo_head(tagger.pseudo_representations[name]) for name in model.tagger.view_names],
                    dim=1,
                )
                per_view_rows.append(by_view.float().cpu().numpy())
            if tagger.pooled_gates is not None:
                gate_rows.append(tagger.pooled_gates.float().cpu().numpy())
            raw_uncertainty = tagger.diagnostics.get("mean_view_uncertainty")
            if torch.is_tensor(raw_uncertainty):
                uncertainty_rows.append(raw_uncertainty.float().cpu().numpy())
            reps = [tagger.pseudo_representations.get(name) for name in model.tagger.view_names]
            if reps and all(row is not None for row in reps):
                stacked = torch.stack([F.normalize(row.float(), dim=-1) for row in reps], dim=1)
                cosine_rows.append(torch.einsum("bvd,bwd->bvw", stacked, stacked).cpu().numpy())
            if masks and start < ablation_limit:
                ablation_stop = min(stop, ablation_limit)
                count = ablation_stop - start
                sub_hlt = type(hlt)(
                    hlt.points[:count], hlt.features[:count], hlt.lorentz_vectors[:count], hlt.mask[:count]
                )
                sub_views = tuple(
                    type(row)(**{name: getattr(row, name)[:count] if torch.is_tensor(getattr(row, name)) and getattr(row, name).ndim > 1 and int(getattr(row, name).shape[0]) == stop - start else getattr(row, name) for name in row.__dataclass_fields__})
                    for row in output.pseudo_views
                )
                for name, mask in masks.items():
                    override = torch.from_numpy(mask).to(device=device)[None, :].expand(count, -1)
                    with amp_autocast_context(amp_enabled):
                        masked = model.tagger.forward_detailed(
                            sub_hlt,
                            sub_views,
                            view_availability_override=override,
                        )
                    ablation_logits.setdefault(name, []).append(masked.logits.float().cpu().numpy())
    logits = np.concatenate(logits_rows, axis=0)
    hlt_logits = np.concatenate(hlt_rows, axis=0) if hlt_rows else None
    per_view_logits = np.concatenate(per_view_rows, axis=0) if per_view_rows else None
    gates = np.concatenate(gate_rows, axis=0) if gate_rows else None
    uncertainties = np.concatenate(uncertainty_rows, axis=0) if uncertainty_rows else None
    cosines = np.concatenate(cosine_rows, axis=0) if cosine_rows else None
    representations = np.concatenate(representation_rows, axis=0)
    ablation_metrics = {
        name: _detailed_metrics(np.concatenate(rows, axis=0), labels[:ablation_limit])
        for name, rows in ablation_logits.items()
    }
    checkpoint_hash = _file_sha256(config.checkpoint_path)
    configuration_hash = payload.get("configuration_hash") or _canonical_hash(
        {"fusion": payload.get("fusion_config"), "reconstructors": payload.get("reconstructors")}
    )
    metrics = _detailed_metrics(logits, labels)
    mechanism = _mechanism_diagnostics(
        logits=logits,
        labels=labels,
        hlt_logits=hlt_logits,
        per_view_logits=per_view_logits,
        gates=gates,
        uncertainties=uncertainties,
        representation_cosines=cosines,
        view_names=model.tagger.view_names,
    )
    source_hashes = {
        name: {
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "model_config_hash": row.get("model_config_hash"),
            "alias_of": row.get("alias_of"),
        }
        for name, row in resolved.source_metadata.items()
    }
    representation_path = model_dir / f"{split}_representations.npz"
    if representation_path.exists() and (split == "final_test" or not config.overwrite):
        raise FileExistsError(f"refusing to overwrite representation cache: {representation_path}")
    representation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        representation_path,
        representation=representations.astype(np.float16),
        labels=labels,
    )
    representation_hash = _file_sha256(representation_path)
    metadata = {
        "ok": True,
        "contract": COARSE_TO_FINE_PREDICTION_CONTRACT,
        "run_id": config.model_name,
        "checkpoint_contract": payload.get("checkpoint_contract"),
        "checkpoint_role": payload.get("checkpoint_role"),
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "configuration_hash": configuration_hash,
        "variant": payload.get("variant"),
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": view.metadata.get("hlt_content_hash"),
        "hlt_profile": HIERARCHY_TARGET_EXPECTED_HLT_PROFILE,
        "hlt_profile_version": HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION,
        "hlt_degradation_strength": HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH,
        "jet_identity_hash": jet_identity_hash(view.jet_ids[:limit]),
        "reconstructor_sources": source_hashes,
        "reconstructor_aliases": dict(resolved.aliases),
        "source_state": payload.get("source_state"),
        "active_view_names": list(model.tagger.view_names),
        "fusion_representation_path": str(representation_path),
        "fusion_representation_sha256": representation_hash,
        "fusion_representation_shape": list(representations.shape),
        "metrics": metrics,
        "mechanism_diagnostics": mechanism,
        "d8_model_val_view_ablations": ablation_metrics if split == "model_val" else {},
        "d8_view_ablation_selection_only": bool(ablation_metrics),
        "offline_inputs_loaded": False,
        "target_cache_loaded": False,
        "deployable_hlt_only": True,
        "final_test_confirmed": split == "final_test" and bool(config.confirm_final_test),
    }
    block = PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=list(view.jet_ids[:limit]),
        metadata=metadata,
    )
    saved = save_prediction_block(block, config.prediction_dir, overwrite=bool(config.overwrite))
    # The shared block helper has a compact metric block; preserve the richer
    # ten-class diagnostics under the canonical Step 9 key.
    saved["metrics"] = metrics
    saved["mechanism_diagnostics"] = mechanism
    saved["d8_model_val_view_ablations"] = metadata["d8_model_val_view_ablations"]
    metadata_path = Path(config.prediction_dir) / config.model_name / f"{split}_predictions_metadata.json"
    _write_json(metadata_path, saved)
    if split == "final_test":
        _write_json(
            final_receipt,
            {
                "contract": COARSE_TO_FINE_PREDICTION_CONTRACT,
                "immutable_final_claim": True,
                "run_id": config.model_name,
                "checkpoint_sha256": checkpoint_hash,
                "source_manifest_hash": manifest_sha,
                "hlt_content_hash": view.metadata.get("hlt_content_hash"),
                "jet_identity_hash": metadata["jet_identity_hash"],
                "prediction_path": str(final_prediction),
                "representation_sha256": representation_hash,
            },
        )
    return saved


def cache_end_to_end_predictions(config: EndToEndPredictionConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    bundle = load_end_to_end_tagger_checkpoint(config.checkpoint_path, device=device)
    rows = {
        split: cache_end_to_end_prediction_split(config, split, model_bundle=bundle)
        for split in config.splits
    }
    report_path = Path(config.prediction_dir) / config.model_name / "prediction_run_report.json"
    existing_report = _read_json(report_path) if report_path.is_file() else {}
    existing_splits = existing_report.get("splits") if isinstance(existing_report.get("splits"), Mapping) else {}
    report = {
        "ok": True,
        "contract": COARSE_TO_FINE_PREDICTION_CONTRACT,
        "config": asdict(config),
        "splits": {**existing_splits, **rows},
        "selection_split": "model_val",
        "fusion_fit_split": "stack_train",
        "final_test_claim_receipt": (
            str(Path(config.prediction_dir) / config.model_name / "final_test_claim_receipt.json")
            if "final_test" in config.splits
            else None
        ),
        "offline_inputs_loaded": False,
    }
    _write_json(report_path, report)
    return report


def cache_prediction_alias(
    prediction_dir: str | Path,
    *,
    source_name: str,
    alias_name: str,
    splits: Sequence[str] = DEFAULT_PREDICTION_SPLITS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Preserve an aliased campaign row without pretending it was retrained."""

    from jetclass_fresh.fusion import load_prediction_block
    import os
    import shutil

    report_path = Path(prediction_dir) / alias_name / "prediction_run_report.json"
    existing_report = _read_json(report_path) if report_path.is_file() else {}
    existing_splits = existing_report.get("splits") if isinstance(existing_report.get("splits"), Mapping) else {}
    rows = {}
    for split in splits:
        source = load_prediction_block(prediction_dir, source_name, split)
        metadata = {
            **source.metadata,
            "run_id": alias_name,
            "alias_of": source_name,
            "shared_checkpoint_sha256": source.metadata.get("checkpoint_sha256"),
            "shared_configuration_hash": source.metadata.get("configuration_hash"),
            "shared_configuration": True,
        }
        alias = PredictionBlock(alias_name, split, source.logits, source.probs, source.labels, source.jet_ids, metadata)
        rows[split] = save_prediction_block(alias, prediction_dir, overwrite=overwrite)
        source_representation = Path(prediction_dir) / source_name / f"{split}_representations.npz"
        alias_representation = Path(prediction_dir) / alias_name / f"{split}_representations.npz"
        if not source_representation.is_file():
            raise FileNotFoundError(f"source prediction alias lacks representations: {source_representation}")
        if alias_representation.exists():
            if not overwrite:
                raise FileExistsError(f"alias representation already exists: {alias_representation}")
            alias_representation.unlink()
        try:
            os.link(source_representation, alias_representation)
        except OSError:
            shutil.copy2(source_representation, alias_representation)
        rows[split]["fusion_representation_path"] = str(alias_representation)
        rows[split]["fusion_representation_sha256"] = _file_sha256(alias_representation)
        metadata_path = Path(prediction_dir) / alias_name / f"{split}_predictions_metadata.json"
        _write_json(metadata_path, rows[split])
        if split == "final_test":
            source_receipt = _read_json(Path(prediction_dir) / source_name / "final_test_claim_receipt.json")
            _write_json(
                Path(prediction_dir) / alias_name / "final_test_claim_receipt.json",
                {
                    **source_receipt,
                    "run_id": alias_name,
                    "alias_of": source_name,
                    "prediction_path": str(Path(prediction_dir) / alias_name / "final_test_predictions.npz"),
                },
            )
    report = {
        "ok": True,
        "contract": COARSE_TO_FINE_PREDICTION_CONTRACT,
        "run_id": alias_name,
        "alias_of": source_name,
        "shared_checkpoint_sha256": next(iter(rows.values())).get("checkpoint_sha256") if rows else None,
        "shared_configuration_hash": next(iter(rows.values())).get("configuration_hash") if rows else None,
        "splits": {**existing_splits, **rows},
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "COARSE_TO_FINE_PREDICTION_CONTRACT",
    "DEFAULT_PREDICTION_SPLITS",
    "EndToEndPredictionConfig",
    "cache_end_to_end_prediction_split",
    "cache_end_to_end_predictions",
    "cache_prediction_alias",
]
