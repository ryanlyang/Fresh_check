"""Stack-train-only F-tier fitting for constrained pseudo-offline models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import (
    PredictionBlock,
    load_prediction_block,
    save_prediction_block,
    softmax_np,
    validate_prediction_alignment,
)
from jetclass_fresh.independent_fusion import (
    DEFAULT_C_GRID,
    fit_stacker_selecting_c_on_val,
    metrics_from_logits,
    stack_feature_matrix,
)


COARSE_TO_FINE_FUSION_CONTRACT = "constrained_coarse_to_fine_step9_fusion_v1"
FUSION_METHODS = ("mean_logits", "simplex_logits", "linear_stacker", "external")
REQUIRED_FUSION_GROUPS = ("F0", "F1", "F2", "F3", "F4", "F5")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(name) for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class FusionGroupSpec:
    name: str
    members: tuple[str, ...]
    method: str
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        members = tuple(str(row).strip() for row in self.members)
        method = str(self.method).strip().lower()
        if not name or not members or any(not row for row in members):
            raise ValueError("fusion group name and members must be non-empty")
        if len(members) != len(set(members)):
            raise ValueError(f"fusion group {name} repeats a member")
        if method not in FUSION_METHODS:
            raise ValueError(f"unknown fusion method {method!r}")
        if method == "external" and len(members) != 1:
            raise ValueError("external representation fusion must name exactly one trained model")
        if name == "F2" and method != "external":
            raise ValueError("F2 is a trained representation/particle-view model, not post-hoc logit averaging")
        expected = {"F0": "mean_logits", "F1": "simplex_logits", "F2": "external", "F4": "mean_logits", "F5": "linear_stacker"}
        if name in expected and method != expected[name]:
            raise ValueError(f"{name} requires method {expected[name]!r}, found {method!r}")
        if name == "F3" and method not in {"simplex_logits", "linear_stacker"}:
            raise ValueError("F3 particle-view plus A0 logit fusion must use simplex_logits or linear_stacker")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class Step9FusionConfig:
    prediction_dir: str
    output_dir: str
    groups: tuple[FusionGroupSpec, ...]
    required_groups: tuple[str, ...] = REQUIRED_FUSION_GROUPS
    c_grid: tuple[float, ...] = tuple(DEFAULT_C_GRID)
    max_iter: int = 2000
    simplex_samples: int = 4096
    seed: int = 29219
    overwrite_predictions: bool = False
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        names = tuple(row.name for row in groups)
        if len(names) != len(set(names)):
            raise ValueError("fusion group names must be unique")
        missing = sorted(set(self.required_groups) - set(names))
        if missing:
            raise ValueError(f"missing required fusion groups: {missing}")
        if not self.confirm_final_test:
            raise ValueError("F-tier evaluation requires explicit final_test confirmation")
        if int(self.max_iter) <= 0 or int(self.simplex_samples) <= 0:
            raise ValueError("max_iter and simplex_samples must be positive")
        object.__setattr__(self, "groups", groups)


def _blocks(prediction_dir: str | Path, members: Sequence[str], split: str) -> list[PredictionBlock]:
    rows = [load_prediction_block(prediction_dir, member, split) for member in members]
    validate_prediction_alignment(rows)
    expected_manifest = {row.metadata.get("source_manifest_hash") for row in rows}
    expected_hlt = {row.metadata.get("hlt_content_hash") for row in rows}
    expected_identity = {row.metadata.get("jet_identity_hash") for row in rows}
    if None in expected_manifest or len(expected_manifest) != 1:
        raise ValueError(f"{split} fusion members have missing/conflicting manifest provenance")
    if None in expected_hlt or len(expected_hlt) != 1:
        raise ValueError(f"{split} fusion members have missing/conflicting HLT cache provenance")
    if None in expected_identity or len(expected_identity) != 1:
        raise ValueError(f"{split} fusion members have missing/conflicting identity provenance")
    if any(row.metadata.get("deployable_hlt_only") is not True for row in rows):
        raise ValueError(f"{split} fusion contains a prediction not marked deployable_hlt_only")
    return rows


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = softmax_np(logits)
    picked = np.clip(probs[np.arange(len(labels)), labels], 1.0e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def _fit_simplex(blocks: Sequence[PredictionBlock], *, samples: int, seed: int) -> dict[str, Any]:
    logits = np.stack([row.logits for row in blocks], axis=1).astype(np.float64)
    labels = blocks[0].labels
    candidates = [np.full(len(blocks), 1.0 / len(blocks), dtype=np.float64)]
    candidates.extend(np.eye(len(blocks), dtype=np.float64))
    if len(blocks) > 1:
        candidates.extend(np.random.default_rng(seed).dirichlet(np.ones(len(blocks)), size=int(samples)))
    best = min(candidates, key=lambda weights: _cross_entropy(np.einsum("bmc,m->bc", logits, weights), labels))
    return {
        "weights": np.asarray(best, dtype=np.float64),
        "fit_split": "stack_train",
        "stack_train_cross_entropy": _cross_entropy(np.einsum("bmc,m->bc", logits, best), labels),
        "candidate_count": len(candidates),
    }


def _fused_logits(method: str, blocks: Sequence[PredictionBlock], fit: Mapping[str, Any]) -> np.ndarray:
    if method == "external":
        return blocks[0].logits
    if method == "mean_logits":
        return np.mean(np.stack([row.logits for row in blocks], axis=0), axis=0)
    if method == "simplex_logits":
        return np.einsum(
            "bmc,m->bc",
            np.stack([row.logits for row in blocks], axis=1),
            np.asarray(fit["weights"], dtype=np.float64),
        ).astype(np.float32)
    if method == "linear_stacker":
        features = stack_feature_matrix(blocks, feature_mode="logits_probs")
        return fit["stacker"].predict_logits(features).astype(np.float32)
    raise AssertionError(method)


def _fit_group(config: Step9FusionConfig, spec: FusionGroupSpec) -> dict[str, Any]:
    train = _blocks(config.prediction_dir, spec.members, "stack_train")
    if spec.method == "simplex_logits":
        return _fit_simplex(train, samples=config.simplex_samples, seed=config.seed + sum(map(ord, spec.name)))
    if spec.method == "linear_stacker":
        val = _blocks(config.prediction_dir, spec.members, "stack_val")
        stacker, selection = fit_stacker_selecting_c_on_val(
            stack_feature_matrix(train, feature_mode="logits_probs"),
            train[0].labels,
            stack_feature_matrix(val, feature_mode="logits_probs"),
            val[0].labels,
            c_grid=config.c_grid,
            max_iter=config.max_iter,
            feature_mode="logits_probs",
            model_names=spec.members,
            num_classes=int(train[0].logits.shape[1]),
        )
        return {"stacker": stacker, "fit_split": "stack_train", "selection": selection}
    return {"fit_split": None, "parameter_free": True}


def run_step9_fusion(config: Step9FusionConfig) -> dict[str, Any]:
    """Fit every learned fuser before opening final_test predictions."""

    output_dir = Path(config.output_dir)
    report_path = output_dir / "fusion_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite locked fusion report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = {spec.name: _fit_group(config, spec) for spec in config.groups}
    rows: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    # final_test is not loaded until all fitting and stack_val selection is done.
    for spec in config.groups:
        fit = fitted[spec.name]
        split_reports = {}
        for split in ("model_val", "stack_train", "stack_val", "final_test"):
            blocks = _blocks(config.prediction_dir, spec.members, split)
            logits = _fused_logits(spec.method, blocks, fit)
            metadata = {
                "ok": True,
                "contract": COARSE_TO_FINE_FUSION_CONTRACT,
                "fusion_group": spec.name,
                "fusion_method": spec.method,
                "members": list(spec.members),
                "description": spec.description,
                "source_manifest_hash": blocks[0].metadata["source_manifest_hash"],
                "hlt_content_hash": blocks[0].metadata["hlt_content_hash"],
                "deployable_hlt_only": True,
                "fit_split": fit.get("fit_split"),
                "selection_split": "stack_val" if spec.method == "linear_stacker" else None,
                "final_test_opened_after_all_fits": split == "final_test",
                "member_checkpoint_hashes": {
                    row.model_name: row.metadata.get("checkpoint_sha256") for row in blocks
                },
                "member_mechanism_summaries": {
                    row.model_name: row.metadata.get("mechanism_diagnostics") for row in blocks
                },
            }
            saved = save_prediction_block(
                PredictionBlock(
                    spec.name,
                    split,
                    logits,
                    softmax_np(logits),
                    blocks[0].labels,
                    blocks[0].jet_ids,
                    metadata,
                ),
                config.prediction_dir,
                overwrite=bool(config.overwrite_predictions),
            )
            metrics = metrics_from_logits(logits, blocks[0].labels)
            split_reports[split] = {"metrics": metrics, "prediction_metadata": saved}
            rows.append(
                {
                    "fusion_group": spec.name,
                    "method": spec.method,
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "cross_entropy": metrics.get("cross_entropy"),
                    "macro_ovr_auc": metrics.get("macro_ovr_auc"),
                    "members": ";".join(spec.members),
                }
            )
        serialized_fit = {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in fit.items()
            if name != "stacker"
        }
        if "stacker" in fit:
            stacker = fit["stacker"]
            stacker_path = output_dir / "stackers" / f"{spec.name}.npz"
            stacker_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                stacker_path,
                coef=stacker.coef,
                intercept=stacker.intercept,
                mean=stacker.mean,
                scale=stacker.scale,
                C=np.asarray([stacker.C]),
                model_names=np.asarray(stacker.model_names),
            )
            serialized_fit["stacker_path"] = str(stacker_path)
        groups[spec.name] = {
            "spec": asdict(spec),
            "fit": serialized_fit,
            "splits": split_reports,
            "representation_particle_view_summary": (
                "trained external early/representation fusion" if spec.method == "external" else "post-hoc logit fusion"
            ),
        }
    _write_csv(output_dir / "fusion_metrics.csv", rows)
    report = {
        "ok": True,
        "contract": COARSE_TO_FINE_FUSION_CONTRACT,
        "required_groups": list(config.required_groups),
        "groups": groups,
        "leakage_contract": {
            "fit_split": "stack_train",
            "hyperparameter_selection_split": "stack_val",
            "final_test_opened_after_all_fits": True,
            "offline_inputs_used": False,
        },
        "fusion_metrics_csv": str(output_dir / "fusion_metrics.csv"),
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "COARSE_TO_FINE_FUSION_CONTRACT",
    "FUSION_METHODS",
    "REQUIRED_FUSION_GROUPS",
    "FusionGroupSpec",
    "Step9FusionConfig",
    "run_step9_fusion",
]
