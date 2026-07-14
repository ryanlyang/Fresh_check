"""Stack-train-only F-tier fitting for constrained pseudo-offline models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
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
FUSION_METHODS = ("mean_logits", "simplex_logits", "linear_stacker", "representation_stacker")
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        if name == "F2" and method != "representation_stacker":
            raise ValueError("F2 requires learned fusion of cached D-tier representations")
        expected = {"F0": "mean_logits", "F1": "simplex_logits", "F2": "representation_stacker", "F4": "mean_logits", "F5": "linear_stacker"}
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
    best_d_candidates: tuple[str, ...] = (
        "D0", "D1", "D2", "D3", "D4", "D5", "D5-B1", "D5-B2", "D6", "D7", "D8"
    )

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        names = tuple(row.name for row in groups)
        if len(names) != len(set(names)):
            raise ValueError("fusion group names must be unique")
        missing = sorted(set(self.required_groups) - set(names))
        if missing:
            raise ValueError(f"missing required fusion groups: {missing}")
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


def _representation_features(
    prediction_dir: str | Path,
    blocks: Sequence[PredictionBlock],
    split: str,
) -> np.ndarray:
    rows = []
    for block in blocks:
        path = Path(prediction_dir) / block.model_name / f"{split}_representations.npz"
        expected_hash = block.metadata.get("fusion_representation_sha256")
        if not path.exists() or not expected_hash:
            raise FileNotFoundError(f"missing attested representation cache for {block.model_name} {split}")
        digest = _file_sha256(path)
        if digest != expected_hash:
            raise ValueError(f"representation hash mismatch for {block.model_name} {split}")
        with np.load(path, allow_pickle=False) as payload:
            representation = np.asarray(payload["representation"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
        if representation.shape[0] != len(block.labels) or not np.array_equal(labels, block.labels):
            raise ValueError(f"representation alignment mismatch for {block.model_name} {split}")
        rows.append(representation)
    return np.concatenate(rows, axis=1)


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


def _fused_logits(
    method: str,
    blocks: Sequence[PredictionBlock],
    fit: Mapping[str, Any],
    *,
    representation_features: np.ndarray | None = None,
) -> np.ndarray:
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
    if method == "representation_stacker":
        if representation_features is None:
            raise ValueError("representation_stacker requires representation features")
        return fit["stacker"].predict_logits(representation_features).astype(np.float32)
    raise AssertionError(method)


def _fit_group(config: Step9FusionConfig, spec: FusionGroupSpec) -> dict[str, Any]:
    train = _blocks(config.prediction_dir, spec.members, "stack_train")
    if spec.method == "simplex_logits":
        return _fit_simplex(train, samples=config.simplex_samples, seed=config.seed + sum(map(ord, spec.name)))
    if spec.method in {"linear_stacker", "representation_stacker"}:
        val = _blocks(config.prediction_dir, spec.members, "stack_val")
        if spec.method == "representation_stacker":
            train_features = _representation_features(config.prediction_dir, train, "stack_train")
            val_features = _representation_features(config.prediction_dir, val, "stack_val")
            feature_mode = "fusion_representations"
        else:
            train_features = stack_feature_matrix(train, feature_mode="logits_probs")
            val_features = stack_feature_matrix(val, feature_mode="logits_probs")
            feature_mode = "logits_probs"
        stacker, selection = fit_stacker_selecting_c_on_val(
            train_features,
            train[0].labels,
            val_features,
            val[0].labels,
            c_grid=config.c_grid,
            max_iter=config.max_iter,
            feature_mode=feature_mode,
            model_names=spec.members,
            num_classes=int(train[0].logits.shape[1]),
        )
        return {"stacker": stacker, "fit_split": "stack_train", "selection": selection}
    return {"fit_split": None, "parameter_free": True}


def _resolve_best_d(config: Step9FusionConfig) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for name in config.best_d_candidates:
        block = _blocks(config.prediction_dir, (name,), "model_val")[0]
        scores[name] = _cross_entropy(block.logits, block.labels)
    selected = min(scores, key=scores.get)
    return selected, scores


def _resolve_group_members(
    groups: Sequence[FusionGroupSpec],
    best_d: str,
) -> tuple[FusionGroupSpec, ...]:
    resolved = []
    for spec in groups:
        member_aliases = {
            "BEST_D": best_d,
            "BEST_D_SEED1": f"{best_d}-seed1",
            "BEST_D_SEED2": f"{best_d}-seed2",
        }
        members = tuple(member_aliases.get(member, member) for member in spec.members)
        members = tuple(dict.fromkeys(members))
        resolved.append(FusionGroupSpec(spec.name, members, spec.method, spec.description))
    return tuple(resolved)


def run_step9_fusion(config: Step9FusionConfig) -> dict[str, Any]:
    """Fit every learned fuser before opening final_test predictions."""

    output_dir = Path(config.output_dir)
    report_path = output_dir / (
        "fusion_final_claim_report.json" if config.confirm_final_test else "fusion_report.json"
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite locked fusion report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_d, best_d_scores = _resolve_best_d(config)
    groups_to_run = _resolve_group_members(config.groups, best_d)
    fitted = {spec.name: _fit_group(config, spec) for spec in groups_to_run}
    rows: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    # final_test is not loaded until all fitting and stack_val selection is done.
    evaluation_splits = ("final_test",) if config.confirm_final_test else (
        "model_val",
        "stack_train",
        "stack_val",
    )
    for spec in groups_to_run:
        fit = fitted[spec.name]
        split_reports = {}
        for split in evaluation_splits:
            blocks = _blocks(config.prediction_dir, spec.members, split)
            representation_features = (
                _representation_features(config.prediction_dir, blocks, split)
                if spec.method == "representation_stacker"
                else None
            )
            logits = _fused_logits(
                spec.method,
                blocks,
                fit,
                representation_features=representation_features,
            )
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
                "selection_split": "stack_val" if spec.method in {"linear_stacker", "representation_stacker"} else None,
                "final_test_opened_after_all_fits": bool(config.confirm_final_test and split == "final_test"),
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
                "learned fusion of D-tier internal jet representations"
                if spec.method == "representation_stacker"
                else "post-hoc logit fusion"
            ),
        }
    _write_csv(output_dir / "fusion_metrics.csv", rows)
    report = {
        "ok": True,
        "contract": COARSE_TO_FINE_FUSION_CONTRACT,
        "required_groups": list(config.required_groups),
        "selected_best_d": best_d,
        "best_d_selection_metric": "model_val.cross_entropy",
        "best_d_model_val_cross_entropy": best_d_scores,
        "groups": groups,
        "leakage_contract": {
            "fit_split": "stack_train",
            "hyperparameter_selection_split": "stack_val",
            "final_test_opened_after_all_fits": bool(config.confirm_final_test),
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
