"""Two-pass view normalization and canonical selected-view cache publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import (
    SELECTED_VIEW_MATERIALIZATION_POLICY,
    build_view_coordinate_binding,
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PARTICLE_VIEW_NORMALIZER_CONTRACT = "particle_view_normalizer_v1"
PARTICLE_VIEW_CACHE_CONTRACT = "particle_view_selected_cache_v1"
PARTICLE_VIEW_FINAL_COORDINATE_CONTRACT = (
    "particle_view_final_coordinate_publication_v1"
)
PARTICLE_VIEW_CACHE_ALLOWED_SPLITS = (
    "train",
    "model_val_stop",
    "model_val_select",
    "stack_val",
)


@dataclass(frozen=True)
class ParticleViewNormalizer:
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    preclip_minimum: tuple[float, ...]
    preclip_maximum: tuple[float, ...]
    symmetric_quantization_range: tuple[float, ...]
    valid_entries: int
    train_split_sha256: str
    generator_checkpoint_sha256: str
    standard_deviation_floor: float = 1.0e-4
    clip_minimum: float = -6.0
    clip_maximum: float = 6.0
    contract: str = PARTICLE_VIEW_NORMALIZER_CONTRACT

    def __post_init__(self) -> None:
        width = len(self.mean)
        if width not in {1, 2, 4, 8}:
            raise ValueError("normalizer width must be 1, 2, 4, or 8")
        for values in (
            self.standard_deviation,
            self.preclip_minimum,
            self.preclip_maximum,
            self.symmetric_quantization_range,
        ):
            if len(values) != width or not all(np.isfinite(values)):
                raise ValueError("normalizer vector shape/finiteness mismatch")
        if any(value < self.standard_deviation_floor for value in self.standard_deviation):
            raise ValueError("normalizer standard deviation is below its floor")
        if self.valid_entries <= 0:
            raise ValueError("normalizer requires valid train entries")
        require_sha256("train_split_sha256", self.train_split_sha256)
        require_sha256(
            "generator_checkpoint_sha256", self.generator_checkpoint_sha256
        )
        if (
            self.standard_deviation_floor != 1.0e-4
            or self.clip_minimum != -6.0
            or self.clip_maximum != 6.0
        ):
            raise ValueError("normalizer floor/clip policy changed")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "mean": list(self.mean),
            "standard_deviation": list(self.standard_deviation),
            "preclip_minimum": list(self.preclip_minimum),
            "preclip_maximum": list(self.preclip_maximum),
            "symmetric_quantization_range": list(
                self.symmetric_quantization_range
            ),
            "valid_entries": self.valid_entries,
            "train_split_sha256": self.train_split_sha256,
            "generator_checkpoint_sha256": self.generator_checkpoint_sha256,
            "standard_deviation_floor": self.standard_deviation_floor,
            "clip_minimum": self.clip_minimum,
            "clip_maximum": self.clip_maximum,
            "fit_split": "train",
            "operation_order": [
                "masked_train_mean",
                "masked_train_population_standard_deviation",
                "standard_deviation_floor_1e-4",
                "standardize",
                "record_preclip_statistics",
                "clip_minus6_plus6",
                "invalid_particle_zeroing",
            ],
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ParticleViewNormalizer":
        if payload.get("contract") != PARTICLE_VIEW_NORMALIZER_CONTRACT:
            raise ValueError("normalizer contract mismatch")
        normalizer = cls(
            mean=tuple(float(value) for value in payload["mean"]),
            standard_deviation=tuple(
                float(value) for value in payload["standard_deviation"]
            ),
            preclip_minimum=tuple(
                float(value) for value in payload["preclip_minimum"]
            ),
            preclip_maximum=tuple(
                float(value) for value in payload["preclip_maximum"]
            ),
            symmetric_quantization_range=tuple(
                float(value)
                for value in payload["symmetric_quantization_range"]
            ),
            valid_entries=int(payload["valid_entries"]),
            train_split_sha256=str(payload["train_split_sha256"]),
            generator_checkpoint_sha256=str(
                payload["generator_checkpoint_sha256"]
            ),
            standard_deviation_floor=float(
                payload["standard_deviation_floor"]
            ),
            clip_minimum=float(payload["clip_minimum"]),
            clip_maximum=float(payload["clip_maximum"]),
        )
        if normalizer.to_payload() != dict(payload):
            raise ValueError("normalizer payload is not canonical")
        return normalizer


def fit_particle_view_normalizer(
    view: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    train_split_sha256: str,
    generator_checkpoint_sha256: str,
) -> ParticleViewNormalizer:
    values = (
        view.detach().cpu().numpy() if isinstance(view, torch.Tensor) else np.asarray(view)
    ).astype(np.float64, copy=False)
    valid = (
        mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    ).astype(bool, copy=False)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("normalizer inputs must be [N,P,D] and [N,P]")
    selected = values[valid]
    if not selected.size or not np.isfinite(selected).all():
        raise ValueError("normalizer train values are empty or non-finite")
    mean = selected.mean(axis=0, dtype=np.float64)
    std = np.maximum(selected.std(axis=0, ddof=0, dtype=np.float64), 1.0e-4)
    standardized = (selected - mean) / std
    ranges = np.maximum(np.max(np.abs(standardized), axis=0), 1.0e-6)
    return ParticleViewNormalizer(
        mean=tuple(float(value) for value in mean),
        standard_deviation=tuple(float(value) for value in std),
        preclip_minimum=tuple(float(value) for value in standardized.min(axis=0)),
        preclip_maximum=tuple(float(value) for value in standardized.max(axis=0)),
        symmetric_quantization_range=tuple(float(value) for value in ranges),
        valid_entries=int(selected.shape[0]),
        train_split_sha256=train_split_sha256,
        generator_checkpoint_sha256=generator_checkpoint_sha256,
    )


def write_particle_view_normalizer(
    path: str | Path, normalizer: ParticleViewNormalizer
) -> dict[str, Any]:
    """Publish the final train-fit normalizer as an immutable hashed artifact."""

    artifact = with_content_hash(normalizer.to_payload())
    write_immutable_json(path, artifact)
    return artifact


def load_particle_view_normalizer(path: str | Path) -> ParticleViewNormalizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_NORMALIZER_CONTRACT
    )
    unhashed = dict(payload)
    unhashed.pop("content_hash")
    normalizer = ParticleViewNormalizer.from_payload(unhashed)
    if normalizer.content_hash != payload["content_hash"]:
        raise ValueError("normalizer artifact hash differs from coordinate hash")
    return normalizer


def normalize_particle_view(
    view: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    normalizer: ParticleViewNormalizer,
):
    if isinstance(view, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError("torch view requires torch mask")
        mean = view.new_tensor(normalizer.mean)
        std = view.new_tensor(normalizer.standard_deviation)
        result = ((view - mean) / std).clamp(-6.0, 6.0)
        return torch.where(mask[:, :, None], result, torch.zeros_like(result))
    values = np.asarray(view)
    valid = np.asarray(mask, dtype=bool)
    result = (
        values.astype(np.float64)
        - np.asarray(normalizer.mean, dtype=np.float64)
    ) / np.asarray(normalizer.standard_deviation, dtype=np.float64)
    result = np.clip(result, -6.0, 6.0)
    result = np.where(valid[:, :, None], result, 0.0)
    return np.ascontiguousarray(result, dtype="<f4")


def quantized_view_diagnostics(
    normalized_view: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    normalizer: ParticleViewNormalizer,
    *,
    bits: int,
) -> dict[str, Any]:
    if bits not in {4, 8}:
        raise ValueError("only deterministic 4-bit/8-bit diagnostics are registered")
    values = (
        normalized_view.detach().cpu().numpy()
        if isinstance(normalized_view, torch.Tensor)
        else np.asarray(normalized_view)
    ).astype(np.float64, copy=False)
    valid = (
        mask.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor)
        else np.asarray(mask)
    ).astype(bool, copy=False)
    levels = (1 << (bits - 1)) - 1
    scale = np.asarray(normalizer.symmetric_quantization_range) / levels
    codes = np.clip(np.rint(values / scale), -levels, levels)
    reconstructed = codes * scale
    error = (reconstructed - values)[valid]
    return {
        "bits": bits,
        "levels": levels,
        "mean_squared_error": float(np.mean(np.square(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "diagnostic_only": True,
    }


def _array_logical_hash(view: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(view.shape).encode())
    digest.update(view.tobytes(order="C"))
    digest.update(str(mask.shape).encode())
    digest.update(mask.tobytes(order="C"))
    return digest.hexdigest()


def publish_selected_view_cache(
    output_dir: str | Path,
    *,
    split: str,
    split_sha256: str,
    ordered_identity_sha256: str,
    raw_view: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    normalizer: ParticleViewNormalizer,
    coordinate_binding_sha256: str,
    target_id: str,
) -> dict[str, Any]:
    if split not in PARTICLE_VIEW_CACHE_ALLOWED_SPLITS:
        raise ValueError("selected true-view caches are forbidden on this split")
    for name, value in (
        ("split_sha256", split_sha256),
        ("ordered_identity_sha256", ordered_identity_sha256),
        ("coordinate_binding_sha256", coordinate_binding_sha256),
    ):
        require_sha256(name, value)
    view = normalize_particle_view(raw_view, mask, normalizer)
    valid = (
        mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    ).astype(bool, copy=False)
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    if view.dtype.str != "<f4" or np.max(np.abs(view[~valid]), initial=0.0) != 0.0:
        raise ValueError("canonical view dtype/invalid-zero policy failed")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / f"{split}_selected_views.npz"
    manifest_path = root / f"{split}_selected_views.json"
    if array_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite selected-view cache {array_path}"
        )
    np.savez(array_path, view=view, mask=valid)
    manifest = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CACHE_CONTRACT,
            "target_id": str(target_id),
            "split": split,
            "split_sha256": split_sha256,
            "ordered_identity_sha256": ordered_identity_sha256,
            "coordinate_binding_sha256": coordinate_binding_sha256,
            "normalizer_sha256": normalizer.content_hash,
            "generator_checkpoint_sha256": normalizer.generator_checkpoint_sha256,
            "array_file": array_path.name,
            "array_file_sha256": sha256_file(array_path),
            "logical_content_sha256": _array_logical_hash(view, valid),
            "view_shape": list(view.shape),
            "mask_shape": list(valid.shape),
            "dtype": "float32",
            "byte_order": "little",
            "invalid_particles_exactly_zero": True,
            "operation_order": SELECTED_VIEW_MATERIALIZATION_POLICY[
                "operation_order"
            ],
            "live_publication_max_abs_tolerance": 1.0e-6,
            "final_test_cache_forbidden": True,
        }
    )
    write_immutable_json(manifest_path, manifest)
    return manifest


def load_selected_view_cache(
    manifest_path: str | Path,
    *,
    expected_coordinate_binding_sha256: str,
    expected_split_sha256: str,
    expected_normalizer_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_selected_view_cache_manifest(manifest)
    expected = {
        "coordinate_binding_sha256": expected_coordinate_binding_sha256,
        "split_sha256": expected_split_sha256,
        "normalizer_sha256": expected_normalizer_sha256,
    }
    for name, value in expected.items():
        require_sha256(name, value)
        if manifest.get(name) != value:
            raise ValueError(f"selected-view cache {name} mismatch")
    array_path = path.parent / manifest["array_file"]
    if sha256_file(array_path) != manifest["array_file_sha256"]:
        raise ValueError("selected-view array file hash mismatch")
    with np.load(array_path, allow_pickle=False) as data:
        view = np.ascontiguousarray(data["view"])
        mask = np.ascontiguousarray(data["mask"])
    if (
        view.dtype.str != "<f4"
        or mask.dtype != np.bool_
        or list(view.shape) != manifest["view_shape"]
        or list(mask.shape) != manifest["mask_shape"]
        or _array_logical_hash(view, mask) != manifest["logical_content_sha256"]
        or np.max(np.abs(view[~mask]), initial=0.0) != 0.0
    ):
        raise ValueError("selected-view cache array contract failed")
    return view, mask, manifest


def validate_selected_view_cache_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate canonical float32 metadata before opening its array file."""

    validate_content_hash(manifest, expected_contract=PARTICLE_VIEW_CACHE_CONTRACT)
    expected_fields = {
        "contract",
        "target_id",
        "split",
        "split_sha256",
        "ordered_identity_sha256",
        "coordinate_binding_sha256",
        "normalizer_sha256",
        "generator_checkpoint_sha256",
        "array_file",
        "array_file_sha256",
        "logical_content_sha256",
        "view_shape",
        "mask_shape",
        "dtype",
        "byte_order",
        "invalid_particles_exactly_zero",
        "operation_order",
        "live_publication_max_abs_tolerance",
        "final_test_cache_forbidden",
        "content_hash",
    }
    if set(manifest) != expected_fields:
        raise ValueError("selected-view cache manifest schema mismatch")
    if (
        manifest["split"] not in PARTICLE_VIEW_CACHE_ALLOWED_SPLITS
        or manifest["split"] == "final_test"
    ):
        raise ValueError("selected-view cache split is forbidden")
    for name in (
        "split_sha256",
        "ordered_identity_sha256",
        "coordinate_binding_sha256",
        "normalizer_sha256",
        "generator_checkpoint_sha256",
        "array_file_sha256",
        "logical_content_sha256",
    ):
        require_sha256(name, manifest[name])
    if (
        not isinstance(manifest["target_id"], str)
        or not manifest["target_id"]
        or not isinstance(manifest["array_file"], str)
        or Path(manifest["array_file"]).name != manifest["array_file"]
    ):
        raise ValueError("selected-view cache target/file name is invalid")
    if (
        manifest["dtype"] != "float32"
        or manifest["byte_order"] != "little"
        or manifest["invalid_particles_exactly_zero"] is not True
        or manifest["operation_order"]
        != SELECTED_VIEW_MATERIALIZATION_POLICY["operation_order"]
        or manifest["live_publication_max_abs_tolerance"] != 1.0e-6
        or manifest["final_test_cache_forbidden"] is not True
    ):
        raise ValueError("selected-view canonical materialization policy changed")
    view_shape = manifest["view_shape"]
    mask_shape = manifest["mask_shape"]
    if (
        not isinstance(view_shape, list)
        or len(view_shape) != 3
        or not isinstance(mask_shape, list)
        or len(mask_shape) != 2
        or view_shape[:2] != mask_shape
        or view_shape[2] not in {1, 2, 4, 8}
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in (*view_shape, *mask_shape)
        )
    ):
        raise ValueError("selected-view cache shapes are invalid")
    return {
        "ok": True,
        "content_hash": manifest["content_hash"],
        "split": manifest["split"],
        "canonical_float32": True,
    }


def audit_live_selected_view_equivalence(
    *,
    raw_view: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    cached_view: np.ndarray | torch.Tensor,
    cached_mask: np.ndarray | torch.Tensor,
    normalizer: ParticleViewNormalizer,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Prove that publication did not change the selected view coordinates."""

    if tolerance != 1.0e-6:
        raise ValueError("selected-view live/cache tolerance changed")
    live = normalize_particle_view(raw_view, mask, normalizer)
    if isinstance(live, torch.Tensor):
        live = live.detach().cpu().numpy()
    cached = (
        cached_view.detach().cpu().numpy()
        if isinstance(cached_view, torch.Tensor)
        else np.asarray(cached_view)
    )
    live_mask = (
        mask.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor)
        else np.asarray(mask)
    ).astype(bool, copy=False)
    stored_mask = (
        cached_mask.detach().cpu().numpy()
        if isinstance(cached_mask, torch.Tensor)
        else np.asarray(cached_mask)
    ).astype(bool, copy=False)
    if live.shape != cached.shape or not np.array_equal(live_mask, stored_mask):
        raise ValueError("live/cache selected-view shape or mask differs")
    if not np.isfinite(cached).all():
        raise ValueError("cached selected view contains non-finite values")
    maximum = float(
        np.max(
            np.abs(
                live.astype(np.float64, copy=False)
                - cached.astype(np.float64, copy=False)
            ),
            initial=0.0,
        )
    )
    if maximum > tolerance:
        raise ValueError(
            "live selected-view publication differs from canonical cache: "
            f"max_abs={maximum:.9g}, tolerance={tolerance:.9g}"
        )
    return {
        "ok": True,
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
        "valid_entries": int(live_mask.sum()),
        "canonical_dtype": "float32",
        "byte_order": "little",
    }


def finalize_selected_view_coordinate(
    output_dir: str | Path,
    *,
    target_id: str,
    target_selection_sha256: str,
    raw_views_by_split: Mapping[str, np.ndarray | torch.Tensor],
    masks_by_split: Mapping[str, np.ndarray | torch.Tensor],
    split_sha256_by_split: Mapping[str, str],
    ordered_identity_sha256_by_split: Mapping[str, str],
    coordinate_parent_hashes: Mapping[str, str],
    coordinate_definition: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute, bind, publish, reload, and audit the selected coordinates."""

    require_sha256("target_selection_sha256", target_selection_sha256)
    splits = set(raw_views_by_split)
    if splits != set(masks_by_split):
        raise ValueError("selected-view raw view/mask split inventory differs")
    required = {"train", "model_val_stop", "model_val_select"}
    if not required.issubset(splits):
        raise ValueError("selected-view publication is missing required splits")
    if not splits.issubset(PARTICLE_VIEW_CACHE_ALLOWED_SPLITS):
        raise ValueError("selected-view publication includes a forbidden split")
    if set(split_sha256_by_split) != splits or set(
        ordered_identity_sha256_by_split
    ) != splits:
        raise ValueError("selected-view split identity inventory differs")
    generator_sha = coordinate_parent_hashes.get(
        "generator_checkpoint_sha256"
    )
    require_sha256("generator_checkpoint_sha256", generator_sha)
    normalizer = fit_particle_view_normalizer(
        raw_views_by_split["train"],
        masks_by_split["train"],
        train_split_sha256=split_sha256_by_split["train"],
        generator_checkpoint_sha256=generator_sha,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    normalizer_artifact = write_particle_view_normalizer(
        root / "selected_view_normalizer.json", normalizer
    )
    parents = dict(coordinate_parent_hashes)
    parents["normalizer_sha256"] = normalizer_artifact["content_hash"]
    coordinate = build_view_coordinate_binding(
        parent_hashes=parents,
        coordinate_definition=coordinate_definition,
    )
    write_immutable_json(root / "selected_view_coordinate.json", coordinate)
    manifests = {}
    audits = {}
    for split in sorted(splits):
        manifest = publish_selected_view_cache(
            root / "caches",
            split=split,
            split_sha256=split_sha256_by_split[split],
            ordered_identity_sha256=ordered_identity_sha256_by_split[split],
            raw_view=raw_views_by_split[split],
            mask=masks_by_split[split],
            normalizer=normalizer,
            coordinate_binding_sha256=coordinate["content_hash"],
            target_id=target_id,
        )
        cache_view, cache_mask, _ = load_selected_view_cache(
            root / "caches" / f"{split}_selected_views.json",
            expected_coordinate_binding_sha256=coordinate["content_hash"],
            expected_split_sha256=split_sha256_by_split[split],
            expected_normalizer_sha256=normalizer_artifact["content_hash"],
        )
        audits[split] = audit_live_selected_view_equivalence(
            raw_view=raw_views_by_split[split],
            mask=masks_by_split[split],
            cached_view=cache_view,
            cached_mask=cache_mask,
            normalizer=normalizer,
        )
        manifests[split] = manifest["content_hash"]
    publication = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_COORDINATE_CONTRACT,
            "target_id": target_id,
            "target_selection_sha256": target_selection_sha256,
            "generator_checkpoint_sha256": generator_sha,
            "normalizer_sha256": normalizer_artifact["content_hash"],
            "coordinate_binding_sha256": coordinate["content_hash"],
            "cache_manifest_sha256_by_split": manifests,
            "live_publication_audits": audits,
            "normalizer_recomputed_after_selection": True,
            "consumer_reinitialization_required_after_publication": True,
            "final_test_materialized": False,
        }
    )
    write_immutable_json(
        root / "selected_view_publication.json", publication
    )
    return publication


__all__ = [
    "PARTICLE_VIEW_CACHE_ALLOWED_SPLITS",
    "PARTICLE_VIEW_CACHE_CONTRACT",
    "PARTICLE_VIEW_FINAL_COORDINATE_CONTRACT",
    "PARTICLE_VIEW_NORMALIZER_CONTRACT",
    "ParticleViewNormalizer",
    "audit_live_selected_view_equivalence",
    "fit_particle_view_normalizer",
    "finalize_selected_view_coordinate",
    "load_particle_view_normalizer",
    "load_selected_view_cache",
    "normalize_particle_view",
    "publish_selected_view_cache",
    "quantized_view_diagnostics",
    "validate_selected_view_cache_manifest",
    "write_particle_view_normalizer",
]
