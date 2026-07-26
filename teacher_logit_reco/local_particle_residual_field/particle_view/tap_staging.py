"""Allocation-local RAM staging for frozen contextual teacher taps.

Only compact float16 tap tensors, boolean masks, and integer jet identities
live in this object.  No function in this module writes token payloads to
persistent storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import (
    canonical_json_bytes,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)


PARTICLE_VIEW_STAGED_TAP_CONTRACT = "particle_view_ram_teacher_tap_v1"
PARTICLE_VIEW_STAGED_TAP_RESERVATION_CONTRACT = (
    "particle_view_ram_teacher_tap_reservation_v1"
)
FLOAT16_CONVERSION_POLICY = (
    "ieee754_binary16_round_to_nearest_ties_to_even_no_stochastic_rounding_v1"
)


def _identity_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.int16: np.dtype("<i2"),
        torch.int32: np.dtype("<i4"),
        torch.int64: np.dtype("<i8"),
    }
    if dtype not in mapping:
        raise ValueError("jet identities must use int16, int32, or int64")
    return mapping[dtype]


def _canonical_stage_arrays(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    jet_identities: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tokens.ndim != 3 or tokens.dtype is not torch.float32:
        raise ValueError("teacher tokens must be float32 [jets,particles,width]")
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0, :]
    if mask.shape != tokens.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("teacher mask must be bool [jets,particles]")
    if jet_identities.ndim not in {1, 2} or jet_identities.shape[0] != tokens.shape[0]:
        raise ValueError("jet identities must have one row per staged jet")
    identity_dtype = _identity_numpy_dtype(jet_identities.dtype)

    source = tokens.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    mask_array = mask.detach().cpu().contiguous().numpy().astype(np.bool_, copy=False)
    valid_source = source[mask_array]
    if not np.isfinite(valid_source).all():
        raise ValueError("valid teacher tap entries contain nonfinite values")
    if valid_source.size and np.abs(valid_source).max() > np.finfo(np.float16).max:
        raise ValueError("teacher taps exceed finite float16 range")
    sanitized = np.where(mask_array[:, :, None], source, np.float32(0.0))
    staged = sanitized.astype("<f2", casting="unsafe", copy=True)
    if not np.isfinite(staged[mask_array]).all():
        raise ValueError("float16 conversion produced nonfinite valid entries")
    identities = (
        jet_identities.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(identity_dtype, casting="safe", copy=True)
    )
    return source, staged, mask_array.copy(), identities


def _logical_content_hash(
    *,
    metadata: Mapping[str, Any],
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(dict(metadata)))
    digest.update(b"\0tokens\0")
    digest.update(tokens.astype("<f2", copy=False).tobytes(order="C"))
    digest.update(b"\0mask\0")
    digest.update(mask.astype(np.bool_, copy=False).tobytes(order="C"))
    digest.update(b"\0identities\0")
    digest.update(identities.tobytes(order="C"))
    return digest.hexdigest()


def build_tap_stage_reservation(
    *,
    source_role: str,
    source_manifest_sha256: str,
    logical_split_sha256: str,
    ordered_identity_sha256: str,
    teacher_checkpoint_sha256: str,
    tap_spec_sha256: str,
    jets: int,
    max_particles: int,
    token_width: int,
    identity_columns: int,
    identity_dtype: str = "int64",
) -> dict[str, Any]:
    if source_role not in {"offline_teacher", "hlt_memory_control"}:
        raise ValueError("source_role must be offline_teacher or hlt_memory_control")
    hashes = {
        "source_manifest_sha256": source_manifest_sha256,
        "logical_split_sha256": logical_split_sha256,
        "ordered_identity_sha256": ordered_identity_sha256,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "tap_spec_sha256": tap_spec_sha256,
    }
    for name, value in hashes.items():
        require_sha256(name, value)
    dimensions = {
        "jets": jets,
        "max_particles": max_particles,
        "token_width": token_width,
        "identity_columns": identity_columns,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions.values()
    ):
        raise ValueError("reservation dimensions must be positive integers")
    if identity_dtype not in {"int16", "int32", "int64"}:
        raise ValueError("identity_dtype is unsupported")
    token_bytes = jets * max_particles * token_width * 2
    mask_bytes = jets * max_particles
    identity_bytes = jets * identity_columns * int(identity_dtype[3:]) // 8
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_STAGED_TAP_RESERVATION_CONTRACT,
            "source_role": source_role,
            **hashes,
            "shape": [jets, max_particles, token_width],
            "token_dtype": "float16",
            "mask_dtype": "bool",
            "identity_shape": [jets, identity_columns],
            "identity_dtype": identity_dtype,
            "conversion_policy": FLOAT16_CONVERSION_POLICY,
            "persistent_storage_allowed": False,
            "reserved_bytes": token_bytes + mask_bytes + identity_bytes,
        }
    )


def validate_tap_stage_reservation(payload: Mapping[str, Any]) -> int:
    validate_content_hash(
        payload,
        expected_contract=PARTICLE_VIEW_STAGED_TAP_RESERVATION_CONTRACT,
    )
    expected_fields = {
        "contract",
        "source_role",
        "source_manifest_sha256",
        "logical_split_sha256",
        "ordered_identity_sha256",
        "teacher_checkpoint_sha256",
        "tap_spec_sha256",
        "shape",
        "token_dtype",
        "mask_dtype",
        "identity_shape",
        "identity_dtype",
        "conversion_policy",
        "persistent_storage_allowed",
        "reserved_bytes",
        "content_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("tap-stage reservation field inventory mismatch")
    for name in (
        "source_manifest_sha256",
        "logical_split_sha256",
        "ordered_identity_sha256",
        "teacher_checkpoint_sha256",
        "tap_spec_sha256",
    ):
        require_sha256(name, payload[name])
    if payload["source_role"] not in {"offline_teacher", "hlt_memory_control"}:
        raise ValueError("tap-stage reservation source role changed")
    if (
        payload["token_dtype"] != "float16"
        or payload["mask_dtype"] != "bool"
        or payload["conversion_policy"] != FLOAT16_CONVERSION_POLICY
        or payload["persistent_storage_allowed"] is not False
    ):
        raise ValueError("tap-stage reservation conversion/storage policy changed")
    if (
        not isinstance(payload["shape"], list)
        or len(payload["shape"]) != 3
        or not isinstance(payload["identity_shape"], list)
        or len(payload["identity_shape"]) != 2
    ):
        raise ValueError("tap-stage reservation shapes are invalid")
    dimensions = [*payload["shape"], *payload["identity_shape"]]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        raise ValueError("tap-stage reservation dimensions are invalid")
    if (
        payload["identity_shape"][0] != payload["shape"][0]
        or payload["mask_dtype"] != "bool"
    ):
        raise ValueError("tap-stage identity/mask dimensions are inconsistent")
    if payload["identity_dtype"] not in {"int16", "int32", "int64"}:
        raise ValueError("tap-stage reservation identity dtype changed")
    token_bytes = math.prod(payload["shape"]) * 2
    mask_bytes = payload["shape"][0] * payload["shape"][1]
    identity_bytes = (
        math.prod(payload["identity_shape"])
        * int(payload["identity_dtype"][3:])
        // 8
    )
    expected_bytes = token_bytes + mask_bytes + identity_bytes
    if payload["reserved_bytes"] != expected_bytes:
        raise ValueError("tap-stage reservation byte accounting mismatch")
    return expected_bytes


@dataclass
class StagedTeacherTap:
    """RAM-resident staged tap plus its immutable logical-content manifest."""

    tokens: torch.Tensor
    mask: torch.Tensor
    jet_identities: torch.Tensor
    manifest: dict[str, Any]

    @property
    def logical_content_hash(self) -> str:
        return str(self.manifest["logical_content_sha256"])

    @property
    def nbytes(self) -> int:
        return (
            self.tokens.nelement() * self.tokens.element_size()
            + self.mask.nelement() * self.mask.element_size()
            + self.jet_identities.nelement() * self.jet_identities.element_size()
        )

    def promote(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        validate_staged_teacher_tap(self)
        return self.tokens.to(device=device, dtype=torch.float32)


def stage_teacher_tap_float16(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    jet_identities: torch.Tensor,
    *,
    reservation: Mapping[str, Any],
) -> StagedTeacherTap:
    """Convert once to canonical binary16 and retain the result in CPU RAM."""

    validate_tap_stage_reservation(reservation)
    source, staged, mask_array, identities = _canonical_stage_arrays(
        tokens, mask, jet_identities
    )
    if list(staged.shape) != reservation["shape"]:
        raise ValueError("staged tap shape differs from reservation")
    expected_identity_shape = list(identities.shape)
    if identities.ndim == 1:
        expected_identity_shape = [identities.shape[0], 1]
        identities = identities[:, None]
    if expected_identity_shape != reservation["identity_shape"]:
        raise ValueError("staged identity shape differs from reservation")
    if str(identities.dtype) != reservation["identity_dtype"]:
        raise ValueError("staged identity dtype differs from reservation")
    metadata = {
        "token_dtype": "float16",
        "token_byte_order": "little",
        "token_shape": list(staged.shape),
        "mask_dtype": "bool",
        "mask_shape": list(mask_array.shape),
        "identity_dtype": str(identities.dtype),
        "identity_shape": list(identities.shape),
        "conversion_policy": FLOAT16_CONVERSION_POLICY,
        "reservation_sha256": reservation["content_hash"],
    }
    logical_hash = _logical_content_hash(
        metadata=metadata,
        tokens=staged,
        mask=mask_array,
        identities=identities,
    )
    manifest = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STAGED_TAP_CONTRACT,
            "reservation_sha256": reservation["content_hash"],
            "source_role": reservation["source_role"],
            "source_manifest_sha256": reservation["source_manifest_sha256"],
            "logical_split_sha256": reservation["logical_split_sha256"],
            "ordered_identity_sha256": reservation["ordered_identity_sha256"],
            "teacher_checkpoint_sha256": reservation[
                "teacher_checkpoint_sha256"
            ],
            "tap_spec_sha256": reservation["tap_spec_sha256"],
            **metadata,
            "logical_content_sha256": logical_hash,
            "valid_token_count": int(mask_array.sum()),
            "ram_bytes": int(staged.nbytes + mask_array.nbytes + identities.nbytes),
            "persistent_storage_allowed": False,
        }
    )
    staged_object = StagedTeacherTap(
        tokens=torch.from_numpy(staged),
        mask=torch.from_numpy(mask_array),
        jet_identities=torch.from_numpy(identities),
        manifest=manifest,
    )
    if staged_object.nbytes != manifest["ram_bytes"]:
        raise RuntimeError("staged RAM byte accounting mismatch")
    validate_staged_teacher_tap(staged_object, reservation=reservation)
    _ = source  # Retained only to make the one-way conversion explicit.
    return staged_object


def validate_staged_teacher_tap(
    staged: StagedTeacherTap,
    *,
    reservation: Mapping[str, Any] | None = None,
) -> str:
    validate_content_hash(
        staged.manifest, expected_contract=PARTICLE_VIEW_STAGED_TAP_CONTRACT
    )
    expected_fields = {
        "contract",
        "reservation_sha256",
        "source_role",
        "source_manifest_sha256",
        "logical_split_sha256",
        "ordered_identity_sha256",
        "teacher_checkpoint_sha256",
        "tap_spec_sha256",
        "token_dtype",
        "token_byte_order",
        "token_shape",
        "mask_dtype",
        "mask_shape",
        "identity_dtype",
        "identity_shape",
        "conversion_policy",
        "logical_content_sha256",
        "valid_token_count",
        "ram_bytes",
        "persistent_storage_allowed",
        "content_hash",
    }
    if set(staged.manifest) != expected_fields:
        raise ValueError("staged teacher-tap manifest field inventory mismatch")
    if staged.tokens.device.type != "cpu" or staged.tokens.dtype is not torch.float16:
        raise ValueError("staged tokens must remain CPU float16")
    if staged.mask.device.type != "cpu" or staged.mask.dtype is not torch.bool:
        raise ValueError("staged mask must remain CPU bool")
    if staged.jet_identities.device.type != "cpu":
        raise ValueError("staged identities must remain in CPU RAM")
    _, tokens, mask, identities = _canonical_stage_arrays(
        staged.tokens.to(dtype=torch.float32),
        staged.mask,
        staged.jet_identities,
    )
    metadata = {
        "token_dtype": "float16",
        "token_byte_order": "little",
        "token_shape": list(tokens.shape),
        "mask_dtype": "bool",
        "mask_shape": list(mask.shape),
        "identity_dtype": str(identities.dtype),
        "identity_shape": list(identities.shape),
        "conversion_policy": FLOAT16_CONVERSION_POLICY,
        "reservation_sha256": staged.manifest["reservation_sha256"],
    }
    for name, value in metadata.items():
        if staged.manifest[name] != value:
            raise ValueError(f"staged teacher tap {name} metadata changed")
    for name in (
        "reservation_sha256",
        "source_manifest_sha256",
        "logical_split_sha256",
        "ordered_identity_sha256",
        "teacher_checkpoint_sha256",
        "tap_spec_sha256",
        "logical_content_sha256",
    ):
        require_sha256(name, staged.manifest[name])
    if staged.manifest["valid_token_count"] != int(mask.sum()):
        raise ValueError("staged teacher tap valid-token count changed")
    if staged.manifest["persistent_storage_allowed"] is not False:
        raise ValueError("staged teacher taps cannot enter persistent storage")
    logical_hash = _logical_content_hash(
        metadata=metadata,
        tokens=tokens,
        mask=mask,
        identities=identities,
    )
    if logical_hash != staged.manifest["logical_content_sha256"]:
        raise ValueError("staged teacher tap logical content changed in RAM")
    if staged.nbytes != staged.manifest["ram_bytes"]:
        raise ValueError("staged teacher tap RAM byte accounting changed")
    if reservation is not None:
        reserved_bytes = validate_tap_stage_reservation(reservation)
        if staged.manifest["reservation_sha256"] != reservation["content_hash"]:
            raise ValueError("staged teacher tap uses a different reservation")
        for name in (
            "source_role",
            "source_manifest_sha256",
            "logical_split_sha256",
            "ordered_identity_sha256",
            "teacher_checkpoint_sha256",
            "tap_spec_sha256",
        ):
            if staged.manifest[name] != reservation[name]:
                raise ValueError(f"staged teacher tap {name} differs from reservation")
        if staged.nbytes != reserved_bytes:
            raise ValueError("staged teacher tap exceeds or underuses reservation")
    return logical_hash


def _rounding_error_audit(
    source_float32: np.ndarray,
    staged_float16: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    source = source_float32[mask].reshape(-1)
    rounded16 = staged_float16[mask].reshape(-1)
    rounded = rounded16.astype(np.float32)
    if source.size == 0:
        return {"max_abs_error": 0.0, "max_half_bracket_ratio": 0.0}
    lower16 = np.where(
        rounded <= source,
        rounded16,
        np.nextafter(rounded16, np.float16(-np.inf)),
    ).astype(np.float32)
    upper16 = np.where(
        rounded >= source,
        rounded16,
        np.nextafter(rounded16, np.float16(np.inf)),
    ).astype(np.float32)
    half_spacing = (upper16 - lower16) * np.float32(0.5)
    error = np.abs(source - rounded)
    tolerance = np.maximum(
        np.float32(np.finfo(np.float32).eps),
        np.abs(half_spacing) * np.float32(1.0e-6),
    )
    if np.any(error > half_spacing + tolerance):
        raise ValueError("float16 conversion violates nearest-even bracket bound")
    ratio = np.divide(
        error,
        half_spacing,
        out=np.zeros_like(error),
        where=half_spacing > 0,
    )
    return {
        "max_abs_error": float(error.max(initial=np.float32(0.0))),
        "max_half_bracket_ratio": float(ratio.max(initial=np.float32(0.0))),
    }


def audit_live_tap_equivalence(
    staged: StagedTeacherTap,
    fresh_float32_tokens: torch.Tensor,
    fresh_mask: torch.Tensor,
    fresh_jet_identities: torch.Tensor,
) -> dict[str, Any]:
    """Require exact masks/identities and bitwise-equal canonical float16 taps."""

    logical_hash = validate_staged_teacher_tap(staged)
    source, converted, mask, identities = _canonical_stage_arrays(
        fresh_float32_tokens,
        fresh_mask,
        fresh_jet_identities,
    )
    staged_bits = staged.tokens.numpy().astype("<f2", copy=False).view("<u2")
    converted_bits = converted.view("<u2")
    if not np.array_equal(staged_bits, converted_bits):
        raise ValueError("live and staged float16 teacher taps differ bitwise")
    if not np.array_equal(staged.mask.numpy(), mask):
        raise ValueError("live and staged teacher masks differ")
    if identities.ndim == 1:
        identities = identities[:, None]
    if not np.array_equal(staged.jet_identities.numpy(), identities):
        raise ValueError("live and staged jet identities differ")
    rounding = _rounding_error_audit(source, converted, mask)
    return {
        "ok": True,
        "logical_content_sha256": logical_hash,
        "bitwise_equal_float16": True,
        "exact_mask": True,
        "exact_identities": True,
        **rounding,
    }


__all__ = [
    "FLOAT16_CONVERSION_POLICY",
    "PARTICLE_VIEW_STAGED_TAP_CONTRACT",
    "PARTICLE_VIEW_STAGED_TAP_RESERVATION_CONTRACT",
    "StagedTeacherTap",
    "audit_live_tap_equivalence",
    "build_tap_stage_reservation",
    "stage_teacher_tap_float16",
    "validate_tap_stage_reservation",
    "validate_staged_teacher_tap",
]
