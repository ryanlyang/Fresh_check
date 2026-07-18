"""Source-family contracts for RAM-bundled, logit-only ABPH scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import canonical_hash
from .variants import resolve_variant_config, variant_spec


ABPH_BUNDLED_SCORING_CONTRACT = "adaptive_binary_bundled_scoring_v1"
ABPH_LOGIT_ONLY_ARTIFACT_CONTRACT = "adaptive_binary_logit_only_artifact_v1"
ABPH_LOGIT_ONLY_ARRAY_NAMES = (
    "logits",
    "labels",
    "jet_ids",
    "source_indices",
)


@dataclass(frozen=True)
class ScoringSourceFamily:
    key: str
    kind: str
    source_names: tuple[str, ...]
    independent_roots: bool = False
    joint_checkpoint_member: str | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.kind:
            raise ValueError("scoring source family lacks its identity")
        if self.kind == "joint_checkpoint":
            if not self.joint_checkpoint_member or self.source_names != (
                self.joint_checkpoint_member,
            ):
                raise ValueError("joint scoring family must name its exact checkpoint")
        elif self.joint_checkpoint_member is not None:
            raise ValueError("non-joint scoring family cannot name a joint checkpoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "source_names": list(self.source_names),
            "independent_roots": self.independent_roots,
            "joint_checkpoint_member": self.joint_checkpoint_member,
        }


def scoring_source_family(member: str) -> ScoringSourceFamily:
    base_name = str(member).split("__seed", 1)[0]
    spec = variant_spec(base_name)
    resolved = resolve_variant_config(base_name)
    run_id = str(spec.run_id)
    if spec.tier == "A":
        return ScoringSourceFamily("hlt_only", "hlt_only", ())
    if spec.tier == "F" and run_id not in {"F1", "F4"}:
        return ScoringSourceFamily(
            f"joint_checkpoint__{member}",
            "joint_checkpoint",
            (str(member),),
            joint_checkpoint_member=str(member),
        )
    if bool(resolved["model"]["fusion"].get("dual_hierarchy")) and run_id != "E11":
        return ScoringSourceFamily(
            "shared_root_dual", "shared_root_dual", ("E7_shared_root_dual",)
        )
    if run_id == "E11":
        return ScoringSourceFamily(
            "independent_root_dual",
            "independent_root_dual",
            ("D1_kt32_mh4_particles", "D2_ca32_mh4_particles"),
            independent_roots=True,
        )
    if str(resolved["model"]["hierarchy"].get("grouping")) == "cambridge_aachen":
        return ScoringSourceFamily(
            "d2_cambridge_aachen", "frozen_reconstructor", ("D2_ca32_mh4_particles",)
        )
    return ScoringSourceFamily(
        "d1_exclusive_kt", "frozen_reconstructor", ("D1_kt32_mh4_particles",)
    )


def group_scoring_members(
    members: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    seen: set[str] = set()
    for raw_member in members:
        member = str(raw_member)
        if member in seen:
            raise ValueError(f"duplicate scoring member {member}")
        seen.add(member)
        grouped.setdefault(scoring_source_family(member).key, []).append(member)
    return {name: tuple(values) for name, values in grouped.items()}


def source_generation_hash(
    family: ScoringSourceFamily,
    *,
    split: str,
    hlt_content_hash: str,
    jet_identity_hash: str,
    generator_source_hash: str,
    checkpoint_hashes: Mapping[str, str],
    consumer_schema_hashes: Sequence[str],
) -> str:
    required = {
        "split": split,
        "hlt_content_hash": hlt_content_hash,
        "jet_identity_hash": jet_identity_hash,
        "generator_source_hash": generator_source_hash,
    }
    if any(value in (None, "") for value in required.values()):
        raise ValueError(f"scoring source generation lacks provenance: {required}")
    if not checkpoint_hashes and family.kind != "hlt_only":
        raise ValueError("pseudo scoring source lacks checkpoint hashes")
    return canonical_hash(
        {
            "contract": ABPH_BUNDLED_SCORING_CONTRACT,
            "family": family.to_dict(),
            **required,
            "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
            "consumer_schema_hashes": list(consumer_schema_hashes),
        }
    )


def encode_logit_only_npz(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    jet_ids: np.ndarray,
    source_indices: np.ndarray,
) -> bytes:
    values = {
        "logits": np.asarray(logits, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "jet_ids": np.asarray(jet_ids, dtype=np.str_),
        "source_indices": np.asarray(source_indices, dtype=np.int64),
    }
    jets = int(values["logits"].shape[0])
    if values["logits"].ndim != 2 or values["logits"].shape[1] < 2:
        raise ValueError("logit-only output has an invalid class dimension")
    if any(values[name].shape[0] != jets for name in ABPH_LOGIT_ONLY_ARRAY_NAMES[1:]):
        raise ValueError("logit-only arrays differ in jet count")
    if not np.isfinite(values["logits"]).all():
        raise FloatingPointError("logit-only output contains nonfinite values")
    output = io.BytesIO()
    np.savez_compressed(output, **values)
    return output.getvalue()


def validate_logit_only_npz(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != set(ABPH_LOGIT_ONLY_ARRAY_NAMES):
            raise ValueError("scoring artifact contains non-logit payload arrays")
        logits = np.asarray(payload["logits"])
        labels = np.asarray(payload["labels"])
        jet_ids = np.asarray(payload["jet_ids"])
        indices = np.asarray(payload["source_indices"])
    if logits.dtype != np.float32 or labels.dtype != np.int64 or indices.dtype != np.int64:
        raise ValueError("scoring artifact dtypes differ from the logit-only contract")
    if jet_ids.dtype.kind != "U":
        raise ValueError("scoring artifact identities must be fixed-width strings")
    jets = int(logits.shape[0])
    if logits.ndim != 2 or any(
        value.shape != (jets,) for value in (labels, jet_ids, indices)
    ):
        raise ValueError("scoring artifact arrays are not aligned")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "contract": ABPH_LOGIT_ONLY_ARTIFACT_CONTRACT,
        "n_jets": jets,
        "n_classes": int(logits.shape[1]),
        "sha256": digest,
        "array_names": list(ABPH_LOGIT_ONLY_ARRAY_NAMES),
    }


__all__ = [
    "ABPH_BUNDLED_SCORING_CONTRACT",
    "ABPH_LOGIT_ONLY_ARRAY_NAMES",
    "ABPH_LOGIT_ONLY_ARTIFACT_CONTRACT",
    "ScoringSourceFamily",
    "encode_logit_only_npz",
    "group_scoring_members",
    "scoring_source_family",
    "source_generation_hash",
    "validate_logit_only_npz",
]
