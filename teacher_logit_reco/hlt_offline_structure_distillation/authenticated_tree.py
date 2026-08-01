"""Fail-closed lazy access to authenticated relational CA-tree splits."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relational_part import (
    ANGULAR_TREE_SHARD_CONTRACT,
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    sha256_file,
    unpack_tree_shard,
)

from .contracts import load_hashed_json, validate_content_hash, with_content_hash


def compatible_artifact_content_hashes(
    artifact: Mapping[str, Any],
) -> frozenset[str]:
    """Return hashes accepted for a source-bound wrapper and its raw payload."""

    validate_content_hash(artifact)
    hashes = {str(artifact["content_hash"])}
    if "source" in artifact:
        raw = dict(artifact)
        raw.pop("content_hash")
        raw.pop("source")
        hashes.add(str(with_content_hash(raw)["content_hash"]))
    return frozenset(hashes)


def resolve_tree_parent_lineage(
    manifest_parents: Mapping[str, Any],
    *,
    hlt_content_sha256: str,
    tree_resource: Mapping[str, Any],
    tree_backend: Mapping[str, Any],
) -> dict[str, str]:
    """Authenticate exact tree lineage through source-bound parent aliases."""

    actual = {str(key): str(value) for key, value in manifest_parents.items()}
    required = {
        "hlt_content_sha256",
        "tree_resource_sha256",
        "backend_manifest_sha256",
    }
    if set(actual) != required:
        raise ValueError("tree split parent schema differs")
    if actual["hlt_content_sha256"] != str(hlt_content_sha256):
        raise ValueError("tree split input-content parent differs")
    if actual["tree_resource_sha256"] not in compatible_artifact_content_hashes(
        tree_resource
    ):
        raise ValueError("tree split resource parent differs")
    if actual["backend_manifest_sha256"] not in compatible_artifact_content_hashes(
        tree_backend
    ):
        raise ValueError("tree split backend parent differs")
    return actual


def _identity_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for raw in values:
        digest.update(str(raw).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class AuthenticatedTreeShard:
    shard_index: int
    start: int
    stop: int
    npz_path: Path
    metadata_path: Path
    npz_sha256: str
    metadata_sha256: str
    identity_sha256: str

    @property
    def event_count(self) -> int:
        return self.stop - self.start


class AuthenticatedTreeSplit:
    """Verify a complete split before exposing bounded, lazy shard loads."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_identities: Sequence[Any] | None = None,
        expected_parents: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.manifest = load_hashed_json(
            self.root / "manifest.json",
            expected_contract=ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
        )
        if (
            expected_parents is not None
            and self.manifest.get("parents") != dict(expected_parents)
        ):
            raise ValueError("tree split parents differ from expected lineage")
        rows = tuple(self.manifest.get("shards", ()))
        expected_npz_names = {
            f"shard_{index:05d}.npz" for index in range(len(rows))
        }
        expected_metadata_names = {
            f"shard_{index:05d}.metadata.json" for index in range(len(rows))
        }
        shard_root = self.root / "shards"
        actual_npz_names = {path.name for path in shard_root.glob("shard_*.npz")}
        actual_metadata_names = {
            path.name for path in shard_root.glob("shard_*.metadata.json")
        }
        if (
            not rows
            or actual_npz_names != expected_npz_names
            or actual_metadata_names != expected_metadata_names
        ):
            raise ValueError("tree split filenames or shard count differ")

        records = []
        cursor = 0
        for index, row in enumerate(rows):
            npz_path = shard_root / f"shard_{index:05d}.npz"
            metadata_path = shard_root / f"shard_{index:05d}.metadata.json"
            metadata = load_hashed_json(
                metadata_path, expected_contract=ANGULAR_TREE_SHARD_CONTRACT
            )
            count = int(row.get("jet_count", -1))
            if (
                int(row.get("shard_index", -1)) != index
                or count <= 0
                or row.get("metadata_sha256") != metadata["content_hash"]
                or row.get("npz_sha256") != metadata.get("npz_sha256")
                or row.get("identity_sha256") != metadata.get("identity_sha256")
                or int(metadata.get("jet_count", -1)) != count
                or metadata.get("parents") != self.manifest.get("parents")
                or sha256_file(npz_path) != row.get("npz_sha256")
            ):
                raise ValueError("tree split shard attestation differs")
            with np.load(npz_path, allow_pickle=False) as packed:
                identities = tuple(str(value) for value in packed["identity"])
            if (
                len(identities) != count
                or _identity_sha256(identities) != row.get("identity_sha256")
                or len(set(identities)) != count
            ):
                raise ValueError("tree split identity coverage differs")
            if expected_identities is not None:
                expected = tuple(
                    str(value)
                    for value in expected_identities[cursor : cursor + count]
                )
                if identities != expected:
                    raise ValueError(
                        "tree split identities differ from expected canonical order"
                    )
            records.append(
                AuthenticatedTreeShard(
                    shard_index=index,
                    start=cursor,
                    stop=cursor + count,
                    npz_path=npz_path,
                    metadata_path=metadata_path,
                    npz_sha256=str(row["npz_sha256"]),
                    metadata_sha256=str(row["metadata_sha256"]),
                    identity_sha256=str(row["identity_sha256"]),
                )
            )
            cursor += count
        if (
            cursor != int(self.manifest.get("jet_count", -1))
            or (
                expected_identities is not None
                and cursor != len(expected_identities)
            )
        ):
            raise ValueError("tree split population coverage differs")
        self.records = tuple(records)
        self.ends = tuple(record.stop for record in records)

    def __len__(self) -> int:
        return self.ends[-1]

    def shard_for_event(self, event_index: int) -> int:
        index = int(event_index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect_right(self.ends, index)

    def _revalidate(self, record: AuthenticatedTreeShard) -> None:
        metadata = load_hashed_json(
            record.metadata_path, expected_contract=ANGULAR_TREE_SHARD_CONTRACT
        )
        if (
            metadata["content_hash"] != record.metadata_sha256
            or metadata.get("npz_sha256") != record.npz_sha256
            or metadata.get("identity_sha256") != record.identity_sha256
            or metadata.get("parents") != self.manifest.get("parents")
            or sha256_file(record.npz_path) != record.npz_sha256
        ):
            raise ValueError("tree shard changed after split authentication")

    def load_shard(
        self, shard_index: int, *, rows: Sequence[int] | None = None
    ) -> tuple[list[str], list[dict[str, Any]]]:
        record = self.records[int(shard_index)]
        self._revalidate(record)
        identities, trees = unpack_tree_shard(record.npz_path, rows=rows)
        if (
            len(identities) != record.event_count
            or _identity_sha256(identities) != record.identity_sha256
        ):
            raise ValueError("loaded tree shard identity attestation differs")
        return identities, trees

    def verified_shard_path(self, shard_index: int) -> Path:
        """Return a shard path only after immediate byte revalidation."""

        record = self.records[int(shard_index)]
        self._revalidate(record)
        return record.npz_path

    def load_event_rows(
        self,
        event_indices: Sequence[int],
        *,
        expected_identities: Sequence[Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Load arbitrary event rows while retaining at most one tree shard.

        Rows are grouped by authenticated shard, decoded selectively, and then
        restored to caller order.  This is the common production path for
        target extraction, normalizer fitting, graph loading, and teacher
        inference; no population-wide identity-to-tree mapping is constructed.
        """

        indices = tuple(int(value) for value in event_indices)
        if not indices:
            return ()
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, event_index in enumerate(indices):
            shard_index = self.shard_for_event(event_index)
            record = self.records[shard_index]
            grouped.setdefault(shard_index, []).append(
                (output_index, event_index - record.start)
            )
        output: list[dict[str, Any] | None] = [None] * len(indices)
        observed_identities: list[str | None] = [None] * len(indices)
        for shard_index, requests in sorted(grouped.items()):
            rows = [local_index for _, local_index in requests]
            all_identities, trees = self.load_shard(shard_index, rows=rows)
            for (output_index, local_index), tree in zip(requests, trees):
                output[output_index] = tree
                observed_identities[output_index] = all_identities[local_index]
        if any(value is None for value in output):
            raise RuntimeError("selective tree loading did not fill every row")
        if expected_identities is not None:
            expected = tuple(str(value) for value in expected_identities)
            if tuple(observed_identities) != expected:
                raise ValueError("selected tree identities differ from active input")
        return tuple(value for value in output if value is not None)

    def event_indices_for_identities(
        self, identities: Sequence[Any]
    ) -> np.ndarray:
        """Resolve a bounded subrole against the authenticated full split."""

        requested = tuple(str(value) for value in identities)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("requested tree identities are empty or duplicated")
        wanted = set(requested)
        positions: dict[str, int] = {}
        for record in self.records:
            self._revalidate(record)
            with np.load(record.npz_path, allow_pickle=False) as packed:
                shard_ids = tuple(str(value) for value in packed["identity"])
            for local_index, identity in enumerate(shard_ids):
                if identity in wanted:
                    if identity in positions:
                        raise ValueError("tree split duplicates a requested identity")
                    positions[identity] = record.start + local_index
        if set(positions) != wanted:
            raise ValueError("tree split lacks requested subrole identities")
        return np.asarray([positions[value] for value in requested], dtype=np.int64)


__all__ = ["AuthenticatedTreeShard", "AuthenticatedTreeSplit"]
