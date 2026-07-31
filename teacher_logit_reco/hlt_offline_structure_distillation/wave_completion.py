"""Race-safe exact-coverage completion for registered row arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ROW_WAVE_COMPLETION_CONTRACT,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


def build_row_wave_completion(
    *,
    wave_id: str,
    expected_rows: Mapping[str, Mapping[str, Any]],
    row_artifacts: Mapping[str, Mapping[str, Any]],
    expected_contract: str,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not wave_id
        or not expected_rows
        or set(row_artifacts) != set(expected_rows)
    ):
        raise ValueError("row-wave coverage differs")
    hashes = {}
    for row_id, artifact in sorted(row_artifacts.items()):
        validate_content_hash(artifact, expected_contract=expected_contract)
        if artifact.get("source") != dict(source):
            raise ValueError("row-wave artifact source differs")
        hashes[row_id] = artifact["content_hash"]
    return with_content_hash(
        {
            "contract": ROW_WAVE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "wave_id": str(wave_id),
            "expected_rows": {
                str(key): dict(value)
                for key, value in sorted(expected_rows.items())
            },
            "row_artifact_hashes": hashes,
            "parent_hashes": dict(sorted(parent_hashes.items())),
            "row_count": len(hashes),
            "exact_row_coverage": True,
            "performance_based_termination": False,
        }
    )


def try_finalize_row_wave(
    *,
    wave_id: str,
    expected_paths: Mapping[str, str | Path],
    expected_rows: Mapping[str, Mapping[str, Any]],
    expected_contract: str,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any] | None:
    paths = {key: Path(value) for key, value in expected_paths.items()}
    if set(paths) != set(expected_rows):
        raise ValueError("row-wave path coverage differs")
    if not all(path.is_file() for path in paths.values()):
        return None
    artifacts = {
        key: load_hashed_json(path, expected_contract=expected_contract)
        for key, path in paths.items()
    }
    artifact = build_row_wave_completion(
        wave_id=wave_id,
        expected_rows=expected_rows,
        row_artifacts=artifacts,
        expected_contract=expected_contract,
        parent_hashes=parent_hashes,
        source=source,
    )
    write_immutable_json(output, artifact)
    return artifact


__all__ = ["build_row_wave_completion", "try_finalize_row_wave"]
