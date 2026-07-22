"""Atomic publication helpers for immutable fusion campaign artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def publish_temporary_file(temporary: str | Path, destination: str | Path, *, overwrite: bool) -> None:
    """Publish a completed same-filesystem temporary file atomically.

    Immutable publication uses a hard link, which fails atomically when the
    destination already exists. Explicitly mutable call sites retain replace
    semantics.
    """

    temporary_path = Path(temporary)
    destination_path = Path(destination)
    if overwrite:
        os.replace(temporary_path, destination_path)
        return
    try:
        os.link(temporary_path, destination_path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination_path}") from exc
    temporary_path.unlink()


__all__ = ["publish_temporary_file"]
