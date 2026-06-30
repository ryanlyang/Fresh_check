"""Label-filter helpers for set-matching command-line tools."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jetclass_fresh.jetclass_data import LABEL_NAMES, load_split_manifest


def label_names_to_manifest_indices(values: Sequence[str], *, manifest_path: str | Path | None = None) -> tuple[int, ...]:
    """Resolve label names against the active manifest's label space.

    Fresh binary manifests remap source JetClass labels to compact labels, e.g.
    QCD/Hgg becomes ``QCD -> 0`` and ``Hgg -> 1``.  The old global JetClass
    mapping would resolve Hgg to 3, accidentally filtering out the compact
    signal rows.  Prefer the manifest class names when a manifest is available;
    fall back to the global 10-class names for legacy callers.
    """

    if not values:
        return ()

    class_names: tuple[str, ...]
    if manifest_path is not None:
        try:
            manifest = load_split_manifest(manifest_path)
            class_names = tuple(str(name) for name in manifest.class_names)
        except FileNotFoundError:
            class_names = tuple(LABEL_NAMES)
    else:
        class_names = tuple(LABEL_NAMES)

    by_name = {name: index for index, name in enumerate(class_names)}
    output: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            output.append(int(text))
            continue
        if text not in by_name:
            raise ValueError(f"Unknown label {text!r}; expected one of {list(class_names)}")
        output.append(by_name[text])

    if len(set(output)) != len(output):
        raise ValueError(f"label filter contains duplicate resolved labels: {output}")
    return tuple(output)
