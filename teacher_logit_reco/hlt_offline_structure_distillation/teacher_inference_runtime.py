"""Default label-blind inference adapter for locked HOSD relational teachers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import hashlib
import numpy as np
import torch

from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part import build_runtime_model, load_hashed_json

from .authenticated_tree import AuthenticatedTreeSplit
from .input_views import load_materialized_input_view
from .teachers import build_relational_teacher_adapter


FORBIDDEN_ARRAY_NAMES = frozenset(
    {"label", "labels", "class", "classes", "target_class", "y"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_label_blind_relational_adapter(
    *,
    teacher_id: str,
    teacher_lock: Mapping[str, Any],
    config: Mapping[str, Any],
):
    """Return the default adapter, identity population, and indexed batch provider."""

    required = {"input_npz", "screening_registry", "relation_normalizer"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"teacher inference config lacks {sorted(missing)}")
    input_path = Path(config["input_npz"])
    if _sha256(input_path) != config.get("input_npz_sha256"):
        raise ValueError("teacher inference input bytes differ from adapter lock")
    arrays, _ = load_materialized_input_view(
        input_path,
        expected_view_kind="canonical_offline",
        expected_source=teacher_lock.get("source"),
    )
    identities = arrays["identities"]
    raw_tokens = arrays["tokens"]
    mask = arrays["mask"]
    if identities.shape[0] != raw_tokens.shape[0] or mask.shape != raw_tokens.shape[:2]:
        raise ValueError("teacher inference input population/shape differs")
    screening = load_hashed_json(config["screening_registry"])
    relation = load_hashed_json(config["relation_normalizer"])
    region = (
        load_hashed_json(config["region_normalizer"])
        if config.get("region_normalizer") is not None
        else None
    )
    run_id = "RPT_BASE" if teacher_id == "O_BASE" else "RPT_FULL_ALL"
    if teacher_id == "O_FULLREL" and region is None:
        raise ValueError("O_FULLREL inference requires REGION normalization")
    tree_split = None
    if teacher_id == "O_FULLREL":
        if config.get("tree_cache_dir") is None:
            raise ValueError("O_FULLREL inference requires a prebuilt tree cache")
        expected_parents = config.get("tree_expected_parents")
        if not isinstance(expected_parents, Mapping):
            raise ValueError("O_FULLREL inference lacks exact tree parents")
        tree_split = AuthenticatedTreeSplit(
            config["tree_cache_dir"],
            expected_identities=identities,
            expected_parents=expected_parents,
        )
    model = build_runtime_model(
        run_id,
        screening_registry=screening,
        normalization_artifact=relation,
        region_normalization_artifact=region,
    )
    row = next(
        value for value in teacher_lock["teachers"] if value["teacher_id"] == teacher_id
    )
    checkpoint = torch.load(
        row["checkpoint_path"], map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("teacher checkpoint lacks model_state_dict")
    model.load_state_dict(state, strict=True)
    device_name = str(config.get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    model.to(device)
    adapter = build_relational_teacher_adapter(teacher_id=teacher_id, model=model)

    def batch_provider(indices: np.ndarray) -> dict[str, Any]:
        selected_tokens = raw_tokens[indices]
        selected_mask = mask[indices]
        inputs = build_particle_transformer_inputs_from_tokens(
            selected_tokens,
            selected_mask,
            source_view="offline",
        )
        batch: dict[str, Any] = {
            "points": torch.from_numpy(inputs.pf_points).float().to(device),
            "features": torch.from_numpy(inputs.pf_features).float().to(device),
            "lorentz_vectors": torch.from_numpy(inputs.pf_vectors).float().to(device),
            "mask": torch.from_numpy(inputs.pf_mask).bool().to(device),
            "raw_tokens": torch.from_numpy(selected_tokens).float().to(device),
        }
        if teacher_id == "O_FULLREL":
            selected_identities = identities[indices]
            batch["region_trees"] = tree_split.load_event_rows(
                indices,
                expected_identities=selected_identities,
            )
        return batch

    return adapter, identities, batch_provider


__all__ = [
    "FORBIDDEN_ARRAY_NAMES",
    "build_label_blind_relational_adapter",
]
