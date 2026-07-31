"""Default label-blind inference adapter for locked HOSD relational teachers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import hashlib
import numpy as np
import torch

from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part import build_runtime_model, load_hashed_json
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard

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
    with np.load(input_path, allow_pickle=False) as archive:
        forbidden = FORBIDDEN_ARRAY_NAMES & set(archive.files)
        if forbidden:
            raise ValueError(
                f"teacher inference input exposes labels: {sorted(forbidden)}"
            )
        required_arrays = {"identity", "raw_tokens", "mask"}
        if not required_arrays.issubset(archive.files):
            raise ValueError("teacher inference input lacks identity/raw_tokens/mask")
        identities = tuple(str(value) for value in archive["identity"].tolist())
        raw_tokens = np.asarray(archive["raw_tokens"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=bool)
    if len(identities) != raw_tokens.shape[0] or mask.shape != raw_tokens.shape[:2]:
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
    tree_by_identity = None
    if teacher_id == "O_FULLREL":
        if config.get("tree_cache_dir") is None:
            raise ValueError("O_FULLREL inference requires a prebuilt tree cache")
        tree_root = Path(config["tree_cache_dir"])
        load_hashed_json(tree_root / "manifest.json")
        tree_by_identity = {}
        for shard_path in sorted((tree_root / "shards").glob("shard_*.npz")):
            shard_identities, shard_trees = unpack_tree_shard(shard_path)
            for identity, tree in zip(shard_identities, shard_trees):
                if identity in tree_by_identity:
                    raise ValueError("tree cache contains duplicate identities")
                tree_by_identity[identity] = tree
        if set(tree_by_identity) != set(identities):
            raise ValueError("tree cache identity population differs from teacher input")
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
            batch["region_trees"] = [
                tree_by_identity[identities[int(index)]] for index in indices
            ]
        return batch

    return adapter, identities, batch_provider


__all__ = [
    "FORBIDDEN_ARRAY_NAMES",
    "build_label_blind_relational_adapter",
]
