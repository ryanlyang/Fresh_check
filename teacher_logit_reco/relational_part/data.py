"""Identity-preserving cached-HLT datasets for Step-6 training/evaluation."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

from .train import DeterministicEpochSampler

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class RelationalJetDataset(torch.utils.data.Dataset if torch is not None else object):
    def __init__(
        self,
        view: Any,
        *,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for relational datasets")
        self.tokens = np.asarray(view.tokens, dtype=np.float32)
        self.mask = np.asarray(view.mask, dtype=bool)
        self.labels = np.asarray(view.labels, dtype=np.int64)
        self.identities = tuple(view.jet_ids)
        self.split = str(view.split)
        self.region_trees = (
            None if region_trees is None else tuple(region_trees)
        )
        if self.region_trees is not None and len(self.region_trees) != len(self.labels):
            raise ValueError("REGION tree count differs from cached HLT jet count")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int):
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "label": self.labels[index],
            "identity": self.identities[index].key(),
            "region_tree": (
                None if self.region_trees is None else self.region_trees[index]
            ),
        }


def collate_relational_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for relational batches")
    tokens = np.stack([sample["tokens"] for sample in samples])
    mask = np.stack([sample["mask"] for sample in samples])
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens, mask, labels=labels, source_view="fixed_hlt"
    )
    output = {
        "points": torch.from_numpy(inputs.pf_points).float(),
        "features": torch.from_numpy(inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(inputs.pf_vectors).float(),
        "mask": torch.from_numpy(inputs.pf_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "raw_tokens": torch.from_numpy(tokens).float(),
        "event_identities": [sample["identity"] for sample in samples],
    }
    trees = [sample["region_tree"] for sample in samples]
    if any(tree is not None for tree in trees):
        if not all(tree is not None for tree in trees):
            raise ValueError("a batch cannot mix present and absent REGION trees")
        output["region_trees"] = trees
    return output


def make_relational_loader(
    dataset: RelationalJetDataset,
    *,
    seed: int,
    training: bool,
    batch_size: int = 64,
    num_workers: int = 0,
):
    if int(batch_size) != 64 or int(num_workers) != 0:
        raise ValueError("Step-6 loaders lock batch_size=64 and num_workers=0")
    sampler = (
        DeterministicEpochSampler(dataset, seed=int(seed))
        if training
        else torch.utils.data.SequentialSampler(dataset)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_relational_batch,
    )


__all__ = [
    "RelationalJetDataset",
    "collate_relational_batch",
    "make_relational_loader",
]
