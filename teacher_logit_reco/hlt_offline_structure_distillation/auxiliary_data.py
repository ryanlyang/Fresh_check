"""Identity-aligned target joins and streamed HLT target construction for Stage D."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (
    DeterministicExpertSampler,
    collate_native_hlt_expert_batch,
)
from teacher_logit_reco.relation_expert_token_bridge.replicas import replica_for

from .normalization import validate_target_normalizer
from .extractors import ExtractorResources, extract_registered_target
from .normalization import NORMALIZED_CLIP

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _normalizer_row(normalizer: Mapping[str, Any], target_id: str) -> Mapping[str, Any]:
    validate_target_normalizer(normalizer)
    rows = [row for row in normalizer["targets"] if row["target_id"] == target_id]
    if len(rows) != 1:
        raise ValueError(f"target normalizer has no unique {target_id} row")
    return rows[0]


def normalize_channel_first_target(
    values: np.ndarray,
    masks: np.ndarray,
    *,
    target_id: str,
    normalizer: Mapping[str, Any],
) -> np.ndarray:
    """Normalize [C,...] streamed values without materializing an N x N cache."""

    raw = np.asarray(values, dtype=np.float32)
    valid = np.asarray(masks, dtype=bool)
    if raw.shape != valid.shape or raw.ndim < 2:
        raise ValueError("streamed target values and masks differ")
    row = _normalizer_row(normalizer, target_id)
    if int(row["component_count"]) != raw.shape[0]:
        raise ValueError("streamed target channel count differs")
    output = np.zeros_like(raw)
    for component in row["components"]:
        index = int(component["component_index"])
        selected = valid[index]
        if component["normalize"]:
            output[index][selected] = np.clip(
                (raw[index][selected] - float(component["center"]))
                / float(component["scale"]),
                *NORMALIZED_CLIP,
            )
        else:
            output[index][selected] = raw[index][selected]
    return output


class StreamedHLTTarget:
    """Recompute an HLT-self or pair target from the selected same-view replica."""

    def __init__(
        self,
        *,
        target_id: str,
        normalizer: Mapping[str, Any],
        resources: ExtractorResources,
        control_kind: str | None = None,
    ) -> None:
        self.target_id = str(target_id)
        self.normalizer = normalizer
        self.resources = resources
        if control_kind not in {None, "target_mean"}:
            raise ValueError("streamed target supports only direct or target-mean values")
        self.control_kind = control_kind

    def __call__(self, sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        target = extract_registered_target(
            self.target_id,
            np.asarray(sample["tokens"], dtype=np.float32)[None],
            np.asarray(sample["mask"], dtype=bool)[None],
            resources=self.resources,
            trees=(
                None
                if sample.get("region_tree") is None
                else (sample["region_tree"],)
            ),
        )
        values = target.values.detach().cpu().numpy()[0]
        masks = target.loss_mask.detach().cpu().numpy()[0].astype(bool)
        if self.control_kind == "target_mean":
            row = _normalizer_row(self.normalizer, self.target_id)
            replacement = np.zeros_like(values)
            for component in row["components"]:
                index = int(component["component_index"])
                replacement[index][masks[index]] = float(
                    component["unconditional_mean"]
                )
            values = replacement
        normalized = normalize_channel_first_target(
            values,
            masks,
            target_id=self.target_id,
            normalizer=self.normalizer,
        )
        if normalized.ndim == 3:
            normalized = np.moveaxis(normalized, 0, -1)
            masks = np.moveaxis(masks, 0, -1)
        return normalized, masks


class HLTArrayDataset(torch.utils.data.Dataset if torch is not None else object):
    """Validated HLT cache arrays after an identity-only role subset join."""

    def __init__(
        self,
        *,
        replica_arrays: Mapping[int, Mapping[str, np.ndarray]],
        labels: np.ndarray,
        identities: Sequence[str],
        logical_role: str,
        realization_policy: str,
        source_indices_by_replica: Mapping[int, Sequence[int]] | None = None,
        region_trees_by_replica: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.identities = tuple(str(value) for value in identities)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.logical_role = str(logical_role)
        self.realization_policy = str(realization_policy)
        self.replicas = {
            int(replica): {
                # Preserve np.memmap objects; np.asarray would erase the
                # storage type and makes accidental whole-array copies easier.
                name: value for name, value in arrays.items()
            }
            for replica, arrays in replica_arrays.items()
        }
        self.source_indices_by_replica = {
            replica: (
                np.arange(len(self.identities), dtype=np.int64)
                if source_indices_by_replica is None
                else np.asarray(source_indices_by_replica[replica], dtype=np.int64)
            )
            for replica in self.replicas
        }
        expected_replicas = (
            {0}
            if self.realization_policy == "R_FIXED"
            else {0, 1, 2, 3}
        )
        if (
            not self.identities
            or len(set(self.identities)) != len(self.identities)
            or self.labels.shape != (len(self.identities),)
            or set(self.replicas) != expected_replicas
            or set(self.source_indices_by_replica) != set(self.replicas)
            or any(
                indices.shape != (len(self.identities),)
                or bool((indices < 0).any())
                or (
                    len(indices) > 0
                    and int(indices.max()) >= len(self.replicas[replica]["tokens"])
                )
                for replica, indices in self.source_indices_by_replica.items()
            )
        ):
            raise ValueError("HLT role-subset dataset population differs")
        self.region_trees_by_replica = (
            None
            if region_trees_by_replica is None
            else {
                int(replica): rows
                for replica, rows in region_trees_by_replica.items()
            }
        )
        if self.region_trees_by_replica is not None and (
            set(self.region_trees_by_replica) != set(self.replicas)
            or any(
                len(rows) != len(self.identities)
                for rows in self.region_trees_by_replica.values()
            )
        ):
            raise ValueError("HLT role-subset tree coverage differs")
        self.zero_based_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) <= 0:
            raise ValueError("HLT role-subset epoch is one-based")
        self.zero_based_epoch = int(epoch) - 1

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        identity = self.identities[index]
        replica = replica_for(
            policy=self.realization_policy,
            logical_role=self.logical_role,
            epoch=self.zero_based_epoch,
            canonical_identity=identity,
        )
        arrays = self.replicas[int(replica)]
        source_index = int(self.source_indices_by_replica[int(replica)][index])
        return {
            "tokens": arrays["tokens"][source_index],
            "mask": arrays["mask"][source_index],
            "measurement_states": arrays["measurement_states"][source_index],
            "label": self.labels[index],
            "identity": identity,
            "replica_id": int(replica),
            "offline_target_tokens": None,
            "offline_target_logits": None,
            "region_tree": (
                None
                if self.region_trees_by_replica is None
                else self.region_trees_by_replica[int(replica)][index]
            ),
        }


class AuxiliaryTargetDataset(torch.utils.data.Dataset if torch is not None else object):
    """Attach one exact target coordinate to a native-HLT dataset."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        target_id: str,
        target_values: np.ndarray | Mapping[int, np.ndarray] | None = None,
        target_masks: np.ndarray | Mapping[int, np.ndarray] | None = None,
        streamed_target: StreamedHLTTarget | None = None,
        control_kind: str | None = None,
        target_parent_hashes: Mapping[str, str],
        donor_indices: Sequence[int] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Stage-D datasets")
        self.base_dataset = base_dataset
        self.target_id = str(target_id)
        self.control_kind = control_kind
        self.target_parent_hashes = dict(target_parent_hashes)
        self.donor_indices = (
            None
            if donor_indices is None
            else np.asarray(donor_indices, dtype=np.int64)
        )
        if self.donor_indices is not None and (
            self.donor_indices.shape != (len(base_dataset),)
            or sorted(self.donor_indices.tolist()) != list(range(len(base_dataset)))
        ):
            raise ValueError("Stage-D donor mapping is not a complete permutation")
        if not self.target_parent_hashes:
            raise ValueError("Stage-D target join requires parent hashes")
        static = target_values is not None or target_masks is not None
        if static == (streamed_target is not None):
            raise ValueError("provide exactly one static or streamed target source")
        self.streamed_target = streamed_target
        self.values = target_values
        self.masks = target_masks
        if static:
            if target_values is None or target_masks is None:
                raise ValueError("static target values and masks must be paired")
            for value in (target_values, target_masks):
                rows = value.values() if isinstance(value, Mapping) else (value,)
                if any(
                    not hasattr(row, "shape")
                    or int(row.shape[0]) != len(base_dataset)
                    for row in rows
                ):
                    raise ValueError("static target population differs")
            for value in (target_values, target_masks):
                if isinstance(value, Mapping) and set(value) != set(
                    base_dataset.replicas
                ):
                    raise ValueError(
                        "replica-specific target coverage differs from HLT views"
                    )
        self.identities = base_dataset.identities

    def set_epoch(self, epoch: int) -> None:
        self.base_dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _select(value: Any, replica: int, index: int) -> np.ndarray:
        array = value[int(replica)] if isinstance(value, Mapping) else value
        return np.asarray(array[index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        if self.streamed_target is not None:
            target_sample = (
                sample
                if self.donor_indices is None
                else self.base_dataset[int(self.donor_indices[index])]
            )
            target, mask = self.streamed_target(target_sample)
        else:
            target = self._select(self.values, sample["replica_id"], index)
            mask = self._select(self.masks, sample["replica_id"], index)
        if target.shape != mask.shape or not np.isfinite(target).all():
            raise ValueError("Stage-D joined target arrays differ or are nonfinite")
        sample["target"] = np.asarray(target, dtype=np.float32)
        sample["target_mask"] = np.asarray(mask, dtype=bool)
        return sample


def collate_auxiliary_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = collate_native_hlt_expert_batch(samples)
    output["target"] = torch.from_numpy(
        np.stack([sample["target"] for sample in samples])
    ).float()
    output["target_mask"] = torch.from_numpy(
        np.stack([sample["target_mask"] for sample in samples])
    ).bool()
    output["identities"] = list(output["event_identities"])
    return output


def make_auxiliary_loader(
    dataset: AuxiliaryTargetDataset,
    *,
    seed: int,
    training: bool,
    batch_size: int,
) -> Any:
    sampler = (
        DeterministicExpertSampler(dataset, seed=int(seed))
        if training
        else torch.utils.data.SequentialSampler(dataset)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_auxiliary_batch,
    )


__all__ = [
    "AuxiliaryTargetDataset",
    "HLTArrayDataset",
    "StreamedHLTTarget",
    "collate_auxiliary_batch",
    "make_auxiliary_loader",
    "normalize_channel_first_target",
]
