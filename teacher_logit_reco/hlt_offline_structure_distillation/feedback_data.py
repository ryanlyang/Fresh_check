"""Identity-bound Stage-E intervention sources and collation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .auxiliary_data import collate_auxiliary_batch
from .contracts import load_hashed_json, require_sha256, with_content_hash
from .stage_d_data_factory import load_stage_d_loaders_from_manifest
from .target_cache import identity_order_sha256

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


STAGE_E_LOADER_MANIFEST_CONTRACT = "hosd_stage_e_loader_manifest_v2"
FEEDBACK_SHUFFLE_SPLIT_BY_ROLE = {
    "model_train": "model_train",
    "val_stop": "val_stop",
    "design_select": "design_select",
    "design_confirm": "design_confirm",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_stage_e_loader_manifest(
    *,
    row: Mapping[str, Any],
    base_loader_manifest: str | Path,
    intervention_sources: Mapping[str, Mapping[str, Any]] | None,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        row.get("control")
        in {
            "SHUFFLED_PREDICTION",
            "SHUFFLED",
            "ORACLE_SUB",
            "ORACLE_TRAINED",
        }
    )
    sources = {
        str(role): {
            **dict(definition),
            "npz_sha256": require_sha256(
                definition.get(
                    "npz_sha256", _sha256_file(definition["npz_path"])
                ),
                name=f"{role}.npz_sha256",
            ),
        }
        for role, definition in (intervention_sources or {}).items()
    }
    expected_roles = (
        {"design_select"}
        if row.get("control") == "ORACLE_SUB"
        else {"model_train", "val_stop", "design_select"}
    )
    if required and set(sources) != expected_roles:
        raise ValueError("Stage-E intervention role coverage differs")
    return with_content_hash(
        {
            "contract": STAGE_E_LOADER_MANIFEST_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "row_id": row["row_id"],
            "base_loader_manifest": str(Path(base_loader_manifest).resolve()),
            "base_loader_manifest_sha256": _sha256_file(base_loader_manifest),
            "intervention_sources": sources,
            "identity_join_before_batching": True,
            "batch_layout_independent": True,
        }
    )


def _load_intervention_npz(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str] | None]:
    with np.load(path, allow_pickle=False) as payload:
        if "identities" not in payload.files:
            raise ValueError("Stage-E intervention lacks identities")
        identities = tuple(str(value) for value in payload["identities"].tolist())
        if len(identities) != len(set(identities)):
            raise ValueError("Stage-E intervention identities are duplicated")
        excluded = {"identities", "donor_identities"}
        fields = sorted(set(payload.files) - excluded)
        if not fields:
            raise ValueError("Stage-E intervention has no value fields")
        values = {
            identity: {
                field: np.asarray(payload[field][index])
                for field in fields
            }
            for index, identity in enumerate(identities)
        }
        donors = (
            None
            if "donor_identities" not in payload.files
            else {
                identity: str(donor)
                for identity, donor in zip(
                    identities, payload["donor_identities"].tolist()
                )
            }
        )
    return values, donors


def load_stage_e_loaders_from_manifest(
    *,
    manifest_path: str | Path,
    campaign_root: Path,
    row: Mapping[str, Any],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_hashed_json(
        manifest_path, expected_contract=STAGE_E_LOADER_MANIFEST_CONTRACT
    )
    if (
        manifest.get("source") != campaign["source"]
        or manifest.get("campaign_spec_sha256") != campaign["content_hash"]
        or manifest.get("row_id") != row["row_id"]
    ):
        raise ValueError("Stage-E loader manifest lineage differs")
    base_path = Path(manifest["base_loader_manifest"])
    if _sha256_file(base_path) != manifest["base_loader_manifest_sha256"]:
        raise ValueError("Stage-E base loader manifest bytes differ")
    loaded = load_stage_d_loaders_from_manifest(
        manifest_path=base_path,
        campaign_root=campaign_root,
        row=row,
        campaign=campaign,
        target_registry=target_registry,
    )
    control = row.get("control")
    kind = (
        "predicted_feedback_override"
        if control in {"SHUFFLED_PREDICTION", "SHUFFLED"}
        else "oracle_feedback"
        if control in {"ORACLE_SUB", "ORACLE_TRAINED"}
        else None
    )
    if kind is None:
        return loaded
    wrapped = {}
    for role, loader_key in (
        ("model_train", "train_loader"),
        ("val_stop", "val_stop_loader"),
        ("design_select", "design_select_loader"),
    ):
        if role not in manifest["intervention_sources"]:
            if control != "ORACLE_SUB":
                raise ValueError("Stage-E intervention role is absent")
            continue
        definition = manifest["intervention_sources"][role]
        path = Path(definition["npz_path"])
        if _sha256_file(path) != definition["npz_sha256"]:
            raise ValueError("Stage-E intervention bytes differ")
        values, donors = _load_intervention_npz(path)
        dataset = FeedbackInterventionDataset(
            loaded[loader_key].dataset,
            intervention=kind,
            values_by_identity=values,
            donor_identity_by_identity=donors,
            parent_hashes={"intervention_npz": definition["npz_sha256"]},
        )
        wrapped[loader_key] = make_feedback_loader(
            dataset,
            seed=int(loaded["sampler_seed_by_role"][role]),
            training=role == "model_train",
            batch_size=int(loaded[loader_key].batch_size),
        )
        loaded["lineage_hashes"][
            f"{role}_feedback_intervention"
        ] = definition["npz_sha256"]
    loaded.update(wrapped)
    loaded["lineage_hashes"]["stage_e_loader_manifest"] = manifest["content_hash"]
    return loaded


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(path)


def materialize_feedback_intervention(
    *,
    row: Mapping[str, Any],
    loader: Any,
    output_path: str | Path,
    role: str,
    prediction_model: Any | None = None,
    shuffle_plan: Mapping[str, Any] | None = None,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Create an identity-bound Stage-E oracle/direct/prediction intervention."""

    if torch is None:
        raise RuntimeError("PyTorch is required for feedback materialization")
    control = str(row.get("control"))
    predicted = control in {"SHUFFLED_PREDICTION", "SHUFFLED"}
    oracle = control in {"ORACLE_SUB", "ORACLE_TRAINED"}
    direct = control == "EXACT_HLT"
    if sum((predicted, oracle, direct)) != 1:
        raise ValueError("row does not require a materialized intervention")
    if predicted != (prediction_model is not None):
        raise ValueError("prediction intervention model coverage differs")
    if predicted != (shuffle_plan is not None):
        raise ValueError("prediction intervention shuffle-plan coverage differs")
    resolved = torch.device(device)
    if prediction_model is not None:
        prediction_model.to(resolved)
        prediction_model.eval()
    identities: list[str] = []
    fields: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            batch_ids = [str(value) for value in batch["identities"]]
            identities.extend(batch_ids)
            if predicted:
                features = batch["features"].to(resolved)
                mask = batch["mask"].to(resolved)
                vectors = batch.get("lorentz_vectors", batch.get("vectors"))
                if vectors is None:
                    raise ValueError(
                        "feedback prediction batch lacks Lorentz vectors"
                    )
                vectors = vectors.to(resolved)
                points = batch.get("points")
                points = (
                    features[:, 15:17]
                    if points is None
                    else points.to(resolved)
                )
                _, values = prediction_model.forward_with_feedback(
                    points, features, vectors, mask
                )
                batch_fields = {
                    key: value.detach().float().cpu().numpy()
                    for key, value in values.items()
                    if hasattr(value, "shape")
                    and int(value.shape[0]) == len(batch_ids)
                    and key
                    in {
                        "value",
                        "mean",
                        "log_variance",
                        "availability_logits",
                    }
                }
            else:
                target = batch["target"].detach().float().cpu().numpy()
                batch_fields = {"value": target}
                if oracle:
                    from .target_schemas import (
                        target_component_availability_groups,
                        target_declarations,
                    )

                    declaration = next(
                        item
                        for item in target_declarations()
                        if item.target_id == row["target_id"]
                    )
                    groups = target_component_availability_groups(
                        row["target_id"], declaration.components
                    )
                    order = tuple(dict.fromkeys(groups))
                    target_mask = batch["target_mask"].detach().bool().cpu().numpy()
                    availability = np.stack(
                        [
                            target_mask[
                                :, [index for index, value in enumerate(groups) if value == group]
                            ].any(axis=1)
                            for group in order
                        ],
                        axis=1,
                    )
                    batch_fields["availability_logits"] = np.where(
                        availability, 20.0, -20.0
                    ).astype(np.float32)
                if row["parameterization"] == "HET" and not direct:
                    batch_fields.update(
                        {
                            "mean": target,
                            "log_variance": np.zeros_like(
                                target, dtype=np.float32
                            ),
                        }
                    )
            for key, value in batch_fields.items():
                fields.setdefault(key, []).append(np.asarray(value))
    if (
        not identities
        or len(identities) != len(set(identities))
        or set(identities) != set(str(value) for value in loader.dataset.identities)
    ):
        raise ValueError("feedback intervention identity coverage differs")
    canonical_identities = tuple(
        str(value) for value in loader.dataset.identities
    )
    observed_index = {
        identity: index for index, identity in enumerate(identities)
    }
    order = np.asarray(
        [observed_index[identity] for identity in canonical_identities],
        dtype=np.int64,
    )
    identities = list(canonical_identities)
    arrays = {
        "identities": np.asarray(identities),
        **{
            key: np.concatenate(chunks, axis=0)[order].astype(
                np.float32, copy=False
            )
            for key, chunks in sorted(fields.items())
        },
    }
    if predicted:
        from .contracts import TARGET_SHUFFLE_PLAN_CONTRACT, validate_content_hash

        validate_content_hash(
            shuffle_plan, expected_contract=TARGET_SHUFFLE_PLAN_CONTRACT
        )
        if (
            role not in FEEDBACK_SHUFFLE_SPLIT_BY_ROLE
            or shuffle_plan["target_id"] != row["target_id"]
            or shuffle_plan["split"]
            != FEEDBACK_SHUFFLE_SPLIT_BY_ROLE[role]
            or shuffle_plan["shuffle_kind"] != "global"
            or shuffle_plan["canonical_identity_order_sha256"]
            != identity_order_sha256(identities)
        ):
            raise ValueError("feedback prediction shuffle-plan lineage differs")
        mapping = np.asarray(
            shuffle_plan["mapping_recipient_to_donor"], dtype=np.int64
        )
        arrays["donor_identities"] = np.asarray(identities)[mapping]
    output = Path(output_path)
    digest = _atomic_save_npz(output, arrays)
    return {
        "path": str(output.resolve()),
        "sha256": digest,
        "event_count": len(identities),
        "fields": sorted(set(arrays) - {"identities", "donor_identities"}),
        "wrong_event_mapping": predicted,
        "oracle_target": oracle,
        "direct_hlt_relation": direct,
    }


class FeedbackInterventionDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    """Attach precomputed feedback by canonical identity before batching."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        intervention: str,
        values_by_identity: Mapping[str, Mapping[str, np.ndarray]],
        donor_identity_by_identity: Mapping[str, str] | None = None,
        parent_hashes: Mapping[str, str],
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Stage-E datasets")
        if intervention not in {
            "predicted_feedback_override",
            "oracle_feedback",
            "direct_pair_features",
        }:
            raise ValueError("unknown Stage-E intervention")
        identities = tuple(str(value) for value in base_dataset.identities)
        if len(identities) != len(set(identities)):
            raise ValueError("Stage-E base identities are not unique")
        if set(values_by_identity) != set(identities):
            raise ValueError("Stage-E intervention identity coverage differs")
        donors = (
            {identity: identity for identity in identities}
            if donor_identity_by_identity is None
            else {
                str(key): str(value)
                for key, value in donor_identity_by_identity.items()
            }
        )
        if set(donors) != set(identities) or set(donors.values()) != set(identities):
            raise ValueError("Stage-E donor mapping is not a complete permutation")
        if donor_identity_by_identity is not None and any(
            key == value for key, value in donors.items()
        ):
            raise ValueError("shuffled feedback donor mapping contains a fixed point")
        if not parent_hashes:
            raise ValueError("Stage-E intervention source lacks authenticated lineage")
        self.base_dataset = base_dataset
        self.intervention = intervention
        self.values = {
            str(identity): {
                str(key): np.asarray(value)
                for key, value in fields.items()
            }
            for identity, fields in values_by_identity.items()
        }
        self.donors = donors
        self.parent_hashes = dict(parent_hashes)
        self.identities = identities
        self.feedback_intervention_ready = True
        self.control_kind = getattr(base_dataset, "control_kind", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        identity = str(sample.get("identities", sample.get("event_identity")))
        donor = self.donors[identity]
        fields = self.values[donor]
        if self.intervention == "direct_pair_features":
            if set(fields) != {"value"}:
                raise ValueError("direct pair intervention requires one value tensor")
            sample[self.intervention] = np.asarray(fields["value"], dtype=np.float32)
        else:
            sample[self.intervention] = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in fields.items()
            }
        return sample


def collate_feedback_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = collate_auxiliary_batch(samples)
    keys = {
        key
        for sample in samples
        for key in (
            "predicted_feedback_override",
            "oracle_feedback",
            "direct_pair_features",
        )
        if key in sample
    }
    if len(keys) > 1:
        raise ValueError("Stage-E batch mixes intervention kinds")
    if not keys:
        return output
    key = next(iter(keys))
    if not all(key in sample for sample in samples):
        raise ValueError("Stage-E batch has partial intervention coverage")
    if key == "direct_pair_features":
        output[key] = torch.from_numpy(
            np.stack([sample[key] for sample in samples])
        ).float()
        return output
    field_names = set(samples[0][key])
    if any(set(sample[key]) != field_names for sample in samples):
        raise ValueError("Stage-E feedback field coverage differs")
    output[key] = {
        field: torch.from_numpy(
            np.stack([sample[key][field] for sample in samples])
        ).float()
        for field in sorted(field_names)
    }
    return output


def make_feedback_loader(
    dataset: Any, *, seed: int, training: bool, batch_size: int
) -> Any:
    from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (
        DeterministicExpertSampler,
    )

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
        collate_fn=collate_feedback_batch,
    )


__all__ = [
    "FeedbackInterventionDataset",
    "STAGE_E_LOADER_MANIFEST_CONTRACT",
    "build_stage_e_loader_manifest",
    "collate_feedback_batch",
    "load_stage_e_loaders_from_manifest",
    "make_feedback_loader",
    "materialize_feedback_intervention",
]
