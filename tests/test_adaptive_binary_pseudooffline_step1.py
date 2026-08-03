from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    HLT_PROFILE_V1,
    fixed_hlt_params_from_profile,
    generate_and_cache_hlt_view,
    jet_identity_hash,
)
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    FileRecord,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_EXPECTED_VARIANT_NAMES,
    ABPH_HIGHDATA_SPLIT_SIZES,
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_PILOT_SPLIT_SIZES,
    ABPH_VARIANT_REGISTRY,
    AdaptiveBinaryHLTOnlyDataset,
    AdaptiveBinaryInputContractConfig,
    GROUP_TARGET_SCHEMA,
    PARTICLE_TARGET_SCHEMA,
    ROOT_LEDGER_SCHEMA,
    SchemaField,
    VersionedTensorSchema,
    abph_hlt_params_dict,
    load_hlt_only_dataset,
    normalize_variant_name,
    registry_manifest,
    resolve_variant_config,
    schema_from_dict,
    schema_manifest,
    validate_hlt_view_contract,
    validate_manifest_contract,
    validate_offline_view_contract,
)


def _toy_manifest(n_jets: int = 10) -> SplitManifest:
    prefixes = list(FILE_PREFIX_TO_LABEL)
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        splits[split] = [
            JetIdentity(
                file=f"{prefixes[label]}_{split_index:03d}.root",
                entry=split_index * 1000 + label,
                label=label,
            )
            for label in range(n_jets)
        ]
    records = [
        FileRecord(path=identity.file, label=identity.label, num_entries=10_000)
        for rows in splits.values()
        for identity in rows
    ]
    return SplitManifest(
        data_dir="toy",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: n_jets for split in SPLIT_ORDER},
        split_seeds={split: index + 100 for index, split in enumerate(SPLIT_ORDER)},
        file_records=records,
        splits=splits,
        metadata={"test": True},
    )


def _toy_view(manifest: SplitManifest, split: str, *, hlt: bool) -> JetView:
    identities = tuple(manifest.splits[split])
    n_jets = len(identities)
    tokens = np.zeros((n_jets, 128, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 128), dtype=bool)
    labels = np.asarray([row.label for row in identities], dtype=np.int64)
    for jet_index in range(n_jets):
        valid = 3 + jet_index % 3
        mask[jet_index, :valid] = True
        for particle_index in range(valid):
            pt = 10.0 + jet_index + particle_index
            eta = -0.2 + 0.05 * particle_index
            phi = -0.3 + 0.07 * particle_index
            tokens[jet_index, particle_index, :5] = (
                pt,
                eta,
                phi,
                pt * np.cosh(eta) + 0.2,
                1.0 if particle_index % 2 == 0 else 0.0,
            )
            tokens[jet_index, particle_index, 5 + particle_index % 5] = 1.0
    identity_sha = jet_identity_hash(identities)
    common = {
        "source_manifest_hash": manifest_hash(manifest),
        "jet_identity_hash": identity_sha,
    }
    if hlt:
        metadata = {
            **common,
            "view": "fixed_hlt",
            "hlt_profile": ABPH_HLT_PROFILE,
            "hlt_profile_version": "v1",
            "hlt_degradation_strength": ABPH_HLT_DEGRADATION_STRENGTH,
            "hlt_params": abph_hlt_params_dict(),
            "hlt_content_hash": "toy-hlt-content",
        }
    else:
        metadata = {
            **common,
            "view": "offline",
            "offline_content_hash": "toy-offline-content",
        }
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=list(identities),
        split=split,
        metadata=metadata,
    )


def _smoke_sizes(n_jets: int = 10) -> dict[str, int]:
    return {split: n_jets for split in SPLIT_ORDER}


def test_schemas_round_trip_and_hash_every_semantic_change():
    for schema in (ROOT_LEDGER_SCHEMA, GROUP_TARGET_SCHEMA, PARTICLE_TARGET_SCHEMA):
        restored = schema_from_dict(schema.to_dict())
        assert restored == schema
        assert restored.feature_order_hash == schema.feature_order_hash

    changed_version = replace(ROOT_LEDGER_SCHEMA, version="v2")
    changed_field = replace(
        ROOT_LEDGER_SCHEMA,
        fields=(replace(ROOT_LEDGER_SCHEMA.fields[0], unit="MeV"), *ROOT_LEDGER_SCHEMA.fields[1:]),
    )
    reordered = replace(
        ROOT_LEDGER_SCHEMA,
        fields=(ROOT_LEDGER_SCHEMA.fields[1], ROOT_LEDGER_SCHEMA.fields[0], *ROOT_LEDGER_SCHEMA.fields[2:]),
    )
    assert changed_version.feature_order_hash != ROOT_LEDGER_SCHEMA.feature_order_hash
    assert changed_field.feature_order_hash != ROOT_LEDGER_SCHEMA.feature_order_hash
    assert reordered.feature_order_hash != ROOT_LEDGER_SCHEMA.feature_order_hash
    assert schema_manifest()["manifest_hash"]


def test_schema_deserialization_rejects_stale_hash():
    payload = ROOT_LEDGER_SCHEMA.to_dict()
    payload["feature_order_hash"] = "stale"
    with pytest.raises(ValueError, match="hash mismatch"):
        VersionedTensorSchema.from_dict(payload)


def test_input_config_locks_pilot_and_highdata_contracts():
    pilot = AdaptiveBinaryInputContractConfig(campaign_mode="pilot")
    highdata = AdaptiveBinaryInputContractConfig(campaign_mode="highdata")
    assert pilot.split_sizes == ABPH_PILOT_SPLIT_SIZES
    assert highdata.split_sizes == ABPH_HIGHDATA_SPLIT_SIZES
    assert highdata.hlt_degradation_strength == 2.5
    assert highdata.max_particles == 128
    with pytest.raises(ValueError, match="locked to HLT profile"):
        AdaptiveBinaryInputContractConfig(hlt_profile=HLT_PROFILE_V1)
    with pytest.raises(ValueError, match="locked to HLT strength"):
        AdaptiveBinaryInputContractConfig(hlt_degradation_strength=1.0)


def test_complete_registry_resolves_every_short_and_full_name():
    assert len(ABPH_EXPECTED_VARIANT_NAMES) == 59
    assert set(ABPH_EXPECTED_VARIANT_NAMES) == set(ABPH_VARIANT_REGISTRY)
    hashes = set()
    for name, spec in ABPH_VARIANT_REGISTRY.items():
        assert normalize_variant_name(spec.run_id) == name
        assert normalize_variant_name(name) == name
        resolved = resolve_variant_config(name)
        assert resolved["variant"]["name"] == name
        assert resolved["variant"]["run_id"] == spec.run_id
        assert resolved["data"]["final_test_teacher_free"] is True
        assert set(resolved["model"]) >= {
            "hlt_part", "root_predictor", "hierarchy", "distribution",
            "renderer", "pseudo_part", "fusion", "hierarchy_modules",
        }
        hashes.add(resolved["resolved_config_hash"])
    assert len(hashes) == len(ABPH_EXPECTED_VARIANT_NAMES)
    assert registry_manifest()["registry_hash"]


def test_registry_encodes_shared_root_oracle_and_kd_safety():
    b2 = resolve_variant_config("B2")
    c5 = resolve_variant_config("C5")
    d1 = resolve_variant_config("D1")
    e7 = resolve_variant_config("E7")
    e11 = resolve_variant_config("E11")
    b4 = resolve_variant_config("B4")
    f4 = resolve_variant_config("F4")
    assert b2["model"]["hierarchy"]["enabled"] is False
    assert b2["model"]["renderer"]["enabled"] is False
    assert c5["model"]["hierarchy"]["enabled"] is True
    assert c5["model"]["renderer"]["enabled"] is False
    assert d1["model"]["renderer"]["enabled"] is True
    assert d1["model"]["fusion"]["enabled"] is False
    assert e7["model"]["root_predictor"]["shared_across_hierarchies"] is True
    assert e7["model"]["distribution"]["shared_compiled_root"] is True
    assert e11["model"]["root_predictor"]["shared_across_hierarchies"] is False
    assert b4["evaluation"]["oracle"] is True
    assert b4["evaluation"]["final_test_eligible"] is False
    assert f4["data"]["requires_teacher_logits"] is True
    assert f4["data"]["final_test_teacher_free"] is True


def test_supplemental_kt8_screen_is_explicit_and_non_gating():
    d7 = resolve_variant_config("D7_kt8_mh4_particles_screen")
    e12 = resolve_variant_config("E12_kt8_mh4_dualcross_screen")

    assert d7["variant"]["dependencies"] == ["C3_kt_8"]
    assert d7["model"]["hierarchy"]["capacities"] == [2, 4, 8]
    assert d7["model"]["renderer"]["enabled"] is True
    assert d7["model"]["distribution"]["stochastic_views"] == 4
    assert d7["evaluation"]["supplemental_screen"] is True

    assert e12["variant"]["dependencies"] == [
        "A0_hlt_part",
        "D7_kt8_mh4_particles_screen",
    ]
    assert e12["data"]["pseudo_sources"] == ["D7_kt8_mh4_particles_screen"]
    assert e12["model"]["hierarchy"]["capacities"] == [2, 4, 8]
    assert e12["evaluation"]["supplemental_screen"] is True


def test_manifest_and_views_fail_closed_on_contract_drift():
    manifest = _toy_manifest()
    validate_manifest_contract(manifest, expected_split_sizes=_smoke_sizes())
    hlt = _toy_view(manifest, "model_train", hlt=True)
    offline = _toy_view(manifest, "model_train", hlt=False)
    validate_hlt_view_contract(hlt, manifest, "model_train", expected_n_jets=10)
    validate_offline_view_contract(
        offline,
        manifest,
        "model_train",
        expected_n_jets=10,
        hlt_view=hlt,
    )

    bad_profile = JetView(
        tokens=hlt.tokens,
        mask=hlt.mask,
        labels=hlt.labels,
        jet_ids=hlt.jet_ids,
        split=hlt.split,
        metadata={**hlt.metadata, "hlt_profile": HLT_PROFILE_V1},
    )
    with pytest.raises(ValueError, match="profile mismatch"):
        validate_hlt_view_contract(bad_profile, manifest, "model_train", expected_n_jets=10)

    bad_manifest = replace(manifest, max_constits=64)
    with pytest.raises(ValueError, match="max_constits"):
        validate_manifest_contract(bad_manifest, expected_split_sizes=_smoke_sizes())

    wrong_split = JetView(
        tokens=hlt.tokens,
        mask=hlt.mask,
        labels=hlt.labels,
        jet_ids=hlt.jet_ids,
        split="model_val",
        metadata=hlt.metadata,
    )
    with pytest.raises(ValueError, match="split mismatch"):
        validate_hlt_view_contract(wrong_split, manifest, "model_train", expected_n_jets=10)


def test_clean_hlt_dataset_has_no_privileged_fields_or_paths():
    manifest = _toy_manifest()
    hlt = _toy_view(manifest, "model_train", hlt=True)
    dataset = AdaptiveBinaryHLTOnlyDataset(hlt, manifest, expected_n_jets=10)
    assert len(dataset) == 10
    assert set(dataset[0]) == {"tokens", "mask", "labels", "indices"}
    assert not hasattr(dataset, "target_fields")
    assert not hasattr(dataset, "offline_tokens")
    assert dataset.metadata["target_cache_loaded"] is False
    assert dataset.metadata["offline_inputs_loaded"] is False
    assert dataset.metadata["teacher_logits_loaded"] is False


def test_on_disk_hlt_only_loader_requires_no_target_cache():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _toy_manifest()
        manifest_path = root / "split_manifest.json.gz"
        cache_dir = root / "hlt_cache"
        save_split_manifest(manifest, manifest_path)
        offline_source = _toy_view(manifest, "model_train", hlt=False)
        generate_and_cache_hlt_view(
            offline_source,
            cache_dir,
            seed=DEFAULT_HLT_SEEDS["model_train"],
            params=fixed_hlt_params_from_profile(ABPH_HLT_PROFILE, 2.5),
            hlt_degradation_strength=2.5,
        )
        dataset = load_hlt_only_dataset(
            manifest_path,
            cache_dir,
            "model_train",
            expected_split_sizes=_smoke_sizes(),
        )
        assert len(dataset) == 10
        assert not (root / "target_cache").exists()
        assert dataset.metadata["target_cache_loaded"] is False
