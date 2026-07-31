from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.bridge_certification import (
    build_bridge_candidate_eligibility,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_selection import (
    BRIDGE_COORDINATE_SELECTION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.step8 import (
    build_locked_target_cache_specification,
    build_postlock_target_policy,
    build_selected_target_lineage,
    build_step8_bundle,
    validate_step8_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (
    audit_target_storage,
    build_sealed_input_preparation,
    build_target_cache_specification,
    fit_target_normalizers,
    identity_order_sha256,
    load_offline_target_cache,
    load_frozen_token_head_reproducer,
    publish_offline_target_cache,
    validate_offline_target_cache,
    validate_sealed_input_preparation,
    validate_target_cache_specification,
    validate_target_normalizer_set,
    validate_target_storage_audit,
    verify_target_batch_logits,
)
from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (
    TokenOnlyExpertHead,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _descriptors():
    return {
        expert: {
            "checkpoint_sha256": hashlib.sha256(
                f"checkpoint:{expert}".encode()
            ).hexdigest(),
            "registration_sha256": hashlib.sha256(
                f"registration:{expert}".encode()
            ).hexdigest(),
            "slot_query_sha256": hashlib.sha256(
                f"queries:{expert}".encode()
            ).hexdigest(),
            "eligibility_sha256": hashlib.sha256(
                f"eligibility:{expert}".encode()
            ).hexdigest(),
        }
        for expert in EXPERT_ORDER
    }


def _spec(identities, labels, *, split="model_train", seed=101):
    return build_target_cache_specification(
        split=split,
        pipeline_seed=seed,
        shape_id="SHAPE_COMPACT",
        allocation={expert: [1, 64] for expert in EXPERT_ORDER},
        target_tuple=["T0_PURE"] * len(EXPERT_ORDER),
        target_descriptors=_descriptors(),
        selected_target_lineage_sha256=SHA_A,
        target_cache_namespace="compact-seed-101",
        locked_coordinate_contract_sha256=SHA_B,
        locked_coordinate_selection_sha256=SHA_C,
        offline_fusion_checkpoint_sha256=SHA_A,
        offline_fusion_registration_sha256=SHA_B,
        normalizer_set_sha256=SHA_C,
        identity_manifest_sha256=SHA_A,
        identity_order_sha256=identity_order_sha256(identities, labels),
        event_count=len(identities),
    )


def _reproducer(scale: float):
    weights = np.arange(1, 11, dtype=np.float32) * np.float32(scale)

    def reproduce(tokens):
        pooled = np.asarray(tokens, dtype=np.float32).mean(
            axis=(1, 2), dtype=np.float32
        )
        return pooled[:, None] * weights[None, :]

    return reproduce


def _cache_fixture(tmp_path: Path):
    events = 13
    identities = [f"jet-{index:03d}" for index in range(events)]
    labels = np.arange(events, dtype=np.int64) % 10
    spec = bind_source(_spec(identities, labels), source_snapshot=SOURCE)
    values = {
        expert: np.full(
            (events, 1, 64),
            1.0 if expert != "TRACK" else 1.0003,
            dtype=np.float32,
        )
        for expert in EXPERT_ORDER
    }
    reproducers = {
        expert: _reproducer(1.0 if expert != "TRACK" else 100.0)
        for expert in EXPERT_ORDER
    }

    def generate(start, stop):
        return {
            "tokens": {
                expert: values[expert][start:stop] for expert in EXPERT_ORDER
            },
            "expert_logits": {
                expert: reproducers[expert](values[expert][start:stop])
                for expert in EXPERT_ORDER
            },
        }

    manifest = publish_offline_target_cache(
        output_dir=tmp_path / "cache",
        specification=spec,
        identities=identities,
        labels=labels,
        generator=generate,
        logit_reproducers=reproducers,
        source_snapshot=SOURCE,
        shard_size=5,
    )
    return spec, manifest, identities, labels, reproducers


def test_policy_and_specification_freeze_token_targets() -> None:
    policy = build_postlock_target_policy()
    assert not policy["prelock"]["final_test_model_outputs_permitted"]
    assert policy["post_finalist_lock"]["final_test_oracle_targets"][
        "requires_exact_stage_m_teacher"
    ]
    identities = ["a", "b"]
    labels = np.array([0, 1], dtype=np.int64)
    specification = _spec(identities, labels)
    assert validate_target_cache_specification(specification) == specification[
        "content_hash"
    ]
    invalid = copy.deepcopy(specification)
    invalid.pop("content_hash")
    invalid["target_tuple"][0] = "T3_LOGIT"
    invalid = with_content_hash(invalid)
    with pytest.raises(ValueError, match="non-token"):
        validate_target_cache_specification(invalid)
    with pytest.raises(ValueError, match="identity population"):
        identity_order_sha256(["same", "same"], labels)


def test_float16_audit_falls_back_without_terminating() -> None:
    tokens = np.full((4, 1, 64), 1.0003, dtype=np.float32)
    reproduce = _reproducer(100.0)
    logits = reproduce(tokens)
    audit = audit_target_storage(
        expert_id="TRACK",
        tokens=tokens,
        stored_expert_logits=logits,
        reproduce_logits=reproduce,
        checkpoint_sha256=SHA_A,
        slot_query_sha256=SHA_B,
        identity_order_sha256=SHA_C,
    )
    validate_target_storage_audit(audit)
    assert not audit["float16_audit_passed"]
    assert audit["selected_storage_dtype"] == "float32"
    assert not audit["float16_failure_stops_workflow"]


@pytest.mark.parametrize(
    ("mode", "payload_key", "prefix"),
    (
        ("T0_PURE", "model_state_dict", "head."),
        ("T2_PROJECT", "offline_target_state_dict", "projected_expert_head."),
    ),
)
def test_frozen_head_reproducer_is_checkpoint_hash_and_mode_bound(
    tmp_path: Path, mode: str, payload_key: str, prefix: str
) -> None:
    torch.manual_seed(8001)
    head = TokenOnlyExpertHead(token_dimension=64)
    head.eval()
    checkpoint = tmp_path / f"{mode}.pt"
    torch.save(
        {
            payload_key: {
                f"{prefix}{name}": value
                for name, value in head.state_dict().items()
            }
        },
        checkpoint,
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reproduce = load_frozen_token_head_reproducer(
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=checkpoint_sha,
        target_mode=mode,
        token_dimension=64,
    )
    tokens = np.random.default_rng(8).normal(
        size=(3, 2, 64)
    ).astype(np.float32)
    with torch.no_grad():
        expected = head(torch.from_numpy(tokens)).numpy()
    reproduced = reproduce(tokens)
    assert np.allclose(reproduced, expected, rtol=0, atol=2.0e-6)
    assert np.array_equal(reproduced.argmax(axis=1), expected.argmax(axis=1))
    with pytest.raises(ValueError, match="checkpoint identity"):
        load_frozen_token_head_reproducer(
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=SHA_A,
            target_mode=mode,
            token_dimension=64,
        )


def test_resumable_cache_roundtrip_normalizers_and_seed_rejection(
    tmp_path: Path,
) -> None:
    spec, manifest, _, _, reproducers = _cache_fixture(tmp_path)
    manifest_path = tmp_path / "cache" / "target_cache_manifest.json"
    assert manifest["shard_count"] == 3
    assert validate_offline_target_cache(
        manifest_path,
        expected_pipeline_seed=101,
        expected_specification_sha256=spec["content_hash"],
    ) == manifest["content_hash"]
    loaded_manifest, arrays = load_offline_target_cache(
        manifest_path,
        expected_pipeline_seed=101,
        expected_specification_sha256=spec["content_hash"],
    )
    assert all(
        arrays["tokens"][expert].dtype == np.float32
        for expert in EXPERT_ORDER
    )
    first_shard = np.load(
        tmp_path / "cache" / "shard_000000.npz", allow_pickle=False
    )
    storage_dtypes = {
        expert: str(first_shard[f"tokens_{expert}"].dtype)
        for expert in EXPERT_ORDER
    }
    verify_target_batch_logits(
        {
            **{
                f"tokens_{expert}": arrays["tokens"][expert]
                for expert in EXPERT_ORDER
            },
            **{
                f"logits_{expert}": arrays["expert_logits"][expert]
                for expert in EXPERT_ORDER
            },
        },
        logit_reproducers=reproducers,
        storage_dtype_by_expert=storage_dtypes,
    )
    assert storage_dtypes["TRACK"] == "float32"
    assert any(
        storage_dtypes[expert] == "float16"
        for expert in EXPERT_ORDER
        if expert != "TRACK"
    )
    normalizers = fit_target_normalizers(
        model_train_manifest_path=manifest_path,
        expected_pipeline_seed=101,
        expected_specification_sha256=spec["content_hash"],
        source_snapshot=SOURCE,
    )
    validate_target_normalizer_set(normalizers)
    assert np.asarray(normalizers["TRACK"]["mean"], dtype=np.float32)[0, 0] == (
        np.float32(1.0003)
    )
    assert not normalizers["normalizer_set"][
        "validation_or_test_statistics_consumed"
    ]
    val_only = copy.deepcopy(manifest)
    val_only.pop("content_hash")
    val_only["split"] = "val_stop"
    val_only = with_content_hash(val_only)
    val_path = tmp_path / "cache" / "val_only_manifest.json"
    import json

    val_path.write_text(json.dumps(val_only), encoding="utf-8")
    with pytest.raises(ValueError, match="training population"):
        fit_target_normalizers(
            model_train_manifest_path=val_path,
            expected_pipeline_seed=101,
            expected_specification_sha256=spec["content_hash"],
            source_snapshot=SOURCE,
        )
    with pytest.raises(ValueError, match="seed/specification"):
        validate_offline_target_cache(
            manifest_path,
            expected_pipeline_seed=202,
            expected_specification_sha256=spec["content_hash"],
        )
    assert loaded_manifest["identity_order_sha256"] == spec[
        "identity_order_sha256"
    ]


def test_cache_resume_and_tamper_fail_closed(tmp_path: Path) -> None:
    spec, _, identities, labels, reproducers = _cache_fixture(tmp_path)

    def forbidden(_start, _stop):
        raise AssertionError("complete cache must be reused")

    reused = publish_offline_target_cache(
        output_dir=tmp_path / "cache",
        specification=spec,
        identities=identities,
        labels=labels,
        generator=forbidden,
        logit_reproducers=reproducers,
        source_snapshot=SOURCE,
        shard_size=5,
    )
    assert reused["complete_coverage"]
    shard = tmp_path / "cache" / "shard_000001.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes differ"):
        validate_offline_target_cache(
            tmp_path / "cache" / "target_cache_manifest.json",
            expected_pipeline_seed=101,
            expected_specification_sha256=spec["content_hash"],
        )


def test_partial_shards_resume_without_regeneration(tmp_path: Path) -> None:
    events = 12
    identities = [f"resume-{index}" for index in range(events)]
    labels = np.arange(events, dtype=np.int64) % 10
    specification = bind_source(
        _spec(identities, labels), source_snapshot=SOURCE
    )
    tokens = {
        expert: np.ones((events, 1, 64), dtype=np.float32)
        for expert in EXPERT_ORDER
    }
    reproducers = {expert: _reproducer(1.0) for expert in EXPERT_ORDER}

    def result(start, stop):
        return {
            "tokens": {
                expert: tokens[expert][start:stop] for expert in EXPERT_ORDER
            },
            "expert_logits": {
                expert: reproducers[expert](tokens[expert][start:stop])
                for expert in EXPERT_ORDER
            },
        }

    def interrupted(start, stop):
        if (start, stop) == (5, 10):
            raise RuntimeError("simulated interruption")
        return result(start, stop)

    with pytest.raises(RuntimeError, match="interruption"):
        publish_offline_target_cache(
            output_dir=tmp_path / "partial",
            specification=specification,
            identities=identities,
            labels=labels,
            generator=interrupted,
            logit_reproducers=reproducers,
            source_snapshot=SOURCE,
            shard_size=5,
        )
    assert (tmp_path / "partial" / "shard_000000.json").is_file()
    calls = []

    def resumed(start, stop):
        calls.append((start, stop))
        return result(start, stop)

    manifest = publish_offline_target_cache(
        output_dir=tmp_path / "partial",
        specification=specification,
        identities=identities,
        labels=labels,
        generator=resumed,
        logit_reproducers=reproducers,
        source_snapshot=SOURCE,
        shard_size=5,
    )
    assert manifest["complete_coverage"]
    assert (0, 5) not in calls
    assert (5, 10) in calls and (10, 12) in calls


def test_target_lineage_is_seed_matched_and_complete() -> None:
    registrations = {}
    eligibility = {}
    slots = {}
    for expert in EXPERT_ORDER:
        checkpoints = {
            seed: hashlib.sha256(f"{expert}:{seed}".encode()).hexdigest()
            for seed in (101, 202, 303)
        }
        registrations[expert] = with_content_hash(
            {
                "contract": "test_registration_v1",
                "schema_version": 1,
                "expert_id": expert,
                "pipeline_seed": 101,
                "checkpoint_sha256": checkpoints[101],
            }
        )
        eligibility[expert] = build_bridge_candidate_eligibility(
            target_mode="T0_PURE",
            expert_id=expert,
            shape_id="SHAPE_COMPACT",
            checkpoint_hashes_by_seed=checkpoints,
        )
        slots[expert] = hashlib.sha256(f"slot:{expert}".encode()).hexdigest()
    lineage = build_selected_target_lineage(
        pipeline_seed=101,
        shape_id="SHAPE_COMPACT",
        target_tuple=["T0_PURE"] * len(EXPERT_ORDER),
        target_registrations=registrations,
        slot_query_hashes=slots,
        eligibility_artifacts=eligibility,
        content_certifications={expert: None for expert in EXPERT_ORDER},
        noninferiority_artifacts={expert: None for expert in EXPERT_ORDER},
    )
    assert lineage["cross_seed_substitution_permitted"] is False
    swapped = copy.deepcopy(registrations)
    swapped["TRACK"] = with_content_hash(
        {
            "contract": "test_registration_v1",
            "schema_version": 1,
            "expert_id": "TRACK",
            "pipeline_seed": 202,
            "checkpoint_sha256": eligibility["TRACK"][
                "checkpoint_hashes_by_seed"
            ]["202"],
        }
    )
    with pytest.raises(ValueError, match="seed/mode"):
        build_selected_target_lineage(
            pipeline_seed=101,
            shape_id="SHAPE_COMPACT",
            target_tuple=["T0_PURE"] * len(EXPERT_ORDER),
            target_registrations=swapped,
            slot_query_hashes=slots,
            eligibility_artifacts=eligibility,
            content_certifications={expert: None for expert in EXPERT_ORDER},
            noninferiority_artifacts={expert: None for expert in EXPERT_ORDER},
        )


def test_locked_coordinate_is_bound_into_canonical_cache_spec() -> None:
    target_lineage = with_content_hash(
        {
            "contract": "retb_selected_target_lineage_v2",
            "schema_version": 2,
            "pipeline_seed": 101,
            "shape_id": "SHAPE_COMPACT",
            "expert_order": list(EXPERT_ORDER),
            "target_tuple": ["T0_PURE"] * len(EXPERT_ORDER),
            "target_descriptors": _descriptors(),
            "cross_seed_substitution_permitted": False,
        }
    )
    normalizer = with_content_hash(
        {"contract": "test_normalizer_set_v1", "schema_version": 1}
    )
    fusion = with_content_hash(
        {
            "contract": "test_fusion_registration_v1",
            "schema_version": 1,
            "checkpoint_sha256": SHA_A,
            "selector_parent_fusion_sha256": SHA_A,
            "selector_parent_normalizer_set_sha256": normalizer[
                "content_hash"
            ],
            "shape_id": "SHAPE_COMPACT",
            "pipeline_seed": 101,
            "target_tuple": ["T0_PURE"] * len(EXPERT_ORDER),
        }
    )
    coordinate_sha = SHA_B
    selection = with_content_hash(
        {
            "contract": BRIDGE_COORDINATE_SELECTION_CONTRACT,
            "schema_version": 1,
            "locked_coordinate_systems": [
                {
                    "coordinate_contract_sha256": coordinate_sha,
                    "target_tuple": ["T0_PURE"] * len(EXPERT_ORDER),
                    "fusion_sha256": SHA_A,
                    "normalizer_set_sha256": normalizer["content_hash"],
                    "target_cache_namespace": "locked-compact-101",
                }
            ],
        }
    )
    identities = ["a", "b"]
    labels = np.array([0, 1], dtype=np.int64)
    specification = build_locked_target_cache_specification(
        split="model_train",
        pipeline_seed=101,
        shape_id="SHAPE_COMPACT",
        allocation={expert: [1, 64] for expert in EXPERT_ORDER},
        coordinate_selection=selection,
        coordinate_contract_sha256=coordinate_sha,
        target_lineage=target_lineage,
        fusion_registration=fusion,
        normalizer_set=normalizer,
        identity_manifest_sha256=SHA_C,
        identity_order_sha256=identity_order_sha256(identities, labels),
        event_count=2,
    )
    validate_target_cache_specification(specification)
    assert specification["selected_target_lineage_sha256"] == target_lineage[
        "content_hash"
    ]
    assert specification["target_cache_namespace"] == "locked-compact-101"


def test_sealed_preparation_and_step8_bundle_are_fail_closed() -> None:
    sealed = build_sealed_input_preparation(
        split="final_test",
        identity_manifest_sha256=SHA_A,
        raw_input_manifest_sha256=SHA_B,
        degraded_hlt_input_manifest_sha256=SHA_C,
        relation_sidecar_manifest_sha256=SHA_A,
        region_sidecar_manifest_sha256=SHA_B,
    )
    validate_sealed_input_preparation(sealed)
    assert not sealed["checkpoint_loading_permitted"]
    assert not sealed["model_outputs_present"]
    invalid = copy.deepcopy(sealed)
    invalid.pop("content_hash")
    invalid["oracle_logits_present"] = True
    with pytest.raises(ValueError, match="forbidden"):
        validate_sealed_input_preparation(with_content_hash(invalid))
    bundle = build_step8_bundle(
        campaign_spec_sha256=SHA_A,
        step7_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        source_snapshot=SOURCE,
    )
    assert validate_step8_bundle(bundle) == bundle["step8_bundle"][
        "content_hash"
    ]
    assert not bundle["step8_bundle"]["performance_based_termination"]


def test_step8_production_entrypoints_exist_and_sealed_cli_has_no_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (
        "scripts/build_retb_step8_contracts.py",
        "scripts/materialize_retb_target_cache_spec.py",
        "scripts/build_retb_target_cache.py",
        "scripts/fit_retb_target_normalizers.py",
        "scripts/prepare_retb_sealed_inputs.py",
        "sbatch/run_retb_build_step8_contracts.sh",
        "sbatch/run_retb_materialize_target_cache_spec.sh",
        "sbatch/run_retb_build_target_cache.sh",
        "sbatch/run_retb_fit_target_normalizers.sh",
        "sbatch/run_retb_prepare_sealed_inputs.sh",
    )
    for relative in expected:
        assert (root / relative).is_file()
    sealed_cli = (root / "scripts/prepare_retb_sealed_inputs.py").read_text()
    assert "--checkpoint" not in sealed_cli
    assert "load_and_validate_campaign_source" in sealed_cli
