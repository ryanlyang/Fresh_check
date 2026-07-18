from __future__ import annotations

import json

import numpy as np
import pytest

from jetclass_fresh.hlt_cache import hash_arrays
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_COMPACT_CODEC_MANIFEST_KEY,
    AdaptiveBinaryHierarchyLayout,
    build_adaptive_binary_targets,
    byte_shuffle,
    byte_unshuffle,
    decode_compact_target_arrays,
    encode_compact_target_arrays,
    narrow_signed_integer_array,
    reconstruct_target_identity_arrays,
)


def _target_arrays() -> tuple[dict[str, np.ndarray], tuple[JetIdentity, ...]]:
    hlt = np.zeros((2, 128, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((2, 128), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    for jet_index, count in enumerate((5, 9)):
        hlt_mask[jet_index, : count - 2] = True
        offline_mask[jet_index, :count] = True
        for particle_index in range(count):
            token = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
            token[:5] = (
                40.0 - particle_index,
                -0.3 + 0.07 * particle_index,
                -0.4 + 0.09 * particle_index,
                45.0 - particle_index,
                -1.0 if particle_index % 2 else 1.0,
            )
            token[5 + particle_index % 5] = 1.0
            token[10:] = (0.01, 0.002, -0.03, 0.004)
            offline[jet_index, particle_index] = token
            if particle_index < count - 2:
                hlt[jet_index, particle_index] = token
    jet_ids = (
        JetIdentity(file="HToBB_010.root", entry=11, label=0),
        JetIdentity(file="HToCC_010.root", entry=19, label=1),
    )
    targets = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=jet_ids,
    )
    arrays = targets.array_dict()
    arrays.update(
        {
            "labels": np.asarray([0, 1], dtype=np.int64),
            "jet_file_indices": np.asarray([0, 1], dtype=np.int32),
            "jet_entries": np.asarray([11, 19], dtype=np.int64),
        }
    )
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}, jet_ids


def _identity_keys() -> tuple[str, ...]:
    return (
        "root_identities",
        "level1_identities",
        "level2_identities",
        "level3_identities",
        "level4_identities",
        "level5_identities",
    )


def test_byte_shuffle_round_trips_float32_bitwise() -> None:
    source = np.asarray(
        [[0.0, -0.0, 1.25], [np.inf, -np.inf, np.nan]], dtype=np.float32
    )
    shuffled = byte_shuffle(source)
    restored = byte_unshuffle(shuffled, dtype=source.dtype.str, shape=source.shape)
    assert restored.dtype == np.float32
    assert restored.shape == source.shape
    assert restored.view(np.uint32).tobytes() == source.view(np.uint32).tobytes()


def test_integer_narrowing_uses_smallest_signed_dtype_and_rejects_overflow() -> None:
    assert narrow_signed_integer_array(np.asarray([-1, 31], dtype=np.int64)).dtype == np.int8
    assert narrow_signed_integer_array(np.asarray([-1, 128], dtype=np.int64)).dtype == np.int16
    with pytest.raises(OverflowError, match="does not fit"):
        narrow_signed_integer_array(
            np.asarray([-1, 128], dtype=np.int64), dtype=np.int8
        )


def test_compact_codec_reconstructs_complete_target_schema_bitwise() -> None:
    logical, jet_ids = _target_arrays()
    encoded, encode_manifest = encode_compact_target_arrays(
        logical,
        omitted_identity_keys=_identity_keys(),
    )
    decoded, decode_manifest = decode_compact_target_arrays(
        encoded,
        expected_encoded_content_hash=encode_manifest["encoded_content_hash"],
    )
    layout = AdaptiveBinaryHierarchyLayout()
    roots, levels = reconstruct_target_identity_arrays(
        jet_ids=jet_ids,
        layout=layout,
        particle_mask=decoded["particle_mask"],
        level_masks=tuple(decoded[f"level{depth + 1}_mask"] for depth in range(5)),
        level_membership=tuple(
            decoded[f"level{depth + 1}_membership"] for depth in range(5)
        ),
    )
    decoded["root_identities"] = roots
    for depth, identities in enumerate(levels):
        decoded[f"level{depth + 1}_identities"] = identities
    assert set(decoded) == set(logical)
    for key in logical:
        assert decoded[key].dtype == logical[key].dtype, key
        assert decoded[key].shape == logical[key].shape, key
        assert decoded[key].tobytes() == logical[key].tobytes(), key
    assert hash_arrays(decoded) == hash_arrays(logical)
    assert decode_manifest["logical_content_hash"] == hash_arrays(logical)
    assert encoded["payload__particle_mask"].shape[-1] == 16
    assert encoded["payload__level5_membership"].shape[-1] == 16
    assert encoded["payload__root_features"].dtype == np.uint8
    assert encoded["payload__valid_offline_counts"].dtype == np.int8


def test_compact_codec_detects_payload_shape_dtype_order_and_manifest_changes() -> None:
    logical, _ = _target_arrays()
    encoded, manifest = encode_compact_target_arrays(
        logical,
        omitted_identity_keys=_identity_keys(),
    )
    changed = {key: value.copy() for key, value in encoded.items()}
    changed["payload__root_features"][0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="encoded content hash"):
        decode_compact_target_arrays(
            changed,
            expected_encoded_content_hash=manifest["encoded_content_hash"],
        )

    changed = {key: value.copy() for key, value in encoded.items()}
    changed["payload__level1_topology"] = changed[
        "payload__level1_topology"
    ].astype(np.int16)
    with pytest.raises(ValueError, match="encoded content hash"):
        decode_compact_target_arrays(
            changed,
            expected_encoded_content_hash=manifest["encoded_content_hash"],
        )

    changed = {key: value.copy() for key, value in encoded.items()}
    payload = json.loads(changed[ABPH_COMPACT_CODEC_MANIFEST_KEY].tobytes())
    payload["array_codecs"]["particle_mask"]["shape"][-1] = 127
    changed[ABPH_COMPACT_CODEC_MANIFEST_KEY] = np.frombuffer(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    with pytest.raises(ValueError, match="manifest hash|encoded content hash"):
        decode_compact_target_arrays(changed)
