#!/usr/bin/env python3
"""Run authoritative FP32 explicit-uu parity against installed Weaver."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier  # noqa: E402
from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    RelationalParticleTransformer,
    build_global_determinism_contract,
    build_pair_base_contract,
    build_relation_family_registry,
    build_rpt_base_model_contract,
    exact_rpt_base_config,
    inspect_weaver_runtime,
    resolve_weaver_pairwise_helper,
    source_snapshot,
    with_content_hash,
    write_immutable_json,
)


PARITY_REPORT_CONTRACT = "relational_part_weaver_parity_report_v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-weaver-version")
    return parser


def _device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _batch(torch, device, *, valid_counts=(7, 3), length=7, seed=9001):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batch = len(valid_counts)
    points = torch.randn(batch, 2, length, generator=generator)
    features = torch.randn(batch, 17, length, generator=generator)
    pt = torch.rand(batch, length, generator=generator) * 40.0 + 0.5
    eta = torch.randn(batch, length, generator=generator) * 0.8
    phi = (torch.rand(batch, length, generator=generator) * 2.0 - 1.0) * np.pi
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px.square() + py.square() + pz.square() + 0.25)
    vectors = torch.stack((px, py, pz, energy), dim=1)
    mask = torch.zeros(batch, 1, length, dtype=torch.bool)
    for row, count in enumerate(valid_counts):
        mask[row, 0, : int(count)] = True
    points = points.masked_fill(~mask, 0.0).float().to(device)
    features = features.masked_fill(~mask, 0.0).float().to(device)
    vectors = vectors.masked_fill(~mask, 0.0).float().to(device)
    return points, features, vectors, mask.to(device)


def _difference(torch, candidate, reference) -> dict[str, float]:
    absolute = (candidate.detach().double() - reference.detach().double()).abs()
    denominator = reference.detach().double().abs().clamp(min=1.0e-30)
    return {
        "maximum_absolute": float(absolute.max().cpu()) if absolute.numel() else 0.0,
        "maximum_relative": (
            float((absolute / denominator).max().cpu()) if absolute.numel() else 0.0
        ),
    }


def _assert_close(torch, candidate, reference, *, atol: float, rtol: float):
    torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
    return _difference(torch, candidate, reference)


def _assert_trimmer_state_restored(
    torch,
    trimmer,
    *,
    counter_before,
) -> None:
    if bool(trimmer.enabled) is not True:
        raise AssertionError(
            "diagnostic capture changed SequenceTrimmer enabled flag"
        )
    if not torch.equal(trimmer._counter, counter_before):
        raise AssertionError(
            "diagnostic capture changed SequenceTrimmer counter"
        )


def _reference_pair_bias(reference, vectors, mask):
    parameters = inspect.signature(reference.mod.pair_embed.forward).parameters
    if "mask" in parameters:
        return reference.mod.pair_embed(vectors, mask=mask)
    return reference.mod.pair_embed(vectors)


def _run(device_name: str, expected_version: str | None) -> dict[str, Any]:
    import torch

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    device = _device(torch, device_name)

    runtime = inspect_weaver_runtime()
    if (
        expected_version is not None
        and runtime["weaver_core_version"] != expected_version
    ):
        raise RuntimeError(
            f"Weaver version {runtime['weaver_core_version']} != "
            f"expected {expected_version}"
        )
    module = __import__(
        "weaver.nn.model.ParticleTransformer", fromlist=["ParticleTransformer"]
    )
    determinism = build_global_determinism_contract()
    tolerance = determinism["parity"]["authoritative_weaver_explicit_uu"]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])

    relation_registry = build_relation_family_registry()
    pair_contract = build_pair_base_contract(
        relation_registry_sha256=relation_registry["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
    )
    model_contract = build_rpt_base_model_contract(
        pair_base_sha256=pair_contract["content_hash"],
        weaver_runtime_sha256=runtime["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
    )

    torch.manual_seed(71)
    reference = ParticleTransformerHLTClassifier(**exact_rpt_base_config()).to(
        device=device, dtype=torch.float32
    )
    torch.manual_seed(71)
    explicit = RelationalParticleTransformer(weaver_module=module).to(
        device=device, dtype=torch.float32
    )
    reference.eval()
    explicit.eval()

    reference_state = reference.state_dict()
    explicit_state = explicit.state_dict()
    if list(reference_state) != list(explicit_state):
        raise AssertionError("state-dictionary key order differs")
    state_maximum = 0.0
    for name in reference_state:
        if (
            reference_state[name].shape != explicit_state[name].shape
            or reference_state[name].dtype != explicit_state[name].dtype
        ):
            raise AssertionError(f"state structure differs at {name}")
        torch.testing.assert_close(
            explicit_state[name], reference_state[name], atol=0.0, rtol=0.0
        )
        if reference_state[name].is_floating_point() and reference_state[name].numel():
            state_maximum = max(
                state_maximum,
                float(
                    (
                        explicit_state[name].double()
                        - reference_state[name].double()
                    )
                    .abs()
                    .max()
                    .cpu()
                ),
            )
    explicit.load_state_dict(reference_state, strict=True)

    points, features, vectors, mask = _batch(torch, device)
    helper_name, helper = resolve_weaver_pairwise_helper(module)
    with torch.no_grad():
        expected_pair = helper(
            vectors.unsqueeze(-1),
            vectors.unsqueeze(-2),
            num_outputs=4,
        )
        actual_pair = explicit.explicit_standard_four(vectors, mask)
        pair_difference = _assert_close(
            torch, actual_pair, expected_pair, atol=atol, rtol=rtol
        )
        reference_bias = _reference_pair_bias(reference, vectors, mask)
        explicit_bias = explicit.mod.pair_embed(
            vectors, uu=actual_pair, mask=mask
        )
        bias_difference = _assert_close(
            torch, explicit_bias, reference_bias, atol=atol, rtol=rtol
        )
        reference_logits = reference(points, features, vectors, mask)
        explicit_logits = explicit(points, features, vectors, mask)
        logit_difference = _assert_close(
            torch, explicit_logits, reference_logits, atol=atol, rtol=rtol
        )
        trim_points, trim_features, trim_vectors, trim_mask = _batch(
            torch,
            device,
            valid_counts=(3, 2, 4, 1, 3, 2, 4, 1),
            length=7,
            seed=9002,
        )
        del trim_points
        trim_uu = explicit.explicit_standard_four(trim_vectors, trim_mask)
        trimmer = explicit.mod.trimmer
        if bool(getattr(trimmer, "enabled", False)) is not True:
            raise RuntimeError("authoritative parity requires active trimming")
        initial_counter = trimmer._counter.detach().clone()
        trimmer_state_restored = False
        try:
            remaining = max(
                0,
                int(trimmer.warmup_steps)
                - int(trimmer._counter.detach().cpu().item()),
            )
            for _ in range(remaining):
                trimmer(
                    trim_features,
                    v=trim_vectors,
                    mask=trim_mask,
                    uu=trim_uu,
                )
            trimmed = trimmer(
                trim_features,
                v=trim_vectors,
                mask=trim_mask,
                uu=trim_uu,
            )
            if int(trimmed[2].shape[-1]) != 4:
                raise AssertionError(
                    "Weaver trimming fixture did not reduce width 7 to 4"
                )
            counter_before_diagnostics = trimmer._counter.detach().clone()
            attention_capture = explicit.diagnostics(
                trim_features, trim_vectors, trim_mask
            )["attention_allocation"]
            _assert_trimmer_state_restored(
                torch,
                trimmer,
                counter_before=counter_before_diagnostics,
            )
            trimmer_state_restored = True
        finally:
            with torch.no_grad():
                trimmer._counter.copy_(initial_counter)

    reference.zero_grad(set_to_none=True)
    explicit.zero_grad(set_to_none=True)
    reference_features = features.detach().clone().requires_grad_(True)
    explicit_features = features.detach().clone().requires_grad_(True)
    weights = torch.linspace(-0.7, 0.9, 10, device=device, dtype=torch.float32)
    (reference(points, reference_features, vectors, mask) * weights).sum().backward()
    (explicit(points, explicit_features, vectors, mask) * weights).sum().backward()
    input_gradient_difference = _assert_close(
        torch,
        explicit_features.grad,
        reference_features.grad,
        atol=atol,
        rtol=rtol,
    )
    parameter_gradient_maximum = {"maximum_absolute": 0.0, "maximum_relative": 0.0}
    reference_parameters = dict(reference.named_parameters())
    explicit_parameters = dict(explicit.named_parameters())
    if list(reference_parameters) != list(explicit_parameters):
        raise AssertionError("parameter key order differs")
    for name in reference_parameters:
        left = explicit_parameters[name].grad
        right = reference_parameters[name].grad
        if (left is None) != (right is None):
            raise AssertionError(f"gradient presence differs at {name}")
        if left is None:
            continue
        difference = _assert_close(torch, left, right, atol=atol, rtol=rtol)
        for field in parameter_gradient_maximum:
            parameter_gradient_maximum[field] = max(
                parameter_gradient_maximum[field], difference[field]
            )

    pad_points, pad_features, pad_vectors, pad_mask = _batch(
        torch, device, valid_counts=(3,), length=7, seed=104
    )
    garbage_points = pad_points.clone()
    garbage_features = pad_features.clone()
    garbage_vectors = pad_vectors.clone()
    garbage_points[:, :, 3:] = 50.0
    garbage_features[:, :, 3:] = -70.0
    garbage_vectors[:, :, 3:] = 30.0
    garbage_vectors[:, 3, 3:] = 100.0
    with torch.no_grad():
        reference_clean = reference(
            pad_points, pad_features, pad_vectors, pad_mask
        )
        reference_garbage = reference(
            garbage_points, garbage_features, garbage_vectors, pad_mask
        )
        explicit_clean = explicit(
            pad_points, pad_features, pad_vectors, pad_mask
        )
        explicit_garbage = explicit(
            garbage_points, garbage_features, garbage_vectors, pad_mask
        )
    padding_reference = _assert_close(
        torch, reference_garbage, reference_clean, atol=atol, rtol=rtol
    )
    padding_explicit = _assert_close(
        torch, explicit_garbage, explicit_clean, atol=atol, rtol=rtol
    )
    padding_cross = _assert_close(
        torch, explicit_clean, reference_clean, atol=atol, rtol=rtol
    )

    one = _batch(torch, device, valid_counts=(1,), length=5, seed=105)
    with torch.no_grad():
        one_reference = reference(*one)
        one_explicit = explicit(*one)
    if not bool(torch.isfinite(one_reference).all()) or not bool(
        torch.isfinite(one_explicit).all()
    ):
        raise FloatingPointError("one-particle parity fixture is nonfinite")
    one_difference = _assert_close(
        torch, one_explicit, one_reference, atol=atol, rtol=rtol
    )

    canonical = build_particle_transformer_inputs_from_tokens(
        np.zeros((1, 5, 14), dtype=np.float32),
        np.zeros((1, 5), dtype=bool),
        source_view="fixed_hlt",
    )
    forced = tuple(
        torch.from_numpy(value).to(device)
        for value in (
            canonical.pf_points,
            canonical.pf_features,
            canonical.pf_vectors,
            canonical.pf_mask,
        )
    )
    with torch.no_grad():
        forced_reference = reference(*forced)
        forced_explicit = explicit(*forced)
    if not bool(torch.isfinite(forced_reference).all()) or not bool(
        torch.isfinite(forced_explicit).all()
    ):
        raise FloatingPointError("forced-nonempty parity fixture is nonfinite")
    forced_difference = _assert_close(
        torch, forced_explicit, forced_reference, atol=atol, rtol=rtol
    )

    return with_content_hash(
        {
            "contract": PARITY_REPORT_CONTRACT,
            "schema_version": 3,
            "ok": True,
            "source": source_snapshot(REPO_ROOT),
            "device": str(device),
            "dtype": "float32",
            "autocast_enabled": False,
            "tf32_enabled": False,
            "deterministic_algorithms": True,
            "atol": atol,
            "rtol": rtol,
            "weaver_runtime": runtime,
            "pair_base_contract_sha256": pair_contract["content_hash"],
            "model_contract_sha256": model_contract["content_hash"],
            "helper_name": helper_name,
            "state_dictionary": {
                "key_order_exact": True,
                "shapes_and_dtypes_exact": True,
                "initial_values_bitwise_exact": True,
                "maximum_absolute_difference": state_maximum,
            },
            "standard_four_pair_features": pair_difference,
            "pair_bias": bias_difference,
            "logits": logit_difference,
            "attention_capture": attention_capture,
            "sequence_trimming_diagnostic": {
                "warmup_deliberately_exhausted": True,
                "partial_batch_size": 8,
                "original_padded_width": 7,
                "maximum_valid_count": 4,
                "ordinary_trimmed_width": 4,
                "diagnostic_captured_width": 7,
                "trimmer_state_restored": trimmer_state_restored,
            },
            "input_gradients": input_gradient_difference,
            "parameter_gradients": parameter_gradient_maximum,
            "padding": {
                "reference_clean_vs_garbage": padding_reference,
                "explicit_clean_vs_garbage": padding_explicit,
                "explicit_vs_reference": padding_cross,
            },
            "one_particle": {
                "finite": True,
                "difference": one_difference,
            },
            "forced_nonempty": {
                "source_row_was_all_invalid": True,
                "canonical_forced_row_count": int(
                    canonical.metadata[
                        "forced_nonempty_particle_transformer_rows"
                    ]
                ),
                "finite": True,
                "difference": forced_difference,
            },
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _run(args.device, args.expect_weaver_version)
    publication = None
    if args.output is not None:
        publication = write_immutable_json(args.output, report)
    print(
        json.dumps(
            {"report": report, "publication": publication},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
