#!/usr/bin/env python3
"""Authoritative FP32 unsplit/split H_BASE parity at blocks 2, 4, and 8."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    AuxiliaryHBaseClassifier,
    HBaseParticleTransformer,
    split_forward_contract,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    with_content_hash,
    write_immutable_json,
)


def _batch(seed: int):
    generator = torch.Generator().manual_seed(seed)
    batch, length = 3, 12
    points = torch.randn(batch, 2, length, generator=generator)
    features = torch.randn(batch, 17, length, generator=generator)
    vectors = torch.randn(batch, 4, length, generator=generator)
    mask = torch.zeros(batch, 1, length, dtype=torch.bool)
    for row, count in enumerate((12, 8, 5)):
        mask[row, 0, :count] = True
    return {
        "points": points.masked_fill(~mask, 0).float(),
        "features": features.masked_fill(~mask, 0).float(),
        "lorentz_vectors": vectors.masked_fill(~mask, 0).float(),
        "mask": mask,
    }


def _gradients(model, batch, *, split: bool):
    model.zero_grad(set_to_none=True)
    features = batch["features"].detach().clone().requires_grad_(True)
    call = {**batch, "features": features}
    if split:
        result = model.forward_with_taps(**call)
        logits = result.logits
    else:
        result = None
        logits = model(**call)
    weight = torch.linspace(-0.4, 0.6, 10)
    (logits * weight).sum().backward()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return logits.detach(), features.grad.detach(), gradients, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args(argv)
    torch.set_autocast_enabled("cpu", False)
    if torch.cuda.is_available():
        torch.set_autocast_enabled("cuda", False)
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    torch.manual_seed(args.seed)
    model = HBaseParticleTransformer(weaver_module=module).float().eval()
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    batch = _batch(args.seed)
    ordinary = _gradients(model, batch, split=False)
    split = _gradients(model, batch, split=True)
    torch.testing.assert_close(split[0], ordinary[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(split[1], ordinary[1], atol=2e-6, rtol=2e-5)
    for name in ordinary[2]:
        left, right = ordinary[2][name], split[2][name]
        if (left is None) != (right is None):
            raise AssertionError(f"gradient presence differs: {name}")
        if left is not None:
            torch.testing.assert_close(right, left, atol=2e-6, rtol=2e-5)
    if list(state_before) != list(model.state_dict()):
        raise AssertionError("split forward changed state-dictionary keys")
    for name, expected in state_before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
    captured = split[3]
    if list(captured.states) != ["TAP_EARLY", "TAP_MID", "TAP_LATE"]:
        raise AssertionError("tap order differs")
    for tap in captured.states:
        if tuple(captured.states[tap].shape[:2]) != tuple(captured.masks[tap].shape):
            raise AssertionError(f"{tap} mask/state shape differs")
    torch.manual_seed(args.seed + 1)
    auxiliary = AuxiliaryHBaseClassifier(
        model,
        target_id="T_OFFLINE_JET_10",
        target_dimension=10,
        input_dimension=128,
        availability_group_count=2,
        parameterization="ABS",
    ).float().eval()
    head_calls = []
    handle = auxiliary.target_head.register_forward_hook(
        lambda *_args: head_calls.append(1)
    )
    try:
        isolated_logits = auxiliary(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
        )
    finally:
        handle.remove()
    if head_calls:
        raise AssertionError("deployable auxiliary forward called the target head")
    torch.testing.assert_close(
        isolated_logits, ordinary[0], atol=1e-6, rtol=1e-5
    )
    tapped_logits, _ = auxiliary.forward_with_aux(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )
    torch.testing.assert_close(
        tapped_logits, ordinary[0], atol=1e-6, rtol=1e-5
    )
    report = with_content_hash({
        "contract": "hosd_weaver_split_forward_parity_v2",
        "schema_version": 2,
        "split_forward_contract_sha256": split_forward_contract()["content_hash"],
        "precision": "FP32_mixed_precision_disabled",
        "taps": ["TAP_EARLY", "TAP_MID", "TAP_LATE"],
        "logits_passed": True,
        "input_and_parameter_gradients_passed": True,
        "masks_exact": True,
        "state_dictionary_keys_shapes_values_exact": True,
        "auxiliary_classification_isolation_passed": True,
        "auxiliary_deployable_forward_called_target_head": False,
        "passed": True,
    })
    if args.output is not None:
        write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
