"""Explicit native-Weaver/outer precision and observed activation checks.

Weaver 0.4.17 enters its own CUDA autocast context. Setting only an outer
context does not enable AMP when ParticleTransformer.use_amp is False.
Scope both settings to the forward, leaving validation/diagnostics in FP32.
This module belongs only to the screen; historical full-scale runs are untouched.
"""
from contextlib import contextmanager

import torch
from torch import nn


POLICY = dict(training="fp16_autocast_with_fp32_parameters_and_grad_scaler",
              evaluation="fp32", native_amp="scoped_to_training_forward",
              audit="observed_linear_and_conv1d_outputs_v1")


@contextmanager
def model_precision(model, device, *, training):
    """Set both autocast controls and restore the native flag even on failure."""
    enabled = bool(training and device.type == "cuda")
    native = getattr(model, "mod", None)
    has_flag = native is not None and hasattr(native, "use_amp")
    previous = native.use_amp if has_flag else None
    try:
        if has_flag:
            native.use_amp = enabled
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
            yield
    finally:
        if has_flag:
            native.use_amp = previous


@contextmanager
def audit_precision(model, *, expected_dtype):
    """Observe real operator outputs, not merely requested autocast settings.

Used on the first benchmark training and validation forwards, not every update.
LayerNorm/softmax are deliberately excluded: AMP may keep those in FP32.
"""
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if parameter_dtypes != [str(torch.float32)]:
        raise RuntimeError(f"Screen parameters must remain FP32: {parameter_dtypes}")
    record = dict(passed=False, expected_activation_dtype=str(expected_dtype),
                  parameter_dtypes=parameter_dtypes, observed_activations={})
    handles = []

    def observer(name):
        def observe(_module, _inputs, output):
            dtype = str(output.dtype)
            record["observed_activations"][name] = dtype
            if output.dtype != expected_dtype:
                raise RuntimeError(f"Precision audit failed at {name}: expected "
                                   f"{expected_dtype}, observed {dtype}")
        return observe

    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                handles.append(module.register_forward_hook(observer(name)))
        yield record
        if not record["observed_activations"]:
            raise RuntimeError("Precision audit observed no Linear/Conv1d outputs")
        record["passed"] = True
    finally:
        for handle in handles:
            handle.remove()
