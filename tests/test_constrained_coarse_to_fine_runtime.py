from __future__ import annotations

from dataclasses import dataclass

import torch

from teacher_logit_reco.constrained_coarse_to_fine import runtime


@dataclass
class _Node:
    dsl_name: str
    active: bool = True


class _FakeRegistry:
    def __init__(self) -> None:
        self._graphs = {
            ("bmm", "AutogradCUDA"): [_Node("triton"), _Node("other")],
        }
        self.disabled: list[str] = []

    def get_dsl_operations(self, name: str) -> list[str]:
        return ["bmm"] if name == "triton" else []

    def deregister_op_overrides(self, *, disable_dsl_names: str) -> None:
        self.disabled.append(disable_dsl_names)
        for nodes in self._graphs.values():
            for node in nodes:
                if node.dsl_name == disable_dsl_names:
                    node.active = False


def test_native_triton_fallback_skips_cpu() -> None:
    report = runtime.configure_torch_native_triton_fallback(torch.device("cpu"), mode="disable", probe=False)

    assert report["fallback_selected"] is False
    assert report["reason"] == "non_cuda_device"


def test_native_triton_fallback_disables_registered_overrides(monkeypatch) -> None:
    registry = _FakeRegistry()
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: registry)

    report = runtime.configure_torch_native_triton_fallback("cuda", mode="disable", probe=False)

    assert registry.disabled == ["triton"]
    assert report["fallback_selected"] is True
    assert report["registered_triton_operations"] == ["bmm"]
    assert report["active_triton_overrides_after_disable"] == 0
    assert report["probe_passed"] is False


def test_native_triton_auto_mode_is_arm_only(monkeypatch) -> None:
    registry = _FakeRegistry()
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: registry)
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")

    report = runtime.configure_torch_native_triton_fallback("cuda", mode="auto", probe=False)

    assert registry.disabled == []
    assert report["fallback_selected"] is False
    assert report["reason"] == "policy_keeps_native_overrides"
