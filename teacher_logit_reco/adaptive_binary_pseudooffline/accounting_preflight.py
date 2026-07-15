"""Reusable real-target acceptance gate for the exact hierarchy accounting path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .binary_accounting import (
    ABPH_AUXILIARY_ADDITIVE_NAMES,
    ABPH_BINARY_ACCOUNTING_CONTRACT,
    ABPH_BINARY_COUNT_SUPPORT,
    ABPH_BINARY_P4_ABS_TOLERANCE,
    ABPH_BINARY_P4_REL_TOLERANCE,
    AccountingState,
    BinarySplitPrediction,
    compile_binary_split,
)
from .cache import iter_adaptive_binary_target_shards
from .root_compiler import minimum_mass_budget
from .root_transforms import ROOT_FEATURE_INDEX, ROOT_SHAPE_FEATURE_NAMES, wrap_phi_tensor
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import (
    GROUP_FEATURE_NAMES,
    PARTICLE_TARGET_NAMES,
    ROOT_FEATURE_NAMES,
    TOPOLOGY_ACTIVE_SPLIT,
    TOPOLOGY_ACTIVE_TERMINAL,
    AdaptiveBinaryTargetBatch,
)


ABPH_STEP4_PREFLIGHT_CONTRACT = "adaptive_binary_pseudooffline_step4_preflight_v1"
ABPH_TARGET_REPLAY_P4_TOLERANCE = 5.0e-3
_HARD_LEDGER_NAMES: tuple[str, ...] = (
    "energy",
    "px",
    "py",
    "pz",
    "constituent_count",
    *(f"count_{name}" for name in ABPH_PID_CATEGORIES),
    "integer_charge",
    "minimum_mass_budget",
)
_PARTICLE_INDEX = {name: index for index, name in enumerate(PARTICLE_TARGET_NAMES)}


def neutral_binary_prediction(
    batch_size: int,
    *,
    device: Any | None = None,
    dtype: Any | None = None,
    requires_grad: bool = False,
) -> BinarySplitPrediction:
    torch = require_torch()
    resolved_dtype = dtype or torch.float32

    def tensor(shape: tuple[int, ...]) -> Any:
        value = torch.zeros(shape, device=device, dtype=resolved_dtype)
        return value.requires_grad_(requires_grad)

    return BinarySplitPrediction(
        topology_logits=tensor((batch_size, 2)),
        count_logits=tensor((batch_size, ABPH_BINARY_COUNT_SUPPORT)),
        type_allocation_logits=tensor((batch_size, len(ABPH_PID_CATEGORIES))),
        charge_logits=tensor((batch_size, 2 * ABPH_MAX_PARTICLES + 1)),
        mass_allocation_logits=tensor((batch_size, 3)),
        direction_raw=tensor((batch_size, 3)),
        collinear_fraction_raw=tensor((batch_size,)),
        auxiliary_fraction_logits=tensor((batch_size, len(ABPH_AUXILIARY_ADDITIVE_NAMES))),
        child_shape_raw=tensor((batch_size, 2, len(ROOT_SHAPE_FEATURE_NAMES) + 1)),
    )


def _group_ledger(features: np.ndarray) -> np.ndarray:
    if tuple(GROUP_FEATURE_NAMES[: len(ROOT_FEATURE_NAMES)]) != tuple(ROOT_FEATURE_NAMES):
        raise RuntimeError("group target schema no longer begins with the root accounting ledger")
    return np.asarray(features[..., : len(ROOT_FEATURE_NAMES)], dtype=np.float32)


def _hard_target_residual(compiled: Any, target_children: Any) -> dict[str, float]:
    torch = require_torch()
    target = torch.as_tensor(target_children, device=compiled.child_ledger.device).float()
    residuals: dict[str, float] = {}
    for name in _HARD_LEDGER_NAMES:
        observed = compiled.child_ledger[0, :, ROOT_FEATURE_INDEX[name]]
        expected = target[:, ROOT_FEATURE_INDEX[name]]
        residuals[name] = float((observed - expected).abs().max().detach().cpu())
    return residuals


def _compile_target_children(parent_ledger: np.ndarray, child_ledgers: np.ndarray) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    parent = AccountingState.from_ledger(torch.as_tensor(parent_ledger[None, :]).float())
    children = torch.as_tensor(child_ledgers).float()
    prediction = neutral_binary_prediction(1)
    child_one_type = torch.stack(
        tuple(children[0, ROOT_FEATURE_INDEX[f"count_{name}"]].round().to(torch.long) for name in ABPH_PID_CATEGORIES)
    )[None, :]
    child_p4 = torch.stack(
        tuple(children[:, ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")),
        dim=-1,
    )[None, :, :]
    result = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=children[0, ROOT_FEATURE_INDEX["constituent_count"]].round()[None].to(torch.long),
        child_one_type_counts_override=child_one_type,
        child_one_charge_override=children[0, ROOT_FEATURE_INDEX["integer_charge"]].round()[None].to(torch.long),
        child_four_vector_override=child_p4,
    )
    return result, _hard_target_residual(result, children)


def _renderer_target_audit(
    targets: AdaptiveBinaryTargetBatch,
    jet_index: int,
    *,
    p4_tolerance: float,
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    final_features = _group_ledger(targets.level_features[-1][jet_index])
    final_mask = targets.level_masks[-1][jet_index]
    particle_rows = np.asarray(targets.particle_targets[jet_index], dtype=np.float64)
    particle_mask = np.asarray(targets.particle_mask[jet_index], dtype=bool)
    microgroup_index = np.rint(
        particle_rows[:, _PARTICLE_INDEX["target_microgroup_index"]]
    ).astype(np.int64)
    eta = particle_rows[:, _PARTICLE_INDEX["eta_hlt_relative"]] + float(targets.hlt_axis_eta[jet_index])
    phi = np.asarray(
        wrap_phi_tensor(
            require_torch().as_tensor(
                particle_rows[:, _PARTICLE_INDEX["phi_hlt_relative"]]
                + float(targets.hlt_axis_phi[jet_index])
            )
        ).cpu(),
        dtype=np.float64,
    )
    pt = particle_rows[:, _PARTICLE_INDEX["pt"]]
    energy = particle_rows[:, _PARTICLE_INDEX["energy"]]
    particle_p4 = np.stack((energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), axis=-1)
    charge = np.rint(particle_rows[:, _PARTICLE_INDEX["charge"]]).astype(np.int64)
    pid_columns = np.stack(
        tuple(particle_rows[:, _PARTICLE_INDEX[f"pid_{name}"]] for name in ABPH_PID_CATEGORIES),
        axis=-1,
    )
    pid_index = np.argmax(pid_columns, axis=-1)
    all_max_p4 = 0.0
    for group_index in np.flatnonzero(final_mask):
        selected = particle_mask & (microgroup_index == int(group_index))
        ledger = final_features[group_index]
        expected_count = int(round(float(ledger[ROOT_FEATURE_INDEX["constituent_count"]])))
        if int(selected.sum()) != expected_count:
            problems.append(f"final group {group_index}: renderer count mismatch")
            continue
        observed_p4 = particle_p4[selected].sum(axis=0)
        expected_p4 = np.asarray(
            [ledger[ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")],
            dtype=np.float64,
        )
        p4_residual = float(np.max(np.abs(observed_p4 - expected_p4), initial=0.0))
        all_max_p4 = max(all_max_p4, p4_residual)
        if p4_residual > p4_tolerance + ABPH_BINARY_P4_REL_TOLERANCE * float(np.max(np.abs(expected_p4))):
            problems.append(f"final group {group_index}: renderer four-vector mismatch {p4_residual:.6g}")
        observed_types = np.bincount(pid_index[selected], minlength=len(ABPH_PID_CATEGORIES))
        expected_types = np.asarray(
            [round(float(ledger[ROOT_FEATURE_INDEX[f"count_{name}"]])) for name in ABPH_PID_CATEGORIES],
            dtype=np.int64,
        )
        if not np.array_equal(observed_types, expected_types):
            problems.append(f"final group {group_index}: renderer type-count mismatch")
        observed_charge = int(charge[selected].sum())
        expected_charge = int(round(float(ledger[ROOT_FEATURE_INDEX["integer_charge"]])))
        if observed_charge != expected_charge:
            problems.append(f"final group {group_index}: renderer charge mismatch")
        observed_floor = float(
            minimum_mass_budget(require_torch().as_tensor(observed_types[None, :])).item()
        )
        expected_floor = float(ledger[ROOT_FEATURE_INDEX["minimum_mass_budget"]])
        if abs(observed_floor - expected_floor) > 2.0e-5:
            problems.append(f"final group {group_index}: renderer minimum-mass mismatch")
    assigned = particle_mask & (microgroup_index >= 0)
    if int(assigned.sum()) != int(particle_mask.sum()):
        problems.append("valid renderer particles are not all assigned to a final group")
    return problems, {
        "n_final_groups": int(final_mask.sum()),
        "n_rendered_particles": int(particle_mask.sum()),
        "max_renderer_four_vector_residual": all_max_p4,
    }


def audit_target_batch_feasibility(
    targets: AdaptiveBinaryTargetBatch,
    *,
    labels: np.ndarray | None = None,
    jet_indices: Sequence[int] | None = None,
    p4_tolerance: float = ABPH_TARGET_REPLAY_P4_TOLERANCE,
) -> dict[str, Any]:
    """Replay actual target ledgers through root, binary, and renderer feasibility."""

    selected = tuple(range(targets.n_jets)) if jet_indices is None else tuple(int(value) for value in jet_indices)
    problems: list[str] = []
    compiler_failures = 0
    transition_count = 0
    carry_count = 0
    renderer_groups = 0
    rendered_particles = 0
    max_hard_target_residual = 0.0
    max_renderer_p4 = 0.0
    near_massless_groups = 0
    boundary_geometry_examples = 0
    rare_type_examples = {name: 0 for name in ("electron", "muon", "other")}
    root_counts: list[int] = []
    class_counts = {name: 0 for name in LABEL_NAMES}
    for jet_index in selected:
        prefix = f"jet {jet_index}"
        if not 0 <= jet_index < targets.n_jets:
            problems.append(f"{prefix}: index outside target batch")
            continue
        if labels is not None:
            class_counts[LABEL_NAMES[int(labels[jet_index])]] += 1
        root_ledger = np.asarray(targets.root_features[jet_index], dtype=np.float32)
        try:
            root_state = AccountingState.from_ledger(require_torch().as_tensor(root_ledger[None, :]))
        except Exception as exc:
            problems.append(f"{prefix}: root feasibility failed: {exc}")
            compiler_failures += 1
            continue
        root_counts.append(int(root_state.constituent_count[0]))
        previous_ledgers = root_ledger[None, :]
        previous_mask = np.asarray((True,), dtype=bool)
        previous_topology = np.asarray(
            (
                int(TOPOLOGY_ACTIVE_TERMINAL)
                if int(root_state.constituent_count[0]) == 1
                else int(TOPOLOGY_ACTIVE_SPLIT),
            ),
            dtype=np.int8,
        )
        for depth_index in range(len(targets.level_features)):
            current_ledgers = _group_ledger(targets.level_features[depth_index][jet_index])
            current_mask = targets.level_masks[depth_index][jet_index]
            current_parents = targets.level_parent_indices[depth_index][jet_index]
            current_topology = targets.level_topology[depth_index][jet_index]
            for parent_index in np.flatnonzero(previous_mask):
                child_indices = np.flatnonzero(current_mask & (current_parents == parent_index))
                parent_ledger = previous_ledgers[parent_index]
                if previous_topology[parent_index] == int(TOPOLOGY_ACTIVE_TERMINAL):
                    carry_count += 1
                    if child_indices.size != 1:
                        problems.append(f"{prefix} depth {depth_index + 1}: terminal carry cardinality mismatch")
                        continue
                    target_child = current_ledgers[child_indices[0]]
                    hard_residual = max(
                        abs(float(parent_ledger[ROOT_FEATURE_INDEX[name]] - target_child[ROOT_FEATURE_INDEX[name]]))
                        for name in _HARD_LEDGER_NAMES
                    )
                    max_hard_target_residual = max(max_hard_target_residual, hard_residual)
                    if hard_residual > p4_tolerance:
                        problems.append(f"{prefix} depth {depth_index + 1}: terminal carry changed hard state")
                    continue
                transition_count += 1
                if child_indices.size != 2:
                    problems.append(f"{prefix} depth {depth_index + 1}: split has {child_indices.size} children")
                    compiler_failures += 1
                    continue
                child_ledgers = current_ledgers[child_indices]
                try:
                    compiled, residuals = _compile_target_children(parent_ledger, child_ledgers)
                    max_local = max(residuals.values(), default=0.0)
                    max_hard_target_residual = max(max_hard_target_residual, max_local)
                    if max_local > p4_tolerance:
                        problems.append(
                            f"{prefix} depth {depth_index + 1}: compiled target residual {max_local:.6g}"
                        )
                    near_massless_groups += int(compiled.diagnostics["near_massless_count"])
                except Exception as exc:
                    problems.append(f"{prefix} depth {depth_index + 1}: binary compile failed: {exc}")
                    compiler_failures += 1
            previous_ledgers = current_ledgers
            previous_mask = current_mask
            previous_topology = current_topology
        renderer_problems, renderer = _renderer_target_audit(
            targets, jet_index, p4_tolerance=p4_tolerance
        )
        problems.extend(f"{prefix}: {value}" for value in renderer_problems)
        renderer_groups += int(renderer["n_final_groups"])
        rendered_particles += int(renderer["n_rendered_particles"])
        max_renderer_p4 = max(max_renderer_p4, float(renderer["max_renderer_four_vector_residual"]))
        particle_rows = targets.particle_targets[jet_index]
        valid = targets.particle_mask[jet_index]
        phi_relative = particle_rows[:, _PARTICLE_INDEX["phi_hlt_relative"]]
        boundary_geometry_examples += int(np.any(valid & (np.abs(phi_relative) > np.pi - 0.10)))
        root_types = {
            name: int(round(float(root_ledger[ROOT_FEATURE_INDEX[f"count_{name}"]])))
            for name in rare_type_examples
        }
        for name, amount in root_types.items():
            rare_type_examples[name] += int(amount > 0)
    report = {
        "ok": not problems,
        "contract": ABPH_STEP4_PREFLIGHT_CONTRACT,
        "compiler_contract": ABPH_BINARY_ACCOUNTING_CONTRACT,
        "grouping": targets.layout.grouping,
        "n_jets": len(selected),
        "n_binary_transitions": transition_count,
        "n_terminal_carries": carry_count,
        "n_renderer_groups": renderer_groups,
        "n_rendered_particles": rendered_particles,
        "compiler_failure_count": compiler_failures,
        "max_hard_target_residual": max_hard_target_residual,
        "max_renderer_four_vector_residual": max_renderer_p4,
        "problems": problems[:100],
        "coverage": {
            "class_counts": class_counts,
            "all_classes_present": labels is None or all(value > 0 for value in class_counts.values()),
            "singleton_examples": sum(value == 1 for value in root_counts),
            "largest_count_value": max(root_counts, default=0),
            "largest_count_examples": sum(value == max(root_counts, default=-1) for value in root_counts),
            "cache_boundary_count_128_present": ABPH_MAX_PARTICLES in root_counts,
            "near_massless_groups": near_massless_groups,
            "boundary_geometry_examples": boundary_geometry_examples,
            "rare_particle_type_examples": rare_type_examples,
        },
    }
    return report


def synthetic_edge_case_preflight() -> dict[str, Any]:
    """Always exercise contract boundaries that may be absent from a finite cache sample."""

    torch = require_torch()
    cases: dict[str, Any] = {}

    def ledger(
        p4: tuple[float, float, float, float],
        types: tuple[int, ...],
        charge: int,
    ) -> Any:
        row = torch.zeros((1, len(ROOT_FEATURE_NAMES)), dtype=torch.float64)
        for index, name in enumerate(("energy", "px", "py", "pz")):
            row[:, ROOT_FEATURE_INDEX[name]] = p4[index]
        count = sum(types)
        row[:, ROOT_FEATURE_INDEX["constituent_count"]] = count
        for index, name in enumerate(ABPH_PID_CATEGORIES):
            row[:, ROOT_FEATURE_INDEX[f"count_{name}"]] = types[index]
        row[:, ROOT_FEATURE_INDEX["integer_charge"]] = charge
        floor = minimum_mass_budget(torch.tensor((types,))).item()
        row[:, ROOT_FEATURE_INDEX["minimum_mass_budget"]] = floor
        row[:, ROOT_FEATURE_INDEX["feasible_charge_min"]] = -count
        row[:, ROOT_FEATURE_INDEX["feasible_charge_max"]] = count
        row[:, ROOT_FEATURE_INDEX["scalar_sum_pt"]] = math.sqrt(p4[1] ** 2 + p4[2] ** 2)
        row[:, ROOT_FEATURE_INDEX["absolute_charge_sum"]] = abs(charge)
        return row

    singleton = AccountingState.from_ledger(ledger((2.0, 0.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0), 0))
    singleton_result = compile_binary_split(singleton, neutral_binary_prediction(1, dtype=torch.float64))
    cases["singleton"] = bool(not singleton_result.split_mask[0] and not singleton_result.child_mask[0].any())

    largest = AccountingState.from_ledger(ledger((250.0, 0.0, 0.0, 0.0), (0, 128, 0, 0, 0, 0), 0))
    largest_children = torch.tensor(
        [[[125.0, 0.0, 0.0, 100.0], [125.0, 0.0, 0.0, -100.0]]],
        dtype=torch.float64,
    )
    largest_result = compile_binary_split(
        largest,
        neutral_binary_prediction(1, dtype=torch.float64),
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((64,)),
        child_one_type_counts_override=torch.tensor(((0, 64, 0, 0, 0, 0),)),
        child_one_charge_override=torch.tensor((0,)),
        child_four_vector_override=largest_children,
    )
    cases["largest_count_128"] = bool(largest_result.diagnostics["ok"])

    lightlike = AccountingState.from_ledger(ledger((100.0, 0.0, 0.0, 100.0), (0, 0, 2, 0, 0, 0), 0))
    lightlike_children = torch.tensor(
        [[[40.0, 0.0, 0.0, 40.0], [60.0, 0.0, 0.0, 60.0]]],
        dtype=torch.float64,
    )
    lightlike_result = compile_binary_split(
        lightlike,
        neutral_binary_prediction(1, dtype=torch.float64),
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((1,)),
        child_one_type_counts_override=torch.tensor(((0, 0, 1, 0, 0, 0),)),
        child_one_charge_override=torch.tensor((0,)),
        child_four_vector_override=lightlike_children,
    )
    cases["near_massless"] = bool(lightlike_result.diagnostics["near_massless_count"] == 1)

    rare_types = (1, 1, 1, 1, 1, 1)
    rare = AccountingState.from_ledger(ledger((20.0, 0.0, 0.0, 0.0), rare_types, 0))
    rare_result = compile_binary_split(
        rare,
        neutral_binary_prediction(1, dtype=torch.float64),
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((3,)),
        child_one_type_counts_override=torch.tensor(((1, 1, 0, 1, 0, 0),)),
        child_one_charge_override=torch.tensor((0,)),
    )
    cases["rare_particle_types"] = bool(rare_result.diagnostics["ok"])
    boundary = wrap_phi_tensor(torch.tensor((math.pi + 1.0e-6, -math.pi - 1.0e-6)))
    cases["boundary_geometry"] = bool(
        torch.all(boundary >= -math.pi) and torch.all(boundary < math.pi)
    )
    return {"ok": all(cases.values()), "cases": cases}


def audit_target_cache_feasibility(
    cache_dir: str | Path,
    *,
    splits: Sequence[str],
    groupings: Sequence[str],
    max_jets_per_class: int = 64,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Stratify real cached targets and run the complete feasibility preflight."""

    quota = int(max_jets_per_class)
    if quota <= 0:
        raise ValueError("max_jets_per_class must be positive")
    reports: dict[str, Any] = {}
    problems: list[str] = []
    for split in tuple(str(value) for value in splits):
        for grouping in tuple(str(value) for value in groupings):
            key = f"{split}/{grouping}"
            local_reports: list[dict[str, Any]] = []
            observed = np.zeros(len(LABEL_NAMES), dtype=np.int64)
            for shard in iter_adaptive_binary_target_shards(
                cache_dir, split, grouping, verify_hash=verify_hash
            ):
                selected: list[int] = []
                for index, label in enumerate(shard.labels):
                    class_index = int(label)
                    if observed[class_index] < quota:
                        selected.append(index)
                        observed[class_index] += 1
                if selected:
                    local_reports.append(
                        audit_target_batch_feasibility(
                            shard.targets,
                            labels=shard.labels,
                            jet_indices=selected,
                        )
                    )
                if bool((observed >= quota).all()):
                    break
            local_problems = [
                problem
                for report in local_reports
                for problem in report.get("problems", ())
            ]
            if not bool(observed.all()):
                local_problems.append("not every JetClass label was represented")
            reports[key] = {
                "ok": not local_problems,
                "n_jets": int(observed.sum()),
                "class_counts": {
                    name: int(observed[index]) for index, name in enumerate(LABEL_NAMES)
                },
                "compiler_failure_count": sum(
                    int(report["compiler_failure_count"]) for report in local_reports
                ),
                "max_hard_target_residual": max(
                    (float(report["max_hard_target_residual"]) for report in local_reports),
                    default=0.0,
                ),
                "max_renderer_four_vector_residual": max(
                    (
                        float(report["max_renderer_four_vector_residual"])
                        for report in local_reports
                    ),
                    default=0.0,
                ),
                "problems": local_problems[:100],
            }
            problems.extend(f"{key}: {problem}" for problem in local_problems)
    synthetic = synthetic_edge_case_preflight()
    if not synthetic["ok"]:
        problems.append("synthetic edge-case matrix failed")
    return {
        "ok": not problems,
        "contract": ABPH_STEP4_PREFLIGHT_CONTRACT,
        "cache_dir": str(Path(cache_dir).resolve()),
        "max_jets_per_class": quota,
        "reports": reports,
        "synthetic_edge_cases": synthetic,
        "problems": problems[:200],
    }


__all__ = [
    "ABPH_STEP4_PREFLIGHT_CONTRACT",
    "ABPH_TARGET_REPLAY_P4_TOLERANCE",
    "audit_target_batch_feasibility",
    "audit_target_cache_feasibility",
    "neutral_binary_prediction",
    "synthetic_edge_case_preflight",
]
