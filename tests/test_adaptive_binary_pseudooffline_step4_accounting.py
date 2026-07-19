from __future__ import annotations

from dataclasses import fields, replace
import math

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_AUXILIARY_ADDITIVE_NAMES,
    ABPH_BINARY_P4_ABS_TOLERANCE,
    ABPH_BINARY_P4_REL_TOLERANCE,
    ABPH_MIN_REQUIRED_BINARY_LOSS_WEIGHT,
    ABPH_MAX_PARTICLES,
    ABPH_PID_CATEGORIES,
    ROOT_FEATURE_INDEX,
    ROOT_FEATURE_NAMES,
    AccountingState,
    AdaptiveBinaryHierarchyLayout,
    BinaryAccountingLossWeights,
    BinarySplitPrediction,
    accounting_state_audit,
    allocate_child_type_counts,
    audit_target_batch_feasibility,
    binary_accounting_manifest,
    build_adaptive_binary_targets,
    compile_binary_split,
    compute_binary_accounting_losses,
    feasible_charge_mask,
    minimum_mass_budget,
    neutral_binary_prediction,
    synthetic_edge_case_preflight,
    two_body_phase_space_split,
    wrap_phi,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.targets import (
    TOPOLOGY_ACTIVE_SPLIT,
    TOPOLOGY_ACTIVE_TERMINAL,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.root_transforms import (
    make_four_vector_mass_representable,
)


_MASS = (0.13957039, 0.0, 0.0, 0.00051099895, 0.1056583755, 0.0)


def _ledger(
    p4: tuple[float, float, float, float],
    type_counts: tuple[int, ...],
    charge: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    row = torch.zeros((1, len(ROOT_FEATURE_NAMES)), dtype=dtype)
    for index, name in enumerate(("energy", "px", "py", "pz")):
        row[:, ROOT_FEATURE_INDEX[name]] = p4[index]
    count = sum(type_counts)
    row[:, ROOT_FEATURE_INDEX["constituent_count"]] = count
    for index, name in enumerate(ABPH_PID_CATEGORIES):
        row[:, ROOT_FEATURE_INDEX[f"count_{name}"]] = type_counts[index]
    row[:, ROOT_FEATURE_INDEX["integer_charge"]] = charge
    floor = minimum_mass_budget(torch.tensor((type_counts,), dtype=dtype))[0]
    row[:, ROOT_FEATURE_INDEX["minimum_mass_budget"]] = floor
    row[:, ROOT_FEATURE_INDEX["feasible_charge_min"]] = -count
    row[:, ROOT_FEATURE_INDEX["feasible_charge_max"]] = count
    pt = math.hypot(p4[1], p4[2])
    row[:, ROOT_FEATURE_INDEX["scalar_sum_pt"]] = pt
    row[:, ROOT_FEATURE_INDEX["absolute_charge_sum"]] = abs(charge)
    for index, name in enumerate(ABPH_PID_CATEGORIES):
        row[:, ROOT_FEATURE_INDEX[f"energy_{name}"]] = p4[0] * type_counts[index] / max(count, 1)
        row[:, ROOT_FEATURE_INDEX[f"scalar_pt_{name}"]] = pt * type_counts[index] / max(count, 1)
    return row


def _random_prediction(
    batch: int,
    *,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> BinarySplitPrediction:
    torch.manual_seed(44)
    template = neutral_binary_prediction(batch, dtype=dtype)
    values = {}
    for field in fields(template):
        value = getattr(template, field.name)
        random = torch.randn(value.shape, dtype=dtype)
        if field.name == "topology_logits":
            random[:, 1] += 5.0
        values[field.name] = random.requires_grad_(requires_grad)
    return BinarySplitPrediction(**values)


def _token(
    pt: float,
    eta: float,
    phi: float,
    *,
    pid: int,
    charge: float,
) -> np.ndarray:
    row = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    momentum = pt * np.cosh(eta)
    row[:5] = (
        pt,
        eta,
        phi,
        np.sqrt(momentum * momentum + _MASS[pid] ** 2),
        charge,
    )
    if pid < 5:
        row[5 + pid] = 1.0
    row[10:] = (0.01, 0.002, -0.03, 0.004)
    return row


def _real_target_batch() -> tuple[object, np.ndarray]:
    n_jets = 10
    hlt = np.zeros((n_jets, 128, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((n_jets, 128), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    identities: list[JetIdentity] = []
    for jet_index in range(n_jets):
        count = 1 if jet_index == 0 else (2 if jet_index == 1 else 5 + jet_index)
        offline_mask[jet_index, :count] = True
        hlt_count = max(1, count - 2)
        hlt_mask[jet_index, :hlt_count] = True
        for particle_index in range(count):
            if jet_index == 1:
                pid = 2
                eta = 0.12
                phi = -0.7
                charge = 0.0
            else:
                pid = particle_index % len(ABPH_PID_CATEGORIES)
                eta = -0.35 + 0.055 * particle_index
                phi = float(wrap_phi(3.11 + 0.061 * particle_index))
                if pid in (0, 3, 4):
                    charge = 1.0 if particle_index % 2 == 0 else -1.0
                elif pid == 5:
                    charge = float((particle_index % 3) - 1)
                else:
                    charge = 0.0
            row = _token(
                38.0 - 0.8 * particle_index + jet_index,
                eta,
                phi,
                pid=pid,
                charge=charge,
            )
            offline[jet_index, particle_index] = row
            if particle_index < hlt_count:
                hlt[jet_index, particle_index] = row
        identities.append(
            JetIdentity(file=f"class_{jet_index}_edge.root", entry=jet_index, label=jet_index)
        )
    targets = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=tuple(identities),
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )
    return targets, np.arange(n_jets, dtype=np.int64)


def test_random_valid_parents_conserve_every_hard_channel():
    rows = []
    for index in range(32):
        count = 2 + (index % 30)
        types = (0, count, 0, 0, 0, 0)
        rows.append(_ledger((200.0 + index, 0.0, 0.0, 0.0), types, 0)[0])
    parent = AccountingState.from_ledger(torch.stack(rows))
    prediction = _random_prediction(parent.batch_size)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.full((parent.batch_size,), int(TOPOLOGY_ACTIVE_SPLIT)),
    )
    assert compiled.diagnostics["ok"]
    assert compiled.diagnostics["hard"]["max_count_residual"] == 0
    assert compiled.diagnostics["hard"]["max_type_count_residual"] == 0
    assert compiled.diagnostics["hard"]["max_charge_residual"] == 0
    # One representability ULP may be added to each child energy so its stored
    # float32 p4 retains the compiled mass floor on the next hierarchy level.
    assert compiled.diagnostics["hard"]["max_four_vector_residual"] <= 2e-5
    assert torch.equal(
        compiled.child_type_counts.sum(dim=2), compiled.child_constituent_count
    )
    assert torch.equal(
        compiled.child_type_counts.sum(dim=1), parent.type_counts
    )


def test_exact_type_transport_respects_capacities_and_has_finite_gradients():
    logits = torch.randn(64, 6, requires_grad=True)
    capacities = torch.randint(0, 8, (64, 6))
    capacities[:, 0] += 2
    parent_count = capacities.sum(dim=-1)
    child_count = torch.maximum(torch.ones_like(parent_count), parent_count // 2)
    child_count = torch.minimum(child_count, parent_count - 1)
    hard, relaxed = allocate_child_type_counts(logits, capacities, child_count)
    assert torch.equal(hard.sum(dim=-1), child_count)
    assert bool((hard <= capacities).all())
    relaxed.square().mean().backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


def test_two_body_layer_recovers_target_direction_and_conserves_boosted_parent():
    parent = torch.tensor(((20.0, 3.0, -2.0, 4.0),), dtype=torch.float64)
    parent_mass = torch.sqrt(parent[:, 0].square() - parent[:, 1:].square().sum(dim=-1))
    masses = torch.tensor(((2.0, 3.0),), dtype=torch.float64, requires_grad=True)
    direction = torch.tensor(((0.3, -0.4, 0.5),), dtype=torch.float64, requires_grad=True)
    first, second, diagnostics = two_body_phase_space_split(
        parent,
        masses,
        direction,
        collinear_fraction=torch.tensor((0.4,), dtype=torch.float64),
    )
    torch.testing.assert_close(first + second, parent, rtol=0.0, atol=1e-11)
    first_mass = torch.sqrt((first[:, 0].square() - first[:, 1:].square().sum(dim=-1)).clamp_min(0))
    second_mass = torch.sqrt((second[:, 0].square() - second[:, 1:].square().sum(dim=-1)).clamp_min(0))
    torch.testing.assert_close(first_mass, masses[:, 0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(second_mass, masses[:, 1], rtol=1e-10, atol=1e-10)
    assert bool(masses.sum(dim=-1) <= parent_mass)
    assert diagnostics["near_massless_count"] == 0
    (first.square().sum() + second.square().sum()).backward()
    assert bool(torch.isfinite(masses.grad).all())
    assert bool(torch.isfinite(direction.grad).all())


def test_near_massless_parent_uses_documented_collinear_branch():
    parent = torch.tensor(((80.0, 0.0, 0.0, 80.0),), dtype=torch.float32)
    first, second, diagnostics = two_body_phase_space_split(
        parent,
        torch.zeros((1, 2)),
        torch.randn(1, 3),
        collinear_fraction=torch.tensor((0.25,)),
    )
    assert diagnostics["near_massless_count"] == 1
    torch.testing.assert_close(first, 0.25 * parent)
    torch.testing.assert_close(second, 0.75 * parent)
    torch.testing.assert_close(first + second, parent)


def test_exactly_lightlike_compiler_has_finite_prediction_gradients():
    parent = AccountingState.from_ledger(
        _ledger((80.0, 0.0, 0.0, 80.0), (0, 0, 2, 0, 0, 0), 0)
    )
    prediction = _random_prediction(1, requires_grad=True)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((1,)),
        child_one_type_counts_override=torch.tensor(((0, 0, 1, 0, 0, 0),)),
    )
    assert compiled.diagnostics["ok"]
    objective = (
        compiled.child_four_vector.square().mean()
        + compiled.relaxed_split_probability.mean()
        + compiled.relaxed_child_constituent_count.square().mean()
        + compiled.relaxed_child_type_counts.square().mean()
        + compiled.child_scalar_sum_pt.square().mean()
    )
    objective.backward()
    gradients = [
        getattr(prediction, field.name).grad
        for field in fields(prediction)
        if getattr(prediction, field.name).grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_tolerance_accepted_parent_mass_roundoff_is_projected_before_split():
    type_counts = (2, 0, 0, 0, 0, 0)
    floor = 2.0 * _MASS[0]
    represented_mass = floor - 1.0e-5
    energy = 20.0
    pz = math.sqrt(energy * energy - represented_mass * represented_mass)
    parent = AccountingState.from_ledger(
        _ledger(
            (energy, 0.0, 0.0, pz),
            type_counts,
            0,
            dtype=torch.float64,
        )
    )
    prediction = _random_prediction(1, dtype=torch.float64)

    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((1,)),
        child_one_type_counts_override=torch.tensor(((1, 0, 0, 0, 0, 0),)),
    )

    assert compiled.diagnostics["ok"]
    residual = (compiled.child_four_vector.sum(dim=1) - parent.four_vector).abs()
    tolerance = (
        ABPH_BINARY_P4_ABS_TOLERANCE
        + ABPH_BINARY_P4_REL_TOLERANCE * parent.four_vector.abs()
    )
    assert bool((residual <= tolerance).all())
    torch.testing.assert_close(
        compiled.child_minimum_mass_budget.sum(dim=1),
        parent.minimum_mass_budget,
        rtol=0.0,
        atol=2.0e-8,
    )
    for child_index in range(2):
        child = AccountingState.from_ledger(compiled.child_ledger[:, child_index])
        assert accounting_state_audit(child)["minimum_mass_margin_min"] >= 0.0


def test_boosted_float32_children_remain_valid_recursive_parents():
    type_counts = (2, 0, 0, 0, 0, 0)
    floor = torch.tensor((2.0 * _MASS[0],), dtype=torch.float32)
    p4 = make_four_vector_mass_representable(
        torch.tensor(((1000.0, 0.0, 0.0, 1000.0),), dtype=torch.float32),
        floor,
    )
    parent = AccountingState.from_ledger(
        _ledger(tuple(float(value) for value in p4[0]), type_counts, 0)
    )
    prediction = _random_prediction(1)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((1,)),
        child_one_type_counts_override=torch.tensor(((1, 0, 0, 0, 0, 0),)),
        child_one_charge_override=torch.tensor((1,)),
    )
    assert compiled.diagnostics["ok"]
    for child_index in range(2):
        child = AccountingState.from_ledger(compiled.child_ledger[:, child_index])
        assert accounting_state_audit(child)["ok"]


def test_tolerance_accepted_boosted_parent_does_not_scale_recursive_child_floors():
    type_counts = (2, 0, 0, 0, 0, 0)
    # This lightlike FP32 representation is accepted for a 1 TeV boosted state
    # because its missing sub-GeV mass lies inside the explicit cancellation
    # tolerance. The split compiler must repair that representation rather
    # than reduce either child's exact type-conditioned mass floor.
    parent = AccountingState.from_ledger(
        _ledger((1000.0, 0.0, 0.0, 1000.0), type_counts, 0)
    )
    assert accounting_state_audit(parent)["minimum_mass_margin_min"] >= 0.0
    prediction = _random_prediction(1)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((1,)),
        child_one_type_counts_override=torch.tensor(((1, 0, 0, 0, 0, 0),)),
        child_one_charge_override=torch.tensor((1,)),
    )
    assert compiled.diagnostics["ok"]
    torch.testing.assert_close(
        compiled.child_minimum_mass_budget.sum(dim=1),
        parent.minimum_mass_budget,
        rtol=0.0,
        atol=2.0e-8,
    )
    for child_index in range(2):
        child = AccountingState.from_ledger(compiled.child_ledger[:, child_index])
        assert accounting_state_audit(child)["minimum_mass_margin_min"] >= 0.0


def test_material_mass_deficit_is_not_hidden_by_ledger_canonicalization():
    type_counts = (128, 0, 0, 0, 0, 0)
    # Representing 128 massive charged hadrons with a lightlike 1 TeV p4 would
    # require an O(100 MeV) energy change, far outside the p4 closure contract.
    # This is physical inconsistency, not floating-point cancellation.
    with pytest.raises(ValueError, match="four-vector mass lies below"):
        AccountingState.from_ledger(
            _ledger((1000.0, 0.0, 0.0, 1000.0), type_counts, 0)
        )


def test_component_safe_correction_handles_mass_domain_cancellation():
    type_counts = (8, 0, 0, 0, 0, 0)
    raw = _ledger((1000.0, 0.0, 0.0, 1000.0), type_counts, 0)
    floor = float(raw[0, ROOT_FEATURE_INDEX["minimum_mass_budget"]])
    mass_tolerance = 3.0e-5 + 5.0e-4 * 1000.0
    assert floor > mass_tolerance

    state = AccountingState.from_ledger(raw)
    report = accounting_state_audit(state)
    assert report["ok"]
    assert report["minimum_mass_margin_min"] >= 0.0
    energy_correction = float(state.four_vector[0, 0] - raw[0, ROOT_FEATURE_INDEX["energy"]])
    component_tolerance = ABPH_BINARY_P4_ABS_TOLERANCE + ABPH_BINARY_P4_REL_TOLERANCE * 1000.0
    assert 0.0 < energy_correction <= component_tolerance


def test_singleton_is_forced_terminal_and_has_no_physical_empty_child():
    parent = AccountingState.from_ledger(
        _ledger((3.0, 0.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0), 0)
    )
    prediction = _random_prediction(1)
    compiled = compile_binary_split(parent, prediction)
    assert int(compiled.topology[0]) == int(TOPOLOGY_ACTIVE_TERMINAL)
    assert not bool(compiled.child_mask.any())
    assert not bool(compiled.split_mask[0])
    with pytest.raises(ValueError, match="singleton"):
        compile_binary_split(
            parent,
            prediction,
            topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        )


def test_soft_auxiliary_heads_are_separate_and_parent_consistent():
    parent = AccountingState.from_ledger(
        _ledger((50.0, 10.0, 0.0, 0.0), (0, 8, 0, 0, 0, 0), 0)
    )
    compiled = compile_binary_split(
        parent,
        _random_prediction(1),
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
    )
    torch.testing.assert_close(compiled.child_scalar_sum_pt.sum(dim=1), parent.scalar_sum_pt)
    torch.testing.assert_close(compiled.child_type_energy.sum(dim=1), parent.type_energy)
    torch.testing.assert_close(compiled.child_type_scalar_pt.sum(dim=1), parent.type_scalar_pt)
    assert compiled.diagnostics["soft"]["scalar_pt_consistency_mae"] < 1e-6
    assert compiled.diagnostics["hard"]["max_four_vector_residual"] < 1e-5


def test_actual_target_hierarchy_and_renderer_replay_with_zero_failures():
    targets, labels = _real_target_batch()
    report = audit_target_batch_feasibility(targets, labels=labels)
    assert report["ok"], report["problems"][:10]
    assert report["compiler_failure_count"] == 0
    assert report["n_binary_transitions"] > 0
    assert report["n_terminal_carries"] > 0
    assert report["n_renderer_groups"] > 0
    assert report["n_rendered_particles"] == int(targets.particle_mask.sum())
    assert report["max_hard_target_residual"] <= 5.0e-3
    assert report["max_renderer_four_vector_residual"] <= 5.0e-3
    assert report["coverage"]["all_classes_present"] is True
    assert report["coverage"]["singleton_examples"] == 1
    assert all(
        value > 0 for value in report["coverage"]["rare_particle_type_examples"].values()
    )


def test_real_target_preflight_is_stable_for_highly_boosted_collinear_jets():
    hlt = np.zeros((1, 128, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((1, 128), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    for particle_index in range(32):
        pid = particle_index % len(ABPH_PID_CATEGORIES)
        pt = 4000.0 - 30.0 * particle_index
        eta = 4.4 + 1.0e-4 * particle_index
        phi = 1.2 + 2.0e-4 * particle_index
        charge = (
            (1.0 if particle_index % 2 == 0 else -1.0)
            if pid in (0, 3, 4)
            else 0.0
        )
        row = _token(pt, eta, phi, pid=pid, charge=charge)
        hlt[0, particle_index] = row
        offline[0, particle_index] = row
        hlt_mask[0, particle_index] = True
        offline_mask[0, particle_index] = True
    targets = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=(JetIdentity(file="boosted.root", entry=0, label=0),),
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )

    report = audit_target_batch_feasibility(targets)

    assert report["ok"], report["problems"][:10]
    assert report["compiler_failure_count"] == 0
    # Absolute float32 parent/child closure grows with the multi-TeV p4 scale;
    # acceptance above is governed by the compiler's relative tolerance.
    assert report["max_hard_target_residual"] < 1.0


def test_synthetic_campaign_edge_matrix_covers_all_required_boundaries():
    report = synthetic_edge_case_preflight()
    assert report["ok"], report
    assert set(report["cases"]) == {
        "singleton",
        "largest_count_128",
        "near_massless",
        "rare_particle_types",
        "boundary_geometry",
    }
    assert all(report["cases"].values())


def test_infeasible_overrides_fail_closed_without_repair():
    parent = AccountingState.from_ledger(
        _ledger((30.0, 0.0, 0.0, 0.0), (0, 4, 0, 0, 0, 0), 0)
    )
    prediction = _random_prediction(1)
    with pytest.raises(ValueError, match="outside the parent budget"):
        compile_binary_split(
            parent,
            prediction,
            topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
            child_one_count_override=torch.tensor((4,)),
        )
    bad_p4 = torch.tensor(
        [[[20.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]]
    )
    with pytest.raises(ValueError, match="does not conserve"):
        compile_binary_split(
            parent,
            prediction,
            topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
            child_one_count_override=torch.tensor((2,)),
            child_one_type_counts_override=torch.tensor(((0, 2, 0, 0, 0, 0),)),
            child_one_charge_override=torch.tensor((0,)),
            child_four_vector_override=bad_p4,
        )


def test_feasible_four_vector_override_is_preserved_exactly():
    parent = AccountingState.from_ledger(
        _ledger((30.0, 0.0, 0.0, 0.0), (0, 4, 0, 0, 0, 0), 0, dtype=torch.float64)
    )
    target = torch.tensor(
        [[[12.0, 3.0, 0.0, 0.0], [18.0, -3.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    compiled = compile_binary_split(
        parent,
        _random_prediction(1, dtype=torch.float64),
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
        child_one_count_override=torch.tensor((2,)),
        child_one_type_counts_override=torch.tensor(((0, 2, 0, 0, 0, 0),)),
        child_one_charge_override=torch.tensor((0,)),
        child_four_vector_override=target,
    )

    torch.testing.assert_close(compiled.child_four_vector, target, rtol=0.0, atol=0.0)
    assert compiled.diagnostics["phase_space_branch"] == "validated_exact_target_override"


def test_all_relaxed_paths_have_finite_gradients():
    parent = AccountingState.from_ledger(
        _ledger((80.0, 5.0, 3.0, -2.0), (2, 4, 2, 2, 2, 2), 0)
    )
    prediction = _random_prediction(1, requires_grad=True)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
    )
    objective = (
        compiled.relaxed_split_probability.sum()
        + compiled.relaxed_child_constituent_count.square().mean()
        + compiled.relaxed_child_type_counts.square().mean()
        + compiled.relaxed_child_charge.square().mean()
        + compiled.child_four_vector.square().mean()
        + compiled.child_scalar_sum_pt.square().mean()
        + compiled.child_shape_features.square().mean()
    )
    objective.backward()
    for field in fields(prediction):
        gradient = getattr(prediction, field.name).grad
        assert gradient is not None, field.name
        assert bool(torch.isfinite(gradient).all()), field.name


def test_required_direct_and_auxiliary_binary_losses_are_finite_and_weighted():
    parent = AccountingState.from_ledger(
        _ledger((90.0, 4.0, -3.0, 2.0), (2, 5, 2, 1, 1, 1), 0)
    )
    prediction = _random_prediction(1, requires_grad=True)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
    )
    output = compute_binary_accounting_losses(
        prediction,
        compiled,
        parent,
        compiled.child_ledger.detach(),
        compiled.child_mask,
        torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
    )
    assert torch.isfinite(output.total)
    assert set(output.components) == {
        "topology_nll",
        "count_nll",
        "type_count_huber",
        "charge_nll",
        "four_vector_huber",
        "scalar_pt_huber",
        "type_energy_huber",
        "type_pt_huber",
        "shape_huber",
        "auxiliary_consistency",
    }
    assert all(
        float(value) >= ABPH_MIN_REQUIRED_BINARY_LOSS_WEIGHT
        for value in BinaryAccountingLossWeights().to_dict().values()
        if isinstance(value, float)
    )
    output.total.backward()
    assert prediction.topology_logits.grad is not None
    assert bool(torch.isfinite(prediction.topology_logits.grad).all())


def test_mixed_precision_inputs_compile_in_float32_within_hard_tolerance():
    parent = AccountingState.from_ledger(
        _ledger((100.0, 8.0, -3.0, 4.0), (0, 12, 0, 0, 0, 0), 0).half()
    )
    prediction = _random_prediction(1, dtype=torch.float16)
    compiled = compile_binary_split(
        parent,
        prediction,
        topology_override=torch.tensor((int(TOPOLOGY_ACTIVE_SPLIT),)),
    )
    assert compiled.child_four_vector.dtype == torch.float32
    assert compiled.diagnostics["ok"]
    assert compiled.diagnostics["hard"]["max_four_vector_residual"] < 5e-4


def test_manifest_hash_binds_compiler_order_mass_table_and_safe_branch():
    first = binary_accounting_manifest()
    second = binary_accounting_manifest()
    assert first == second
    assert len(first["compiler_hash"]) == 64
    assert first["compiler_order"][0] == "topology"
    assert first["near_massless_branch"] == "positive_collinear_fraction"


def test_parent_ledger_validation_rejects_inconsistent_mass_budget():
    row = _ledger((20.0, 0.0, 0.0, 0.0), (2, 0, 0, 0, 0, 0), 0)
    row[:, ROOT_FEATURE_INDEX["minimum_mass_budget"]] = 0.0
    report = accounting_state_audit(AccountingState.from_ledger(row, validate=False))
    assert not report["ok"]
    with pytest.raises(ValueError, match="minimum-mass"):
        AccountingState.from_ledger(row)
