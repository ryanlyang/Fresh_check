from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from teacher_logit_reco.adaptive_binary_pseudooffline.tagger import (
    HierarchyAwareDualStreamTagger,
    NativeStagewiseParticleTransformer,
    PseudoViewInputs,
    TreeRelationBias,
    WeaverStagewiseParticleTransformer,
    build_variant_hierarchy_aware_tagger,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_NEURAL_TAGGER_VARIANTS,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.hypothesis_distribution import (
    ABPH_PRIMARY_HYPOTHESIS_NAMES,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.prediction_cache import (
    DeployablePseudoViewBatch,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.pseudo_consumer import (
    ABPH_RENDERER_ONLY_FRONTIER_FIELDS,
    ABPH_RENDERER_ONLY_PARTICLE_FIELDS,
    consumer_pseudo_array_names,
    renderer_only_array_names,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.distributed import (
    DistributedRuntime,
    parameter_state_hash,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_runtime import (
    _materialize_tagger_dynamic_state,
    _uninitialized_state_names,
)


def _synthetic_inputs(*, dual: bool = False, views: int = 3, latent_dim: int = 4):
    torch.manual_seed(1210)
    batch = 2
    particles = 128
    hierarchy_names = (
        ("exclusive_kt", "cambridge_aachen") if dual else ("exclusive_kt",)
    )
    shared_root = torch.randn(batch, 8)
    arrays = {
        "shared_root_ledger": shared_root,
        "hypothesis_latent": torch.randn(batch, views, latent_dim),
        "hypothesis_prior_log_prob": torch.randn(batch, views),
    }
    for hierarchy_index, hierarchy in enumerate(hierarchy_names):
        mask = torch.zeros(batch, views, particles, dtype=torch.bool)
        mask[..., :6] = True
        pt = torch.rand(batch, views, particles) + 0.5
        eta = torch.randn(batch, views, particles) * 0.15
        phi = torch.randn(batch, views, particles) * 0.15
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = torch.sqrt(px.square() + py.square() + pz.square() + 0.02)
        canonical = torch.zeros(batch, views, particles, 17)
        canonical[..., 0] = pt
        canonical[..., 1] = eta
        canonical[..., 2] = phi
        canonical[..., 3] = energy
        canonical[..., 4] = 1.0
        canonical[..., 5] = 1.0
        canonical = canonical * mask.unsqueeze(-1)
        side = torch.randn(batch, views, particles, 6) * mask.unsqueeze(-1)
        uncertainty = torch.rand(batch, views, particles) * mask
        group_indices = torch.full((batch, views, particles), -1, dtype=torch.long)
        group_indices[..., :6] = torch.arange(6) % 4
        local_indices = torch.full_like(group_indices, -1)
        local_indices[..., :6] = torch.arange(6)
        prefix = f"particle__{hierarchy}__"
        arrays.update(
            {
                prefix + "canonical_features": canonical,
                prefix + "side_channels": side,
                prefix + "four_vector": torch.stack((energy, px, py, pz), dim=-1)
                * mask.unsqueeze(-1),
                prefix + "mass": torch.full_like(pt, 0.14) * mask,
                prefix + "mask": mask,
                prefix + "group_indices": group_indices,
                prefix + "local_slot_indices": local_indices,
                prefix + "uncertainty": uncertainty,
                prefix + "slot_hidden": torch.randn(batch, views, particles, 8)
                * mask.unsqueeze(-1),
            }
        )
        for depth, capacity in enumerate((1, 2, 4)):
            frontier_mask = torch.ones(batch, views, capacity, dtype=torch.bool)
            if capacity > 1:
                frontier_mask[..., -1] = False
            ledger = torch.randn(batch, views, capacity, 8)
            if depth == 0:
                ledger[..., 0, :] = shared_root[:, None, :]
            hidden = torch.randn(batch, views, capacity, 16)
            support = torch.randn(batch, views, capacity, 3)
            parent = torch.full((batch, views, capacity), -1, dtype=torch.long)
            if depth == 1:
                parent[:] = 0
            elif depth == 2:
                parent[:] = torch.tensor([0, 0, 1, 1])
            fp = f"frontier__{hierarchy}__depth_{depth:02d}__"
            root_uncertainty = torch.rand(batch, views, capacity) * frontier_mask
            if depth == 0 and hierarchy_index > 0:
                root_uncertainty = arrays[
                    "frontier__exclusive_kt__depth_00__uncertainty"
                ].clone()
            arrays.update(
                {
                    fp + "ledger": ledger * frontier_mask.unsqueeze(-1),
                    fp + "hidden": hidden * frontier_mask.unsqueeze(-1),
                    fp + "support": support * frontier_mask.unsqueeze(-1),
                    fp + "uncertainty": root_uncertainty,
                    fp + "mask": frontier_mask,
                    fp + "topology": torch.ones(batch, views, capacity, dtype=torch.long)
                    * frontier_mask,
                    fp + "parent_indices": parent,
                    fp + "source_child_indices": torch.arange(capacity)
                    .reshape(1, 1, capacity)
                    .expand(batch, views, -1),
                }
            )
    pseudo = PseudoViewInputs(
        arrays=arrays,
        view_names=tuple(f"view_{index}" for index in range(views)),
        hierarchy_names=hierarchy_names,
        frontier_depths={name: 3 for name in hierarchy_names},
        diagnostics={
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
        },
    )
    hlt_mask = torch.zeros(batch, particles, dtype=torch.bool)
    hlt_mask[:, :8] = True
    hlt = torch.zeros(batch, particles, 14)
    hlt[:, :8, 0] = torch.rand(batch, 8) + 1.0
    hlt[:, :8, 1:3] = torch.randn(batch, 8, 2) * 0.1
    hlt[:, :8, 3] = hlt[:, :8, 0] * torch.cosh(hlt[:, :8, 1]) + 0.1
    hlt[:, :8, 4] = 1.0
    hlt[:, :8, 5] = 1.0
    return hlt, hlt_mask, pseudo


def _model(*, dual: bool = False, independent_roots: bool = False):
    torch.manual_seed(19)
    hlt = NativeStagewiseParticleTransformer(
        input_dim=17, model_dim=16, num_layers=3, num_heads=4, num_classes=10
    )
    pseudo = NativeStagewiseParticleTransformer(
        input_dim=17, model_dim=16, num_layers=3, num_heads=4, num_classes=10
    )
    return HierarchyAwareDualStreamTagger(
        hlt_backbone=hlt,
        pseudo_backbone=pseudo,
        num_classes=10,
        fusion_dim=16,
        fusion_heads=4,
        fusion_blocks_per_location=1,
        fusion_locations=(1, 2, 3),
        view_aggregator_blocks=1,
        hypothesis_latent_dim=4,
        dropout=0.0,
        dual_hierarchy=dual,
        independent_roots=independent_roots,
    )


def test_e12_materializes_dynamic_pseudo_projections_before_state_hashing():
    seed = 24731
    hlt, hlt_mask, pseudo = _synthetic_inputs(views=5, latent_dim=64)
    model = build_variant_hierarchy_aware_tagger(
        "E12_kt8_mh4_dualcross_screen", smoke=True
    )
    before = _uninitialized_state_names(model)
    assert before

    runtime = DistributedRuntime(
        rank=0,
        world_size=1,
        local_rank=0,
        backend="none",
        device_type="cpu",
    )
    report = _materialize_tagger_dynamic_state(
        model,
        lambda: model(hlt, hlt_mask, pseudo),
        distributed_runtime=runtime,
        seed=seed,
    )

    assert report["required"] is True
    assert report["materialized_state_count"] == len(before)
    assert report["remaining_uninitialized_state_names"] == []
    assert _uninitialized_state_names(model) == ()
    assert len(parameter_state_hash(model)) == 64

    expected = torch.rand(4, generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(torch.rand(4), expected, rtol=0.0, atol=0.0)


def _differentiable_copy(pseudo):
    arrays = {}
    for name, value in pseudo.arrays.items():
        copied = value.detach().clone()
        if copied.is_floating_point():
            copied.requires_grad_(True)
        arrays[name] = copied
    return PseudoViewInputs(
        arrays=arrays,
        view_names=pseudo.view_names,
        hierarchy_names=pseudo.hierarchy_names,
        frontier_depths=pseudo.frontier_depths,
        diagnostics=pseudo.diagnostics,
    )


@pytest.mark.parametrize("dual", (False, True))
def test_consumer_only_pseudo_matches_full_logits_and_gradients(dual):
    hlt, hlt_mask, template = _synthetic_inputs(dual=dual)
    full = _differentiable_copy(template)
    consumer_source = _differentiable_copy(template)
    consumer = consumer_source.to_consumer_only()
    expected_names = set(
        consumer_pseudo_array_names(
            consumer.hierarchy_names, consumer.frontier_depths
        )
    )
    assert set(consumer.arrays) == expected_names
    assert not (
        set(consumer.arrays)
        & set(renderer_only_array_names(consumer.hierarchy_names, consumer.frontier_depths))
    )
    retained = "particle__exclusive_kt__canonical_features"
    assert consumer.arrays[retained] is consumer_source.arrays[retained]

    full_model = _model(dual=dual).eval()
    with torch.no_grad():
        full_model(hlt, hlt_mask, template)
    consumer_model = copy.deepcopy(full_model).eval()
    full_logits = full_model(hlt, hlt_mask, full).logits
    consumer_logits = consumer_model(hlt, hlt_mask, consumer).logits
    torch.testing.assert_close(full_logits, consumer_logits, rtol=0.0, atol=0.0)

    full_logits.square().mean().backward()
    consumer_logits.square().mean().backward()
    for (full_name, full_parameter), (consumer_name, consumer_parameter) in zip(
        full_model.named_parameters(), consumer_model.named_parameters()
    ):
        assert full_name == consumer_name
        if full_parameter.grad is None or consumer_parameter.grad is None:
            assert full_parameter.grad is None and consumer_parameter.grad is None
        else:
            torch.testing.assert_close(
                full_parameter.grad,
                consumer_parameter.grad,
                rtol=0.0,
                atol=0.0,
            )
    torch.testing.assert_close(
        full.arrays[retained].grad,
        consumer.arrays[retained].grad,
        rtol=0.0,
        atol=0.0,
    )


def test_consumer_contract_lists_every_removed_renderer_field():
    _, _, pseudo = _synthetic_inputs(dual=True)
    removed = set(renderer_only_array_names(pseudo.hierarchy_names, pseudo.frontier_depths))
    for hierarchy in pseudo.hierarchy_names:
        assert all(
            f"particle__{hierarchy}__{field}" in removed
            for field in ABPH_RENDERER_ONLY_PARTICLE_FIELDS
        )
        for depth in range(pseudo.frontier_depths[hierarchy]):
            assert all(
                f"frontier__{hierarchy}__depth_{depth:02d}__{field}" in removed
                for field in ABPH_RENDERER_ONLY_FRONTIER_FIELDS
            )


@pytest.mark.parametrize("variant_name", ABPH_NEURAL_TAGGER_VARIANTS)
def test_every_registered_neural_tagger_reads_consumer_schema(variant_name):
    model = build_variant_hierarchy_aware_tagger(variant_name, smoke=True).eval()
    hlt, hlt_mask, pseudo = _synthetic_inputs(
        dual=model.dual_hierarchy,
        views=3,
        latent_dim=64,
    )
    pseudo = pseudo.to_consumer_only()
    independent_roots = None
    if model.independent_roots:
        root = pseudo.arrays["shared_root_ledger"]
        independent_roots = {
            "exclusive_kt": root,
            "cambridge_aachen": root + 0.125,
        }
    with torch.no_grad():
        output = model(
            hlt,
            hlt_mask,
            pseudo,
            independent_root_ledgers=independent_roots,
        )
    assert output.logits.shape == (hlt.shape[0], 10)


class _FakeEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(17, 8)

    def forward(self, features):
        return self.projection(features.transpose(1, 2))


class _FakeParticleBlock(torch.nn.Module):
    num_heads = 2

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, tokens, x_cls=None, padding_mask=None, attn_mask=None):
        del attn_mask
        if x_cls is not None:
            weights = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1.0)
            return x_cls + self.projection(pooled)
        output = tokens + self.projection(tokens)
        return output * (~padding_mask).unsqueeze(-1).to(output.dtype)


class _FakeWeaver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = _FakeEmbed()
        self.pair_embed = None
        self.blocks = torch.nn.ModuleList([_FakeParticleBlock(), _FakeParticleBlock()])
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 8))
        self.cls_blocks = torch.nn.ModuleList([_FakeParticleBlock()])
        self.norm = torch.nn.LayerNorm(8)
        self.fc = torch.nn.Linear(8, 10)

    def forward(self, features, v=None, mask=None):
        del v
        valid = mask[:, 0].bool()
        tokens = self.embed(features)
        for block in self.blocks:
            tokens = block(tokens, padding_mask=~valid)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        for block in self.cls_blocks:
            cls = block(tokens, x_cls=cls, padding_mask=~valid)
        return self.fc(self.norm(cls).squeeze(1))


def test_initial_fusion_is_near_baseline_and_every_stack_gets_gradient():
    hlt, mask, pseudo = _synthetic_inputs()
    model = _model()
    report = model.calibration_report(hlt, mask, pseudo)
    assert report["mean_absolute_logit_difference"] <= 1.0e-3
    assert report["top1_changes"] == 0

    model.train()
    output = model(hlt, mask, pseudo)
    torch.nn.functional.cross_entropy(output.logits, torch.tensor([1, 2])).backward()
    gradient_report = model.fusion_gradient_report()
    assert gradient_report["all_nonzero"]


def test_masks_make_padded_particle_and_hierarchy_values_inert():
    hlt, mask, pseudo = _synthetic_inputs()
    model = _model().eval()
    changed_arrays = dict(pseudo.arrays)
    particle_mask = changed_arrays["particle__exclusive_kt__mask"]
    for field in ("canonical_features", "side_channels", "four_vector", "uncertainty"):
        key = f"particle__exclusive_kt__{field}"
        changed = changed_arrays[key].clone()
        changed[~particle_mask] = 10000.0
        changed_arrays[key] = changed
    frontier_mask = changed_arrays["frontier__exclusive_kt__depth_02__mask"]
    for field in ("ledger", "hidden", "support", "uncertainty"):
        key = f"frontier__exclusive_kt__depth_02__{field}"
        changed = changed_arrays[key].clone()
        changed[~frontier_mask] = -9000.0
        changed_arrays[key] = changed
    changed_pseudo = copy.copy(pseudo)
    object.__setattr__(changed_pseudo, "arrays", changed_arrays)
    with torch.no_grad():
        first = model(hlt, mask, pseudo).logits
        second = model(hlt, mask, changed_pseudo).logits
    torch.testing.assert_close(first, second, atol=1.0e-6, rtol=1.0e-6)


def test_tree_relations_are_particle_permutation_equivariant():
    module = TreeRelationBias(num_heads=4)
    ancestors = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3]]])
    uncertainty = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    order = torch.tensor([2, 0, 3, 1])
    original = module(ancestors, uncertainty, mask)
    permuted = module(ancestors[:, order], uncertainty[:, order], mask[:, order])
    torch.testing.assert_close(
        permuted, original[:, :, order][:, :, :, order], atol=1.0e-6, rtol=1.0e-6
    )


def test_hypothesis_order_is_invariant_and_hlt_only_path_is_exact():
    hlt, mask, pseudo = _synthetic_inputs()
    model = _model().eval()
    with torch.no_grad():
        original = model(hlt, mask, pseudo)
        permuted = model(hlt, mask, pseudo.permute_views([2, 0, 1]))
        hlt_only = model(hlt, mask)
    torch.testing.assert_close(original.logits, permuted.logits, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(hlt_only.logits, hlt_only.baseline_logits)
    assert hlt_only.diagnostics["offline_inputs_loaded"] is False
    assert hlt_only.diagnostics["teacher_logits_loaded"] is False
    model.set_all_residual_scales(0.0)
    with torch.no_grad():
        zero_residual = model(hlt, mask, pseudo)
    torch.testing.assert_close(
        zero_residual.logits, zero_residual.baseline_logits, atol=0.0, rtol=0.0
    )


def test_e7_uses_one_shared_root_and_only_e11_accepts_independent_roots():
    hlt, mask, pseudo = _synthetic_inputs(dual=True)
    e7 = _model(dual=True).eval()
    with torch.no_grad():
        output = e7(hlt, mask, pseudo)
    provenance = output.diagnostics["root_provenance"]
    assert provenance["shared_root"] is True
    assert provenance["root_hash_count"] == 1
    assert len(set(provenance["branch_root_hashes"].values())) == 1
    independent = {
        "exclusive_kt": pseudo.arrays["shared_root_ledger"],
        "cambridge_aachen": pseudo.arrays["shared_root_ledger"] + 1.0,
    }
    with pytest.raises(ValueError, match="only E11"):
        e7(hlt, mask, pseudo, independent_root_ledgers=independent)

    e11 = _model(dual=True, independent_roots=True).eval()
    with torch.no_grad():
        diagnostic = e11(hlt, mask, pseudo, independent_root_ledgers=independent)
    assert diagnostic.diagnostics["root_provenance"]["shared_root"] is False
    assert diagnostic.diagnostics["root_provenance"]["root_hash_count"] == 2


def test_generated_deployable_cache_batch_is_a_teacher_free_tagger_input():
    hlt, mask, pseudo = _synthetic_inputs(views=5)
    deployable = DeployablePseudoViewBatch(
        arrays={name: value.numpy() for name, value in pseudo.arrays.items()},
        view_names=ABPH_PRIMARY_HYPOTHESIS_NAMES,
        hierarchy_names=pseudo.hierarchy_names,
        frontier_depths=pseudo.frontier_depths,
        diagnostics={
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
        },
    )
    model = _model().eval()
    with torch.no_grad():
        output = model(hlt, mask, deployable)
    assert output.logits.shape == (2, 10)
    assert output.diagnostics["offline_inputs_loaded"] is False
    assert output.diagnostics["teacher_logits_loaded"] is False


def test_stagewise_weaver_adapter_matches_the_reference_forward():
    torch.manual_seed(88)
    adapter = WeaverStagewiseParticleTransformer(_FakeWeaver())
    features = torch.randn(2, 17, 8)
    four_vector = torch.randn(2, 4, 8)
    mask = torch.ones(2, 1, 8, dtype=torch.bool)
    report = adapter.reference_parity_report(features, four_vector, mask)
    assert report["ok"]
    assert report["maximum_absolute_difference"] <= 1.0e-6
