from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    GlobalTargetHead,
    HBaseParticleTransformer,
    HOSDTrainingProtocol,
    PairTargetHead,
    ParticleTargetHead,
    build_baseline_registry,
    build_stage_c_plan,
    build_tap_probe,
    classification_kd_loss,
    continuous_probe_metrics,
    deterministic_pair_indices,
    heteroscedastic_nll,
    latent_probe_metrics,
    pair_probe_metrics,
    particle_net_config,
    split_forward_contract,
    statistical_references,
    target_head_registry,
    teacher_probe_metrics,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_c_training import (
    train_stage_c_baseline,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


class _PairEmbed(torch.nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.pairwise_lv_dim = 4
        self.pairwise_input_dim = 0
        self.is_symmetric = True
        self.out_dim = heads
        self.remove_self_pair = False
        self.sparse_eval = (False, False)
        self.embed = torch.nn.Sequential(
            torch.nn.BatchNorm1d(4),
            torch.nn.Conv1d(4, 7, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(7, heads, 1),
        )

    def forward(self, v, uu=None, mask=None):
        del v, mask
        batch, _, length, _ = uu.shape
        return self.embed(uu.reshape(batch, 4, -1)).reshape(batch, -1, length, length)


class _ParticleBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(dimension)

    def forward(self, x, padding_mask=None, attn_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.flatten(0, 1)
        update, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.norm(x + update)


class _ClassBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )

    def forward(self, x, x_cls=None, padding_mask=None):
        context = torch.cat((x_cls, x), dim=1)
        cls_mask = torch.zeros(
            padding_mask.shape[0], 1, dtype=torch.bool, device=x.device
        )
        result, _ = self.attn(
            x_cls,
            context,
            context,
            key_padding_mask=torch.cat((cls_mask, padding_mask), dim=1),
            need_weights=False,
        )
        return x_cls + result


class _Transformer(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        dimension = 16
        heads = int(config["num_heads"])
        self.pair_extra_dim = int(config.get("pair_extra_dim", 0))
        self.use_amp = False
        self.pair_embed = _PairEmbed(heads)
        self.input = torch.nn.Linear(17, dimension)
        self.blocks = torch.nn.ModuleList(
            [_ParticleBlock(dimension, heads) for _ in range(8)]
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dimension))
        self.cls_blocks = torch.nn.ModuleList(
            [_ClassBlock(dimension, heads) for _ in range(2)]
        )
        self.norm = torch.nn.LayerNorm(dimension)
        self.fc = torch.nn.Linear(dimension, 10)

    def embed(self, features):
        return self.input(features.transpose(1, 2))

    def forward(self, features, v=None, mask=None, uu=None):
        x = self.embed(features)
        padding = ~mask[:, 0].bool()
        bias = self.pair_embed(v, uu=uu, mask=mask)
        for block in self.blocks:
            x = block(x, padding_mask=padding, attn_mask=bias)
        cls = self.cls_token.expand(x.shape[0], 1, -1)
        for block in self.cls_blocks:
            cls = block(x, x_cls=cls, padding_mask=padding)
        return self.fc(self.norm(cls).squeeze(1))


def _weaver():
    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat([base + index for index in range(num_outputs)], dim=1)

    return SimpleNamespace(
        ParticleTransformer=lambda **config: _Transformer(**config),
        pairwise_lv_fts=pairwise,
    )


def _batch():
    torch.manual_seed(31)
    mask = torch.tensor(
        [[[True, True, True, False]], [[True, True, False, False]]]
    )
    return {
        "points": torch.zeros(2, 2, 4),
        "features": torch.randn(2, 17, 4).masked_fill(~mask, 0),
        "lorentz_vectors": torch.randn(2, 4, 4).masked_fill(~mask, 0),
        "mask": mask,
    }


def test_split_forward_disabled_and_tapped_paths_preserve_exact_model() -> None:
    torch.manual_seed(17)
    model = HBaseParticleTransformer(weaver_module=_weaver()).eval()
    batch = _batch()
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    ordinary = model(**batch)
    disabled = model.forward_with_taps(**batch, capture=()).logits
    tapped = model.forward_with_taps(**batch)
    torch.testing.assert_close(disabled, ordinary, atol=0, rtol=0)
    torch.testing.assert_close(tapped.logits, ordinary, atol=0, rtol=0)
    assert list(tapped.states) == ["TAP_EARLY", "TAP_MID", "TAP_LATE"]
    assert [tapped.states[name].shape for name in tapped.states] == [
        (2, 4, 16),
        (2, 4, 16),
        (2, 4, 16),
    ]
    assert torch.equal(tapped.masks["TAP_MID"], batch["mask"][:, 0])
    assert list(model.state_dict()) == list(state_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], atol=0, rtol=0)
    changed = model.forward_with_taps(
        **batch,
        post_mid_transform=lambda state, mask: state.masked_fill(
            mask.unsqueeze(-1), 0
        ),
    )
    assert not torch.equal(changed.logits, ordinary)
    contract = split_forward_contract()
    assert contract["taps"] == {"TAP_EARLY": 2, "TAP_MID": 4, "TAP_LATE": 8}
    assert contract["disabled_path"]["state_dictionary_changed"] is False


def test_split_forward_preserves_input_and_parameter_gradients() -> None:
    torch.manual_seed(29)
    model = HBaseParticleTransformer(weaver_module=_weaver()).eval()
    batch = _batch()
    feature = batch["features"].clone().requires_grad_(True)
    ordinary = model(**{**batch, "features": feature})
    ordinary.sum().backward()
    expected_input = feature.grad.clone()
    expected_parameter = {
        name: None if value.grad is None else value.grad.clone()
        for name, value in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    feature = batch["features"].clone().requires_grad_(True)
    tapped = model.forward_with_taps(**{**batch, "features": feature})
    tapped.logits.sum().backward()
    torch.testing.assert_close(feature.grad, expected_input, atol=0, rtol=0)
    for name, value in model.named_parameters():
        expected = expected_parameter[name]
        assert (value.grad is None) == (expected is None)
        if expected is not None:
            torch.testing.assert_close(value.grad, expected, atol=0, rtol=0)


def test_registered_heads_masks_heteroscedasticity_and_pair_symmetry() -> None:
    torch.manual_seed(41)
    states = torch.randn(2, 5, 128)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]]
    )
    global_head = GlobalTargetHead(
        6, availability_groups=2, heteroscedastic=True
    )
    result = global_head(states, mask)
    assert result["mean"].shape == (2, 6)
    assert result["availability_logits"].shape == (2, 2)
    assert bool((result["log_variance"] >= -8).all())
    assert bool((result["log_variance"] <= 5).all())
    nll = heteroscedastic_nll(
        result["mean"],
        result["log_variance"],
        torch.zeros_like(result["mean"]),
        torch.ones_like(result["mean"], dtype=torch.bool),
    )
    assert torch.isfinite(nll)
    particle = ParticleTargetHead(128, 3)(states, mask)
    assert particle.shape == (2, 5, 3)
    assert int(particle[~mask].count_nonzero()) == 0
    pair, pair_mask = PairTargetHead(128, 4, symmetric=True)(states, mask)
    torch.testing.assert_close(pair, pair.transpose(1, 2), atol=0, rtol=0)
    assert not bool(pair_mask.diagonal(dim1=1, dim2=2).any())
    assert target_head_registry()["pair_sampling"]["rng_library_used"] is False


def test_pair_sampling_is_sha256_deterministic_stratified_and_capped() -> None:
    pair_ids = [f"{left}:{right}" for left in range(50) for right in range(50) if left != right]
    positive = [index % 3 == 0 for index in range(len(pair_ids))]
    first = deterministic_pair_indices(
        epoch=2,
        identity="jet-9",
        target_id="T_HLT_REGION_PAIR_8",
        pair_ids=pair_ids,
        positive=positive,
    )
    second = deterministic_pair_indices(
        epoch=2,
        identity="jet-9",
        target_id="T_HLT_REGION_PAIR_8",
        pair_ids=pair_ids,
        positive=positive,
    )
    assert first == second
    assert sum(positive[index] for index in first) == 512
    assert sum(not positive[index] for index in first) == 512
    continuous = deterministic_pair_indices(
        epoch=2,
        identity="jet-9",
        target_id="T_HLT_TRACK_PAIR_13",
        pair_ids=pair_ids,
    )
    assert len(continuous) == 1024


def _target_registry():
    target_ids = [
        "T_OFFLINE_JET_10",
        "T_OFFLINE_COMPOSITION_16",
        "T_OFFLINE_TRACK_32",
        "T_OFFLINE_DENSITY_22",
        "T_OFFLINE_CA_TREE_26",
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
        *[f"T_OFFLINE_RELATION_{name}" for name in ("BASE4", "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION")],
        "T_OFFLINE_LOGITS_O_BASE",
        "T_OFFLINE_LOGITS_O_FULLREL",
        "T_OFFLINE_POOLED_LATENT",
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    ]
    return with_content_hash({
        "contract": "hosd_structure_target_registry_v1",
        "schema_version": 1,
        "targets": [
            {"target_id": value, "executable_current_source": True}
            for value in target_ids
        ],
    })


def test_exact_baselines_and_complete_bounded_probe_plan() -> None:
    baseline = build_baseline_registry(source=SOURCE)
    assert baseline["baseline_order"] == [
        "H_BASE",
        "H_BASE_BEAM_BUDGET",
        "H_BASE_LONG",
        "H_PARTICLENET",
        "H_KD_LOGIT_O_BASE",
        "H_KD_LOGIT_O_FULLREL",
        "H_NATIVE_REL_AUX",
    ]
    beam = next(
        row
        for row in baseline["baselines"]
        if row["baseline_id"] == "H_BASE_BEAM_BUDGET"
    )
    assert beam["epochs"] == 5
    assert beam["selection_eligible"] is False
    assert [row["epochs"] for row in baseline["baselines"]] == [
        40,
        5,
        80,
        40,
        40,
        40,
        40,
    ]
    pn = particle_net_config()
    assert pn["conv_params"] == [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ]
    assert pn["fc_params"] == [(256, 0.1)]
    plan = build_stage_c_plan(
        campaign_spec_sha256="c" * 64,
        target_registry=_target_registry(),
        baseline_registry=baseline,
        source=SOURCE,
    )
    assert len(plan["baseline_rows"]) == 7
    assert plan["probe_row_count"] == 157
    assert plan["probe_row_count"] <= 162
    assert len(plan["probe_target_order"]) == 18
    assert all(not row["selection_eligible"] for row in plan["probe_rows"])
    assert plan["all_targets_continue_to_stage_d"]
    assert not plan["performance_can_cancel_or_omit"]


def test_kd_probes_statistics_and_metrics_are_numerically_defined() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    labels = torch.tensor([0, 1])
    loss, pieces = classification_kd_loss(student, labels, teacher)
    assert pieces["temperature_2_kl"] is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    target = np.asarray([[1.0, 2.0], [2.0, 4.0], [9.0, 8.0], [8.0, 7.0]])
    mask = np.ones_like(target, dtype=bool)
    classes = np.asarray([0, 0, 1, 1])
    stats = statistical_references(
        target, mask, classes, target_kind="continuous"
    )
    assert stats["P_PRIOR"] == [5.0, 5.5]
    assert stats["P_CLASS_CONDITIONAL_ORACLE"]["0"] == [1.5, 3.0]
    metrics = continuous_probe_metrics(target + 0.5, target, mask)
    assert metrics["normalized_mae"] == pytest.approx(0.5)
    assert metrics["valid_target_coverage"] == 1.0
    teacher_metrics = teacher_probe_metrics(target, target)
    assert teacher_metrics["temperature_2_kl"] == pytest.approx(0)
    assert teacher_metrics["top1_agreement"] == 1.0
    latent = latent_probe_metrics(target, target, mask)
    assert latent["normalized_rmse"] == 0
    assert latent["linear_cka"] == pytest.approx(1)
    pair_target = np.zeros((2, 3, 3, 4), dtype=np.float32)
    pair_mask = np.ones_like(pair_target, dtype=bool)
    diagonal = np.eye(3, dtype=bool)
    pair_mask[:, diagonal] = False
    pair_target[..., 0] = np.asarray(
        [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]] * 2
    )
    pair = pair_probe_metrics(
        pair_target, pair_target, pair_mask, binary_channels=1
    )
    assert len(pair["binary_channels"]) == 1
    assert pair["continuous_channels"]["normalized_rmse"] == 0
    linear = build_tap_probe(
        probe_kind="P_LINEAR",
        tap="TAP_EARLY",
        input_dimension=128,
        target_dimension=5,
    )
    assert linear(torch.randn(2, 3, 128), torch.ones(2, 3, dtype=torch.bool))["value"].shape == (2, 5)
    HOSDTrainingProtocol().validate()
    HOSDTrainingProtocol(maximum_epochs=80).validate()


def test_stage_c_scripts_and_authoritative_parity_entrypoint_exist() -> None:
    for name in (
        "train_hosd_baseline.py",
        "train_hosd_probe.py",
        "aggregate_hosd_predictability.py",
        "build_hosd_probe_taps.py",
        "materialize_hosd_probe_inputs.py",
        "materialize_hosd_pair_probe_inputs.py",
        "validate_hosd_weaver_parity.py",
    ):
        text = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "--campaign-root" in text or name == "validate_hosd_weaver_parity.py"
    parity = (REPO_ROOT / "scripts" / "validate_hosd_weaver_parity.py").read_text(
        encoding="utf-8"
    )
    assert "atol=1e-6" in parity
    assert "atol=2e-6" in parity
    assert "set_autocast_enabled" in parity


def test_probe_tap_loader_resolves_authenticated_design_select_subrole(
    tmp_path: Path, monkeypatch
) -> None:
    script = REPO_ROOT / "scripts" / "build_hosd_probe_taps.py"
    spec = importlib.util.spec_from_file_location("hosd_probe_tap_runtime", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source_identities = np.asarray(["jet-a", "jet-b", "jet-c"])
    arrays = {"identities": source_identities}
    metadata = {"logical_role": "val_design"}
    monkeypatch.setattr(
        module, "load_hlt_v3_cache", lambda path: (arrays, metadata)
    )
    labels_path = tmp_path / "design_select.npz"
    np.savez(
        labels_path,
        identities=np.asarray(["jet-c", "jet-a"]),
        labels=np.asarray([2, 0], dtype=np.int64),
    )
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(module, "NativeHLTExpertDataset", fake_dataset)
    module._dataset(
        {0: tmp_path / "val_design_cache"},
        labels_path,
        "design_select",
        realization_policy="R_FIXED",
    )

    assert captured["logical_role"] == "design_select"
    assert captured["source_logical_role"] == "val_design"
    assert captured["identities"] == ("jet-c", "jet-a")
    assert captured["source_indices_by_replica"][0].tolist() == [2, 0]


def test_miniature_fixed_budget_baseline_trainer_never_stops_on_performance(
    tmp_path: Path,
) -> None:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            generator = torch.Generator().manual_seed(7)
            self.features = torch.randn(20, 17, 4, generator=generator)
            self.labels = torch.arange(20) % 10
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(int(epoch))

        def __len__(self):
            return 20

        def __getitem__(self, index):
            return {
                "points": torch.zeros(2, 4),
                "features": self.features[index],
                "vectors": torch.ones(4, 4),
                "mask": torch.ones(1, 4, dtype=torch.bool),
                "labels": self.labels[index],
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(17, 10)

        def forward(self, points, features, vectors, mask):
            del points, vectors
            pooled = (
                features * mask.float()
            ).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            return self.output(pooled)

    train_data, val_data = Dataset(), Dataset()
    train = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=False)
    val = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=False)
    completion = train_stage_c_baseline(
        model=Model(),
        train_loader=train,
        val_stop_loader=val,
        output_dir=tmp_path / "baseline",
        baseline_id="H_BASE",
        seed=101,
        baseline_registry_sha256="a" * 64,
        campaign_spec_sha256="b" * 64,
        lineage_hashes={"target_registry": "c" * 64},
        protocol=HOSDTrainingProtocol(
            maximum_epochs=2, campaign_profile="miniature_test"
        ),
        device="cpu",
        source=SOURCE,
    )
    assert completion["epochs_completed"] == 2
    assert completion["optimizer_updates_completed"] == 2
    assert completion["early_stopping"] is False
    assert completion["performance_based_termination"] is False
    assert train_data.epochs == [1, 2]
    assert not (tmp_path / "baseline" / "last.pt").exists()
    checkpoint = torch.load(
        tmp_path / "baseline" / "best_model_val.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["source"] == SOURCE
    assert checkpoint["lineage_hashes"] == {"target_registry": "c" * 64}
