"""New factorial cells, original preprocessing, lineage and restart regressions."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_relational_part_offline_ablations as D
from teacher_logit_reco.relational_part import offline_ablation_data as data
from teacher_logit_reco.relational_part.offline_ablations import MODEL_SPECS, SharedBiasEdgeValue, build_ablation_model
from teacher_logit_reco.relational_part.model import RelationalParticleTransformer
from teacher_logit_reco.relational_part.train import TrainingConfig
from teacher_logit_reco.relational_part.ca_tree import pack_tree_shard
from teacher_logit_reco.relational_part.region_tree import tree_content_sha256
from teacher_logit_reco.relational_part.data import RelationalJetDataset, make_relational_loader
from tests.test_relational_particle_transformer_step5 import _artifacts
from tests.test_relational_particle_transformer_step6 import (
    _FakeArchitectureTransformer, _FakePairEmbed, _CustomWeaverAttention,
)


class Trimmer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self._counter = 6

    def forward(self, x, v, mask, uu):
        if self.enabled:
            self._counter += 1
            if self.training:
                order = torch.rand_like(mask.float()).masked_fill(~mask, -1).argsort(-1, descending=True)
                x = x.gather(-1, order.expand_as(x))
                v = v.gather(-1, order.expand_as(v))
                mask = mask.gather(-1, order)
                uu = uu.gather(-1, order.unsqueeze(-2).expand_as(uu))
                uu = uu.gather(-2, order.unsqueeze(-1).expand_as(uu))
            count = int(mask.sum(-1).max())
            if self.training:
                count = max(1, count - random.choice((0, 1)))
            x, v, mask, uu = x[..., :count], v[..., :count], mask[..., :count], uu[..., :count, :count]
        return x, v, mask, uu


class Transformer(_FakeArchitectureTransformer):
    def __init__(self, *, custom=False, sparse=False, **config):
        config.setdefault("pair_extra_dim", 0)
        super().__init__(**{**config, "pair_extra_dim": config["pair_extra_dim"] or 4})
        self.use_amp = False
        self.pair_extra_dim = config.get("pair_extra_dim", 0)
        pair = _FakePairEmbed(self.pair_extra_dim or 4, config["num_heads"])
        pair.out_dim, pair.remove_self_pair = config["num_heads"], False
        pair.sparse_eval = (sparse, sparse)
        if not self.pair_extra_dim:
            pair.pairwise_lv_dim, pair.pairwise_input_dim = 4, 0
            pair.is_symmetric = True
            pair.embed = pair.fts_embed
            del pair.fts_embed
        self.pair_embed = pair
        self.trimmer = Trimmer()
        self.custom = custom
        if custom:
            self.blocks = torch.nn.ModuleList([CustomBlock() for _ in range(8)])

    def forward(self, x, v=None, mask=None, uu=None):
        x, v, mask, uu = self.trimmer(x, v, mask, uu)
        bias = self.pair_embed(v, uu=uu, mask=mask)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, padding_mask=~mask[:, 0], attn_mask=bias)
        x_cls = self.cls_token.expand(len(x), -1, -1)
        for block in self.cls_blocks:
            x_cls = block(x, x_cls=x_cls, padding_mask=~mask[:, 0])
        return self.fc(self.norm(x_cls).squeeze(1))


class CustomBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _CustomWeaverAttention(16, 8)
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, x, padding_mask=None, attn_mask=None):
        update, _ = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        return self.norm(x + update)


def weaver(custom=False, sparse=False):
    from types import ModuleType
    def pairwise(xi, xj, num_outputs=4):
        return torch.cat([xi[:, :1] + xj[:, :1] + i for i in range(num_outputs)], 1)
    module = ModuleType("fake_installed_weaver")
    module.ParticleTransformer = lambda **cfg: Transformer(custom=custom, sparse=sparse, **cfg)
    module.pairwise_lv_fts = pairwise
    return module


@pytest.fixture
def batch_and_normalizers():
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
    from teacher_logit_reco.relational_part.region_tree import build_reference_tree
    raw, mask, _, _, trees, _, normalizer, region = _artifacts()
    raw = np.pad(raw, ((0, 0), (0, 2), (0, 0)))
    mask = np.pad(mask, ((0, 0), (0, 2)))
    for i, count in enumerate((6, 5, 2, 1)):
        mask[i, count:] = False
    raw[~mask] = 0
    inputs = build_particle_transformer_inputs_from_tokens(raw, mask, source_view="offline")
    trees = [build_reference_tree(inputs.pf_vectors[i].T, raw[i], mask[i]) for i in range(4)]
    batch = {"points": torch.from_numpy(inputs.pf_points), "features": torch.from_numpy(inputs.pf_features),
             "lorentz_vectors": torch.from_numpy(inputs.pf_vectors), "mask": torch.from_numpy(inputs.pf_mask).bool(),
             "raw_tokens": torch.from_numpy(raw), "region_trees": trees}
    return batch, normalizer, region


@pytest.mark.parametrize("selected,custom,sparse", [(False, False, False), (False, True, True), (True, False, False), (True, True, False)])
def test_shared_zero_ev_logits_gradients_bn_and_active_trim(batch_and_normalizers, selected, custom, sparse):
    batch, norm, region = batch_and_normalizers
    module = weaver(custom, sparse)
    reference = (build_ablation_model("OFF_RPT_SELECTED_SHARED", normalization_artifact=norm,
                 region_normalization_artifact=region, weaver_module=module) if selected
                 else RelationalParticleTransformer(weaver_module=module))
    result = D.zero_message_parity(reference, batch, device="cpu")
    assert result["batchnorm_buffers_and_trimmer_counters_exact"]
    assert reference.mod.trimmer.enabled
    assert reference.mod.trimmer._counter == 7


@pytest.mark.parametrize("run_id", list(MODEL_SPECS))
@pytest.mark.parametrize("custom", [False, True])
def test_four_new_factories_gradients_and_strict_reload(batch_and_normalizers, run_id, custom):
    from teacher_logit_reco.relational_part.evaluation import model_forward
    batch, norm, region = batch_and_normalizers
    model = build_ablation_model(run_id, normalization_artifact=norm, region_normalization_artifact=region,
                                 weaver_module=weaver(custom)).eval()
    output = model_forward(model, batch)
    assert output.shape == (4, 10) and torch.isfinite(output).all()
    torch.nn.functional.cross_entropy(output, torch.arange(4)).backward()
    edges = [p for name, p in model.named_parameters() if name.endswith("edge_projection")]
    assert len(edges) == (8 if MODEL_SPECS[run_id]["edge_value"] else 0)
    assert len({p.data_ptr() for p in edges}) == len(edges)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in edges)
    restored = build_ablation_model(run_id, normalization_artifact=norm, region_normalization_artifact=region,
                                    weaver_module=weaver(custom)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model_forward(restored, batch), output, atol=0, rtol=0)


@pytest.mark.parametrize("custom", [False, True])
def test_shared_ev_padding_invariance_and_clears_on_failure(custom):
    reference = RelationalParticleTransformer(weaver_module=weaver(custom))
    model = SharedBiasEdgeValue(reference).eval()
    mask = torch.tensor([[[1, 1, 1, 0, 0]], [[1, 0, 0, 0, 0]]], dtype=torch.bool)
    batch = {"points": torch.zeros(2, 2, 5), "features": torch.randn(2, 17, 5),
             "lorentz_vectors": torch.randn(2, 4, 5), "mask": mask}
    wanted = model(**batch)
    changed = {k: v.clone() for k, v in batch.items()}
    for key in ("features", "lorentz_vectors"):
        changed[key].masked_fill_(~mask, 10000.)
    torch.testing.assert_close(model(**changed), wanted, atol=0, rtol=0)
    assert all(edge._relation_stem is None for edge in model.edge_attention)
    def failure(*args):
        raise RuntimeError("synthetic block failure")
    hook = reference.mod.blocks[0].register_forward_pre_hook(failure)
    with pytest.raises(RuntimeError, match="synthetic"):
        model(**batch)
    hook.remove()
    assert all(edge._relation_stem is None for edge in model.edge_attention)


def test_tree_array_covers_original_splits_once():
    tasks = data.tree_tasks()
    assert len(tasks) == 126
    for split, count in data.SPLITS.items():
        rows = [t for t in tasks if t["split"] == split]
        assert rows[0]["start"] == 0 and rows[-1]["stop"] == count
        assert all(a["stop"] == b["start"] for a, b in zip(rows, rows[1:]))
        assert [r["shard"] for r in rows] == list(range(len(rows)))
    assert tasks[-1]["stop"] - tasks[-1]["start"] == 5000


def test_mmap_and_lazy_trees_match_original_sampler_and_collator(tmp_path, monkeypatch):
    raw, mask, _, ids, trees, *_ = _artifacts()
    view = SimpleNamespace(tokens=raw, mask=mask, labels=np.arange(4), jet_ids=ids,
                           split="model_train", metadata={"view": "offline"})
    files, arrays = data.view_arrays(view)
    from jetclass_fresh.hlt_cache import hash_arrays
    from teacher_logit_reco.architecture_view_part.train import save_cached_offline_view
    historical = save_cached_offline_view(view, tmp_path / "historical")
    assert hash_arrays(arrays) == historical["offline_content_hash"]
    assert files == historical["jet_files"]
    cache = tmp_path / "inputs/cache/model_train"
    cache.mkdir(parents=True)
    hashes = {}
    for name, values in arrays.items():
        np.save(cache / (name + ".npy"), values, allow_pickle=False)
        hashes[name + ".npy"] = D.F.digest(cache / (name + ".npy"))
    receipt = D.F.hashed({"files": hashes, "jet_files": files})
    D.F.write_json(cache / "receipt.json", receipt)
    tree_path = tmp_path / "inputs/trees/model_train/shard_000.npz"
    tree_path.parent.mkdir(parents=True)
    packed = pack_tree_shard(trees, ids)
    np.savez_compressed(tree_path, **packed)
    tree_receipt = D.F.hashed({"npz_sha256": D.F.digest(tree_path)})
    D.F.write_json(tree_path.with_suffix(".json"), tree_receipt)
    monkeypatch.setattr(data, "SPLITS", {"model_train": 4})
    binding = {"splits": {"model_train": {"cache": receipt["content_hash"], "trees": [tree_receipt["content_hash"]]}}}
    dataset = data.AblationDataset(tmp_path, "model_train", uses_region=True, binding=binding)
    assert isinstance(dataset.arrays["tokens"], np.memmap)
    for i, tree in enumerate(trees):
        assert tree_content_sha256(dataset[i]["region_tree"]) == tree_content_sha256(tree)
    original = RelationalJetDataset(view, region_trees=trees)
    for training in (True, False):
        left = make_relational_loader(original, seed=303, training=training)
        right = make_relational_loader(dataset, seed=303, training=training)
        if training:
            left.sampler.set_epoch(3)
            right.sampler.set_epoch(3)
        a, b = next(iter(left)), next(iter(right))
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        assert a["event_identities"] == b["event_identities"]
    with tree_path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Tree shard"):
        data.AblationDataset(tmp_path, "model_train", uses_region=True, binding=binding)
    with (cache / "labels.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Cache file"):
        data.open_arrays(cache, receipt)
    with pytest.raises(ValueError, match="Unsafe"):
        data.open_arrays(cache, {"files": {"../escape.npy": "a"}})


def _workflow(tmp_path, monkeypatch):
    spec = {"content_hash": "a" * 64, "parent_campaign_sha256": "b" * 64,
            "test_data": "/read-only-test", "models": MODEL_SPECS}
    determinism = {"content_hash": "d" * 64}
    parents = {"registry/global_determinism.json": determinism,
               "inputs/offline_cache_binding.json": {"campaign_sha256": spec["parent_campaign_sha256"]},
               "inputs/relation_normalization.json": {"content_hash": "e" * 64},
               "inputs/region_normalization.json": {"content_hash": "f" * 64}}
    for seed in D.F.SEEDS:
        parents[f"runs/OFF_RPT_BASE/seed_{seed}/checkpoint_registration.json"] = {
            "training_contract_sha256": TrainingConfig(seed=seed).artifact(global_determinism_sha256=determinism["content_hash"])["content_hash"],
            "precision_mode": "bf16"}
    monkeypatch.setattr(D, "load_campaign", lambda *args: spec)
    monkeypatch.setattr(D, "parent_json", lambda _, name: parents[name])
    monkeypatch.setattr(D, "backend_for", lambda _: None)
    args = argparse.Namespace(output=tmp_path)
    D.prepare(args)
    D.prepare(args)  # Immutable idempotent preparation.
    tasks = D.registered_tasks(tmp_path, spec)["tasks"]
    assert len(tasks) == 12 and all(t["run_id"] not in D.F.MODELS for t in tasks)
    registry = D.F.read_json(tmp_path / "training_tasks.json")
    binding = D.F.hashed({"campaign_sha256": spec["content_hash"]})
    D.F.write_json(tmp_path / "data_binding.json", binding)
    for task in tasks:
        directory = tmp_path / "runs" / task["run_id"] / f"seed_{task['seed']}"
        directory.mkdir(parents=True)
        (directory / "best_model_val.pt").write_bytes(b"fake checkpoint fixture")
        metrics = D.F.hashed({"accuracy": 0.01})  # Terrible results must still lock.
        D.F.write_json(directory / "val_select_metrics.json", metrics)
        reg = D.F.hashed({"run_id": task["run_id"], "seed": task["seed"],
            "model_contract_sha256": task["model_contract_sha256"], "run_registry_sha256": registry["content_hash"],
            "lineage_hashes": {"campaign_spec": spec["content_hash"], "data_binding": binding["content_hash"],
                               "relation_normalization": "e" * 64, "region_normalization": "f" * 64},
            "training_contract_sha256": TrainingConfig(seed=task["seed"]).artifact(global_determinism_sha256=determinism["content_hash"])["content_hash"],
            "checkpoint_sha256": D.F.digest(directory / "best_model_val.pt"), "offline_tagger_inference": True,
            "precision_mode": "bf16", "val_select_used_for_checkpoint_selection": False,
            "val_select_metrics_sha256": metrics["content_hash"], "selected_epoch": 15})
        D.F.write_json(directory / "checkpoint_registration.json", reg)
    old = D.F.hashed({"files": [{"name": name} for name in sorted(D.F.NAMES)], "event_count": 20_000_000,
                     "class_order": list(D.F.CLASS_NAMES), "chunk_size": 10000, "batch_size": 64,
                     "precision": "fp32", "metric_policy": {}, "rejection_targets": list(D.F.TARGETS)})
    D.F.write_json(tmp_path / "reference_plan.json", old)
    D.lock(args)
    D.lock(args)
    return spec, args


def test_prepare_lock_all_twelve_even_when_accuracy_is_bad(tmp_path, monkeypatch):
    spec, args = _workflow(tmp_path, monkeypatch)
    plan = D.test_plan(tmp_path, spec)
    assert len(plan["tasks"]) == 12 and len(plan["assignments"]) == 20
    assert plan["used_for_model_selection"] is False
    locked = D.F.read_json(tmp_path / "selection/locked_models.json")
    assert locked["all_twelve_included"] and not locked["performance_gate"]
    assert {r["validation"]["accuracy"] for r in locked["rows"]} == {0.01}
    # Authenticated completed training reuse must not construct/train any model.
    D.F.write_json(tmp_path / "runtime_validation.json", D.F.hashed({"campaign_sha256": spec["content_hash"], "passed": True}))
    monkeypatch.setattr(D, "model_for", lambda *a: pytest.fail("completed training was restarted"))
    args.task_index = 0
    # Windows lacks POSIX flock; replace only the lock for this isolated fixture.
    from contextlib import nullcontext
    monkeypatch.setattr(D.F, "exclusive", lambda *a: nullcontext())
    D.train(args)


@pytest.mark.parametrize("mutation", ["tasks", "rows", "precision", "assignment", "checkpoint"])
def test_test_lock_rejects_changed_lineage(tmp_path, monkeypatch, mutation):
    spec, _ = _workflow(tmp_path, monkeypatch)
    path = tmp_path / "test/evaluation_plan.json"
    value = D.F.read_json(path)
    if mutation == "tasks":
        value["tasks"][0] = value["tasks"][1]
    elif mutation == "precision":
        value["precision"] = "bf16"
    elif mutation == "assignment":
        value["assignments"][0] = []
    elif mutation == "rows":
        path = tmp_path / "selection/locked_models.json"
        value = D.F.read_json(path)
        value["rows"] = value["rows"][:-1]
    else:
        (tmp_path / value["tasks"][0]["checkpoint"]).write_bytes(b"different")
    if mutation != "checkpoint":
        value.pop("content_hash")
        path.write_text(__import__("json").dumps(D.F.hashed(value)))
    with pytest.raises(ValueError, match="differs|changed"):
        D.test_plan(tmp_path, spec)


def test_resource_budget_fails_before_tree_or_inference_writes(tmp_path, monkeypatch):
    spec = {"prediction_bytes_uncompressed": 9_600_000_000, "headroom_bytes": 6 * 1024**3}
    monkeypatch.setattr(D, "load_campaign", lambda *a: spec)
    for split in data.SPLITS:
        D.F.write_json(tmp_path / "inputs/cache" / split / "receipt.json", D.F.hashed({"tree_storage_bound_bytes": 1000}))
    monkeypatch.setattr(D.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 1024**3))
    D.storage(SimpleNamespace(output=tmp_path))
    monkeypatch.setattr(D.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="no data was deleted"):
        D.storage(SimpleNamespace(output=tmp_path))


def test_combined_report_reuses_old_results_and_all_factorial_contrasts(tmp_path, monkeypatch):
    spec, args = _workflow(tmp_path, monkeypatch)
    plan = D.test_plan(tmp_path, spec)
    def metric(accuracy, saturated=False):
        return D.F.hashed({"accuracy": accuracy, "qcd_signal_rejection": {
            signal: {str(t): {"background_rejection": None if saturated else 100.} for t in D.F.TARGETS}
            for signal in D.F.CLASS_NAMES[1:]}})
    old_results = [{"task": {"run_id": name, "seed": seed}, "metrics": metric(.8 + i*.001)}
                   for i, name in enumerate(D.F.MODELS) for seed in D.F.SEEDS]
    D.F.write_json(tmp_path / "reference_report.json", D.F.hashed({"per_model_results": old_results}))
    for i, task in enumerate(plan["tasks"]):
        D.F.write_json(tmp_path / "test/metrics" / f"model_{i:02d}.json", D.F.hashed({
            "plan_sha256": plan["content_hash"], "task": task, "file_completion_hashes": {"old": [], "new": []},
            "metrics": metric(.81 + (i//3)*.001, saturated=i == 3), "paired_accuracy": {}}))
    D.report(args)
    D.report(args)
    report = D.F.read_json(tmp_path / "reports/offline_factorial_report.json")
    assert len(report["models"]) == 8 and len(report["per_model_results"]) == 24
    assert report["per_model_results"][:12] == old_results
    assert report["contrasts"]["standard"]["EV_at_shared"]["mean_accuracy_difference"] == pytest.approx(.011)
    assert report["contrasts"]["standard"]["EV_at_layerwise"]["mean_accuracy_difference"] == pytest.approx(-.009)
    assert report["contrasts"]["standard"]["EV_by_layerwise_interaction"]["mean_accuracy_difference"] == pytest.approx(-.02)
    assert len(report["contrasts"]["selected_minus_standard"]) == 4
    assert report["models"]["OFF_RPT_BASE_SHARED_EDGEVALUE"]["qcd_signal_rejection"]["Hbb"]["0.5"]["mean"] is None
    markdown = (tmp_path / "reports/offline_factorial_report.md").read_text()
    assert "50% signal efficiency" in markdown and "SAT" in markdown and "AFTER inspection" in markdown


def test_bootstrap_freezes_old_runtime_and_does_not_touch_parents(tmp_path, monkeypatch):
    parent, previous, old_source, test = [tmp_path / p for p in ("parent", "previous", "old-source", "test")]
    for path in (parent, previous, old_source, test):
        path.mkdir()
    (old_source / "jetclass_fixed_hlt.py").write_text("# historical runtime\n")
    old_sources = D.F.source_snapshot(old_source)
    campaign = D.F.hashed({"test": "parent"})
    D.F.write_json(parent / "campaign_spec.json", campaign)
    for name in D.EXTRA_PARENTS:
        D.F.write_json(parent / name, D.F.hashed({"test": name}))
    tasks = [{"run_id": name, "seed": seed} for name in D.F.MODELS for seed in D.F.SEEDS]
    plan = D.F.hashed({"contract": D.F.VERSION, "parent_campaign_sha256": campaign["content_hash"],
        "tasks": tasks, "parent_artifacts": {"campaign_spec.json": D.F.digest(parent / "campaign_spec.json")},
            "files": [{"name": "synthetic.root", "entries": 1, "sha256": "1" * 64}], "chunk_size": 1,
        "source_root": str(old_source), "source_files": old_sources, "environment": {"test": "cpu"}})
    D.F.write_json(previous / "evaluation_plan.json", plan)
    (test / "synthetic.root").write_bytes(b"read only input")
    path = D.F.chunk_path(previous, 0, 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"read only logits")
    sidecar = D.F.hashed({"metadata": D.F.chunk_metadata(plan, 0, 0, 1), "npz_sha256": D.F.digest(path)})
    D.F.write_json(path.with_suffix(".json"), sidecar)
    done = D.F.hashed({"contract": D.F.VERSION + "_file_complete", "plan_sha256": plan["content_hash"],
                      "file_index": 0, "chunks": [sidecar["content_hash"]]})
    D.F.write_json(previous / "predictions/file_000/complete.json", done)
    rows = [D.F.hashed({"task": t, "plan_sha256": plan["content_hash"], "metrics": D.F.hashed({"accuracy": .8}),
                        "file_completion_hashes": [done["content_hash"]]}) for t in tasks]
    report = D.F.hashed({"per_model_results": rows, "event_count_per_model": 20_000_000, "plan_sha256": plan["content_hash"]})
    D.F.write_json(previous / "reports/full_test_report.json", report)
    monkeypatch.setattr(D.F, "parent_artifacts", lambda _: (campaign, tasks, plan["parent_artifacts"].copy()))
    monkeypatch.setattr(D.shutil, "disk_usage", lambda _: SimpleNamespace(free=100*1024**3))
    before = {p: D.F.digest(p) for folder in (parent, previous, old_source, test) for p in folder.rglob("*") if p.is_file()}
    args = SimpleNamespace(parent=parent, previous=previous, output=tmp_path / "new", test_data=test, train_data=tmp_path)
    D.bootstrap(args)
    assert before == {p: D.F.digest(p) for p in before}
    assert D.load_campaign(args.output, runtime=False)["followup_after_inspection_of_500k_and_20M_results"]
    # Parent repository commits/pyc are irrelevant; frozen actual source is not.
    (args.output / "source/unrelated.md").write_text("later unrelated work")
    D.load_campaign(args.output, runtime=False)
    with (args.output / "source" / D.NEW_FILES[0]).open("a") as stream:
        stream.write("\n# drift\n")
    with pytest.raises(ValueError, match="source changed"):
        D.load_campaign(args.output, runtime=False)
    with pytest.raises(FileExistsError, match="--resume"):
        D.bootstrap(args)


def test_submission_contract_and_shell_syntax():
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[1]
    submit = root / "sbatch/submit_relational_part_offline_ablations.sh"
    worker = root / "sbatch/run_rpt_offline_ablations.sh"
    text = submit.read_text()
    for flag in ("--array=0-2%1", "--array=\"0-125%", "--array=\"0-11%", "--array=\"0-19%", "--array=0-11%3"):
        assert flag in text
    assert "--kill-on-invalid-dep=yes" in text and "--dependency=\"afterok:" in text
    assert "still live" in text and "flock -n" in text and "--resume" in text
    assert not any(line.lstrip().startswith("rm ") for line in (text + worker.read_text()).splitlines())
    bash = str(Path("C:/Program Files/Git/bin/bash.exe")) if Path("C:/Program Files/Git/bin/bash.exe").exists() else shutil.which("bash")
    if bash:
        for path in (submit, worker):
            checked = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
            assert checked.returncode == 0, checked.stderr
        dry = subprocess.run([bash, str(submit), "--dry-run"], capture_output=True, text=True)
        assert dry.returncode == 0, dry.stderr
        assert "train[12]" in dry.stdout and "infer[20]" in dry.stdout


def test_global_aggregation_pairs_old_and_new_models_and_authenticates_restart(tmp_path, monkeypatch):
    previous = tmp_path / "old-evaluation"
    rows = [{"label": label, "name": f"class_{label}.root", "entries": 10, "sha256": str(label)*64} for label in range(10)]
    old_tasks = [{"run_id": name, "seed": seed} for name in D.F.MODELS for seed in D.F.SEEDS]
    tasks = [{"run_id": name, "seed": seed} for name in MODEL_SPECS for seed in D.F.SEEDS]
    base = {"files": rows, "chunk_size": 10, "event_count": 100}
    old = D.F.hashed({**base, "contract": D.F.VERSION, "tasks": old_tasks})
    plan = D.F.hashed({**base, "contract": D.VERSION + "_test", "tasks": tasks})
    D.F.write_json(tmp_path / "reference_plan.json", old)
    rng = np.random.default_rng(112)
    wanted = []
    old_completions = []
    for index, row in enumerate(rows):
        for root, current in ((previous, old), (tmp_path / "test", plan)):
            path = D.F.chunk_path(root, index, 0)
            path.parent.mkdir(parents=True)
            values = {f"logits_{i:02d}": rng.normal(size=(10, 10)).astype(np.float32) for i in range(12)}
            if current is plan:
                wanted.append(values["logits_00"])
            np.savez_compressed(path, **values)
            metadata = D.F.chunk_metadata(current, index, 0, 10)
            receipt = D.F.hashed({"metadata": metadata, "npz_sha256": D.F.digest(path)})
            D.F.write_json(path.with_suffix(".json"), receipt)
            done = D.F.hashed({"contract": current["contract"] + "_file_complete", "plan_sha256": current["content_hash"],
                              "file_index": index, "chunks": [receipt["content_hash"]]})
            D.F.write_json(root / f"predictions/file_{index:03d}/complete.json", done)
            if current is old:
                old_completions.append(done["content_hash"])
    D.F.write_json(tmp_path / "reference_report.json", D.F.hashed({"per_model_results": [
        {"file_completion_hashes": old_completions} for _ in old_tasks]}))
    monkeypatch.setattr(D, "load_campaign", lambda _: {"previous_evaluation": str(previous)})
    # Bypass ONLY the production 20M-size lock for this 100-event fixture.
    monkeypatch.setattr(D, "test_plan", lambda *a: plan)
    args = SimpleNamespace(output=tmp_path, task_index=0)
    D.aggregate(args)
    D.aggregate(args)
    metric = D.F.read_json(tmp_path / "test/metrics/model_00.json")
    assert metric["metrics"] == D.F.calculate_metrics(np.concatenate(wanted), np.repeat(np.arange(10), 10))
    assert len(metric["paired_accuracy"]) == 7
    assert all(row["sampling_unit"] == "paired_event_identity" for row in metric["paired_accuracy"].values())
    with D.F.chunk_path(previous, 0, 0).open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="bytes"):
        D.aggregate(args)
