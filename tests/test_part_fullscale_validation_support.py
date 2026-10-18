"""Full validation summaries must retain population identity and all seeds."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/part_fullscale_validation_support.py"
SPEC = importlib.util.spec_from_file_location("fullscale_validation_support_tests", PATH)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def records():
    results = []
    for architecture in support.ARCHITECTURES:
        for index, seed in enumerate(support.SEEDS):
            rejection = {}
            for signal, targets in support.REJECTION_TARGETS.items():
                rejection[signal] = {}
                for target in targets:
                    accepted = math.ceil(float(target) * support.CLASS_EVENTS)
                    qcd = ((100, 200, 400) if architecture == "part" else (50, 100, 100))[index]
                    rejection[signal][target] = {
                        "threshold_log_odds": 2.0, "target_signal_efficiency": float(target),
                        "achieved_signal_efficiency": accepted / support.CLASS_EVENTS,
                        "signal_accepted": accepted, "signal_total": support.CLASS_EVENTS,
                        "qcd_accepted": qcd, "qcd_total": support.CLASS_EVENTS,
                        "rejection": support.CLASS_EVENTS / qcd,
                    }
            results.append({
                "task": {"architecture": architecture, "seed": seed},
                "evaluation_sha256": "same-evaluation", "inventory_sha256": "same-validation",
                "checkpoint_sha256": f"{architecture}-{seed}",
                "metrics": {
                    "events": support.EVENTS, "class_support": {name: support.CLASS_EVENTS for name in support.CLASS_NAMES},
                    "accuracy": 0.8 + index * 0.01 + (0.005 if architecture == "part_ev" else 0),
                    "macro_auc_ovo": 0.98, "macro_auc_ovr": 0.985, "qcd_rejection": rejection,
                },
            })
    return results


def test_full_report_uses_matched_seeds_and_distinguishes_percent_from_percentage_points():
    inputs = records()
    unchanged = copy.deepcopy(inputs)
    report = support.build_report(list(reversed(inputs)))
    assert inputs == unchanged
    assert report["split"] == "validation" and report["events_per_checkpoint"] == 5_000_000
    assert report["seeds"] == [101, 202, 303]
    assert report["per_seed_results"] == inputs
    assert report["models"]["part"]["accuracy"]["mean"] == pytest.approx(0.81)
    assert report["models"]["part"]["accuracy"]["seed_std"] == pytest.approx(0.01)
    accuracy = report["comparisons"]["accuracy_gain_percentage_points"]
    assert list(accuracy["per_seed"].values()) == pytest.approx([0.5, 0.5, 0.5])
    assert accuracy["mean"] == pytest.approx(0.5)  # Not 0.625% relative accuracy gain.
    row = report["comparisons"]["qcd_rejection"]["Hbb"]["0.5"]
    assert row["per_seed_ratio"] == {"101": 2, "202": 2, "303": 4}
    assert row["per_seed_gain_percent"] == {"101": 100, "202": 100, "303": 300}
    assert row["mean_paired_gain_percent"] == pytest.approx(500 / 3)
    assert row["ratio_of_means"] == pytest.approx(20_000 / 8_750)
    assert row["ratio_of_means_gain_percent"] == pytest.approx(100 * (20_000 / 8_750 - 1))
    assert row["ratio_of_means_gain_percent"] != row["mean_paired_gain_percent"]
    assert report["models"]["part"]["qcd_rejection"]["Hbb"]["0.5"]["passing_counts"]["303"]["qcd_accepted"] == 400


@pytest.mark.parametrize("index", [0, 3])
def test_zero_qcd_seed_is_not_dropped_from_aggregates_or_gains(index):
    inputs = records()
    row = inputs[index]["metrics"]["qcd_rejection"]["Hbb"]["0.3"]
    row.update(qcd_accepted=0, rejection=None)
    report = support.build_report(inputs)
    model = inputs[index]["task"]["architecture"]
    summary = report["models"][model]["qcd_rejection"]["Hbb"]["0.3"]
    assert summary["mean"] is None and summary["seed_std"] is None
    assert summary["zero_background_seeds"] == 1
    assert summary["passing_counts"]["101"]["qcd_accepted"] == 0
    gains = report["comparisons"]["qcd_rejection"]["Hbb"]["0.3"]
    assert gains["per_seed_gain_percent"]["101"] is None
    assert gains["per_seed_gain_percent"]["202"] == 100
    for key in ("mean_paired_gain_percent", "paired_gain_seed_std_percent", "ratio_of_means", "ratio_of_means_gain_percent"):
        assert gains[key] is None
    markdown = support.markdown_report(report)
    assert "ZB 1/3" in markdown and "undefined" in markdown


@pytest.mark.parametrize("mutation", [
    "missing_seed", "extra_seed", "duplicate", "wrong_model", "wrong_seed", "mixed_evaluation", "mixed_inventory",
    "missing_checkpoint", "missing_metrics", "missing_auc", "wrong_events", "wrong_support", "missing_class",
    "missing_target", "wrong_target", "wrong_qcd_population", "wrong_signal_population", "bad_count",
    "below_signal_target", "bad_achieved", "bad_rejection", "zero_with_finite", "nonzero_with_null",
])
def test_incomplete_or_inconsistent_metrics_are_rejected(mutation):
    rows = records()
    metrics = rows[0]["metrics"]
    rejection = metrics["qcd_rejection"]["Hbb"]["0.5"]
    if mutation == "missing_seed": rows.pop()
    elif mutation == "extra_seed": rows.append(copy.deepcopy(rows[0]))
    elif mutation == "duplicate": rows[1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_model": rows[0]["task"]["architecture"] = "part_plain"
    elif mutation == "wrong_seed": rows[0]["task"]["seed"] = 404
    elif mutation == "mixed_evaluation": rows[0]["evaluation_sha256"] = "other"
    elif mutation == "mixed_inventory": rows[0]["inventory_sha256"] = "other"
    elif mutation == "missing_checkpoint": rows[0].pop("checkpoint_sha256")
    elif mutation == "missing_metrics": rows[0].pop("metrics")
    elif mutation == "missing_auc": metrics.pop("macro_auc_ovo")
    elif mutation == "wrong_events": metrics["events"] = 20_000_000
    elif mutation == "wrong_support": metrics["class_support"]["QCD"] = 2_000_000
    elif mutation == "missing_class": metrics["qcd_rejection"].pop("Tbl")
    elif mutation == "missing_target": metrics["qcd_rejection"]["Hqql"].pop("0.99")
    elif mutation == "wrong_target": rejection["target_signal_efficiency"] = 0.3
    elif mutation == "wrong_qcd_population": rejection["qcd_total"] = 2_000_000
    elif mutation == "wrong_signal_population": rejection["signal_total"] = 2_000_000
    elif mutation == "bad_count": rejection["qcd_accepted"] = -1
    elif mutation == "below_signal_target": rejection["signal_accepted"] = 1
    elif mutation == "bad_achieved": rejection["achieved_signal_efficiency"] = 0.25
    elif mutation == "bad_rejection": rejection["rejection"] += 1
    elif mutation == "zero_with_finite": rejection["qcd_accepted"] = 0
    elif mutation == "nonzero_with_null": rejection["rejection"] = None
    with pytest.raises(ValueError):
        support.build_report(rows)


@pytest.mark.parametrize("field", ["accuracy", "macro_auc_ovo", "macro_auc_ovr"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, None, True])
def test_scalar_metrics_must_be_finite_probabilities(field, value):
    rows = records()
    rows[0]["metrics"][field] = value
    with pytest.raises(ValueError):
        support.build_report(rows)


def test_markdown_describes_validation_and_all_operating_points_without_significance_claims():
    text = support.markdown_report(support.build_report(records()))
    assert "validation results, not final-test results" in text
    assert "selected using training-time validation accuracy" in text
    assert "+0.5000 percentage points" in text
    assert "## 30% signal efficiency" in text and "## 50% signal efficiency" in text
    assert "| Hqql | 99% |" in text and "| Tbl | 99.5% |" in text
    assert "| Hqql | 30% |" not in text and "| Tbl | 50% |" not in text
    assert "mean ParT-EV rejection / mean ParT rejection" in text
    assert "100 / 200 / 400" in text
    assert "No statistical significance is inferred" in text


@pytest.fixture
def loader_mocks(monkeypatch):
    import torch.utils.data
    import weaver.utils.dataset

    calls = {}
    config = SimpleNamespace(_missing_standardization_info=False, weight_name=None)
    record = {"name": "example.root", "path": "/frozen/example.root", "num_entries": 17}
    inventory = {"splits": {"val": {"files": [record]}}}

    def rank_files(split, selected):
        calls["rank_files"] = (split, selected)
        return {"val": [record["path"]]}

    def entries(actual_inventory, split, files):
        assert actual_inventory is inventory
        calls["entries"] = (split, files)
        return {record["path"]: record["num_entries"]}

    def dataset(**kwargs):
        calls["dataset_options"] = kwargs
        return SimpleNamespace(config=config)

    def seeded(*args):
        calls["seeded_args"] = args
        return "seeded frozen dataset"

    def dataloader(actual_dataset, **kwargs):
        calls["loader"] = (actual_dataset, kwargs)
        return "finite loader"

    frozen = SimpleNamespace(_SeededDataset=seeded, _stream_seed=lambda *args: 123,
                             _validated_config=lambda value: Path("frozen/JetClass_full.yaml"),
                             _file_entry_counts=entries, rank_file_dict=rank_files)
    monkeypatch.setitem(sys.modules, "part_ev_fullscale.data", frozen)
    monkeypatch.setattr(weaver.utils.dataset, "SimpleIterDataset", dataset)
    monkeypatch.setattr(torch.utils.data, "DataLoader", dataloader)
    return calls, config, inventory, record


def test_finite_loader_uses_frozen_preprocessing_all_entries_and_keeps_tail(loader_mocks):
    calls, config, inventory, record = loader_mocks
    assert support.finite_validation_loader(inventory, record, batch_size=8) == "finite loader"
    assert calls["dataset_options"] == {
        "file_dict": {"val": [record["path"]]}, "data_config_file": str(Path("frozen/JetClass_full.yaml")),
        "for_training": False, "load_range_and_fraction": ((0, 1), 1), "fetch_by_files": True,
        "fetch_step": 1, "file_fraction": 1, "remake_weights": False, "async_load": False,
        "infinity_mode": False, "in_memory": False, "name": "fullscale_finite_validation",
    }
    seeded = calls["seeded_args"]
    assert seeded[2:] == ({record["path"]: 17}, 123, False)
    assert seeded[0].config is config and seeded[1] == calls["dataset_options"]
    dataset, options = calls["loader"]
    assert dataset == "seeded frozen dataset"
    assert options["batch_size"] == 8 and options["drop_last"] is False
    assert options["num_workers"] == 0 and options["pin_memory"] is True
    assert options["generator"].initial_seed() == 123


@pytest.mark.parametrize("field,value", [("_missing_standardization_info", True), ("weight_name", "weights")])
def test_finite_loader_rejects_nonmanual_or_weighted_preprocessing(loader_mocks, field, value):
    calls, config, inventory, record = loader_mocks
    setattr(config, field, value)
    with pytest.raises(ValueError, match="manual and unweighted"):
        support.finite_validation_loader(inventory, record)
    assert "loader" not in calls


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_loader_rejects_invalid_batch_size_before_importing_dataset(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        support.finite_validation_loader({}, {}, batch_size)


@pytest.mark.parametrize("duplicate", [False, True])
def test_loader_rejects_file_outside_frozen_unique_inventory(loader_mocks, duplicate):
    calls, _, inventory, record = loader_mocks
    if duplicate:
        inventory["splits"]["val"]["files"].append(record)
    else:
        record = {**record, "path": "/different/example.root"}
    with pytest.raises(ValueError, match="exactly once"):
        support.finite_validation_loader(inventory, record)
    assert "dataset_options" not in calls
