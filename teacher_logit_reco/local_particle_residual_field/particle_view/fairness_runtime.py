"""Production runtime for the post-selection Stage-G fairness closure."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from .campaign import _FAIRNESS_CONTROL_IDS, _WINNER_FAMILIES
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .controls import DirectControlCandidate, build_stage_g_control_plan
from .direct_control import (
    _direct_model,
    particle_transformer_parameter_count,
    particle_transformer_semantic_flops,
)
from .ledger import build_fairness_budget_accounting
from .offline_teacher import (
    build_predeclared_direct_control_grid,
    teacher_learning_rate,
)
from .post_target_runtime import _artifact, _task_artifacts, _teacher_from_task
from .registry import validate_particle_view_registry
from .runtime_data import (
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    validate_runtime_data_config,
)
from .selection import build_selected_path_fairness_ledger
from .teacher_train import evaluate_particle_view_teacher


PARTICLE_VIEW_FAIRNESS_FACTORY_CONFIG_CONTRACT = (
    "particle_view_fairness_factory_config_v1"
)
PARTICLE_VIEW_FAIRNESS_INPUT_INDEX_CONTRACT = (
    "particle_view_fairness_input_index_v1"
)
PARTICLE_VIEW_STAGE_G_RESULT_CONTRACT = "particle_view_stage_g_result_v1"
PARTICLE_VIEW_STAGE_G_ALIAS_CONTRACT = "particle_view_stage_g_alias_v1"


def _direct_candidates() -> list[dict[str, Any]]:
    rows = []
    for config in build_predeclared_direct_control_grid()["candidates"]:
        parameters = particle_transformer_parameter_count(config)
        flops = sum(particle_transformer_semantic_flops(config).values())
        rows.append(
            {
                **config,
                "deployed_parameters": parameters,
                "forward_flops": flops,
                "config_sha256": canonical_sha256(config),
            }
        )
    return rows


def build_fairness_input_index(
    *,
    selection: Mapping[str, Any],
    configurations: Mapping[str, Mapping[int, Mapping[str, Mapping[str, str]]]],
    flop_fixture_sha256: str,
    flop_counter_sha256: str,
) -> dict[str, Any]:
    validate_content_hash(
        selection, expected_contract="particle_view_winner_selection_v1"
    )
    required_configurations = {
        selection["selected_privileged_scientific_model"]["configuration_id"],
        selection["selected_pre_stage_g_hlt_deployable_model"][
            "configuration_id"
        ],
    }
    if set(configurations) != required_configurations:
        raise ValueError("fairness input configurations differ from winners")
    normalized = {}
    for configuration_id in sorted(configurations):
        replicas = configurations[configuration_id]
        if set(replicas) != {101, 202, 303}:
            raise ValueError("fairness inputs require seeds 101/202/303")
        normalized[configuration_id] = {}
        for seed in (101, 202, 303):
            binding = replicas[seed]
            if set(binding) != {"training_ledger", "resource_profile"}:
                raise ValueError("fairness replica binding inventory changed")
            normalized[configuration_id][str(seed)] = {}
            for kind in ("training_ledger", "resource_profile"):
                source = binding[kind]
                if set(source) != {"path", "sha256"}:
                    raise ValueError("fairness artifact binding changed")
                path = Path(source["path"]).resolve()
                digest = require_sha256(
                    f"{kind}.sha256", source["sha256"]
                )
                if not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("fairness source artifact is absent/stale")
                normalized[configuration_id][str(seed)][kind] = {
                    "path": str(path),
                    "sha256": digest,
                }
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FAIRNESS_INPUT_INDEX_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "flop_fixture_sha256": require_sha256(
                "flop_fixture_sha256", flop_fixture_sha256
            ),
            "flop_counter_sha256": require_sha256(
                "flop_counter_sha256", flop_counter_sha256
            ),
            "configurations": normalized,
            "configuration_count": len(normalized),
            "replica_count": sum(len(rows) for rows in normalized.values()),
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    validate_fairness_input_index(
        artifact, selection=selection, verify_files=True
    )
    return artifact


def validate_fairness_input_index(
    payload: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    verify_files: bool,
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_FAIRNESS_INPUT_INDEX_CONTRACT
    )
    validate_content_hash(
        selection, expected_contract="particle_view_winner_selection_v1"
    )
    expected = {
        "contract",
        "selection_sha256",
        "flop_fixture_sha256",
        "flop_counter_sha256",
        "configurations",
        "configuration_count",
        "replica_count",
        "stack_val_loaded",
        "final_test_loaded",
        "content_hash",
    }
    required = {
        selection["selected_privileged_scientific_model"]["configuration_id"],
        selection["selected_pre_stage_g_hlt_deployable_model"][
            "configuration_id"
        ],
    }
    if (
        set(payload) != expected
        or payload["selection_sha256"] != selection["content_hash"]
        or set(payload["configurations"]) != required
        or payload["configuration_count"] != len(required)
        or payload["replica_count"] != 3 * len(required)
        or payload["stack_val_loaded"] is not False
        or payload["final_test_loaded"] is not False
    ):
        raise ValueError("fairness input index policy changed")
    require_sha256("flop_fixture_sha256", payload["flop_fixture_sha256"])
    require_sha256("flop_counter_sha256", payload["flop_counter_sha256"])
    for replicas in payload["configurations"].values():
        if set(replicas) != {"101", "202", "303"}:
            raise ValueError("fairness input seed inventory changed")
        for binding in replicas.values():
            if set(binding) != {"training_ledger", "resource_profile"}:
                raise ValueError("fairness input binding inventory changed")
            for source in binding.values():
                if set(source) != {"path", "sha256"}:
                    raise ValueError("fairness source fields changed")
                digest = require_sha256("source.sha256", source["sha256"])
                path = Path(source["path"]).resolve()
                if verify_files and (
                    not path.is_file() or sha256_file(path) != digest
                ):
                    raise ValueError("fairness source artifact changed")
    return {"ok": True, "content_hash": payload["content_hash"]}


def build_fairness_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    validate_runtime_data_config(runtime_data_config, verify_cache_files=True)
    if not isinstance(device, str) or not device or int(num_workers) < 0:
        raise ValueError("fairness runtime settings are invalid")
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_val_batches", max_val_batches),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when set")
    candidates = _direct_candidates()
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FAIRNESS_FACTORY_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "runtime": {
                "device": device,
                "num_workers": int(num_workers),
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
            },
            "winner_families": list(_WINNER_FAMILIES),
            "fairness_control_ids": list(_FAIRNESS_CONTROL_IDS),
            "direct_candidates": candidates,
            "direct_candidates_sha256": canonical_sha256(candidates),
            "direct_trial_grid": {
                "learning_rate": [1.0e-4, 3.0e-4],
                "weight_decay": [1.0e-5, 1.0e-4],
                "dropout": [0.05, 0.10],
            },
            "direct_trial_count": 8,
            "a0_checkpoint_kinds": [
                "exact_matched_update",
                "best_model_val_stop_within_budget",
            ],
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    validate_fairness_factory_config(artifact)
    return artifact


def validate_fairness_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_FAIRNESS_FACTORY_CONFIG_CONTRACT
    )
    expected = {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "runtime",
        "winner_families",
        "fairness_control_ids",
        "direct_candidates",
        "direct_candidates_sha256",
        "direct_trial_grid",
        "direct_trial_count",
        "a0_checkpoint_kinds",
        "performance_gates",
        "quality_warnings_stop_execution",
        "stack_val_loaded",
        "final_test_loaded",
        "content_hash",
    }
    if set(payload) != expected:
        raise ValueError("fairness factory field inventory mismatch")
    validate_runtime_data_config(
        payload["runtime_data_config"], verify_cache_files=False
    )
    runtime = payload["runtime"]
    if set(runtime) != {
        "device",
        "num_workers",
        "max_train_batches",
        "max_val_batches",
    }:
        raise ValueError("fairness runtime field inventory mismatch")
    candidates = _direct_candidates()
    if (
        payload["runtime_data_config_sha256"]
        != payload["runtime_data_config"]["content_hash"]
        or payload["winner_families"] != list(_WINNER_FAMILIES)
        or payload["fairness_control_ids"] != list(_FAIRNESS_CONTROL_IDS)
        or payload["direct_candidates"] != candidates
        or payload["direct_candidates_sha256"] != canonical_sha256(candidates)
        or payload["direct_trial_grid"]
        != {
            "learning_rate": [1.0e-4, 3.0e-4],
            "weight_decay": [1.0e-5, 1.0e-4],
            "dropout": [0.05, 0.10],
        }
        or payload["direct_trial_count"] != 8
        or payload["a0_checkpoint_kinds"]
        != ["exact_matched_update", "best_model_val_stop_within_budget"]
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
        or payload["stack_val_loaded"] is not False
        or payload["final_test_loaded"] is not False
        or not isinstance(runtime["device"], str)
        or not runtime["device"]
        or int(runtime["num_workers"]) < 0
        or any(
            value is not None and int(value) <= 0
            for value in (
                runtime["max_train_batches"],
                runtime["max_val_batches"],
            )
        )
    ):
        raise ValueError("fairness production policy changed")
    return {"ok": True, "content_hash": payload["content_hash"]}


def _resolve_fairness_inputs(
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = _task_artifacts(
        root, registry, "SELECT_WINNER_FAMILIES", seed
    )
    selection = load_hashed_json(
        _artifact(artifacts, "winner_selection.json")
    )
    index = load_hashed_json(
        _artifact(artifacts, "fairness_input_index.json")
    )
    validate_fairness_input_index(
        index, selection=selection, verify_files=True
    )
    ledgers: dict[str, dict[int, dict[str, Any]]] = {}
    resources: dict[str, dict[int, dict[str, Any]]] = {}
    for configuration_id, replicas in index["configurations"].items():
        if set(map(int, replicas)) != {101, 202, 303}:
            raise ValueError("fairness inputs require all three seeds")
        ledgers[configuration_id] = {}
        resources[configuration_id] = {}
        for raw_seed, binding in replicas.items():
            replica_seed = int(raw_seed)
            ledger_path = Path(binding["training_ledger"]["path"]).resolve()
            resource_path = Path(binding["resource_profile"]["path"]).resolve()
            if (
                sha256_file(ledger_path)
                != binding["training_ledger"]["sha256"]
                or sha256_file(resource_path)
                != binding["resource_profile"]["sha256"]
            ):
                raise ValueError("fairness source artifact changed")
            ledger = load_hashed_json(ledger_path)
            resource = load_hashed_json(resource_path)
            ledgers[configuration_id][replica_seed] = ledger
            resources[configuration_id][replica_seed] = resource
    return selection, {
        "index": index,
        "training_ledgers": ledgers,
        "resource_profiles": resources,
    }


def _publish_fairness_ledger(
    *,
    output_dir: str,
    selection: Mapping[str, Any],
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
    a0_checkpoint_by_seed: Mapping[int, str],
    a0_config_sha256: str,
) -> None:
    data = config["runtime_data_config"]
    unified = load_hashed_json(data["unified_manifest"]["path"])
    train_identity = unified["logical_splits"]["train"][
        "ordered_identity_sha256"
    ]
    fairness = build_selected_path_fairness_ledger(
        selection=selection,
        replica_training_ledgers=inputs["training_ledgers"],
        resource_profiles=inputs["resource_profiles"],
        train_identity_sha256=train_identity,
        flop_fixture_sha256=inputs["index"]["flop_fixture_sha256"],
        flop_counter_sha256=inputs["index"]["flop_counter_sha256"],
    )
    output = Path(output_dir)
    write_immutable_json(output / "selected_path_fairness_ledger.json", fairness)
    accounting = build_fairness_budget_accounting(
        selected_path_fairness_ledger=fairness,
        label_exposure_ledgers=[
            ledger
            for replicas in inputs["training_ledgers"].values()
            for ledger in replicas.values()
        ],
    )
    write_immutable_json(output / "fairness_budget_accounting.json", accounting)
    candidates = [
        DirectControlCandidate(
            row["config_id"],
            int(row["deployed_parameters"]),
            int(row["forward_flops"]),
            row["config_sha256"],
        )
        for row in config["direct_candidates"]
    ]
    plan = build_stage_g_control_plan(
        fairness_ledger=fairness,
        candidates=candidates,
        a0_checkpoint_by_seed=a0_checkpoint_by_seed,
        a0_config_sha256=a0_config_sha256,
    )
    write_immutable_json(output / "stage_g_control_plan.json", plan)
    write_immutable_json(
        output / "fairness_publication.json",
        with_content_hash(
            {
                "contract": "particle_view_fairness_publication_v1",
                "selection_sha256": selection["content_hash"],
                "fairness_input_index_sha256": inputs["index"]["content_hash"],
                "fairness_ledger_sha256": fairness["content_hash"],
                "fairness_budget_accounting_sha256": accounting[
                    "content_hash"
                ],
                "stage_g_control_plan_sha256": plan["content_hash"],
                "stage_g_controls_may_start": True,
                "performance_gate_used": False,
            }
        ),
    )


def _resolve_a0_bindings(
    root: Path,
    registry: Mapping[str, Any],
) -> tuple[dict[int, str], str]:
    hashes = {}
    config_hashes = set()
    for seed in (101, 202, 303):
        registration, checkpoint, _ = _teacher_from_task(
            root, registry, "A0_VIEW", seed
        )
        hashes[seed] = sha256_file(checkpoint)
        config_hashes.add(
            canonical_sha256(registration["recipe"]["architecture"])
        )
    if len(config_hashes) != 1:
        raise ValueError("A0 replicas disagree on architecture")
    return hashes, next(iter(config_hashes))


def build_fairness_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_fairness_factory_config(config)
    validate_particle_view_registry(registry)
    if (
        operation != "fairness_closure"
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("fairness task identity changed")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    if run_id == "SELECTED_PATH_FAIRNESS_LEDGER":
        selection, inputs = _resolve_fairness_inputs(root, registry, seed)
        checkpoints, architecture_sha = _resolve_a0_bindings(root, registry)
        return {
            "kwargs": {
                "output_dir": str(output),
                "selection": selection,
                "inputs": inputs,
                "config": dict(config),
                "a0_checkpoint_by_seed": checkpoints,
                "a0_config_sha256": architecture_sha,
            },
            "artifact_paths": [
                str(output / "selected_path_fairness_ledger.json"),
                str(output / "fairness_budget_accounting.json"),
                str(output / "stage_g_control_plan.json"),
                str(output / "fairness_publication.json"),
            ],
            "action": _publish_fairness_ledger,
        }
    if not run_id.startswith("FAIR_"):
        raise ValueError("unknown fairness run")
    ledger_artifacts = _task_artifacts(
        root, registry, "SELECTED_PATH_FAIRNESS_LEDGER", seed
    )
    fairness = load_hashed_json(
        _artifact(ledger_artifacts, "selected_path_fairness_ledger.json")
    )
    plan = load_hashed_json(
        _artifact(ledger_artifacts, "stage_g_control_plan.json")
    )
    return _prepare_stage_g_control(
        config=config,
        registry=registry,
        root=root,
        output=output,
        run_id=run_id,
        seed=int(seed),
        fairness=fairness,
        plan=plan,
    )


def _resolved_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _move_batch(raw: Mapping[str, Any], device: torch.device):
    return {
        key: (
            value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in raw.items()
    }


def _forward(model, batch):
    return model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )


def _train_ce_trajectory(
    *,
    model,
    train_loader,
    stop_loader,
    output: Path,
    seed: int,
    device: str,
    exact_updates: int | None,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    patience: int | None,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    resolved = _resolved_device(device)
    model = model.to(resolved)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    updates = 0
    best = None
    stale = 0
    epoch = 0
    target_updates = exact_updates
    if target_updates == 0:
        metrics = evaluate_particle_view_teacher(
            model,
            stop_loader,
            device=resolved,
            max_batches=max_val_batches,
        )
        row = {"epoch": 0, "optimizer_updates": 0, **metrics}
        rows.append(row)
        best = {"order": (0.0, 0.0, 0), "row": row}
        payload = {
            "contract": "particle_view_stage_g_checkpoint_v1",
            "seed": seed,
            "optimizer_updates": 0,
            "model_state_dict": model.state_dict(),
            "model_val_stop": metrics,
        }
        torch.save(payload, output / "best_model_val_stop_within_budget.pt")
        torch.save(payload, output / "exact_matched_update.pt")
        return {
            "model": model,
            "rows": rows,
            "optimizer_updates": 0,
            "best": row,
        }
    while (
        updates < target_updates
        if target_updates is not None
        else epoch < maximum_epochs
    ):
        epoch += 1
        model.train()
        batches = 0
        for raw in train_loader:
            if max_train_batches is not None and batches >= max_train_batches:
                break
            if target_updates is not None and updates >= target_updates:
                break
            batch = _move_batch(raw, resolved)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = torch.nn.functional.cross_entropy(
                logits, batch["labels"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Stage-G CE loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if target_updates is not None:
                total = max(target_updates, 1)
                lr = teacher_learning_rate(
                    update_index=min(updates, total - 1),
                    total_updates=total,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
            optimizer.step()
            updates += 1
            batches += 1
        if batches == 0:
            raise ValueError("Stage-G training loader is empty")
        metrics = evaluate_particle_view_teacher(
            model,
            stop_loader,
            device=resolved,
            max_batches=max_val_batches,
        )
        row = {"epoch": epoch, "optimizer_updates": updates, **metrics}
        rows.append(row)
        order = (
            -float(metrics["accuracy"]),
            float(metrics["cross_entropy"]),
            epoch,
        )
        if best is None or order < best["order"]:
            best = {"order": order, "row": row}
            stale = 0
            torch.save(
                {
                    "contract": "particle_view_stage_g_checkpoint_v1",
                    "seed": seed,
                    "optimizer_updates": updates,
                    "model_state_dict": model.state_dict(),
                    "model_val_stop": metrics,
                },
                output / "best_model_val_stop_within_budget.pt",
            )
        else:
            stale += 1
        if (
            target_updates is None
            and patience is not None
            and stale >= patience
        ):
            break
        if epoch > 100000:
            raise RuntimeError("Stage-G exact-update trajectory did not finish")
    if target_updates is not None:
        torch.save(
            {
                "contract": "particle_view_stage_g_checkpoint_v1",
                "seed": seed,
                "optimizer_updates": updates,
                "model_state_dict": model.state_dict(),
                "model_val_stop": rows[-1],
            },
            output / "exact_matched_update.pt",
        )
    return {
        "model": model,
        "rows": rows,
        "optimizer_updates": updates,
        "best": best["row"],
    }


def _write_stage_g_result(
    *,
    output: Path,
    run_id: str,
    job: Mapping[str, Any],
    checkpoint: Path,
    rows: list[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> None:
    curves = with_content_hash(
        {
            "contract": "particle_view_stage_g_curves_v1",
            "run_id": run_id,
            "job_sha256": canonical_sha256(job),
            "epochs": rows,
            "quality_gate_used": False,
        }
    )
    write_immutable_json(output / "training_curves.json", curves)
    result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STAGE_G_RESULT_CONTRACT,
            "run_id": run_id,
            "configuration_id": job["configuration_id"],
            "control_id": job["control_id"],
            "seed": job["seed"],
            "fairness_entry_sha256": job["fairness_entry_sha256"],
            "winner_bundle_sha256": job["winner_bundle_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_curves_sha256": curves["content_hash"],
            "stack_val_eligible": True,
            "final_test_eligible": False,
            "hlt_only_inference": True,
            "performance_gate_used": False,
            **dict(extra),
        }
    )
    write_immutable_json(output / "stage_g_result.json", result)


def _run_a0_stage_g(
    *,
    model,
    train_loader,
    stop_loader,
    output_dir: str,
    run_id: str,
    job: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    output = Path(output_dir)
    trained = _train_ce_trajectory(
        model=model,
        train_loader=train_loader,
        stop_loader=stop_loader,
        output=output,
        seed=int(job["seed"]),
        device=runtime["device"],
        exact_updates=int(job["exact_optimizer_update_budget"]),
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        maximum_epochs=100000,
        patience=None,
        max_train_batches=runtime["max_train_batches"],
        max_val_batches=runtime["max_val_batches"],
    )
    checkpoint = output / "best_model_val_stop_within_budget.pt"
    _write_stage_g_result(
        output=output,
        run_id=run_id,
        job=job,
        checkpoint=checkpoint,
        rows=trained["rows"],
        extra={
            "exact_optimizer_update_budget": int(
                job["exact_optimizer_update_budget"]
            ),
            "exact_matched_checkpoint_sha256": sha256_file(
                output / "exact_matched_update.pt"
            ),
            "primary_comparison_checkpoint": (
                "best_model_val_stop_within_budget"
            ),
        },
    )


def _run_direct_stage_g(
    *,
    candidate: Mapping[str, Any],
    train_loader,
    stop_loader,
    output_dir: str,
    run_id: str,
    job: Mapping[str, Any],
    runtime: Mapping[str, Any],
    trial_grid: Mapping[str, Any],
) -> None:
    output = Path(output_dir)
    trials = []
    for learning_rate in trial_grid["learning_rate"]:
        for weight_decay in trial_grid["weight_decay"]:
            for dropout in trial_grid["dropout"]:
                trial_id = (
                    f"lr{learning_rate:g}_wd{weight_decay:g}_do{dropout:g}"
                )
                # Every hyperparameter trial starts from the same
                # same-seed initialization; only the declared trial
                # hyperparameters may differ.
                torch.manual_seed(int(job["seed"]))
                model = _direct_model(candidate)
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = float(dropout)
                trial_root = output / "trials" / trial_id
                trained = _train_ce_trajectory(
                    model=model,
                    train_loader=train_loader,
                    stop_loader=stop_loader,
                    output=trial_root,
                    seed=int(job["seed"]),
                    device=runtime["device"],
                    exact_updates=None,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    maximum_epochs=40,
                    patience=8,
                    max_train_batches=runtime["max_train_batches"],
                    max_val_batches=runtime["max_val_batches"],
                )
                trials.append(
                    {
                        "trial_id": trial_id,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "best": trained["best"],
                        "checkpoint": str(
                            trial_root
                            / "best_model_val_stop_within_budget.pt"
                        ),
                        "checkpoint_sha256": sha256_file(
                            trial_root
                            / "best_model_val_stop_within_budget.pt"
                        ),
                        "rows": trained["rows"],
                    }
                )
    winner = min(
        trials,
        key=lambda row: (
            -float(row["best"]["accuracy"]),
            float(row["best"]["cross_entropy"]),
            row["trial_id"],
        ),
    )
    checkpoint = Path(winner["checkpoint"])
    selected = output / "best_model_val_stop_within_budget.pt"
    selected.write_bytes(checkpoint.read_bytes())
    summary = with_content_hash(
        {
            "contract": "particle_view_stage_g_direct_trials_v1",
            "run_id": run_id,
            "candidate_config_sha256": candidate["config_sha256"],
            "trials": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"rows", "checkpoint"}
                }
                for row in trials
            ],
            "selected_trial_id": winner["trial_id"],
            "trial_count": len(trials),
        }
    )
    write_immutable_json(output / "direct_trial_summary.json", summary)
    for row in trials:
        Path(row["checkpoint"]).unlink()
    _write_stage_g_result(
        output=output,
        run_id=run_id,
        job=job,
        checkpoint=selected,
        rows=winner["rows"],
        extra={
            "direct_trial_summary_sha256": summary["content_hash"],
            "direct_match_sha256": job["direct_match_sha256"],
            "requested_tolerance_met": job[
                "requested_tolerance_met"
            ],
            "selected_direct_config_sha256": candidate["config_sha256"],
        },
    )


def _prepare_stage_g_control(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    root: Path,
    output: Path,
    run_id: str,
    seed: int,
    fairness: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        fairness,
        expected_contract="particle_view_selected_path_fairness_ledger_v1",
    )
    validate_content_hash(
        plan, expected_contract="particle_view_stage_g_control_plan_v1"
    )
    if plan["fairness_ledger_sha256"] != fairness["content_hash"]:
        raise ValueError("Stage-G plan belongs to another fairness ledger")
    remainder = run_id[len("FAIR_") :]
    family = next(
        (
            candidate
            for candidate in _WINNER_FAMILIES
            if remainder.startswith(candidate + "_")
        ),
        None,
    )
    if family is None:
        raise ValueError("fairness winner family is invalid")
    control_id = remainder[len(family) + 1 :]
    if control_id not in _FAIRNESS_CONTROL_IDS:
        raise ValueError("fairness control ID is invalid")
    selection_family = {
        "PRIVILEGED_SCIENTIFIC": "selected_privileged_scientific_model",
        "PRE_STAGE_G_DEPLOYABLE": (
            "selected_pre_stage_g_hlt_deployable_model"
        ),
    }[family]
    entries = [
        entry
        for entry in fairness["entries"]
        if selection_family in entry["winner_families"]
    ]
    if len(entries) != 1:
        raise ValueError("fairness family does not resolve to one entry")
    if (
        family == "PRIVILEGED_SCIENTIFIC"
        and set(entries[0]["winner_families"])
        == {
            "selected_privileged_scientific_model",
            "selected_pre_stage_g_hlt_deployable_model",
        }
    ):
        canonical_run_id = (
            f"FAIR_PRE_STAGE_G_DEPLOYABLE_{control_id}"
        )
        canonical_artifacts = _task_artifacts(
            root, registry, canonical_run_id, seed
        )
        required_names = [
            "best_model_val_stop_within_budget.pt",
            "training_curves.json",
            "stage_g_result.json",
        ]
        if control_id in {"SELECTED_PARAMETER_MATCH", "SELECTED_FLOP_MATCH"}:
            required_names.append("direct_trial_summary.json")
        paths = [_artifact(canonical_artifacts, name) for name in required_names]
        alias_path = output / "stage_g_alias.json"
        return {
            "kwargs": {
                "output_path": str(alias_path),
                "run_id": run_id,
                "canonical_run_id": canonical_run_id,
                "seed": seed,
                "fairness_entry_sha256": entries[0][
                    "fairness_entry_sha256"
                ],
                "canonical_artifacts": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for path in paths
                ],
            },
            "artifact_paths": [*(str(path) for path in paths), str(alias_path)],
            "action": _publish_stage_g_alias,
        }
    configuration_id = entries[0]["configuration_id"]
    jobs = [
        row
        for row in plan["jobs"]
        if (
            row["configuration_id"] == configuration_id
            and row["control_id"] == control_id
            and int(row["seed"]) == seed
        )
    ]
    if len(jobs) != 1:
        raise ValueError("Stage-G job resolution is ambiguous")
    job = jobs[0]
    runtime = config["runtime"]
    data = config["runtime_data_config"]
    train = load_aligned_logical_jet_view(data, "train")
    stop = load_aligned_logical_jet_view(data, "model_val_stop")
    train_loader = make_logical_data_loader(
        train,
        mode="fixed_hlt",
        batch_size=128,
        shuffle=True,
        num_workers=runtime["num_workers"],
        seed=seed,
    )
    stop_loader = make_logical_data_loader(
        stop,
        mode="fixed_hlt",
        batch_size=128,
        shuffle=False,
        num_workers=runtime["num_workers"],
        seed=seed + 1,
    )
    common = {
        "train_loader": train_loader,
        "stop_loader": stop_loader,
        "output_dir": str(output),
        "run_id": run_id,
        "job": job,
        "runtime": runtime,
    }
    if control_id in {
        "A0_VIEW_LONG_DEPLOY",
        "A0_VIEW_TOTAL_LABEL_BUDGET",
    }:
        _, _, model = _teacher_from_task(
            root, registry, "A0_VIEW", seed
        )
        return {
            "kwargs": {"model": deepcopy(model), **common},
            "artifact_paths": [
                str(output / "exact_matched_update.pt"),
                str(output / "best_model_val_stop_within_budget.pt"),
                str(output / "training_curves.json"),
                str(output / "stage_g_result.json"),
            ],
            "action": _run_a0_stage_g,
        }
    selected = job["selected_direct_config"]
    candidate = next(
        row
        for row in config["direct_candidates"]
        if row["config_id"] == selected["config_id"]
    )
    if (
        candidate["config_sha256"] != selected["config_sha256"]
        or candidate["deployed_parameters"]
        != selected["deployed_parameters"]
        or candidate["forward_flops"] != selected["forward_flops"]
    ):
        raise ValueError("selected direct control profile changed")
    return {
        "kwargs": {
            "candidate": candidate,
            "trial_grid": config["direct_trial_grid"],
            **common,
        },
        "artifact_paths": [
            str(output / "best_model_val_stop_within_budget.pt"),
            str(output / "training_curves.json"),
            str(output / "direct_trial_summary.json"),
            str(output / "stage_g_result.json"),
        ],
        "action": _run_direct_stage_g,
    }


def _publish_stage_g_alias(
    *,
    output_path: str,
    run_id: str,
    canonical_run_id: str,
    seed: int,
    fairness_entry_sha256: str,
    canonical_artifacts: list[dict[str, str]],
) -> None:
    write_immutable_json(
        output_path,
        with_content_hash(
            {
                "contract": PARTICLE_VIEW_STAGE_G_ALIAS_CONTRACT,
                "run_id": run_id,
                "canonical_run_id": canonical_run_id,
                "seed": int(seed),
                "fairness_entry_sha256": require_sha256(
                    "fairness_entry_sha256", fairness_entry_sha256
                ),
                "canonical_artifacts": canonical_artifacts,
                "independent_retraining_performed": False,
                "numerically_identical_by_construction": True,
            }
        ),
    )


def build_fairness_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_fairness_factory_config(load_hashed_json(path))
    common = {
        "operation": "fairness_closure",
        "factory": (
            "teacher_logit_reco.local_particle_residual_field."
            "particle_view.fairness_runtime:build_fairness_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    specs = {"SELECTED_PATH_FAIRNESS_LEDGER": dict(common)}
    for family in _WINNER_FAMILIES:
        for control_id in _FAIRNESS_CONTROL_IDS:
            specs[f"FAIR_{family}_{control_id}"] = dict(common)
    return specs


__all__ = [
    "PARTICLE_VIEW_FAIRNESS_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_FAIRNESS_INPUT_INDEX_CONTRACT",
    "PARTICLE_VIEW_STAGE_G_RESULT_CONTRACT",
    "build_fairness_input_index",
    "build_fairness_factory",
    "build_fairness_factory_config",
    "build_fairness_task_specs",
    "validate_fairness_factory_config",
    "validate_fairness_input_index",
]
