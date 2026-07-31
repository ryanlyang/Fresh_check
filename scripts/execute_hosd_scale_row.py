#!/usr/bin/env python3
"""Execute one source-bound Stage-J teacher, target, or student coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_NORMALIZER_COMPLETION_CONTRACT,
    SCALE_TARGET_COMPLETION_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.scale_runtime import (  # noqa: E402
    PAIR_TARGETS,
    build_scale_target_completion,
    build_scale_target_wave_completion,
    fit_pair_normalizer_from_views,
    merge_target_normalizers,
    offline_to_hlt_target,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(values: list[str], *, name: str, exact_replicas: bool = True) -> dict[int, Path]:
    output = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"{name} requires REPLICA=PATH")
        replica = int(key)
        if replica not in range(4) or replica in output:
            raise ValueError(f"{name} has an invalid or duplicate replica")
        output[replica] = Path(raw_path).resolve()
    if exact_replicas and set(output) != set(range(4)):
        raise ValueError(f"{name} requires exact replicas 0,1,2,3")
    return output


def _load_cache(path: Path):
    from teacher_logit_reco.hlt_offline_structure_distillation import load_target_cache_sharded
    from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
        TARGET_CACHE_SPEC_CONTRACT,
    )

    spec = load_hashed_json(
        path / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
    )
    return load_target_cache_sharded(path, cache_spec=spec)


def _component_kinds(registry, target_id: str) -> tuple[str, ...]:
    rows = [row for row in registry["targets"] if row["target_id"] == target_id]
    if len(rows) != 1:
        raise ValueError("scale target registry coordinate differs")
    return tuple(
        "continuous"
        if str(component["component_kind"]).endswith("continuous")
        else "binary"
        if str(component["component_kind"]).endswith("binary")
        else "categorical"
        for component in rows[0]["components"]
    )


def _fit_cache_normalizer(
    cache_path: Path,
    *,
    target_id_for_kinds: str,
    registry,
    source,
    normalization_role: str,
):
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        fit_sharded_target_normalizer,
    )

    cache = _load_cache(cache_path)
    kinds = _component_kinds(registry, target_id_for_kinds)
    return fit_sharded_target_normalizer(
        [cache],
        target_id=target_id_for_kinds,
        fitting_population="target_scale",
        source=source,
        component_kinds=kinds,
        normalization_role=normalization_role,
        workspace=cache_path.parent / ".normalizer_workspace",
    )


def _build_physical_cache(
    *,
    root: Path,
    input_npz: Path,
    output: Path,
    target_id: str,
    artifact_kind: str,
    cache_id: str,
    relation_normalizer: Path,
    tree_backend_manifest: Path | None,
    tree_cache: Path | None,
    replica: int | None,
) -> None:
    from scripts.build_hosd_targets import main as build_targets

    argv = [
        "--campaign-root",
        str(root),
        "--input-npz",
        str(input_npz),
        "--output-dir",
        str(output),
        "--split",
        "scale_train",
        "--artifact-kind",
        artifact_kind,
        "--target-id",
        target_id,
        "--relation-normalizer",
        str(relation_normalizer),
        "--cache-id",
        cache_id,
    ]
    if replica is not None:
        argv.extend(["--hlt-replica-id", str(replica)])
    if "CA_TREE" in target_id or "REGION" in target_id:
        if tree_backend_manifest is None or tree_cache is None:
            raise ValueError("scale tree-derived target lacks tree resources")
        argv.extend(
            [
                "--tree-backend-manifest",
                str(tree_backend_manifest),
                "--tree-cache-dir",
                str(tree_cache),
            ]
        )
    if build_targets(argv):
        raise RuntimeError("scale target-cache builder returned nonzero")


def _build_residual_cache(
    *,
    root: Path,
    canonical: Path,
    hlt: Path,
    output: Path,
    target_id: str,
    hlt_target_id: str,
    replica: int,
) -> None:
    from scripts.build_hosd_target_derivatives import main as build_derivatives

    pairs = output.parent / f".{output.name}.target_pairs.json"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps({target_id: hlt_target_id}, sort_keys=True)
    if pairs.exists() and pairs.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("scale residual target-pair declaration differs")
    pairs.write_text(encoded, encoding="utf-8")
    code = build_derivatives(
        [
            "--campaign-root",
            str(root),
            "--canonical-cache",
            str(canonical),
            "--hlt-cache",
            str(hlt),
            "--target-pairs",
            str(pairs),
            "--output-dir",
            str(output),
            "--kind",
            "residual",
            "--cache-id",
            f"scale_residual_{target_id}_{replica}",
        ]
    )
    if code:
        raise RuntimeError("scale residual builder returned nonzero")


def _target(args, root: Path, campaign, plan) -> int:
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        build_heteroscedastic_metadata,
        fit_sharded_latent_whitening,
    )

    if args.target_id is None:
        raise ValueError("scale target execution lacks target ID")
    rows = [
        row for row in plan["target_refit_rows"]
        if row["target_id"] == args.target_id
    ]
    if len(rows) != 1:
        raise ValueError("scale target row is absent or duplicated")
    plan_row = rows[0]
    output = root / "scale_up" / "targets" / args.target_id
    completion_path = output / "completion.json"
    if completion_path.is_file():
        existing = load_hashed_json(
            completion_path, expected_contract=SCALE_TARGET_COMPLETION_CONTRACT
        )
        if (
            existing.get("source") != campaign["source"]
            or existing.get("scale_execution_plan_sha256") != plan["content_hash"]
            or existing.get("target_id") != args.target_id
        ):
            raise ValueError("reusable scale target completion lineage differs")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    registry = load_hashed_json(
        root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    offline_relation = load_hashed_json(
        args.scale_offline_relation_normalizer
    )
    hlt_relation = load_hashed_json(args.scale_hlt_relation_normalizer)
    artifact_hashes: dict[str, str] = {}
    definitions: dict[str, dict] = {}
    required = set(plan_row["required_parameterizations"])
    normalizer_path = output / "target_normalizer.json"
    residual_normalizer_path = output / "residual_normalizer.json"
    whitening_path = output / "latent_whitening.json"

    if args.target_id in PAIR_TARGETS:
        views = _mapping(args.scale_hlt_input, name="--scale-hlt-input")
        trees = (
            _mapping(args.scale_hlt_tree, name="--scale-hlt-tree")
            if args.target_id == "T_HLT_REGION_PAIR_8"
            else None
        )
        tree_completion = (
            load_hashed_json(root / "scale_up" / "trees" / "completion.json")
            if trees is not None
            else None
        )
        normalizer = fit_pair_normalizer_from_views(
            target_id=args.target_id,
            view_paths_by_replica=views,
            relation_normalizer=hlt_relation,
            tree_paths_by_replica=trees,
            fitting_population="target_scale",
            split="scale_train",
            source=campaign["source"],
            tree_resource_sha256=(
                None if tree_completion is None else tree_completion["tree_resource_sha256"]
            ),
            tree_backend_sha256=(
                None if tree_completion is None else tree_completion["backend_manifest_sha256"]
            ),
        )
        write_immutable_json(normalizer_path, normalizer)
        artifact_hashes["target_normalizer"] = normalizer["content_hash"]
        for parameterization in required:
            if parameterization != "ABS":
                raise ValueError("streamed HLT pair target supports only ABS")
            definitions[parameterization] = {
                "mode": "stream_same_view",
                "normalizer": str(normalizer_path.resolve()),
                "relation_normalizer": str(
                    args.scale_hlt_relation_normalizer.resolve()
                ),
                "control_kind": None,
            }
    elif args.target_id.startswith(
        ("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_")
    ):
        teacher_id = (
            "O_FULLREL"
            if args.target_id == "T_OFFLINE_LOGITS_O_FULLREL"
            else "O_BASE"
        )
        cache_path = root / "scale_up" / "teacher_outputs" / teacher_id
        cache = _load_cache(cache_path)
        if args.target_id not in cache.values:
            raise ValueError("scale teacher output cache lacks target")
        normalizer = _fit_cache_normalizer(
            cache_path,
            target_id_for_kinds=args.target_id,
            registry=registry,
            source=campaign["source"],
            normalization_role="target",
        )
        write_immutable_json(normalizer_path, normalizer)
        artifact_hashes.update(
            {
                "teacher_target_cache": cache.manifest["content_hash"],
                "target_normalizer": normalizer["content_hash"],
            }
        )
        if args.target_id == "T_OFFLINE_POOLED_LATENT":
            lock = load_hashed_json(
                root / "scale_up" / "teachers" / "teacher_lock.json"
            )
            whitening = fit_sharded_latent_whitening(
                cache.values[args.target_id],
                teacher_lock_sha256=lock["content_hash"],
                fitting_population="target_scale",
                source=campaign["source"],
            )
            write_immutable_json(whitening_path, whitening)
            artifact_hashes["latent_whitening"] = whitening["content_hash"]
        for parameterization in required:
            definition = {
                "mode": "static_cache",
                "caches": {"shared": str(cache_path.resolve())},
                "control_kind": None,
            }
            if parameterization == "WHITENED_ABS":
                definition["whitening"] = str(whitening_path.resolve())
            elif parameterization != "KD":
                definition["normalizer"] = str(normalizer_path.resolve())
            definitions[parameterization] = definition
    elif args.target_id.startswith("T_HLT_SELF_"):
        views = _mapping(args.scale_hlt_input, name="--scale-hlt-input")
        trees = (
            _mapping(args.scale_hlt_tree, name="--scale-hlt-tree")
            if "CA_TREE" in args.target_id or "REGION" in args.target_id
            else {}
        )
        caches = {}
        for replica in range(4):
            cache_path = output / "hlt" / f"replica_{replica}"
            _build_physical_cache(
                root=root,
                input_npz=views[replica],
                output=cache_path,
                target_id=args.target_id,
                artifact_kind="hlt_analogue",
                cache_id=f"scale_{args.target_id}_{replica}",
                relation_normalizer=args.scale_hlt_relation_normalizer,
                tree_backend_manifest=args.tree_backend_manifest,
                tree_cache=trees.get(replica),
                replica=replica,
            )
            cache = _load_cache(cache_path)
            artifact_hashes[f"hlt_cache_{replica}"] = cache.manifest["content_hash"]
            caches[str(replica)] = str(cache_path.resolve())
        # Identical coordinate IDs appear in each replica normalizer. A
        # same-view target requires one shared R_MULTI statistic, so fit from
        # the concatenated caches instead of retaining per-replica rows.
        # Rebuild the shared row directly over exact concatenated samples.
        from teacher_logit_reco.hlt_offline_structure_distillation import (
            fit_sharded_target_normalizer,
        )
        loaded = [_load_cache(Path(caches[str(index)])) for index in range(4)]
        normalizer = fit_sharded_target_normalizer(
            loaded,
            target_id=args.target_id,
            fitting_population="target_scale",
            source=campaign["source"],
            component_kinds=_component_kinds(registry, args.target_id),
            workspace=output / ".normalizer_workspace",
        )
        write_immutable_json(normalizer_path, normalizer)
        artifact_hashes["target_normalizer"] = normalizer["content_hash"]
        if "HET" in required:
            hetero = build_heteroscedastic_metadata(
                normalizer, source=campaign["source"]
            )
            write_immutable_json(
                output / "heteroscedastic_metadata.json", hetero
            )
            artifact_hashes["heteroscedastic_metadata"] = hetero["content_hash"]
        for parameterization in required:
            definitions[parameterization] = {
                "mode": "static_cache",
                "caches": caches,
                "normalizer": str(normalizer_path.resolve()),
                "control_kind": None,
            }
    else:
        if args.scale_offline_input is None:
            raise ValueError("scale offline target lacks offline input")
        canonical = output / "canonical"
        _build_physical_cache(
            root=root,
            input_npz=args.scale_offline_input,
            output=canonical,
            target_id=args.target_id,
            artifact_kind="canonical_offline",
            cache_id=f"scale_{args.target_id}_canonical",
            relation_normalizer=args.scale_offline_relation_normalizer,
            tree_backend_manifest=args.tree_backend_manifest,
            tree_cache=args.scale_offline_tree,
            replica=None,
        )
        canonical_cache = _load_cache(canonical)
        normalizer = _fit_cache_normalizer(
            canonical,
            target_id_for_kinds=args.target_id,
            registry=registry,
            source=campaign["source"],
            normalization_role="target",
        )
        write_immutable_json(normalizer_path, normalizer)
        artifact_hashes.update(
            {
                "canonical_cache": canonical_cache.manifest["content_hash"],
                "target_normalizer": normalizer["content_hash"],
            }
        )
        hlt_target = offline_to_hlt_target(args.target_id)
        residual_caches, residual_normalizers = {}, []
        views = _mapping(args.scale_hlt_input, name="--scale-hlt-input")
        trees = (
            _mapping(args.scale_hlt_tree, name="--scale-hlt-tree")
            if "CA_TREE" in hlt_target or "REGION" in hlt_target
            else {}
        )
        for replica in range(4):
            hlt_cache_path = output / "hlt" / f"replica_{replica}"
            _build_physical_cache(
                root=root,
                input_npz=views[replica],
                output=hlt_cache_path,
                target_id=hlt_target,
                artifact_kind="hlt_analogue",
                cache_id=f"scale_{hlt_target}_{replica}",
                relation_normalizer=args.scale_hlt_relation_normalizer,
                tree_backend_manifest=args.tree_backend_manifest,
                tree_cache=trees.get(replica),
                replica=replica,
            )
            hlt_cache = _load_cache(hlt_cache_path)
            artifact_hashes[f"hlt_analogue_cache_{replica}"] = (
                hlt_cache.manifest["content_hash"]
            )
            if "RES" in required:
                residual = output / "residual" / f"replica_{replica}"
                _build_residual_cache(
                    root=root,
                    canonical=canonical,
                    hlt=hlt_cache_path,
                    output=residual,
                    target_id=args.target_id,
                    hlt_target_id=hlt_target,
                    replica=replica,
                )
                residual_cache = _load_cache(residual)
                artifact_hashes[f"residual_cache_{replica}"] = residual_cache.manifest[
                    "content_hash"
                ]
                residual_caches[str(replica)] = str(residual.resolve())
                residual_normalizers.append(
                    _fit_cache_normalizer(
                        residual,
                        target_id_for_kinds=args.target_id,
                        registry=registry,
                        source=campaign["source"],
                        normalization_role="residual",
                    )
                )
        if "RES" in required:
            residual_normalizer = merge_target_normalizers(
                residual_normalizers,
                fitting_population="target_scale",
                normalization_role="residual",
                parent_hashes={
                    f"residual_cache_{replica}": artifact_hashes[
                        f"residual_cache_{replica}"
                    ]
                    for replica in range(4)
                },
                source=campaign["source"],
            )
            write_immutable_json(residual_normalizer_path, residual_normalizer)
            artifact_hashes["residual_normalizer"] = residual_normalizer[
                "content_hash"
            ]
        if "HET" in required:
            hetero = build_heteroscedastic_metadata(
                normalizer, source=campaign["source"]
            )
            write_immutable_json(output / "heteroscedastic_metadata.json", hetero)
            artifact_hashes["heteroscedastic_metadata"] = hetero["content_hash"]
        for parameterization in required:
            if parameterization == "RES":
                definitions[parameterization] = {
                    "mode": "static_cache",
                    "caches": residual_caches,
                    "normalizer": str(residual_normalizer_path.resolve()),
                    "control_kind": None,
                }
            else:
                definitions[parameterization] = {
                    "mode": "static_cache",
                    "caches": {"shared": str(canonical.resolve())},
                    "normalizer": str(normalizer_path.resolve()),
                    "control_kind": None,
                }
    completion = build_scale_target_completion(
        scale_plan=plan,
        target_id=args.target_id,
        artifact_hashes=artifact_hashes,
        training_target_definitions=definitions,
        source=campaign["source"],
    )
    write_immutable_json(completion_path, completion)
    completed = []
    for row in plan["target_refit_rows"]:
        path = root / "scale_up" / "targets" / row["target_id"] / "completion.json"
        if not path.is_file():
            print(json.dumps({"target_id": args.target_id, "coverage_complete": False}))
            return 0
        completed.append(
            load_hashed_json(path, expected_contract=SCALE_TARGET_COMPLETION_CONTRACT)
        )
    wave = build_scale_target_wave_completion(
        scale_plan=plan, completions=completed, source=campaign["source"]
    )
    write_immutable_json(root / "scale_up" / "target_completion.json", wave)
    print(json.dumps({"target_id": args.target_id, "coverage_complete": True}))
    return 0


def _graph(args, root: Path) -> int:
    from scripts.execute_hosd_scale_graph import main as execute_graph

    required = (
        args.graph_id,
        args.seed,
        args.scale_train_labels,
        args.val_stop_labels,
        args.design_confirm_labels,
        args.stage_d_loader_root,
    )
    if any(value is None for value in required):
        raise ValueError("scale graph execution inputs are incomplete")
    argv = [
        "--campaign-root",
        str(root),
        "--graph-id",
        str(args.graph_id),
        "--seed",
        str(args.seed),
        "--scale-train-labels",
        str(args.scale_train_labels),
        "--val-stop-labels",
        str(args.val_stop_labels),
        "--design-confirm-labels",
        str(args.design_confirm_labels),
        "--stage-d-loader-root",
        str(args.stage_d_loader_root),
        "--device",
        str(args.device),
    ]
    for option, values in (
        ("--scale-train-cache", args.scale_train_cache),
        ("--val-stop-cache", args.val_stop_cache),
        ("--design-confirm-cache", args.design_confirm_cache),
        ("--scale-train-tree", args.scale_train_tree),
    ):
        for value in values:
            argv.extend([option, str(value)])
    for value in args.scale_native_relation:
        argv.extend(["--scale-native-relation", str(value)])
    for option, value in (
        ("--design-confirm-native-relation", args.design_confirm_native_relation),
    ):
        if value is not None:
            argv.extend([option, str(value)])
    return execute_graph(argv)


def _teacher(args, root: Path) -> int:
    from scripts.train_hosd_offline_teacher import main as train_teacher

    output = root / "scale_up" / "teachers" / str(args.teacher_id)
    normalizer_completion = load_hashed_json(
        root / "scale_up" / "normalization" / "completion.json",
        expected_contract=SCALE_NORMALIZER_COMPLETION_CONTRACT,
    )
    normalizer_hashes_path = (
        root
        / "scale_up"
        / "normalization"
        / "teacher_normalizer_hashes.json"
    )
    if (
        _sha256_file(normalizer_hashes_path)
        != normalizer_completion["teacher_normalizer_hashes_sha256"]
    ):
        raise ValueError(
            "scale teacher normalizer-hash bytes differ from completion lock"
        )
    argv = [
        "--campaign-root",
        str(root),
        "--model-contract-o-base",
        str(args.model_contract_o_base),
        "--model-contract-o-fullrel",
        str(args.model_contract_o_fullrel),
        "--normalizer-hashes",
        str(normalizer_hashes_path),
        "--population",
        "target_scale",
        "--training-manifest-output",
        str(root / "scale_up" / "teachers" / "training_manifest.json"),
        "--teacher-id",
        str(args.teacher_id),
        "--execute-training",
        "--offline-manifest",
        str(args.offline_manifest),
        "--validation-partition",
        str(args.validation_partition),
        "--screening-registry",
        str(args.screening_registry),
        "--relation-registry",
        str(args.relation_registry),
        "--relation-normalizer",
        str(args.scale_offline_relation_normalizer),
        "--global-determinism",
        str(args.global_determinism),
        "--training-output-dir",
        str(output),
        "--completion-output",
        str(output / "completion.json"),
        "--device",
        str(args.device),
        "--training-input-npz",
        str(
            root
            / "scale_up"
            / "inputs"
            / "offline"
            / "scale_train.npz"
        ),
        "--training-labels-npz",
        str(args.scale_train_labels),
    ]
    if args.data_dir is not None:
        argv.extend(["--data-dir", str(args.data_dir)])
    if args.scale_offline_region_normalizer is not None:
        argv.extend(
            [
                "--region-normalizer",
                str(args.scale_offline_region_normalizer),
            ]
        )
    argv.extend(
        [
            "--tree-root",
            str(root / "scale_up" / "trees" / "offline"),
        ]
    )
    if args.validation_tree_root is not None:
        argv.extend(
            ["--validation-tree-root", str(args.validation_tree_root)]
        )
    code = train_teacher(argv)
    hashes = {}
    for teacher_id in ("O_BASE", "O_FULLREL"):
        completion_path = (
            root
            / "scale_up"
            / "teachers"
            / teacher_id
            / "completion.json"
        )
        if not completion_path.is_file():
            return code
        completion = load_hashed_json(completion_path)
        hashes[teacher_id] = completion["content_hash"]
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    wave = with_content_hash(
        {
            "contract": "hosd_scale_teacher_training_wave_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "scale_execution_plan_sha256": plan["content_hash"],
            "teacher_completion_hashes": hashes,
            "teacher_ids": ["O_BASE", "O_FULLREL"],
            "coverage_exact": True,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(
        root / "scale_up" / "teacher_completion.json", wave
    )
    return code


def _teacher_lock(args, root: Path) -> int:
    from scripts.lock_hosd_teachers import main as lock_teachers

    return lock_teachers(
        [
            "--campaign-root",
            str(root),
            "--training-manifest",
            str(root / "scale_up" / "teachers" / "training_manifest.json"),
            "--o-base-completion",
            str(root / "scale_up" / "teachers" / "O_BASE" / "completion.json"),
            "--o-fullrel-completion",
            str(
                root / "scale_up" / "teachers" / "O_FULLREL" / "completion.json"
            ),
            "--output",
            str(root / "scale_up" / "teachers" / "teacher_lock.json"),
        ]
    )


def _teacher_inference(args, root: Path) -> int:
    from scripts.infer_hosd_teacher_targets import main as infer_teacher

    adapter_config = (
        args.teacher_adapter_config
        if args.teacher_adapter_config is not None
        else root
        / "scale_up"
        / "teacher_outputs"
        / "adapter_configs"
        / f"{args.teacher_id}.json"
    )
    code = infer_teacher(
        [
            "--campaign-root",
            str(root),
            "--teacher-lock",
            str(root / "scale_up" / "teachers" / "teacher_lock.json"),
            "--teacher-id",
            str(args.teacher_id),
            "--split",
            "scale_train",
            "--output-dir",
            str(root / "scale_up" / "teacher_outputs" / str(args.teacher_id)),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    hashes = {}
    for teacher_id in ("O_BASE", "O_FULLREL"):
        path = (
            root
            / "scale_up"
            / "teacher_outputs"
            / teacher_id
            / "target_manifest.json"
        )
        if not path.is_file():
            return code
        manifest = load_hashed_json(path)
        hashes[teacher_id] = manifest["content_hash"]
    completion = with_content_hash(
        {
            "contract": "hosd_scale_teacher_output_completion_v1",
            "schema_version": 1,
            "source": load_and_validate_campaign(
                root, repo_root=REPO_ROOT
            )["source"],
            "scale_execution_plan_sha256": load_hashed_json(
                root / "scale_up" / "execution_plan.json",
                expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
            )["content_hash"],
            "teacher_output_manifest_hashes": hashes,
            "coverage_exact": True,
        }
    )
    write_immutable_json(
        root / "scale_up" / "teacher_outputs" / "completion.json",
        completion,
    )
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("teacher", "teacher-lock", "teacher-inference", "target", "graph"),
        required=True,
    )
    parser.add_argument("--teacher-id", choices=("O_BASE", "O_FULLREL"))
    parser.add_argument("--target-id")
    parser.add_argument("--graph-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-contract-o-base", type=Path)
    parser.add_argument("--model-contract-o-fullrel", type=Path)
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--validation-partition", type=Path)
    parser.add_argument("--screening-registry", type=Path)
    parser.add_argument("--relation-registry", type=Path)
    parser.add_argument("--scale-offline-relation-normalizer", type=Path)
    parser.add_argument("--scale-hlt-relation-normalizer", type=Path)
    parser.add_argument("--scale-offline-region-normalizer", type=Path)
    parser.add_argument("--scale-hlt-region-normalizer", type=Path)
    parser.add_argument("--global-determinism", type=Path)
    parser.add_argument("--validation-tree-root", type=Path)
    parser.add_argument("--scale-offline-input", type=Path)
    parser.add_argument("--scale-hlt-input", action="append", default=[])
    parser.add_argument("--scale-offline-tree", type=Path)
    parser.add_argument("--scale-hlt-tree", action="append", default=[])
    parser.add_argument("--tree-backend-manifest", type=Path)
    parser.add_argument("--teacher-adapter-config", type=Path)
    parser.add_argument("--scale-train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--design-confirm-cache", action="append", default=[])
    parser.add_argument("--scale-train-tree", action="append", default=[])
    parser.add_argument("--scale-train-labels", type=Path)
    parser.add_argument("--val-stop-labels", type=Path)
    parser.add_argument("--design-confirm-labels", type=Path)
    parser.add_argument("--stage-d-loader-root", type=Path)
    parser.add_argument(
        "--scale-native-relation", action="append", default=[]
    )
    parser.add_argument("--design-confirm-native-relation", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    if plan.get("source") != campaign["source"]:
        raise ValueError("scale execution plan source differs")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "teacher_id": args.teacher_id,
                    "target_id": args.target_id,
                    "graph_id": args.graph_id,
                    "seed": args.seed,
                    "executed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "teacher":
        required = (
            args.teacher_id,
            args.model_contract_o_base,
            args.model_contract_o_fullrel,
            args.offline_manifest,
            args.validation_partition,
            args.screening_registry,
            args.relation_registry,
            args.scale_offline_relation_normalizer,
            args.scale_offline_region_normalizer,
            args.global_determinism,
            args.scale_train_labels,
        )
        if any(value is None for value in required):
            raise ValueError("scale teacher execution inputs are incomplete")
        return _teacher(args, root)
    if args.mode == "teacher-lock":
        return _teacher_lock(args, root)
    if args.mode == "teacher-inference":
        if args.teacher_id is None:
            raise ValueError("scale teacher inference lacks teacher ID")
        return _teacher_inference(args, root)
    if args.mode == "target":
        if (
            args.scale_offline_relation_normalizer is None
            or args.scale_hlt_relation_normalizer is None
        ):
            raise ValueError("scale target execution lacks relation normalizers")
        return _target(args, root, campaign, plan)
    if args.mode == "graph":
        return _graph(args, root)
    raise AssertionError(f"unreachable scale mode {args.mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
