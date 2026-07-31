from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.bootstrap_retb_input_tasks import (
    build_stage_a_task_manifests,
    main as bootstrap_main,
)
from teacher_logit_reco.relation_expert_token_bridge import (
    build_production_graph,
    validate_task_manifest_for_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (
    bind_fitted_normalizer,
    bind_fitted_region_normalizer,
    build_stage_a_contract_bundle,
    build_stage_a_input_audit,
    build_stage_a_normalizer_bundle,
    load_authenticated_tree_selection,
    padding_is_exact_zero,
    validate_stage_a_contract_bundle,
    validate_stage_a_input_audit,
    validate_stage_a_normalizer_bundle,
)
from teacher_logit_reco.relational_part import (
    build_reference_tree,
    finalize_tree_split,
    fit_region_normalization,
    fit_relation_normalization,
    write_tree_shard,
)


ROOT = Path(__file__).resolve().parents[1]


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _campaign(*, miniature: bool) -> dict:
    parent_names = (
        "artifact_layout",
        "final_select_label_manifest",
        "global_determinism",
        "hlt_replica_manifest",
        "raw_input_schema",
        "scale_train_manifest",
        "split_audit",
        "split_manifest",
        "storage_measurements",
        "validation_partition_manifest",
    )
    return build_campaign_spec(
        campaign_id="retb_stage_a_test",
        campaign_profile=(
            "miniature_test" if miniature else "production_500k_scale3m"
        ),
        source_snapshot=_source(),
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parent_names)
        },
        run_registry_hashes={"expert": "f" * 64},
    )


def _contracts(campaign: dict, *, miniature: bool) -> dict:
    return build_stage_a_contract_bundle(
        campaign_spec=campaign,
        model_train_identity_count=20 if miniature else 500_000,
        scale_train_identity_count=40 if miniature else 3_000_000,
        source_snapshot=_source(),
    )


def _sample(
    jets: int = 4, particles: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.ones((jets, particles), dtype=bool)
    vectors = np.zeros((jets, particles, 4), dtype=np.float64)
    identities = [f"stage-a-{index}" for index in range(jets)]
    for row in range(jets):
        pt = np.asarray([12.0, 8.0, 5.0, 3.0, 2.0, 1.0]) + row * 0.1
        eta = np.asarray([0.0, 0.04, 0.3, -0.5, 0.8, -1.0])
        phi = np.asarray([0.0, 0.03, 0.5, -1.0, 2.0, -2.5])
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px**2 + py**2 + pz**2 + (0.2 + 0.1 * row) ** 2)
        vectors[row] = np.stack((px, py, pz, energy), axis=1)
        tokens[row, :, :4] = np.stack((pt, eta, phi, energy), axis=1)
        tokens[row, :, 4] = np.asarray([1, 0, 0, -1, 1, -1])
        for particle in range(particles):
            tokens[row, particle, 5 + particle % 5] = 1.0
        tokens[row, :, 10] = np.linspace(-0.02, 0.03, particles)
        tokens[row, :, 11] = 0.002 + row * 0.0001
        tokens[row, :, 12] = np.linspace(-0.03, 0.05, particles)
        tokens[row, :, 13] = 0.004 + row * 0.0001
    return tokens, mask, vectors, identities


def _fit_bound_domain(
    *,
    logical_domain: str,
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: list[str],
    trees: list[dict],
    contracts: dict,
    campaign: dict,
    view_hashes: list[str],
    tree_hashes: list[str],
) -> tuple[dict, dict]:
    recipe = contracts["normalizer_population_registry"]["recipes"][
        logical_domain
    ]
    relation = fit_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=contracts["normalization_contract"],
        relation_registry=contracts["relation_family_registry"],
        raw_input_schema=contracts["inherited_raw_input_schema"],
        hlt_binding_sha256=contracts["stage_a_contract_bundle"][
            "content_hash"
        ],
        source_manifest_sha256=recipe["identity_manifest_sha256"],
        hlt_model_train_content_sha256="c" * 64,
    )
    relation = bind_fitted_normalizer(
        relation,
        logical_domain=logical_domain,
        population_recipe=recipe,
        identity_manifest_sha256=recipe["identity_manifest_sha256"],
        view_content_sha256s=view_hashes,
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=_source(),
    )
    region = fit_region_normalization(
        tokens,
        mask,
        identities,
        trees,
        relation_normalization_artifact=relation,
        angular_tree_resource_sha256=contracts["angular_tree_resource"][
            "content_hash"
        ],
    )
    region = bind_fitted_region_normalizer(
        region,
        relation_normalizer=relation,
        logical_domain=logical_domain,
        population_recipe=recipe,
        tree_manifest_sha256s=tree_hashes,
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=_source(),
    )
    return relation, region


def test_stage_a_contracts_and_numeric_normalizers_are_domain_bound() -> None:
    campaign = _campaign(miniature=True)
    contracts = _contracts(campaign, miniature=True)
    validate_stage_a_contract_bundle(contracts, campaign_spec=campaign)
    assert contracts["stage_a_contract_bundle"]["normalizer_sampling"][
        "shared_hlt_sample_is_replica_balanced"
    ] is True
    tokens, mask, vectors, identities = _sample()
    trees = [
        build_reference_tree(vectors[index], tokens[index], mask[index])
        for index in range(len(tokens))
    ]
    offline_relation, offline_region = _fit_bound_domain(
        logical_domain="offline_500k",
        tokens=tokens,
        mask=mask,
        identities=identities,
        trees=trees,
        contracts=contracts,
        campaign=campaign,
        view_hashes=["1" * 64],
        tree_hashes=["2" * 64],
    )
    hlt_tokens = np.repeat(tokens, 4, axis=0)
    hlt_mask = np.repeat(mask, 4, axis=0)
    hlt_trees = [
        tree for tree in trees for _replica in range(4)
    ]
    hlt_identities = [
        f"{identity}@retb_replica_{replica}"
        for identity in identities
        for replica in range(4)
    ]
    hlt_relation, hlt_region = _fit_bound_domain(
        logical_domain="shared_hlt_500k",
        tokens=hlt_tokens,
        mask=hlt_mask,
        identities=hlt_identities,
        trees=hlt_trees,
        contracts=contracts,
        campaign=campaign,
        view_hashes=[f"{index + 3:064x}" for index in range(4)],
        tree_hashes=[f"{index + 7:064x}" for index in range(4)],
    )
    bundle = build_stage_a_normalizer_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        stage_a_contract_bundle_sha256=contracts[
            "stage_a_contract_bundle"
        ]["content_hash"],
        population_registry=contracts["normalizer_population_registry"],
        offline_relation=offline_relation,
        offline_region=offline_region,
        shared_hlt_relation=hlt_relation,
        shared_hlt_region=hlt_region,
        source_snapshot=_source(),
    )
    validate_stage_a_normalizer_bundle(
        bundle,
        artifacts={
            "offline_500k_relation": offline_relation,
            "offline_500k_region": offline_region,
            "shared_hlt_500k_relation": hlt_relation,
            "shared_hlt_500k_region": hlt_region,
        },
    )
    assert offline_relation["logical_domain"] == "offline_500k"
    assert hlt_relation["logical_domain"] == "shared_hlt_500k"
    assert offline_relation["content_hash"] != hlt_relation["content_hash"]
    assert bundle["offline_and_hlt_interchangeable"] is False


def test_stage_a_relation_audit_rebuilds_every_owner_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import audit_retb_stage_a_inputs as audit_script

    campaign = _campaign(miniature=True)
    contracts = _contracts(campaign, miniature=True)
    tokens, mask, vectors, identities = _sample()
    trees = [
        build_reference_tree(vectors[index], tokens[index], mask[index])
        for index in range(len(tokens))
    ]
    relation, region = _fit_bound_domain(
        logical_domain="offline_500k",
        tokens=tokens,
        mask=mask,
        identities=identities,
        trees=trees,
        contracts=contracts,
        campaign=campaign,
        view_hashes=["1" * 64],
        tree_hashes=["2" * 64],
    )

    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat(
            [base + index for index in range(num_outputs)], dim=1
        )

    fake_weaver = SimpleNamespace(pairwise_lv_fts=pairwise)
    monkeypatch.setattr(
        audit_script.importlib,
        "import_module",
        lambda _name: fake_weaver,
    )
    views = audit_script._relation_owner_views(
        tokens=tokens,
        mask=mask,
        trees=trees,
        relation_normalizer=relation,
        region_normalizer=region,
        source_view="stage_a_test",
    )
    assert list(views) == [
        "standard_four",
        "PT",
        "TRACK",
        "PID",
        "CHARGE",
        "DENSITY",
        "REGION",
    ]
    assert views["standard_four"].shape == (4, 4, 6, 6)
    assert views["REGION"].shape == (4, 41, 6, 6)
    assert all(np.isfinite(value).all() for value in views.values())


def test_stage_a_bootstrap_builds_complete_production_and_miniature_manifests(
    tmp_path: Path,
) -> None:
    for miniature in (True, False):
        root = tmp_path / ("miniature" if miniature else "production")
        campaign = _campaign(miniature=miniature)
        graph = build_production_graph(
            campaign_root=root,
            campaign_id=campaign["campaign_id"],
            source_commit="a" * 40,
            source_status_sha256="b" * 64,
            storage_measurements_sha256=campaign[
                "parent_artifact_hashes"
            ]["storage_measurements"],
            miniature=miniature,
        )
        contracts = _contracts(campaign, miniature=miniature)
        manifests = build_stage_a_task_manifests(
            campaign=campaign,
            graph=graph,
            campaign_root=root,
            data_dir="/data/jetclass",
            stage_a_contracts=contracts,
        )
        assert set(manifests) == {
            "offline_input_cache",
            "hlt_v3_cache",
            "region_tree_cache",
            "region_tree_finalize",
            "normalizers_500k",
            "input_audit",
        }
        assert manifests["offline_input_cache"]["task_count"] == 6
        assert manifests["hlt_v3_cache"]["task_count"] == 12
        assert (
            manifests["region_tree_cache"]["task_count"]
            == 9
        )
        assert all(
            manifests[name]["task_count"] == 1
            for name in (
                "region_tree_finalize",
                "normalizers_500k",
                "input_audit",
            )
        )
        for manifest in manifests.values():
            validate_task_manifest_for_graph(
                manifest,
                production_graph=graph,
                campaign_root=root,
                repo_root=ROOT,
            )
        hlt_commands = [
            row["argv"] for row in manifests["hlt_v3_cache"]["rows"]
        ]
        assert all("--realization-policy" in argv for argv in hlt_commands)
        tree_commands = [
            row["argv"] for row in manifests["region_tree_cache"]["rows"]
            if "--replica-id" in row["argv"]
        ]
        assert all("--realization-policy" in argv for argv in tree_commands)
        production_tree_outputs = sum(
            len(row["expected_outputs"])
            for row in manifests["region_tree_cache"]["rows"]
        )
        assert production_tree_outputs == (
            18 if miniature else 540
        )


def test_stage_a_bootstrap_cli_dry_run_resolves_every_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = source_snapshot(ROOT)
    campaign_root = tmp_path / "retb_stage_a_cli"
    campaign_root.mkdir()
    parent_names = (
        "artifact_layout",
        "final_select_label_manifest",
        "global_determinism",
        "hlt_replica_manifest",
        "raw_input_schema",
        "scale_train_manifest",
        "split_audit",
        "split_manifest",
        "storage_measurements",
        "validation_partition_manifest",
    )
    campaign = build_campaign_spec(
        campaign_id="retb_stage_a_cli",
        campaign_profile="miniature_test",
        source_snapshot=source,
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parent_names)
        },
        run_registry_hashes={"runs": "f" * 64},
    )
    write_immutable_json(campaign_root / "campaign_spec.json", campaign)
    graph = build_production_graph(
        campaign_root=campaign_root,
        campaign_id=campaign["campaign_id"],
        source_commit=str(source["source_commit"]),
        source_status_sha256=str(source["source_status_sha256"]),
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    graph_path = campaign_root / "production_graph.json"
    write_immutable_json(graph_path, graph)
    arguments = [
        "--campaign-root",
        str(campaign_root),
        "--production-graph",
        str(graph_path),
        "--data-dir",
        "/data/jetclass",
    ]
    assert bootstrap_main([*arguments, "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"region_tree_cache": 9' in output
    assert '"input_audit": 1' in output
    assert not (campaign_root / "job_ledgers" / "tasks").exists()
    assert bootstrap_main(arguments) == 0
    assert {
        path.stem
        for path in (campaign_root / "job_ledgers" / "tasks").glob("*.json")
    } == {
        "offline_input_cache",
        "hlt_v3_cache",
        "region_tree_cache",
        "region_tree_finalize",
        "normalizers_500k",
        "input_audit",
            "stage_b_offline_experts",
            "stage_c_offline_expert_confirmations",
            "stage_c_offline_fusion_cache",
            "stage_c_offline_fusions",
        "stage_d_hlt_experts",
        "stage_d_hlt_fusions",
        "stage_e_bridge_pilots",
    }
    assert (
        campaign_root / "registry" / "retb_stage_a_contract_bundle.json"
    ).is_file()
    assert (
        campaign_root / "registry" / "retb_static_experiment_bundle.json"
    ).is_file()


def test_nonarray_manifest_requires_manifest_driven_graph_node(
    tmp_path: Path,
) -> None:
    campaign = _campaign(miniature=True)
    graph = build_production_graph(
        campaign_root=tmp_path,
        campaign_id=campaign["campaign_id"],
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    contracts = _contracts(campaign, miniature=True)
    manifests = build_stage_a_task_manifests(
        campaign=campaign,
        graph=graph,
        campaign_root=tmp_path,
        data_dir="/data/jetclass",
        stage_a_contracts=contracts,
    )
    validate_task_manifest_for_graph(
        manifests["input_audit"],
        production_graph=graph,
        campaign_root=tmp_path,
        repo_root=ROOT,
    )
    direct = {
        **manifests["input_audit"],
        "node_id": "campaign_bootstrap",
    }
    direct["rows"] = [
        {
            **direct["rows"][0],
            "task_id": "campaign_bootstrap:0",
        }
    ]
    direct.pop("content_hash")
    from teacher_logit_reco.relation_expert_token_bridge.contracts import (
        with_content_hash,
    )

    direct = with_content_hash(direct)
    with pytest.raises(ValueError, match="production graph"):
        validate_task_manifest_for_graph(
            direct,
            production_graph=graph,
            campaign_root=tmp_path,
            repo_root=ROOT,
        )


def test_stage_a_input_audit_contract_requires_exact_safe_views() -> None:
    values = np.zeros((2, 3, 14), dtype=np.float32)
    mask = np.asarray([[True, True, False], [True, False, False]])
    assert padding_is_exact_zero(values, mask)
    row = {
        "view_id": "offline:model_train",
        "logical_role": "model_train",
        "metadata_sha256": "1" * 64,
        "identity_manifest_sha256": "2" * 64,
        "event_count": 2,
        "particle_capacity": 128,
        "raw_particle_field_count": 14,
        "tokens_dtype": "float32",
        "mask_dtype": "bool",
        "finite_valid_tokens": True,
        "padding_zero_exact": True,
        "identities_unique": True,
        "identity_order_sha256": "3" * 64,
        "valid_particle_count": 3,
        "all_empty_jet_count": 0,
    }
    audit = build_stage_a_input_audit(
        campaign_spec_sha256="4" * 64,
        offline_views=[row],
        hlt_views=[
            {
                **row,
                "view_id": "hlt:model_train:r0:R_MULTI",
                "replica_id": 0,
                "realization_policy": "R_MULTI",
            }
        ],
        tree_index_sha256="5" * 64,
        normalizer_bundle_sha256="6" * 64,
        hlt_v3_degradation_audit_sha256="7" * 64,
        source_snapshot=_source(),
    )
    validate_stage_a_input_audit(audit)
    assert audit["identity_alignment_exact"] is True
    assert audit["validation_or_test_statistics_used_for_normalization"] is False


def test_authenticated_tree_selection_revalidates_npz_bytes(
    tmp_path: Path,
) -> None:
    tokens, mask, vectors, identities = _sample()
    trees = [
        build_reference_tree(vectors[index], tokens[index], mask[index])
        for index in range(len(tokens))
    ]
    metadata_paths = []
    for shard_index, start in enumerate((0, 2)):
        path = (
            tmp_path / "shards" / f"shard_{shard_index:05d}.npz"
        )
        write_tree_shard(
            path,
            trees[start : start + 2],
            identities[start : start + 2],
            hlt_content_sha256="1" * 64,
            tree_resource_sha256="2" * 64,
            backend_manifest_sha256="3" * 64,
        )
        metadata_paths.append(path.with_suffix(".metadata.json"))
    finalize_tree_split(
        tmp_path / "manifest.json",
        metadata_paths,
        split="offline:model_train",
        expected_jet_count=4,
        hlt_content_sha256="1" * 64,
        tree_resource_sha256="2" * 64,
        backend_manifest_sha256="3" * 64,
    )
    selected, manifest = load_authenticated_tree_selection(
        tmp_path, [identities[3], identities[0]]
    )
    assert len(selected) == 2
    assert manifest["jet_count"] == 4
    corrupted = tmp_path / "shards" / "shard_00001.npz"
    corrupted.write_bytes(corrupted.read_bytes() + b"corruption")
    with pytest.raises(FileExistsError, match="stale or incompatible"):
        load_authenticated_tree_selection(tmp_path, [identities[3]])
