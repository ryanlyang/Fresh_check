from pathlib import Path

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_REQUIRED_CAMPAIGN_VARIANT_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "sbatch" / "submit_adaptive_binary_kt8_screen_tigris.sh"
TAGGER_RUNTIME = (
    ROOT
    / "teacher_logit_reco"
    / "adaptive_binary_pseudooffline"
    / "tagger_runtime.py"
)


def test_kt8_screen_does_not_expand_canonical_campaign_graph() -> None:
    assert "D7_kt8_mh4_particles_screen" not in ABPH_REQUIRED_CAMPAIGN_VARIANT_NAMES
    assert "E12_kt8_mh4_dualcross_screen" not in ABPH_REQUIRED_CAMPAIGN_VARIANT_NAMES


def test_kt8_submitter_is_guarded_and_orders_renderer_before_tagger() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    assert "C3_kt_8" in source
    assert "D7_kt8_mh4_particles_screen" in source
    assert "E12_kt8_mh4_dualcross_screen" in source
    assert "selected_checkpoint_provenance" in source
    assert 'provenance.get("file_sha256")' in source
    assert 'checkpoint.get("final_test_loaded") is not False' in source
    assert "ABPH_KT8_CONFIRM_RESUBMIT" in source
    assert 'archives/kt8_screen_resubmit_${stamp}' in source
    assert 'scancel "${previous_jobs[@]}"' in source
    assert "for artifact_kind in runtime_batch_measurements runtime_batch_contracts" in source
    assert 'if [[ ! -f "${renderer_checkpoint}" ]]; then' in source
    assert 'failed_tagger_run="${ABPH_ROOT}/runs/${tagger_variant}"' in source
    assert 'squeue -h -j "${previous_job_csv}"' in source
    assert "ABPH_RECONSTRUCTOR_PARALLELISM=ddp8" in source
    assert "--nodes=8 --ntasks=8" in source
    assert 'tagger_dependency=("--dependency=afterok:${renderer_job}")' in source
    assert "ABPH_TAGGER_PARALLELISM=single" in source
    assert "--job-name=abph_kt8_tagger" in source
    assert "CONFIRM_FINAL_TEST" not in source


def test_tagger_materializes_dynamic_state_before_hash_optimizer_and_ddp() -> None:
    source = TAGGER_RUNTIME.read_text(encoding="utf-8")
    materialize = source.index(
        "dynamic_state_materialization = _materialize_tagger_dynamic_state("
    )
    state_hash = source.index(
        "initial_parameter_state_hash = verify_common_parameter_state(",
        materialize,
    )
    optimizer = source.index("optimizer = torch.optim.AdamW(", state_hash)
    ddp = source.index("ddp_wrapper: Any = build_tagger_ddp_wrapper(", optimizer)
    assert materialize < state_hash < optimizer < ddp
