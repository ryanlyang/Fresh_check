from pathlib import Path

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_REQUIRED_CAMPAIGN_VARIANT_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "sbatch" / "submit_adaptive_binary_kt8_screen_tigris.sh"


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
    assert "ABPH_RECONSTRUCTOR_PARALLELISM=ddp8" in source
    assert "--nodes=8 --ntasks=8" in source
    assert 'tagger_dependency=("--dependency=afterok:${renderer_job}")' in source
    assert "ABPH_TAGGER_PARALLELISM=single" in source
    assert "--job-name=abph_kt8_tagger" in source
    assert "CONFIRM_FINAL_TEST" not in source
