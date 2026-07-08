import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO_ROOT / "sbatch"


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


class HLTTriviewDebugSlurmTests(unittest.TestCase):
    def test_source_models_default_to_scratch_stable_training(self):
        runner = _read("run_pd10_train_hlt_triview_source.sh")
        submitter = _read("submit_pd10_hlt_triview_debug.sh")

        self.assertIn('PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:=}', runner)
        self.assertIn('PD10_HLT_TRIVIEW_SOURCE_LR:=0.001', runner)
        self.assertIn('PD10_HLT_TRIVIEW_SOURCE_NO_AMP:=1', runner)
        self.assertIn('--warm-start-checkpoint "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}"', runner)

        self.assertIn('PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START:=0', submitter)
        self.assertIn('PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:=}', submitter)
        self.assertIn('requires PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT', submitter)
        self.assertIn('source_warm_start_checkpoint=${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:-none}', submitter)


if __name__ == "__main__":
    unittest.main()
