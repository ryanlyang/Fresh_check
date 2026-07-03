from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import save_prediction_block
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.privileged_distill_10class import (
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_STUDENT_REPRESENTATION_WRAPPER_CONTRACT,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    PD10ParticleDualViewTeacherCacheConfig,
    PD10StudentDistillationDataset,
    PD10StudentRepresentationAdapter,
    PD10StudentRepresentationOutput,
    PD10TeacherRepresentationCacheConfig,
    assert_pd10_student_batch_hlt_only,
    build_pd10_particle_dual_view_logit_block,
    build_pd10_student_dataset_from_view,
    build_pd10_teacher_representation_block,
    collate_pd10_student_batch,
    load_pd10_student_teacher_block,
    load_pd10_student_teacher_representation_block,
    pd10_student_forward_with_optional_representation,
    pd10_student_representation_loss,
    save_pd10_teacher_representation_block,
)


def make_hlt_view(labels=(0, 1, 2), *, split="model_val") -> JetView:
    labels_np = np.asarray(labels, dtype=np.int64)
    n_jets = int(labels_np.shape[0])
    tokens = np.zeros((n_jets, 4, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.ones((n_jets, 4), dtype=bool)
    for jet in range(n_jets):
        for particle in range(4):
            pt = 20.0 + float(jet) + 0.5 * float(particle)
            eta = 0.01 * float(jet + particle)
            phi = -0.02 * float(particle)
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt + 1.0
            tokens[jet, particle, 4] = (-1.0, 0.0, 1.0, 0.0)[particle]
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 10] = 0.01 * float(particle)
            tokens[jet, particle, 11] = 0.02
            tokens[jet, particle, 12] = 0.03 * float(jet)
            tokens[jet, particle, 13] = 0.04
    jet_ids = [
        JetIdentity(file=f"pd10_class{int(label)}.root", entry=10_000 + index, label=int(label))
        for index, label in enumerate(labels_np)
    ]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels_np,
        jet_ids=jet_ids,
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}-hlt-hash"},
    )


def make_particle_cache_config(root: Path) -> PD10ParticleDualViewTeacherCacheConfig:
    checkpoint = root / "particle_dual_view.pt"
    checkpoint.write_bytes(b"checkpoint")
    return PD10ParticleDualViewTeacherCacheConfig(
        checkpoint=str(checkpoint),
        manifest_path=str(root / "split_manifest.json.gz"),
        hlt_cache_dir=str(root / "hlt_cache"),
        logit_output_dir=str(root / "teacher_logits"),
        representation_output_dir=str(root / "teacher_representations"),
        splits=("model_val",),
    )


def make_particle_logit_block(view: JetView, root: Path):
    cfg = make_particle_cache_config(root)
    logits = np.full((len(view.labels), PD10_NUM_CLASSES), -2.0, dtype=np.float32)
    logits[np.arange(len(view.labels)), view.labels] = 3.0
    return build_pd10_particle_dual_view_logit_block(
        cfg,
        view.split,
        logits=logits,
        labels=view.labels,
        jet_ids=view.jet_ids,
        source_metadata={"hlt_content_hash": "hlt-hash", "source_manifest_hash": "manifest-hash"},
        checkpoint_payload={"epoch": 1, "experiment_step": "unit_test"},
    )


def make_representation_block(view: JetView, root: Path, *, reverse: bool = False):
    rep_cfg = PD10TeacherRepresentationCacheConfig(
        teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
        output_dir=str(root / "teacher_representations"),
        splits=(view.split,),
    )
    order = np.arange(len(view.labels), dtype=np.int64)
    if reverse:
        order = order[::-1]
    reps = np.arange(len(view.labels) * PD10_REPRESENTATION_DIM, dtype=np.float32).reshape(
        len(view.labels),
        PD10_REPRESENTATION_DIM,
    )
    reps = reps[order]
    return build_pd10_teacher_representation_block(
        rep_cfg,
        view.split,
        representations=reps,
        labels=np.asarray(view.labels)[order],
        jet_ids=[view.jet_ids[int(index)] for index in order],
        source_metadata={"hlt_content_hash": "hlt-hash", "source_manifest_hash": "manifest-hash"},
        extra_metadata={"checkpoint_sha256": "abc123"},
    )


class PD10V2Step3StudentRepresentationTests(unittest.TestCase):
    def test_student_dataset_collates_logits_and_teacher_representations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = make_hlt_view(labels=(0, 1, 2))
            logit_block = make_particle_logit_block(view, root)
            rep_block = make_representation_block(view, root)

            dataset = PD10StudentDistillationDataset(
                view,
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                teacher_block=logit_block,
                teacher_representation_block=rep_block,
            )
            self.assertTrue(dataset.has_teacher_logits)
            self.assertTrue(dataset.has_teacher_representations)
            self.assertEqual(dataset.to_metadata()["teacher_target"], PD10_TEACHER_PARTICLE_DUAL_VIEW)
            self.assertTrue(dataset.to_metadata()["returns_teacher_logits"])
            self.assertTrue(dataset.to_metadata()["returns_teacher_representations"])
            self.assertFalse(dataset.to_metadata()["inference_export_requires_teacher_representations"])

            batch = collate_pd10_student_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["teacher_logits"].shape), (2, PD10_NUM_CLASSES))
            self.assertEqual(tuple(batch["teacher_representations"].shape), (2, PD10_REPRESENTATION_DIM))
            self.assertTrue(batch["has_teacher_logits"])
            self.assertTrue(batch["has_teacher_representations"])
            assert_pd10_student_batch_hlt_only(batch)

    def test_representation_only_dataset_does_not_require_teacher_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = make_hlt_view(labels=(0, 1, 2))
            rep_block = make_representation_block(view, root)

            dataset = PD10StudentDistillationDataset(
                view,
                teacher_target="particle_dual_view",
                teacher_representation_block=rep_block,
                max_jets=2,
            )
            self.assertFalse(dataset.has_teacher_logits)
            self.assertTrue(dataset.has_teacher_representations)
            self.assertNotIn("teacher_logits", dataset[0])
            batch = collate_pd10_student_batch([dataset[0], dataset[1]])
            self.assertNotIn("teacher_logits", batch)
            self.assertEqual(tuple(batch["teacher_representations"].shape), (2, PD10_REPRESENTATION_DIM))
            assert_pd10_student_batch_hlt_only(batch)

    def test_logit_and_representation_blocks_must_share_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = make_hlt_view(labels=(0, 1, 2))
            logit_block = make_particle_logit_block(view, root)
            reversed_rep_block = make_representation_block(view, root, reverse=True)

            with self.assertRaises(ValueError):
                PD10StudentDistillationDataset(
                    view,
                    teacher_target="particle_dual_view",
                    teacher_block=logit_block,
                    teacher_representation_block=reversed_rep_block,
                )

    def test_student_loaders_support_particle_logits_and_representation_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = make_hlt_view(labels=(0, 1, 2))
            logit_cfg = make_particle_cache_config(root)
            logit_block = make_particle_logit_block(view, root)
            rep_block = make_representation_block(view, root)
            save_prediction_block(logit_block, logit_cfg.logit_output_dir)
            save_pd10_teacher_representation_block(rep_block, root / "teacher_representations")

            loaded_logits = load_pd10_student_teacher_block(
                logit_cfg.logit_output_dir,
                "particle_dual_view",
                view.split,
            )
            loaded_reps = load_pd10_student_teacher_representation_block(
                root / "teacher_representations",
                "particle_dual_view",
                view.split,
            )
            dataset = build_pd10_student_dataset_from_view(
                view,
                teacher_target="particle_dual_view",
                teacher_logit_dir=logit_cfg.logit_output_dir,
                teacher_representation_dir=root / "teacher_representations",
            )

            self.assertEqual(loaded_logits.logits.shape, (3, PD10_NUM_CLASSES))
            self.assertEqual(loaded_reps.representations.shape, (3, PD10_REPRESENTATION_DIM))
            self.assertTrue(dataset.has_teacher_logits)
            self.assertTrue(dataset.has_teacher_representations)

    def test_student_representation_adapter_keeps_logits_only_forward_default(self):
        try:
            torch = require_torch()
        except ImportError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

        class TinyStudent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.classifier = torch.nn.Linear(17, PD10_NUM_CLASSES)
                self.config = {"model": "tiny_unit_test_student"}

            def forward(self, points, features, lorentz_vectors, mask):
                mask_float = mask.float()
                pooled = (features.float() * mask_float).sum(dim=-1) / torch.clamp(mask_float.sum(dim=-1), min=1.0)
                return self.classifier(pooled)

        view = make_hlt_view(labels=(0, 1))
        dataset = PD10StudentDistillationDataset(view, teacher_target="none")
        batch = collate_pd10_student_batch([dataset[0], dataset[1]])
        model = PD10StudentRepresentationAdapter(TinyStudent(), representation_dim=PD10_REPRESENTATION_DIM)

        logits = pd10_student_forward_with_optional_representation(model, batch, return_representation=False)
        output = pd10_student_forward_with_optional_representation(model, batch, return_representation=True)

        self.assertEqual(tuple(logits.shape), (2, PD10_NUM_CLASSES))
        self.assertIsInstance(output, PD10StudentRepresentationOutput)
        self.assertEqual(tuple(output.logits.shape), (2, PD10_NUM_CLASSES))
        self.assertEqual(tuple(output.student_repr_raw.shape), (2, 17))
        self.assertEqual(tuple(output.student_repr_projected.shape), (2, PD10_REPRESENTATION_DIM))
        norms = torch.linalg.vector_norm(output.student_repr_projected.detach(), dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5))
        self.assertEqual(model.config["representation_wrapper_contract"], PD10_STUDENT_REPRESENTATION_WRAPPER_CONTRACT)

        zero_loss = pd10_student_representation_loss(
            output.student_repr_projected,
            output.student_repr_projected.detach(),
        )
        self.assertLess(float(zero_loss.detach().cpu().item()), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
