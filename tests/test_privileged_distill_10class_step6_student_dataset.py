from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM
from teacher_logit_reco.privileged_distill_10class import (
    PD10_NUM_CLASSES,
    PD10_STEP4_EXPERIMENT_STEP,
    PD10_STEP5_EXPERIMENT_STEP,
    PD10_STEP6_EXPERIMENT_STEP,
    PD10_STUDENT_ALLOWED_INPUTS,
    PD10_STUDENT_DATASET_CONTRACT,
    PD10_TEACHER_LOGIT_CACHE_CONTRACT,
    PD10_DUAL_VIEW_LOGIT_MODEL_NAME,
    PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
    PD10StudentDistillationDataset,
    align_hlt_view_to_teacher_block,
    assert_pd10_student_batch_hlt_only,
    build_pd10_student_dataset_from_view,
    collate_pd10_student_batch,
    load_pd10_student_teacher_block,
    make_pd10_student_data_loader,
    move_pd10_student_batch_to_device,
    pd10_part_teacher_model_name,
)


def make_hlt_view(labels=(0, 1, 2, 3), *, split="model_val") -> JetView:
    labels_np = np.asarray(labels, dtype=np.int64)
    tokens = np.zeros((len(labels_np), 3, RAW_TOKEN_DIM), dtype=np.float32)
    for row in range(len(labels_np)):
        tokens[row, :, 0] = float(row + 1)
        tokens[row, :, 1] = 0.01 * float(row + 1)
        tokens[row, :, 2] = 0.02 * float(row + 1)
        tokens[row, :, 3] = float(row + 2)
    mask = np.ones((len(labels_np), 3), dtype=bool)
    jet_ids = [
        JetIdentity(file=f"class{int(label)}.root", entry=100 + index, label=int(label))
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


def make_teacher_block(
    target: str,
    view: JetView,
    *,
    indices=None,
    logits_offset: float = 0.0,
) -> PredictionBlock:
    if indices is None:
        indices = np.arange(len(view.labels), dtype=np.int64)
    labels = np.asarray(view.labels)[indices].astype(np.int64)
    logits = np.full((len(labels), PD10_NUM_CLASSES), -1.0 + logits_offset, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 2.0 + logits_offset
    model_name = (
        PD10_DUAL_VIEW_LOGIT_MODEL_NAME
        if target == "dual_view"
        else pd10_part_teacher_model_name(target)
    )
    metadata = {
        "teacher_target": target,
        "model_name": model_name,
        "split": view.split,
        "num_classes": PD10_NUM_CLASSES,
        "student_deployment_inputs": "HLT_only",
        "teacher_logits_train_time_only": True,
    }
    if target == "dual_view":
        metadata.update(
            {
                "contract": PD10_DUAL_VIEW_LOGIT_TEACHER_CONTRACT,
                "experiment_step": PD10_STEP5_EXPERIMENT_STEP,
                "source_view": "hlt_plus_offline_teacher_logits",
                "allowed_inputs": "HLT_plus_offline_train_time_privileged",
                "uses_hlt_teacher_logits": True,
                "uses_offline_teacher_logits": True,
                "uses_raw_offline_particles": False,
                "input_hlt_prediction_content_hash": "hlt-pred-hash",
                "input_offline_prediction_content_hash": "offline-pred-hash",
            }
        )
    else:
        metadata.update(
            {
                "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
                "experiment_step": PD10_STEP4_EXPERIMENT_STEP,
            }
        )
        if target == "hlt":
            metadata.update(
                {
                    "source_view": "fixed_hlt",
                    "allowed_inputs": "HLT_only",
                    "hlt_content_hash": "teacher-hlt-hash",
                    "no_offline_inputs_loaded": True,
                }
            )
        else:
            metadata.update(
                {
                    "source_view": "offline",
                    "allowed_inputs": "offline_only_train_time_privileged",
                    "no_hlt_inputs_loaded": True,
                    "offline_privileged_inputs_loaded": True,
                }
            )
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=[view.jet_ids[int(index)] for index in indices],
        metadata=metadata,
    )


class PD10Step6StudentDatasetTests(unittest.TestCase):
    def test_teacher_logits_align_hlt_rows_by_jet_identity_subset(self):
        view = make_hlt_view(labels=(0, 1, 2, 3))
        teacher = make_teacher_block("offline", view, indices=np.asarray([2, 0], dtype=np.int64))

        aligned = align_hlt_view_to_teacher_block(view, teacher, teacher_target="offline")
        dataset = PD10StudentDistillationDataset(view, teacher_target="offline", teacher_block=teacher)

        self.assertEqual([jet.key() for jet in aligned.jet_ids], [view.jet_ids[2].key(), view.jet_ids[0].key()])
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset.has_teacher_logits)
        self.assertTrue(np.array_equal(dataset.labels, teacher.labels))
        self.assertEqual([jet.key() for jet in dataset.jet_ids], [jet.key() for jet in teacher.jet_ids])
        self.assertEqual(dataset.to_metadata()["contract"], PD10_STUDENT_DATASET_CONTRACT)
        self.assertEqual(dataset.to_metadata()["experiment_step"], PD10_STEP6_EXPERIMENT_STEP)
        self.assertFalse(dataset.to_metadata()["returns_offline_particles"])

    def test_collate_returns_hlt_only_part_inputs_teacher_logits_and_jet_ids(self):
        view = make_hlt_view(labels=(0, 1, 2))
        teacher = make_teacher_block("hlt", view)
        dataset = PD10StudentDistillationDataset(view, teacher_target="hlt", teacher_block=teacher)

        batch = collate_pd10_student_batch([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["points"].shape[:2]), (2, 2))
        self.assertEqual(tuple(batch["features"].shape[:2]), (2, 17))
        self.assertEqual(tuple(batch["lorentz_vectors"].shape[:2]), (2, 4))
        self.assertEqual(tuple(batch["mask"].shape), (2, 1, 3))
        self.assertEqual(tuple(batch["teacher_logits"].shape), (2, PD10_NUM_CLASSES))
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertEqual(batch["jet_keys"], [view.jet_ids[0].key(), view.jet_ids[1].key()])
        self.assertEqual(batch["student_allowed_inputs"], PD10_STUDENT_ALLOWED_INPUTS)
        self.assertFalse(batch["returns_offline_particles"])
        self.assertNotIn("offline_tokens", batch)
        assert_pd10_student_batch_hlt_only(batch)

    def test_missing_teacher_logits_fail_loudly(self):
        view = make_hlt_view(labels=(0, 1))

        with self.assertRaises(FileNotFoundError):
            PD10StudentDistillationDataset(view, teacher_target="offline")
        with self.assertRaises(FileNotFoundError):
            build_pd10_student_dataset_from_view(view, teacher_target="hlt", teacher_logit_dir=None)

    def test_alignment_refuses_mismatched_teacher_rows(self):
        view = make_hlt_view(labels=(0, 1))
        teacher = make_teacher_block("hlt", view)
        teacher.jet_ids[1] = JetIdentity(file="missing.root", entry=999, label=1)

        with self.assertRaises(ValueError):
            PD10StudentDistillationDataset(view, teacher_target="hlt", teacher_block=teacher)

    def test_inference_export_path_does_not_require_teacher_logits(self):
        view = make_hlt_view(labels=(0, 1, 2))
        dataset = PD10StudentDistillationDataset(view, teacher_target="none", max_jets=2)

        self.assertFalse(dataset.has_teacher_logits)
        self.assertNotIn("teacher_logits", dataset[0])
        batch = collate_pd10_student_batch([dataset[0], dataset[1]])
        self.assertNotIn("teacher_logits", batch)
        self.assertFalse(batch["has_teacher_logits"])
        self.assertFalse(dataset.to_metadata()["inference_export_requires_teacher_logits"])
        assert_pd10_student_batch_hlt_only(batch)

    def test_loader_and_move_batch_preserve_non_tensor_jet_metadata(self):
        view = make_hlt_view(labels=(0, 1, 2))
        dataset = PD10StudentDistillationDataset(view, teacher_target="none")
        loader = make_pd10_student_data_loader(dataset, batch_size=2, shuffle=False, seed=7)
        batch = next(iter(loader))
        moved = move_pd10_student_batch_to_device(batch, "cpu")

        self.assertEqual(moved["jet_keys"], [view.jet_ids[0].key(), view.jet_ids[1].key()])
        self.assertEqual(moved["labels"].device.type, "cpu")
        self.assertIsInstance(moved["jet_files"][0], str)

    def test_load_teacher_blocks_supports_hlt_offline_and_dual_view_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = make_hlt_view(labels=(0, 1, 2))
            for target in ("hlt", "offline", "dual_view"):
                block = make_teacher_block(target, view)
                save_prediction_block(block, root)

            hlt = load_pd10_student_teacher_block(root, "hlt", view.split)
            offline = load_pd10_student_teacher_block(root, "offline", view.split)
            dual = load_pd10_student_teacher_block(root, "dual_view", view.split)

            self.assertEqual(hlt.logits.shape, (3, PD10_NUM_CLASSES))
            self.assertEqual(offline.metadata["teacher_target"], "offline")
            self.assertEqual(dual.metadata["teacher_target"], "dual_view")


if __name__ == "__main__":
    unittest.main()
