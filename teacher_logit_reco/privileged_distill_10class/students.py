"""PD10 Step 7 HLT-only student KD training."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import (
    PredictionBlock,
    load_hlt_model_from_checkpoint,
    sanitize_prediction_logits,
    save_prediction_block,
    softmax_np,
)
from jetclass_fresh.hlt_baseline import (
    build_hlt_classifier,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetView
from .config import (
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_REPRESENTATION_BETA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_EXPERIMENT_NAME,
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_REPRESENTATION_MODE_COSINE,
    PD10_REPRESENTATION_MODE_NONE,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_STUDENT_INIT_SCRATCH,
    PD10_STUDENT_INIT_WARM_START,
    PD10_TARGET_CONFIDENCE_WEIGHTED,
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_REP_ONLY,
    PD10_TARGET_TOP3,
    PD10_TEACHER_NONE,
    PD10_TOP_K,
    default_pd10_experiment_layout,
    normalize_pd10_extended_student_target_mode,
    normalize_pd10_extended_teacher_target,
    pd10_base_logit_target_mode,
    pd10_extended_student_variant_name,
    pd10_target_mode_uses_logits,
    pd10_target_mode_uses_representations,
    normalize_pd10_student_init_mode,
    normalize_pd10_representation_mode,
)
from .student_data import (
    PD10StudentDistillationDataset,
    PD10_STUDENT_ALLOWED_INPUTS,
    assert_pd10_student_batch_hlt_only,
    load_pd10_student_dataset,
    make_pd10_student_data_loader,
    move_pd10_student_batch_to_device,
)
from .metrics import pd10_prediction_metrics_from_logits
from .teachers import sha256_file


PD10_STEP7_EXPERIMENT_STEP = "pd10_step7_student_kd_training"
PD10_STUDENT_TRAINING_CONTRACT = "pd10_hlt_only_student_kd_training_v1"
PD10_STUDENT_DEFAULT_SEED = 2205
PD10_STUDENT_SCRATCH_DEFAULT_EPOCHS = 20
PD10_STUDENT_WARM_START_DEFAULT_EPOCHS = 8
PD10_STUDENT_SCRATCH_DEFAULT_LR = 1.0e-3
PD10_STUDENT_WARM_START_DEFAULT_LR = 3.0e-5
PD10_STUDENT_DEFAULT_WEIGHT_DECAY = 1.0e-4
PD10_STUDENT_DEFAULT_BATCH_SIZE = 128
PD10_STUDENT_SCRATCH_DEFAULT_KD_WARMUP_EPOCHS = 3
PD10_STUDENT_WARM_START_DEFAULT_KD_WARMUP_EPOCHS = 1
PD10_STUDENT_PREDICTION_CACHE_CONTRACT = "pd10_hlt_only_student_prediction_cache_v1"
PD10_V2_STEP3_EXPERIMENT_STEP = "pd10_v2_step3_representation_aware_student_training"
PD10_STUDENT_REPRESENTATION_WRAPPER_CONTRACT = "pd10_student_representation_wrapper_v1"


@dataclass(frozen=True)
class PD10StudentRepresentationOutput:
    """Student outputs needed for optional representation KD."""

    logits: Any
    student_repr_raw: Any
    student_repr_projected: Any


class PD10StudentRepresentationAdapter:
    """Factory returning a torch Module that keeps logits-only forward as the default path."""

    def __new__(
        cls,
        base_model,
        *,
        raw_representation_dim: int | None = None,
        representation_dim: int = PD10_REPRESENTATION_DIM,
    ):
        torch = require_torch()

        class _PD10StudentRepresentationAdapterModule(torch.nn.Module):
            def __init__(self, wrapped_model) -> None:
                super().__init__()
                inferred_dim = self._infer_raw_representation_dim(wrapped_model)
                raw_dim = inferred_dim if raw_representation_dim is None else int(raw_representation_dim)
                if raw_dim is None or int(raw_dim) <= 0:
                    raise ValueError(
                        "Representation KD requires a discoverable student ParT hidden dimension; "
                        "pass raw_representation_dim explicitly if the model is custom."
                    )
                if int(representation_dim) <= 0:
                    raise ValueError("representation_dim must be positive")
                self.base_model = wrapped_model
                self.raw_representation_dim = int(raw_dim)
                self.representation_dim = int(representation_dim)
                self.student_representation_projector = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.raw_representation_dim),
                    torch.nn.Linear(self.raw_representation_dim, self.representation_dim),
                )
                self.representation_capture_source = self._representation_capture_source()
                self.config = {
                    **dict(getattr(wrapped_model, "config", {}) or {}),
                    "representation_wrapper_contract": PD10_STUDENT_REPRESENTATION_WRAPPER_CONTRACT,
                    "representation_source": self.representation_capture_source,
                    "raw_representation_dim": self.raw_representation_dim,
                    "representation_dim": self.representation_dim,
                    "student_deployment_inputs": PD10_STUDENT_ALLOWED_INPUTS,
                    "inference_requires_teacher_representations": False,
                }

            @staticmethod
            def _infer_raw_representation_dim(wrapped_model) -> int | None:
                cfg = dict(getattr(wrapped_model, "config", {}) or {})
                embed_dims = cfg.get("embed_dims")
                if isinstance(embed_dims, (list, tuple)) and embed_dims:
                    return int(embed_dims[-1])
                for key in ("embed_dim", "branch_dim", "classifier_hidden_dim"):
                    value = cfg.get(key)
                    if value is not None:
                        return int(value)
                mod = getattr(wrapped_model, "mod", None)
                for module in (
                    getattr(mod, "fc", None),
                    getattr(mod, "head", None),
                    getattr(mod, "classifier", None),
                    getattr(wrapped_model, "classifier", None),
                ):
                    if hasattr(module, "in_features"):
                        return int(module.in_features)
                    if isinstance(module, torch.nn.Sequential) and len(module) > 0:
                        first = module[0]
                        if hasattr(first, "in_features"):
                            return int(first.in_features)
                return None

            @staticmethod
            def _extract_logits(output):
                if isinstance(output, PD10StudentRepresentationOutput):
                    return output.logits
                if isinstance(output, Mapping) and "logits" in output:
                    return output["logits"]
                if isinstance(output, (tuple, list)):
                    if not output:
                        raise ValueError("student model returned an empty tuple/list")
                    return output[0]
                return output

            def _representation_capture_source(self) -> str:
                mod = getattr(self.base_model, "mod", None)
                if isinstance(getattr(mod, "fc", None), torch.nn.Module):
                    return "student_part_hidden_pre_fc"
                if isinstance(getattr(mod, "head", None), torch.nn.Module):
                    return "student_part_hidden_pre_head"
                if isinstance(getattr(mod, "classifier", None), torch.nn.Module):
                    return "student_part_hidden_pre_classifier"
                if isinstance(getattr(mod, "norm", None), torch.nn.Module):
                    return "student_part_hidden_norm"
                if isinstance(getattr(self.base_model, "classifier", None), torch.nn.Module):
                    return "student_hidden_pre_classifier"
                return "unavailable"

            @staticmethod
            def _first_tensor(value):
                if torch.is_tensor(value):
                    return value
                if isinstance(value, Mapping):
                    for item in value.values():
                        tensor = _PD10StudentRepresentationAdapterModule._first_tensor(item)
                        if tensor is not None:
                            return tensor
                if isinstance(value, (tuple, list)):
                    for item in value:
                        tensor = _PD10StudentRepresentationAdapterModule._first_tensor(item)
                        if tensor is not None:
                            return tensor
                return None

            def _register_representation_hook(self, captured: list[Any]):
                def pre_hook(_module, inputs):
                    tensor = self._first_tensor(inputs)
                    if tensor is not None:
                        captured.append(tensor)

                def post_hook(_module, _inputs, output):
                    tensor = self._first_tensor(output)
                    if tensor is not None:
                        captured.append(tensor)

                mod = getattr(self.base_model, "mod", None)
                for module in (
                    getattr(mod, "fc", None),
                    getattr(mod, "head", None),
                    getattr(mod, "classifier", None),
                ):
                    if isinstance(module, torch.nn.Module):
                        return module.register_forward_pre_hook(pre_hook)
                norm = getattr(mod, "norm", None)
                if isinstance(norm, torch.nn.Module):
                    return norm.register_forward_hook(post_hook)
                classifier = getattr(self.base_model, "classifier", None)
                if isinstance(classifier, torch.nn.Module):
                    return classifier.register_forward_pre_hook(pre_hook)
                raise RuntimeError(
                    "Representation KD could not find a ParT hidden-state capture point "
                    "(expected mod.fc, mod.head, mod.classifier, or mod.norm)."
                )

            def _coerce_hidden_representation(self, raw, *, batch_size: int):
                tensor = self._first_tensor(raw)
                if tensor is None:
                    raise RuntimeError("Representation hook did not capture a tensor")
                if tensor.ndim == 3:
                    if int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == int(batch_size):
                        tensor = tensor[0]
                    elif int(tensor.shape[0]) == int(batch_size):
                        tensor = tensor[:, 0, :]
                    elif int(tensor.shape[1]) == int(batch_size):
                        tensor = tensor[0]
                    else:
                        raise ValueError(
                            f"Cannot interpret captured student representation shape {tuple(tensor.shape)} "
                            f"for batch size {int(batch_size)}"
                        )
                if tensor.ndim != 2:
                    raise ValueError(
                        f"student hidden representation must be rank-2 after pooling, got {tuple(tensor.shape)}"
                    )
                if int(tensor.shape[0]) != int(batch_size) and int(tensor.shape[1]) == int(batch_size):
                    tensor = tensor.transpose(0, 1)
                if int(tensor.shape[0]) != int(batch_size):
                    raise ValueError(
                        f"student hidden representation batch mismatch: "
                        f"{tuple(tensor.shape)} for batch size {int(batch_size)}"
                    )
                if int(tensor.shape[1]) != self.raw_representation_dim:
                    raise ValueError(
                        f"student hidden representation width {int(tensor.shape[1])} does not match "
                        f"raw_representation_dim={self.raw_representation_dim}"
                    )
                return tensor

            def forward(self, points, features, lorentz_vectors, mask, *, return_representation: bool = False):
                if not return_representation:
                    return self._extract_logits(self.base_model(points, features, lorentz_vectors, mask))
                captured: list[Any] = []
                handle = self._register_representation_hook(captured)
                try:
                    output = self.base_model(points, features, lorentz_vectors, mask)
                finally:
                    handle.remove()
                logits = self._extract_logits(output)
                raw = self._coerce_hidden_representation(captured[-1] if captured else None, batch_size=int(logits.shape[0]))
                projected = self.student_representation_projector(raw)
                projected = torch.nn.functional.normalize(projected, p=2, dim=-1, eps=1.0e-12)
                return PD10StudentRepresentationOutput(
                    logits=logits,
                    student_repr_raw=raw,
                    student_repr_projected=projected,
                )

        return _PD10StudentRepresentationAdapterModule(base_model)


def pd10_student_forward_with_optional_representation(
    model,
    batch: Mapping[str, Any],
    *,
    return_representation: bool = False,
) -> Any:
    """Forward helper for Step 4 while preserving the logits-only model contract."""

    if not return_representation:
        return model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
    try:
        return model(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
            return_representation=True,
        )
    except TypeError as exc:
        raise TypeError(
            "Representation KD requires a student model wrapped with PD10StudentRepresentationAdapter"
        ) from exc


def pd10_student_representation_loss(
    student_repr_projected,
    teacher_representations=None,
    *,
    mode: str = PD10_REPRESENTATION_MODE_COSINE,
):
    """Representation KD loss for normalized student/teacher vectors."""

    torch = require_torch()
    rep_mode = normalize_pd10_representation_mode(mode)
    if rep_mode == PD10_REPRESENTATION_MODE_NONE or teacher_representations is None:
        return student_repr_projected.new_zeros(())
    if rep_mode != PD10_REPRESENTATION_MODE_COSINE:
        raise AssertionError(f"Unhandled PD10 representation mode {rep_mode!r}")
    if student_repr_projected.ndim != 2 or teacher_representations.ndim != 2:
        raise ValueError("student and teacher representations must both have shape [N, D]")
    if tuple(student_repr_projected.shape) != tuple(teacher_representations.shape):
        raise ValueError(
            f"student/teacher representation shape mismatch: "
            f"{tuple(student_repr_projected.shape)} != {tuple(teacher_representations.shape)}"
        )
    student = torch.nn.functional.normalize(student_repr_projected.float(), p=2, dim=-1, eps=1.0e-12)
    teacher = torch.nn.functional.normalize(teacher_representations.float(), p=2, dim=-1, eps=1.0e-12)
    return torch.mean(1.0 - torch.sum(student * teacher, dim=-1))


@dataclass(frozen=True)
class PD10StudentTrainConfig:
    """Config for one HLT-only PD10 student condition."""

    student_init: str
    teacher_target: str
    output_dir: str
    hlt_cache_dir: str
    teacher_logit_cache: str | None = None
    teacher_representation_cache: str | None = None
    baseline_checkpoint: str | None = None
    target_mode: str = PD10_TARGET_FULL_LOGITS
    temperature: float = PD10_DEFAULT_TEMPERATURE
    kd_alpha: float = PD10_DEFAULT_ALPHA
    kd_warmup_epochs: int | None = None
    top_k: int = PD10_TOP_K
    representation_beta: float = PD10_DEFAULT_REPRESENTATION_BETA
    representation_dim: int = PD10_REPRESENTATION_DIM
    representation_mode: str = PD10_REPRESENTATION_MODE_COSINE
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = PD10_STUDENT_DEFAULT_SEED
    batch_size: int = PD10_STUDENT_DEFAULT_BATCH_SIZE
    epochs: int | None = None
    lr: float | None = None
    weight_decay: float = PD10_STUDENT_DEFAULT_WEIGHT_DECAY
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    model_size: str = "base"
    compile_model: bool = False
    confirm_final_test: bool = False
    evaluate_final_test: bool = True
    align_prediction_to_teacher_cache: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        init = normalize_pd10_student_init_mode(self.student_init)
        teacher = normalize_pd10_extended_teacher_target(self.teacher_target)
        target_mode = normalize_pd10_extended_student_target_mode(self.target_mode)
        uses_logits = teacher != PD10_TEACHER_NONE and pd10_target_mode_uses_logits(target_mode)
        uses_representations = teacher != PD10_TEACHER_NONE and pd10_target_mode_uses_representations(target_mode)
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"PD10 student training split order must be {PD10_SPLIT_ORDER}")
        if self.evaluate_final_test and not self.confirm_final_test:
            raise ValueError("PD10 student final-test evaluation requires confirm_final_test=True")
        if init == PD10_STUDENT_INIT_WARM_START and not self.baseline_checkpoint:
            raise ValueError("warm_start students require baseline_checkpoint")
        if init == PD10_STUDENT_INIT_SCRATCH and self.baseline_checkpoint:
            raise ValueError("scratch students must not receive baseline_checkpoint")
        if teacher == PD10_TEACHER_NONE:
            if target_mode != PD10_TARGET_FULL_LOGITS:
                raise ValueError("CE-only students must use target_mode='full_logits'")
            kd_alpha = 0.0
            kd_warmup = 0
            representation_beta = 0.0
            representation_mode = PD10_REPRESENTATION_MODE_NONE
        else:
            if uses_logits and not self.teacher_logit_cache:
                raise ValueError("logit-KD students require teacher_logit_cache")
            if uses_representations and not self.teacher_representation_cache:
                raise ValueError("representation-KD students require teacher_representation_cache")
            kd_alpha = float(self.kd_alpha) if uses_logits else 0.0
            if uses_logits and (kd_alpha <= 0.0 or kd_alpha > 1.0):
                raise ValueError("kd_alpha must be in (0, 1] for logit-KD students")
            representation_mode = normalize_pd10_representation_mode(self.representation_mode)
            representation_beta = float(self.representation_beta) if uses_representations else 0.0
            if uses_representations:
                if representation_mode == PD10_REPRESENTATION_MODE_NONE:
                    raise ValueError("representation-KD students require representation_mode!='none'")
                if representation_beta <= 0.0:
                    raise ValueError("representation_beta must be positive for representation-KD students")
            else:
                representation_mode = PD10_REPRESENTATION_MODE_NONE
            kd_warmup = (
                PD10_STUDENT_WARM_START_DEFAULT_KD_WARMUP_EPOCHS
                if init == PD10_STUDENT_INIT_WARM_START
                else PD10_STUDENT_SCRATCH_DEFAULT_KD_WARMUP_EPOCHS
            )
            if self.kd_warmup_epochs is not None:
                kd_warmup = int(self.kd_warmup_epochs)
            if kd_warmup < 0:
                raise ValueError("kd_warmup_epochs cannot be negative")
        temperature = float(self.temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if int(self.top_k) <= 0 or int(self.top_k) > PD10_NUM_CLASSES:
            raise ValueError(f"top_k must be in [1, {PD10_NUM_CLASSES}]")
        representation_dim = int(self.representation_dim)
        if representation_dim <= 0:
            raise ValueError("representation_dim must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        epochs = self.epochs
        if epochs is None:
            epochs = (
                PD10_STUDENT_WARM_START_DEFAULT_EPOCHS
                if init == PD10_STUDENT_INIT_WARM_START
                else PD10_STUDENT_SCRATCH_DEFAULT_EPOCHS
            )
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        lr = self.lr
        if lr is None:
            lr = (
                PD10_STUDENT_WARM_START_DEFAULT_LR
                if init == PD10_STUDENT_INIT_WARM_START
                else PD10_STUDENT_SCRATCH_DEFAULT_LR
            )
        if float(lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.model_size not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "student_init", init)
        object.__setattr__(self, "teacher_target", teacher)
        object.__setattr__(self, "target_mode", target_mode)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "kd_alpha", kd_alpha)
        object.__setattr__(self, "kd_warmup_epochs", kd_warmup)
        object.__setattr__(self, "representation_beta", representation_beta)
        object.__setattr__(self, "representation_dim", representation_dim)
        object.__setattr__(self, "representation_mode", representation_mode)
        object.__setattr__(self, "epochs", int(epochs))
        object.__setattr__(self, "lr", float(lr))

    @property
    def uses_teacher(self) -> bool:
        return self.teacher_target != PD10_TEACHER_NONE

    @property
    def uses_logit_teacher(self) -> bool:
        return self.uses_teacher and pd10_target_mode_uses_logits(self.target_mode)

    @property
    def uses_representations(self) -> bool:
        return self.uses_teacher and pd10_target_mode_uses_representations(self.target_mode)

    @property
    def logit_target_mode(self) -> str:
        return pd10_base_logit_target_mode(self.target_mode)

    @property
    def variant_name(self) -> str:
        return pd10_extended_student_variant_name(
            self.student_init,
            self.teacher_target,
            self.target_mode,
            temperature=self.temperature,
            kd_alpha=self.kd_alpha if self.uses_teacher else PD10_DEFAULT_ALPHA,
            top_k=self.top_k,
            representation_beta=self.representation_beta
            if self.uses_representations
            else PD10_DEFAULT_REPRESENTATION_BETA,
            representation_mode=self.representation_mode,
        )

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "last.pt"


def pd10_student_dir(
    student_init: str,
    teacher_target: str = PD10_TEACHER_NONE,
    target_mode: str = PD10_TARGET_FULL_LOGITS,
    *,
    temperature: float = PD10_DEFAULT_TEMPERATURE,
    kd_alpha: float = PD10_DEFAULT_ALPHA,
    top_k: int = PD10_TOP_K,
    representation_beta: float = PD10_DEFAULT_REPRESENTATION_BETA,
    representation_mode: str = PD10_REPRESENTATION_MODE_COSINE,
    output_root: str | Path = "checkpoints",
) -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    name = pd10_extended_student_variant_name(
        student_init,
        teacher_target,
        target_mode,
        temperature=temperature,
        kd_alpha=kd_alpha,
        top_k=top_k,
        representation_beta=representation_beta,
        representation_mode=representation_mode,
    )
    return layout.student_dir(name)


def pd10_student_checkpoint(
    student_init: str,
    teacher_target: str = PD10_TEACHER_NONE,
    target_mode: str = PD10_TARGET_FULL_LOGITS,
    *,
    temperature: float = PD10_DEFAULT_TEMPERATURE,
    kd_alpha: float = PD10_DEFAULT_ALPHA,
    top_k: int = PD10_TOP_K,
    representation_beta: float = PD10_DEFAULT_REPRESENTATION_BETA,
    representation_mode: str = PD10_REPRESENTATION_MODE_COSINE,
    output_root: str | Path = "checkpoints",
) -> Path:
    return pd10_student_dir(
        student_init,
        teacher_target,
        target_mode,
        temperature=temperature,
        kd_alpha=kd_alpha,
        top_k=top_k,
        representation_beta=representation_beta,
        representation_mode=representation_mode,
        output_root=output_root,
    ) / "best_model_val.pt"


def pd10_student_prediction_dir(output_dir: str | Path) -> Path:
    """Return the prediction-cache root inside one student output directory."""

    return Path(output_dir) / "student_predictions"


def _amp_state_targets(model: Any) -> list[Any]:
    targets: list[Any] = []
    seen: set[int] = set()
    for obj in (model, getattr(model, "mod", None), getattr(model, "module", None)):
        if obj is None or not hasattr(obj, "use_amp"):
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        targets.append(obj)
        seen.add(obj_id)
    return targets


@contextmanager
def _disable_model_amp_for_eval(model: Any):
    targets = _amp_state_targets(model)
    previous = [(target, getattr(target, "use_amp")) for target in targets]
    try:
        for target, _state in previous:
            setattr(target, "use_amp", False)
        yield bool(previous)
    finally:
        for target, state in previous:
            setattr(target, "use_amp", state)


def pd10_effective_kd_alpha(config: PD10StudentTrainConfig, epoch: int) -> float:
    if not config.uses_logit_teacher:
        return 0.0
    warmup = int(config.kd_warmup_epochs or 0)
    if warmup <= 0:
        return float(config.kd_alpha)
    return float(config.kd_alpha) * min(1.0, float(epoch) / float(warmup))


def pd10_effective_representation_beta(config: PD10StudentTrainConfig, epoch: int) -> float:
    if not config.uses_representations:
        return 0.0
    warmup = int(config.kd_warmup_epochs or 0)
    if warmup <= 0:
        return float(config.representation_beta)
    return float(config.representation_beta) * min(1.0, float(epoch) / float(warmup))


def _teacher_distribution(teacher_logits, *, temperature: float, target_mode: str, top_k: int):
    torch = require_torch()
    teacher_prob = torch.softmax(teacher_logits / float(temperature), dim=-1)
    if target_mode != PD10_TARGET_TOP3:
        return teacher_prob
    values, indices = torch.topk(teacher_prob, k=int(top_k), dim=-1)
    values = values / torch.clamp(values.sum(dim=-1, keepdim=True), min=1.0e-12)
    sparse = torch.zeros_like(teacher_prob)
    sparse.scatter_(dim=-1, index=indices, src=values)
    return sparse


def pd10_kd_loss(
    student_logits,
    teacher_logits,
    *,
    temperature: float,
    target_mode: str,
    top_k: int = PD10_TOP_K,
):
    torch = require_torch()
    mode = pd10_base_logit_target_mode(target_mode)
    teacher_prob = _teacher_distribution(
        teacher_logits,
        temperature=float(temperature),
        target_mode=mode,
        top_k=int(top_k),
    )
    student_log_prob = torch.log_softmax(student_logits / float(temperature), dim=-1)
    per_row = torch.sum(
        teacher_prob * (torch.log(torch.clamp(teacher_prob, min=1.0e-12)) - student_log_prob),
        dim=-1,
    ) * float(temperature) * float(temperature)
    if mode == PD10_TARGET_CONFIDENCE_WEIGHTED:
        confidence = torch.softmax(teacher_logits, dim=-1).max(dim=-1).values
        floor = 1.0 / float(PD10_NUM_CLASSES)
        weights = torch.clamp((confidence - floor) / (1.0 - floor), min=0.0, max=1.0)
        return torch.mean(weights * per_row)
    return torch.mean(per_row)


def pd10_student_loss(
    student_logits,
    labels,
    *,
    teacher_logits=None,
    teacher_representations=None,
    student_repr_projected=None,
    target_mode: str = PD10_TARGET_FULL_LOGITS,
    temperature: float = PD10_DEFAULT_TEMPERATURE,
    kd_alpha: float = 0.0,
    representation_beta: float = 0.0,
    representation_mode: str = PD10_REPRESENTATION_MODE_NONE,
    top_k: int = PD10_TOP_K,
) -> tuple[Any, dict[str, float]]:
    torch = require_torch()
    ce_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    alpha = float(kd_alpha)
    beta = float(representation_beta)
    kd = student_logits.new_zeros(())
    has_logit_teacher = teacher_logits is not None
    if has_logit_teacher:
        kd = pd10_kd_loss(
            student_logits,
            teacher_logits,
            temperature=float(temperature),
            target_mode=target_mode,
            top_k=int(top_k),
        )
    if has_logit_teacher and alpha > 0.0:
        total = (1.0 - alpha) * ce_loss + alpha * kd
    else:
        total = ce_loss
        alpha = 0.0
    rep = student_logits.new_zeros(())
    has_rep_teacher = teacher_representations is not None and student_repr_projected is not None
    if has_rep_teacher:
        rep = pd10_student_representation_loss(
            student_repr_projected,
            teacher_representations,
            mode=representation_mode,
        )
    if has_rep_teacher and beta > 0.0:
        total = total + beta * rep
    else:
        beta = 0.0
    kd_value = float(kd.detach().cpu().item())
    rep_value = float(rep.detach().cpu().item())
    return total, {
        "loss": float(total.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "kd_loss": kd_value,
        "rep_loss": rep_value,
        "effective_kd_alpha": alpha,
        "effective_representation_beta": beta,
        "teacher_student_logit_kl": kd_value if has_logit_teacher else float("nan"),
        "teacher_student_representation_cosine": (1.0 - rep_value) if has_rep_teacher else float("nan"),
    }


def _accuracy_from_logits(student_logits, labels) -> tuple[int, int]:
    preds = student_logits.argmax(dim=1)
    return int((preds == labels).sum().detach().cpu().item()), int(labels.numel())


def run_pd10_student_epoch(
    model,
    loader,
    *,
    config: PD10StudentTrainConfig,
    device,
    epoch: int,
    optimizer=None,
    scaler=None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_rep = 0.0
    total_teacher_student_logit_kl = 0.0
    total_teacher_student_logit_kl_seen = 0
    total_teacher_student_representation_cosine = 0.0
    total_teacher_student_representation_cosine_seen = 0
    total_correct = 0
    total_seen = 0
    effective_alpha = pd10_effective_kd_alpha(config, epoch)
    effective_beta = pd10_effective_representation_beta(config, epoch)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            assert_pd10_student_batch_hlt_only(batch)
            batch = move_pd10_student_batch_to_device(batch, device)
            if config.uses_logit_teacher and "teacher_logits" not in batch:
                raise ValueError("KD student batch is missing teacher_logits")
            if not config.uses_logit_teacher and "teacher_logits" in batch:
                raise ValueError("student batch unexpectedly includes teacher_logits")
            if config.uses_representations and "teacher_representations" not in batch:
                raise ValueError("representation-KD student batch is missing teacher_representations")
            if not config.uses_representations and "teacher_representations" in batch:
                raise ValueError("student batch unexpectedly includes teacher_representations")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(config.amp and device.type == "cuda")
            if autocast_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_context = torch.amp.autocast("cuda", enabled=True)
            elif autocast_enabled:
                autocast_context = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_context = nullcontext()
            with autocast_context:
                student_repr_projected = None
                if config.uses_representations:
                    student_output = pd10_student_forward_with_optional_representation(
                        model,
                        batch,
                        return_representation=True,
                    )
                    logits = student_output.logits
                    student_repr_projected = student_output.student_repr_projected
                else:
                    logits = pd10_student_forward_with_optional_representation(
                        model,
                        batch,
                        return_representation=False,
                    )
                loss, loss_parts = pd10_student_loss(
                    logits,
                    batch["labels"],
                    teacher_logits=batch.get("teacher_logits"),
                    teacher_representations=batch.get("teacher_representations"),
                    student_repr_projected=student_repr_projected,
                    target_mode=config.target_mode,
                    temperature=config.temperature,
                    kd_alpha=effective_alpha,
                    representation_beta=effective_beta,
                    representation_mode=config.representation_mode,
                    top_k=config.top_k,
                )
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if config.grad_clip_norm and config.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip_norm and config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
                    optimizer.step()
            batch_size = int(batch["labels"].numel())
            total_loss += float(loss_parts["loss"]) * batch_size
            total_ce += float(loss_parts["ce_loss"]) * batch_size
            total_kd += float(loss_parts["kd_loss"]) * batch_size
            total_rep += float(loss_parts.get("rep_loss", 0.0)) * batch_size
            logit_kl = loss_parts.get("teacher_student_logit_kl")
            if logit_kl is not None and np.isfinite(float(logit_kl)):
                total_teacher_student_logit_kl += float(logit_kl) * batch_size
                total_teacher_student_logit_kl_seen += batch_size
            rep_cosine = loss_parts.get("teacher_student_representation_cosine")
            if rep_cosine is not None and np.isfinite(float(rep_cosine)):
                total_teacher_student_representation_cosine += float(rep_cosine) * batch_size
                total_teacher_student_representation_cosine_seen += batch_size
            correct, seen = _accuracy_from_logits(logits.detach(), batch["labels"])
            total_correct += correct
            total_seen += seen
    if total_seen == 0:
        return {
            "loss": float("nan"),
            "ce_loss": float("nan"),
            "kd_loss": 0.0,
            "rep_loss": 0.0,
            "accuracy": 0.0,
            "n_jets": 0,
            "effective_kd_alpha": float(effective_alpha),
            "effective_representation_beta": float(effective_beta),
            "teacher_student_logit_kl": None,
            "teacher_student_representation_cosine": None,
        }
    return {
        "loss": total_loss / float(total_seen),
        "ce_loss": total_ce / float(total_seen),
        "kd_loss": total_kd / float(total_seen),
        "rep_loss": total_rep / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
        "effective_kd_alpha": float(effective_alpha),
        "effective_representation_beta": float(effective_beta),
        "teacher_student_logit_kl": None
        if total_teacher_student_logit_kl_seen == 0
        else total_teacher_student_logit_kl / float(total_teacher_student_logit_kl_seen),
        "teacher_student_representation_cosine": None
        if total_teacher_student_representation_cosine_seen == 0
        else total_teacher_student_representation_cosine / float(total_teacher_student_representation_cosine_seen),
    }


def collect_pd10_student_prediction_block(
    model,
    loader,
    *,
    config: PD10StudentTrainConfig,
    device,
    split: str,
    checkpoint_payload: Mapping[str, Any] | None = None,
    max_batches: int | None = None,
) -> PredictionBlock:
    """Collect HLT-only logits for a deployable selected student checkpoint."""

    torch = require_torch()
    model.eval()
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    jet_ids: list[Any] = []
    with torch.no_grad():
        with _disable_model_amp_for_eval(model) as amp_disabled:
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                assert_pd10_student_batch_hlt_only(batch)
                if "teacher_logits" in batch or "teacher_representations" in batch:
                    raise ValueError("student prediction cache must be built from teacher-free HLT batches")
                batch = move_pd10_student_batch_to_device(batch, device)
                logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
                logits_chunks.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
                labels_chunks.append(batch["labels"].detach().cpu().numpy().astype(np.int64, copy=False))
                jet_ids.extend(batch["jet_ids"])
    if not labels_chunks:
        raise ValueError(f"student prediction cache for {split} collected no jets")
    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32, copy=False)
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.int64, copy=False)
    logits_np, logit_sanitization = sanitize_prediction_logits(
        logits_np,
        model_name=config.variant_name,
        split=split,
    )
    logit_sanitization["amp_disabled_for_eval"] = bool(amp_disabled)
    checkpoint_payload = dict(checkpoint_payload or {})
    metadata = {
        "contract": PD10_STUDENT_PREDICTION_CACHE_CONTRACT,
        "training_contract": PD10_STUDENT_TRAINING_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP7_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.variant_name,
        "model_kind": "pd10_hlt_only_student",
        "architecture": "part",
        "split": split,
        "split_expected_size": int(PD10_SPLIT_SIZES[split]),
        "student_init": config.student_init,
        "teacher_target_train_time": config.teacher_target,
        "target_mode": config.target_mode,
        "temperature": float(config.temperature),
        "kd_alpha": float(config.kd_alpha),
        "representation_beta": float(config.representation_beta),
        "representation_dim": int(config.representation_dim),
        "representation_mode": config.representation_mode,
        "uses_logit_teacher_train_time": bool(config.uses_logit_teacher),
        "uses_representations_train_time": bool(config.uses_representations),
        "top_k": int(config.top_k),
        "source_view": "fixed_hlt",
        "allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
        "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
        "selected_by_model_val": True,
        "prediction_aligned_to_teacher_cache": bool(config.align_prediction_to_teacher_cache),
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path) if config.checkpoint_path.exists() else None,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
        "inference_requires_teacher_logits": False,
        "inference_requires_teacher_representations": False,
        "inference_requires_offline_inputs": False,
        "inference_export_requires_teacher_features": False,
        "no_offline_inputs_loaded": True,
        "offline_privileged_inputs_loaded": False,
        "final_test_after_model_val_selection": split == config.final_test_split,
        "logit_sanitization": logit_sanitization,
    }
    return PredictionBlock(
        model_name=config.variant_name,
        split=split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=jet_ids,
        metadata=metadata,
    )


def collect_and_save_pd10_student_predictions(
    model,
    dataset,
    *,
    config: PD10StudentTrainConfig,
    device,
    split: str,
    checkpoint_payload: Mapping[str, Any] | None = None,
    max_batches: int | None = None,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader = make_pd10_student_data_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + (11 if split == config.val_split else 12),
    )
    block = collect_pd10_student_prediction_block(
        model,
        loader,
        config=config,
        device=device,
        split=split,
        checkpoint_payload=checkpoint_payload,
        max_batches=max_batches,
    )
    prediction_dir = pd10_student_prediction_dir(config.output_dir)
    metadata = save_prediction_block(block, prediction_dir, overwrite=bool(config.overwrite))
    metrics = pd10_prediction_metrics_from_logits(
        block.logits,
        block.labels,
        validation_thresholds_by_class=validation_thresholds_by_class,
        validation_binary_thresholds=validation_binary_thresholds,
    )
    if "cross_entropy" in metrics:
        metrics.setdefault("loss", metrics["cross_entropy"])
        metrics.setdefault("ce_loss", metrics["cross_entropy"])
    metrics.setdefault("kd_loss", 0.0)
    metrics.setdefault("rep_loss", 0.0)
    metrics.setdefault("effective_kd_alpha", 0.0)
    metrics.setdefault("effective_representation_beta", 0.0)
    logit_sanitization = dict(block.metadata.get("logit_sanitization") or {})
    metrics.update(
        {
            "n_jets": int(block.labels.shape[0]),
            "prediction_content_hash": metadata.get("prediction_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
            "prediction_npz_path": metadata.get("npz_path"),
            "prediction_metadata_path": metadata.get("metadata_path"),
            "metrics_source": "student_prediction_cache",
            "logit_sanitization": logit_sanitization,
            "nonfinite_logit_row_count": int(logit_sanitization.get("nonfinite_row_count", 0) or 0),
            "nonfinite_logit_value_count": int(logit_sanitization.get("nonfinite_value_count", 0) or 0),
            "nonfinite_logit_rows_repaired": bool(logit_sanitization.get("repaired", False)),
        }
    )
    save_json(Path(config.output_dir) / f"{split}_prediction_metrics.json", metrics)
    return metrics, metadata


def _teacher_free_dataset_from_aligned_dataset(
    dataset: PD10StudentDistillationDataset,
    *,
    reference_teacher_target: str,
) -> PD10StudentDistillationDataset:
    """Strip train-time teacher fields from an already teacher-aligned HLT dataset."""

    view = JetView(
        tokens=np.asarray(dataset.tokens, dtype=np.float32),
        mask=np.asarray(dataset.hlt_mask, dtype=bool),
        labels=np.asarray(dataset.labels, dtype=np.int64),
        jet_ids=list(dataset.jet_ids),
        split=dataset.split,
        metadata={
            **dict(dataset.hlt_metadata),
            "view": "fixed_hlt",
            "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
            "returns_offline_particles": False,
            "prediction_subset_reference_teacher_target": str(reference_teacher_target),
            "subset_selection": {
                "strategy": "teacher_cache_aligned_then_teacher_free_for_prediction",
                "reference_teacher_target": str(reference_teacher_target),
                "selected_n_jets": int(len(dataset.labels)),
            },
        },
    )
    return PD10StudentDistillationDataset(view, teacher_target=PD10_TEACHER_NONE)


def _load_teacher_free_prediction_dataset(
    config: PD10StudentTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> PD10StudentDistillationDataset:
    """Build an HLT-only prediction dataset.

    The default path intentionally does not touch teacher caches, so selected
    model_val/final_test prediction works from HLT particles alone.  Teacher-row
    alignment is an explicit diagnostic mode for smoke comparisons.
    """

    if not bool(config.align_prediction_to_teacher_cache):
        return load_pd10_student_dataset(
            config.hlt_cache_dir,
            split,
            teacher_target=PD10_TEACHER_NONE,
            teacher_logit_dir=None,
            teacher_representation_dir=None,
            max_jets=max_jets,
        )

    reference_teacher = config.teacher_target
    if reference_teacher == PD10_TEACHER_NONE and config.teacher_logit_cache:
        reference_teacher = PD10_TEACHER_HLT
    if reference_teacher == PD10_TEACHER_NONE:
        return load_pd10_student_dataset(
            config.hlt_cache_dir,
            split,
            teacher_target=PD10_TEACHER_NONE,
            teacher_logit_dir=None,
            max_jets=max_jets,
        )

    teacher_logit_dir = config.teacher_logit_cache
    teacher_representation_dir = (
        config.teacher_representation_cache
        if reference_teacher == config.teacher_target and config.uses_representations
        else None
    )
    if teacher_logit_dir is None and teacher_representation_dir is None:
        if config.teacher_target == PD10_TEACHER_NONE:
            return load_pd10_student_dataset(
                config.hlt_cache_dir,
                split,
                teacher_target=PD10_TEACHER_NONE,
                teacher_logit_dir=None,
                max_jets=max_jets,
            )
        return load_pd10_student_dataset(
            config.hlt_cache_dir,
            split,
            teacher_target=PD10_TEACHER_NONE,
            teacher_logit_dir=None,
            teacher_representation_dir=None,
            max_jets=max_jets,
        )
    aligned = load_pd10_student_dataset(
        config.hlt_cache_dir,
        split,
        teacher_target=reference_teacher,
        teacher_logit_dir=teacher_logit_dir,
        teacher_representation_dir=teacher_representation_dir,
        max_jets=max_jets,
    )
    return _teacher_free_dataset_from_aligned_dataset(
        aligned,
        reference_teacher_target=reference_teacher,
    )


def _load_or_build_student_model(config: PD10StudentTrainConfig, *, device):
    if config.student_init == PD10_STUDENT_INIT_WARM_START:
        model, payload = load_hlt_model_from_checkpoint(config.baseline_checkpoint, device=device)
        return model, payload
    model = build_hlt_classifier(num_classes=PD10_NUM_CLASSES, model_size=config.model_size)
    return model.to(device), None


def _pd10_model_has_representation_adapter(model) -> bool:
    return hasattr(model, "base_model") and hasattr(model, "student_representation_projector")


def _maybe_wrap_student_for_representation(config: PD10StudentTrainConfig, model):
    if not config.uses_representations or _pd10_model_has_representation_adapter(model):
        return model
    return PD10StudentRepresentationAdapter(
        model,
        representation_dim=int(config.representation_dim),
    )


def _deployable_student_model(model):
    return model.base_model if _pd10_model_has_representation_adapter(model) else model


def _representation_projector_state_dict(model):
    if not _pd10_model_has_representation_adapter(model):
        return None
    return model.student_representation_projector.state_dict()


def _checkpoint_payload(
    model,
    optimizer,
    *,
    config: PD10StudentTrainConfig,
    epoch: int,
    metrics: Mapping[str, Any],
    baseline_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    deployable_model = _deployable_student_model(model)
    representation_state = _representation_projector_state_dict(model)
    return {
        "contract": PD10_STUDENT_TRAINING_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP7_EXPERIMENT_STEP,
        "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
        "student_init": config.student_init,
        "teacher_target": config.teacher_target,
        "target_mode": config.target_mode,
        "temperature": float(config.temperature),
        "kd_alpha": float(config.kd_alpha),
        "kd_warmup_epochs": int(config.kd_warmup_epochs or 0),
        "variant_name": config.variant_name,
        "epoch": int(epoch),
        "model_state_dict": deployable_model.state_dict(),
        "representation_projector_state_dict": representation_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "metrics": dict(metrics),
        "baseline_checkpoint": config.baseline_checkpoint,
        "baseline_checkpoint_sha256": (
            sha256_file(config.baseline_checkpoint) if config.baseline_checkpoint else None
        ),
        "baseline_metadata": dict(baseline_metadata or {}),
        "model_config": getattr(deployable_model, "config", {}),
        "representation_wrapper_config": getattr(model, "config", {}) if representation_state is not None else None,
        "label_names": list(LABEL_NAMES),
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
        "inference_requires_teacher_logits": False,
        "inference_requires_teacher_representations": False,
        "inference_requires_offline_inputs": False,
        "inference_export_requires_teacher_features": False,
        "representation_beta": float(config.representation_beta),
        "representation_mode": config.representation_mode,
        "representation_dim": int(config.representation_dim),
        "uses_logit_teacher": bool(config.uses_logit_teacher),
        "uses_representations": bool(config.uses_representations),
        "deployable_checkpoint_state": "base_hlt_student_only",
    }


def train_pd10_student(
    config: PD10StudentTrainConfig,
    *,
    model=None,
    train_dataset=None,
    val_dataset=None,
    final_test_dataset=None,
) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"PD10 student checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or load_pd10_student_dataset(
        config.hlt_cache_dir,
        config.train_split,
        teacher_target=config.teacher_target,
        teacher_logit_dir=config.teacher_logit_cache,
        teacher_representation_dir=config.teacher_representation_cache,
        max_jets=config.max_train_jets,
    )
    val_dataset = val_dataset or load_pd10_student_dataset(
        config.hlt_cache_dir,
        config.val_split,
        teacher_target=config.teacher_target,
        teacher_logit_dir=config.teacher_logit_cache,
        teacher_representation_dir=config.teacher_representation_cache,
        max_jets=config.max_val_jets,
    )
    train_loader = make_pd10_student_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_pd10_student_data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    baseline_metadata = None
    provided_model = model is not None
    if model is None:
        model, baseline_metadata = _load_or_build_student_model(config, device=device)
    else:
        model = model.to(device)
    model = _maybe_wrap_student_for_representation(config, model).to(device)
    checkpoint_model = model
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        checkpoint_model = getattr(model, "_orig_mod", checkpoint_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = None
    if bool(config.amp and device.type == "cuda"):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    run_metadata = {
        "ok": True,
        "contract": PD10_STUDENT_TRAINING_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP7_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "student_init": config.student_init,
        "teacher_target": config.teacher_target,
        "target_mode": config.target_mode,
        "temperature": float(config.temperature),
        "kd_alpha": float(config.kd_alpha),
        "kd_warmup_epochs": int(config.kd_warmup_epochs or 0),
        "teacher_logit_cache": config.teacher_logit_cache,
        "teacher_representation_cache": config.teacher_representation_cache,
        "representation_beta": float(config.representation_beta),
        "representation_dim": int(config.representation_dim),
        "representation_mode": config.representation_mode,
        "uses_logit_teacher": bool(config.uses_logit_teacher),
        "uses_representations": bool(config.uses_representations),
        "baseline_checkpoint": config.baseline_checkpoint,
        "baseline_checkpoint_sha256": (
            sha256_file(config.baseline_checkpoint) if config.baseline_checkpoint else None
        ),
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
        "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
        "teacher_logits_train_time_only": True,
        "teacher_representations_train_time_only": True,
        "inference_requires_teacher_logits": False,
        "inference_requires_teacher_representations": False,
        "inference_requires_offline_inputs": False,
        "inference_export_requires_teacher_features": False,
        "config": asdict(config),
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_ce = float("inf")
    best_val_accuracy = -1.0
    best_epoch = -1
    best_row: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_pd10_student_epoch(
            model,
            train_loader,
            config=config,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_pd10_student_epoch(
            model,
            val_loader,
            config=config,
            device=device,
            epoch=epoch,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        torch.save(
            _checkpoint_payload(
                checkpoint_model,
                optimizer,
                config=config,
                epoch=epoch,
                metrics=row,
                baseline_metadata=baseline_metadata,
            ),
            config.last_checkpoint_path,
        )
        val_ce = float(val_metrics["ce_loss"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = val_ce < best_val_ce or (
            np.isclose(val_ce, best_val_ce) and val_accuracy > best_val_accuracy
        )
        if improved:
            best_val_ce = val_ce
            best_val_accuracy = val_accuracy
            best_epoch = int(epoch)
            best_row = row
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    checkpoint_model,
                    optimizer,
                    config=config,
                    epoch=epoch,
                    metrics=row,
                    baseline_metadata=baseline_metadata,
                ),
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    report: dict[str, Any] = {
        **run_metadata,
        "best_epoch": int(best_epoch),
        "selection_metric": "model_val_ce_loss",
        "selection_tie_breaker": "model_val_accuracy",
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_ce_loss": float(best_val_ce),
        "best_model_val_cross_entropy": float(best_val_ce),
        "best_model_val_metrics": dict(best_row["model_val"]) if best_row else None,
        "best_model_val_training_metrics": dict(best_row["model_val"]) if best_row else None,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(config.checkpoint_path),
        "last_checkpoint": str(config.last_checkpoint_path),
        "no_final_test_used_for_selection": True,
    }

    selected_model = None
    selected_payload: Mapping[str, Any] | None = None
    if not provided_model:
        selected_model, selected_payload = load_hlt_model_from_checkpoint(config.checkpoint_path, device=device)
    elif config.evaluate_final_test:
        selected_model, selected_payload = load_hlt_model_from_checkpoint(config.checkpoint_path, device=device)

    model_val_thresholds = None
    model_val_binary_thresholds = None
    if selected_model is not None:
        try:
            model_val_prediction_dataset = _load_teacher_free_prediction_dataset(
                config,
                config.val_split,
                max_jets=config.max_val_jets,
            )
            model_val_metrics, model_val_metadata = collect_and_save_pd10_student_predictions(
                selected_model,
                model_val_prediction_dataset,
                config=config,
                device=device,
                split=config.val_split,
                checkpoint_payload=selected_payload,
                max_batches=config.max_val_batches,
            )
            selected_model_val_diagnostics = dict(best_row["model_val"]) if best_row else {}
            for key in (
                "kd_loss",
                "rep_loss",
                "effective_kd_alpha",
                "effective_representation_beta",
                "teacher_student_logit_kl",
                "teacher_student_representation_cosine",
            ):
                if selected_model_val_diagnostics.get(key) is not None:
                    model_val_metrics[key] = selected_model_val_diagnostics.get(key)
            model_val_metrics["teacher_student_diagnostics_source"] = (
                "selected_model_val_training_loader"
                if selected_model_val_diagnostics
                else None
            )
            model_val_thresholds = model_val_metrics.get("score_thresholds_by_class")
            model_val_binary_thresholds = model_val_metrics.get("binary_score_thresholds")
            report["model_val_metrics"] = model_val_metrics
            report["selected_model_val_metrics"] = model_val_metrics
            report["model_val_prediction_cache"] = model_val_metadata
            report["model_val_prediction_dataset"] = model_val_prediction_dataset.to_metadata()
            report["model_val_prediction_teacher_free"] = True
        except Exception as exc:  # pragma: no cover - protects long compute jobs from report-side failures
            report["model_val_prediction_cache_error"] = str(exc)
    else:
        report["model_val_prediction_cache_skipped"] = (
            "custom in-memory model supplied; selected checkpoint was not reloaded for teacher-free prediction caching"
        )

    if config.evaluate_final_test:
        if selected_model is None:
            selected_model, selected_payload = load_hlt_model_from_checkpoint(config.checkpoint_path, device=device)
        final_test_dataset = final_test_dataset or _load_teacher_free_prediction_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_metrics, final_metadata = collect_and_save_pd10_student_predictions(
            selected_model,
            final_test_dataset,
            config=config,
            device=device,
            split=config.final_test_split,
            checkpoint_payload=selected_payload,
            max_batches=config.max_final_test_batches,
            validation_thresholds_by_class=model_val_thresholds,
            validation_binary_thresholds=model_val_binary_thresholds,
        )
        report["final_test_metrics"] = final_metrics
        report["final_test_prediction_cache"] = final_metadata
        report["final_test_dataset"] = final_test_dataset.to_metadata()
        report["final_test_evaluation_teacher_free"] = True
    else:
        report["final_test_metrics"] = None
        report["final_test_evaluation_skipped"] = True

    save_json(output_dir / "run_report.json", report)
    save_json(output_dir / "model_val_report.json", report)
    if config.evaluate_final_test:
        save_json(output_dir / "final_test_report.json", report)
    return report


__all__ = [
    "PD10_STEP7_EXPERIMENT_STEP",
    "PD10_STUDENT_DEFAULT_BATCH_SIZE",
    "PD10_STUDENT_DEFAULT_SEED",
    "PD10_STUDENT_DEFAULT_WEIGHT_DECAY",
    "PD10_STUDENT_SCRATCH_DEFAULT_EPOCHS",
    "PD10_STUDENT_SCRATCH_DEFAULT_KD_WARMUP_EPOCHS",
    "PD10_STUDENT_SCRATCH_DEFAULT_LR",
    "PD10_STUDENT_PREDICTION_CACHE_CONTRACT",
    "PD10_STUDENT_REPRESENTATION_WRAPPER_CONTRACT",
    "PD10_STUDENT_TRAINING_CONTRACT",
    "PD10_STUDENT_WARM_START_DEFAULT_EPOCHS",
    "PD10_STUDENT_WARM_START_DEFAULT_KD_WARMUP_EPOCHS",
    "PD10_STUDENT_WARM_START_DEFAULT_LR",
    "PD10_V2_STEP3_EXPERIMENT_STEP",
    "PD10StudentRepresentationAdapter",
    "PD10StudentRepresentationOutput",
    "PD10StudentTrainConfig",
    "collect_and_save_pd10_student_predictions",
    "collect_pd10_student_prediction_block",
    "pd10_effective_kd_alpha",
    "pd10_effective_representation_beta",
    "pd10_kd_loss",
    "pd10_student_forward_with_optional_representation",
    "pd10_student_checkpoint",
    "pd10_student_dir",
    "pd10_student_loss",
    "pd10_student_prediction_dir",
    "pd10_student_representation_loss",
    "run_pd10_student_epoch",
    "train_pd10_student",
]
