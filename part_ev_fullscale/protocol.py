"""Immutable experiment recipe and small, atomic campaign artifacts."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile

SCHEMA = "part_ev_jetclass100m_v1"
SEEDS = (101, 202, 303)
COMPARISONS = {
    "standard": ("part", "part_ev"),
    "plain": ("part_plain", "part_plain_ev"),
}
TRAINING_RECIPES = ("original", "adamw")
TASKS = [{"architecture": architecture, "seed": seed}
         for architecture in ("part", "part_ev") for seed in SEEDS]
CLASS_NAMES = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")
RECIPE = {
    "global_batch_size": 512,
    "epochs": 50,
    "steps_per_epoch": 20000,
    "total_steps": 1000000,
    "validation_events_per_epoch": 1280000,
    "validation_population": 5000000,
    "training_population": 100000000,
    "test_population": 20000000,
    "optimizer": "Weaver_RAdam_plus_Lookahead",
    "betas": [0.95, 0.999], "eps": 1e-5, "weight_decay": 0.0,
    "lookahead_k": 6, "lookahead_alpha": 0.5,
    "start_lr": 1e-3, "final_lr": 1e-5,
    "lr_decay_epochs": list(range(35, 50)),
    "gradient_clipping": None,
    "precision": "fp16_autocast_with_fp32_parameters_and_grad_scaler",
    "checkpoint_selection": "highest_validation_accuracy_first_tie",
    "early_stopping": False,
    "pair_bias": "shared", "selected_relations": False,
    "padding": "official_Weaver_wrap_with_constant_mask",
    "particle_limit": 128, "workers_per_rank": 2, "fetch_step": 0.01,
    "ddp_batchnorm": "SyncBatchNorm_as_reference_launcher",
    "ddp_trimming": "native_per_rank_SequenceTrimmer_as_reference_launcher",
    "ddp_schedule_adaptation": "keep_single_GPU_global_batch_and_step_budget_not_upstream_local_batch_scaling",
    "stream": "continuous_across_validation_epochs",
    "resume_stream": "deterministic_replay_without_forward_or_optimizer_steps",
    "validation_stream": "separate_continuous_rank0_stream",
    "weaver_version": "0.4.17",
}


def tasks_for(comparison="standard", training_recipe="original"):
    if comparison not in COMPARISONS:
        raise ValueError("Unknown comparison; choose standard or plain")
    if training_recipe not in TRAINING_RECIPES:
        raise ValueError("Unknown training recipe; choose original or adamw")
    if training_recipe == "adamw" and comparison != "standard":
        raise ValueError("The AdamW pilot is defined only for the standard comparison")
    seeds = (101,) if training_recipe == "adamw" else SEEDS
    return [{"architecture": architecture, "seed": seed}
            for architecture in COMPARISONS[comparison] for seed in seeds]


def recipe_for(comparison="standard", training_recipe="original"):
    tasks_for(comparison, training_recipe)  # Reject unsupported combinations first.
    recipe = copy.deepcopy(RECIPE)
    if comparison == "plain":
        recipe["pair_bias"] = "none"
    if training_recipe == "adamw":
        recipe.update({
            "optimizer": "AdamW",
            "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4,
            "lookahead_k": None, "lookahead_alpha": None,
            "lr_decay_epochs": [],
            "lr_schedule": "linear_warmup_then_cosine", "warmup_steps": 50000,
            "gradient_clipping": 1.0,
            "weight_decay_policy": "exclude_bias_normalization_and_cls_token_include_edge_projection",
        })
    return recipe


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def file_lock(path, *, blocking=True):
    """A kernel lock, released even on process death; never a stale PID sentinel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def local_batch_size(world_size):
    if world_size not in (1, 2, 4):
        raise ValueError("This protocol supports exactly 1, 2 or 4 GPUs")
    return RECIPE["global_batch_size"] // world_size


def learning_rate(step, recipe=None):
    """LR for a zero-indexed update; preserve the original milestone arithmetic.

    The AdamW pilot uses one-based update ordinals for its warm-up/cosine
    schedule: update 50,000 reaches the peak and update 1,000,000 the floor.
    """
    recipe = RECIPE if recipe is None else recipe
    if not 0 <= step < recipe["total_steps"]:
        raise ValueError("Training step outside the frozen budget")
    if recipe.get("lr_schedule") == "linear_warmup_then_cosine":
        ordinal = step + 1
        warmup = recipe["warmup_steps"]
        if not 0 < warmup < recipe["total_steps"]:
            raise ValueError("Warm-up must be positive and shorter than the training budget")
        if ordinal <= warmup:
            return recipe["start_lr"] * ordinal / warmup
        progress = (ordinal - warmup) / (recipe["total_steps"] - warmup)
        return recipe["final_lr"] + 0.5 * (recipe["start_lr"] - recipe["final_lr"]) * (
            1 + math.cos(math.pi * progress))
    if "lr_schedule" in recipe:
        raise ValueError("Unknown learning-rate schedule")
    epoch = step // recipe["steps_per_epoch"]
    decays = sum(epoch >= milestone for milestone in recipe["lr_decay_epochs"])
    return recipe["start_lr"] * (0.01 ** (1 / 15)) ** decays


def environment():
    import torch
    import weaver
    package = Path(weaver.__file__).parent
    version = importlib.metadata.version("weaver-core")
    if version != RECIPE["weaver_version"]:
        raise RuntimeError(f"Need weaver-core=={RECIPE['weaver_version']}; found {version}. "
                           "Use a dedicated environment; do not modify an active campaign environment.")
    sources = ("nn/model/ParticleTransformer.py", "utils/dataset.py", "utils/data/config.py",
               "utils/data/tools.py", "utils/data/fileio.py", "utils/nn/optimizer/radam.py")
    return {
        "python": platform.python_version(), "architecture": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "weaver-core", "numpy", "awkward", "uproot", "PyYAML", "scipy", "scikit-learn")},
        "torch_cuda": torch.version.cuda,
        "weaver_sources": {name: file_sha256(package / name) for name in sources},
    }


def load_campaign(output, *, check_environment=True):
    spec = read_json(Path(output) / "campaign.json")
    content_hash = spec.pop("content_hash", None)
    if digest(spec) != content_hash or spec.get("schema") != SCHEMA:
        raise ValueError("Campaign metadata was changed or is incompatible")
    spec["content_hash"] = content_hash
    # Existing campaigns predate these fields; never insert defaults into their
    # metadata, since doing so would change their frozen content hashes.
    comparison = spec.get("comparison", "standard")
    training_recipe = spec.get("training_recipe", "original")
    if (spec["recipe"] != recipe_for(comparison, training_recipe)
            or spec["tasks"] != tasks_for(comparison, training_recipe)):
        raise ValueError("Campaign differs from the frozen comparison/training-recipe protocol")
    config = Path(__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
    if spec["data_config_sha256"] != file_sha256(config):
        raise ValueError("Official input preprocessing YAML changed after preparation")
    if check_environment and spec["environment"] != environment():
        raise ValueError("Runtime environment differs from prepare; use the same pinned environment")
    return spec


def task_directory(output, task_index, *, tasks=None):
    tasks = TASKS if tasks is None else tasks
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"Task index must be 0..{len(tasks) - 1}")
    task = tasks[task_index]
    return Path(output) / "runs" / task["architecture"] / f"seed_{task['seed']}"
