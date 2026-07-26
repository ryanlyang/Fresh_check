"""Exact Weaver standard-four pair construction and runtime provenance."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from .contracts import require_sha256, with_content_hash

try:  # Keep Step-1 contract imports usable without the training stack.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


PAIR_BASE_CONTRACT = "relational_part_pair_base_v1"
WEAVER_RUNTIME_CONTRACT = "relational_part_weaver_runtime_v1"
STANDARD_FOUR_FEATURE_NAMES = ("lnkt", "lnz", "lndelta", "lnm2")
STANDARD_FOUR_CHANNELS = 4
_WEAVER_MODULE = "weaver.nn.model.ParticleTransformer"


def require_torch():
    if _torch is None:  # pragma: no cover - environment dependent
        raise ImportError("relational Particle Transformer models require PyTorch")
    return _torch


def _import_weaver_module():
    try:
        return importlib.import_module(_WEAVER_MODULE)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Step 2 requires the real weaver-core ParticleTransformer module"
        ) from exc


def resolve_weaver_pairwise_helper(module: Any | None = None) -> tuple[str, Any]:
    """Resolve only the standard proton-proton helper APIs we authenticate."""

    module = _import_weaver_module() if module is None else module
    for name in ("pairwise_lv_fts", "pairwise_lv_fts_pp"):
        helper = getattr(module, name, None)
        if helper is None:
            continue
        signature = inspect.signature(helper)
        parameters = tuple(signature.parameters)
        if len(parameters) < 3 or parameters[:3] != ("xi", "xj", "num_outputs"):
            raise RuntimeError(
                f"Weaver {name} has an unsupported signature {signature}"
            )
        return name, helper
    raise RuntimeError(
        "installed Weaver exports neither pairwise_lv_fts nor pairwise_lv_fts_pp"
    )


def _validate_inputs(lorentz_vectors: Any, mask: Any | None) -> tuple[Any, Any | None]:
    torch = require_torch()
    if not isinstance(lorentz_vectors, torch.Tensor):
        raise TypeError("lorentz_vectors must be a torch.Tensor")
    if lorentz_vectors.ndim != 3 or int(lorentz_vectors.shape[1]) != 4:
        raise ValueError(
            "lorentz_vectors must have shape [batch,4,constituents], got "
            f"{tuple(lorentz_vectors.shape)}"
        )
    if not lorentz_vectors.is_floating_point():
        raise TypeError("lorentz_vectors must use a floating dtype")
    if not bool(torch.isfinite(lorentz_vectors).all()):
        raise FloatingPointError("lorentz_vectors contain NaN or infinity")
    if int(lorentz_vectors.shape[0]) <= 0 or int(lorentz_vectors.shape[2]) <= 0:
        raise ValueError("pair construction requires nonempty batch and sequence axes")
    if mask is None:
        return lorentz_vectors, None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    expected = (
        int(lorentz_vectors.shape[0]),
        1,
        int(lorentz_vectors.shape[2]),
    )
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask must have shape {expected}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("non-boolean masks must contain only zero or one")
        mask = mask.bool()
    return lorentz_vectors, mask


def build_standard_four_pair_features(
    lorentz_vectors: Any,
    *,
    mask: Any | None = None,
    module: Any | None = None,
) -> Any:
    """Generate Weaver's exact dense standard-four tensor as ``uu``.

    Padding is deliberately not zeroed here.  The reference Weaver path first
    computes its physical four-vector features and then relies on the attention
    padding mask.  Altering padded raw values before the shared pair encoder
    would break the Step-2 parity target.
    """

    torch = require_torch()
    lorentz_vectors, _ = _validate_inputs(lorentz_vectors, mask)
    helper_name, helper = resolve_weaver_pairwise_helper(module)
    # Weaver 0.4's generic PairEmbed constructs physical features inside a
    # no-grad block.  The newer pp-specific implementation intentionally does
    # not.  Preserve that input-gradient behavior as part of runtime parity.
    context = torch.no_grad() if helper_name == "pairwise_lv_fts" else nullcontext()
    with context:
        pair = helper(
            lorentz_vectors.unsqueeze(-1),
            lorentz_vectors.unsqueeze(-2),
            num_outputs=STANDARD_FOUR_CHANNELS,
        )
    expected = (
        int(lorentz_vectors.shape[0]),
        STANDARD_FOUR_CHANNELS,
        int(lorentz_vectors.shape[2]),
        int(lorentz_vectors.shape[2]),
    )
    if tuple(pair.shape) != expected:
        raise RuntimeError(
            f"Weaver standard-four helper returned {tuple(pair.shape)}, "
            f"expected {expected}"
        )
    if pair.dtype != lorentz_vectors.dtype:
        raise RuntimeError(
            "Weaver standard-four helper changed dtype: "
            f"{lorentz_vectors.dtype} -> {pair.dtype}"
        )
    if not bool(torch.isfinite(pair).all()):
        raise FloatingPointError("Weaver standard-four pair tensor is nonfinite")
    return pair


def build_pair_base_contract(
    *,
    relation_registry_sha256: str,
    global_determinism_sha256: str,
) -> dict[str, Any]:
    """Build the training-independent Step-2 pair-path contract."""

    return with_content_hash(
        {
            "contract": PAIR_BASE_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "standard_four": {
                "feature_names": list(STANDARD_FOUR_FEATURE_NAMES),
                "channel_count": STANDARD_FOUR_CHANNELS,
                "source": "installed_Weaver_proton_proton_pairwise_helper",
                "helper_names_allowed": [
                    "pairwise_lv_fts",
                    "pairwise_lv_fts_pp",
                ],
                "layout": "[batch,channels,query,context]",
                "dtype": "same_as_lorentz_vectors",
                "persistent_cache_allowed": False,
                "padding_raw_feature_zeroing_allowed": False,
            },
            "explicit_path": {
                "argument": "uu",
                "lorentz_vectors_still_passed_as_v": True,
                "pair_embed_uses_explicit_uu_only": True,
                "symmetric_lower_triangle_embedding": True,
                "state_dictionary_keys_shapes_dtypes_preserved": True,
                "reference_pair_encoder_parameters_reused": True,
            },
            "authoritative_parity": {
                "precision": "float32",
                "autocast": False,
                "required": [
                    "standard_four_features",
                    "pair_bias",
                    "logits",
                    "input_gradients",
                    "parameter_gradients",
                    "state_dictionary",
                    "padding",
                    "forced_nonempty",
                    "one_particle",
                ],
            },
        }
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_source_identity(module: Any) -> tuple[str, str]:
    source_name = inspect.getsourcefile(module) or str(module.__file__)
    source_path = Path(source_name)
    if source_path.is_file():
        return str(source_path.resolve()), _sha256_path(source_path)
    loader = getattr(module, "__loader__", None)
    get_data = getattr(loader, "get_data", None)
    if callable(get_data):
        try:
            encoded = get_data(source_name)
        except (OSError, KeyError):
            encoded = None
        if encoded is not None:
            return source_name, hashlib.sha256(encoded).hexdigest()
    try:
        encoded = inspect.getsource(module).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise FileNotFoundError(
            f"Weaver source is unavailable for hashing: {source_name}"
        ) from exc
    return source_name, hashlib.sha256(encoded).hexdigest()


def inspect_weaver_runtime(module: Any | None = None) -> dict[str, Any]:
    """Record installed source, versions, and exact callable signatures."""

    torch = require_torch()
    module = _import_weaver_module() if module is None else module
    helper_name, helper = resolve_weaver_pairwise_helper(module)
    module_file, module_sha256 = _module_source_identity(module)
    try:
        package_version = importlib.metadata.version("weaver-core")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unregistered_editable_or_test_fixture"
    pair_embed = getattr(module, "PairEmbed", None)
    transformer = getattr(module, "ParticleTransformer", None)
    if pair_embed is None or transformer is None:
        raise RuntimeError("Weaver module lacks PairEmbed or ParticleTransformer")
    return with_content_hash(
        {
            "contract": WEAVER_RUNTIME_CONTRACT,
            "schema_version": 1,
            "module": _WEAVER_MODULE,
            "module_file": module_file,
            "module_sha256": module_sha256,
            "weaver_core_version": package_version,
            "torch_version": str(torch.__version__),
            "helper_name": helper_name,
            "signatures": {
                "pairwise_helper": str(inspect.signature(helper)),
                "ParticleTransformer.__init__": str(
                    inspect.signature(transformer.__init__)
                ),
                "ParticleTransformer.forward": str(
                    inspect.signature(transformer.forward)
                ),
                "PairEmbed.forward": str(inspect.signature(pair_embed.forward)),
            },
        }
    )


__all__ = [
    "PAIR_BASE_CONTRACT",
    "STANDARD_FOUR_CHANNELS",
    "STANDARD_FOUR_FEATURE_NAMES",
    "WEAVER_RUNTIME_CONTRACT",
    "build_pair_base_contract",
    "build_standard_four_pair_features",
    "inspect_weaver_runtime",
    "require_torch",
    "resolve_weaver_pairwise_helper",
]
