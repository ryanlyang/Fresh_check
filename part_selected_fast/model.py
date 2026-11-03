"""Parameter-identical execution adapter; no historical global state mutation."""
import types
import torch
from part_selected_screen.model import build_model as reference_model
from part_selected_screen.features import Relations
from .trees import region_from_packed


def build_model(architecture, *, normalization, backend, provider, seed=101, trim=True, use_amp=False,
                cache_validation=False, compile_region=False):
    model = reference_model(architecture, normalization=normalization, backend=backend,
                            seed=seed, trim=trim, use_amp=use_amp)
    relation = model.mod.pair_embed.relations
    feature_fn = region_from_packed
    if compile_region:
        feature_fn = torch.compile(feature_fn, dynamic=True, fullgraph=True)

    def trees(v,r,m,unused):
        return provider(v,r,m,use_cache=cache_validation and not relation.training)

    # Reuse the precise existing arithmetic/autograd order. Only the CPU tree
    # transport/packing path is replaced. Functions live on this instance.
    for name in ('forward', '_forward_reference'):
        original = getattr(Relations,name)
        namespace = dict(original.__globals__, trees_for=trees, region_features=feature_fn)
        if name == '_forward_reference':
            raw = original.__globals__['raw_features']
            namespace['raw_features'] = types.FunctionType(raw.__code__, dict(raw.__globals__,
                trees_for=trees, region_features=feature_fn), raw.__name__)
        function = types.FunctionType(original.__code__, namespace, original.__name__)
        object.__setattr__(relation,name,types.MethodType(function,relation))
    object.__setattr__(model,'fast_provider',provider)
    return model
