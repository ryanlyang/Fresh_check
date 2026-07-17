"""Coherent mean-plus-stochastic hierarchy hypotheses and their objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import require_torch

from .binary_accounting import AccountingState
from .conditional_latent import (
    ABPH_CONDITIONAL_LATENT_CONTRACT,
    ABPH_FIXED_EVALUATION_SEED,
    ABPH_PRIMARY_STOCHASTIC_HYPOTHESES,
    ConditionalLatentConfig,
    ConditionalLatentContextEncoder,
    ConditionalSplinePrior,
    TrainingOnlyHierarchyPosterior,
    VariationalLatentSample,
)
from .hierarchy_alignment import HierarchyTargetTensors
from .hierarchy_decoder import RecursiveHierarchyDecoder, RecursiveHierarchyOutput
from .targets import ROOT_FEATURE_NAMES


try:  # Keep schemas importable in cache-only environments.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ABPH_HYPOTHESIS_DISTRIBUTION_CONTRACT = (
    "adaptive_binary_pseudooffline_hypothesis_distribution_v1"
)
ABPH_PRIMARY_HYPOTHESIS_NAMES: tuple[str, ...] = (
    "mean",
    "stochastic_1",
    "stochastic_2",
    "stochastic_3",
    "stochastic_4",
)


def conditional_distribution_weight(training_fraction: float) -> float:
    """Locked 0.05 -> 0.25 warmup over the first 10% of updates."""

    progress = float(training_fraction)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("training_fraction must lie in [0, 1]")
    return 0.05 + 0.20 * min(progress / 0.10, 1.0)


@dataclass(frozen=True)
class HypothesisIdentity:
    index: int
    name: str
    kind: str
    sampling_seed: int | None
    report_identity: str

    @property
    def stochastic(self) -> bool:
        return self.kind == "stochastic"


@dataclass(frozen=True)
class HypothesisLatentSet:
    values: Any
    prior_log_prob: Any
    identities: tuple[HypothesisIdentity, ...]
    evaluation_seed: int
    diagnostics: Mapping[str, Any]

    @property
    def count(self) -> int:
        return len(self.identities)


@dataclass(frozen=True)
class HierarchyHypothesis:
    identity: HypothesisIdentity
    latent: Any
    prior_log_prob: Any
    hierarchy_outputs: Mapping[str, RecursiveHierarchyOutput]


@dataclass(frozen=True)
class MultiHypothesisHierarchyOutput:
    hypotheses: tuple[HierarchyHypothesis, ...]
    latent_set: HypothesisLatentSet
    shared_root_ledger: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class DistributionLossWeights:
    split_nll: float = 1.0
    particle_nll: float = 1.0
    latent_kl: float = 0.05
    energy_score: float = 0.25
    calibration: float = 0.10
    bounded_anti_collapse: float = 0.05
    minimum_observable_pair_distance: float = 0.02
    calibration_temperature: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "split_nll",
            "particle_nll",
            "latent_kl",
            "energy_score",
            "calibration",
            "bounded_anti_collapse",
            "minimum_observable_pair_distance",
            "calibration_temperature",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if float(self.calibration_temperature) <= 0.0:
            raise ValueError("calibration_temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributionLossOutput:
    total: Any
    split_nll: Any
    particle_nll: Any
    latent_kl: Any
    energy_score: Any
    calibration_loss: Any
    anti_collapse_penalty: Any
    metrics: Mapping[str, Any]


def _content_hash(values: Any) -> str:
    tensor = require_torch().as_tensor(values).detach().contiguous().cpu()
    payload = tensor.numpy().tobytes(order="C")
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + payload).hexdigest()


def _primary_identities(seed: int) -> tuple[HypothesisIdentity, ...]:
    identities = [
        HypothesisIdentity(
            index=0,
            name="mean",
            kind="deterministic_mean",
            sampling_seed=None,
            report_identity="abph_conditional_mean_v1",
        )
    ]
    for index in range(ABPH_PRIMARY_STOCHASTIC_HYPOTHESES):
        identities.append(
            HypothesisIdentity(
                index=index + 1,
                name=f"stochastic_{index + 1}",
                kind="stochastic",
                sampling_seed=int(seed),
                report_identity=(
                    f"abph_stochastic_{index + 1:02d}_fixed_seed_{int(seed)}_v1"
                ),
            )
        )
    return tuple(identities)


class ConditionalHierarchyHypothesisModel(_ModuleBase):
    """HLT-conditioned prior plus a training-only offline posterior."""

    def __init__(self, config: ConditionalLatentConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConditionalLatentConfig()
        self.context_encoder = ConditionalLatentContextEncoder(self.config)
        self.prior = ConditionalSplinePrior(self.config)
        self.training_only_posterior = TrainingOnlyHierarchyPosterior(self.config)

    def encode_deployment_context(
        self,
        hlt_particle_evidence: Any,
        hlt_particle_mask: Any,
        hlt_jet_embedding: Any,
        root_semantic_tokens: Any,
        shared_root_ledger: Any,
    ) -> Any:
        return self.context_encoder(
            hlt_particle_evidence,
            hlt_particle_mask,
            hlt_jet_embedding,
            root_semantic_tokens,
            shared_root_ledger,
        )

    def deployment_hypotheses(
        self,
        context: Any,
        *,
        evaluation_seed: int = ABPH_FIXED_EVALUATION_SEED,
    ) -> HypothesisLatentSet:
        torch = require_torch()
        mean = self.prior.deterministic_mean(context)
        stochastic, stochastic_log_prob = self.prior.sample(
            context,
            count=ABPH_PRIMARY_STOCHASTIC_HYPOTHESES,
            seed=int(evaluation_seed),
        )
        mean_log_prob = self.prior.log_prob(mean, context)
        values = torch.cat((mean[:, None, :], stochastic), dim=1)
        log_prob = torch.cat((mean_log_prob[:, None], stochastic_log_prob), dim=1)
        identities = _primary_identities(int(evaluation_seed))
        pairwise = torch.cdist(stochastic, stochastic, p=2)
        off_diagonal = ~torch.eye(
            stochastic.shape[1], dtype=torch.bool, device=stochastic.device
        )
        return HypothesisLatentSet(
            values=values,
            prior_log_prob=log_prob,
            identities=identities,
            evaluation_seed=int(evaluation_seed),
            diagnostics={
                "contract": ABPH_HYPOTHESIS_DISTRIBUTION_CONTRACT,
                "conditional_latent_contract": ABPH_CONDITIONAL_LATENT_CONTRACT,
                "hypothesis_count": len(identities),
                "mean_count": 1,
                "stochastic_count": ABPH_PRIMARY_STOCHASTIC_HYPOTHESES,
                "fixed_evaluation_seed": int(evaluation_seed),
                "offline_target_consumed": False,
                "node_local_noise": False,
                "latent_pair_distance_mean": pairwise[:, off_diagonal].mean().detach(),
                "report_identities": [item.report_identity for item in identities],
            },
        )

    def training_posterior_sample(
        self,
        context: Any,
        offline_target_hierarchy: HierarchyTargetTensors,
        *,
        seed: int | None = None,
    ) -> VariationalLatentSample:
        """Training-only q(z | HLT, offline hierarchy); never used for prediction."""

        return self.training_only_posterior.sample(
            context,
            offline_target_hierarchy,
            self.prior,
            seed=seed,
        )


class MultiHypothesisHierarchyReconstructor(_ModuleBase):
    """Roll every hierarchy/view below one exact, already-compiled root state."""

    def __init__(
        self,
        decoders: RecursiveHierarchyDecoder | Mapping[str, RecursiveHierarchyDecoder],
        latent_model: ConditionalHierarchyHypothesisModel | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        if isinstance(decoders, RecursiveHierarchyDecoder):
            resolved = {"primary": decoders}
        else:
            resolved = dict(decoders)
        if not resolved or any(not str(name).strip() for name in resolved):
            raise ValueError("at least one named hierarchy decoder is required")
        if any(not isinstance(value, RecursiveHierarchyDecoder) for value in resolved.values()):
            raise TypeError("every hierarchy decoder must be RecursiveHierarchyDecoder")
        self.decoders = torch.nn.ModuleDict(resolved)
        first = next(iter(resolved.values()))
        if latent_model is None:
            latent_model = ConditionalHierarchyHypothesisModel(
                ConditionalLatentConfig(
                    hlt_evidence_dim=first.config.hlt_input_dims[0],
                    hlt_jet_dim=first.config.d_model,
                    root_semantic_dim=first.config.root_semantic_dim,
                    latent_dim=first.config.latent_dim,
                )
            )
        self.latent_model = latent_model
        for name, decoder in resolved.items():
            if int(decoder.config.latent_dim) != int(latent_model.config.latent_dim):
                raise ValueError(f"decoder {name} and latent dimensions differ")
            if int(decoder.config.hlt_input_dims[0]) != int(
                latent_model.config.hlt_evidence_dim
            ):
                raise ValueError(f"decoder {name} first HLT evidence dimension differs")
            if int(decoder.config.root_semantic_dim) != int(
                latent_model.config.root_semantic_dim
            ):
                raise ValueError(f"decoder {name} root semantic dimension differs")

    @staticmethod
    def _first_hlt_source(hlt_particle_embeddings: Any) -> Any:
        if isinstance(hlt_particle_embeddings, (tuple, list)):
            if not hlt_particle_embeddings:
                raise ValueError("HLT particle embedding sources are empty")
            return hlt_particle_embeddings[0]
        return hlt_particle_embeddings

    def encode_deployment_context(
        self,
        root_state: AccountingState,
        root_hidden: Any,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
    ) -> Any:
        return self.latent_model.encode_deployment_context(
            self._first_hlt_source(hlt_particle_embeddings),
            hlt_particle_mask,
            root_hidden,
            root_semantic_tokens,
            root_state.ledger,
        )

    def deployment_rollout(
        self,
        root_state: AccountingState,
        root_hidden: Any,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
        hlt_support_features: Any,
        *,
        evaluation_seed: int = ABPH_FIXED_EVALUATION_SEED,
    ) -> MultiHypothesisHierarchyOutput:
        latent_set = self.prepare_deployment_hypotheses(
            root_state,
            root_hidden,
            root_semantic_tokens,
            hlt_particle_embeddings,
            hlt_particle_mask,
            evaluation_seed=int(evaluation_seed),
        )
        hypotheses = self.rollout_deployment_hypotheses(
            root_state,
            root_hidden,
            root_semantic_tokens,
            hlt_particle_embeddings,
            hlt_particle_mask,
            hlt_support_features,
            latent_set=latent_set,
            hypothesis_indices=range(latent_set.count),
        )
        return self.assemble_deployment_rollout(
            root_state,
            latent_set=latent_set,
            hypotheses=hypotheses,
        )

    def prepare_deployment_hypotheses(
        self,
        root_state: AccountingState,
        root_hidden: Any,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
        *,
        evaluation_seed: int = ABPH_FIXED_EVALUATION_SEED,
    ) -> HypothesisLatentSet:
        """Create the ordered mean-plus-stochastic latent set exactly once."""

        context = self.encode_deployment_context(
            root_state,
            root_hidden,
            root_semantic_tokens,
            hlt_particle_embeddings,
            hlt_particle_mask,
        )
        return self.latent_model.deployment_hypotheses(
            context, evaluation_seed=int(evaluation_seed)
        )

    def rollout_deployment_hypotheses(
        self,
        root_state: AccountingState,
        root_hidden: Any,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
        hlt_support_features: Any,
        *,
        latent_set: HypothesisLatentSet,
        hypothesis_indices: Sequence[int],
    ) -> tuple[HierarchyHypothesis, ...]:
        """Roll out an audited ordered subset of one prepared latent set."""

        indices = tuple(int(value) for value in hypothesis_indices)
        if not indices:
            return ()
        if indices != tuple(sorted(set(indices))):
            raise ValueError("hypothesis indices must be unique and increasing")
        if indices[0] < 0 or indices[-1] >= latent_set.count:
            raise IndexError("hypothesis index is outside the prepared latent set")
        hypotheses: list[HierarchyHypothesis] = []
        for hypothesis_index in indices:
            identity = latent_set.identities[hypothesis_index]
            outputs = {}
            for hierarchy_name, decoder in self.decoders.items():
                output = decoder(
                    root_state,
                    root_hidden,
                    root_semantic_tokens,
                    hlt_particle_embeddings,
                    hlt_particle_mask,
                    hlt_support_features,
                    mode="rollout",
                    hypothesis_latent=latent_set.values[:, hypothesis_index],
                )
                outputs[hierarchy_name] = output
            hypotheses.append(
                HierarchyHypothesis(
                    identity=identity,
                    latent=latent_set.values[:, hypothesis_index],
                    prior_log_prob=latent_set.prior_log_prob[:, hypothesis_index],
                    hierarchy_outputs=outputs,
                )
            )
        return tuple(hypotheses)

    def assemble_deployment_rollout(
        self,
        root_state: AccountingState,
        *,
        latent_set: HypothesisLatentSet,
        hypotheses: Sequence[HierarchyHypothesis],
    ) -> MultiHypothesisHierarchyOutput:
        """Assemble the complete schema after indexed rollout without recomputation."""

        torch = require_torch()
        ordered = tuple(hypotheses)
        if len(ordered) != latent_set.count:
            raise ValueError("deployment assembly requires every prepared hypothesis")
        for index, hypothesis in enumerate(ordered):
            if hypothesis.identity is not latent_set.identities[index]:
                raise ValueError("deployment hypotheses do not preserve latent identity objects")
            if hypothesis.latent is not latent_set.values[:, index]:
                # Tensor slicing creates a new view object on each expression, so
                # exact storage/value identity is the enforceable contract here.
                expected = latent_set.values[:, index]
                if not torch.equal(hypothesis.latent, expected):
                    raise ValueError("deployment hypothesis latent differs from its prepared value")
            if set(hypothesis.hierarchy_outputs) != set(self.decoders):
                raise ValueError("deployment hypothesis hierarchy membership mismatch")
        root_ledgers = [
            output.root_frontier.ledger
            for hypothesis in ordered
            for output in hypothesis.hierarchy_outputs.values()
        ]
        stacked_roots = torch.stack(root_ledgers, dim=1)
        shared = root_state.ledger[:, None, None, :]
        exact = bool(torch.equal(stacked_roots, shared.expand_as(stacked_roots)))
        maximum_variance = float(
            stacked_roots.to(torch.float64).var(dim=1, unbiased=False).max().detach().cpu()
        )
        if not exact or maximum_variance != 0.0:
            raise RuntimeError("primary hypotheses did not preserve the exact shared root")
        return MultiHypothesisHierarchyOutput(
            hypotheses=ordered,
            latent_set=latent_set,
            shared_root_ledger=root_state.ledger,
            diagnostics={
                "contract": ABPH_HYPOTHESIS_DISTRIBUTION_CONTRACT,
                "shared_root_compiled_once_upstream": True,
                "root_compiler_calls_inside_hypothesis_model": 0,
                "root_sampled": False,
                "exact_root_identity_across_all_hypotheses_and_hierarchies": exact,
                "root_hard_maximum_variance": maximum_variance,
                "shared_root_content_hash": _content_hash(root_state.ledger),
                "hierarchy_names": list(self.decoders.keys()),
                "offline_target_consumed": False,
                "deployment_hypothesis_selection": "all_fixed_views_returned",
            },
        )

    def forward(self, *args: Any, **kwargs: Any) -> MultiHypothesisHierarchyOutput:
        return self.deployment_rollout(*args, **kwargs)


def deployment_hypothesis_indices(
    hypotheses: HypothesisLatentSet,
    *,
    offline_target_scores: Any | None = None,
) -> tuple[int, ...]:
    if offline_target_scores is not None:
        raise ValueError("offline targets may not select deployment hypotheses")
    return tuple(range(hypotheses.count))


def model_val_oracle_best_hypothesis(
    offline_target_scores: Any,
    *,
    split: str,
    analysis_only: bool,
) -> Mapping[str, Any]:
    if str(split) != "model_val" or not bool(analysis_only):
        raise ValueError("oracle best-hypothesis selection is model_val analysis only")
    scores = require_torch().as_tensor(offline_target_scores)
    if scores.ndim != 2 or scores.shape[1] != len(ABPH_PRIMARY_HYPOTHESIS_NAMES):
        raise ValueError("oracle scores must have shape [B, 5]")
    return {
        "analysis_only": True,
        "deployable": False,
        "split": "model_val",
        "label": "offline_oracle_best_hypothesis_ceiling",
        "best_indices": scores.argmin(dim=1),
        "best_score_mean": scores.min(dim=1).values.mean(),
    }


def _mean_tensor(values: Any, *, reference: Any) -> Any:
    torch = require_torch()
    tensor = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    return tensor.mean()


def _effective_rank(values: Any) -> Any:
    torch = require_torch()
    centered = values - values.mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular.square()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    return entropy.exp()


def compute_distribution_losses(
    posterior_sample: VariationalLatentSample,
    stochastic_observables: Any,
    target_observables: Any,
    *,
    split_negative_log_likelihood: Any,
    particle_negative_log_likelihood: Any,
    weights: DistributionLossWeights | None = None,
) -> DistributionLossOutput:
    """Proper variational/energy objective plus bounded anti-collapse pressure."""

    torch = require_torch()
    resolved = weights or DistributionLossWeights()
    views = torch.as_tensor(stochastic_observables)
    targets = torch.as_tensor(target_observables, device=views.device, dtype=views.dtype)
    if views.ndim != 3 or views.shape[1] < 2:
        raise ValueError("stochastic observables must have shape [B, H>=2, O]")
    if targets.shape != (views.shape[0], views.shape[2]):
        raise ValueError("target observables must have shape [B, O]")
    split_nll = _mean_tensor(split_negative_log_likelihood, reference=views)
    particle_nll = _mean_tensor(particle_negative_log_likelihood, reference=views)
    latent_kl = posterior_sample.monte_carlo_kl.mean()
    target_distance = torch.linalg.vector_norm(views - targets[:, None, :], dim=-1)
    pairwise = torch.cdist(views, views, p=2)
    off_diagonal = ~torch.eye(views.shape[1], dtype=torch.bool, device=views.device)
    pair_distance = pairwise[:, off_diagonal]
    energy_score = target_distance.mean() - 0.5 * pairwise.mean()
    anti_collapse = torch.relu(
        views.new_tensor(float(resolved.minimum_observable_pair_distance))
        - pair_distance.mean(dim=1)
    ).mean()

    mean = views.mean(dim=1)
    standard_deviation = views.std(dim=1, unbiased=False).clamp_min(1.0e-6)
    absolute_error = (targets - mean).abs()
    calibration_terms = []
    coverages = {}
    for label, z_score, desired in (("50", 0.67448975, 0.50), ("90", 1.64485363, 0.90)):
        margin = z_score * standard_deviation - absolute_error
        soft_coverage = torch.sigmoid(
            margin / float(resolved.calibration_temperature)
        ).mean()
        calibration_terms.append((soft_coverage - desired).square())
        coverages[f"coverage_{label}"] = soft_coverage.detach()
    calibration_loss = torch.stack(calibration_terms).mean()
    total = (
        float(resolved.split_nll) * split_nll
        + float(resolved.particle_nll) * particle_nll
        + float(resolved.latent_kl) * latent_kl
        + float(resolved.energy_score) * energy_score
        + float(resolved.calibration) * calibration_loss
        + float(resolved.bounded_anti_collapse) * anti_collapse
    )
    finite = all(
        bool(torch.isfinite(value))
        for value in (
            total,
            split_nll,
            particle_nll,
            latent_kl,
            energy_score,
            calibration_loss,
            anti_collapse,
        )
    )
    if not finite:
        raise FloatingPointError("nonfinite conditional-distribution objective")
    metrics = {
        "contract": ABPH_HYPOTHESIS_DISTRIBUTION_CONTRACT,
        "weights": resolved.to_dict(),
        "view_pair_distance_mean": pair_distance.mean().detach(),
        "view_pair_distance_min_mean": pair_distance.min(dim=1).values.mean().detach(),
        "hypothesis_effective_rank_mean": _effective_rank(views).mean().detach(),
        "collapsed_example_fraction": (
            pair_distance.mean(dim=1) < float(resolved.minimum_observable_pair_distance)
        ).to(views.dtype).mean().detach(),
        "predictive_standard_deviation_mean": standard_deviation.mean().detach(),
        "absolute_error_mean": absolute_error.mean().detach(),
        "anti_collapse_scope": "caller_supplied_fine_structure_observables",
        **coverages,
    }
    return DistributionLossOutput(
        total=total,
        split_nll=split_nll,
        particle_nll=particle_nll,
        latent_kl=latent_kl,
        energy_score=energy_score,
        calibration_loss=calibration_loss,
        anti_collapse_penalty=anti_collapse,
        metrics=metrics,
    )


__all__ = [
    "ABPH_HYPOTHESIS_DISTRIBUTION_CONTRACT",
    "ABPH_PRIMARY_HYPOTHESIS_NAMES",
    "ConditionalHierarchyHypothesisModel",
    "DistributionLossOutput",
    "DistributionLossWeights",
    "HierarchyHypothesis",
    "HypothesisIdentity",
    "HypothesisLatentSet",
    "MultiHypothesisHierarchyOutput",
    "MultiHypothesisHierarchyReconstructor",
    "compute_distribution_losses",
    "conditional_distribution_weight",
    "deployment_hypothesis_indices",
    "model_val_oracle_best_hypothesis",
]
