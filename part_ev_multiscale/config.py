"""Locked scientific and platform contracts (no resource-tuning model flags)."""
from copy import deepcopy
from part_ev_fullscale.protocol import RECIPE, digest

SCHEMA = 'part_ev_multiscale_v1'
# Operational admission only: do not change the scientific/benchmark policy or
# invalidate already-frozen campaigns just to relax speculative reservations.
STORAGE_POLICY = 'campaign_remaining_plus_reserve_aggregate_advisory_v1'
SCIENCE = dict(version='nested_multiscale_state_layerwise_v1', width=128, fine=16, coarse=4, heads=8,
    stages_after=[2, 4, 6, 8], ffn_width=512, norm_epsilon=1e-5, dropout=0.,
    biased_projections=True, independent_stages=True, checkpoint_stages=True,
    initialization='isolated_xavier_four_zero_feedback_one_zero_fusion_v1',
    geometry='fp32_escheme_angular_canonical_distinct_atoms_exact_ties_v1',
    coordinates='rapidity_wrapped_phi_eps1e-6', atom_key='px_py_pz_E_signed_zero_normalized',
    ties='lexicographic_minimum_atom_keys', cuts='one_history_16_then_4',
    descriptors=['log_pt_share','log1p_mass','delta_y','sin_delta_phi','cos_delta_phi',
                 'log1p_original_occupancy','log1p_pt_weighted_width'],
    fine_attention='coarse_siblings_including_self', coarse_attention='global',
    topdown='all_active_coarse', feedback='own_fine_own_coarse_and_particle',
    max_pool='scatter_reduce_amax_equal_tie_gradients', norms_per_stage=14,
    precision='fp32_norm_scores_softmax_reductions_residuals_fp16_linears_v1',
    native='full_noncompact_state_layerwise_width32_query64', seed=101)
COUNTS = dict(control=2327462, multiscale=5366950)
TASKS = [dict(architecture=k, run_id=r, seed=101, recipe='ranger_full_v1') for k, r in
         [('control', 'LW_STATE_EV_CONTROL'), ('multiscale', 'MULTISCALE_STATE_EV_LAYERWISE')]]
PLATFORMS = {
    'sporc': dict(machine='x86_64', partition='tier3', gpu='a100', gpus=2,
        local_batch=256, workers=1, threads=2, cpus=8, memory_gib=128,
        batchnorm='SyncBatchNorm', python='/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python'),
    'tigris': dict(machine='aarch64', partition='tigris', gpu='gh200', gpus=1,
        local_batch=512, workers=2, threads=4, cpus=12, memory_gib=128,
        batchnorm='BatchNorm', python='/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python'),
}
POLICY = dict(warmup=20, measured=4200, windows=[1000,1000,1000,1000,200],
    fixed_warmup=10, fixed_steps=40, validation_events=128000, validation_batch=512,
    inference_batch=256, inference_warmup=10, inference_steps=20,
    latency_warmup=10, latency_steps=100, checkpoint_interval=1000,
    heartbeat=100, max_seconds=18000, shutdown_margin=900,
    max_skip_fraction=.05, memory_fraction=.90, margin=1.30,
    reserve_bytes=4*2**30, benchmark_bytes=2*2**30)
PARITY = dict(forward_rtol=1e-5, forward_atol=1e-6, gradient_rtol=1e-4,
    gradient_atol=1e-5, amp_forward_rtol=2e-3, amp_forward_atol=2e-3,
    amp_gradient_rtol=1e-2, amp_gradient_atol=1e-3,
    ddp_forward_rtol=5e-4, ddp_gradient_rtol=2e-3, ddp_atol=2e-5,
    restart_absolute=1e-4, restart_atol=1e-5, restart_rtol=1e-4,
    loss_absolute=2e-4, loss_atol=1e-4, loss_rtol=1e-5)


def recipe():
    row=deepcopy(RECIPE)
    # Preserve optimizer/LR/exposure arithmetic, but do not mislabel this LW
    # backbone or the explicitly platform-specific loader/normalization policy.
    row.update(evaluation_population=5_000_000,pair_bias='layerwise',
        workers_per_rank='resources.workers',ddp_batchnorm='resources.batchnorm')
    return row


def science_hash():
    return digest(dict(science=SCIENCE, counts=COUNTS, recipe=recipe(), parity=PARITY))


def arm_budget(parameters):
    return (51*(8*parameters+2**20) + 3*(24*parameters+16*2**20)
            + 5_000_000*48+64*2**20 + 2**30)


def campaign_budget():
    return POLICY['benchmark_bytes'] + sum(arm_budget(p) for p in COUNTS.values())

