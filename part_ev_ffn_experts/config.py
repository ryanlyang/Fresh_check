"""Frozen model, auxiliary loss, precision and independent runtime contracts."""
from copy import deepcopy
from part_ev_fullscale.protocol import RECIPE,digest

SCHEMA='part_ev_ffn_experts_v1'
STORAGE_POLICY='campaign_remaining_plus_reserve_aggregate_advisory_v1'
SCIENCE=dict(version='contextual_ffn_experts_state_layerwise_v1',width=128,
    blocks=[2,3,4,5,6,7],experts=4,top_k=2,expert_width=256,router_width=32,
    norm_epsilon=1e-5,expert_order='linear_gelu_layernorm_linear',
    context='differentiable_masked_mean_then_affine_norm',router_inputs=['u','g','u*g'],
    routing='stable_descending_expert_id_ties',combination='selected_two_softmax',
    dispatch='dropless_disjoint_assignment_slots_fixed_expert_order',
    empty_experts='graph_connected_empty_inputs_no_dummy_tokens',
    initialization='isolated_xavier_inputs_normal_router_head_zero_expert_outputs_v1',
    checkpoint='expert_mlp_nonreentrant_no_rng_no_rerouting',
    precision='fp32_router_norms_reductions_fp16_native_and_expert_linears_v1',
    alpha=.001,balance='four_probability_times_detached_top2_fraction_mean_six_banks',
    ddp_balance='detached_global_statistics_world_scaled_local_gradient_value_corrected_v1',
    native='full_noncompact_state_layerwise_width32_query64',seed=101)
COUNTS=dict(control=2327462,experts=3998078)
TASKS=[dict(architecture=a,run_id=r,seed=101,recipe='ranger_full_v1') for a,r in
       [('control','LW_STATE_EV_CONTROL'),('experts','EXPERTS_STATE_EV_LAYERWISE')]]
PLATFORMS={
 'sporc':dict(machine='x86_64',partition='tier3',gpu='a100',gpus=2,local_batch=256,
    workers=1,threads=2,cpus=8,memory_gib=128,batchnorm='SyncBatchNorm',
    python='/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python'),
 'tigris':dict(machine='aarch64',partition='tigris',gpu='gh200',gpus=1,local_batch=512,
    workers=2,threads=4,cpus=12,memory_gib=128,batchnorm='BatchNorm',
    python='/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python')}
POLICY=dict(warmup=20,measured=4200,windows=[1000,1000,1000,1000,200],fixed_warmup=10,
    fixed_steps=40,validation_events=128000,validation_batch=512,inference_batch=256,
    inference_warmup=10,inference_steps=20,latency_warmup=10,latency_steps=100,
    checkpoint_interval=1000,heartbeat=100,max_seconds=18000,shutdown_margin=900,
    max_skip_fraction=.05,memory_fraction=.90,margin=1.30,reserve_bytes=4*2**30,
    benchmark_bytes=2*2**30)
PARITY=dict(forward_rtol=1e-5,forward_atol=1e-6,gradient_rtol=1e-4,gradient_atol=1e-5,
    amp_forward_rtol=2e-3,amp_forward_atol=2e-3,amp_gradient_rtol=1e-2,amp_gradient_atol=1e-3,
    ddp_forward_rtol=5e-4,ddp_gradient_rtol=2e-3,ddp_atol=2e-5,
    restart_absolute=1e-4,restart_atol=1e-5,restart_rtol=1e-4,
    loss_absolute=2e-4,loss_atol=1e-4,loss_rtol=1e-5,route_ids='exact',route_counts='exact')


def recipe():
    row=deepcopy(RECIPE)
    row.update(evaluation_population=5000000,pair_bias='layerwise',
               workers_per_rank='resources.workers',ddp_batchnorm='resources.batchnorm')
    return row


def science_hash():return digest(dict(science=SCIENCE,counts=COUNTS,recipe=recipe(),parity=PARITY))


def arm_budget(p):return 51*(8*p+2**20)+3*(24*p+16*2**20)+5000000*48+64*2**20+2**30


def campaign_budget():return POLICY['benchmark_bytes']+sum(arm_budget(p) for p in COUNTS.values())
