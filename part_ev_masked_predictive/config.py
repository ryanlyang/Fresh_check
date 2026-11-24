"""Frozen location-conditioned latent-prediction and platform contracts."""
from copy import deepcopy
import math
from part_ev_fullscale.protocol import RECIPE,digest

SCHEMA='part_ev_masked_predictive_v1'
STORAGE_POLICY='campaign_remaining_plus_reserve_aggregate_advisory_v1'
COUNTS=dict(control=2327462,masked=2327462)
TRAINABLE=dict(control=2327462,masked=2626598)
TEACHER=1926412
PREDICTOR=299136
SCIENCE=dict(version='masked_latent_state_layerwise_v1',seed=101,
    backbone='full_noncompact_state_layerwise_width32_query64',teacher_taps=[6,8],
    selection='half_local_without_replacement_half_uniform_half_patch',capacity=128,
    mask_count='n>=4:min(n-2,max(1,n//4))',canonical='p4_then_independent_channels_5_to_14',
    random='four_independent_per_rank_cpu_generators_fixed_capacity_v1',
    rebase='visible_vector_pt_energy_signed_pseudorapidity_yaml_transforms_eps1e-8',
    independent_channels=list(range(5,15)),rebuilt_channels=[0,1,2,3,4,15,16],
    padding='wrap_visible_only_mask_zero',teacher='particle_only_previous_step_ema',
    target='nonaffine_ln_average_ln_H6_ln_H8_eps1e-5',
    predictor=dict(width=128,blocks=2,heads=4,ffn=256,position=[3,128,128],norm_epsilon=1e-5,
        dropout=0.,biased=True,query_self_attention=False,initialization='isolated_xavier_mask_normal_0.02'),
    loss='smooth_l1_beta1_mean_channels_then_targets_then_global_eligible_jets',
    lambda_schedule='0_to20k_linear_to40k_0.1_to800k_linear_to900k_then0',
    momentum='cosine_0.996_to_0.9999_attempt_indexed_1M',
    auxiliary_bn='functional_immutable_preclean_B0_eval_arithmetic',auxiliary_dropout=0.,
    auxiliary_trim=False,auxiliary_microbatch=32,
    ddp=dict(broadcast_buffers=False,find_unused_parameters=False,static_graph=False,
             accumulation='clean_and_nonfinal_slots_no_sync_final_slot_sync_all_parameter_zero_anchors'),
    transaction='one_scaler_ranger_step_then_successful_parameter_ema_and_always_B1_copy',
    teacher_optimizer=False,export='unmodified_student_only')
TASKS=[dict(architecture=a,run_id=r,seed=101,recipe='ranger_full_v1') for a,r in (
    ('control','LW_STATE_EV_CONTROL'),('masked','MASKED_PREDICTIVE_STATE_EV_LAYERWISE'))]
PLATFORMS={
    'sporc':dict(machine='x86_64',partition='tier3',gpu='a100',gpus=2,local_batch=256,workers=1,
        threads=2,cpus=8,memory_gib=128,batchnorm='SyncBatchNorm',broadcast_buffers=False,
        microbatch=32,auxiliary_slots=4,python='/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python'),
    'tigris':dict(machine='aarch64',partition='tigris',gpu='gh200',gpus=1,local_batch=512,workers=2,
        threads=4,cpus=12,memory_gib=128,batchnorm='BatchNorm',broadcast_buffers=False,
        microbatch=32,auxiliary_slots=8,python='/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python')}
POLICY=dict(warmup=20,measured=4200,windows=[1000,1000,1000,1000,200],clean_warmup=20,clean_steps=200,
    fixed_warmup=10,fixed_steps=40,validation_events=128000,validation_batch=512,inference_batch=256,
    inference_warmup=10,inference_steps=20,latency_warmup=10,latency_steps=100,checkpoint_interval=1000,
    heartbeat=100,max_seconds=18000,shutdown_margin=900,max_skip_fraction=.05,memory_fraction=.90,
    margin=1.30,reserve_bytes=4*2**30,benchmark_bytes=2*2**30,
    active_attempts=879999,clean_attempts=120001,diagnostic_schedule='explicit_label_not_production_cursor_v1')
PARITY=dict(forward_rtol=1e-5,forward_atol=1e-6,gradient_rtol=1e-4,gradient_atol=1e-5,
    amp_forward_rtol=2e-3,amp_forward_atol=2e-3,amp_gradient_rtol=1e-2,amp_gradient_atol=1e-3,
    ddp_forward_rtol=5e-4,ddp_gradient_rtol=2e-3,ddp_atol=2e-5,restart_absolute=1e-4,
    restart_atol=1e-5,restart_rtol=1e-4,loss_absolute=2e-4,loss_atol=1e-4,loss_rtol=1e-5)


def coefficient(u):
    if not 1<=u<=1000000:raise ValueError('One-based production schedule outside full budget')
    if u<=20000 or u>=900000:return 0.
    if u<40000:return .1*(u-20000)/20000
    if u<=800000:return .1
    return .1*(900000-u)/100000


def momentum(u):
    if not 1<=u<=1000000:raise ValueError('Invalid EMA attempt label')
    return .9999-(.9999-.996)*(1+math.cos(math.pi*(u-1)/999999))/2


def recipe():
    r=deepcopy(RECIPE);r.update(evaluation_population=5000000,pair_bias='layerwise',
        workers_per_rank='resources.workers',ddp_batchnorm='resources.batchnorm',ddp_broadcast_buffers=False)
    return r


def science_hash():return digest(dict(science=SCIENCE,counts=COUNTS,trainable=TRAINABLE,
    teacher=TEACHER,predictor=PREDICTOR,recipe=recipe(),parity=PARITY))


def arm_budget(arm):
    p=COUNTS[arm];q=TRAINABLE[arm];r=TEACHER if arm=='masked' else 0
    return 51*(8*p+2**20)+3*(24*q+4*r+16*2**20)+5000000*48+64*2**20+2**30


def campaign_budget():return POLICY['benchmark_bytes']+sum(arm_budget(a) for a in COUNTS)
