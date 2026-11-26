SCHEMA = 'masked_execution_speed_v1'
VARIANTS = [dict(name='baseline32', microbatch=32, optimized=False)] + [
    dict(name=f'fast{n}', microbatch=n, optimized=True) for n in (32, 64, 128, 256)]
POLICY = dict(warmup=20, measured=300, windows=3, clean_steps=100, fixed_steps=10,
              memory_fraction=.90, skip_fraction=.05, max_seconds=18000,
              reserve_bytes=4*2**30, output_budget=2*2**30, margin=1.30,
              parity_batch=512, parity_particles=97, authorizes_training=False)
# Byte-for-byte reference code, including the model/data/optimizer dependencies.
REFERENCE_PREFIXES = ('part_ev_masked_predictive/', 'part_ev_fullscale/',
    'part_ev_state_compare/', 'part_ev_state_speed/', 'part_ev_message_study/',
    'part_ev_screen/', 'part_layerwise_screen/', 'part_ev_gpu_scaling/')
REFERENCE_FILES = ('scripts/run_part_ev_screen.py', 'part_ev_workspace/diagnostics.py')
