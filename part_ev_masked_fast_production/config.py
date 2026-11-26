from copy import deepcopy
from part_ev_masked_predictive import config as O

SCHEMA = 'masked_fast256_production_v1'
RESOURCES = dict(O.PLATFORMS['tigris'], microbatch=256, auxiliary_slots=1)
SCIENCE = dict(deepcopy(O.SCIENCE), auxiliary_microbatch=256)
TASK = dict(architecture='masked', run_id='MASKED_PREDICTIVE_FAST256_STATE_EV_LAYERWISE',
            seed=101, recipe='ranger_full_v1')
POLICY = dict(O.POLICY, parity_repeats=2, parity_batch=512, parity_particles=97,
              budget_bytes=O.POLICY['benchmark_bytes']+O.arm_budget('masked'))
VARIANT = dict(name='fast256', optimized=True, microbatch=256)


def execution_hash():
    return O.digest(dict(schema=SCHEMA,science=SCIENCE,resources=RESOURCES,task=TASK,
                         recipe=O.recipe(),policy=POLICY,parity=O.PARITY))
