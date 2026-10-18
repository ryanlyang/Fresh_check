"""Versioned initial architectures; no training-recipe knobs."""
CONFIG = dict(version="grouped_attention_v1", groups=8, width=128, heads=8,
    insertion_block=4, group_ffn=256, physical_hidden=64, rounds=3,
    temperature=1.0, locality=1.0, radius_floor=0.05, gamma=1e-3,
    epsilon=1e-6, norm_epsilon=1e-5, added_dropout=0.0,
    random_seed=731019, geometry_precision="fp32", assignment_floor=0.0,
    routing_metadata="canonical_post_trim_input_bits_v1")

ARCHITECTURES = ("fixed", "learned", "hybrid", "random")
TASKS = [dict(run_id=f"PART_GROUP_{name.upper()}_RANGER", architecture=name,
              seed=101, recipe="ranger_short_v1") for name in ARCHITECTURES]
