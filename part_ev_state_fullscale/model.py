"""Reuse the tested models without changing historical implementations."""
from part_ev_fullscale.model import build_model as reference_model
from part_ev_state_speed.model import build_model as state_model, CONFIGS
from part_ev_message_study.model import diagnostic_mode, diagnostics

COUNTS = {"part": 2143486, "part_ev": 2209022, "state_late4": 2266366}
CONFIG = dict(pair_bias="shared", ordinary_ev_blocks=list(range(8)),
              state=CONFIGS["state_late4"], initialization="fresh_seeded_no_donor_weights")


def build_model(architecture, *, seed=101, trim=True, use_amp=False):
    if architecture == "state_late4":
        model = state_model(architecture, seed=seed, trim=trim, use_amp=use_amp)
    elif architecture in ("part", "part_ev"):
        model = reference_model(architecture, trim=trim, use_amp=use_amp)
    else:
        raise ValueError("Unknown full-schedule architecture")
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError("Parameter contract changed")
    return model
