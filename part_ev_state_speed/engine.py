"""Private engine/benchmark modules; no mutation of imported historical modules."""
import importlib.util
from pathlib import Path
from part_ev_message_study.engine import create_engine as base_engine
from . import model, protocol


def create_engine():
    engine = base_engine()
    engine.P = protocol
    engine.build_model = model.build_model
    engine.diagnostic_mode = model.diagnostic_mode
    engine.diagnostics = model.diagnostics
    return engine


def benchmark_driver(engine, parity):
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_ev_message_study.py"
    spec = importlib.util.spec_from_file_location("_state_speed_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.P, module.D = protocol, engine
    module.build_model = model.build_model
    module.initial_parity = parity
    return module
