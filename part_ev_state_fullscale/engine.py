"""Private engine namespace: inherited loops are budget-driven, not ten-check loops."""
import importlib.util
from pathlib import Path
from . import model, protocol


def create_engine():
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_ev_screen.py"
    spec = importlib.util.spec_from_file_location("_state_fullscale_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.P = protocol
    engine.build_model = model.build_model
    engine.diagnostic_mode = model.diagnostic_mode
    engine.diagnostics = model.diagnostics
    return engine
