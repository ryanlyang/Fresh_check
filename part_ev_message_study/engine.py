"""Private adapter instance of the unchanged screen training/evaluation engine.

No globals in the imported original driver or its protocol are mutated.
All replacements are in this separately loaded module namespace.
"""
import importlib.util
from pathlib import Path
from . import model, protocol


def create_engine():
    path = Path(__file__).resolve().parents[1] / "scripts/run_part_ev_screen.py"
    spec = importlib.util.spec_from_file_location("_ev_message_private_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.P = protocol
    engine.build_model = model.build_model
    engine.diagnostic_mode = model.diagnostic_mode
    engine.diagnostics = model.diagnostics
    return engine
