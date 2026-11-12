"""Probe real import order in clean processes, not an already-loaded pytest VM."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def first_scientific_imports(statement):
    # NumPy is really imported; stop BEFORE loading Torch's native runtime.
    # This proves startup order on CPU-only hosts without pretending that our
    # Windows library set can validate Tigris's exact OpenMP evidence.
    code = '''
import importlib.abc
import json
import runpy
import sys

assert "numpy" not in sys.modules and "torch" not in sys.modules
observed = []
class BeforeTorch(Exception): pass
class Probe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("numpy", "torch"):
            observed.append(fullname)
            if fullname == "torch":
                raise BeforeTorch()
        return None
sys.meta_path.insert(0, Probe())
try:
    exec(STATEMENT)
except BeforeTorch:
    pass
print(json.dumps(observed))
'''.replace('STATEMENT', repr(statement))
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                            check=True, capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout.strip())


@pytest.mark.parametrize('statement', [
    'import scripts.run_part_selected_compact',
    'runpy.run_module("scripts.run_part_selected_compact", run_name="__main__")',
    'runpy.run_module("scripts.run_part_selected_compact", run_name="__mp_main__")',
])
def test_worker_and_spawn_use_approved_selected_import_order(statement):
    original = first_scientific_imports('import part_selected_fullscale.protocol')
    assert original == ['numpy', 'torch']
    assert first_scientific_imports(statement) == original


def test_submitter_remains_stdlib_only():
    # Submission is run by system Python, unlike the dedicated workers. It must
    # not start either scientific runtime just to freeze/submit/review a plan.
    assert first_scientific_imports('import scripts.submit_part_selected_compact') == []
