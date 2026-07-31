"""Exact command-surface checks for checkpoint-free Stage-N preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


PRELOCK_INPUT_ENTRYPOINT = "scripts/prepare_retb_final_test_inputs.py"
PRELOCK_INPUT_OPTION_ORDER = (
    "--campaign-root",
    "--configuration",
    "--output",
)


def validate_prelock_input_row_access(
    argv: Sequence[object],
    expected_outputs: Sequence[object],
) -> None:
    """Require the exact raw-input-only prelock worker interface.

    Path values are deliberately not inspected for words such as
    ``checkpoint``: the authenticated campaign root itself conventionally
    lives below a directory named ``checkpoints``.  Safety comes from the
    exact option surface and exact campaign-relative paths instead.
    """

    command = [str(value) for value in argv]
    if (
        len(command) != 8
        or command[1].replace("\\", "/") != PRELOCK_INPUT_ENTRYPOINT
        or tuple(command[2::2]) != PRELOCK_INPUT_OPTION_ORDER
    ):
        raise ValueError("prelock final-input command surface differs")
    root = Path(command[3]).resolve()
    configuration = Path(command[5]).resolve()
    output = Path(command[7]).resolve()
    expected_configuration = (
        root / "inputs" / "stage_n" / "prelock_input_configuration.json"
    ).resolve()
    expected_output = (
        root / "inputs" / "stage_n" / "prelock_final_inputs.json"
    ).resolve()
    if configuration != expected_configuration or output != expected_output:
        raise ValueError("prelock final-input paths differ")
    shared = root / "inputs" / "stage_n" / "shared"
    required_outputs = {
        expected_output,
        *{
            (shared / f"retb_{split}_shared_HLT_inputs{suffix}").resolve()
            for split in ("stack_val", "final_test")
            for suffix in (".json", ".pt")
        },
    }
    actual_outputs = {Path(str(value)).resolve() for value in expected_outputs}
    if actual_outputs != required_outputs:
        raise ValueError("prelock final-input output coverage differs")


__all__ = [
    "PRELOCK_INPUT_ENTRYPOINT",
    "PRELOCK_INPUT_OPTION_ORDER",
    "validate_prelock_input_row_access",
]
