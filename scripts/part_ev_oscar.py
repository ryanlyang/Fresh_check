#!/usr/bin/env python3
"""Portable ParT/State-EV export and independent Oscar test workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_oscar.launcher import main

if __name__ == '__main__':
    main()
