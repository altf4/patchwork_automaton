#!/usr/bin/env python3
"""Run the standalone Hartford supervisor from a staged source directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from patchwork_automaton.rl.hartford_guard import main

if __name__ == '__main__':
    raise SystemExit(main())
