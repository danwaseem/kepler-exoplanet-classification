#!/usr/bin/env python3
"""Person 3 wrapper for the shared Stage 1 entry point."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "scripts" / "run_stage1.py"), run_name="__main__")
