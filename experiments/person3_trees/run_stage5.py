#!/usr/bin/env python3
"""Wrapper: Person 3 Stage 5 entry point."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "run_stage5.py"), run_name="__main__")
