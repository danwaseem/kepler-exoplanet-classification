#!/usr/bin/env python3
"""Run Stage 1 shared dataset audit from the repository root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet.stage1 import main

if __name__ == "__main__":
    main()
