#!/usr/bin/env python3
"""Run Stage 5 feature importance from the repository root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet.stage5 import main

if __name__ == "__main__":
    main()
