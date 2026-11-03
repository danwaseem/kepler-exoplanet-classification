#!/usr/bin/env python3
"""Run the Stage 5B centroid sensitivity check from the repository root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exoplanet.stage5b import main

if __name__ == "__main__":
    main()
