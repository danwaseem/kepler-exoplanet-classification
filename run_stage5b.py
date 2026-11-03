#!/usr/bin/env python3
"""Run the Stage 5B centroid sensitivity check from the project root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from exoplanet.stage5b import main

if __name__ == "__main__":
    main()
