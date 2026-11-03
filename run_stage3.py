#!/usr/bin/env python3
"""Run Stage 3 Decision Tree tuning from the project root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from exoplanet.stage3 import main

if __name__ == "__main__":
    main()
