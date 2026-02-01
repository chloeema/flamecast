#!/usr/bin/env python3
"""
Visualize fire-weather data.

Usage:
    python scripts/visualize.py --year 2023
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import main

if __name__ == "__main__":
    main()

