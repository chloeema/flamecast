#!/usr/bin/env python3
"""
Download NASA POWER weather data.

Usage:
    python scripts/download_weather.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.download import main

if __name__ == "__main__":
    main()

