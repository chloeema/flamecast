#!/usr/bin/env python3
"""
Align fire events with weather grid.

Usage:
    python scripts/align_data.py --nfdb data/raw/NFDB_point_20240613.txt \
        --weather data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv \
        --output data/processed/aligned_data
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.align import main, AlignmentConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align fire and weather data")
    parser.add_argument("--nfdb", default="data/raw/NFDB_point_20240613.txt", 
                       help="Path to NFDB fire data")
    parser.add_argument("--weather", 
                       default="data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv", 
                       help="Path to NASA POWER weather data")
    parser.add_argument("--output", default="data/processed/aligned_data", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    config = AlignmentConfig(
        nfdb_csv=args.nfdb,
        power_csv=args.weather,
        output_dir=args.output
    )
    
    main(config)

