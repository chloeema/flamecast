#!/usr/bin/env python3
"""
Pretrain the weather foundation encoder using NEPA-style next-embedding prediction.

Usage:
    python scripts/pretrain_foundation.py --epochs 100 --batch-size 4
    
    # With custom paths
    python scripts/pretrain_foundation.py \
        --weather data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv \
        --output outputs/checkpoints/foundation
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.weather_foundation import pretrain, PretrainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Foundation Pretraining")
    
    parser.add_argument("--weather", 
                       default="data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv",
                       help="Path to weather CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--clip-length", type=int, default=30, help="Clip length in days")
    parser.add_argument("--clips-per-epoch", type=int, default=5000, help="Clips per epoch")
    parser.add_argument("--output", default="outputs/checkpoints/foundation", help="Output directory")
    
    args = parser.parse_args()
    
    config = PretrainConfig(
        weather_csv=args.weather,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        clip_length=args.clip_length,
        clips_per_epoch=args.clips_per_epoch,
        output_dir=args.output
    )
    
    pretrain(config)

