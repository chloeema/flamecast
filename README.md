# Chloe's Science Fair: Wildfire Prediction with Weather Foundation Models

Predicting wildfire severity using self-supervised learning on spatiotemporal weather data.

## Project Structure

```
chloe_science_fair/
├── README.md
├── requirements.txt
│
├── configs/                    # Configuration files (future)
│
├── data/
│   ├── raw/                    # Original data (not in git)
│   │   ├── NFDB_point_20240613.txt      # Canadian fire database
│   │   └── nasa_power_weather_hires/     # NASA POWER weather grid
│   └── processed/              # Processed/aligned data
│       └── aligned_data/
│
├── src/                        # Source code
│   ├── data/
│   │   ├── download.py         # NASA POWER data download
│   │   └── align.py            # Fire-weather alignment
│   ├── models/
│   │   ├── weather_foundation.py  # Foundation encoder (Stage 1)
│   │   └── fire_segmentation.py   # Fire prediction (Stage 2)
│   └── utils/
│       └── visualization.py    # Data visualization
│
├── scripts/                    # Entry point scripts
│   ├── download_weather.py
│   ├── align_data.py
│   ├── pretrain_foundation.py
│   └── visualize.py
│
├── outputs/                    # Model outputs (not in git)
│   ├── checkpoints/
│   └── results/
│
└── docs/
    └── implementation_plan.md  # Detailed technical plan
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download weather data
```bash
python scripts/download_weather.py
```

### 3. Align fire events with weather grid
```bash
python scripts/align_data.py
```

### 4. Pretrain the weather foundation encoder
```bash
python scripts/pretrain_foundation.py --epochs 100
```

### 5. Visualize the data
```bash
python scripts/visualize.py --year 2023
```

## Architecture

### Stage 1: Weather Foundation Encoder (NEPA-style)
- **Input**: 30-day weather clips (T×C×H×W)
- **Patch Embedding**: 6×6 Conv2d → 256-dim tokens
- **Position Encoding**: 3D RoPE for (t, x, y)
- **Backbone**: 6-layer causal Transformer
- **Loss**: Multi-horizon next-embedding prediction (Δ = 1, 3, 7 days)

### Stage 2: Fire Severity Prediction
- Fine-tune encoder on fire severity classification
- 5 classes: C0_tiny, C1_small, C2_medium, C3_large, C4_extreme

## Data Sources

- **Weather**: NASA POWER (0.5° resolution, 1981-2024)
- **Fires**: Canadian National Fire Database (NFDB)

## Severity Classes

| Class | Label | Size (hectares) |
|-------|-------|-----------------|
| C0 | tiny | 0 – 1 |
| C1 | small | 1 – 10 |
| C2 | medium | 10 – 100 |
| C3 | large | 100 – 1,000 |
| C4 | extreme | 1,000+ |

