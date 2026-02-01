# Revised Implementation Plan: Weather-NEPA Foundation Encoder (Stage 1) → Fire Severity per Tile (Stage 2)

This plan builds a **foundation-style encoder for gridded weather measurements** by pretraining with a **Next-Embedding Prediction** objective (NEPA-style). Then it fine-tunes the pretrained encoder to predict **wildfire severity classes per tile** using CNFDB point fire events aligned to NASA POWER tiles.

---

## 0) High-Level Goal

### Stage 1 — Weather Foundation Encoder
Learn a representation that captures the **underlying dynamics of weather systems** (spatiotemporal evolution) from long sequences of weather maps.

### Stage 2 — Fire Severity Downstream Task
Use the pretrained weather encoder to predict **fire severity class** (binned `SIZE_HA`) for a tile given pre-ignition weather history (and optional context).

---

## 1) Data Sources

### 1.1 NASA POWER (Primary Weather Data for v1)
Daily gridded weather variables over a target region.

Recommended channels (v1, daily):
- Temperature: `T2M`, `T2M_MAX`, `T2M_MIN`
- Humidity: `RH2M` (optionally `QV2M`)
- Wind: `WS2M` (or `WS10M`)
- Precip: `PRECTOTCORR`
- Soil moisture: `GWETPROF`
- Radiation: `ALLSKY_SFC_SW_DWN`
- Clouds: `CLOUD_AMT`
- Optional seasonal: `FRSNO`

**Normalization**
- Per-channel z-score over training years/region
- (Optional later) month-wise normalization to remove seasonal mean shifts

### 1.2 CNFDB Point Fires (Downstream Labels)
Use ignition date + ignition coordinates to align with POWER.

Fields required:
- ignition: `LATITUDE`, `LONGITUDE`, `YEAR/MONTH/DAY` or `DATE`
- size: `SIZE_HA`
Optional fields:
- `CAUSE` (H/L/N/U), `PROVINCE`, `AGENCY`

---

## 2) Spatial Tokenization / Tile Strategy

### 2.1 Base grid key
Use POWER’s native grid for v1:
- `tile_id := (LAT, LON)` from POWER output rows
- Maintain stable indexing:
  - `tile_row`, `tile_col` (derived from sorted unique LAT/LON)

### 2.2 Patch tokens (NEPA-style)
Token = **spatial patch per day** (not raw cell).

Start with **2×2 POWER-cell patches**:
- physical scale is already large (≈100 km)
- reduces token count 4× compared to 1×1
- preserves mesoscale gradients

Patch grid:
- `Hp = H / 2`, `Wp = W / 2`

---

## 3) Weather Encoder Input Representation (Spatiotemporal “Weather Video”)

Daily frame:
- `X_t ∈ R[C × H × W]`

Clip:
- `X ∈ R[T × C × H × W]`
Typical pretraining clip length:
- `T = 30` (start), later 60–90 for stronger dynamics

---

## 4) Patch Embedding (Conv2d) — Revised Recommendation

### 4.1 Why Conv2d patch embedding
Aligns with NEPA: a light, trainable projection to embeddings, learned end-to-end.

### 4.2 Embedding layer
Apply per day:
- `Conv2d(C → d_model, kernel=(2,2), stride=(2,2))`

Outputs per day:
- `E_t ∈ R[d_model × Hp × Wp]`

Flatten to tokens:
- `Z_t ∈ R[(Hp*Wp) × d_model]`

Stack time:
- `Z ∈ R[(T*Hp*Wp) × d_model]`

### 4.3 Post-embed normalization (weather-specific)
After flatten:
- `LayerNorm(d_model)` to stabilize across variables/units

---

## 5) Positional Encoding (Spatiotemporal Coordinates)

Each token has coordinates `(t, x, y)`:
- `t ∈ [0..T-1]`
- `x ∈ [0..Wp-1]`
- `y ∈ [0..Hp-1]`

Preferred:
- **3D RoPE** (t, x, y) in attention (recommended)
Fallback:
- learned embeddings for each axis:
  - `pos_t[t] + pos_x[x] + pos_y[y]`

Allocations (suggested):
- emphasize time in RoPE dims (e.g., 50% temporal, 25% x, 25% y)

---

## 6) Stage 1 Pretraining: Weather-NEPA Next-Embedding Prediction

### 6.1 Model architecture
Use a Transformer backbone with causal masking:
- input tokens: `Z`
- output hidden states: `H`
A prediction head `P` maps from hidden state to predicted embedding:
- `\hat{z} = P(h)`, output dim = `d_model`

Recommended scalability:
- **factorized attention** (optional early):
  - temporal attention + local spatial attention
- start with standard attention for smaller regions, then optimize

### 6.2 Key adaptation for weather: make “next” primarily temporal
Pure raster-next training overly teaches spatial interpolation.
Instead, define targets explicitly and weight them.

#### Target A (primary): next time, same spatial patch
For patch index `i` at time `t`:
- input context includes tokens up to `(t, i)` causally
- predict embedding at `(t+1, i)`:
  - `\hat{z}_{t+1,i} = P(h_{t,i})`

Loss:
- cosine similarity (normalized dot) or L2 on normalized vectors:
  - `L_time = mean(1 - cos(\hat{z}_{t+1,i}, sg(z_{t+1,i})))`

#### Target B (secondary): spatial neighbor within same time slice
Optionally predict `(t, i+1)` for some i:
- `L_space = mean(1 - cos(\hat{z}_{t,i+1}, sg(z_{t,i+1})))`

#### Target C (recommended): multi-horizon temporal prediction
Predict future embeddings at horizons:
- `Δ ∈ {1, 3, 7}` days (or token offsets corresponding to days)
- `L_multi = Σ_Δ w_Δ mean(1 - cos(\hat{z}_{t+Δ,i}, sg(z_{t+Δ,i})))`

#### Total loss
Choose weights to emphasize dynamics:
- `λ_time` high, `λ_multi` high, `λ_space` low

Example:
- `λ_time=1.0`, `λ_multi=1.0`, `λ_space=0.1`

Total:
- `L = λ_time*L_time + λ_multi*L_multi + λ_space*L_space`

### 6.3 Causal masking
Causal attention in flattened sequence order:
- time-major ordering (day-by-day raster scan)
- ensures representation learns to accumulate context

### 6.4 Data sampling for pretraining
Sample random clips:
- choose random `(t_start, region crop)` windows
- clip length T=30 (start)
- include seasonal stratification for regime diversity

Augmentations (light):
- random channel dropout (small probability)
- random spatial crop (if region large)
- optional masked blocks (later)

### 6.5 Outputs / checkpoints
Persist:
- patch embedder (Conv2d + LN)
- transformer backbone
- prediction head (optional to keep, usually discard in downstream)

---

## 7) Stage 2 Fine-tuning: Fire Severity per Tile

### 7.1 Define severity classes from CNFDB `SIZE_HA`
Example bins (hectares):
- C0: `< 1`
- C1: `[1, 10)`
- C2: `[10, 100)`
- C3: `[100, 1000)`
- C4: `>= 1000`

Tune thresholds based on class balance and decision needs.

### 7.2 Align CNFDB events to POWER tiles
For each fire event:
1) map `(lat, lon)` to nearest POWER cell center `(LAT, LON)` → tile index
2) ignition date `t0` maps to a POWER day index

### 7.3 Downstream input construction
For each labeled event:
- time window: last `W=14` days before ignition:
  - `[t0-W+1, t0]`
- spatial context: local crop around tile:
  - `K×K` patches centered on ignition patch (start K=5)

Build clip tensor:
- `X_clip ∈ R[W × C × (K*2) × (K*2)]` in POWER-cell units
Or directly in patch coordinates:
- `X_clip_patches ∈ R[W × K × K × patch(C)]`

Tokenize with the **same Conv2d patch embedder** used in pretraining.

### 7.4 Fine-tuning attention mode
Pretraining used causal attention.
Fine-tuning options:
- **Bidirectional attention** (recommended for classification)
- Keep positional encoding consistent

### 7.5 Readout / pooling strategy
Choose one:
- **Query token** appended at end (recommended):
  - output representation = last query token embedding
- **Last token** (NEPA-style):
  - choose ordering so the last token corresponds to ignition patch at `t0`
- **Attention pooling** (robust alternative)

### 7.6 Loss and imbalance handling
Use weighted cross-entropy:
- weights inversely proportional to class frequency (smoothed)
Track:
- macro-F1
- tail recall (C3+C4)
- confusion matrix
- ordinal distance |pred-true|

### 7.7 Evaluation splits (avoid leakage)
- Time split: train early years, test later years
- Optional spatial holdout: hold out provinces/regions

---

## 8) Recommended Baselines and Ablations

### Baselines
1) Supervised-only Transformer on weather clips (no pretraining)
2) Simple tabular model on engineered rolling stats (XGBoost)

### Ablations
- patch size: 1×1 vs 2×2 vs 4×4
- loss weights: temporal-heavy vs balanced
- horizons: Δ={1} vs {1,3,7}
- attention: causal fine-tune vs bidirectional fine-tune
- pooling: query token vs last token vs avg pool

---

## 9) Engineering Notes

### Data storage
- Store POWER in partitioned Parquet by date (fast slicing)
- Cache pretraining clips / downstream clips if I/O is bottleneck

### Compute considerations
- Token count grows with region size and T
- Prefer crops for pretraining if full-region is too large
- Factorized attention becomes important at continental scale

---

## 10) Milestones

### Milestone 1 — Data + Alignment
- [ ] POWER downloader + Parquet writer
- [ ] CNFDB loader + severity binning
- [ ] Fire-to-tile mapping validation

### Milestone 2 — Stage 1 Pretraining
- [ ] Conv2d patch embed + LN + (t,x,y) position
- [ ] Causal Transformer + NEPA multi-horizon temporal loss
- [ ] Pretraining runs + representation sanity checks

### Milestone 3 — Stage 2 Fine-tuning
- [ ] Event clip builder (W×K×K)
- [ ] Fine-tune encoder + classifier head
- [ ] Evaluate tail classes and temporal holdout

### Milestone 4 — Scale + Generalization
- [ ] Expand region / longer clips
- [ ] Add optional static geo / human features (two-tower)
- [ ] Consider upgrading weather source (ERA5/MERRA-2)

---
