#!/usr/bin/env python3
"""
Weather Foundation Encoder - Stage 1 Pretraining (NEPA-style)

Next-Embedding Prediction for weather spatiotemporal data.
Learns rich representations of weather dynamics by predicting future embeddings.

Architecture:
    - Conv2d patch embedding (2×2 patches)
    - 3D RoPE positional encoding (t, x, y)
    - Causal Transformer backbone
    - Multi-horizon prediction head (Δ = 1, 3, 7 days)

Usage:
    python weather_foundation_pretrain.py --epochs 100 --batch-size 16

References:
    - JEPA/V-JEPA: Joint Embedding Predictive Architectures
    - RoPE: Rotary Position Embedding (Su et al., 2021)
"""

import os
import math
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PretrainConfig:
    """Configuration for weather foundation pretraining."""
    
    # Data
    weather_csv: str = "data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv"
    
    # Weather channels to use
    channels: List[str] = field(default_factory=lambda: [
        "T2M_MAX", "T2M_MIN", "RH2M", "QV2M", "WS2M",
        "PRECTOTCORR", "GWETPROF", "FRSNO", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT"
    ])
    
    # Temporal
    clip_length: int = 30  # Days per clip (T)
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])  # Multi-horizon
    
    # Spatial patching
    # With 60×150 grid and 6×6 patches: 10×25 = 250 patches/frame
    # With T=30 days: 30 × 250 = 7,500 tokens (manageable)
    patch_size: int = 6  # 6×6 grid cells per patch (~330×200 km at 55°N)
    
    # Model architecture
    d_model: int = 256  # Embedding dimension
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024  # Feed-forward hidden dim
    dropout: float = 0.1
    
    # RoPE settings
    rope_theta: float = 10000.0
    rope_time_fraction: float = 0.5  # Fraction of dims for temporal encoding
    
    # Training
    # With 7,500 tokens/sample, batch_size=4 uses ~30K tokens/batch
    # Gradient accumulation compensates for smaller batch size
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Loss weights
    lambda_time: float = 1.0  # Next-step temporal loss
    lambda_multi: float = 1.0  # Multi-horizon loss
    lambda_space: float = 0.1  # Spatial neighbor loss (optional)
    
    # Data sampling
    clips_per_epoch: int = 10000
    train_years: Tuple[int, int] = (1984, 2018)
    val_years: Tuple[int, int] = (2019, 2020)
    
    # Checkpointing
    output_dir: str = "outputs/checkpoints/foundation"
    save_every_epochs: int = 5
    
    # Memory optimization
    # Effective batch = batch_size × gradient_accumulation = 4 × 8 = 32
    gradient_accumulation_steps: int = 8
    use_amp: bool = True  # Automatic mixed precision


# ============================================================================
# DATA LOADING (Memory-Efficient)
# ============================================================================

class WeatherGridIndex:
    """
    Memory-efficient weather grid indexer.
    Only loads coordinate metadata, not all data.
    """
    
    def __init__(self, csv_path: str, channels: List[str]):
        print(f"Building weather grid index from {csv_path}...")
        
        self.csv_path = csv_path
        self.channels = channels
        
        # Extract grid metadata (memory-efficient)
        self._build_index()
        
    def _build_index(self):
        """Build spatial and temporal index without loading all data."""
        
        # Read first chunk to get structure
        chunk = pd.read_csv(self.csv_path, nrows=100000)
        
        # Get available channels
        self.available_channels = [c for c in self.channels if c in chunk.columns]
        if len(self.available_channels) < len(self.channels):
            missing = set(self.channels) - set(self.available_channels)
            print(f"  Warning: Missing channels: {missing}")
        
        # Get unique coordinates
        location_cols = ['LAT', 'LON', 'YEAR', 'DOY']
        
        print("  Scanning for unique coordinates...")
        unique_lats = set()
        unique_lons = set()
        years = set()
        
        chunk_size = 100000
        reader = pd.read_csv(self.csv_path, usecols=location_cols, chunksize=chunk_size)
        
        stable_count = 0
        for chunk in tqdm(reader, desc="  Indexing", unit=" chunks"):
            before_lats = len(unique_lats)
            before_lons = len(unique_lons)
            
            unique_lats.update(chunk['LAT'].unique())
            unique_lons.update(chunk['LON'].unique())
            years.update(chunk['YEAR'].unique())
            
            # Early exit if stable
            if len(unique_lats) == before_lats and len(unique_lons) == before_lons:
                stable_count += 1
                if stable_count >= 5:
                    break
            else:
                stable_count = 0
        
        self.unique_lats = sorted(unique_lats)
        self.unique_lons = sorted(unique_lons)
        self.years = sorted(years)
        
        self.lat_to_idx = {lat: i for i, lat in enumerate(self.unique_lats)}
        self.lon_to_idx = {lon: i for i, lon in enumerate(self.unique_lons)}
        
        self.grid_h = len(self.unique_lats)
        self.grid_w = len(self.unique_lons)
        self.n_channels = len(self.available_channels)
        
        print(f"  Grid: {self.grid_w} × {self.grid_h} (W × H)")
        print(f"  Years: {min(self.years)} - {max(self.years)}")
        print(f"  Channels: {self.n_channels}")
        
    def get_date_range(self, year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get date range for a year."""
        return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


class WeatherClipSampler(IterableDataset):
    """
    Memory-efficient sampler that loads clips on-demand.
    Uses chunked reading to handle large CSV files.
    """
    
    def __init__(self, 
                 csv_path: str,
                 grid_index: WeatherGridIndex,
                 clip_length: int = 30,
                 year_range: Tuple[int, int] = (1984, 2018),
                 clips_per_epoch: int = 10000,
                 normalize: bool = True):
        
        self.csv_path = csv_path
        self.grid = grid_index
        self.clip_length = clip_length
        self.year_range = year_range
        self.clips_per_epoch = clips_per_epoch
        self.normalize = normalize
        
        # Channel stats for normalization (computed lazily)
        self._channel_stats = None
        
        # Pre-compute valid date ranges
        self.valid_years = [y for y in self.grid.years 
                          if year_range[0] <= y <= year_range[1]]
        
        print(f"WeatherClipSampler: {len(self.valid_years)} years, {clips_per_epoch} clips/epoch")
        
    def _compute_channel_stats(self):
        """Compute channel mean/std from a sample of data."""
        if self._channel_stats is not None:
            return
        
        print("  Computing channel statistics...")
        
        # Sample ~1M rows for stats
        sample_chunks = []
        reader = pd.read_csv(self.csv_path, 
                            usecols=self.grid.available_channels,
                            chunksize=100000)
        
        for i, chunk in enumerate(reader):
            sample_chunks.append(chunk)
            if i >= 10:  # 1M rows
                break
        
        sample = pd.concat(sample_chunks)
        
        self._channel_stats = {}
        for ch in self.grid.available_channels:
            values = sample[ch].replace(-999, np.nan).dropna()
            self._channel_stats[ch] = {
                'mean': float(values.mean()),
                'std': float(values.std()) + 1e-6
            }
        
        print(f"  Channel stats computed from {len(sample):,} rows")
    
    def _load_clip(self, year: int, doy_start: int) -> Optional[np.ndarray]:
        """
        Load a single weather clip.
        
        Returns:
            clip: (T, C, H, W) array or None if not available
        """
        # Read data for this date range
        doy_end = doy_start + self.clip_length
        
        # Build filter for this date range
        cols_to_read = ['LAT', 'LON', 'YEAR', 'DOY'] + self.grid.available_channels
        
        clip_data = []
        reader = pd.read_csv(self.csv_path, usecols=cols_to_read, chunksize=500000)
        
        for chunk in reader:
            mask = (chunk['YEAR'] == year) & (chunk['DOY'] >= doy_start) & (chunk['DOY'] < doy_end)
            if mask.any():
                clip_data.append(chunk[mask])
        
        if not clip_data:
            return None
        
        df = pd.concat(clip_data)
        
        # Build (T, C, H, W) tensor
        clip = np.full((self.clip_length, self.grid.n_channels, 
                       self.grid.grid_h, self.grid.grid_w), 
                      np.nan, dtype=np.float32)
        
        for _, row in df.iterrows():
            t = int(row['DOY'] - doy_start)
            lat_idx = self.grid.lat_to_idx.get(row['LAT'], -1)
            lon_idx = self.grid.lon_to_idx.get(row['LON'], -1)
            
            if 0 <= t < self.clip_length and lat_idx >= 0 and lon_idx >= 0:
                for c, ch in enumerate(self.grid.available_channels):
                    val = row[ch]
                    if val != -999:
                        clip[t, c, lat_idx, lon_idx] = val
        
        # Normalize
        if self.normalize and self._channel_stats:
            for c, ch in enumerate(self.grid.available_channels):
                stats = self._channel_stats[ch]
                clip[:, c] = (clip[:, c] - stats['mean']) / stats['std']
        
        # Fill NaN with 0 (after normalization)
        clip = np.nan_to_num(clip, nan=0.0)
        
        return clip
    
    def __iter__(self):
        """Generate random clips."""
        
        # Ensure channel stats are computed
        self._compute_channel_stats()
        
        for _ in range(self.clips_per_epoch):
            # Random year and start day
            year = np.random.choice(self.valid_years)
            max_doy = 365 - self.clip_length - 7  # Leave room for prediction
            doy_start = np.random.randint(1, max_doy + 1)
            
            clip = self._load_clip(year, doy_start)
            
            if clip is not None:
                yield torch.from_numpy(clip)


class PreloadedWeatherDataset(Dataset):
    """
    Preloaded dataset for faster training.
    Loads a subset of data into memory.
    """
    
    def __init__(self, 
                 csv_path: str,
                 grid_index: WeatherGridIndex,
                 clip_length: int = 30,
                 year_range: Tuple[int, int] = (1984, 2018),
                 max_clips: int = 5000,
                 cache_file: Optional[str] = None):
        
        self.clip_length = clip_length
        self.grid = grid_index
        
        # Check for cache
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached clips from {cache_file}...")
            data = np.load(cache_file)
            self.clips = torch.from_numpy(data['clips'])
            self._channel_stats = dict(np.load(cache_file, allow_pickle=True)['stats'].item())
            print(f"  Loaded {len(self.clips)} clips from cache")
            return
        
        print(f"Preloading weather clips (this may take a while)...")
        
        # Use sampler to get clips
        sampler = WeatherClipSampler(
            csv_path, grid_index, clip_length, year_range, max_clips
        )
        
        clips = []
        for clip in tqdm(sampler, total=max_clips, desc="  Loading clips"):
            clips.append(clip)
            if len(clips) >= max_clips:
                break
        
        self.clips = torch.stack(clips)
        self._channel_stats = sampler._channel_stats
        
        print(f"  Loaded {len(self.clips)} clips")
        
        # Save cache
        if cache_file:
            print(f"  Saving cache to {cache_file}...")
            np.savez_compressed(cache_file, 
                              clips=self.clips.numpy(),
                              stats=self._channel_stats)
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        return self.clips[idx]


# ============================================================================
# 3D ROTARY POSITION EMBEDDING (RoPE)
# ============================================================================

class RotaryEmbedding3D(nn.Module):
    """
    3D Rotary Position Embedding for (t, x, y) coordinates.
    
    Allocates dimensions:
        - time_fraction for temporal dimension
        - (1 - time_fraction) / 2 for each spatial dimension
    """
    
    def __init__(self, 
                 dim: int, 
                 max_t: int = 100,
                 max_x: int = 100,
                 max_y: int = 100,
                 theta: float = 10000.0,
                 time_fraction: float = 0.5):
        super().__init__()
        
        self.dim = dim
        self.max_t = max_t
        self.max_x = max_x
        self.max_y = max_y
        
        # Allocate dimensions
        dim_t = int(dim * time_fraction)
        dim_spatial = dim - dim_t
        dim_x = dim_spatial // 2
        dim_y = dim_spatial - dim_x
        
        self.dim_t = dim_t
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        # Precompute frequencies
        self.register_buffer('freqs_t', self._compute_freqs(dim_t, max_t, theta))
        self.register_buffer('freqs_x', self._compute_freqs(dim_x, max_x, theta))
        self.register_buffer('freqs_y', self._compute_freqs(dim_y, max_y, theta))
        
    def _compute_freqs(self, dim: int, max_pos: int, theta: float) -> torch.Tensor:
        """Compute frequency bands for RoPE."""
        if dim == 0:
            return torch.zeros(1, 1)
        
        # Frequency bands
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # Position indices
        pos = torch.arange(max_pos).float()
        
        # Outer product: (max_pos, dim/2)
        freqs = torch.einsum('p,f->pf', pos, freqs)
        
        # Stack sin and cos: (max_pos, dim)
        freqs = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
        freqs = freqs.reshape(max_pos, dim)
        
        return freqs
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Get rotary embeddings for given coordinates.
        
        Args:
            t: (seq_len,) temporal positions
            x: (seq_len,) x spatial positions
            y: (seq_len,) y spatial positions
            
        Returns:
            rope: (seq_len, dim) rotary embeddings
        """
        # Clamp to valid range
        t = t.clamp(0, self.max_t - 1).long()
        x = x.clamp(0, self.max_x - 1).long()
        y = y.clamp(0, self.max_y - 1).long()
        
        # Look up frequencies
        rope_t = self.freqs_t[t]  # (seq_len, dim_t)
        rope_x = self.freqs_x[x]  # (seq_len, dim_x)
        rope_y = self.freqs_y[y]  # (seq_len, dim_y)
        
        # Concatenate
        rope = torch.cat([rope_t, rope_x, rope_y], dim=-1)
        
        return rope


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        q: (batch, heads, seq, head_dim)
        k: (batch, heads, seq, head_dim)
        rope: (seq, head_dim)
        
    Returns:
        q_rot, k_rot: Rotated query and key tensors
    """
    # Reshape rope for broadcasting
    rope = rope.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    
    # Split into cos and sin components
    cos = rope[..., ::2].repeat_interleave(2, dim=-1)
    sin = rope[..., 1::2].repeat_interleave(2, dim=-1)
    
    # Rotate
    def rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot


# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert weather grid to patch embeddings using Conv2d.
    
    Input: (B, T, C, H, W) - weather video
    Output: (B, T*Hp*Wp, d_model) - patch tokens
    """
    
    def __init__(self, 
                 n_channels: int,
                 patch_size: int = 2,
                 d_model: int = 256):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Conv2d for patch embedding (applied per frame)
        self.proj = nn.Conv2d(
            n_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: (B, T, C, H, W) weather video
            
        Returns:
            tokens: (B, T*Hp*Wp, d_model) patch embeddings
            shape: (T, Hp, Wp) shape info for reconstruction
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames at once
        x = x.reshape(B * T, C, H, W)
        
        # Apply patch embedding
        x = self.proj(x)  # (B*T, d_model, Hp, Wp)
        
        _, _, Hp, Wp = x.shape
        
        # Reshape to tokens
        x = x.reshape(B, T, self.d_model, Hp, Wp)
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, Hp, Wp, d_model)
        x = x.reshape(B, T * Hp * Wp, self.d_model)
        
        # Normalize
        x = self.norm(x)
        
        return x, (T, Hp, Wp)


class CausalMultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking and RoPE support."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                rope: Optional[torch.Tensor] = None,
                causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, seq, d_model)
            rope: (seq, head_dim) rotary embeddings
            causal: Whether to use causal masking
            
        Returns:
            out: (B, seq, d_model)
        """
        B, seq, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if rope is not None:
            q, k = apply_rotary_emb(q, k, rope)
        
        # Use PyTorch 2.0+ scaled_dot_product_attention (flash attention when available)
        # This is much more memory efficient for long sequences
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=causal
        )
        out = out.transpose(1, 2).reshape(B, seq, self.d_model)
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.norm1(x), rope)
        
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class WeatherTransformer(nn.Module):
    """
    Weather Foundation Transformer.
    
    Processes spatiotemporal weather data with causal attention and 3D RoPE.
    """
    
    def __init__(self, config: PretrainConfig, grid_h: int, grid_w: int, n_channels: int):
        super().__init__()
        
        self.config = config
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Patch dimensions
        self.patch_h = grid_h // config.patch_size
        self.patch_w = grid_w // config.patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_channels=n_channels,
            patch_size=config.patch_size,
            d_model=config.d_model
        )
        
        # 3D RoPE
        self.rope = RotaryEmbedding3D(
            dim=config.d_model // config.n_heads,
            max_t=config.clip_length,
            max_x=self.patch_w,
            max_y=self.patch_h,
            theta=config.rope_theta,
            time_fraction=config.rope_time_fraction
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
    def _get_coordinates(self, T: int, Hp: int, Wp: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate (t, x, y) coordinates for all tokens."""
        # Time-major ordering: iterate t, then y, then x
        t = torch.arange(T, device=device).repeat_interleave(Hp * Wp)
        
        y = torch.arange(Hp, device=device).repeat(T * Wp)
        y = y.reshape(T, Wp, Hp).permute(0, 2, 1).flatten()
        
        x = torch.arange(Wp, device=device).repeat(T * Hp)
        
        return t, x, y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) weather video
            
        Returns:
            h: (B, seq, d_model) hidden states
        """
        B = x.shape[0]
        
        # Patch embedding
        tokens, (T, Hp, Wp) = self.patch_embed(x)
        
        # Get coordinates and RoPE
        t, px, py = self._get_coordinates(T, Hp, Wp, x.device)
        rope = self.rope(t, px, py)
        
        # Transformer blocks
        h = tokens
        for block in self.blocks:
            h = block(h, rope)
        
        h = self.final_norm(h)
        
        return h


# ============================================================================
# NEPA PREDICTION HEAD AND LOSS
# ============================================================================

class NEPAPredictionHead(nn.Module):
    """
    Prediction head for next-embedding prediction.
    
    Predicts future embeddings from current hidden states.
    """
    
    def __init__(self, d_model: int, horizons: List[int] = [1, 3, 7]):
        super().__init__()
        
        self.horizons = horizons
        
        # Separate predictor for each horizon
        self.predictors = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            for h in horizons
        })
        
    def forward(self, h: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Predict future embedding.
        
        Args:
            h: (B, seq, d_model) hidden states
            horizon: Prediction horizon (days ahead)
            
        Returns:
            pred: (B, seq, d_model) predicted embeddings
        """
        return self.predictors[str(horizon)](h)


def compute_nepa_loss(
    h: torch.Tensor,
    target_embeddings: torch.Tensor,
    predictor: NEPAPredictionHead,
    T: int, Hp: int, Wp: int,
    config: PretrainConfig
) -> Dict[str, torch.Tensor]:
    """
    Compute NEPA loss with multi-horizon predictions.
    
    Args:
        h: (B, T*Hp*Wp, d_model) hidden states from encoder
        target_embeddings: (B, T*Hp*Wp, d_model) target embeddings (stop-gradient)
        predictor: Prediction head
        T, Hp, Wp: Shape info
        config: Training config
        
    Returns:
        losses: Dictionary of loss components
    """
    B, seq_len, d_model = h.shape
    device = h.device
    
    # Reshape to (B, T, Hp*Wp, d_model)
    h = h.reshape(B, T, Hp * Wp, d_model)
    targets = target_embeddings.reshape(B, T, Hp * Wp, d_model)
    
    losses = {}
    
    # Multi-horizon temporal prediction
    for horizon in config.prediction_horizons:
        if horizon >= T:
            continue
        
        # Source: tokens at time t
        # Target: tokens at time t+horizon (same spatial location)
        src = h[:, :-horizon]  # (B, T-horizon, Hp*Wp, d_model)
        tgt = targets[:, horizon:]  # (B, T-horizon, Hp*Wp, d_model)
        
        # Flatten for prediction
        src_flat = src.reshape(B, -1, d_model)
        tgt_flat = tgt.reshape(B, -1, d_model)
        
        # Predict
        pred = predictor(src_flat, horizon)
        
        # Cosine similarity loss
        pred_norm = F.normalize(pred, dim=-1)
        tgt_norm = F.normalize(tgt_flat, dim=-1)
        
        cosine_sim = (pred_norm * tgt_norm).sum(dim=-1)
        loss = 1 - cosine_sim.mean()
        
        losses[f'loss_h{horizon}'] = loss
    
    # Weighted total loss
    total_loss = torch.tensor(0.0, device=device)
    
    if 'loss_h1' in losses:
        total_loss = total_loss + config.lambda_time * losses['loss_h1']
    
    for h in config.prediction_horizons[1:]:  # Multi-horizon
        key = f'loss_h{h}'
        if key in losses:
            total_loss = total_loss + config.lambda_multi * losses[key] / len(config.prediction_horizons[1:])
    
    losses['total'] = total_loss
    
    return losses


# ============================================================================
# FULL MODEL
# ============================================================================

class WeatherFoundationModel(nn.Module):
    """
    Complete weather foundation model for pretraining.
    """
    
    def __init__(self, config: PretrainConfig, grid_h: int, grid_w: int, n_channels: int):
        super().__init__()
        
        self.config = config
        
        # Encoder (will be saved for downstream)
        self.encoder = WeatherTransformer(config, grid_h, grid_w, n_channels)
        
        # Prediction head (discarded after pretraining)
        self.predictor = NEPAPredictionHead(config.d_model, config.prediction_horizons)
        
        # Store shape info
        self.patch_h = self.encoder.patch_h
        self.patch_w = self.encoder.patch_w
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Forward pass for pretraining.
        
        Args:
            x: (B, T, C, H, W) weather video
            
        Returns:
            h: (B, seq, d_model) hidden states
            shape: (T, Hp, Wp) shape info
        """
        B, T, C, H, W = x.shape
        h = self.encoder(x)
        return h, (T, self.patch_h, self.patch_w)
    
    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute pretraining loss.
        
        Args:
            x: (B, T, C, H, W) weather video
            
        Returns:
            losses: Dictionary of loss components
        """
        # Get encoder outputs
        h, (T, Hp, Wp) = self.forward(x)
        
        # Target embeddings are the patch embeddings (stop gradient)
        with torch.no_grad():
            target_tokens, _ = self.encoder.patch_embed(x)
        
        # Compute NEPA loss
        losses = compute_nepa_loss(
            h, target_tokens.detach(),
            self.predictor,
            T, Hp, Wp,
            self.config
        )
        
        return losses


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: WeatherFoundationModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: PretrainConfig,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    
    total_loss = 0
    horizon_losses = {f'h{h}': 0 for h in config.prediction_horizons}
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            losses = model.compute_loss(batch)
            loss = losses['total'] / config.gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        # Track losses
        total_loss += losses['total'].item()
        for h in config.prediction_horizons:
            key = f'loss_h{h}'
            if key in losses:
                horizon_losses[f'h{h}'] += losses[key].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss/n_batches:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return {
        'loss': total_loss / n_batches,
        **{k: v / n_batches for k, v in horizon_losses.items()}
    }


def validate(
    model: WeatherFoundationModel,
    dataloader: DataLoader,
    config: PretrainConfig,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    
    model.eval()
    
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                losses = model.compute_loss(batch)
            
            total_loss += losses['total'].item()
            n_batches += 1
    
    return {'loss': total_loss / n_batches}


def pretrain(config: PretrainConfig):
    """Main pretraining loop."""
    
    print("\n" + "=" * 70)
    print("WEATHER FOUNDATION ENCODER - NEPA PRETRAINING")
    print("=" * 70)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)
    print(f"Config saved to {config_path}")
    
    # Build grid index
    print("\n[1/5] Building grid index...")
    grid_index = WeatherGridIndex(config.weather_csv, config.channels)
    
    # Create datasets
    print("\n[2/5] Creating datasets...")
    
    # Use cached dataset for faster training
    train_cache = os.path.join(config.output_dir, "train_clips.npz")
    val_cache = os.path.join(config.output_dir, "val_clips.npz")
    
    train_dataset = PreloadedWeatherDataset(
        config.weather_csv, grid_index,
        clip_length=config.clip_length,
        year_range=config.train_years,
        max_clips=config.clips_per_epoch,
        cache_file=train_cache
    )
    
    val_dataset = PreloadedWeatherDataset(
        config.weather_csv, grid_index,
        clip_length=config.clip_length,
        year_range=config.val_years,
        max_clips=config.clips_per_epoch // 5,
        cache_file=val_cache
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    print("\n[3/5] Creating model...")
    model = WeatherFoundationModel(
        config,
        grid_h=grid_index.grid_h,
        grid_w=grid_index.grid_w,
        n_channels=grid_index.n_channels
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == 'cuda' else None
    
    # Training loop
    print("\n[4/5] Training...")
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, device, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, config, device)
        
        # Log
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            **{f'train_{k}': v for k, v in train_metrics.items() if k != 'loss'}
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.encoder.state_dict(),  # Save encoder only
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': asdict(config),
                'grid_info': {
                    'grid_h': grid_index.grid_h,
                    'grid_w': grid_index.grid_w,
                    'n_channels': grid_index.n_channels,
                    'channels': grid_index.available_channels
                }
            }, best_path)
            print(f"  New best model saved to {best_path}")
        
        # Periodic checkpoint
        if (epoch + 1) % config.save_every_epochs == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
    
    # Save history
    print("\n[5/5] Saving results...")
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Foundation Pretraining")
    
    parser.add_argument("--weather", default="nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv",
                       help="Path to weather CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--clip-length", type=int, default=30, help="Clip length in days")
    parser.add_argument("--clips-per-epoch", type=int, default=5000, help="Clips per epoch")
    parser.add_argument("--output", default="weather_foundation_checkpoints", help="Output directory")
    
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

