# -*- coding: utf-8 -*-
"""
Fire Occurrence Segmentation using V-JEPA2

This script uses V-JEPA2 as a spatiotemporal backbone with a segmentation head to predict
fire occurrence per grid cell.

Task: Given X days of weather data, predict which grid cells will have fires in the next Y days.

Input:  (T, C, H, W) = (14 days, 3 channels, grid_h, grid_w) weather video
Output: (H, W) binary mask or (num_classes, H, W) severity mask
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Data paths (use hires 0.5° resolution data)
    weather_csv: str = "data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv"
    aligned_fires_parquet: str = "data/processed/aligned_data/aligned_fires.parquet"
    grid_info_json: str = "data/processed/aligned_data/grid_info.json"
    
    # Temporal parameters
    lookback_days: int = 14      # Days of weather history as input
    prediction_window: int = 7   # Predict fires in next N days
    
    # Weather channels (3 for RGB-like input to V-JEPA2)
    weather_channels: List[str] = field(default_factory=lambda: ["T2M_MAX", "RH2M", "PRECTOTCORR"])
    
    # Segmentation classes
    # 0 = no fire, 1 = fire (binary) OR multi-class with severity
    num_classes: int = 2  # Binary: fire / no-fire
    use_severity_classes: bool = False  # If True, num_classes = 6 (no-fire + 5 severity)
    
    # Model parameters
    vjepa_model: str = "facebook/vjepa2-vitl-fpc64-256"
    target_size: int = 224  # V-JEPA2 input resolution
    freeze_backbone: bool = True
    
    # Training parameters
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 30
    early_stopping_patience: int = 7
    
    # Loss weighting (fires are rare)
    pos_weight: float = 50.0  # Weight for fire class (handles imbalance)
    
    # Data sampling
    samples_per_year: int = 52  # ~1 sample per week per year
    train_years: Tuple[int, int] = (1984, 2018)
    val_years: Tuple[int, int] = (2019, 2020)
    test_years: Tuple[int, int] = (2021, 2024)
    
    # Output
    output_dir: str = "outputs/checkpoints/fire_segmentation"


# ============================================================================
# WEATHER DATA LOADER (Full Grid)
# ============================================================================

class WeatherGridLoader:
    """Loads full weather grid for segmentation task."""
    
    def __init__(self, csv_path: str, channels: List[str]):
        print(f"Loading weather data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.channels = [c for c in channels if c in self.df.columns]
        
        if len(self.channels) < len(channels):
            missing = set(channels) - set(self.channels)
            print(f"  Warning: Missing channels: {missing}")
        
        # Build spatial index
        self.unique_lats = sorted(self.df['LAT'].unique())
        self.unique_lons = sorted(self.df['LON'].unique())
        self.lat_to_idx = {lat: i for i, lat in enumerate(self.unique_lats)}
        self.lon_to_idx = {lon: i for i, lon in enumerate(self.unique_lons)}
        
        self.grid_h = len(self.unique_lats)
        self.grid_w = len(self.unique_lons)
        
        print(f"  Grid size: {self.grid_w} x {self.grid_h} (W x H)")
        print(f"  Channels: {self.channels}")
        
        # Build date column
        if 'YEAR' in self.df.columns and 'DOY' in self.df.columns:
            self.df['DATE'] = pd.to_datetime(
                self.df['YEAR'].astype(str) + self.df['DOY'].astype(str).str.zfill(3),
                format='%Y%j'
            )
        
        self.date_range = (self.df['DATE'].min(), self.df['DATE'].max())
        self.available_dates = sorted(self.df['DATE'].unique())
        print(f"  Date range: {self.date_range[0].date()} to {self.date_range[1].date()}")
        
        # Compute normalization stats
        print("  Computing normalization statistics...")
        self.channel_stats = {}
        for ch in self.channels:
            valid = self.df[ch].replace(-999, np.nan).dropna()
            self.channel_stats[ch] = {
                'mean': valid.mean(),
                'std': valid.std() + 1e-8
            }
        
        # Create indexed lookup
        print("  Building index...")
        self.df_indexed = self.df.set_index(['DATE', 'LAT', 'LON']).sort_index()
        print("  Done!")
    
    def get_weather_frame(self, date: pd.Timestamp) -> Optional[np.ndarray]:
        """Get full grid for a single day. Returns (C, H, W)."""
        try:
            day_data = self.df_indexed.loc[date]
        except KeyError:
            return None
        
        frame = np.full((len(self.channels), self.grid_h, self.grid_w), np.nan, dtype=np.float32)
        
        for idx_tuple, row in day_data.iterrows():
            if isinstance(idx_tuple, tuple):
                lat, lon = idx_tuple
            else:
                lat = idx_tuple
                lon = row.name if hasattr(row, 'name') else row.get('LON', None)
            
            lat_idx = self.lat_to_idx.get(lat, -1)
            lon_idx = self.lon_to_idx.get(lon, -1)
            
            if lat_idx >= 0 and lon_idx >= 0:
                for c_idx, ch in enumerate(self.channels):
                    val = row[ch] if ch in row.index else np.nan
                    if val == -999:
                        val = np.nan
                    # Normalize
                    if not np.isnan(val):
                        val = (val - self.channel_stats[ch]['mean']) / self.channel_stats[ch]['std']
                    frame[c_idx, lat_idx, lon_idx] = val
        
        return frame
    
    def get_weather_sequence(self, end_date: pd.Timestamp, num_days: int) -> Optional[np.ndarray]:
        """Get weather video sequence. Returns (T, C, H, W)."""
        start_date = end_date - pd.Timedelta(days=num_days - 1)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        sequence = np.full((num_days, len(self.channels), self.grid_h, self.grid_w), 
                          np.nan, dtype=np.float32)
        
        for t, date in enumerate(dates):
            frame = self.get_weather_frame(date)
            if frame is not None:
                sequence[t] = frame
        
        # Fill NaN with 0 (normalized mean)
        sequence = np.nan_to_num(sequence, nan=0.0)
        return sequence


# ============================================================================
# FIRE MASK GENERATOR
# ============================================================================

class FireMaskGenerator:
    """Generates fire occurrence masks from aligned fire data."""
    
    def __init__(self, parquet_path: str, grid_h: int, grid_w: int,
                 unique_lats: List[float], unique_lons: List[float],
                 use_severity: bool = False):
        print(f"Loading fire data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.df['fire_date'] = pd.to_datetime(self.df['fire_date'])
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_severity = use_severity
        
        # Map grid coordinates to indices
        self.lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
        self.lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
        
        # Create date index for fast lookup
        self.df_by_date = self.df.set_index('fire_date').sort_index()
        
        print(f"  Loaded {len(self.df):,} fire events")
        print(f"  Date range: {self.df['fire_date'].min().date()} to {self.df['fire_date'].max().date()}")
    
    def get_fire_mask(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> np.ndarray:
        """
        Get fire mask for a date range.
        
        Returns:
            Binary mask (H, W) with 1 where fires occurred, 0 elsewhere
            OR severity mask (H, W) with 0=no fire, 1-5=severity class
        """
        if self.use_severity:
            mask = np.zeros((self.grid_h, self.grid_w), dtype=np.int64)
        else:
            mask = np.zeros((self.grid_h, self.grid_w), dtype=np.int64)
        
        try:
            fires_in_range = self.df_by_date.loc[start_date:end_date]
        except KeyError:
            return mask
        
        for _, fire in fires_in_range.iterrows():
            lat_idx = self.lat_to_idx.get(fire['grid_lat'], -1)
            lon_idx = self.lon_to_idx.get(fire['grid_lon'], -1)
            
            if lat_idx >= 0 and lon_idx >= 0:
                if self.use_severity:
                    # Use max severity if multiple fires in same cell
                    severity = int(fire['severity_class']) + 1  # 1-5 (0 = no fire)
                    mask[lat_idx, lon_idx] = max(mask[lat_idx, lon_idx], severity)
                else:
                    mask[lat_idx, lon_idx] = 1
        
        return mask


# ============================================================================
# SEGMENTATION DATASET
# ============================================================================

class FireSegmentationDataset(Dataset):
    """
    Dataset for fire segmentation.
    
    Each sample:
        - Input: (T, C, H, W) weather sequence for lookback_days
        - Target: (H, W) fire mask for prediction_window days after input
    """
    
    def __init__(self, 
                 weather: WeatherGridLoader,
                 fires: FireMaskGenerator,
                 sample_dates: List[pd.Timestamp],
                 lookback_days: int = 14,
                 prediction_window: int = 7,
                 target_size: int = 224):
        
        self.weather = weather
        self.fires = fires
        self.sample_dates = sample_dates
        self.lookback_days = lookback_days
        self.prediction_window = prediction_window
        self.target_size = target_size
        
        print(f"FireSegmentationDataset: {len(sample_dates)} samples")
    
    def __len__(self):
        return len(self.sample_dates)
    
    def __getitem__(self, idx):
        # This is the last day of the input weather sequence
        input_end_date = self.sample_dates[idx]
        
        # Get weather sequence (lookback_days ending at input_end_date)
        weather_seq = self.weather.get_weather_sequence(input_end_date, self.lookback_days)
        
        # Get fire mask (fires in the prediction_window AFTER input)
        pred_start = input_end_date + pd.Timedelta(days=1)
        pred_end = input_end_date + pd.Timedelta(days=self.prediction_window)
        fire_mask = self.fires.get_fire_mask(pred_start, pred_end)
        
        # Convert to tensors
        weather_tensor = torch.from_numpy(weather_seq).float()  # (T, C, H, W)
        mask_tensor = torch.from_numpy(fire_mask).long()  # (H, W) at grid size
        
        # Resize weather to 224x224 for V-JEPA2 input
        T, C, H, W = weather_tensor.shape
        weather_flat = weather_tensor.reshape(T * C, H, W)
        weather_resized = F.interpolate(
            weather_flat.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        weather_tensor = weather_resized.reshape(T, C, self.target_size, self.target_size)
        
        # Keep mask at original grid size (18x30) - model outputs at this size
        # No resizing needed!
        
        # Normalize to [0, 1] for V-JEPA2 (expects video-like input)
        weather_tensor = torch.sigmoid(weather_tensor)
        
        # Convert to V-JEPA2 format: (T, H, W, C) uint8
        frames_numpy = weather_tensor.permute(0, 2, 3, 1).numpy()
        frames_numpy = (frames_numpy * 255).astype(np.uint8)
        
        return {
            'frames': frames_numpy,
            'mask': mask_tensor,  # (H, W) at original grid size
            'date': str(input_end_date.date()),
            'grid_h': H,
            'grid_w': W
        }


# ============================================================================
# V-JEPA2 SEGMENTATION MODEL
# ============================================================================

class VJEPA2Segmentation(nn.Module):
    """
    V-JEPA2 backbone with lightweight segmentation head.
    
    V-JEPA2 outputs 16x16 patch embeddings (for 224x224 input).
    Our target grid is small (18x30), so we use a simple head:
    - 2-layer MLP to reduce hidden_dim
    - Bilinear resize to output grid size
    
    No fancy FPN needed for such a small output!
    """
    
    def __init__(self, vjepa_model_name: str, num_classes: int = 2, 
                 freeze_backbone: bool = True, target_size: int = 224,
                 output_h: int = 39, output_w: int = 55):
        super().__init__()
        
        from transformers import AutoModel, AutoConfig
        
        self.target_size = target_size
        self.num_classes = num_classes
        self.output_h = output_h
        self.output_w = output_w
        
        # Load V-JEPA2 backbone
        print(f"Loading V-JEPA2 backbone: {vjepa_model_name}")
        self.backbone = AutoModel.from_pretrained(vjepa_model_name)
        config = AutoConfig.from_pretrained(vjepa_model_name)
        
        # Get hidden dimension from config
        self.hidden_dim = getattr(config, 'hidden_size', 1024)
        self.patch_size = getattr(config, 'patch_size', 14)
        
        # Calculate spatial dimensions after patching
        # For 224x224 with patch_size=14: 16x16 patches
        self.patch_h = target_size // self.patch_size
        self.patch_w = target_size // self.patch_size
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  V-JEPA2 spatial output: {self.patch_h}x{self.patch_w}")
        print(f"  Target grid: {output_h}x{output_w}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("  Freezing backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Simple segmentation head
        # V-JEPA2 16x16 features -> 18x30 grid predictions
        # That's roughly the same resolution, so just need projection + resize
        self.head = nn.Sequential(
            # Project hidden_dim down
            nn.Conv2d(self.hidden_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        
        print(f"  Head: {self.hidden_dim} -> 256 -> 64 -> {num_classes}")
    
    def forward(self, pixel_values):
        """
        Forward pass.
        
        Args:
            pixel_values: (B, T, C, H, W) video tensor
            
        Returns:
            logits: (B, num_classes, output_h, output_w) segmentation logits
        """
        # Get V-JEPA2 features
        outputs = self.backbone(pixel_values=pixel_values)
        
        # V-JEPA2 outputs last_hidden_state: (B, num_patches, hidden_dim)
        # We need to reshape to spatial format
        features = outputs.last_hidden_state  # (B, num_patches, hidden_dim)
        
        B = features.shape[0]
        
        # Remove CLS token if present (usually first token)
        expected_patches = self.patch_h * self.patch_w
        if features.shape[1] > expected_patches:
            # Likely has CLS token(s), take last expected_patches
            features = features[:, -expected_patches:, :]
        
        # Reshape to spatial: (B, H_p, W_p, hidden_dim) -> (B, hidden_dim, H_p, W_p)
        features = features.reshape(B, self.patch_h, self.patch_w, self.hidden_dim)
        features = features.permute(0, 3, 1, 2)  # (B, hidden_dim, 16, 16)
        
        # Simple head: project channels
        logits = self.head(features)  # (B, num_classes, 16, 16)
        
        # Resize to target grid (16x16 -> 18x30)
        logits = F.interpolate(logits, size=(self.output_h, self.output_w), 
                               mode='bilinear', align_corners=False)
        
        return logits


# ============================================================================
# TRAINING
# ============================================================================

def generate_sample_dates(weather: WeatherGridLoader, 
                         year_range: Tuple[int, int],
                         lookback_days: int,
                         prediction_window: int,
                         samples_per_year: int) -> List[pd.Timestamp]:
    """Generate evenly spaced sample dates within year range."""
    
    start_year, end_year = year_range
    dates = []
    
    for year in range(start_year, end_year + 1):
        # Sample dates throughout the year (skip winter for fire season focus)
        # Fire season roughly April-October in Canada
        year_start = pd.Timestamp(f'{year}-04-01')
        year_end = pd.Timestamp(f'{year}-10-31')
        
        # Adjust for lookback
        year_start = year_start + pd.Timedelta(days=lookback_days)
        year_end = year_end - pd.Timedelta(days=prediction_window)
        
        if year_start >= year_end:
            continue
        
        # Ensure within weather data range
        if year_start < weather.date_range[0] + pd.Timedelta(days=lookback_days):
            year_start = weather.date_range[0] + pd.Timedelta(days=lookback_days)
        if year_end > weather.date_range[1] - pd.Timedelta(days=prediction_window):
            year_end = weather.date_range[1] - pd.Timedelta(days=prediction_window)
        
        if year_start >= year_end:
            continue
        
        # Generate evenly spaced dates
        date_range = (year_end - year_start).days
        if date_range > 0:
            step = max(1, date_range // samples_per_year)
            current = year_start
            while current <= year_end:
                dates.append(current)
                current += pd.Timedelta(days=step)
    
    return dates


def train_fire_segmentation(config: Config):
    """Train V-JEPA2 fire segmentation model."""
    
    print("\n" + "=" * 70)
    print("FIRE SEGMENTATION WITH V-JEPA2")
    print("=" * 70)
    
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\n[1/4] Loading data...")
    weather = WeatherGridLoader(config.weather_csv, config.weather_channels)
    
    fires = FireMaskGenerator(
        config.aligned_fires_parquet,
        grid_h=weather.grid_h,
        grid_w=weather.grid_w,
        unique_lats=weather.unique_lats,
        unique_lons=weather.unique_lons,
        use_severity=config.use_severity_classes
    )
    
    # Generate sample dates for train/val/test
    print("\n[2/4] Creating datasets...")
    
    train_dates = generate_sample_dates(
        weather, config.train_years, 
        config.lookback_days, config.prediction_window,
        config.samples_per_year
    )
    val_dates = generate_sample_dates(
        weather, config.val_years,
        config.lookback_days, config.prediction_window,
        config.samples_per_year
    )
    test_dates = generate_sample_dates(
        weather, config.test_years,
        config.lookback_days, config.prediction_window,
        config.samples_per_year
    )
    
    print(f"  Train samples: {len(train_dates)} ({config.train_years[0]}-{config.train_years[1]})")
    print(f"  Val samples: {len(val_dates)} ({config.val_years[0]}-{config.val_years[1]})")
    print(f"  Test samples: {len(test_dates)} ({config.test_years[0]}-{config.test_years[1]})")
    
    # Create datasets
    train_dataset = FireSegmentationDataset(
        weather, fires, train_dates,
        config.lookback_days, config.prediction_window, config.target_size
    )
    val_dataset = FireSegmentationDataset(
        weather, fires, val_dates,
        config.lookback_days, config.prediction_window, config.target_size
    )
    test_dataset = FireSegmentationDataset(
        weather, fires, test_dates,
        config.lookback_days, config.prediction_window, config.target_size
    )
    
    # Load V-JEPA2 processor
    from transformers import AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained(config.vjepa_model)
    
    def collate_fn(batch):
        frames_list = [item['frames'] for item in batch]
        masks = torch.stack([item['mask'] for item in batch])
        
        inputs = processor(frames_list, return_tensors="pt")
        
        # Find pixel values key
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        elif 'pixel_values_videos' in inputs:
            pixel_values = inputs['pixel_values_videos']
        else:
            pixel_values = list(inputs.values())[0]
        
        return {'pixel_values': pixel_values, 'masks': masks}
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Create model
    print("\n[3/4] Creating model...")
    num_classes = 6 if config.use_severity_classes else config.num_classes
    
    model = VJEPA2Segmentation(
        config.vjepa_model,
        num_classes=num_classes,
        freeze_backbone=config.freeze_backbone,
        target_size=config.target_size,
        output_h=weather.grid_h,  # 18
        output_w=weather.grid_w   # 30
    )
    model = model.to(device)
    
    # Loss and optimizer
    # Use weighted BCE for binary, weighted CE for multi-class
    if num_classes == 2:
        # Binary: use BCE with pos_weight
        pos_weight = torch.tensor([config.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # Multi-class: weight fire classes more
        weights = torch.ones(num_classes).to(device)
        weights[1:] = config.pos_weight  # Weight fire classes
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print(f"\n[4/4] Training for {config.num_epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_tp, train_fp, train_fn = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['masks'].to(device)
            
            optimizer.zero_grad()
            logits = model(pixel_values)
            
            if num_classes == 2:
                # Binary: logits shape (B, 2, H, W), take fire channel
                loss = criterion(logits[:, 1], masks.float())
                preds = (torch.sigmoid(logits[:, 1]) > 0.5).long()
            else:
                loss = criterion(logits, masks)
                preds = logits.argmax(dim=1)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # IoU computation (fire class)
            pred_fire = (preds > 0)
            true_fire = (masks > 0)
            train_tp += (pred_fire & true_fire).sum().item()
            train_fp += (pred_fire & ~true_fire).sum().item()
            train_fn += (~pred_fire & true_fire).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_iou = train_tp / (train_tp + train_fp + train_fn + 1e-8)
        
        # Validation
        model.eval()
        val_loss = 0
        val_tp, val_fp, val_fn = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                pixel_values = batch['pixel_values'].to(device)
                masks = batch['masks'].to(device)
                
                logits = model(pixel_values)
                
                if num_classes == 2:
                    loss = criterion(logits[:, 1], masks.float())
                    preds = (torch.sigmoid(logits[:, 1]) > 0.5).long()
                else:
                    loss = criterion(logits, masks)
                    preds = logits.argmax(dim=1)
                
                val_loss += loss.item()
                
                pred_fire = (preds > 0)
                true_fire = (masks > 0)
                val_tp += (pred_fire & true_fire).sum().item()
                val_fp += (pred_fire & ~true_fire).sum().item()
                val_fn += (~pred_fire & true_fire).sum().item()
        
        val_loss /= len(val_loader)
        val_iou = val_tp / (val_tp + val_fp + val_fn + 1e-8)
        
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    
    model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
    model.eval()
    
    test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['masks'].to(device)
            
            logits = model(pixel_values)
            
            if num_classes == 2:
                preds = (torch.sigmoid(logits[:, 1]) > 0.5).long()
            else:
                preds = logits.argmax(dim=1)
            
            pred_fire = (preds > 0)
            true_fire = (masks > 0)
            
            test_tp += (pred_fire & true_fire).sum().item()
            test_fp += (pred_fire & ~true_fire).sum().item()
            test_fn += (~pred_fire & true_fire).sum().item()
            test_tn += (~pred_fire & ~true_fire).sum().item()
    
    # Metrics
    precision = test_tp / (test_tp + test_fp + 1e-8)
    recall = test_tp / (test_tp + test_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = test_tp / (test_tp + test_fp + test_fn + 1e-8)
    
    print(f"\nTest Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  IoU:       {iou:.4f}")
    print(f"  TP: {test_tp:,}, FP: {test_fp:,}, FN: {test_fn:,}, TN: {test_tn:,}")
    
    # Save results
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'history': history
    }
    
    with open(os.path.join(config.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(config.output_dir, "training_history.csv"), index=False)
    
    print(f"\nResults saved to {config.output_dir}/")
    
    return model, results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Check files exist
    if not os.path.exists(config.weather_csv):
        print(f"ERROR: Weather data not found at {config.weather_csv}")
        print("Please run download_nasa_power.py first")
        exit(1)
    
    if not os.path.exists(config.aligned_fires_parquet):
        print(f"ERROR: Aligned fire data not found at {config.aligned_fires_parquet}")
        print("Please run align_fire_weather.py first")
        exit(1)
    
    model, results = train_fire_segmentation(config)

