#!/usr/bin/env python3
"""
Fire-Weather Data Alignment

Aligns NFDB fire events with NASA POWER weather grid.
Produces a unified dataset for downstream analysis and training.

Usage:
    python align_fire_weather.py

Output:
    - aligned_fire_weather.parquet: Fire events with matched grid coordinates
    - grid_fire_counts.parquet: Fire counts per grid cell
    - alignment_stats.json: Statistics about the alignment
"""

import os
import json
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def log_step(step_num: int, total_steps: int, message: str):
    """Print a formatted step message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] STEP {step_num}/{total_steps}: {message}")
    print("-" * 70)


def log_substep(message: str, indent: int = 2):
    """Print a substep message."""
    indent_str = " " * indent
    print(f"{indent_str}{message}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AlignmentConfig:
    # Input files
    nfdb_csv: str = "data/raw/NFDB_point_20240613.txt"
    power_csv: str = "data/raw/nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv"
    
    # Output directory
    output_dir: str = "data/processed/aligned_data"
    
    # Severity bins (hectares)
    severity_bins: List[float] = None
    severity_labels: List[str] = None
    
    # Spatial filtering (optional - set to None to use full grid)
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    
    # Year filtering
    year_min: int = 1981
    year_max: int = 2024
    
    def __post_init__(self):
        if self.severity_bins is None:
            self.severity_bins = [0, 1, 10, 100, 1000, float('inf')]
        if self.severity_labels is None:
            self.severity_labels = ["C0_tiny", "C1_small", "C2_medium", "C3_large", "C4_extreme"]


# ============================================================================
# WEATHER GRID LOADER
# ============================================================================

def load_weather_grid(csv_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load NASA POWER weather data and extract grid metadata.
    
    Returns:
        df: Weather dataframe (minimal, only for metadata)
        grid_info: Dictionary with grid metadata
    """
    start_time = time.time()
    log_substep(f"Loading weather grid from {csv_path}")
    
    # Load only the columns needed for grid metadata to save memory
    # Read in chunks to handle large files efficiently
    log_substep("Reading grid coordinates in chunks (memory-efficient)...")
    location_cols = ['LAT', 'LON', 'YEAR', 'DOY']
    
    # Read just location columns in chunks to extract unique coordinates
    unique_lats_set = set()
    unique_lons_set = set()
    date_min = None
    date_max = None
    weather_channels = None
    
    chunk_size = 100000  # Process 100k rows at a time
    max_chunks_for_coords = 100  # Only need to read enough chunks to get all unique coordinates
    stable_chunks_required = 5  # Require this many consecutive chunks with no new coords before stopping
    
    log_substep(f"Scanning file to extract grid coordinates (up to {max_chunks_for_coords} chunks)...")
    
    # Try C engine first (faster), fall back to Python engine if it fails
    try:
        reader = pd.read_csv(csv_path, usecols=location_cols, chunksize=chunk_size, engine='c')
    except Exception as e:
        log_substep(f"  C engine failed, using Python engine (slower but more robust)...")
        reader = pd.read_csv(csv_path, usecols=location_cols, chunksize=chunk_size, engine='python')
    
    total_rows_processed = 0
    chunks_read = 0
    consecutive_stable_chunks = 0  # Track how many chunks found no new coordinates
    pbar = tqdm(desc="  Extracting grid coordinates", unit=" chunks", total=max_chunks_for_coords)
    
    # We only need to read enough chunks to get all unique coordinates
    # Grid coordinates should be consistent across the file
    for chunk in reader:
        chunk_rows = len(chunk)
        total_rows_processed += chunk_rows
        chunks_read += 1
        pbar.update(1)
        
        unique_lats_before = len(unique_lats_set)
        unique_lons_before = len(unique_lons_set)
        
        unique_lats_set.update(chunk['LAT'].unique())
        unique_lons_set.update(chunk['LON'].unique())
        
        # Track date range
        if 'YEAR' in chunk.columns and 'DOY' in chunk.columns:
            chunk_dates = pd.to_datetime(
                chunk['YEAR'].astype(str) + chunk['DOY'].astype(str).str.zfill(3),
                format='%Y%j',
                errors='coerce'
            )
            chunk_date_min = chunk_dates.min()
            chunk_date_max = chunk_dates.max()
            if date_min is None or (pd.notna(chunk_date_min) and chunk_date_min < date_min):
                date_min = chunk_date_min
            if date_max is None or (pd.notna(chunk_date_max) and chunk_date_max > date_max):
                date_max = chunk_date_max
        
        # Early exit: if we've found all unique coordinates, we can stop
        # Check if we're still finding new coordinates
        if (len(unique_lats_set) == unique_lats_before and 
            len(unique_lons_set) == unique_lons_before):
            consecutive_stable_chunks += 1
            if consecutive_stable_chunks >= stable_chunks_required:
                log_substep(f"  All unique coordinates found after {chunks_read} chunks (stable for {stable_chunks_required}), stopping early")
                break
        else:
            consecutive_stable_chunks = 0  # Reset counter if we found new coords
            pbar.set_postfix(lats=len(unique_lats_set), lons=len(unique_lons_set))
        
        # Also stop if we've read enough chunks
        if chunks_read >= max_chunks_for_coords:
            log_substep(f"  Reached max chunks limit ({max_chunks_for_coords}), stopping")
            break
    
    pbar.close()
    elapsed = time.time() - start_time
    log_substep(f"Processed {chunks_read:,} chunks ({total_rows_processed:,} rows) in {elapsed:.1f}s")
    log_substep(f"Found {len(unique_lats_set):,} unique latitudes, {len(unique_lons_set):,} unique longitudes")
    
    # Get weather channels by reading just the header
    log_substep("Reading column names...")
    df_sample = pd.read_csv(csv_path, nrows=1)
    location_cols_all = ['LAT', 'LON', 'YEAR', 'DOY', 'DATE', 'MM', 'DD', 'YEAR_REQUESTED']
    weather_channels = [c for c in df_sample.columns if c not in location_cols_all]
    del df_sample  # Free memory
    
    # Extract unique coordinates
    log_substep("Extracting unique grid coordinates...")
    unique_lats = sorted(unique_lats_set)
    unique_lons = sorted(unique_lons_set)
    
    # Compute grid spacing
    lat_spacing = unique_lats[1] - unique_lats[0] if len(unique_lats) > 1 else 0.5
    lon_spacing = unique_lons[1] - unique_lons[0] if len(unique_lons) > 1 else 0.5
    
    grid_info = {
        'unique_lats': unique_lats,
        'unique_lons': unique_lons,
        'lat_spacing': lat_spacing,
        'lon_spacing': lon_spacing,
        'lat_min': min(unique_lats),
        'lat_max': max(unique_lats),
        'lon_min': min(unique_lons),
        'lon_max': max(unique_lons),
        'n_lats': len(unique_lats),
        'n_lons': len(unique_lons),
        'date_min': date_min,
        'date_max': date_max,
        'weather_channels': weather_channels,
    }
    
    log_substep(f"Grid size: {grid_info['n_lons']} x {grid_info['n_lats']} (lon x lat)")
    log_substep(f"Lat range: {grid_info['lat_min']} to {grid_info['lat_max']} (spacing: {lat_spacing}°)")
    log_substep(f"Lon range: {grid_info['lon_min']} to {grid_info['lon_max']} (spacing: {lon_spacing}°)")
    log_substep(f"Weather channels: {len(weather_channels)} channels found")
    
    elapsed = time.time() - start_time
    log_substep(f"✓ Weather grid loaded in {elapsed:.1f}s")
    
    # Return minimal dataframe (just for compatibility, not actually used)
    df = pd.DataFrame({'LAT': unique_lats[:1], 'LON': unique_lons[:1]})
    
    return df, grid_info


# ============================================================================
# FIRE DATA LOADER
# ============================================================================

def load_fire_data(csv_path: str, config: AlignmentConfig) -> pd.DataFrame:
    """
    Load and preprocess NFDB fire data.
    
    Returns:
        df: Fire events dataframe with standardized columns
    """
    start_time = time.time()
    log_substep(f"Loading fire data from {csv_path}...")
    
    # Try comma separator first (based on the file structure we saw)
    log_substep("Reading CSV file...")
    
    # Use chunking with tqdm for large files
    # First, try to get file size to estimate progress
    try:
        file_size = os.path.getsize(csv_path)
        file_size_mb = file_size / (1024 * 1024)
        log_substep(f"  File size: {file_size_mb:.1f} MB")
        
        # Use chunking with tqdm for files larger than 10MB
        if file_size_mb > 10:
            chunk_size = 50000
            chunks = []
            reader = pd.read_csv(csv_path, low_memory=False, chunksize=chunk_size)
            pbar = tqdm(desc="  Reading fire data", unit=" rows", unit_scale=True)
            
            for chunk in reader:
                chunks.append(chunk)
                pbar.update(len(chunk))
            
            pbar.close()
            df = pd.concat(chunks, ignore_index=True)
            log_substep(f"Loaded {len(df):,} fire records (chunked read)")
        else:
            # Small file, read directly
            df = pd.read_csv(csv_path, low_memory=False)
            log_substep(f"Loaded {len(df):,} fire records (direct read)")
    except Exception as e:
        # Fall back to regular read if chunking fails
        log_substep(f"  Note: Using direct read (chunking not available)")
        df = pd.read_csv(csv_path, low_memory=False)
        log_substep(f"Loaded {len(df):,} fire records (direct read)")
    log_substep(f"Columns: {len(df.columns)} total ({', '.join(list(df.columns)[:5])}...)")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Parse fire date
    if 'rep_date' in df.columns:
        df['fire_date'] = pd.to_datetime(df['rep_date'], errors='coerce')
    elif 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
        df['fire_date'] = pd.to_datetime(
            df[['year', 'month', 'day']].rename(columns={'year': 'year', 'month': 'month', 'day': 'day'}),
            errors='coerce'
        )
    
    # Parse out date (for duration calculation)
    if 'out_date' in df.columns:
        df['out_date'] = pd.to_datetime(df['out_date'], errors='coerce')
        df['duration_days'] = (df['out_date'] - df['fire_date']).dt.days
    
    # Ensure we have lat/lon
    if 'latitude' in df.columns:
        df['lat'] = df['latitude']
    if 'longitude' in df.columns:
        df['lon'] = df['longitude']
    
    # Bin severity
    if 'size_ha' in df.columns:
        df['severity_class'] = pd.cut(
            df['size_ha'],
            bins=config.severity_bins,
            labels=range(len(config.severity_bins) - 1),
            include_lowest=True
        ).astype(float).fillna(-1).astype(int)
        
        df['severity_label'] = df['severity_class'].apply(
            lambda x: config.severity_labels[x] if 0 <= x < len(config.severity_labels) else 'unknown'
        )
    
    # Filter by year
    if 'year' in df.columns:
        log_substep(f"Filtering by year range: {config.year_min}-{config.year_max}...")
        before = len(df)
        df = df[(df['year'] >= config.year_min) & (df['year'] <= config.year_max)]
        log_substep(f"  {before:,} -> {len(df):,} records after year filter")
    
    # Filter valid records
    log_substep("Filtering valid records (lat/lon/date)...")
    before = len(df)
    valid_mask = (
        df['lat'].notna() &
        df['lon'].notna() &
        df['fire_date'].notna()
    )
    df = df[valid_mask].copy()
    log_substep(f"  {before:,} -> {len(df):,} records after validity filter")
    
    elapsed = time.time() - start_time
    log_substep(f"✓ Fire data loaded and preprocessed in {elapsed:.1f}s")
    
    return df


# ============================================================================
# ALIGNMENT FUNCTIONS
# ============================================================================

def find_nearest_grid_point(lat: float, lon: float, 
                           unique_lats: List[float], unique_lons: List[float]) -> Tuple[float, float, int, int]:
    """Find the nearest grid point and its indices."""
    lat_idx = np.argmin(np.abs(np.array(unique_lats) - lat))
    lon_idx = np.argmin(np.abs(np.array(unique_lons) - lon))
    
    return unique_lats[lat_idx], unique_lons[lon_idx], lat_idx, lon_idx


def align_fires_to_grid(fire_df: pd.DataFrame, grid_info: dict) -> pd.DataFrame:
    """
    Align fire events to the nearest weather grid points.
    
    Adds columns:
        - grid_lat, grid_lon: Nearest grid point coordinates
        - grid_lat_idx, grid_lon_idx: Grid indices (for array indexing)
        - dist_to_grid_km: Distance from fire to grid point center
    """
    start_time = time.time()
    log_substep(f"Aligning {len(fire_df):,} fire events to weather grid...")
    
    unique_lats = grid_info['unique_lats']
    unique_lons = grid_info['unique_lons']
    
    # Vectorized nearest neighbor search (much faster than iterrows)
    lat_array = np.array(unique_lats)
    lon_array = np.array(unique_lons)
    
    # Process in batches to avoid memory issues with very large fire datasets
    # Memory usage: batch_size * (n_lats + n_lons) floats
    batch_size = 5000  # Process 5k fires at a time
    n_fires = len(fire_df)
    n_batches = (n_fires + batch_size - 1) // batch_size
    
    log_substep(f"Processing {n_fires:,} fires in {n_batches:,} batches (batch size: {batch_size:,})...")
    
    lat_idxs = np.empty(n_fires, dtype=np.int32)
    lon_idxs = np.empty(n_fires, dtype=np.int32)
    
    pbar = tqdm(total=n_fires, desc="  Mapping fires to grid", unit=" fires", unit_scale=True)
    
    for i in range(0, n_fires, batch_size):
        end_idx = min(i + batch_size, n_fires)
        batch = fire_df.iloc[i:end_idx]
        batch_size_actual = end_idx - i
        
        fire_lats = batch['lat'].values
        fire_lons = batch['lon'].values
        
        # Find nearest lat index for batch
        lat_diffs = np.abs(fire_lats[:, np.newaxis] - lat_array[np.newaxis, :])
        lat_idxs[i:end_idx] = np.argmin(lat_diffs, axis=1)
        
        # Find nearest lon index for batch
        lon_diffs = np.abs(fire_lons[:, np.newaxis] - lon_array[np.newaxis, :])
        lon_idxs[i:end_idx] = np.argmin(lon_diffs, axis=1)
        
        pbar.update(batch_size_actual)
    
    pbar.close()
    
    # Get grid coordinates
    log_substep("Computing grid coordinates and distances...")
    grid_lats = lat_array[lat_idxs]
    grid_lons = lon_array[lon_idxs]
    
    fire_df = fire_df.copy()
    fire_df['grid_lat'] = grid_lats
    fire_df['grid_lon'] = grid_lons
    fire_df['grid_lat_idx'] = lat_idxs
    fire_df['grid_lon_idx'] = lon_idxs
    
    # Calculate distance to grid center (approximate, in km)
    # At mid-latitudes: 1° lat ≈ 111 km, 1° lon ≈ 85 km (varies with latitude)
    lat_diff = fire_df['lat'] - fire_df['grid_lat']
    lon_diff = fire_df['lon'] - fire_df['grid_lon']
    
    # Approximate conversion
    lat_km = lat_diff * 111.0
    lon_km = lon_diff * 85.0 * np.cos(np.radians(fire_df['lat']))
    fire_df['dist_to_grid_km'] = np.sqrt(lat_km**2 + lon_km**2)
    
    log_substep(f"  Mean distance to grid center: {fire_df['dist_to_grid_km'].mean():.1f} km")
    log_substep(f"  Max distance to grid center: {fire_df['dist_to_grid_km'].max():.1f} km")
    
    elapsed = time.time() - start_time
    log_substep(f"✓ Alignment complete in {elapsed:.1f}s")
    
    return fire_df


def filter_fires_in_grid(fire_df: pd.DataFrame, grid_info: dict) -> pd.DataFrame:
    """Filter fires to only those within the weather grid bounds."""
    start_time = time.time()
    log_substep("Filtering fires to grid bounds...")
    
    before = len(fire_df)
    mask = (
        (fire_df['lat'] >= grid_info['lat_min']) &
        (fire_df['lat'] <= grid_info['lat_max']) &
        (fire_df['lon'] >= grid_info['lon_min']) &
        (fire_df['lon'] <= grid_info['lon_max'])
    )
    
    # Also filter by date range if available
    if grid_info['date_min'] is not None:
        mask &= (fire_df['fire_date'] >= grid_info['date_min'])
    if grid_info['date_max'] is not None:
        mask &= (fire_df['fire_date'] <= grid_info['date_max'])
    
    filtered_df = fire_df[mask].copy()
    
    log_substep(f"  {before:,} -> {len(filtered_df):,} fires within grid bounds")
    elapsed = time.time() - start_time
    log_substep(f"✓ Filtering complete in {elapsed:.1f}s")
    
    return filtered_df


def compute_grid_fire_counts(fire_df: pd.DataFrame, grid_info: dict) -> pd.DataFrame:
    """
    Compute fire statistics per grid cell.
    
    Returns dataframe with:
        - grid_lat, grid_lon
        - total_fires
        - fires per severity class
        - total_area_ha
        - mean_size_ha
    """
    start_time = time.time()
    log_substep("Computing fire counts per grid cell...")
    
    # Group by grid cell
    log_substep("  Aggregating fire statistics...")
    grouped = fire_df.groupby(['grid_lat', 'grid_lon'])
    
    stats = grouped.agg({
        'fid': 'count',  # Total fires
        'size_ha': ['sum', 'mean', 'max'],
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['grid_lat', 'grid_lon', 'total_fires', 'total_area_ha', 'mean_size_ha', 'max_size_ha']
    
    # Add severity counts as separate columns (parquet-friendly)
    log_substep("  Computing severity class counts...")
    severity_counts = fire_df.groupby(['grid_lat', 'grid_lon', 'severity_class']).size().unstack(fill_value=0)
    severity_counts.columns = [f'severity_c{int(c)}' for c in severity_counts.columns]
    severity_counts = severity_counts.reset_index()
    
    # Merge severity counts into stats
    stats = stats.merge(severity_counts, on=['grid_lat', 'grid_lon'], how='left')
    
    # Fill NaN severity columns with 0
    severity_cols = [c for c in stats.columns if c.startswith('severity_c')]
    stats[severity_cols] = stats[severity_cols].fillna(0).astype(int)
    
    # Add grid indices
    log_substep("  Adding grid indices...")
    lat_to_idx = {lat: i for i, lat in enumerate(grid_info['unique_lats'])}
    lon_to_idx = {lon: i for i, lon in enumerate(grid_info['unique_lons'])}
    
    stats['grid_lat_idx'] = stats['grid_lat'].map(lat_to_idx)
    stats['grid_lon_idx'] = stats['grid_lon'].map(lon_to_idx)
    
    log_substep(f"  Grid cells with fires: {len(stats):,}")
    log_substep(f"  Max fires in single cell: {stats['total_fires'].max():,}")
    
    elapsed = time.time() - start_time
    log_substep(f"✓ Grid statistics computed in {elapsed:.1f}s")
    
    return stats


def compute_daily_fire_counts(fire_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily fire counts for time series visualization."""
    
    daily = fire_df.groupby(fire_df['fire_date'].dt.date).agg({
        'fid': 'count',
        'size_ha': 'sum'
    }).reset_index()
    
    daily.columns = ['date', 'fire_count', 'total_area_ha']
    daily['date'] = pd.to_datetime(daily['date'])
    
    return daily


# ============================================================================
# MAIN
# ============================================================================

def main(config: AlignmentConfig):
    """Run the alignment pipeline."""
    
    total_start_time = time.time()
    
    print("\n" + "=" * 70)
    print("FIRE-WEATHER DATA ALIGNMENT")
    print("=" * 70)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    total_steps = 7
    
    # Step 1: Load weather grid
    log_step(1, total_steps, "Loading weather grid metadata")
    weather_df, grid_info = load_weather_grid(config.power_csv)
    
    # Step 2: Load fire data
    log_step(2, total_steps, "Loading and preprocessing fire data")
    fire_df = load_fire_data(config.nfdb_csv, config)
    
    # Step 3: Filter fires to grid bounds
    log_step(3, total_steps, "Filtering fires to grid spatial bounds")
    fire_df = filter_fires_in_grid(fire_df, grid_info)
    
    # Step 4: Align fires to grid
    log_step(4, total_steps, "Aligning fires to nearest grid points")
    fire_df = align_fires_to_grid(fire_df, grid_info)
    
    # Step 5: Compute grid-level statistics
    log_step(5, total_steps, "Computing grid-level fire statistics")
    grid_stats = compute_grid_fire_counts(fire_df, grid_info)
    
    # Step 6: Compute daily counts
    log_step(6, total_steps, "Computing daily fire counts")
    start_time = time.time()
    daily_counts = compute_daily_fire_counts(fire_df)
    elapsed = time.time() - start_time
    log_substep(f"✓ Daily counts computed in {elapsed:.1f}s")
    
    # Step 7: Save outputs
    log_step(7, total_steps, "Saving output files")
    
    # Save aligned fire data
    log_substep("Saving aligned fire data...")
    fire_output = os.path.join(config.output_dir, "aligned_fires.parquet")
    fire_df.to_parquet(fire_output, index=False)
    log_substep(f"  ✓ {fire_output}")
    
    # Also save as CSV for easy inspection
    fire_csv = os.path.join(config.output_dir, "aligned_fires.csv")
    log_substep("Saving aligned fire data (CSV)...")
    fire_df.to_csv(fire_csv, index=False)
    log_substep(f"  ✓ {fire_csv}")
    
    # Save grid statistics
    log_substep("Saving grid statistics...")
    grid_output = os.path.join(config.output_dir, "grid_fire_counts.parquet")
    grid_stats.to_parquet(grid_output, index=False)
    log_substep(f"  ✓ {grid_output}")
    
    # Save daily counts
    log_substep("Saving daily counts...")
    daily_output = os.path.join(config.output_dir, "daily_fire_counts.csv")
    daily_counts.to_csv(daily_output, index=False)
    log_substep(f"  ✓ {daily_output}")
    
    # Save grid info as JSON
    log_substep("Saving grid metadata...")
    grid_info_serializable = {
        k: v if not isinstance(v, (list, np.ndarray)) else 
           (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in grid_info.items()
        if not isinstance(v, pd.Timestamp)
    }
    grid_info_serializable['date_min'] = str(grid_info['date_min']) if grid_info['date_min'] else None
    grid_info_serializable['date_max'] = str(grid_info['date_max']) if grid_info['date_max'] else None
    
    grid_info_output = os.path.join(config.output_dir, "grid_info.json")
    with open(grid_info_output, 'w') as f:
        json.dump(grid_info_serializable, f, indent=2)
    log_substep(f"  ✓ {grid_info_output}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"Total fires aligned: {len(fire_df):,}")
    print(f"Date range: {fire_df['fire_date'].min().date()} to {fire_df['fire_date'].max().date()}")
    print(f"Grid cells with fires: {len(grid_stats):,} / {grid_info['n_lats'] * grid_info['n_lons']:,}")
    print(f"\nSeverity distribution:")
    print(fire_df['severity_label'].value_counts().sort_index())
    print(f"\nFires by cause (top 5):")
    if 'cause' in fire_df.columns:
        print(fire_df['cause'].value_counts().head())
    
    # Summary stats to save
    log_substep("Saving alignment summary...")
    summary = {
        'total_fires': len(fire_df),
        'date_min': str(fire_df['fire_date'].min().date()),
        'date_max': str(fire_df['fire_date'].max().date()),
        'grid_cells_with_fires': len(grid_stats),
        'total_grid_cells': grid_info['n_lats'] * grid_info['n_lons'],
        'severity_distribution': fire_df['severity_label'].value_counts().to_dict(),
        'mean_dist_to_grid_km': float(fire_df['dist_to_grid_km'].mean()),
        'config': asdict(config)
    }
    
    summary_output = os.path.join(config.output_dir, "alignment_summary.json")
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log_substep(f"  ✓ {summary_output}")
    
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"ALIGNMENT COMPLETE - Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print("=" * 70)
    
    return fire_df, grid_stats, grid_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align fire and weather data")
    parser.add_argument("--nfdb", default="NFDB_point_20240613.txt", help="Path to NFDB fire data")
    parser.add_argument("--weather", default="nasa_power_weather_hires/power_daily_canada_1981_2024_hires.csv", 
                       help="Path to NASA POWER weather data")
    parser.add_argument("--output", default="aligned_data", help="Output directory")
    
    args = parser.parse_args()
    
    config = AlignmentConfig(
        nfdb_csv=args.nfdb,
        power_csv=args.weather,
        output_dir=args.output
    )
    
    main(config)

