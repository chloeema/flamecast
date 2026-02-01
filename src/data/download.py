import os
import asyncio
import hashlib
from io import StringIO

import aiohttp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- CONFIG ----

# NASA POWER daily point endpoint (for high resolution 0.5° data)
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Meteorological parameters to request (<= 20 per request)
# With point API, we can request ALL parameters in a single call!
POWER_PARAMS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",        # <-- relative humidity (air humidity)
    "QV2M",        # <-- specific humidity (optional but nice)
    "PRECTOTCORR",
    "WS2M",
    "GWETPROF",
    "FRSNO",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
]


# Example: rough bounding box for part of Canada (tweak for your study region)
LAT_MIN = 40.0
LAT_MAX = 70.0
LON_MIN = -140.0
LON_MAX = -50.0

# Grid resolution (NASA POWER native resolution is 0.5°)
GRID_RESOLUTION = 0.5  # degrees

START_YEAR = 1981
END_YEAR = 2024

OUTPUT_DIR = "data/raw/nasa_power_weather_hires"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "power_daily_canada_1981_2024_hires.csv")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")  # Directory for cached point results

# Async configuration
BATCH_SIZE = 10  # Number of concurrent requests (point API is lighter weight)
MAX_RETRIES = 5  # Retry failed requests
BATCH_DELAY = 1.0  # Seconds to wait between batches
RETRY_DELAY_429 = 30.0  # Seconds to wait when hitting 429 rate limit

# Memory management
MAX_MEMORY_CHUNK_SIZE = 50  # Maximum number of files to process in memory at once
MERGE_CHUNK_ROWS = 100000  # Number of rows to process at a time when merging


# ---- HELPER ----

def get_cache_filename_point(lat, lon, start_year, end_year, community):
    """Generate a unique cache filename for a point request (all params, multi-year)."""
    cache_key = f"point_{lat}_{lon}_{start_year}_{end_year}_{community}"
    # Use hash to avoid filesystem issues with special characters
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{cache_hash}.csv")


def get_cache_filename(param, lat_min_tile, lat_max_tile, lon_min_tile, lon_max_tile, year, community):
    """Generate a unique cache filename for a regional request (legacy)."""
    cache_key = f"{param}_{lat_min_tile}_{lat_max_tile}_{lon_min_tile}_{lon_max_tile}_{year}_{community}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{cache_hash}.csv")


def load_from_cache(cache_file):
    """Load a dataframe from cache if it exists."""
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            return df
        except Exception as e:
            print(f"    Warning: Failed to load cache file {cache_file}: {e}")
            return None
    return None


def save_to_cache(df, cache_file):
    """Save a dataframe to cache."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_csv(cache_file, index=False)


def get_location_cols(df):
    """Get the actual location/date columns that exist in a dataframe."""
    # Common location/date columns in NASA POWER data
    possible_cols = ['LAT', 'LON', 'YEAR', 'MM', 'DD', 'DOY', 'YEAR_REQUESTED']
    # Return only columns that actually exist in the dataframe
    return [col for col in possible_cols if col in df.columns]


def get_common_location_cols(dataframes):
    """Get location columns that exist in ALL dataframes."""
    if not dataframes:
        return []
    
    # Start with columns from first dataframe
    common_cols = set(get_location_cols(dataframes[0]))
    
    # Intersect with columns from all other dataframes
    for df in dataframes[1:]:
        df_cols = set(get_location_cols(df))
        common_cols = common_cols.intersection(df_cols)
    
    return sorted(list(common_cols))


async def download_point_request(session, lat, lon, parameters, start_year, end_year, community, use_cache=True):
    """Download data for a single point (all parameters, full year range) asynchronously."""
    cache_file = get_cache_filename_point(lat, lon, start_year, end_year, community)
    
    # Check cache first
    if use_cache:
        cached_df = load_from_cache(cache_file)
        if cached_df is not None:
            return {
                "lat": lat,
                "lon": lon,
                "df": cached_df,
                "cached": True
            }
    
    # Point API allows multiple parameters in one request (comma-separated)
    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "latitude": lat,
        "longitude": lon,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "CSV",
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=180)) as response:
                if response.status == 429:
                    wait_time = RETRY_DELAY_429 * (attempt + 1)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP 429 (Rate Limited): {error_text[:200]}")
                elif response.status != 200:
                    error_text = await response.text()
                    if response.status == 422:
                        # 422 = validation error (e.g., ocean point for land-only param)
                        # Don't print for every point - just return None
                        return None
                    else:
                        raise Exception(f"HTTP {response.status}: {error_text[:200]}")
                
                text = await response.text()
                
                # Parse CSV response
                lines = text.splitlines()
                data_start_idx = None
                for i, line in enumerate(lines):
                    if line.strip() == "-END HEADER-":
                        data_start_idx = i + 1
                        break
                
                if data_start_idx is None:
                    data_lines = [ln for ln in lines if not ln.startswith("#") 
                                 and not ln.startswith("-BEGIN") 
                                 and not ln.startswith("-END")
                                 and ln.strip()]
                else:
                    data_lines = lines[data_start_idx:]
                
                if not data_lines:
                    return None
                
                csv_text = "\n".join(data_lines)
                df_point = pd.read_csv(StringIO(csv_text))
                
                # Add LAT/LON columns (point API doesn't include them)
                df_point["LAT"] = lat
                df_point["LON"] = lon
                
                # Save to cache immediately
                if use_cache:
                    save_to_cache(df_point, cache_file)
                
                return {
                    "lat": lat,
                    "lon": lon,
                    "df": df_point,
                    "cached": False
                }
        except Exception as e:
            if "429" in str(e) or "Rate Limited" in str(e):
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_429 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                    continue
            if attempt == MAX_RETRIES - 1:
                if "422" not in str(e):
                    print(f"    Failed after {MAX_RETRIES} attempts for point ({lat}, {lon}): {e}")
                return None
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def download_single_request(session, param, lat_min_tile, lat_max_tile, lon_min_tile, lon_max_tile, year, community, use_cache=True):
    """Download a single regional API request asynchronously (legacy, for reference)."""
    cache_file = get_cache_filename(param, lat_min_tile, lat_max_tile, lon_min_tile, lon_max_tile, year, community)
    
    if use_cache:
        cached_df = load_from_cache(cache_file)
        if cached_df is not None:
            return {
                "param": param,
                "df": cached_df,
                "tile": (lat_min_tile, lat_max_tile, lon_min_tile, lon_max_tile),
                "year": year,
                "cached": True
            }
    
    # Regional API endpoint
    regional_url = "https://power.larc.nasa.gov/api/temporal/daily/regional"
    params = {
        "parameters": param,
        "community": community,
        "latitude-min": lat_min_tile,
        "latitude-max": lat_max_tile,
        "longitude-min": lon_min_tile,
        "longitude-max": lon_max_tile,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "CSV",
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(regional_url, params=params, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 429:
                    wait_time = RETRY_DELAY_429 * (attempt + 1)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP 429 (Rate Limited): {error_text[:200]}")
                elif response.status != 200:
                    error_text = await response.text()
                    if response.status == 422:
                        full_error = error_text[:500] if len(error_text) > 500 else error_text
                        print(f"    HTTP 422 for {param} tile ({lat_min_tile}-{lat_max_tile}, {lon_min_tile}-{lon_max_tile}) year {year}: {full_error}")
                        return None
                    else:
                        raise Exception(f"HTTP {response.status}: {error_text[:200]}")
                
                text = await response.text()
                lines = text.splitlines()
                data_start_idx = None
                for i, line in enumerate(lines):
                    if line.strip() == "-END HEADER-":
                        data_start_idx = i + 1
                        break
                
                if data_start_idx is None:
                    data_lines = [ln for ln in lines if not ln.startswith("#") 
                                 and not ln.startswith("-BEGIN") 
                                 and not ln.startswith("-END")
                                 and ln.strip()]
                else:
                    data_lines = lines[data_start_idx:]
                
                if not data_lines:
                    return None
                
                csv_text = "\n".join(data_lines)
                df_tile = pd.read_csv(StringIO(csv_text))
                df_tile["YEAR_REQUESTED"] = year
                
                if use_cache:
                    save_to_cache(df_tile, cache_file)
                
                return {
                    "param": param,
                    "df": df_tile,
                    "tile": (lat_min_tile, lat_max_tile, lon_min_tile, lon_max_tile),
                    "year": year,
                    "cached": False
                }
        except Exception as e:
            if "429" in str(e) or "Rate Limited" in str(e):
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_429 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                    continue
            if attempt == MAX_RETRIES - 1:
                if "422" not in str(e):
                    print(f"    Failed after {MAX_RETRIES} attempts for {param} tile ({lat_min_tile}-{lat_max_tile}, {lon_min_tile}-{lon_max_tile}) year {year}: {e}")
                return None
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def download_batch(session, semaphore, requests_list, use_cache=True):
    """Download a batch of regional requests with concurrency control (legacy)."""
    async def download_with_semaphore(req):
        async with semaphore:
            return await download_single_request(session, **req, use_cache=use_cache)
    
    results = await asyncio.gather(*[download_with_semaphore(req) for req in requests_list])
    return [r for r in results if r is not None]


async def download_point_batch(session, semaphore, point_requests, parameters, start_year, end_year, community, use_cache=True):
    """Download a batch of point requests with concurrency control."""
    async def download_with_semaphore(req):
        async with semaphore:
            return await download_point_request(
                session, req["lat"], req["lon"], 
                parameters, start_year, end_year, community, use_cache
            )
    
    results = await asyncio.gather(*[download_with_semaphore(req) for req in point_requests])
    return [r for r in results if r is not None]


def scan_cache_directory_points(parameters, community="AG"):
    """Scan cache directory for point API cache files."""
    all_cache_files = []
    
    if not os.path.exists(CACHE_DIR):
        return all_cache_files
    
    print(f"Scanning cache directory: {CACHE_DIR}")
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.csv') and not f.startswith('combined_') and not f.startswith('merged_')]
    
    print(f"Found {len(cache_files)} cache files")
    
    # Just collect valid cache files (point API files have all params in one file)
    for cache_file in tqdm(cache_files, desc="Scanning cache files", unit="file"):
        cache_path = os.path.join(CACHE_DIR, cache_file)
        try:
            # Quick validation - just check header
            df = pd.read_csv(cache_path, nrows=1)
            if 'LAT' in df.columns and 'LON' in df.columns:
                all_cache_files.append(cache_path)
            del df
        except Exception:
            continue
    
    print(f"Found {len(all_cache_files)} valid point cache files")
    return all_cache_files


def scan_cache_directory(parameters, community="AG"):
    """Scan cache directory and build metadata list from existing cache files (legacy regional API)."""
    all_results_metadata = []
    
    if not os.path.exists(CACHE_DIR):
        return all_results_metadata, {}
    
    print(f"Scanning cache directory: {CACHE_DIR}")
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.csv') and not f.startswith('combined_') and not f.startswith('merged_')]
    
    print(f"Found {len(cache_files)} cache files")
    
    param_location_keys = {}
    
    for cache_file in tqdm(cache_files, desc="Scanning cache files", unit="file"):
        cache_path = os.path.join(CACHE_DIR, cache_file)
        try:
            df = pd.read_csv(cache_path)
            location_cols = get_location_cols(df)
            param_cols = [col for col in df.columns if col not in location_cols]
            
            if param_cols:
                param_name = param_cols[0]
                matched_param = None
                for req_param in parameters:
                    if req_param == param_name or param_name.startswith(req_param):
                        matched_param = req_param
                        break
                
                if matched_param:
                    if 'LAT' in df.columns and 'LON' in df.columns:
                        all_results_metadata.append({
                            "param": matched_param,
                            "cache_file": cache_path,
                            "tile": (None, None, None, None),
                            "year": None,
                            "cached": True
                        })
                        
                        if matched_param not in param_location_keys:
                            param_location_keys[matched_param] = set()
                        
                        key_cols = [c for c in ['LAT', 'LON', 'YEAR', 'MM', 'DD'] if c in df.columns]
                        if key_cols:
                            unique_keys = df[key_cols].drop_duplicates()
                            keys_as_tuples = set(map(tuple, unique_keys.itertuples(index=False, name=None)))
                            param_location_keys[matched_param].update(keys_as_tuples)
                
                del df
        except Exception:
            continue
    
    unique_params = {}
    for meta in all_results_metadata:
        param = meta["param"]
        if param not in unique_params:
            unique_params[param] = []
        unique_params[param].append(meta)
    
    print(f"\n{'='*60}")
    print("TILE COVERAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Found cached data for {len(unique_params)} parameters:")
    
    for param, metas in sorted(unique_params.items()):
        num_keys = len(param_location_keys.get(param, set()))
        print(f"  {param}: {len(metas)} cache files, {num_keys:,} unique location-time points")
    
    if param_location_keys:
        all_params = list(param_location_keys.keys())
        intersection = param_location_keys[all_params[0]].copy()
        for param in all_params[1:]:
            intersection &= param_location_keys[param]
        
        print(f"\n{'='*60}")
        print(f"INTERSECTION: {len(intersection):,} location-time points common to ALL {len(all_params)} parameters")
        print(f"{'='*60}\n")
    
    print(f"Total cache entries: {len(all_results_metadata)}")
    
    return all_results_metadata, param_location_keys


def download_power_region(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start_year,
    end_year,
    parameters,
    output_csv,
    community="AG",
    use_cache=True,
    compile_only=False,
    grid_resolution=None,
):
    """
    Download daily NASA POWER data for a lat/lon bounding box and year range.

    - Uses the /temporal/daily/point endpoint for high resolution (0.5°) data
    - Downloads ALL parameters in a single request per point (efficient!)
    - Each point request covers the full year range
    - Uses async/await for faster parallel downloads
    - Caches individual point results to disk for resumption
    - Returns a pandas DataFrame and writes a CSV to disk
    
    Args:
        compile_only: If True, skip download phase and only process existing cache files
        grid_resolution: Grid spacing in degrees (default: GRID_RESOLUTION from config)
    """
    import gc
    
    if grid_resolution is None:
        grid_resolution = GRID_RESOLUTION
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Generate grid points at specified resolution
    lats = np.arange(lat_min + grid_resolution/2, lat_max, grid_resolution)
    lons = np.arange(lon_min + grid_resolution/2, lon_max, grid_resolution)
    
    # Build list of all points
    all_points = []
    for lat in lats:
        for lon in lons:
            all_points.append({"lat": round(lat, 2), "lon": round(lon, 2)})
    
    total_points = len(all_points)
    
    print(f"\n{'='*60}")
    print("NASA POWER HIGH-RESOLUTION POINT DATA DOWNLOAD")
    print(f"{'='*60}")
    print(f"Grid resolution: {grid_resolution}°")
    print(f"Latitude range: {lat_min}° to {lat_max}° ({len(lats)} points)")
    print(f"Longitude range: {lon_min}° to {lon_max}° ({len(lons)} points)")
    print(f"Total grid points: {total_points:,}")
    print(f"Image resolution: {len(lons)} x {len(lats)} pixels")
    print(f"Year range: {start_year} to {end_year} ({end_year - start_year + 1} years)")
    print(f"Parameters: {len(parameters)} (all fetched in single request per point)")
    print(f"Total API calls needed: {total_points:,}")
    print(f"Batch size: {BATCH_SIZE} concurrent requests")
    print(f"{'='*60}\n")
    
    # If compile_only, just process existing cache files
    if compile_only:
        print("=== COMPILE-ONLY MODE: Processing existing cache files ===\n")
        cache_files = scan_cache_directory_points(parameters, community)
        if not cache_files:
            raise ValueError("No cache files found. Run without compile_only=True to download data first.")
    else:
        # Check which points are already cached
        cached_points = set()
        uncached_points = []
        
        print("Checking cache for existing data...")
        for point in tqdm(all_points, desc="Checking cache", unit="point"):
            cache_file = get_cache_filename_point(point["lat"], point["lon"], start_year, end_year, community)
            if os.path.exists(cache_file):
                cached_points.add((point["lat"], point["lon"]))
            else:
                uncached_points.append(point)
        
        print(f"  Already cached: {len(cached_points):,} points")
        print(f"  Need to download: {len(uncached_points):,} points")
        
        if uncached_points:
            # Run async downloads for uncached points
        async def run_downloads():
            semaphore = asyncio.Semaphore(BATCH_SIZE)
            async with aiohttp.ClientSession() as session:
                    downloaded_count = 0
                    failed_count = 0
                    total_batches = (len(uncached_points) + BATCH_SIZE - 1) // BATCH_SIZE
                
                    for i in range(0, len(uncached_points), BATCH_SIZE):
                        batch = uncached_points[i:i + BATCH_SIZE]
                    batch_num = i // BATCH_SIZE + 1
                        
                        if batch_num % 10 == 0 or batch_num == 1:
                            print(f"  Downloading batch {batch_num}/{total_batches} ({downloaded_count:,} done, {failed_count:,} failed)...")
                    
                        batch_results = await download_point_batch(
                            session, semaphore, batch,
                            parameters, start_year, end_year, community, use_cache
                        )
                        
                        downloaded_count += len(batch_results)
                        failed_count += len(batch) - len(batch_results)
                        
                        # Free memory from results (data is already cached to disk)
                        for result in batch_results:
                            if "df" in result:
                        del result["df"]
                    
                        # Add delay between batches
                        if i + BATCH_SIZE < len(uncached_points):
                        await asyncio.sleep(BATCH_DELAY)
                
                    return downloaded_count, failed_count
    
            print(f"\nStarting downloads...")
            downloaded_count, failed_count = asyncio.run(run_downloads())
            print(f"\nDownload complete: {downloaded_count:,} successful, {failed_count:,} failed (ocean/invalid points)")
        
        # Get list of all cache files
        cache_files = scan_cache_directory_points(parameters, community)
    
    # Combine all point cache files into a single output file
    # Point API files already contain ALL parameters, so no merging needed!
    print(f"\nCombining {len(cache_files):,} point files into output...")
    
    header_written = False
    rows_written = 0
    
    for i, cache_file in enumerate(tqdm(cache_files, desc="Combining files", unit="file")):
        try:
            df = pd.read_csv(cache_file)
            
            if not header_written:
                df.to_csv(output_csv, index=False, mode='w')
                header_written = True
            else:
                df.to_csv(output_csv, index=False, mode='a', header=False)
            
            rows_written += len(df)
            del df
            
            # Periodic garbage collection
            if (i + 1) % 100 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"  Warning: Failed to process {cache_file}: {e}")
            continue
    
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows written: {rows_written:,}")
    print(f"Output file: {output_csv}")
    
    # Calculate actual grid coverage
    df_sample = pd.read_csv(output_csv, usecols=['LAT', 'LON'], nrows=1000000)
    unique_lats = df_sample['LAT'].nunique()
    unique_lons = df_sample['LON'].nunique()
    print(f"Grid coverage: {unique_lons} x {unique_lats} pixels (from sample)")
    print(f"{'='*60}\n")
    
    # Return a sample for display
    df = pd.read_csv(output_csv, nrows=1000)
    return df


if __name__ == "__main__":
    import time
    import sys
    
    # Check if --compile-only flag is passed
    compile_only = "--compile-only" in sys.argv or "-c" in sys.argv
    
    start_time = time.time()
    
    df_power = download_power_region(
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        start_year=START_YEAR,
        end_year=END_YEAR,
        parameters=POWER_PARAMS,
        output_csv=OUTPUT_CSV,
        compile_only=compile_only,
    )

    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nFirst few rows:")
    print(df_power.head())
