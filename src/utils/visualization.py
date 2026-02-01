#!/usr/bin/env python3
"""
Interactive Fire-Weather Visualization

Provides an interactive visualization to explore:
- Weather channels over time (slide through dates)
- Fire events appearing on the map
- Channel comparisons

Usage:
    # In Jupyter notebook:
    from visualize_fire_weather import FireWeatherExplorer
    explorer = FireWeatherExplorer("nasa_power_weather_hires/...", "aligned_data/aligned_fires.parquet")
    explorer.show()
    
    # Or run standalone to generate HTML:
    python visualize_fire_weather.py --output visualization.html
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None  # Placeholder for type hints
    px = None
    make_subplots = None
    print("Warning: plotly not installed. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.widgets import Slider, RadioButtons, Button
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# DATA LOADER
# ============================================================================

class WeatherGridData:
    """Efficient loader for weather grid data with date indexing."""
    
    # NASA POWER uses -999 as fill value for missing data
    FILL_VALUE = -999.0
    
    def __init__(self, csv_path: str, sample_frac: float = 1.0):
        print(f"Loading weather data from {csv_path}...")
        
        self.df = pd.read_csv(csv_path)
        
        if sample_frac < 1.0:
            self.df = self.df.sample(frac=sample_frac, random_state=42)
        
        # Build date column
        if 'YEAR' in self.df.columns and 'DOY' in self.df.columns:
            self.df['DATE'] = pd.to_datetime(
                self.df['YEAR'].astype(str) + self.df['DOY'].astype(str).str.zfill(3),
                format='%Y%j'
            )
        
        # Get unique coordinates
        self.unique_lats = sorted(self.df['LAT'].unique())
        self.unique_lons = sorted(self.df['LON'].unique())
        self.lat_to_idx = {lat: i for i, lat in enumerate(self.unique_lats)}
        self.lon_to_idx = {lon: i for i, lon in enumerate(self.unique_lons)}
        
        # Get available channels
        location_cols = ['LAT', 'LON', 'YEAR', 'DOY', 'DATE', 'MM', 'DD', 'YEAR_REQUESTED']
        self.channels = [c for c in self.df.columns if c not in location_cols]
        
        # Replace -999 fill values with NaN for all weather channels
        print("  Replacing -999 fill values with NaN...")
        for ch in self.channels:
            fill_count = (self.df[ch] == self.FILL_VALUE).sum()
            if fill_count > 0:
                print(f"    {ch}: {fill_count:,} fill values replaced")
                self.df.loc[self.df[ch] == self.FILL_VALUE, ch] = np.nan
        
        # Get date range
        self.dates = sorted(self.df['DATE'].unique())
        self.date_min = min(self.dates)
        self.date_max = max(self.dates)
        
        # Get available years
        self.years = sorted(self.df['YEAR'].unique())
        
        # Compute normalization stats (excluding NaN)
        self.channel_stats = {}
        print("  Computing channel statistics...")
        for ch in self.channels:
            valid = self.df[ch].dropna()
            if len(valid) > 0:
                self.channel_stats[ch] = {
                    'mean': valid.mean(),
                    'std': valid.std(),
                    'min': valid.min(),
                    'max': valid.max(),
                    'p5': valid.quantile(0.05),
                    'p95': valid.quantile(0.95),
                    'valid_count': len(valid),
                    'missing_pct': (1 - len(valid) / len(self.df)) * 100,
                }
                print(f"    {ch}: range [{self.channel_stats[ch]['p5']:.2f}, {self.channel_stats[ch]['p95']:.2f}], {self.channel_stats[ch]['missing_pct']:.1f}% missing")
            else:
                self.channel_stats[ch] = {
                    'mean': 0, 'std': 1, 'min': 0, 'max': 1, 'p5': 0, 'p95': 1,
                    'valid_count': 0, 'missing_pct': 100,
                }
                print(f"    {ch}: NO VALID DATA")
        
        print(f"  Loaded {len(self.df):,} rows")
        print(f"  Grid: {len(self.unique_lons)} x {len(self.unique_lats)}")
        print(f"  Years: {min(self.years)} to {max(self.years)} ({len(self.years)} years)")
        print(f"  Channels: {self.channels}")
    
    def get_frame(self, date: pd.Timestamp, channel: str, normalize: bool = False) -> np.ndarray:
        """Get a single day's data as a 2D grid."""
        
        day_data = self.df[self.df['DATE'] == date]
        
        H, W = len(self.unique_lats), len(self.unique_lons)
        grid = np.full((H, W), np.nan)
        
        for _, row in day_data.iterrows():
            lat_idx = self.lat_to_idx.get(row['LAT'], -1)
            lon_idx = self.lon_to_idx.get(row['LON'], -1)
            if lat_idx >= 0 and lon_idx >= 0:
                val = row[channel]
                if normalize and not np.isnan(val):
                    stats = self.channel_stats[channel]
                    val = (val - stats['mean']) / (stats['std'] + 1e-8)
                grid[lat_idx, lon_idx] = val
        
        return grid
    
    def get_available_dates(self, year: Optional[int] = None) -> List[pd.Timestamp]:
        """Get list of available dates, optionally filtered by year."""
        if year:
            return [d for d in self.dates if d.year == year]
        return self.dates


class FireEventData:
    """Loader for aligned fire event data."""
    
    def __init__(self, parquet_path: str):
        print(f"Loading fire data from {parquet_path}...")
        
        if parquet_path.endswith('.parquet'):
            self.df = pd.read_parquet(parquet_path)
        else:
            self.df = pd.read_csv(parquet_path)
        
        # Ensure date column
        if 'fire_date' in self.df.columns:
            self.df['fire_date'] = pd.to_datetime(self.df['fire_date'])
        
        print(f"  Loaded {len(self.df):,} fire events")
        print(f"  Date range: {self.df['fire_date'].min().date()} to {self.df['fire_date'].max().date()}")
    
    def get_fires_on_date(self, date: pd.Timestamp, window_days: int = 0) -> pd.DataFrame:
        """Get fires on a specific date (or within a window)."""
        
        if window_days > 0:
            start = date - pd.Timedelta(days=window_days)
            end = date + pd.Timedelta(days=window_days)
            mask = (self.df['fire_date'] >= start) & (self.df['fire_date'] <= end)
        else:
            mask = self.df['fire_date'].dt.date == date.date()
        
        return self.df[mask]
    
    def get_fires_in_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Get fires within a date range."""
        mask = (self.df['fire_date'] >= start_date) & (self.df['fire_date'] <= end_date)
        return self.df[mask]


# ============================================================================
# PLOTLY INTERACTIVE VISUALIZATION
# ============================================================================

class FireWeatherExplorer:
    """Interactive explorer for fire-weather data using Plotly."""
    
    def __init__(self, weather_csv: str, fire_parquet: str):
        self.weather = WeatherGridData(weather_csv)
        self.fires = FireEventData(fire_parquet)
        
        # Default settings
        self.current_date = self.weather.dates[len(self.weather.dates) // 2]
        self.current_channel = self.weather.channels[0]
        self.fire_window = 3  # Days before/after to show fires
    
    def create_weather_heatmap(self, date: pd.Timestamp, channel: str):
        """Create a heatmap of weather data for a given date and channel."""
        
        grid = self.weather.get_frame(date, channel)
        stats = self.weather.channel_stats[channel]
        
        # Create figure
        fig = go.Figure()
        
        # Weather heatmap
        fig.add_trace(go.Heatmap(
            z=grid,
            x=self.weather.unique_lons,
            y=self.weather.unique_lats,
            colorscale='RdYlBu_r' if 'T2M' in channel else 'Blues',
            zmin=stats['p5'],
            zmax=stats['p95'],
            colorbar=dict(title=channel),
            hovertemplate='Lat: %{y:.1f}<br>Lon: %{x:.1f}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        # Add fire events
        fires_df = self.fires.get_fires_on_date(date, window_days=self.fire_window)
        
        if len(fires_df) > 0:
            # Size based on fire size (log scale)
            sizes = np.log1p(fires_df['size_ha'].fillna(1)) * 3 + 5
            
            # Color based on severity
            colors = fires_df['severity_class'].fillna(0)
            
            fig.add_trace(go.Scatter(
                x=fires_df['lon'],
                y=fires_df['lat'],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='Reds',
                    cmin=0,
                    cmax=4,
                    line=dict(width=1, color='black'),
                    symbol='circle',
                ),
                text=fires_df.apply(
                    lambda r: f"Date: {r['fire_date'].date()}<br>"
                             f"Size: {r['size_ha']:.1f} ha<br>"
                             f"Severity: {r.get('severity_label', 'N/A')}",
                    axis=1
                ),
                hoverinfo='text',
                name=f'Fires (±{self.fire_window} days)'
            ))
        
        fig.update_layout(
            title=f'{channel} - {date.date()} ({len(fires_df)} fires in window)',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            height=600,
            width=900,
        )
        
        return fig
    
    def create_time_slider_figure(self, channel: str, year: int):
        """Create a figure with a time slider for animation."""
        
        dates = self.weather.get_available_dates(year)
        
        if len(dates) == 0:
            print(f"No data for year {year}")
            return None
        
        # Sample dates for performance (every 7 days)
        dates = dates[::7]
        
        # Create frames
        frames = []
        for date in dates:
            grid = self.weather.get_frame(date, channel)
            fires_df = self.fires.get_fires_on_date(date, window_days=3)
            
            frame_data = [
                go.Heatmap(
                    z=grid,
                    x=self.weather.unique_lons,
                    y=self.weather.unique_lats,
                    colorscale='RdYlBu_r' if 'T2M' in channel else 'Blues',
                )
            ]
            
            if len(fires_df) > 0:
                sizes = np.log1p(fires_df['size_ha'].fillna(1)) * 3 + 5
                frame_data.append(go.Scatter(
                    x=fires_df['lon'],
                    y=fires_df['lat'],
                    mode='markers',
                    marker=dict(size=sizes, color='red', line=dict(width=1, color='black')),
                ))
            
            frames.append(go.Frame(data=frame_data, name=str(date.date())))
        
        # Initial figure
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames,
            layout=go.Layout(
                title=f'{channel} - {year}',
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'y': 0,
                    'x': 0.1,
                    'xanchor': 'right',
                    'yanchor': 'top',
                    'buttons': [
                        {'label': '▶', 'method': 'animate', 
                         'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': '⏸', 'method': 'animate',
                         'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
                    ]
                }],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'prefix': 'Date: ',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'steps': [
                        {'args': [[str(dates[i].date())], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                         'label': str(dates[i].date()),
                         'method': 'animate'}
                        for i in range(len(dates))
                    ]
                }]
            )
        )
        
        return fig
    
    def create_channel_comparison(self, date: pd.Timestamp, channels: List[str]):
        """Create a multi-panel comparison of different channels."""
        
        n_channels = len(channels)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=channels,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        fires_df = self.fires.get_fires_on_date(date, window_days=3)
        
        for i, channel in enumerate(channels):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            grid = self.weather.get_frame(date, channel)
            stats = self.weather.channel_stats[channel]
            
            fig.add_trace(
                go.Heatmap(
                    z=grid,
                    x=self.weather.unique_lons,
                    y=self.weather.unique_lats,
                    colorscale='RdYlBu_r' if 'T2M' in channel else 'Blues',
                    zmin=stats['p5'],
                    zmax=stats['p95'],
                    showscale=True,
                    colorbar=dict(len=0.3, y=1 - (row - 0.5) / n_rows),
                ),
                row=row, col=col
            )
            
            # Add fires
            if len(fires_df) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fires_df['lon'],
                        y=fires_df['lat'],
                        mode='markers',
                        marker=dict(size=8, color='red', line=dict(width=1, color='black')),
                        showlegend=(i == 0),
                        name='Fires'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f'Channel Comparison - {date.date()} ({len(fires_df)} fires)',
            height=300 * n_rows,
            width=400 * n_cols,
        )
        
        return fig
    
    def create_fire_timeline(self, year: int):
        """Create a timeline of fire activity for a year."""
        
        start = pd.Timestamp(f'{year}-01-01')
        end = pd.Timestamp(f'{year}-12-31')
        
        fires_year = self.fires.get_fires_in_range(start, end)
        
        if len(fires_year) == 0:
            print(f"No fires in {year}")
            return None
        
        # Daily counts
        daily = fires_year.groupby(fires_year['fire_date'].dt.date).agg({
            'fid': 'count',
            'size_ha': 'sum'
        }).reset_index()
        daily.columns = ['date', 'count', 'total_area']
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=daily['date'], y=daily['count'], name='Fire Count', marker_color='orange'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=daily['date'], y=daily['total_area'], name='Total Area (ha)', 
                      line=dict(color='red', width=2)),
            secondary_y=True
        )
        
        fig.update_layout(
            title=f'Fire Activity Timeline - {year}',
            xaxis_title='Date',
            height=400,
        )
        fig.update_yaxes(title_text='Fire Count', secondary_y=False)
        fig.update_yaxes(title_text='Total Area (ha)', secondary_y=True)
        
        return fig
    
    def show_interactive(self, year: int = None, channel: str = None):
        """Launch interactive visualization (for Jupyter)."""
        
        if not HAS_PLOTLY:
            print("Please install plotly: pip install plotly")
            return
        
        if year is None:
            year = max(self.weather.years)  # Default to most recent year
        
        if channel is None:
            channel = self.weather.channels[0]
        
        print(f"Available years: {min(self.weather.years)} - {max(self.weather.years)}")
        print(f"Available channels: {self.weather.channels}")
        print(f"Showing: year={year}, channel={channel}")
        
        # Show timeline
        timeline = self.create_fire_timeline(year)
        if timeline:
            timeline.show()
        
        # Show animated map
        animated = self.create_time_slider_figure(channel, year)
        if animated:
            animated.show()
        
        # Show channel comparison for mid-summer
        mid_date = pd.Timestamp(f'{year}-07-15')
        comparison = self.create_channel_comparison(mid_date, self.weather.channels[:6])
        comparison.show()
    
    def save_html(self, output_path: str, year: int = None, channel: str = None, years_to_process: list = None):
        """Save visualization as standalone HTML with interactive controls including date slider."""
        
        if not HAS_PLOTLY:
            print("Please install plotly: pip install plotly")
            return
        
        if channel is None:
            channel = self.weather.channels[0]
        
        if year is None:
            year = max(self.weather.years)  # Default to most recent year
        
        # Store years_to_process for _generate_all_data_json
        self._years_to_process = years_to_process
        
        # For HTML dropdown, show only the processed years
        if years_to_process:
            available_years = sorted(years_to_process)
        else:
            available_years = self.weather.years
        available_channels = self.weather.channels
        
        print(f"Generating visualization for year {year}, channel {channel}")
        if years_to_process:
            print(f"Processing years: {years_to_process}")
        else:
            print(f"Processing all years: {min(self.weather.years)} - {max(self.weather.years)}")
        print(f"Available channels: {available_channels}")
        
        # Create HTML with embedded JavaScript for interactivity
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Fire-Weather Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1, h2 {{ color: #fff; }}
        .controls {{ 
            background: #16213e; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .control-group {{ display: inline-block; margin-right: 20px; margin-bottom: 10px; }}
        label {{ font-weight: bold; margin-right: 8px; color: #aaa; }}
        select {{ 
            padding: 8px 12px; 
            font-size: 14px; 
            border-radius: 4px;
            border: 1px solid #444;
            background: #0f3460;
            color: #fff;
        }}
        input[type="range"] {{
            width: 400px;
            accent-color: #e94560;
        }}
        .date-display {{
            display: inline-block;
            background: #e94560;
            color: #fff;
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 100px;
            text-align: center;
        }}
        .chart-container {{ 
            background: #16213e; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .info {{ 
            background: #0f3460; 
            padding: 10px; 
            border-radius: 4px; 
            margin-bottom: 15px;
            font-size: 13px;
            color: #aaa;
        }}
        .stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }}
        .stat-card {{
            background: #0f3460;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #e94560;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #fff; }}
        .stat-label {{ font-size: 12px; color: #888; }}
        .slider-row {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 10px;
        }}
        .playback-btn {{
            background: #e94560;
            border: none;
            color: #fff;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .playback-btn:hover {{
            background: #ff6b6b;
        }}
    </style>
</head>
<body>
    <h1>🔥 Fire-Weather Data Explorer</h1>
    
    <div class="controls">
        <div class="control-group">
            <label for="year-select">Year:</label>
            <select id="year-select" onchange="onYearChange()">
                {"".join(f'<option value="{y}" {"selected" if y == year else ""}>{y}</option>' for y in available_years)}
            </select>
        </div>
        <div class="control-group">
            <label for="channel-select">Weather Channel:</label>
            <select id="channel-select" onchange="updateMap()">
                {"".join(f'<option value="{ch}" {"selected" if ch == channel else ""}>{ch}</option>' for ch in available_channels)}
            </select>
        </div>
    </div>
    
    <div class="info">
        <strong>Data Info:</strong> 
        Grid: {len(self.weather.unique_lons)} x {len(self.weather.unique_lats)} | 
        Years: {min(available_years)}-{max(available_years)} | 
        Total fires: {len(self.fires.df):,}
    </div>
    
    <div id="stats-container" class="stats"></div>
    
    <div class="chart-container">
        <h2>Fire Activity Timeline (click to jump to date)</h2>
        <div id="timeline-chart"></div>
    </div>
    
    <div class="chart-container">
        <h2 id="map-title">Weather Map with Fires</h2>
        <div class="slider-row" style="margin-bottom: 15px;">
            <label>Date:</label>
            <input type="range" id="date-slider" min="0" max="51" value="25" oninput="onSliderChange()" style="flex: 1;">
            <span class="date-display" id="current-date">--</span>
            <button class="playback-btn" id="play-btn" onclick="togglePlayback()">▶ Play</button>
        </div>
        <div id="map-chart"></div>
    </div>
    
    <div class="chart-container">
        <h2 id="comparison-title">All Channels Comparison</h2>
        <div id="comparison-chart"></div>
    </div>

    <script>
        // Store all precomputed data
        const allData = {self._generate_all_data_json()};
        
        let isPlaying = false;
        let playInterval = null;
        let currentWeekIndex = 25; // Default to ~late June
        
        function getWeekDates(year) {{
            // Generate weekly dates for the year
            const dates = [];
            for (let week = 0; week < 52; week++) {{
                const d = new Date(year, 0, 1 + week * 7);
                dates.push(d.toISOString().split('T')[0]);
            }}
            return dates;
        }}
        
        function onYearChange() {{
            updateStats();
            updateTimeline();
            updateMap();
            updateComparison();
        }}
        
        function onSliderChange() {{
            currentWeekIndex = parseInt(document.getElementById('date-slider').value);
            updateMap();
            updateComparison();
        }}
        
        function togglePlayback() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('play-btn');
            if (isPlaying) {{
                btn.innerText = '⏸ Pause';
                playInterval = setInterval(() => {{
                    currentWeekIndex = (currentWeekIndex + 1) % 52;
                    document.getElementById('date-slider').value = currentWeekIndex;
                    updateMap();
                    updateComparison();
                }}, 500);
            }} else {{
                btn.innerText = '▶ Play';
                clearInterval(playInterval);
            }}
        }}
        
        function updateStats() {{
            const year = parseInt(document.getElementById('year-select').value);
            const yearData = allData.fires_by_year[year] || {{count: 0, area: 0}};
            document.getElementById('stats-container').innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${{yearData.count.toLocaleString()}}</div>
                    <div class="stat-label">Fires in ${{year}}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{Math.round(yearData.area).toLocaleString()}} ha</div>
                    <div class="stat-label">Total Area Burned</div>
                </div>
            `;
        }}
        
        function updateTimeline() {{
            const year = parseInt(document.getElementById('year-select').value);
            const data = allData.timelines[year];
            if (!data) {{
                Plotly.newPlot('timeline-chart', [], {{title: 'No data for ' + year}});
                return;
            }}
            
            const trace1 = {{
                x: data.dates,
                y: data.counts,
                type: 'bar',
                name: 'Fire Count',
                marker: {{color: '#e94560'}}
            }};
            
            const trace2 = {{
                x: data.dates,
                y: data.areas,
                type: 'scatter',
                mode: 'lines',
                name: 'Total Area (ha)',
                yaxis: 'y2',
                line: {{color: '#ff9f43', width: 2}}
            }};
            
            const layout = {{
                title: 'Fire Activity - ' + year,
                xaxis: {{title: 'Date', color: '#aaa'}},
                yaxis: {{title: 'Fire Count', side: 'left', color: '#aaa'}},
                yaxis2: {{title: 'Total Area (ha)', side: 'right', overlaying: 'y', color: '#aaa'}},
                height: 300,
                showlegend: true,
                paper_bgcolor: '#16213e',
                plot_bgcolor: '#0f3460',
                font: {{color: '#aaa'}},
                legend: {{font: {{color: '#aaa'}}}}
            }};
            
            Plotly.newPlot('timeline-chart', [trace1, trace2], layout);
            
            // Add click handler to jump to date
            document.getElementById('timeline-chart').on('plotly_click', function(data) {{
                if (data.points && data.points[0]) {{
                    const clickedDate = data.points[0].x;
                    // Find which week this corresponds to
                    const dateObj = new Date(clickedDate);
                    const startOfYear = new Date(dateObj.getFullYear(), 0, 1);
                    const weekNum = Math.floor((dateObj - startOfYear) / (7 * 24 * 60 * 60 * 1000));
                    currentWeekIndex = Math.min(51, Math.max(0, weekNum));
                    document.getElementById('date-slider').value = currentWeekIndex;
                    updateMap();
                    updateComparison();
                }}
            }});
        }}
        
        function updateMap() {{
            const year = parseInt(document.getElementById('year-select').value);
            const channel = document.getElementById('channel-select').value;
            const weekDates = getWeekDates(year);
            const currentDate = weekDates[currentWeekIndex];
            
            document.getElementById('current-date').innerText = currentDate;
            
            // Find the closest available week in data
            const mapKey = year + '_' + channel + '_' + currentWeekIndex;
            const mapData = allData.week_maps[mapKey];
            
            document.getElementById('map-title').innerText = channel + ' Map with Fires - ' + currentDate;
            
            if (!mapData) {{
                // Fall back to July data if week data not available
                const fallbackKey = year + '_' + channel;
                const fallbackData = allData.maps[fallbackKey];
                if (!fallbackData) {{
                    Plotly.newPlot('map-chart', [], {{title: 'No map data for ' + currentDate}});
                    return;
                }}
            }}
            
            const weatherData = mapData || allData.maps[year + '_' + channel];
            if (!weatherData) {{
                Plotly.newPlot('map-chart', [], {{title: 'No map data'}});
                return;
            }}
            
            // Get colorscale based on channel
            let colorscale = 'Blues';
            let reversescale = false;
            if (channel.includes('T2M') || channel.includes('TEMP')) {{
                colorscale = 'RdYlBu';
                reversescale = true;
            }} else if (channel.includes('PREC') || channel.includes('RH')) {{
                colorscale = 'Blues';
            }} else if (channel.includes('WIND') || channel.includes('WS')) {{
                colorscale = 'Greens';
            }}
            
            const traces = [{{
                z: weatherData.values,
                x: allData.lons,
                y: allData.lats,
                type: 'heatmap',
                colorscale: colorscale,
                reversescale: reversescale,
                colorbar: {{title: channel, tickfont: {{color: '#aaa'}}, titlefont: {{color: '#aaa'}}}},
                hovertemplate: 'Lat: %{{y:.1f}}<br>Lon: %{{x:.1f}}<br>Value: %{{z:.2f}}<extra></extra>'
            }}];
            
            // Get fires within 2 weeks of current date
            const fires = allData.week_fires[year + '_' + currentWeekIndex];
            
            if (fires && fires.lats.length > 0) {{
                traces.push({{
                    x: fires.lons,
                    y: fires.lats,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: fires.sizes.map(s => Math.log(s + 1) * 3 + 5),
                        color: '#ff6b6b',
                        opacity: 0.8,
                        line: {{width: 1, color: '#fff'}}
                    }},
                    text: fires.texts,
                    hoverinfo: 'text',
                    name: 'Fires (±7 days)'
                }});
            }}
            
            const layout = {{
                xaxis: {{title: 'Longitude', color: '#aaa'}},
                yaxis: {{title: 'Latitude', color: '#aaa'}},
                height: 550,
                margin: {{t: 30}},
                paper_bgcolor: '#16213e',
                plot_bgcolor: '#0f3460',
                font: {{color: '#aaa'}}
            }};
            
            Plotly.newPlot('map-chart', traces, layout);
        }}
        
        function updateComparison() {{
            const year = parseInt(document.getElementById('year-select').value);
            const weekDates = getWeekDates(year);
            const currentDate = weekDates[currentWeekIndex];
            
            document.getElementById('comparison-title').innerText = 'All Channels - ' + currentDate;
            
            // Build comparison data from available week maps
            const channels = allData.channels;
            const compData = {{}};
            
            channels.forEach(ch => {{
                const mapKey = year + '_' + ch + '_' + currentWeekIndex;
                const data = allData.week_maps[mapKey];
                if (data) {{
                    compData[ch] = data.values;
                }}
            }});
            
            const availableChannels = Object.keys(compData);
            if (availableChannels.length === 0) {{
                Plotly.newPlot('comparison-chart', [], {{title: 'No comparison data for ' + currentDate}});
                return;
            }}
            
            const nCols = Math.min(3, availableChannels.length);
            const nRows = Math.ceil(availableChannels.length / nCols);
            
            const traces = [];
            const annotations = [];
            
            availableChannels.forEach((ch, i) => {{
                const row = Math.floor(i / nCols);
                const col = i % nCols;
                
                let colorscale = 'Blues';
                let reversescale = false;
                if (ch.includes('T2M') || ch.includes('TEMP')) {{
                    colorscale = 'RdYlBu';
                    reversescale = true;
                }} else if (ch.includes('WIND') || ch.includes('WS')) {{
                    colorscale = 'Greens';
                }}
                
                traces.push({{
                    z: compData[ch],
                    x: allData.lons,
                    y: allData.lats,
                    type: 'heatmap',
                    colorscale: colorscale,
                    reversescale: reversescale,
                    showscale: true,
                    xaxis: 'x' + (i + 1),
                    yaxis: 'y' + (i + 1),
                    name: ch
                }});
                annotations.push({{
                    text: ch,
                    x: (col + 0.5) / nCols,
                    y: 1 - row / nRows + 0.02,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: {{size: 12, color: '#fff'}}
                }});
            }});
            
            // Simple grid layout
            const layout = {{
                grid: {{rows: nRows, columns: nCols, pattern: 'independent'}},
                height: 300 * nRows,
                showlegend: false,
                annotations: annotations,
                paper_bgcolor: '#16213e',
                plot_bgcolor: '#0f3460',
                font: {{color: '#aaa'}}
            }};
            
            Plotly.newPlot('comparison-chart', traces, layout);
        }}
        
        // Initial load
        updateStats();
        updateTimeline();
        updateMap();
        updateComparison();
    </script>
</body>
</html>'''
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Saved visualization to {output_path}")
    
    def _generate_all_data_json(self):
        """Generate JSON data for all years and channels with weekly snapshots."""
        import json
        
        # Determine which years to process
        if hasattr(self, '_years_to_process') and self._years_to_process:
            years_to_process = self._years_to_process
        else:
            years_to_process = self.weather.years
        
        data = {
            'lats': [float(x) for x in self.weather.unique_lats],
            'lons': [float(x) for x in self.weather.unique_lons],
            'channels': self.weather.channels,
            'years': [int(y) for y in years_to_process],
            'timelines': {},
            'maps': {},          # Fallback: July data per year/channel
            'week_maps': {},     # Weekly: year_channel_weeknum
            'week_fires': {},    # Weekly fires: year_weeknum
            'fires_by_year': {},
        }
        
        # Generate data for each year
        for year in years_to_process:
            print(f"  Processing year {year}...")
            
            # Timeline data
            start = pd.Timestamp(f'{year}-01-01')
            end = pd.Timestamp(f'{year}-12-31')
            fires_year = self.fires.get_fires_in_range(start, end)
            
            if len(fires_year) > 0:
                daily = fires_year.groupby(fires_year['fire_date'].dt.date).agg({
                    'fid': 'count',
                    'size_ha': 'sum'
                }).reset_index()
                daily.columns = ['date', 'count', 'area']
                
                data['timelines'][int(year)] = {
                    'dates': [str(d) for d in daily['date']],
                    'counts': [int(c) for c in daily['count']],
                    'areas': [float(a) for a in daily['area']]
                }
                
                data['fires_by_year'][int(year)] = {
                    'count': int(len(fires_year)),
                    'area': float(fires_year['size_ha'].sum())
                }
            else:
                data['fires_by_year'][int(year)] = {'count': 0, 'area': 0}
            
            # Generate weekly snapshots - every 2 weeks for good temporal resolution
            # This gives 26 samples per year, enough to see seasonal changes
            week_intervals = list(range(0, 52, 2))  # Every 2 weeks: 0, 2, 4, ..., 50
            
            for week_idx in week_intervals:
                # Calculate date for this week
                week_date = pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=week_idx * 7)
                if week_date.year != year:
                    week_date = pd.Timestamp(f'{year}-12-31')
                
                # Get weather data for each channel
                for channel in self.weather.channels:
                    grid = self.weather.get_frame(week_date, channel)
                    if grid is not None:
                        grid_list = [[None if np.isnan(v) else float(v) for v in row] for row in grid]
                        # Store for this week and next week (since we sample every 2 weeks)
                        for wi in [week_idx, week_idx + 1]:
                            if 0 <= wi < 52:
                                key = f'{year}_{channel}_{wi}'
                                data['week_maps'][key] = {'values': grid_list}
                
                # Get fires within ±7 days of this week
                week_start = week_date - pd.Timedelta(days=7)
                week_end = week_date + pd.Timedelta(days=7)
                fires_week = self.fires.get_fires_in_range(week_start, week_end)
                
                if len(fires_week) > 0:
                    fire_data = {
                        'lats': [float(x) for x in fires_week['lat'].values],
                        'lons': [float(x) for x in fires_week['lon'].values],
                        'sizes': [float(x) if pd.notna(x) else 1.0 for x in fires_week['size_ha'].values],
                        'texts': [f"Date: {row['fire_date'].date()}<br>Size: {row['size_ha']:.1f} ha" 
                                 for _, row in fires_week.iterrows()]
                    }
                else:
                    fire_data = {'lats': [], 'lons': [], 'sizes': [], 'texts': []}
                
                # Store for this week and next week
                for wi in [week_idx, week_idx + 1]:
                    if 0 <= wi < 52:
                        key = f'{year}_{wi}'
                        data['week_fires'][key] = fire_data
            
            # Fallback July map data
            mid_date = pd.Timestamp(f'{year}-07-15')
            for channel in self.weather.channels:
                grid = self.weather.get_frame(mid_date, channel)
                if grid is not None:
                    grid_list = [[None if np.isnan(v) else float(v) for v in row] for row in grid]
                    data['maps'][f'{year}_{channel}'] = {'values': grid_list}
        
        return json.dumps(data)


# ============================================================================
# MATPLOTLIB INTERACTIVE (ALTERNATIVE)
# ============================================================================

def matplotlib_explorer(weather_csv: str, fire_parquet: str, year: int = 2023):
    """Simple matplotlib-based interactive explorer."""
    
    if not HAS_MATPLOTLIB:
        print("Please install matplotlib")
        return
    
    weather = WeatherGridData(weather_csv)
    fires = FireEventData(fire_parquet)
    
    # Get dates for the year
    dates = weather.get_available_dates(year)
    if not dates:
        print(f"No data for {year}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot
    channel = weather.channels[0]
    date_idx = len(dates) // 2
    grid = weather.get_frame(dates[date_idx], channel)
    
    im = ax.imshow(grid, extent=[
        min(weather.unique_lons), max(weather.unique_lons),
        min(weather.unique_lats), max(weather.unique_lats)
    ], origin='lower', aspect='auto', cmap='RdYlBu_r')
    
    # Fire scatter
    fires_df = fires.get_fires_on_date(dates[date_idx], window_days=3)
    scatter = ax.scatter([], [], c='red', s=50, edgecolors='black', linewidth=1, label='Fires')
    
    plt.colorbar(im, ax=ax, label=channel)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{channel} - {dates[date_idx].date()}')
    
    # Date slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Date', 0, len(dates) - 1, valinit=date_idx, valstep=1)
    
    # Channel selector
    ax_radio = plt.axes([0.02, 0.4, 0.12, 0.3])
    radio = RadioButtons(ax_radio, weather.channels[:6])
    
    def update(val):
        idx = int(slider.val)
        date = dates[idx]
        ch = radio.value_selected
        
        grid = weather.get_frame(date, ch)
        im.set_data(grid)
        im.set_clim(weather.channel_stats[ch]['p5'], weather.channel_stats[ch]['p95'])
        
        fires_df = fires.get_fires_on_date(date, window_days=3)
        if len(fires_df) > 0:
            scatter.set_offsets(np.c_[fires_df['lon'], fires_df['lat']])
            sizes = np.log1p(fires_df['size_ha'].fillna(1)) * 10 + 20
            scatter.set_sizes(sizes)
        else:
            scatter.set_offsets(np.empty((0, 2)))
        
        ax.set_title(f'{ch} - {date.date()} ({len(fires_df)} fires)')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    radio.on_clicked(update)
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive fire-weather visualization")
    parser.add_argument("--weather", default="nasa_power_weather_old/power_daily_canada_1981_2024.csv",
                       help="Path to weather CSV")
    parser.add_argument("--fires", default="aligned_data/aligned_fires.parquet",
                       help="Path to aligned fires parquet/csv")
    parser.add_argument("--year", type=int, default=2023, help="Year to visualize")
    parser.add_argument("--output", default=None, help="Output HTML file (optional)")
    parser.add_argument("--backend", choices=['plotly', 'matplotlib'], default='plotly',
                       help="Visualization backend")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: only process selected year ±2 years")
    parser.add_argument("--years", type=str, default=None,
                       help="Specific years to process (comma-separated, e.g., '2020,2021,2022,2023')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weather):
        print(f"Weather file not found: {args.weather}")
        print("Please run download_nasa_power.py first")
        return
    
    if not os.path.exists(args.fires):
        print(f"Fire file not found: {args.fires}")
        print("Please run align_fire_weather.py first")
        return
    
    if args.backend == 'matplotlib':
        matplotlib_explorer(args.weather, args.fires, args.year)
    else:
        explorer = FireWeatherExplorer(args.weather, args.fires)
        
        # Determine which years to process
        if args.years:
            years_to_process = [int(y.strip()) for y in args.years.split(',')]
        elif args.fast:
            years_to_process = list(range(args.year - 2, args.year + 3))
        else:
            years_to_process = None  # All years
        
        if years_to_process:
            # Filter to only available years
            years_to_process = [y for y in years_to_process if y in explorer.weather.years]
            print(f"Processing years: {years_to_process}")
        
        if args.output:
            explorer.save_html(args.output, args.year, years_to_process=years_to_process)
        else:
            # Print usage for Jupyter
            print("\n" + "=" * 60)
            print("FIRE-WEATHER EXPLORER")
            print("=" * 60)
            print("\nTo use interactively in Jupyter:")
            print("  from visualize_fire_weather import FireWeatherExplorer")
            print(f"  explorer = FireWeatherExplorer('{args.weather}', '{args.fires}')")
            print(f"  explorer.show_interactive(year={args.year})")
            print("\nOr to save as HTML:")
            print("  explorer.save_html('output.html')")
            print("\nOr run with --output flag:")
            print(f"  python visualize_fire_weather.py --output visualization.html --year {args.year}")
            print("\nFor faster generation, use --fast or --years:")
            print(f"  python visualize_fire_weather.py --output viz.html --year {args.year} --fast")
            print(f"  python visualize_fire_weather.py --output viz.html --years 2020,2021,2022,2023")
            print("=" * 60)


if __name__ == "__main__":
    main()

