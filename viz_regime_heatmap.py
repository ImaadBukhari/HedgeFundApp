#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for options-implied probability distributions.
Overlays SPY historical prices with forward-looking probability heatmaps.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import yfinance as yf
from typing import Tuple, Dict, Optional

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data/opra_spy_snapshots/snapshots"
OUTPUT_DIR = "output/distributions"

# Return grid for heatmaps
R_MIN = -0.25  # -25% return
R_MAX = 0.25   # +25% return
N_RETURN_BINS = 100

# Default lookback for historical price chart
DEFAULT_LOOKBACK_YEARS = 2

# =============================================================================
# Breeden-Litzenberger Implementation
# =============================================================================

def smooth_call_prices(strikes: np.ndarray, prices: np.ndarray, lambda_reg: float = 1e-2) -> np.ndarray:
    """Smooth call prices using ridge regression on second differences."""
    n = len(prices)
    if n < 5:
        return prices.copy()
    
    D2 = np.zeros((n-2, n))
    for i in range(n-2):
        D2[i, i:i+3] = [1, -2, 1]
    
    I = np.eye(n)
    A = I + lambda_reg * (D2.T @ D2)
    smoothed = np.linalg.solve(A, prices)
    return smoothed

def extract_implied_density(strikes: np.ndarray, call_prices: np.ndarray, 
                            spot: float, r: float = 0.0, T: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract risk-neutral probability density from call prices using Breeden-Litzenberger.
    Returns: (strikes, density)
    """
    # Sort by strike
    idx = np.argsort(strikes)
    K = strikes[idx]
    C = call_prices[idx]
    
    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(K) > 1e-6])
    K = K[unique_mask]
    C = C[unique_mask]
    
    # Enforce monotonicity (calls must decrease with strike)
    for i in range(1, len(C)):
        C[i] = min(C[i], C[i-1])
    
    if len(K) < 5:
        return K, np.zeros_like(K)
    
    # Smooth
    C_smooth = smooth_call_prices(K, C, lambda_reg=1e-2)
    
    # Interpolate to uniform grid
    K_min, K_max = K.min(), K.max()
    n_points = max(100, len(K) * 2)
    K_uniform = np.linspace(K_min, K_max, n_points)
    
    interp = interp1d(K, C_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
    C_uniform = interp(K_uniform)
    
    # Ensure monotonicity
    for i in range(1, len(C_uniform)):
        C_uniform[i] = min(C_uniform[i], C_uniform[i-1])
    
    # Second derivative (density)
    dK = K_uniform[1] - K_uniform[0]
    d2C = np.gradient(np.gradient(C_uniform, dK), dK)
    
    # Breeden-Litzenberger
    if T is None:
        T = 45 / 365.0
    density = np.exp(r * T) * d2C
    density = np.maximum(density, 0.0)
    
    # Normalize
    total_mass = np.trapz(density, K_uniform)
    if total_mass > 0:
        density = density / total_mass
    
    return K_uniform, density

def compute_metrics(strikes: np.ndarray, density: np.ndarray, spot: float) -> Dict[str, float]:
    """Compute distribution metrics."""
    mean = np.trapz(strikes * density, strikes)
    variance = np.trapz((strikes - mean)**2 * density, strikes)
    std = np.sqrt(variance)
    
    # Median via CDF
    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]
    
    # Left tail probability (P(S_T <= 0.95 * spot))
    threshold = 0.95 * spot
    left_tail_prob = np.trapz(density[strikes <= threshold], strikes[strikes <= threshold])
    
    # Skew proxy
    skew_proxy = (mean - median) / std if std > 0 else 0.0
    
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'left_tail_prob': left_tail_prob,
        'skew_proxy': skew_proxy
    }

# =============================================================================
# Data Loading
# =============================================================================

def find_snapshot_file(target_date: str) -> Optional[str]:
    """Find the closest snapshot file for a given date. Prefers 16:30 snapshots."""
    pattern = os.path.join(DATA_DIR, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        return None
    
    target_dt = pd.to_datetime(target_date)
    candidates = []
    
    for filepath in files:
        try:
            # Parse from filename: YYYY-MM-DD_exp... or YYYY-MM-DD_exp..._t...
            basename = os.path.basename(filepath)
            date_str = basename.split('_')[0]
            file_date = pd.to_datetime(date_str)
            
            diff = abs((file_date - target_dt).days)
            # Check if it's a 16:30 snapshot (market close)
            is_close = 't1630' in basename or '16:30' in basename
            
            candidates.append((diff, is_close, filepath))
        except Exception:
            continue
    
    if not candidates:
        return None
    
    # Sort by: 1) date difference, 2) prefer 16:30 snapshots
    candidates.sort(key=lambda x: (x[0], not x[1]))
    
    return candidates[0][2]

def load_snapshot(date: str) -> Tuple[pd.DataFrame, str]:
    """
    Load snapshot for a given date.
    Returns: (dataframe, filepath)
    """
    filepath = find_snapshot_file(date)
    if filepath is None:
        raise FileNotFoundError(f"No snapshot file found for date {date}")
    
    df = pd.read_csv(filepath, compression='gzip')
    
    # Get date from dataframe or filename
    if 'date' in df.columns:
        file_date = df['date'].iloc[0]
    else:
        basename = os.path.basename(filepath)
        file_date = basename.split('_')[0]
    
    print(f"Loaded snapshot file: {os.path.basename(filepath)}")
    print(f"Snapshot date: {file_date}")
    
    return df, filepath

def extract_density(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, float, Dict]:
    """
    Extract implied density from snapshot dataframe.
    Returns: (return_grid, density, dte, spot, metrics)
    """
    # Filter to calls only, valid prices
    calls = df[(df['opt_type'] == 'call') & 
              (df['mid'].notna()) & 
              (df['mid'] > 0) &
              (df['strike'].notna()) &
              np.isfinite(df['strike']) &
              np.isfinite(df['mid'])].copy()
    
    if len(calls) < 20:
        raise ValueError(f"Insufficient call contracts: {len(calls)} < 20 required")
    
    # Get spot
    spot = float(calls['underlying_spot'].iloc[0])
    
    # Get expiration and DTE
    if 'expiration' in df.columns and df['expiration'].notna().any():
        expiration = pd.to_datetime(df['expiration'].iloc[0])
        snapshot_date = pd.to_datetime(df['date'].iloc[0])
        dte = (expiration - snapshot_date).days
    elif 'dte' in df.columns and df['dte'].notna().any():
        dte = int(df['dte'].iloc[0])
    else:
        dte = 45  # Default
        print(f"Warning: No expiration found, assuming DTE={dte}")
    
    T = dte / 365.0
    
    # Get strikes and prices
    strikes = calls['strike'].values
    prices = calls['mid'].values
    
    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(np.sort(strikes)) > 1e-3])
    strikes = np.sort(strikes)[unique_mask]
    prices = prices[np.argsort(calls['strike'].values)][unique_mask]
    
    if len(strikes) < 10:
        raise ValueError(f"Too few unique strikes after deduplication: {len(strikes)}")
    
    # Extract density
    K, density = extract_implied_density(strikes, prices, spot, r=0.0, T=T)
    
    if np.sum(density) < 0.1:
        raise ValueError("Invalid density extracted (sum too small)")
    
    # Convert to return space
    r_grid = np.linspace(R_MIN, R_MAX, N_RETURN_BINS)
    K_from_r = spot * (1 + r_grid)
    
    # Interpolate density to return grid
    interp = interp1d(K, density, kind='linear', bounds_error=False, fill_value=0.0)
    density_r = interp(K_from_r)
    density_r = np.maximum(density_r, 0.0)
    
    # Normalize
    if density_r.sum() > 0:
        density_r = density_r / density_r.sum()
    
    # Compute metrics
    metrics = compute_metrics(K, density, spot)
    
    print(f"Spot: ${spot:.2f}, DTE: {dte} days")
    print(f"Mean: ${metrics['mean']:.2f}, Median: ${metrics['median']:.2f}, Std: ${metrics['std']:.2f}")
    print(f"Skew proxy: {metrics['skew_proxy']:.4f}, Left tail prob: {metrics['left_tail_prob']:.2%}")
    
    return r_grid, density_r, dte, spot, metrics

# =============================================================================
# Visualization
# =============================================================================

def plot_single_report(date: str, lookback_years: float = DEFAULT_LOOKBACK_YEARS):
    """
    Create single-date report with SPY history and forward probability heatmap.
    """
    # Load snapshot
    df, filepath = load_snapshot(date)
    r_grid, density, dte, spot, metrics = extract_density(df)
    
    snapshot_dt = pd.to_datetime(date)
    
    # Load SPY historical prices
    start_date = snapshot_dt - pd.DateOffset(years=lookback_years)
    end_date = snapshot_dt + pd.Timedelta(days=1)
    
    print(f"\nLoading SPY history from {start_date.date()} to {snapshot_dt.date()}...")
    spy = yf.download("SPY", start=start_date.strftime("%Y-%m-%d"),
                     end=end_date.strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    
    if spy.empty or 'Close' not in spy.columns:
        raise ValueError(f"No SPY price data found for date range")
    
    spy_prices = spy['Close']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1.2])
    
    # Top panel: SPY historical price
    ax1.plot(spy_prices.index, spy_prices.values, linewidth=2, color='#2E86AB', label='SPY Adjusted Close')
    ax1.axvline(snapshot_dt, color='red', linestyle='--', linewidth=2.5, 
               label=f'Snapshot Date: {date}', alpha=0.8)
    ax1.axhline(spot, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Snapshot Spot: ${spot:.2f}')
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'SPY Historical Price (Last {lookback_years} Years)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    # Bottom panel: Forward probability heatmap
    # Create matrix: each row is the same density (replicated across time)
    dte_days = int(dte)
    heatmap_data = np.tile(density, (dte_days + 1, 1))
    
    # Optional: slight Gaussian blur along y-axis for aesthetics
    if dte_days > 5:
        heatmap_data = gaussian_filter1d(heatmap_data, sigma=0.5, axis=0)
    
    # Use log scale for better visibility of tails
    heatmap_data_log = np.log10(heatmap_data + 1e-12)
    
    im = ax2.imshow(heatmap_data_log, aspect='auto', origin='lower',
                    extent=[R_MIN*100, R_MAX*100, 0, dte_days],
                    cmap='viridis', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label('log₁₀(Probability Density)', rotation=270, labelpad=20, fontsize=11)
    
    # Reference lines
    ax2.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8, label='0% Return')
    ax2.axvline(-5, color='red', linestyle='--', linewidth=2, alpha=0.8, label='-5% Return (Downside)')
    
    ax2.set_xlabel('Return at Expiry (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Days to Expiry', fontsize=12, fontweight='bold')
    ax2.set_title(f'Options-Implied Probability Distribution (Forward from {date})\n'
                 f'Implied distribution at expiry (replicated across horizon for visualization)',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.2, color='white')
    
    # Add text box with metrics
    stats_text = (
        f"Mean Return: {((metrics['mean']/spot - 1)*100):.2f}%\n"
        f"Std Dev: {((metrics['std']/spot)*100):.2f}%\n"
        f"Skew: {metrics['skew_proxy']:.3f}\n"
        f"P(≤-5%): {metrics['left_tail_prob']:.1%}"
    )
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"report_{date}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    return fig

def plot_time_return_heatmap(n_samples: Optional[int] = None):
    """
    Create multi-date heatmap showing probability distributions across time.
    Deduplicates by date, preferring 16:30 snapshots.
    """
    pattern = os.path.join(DATA_DIR, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {DATA_DIR}")
    
    print(f"Found {len(files)} snapshot files, deduplicating by date...")
    
    # Group files by date, preferring 16:30 snapshots
    date_to_file = {}
    for filepath in files:
        try:
            basename = os.path.basename(filepath)
            date_str = basename.split('_')[0]
            date_key = pd.to_datetime(date_str)
            
            # Prefer 16:30 snapshots
            is_close = 't1630' in basename or '16:30' in basename
            
            if date_key not in date_to_file:
                date_to_file[date_key] = (filepath, is_close)
            else:
                # Replace if this is a 16:30 snapshot and current isn't
                current_file, current_is_close = date_to_file[date_key]
                if is_close and not current_is_close:
                    date_to_file[date_key] = (filepath, is_close)
        except Exception:
            continue
    
    # Sort by date
    sorted_dates = sorted(date_to_file.keys())
    
    if n_samples is not None:
        sorted_dates = sorted_dates[:n_samples]
    
    print(f"Processing {len(sorted_dates)} unique snapshot dates...")
    
    dates_list = []
    density_matrix = []
    
    for date_key in sorted_dates:
        filepath, _ = date_to_file[date_key]
        try:
            df = pd.read_csv(filepath, compression='gzip')
            
            # Extract density
            r_grid, density, dte, spot, metrics = extract_density(df)
            
            dates_list.append(date_key)
            density_matrix.append(density)
            
        except Exception as e:
            print(f"Skipping {os.path.basename(filepath)}: {e}")
            continue
    
    if len(density_matrix) < 2:
        raise ValueError(f"Need at least 2 valid snapshots. Got {len(density_matrix)}")
    
    # Convert to array
    density_array = np.array(density_matrix)
    dates_array = np.array(dates_list)
    
    # Sort by date
    sort_idx = np.argsort(dates_array)
    density_array = density_array[sort_idx]
    dates_array = dates_array[sort_idx]
    
    # Log scale for visibility
    density_log = np.log10(density_array + 1e-12)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(dates_array) * 0.3)))
    
    # Create heatmap
    im = ax.imshow(density_log, aspect='auto', origin='lower',
                   extent=[R_MIN*100, R_MAX*100, 0, len(dates_array)],
                   cmap='viridis', interpolation='bilinear')
    
    # Set y-axis labels to dates
    n_dates = len(dates_array)
    if n_dates <= 20:
        step = 1
    elif n_dates <= 50:
        step = max(1, n_dates // 15)
    else:
        step = max(1, n_dates // 20)
    
    yticks = list(range(0, n_dates, step))
    if n_dates - 1 not in yticks:
        yticks.append(n_dates - 1)
    ytick_labels = [dates_array[i].strftime('%Y-%m-%d') for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    
    # Reference lines
    ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(-5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('log₁₀(Probability Density)', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_xlabel('Return at Expiry (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Snapshot Date', fontsize=12, fontweight='bold')
    ax.set_title('Options-Implied Probability Distributions Over Time\n'
                f'({len(dates_array)} snapshots from {dates_array[0].date()} to {dates_array[-1].date()})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "time_return_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    return fig

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize options-implied probability distributions with SPY historical prices'
    )
    parser.add_argument('--date', type=str, help='Snapshot date (YYYY-MM-DD) for single-date report')
    parser.add_argument('--lookback_years', type=float, default=DEFAULT_LOOKBACK_YEARS,
                       help=f'Years of SPY history to show (default: {DEFAULT_LOOKBACK_YEARS})')
    parser.add_argument('--all', action='store_true', help='Generate multi-date heatmap')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Limit number of snapshots for --all (default: all)')
    
    args = parser.parse_args()
    
    if args.all:
        print("Generating multi-date heatmap...")
        plot_time_return_heatmap(n_samples=args.n_samples)
        print("\nDone!")
    elif args.date:
        print(f"Generating single-date report for {args.date}...")
        plot_single_report(args.date, lookback_years=args.lookback_years)
        print("\nDone!")
    else:
        parser.print_help()
        print("\nError: Must specify either --date or --all")
        sys.exit(1)

if __name__ == "__main__":
    main()

