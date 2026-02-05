#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze option prices from CSV files to extract implied probability distributions.
Shows 2 examples with different skews (different market regimes) using heatmaps.
"""

import os
import glob
import math
import argparse
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data/opra_spy_snapshots/snapshots"
OUTPUT_DIR = "output/distributions"

# =============================================================================
# Breeden-Litzenberger Implementation
# =============================================================================

def smooth_call_prices(strikes: np.ndarray, prices: np.ndarray, lambda_reg: float = 1e-2) -> np.ndarray:
    """
    Smooth call prices using ridge regression on second differences.
    Minimizes: ||C_smooth - C||^2 + lambda * ||D2 C_smooth||^2
    """
    n = len(prices)
    if n < 5:
        return prices.copy()
    
    # Second difference matrix
    D2 = np.zeros((n-2, n))
    for i in range(n-2):
        D2[i, i:i+3] = [1, -2, 1]
    
    # Ridge solution
    I = np.eye(n)
    A = I + lambda_reg * (D2.T @ D2)
    smoothed = np.linalg.solve(A, prices)
    return smoothed

def extract_implied_density(strikes: np.ndarray, call_prices: np.ndarray, 
                            spot: float, r: float = 0.0, T: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract risk-neutral probability density from call prices using Breeden-Litzenberger.
    
    f(K) = exp(r*T) * d²C/dK²
    
    Returns:
        strikes: strike prices (uniform grid)
        density: probability density f(K)
    """
    # Sort by strike
    idx = np.argsort(strikes)
    K = strikes[idx]
    C = call_prices[idx]
    
    # Remove duplicates and ensure monotonicity (calls should decrease with strike)
    unique_mask = np.concatenate([[True], np.diff(K) > 1e-6])
    K = K[unique_mask]
    C = C[unique_mask]
    
    # Ensure calls are decreasing (enforce arbitrage bounds)
    for i in range(1, len(C)):
        C[i] = min(C[i], C[i-1])
    
    if len(K) < 5:
        return K, np.zeros_like(K)
    
    # Smooth call prices
    C_smooth = smooth_call_prices(K, C, lambda_reg=1e-2)
    
    # Interpolate to uniform grid for better derivative estimation
    K_min, K_max = K.min(), K.max()
    n_points = max(100, len(K) * 2)
    K_uniform = np.linspace(K_min, K_max, n_points)
    
    # Interpolate smoothed prices
    interp = interp1d(K, C_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
    C_uniform = interp(K_uniform)
    
    # Ensure monotonicity on uniform grid
    for i in range(1, len(C_uniform)):
        C_uniform[i] = min(C_uniform[i], C_uniform[i-1])
    
    # Compute second derivative (density)
    dK = K_uniform[1] - K_uniform[0]
    d2C = np.gradient(np.gradient(C_uniform, dK), dK)
    
    # Apply Breeden-Litzenberger formula
    if T is None:
        T = 45 / 365.0  # Default ~45 days
    density = np.exp(r * T) * d2C
    
    # Enforce non-negativity and normalize
    density = np.maximum(density, 0.0)
    total_mass = np.trapz(density, K_uniform)
    if total_mass > 0:
        density = density / total_mass
    
    return K_uniform, density

def compute_skew_metrics(strikes: np.ndarray, density: np.ndarray, spot: float) -> Dict[str, float]:
    """
    Compute skew metrics from the probability distribution.
    """
    # Mean
    mean = np.trapz(strikes * density, strikes)
    
    # Variance and std
    variance = np.trapz((strikes - mean)**2 * density, strikes)
    std = np.sqrt(variance)
    
    # Skewness (third moment)
    skewness = np.trapz(((strikes - mean) / std)**3 * density, strikes) if std > 0 else 0.0
    
    # Median (via CDF)
    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]
    
    # Skew proxy: (mean - median) / std
    skew_proxy = (mean - median) / std if std > 0 else 0.0
    
    # Left tail probability (P(S_T <= 0.95 * spot))
    threshold = 0.95 * spot
    left_tail_prob = np.trapz(density[strikes <= threshold], strikes[strikes <= threshold])
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'skew_proxy': skew_proxy,
        'median': median,
        'left_tail_prob': left_tail_prob,
        'spot': spot
    }

def compute_regime_features(strikes: np.ndarray, density: np.ndarray, spot: float) -> Dict[str, float]:
    """
    Compute full regime features for classification.
    Returns:
        - std_ret: implied std in return space
        - skew_proxy: (mean - median) / std
        - p_left_5: P(S_T / spot - 1 <= -5%)
        - p_right_5: P(S_T / spot - 1 >= +5%)
    """
    # Convert to return space
    returns = (strikes / spot) - 1.0
    
    # Mean return
    mean_ret = np.trapz(returns * density, strikes)
    
    # Std in return space
    variance_ret = np.trapz((returns - mean_ret)**2 * density, strikes)
    std_ret = np.sqrt(variance_ret)
    
    # Median (via CDF)
    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]
    mean = np.trapz(strikes * density, strikes)
    std = np.sqrt(np.trapz((strikes - mean)**2 * density, strikes))
    skew_proxy = (mean - median) / std if std > 0 else 0.0
    
    # Tail probabilities in return space
    # P(S_T / spot - 1 <= -5%)
    threshold_left = spot * 0.95  # -5% return
    p_left_5 = np.trapz(density[strikes <= threshold_left], strikes[strikes <= threshold_left])
    
    # P(S_T / spot - 1 >= +5%)
    threshold_right = spot * 1.05  # +5% return
    p_right_5 = np.trapz(density[strikes >= threshold_right], strikes[strikes >= threshold_right])
    
    return {
        'std_ret': std_ret,
        'skew_proxy': skew_proxy,
        'p_left_5': p_left_5,
        'p_right_5': p_right_5
    }

def compute_regime_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute composite regime scores.
    - stress_score: high downside tail prob + high vol + negative skew
    - calm_score: low downside tail prob + low vol
    - risk_on_score: high right-tail prob + positive skew
    """
    std_ret = features['std_ret']
    skew_proxy = features['skew_proxy']
    p_left_5 = features['p_left_5']
    p_right_5 = features['p_right_5']
    
    # Normalize components to [0, 1] range for scoring
    # Assume reasonable ranges: std_ret in [0, 0.5], tail probs in [0, 1]
    std_norm = min(std_ret / 0.5, 1.0)  # Cap at 50% vol
    p_left_norm = min(p_left_5, 1.0)
    p_right_norm = min(p_right_5, 1.0)
    skew_norm = (skew_proxy + 0.5) / 1.0  # Map [-0.5, 0.5] to [0, 1], negative skew -> higher stress
    
    # Stress score: high downside prob + high vol + negative skew
    stress_score = (p_left_norm * 0.5 + std_norm * 0.3 + (1 - skew_norm) * 0.2)
    
    # Calm score: low downside prob + low vol
    calm_score = ((1 - p_left_norm) * 0.6 + (1 - std_norm) * 0.4)
    
    # Risk-on score: high right-tail prob + positive skew
    risk_on_score = (p_right_norm * 0.6 + max(0, skew_proxy) * 0.4)
    
    return {
        'stress_score': stress_score,
        'calm_score': calm_score,
        'risk_on_score': risk_on_score
    }

# =============================================================================
# Data Loading
# =============================================================================

def load_snapshot_file(filepath: str) -> pd.DataFrame:
    """Load a single snapshot CSV file."""
    return pd.read_csv(filepath, compression='gzip')

def find_snapshot_by_date(data_dir: str, target_date: str) -> Optional[Dict]:
    """
    Find and load a snapshot for a specific date.
    Returns snapshot dict or None if not found.
    """
    pattern = os.path.join(data_dir, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    
    target_dt = pd.to_datetime(target_date)
    best_file = None
    best_diff = None
    
    for filepath in files:
        try:
            basename = os.path.basename(filepath)
            date_str = basename.split('_')[0]
            file_date = pd.to_datetime(date_str)
            
            diff = abs((file_date - target_dt).days)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_file = filepath
        except Exception:
            continue
    
    if best_file is None or (best_diff is not None and best_diff > 7):
        return None
    
    try:
        df = load_snapshot_file(best_file)
        if df.empty:
            return None
        
        spot = df['underlying_spot'].iloc[0] if 'underlying_spot' in df.columns else None
        if spot is None or spot <= 0:
            return None
        
        calls = df[(df['opt_type'] == 'call') & 
                  (df['mid'].notna()) & 
                  (df['mid'] > 0) &
                  (df['strike'].notna())].copy()
        
        if len(calls) < 20:
            return None
        
        strikes = calls['strike'].values
        prices = calls['mid'].values
        
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        prices = prices[idx]
        
        unique_mask = np.concatenate([[True], np.diff(strikes) > 1e-3])
        strikes = strikes[unique_mask]
        prices = prices[unique_mask]
        
        if len(strikes) < 10:
            return None
        
        K, density = extract_implied_density(strikes, prices, spot)
        
        if np.sum(density) < 0.1:
            return None
        
        metrics = compute_skew_metrics(K, density, spot)
        
        return {
            'filepath': best_file,
            'df': df,
            'strikes': K,
            'density': density,
            'metrics': metrics,
            'date': df['date'].iloc[0] if 'date' in df.columns else os.path.basename(best_file)
        }
    except Exception as e:
        print(f"Error processing {best_file}: {e}")
        return None

def analyze_all_snapshots(data_dir: str) -> pd.DataFrame:
    """
    Analyze all snapshot files and compute regime features and scores.
    Returns DataFrame with all snapshots and their metrics.
    """
    pattern = os.path.join(data_dir, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    
    if len(files) < 2:
        raise ValueError(f"Need at least 2 snapshot files. Found {len(files)} in {data_dir}")
    
    print(f"Found {len(files)} snapshot files. Analyzing all...")
    
    results = []
    
    for filepath in files:
        try:
            df = load_snapshot_file(filepath)
            if df.empty:
                continue
            
            # Get spot price
            spot = df['underlying_spot'].iloc[0] if 'underlying_spot' in df.columns else None
            if spot is None or spot <= 0:
                continue
            
            # Filter to calls only and valid prices
            calls = df[(df['opt_type'] == 'call') & 
                      (df['mid'].notna()) & 
                      (df['mid'] > 0) &
                      (df['strike'].notna())].copy()
            
            if len(calls) < 20:  # Need enough strikes
                continue
            
            # Extract density
            strikes_raw = calls['strike'].values
            prices = calls['mid'].values
            
            # Sort and filter
            idx = np.argsort(strikes_raw)
            strikes_raw = strikes_raw[idx]
            prices = prices[idx]
            
            # Remove duplicates
            unique_mask = np.concatenate([[True], np.diff(strikes_raw) > 1e-3])
            strikes_raw = strikes_raw[unique_mask]
            prices = prices[unique_mask]
            
            if len(strikes_raw) < 10:
                continue
            
            # Extract implied density
            K, density = extract_implied_density(strikes_raw, prices, spot)
            
            if np.sum(density) < 0.1:  # Invalid density
                continue
            
            # Compute full regime features
            features = compute_regime_features(K, density, spot)
            
            # Compute regime scores
            scores = compute_regime_scores(features)
            
            # Get date
            date_str = df['date'].iloc[0] if 'date' in df.columns else os.path.basename(filepath).split('_')[0]
            
            # Store full snapshot data
            results.append({
                'filepath': filepath,
                'date': date_str,
                'df': df,
                'strikes': K,
                'density': density,
                'spot': spot,
                'std_ret': features['std_ret'],
                'skew_proxy': features['skew_proxy'],
                'p_left_5': features['p_left_5'],
                'p_right_5': features['p_right_5'],
                'stress_score': scores['stress_score'],
                'calm_score': scores['calm_score'],
                'risk_on_score': scores['risk_on_score']
            })
            
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {e}")
            continue
    
    if len(results) < 2:
        raise ValueError(f"Could not process enough snapshots. Got {len(results)} valid snapshots.")
    
    print(f"Successfully processed {len(results)} snapshots.")
    
    # Convert to DataFrame for easier manipulation
    df_results = pd.DataFrame([{
        'filepath': r['filepath'],
        'date': r['date'],
        'std_ret': r['std_ret'],
        'skew_proxy': r['skew_proxy'],
        'p_left_5': r['p_left_5'],
        'p_right_5': r['p_right_5'],
        'stress_score': r['stress_score'],
        'calm_score': r['calm_score'],
        'risk_on_score': r['risk_on_score']
    } for r in results])
    
    # Store full snapshot dicts separately for later use
    df_results['_snapshot_dict'] = results
    
    return df_results

def select_extreme_snapshots(df_results: pd.DataFrame, top_n: int = 2) -> Tuple[list, list]:
    """
    Select top N most stressed and top N calmest snapshots.
    Returns: (stress_snapshots, calm_snapshots)
    """
    # Sort by stress_score (descending for stress, ascending for calm)
    df_stress = df_results.nlargest(top_n, 'stress_score')
    df_calm = df_results.nsmallest(top_n, 'stress_score')  # Low stress = calm
    
    stress_snapshots = df_stress['_snapshot_dict'].tolist()
    calm_snapshots = df_calm['_snapshot_dict'].tolist()
    
    return stress_snapshots, calm_snapshots

# =============================================================================
# Visualization
# =============================================================================

def create_distribution_plot(snapshot: Dict, title: str, ax=None):
    """
    Create a clean, readable visualization of the implied probability distribution.
    
    What this shows:
    - The probability distribution of where SPY price might be at expiration
    - Derived from option prices using Breeden-Litzenberger
    - Different shapes indicate different market expectations (regimes)
    """
    strikes = snapshot['strikes']
    density = snapshot['density']
    metrics = snapshot['metrics']
    spot = metrics['spot']
    mean = metrics['mean']
    median = metrics['median']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Main density curve - filled area with gradient
    ax.fill_between(strikes, 0, density, alpha=0.6, color='steelblue', 
                    label='Implied Probability Density')
    ax.plot(strikes, density, color='darkblue', linewidth=2.5, alpha=0.9)
    
    # Add reference lines
    ax.axvline(spot, color='red', linestyle='--', linewidth=2.5, 
               label=f'Current Spot: ${spot:.2f}', alpha=0.8)
    ax.axvline(mean, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Expected Mean: ${mean:.2f}', alpha=0.8)
    ax.axvline(median, color='green', linestyle='--', linewidth=2.5, 
               label=f'Median: ${median:.2f}', alpha=0.8)
    
    # Shade left tail (downside risk area)
    left_tail_threshold = 0.95 * spot
    left_mask = strikes <= left_tail_threshold
    if np.any(left_mask):
        ax.fill_between(strikes[left_mask], 0, density[left_mask], 
                       alpha=0.4, color='crimson', label='Left Tail Risk')
    
    # Labels and formatting
    ax.set_xlabel('SPY Price at Expiration ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Add statistics in a clean box
    skew_proxy = metrics['skew_proxy']
    skew_label = "Left-Skewed (Stress)" if skew_proxy < -0.1 else "Right-Skewed (Calm)" if skew_proxy > 0.1 else "Symmetric"
    
    stats_text = (
        f"Regime: {skew_label}\n"
        f"Skew Proxy: {skew_proxy:.3f}\n"
        f"Std Deviation: ${metrics['std']:.2f}\n"
        f"Downside Risk (≤95% spot): {metrics['left_tail_prob']:.1%}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=1.5))
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    return ax

def create_comparison_plot(snapshot1: Dict, snapshot2: Dict, 
                           label1: str, label2: str):
    """
    Create a side-by-side comparison of two distributions.
    Shows how different market regimes have different probability distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Implied Probability Distributions: Market Regime Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Get common strike range for consistent visualization
    strikes1 = snapshot1['strikes']
    strikes2 = snapshot2['strikes']
    all_strikes = np.concatenate([strikes1, strikes2])
    strike_min = all_strikes.min() * 0.98
    strike_max = all_strikes.max() * 1.02
    
    # Create both plots
    create_distribution_plot(snapshot1, label1, ax=axes[0])
    create_distribution_plot(snapshot2, label2, ax=axes[1])
    
    # Set consistent x-axis limits for comparison
    axes[0].set_xlim(strike_min, strike_max)
    axes[1].set_xlim(strike_min, strike_max)
    
    # Get consistent y-axis limits
    max_density = max(snapshot1['density'].max(), snapshot2['density'].max())
    axes[0].set_ylim(0, max_density * 1.15)
    axes[1].set_ylim(0, max_density * 1.15)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze options-implied probability distributions and compare market regimes'
    )
    parser.add_argument('--date1', type=str, default=None,
                       help='First date for comparison (YYYY-MM-DD). If not specified, auto-selects extremes.')
    parser.add_argument('--date2', type=str, default=None,
                       help='Second date for comparison (YYYY-MM-DD). If not specified, auto-selects extremes.')
    parser.add_argument('--top_n', type=int, default=2,
                       help='Number of extreme snapshots to select for each regime (default: 2)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please run data_downloader.py first to download option data.")
        return
    
    # Find snapshots
    if args.date1 and args.date2:
        # Use specified dates
        print(f"Loading snapshots for specified dates...")
        print(f"  Date 1: {args.date1}")
        print(f"  Date 2: {args.date2}")
        
        stress_snapshot = find_snapshot_by_date(DATA_DIR, args.date1)
        calm_snapshot = find_snapshot_by_date(DATA_DIR, args.date2)
        
        if stress_snapshot is None:
            print(f"Error: Could not find snapshot for date {args.date1}")
            return
        if calm_snapshot is None:
            print(f"Error: Could not find snapshot for date {args.date2}")
            return
        
        # Compute features for specified dates
        features1 = compute_regime_features(stress_snapshot['strikes'], stress_snapshot['density'], stress_snapshot['metrics']['spot'])
        features2 = compute_regime_features(calm_snapshot['strikes'], calm_snapshot['density'], calm_snapshot['metrics']['spot'])
        
        # Determine which is stress/calm based on stress_score
        scores1 = compute_regime_scores(features1)
        scores2 = compute_regime_scores(features2)
        
        if scores1['stress_score'] < scores2['stress_score']:
            # Swap them
            stress_snapshot, calm_snapshot = calm_snapshot, stress_snapshot
            scores1, scores2 = scores2, scores1
        
        print(f"\nLoaded snapshots:")
        print(f"  Date 1 ({args.date1}): stress_score={scores1['stress_score']:.4f}, p_left_5={features1['p_left_5']:.4f}")
        print(f"  Date 2 ({args.date2}): stress_score={scores2['stress_score']:.4f}, p_left_5={features2['p_left_5']:.4f}")
        
        # Convert to list format for compatibility
        stress_snapshots = [stress_snapshot]
        calm_snapshots = [calm_snapshot]
    else:
        # Analyze all snapshots and select extremes
        print("Analyzing all snapshots...")
        df_results = analyze_all_snapshots(DATA_DIR)
        
        # Print ranked table
        print("\n" + "="*100)
        print("RANKED SNAPSHOTS BY STRESS SCORE")
        print("="*100)
        df_display = df_results[['date', 'stress_score', 'p_left_5', 'std_ret', 'skew_proxy']].copy()
        df_display = df_display.sort_values('stress_score', ascending=False)
        df_display['stress_score'] = df_display['stress_score'].map('{:.4f}'.format)
        df_display['p_left_5'] = df_display['p_left_5'].map('{:.4f}'.format)
        df_display['std_ret'] = df_display['std_ret'].map('{:.4f}'.format)
        df_display['skew_proxy'] = df_display['skew_proxy'].map('{:.4f}'.format)
        print(df_display.to_string(index=False))
        print("="*100 + "\n")
        
        # Select extreme snapshots
        stress_snapshots, calm_snapshots = select_extreme_snapshots(df_results, top_n=args.top_n)
        
        print(f"Selected top {args.top_n} most stressed snapshots:")
        for i, snap in enumerate(stress_snapshots, 1):
            print(f"  {i}. {snap['date']}: stress_score={snap['stress_score']:.4f}, p_left_5={snap['p_left_5']:.4f}, std_ret={snap['std_ret']:.4f}")
        
        print(f"\nSelected top {args.top_n} calmest snapshots:")
        for i, snap in enumerate(calm_snapshots, 1):
            print(f"  {i}. {snap['date']}: stress_score={snap['stress_score']:.4f}, p_left_5={snap['p_left_5']:.4f}, std_ret={snap['std_ret']:.4f}")
        
        # Use the most extreme of each
        stress_snapshot = stress_snapshots[0]
        calm_snapshot = calm_snapshots[0]
    
    # Create visualizations
    print("\nCreating visualizations...")
    print("\nWhat these graphs show:")
    print("  - The implied probability distribution of where SPY price might be at expiration")
    print("  - Derived from option prices using Breeden-Litzenberger formula")
    print("  - Different shapes indicate different market expectations (regimes)")
    print("  - Stress regime: high downside risk, high volatility, negative skew")
    print("  - Calm regime: low downside risk, low volatility\n")
    
    # Compute metrics for display (compatible with old format)
    stress_metrics = compute_skew_metrics(stress_snapshot['strikes'], stress_snapshot['density'], stress_snapshot['spot'])
    calm_metrics = compute_skew_metrics(calm_snapshot['strikes'], calm_snapshot['density'], calm_snapshot['spot'])
    
    # Create snapshot dicts in expected format
    stress_snap_dict = {
        'filepath': stress_snapshot['filepath'],
        'df': stress_snapshot['df'],
        'strikes': stress_snapshot['strikes'],
        'density': stress_snapshot['density'],
        'metrics': stress_metrics,
        'date': stress_snapshot['date']
    }
    
    calm_snap_dict = {
        'filepath': calm_snapshot['filepath'],
        'df': calm_snapshot['df'],
        'strikes': calm_snapshot['strikes'],
        'density': calm_snapshot['density'],
        'metrics': calm_metrics,
        'date': calm_snapshot['date']
    }
    
    # Individual plots
    fig1 = plt.figure(figsize=(14, 8))
    create_distribution_plot(
        stress_snap_dict,
        f"Stress Regime - {stress_snap_dict['date']}\n(High Downside Risk, High Volatility)",
        ax=plt.gca()
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "stress_regime_distribution.png"), 
                bbox_inches='tight', facecolor='white', dpi=150)
    print(f"Saved: {OUTPUT_DIR}/stress_regime_distribution.png")
    
    fig2 = plt.figure(figsize=(14, 8))
    create_distribution_plot(
        calm_snap_dict,
        f"Calm Regime - {calm_snap_dict['date']}\n(Low Downside Risk, Low Volatility)",
        ax=plt.gca()
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "calm_regime_distribution.png"), 
                bbox_inches='tight', facecolor='white', dpi=150)
    print(f"Saved: {OUTPUT_DIR}/calm_regime_distribution.png")
    
    # Comparison plot
    fig3 = create_comparison_plot(
        stress_snap_dict,
        calm_snap_dict,
        f"Stress Regime - {stress_snap_dict['date']}\n(Stress Score: {stress_snapshot['stress_score']:.3f}, P(≤-5%): {stress_snapshot['p_left_5']:.2%})",
        f"Calm Regime - {calm_snap_dict['date']}\n(Stress Score: {calm_snapshot['stress_score']:.3f}, P(≤-5%): {calm_snapshot['p_left_5']:.2%})"
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "regime_comparison.png"), 
                bbox_inches='tight', facecolor='white', dpi=150)
    print(f"Saved: {OUTPUT_DIR}/regime_comparison.png")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nStress Regime ({stress_snap_dict['date']}):")
    print(f"  stress_score        : {stress_snapshot['stress_score']:10.4f}")
    print(f"  std_ret             : {stress_snapshot['std_ret']:10.4f}")
    print(f"  p_left_5            : {stress_snapshot['p_left_5']:10.4f}")
    print(f"  p_right_5           : {stress_snapshot['p_right_5']:10.4f}")
    print(f"  skew_proxy          : {stress_snapshot['skew_proxy']:10.4f}")
    print(f"  spot                : {stress_metrics['spot']:10.2f}")
    
    print(f"\nCalm Regime ({calm_snap_dict['date']}):")
    print(f"  stress_score        : {calm_snapshot['stress_score']:10.4f}")
    print(f"  std_ret             : {calm_snapshot['std_ret']:10.4f}")
    print(f"  p_left_5            : {calm_snapshot['p_left_5']:10.4f}")
    print(f"  p_right_5           : {calm_snapshot['p_right_5']:10.4f}")
    print(f"  skew_proxy          : {calm_snapshot['skew_proxy']:10.4f}")
    print(f"  spot                : {calm_metrics['spot']:10.2f}")
    
    print("\n" + "="*80)
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    main()

