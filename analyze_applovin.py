#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze RKLB (Rocket Lab USA) current options-implied probability distribution.
Uses yfinance to get current options prices and applies Breeden-Litzenberger method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing import Tuple, Dict
from datetime import datetime

# =============================================================================
# Breeden-Litzenberger Implementation (same as analyze_distributions.py)
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
                            spot: float, r: float = 0.0, T: float = None,
                            max_violation: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Extract risk-neutral probability density from call prices using Breeden-Litzenberger.
    
    f(K) = exp(r*T) * d²C/dK²
    
    Returns:
        strikes: strike prices (uniform grid)
        density: probability density f(K)
        quality_info: dict with quality metrics
    """
    quality_info = {'valid': True, 'warnings': []}
    
    # Sort by strike
    idx = np.argsort(strikes)
    K = strikes[idx]
    C = call_prices[idx]
    
    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(K) > 1e-6])
    K = K[unique_mask]
    C = C[unique_mask]
    
    # Clip tiny violations instead of aggressive flattening
    # If violation is large, drop that strike
    valid_mask = np.ones(len(K), dtype=bool)
    for i in range(1, len(K)):
        if C[i] > C[i-1]:
            violation = C[i] - C[i-1]
            if violation > max_violation:
                # Large violation: drop this strike
                valid_mask[i] = False
                quality_info['warnings'].append(f"Dropped strike {K[i]:.2f} due to large monotonicity violation")
            else:
                # Small violation: clip it
                C[i] = C[i-1] - 1e-6
    
    K = K[valid_mask]
    C = C[valid_mask]
    
    if len(K) < 5:
        quality_info['valid'] = False
        quality_info['warnings'].append("Too few strikes after filtering")
        return K, np.zeros_like(K), quality_info
    
    # Smooth call prices
    C_smooth = smooth_call_prices(K, C, lambda_reg=1e-2)
    
    # Interpolate to uniform grid for better derivative estimation
    K_min, K_max = K.min(), K.max()
    n_points = max(100, len(K) * 2)
    K_uniform = np.linspace(K_min, K_max, n_points)
    
    # Use linear interpolation instead of cubic
    interp = interp1d(K, C_smooth, kind='linear', bounds_error=False, fill_value='extrapolate')
    C_uniform = interp(K_uniform)
    
    # Clip tiny violations on uniform grid
    for i in range(1, len(C_uniform)):
        if C_uniform[i] > C_uniform[i-1]:
            C_uniform[i] = C_uniform[i-1] - 1e-6
    
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
    
    # Quality checks
    quality_info.update(check_density_quality(K_uniform, density, spot))
    
    return K_uniform, density, quality_info

def check_density_quality(strikes: np.ndarray, density: np.ndarray, spot: float) -> Dict[str, any]:
    """
    Check density quality and reject if problematic.
    Returns dict with 'valid', 'warnings', and quality metrics.
    """
    quality = {'valid': True, 'warnings': [], 'metrics': {}}
    
    # Check 1: Mass concentration at left boundary
    # If >60% of mass is in the lowest 10% of strikes → reject
    strike_range = strikes.max() - strikes.min()
    left_boundary = strikes.min() + 0.1 * strike_range
    left_mask = strikes <= left_boundary
    left_mass = np.trapz(density[left_mask], strikes[left_mask])
    
    quality['metrics']['left_mass_pct'] = left_mass * 100
    
    if left_mass > 0.6:
        quality['valid'] = False
        quality['warnings'].append(f"Too much mass ({left_mass:.1%}) in lowest 10% of strikes")
    
    # Check 2: Too many local maxima (spikes)
    # Count local maxima
    peaks, _ = find_peaks(density, height=np.max(density) * 0.1)  # Peaks > 10% of max
    n_peaks = len(peaks)
    n_strikes = len(strikes)
    peak_ratio = n_peaks / n_strikes
    
    quality['metrics']['n_peaks'] = n_peaks
    quality['metrics']['peak_ratio'] = peak_ratio
    
    if peak_ratio > 0.3:  # More than 30% of points are peaks
        quality['valid'] = False
        quality['warnings'].append(f"Too many local maxima ({n_peaks} peaks, {peak_ratio:.1%} of strikes)")
    
    # Check 3: Density sum should be close to 1
    total_mass = np.trapz(density, strikes)
    quality['metrics']['total_mass'] = total_mass
    
    if total_mass < 0.5 or total_mass > 1.5:
        quality['warnings'].append(f"Density mass ({total_mass:.3f}) is far from 1.0")
    
    return quality

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
# Data Loading from yfinance
# =============================================================================

def get_applovin_options(ticker: str, target_dte: int = 45, dte_window: int = 7,
                         strike_band: Tuple[float, float] = (0.7, 1.3),
                         max_spread_pct: float = 0.5) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Get options chain from yfinance with quality filters.
    Returns: (options_df, spot_price, puts_df)
    
    Args:
        strike_band: (min_ratio, max_ratio) relative to spot for filtering strikes
        max_spread_pct: Maximum bid-ask spread as % of mid price (0.5 = 50%)
    """
    print(f"Fetching {ticker} stock and options data from yfinance...")
    
    # Get stock data for current price
    stock = yf.Ticker(ticker)
    info = stock.info
    spot = info.get('currentPrice') or info.get('regularMarketPrice')
    
    if spot is None:
        # Fallback: get from history
        hist = stock.history(period="1d")
        if not hist.empty:
            spot = hist['Close'].iloc[-1]
        else:
            raise ValueError(f"Could not get current price for {ticker}")
    
    print(f"Current spot price: ${spot:.2f}")
    
    # Get options chain
    try:
        expirations = stock.options
        if not expirations:
            raise ValueError(f"No options expirations found for {ticker}")
        
        print(f"Found {len(expirations)} expiration dates")
        
        # Find expiration closest to target DTE
        today = pd.Timestamp.now().normalize()
        target_date = today + pd.Timedelta(days=target_dte)
        
        best_exp = None
        best_diff = None
        
        for exp_str in expirations:
            exp_date = pd.Timestamp(exp_str)
            diff = abs((exp_date - target_date).days)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_exp = exp_str
        
        if best_exp is None or best_diff > dte_window:
            # Use nearest expiration
            exp_dates = [pd.Timestamp(e) for e in expirations]
            best_exp = min(exp_dates, key=lambda x: abs((x - today).days - target_dte))
            best_exp = best_exp.strftime('%Y-%m-%d')
        
        print(f"Selected expiration: {best_exp} (DTE: {(pd.Timestamp(best_exp) - today).days} days)")
        
        # Try to get options chain for this expiration
        calls = pd.DataFrame()
        puts = pd.DataFrame()
        
        # Try the selected expiration first
        expirations_to_try = [best_exp] + [e for e in expirations if e != best_exp]
        expirations_to_try = sorted(expirations_to_try, key=lambda x: abs((pd.Timestamp(x) - target_date).days))[:10]
        
        for exp_str in expirations_to_try:
            try:
                opt_chain = stock.option_chain(exp_str)
                calls = opt_chain.calls
                puts = opt_chain.puts if hasattr(opt_chain, 'puts') else pd.DataFrame()
                if not calls.empty:
                    best_exp = exp_str
                    print(f"Successfully retrieved options for expiration: {best_exp}")
                    break
            except Exception as e:
                continue
        
        if calls.empty:
            raise ValueError(f"No call options found for any expiration. Tried {len(expirations_to_try)} expirations.")
        
        print(f"Found {len(calls)} call contracts (raw)")
        if not puts.empty:
            print(f"Found {len(puts)} put contracts (raw)")
        
        # Filter to valid strikes
        calls = calls[
            (calls['strike'].notna()) &
            (calls['strike'] > 0)
        ].copy()
        
        # Filter to strike band near spot
        strike_min = spot * strike_band[0]
        strike_max = spot * strike_band[1]
        calls = calls[(calls['strike'] >= strike_min) & (calls['strike'] <= strike_max)].copy()
        print(f"  After strike band filter [{strike_min:.2f}, {strike_max:.2f}]: {len(calls)} contracts")
        
        # Compute mid prices from bid/ask
        if 'bid' in calls.columns and 'ask' in calls.columns:
            calls['bid'] = pd.to_numeric(calls['bid'], errors='coerce').fillna(0)
            calls['ask'] = pd.to_numeric(calls['ask'], errors='coerce').fillna(0)
            calls['mid'] = 0.5 * (calls['bid'] + calls['ask'])
            
            # Compute spread
            calls['spread'] = calls['ask'] - calls['bid']
            calls['spread_pct'] = calls['spread'] / (calls['mid'] + 1e-10) * 100
            
            # Filter out bad spreads
            valid_spread = (calls['spread_pct'] <= max_spread_pct * 100) | (calls['spread'] <= 0.5)
            calls = calls[valid_spread].copy()
            print(f"  After spread filter (max {max_spread_pct*100:.1f}%): {len(calls)} contracts")
            
            # Use lastPrice as fallback for zero mid
            if 'lastPrice' in calls.columns:
                calls.loc[calls['mid'] <= 0, 'mid'] = pd.to_numeric(
                    calls.loc[calls['mid'] <= 0, 'lastPrice'], errors='coerce'
                )
        else:
            # Fallback to lastPrice
            if 'lastPrice' in calls.columns:
                calls['mid'] = pd.to_numeric(calls['lastPrice'], errors='coerce')
            else:
                raise ValueError("No bid/ask or lastPrice available")
        
        # Filter to positive mid prices
        calls = calls[calls['mid'] > 0].copy()
        
        # Deprioritize low volume/zero OI contracts (but don't drop yet)
        if 'volume' in calls.columns:
            calls['volume'] = pd.to_numeric(calls['volume'], errors='coerce').fillna(0)
        if 'openInterest' in calls.columns:
            calls['openInterest'] = pd.to_numeric(calls['openInterest'], errors='coerce').fillna(0)
        
        # Sort by volume/OI for prioritization
        if 'volume' in calls.columns and 'openInterest' in calls.columns:
            calls['liquidity'] = calls['volume'] + calls['openInterest']
            calls = calls.sort_values('liquidity', ascending=False)
        
        if len(calls) < 5:
            raise ValueError(f"Insufficient valid call options: {len(calls)} < 5")
        
        print(f"Using {len(calls)} call contracts with valid prices and spreads")
        
        # Optionally return puts for put-call parity analysis
        if not puts.empty:
            # Apply similar filters to puts
            puts = puts[
                (puts['strike'].notna()) &
                (puts['strike'] > 0) &
                (puts['strike'] >= strike_min) &
                (puts['strike'] <= strike_max)
            ].copy()
            
            if 'bid' in puts.columns and 'ask' in puts.columns:
                puts['bid'] = pd.to_numeric(puts['bid'], errors='coerce').fillna(0)
                puts['ask'] = pd.to_numeric(puts['ask'], errors='coerce').fillna(0)
                puts['mid'] = 0.5 * (puts['bid'] + puts['ask'])
                puts['spread'] = puts['ask'] - puts['bid']
                puts['spread_pct'] = puts['spread'] / (puts['mid'] + 1e-10) * 100
                valid_spread = (puts['spread_pct'] <= max_spread_pct * 100) | (puts['spread'] <= 0.5)
                puts = puts[valid_spread].copy()
                
                if 'lastPrice' in puts.columns:
                    puts.loc[puts['mid'] <= 0, 'mid'] = pd.to_numeric(
                        puts.loc[puts['mid'] <= 0, 'lastPrice'], errors='coerce'
                    )
            elif 'lastPrice' in puts.columns:
                puts['mid'] = pd.to_numeric(puts['lastPrice'], errors='coerce')
            
            puts = puts[puts['mid'] > 0].copy()
            print(f"Using {len(puts)} put contracts with valid prices and spreads")
        
        return calls, spot, puts if not puts.empty else None
        
    except Exception as e:
        raise RuntimeError(f"Error fetching options data: {e}")

# =============================================================================
# Visualization
# =============================================================================

def plot_distribution(strikes: np.ndarray, density: np.ndarray, spot: float, 
                     features: Dict, scores: Dict, ticker: str):
    """
    Plot the implied probability distribution.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Main density curve
    ax.fill_between(strikes, 0, density, alpha=0.6, color='steelblue', 
                    label='Implied Probability Density')
    ax.plot(strikes, density, color='darkblue', linewidth=2.5, alpha=0.9)
    
    # Compute mean and median for display
    mean = np.trapz(strikes * density, strikes)
    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]
    
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
                       alpha=0.4, color='crimson', label='Left Tail Risk (≤-5%)')
    
    # Shade right tail (upside potential)
    right_tail_threshold = 1.05 * spot
    right_mask = strikes >= right_tail_threshold
    if np.any(right_mask):
        ax.fill_between(strikes[right_mask], 0, density[right_mask], 
                       alpha=0.4, color='green', label='Right Tail (≥+5%)')
    
    # Labels and formatting
    ax.set_xlabel(f'{ticker} Price at Expiration ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title(f'{ticker} Options-Implied Probability Distribution\n'
                f'Current Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                fontsize=15, fontweight='bold', pad=15)
    
    # Add statistics in a clean box
    regime_label = "STRESS" if scores['stress_score'] > 0.4 else "CALM" if scores['calm_score'] > 0.6 else "NEUTRAL"
    
    stats_text = (
        f"Regime: {regime_label}\n"
        f"Stress Score: {scores['stress_score']:.3f}\n"
        f"Calm Score: {scores['calm_score']:.3f}\n"
        f"Risk-On Score: {scores['risk_on_score']:.3f}\n"
        f"\n"
        f"Std (Return): {features['std_ret']:.2%}\n"
        f"Skew Proxy: {features['skew_proxy']:.3f}\n"
        f"P(≤-5%): {features['p_left_5']:.1%}\n"
        f"P(≥+5%): {features['p_right_5']:.1%}"
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
    
    plt.tight_layout()
    return fig

# =============================================================================
# Main
# =============================================================================

def main():
    ticker = "JPM"
    output_dir = "output/distributions"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print(f"ANALYZING {ticker} OPTIONS-IMPLIED DISTRIBUTION")
    print("="*80)
    print()
    
    try:
        # Get options data
        result = get_applovin_options(ticker)
        if len(result) == 3:
            calls_df, spot, puts_df = result
        else:
            calls_df, spot = result
            puts_df = None
        
        # Extract strikes and prices from calls
        strikes = calls_df['strike'].values
        prices = calls_df['mid'].values
        
        # Optionally use puts via put-call parity for OTM strikes
        # For now, we'll use calls only, but this could be enhanced
        if puts_df is not None:
            print(f"\nNote: {len(puts_df)} put contracts available (not used in current analysis)")
        
        # Sort and remove duplicates
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        prices = prices[idx]
        
        unique_mask = np.concatenate([[True], np.diff(strikes) > 1e-3])
        strikes = strikes[unique_mask]
        prices = prices[unique_mask]
        
        if len(strikes) < 5:
            raise ValueError(f"Insufficient unique strikes: {len(strikes)}")
        
        print(f"\nExtracting implied density from {len(strikes)} call options...")
        
        # Extract implied density
        K, density, quality_info = extract_implied_density(strikes, prices, spot)
        
        # Check quality
        if not quality_info['valid']:
            print("\nWARNING: Density quality checks failed!")
            for warning in quality_info['warnings']:
                print(f"  - {warning}")
            print("\nProceeding with low-confidence results...")
        elif quality_info['warnings']:
            print("\nQuality warnings:")
            for warning in quality_info['warnings']:
                print(f"  - {warning}")
        
        if np.sum(density) < 0.1:
            raise ValueError("Invalid density extracted (sum too small)")
        
        print("Computing regime features and scores...")
        
        # Compute regime features
        features = compute_regime_features(K, density, spot)
        
        # Compute regime scores
        scores = compute_regime_scores(features)
        
        # Print results
        print("\n" + "="*80)
        print("REGIME ANALYSIS RESULTS")
        print("="*80)
        print(f"\nCurrent Spot Price: ${spot:.2f}")
        print(f"\nRegime Features:")
        print(f"  Std (Return Space):     {features['std_ret']:.4f} ({features['std_ret']:.2%})")
        print(f"  Skew Proxy:             {features['skew_proxy']:.4f}")
        print(f"  P(Return ≤ -5%):       {features['p_left_5']:.4f} ({features['p_left_5']:.2%})")
        print(f"  P(Return ≥ +5%):        {features['p_right_5']:.4f} ({features['p_right_5']:.2%})")
        
        # Print quality metrics if available
        if 'metrics' in quality_info:
            print(f"\nDensity Quality Metrics:")
            for key, value in quality_info['metrics'].items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:10.4f}")
                else:
                    print(f"  {key:20s}: {value}")
        
        print(f"\nRegime Scores:")
        print(f"  Stress Score:           {scores['stress_score']:.4f}")
        print(f"  Calm Score:             {scores['calm_score']:.4f}")
        print(f"  Risk-On Score:           {scores['risk_on_score']:.4f}")
        
        # Determine regime
        if scores['stress_score'] > 0.4:
            regime = "STRESS"
            interpretation = "High downside risk, high volatility, negative skew"
        elif scores['calm_score'] > 0.6:
            regime = "CALM"
            interpretation = "Low downside risk, low volatility"
        elif scores['risk_on_score'] > 0.5:
            regime = "RISK-ON"
            interpretation = "High upside potential, positive skew"
        else:
            regime = "NEUTRAL"
            interpretation = "Moderate risk characteristics"
        
        print(f"\nRegime Classification: {regime}")
        print(f"  Interpretation: {interpretation}")
        print("="*80)
        
        # Create visualization
        print("\nGenerating visualization...")
        fig = plot_distribution(K, density, spot, features, scores, ticker)
        
        output_path = os.path.join(output_dir, f"{ticker}_distribution_{datetime.now().strftime('%Y%m%d')}.png")
        fig.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=150)
        print(f"\nSaved: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

