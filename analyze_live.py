#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze any ticker's live options-implied probability distribution.
Fetches options data from yfinance and applies Breeden-Litzenberger.

Usage:
  python analyze_live.py              # defaults to JPM
  python analyze_live.py --ticker AAPL --dte 30
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from hedge.breeden_litzenberger import extract_implied_density
from hedge.regime import compute_regime_features, compute_regime_scores

OUTPUT_DIR = "output/distributions"


# ---------------------------------------------------------------------------
# Data loading from yfinance
# ---------------------------------------------------------------------------

def get_live_options(
    ticker: str,
    target_dte: int = 45,
    dte_window: int = 7,
    strike_band: Tuple[float, float] = (0.7, 1.3),
    max_spread_pct: float = 0.5,
) -> Tuple[pd.DataFrame, float, Optional[pd.DataFrame]]:
    """
    Fetch live options chain for ticker from yfinance.
    Returns (calls_df, spot_price, puts_df_or_None).
    """
    print(f"Fetching {ticker} options data from yfinance...")

    stock = yf.Ticker(ticker)
    info = stock.info
    spot = info.get("currentPrice") or info.get("regularMarketPrice")

    if spot is None:
        hist = stock.history(period="1d")
        if not hist.empty:
            spot = hist["Close"].iloc[-1]
        else:
            raise ValueError(f"Could not get current price for {ticker}")

    spot = float(spot)
    print(f"Spot price: ${spot:.2f}")

    expirations = stock.options
    if not expirations:
        raise ValueError(f"No options expirations found for {ticker}")

    today = pd.Timestamp.now().normalize()
    target_date = today + pd.Timedelta(days=target_dte)

    # Pick expiration closest to target DTE
    expirations_sorted = sorted(
        expirations, key=lambda e: abs((pd.Timestamp(e) - target_date).days)
    )

    calls = pd.DataFrame()
    puts = pd.DataFrame()
    best_exp = None

    for exp_str in expirations_sorted[:10]:
        try:
            chain = stock.option_chain(exp_str)
            if not chain.calls.empty:
                calls = chain.calls
                puts = getattr(chain, "puts", pd.DataFrame())
                best_exp = exp_str
                dte_actual = (pd.Timestamp(exp_str) - today).days
                print(f"Selected expiration: {best_exp} (DTE: {dte_actual})")
                break
        except Exception:
            continue

    if calls.empty or best_exp is None:
        raise ValueError("No call options found for any near-term expiration")

    print(f"Raw calls: {len(calls)}")

    # Strike band filter
    strike_min = spot * strike_band[0]
    strike_max = spot * strike_band[1]
    calls = calls[calls["strike"].between(strike_min, strike_max)].copy()

    # Mid price from bid/ask
    if "bid" in calls.columns and "ask" in calls.columns:
        calls["bid"] = pd.to_numeric(calls["bid"], errors="coerce").fillna(0)
        calls["ask"] = pd.to_numeric(calls["ask"], errors="coerce").fillna(0)
        calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])
        calls["spread_pct"] = (calls["ask"] - calls["bid"]) / (calls["mid"] + 1e-10) * 100
        calls = calls[
            (calls["spread_pct"] <= max_spread_pct * 100) | ((calls["ask"] - calls["bid"]) <= 0.5)
        ].copy()
        if "lastPrice" in calls.columns:
            mask = calls["mid"] <= 0
            calls.loc[mask, "mid"] = pd.to_numeric(calls.loc[mask, "lastPrice"], errors="coerce")
    elif "lastPrice" in calls.columns:
        calls["mid"] = pd.to_numeric(calls["lastPrice"], errors="coerce")
    else:
        raise ValueError("No bid/ask or lastPrice available")

    calls = calls[calls["mid"] > 0].copy()

    if len(calls) < 5:
        raise ValueError(f"Only {len(calls)} valid call contracts after filtering")

    print(f"Using {len(calls)} call contracts")

    # Optionally return puts
    if not puts.empty:
        puts = puts[puts["strike"].between(strike_min, strike_max)].copy()
        if "bid" in puts.columns and "ask" in puts.columns:
            puts["bid"] = pd.to_numeric(puts["bid"], errors="coerce").fillna(0)
            puts["ask"] = pd.to_numeric(puts["ask"], errors="coerce").fillna(0)
            puts["mid"] = 0.5 * (puts["bid"] + puts["ask"])
            puts["spread_pct"] = (puts["ask"] - puts["bid"]) / (puts["mid"] + 1e-10) * 100
            puts = puts[
                (puts["spread_pct"] <= max_spread_pct * 100) | ((puts["ask"] - puts["bid"]) <= 0.5)
            ].copy()
            if "lastPrice" in puts.columns:
                mask = puts["mid"] <= 0
                puts.loc[mask, "mid"] = pd.to_numeric(puts.loc[mask, "lastPrice"], errors="coerce")
        elif "lastPrice" in puts.columns:
            puts["mid"] = pd.to_numeric(puts["lastPrice"], errors="coerce")
        puts = puts[puts["mid"] > 0].copy()
        print(f"Using {len(puts)} put contracts")

    return calls, spot, puts if not puts.empty else None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_distribution(
    strikes: np.ndarray,
    density: np.ndarray,
    spot: float,
    features: Dict,
    scores: Dict,
    ticker: str,
):
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.fill_between(strikes, 0, density, alpha=0.6, color="steelblue",
                    label="Implied Probability Density")
    ax.plot(strikes, density, color="darkblue", linewidth=2.5, alpha=0.9)

    mean = np.trapz(strikes * density, strikes)
    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]

    ax.axvline(spot, color="red", linestyle="--", linewidth=2.5,
               label=f"Spot: ${spot:.2f}", alpha=0.8)
    ax.axvline(mean, color="orange", linestyle="--", linewidth=2.5,
               label=f"Mean: ${mean:.2f}", alpha=0.8)
    ax.axvline(median, color="green", linestyle="--", linewidth=2.5,
               label=f"Median: ${median:.2f}", alpha=0.8)

    left_mask = strikes <= 0.95 * spot
    if np.any(left_mask):
        ax.fill_between(strikes[left_mask], 0, density[left_mask],
                        alpha=0.4, color="crimson", label="Left Tail (≤-5%)")
    right_mask = strikes >= 1.05 * spot
    if np.any(right_mask):
        ax.fill_between(strikes[right_mask], 0, density[right_mask],
                        alpha=0.4, color="green", label="Right Tail (≥+5%)")

    if scores["stress_score"] > 0.4:
        regime_label = "STRESS"
    elif scores["calm_score"] > 0.6:
        regime_label = "CALM"
    elif scores["risk_on_score"] > 0.5:
        regime_label = "RISK-ON"
    else:
        regime_label = "NEUTRAL"

    stats_text = (
        f"Regime: {regime_label}\n"
        f"Stress: {scores['stress_score']:.3f}  "
        f"Calm: {scores['calm_score']:.3f}  "
        f"Risk-On: {scores['risk_on_score']:.3f}\n"
        f"\n"
        f"Std (Return): {features['std_ret']:.2%}\n"
        f"Skew Proxy: {features['skew_proxy']:.3f}\n"
        f"P(≤-5%): {features['p_left_5']:.1%}\n"
        f"P(≥+5%): {features['p_right_5']:.1%}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.9,
                      edgecolor="gray", linewidth=1.5))

    ax.set_xlabel(f"{ticker} Price at Expiration ($)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{ticker} Options-Implied Probability Distribution\n"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f9fa")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live options-implied regime analysis for any ticker"
    )
    parser.add_argument("--ticker", type=str, default="JPM")
    parser.add_argument("--dte", type=int, default=45,
                        help="Target days-to-expiry (default: 45)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print(f"ANALYZING {ticker} OPTIONS-IMPLIED DISTRIBUTION")
    print("=" * 80)

    try:
        calls_df, spot, puts_df = get_live_options(ticker, target_dte=args.dte)

        if puts_df is not None:
            print(f"({len(puts_df)} put contracts available, not used in analysis)")

        strikes = calls_df["strike"].values
        prices = calls_df["mid"].values
        idx = np.argsort(strikes)
        strikes, prices = strikes[idx], prices[idx]
        unique_mask = np.concatenate([[True], np.diff(strikes) > 1e-3])
        strikes, prices = strikes[unique_mask], prices[unique_mask]

        if len(strikes) < 5:
            raise ValueError(f"Insufficient unique strikes: {len(strikes)}")

        print(f"\nExtracting implied density from {len(strikes)} calls...")
        K, density, quality = extract_implied_density(strikes, prices, spot)

        if not quality["valid"]:
            print("\nWARNING: Density quality checks failed!")
            for w in quality["warnings"]:
                print(f"  - {w}")
        elif quality["warnings"]:
            for w in quality["warnings"]:
                print(f"  Quality note: {w}")

        if np.sum(density) < 0.1:
            raise ValueError("Invalid density (sum too small)")

        features = compute_regime_features(K, density, spot)
        scores = compute_regime_scores(features)

        print("\n" + "=" * 80)
        print("REGIME ANALYSIS")
        print("=" * 80)
        print(f"Spot: ${spot:.2f}")
        print(f"\nFeatures:")
        print(f"  Std (return):    {features['std_ret']:.4f} ({features['std_ret']:.2%})")
        print(f"  Skew proxy:      {features['skew_proxy']:.4f}")
        print(f"  P(≤-5%):         {features['p_left_5']:.4f} ({features['p_left_5']:.2%})")
        print(f"  P(≥+5%):         {features['p_right_5']:.4f} ({features['p_right_5']:.2%})")

        if "metrics" in quality:
            print("\nDensity quality:")
            for k, v in quality["metrics"].items():
                print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

        print(f"\nScores:")
        print(f"  Stress:  {scores['stress_score']:.4f}")
        print(f"  Calm:    {scores['calm_score']:.4f}")
        print(f"  Risk-On: {scores['risk_on_score']:.4f}")

        if scores["stress_score"] > 0.4:
            regime, interp = "STRESS", "High downside risk, high vol, negative skew"
        elif scores["calm_score"] > 0.6:
            regime, interp = "CALM", "Low downside risk, low vol"
        elif scores["risk_on_score"] > 0.5:
            regime, interp = "RISK-ON", "High upside potential, positive skew"
        else:
            regime, interp = "NEUTRAL", "Moderate risk characteristics"

        print(f"\nRegime: {regime} — {interp}")
        print("=" * 80)

        fig = plot_distribution(K, density, spot, features, scores, ticker)
        out_path = os.path.join(
            OUTPUT_DIR, f"{ticker}_distribution_{datetime.now().strftime('%Y%m%d')}.png"
        )
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=150)
        print(f"\nSaved: {out_path}")
        plt.show()

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
