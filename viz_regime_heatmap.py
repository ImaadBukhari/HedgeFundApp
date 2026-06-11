#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize options-implied probability distributions with SPY historical prices.

Usage:
  python viz_regime_heatmap.py --date 2024-03-15          # single-date report
  python viz_regime_heatmap.py --all                      # multi-date heatmap
  python viz_regime_heatmap.py --all --n_samples 20
"""

import argparse
import glob
import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from hedge.breeden_litzenberger import extract_implied_density
from hedge.regime import compute_skew_metrics

DATA_DIR = "data/opra_spy_snapshots/snapshots"
OUTPUT_DIR = "output/distributions"

R_MIN = -0.25
R_MAX = 0.25
N_RETURN_BINS = 100
DEFAULT_LOOKBACK_YEARS = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_snapshot_file(target_date: str) -> Optional[str]:
    """Find the closest snapshot file for a given date, preferring 16:30 closes."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv.gz")))
    if not files:
        return None

    target_dt = pd.to_datetime(target_date)
    candidates = []

    for filepath in files:
        try:
            date_str = os.path.basename(filepath).split("_")[0]
            diff = abs((pd.to_datetime(date_str) - target_dt).days)
            is_close = "t1630" in filepath or "16:30" in filepath
            candidates.append((diff, not is_close, filepath))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def load_snapshot(date: str) -> Tuple[pd.DataFrame, str]:
    filepath = find_snapshot_file(date)
    if filepath is None:
        raise FileNotFoundError(f"No snapshot file found for date {date}")

    df = pd.read_csv(filepath, compression="gzip")
    file_date = (
        df["date"].iloc[0] if "date" in df.columns else os.path.basename(filepath).split("_")[0]
    )
    print(f"Loaded: {os.path.basename(filepath)} ({file_date})")
    return df, filepath


def extract_density_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, float, Dict]:
    """
    Extract density from a snapshot dataframe and project it into return space.
    Returns: (return_grid, density_on_return_grid, dte, spot, metrics)
    """
    calls = df[
        (df["opt_type"] == "call")
        & df["mid"].notna()
        & (df["mid"] > 0)
        & df["strike"].notna()
        & np.isfinite(df["strike"])
        & np.isfinite(df["mid"])
    ].copy()

    if len(calls) < 20:
        raise ValueError(f"Only {len(calls)} calls — need at least 20")

    spot = float(calls["underlying_spot"].iloc[0])

    if "expiration" in df.columns and df["expiration"].notna().any():
        expiration = pd.to_datetime(df["expiration"].iloc[0])
        snapshot_date = pd.to_datetime(df["date"].iloc[0])
        dte = (expiration - snapshot_date).days
    elif "dte" in df.columns and df["dte"].notna().any():
        dte = int(df["dte"].iloc[0])
    else:
        dte = 45

    T = dte / 365.0

    strikes = calls["strike"].values
    prices = calls["mid"].values
    idx = np.argsort(strikes)
    strikes, prices = strikes[idx], prices[idx]
    unique_mask = np.concatenate([[True], np.diff(strikes) > 1e-3])
    strikes = np.sort(strikes)[unique_mask]
    prices = prices[np.argsort(calls["strike"].values)][unique_mask]

    if len(strikes) < 10:
        raise ValueError(f"Too few unique strikes: {len(strikes)}")

    K, density, _ = extract_implied_density(strikes, prices, spot, r=0.0, T=T)

    if np.sum(density) < 0.1:
        raise ValueError("Invalid density extracted")

    r_grid = np.linspace(R_MIN, R_MAX, N_RETURN_BINS)
    K_from_r = spot * (1 + r_grid)
    interp = interp1d(K, density, kind="linear", bounds_error=False, fill_value=0.0)
    density_r = np.maximum(interp(K_from_r), 0.0)
    if density_r.sum() > 0:
        density_r = density_r / density_r.sum()

    metrics = compute_skew_metrics(K, density, spot)
    print(f"Spot: ${spot:.2f}, DTE: {dte}, Mean: ${metrics['mean']:.2f}, "
          f"Skew: {metrics['skew_proxy']:.4f}, P(≤-5%): {metrics['left_tail_prob']:.2%}")

    return r_grid, density_r, dte, spot, metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_single_report(date: str, lookback_years: float = DEFAULT_LOOKBACK_YEARS):
    df, _ = load_snapshot(date)
    r_grid, density, dte, spot, metrics = extract_density_from_df(df)

    snapshot_dt = pd.to_datetime(date)
    start_date = snapshot_dt - pd.DateOffset(years=lookback_years)

    print(f"\nLoading SPY history {start_date.date()} → {snapshot_dt.date()}...")
    spy = yf.download("SPY", start=start_date.strftime("%Y-%m-%d"),
                      end=(snapshot_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)

    if spy.empty or "Close" not in spy.columns:
        raise ValueError("No SPY price data found for date range")

    spy_prices = spy["Close"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1.2])

    ax1.plot(spy_prices.index, spy_prices.values, linewidth=2, color="#2E86AB",
             label="SPY Adjusted Close")
    ax1.axvline(snapshot_dt, color="red", linestyle="--", linewidth=2.5,
                label=f"Snapshot: {date}", alpha=0.8)
    ax1.axhline(spot, color="orange", linestyle=":", linewidth=1.5, alpha=0.6,
                label=f"Spot: ${spot:.2f}")
    ax1.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax1.set_ylabel("SPY Price ($)", fontsize=12, fontweight="bold")
    ax1.set_title(f"SPY Historical Price (Last {lookback_years} Years)",
                  fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    dte_days = int(dte)
    heatmap_data = np.tile(density, (dte_days + 1, 1))
    if dte_days > 5:
        heatmap_data = gaussian_filter1d(heatmap_data, sigma=0.5, axis=0)
    heatmap_log = np.log10(heatmap_data + 1e-12)

    im = ax2.imshow(heatmap_log, aspect="auto", origin="lower",
                    extent=[R_MIN * 100, R_MAX * 100, 0, dte_days],
                    cmap="viridis", interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label("log₁₀(Probability Density)", rotation=270, labelpad=20, fontsize=11)

    ax2.axvline(0, color="white", linestyle="--", linewidth=2, alpha=0.8, label="0% Return")
    ax2.axvline(-5, color="red", linestyle="--", linewidth=2, alpha=0.8, label="-5% Downside")
    ax2.set_xlabel("Return at Expiry (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Days to Expiry", fontsize=12, fontweight="bold")
    ax2.set_title(f"Options-Implied Probability Distribution (Forward from {date})\n"
                  "Implied distribution at expiry, replicated across horizon",
                  fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.2, color="white")

    stats_text = (
        f"Mean Return: {((metrics['mean'] / spot - 1) * 100):.2f}%\n"
        f"Std Dev: {(metrics['std'] / spot * 100):.2f}%\n"
        f"Skew: {metrics['skew_proxy']:.3f}\n"
        f"P(≤-5%): {metrics['left_tail_prob']:.1%}"
    )
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9, edgecolor="gray"))

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"report_{date}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")
    return fig


def plot_time_return_heatmap(n_samples: Optional[int] = None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {DATA_DIR}")

    print(f"Found {len(files)} files — deduplicating by date...")

    # One file per date, preferring 16:30 closes
    date_to_file = {}
    for filepath in files:
        try:
            date_str = os.path.basename(filepath).split("_")[0]
            date_key = pd.to_datetime(date_str)
            is_close = "t1630" in filepath
            if date_key not in date_to_file or (is_close and not date_to_file[date_key][1]):
                date_to_file[date_key] = (filepath, is_close)
        except Exception:
            continue

    sorted_dates = sorted(date_to_file.keys())
    if n_samples is not None:
        sorted_dates = sorted_dates[:n_samples]

    print(f"Processing {len(sorted_dates)} unique dates...")

    dates_list = []
    density_matrix = []

    for date_key in sorted_dates:
        filepath, _ = date_to_file[date_key]
        try:
            df = pd.read_csv(filepath, compression="gzip")
            r_grid, density, _, _, _ = extract_density_from_df(df)
            dates_list.append(date_key)
            density_matrix.append(density)
        except Exception as e:
            print(f"Skipped {os.path.basename(filepath)}: {e}")

    if len(density_matrix) < 2:
        raise ValueError(f"Need at least 2 valid snapshots; got {len(density_matrix)}")

    density_array = np.array(density_matrix)
    dates_array = np.array(dates_list)
    sort_idx = np.argsort(dates_array)
    density_array = density_array[sort_idx]
    dates_array = dates_array[sort_idx]

    density_log = np.log10(density_array + 1e-12)

    fig, ax = plt.subplots(figsize=(16, max(8, len(dates_array) * 0.3)))
    im = ax.imshow(density_log, aspect="auto", origin="lower",
                   extent=[R_MIN * 100, R_MAX * 100, 0, len(dates_array)],
                   cmap="viridis", interpolation="bilinear")

    n_dates = len(dates_array)
    step = 1 if n_dates <= 20 else max(1, n_dates // 15) if n_dates <= 50 else max(1, n_dates // 20)
    yticks = list(range(0, n_dates, step))
    if n_dates - 1 not in yticks:
        yticks.append(n_dates - 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([dates_array[i].strftime("%Y-%m-%d") for i in yticks], fontsize=9)

    ax.axvline(0, color="white", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(-5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("log₁₀(Probability Density)", rotation=270, labelpad=20, fontsize=11)

    ax.set_xlabel("Return at Expiry (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Snapshot Date", fontsize=12, fontweight="bold")
    ax.set_title(
        "Options-Implied Probability Distributions Over Time\n"
        f"({len(dates_array)} snapshots from "
        f"{dates_array[0].date()} to {dates_array[-1].date()})",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, color="white")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "time_return_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize options-implied probability distributions with SPY history"
    )
    parser.add_argument("--date", type=str, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--lookback_years", type=float, default=DEFAULT_LOOKBACK_YEARS)
    parser.add_argument("--all", action="store_true", help="Generate multi-date heatmap")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        print("Generating multi-date heatmap...")
        plot_time_return_heatmap(n_samples=args.n_samples)
    elif args.date:
        print(f"Generating single-date report for {args.date}...")
        plot_single_report(args.date, lookback_years=args.lookback_years)
    else:
        parser.print_help()
        print("\nError: Must specify --date or --all")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
