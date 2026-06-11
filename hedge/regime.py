#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Options-implied regime classification.

Provides:
  - compute_skew_metrics   : descriptive stats from a density
  - compute_regime_features: std, skew, tail probs in return space
  - compute_regime_scores  : composite stress / calm / risk-on scores
  - analyze_all_snapshots  : batch process CSV snapshot directory
  - select_extreme_snapshots: pick most-stressed and calmest snapshots
"""

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from hedge.breeden_litzenberger import extract_implied_density


def compute_skew_metrics(strikes: np.ndarray, density: np.ndarray,
                         spot: float) -> Dict[str, float]:
    """Descriptive statistics from an implied density (price space)."""
    mean = np.trapz(strikes * density, strikes)
    variance = np.trapz((strikes - mean) ** 2 * density, strikes)
    std = np.sqrt(variance)

    skewness = (
        np.trapz(((strikes - mean) / std) ** 3 * density, strikes) if std > 0 else 0.0
    )

    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]

    skew_proxy = (mean - median) / std if std > 0 else 0.0

    threshold = 0.95 * spot
    left_tail_prob = np.trapz(
        density[strikes <= threshold], strikes[strikes <= threshold]
    )

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "skew_proxy": skew_proxy,
        "median": median,
        "left_tail_prob": left_tail_prob,
        "spot": spot,
    }


def compute_regime_features(strikes: np.ndarray, density: np.ndarray,
                             spot: float) -> Dict[str, float]:
    """
    Regime features in return space:
      std_ret    : implied return std
      skew_proxy : (mean - median) / std
      p_left_5   : P(return ≤ -5%)
      p_right_5  : P(return ≥ +5%)
    """
    returns = (strikes / spot) - 1.0
    mean_ret = np.trapz(returns * density, strikes)
    variance_ret = np.trapz((returns - mean_ret) ** 2 * density, strikes)
    std_ret = np.sqrt(variance_ret)

    mean = np.trapz(strikes * density, strikes)
    std = np.sqrt(np.trapz((strikes - mean) ** 2 * density, strikes))
    cdf = np.cumsum(density) * (strikes[1] - strikes[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median = strikes[median_idx] if median_idx < len(strikes) else strikes[-1]
    skew_proxy = (mean - median) / std if std > 0 else 0.0

    p_left_5 = np.trapz(
        density[strikes <= spot * 0.95], strikes[strikes <= spot * 0.95]
    )
    p_right_5 = np.trapz(
        density[strikes >= spot * 1.05], strikes[strikes >= spot * 1.05]
    )

    return {
        "std_ret": std_ret,
        "skew_proxy": skew_proxy,
        "p_left_5": p_left_5,
        "p_right_5": p_right_5,
    }


def compute_regime_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Composite regime scores from regime features.

    stress_score : high downside tail + high vol + negative skew → [0, 1]
    calm_score   : low downside tail + low vol → [0, 1]
    risk_on_score: high right-tail + positive skew → [0, 1]
    """
    std_ret = features["std_ret"]
    skew_proxy = features["skew_proxy"]
    p_left_5 = features["p_left_5"]
    p_right_5 = features["p_right_5"]

    std_norm = min(std_ret / 0.5, 1.0)
    p_left_norm = min(p_left_5, 1.0)
    p_right_norm = min(p_right_5, 1.0)
    skew_norm = (skew_proxy + 0.5) / 1.0

    stress_score = p_left_norm * 0.5 + std_norm * 0.3 + (1 - skew_norm) * 0.2
    calm_score = (1 - p_left_norm) * 0.6 + (1 - std_norm) * 0.4
    risk_on_score = p_right_norm * 0.6 + max(0.0, skew_proxy) * 0.4

    return {
        "stress_score": stress_score,
        "calm_score": calm_score,
        "risk_on_score": risk_on_score,
    }


# ---------------------------------------------------------------------------
# Snapshot batch processing
# ---------------------------------------------------------------------------

def _load_calls_from_snapshot(filepath: str):
    """Return (calls_df, spot) for a snapshot file, or raise on failure."""
    df = pd.read_csv(filepath, compression="gzip")
    if df.empty:
        raise ValueError("Empty file")

    spot = df["underlying_spot"].iloc[0] if "underlying_spot" in df.columns else None
    if spot is None or spot <= 0:
        raise ValueError("Missing or invalid spot price")

    calls = df[
        (df["opt_type"] == "call")
        & df["mid"].notna()
        & (df["mid"] > 0)
        & df["strike"].notna()
    ].copy()

    if len(calls) < 20:
        raise ValueError(f"Only {len(calls)} calls — need at least 20")

    return calls, float(spot), df


def analyze_all_snapshots(data_dir: str) -> pd.DataFrame:
    """
    Process every CSV snapshot in data_dir and return a DataFrame with
    regime features and scores for each date.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv.gz")))
    if len(files) < 2:
        raise ValueError(
            f"Need at least 2 snapshot files; found {len(files)} in {data_dir}"
        )

    print(f"Found {len(files)} snapshot files — analysing...")

    results = []
    for filepath in files:
        try:
            calls, spot, df = _load_calls_from_snapshot(filepath)

            strikes_raw = calls["strike"].values
            prices = calls["mid"].values
            idx = np.argsort(strikes_raw)
            strikes_raw, prices = strikes_raw[idx], prices[idx]
            unique_mask = np.concatenate([[True], np.diff(strikes_raw) > 1e-3])
            strikes_raw, prices = strikes_raw[unique_mask], prices[unique_mask]

            if len(strikes_raw) < 10:
                continue

            K, density, _ = extract_implied_density(strikes_raw, prices, spot)
            if np.sum(density) < 0.1:
                continue

            features = compute_regime_features(K, density, spot)
            scores = compute_regime_scores(features)
            date_str = (
                df["date"].iloc[0]
                if "date" in df.columns
                else os.path.basename(filepath).split("_")[0]
            )

            results.append(
                {
                    "filepath": filepath,
                    "date": date_str,
                    "df": df,
                    "strikes": K,
                    "density": density,
                    "spot": spot,
                    **features,
                    **scores,
                }
            )
        except Exception as e:
            print(f"Skipped {os.path.basename(filepath)}: {e}")

    if len(results) < 2:
        raise ValueError(
            f"Could not process enough snapshots — got {len(results)} valid."
        )

    print(f"Processed {len(results)} snapshots.")

    df_results = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("df", "strikes", "density")}
            for r in results
        ]
    )
    df_results["_snapshot_dict"] = results
    return df_results


def select_extreme_snapshots(
    df_results: pd.DataFrame, top_n: int = 2
) -> Tuple[list, list]:
    """
    Return (stress_snapshots, calm_snapshots) — each a list of snapshot dicts.
    Stress = highest stress_score; calm = lowest stress_score.
    """
    stress = df_results.nlargest(top_n, "stress_score")["_snapshot_dict"].tolist()
    calm = df_results.nsmallest(top_n, "stress_score")["_snapshot_dict"].tolist()
    return stress, calm


def find_snapshot_by_date(data_dir: str, target_date: str) -> Optional[Dict]:
    """
    Find and load the snapshot closest to target_date (within 7 days).
    Returns a snapshot dict (same shape as analyze_all_snapshots rows) or None.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv.gz")))
    target_dt = pd.to_datetime(target_date)

    best_file, best_diff = None, None
    for filepath in files:
        try:
            date_str = os.path.basename(filepath).split("_")[0]
            diff = abs((pd.to_datetime(date_str) - target_dt).days)
            if best_diff is None or diff < best_diff:
                best_diff, best_file = diff, filepath
        except Exception:
            continue

    if best_file is None or (best_diff is not None and best_diff > 7):
        return None

    try:
        calls, spot, df = _load_calls_from_snapshot(best_file)
        strikes_raw = calls["strike"].values
        prices = calls["mid"].values
        idx = np.argsort(strikes_raw)
        strikes_raw, prices = strikes_raw[idx], prices[idx]
        unique_mask = np.concatenate([[True], np.diff(strikes_raw) > 1e-3])
        strikes_raw, prices = strikes_raw[unique_mask], prices[unique_mask]

        if len(strikes_raw) < 10:
            return None

        K, density, _ = extract_implied_density(strikes_raw, prices, spot)
        if np.sum(density) < 0.1:
            return None

        metrics = compute_skew_metrics(K, density, spot)
        features = compute_regime_features(K, density, spot)
        scores = compute_regime_scores(features)
        date_str = (
            df["date"].iloc[0]
            if "date" in df.columns
            else os.path.basename(best_file)
        )

        return {
            "filepath": best_file,
            "df": df,
            "strikes": K,
            "density": density,
            "metrics": metrics,
            "spot": spot,
            "date": date_str,
            **features,
            **scores,
        }
    except Exception as e:
        print(f"Error processing {best_file}: {e}")
        return None
