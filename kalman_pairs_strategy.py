#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kalman-ML Pairs Strategy.

Pairs of equities are tracked via a Kalman filter (time-varying alpha/beta),
and an XGBoost gate decides when to activate the hedge based on spread features.

Usage:
  python kalman_pairs_strategy.py
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from hedge.kalman import KalmanResult, StrategyParams, kalman_time_varying_beta
from hedge.ml_gate import (
    FEATURE_COLS,
    BacktestResult,
    backtest_pair_with_gate,
    calmar_ratio,
    make_features_labels,
    sharpe_ratio,
    train_classifier,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

UNIVERSE = [
    # Mega/large-cap tech + comm
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ORCL", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM", "IBM",
    "SHOP", "UBER", "SNOW", "PANW", "NOW", "MU", "AMAT", "TXN", "MRVL", "ASML",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "V", "MA", "AXP", "PYPL", "SQ",
    # Industrials
    "CAT", "HON", "GE", "LMT", "NOC", "BA", "DE", "MMM", "UNP",
    # Staples & Discretionary
    "PG", "KO", "PEP", "COST", "WMT", "TGT", "HD", "LOW", "NKE", "MCD", "SBUX",
    # Healthcare
    "UNH", "JNJ", "PFE", "LLY", "MRK", "ABT", "TMO", "BMY", "GILD", "AMGN",
    # Energy & Materials
    "XOM", "CVX", "COP", "SLB", "EOG", "SHEL", "BP", "SCCO", "FCX", "NEM",
    # Utilities & REITs
    "NEE", "DUK", "SO", "AEP", "PLD", "AMT",
    # Index proxy
    "SPY",
]

NUM_PAIRS = 40
START = "2015-01-01"
END: Optional[str] = None

PARAMS = StrategyParams()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_data(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def select_pairs_by_corr(
    returns: pd.DataFrame, num_pairs: int = NUM_PAIRS
) -> List[Tuple[str, str]]:
    """Pick the top num_pairs most-correlated pairs (excluding SPY)."""
    tickers = [t for t in returns.columns if t != "SPY"]
    R = returns[tickers].corr().abs()
    np.fill_diagonal(R.values, 0.0)

    pairs: set = set()
    for (a, b), _ in R.unstack().sort_values(ascending=False).items():
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in pairs:
            continue
        pairs.add(key)
        if len(pairs) >= num_pairs:
            break
    return list(pairs)


def split_pairs(
    pairs: List[Tuple[str, str]], n_train: int = 30, n_test: int = 10
) -> Tuple[List, List]:
    assert len(pairs) >= n_train + n_test
    return pairs[:n_train], pairs[n_train: n_train + n_test]


def aggregate_metrics(results: List[BacktestResult]) -> Dict[str, float]:
    if not results:
        return {"sharpe": 0.0, "calmar": 0.0}
    panel = pd.concat([r.daily_returns for r in results], axis=1).fillna(0.0)
    eq = panel.mean(axis=1)
    return {
        "sharpe": sharpe_ratio(eq),
        "calmar": calmar_ratio(eq),
        "n_days": len(eq),
    }


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def coarse_tune(
    returns: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    params: StrategyParams,
) -> StrategyParams:
    q_grid = [1e-5]
    r_grid = [2e-3]
    gate_grid = [(0.5, 0.4)]
    vt_grid = [0.25]

    best: Optional[StrategyParams] = None
    best_score = -1e9
    best_ann_return = 0.0
    best_sharpe = 0.0

    tune_pairs = pairs[:10]

    for q in q_grid:
        for r in r_grid:
            for (on, off) in gate_grid:
                for vt in vt_grid:
                    P = StrategyParams(
                        process_noise=q, measurement_noise=r,
                        vol_target_annual=vt, gate_on=on, gate_off=off,
                        horizon_days=params.horizon_days,
                        cost_bps_roundtrip=params.cost_bps_roundtrip,
                        min_lookback_days=params.min_lookback_days,
                    )

                    feat_frames = []
                    for A, B in tune_pairs:
                        rA = returns[A].dropna()
                        rB = returns[B].dropna()
                        kal = kalman_time_varying_beta(rA, rB, P.process_noise, P.measurement_noise)
                        feat_frames.append(
                            make_features_labels(rA, rB, kal, P.horizon_days, tickerA=A, tickerB=B)
                        )

                    all_df = pd.concat(feat_frames).dropna()
                    model = train_classifier(all_df)

                    tune_results = []
                    for A, B in tune_pairs:
                        rA = returns[A].dropna()
                        rB = returns[B].dropna()
                        kal = kalman_time_varying_beta(rA, rB, P.process_noise, P.measurement_noise)
                        tune_results.append(
                            backtest_pair_with_gate(rA, rB, kal, model, P, f"{A}-{B}")
                        )

                    agg = aggregate_metrics(tune_results)
                    panel = pd.concat([r.daily_returns for r in tune_results], axis=1).fillna(0.0)
                    ann_return = panel.mean(axis=1).mean() * 252

                    print(
                        f"[TUNE] Q={P.process_noise:.1e} R={P.measurement_noise:.1e} "
                        f"Q/R={P.process_noise/P.measurement_noise:.2f} "
                        f"Sharpe={agg['sharpe']:.3f} Return={ann_return:.2%} "
                        f"Calmar={agg['calmar']:.2f}"
                    )

                    score = (
                        0.5 * agg["sharpe"]
                        + 0.4 * min(ann_return, 0.15) / 0.10
                        + 0.1 * agg["calmar"]
                    )
                    if score > best_score:
                        best_score = score
                        best = P
                        best_ann_return = ann_return
                        best_sharpe = agg["sharpe"]

    print(f"[TUNER] Best: Sharpe={best_sharpe:.3f}  Return={best_ann_return:.2%}  Params={best}")
    return best if best is not None else params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Downloading data...")
    prices = download_data(UNIVERSE, START, END)
    prices = prices.dropna(how="all", axis=0).dropna(how="any", axis=1)
    rets = daily_returns(prices)

    pairs = select_pairs_by_corr(rets)
    print(f"Selected {len(pairs)} pairs. First 5: {pairs[:5]}")

    train_pairs, test_pairs = split_pairs(pairs)
    print(f"Train: {len(train_pairs)}  Test: {len(test_pairs)}")

    tuned_params = coarse_tune(rets, train_pairs, PARAMS)
    print("Tuned params:", tuned_params)

    # Build training set for ML
    feat_frames = []
    for A, B in train_pairs:
        rA = rets[A].dropna()
        rB = rets[B].dropna()
        kal = kalman_time_varying_beta(rA, rB, tuned_params.process_noise,
                                       tuned_params.measurement_noise)
        feat_frames.append(
            make_features_labels(rA, rB, kal, tuned_params.horizon_days, tickerA=A, tickerB=B)
        )

    train_df = pd.concat(feat_frames).dropna()
    model = train_classifier(train_df)

    # Backtest
    def run_backtest(pair_list):
        results = []
        for A, B in pair_list:
            rA = rets[A].dropna()
            rB = rets[B].dropna()
            kal = kalman_time_varying_beta(rA, rB, tuned_params.process_noise,
                                           tuned_params.measurement_noise)
            results.append(
                backtest_pair_with_gate(rA, rB, kal, model, tuned_params, f"{A}-{B}")
            )
        return results

    train_results = run_backtest(train_pairs)
    test_results = run_backtest(test_pairs)

    train_agg = aggregate_metrics(train_results)
    test_agg = aggregate_metrics(test_results)
    print(f"[TRAIN] Sharpe: {train_agg['sharpe']:.3f}  Calmar: {train_agg['calmar']:.3f}")
    print(f"[TEST]  Sharpe: {test_agg['sharpe']:.3f}  Calmar: {test_agg['calmar']:.3f}")

    out_dir = "kalman_ml_pairs_output"
    os.makedirs(out_dir, exist_ok=True)

    def results_to_df(results: List[BacktestResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            d = dict(r.details)
            d["sharpe"] = r.sharpe
            d["calmar"] = r.calmar
            rows.append(d)
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    results_to_df(train_results).to_csv(os.path.join(out_dir, "train_pair_results.csv"), index=False)
    results_to_df(test_results).to_csv(os.path.join(out_dir, "test_pair_results.csv"), index=False)

    def equity_curve(results: List[BacktestResult]) -> pd.Series:
        panel = pd.concat([r.daily_returns for r in results], axis=1).fillna(0.0)
        return (1 + panel.mean(axis=1)).cumprod()

    equity_curve(train_results).to_csv(os.path.join(out_dir, "equity_curve_train.csv"))
    equity_curve(test_results).to_csv(os.path.join(out_dir, "equity_curve_test.csv"))

    with open(os.path.join(out_dir, "tuned_params.txt"), "w") as f:
        f.write(str(tuned_params))

    model_path = os.path.join(out_dir, "kalman_ml_model.pkl")
    try:
        import joblib
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
    except Exception as e:
        print(f"Model save failed: {e}", file=sys.stderr)

    print(f"Outputs saved in ./{out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
