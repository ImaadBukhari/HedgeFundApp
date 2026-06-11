# HedgeFundApp — Codebase Overview

## What this is

A quantitative finance toolkit with two distinct workstreams:

1. **Pairs trading / equity hedging** — uses a Kalman filter to estimate a time-varying hedge ratio (beta) between two correlated stocks, then uses an XGBoost classifier to decide when to actually activate the hedge.
2. **Options-implied regime detection** — extracts risk-neutral probability densities from options prices (Breeden-Litzenberger) to classify the market as stressed, calm, or risk-on.

The two workstreams currently run independently but are designed to complement each other: regime signals could gate or adjust the pair-hedge ratios.

---

## The `hedge/` package — core logic

| File | What it does |
|---|---|
| `kalman.py` | Kalman filter that tracks time-varying alpha/beta between two return series. Also defines `StrategyParams` (shared config) and `KalmanResult`. |
| `ml_gate.py` | XGBoost classifier that learns when hedging is profitable. Takes Kalman output, builds features (spread z-score, beta change, rolling correlation, etc.), and produces a hedge ON/OFF signal with hysteresis. Also contains backtesting logic with vol-targeting and transaction costs. |
| `regime.py` | Options-implied regime classification. Consumes densities from `breeden_litzenberger.py`, computes stress/calm/risk-on scores, and can batch-process a directory of options snapshots. |
| `breeden_litzenberger.py` | Extracts a risk-neutral probability density from call prices via the Breeden-Litzenberger formula (second derivative of call price w.r.t. strike). Includes smoothing and quality checks. |

**`kalman.py` → `ml_gate.py`** is a direct dependency (ml_gate imports `KalmanResult` and `StrategyParams`).  
**`breeden_litzenberger.py` → `regime.py`** is a direct dependency (regime imports `extract_implied_density`).

---

## Top-level scripts

| File | What it does |
|---|---|
| `kalman_pairs_strategy.py` | End-to-end runner: downloads prices via yfinance, selects correlated pairs, runs Kalman + ML training + backtest, saves results/equity curves to `kalman_ml_pairs_output/`. |
| `analyze_live.py` | Fetches live options for any ticker via yfinance and runs Breeden-Litzenberger + regime scoring. Produces a distribution plot. Entry point: `python analyze_live.py --ticker AAPL --dte 30`. |
| `analyze_distributions.py` | Compares implied distributions across historical SPY snapshots (stored as `.csv.gz` files). Identifies the most-stressed vs calmest dates. Requires local data in `data/opra_spy_snapshots/snapshots/`. |
| `viz_regime_heatmap.py` | Visualization utility — likely plots regime scores across dates as a heatmap (uses the `hedge/regime` batch output). |
| `data_downloader.py` | Downloads and saves options snapshot data for historical analysis. |

---

## How the pieces fit together toward your goal

**Goal: advise on hedging ratios for stock portfolios.**

The building blocks are mostly already here:

- **Hedge ratio** → `kalman.py` outputs a time-varying `beta` for any stock pair. This is the hedge ratio.
- **When to hedge** → `ml_gate.py` decides whether the hedge should be on or off based on recent spread behaviour.
- **Market regime context** → `regime.py` + `breeden_litzenberger.py` read live options to classify whether markets are stressed/calm — this could adjust how aggressively you hedge.

What's missing for an advisory tool:
- A function that takes an arbitrary portfolio (not just a pair) and recommends hedge ratios per position.
- A way to surface regime scores alongside hedge ratio recommendations in a single output.
- The backend (`backend/`) exists (FastAPI + a database) but is largely a skeleton — the quant logic in `hedge/` is not yet wired into any API endpoints.
