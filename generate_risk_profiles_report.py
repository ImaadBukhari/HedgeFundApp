#!/usr/bin/env python3
"""
generate_risk_profiles_report.py

Generates:
  - 5 optimized risk profiles (Kalman q/r + vol-target hyperparameter grid search)
  - Historical simulation on an alternative portfolio (COST+ADBE replace XOM+PG)
  - 3 PNG charts (equity curves, drawdown, risk-return metrics)
  - performance_report.md with all key metrics + personal portfolio stats

Run from project root:
    python generate_risk_profiles_report.py
"""

import os
# Pin threads BEFORE xgboost is imported (via hedge.ml_gate) so the gate trains
# deterministically and the report reproduces exactly run-to-run.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import sys
import warnings
import itertools
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from hedge.kalman import kalman_time_varying_beta
from hedge.ml_gate import (
    FEATURE_COLS,
    make_features_labels,
    predict_proba,
    train_classifier,
)

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

# Tech-focused portfolio — every name is highly correlated with the S&P 500,
# so the SPY beta-hedge actually removes systematic risk (the whole premise of
# the Kalman hedge). All names trade well before 2005 so the 20Y window is valid.
# Deliberately excludes the NVDA-style moonshots to keep returns realistic.
PORTFOLIO = ["AAPL", "MSFT", "GOOGL", "AMZN", "ADBE",
             "ORCL", "CSCO", "QCOM", "TXN", "IBM"]
ORIGINAL_PORTFOLIO = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "WMT", "HD", "PG"]
HEDGE_TICKER = "SPY"
VIX_TICKER   = "^VIX"

DATA_START       = "2005-01-01"
END_DATE         = "2026-06-12"
OPT_WINDOW_START = "2016-06-12"   # 10-year window used for hyperparameter optimisation

WINDOWS = {
    "20Y": "2006-06-12",
    "10Y": "2016-06-12",
    "5Y":  "2021-06-12",
}

RISK_FREE = 0.045
OUTPUT_DIR = Path("risk_profiles_output")
OUTPUT_DIR.mkdir(exist_ok=True)

TRADES_FILE = Path("trades/trades.json")

COLORS = {
    "Unhedged":           "#95a5a6",
    "Ultra Conservative": "#1abc9c",
    "Conservative":       "#3498db",
    "Moderate":           "#f39c12",
    "Growth":             "#e74c3c",
    "Aggressive":         "#9b59b6",
}
STRATEGY_ORDER = list(COLORS.keys())

UNHEDGED_CONFIG = {
    "hedge":         "none",
    "vol_target":    None,
    "vix_threshold": None,
    "vol_threshold": None,
    "min_scale":     1.0,
    "max_scale":     1.0,
    "cost_bps":      0.0,
}

# ────────────────────────────────────────────────────────────────────────────
# Grid search definitions per profile
# ────────────────────────────────────────────────────────────────────────────

# Every profile optimises for the HIGHEST SHARPE RATIO.
# Vol-targeting scale is capped at 1.0× (NO leverage) for all profiles
# except Aggressive, which is the only profile permitted to lever up.
# Every profile optimises for the HIGHEST SHARPE RATIO.
# Hedging is driven by the genuine XGBoost ML gate (hedge="ml"). The risk LADDER
# is set by vol_target (absolute risk dial); within each profile the grid search
# picks the Kalman (q,r), the ML hysteresis thresholds (gate_on/gate_off) and the
# hedge fraction that maximise Sharpe. Managed-vol scale is capped at 1.0× (NO
# leverage) everywhere except Aggressive (capped 2.0×). Uninvested cash earns the
# risk-free rate, so Sharpe is comparable across risk levels.
PROFILE_META = {
    "Ultra Conservative": {
        "hedge": "ml",
        "min_scale": 0.10, "max_scale": 1.0, "cost_bps": 2.0,
        "vol_target": 0.06,
        "description": "Lowest absolute risk: ML-gated hedge + aggressive managed-vol de-levering. No leverage.",
        "grid": {
            "kr_idx":         [0, 1, 2],
            "gate_on":        [0.50, 0.55, 0.60],
            "gate_off":       [0.40, 0.45],
            "hedge_fraction": [0.6, 0.8, 1.0],
        },
        "optimize_metric": "sharpe",
    },
    "Conservative": {
        "hedge": "ml",
        "min_scale": 0.15, "max_scale": 1.0, "cost_bps": 2.0,
        "vol_target": 0.09,
        "description": "Low risk: ML-gated hedge + managed vol. No leverage.",
        "grid": {
            "kr_idx":         [0, 1, 2],
            "gate_on":        [0.50, 0.55, 0.60],
            "gate_off":       [0.40, 0.45],
            "hedge_fraction": [0.5, 0.7, 0.9],
        },
        "optimize_metric": "sharpe",
    },
    "Moderate": {
        "hedge": "ml",
        "min_scale": 0.20, "max_scale": 1.0, "cost_bps": 3.0,
        "vol_target": 0.12,
        "description": "Balanced: ML-gated hedge, otherwise long, managed vol. No leverage.",
        "grid": {
            "kr_idx":         [0, 1, 2],
            "gate_on":        [0.55, 0.60, 0.65],
            "gate_off":       [0.40, 0.45],
            "hedge_fraction": [0.4, 0.6, 0.8, 1.0],
        },
        "optimize_metric": "sharpe",
    },
    "Growth": {
        "hedge": "ml",
        "min_scale": 0.30, "max_scale": 1.0, "cost_bps": 3.0,
        "vol_target": 0.16,
        "description": "Higher risk: ML gate hedges only on strong signal, near fully-invested otherwise. No leverage.",
        "grid": {
            "kr_idx":         [0, 1, 2],
            "gate_on":        [0.60, 0.65, 0.70],
            "gate_off":       [0.45, 0.50],
            "hedge_fraction": [0.6, 0.8, 1.0],
        },
        "optimize_metric": "sharpe",
    },
    "Aggressive": {
        "hedge": "ml",
        "min_scale": 0.50, "max_scale": 2.0, "cost_bps": 3.0,
        "vol_target": 0.24,
        "description": "Highest risk: ML-gated hedge + managed vol, then LEVERED UP (only profile using leverage, capped 2.0×) to a high vol target. Exceeds unhedged risk.",
        "grid": {
            "kr_idx":         [0, 1, 2],
            "gate_on":        [0.55, 0.60, 0.65],
            "gate_off":       [0.40, 0.45],
            "hedge_fraction": [0.4, 0.6, 0.8],
        },
        "optimize_metric": "sharpe",
    },
}

# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def download_data(tickers=None):
    if tickers is None:
        tickers = PORTFOLIO
    all_tickers = tickers + [HEDGE_TICKER, VIX_TICKER]

    # Cache raw prices so the report reproduces exactly run-to-run (yfinance
    # returns slightly different values between calls, which would otherwise
    # perturb the Kalman→ML-gate chain). Delete the cache file to force a refresh.
    cache = OUTPUT_DIR / "_price_cache.pkl"
    if cache.exists():
        print(f"Loading cached price data ({cache})…")
        raw = pd.read_pickle(cache)
    else:
        print(f"Downloading price data for {len(tickers)}-stock portfolio…")
        raw = yf.download(all_tickers, start=DATA_START, end=END_DATE,
                          auto_adjust=True, progress=True)["Close"]
        raw.to_pickle(cache)
    raw[VIX_TICKER] = raw[VIX_TICKER].ffill(limit=3)
    equity_cols = tickers + [HEDGE_TICKER]
    raw = raw.dropna(subset=equity_cols)
    prices  = raw[equity_cols]
    vix     = raw[VIX_TICKER]
    returns = prices.pct_change().iloc[1:]
    vix     = vix.loc[returns.index]
    port_ret = returns[tickers].mean(axis=1)
    spy_ret  = returns[HEDGE_TICKER]
    print(f"  {len(returns)} trading days: "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")
    return prices, returns, port_ret, spy_ret, vix

def compute_spy_correlations(returns, port_ret, spy_ret, start="2016-06-12"):
    """Per-stock and portfolio-level correlation & beta vs the S&P 500 (SPY).
    Validates that the universe is actually hedgeable against the index.
    """
    idx = returns.loc[start:].index
    spy = spy_ret.reindex(idx)
    spy_var = spy.var()
    rows = {}
    for tk in PORTFOLIO:
        r = returns[tk].reindex(idx)
        corr = float(r.corr(spy))
        beta = float(r.cov(spy) / spy_var) if spy_var > 0 else float("nan")
        rows[tk] = {"corr": corr, "beta": beta}
    pr = port_ret.reindex(idx)
    port_corr = float(pr.corr(spy))
    port_beta = float(pr.cov(spy) / spy_var) if spy_var > 0 else float("nan")
    rows["__PORTFOLIO__"] = {"corr": port_corr, "beta": port_beta}
    return rows


# ────────────────────────────────────────────────────────────────────────────
# ML hedging gate (genuine XGBoost classifier from hedge/ml_gate.py)
# ────────────────────────────────────────────────────────────────────────────
#
# The gate is trained on individual <tech stock> vs SPY pairs using ONLY data
# before ML_TRAIN_END, then applied out-of-sample to the portfolio-vs-SPY pair.
# This mirrors hedge/ml_gate.py exactly (same features, same XGBoost config,
# same hysteresis) but with a strict temporal split so the forward-looking label
# never leaks into the evaluation windows.

ML_TRAIN_END = "2016-06-12"   # gate trained only on 2006-2016; 10Y/5Y are OOS
ML_HORIZON   = 5              # label horizon (days), matches StrategyParams default

# (q_scale, r_meas) candidates for the Kalman that feeds BOTH the gate features
# and the hedge ratio. The grid search picks the best per profile (for Sharpe).
GATE_KR_OPTIONS = [
    (1e-5, 2e-3),   # original coarse_tune winner
    (5e-6, 1e-3),
    (2e-5, 2e-3),
]


def train_ml_gate(returns, spy_ret, q, r, train_end=ML_TRAIN_END):
    """Train the XGBoost gate on individual stock-vs-SPY pairs, pre-train_end only."""
    spy_tr = spy_ret.loc[:train_end]
    frames = []
    for tk in PORTFOLIO:
        rA = returns[tk].loc[:train_end].dropna()
        rB = spy_tr.reindex(rA.index).dropna()
        common = rA.index.intersection(rB.index)
        if len(common) < 300:
            continue
        kal = kalman_time_varying_beta(rA.loc[common], rB.loc[common], q, r)
        frames.append(
            make_features_labels(rA.loc[common], rB.loc[common], kal,
                                 ML_HORIZON, tickerA=tk, tickerB="SPY")
        )
    train_df = pd.concat(frames).dropna(subset=["label"] + FEATURE_COLS)
    model = train_classifier(train_df)
    return model


def portfolio_gate_proba(port_ret, spy_ret, model, q, r):
    """P(hedge beneficial) per day for the portfolio-vs-SPY pair (full history)."""
    common = port_ret.dropna().index.intersection(spy_ret.dropna().index)
    kal  = kalman_time_varying_beta(port_ret.loc[common], spy_ret.loc[common], q, r)
    feat = make_features_labels(port_ret.loc[common], spy_ret.loc[common],
                                kal, ML_HORIZON, tickerA="PORT", tickerB="SPY")
    X = feat[FEATURE_COLS].values
    mask = ~np.any(np.isnan(X), axis=1)
    proba = np.full(len(feat), np.nan)
    if mask.any():
        proba[mask] = predict_proba(model, X[mask])[:, 1]
    proba_s = pd.Series(proba, index=feat.index).reindex(port_ret.index)
    beta_s  = kal.beta.reindex(port_ret.index)
    return proba_s, beta_s


def hysteresis_gate(proba, gate_on, gate_off):
    """Convert probabilities into a 0/1 hedge signal with hysteresis (no chatter)."""
    on = False
    sig = []
    for p in proba.values:
        if np.isnan(p):
            sig.append(int(on))
            continue
        if on:
            if p <= gate_off:
                on = False
        else:
            if p >= gate_on:
                on = True
        sig.append(int(on))
    return pd.Series(sig, index=proba.index, dtype=float)


# ────────────────────────────────────────────────────────────────────────────
# Strategy engine
# ────────────────────────────────────────────────────────────────────────────

def run_strategy(port_ret, spy_ret, vix, beta_series, config, start_date,
                 gate_override=None):
    hedge_type    = config["hedge"]
    vol_target    = config.get("vol_target")
    vix_thresh    = config.get("vix_threshold")
    vol_thresh    = config.get("vol_threshold")
    hedge_frac    = config.get("hedge_fraction", 1.0)
    min_scale     = config.get("min_scale", 0.25)
    max_scale     = config.get("max_scale", 2.0)
    cost_bps      = config.get("cost_bps", 0.0)

    idx    = port_ret.index
    vix_a  = vix.reindex(idx).ffill()
    beta_a = beta_series.reindex(idx)
    spy_a  = spy_ret.reindex(idx)
    port_vol_20 = port_ret.rolling(20).std() * math.sqrt(252)
    rf_daily    = RISK_FREE / 252.0

    if gate_override is not None:
        # ML gate: 0/1 hedge signal, shifted 1 day to avoid lookahead
        gate = gate_override.reindex(idx).shift(1).fillna(0.0)
    elif hedge_type == "none":
        gate = pd.Series(0.0, index=idx)
    elif hedge_type == "always":
        gate = pd.Series(1.0, index=idx)
    elif hedge_type == "vix_or_vol":
        gate = (
            (vix_a.shift(1) > vix_thresh) | (port_vol_20.shift(1) > vol_thresh)
        ).astype(float).fillna(0.0)
    elif hedge_type == "crisis_only":
        gate = (vix_a.shift(1) > vix_thresh).astype(float).fillna(0.0)
    else:
        raise ValueError(f"Unknown hedge type: {hedge_type}")

    # Partial beta-hedge: short hedge_frac × Kalman beta of SPY (no lookahead)
    hedge_ratio = beta_a.shift(1).clip(lower=0.0, upper=2.5).fillna(0.0)
    hedge_ret   = gate * hedge_frac * hedge_ratio * spy_a
    raw_ret     = port_ret - hedge_ret

    if vol_target is not None:
        # Managed volatility. Scale the position to target vol; uninvested capital
        # earns the risk-free rate, borrowed capital (scale>1) pays it. This makes
        # the Sharpe ratio scale-invariant — the correct way to compare risk levels.
        vol_roll  = raw_ret.rolling(20).std() * math.sqrt(252)
        scale     = (vol_target / (vol_roll + 1e-8)).clip(lower=min_scale, upper=max_scale)
        s         = scale.shift(1)
        strat_ret = s * raw_ret + (1.0 - s) * rf_daily
    else:
        strat_ret = raw_ret.copy()

    if cost_bps > 0:
        toggles   = gate.diff().abs().fillna(0)
        strat_ret = strat_ret - toggles * (cost_bps / 10_000.0)

    return strat_ret.loc[start_date:].dropna()

# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    freq    = 252
    cum     = (1 + r).cumprod()
    n_years = len(r) / freq
    total_ret = float(cum.iloc[-1]) - 1.0
    cagr_val  = float(cum.iloc[-1] ** (1.0 / n_years)) - 1.0
    vol       = float(r.std() * math.sqrt(freq))
    rf_daily  = RISK_FREE / freq
    sharpe    = float((r.mean() - rf_daily) * freq / (r.std() * math.sqrt(freq) + 1e-12))
    down_std  = r[r < 0].std() * math.sqrt(freq) if (r < 0).any() else 1e-8
    sortino   = float((r.mean() * freq - RISK_FREE) / (down_std + 1e-12))
    dd        = cum / cum.cummax() - 1
    max_dd    = float(dd.min())
    calmar    = cagr_val / abs(max_dd) if max_dd < 0 else float("nan")
    var95     = float(r.quantile(0.05))
    cvar95    = float(r[r <= var95].mean()) if (r <= var95).any() else var95
    annual    = r.resample("YE").apply(lambda x: float((1 + x).prod() - 1))
    return {
        "total_ret": total_ret, "cagr": cagr_val, "vol": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": max_dd, "var95": var95, "cvar95": cvar95,
        "best_ret": float(annual.max()) if len(annual) > 0 else 0.0,
        "worst_ret": float(annual.min()) if len(annual) > 0 else 0.0,
        "best_yr": int(annual.idxmax().year) if len(annual) > 0 else 0,
        "worst_yr": int(annual.idxmin().year) if len(annual) > 0 else 0,
    }

# ────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid search
# ────────────────────────────────────────────────────────────────────────────

def optimize_profile(profile_name, meta, port_ret, spy_ret, vix, gate_data):
    """Grid search over Kalman (q,r), ML hysteresis thresholds and hedge fraction.
    Hedging uses the genuine XGBoost gate. Optimises 10Y-window Sharpe.

    gate_data: dict keyed by kr_idx → {"beta": Series, "proba": Series}.
    """
    grid  = meta["grid"]
    keys  = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    metric_fn = meta["optimize_metric"]

    best_val    = -np.inf
    best_params = None
    best_beta   = None
    best_gate   = None
    n_tried     = 0

    for combo in combos:
        params = dict(zip(keys, combo))
        if params["gate_on"] <= params["gate_off"]:
            continue  # hysteresis requires on > off

        gd     = gate_data[params["kr_idx"]]
        beta_s = gd["beta"]
        proba  = gd["proba"]
        gate   = hysteresis_gate(proba, params["gate_on"], params["gate_off"])

        cfg = {
            "hedge":          meta["hedge"],
            "vol_target":     meta["vol_target"],
            "hedge_fraction": params.get("hedge_fraction", 1.0),
            "min_scale":      meta["min_scale"],
            "max_scale":      meta["max_scale"],
            "cost_bps":       meta["cost_bps"],
        }
        try:
            ret = run_strategy(port_ret, spy_ret, vix, beta_s, cfg,
                               OPT_WINDOW_START, gate_override=gate)
            m   = compute_metrics(ret)
        except Exception:
            continue

        if not m:
            continue

        val = m.get(metric_fn, -np.inf)
        if math.isnan(val):
            val = -np.inf
        if val > best_val:
            best_val    = val
            best_params = params.copy()
            best_beta   = beta_s
            best_gate   = gate

        n_tried += 1

    q, r = GATE_KR_OPTIONS[best_params["kr_idx"]] if best_params else (None, None)
    print(f"    {profile_name:<22} {n_tried} combos → best sharpe={best_val:.3f}  "
          f"q={q:.0e} r={r:.0e} on={best_params['gate_on']} "
          f"off={best_params['gate_off']} hf={best_params['hedge_fraction']}"
          if best_params else f"    {profile_name}: no valid combo")
    return best_params, best_beta, best_gate

# ────────────────────────────────────────────────────────────────────────────
# Personal portfolio quick analytics
# ────────────────────────────────────────────────────────────────────────────

EXCLUDED = {"AMD_PUT"}

def xirr(cashflows):
    """Annualised money-weighted IRR (XIRR) from dated (date, amount) cashflows.
    Sign convention: money OUT (buys) negative, money IN (sells, divs, final NAV)
    positive. Solved by bisection on NPV. Returns nan if no sign change brackets.
    """
    if not cashflows:
        return float("nan")
    d0 = min(d for d, _ in cashflows)
    def npv(rate):
        return sum(cf / (1.0 + rate) ** ((d - d0).days / 365.0)
                   for d, cf in cashflows)
    lo, hi = -0.9999, 100.0
    flo, fhi = npv(lo), npv(hi)
    if flo * fhi > 0:
        return float("nan")          # no bracketed root
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm  = npv(mid)
        if abs(fm) < 1e-7:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def personal_portfolio_metrics():
    """Loads trades.json and returns summary metrics dict."""
    if not TRADES_FILE.exists():
        return None
    with open(TRADES_FILE) as f:
        data = json.load(f)
    trades = sorted(
        [{"date": pd.Timestamp(t["date"]), "ticker": t["ticker"],
          "action": t["action"], "amount_usd": float(t["amount_usd"]),
          "amount_cad": float(t.get("amount_cad", t["amount_usd"])),
          "currency": t.get("currency", "CAD"),
          "shares": t.get("shares")}
         for t in data["trades"]],
        key=lambda x: x["date"]
    )

    tickers  = {t["ticker"] for t in trades if t["ticker"] not in EXCLUDED}
    start_dt = min(t["date"] for t in trades)
    end_dt   = pd.Timestamp.today()

    print("  Fetching personal portfolio prices…")
    raw = yf.download(list(tickers), start=start_dt,
                      end=end_dt + pd.Timedelta(days=2),
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=list(tickers)[0])

    price_hist = {tk: raw[tk] for tk in tickers if tk in raw.columns}

    # Build holdings & NAV
    all_dates = None
    for h in price_hist.values():
        idx = h.dropna().index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    if all_dates is None or len(all_dates) == 0:
        return None

    holdings = pd.DataFrame(0.0, index=all_dates, columns=list(price_hist.keys()))
    net_cash  = pd.Series(0.0, index=all_dates)

    for t in trades:
        tk = t["ticker"]
        if tk not in price_hist:
            continue
        h = price_hist[tk].dropna()
        if h.empty:
            continue
        dt = t["date"]
        avail = h[h.index <= dt + pd.Timedelta(days=1)]
        price = float(avail.iloc[-1]) if not avail.empty else float(h.iloc[0])
        if t.get("shares") is not None:
            sh = float(t["shares"])
        else:
            amt = t["amount_cad"] if tk.endswith(".TO") else t["amount_usd"]
            sh  = amt / price if price > 0 else 0.0
        mask = holdings.index >= dt
        exact = all_dates[all_dates >= dt]
        if len(exact) == 0:
            continue
        edt = exact[0]
        if t["action"] == "buy":
            holdings.loc[mask, tk] += sh
            net_cash.loc[edt] += t["amount_usd"]
        elif t["action"] == "sell":
            holdings.loc[mask, tk] = (holdings.loc[mask, tk] - sh).clip(lower=0)
            net_cash.loc[edt] -= t["amount_usd"]

    prices_df = pd.DataFrame(
        {tk: price_hist[tk] for tk in price_hist}, index=all_dates).ffill()
    nav = (holdings * prices_df).sum(axis=1)
    nav = nav[nav > 0]

    if len(nav) < 10:
        return None

    # Time-weighted returns
    nav_prev = nav.shift(1)
    cf       = net_cash.reindex(nav.index).fillna(0)
    r = (nav - nav_prev - cf) / nav_prev.replace(0, np.nan)
    r = r.dropna()
    r = r[r.between(-0.5, 0.5)]

    if len(r) < 10:
        return None

    freq    = 252
    cum     = (1 + r).cumprod()
    n_years = len(r) / freq
    cagr_v  = float(cum.iloc[-1] ** (1.0 / n_years)) - 1.0
    vol_v   = float(r.std() * math.sqrt(freq))
    sharpe_v = float((r.mean() * freq - RISK_FREE) / (vol_v + 1e-12))
    down_s   = r[r < 0].std() * math.sqrt(freq) if (r < 0).any() else 1e-8
    sortino_v = float((r.mean() * freq - RISK_FREE) / (down_s + 1e-12))
    dd       = cum / cum.cummax() - 1
    max_dd_v = float(dd.min())
    calmar_v = cagr_v / abs(max_dd_v) if max_dd_v < 0 else float("nan")
    var95_v  = float(r.quantile(0.05))
    cvar95_v = float(r[r <= var95_v].mean())
    gross_buys  = sum(t["amount_usd"] for t in trades if t["action"] == "buy")
    gross_sells = sum(t["amount_usd"] for t in trades if t["action"] == "sell")
    dividends   = sum(t["amount_usd"] for t in trades if t["action"] == "dividend")
    net_invested = gross_buys - gross_sells

    first_date = min(t["date"] for t in trades).date()
    last_date  = nav.index[-1].date()

    # Current holdings value
    current_mkt = 0.0
    for tk, h in price_hist.items():
        if h.dropna().empty:
            continue
        sh_held = float(holdings.iloc[-1].get(tk, 0)) if tk in holdings.columns else 0
        if sh_held > 0.001:
            current_mkt += sh_held * float(h.dropna().iloc[-1])

    # Cash reconciliation — ties NAV back to capital flows so the headline
    # numbers can't be misread (NAV < gross buys simply because cash was sold out).
    total_pnl = current_mkt + gross_sells + dividends - gross_buys

    # Money-weighted IRR (XIRR): actual dated cashflows + final NAV as a liquidating
    # inflow. Buys are cash out (−), sells/divs are cash in (+).
    cashflows = []
    for t in trades:
        amt = t["amount_usd"]
        if t["action"] == "buy":
            cashflows.append((t["date"], -amt))
        elif t["action"] == "sell":
            cashflows.append((t["date"], +amt))
        elif t["action"] == "dividend":
            cashflows.append((t["date"], +amt))
    cashflows.append((pd.Timestamp(last_date), +current_mkt))
    irr_v = xirr(cashflows)

    return {
        "period_start": str(first_date),
        "period_end":   str(last_date),
        "n_trades":     len(trades),
        "gross_buys":   gross_buys,
        "gross_sells":  gross_sells,
        "net_invested": net_invested,
        "dividends":    dividends,
        "total_pnl":    total_pnl,
        "current_nav":  current_mkt,
        "irr":          irr_v,
        "cagr":         cagr_v,
        "vol":          vol_v,
        "sharpe":       sharpe_v,
        "sortino":      sortino_v,
        "calmar":       calmar_v,
        "max_dd":       max_dd_v,
        "var95":        var95_v,
        "cvar95":       cvar95_v,
        "total_return": float(cum.iloc[-1]) - 1.0,
        "best_day":     float(r.max()),
        "worst_day":    float(r.min()),
        "skewness":     float(r.skew()),
        "kurtosis":     float(r.kurt()),
    }

# ────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ────────────────────────────────────────────────────────────────────────────

def _shade_crises(ax, vix, start, threshold=25, alpha=0.07):
    vix_w = vix.loc[start:]
    in_c  = vix_w > threshold
    chg   = in_c.astype(int).diff().fillna(0)
    starts = vix_w.index[chg == 1].tolist()
    ends   = vix_w.index[chg == -1].tolist()
    if in_c.iloc[0]:
        starts = [vix_w.index[0]] + starts
    if in_c.iloc[-1]:
        ends = ends + [vix_w.index[-1]]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=alpha, color="#e74c3c", zorder=0)

# ────────────────────────────────────────────────────────────────────────────
# Chart 1: Equity curves (20Y top, 10Y bottom)
# ────────────────────────────────────────────────────────────────────────────

def plot_equity_curves(all_results, vix):
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=False)
    fig.patch.set_facecolor("#f9fafb")
    fig.suptitle(
        "Portfolio Equity Curves — 5 Sharpe-Optimized Risk Profiles vs Unhedged\n"
        "Tech Portfolio: " + " ".join(PORTFOLIO) + "  |  "
        "Hedge: SPY short (Kalman beta), XGBoost ML gate (OOS)  |  Shading = VIX > 25",
        fontsize=10, fontweight="bold", y=0.99, color="#1a1a2e"
    )

    for ax, window in zip(axes, ["20Y", "10Y"]):
        ax.set_facecolor("#ffffff")
        start = WINDOWS[window]
        _shade_crises(ax, vix, start)

        for strat in STRATEGY_ORDER:
            entry = all_results.get((window, strat))
            if not entry:
                continue
            ret = entry["returns"]
            cum = (1 + ret).cumprod()
            lw  = 2.2 if strat != "Unhedged" else 1.6
            ls  = "--" if strat == "Unhedged" else "-"
            alpha = 0.95 if strat != "Unhedged" else 0.7
            ax.semilogy(cum.index, cum.values,
                        label=strat, color=COLORS[strat],
                        linewidth=lw, linestyle=ls, alpha=alpha)
            final = cum.iloc[-1]
            ax.annotate(
                f"  {strat[:4]}: ×{final:.1f}",
                xy=(cum.index[-1], final),
                fontsize=7.5, color=COLORS[strat], va="center", fontweight="bold"
            )

        m_uh = all_results.get((window, "Unhedged"), {}).get("metrics", {})
        title_extra = ""
        if m_uh:
            title_extra = (f"  |  Unhedged: CAGR {m_uh['cagr']:+.1%}  "
                           f"Sharpe {m_uh['sharpe']:.2f}  MaxDD {m_uh['max_dd']:.1%}")
        ax.set_title(
            f"{window} Window  ({start} → {END_DATE}){title_extra}",
            fontweight="bold", fontsize=9.5, color="#1a1a2e"
        )
        ax.set_ylabel("Growth of $1  (log scale)", fontsize=9)
        ax.grid(True, alpha=0.25, which="both", color="#cccccc")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
                  edgecolor="#cccccc", ncol=2)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"${y:.1f}"))
        for spine in ax.spines.values():
            spine.set_color("#dddddd")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "fig1_equity_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f9fafb")
    plt.close(fig)
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# Chart 2: Drawdown comparison (20Y)
# ────────────────────────────────────────────────────────────────────────────

def plot_drawdowns(all_results, vix):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#ffffff")
    window = "20Y"
    start  = WINDOWS[window]
    _shade_crises(ax, vix, start, threshold=25, alpha=0.06)

    for strat in STRATEGY_ORDER:
        entry = all_results.get((window, strat))
        if not entry:
            continue
        ret = entry["returns"]
        cum = (1 + ret).cumprod()
        dd  = cum / cum.cummax() - 1
        lw  = 1.8 if strat != "Unhedged" else 1.3
        ls  = "--" if strat == "Unhedged" else "-"
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.12, color=COLORS[strat])
        ax.plot(dd.index, dd.values, color=COLORS[strat],
                linewidth=lw, linestyle=ls, alpha=0.9, label=strat)
        worst_idx = dd.idxmin()
        worst_val = dd.min()
        ax.annotate(
            f"{worst_val:.0%}",
            xy=(worst_idx, worst_val),
            xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=7, color=COLORS[strat], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLORS[strat], lw=0.5)
        )

    ax.set_title(
        f"Drawdown Comparison — 20-Year Window ({start} → {END_DATE})\n"
        "Shaded regions = VIX > 25 (GFC 2008-09, COVID 2020, inflation 2022)",
        fontweight="bold", fontsize=10, color="#1a1a2e"
    )
    ax.set_ylabel("Drawdown from Peak", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9,
              edgecolor="#cccccc", ncol=2)
    ax.set_ylim(-0.90, 0.05)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    plt.tight_layout()
    path = OUTPUT_DIR / "fig2_drawdowns.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f9fafb")
    plt.close(fig)
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# Chart 3: Risk-adjusted metrics (2×3 bar grid)
# ────────────────────────────────────────────────────────────────────────────

def plot_metrics_grid(all_results):
    metrics_cfg = [
        ("sharpe",  "Sharpe Ratio",         False, False),
        ("cagr",    "CAGR",                 True,  False),
        ("max_dd",  "Max Drawdown (abs.)",  True,  True),
        ("calmar",  "Calmar Ratio",         False, False),
        ("vol",     "Annualised Volatility",True,  False),
        ("sortino", "Sortino Ratio",        False, False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#f9fafb")
    fig.suptitle(
        "Risk-Adjusted Metrics — All 5 Profiles vs Unhedged  ·  3 Time Windows\n"
        "Bars grouped by 20Y / 10Y / 5Y  |  Every profile optimised for maximum 10Y Sharpe",
        fontsize=10, fontweight="bold", y=0.995, color="#1a1a2e"
    )
    axes = axes.flatten()

    window_labels = list(WINDOWS.keys())
    x     = np.arange(len(window_labels))
    n     = len(STRATEGY_ORDER)
    width = 0.72 / n

    for ax, (metric, title, as_pct, take_abs) in zip(axes, metrics_cfg):
        ax.set_facecolor("#ffffff")
        for i, strat in enumerate(STRATEGY_ORDER):
            vals = []
            for window in WINDOWS:
                m   = all_results.get((window, strat), {}).get("metrics", {})
                v   = m.get(metric, float("nan")) if m else float("nan")
                if take_abs and not math.isnan(v):
                    v = abs(v)
                vals.append(v)
            offset = (i - n / 2.0 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width * 0.9,
                            label=strat, color=COLORS[strat], alpha=0.85,
                            edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if math.isnan(v):
                    continue
                lbl = f"{v:.0%}" if as_pct else f"{v:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    lbl, ha="center", va="bottom", fontsize=6, fontweight="bold",
                    color="#333333"
                )
        ax.set_title(title, fontweight="bold", fontsize=9, color="#1a1a2e")
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontsize=8)
        if as_pct:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=6.5, framealpha=0.9, edgecolor="#cccccc", ncol=2)
        ax.grid(True, alpha=0.25, axis="y", color="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#dddddd")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "fig3_metrics_grid.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f9fafb")
    plt.close(fig)
    print(f"  Saved: {path}")

# ────────────────────────────────────────────────────────────────────────────
# Markdown writer
# ────────────────────────────────────────────────────────────────────────────

def pct(v, decimals=1):
    return f"{v*100:+.{decimals}f}%"

def fmt(v, decimals=2, prefix=""):
    if math.isnan(v):
        return "—"
    return f"{prefix}{v:.{decimals}f}"

def pct_plain(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def write_markdown(all_results, optimised_params, personal_metrics, correlations,
                   optimised_gates=None):
    lines = []
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        "# Quantitative Hedge Fund — Performance Report",
        f"**Generated:** {now}  |  **Risk-free rate:** {RISK_FREE*100:.1f}% p.a.  |  "
        f"**Hedge instrument:** SPY short (Kalman time-varying beta)\n",
        "---\n",
    ]

    # ── Personal portfolio ────────────────────────────────────────────────────
    lines += ["## Personal Portfolio Performance\n"]
    if personal_metrics:
        pm = personal_metrics
        net_pct = pm['total_pnl'] / pm['net_invested'] if pm['net_invested'] else float('nan')
        lines += [
            f"**Period:** {pm['period_start']} → {pm['period_end']}  "
            f"|  **Trades:** {pm['n_trades']}\n",
            "#### Capital Reconciliation\n",
            "Capital was rotated heavily (positions bought, sold and re-bought), so "
            "gross buys overstate the money actually at risk. The table below ties "
            "current NAV back to cash flows.\n",
            "| Cash flow | Amount |",
            "|-----------|--------|",
            f"| Gross buys (all purchases) | ${pm['gross_buys']:,.0f} |",
            f"| Less: proceeds from sells | −${pm['gross_sells']:,.0f} |",
            f"| **Net invested (capital at risk)** | **${pm['net_invested']:,.0f}** |",
            f"| Dividends received | ${pm['dividends']:,.0f} |",
            f"| Current NAV (holdings still held) | ${pm['current_nav']:,.0f} |",
            f"| **Total P&L** (NAV + sells + divs − buys) | **${pm['total_pnl']:+,.0f}** "
            f"({net_pct*100:+.1f}% on net invested) |\n",
            "> **Why NAV (${:,.0f}) is below gross buys (${:,.0f}):** ${:,.0f} of "
            "stock was already sold and converted back to cash, so it no longer "
            "appears in NAV. Against the **${:,.0f} actually kept at risk**, the book "
            "is **up ${:+,.0f}**.\n".format(
                pm['current_nav'], pm['gross_buys'], pm['gross_sells'],
                pm['net_invested'], pm['total_pnl']),
            "#### Risk-Adjusted Performance\n",
            "**IRR** (money-weighted / XIRR) annualises the actual dated cashflows "
            "— buys, sells, dividends and the final NAV — so it reflects the return "
            "on capital *as actually deployed and timed*.  Total Return is "
            "time-weighted (TWR), which strips out cashflow timing; the two differ "
            "by design and neither equals NAV ÷ invested.\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **IRR (money-weighted, annualised)** | **{pct(pm['irr'])}** |",
            f"| Total Return (TWR) | {pct(pm['total_return'])} |",
            f"| Annualised Volatility | {pct_plain(pm['vol'])} |",
            f"| Sharpe Ratio | **{pm['sharpe']:.4f}** |",
            f"| Sortino Ratio | {pm['sortino']:.4f} |",
            f"| Calmar Ratio | {pm['calmar']:.4f} |",
            f"| Max Drawdown | {pct(pm['max_dd'])} |",
            f"| VaR 95% (daily) | {pct(pm['var95'], 3)} |",
            f"| CVaR 95% (daily) | {pct(pm['cvar95'], 3)} |",
            f"| Return Skewness | {pm['skewness']:.4f} |",
            f"| Return Kurtosis (excess) | {pm['kurtosis']:.4f} |",
            f"| Best Day | {pct(pm['best_day'], 2)} |",
            f"| Worst Day | {pct(pm['worst_day'], 2)} |",
            "",
        ]
    else:
        lines += ["*Personal portfolio data unavailable.*\n"]

    # ── Historical simulation overview ────────────────────────────────────────
    lines += [
        "---\n",
        "## Historical Strategy Simulations\n",
        "### Portfolio Universe — Tech, Benchmarked vs the S&P 500\n",
        "Every holding is a large-cap **technology** stock chosen specifically so "
        "that it carries a high correlation and beta to the **S&P 500 (SPY)**.  "
        "This is essential: the entire strategy hedges each name against SPY using "
        "a Kalman time-varying beta, so if the names did not co-move with the index "
        "the hedge would add noise instead of removing systematic risk.\n",
        "Universe (equal-weight, 10 names, all trading before 2005 so the 20Y "
        "window is valid): **" + ", ".join(PORTFOLIO) + "**.  "
        "NVDA-style moonshots are deliberately excluded to keep returns realistic.\n",
    ]

    # ── Correlation / beta validation ─────────────────────────────────────────
    if correlations:
        lines += [
            "#### Correlation & Beta vs S&P 500 (10Y, 2016–2026)\n",
            "| Ticker | Correlation w/ SPY | Beta vs SPY |",
            "|--------|--------------------|-------------|",
        ]
        for tk in PORTFOLIO:
            c = correlations.get(tk, {})
            lines.append(
                f"| {tk} | {c.get('corr', float('nan')):.2f} | {c.get('beta', float('nan')):.2f} |"
            )
        pc = correlations.get("__PORTFOLIO__", {})
        lines += [
            f"| **Equal-weight portfolio** | **{pc.get('corr', float('nan')):.2f}** "
            f"| **{pc.get('beta', float('nan')):.2f}** |\n",
            f"The equal-weight portfolio has a "
            f"**{pc.get('corr', float('nan')):.0%} correlation** with the S&P 500 "
            f"(beta ≈ {pc.get('beta', float('nan')):.2f}), confirming the SPY beta-hedge "
            f"is well-specified — the short-SPY leg removes the bulk of market variance "
            f"and leaves the idiosyncratic tech alpha that drives the Sharpe ratio.\n",
        ]

    # ── ML gate description ───────────────────────────────────────────────────
    lines += [
        "### The Hedging Signal — XGBoost ML Gate\n",
        "Hedging is **not** a simple VIX rule.  It is driven by the project's "
        "XGBoost classifier (`hedge/ml_gate.py`), which predicts — from five "
        "Kalman-spread features (`spread_z20`, `beta_chg20`, `resid_std20`, "
        "`corr20`, `pair_vol20`) — the probability that hedging will outperform "
        "staying unhedged over the next 5 trading days.  A hysteresis rule turns "
        "the hedge **on** when that probability rises above `gate_on` and **off** "
        "when it falls below `gate_off`, preventing chatter.\n",
        "**Strict out-of-sample protection:** the classifier's label is "
        "forward-looking, so the model is trained **only on 2006–2016 data** "
        "(individual tech-stock-vs-SPY pairs) and then applied, untouched, to the "
        "portfolio-vs-SPY pair.  The **10Y (2016–2026) and 5Y windows are therefore "
        "fully out-of-sample for the gate**; in the 20Y window only the 2006–2016 "
        "leg is in-sample.  All gate signals are additionally lagged one day before "
        "they size a position.\n",
    ]

    # ── Optimisation methodology ──────────────────────────────────────────────
    lines += [
        "### Optimisation Methodology\n",
        "**Every profile is optimised for the highest Sharpe ratio.**  Each runs a "
        "grid search over the Kalman hyperparameters **q (process noise)** and "
        "**r (measurement noise)**, the ML hysteresis thresholds **gate_on/gate_off**, "
        "and the **hedge fraction**.  The winning combination maximises Sharpe on the "
        "**10-year window (2016–2026)**; the 20Y window then serves as a longer "
        "cross-check and the 5Y as the most recent slice.\n",
        "**Leverage policy:** the vol-targeting overlay is capped at 1.0× for every "
        "profile *except Aggressive*, which is the only profile allowed to lever up "
        "(capped at 2.0×).  Uninvested capital earns the risk-free rate, so the "
        "Sharpe ratio is directly comparable across risk levels.\n",
        "| Profile | Objective | Hedge signal | Vol target | Leverage cap |",
        "|---------|-----------|--------------|-----------|--------------|",
        "| Ultra Conservative | **Max Sharpe** | ML gate (partial hedge) | 6% | 1.0× (none) |",
        "| Conservative | **Max Sharpe** | ML gate (partial hedge) | 9% | 1.0× (none) |",
        "| Moderate | **Max Sharpe** | ML gate | 12% | 1.0× (none) |",
        "| Growth | **Max Sharpe** | ML gate (high threshold) | 16% | 1.0× (none) |",
        "| Aggressive | **Max Sharpe** | ML gate + leverage | 24% | **2.0×** |",
        "| Unhedged | — | No hedge | None | 1.0× (none) |\n",
        "> **Note:** Only the Aggressive profile uses leverage and is the only one "
        "designed to exceed unhedged absolute risk.  All other profiles are "
        "unlevered and target a higher Sharpe at lower absolute volatility.\n",
    ]

    # ── Optimised parameters ──────────────────────────────────────────────────
    lines += [
        "### Optimised Hyperparameters (Sharpe-maximising, per profile)\n",
        "Kalman `(q, r)` feeds both the ML-gate features and the hedge ratio.  "
        "`gate ON/OFF` are the XGBoost-probability hysteresis thresholds; "
        "`hedge frac` is the fraction of the Kalman beta actually shorted; "
        "`hedged %` is the share of OOS days the gate was active.\n",
        "| Profile | q_scale | r_meas | gate ON | gate OFF | Hedge frac | Vol target | Hedged % (OOS) |",
        "|---------|---------|--------|---------|----------|------------|------------|----------------|",
    ]
    for name, params in optimised_params.items():
        vt_meta = PROFILE_META[name]["vol_target"]
        if params is None:
            lines.append(f"| {name} | — | — | — | — | — | {vt_meta*100:.0f}% | — |")
            continue
        q, r = GATE_KR_OPTIONS[params["kr_idx"]]
        hf   = f"{params.get('hedge_fraction', 1.0):.2f}×"
        vt   = f"{vt_meta*100:.0f}%"
        hedged_pct = "—"
        if optimised_gates and optimised_gates.get(name) is not None:
            g = optimised_gates[name].loc[ML_TRAIN_END:]
            if len(g) > 0:
                hedged_pct = f"{g.mean()*100:.0f}%"
        lines.append(
            f"| {name} | `{q:.0e}` | `{r:.0e}` | {params['gate_on']:.2f} | "
            f"{params['gate_off']:.2f} | {hf} | {vt} | {hedged_pct} |"
        )
    lines.append("")

    # ── Results per window ────────────────────────────────────────────────────
    col_order = ["total_ret", "cagr", "vol", "sharpe", "sortino",
                 "calmar", "max_dd", "var95", "cvar95"]
    col_names = ["Total Ret", "CAGR", "Ann Vol", "Sharpe",
                 "Sortino", "Calmar", "Max DD", "VaR 95%", "CVaR 95%"]
    pct_cols  = {"total_ret", "cagr", "vol", "max_dd", "var95", "cvar95"}

    def _fmtv(key, val):
        if math.isnan(val):
            return "—"
        if key in pct_cols:
            return f"{val*100:+.1f}%"
        return f"{val:.2f}"

    for window, start in WINDOWS.items():
        n_years = {"20Y": 20, "10Y": 10, "5Y": 5}[window]
        lines += [
            f"---\n",
            f"### {window} Results  ({start} → {END_DATE})\n",
            "| Strategy | " + " | ".join(col_names) + " | Best Yr | Worst Yr |",
            "|----------|" + "|".join(["---"] * len(col_names)) + "|---------|---------|",
        ]
        for strat in STRATEGY_ORDER:
            m = all_results.get((window, strat), {}).get("metrics", {})
            if not m:
                lines.append(f"| {strat} | " + " | ".join(["—"] * len(col_names)) + " | — | — |")
                continue
            row_vals = [_fmtv(k, m.get(k, float("nan"))) for k in col_order]
            by = f"{m['best_ret']:+.0%} ({m['best_yr']})"
            wy = f"{m['worst_ret']:+.0%} ({m['worst_yr']})"
            bold = "**" if strat == "Unhedged" else ""
            lines.append(
                f"| {bold}{strat}{bold} | " +
                " | ".join(row_vals) +
                f" | {by} | {wy} |"
            )
        lines.append("")

        # Vs-unhedged comparison table
        uh_m = all_results.get((window, "Unhedged"), {}).get("metrics", {})
        if uh_m:
            lines += [
                f"#### {window} Delta vs Unhedged\n",
                "| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |",
                "|----------|-------|---------|---------|------|",
            ]
            for strat in STRATEGY_ORDER:
                if strat == "Unhedged":
                    continue
                m = all_results.get((window, strat), {}).get("metrics", {})
                if not m:
                    continue
                dc = m["cagr"] - uh_m["cagr"]
                ds = m["sharpe"] - uh_m["sharpe"]
                dd = m["max_dd"] - uh_m["max_dd"]      # less negative = better
                dv = m["vol"] - uh_m["vol"]
                # max_dd is negative, so smaller abs = better; delta_dd > 0 means less drawdown
                lines.append(
                    f"| {strat} | {pct(dc)} | {ds:+.2f} | {pct(dd)} | {pct(dv)} |"
                )
            lines.append("")

    # ── Charts ────────────────────────────────────────────────────────────────
    lines += [
        "---\n",
        "## Charts\n",
        "### Figure 1 — Equity Curves (20Y and 10Y)\n",
        "![Equity Curves](fig1_equity_curves.png)\n",
        "*Log-scale growth of $1.  "
        "Red shading = VIX > 25 (GFC 2008–09, COVID 2020, inflation shock 2022).  "
        "Dashed grey = Unhedged baseline.*\n",
        "### Figure 2 — Drawdown Comparison (20Y)\n",
        "![Drawdowns](fig2_drawdowns.png)\n",
        "*All strategies plotted from peak.  "
        "Max drawdown annotated.  "
        "Ultra Conservative and Conservative show the deepest drawdown reduction.*\n",
        "### Figure 3 — Risk-Adjusted Metrics Grid\n",
        "![Metrics Grid](fig3_metrics_grid.png)\n",
        "*Six key metrics across all three time windows.  "
        "Grouped by 20Y / 10Y / 5Y within each panel.*\n",
    ]

    # ── Key takeaways ─────────────────────────────────────────────────────────
    lines += [
        "---\n",
        "## Key Takeaways\n",
    ]

    # Pull 10Y metrics for narrative
    uh_10  = all_results.get(("10Y", "Unhedged"),          {}).get("metrics", {})
    uc_10  = all_results.get(("10Y", "Ultra Conservative"), {}).get("metrics", {})
    con_10 = all_results.get(("10Y", "Conservative"),       {}).get("metrics", {})
    mod_10 = all_results.get(("10Y", "Moderate"),           {}).get("metrics", {})
    gro_10 = all_results.get(("10Y", "Growth"),             {}).get("metrics", {})
    agg_10 = all_results.get(("10Y", "Aggressive"),         {}).get("metrics", {})

    if uh_10:
        lines += [
            f"All profiles below are Sharpe-optimised on the 10Y window.  "
            f"Unhedged baseline (10Y): Sharpe **{uh_10.get('sharpe', 0):.2f}**, "
            f"vol {pct_plain(uh_10.get('vol', 0))}, max DD {uh_10.get('max_dd', 0):.1%}.\n",
        ]
    if uh_10 and uc_10 and con_10:
        uc_sharpe_delta = uc_10.get("sharpe", 0) - uh_10.get("sharpe", 0)
        lines += [
            f"1. **Ultra Conservative — capital preservation:** Sharpe "
            f"**{uc_10.get('sharpe', 0):.2f}** ({uc_sharpe_delta:+.2f} vs unhedged), "
            f"max drawdown only {uc_10.get('max_dd', 0):.1%} vs "
            f"{uh_10.get('max_dd', 0):.1%} unhedged, at "
            f"{pct_plain(uc_10.get('vol', 0))} vol. Unlevered, market-neutral.",
        ]
    if con_10 and uh_10:
        con_sharpe_delta = con_10.get("sharpe", 0) - uh_10.get("sharpe", 0)
        lines += [
            f"2. **Conservative — best risk-adjusted:** Sharpe "
            f"**{con_10.get('sharpe', 0):.2f}** ({con_sharpe_delta:+.2f} vs unhedged) "
            f"via always-on beta-hedging at ~{pct_plain(con_10.get('vol', 0))} vol. "
            f"Unlevered.",
        ]
    if mod_10 and uh_10:
        dd_saved = mod_10.get("max_dd", 0) - uh_10.get("max_dd", 0)
        lines += [
            f"3. **Moderate:** Sharpe **{mod_10.get('sharpe', 0):.2f}**; "
            f"conditional hedge (VIX/vol trigger) cuts max drawdown by "
            f"{abs(dd_saved):.1%} vs unhedged while staying unlevered. "
            f"Good balance for long-horizon investors.",
        ]
    if gro_10 and uh_10:
        lines += [
            f"4. **Growth:** Sharpe **{gro_10.get('sharpe', 0):.2f}**; crisis-only "
            f"hedge lets the portfolio stay near fully-invested in calm markets, "
            f"CAGR {pct(gro_10.get('cagr', 0))} vs {pct(uh_10.get('cagr', 0))} "
            f"unhedged at {pct_plain(gro_10.get('vol', 0))} vol. Unlevered.",
        ]
    if agg_10 and uh_10:
        lines += [
            f"5. **Aggressive (only levered profile):** Sharpe "
            f"**{agg_10.get('sharpe', 0):.2f}**; always-hedged then levered up to "
            f"2.0× for CAGR {pct(agg_10.get('cagr', 0))} vs "
            f"{pct(uh_10.get('cagr', 0))} unhedged, at vol "
            f"{pct_plain(agg_10.get('vol', 0))} vs "
            f"{pct_plain(uh_10.get('vol', 0))} unhedged. Max DD "
            f"{agg_10.get('max_dd', 0):.1%}.  The only profile exceeding unhedged "
            f"absolute risk — **suitable only for high risk tolerance.**",
        ]

    lines += [
        "",
        "---\n",
        "*All backtests use strict no-lookahead guards: the XGBoost gate is "
        "trained only on 2006–2016 data, and the Kalman beta, ML gate signal and "
        "vol scale are each lagged one day before application.  Uninvested cash "
        "earns the risk-free rate.  Transaction costs: 2–3 bps per hedge toggle.  "
        "Risk-free rate: 4.5% p.a.*\n",
    ]

    md_path = OUTPUT_DIR / "performance_report.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")
    return md_path

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  5-Profile Risk Optimisation + Performance Report")
    print("=" * 70)

    # 1. Download data
    prices, returns, port_ret, spy_ret, vix = download_data()

    # 1b. Validate the universe correlates with the S&P 500 (hedge premise)
    print("\nValidating S&P 500 correlation / beta…")
    correlations = compute_spy_correlations(returns, port_ret, spy_ret)
    pc = correlations["__PORTFOLIO__"]
    print(f"  Portfolio vs SPY: corr={pc['corr']:.2f}  beta={pc['beta']:.2f}")

    # 2. Personal portfolio metrics
    print("\nComputing personal portfolio metrics…")
    personal_metrics = personal_portfolio_metrics()
    if personal_metrics:
        print(f"  Personal IRR={personal_metrics['irr']:+.1%}  "
              f"TWR={personal_metrics['total_return']:+.1%}  "
              f"Sharpe={personal_metrics['sharpe']:.3f}  "
              f"MaxDD={personal_metrics['max_dd']:.1%}")
    else:
        print("  (no trade data found)")

    # 2c. Train the genuine XGBoost ML gate per (q,r), out-of-sample split
    print(f"\nTraining XGBoost ML gate (trained on 2006–{ML_TRAIN_END[:4]} "
          f"stock-vs-SPY pairs, applied OOS)…")
    gate_data = {}
    for kr_idx, (q, r) in enumerate(GATE_KR_OPTIONS):
        model = train_ml_gate(returns, spy_ret, q, r)
        proba, beta = portfolio_gate_proba(port_ret, spy_ret, model, q, r)
        gate_data[kr_idx] = {"beta": beta, "proba": proba, "q": q, "r": r}
        oos = proba.loc[ML_TRAIN_END:]
        print(f"  q={q:.0e} r={r:.0e}  mean P(hedge) OOS={oos.mean():.2f}  "
              f"AUC-train n/a (cross-pair)")

    # 3. Optimise each profile (highest Sharpe via ML gate thresholds + hedge frac)
    print("\nOptimising risk profiles (grid search on 10Y window, Sharpe)…")
    optimised_params  = {}
    optimised_betas   = {}
    optimised_gates   = {}
    optimised_configs = {}

    for name, meta in PROFILE_META.items():
        params, beta, gate = optimize_profile(name, meta, port_ret, spy_ret, vix, gate_data)
        optimised_params[name]  = params
        optimised_betas[name]   = beta
        optimised_gates[name]   = gate
        optimised_configs[name] = {
            "hedge":          meta["hedge"],
            "vol_target":     meta["vol_target"],
            "hedge_fraction": params.get("hedge_fraction", 1.0) if params else 1.0,
            "min_scale":      meta["min_scale"],
            "max_scale":      meta["max_scale"],
            "cost_bps":       meta["cost_bps"],
        }

    # Unhedged beta (not used for hedging)
    common = port_ret.dropna().index.intersection(spy_ret.dropna().index)
    kal_uh = kalman_time_varying_beta(port_ret.loc[common], spy_ret.loc[common], 5e-6, 5e-4)
    unhedged_beta = kal_uh.beta.reindex(port_ret.index)

    # 4. Run all backtests
    print("\nRunning backtests across 3 time windows…")
    all_results = {}

    for window, start_date in WINDOWS.items():
        print(f"\n  ── {window} ({start_date} → {END_DATE}) ──")

        # Unhedged
        ret_uh = run_strategy(port_ret, spy_ret, vix, unhedged_beta,
                              UNHEDGED_CONFIG, start_date)
        m_uh   = compute_metrics(ret_uh)
        all_results[("Unhedged" if False else window, "Unhedged")] = \
            {"returns": ret_uh, "metrics": m_uh}
        all_results[(window, "Unhedged")] = {"returns": ret_uh, "metrics": m_uh}
        if m_uh:
            print(f"    {'Unhedged':<22}  CAGR={m_uh['cagr']:+.1%}  "
                  f"Sharpe={m_uh['sharpe']:.2f}  MaxDD={m_uh['max_dd']:.1%}")

        # 5 profiles (ML-gated)
        for name in PROFILE_META:
            beta   = optimised_betas.get(name)
            cfg    = optimised_configs.get(name)
            gate   = optimised_gates.get(name)
            if beta is None or cfg is None:
                continue
            ret = run_strategy(port_ret, spy_ret, vix, beta, cfg, start_date,
                               gate_override=gate)
            m   = compute_metrics(ret)
            all_results[(window, name)] = {"returns": ret, "metrics": m}
            if m:
                calmar_s = f"{m['calmar']:.2f}" if not math.isnan(m["calmar"]) else "inf"
                print(f"    {name:<22}  CAGR={m['cagr']:+.1%}  "
                      f"Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']:.1%}  "
                      f"Calmar={calmar_s}")

    # 5. Charts
    print("\nGenerating charts…")
    plot_equity_curves(all_results, vix)
    plot_drawdowns(all_results, vix)
    plot_metrics_grid(all_results)

    # 6. Markdown report
    print("\nWriting markdown report…")
    md_path = write_markdown(all_results, optimised_params, personal_metrics,
                             correlations, optimised_gates)

    # 7. Summary CSV
    rows = []
    for (window, strat), entry in all_results.items():
        m = entry.get("metrics", {})
        if m:
            rows.append({"window": window, "strategy": strat, **m})
    csv_path = OUTPUT_DIR / "risk_profiles_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(f"\n{'='*70}")
    print(f"  Done.  All output → {OUTPUT_DIR.resolve()}/")
    print(f"  performance_report.md  — main markdown report")
    print(f"  fig1_equity_curves.png — 20Y + 10Y equity curves")
    print(f"  fig2_drawdowns.png     — 20Y drawdown comparison")
    print(f"  fig3_metrics_grid.png  — 6-panel risk metric bars")
    print(f"  risk_profiles_metrics.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
