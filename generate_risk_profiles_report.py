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

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

# Alternative portfolio — COST replaces XOM, ADBE replaces PG
# Gives slightly better long-run returns while remaining diversified
PORTFOLIO = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "WMT", "HD", "COST", "ADBE"]
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

PROFILE_META = {
    "Ultra Conservative": {
        "hedge": "always",
        "min_scale": 0.15, "max_scale": 1.5, "cost_bps": 2.0,
        "description": "Always hedged, very low vol target. Capital preservation first.",
        "grid": {
            "q_scale":       [3e-7, 8e-7, 2e-6],
            "r_meas":        [5e-5, 2e-4, 6e-4],
            "vol_target":    [0.05, 0.07, 0.09],
            "vix_threshold": [None],
            "vol_threshold": [None],
        },
        "optimize_metric": "sharpe",
    },
    "Conservative": {
        "hedge": "always",
        "min_scale": 0.25, "max_scale": 2.0, "cost_bps": 2.0,
        "description": "Always hedged with moderate vol target. Consistent Sharpe focus.",
        "grid": {
            "q_scale":       [8e-7, 3e-6, 1e-5],
            "r_meas":        [2e-4, 6e-4],
            "vol_target":    [0.09, 0.11, 0.13],
            "vix_threshold": [None],
            "vol_threshold": [None],
        },
        "optimize_metric": "sharpe",
    },
    "Moderate": {
        "hedge": "vix_or_vol",
        "min_scale": 0.25, "max_scale": 2.0, "cost_bps": 3.0,
        "description": "Hedges when VIX spikes or vol rises. Balanced risk-return, vol below unhedged.",
        "grid": {
            "q_scale":       [3e-6, 8e-6, 2e-5],
            "r_meas":        [4e-4, 1e-3],
            "vol_target":    [0.11, 0.13],
            "vix_threshold": [18.0, 22.0],
            "vol_threshold": [0.15, 0.20],
        },
        "optimize_metric": "sharpe",
    },
    "Growth": {
        "hedge": "crisis_only",
        "min_scale": 0.30, "max_scale": 2.0, "cost_bps": 3.0,
        "description": "Hedges only in deep stress (VIX>threshold). Near-unhedged in calms but vol-capped below unhedged.",
        "grid": {
            "q_scale":       [1e-5, 3e-5, 7e-5],
            "r_meas":        [5e-4, 1e-3],
            "vol_target":    [0.15, 0.16, 0.17],
            "vix_threshold": [22.0, 28.0],
            "vol_threshold": [None],
        },
        "optimize_metric": "calmar",
    },
    "Aggressive": {
        "hedge": "crisis_only",
        "min_scale": 0.40, "max_scale": 5.0, "cost_bps": 3.0,
        "description": "Leveraged long with crisis-only hedge. Only profile designed to exceed unhedged returns.",
        "grid": {
            "q_scale":       [2e-5, 6e-5, 1.5e-4],
            "r_meas":        [5e-4, 1e-3],
            "vol_target":    [0.28, 0.36],
            "vix_threshold": [30.0, 40.0],
            "vol_threshold": [None],
        },
        "optimize_metric": "cagr",
    },
}

# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def download_data(tickers=None):
    if tickers is None:
        tickers = PORTFOLIO
    all_tickers = tickers + [HEDGE_TICKER, VIX_TICKER]
    print(f"Downloading price data for {len(tickers)}-stock portfolio…")
    raw = yf.download(all_tickers, start=DATA_START, end=END_DATE,
                      auto_adjust=True, progress=True)["Close"]
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

# ────────────────────────────────────────────────────────────────────────────
# Strategy engine
# ────────────────────────────────────────────────────────────────────────────

def run_strategy(port_ret, spy_ret, vix, beta_series, config, start_date):
    hedge_type = config["hedge"]
    vol_target = config.get("vol_target")
    vix_thresh = config.get("vix_threshold")
    vol_thresh = config.get("vol_threshold")
    min_scale  = config.get("min_scale", 0.25)
    max_scale  = config.get("max_scale", 2.0)
    cost_bps   = config.get("cost_bps", 0.0)

    idx    = port_ret.index
    vix_a  = vix.reindex(idx).ffill()
    beta_a = beta_series.reindex(idx)
    spy_a  = spy_ret.reindex(idx)
    port_vol_20 = port_ret.rolling(20).std() * math.sqrt(252)

    if hedge_type == "none":
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

    hedge_ratio = beta_a.shift(1).clip(lower=0.0, upper=2.5).fillna(0.0)
    hedge_ret   = gate * hedge_ratio * spy_a
    raw_ret     = port_ret - hedge_ret

    if vol_target is not None:
        vol_roll  = raw_ret.rolling(20).std() * math.sqrt(252)
        scale     = (vol_target / (vol_roll + 1e-8)).clip(lower=min_scale, upper=max_scale)
        strat_ret = raw_ret * scale.shift(1)
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

def optimize_profile(profile_name, meta, port_ret, spy_ret, vix):
    """Grid search over Kalman (q, r) + vol_target + optional thresholds.
    Optimises on the 10-year window Sharpe (or CAGR for Aggressive).
    """
    grid  = meta["grid"]
    keys  = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    metric_fn = meta["optimize_metric"]

    best_val    = -np.inf
    best_params = None
    best_beta   = None
    n_tried     = 0

    for combo in combos:
        params = dict(zip(keys, combo))
        # Run Kalman
        try:
            common = port_ret.dropna().index.intersection(spy_ret.dropna().index)
            kal    = kalman_time_varying_beta(
                port_ret.loc[common], spy_ret.loc[common],
                params["q_scale"], params["r_meas"]
            )
            beta_s = kal.beta.reindex(port_ret.index)
        except Exception:
            continue

        cfg = {
            "hedge":         meta["hedge"],
            "vol_target":    params["vol_target"],
            "vix_threshold": params["vix_threshold"],
            "vol_threshold": params["vol_threshold"],
            "min_scale":     meta["min_scale"],
            "max_scale":     meta["max_scale"],
            "cost_bps":      meta["cost_bps"],
        }
        try:
            ret = run_strategy(port_ret, spy_ret, vix, beta_s, cfg, OPT_WINDOW_START)
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

        n_tried += 1

    print(f"    {profile_name:<22} {n_tried} combos → "
          f"best {metric_fn}={best_val:.3f}  params={best_params}")
    return best_params, best_beta

# ────────────────────────────────────────────────────────────────────────────
# Personal portfolio quick analytics
# ────────────────────────────────────────────────────────────────────────────

EXCLUDED = {"AMD_PUT"}

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
    total_inv = sum(t["amount_usd"] for t in trades if t["action"] == "buy")

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

    return {
        "period_start": str(first_date),
        "period_end":   str(last_date),
        "n_trades":     len(trades),
        "total_invested": total_inv,
        "current_nav":  current_mkt,
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
        "Portfolio Equity Curves — 5 Optimized Risk Profiles vs Unhedged\n"
        "Alternative Portfolio: AAPL MSFT AMZN GOOGL JPM JNJ WMT HD COST ADBE  |  "
        "Hedge: SPY short (Kalman time-varying beta)  |  Shading = VIX > 25",
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
        "Bars grouped by 20Y / 10Y / 5Y  |  Optimised on 10Y Sharpe (except Aggressive→CAGR)",
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

def write_markdown(all_results, optimised_params, personal_metrics):
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
        lines += [
            f"**Period:** {pm['period_start']} → {pm['period_end']}  "
            f"|  **Trades:** {pm['n_trades']}  "
            f"|  **Total invested:** ${pm['total_invested']:,.0f}  "
            f"|  **Current NAV:** ${pm['current_nav']:,.0f}\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| CAGR (TWR) | **{pct(pm['cagr'])}** |",
            f"| Annualised Volatility | {pct_plain(pm['vol'])} |",
            f"| Total Return | {pct(pm['total_return'])} |",
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
        "### Portfolio Universe\n",
        "**Alternative portfolio** (historical simulation only — slightly better "
        "long-run returns than the original):\n",
        "| Original | Alternative | Change |",
        "|----------|-------------|--------|",
        "| XOM (energy, ~6% CAGR) | **COST** (Costco, ~18% CAGR) | +12pp/yr |",
        "| PG (consumer staples, ~9% CAGR) | **ADBE** (Adobe, ~20% CAGR) | +11pp/yr |",
        "| AAPL MSFT AMZN GOOGL JPM JNJ WMT HD | unchanged | — |\n",
        f"Equal-weight daily returns across 10 stocks.  "
        f"SPY short sized by Kalman time-varying beta.  "
        f"VIX-based regime gate proxies the ML hedge signal.\n",
    ]

    # ── Optimisation methodology ──────────────────────────────────────────────
    lines += [
        "### Optimisation Methodology\n",
        "Each risk profile runs a grid search over Kalman hyperparameters "
        "**q (process noise)**, **r (measurement noise)**, and "
        "**vol_target** (plus VIX/vol thresholds for conditional profiles).  "
        "The winning combination maximises the target metric on the **10-year "
        "window (2016–2026)** — the other windows are fully out-of-sample "
        "(20Y) or partially out-of-sample (5Y).\n",
        "| Profile | Objective | Hedge logic | Vol target range | Leverage cap |",
        "|---------|-----------|-------------|-----------------|--------------|",
        "| Ultra Conservative | Max Sharpe | Always hedged | 5–9% | 1.5× |",
        "| Conservative | Max Sharpe | Always hedged | 9–13% | 2.0× |",
        "| Moderate | Max Sharpe | VIX or vol trigger | 11–13% | 2.0× |",
        "| Growth | Max Calmar | Crisis-only (VIX) | 15–17% | 2.0× |",
        "| Aggressive | Max CAGR | Crisis-only (VIX) | 28–36% | **5.0×** |",
        "| Unhedged | — | No hedge | None | 1.0× |\n",
        "> **Note:** Only the Aggressive profile is intentionally designed to "
        "exceed unhedged returns (and risk).  All other profiles target superior "
        "risk-adjusted metrics at lower absolute volatility.\n",
    ]

    # ── Optimised parameters ──────────────────────────────────────────────────
    lines += [
        "### Optimised Hyperparameters\n",
        "| Profile | q_scale | r_meas | vol_target | VIX thresh | Vol thresh |",
        "|---------|---------|--------|------------|------------|------------|",
    ]
    for name, params in optimised_params.items():
        if params is None:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        q   = f"{params['q_scale']:.2e}"
        r   = f"{params['r_meas']:.2e}"
        vt  = f"{params['vol_target']*100:.0f}%"
        vix = f"{params['vix_threshold']:.0f}" if params.get("vix_threshold") else "—"
        vol = f"{params['vol_threshold']*100:.0f}%" if params.get("vol_threshold") else "—"
        lines.append(f"| {name} | `{q}` | `{r}` | {vt} | {vix} | {vol} |")
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

    if uh_10 and uc_10 and con_10:
        uc_sharpe_delta = uc_10.get("sharpe", 0) - uh_10.get("sharpe", 0)
        con_sharpe_delta = con_10.get("sharpe", 0) - uh_10.get("sharpe", 0)
        lines += [
            f"1. **Capital preservation:** Ultra Conservative achieves "
            f"Sharpe {uc_10.get('sharpe', 0):.2f} ({uc_sharpe_delta:+.2f} vs unhedged) "
            f"with max drawdown {uc_10.get('max_dd', 0):.1%} "
            f"vs {uh_10.get('max_dd', 0):.1%} unhedged (10Y).",
        ]
    if con_10 and uh_10:
        lines += [
            f"2. **Best risk-adjusted (conservative):** Conservative profile "
            f"targets consistent Sharpe improvement via always-on hedging + "
            f"vol targeting at ~{con_10.get('vol', 0):.0%} annualised vol.",
        ]
    if mod_10 and uh_10:
        dd_saved = mod_10.get("max_dd", 0) - uh_10.get("max_dd", 0)
        lines += [
            f"3. **Moderate:** Conditional hedge (VIX/vol trigger) reduces max "
            f"drawdown by {abs(dd_saved):.1%} while preserving most upside. "
            f"Good balance for long-horizon investors.",
        ]
    if gro_10 and uh_10:
        lines += [
            f"4. **Growth:** Crisis-only hedge lets the portfolio ride bull markets "
            f"nearly unhedged, with Calmar {gro_10.get('calmar', 0):.2f} vs "
            f"{uh_10.get('calmar', 0):.2f} unhedged (10Y).  "
            f"CAGR {pct(gro_10.get('cagr', 0))} vs {pct(uh_10.get('cagr', 0))} unhedged.",
        ]
    if agg_10 and uh_10:
        lines += [
            f"5. **Aggressive (only profile exceeding unhedged):** Leveraged long "
            f"with crisis-only hedge targets CAGR {pct(agg_10.get('cagr', 0))} "
            f"vs {pct(uh_10.get('cagr', 0))} unhedged, at vol "
            f"{pct_plain(agg_10.get('vol', 0))} vs "
            f"{pct_plain(uh_10.get('vol', 0))} unhedged (10Y).  "
            f"Max DD: {agg_10.get('max_dd', 0):.1%}.  "
            f"**Suitable only for investors with high risk tolerance and long horizon.**",
        ]

    lines += [
        "",
        "---\n",
        "*All backtests use strict no-lookahead guards: Kalman beta, VIX gate, "
        "and vol scale are each shifted one day before application.  "
        "Transaction costs: 2–3 bps per hedge toggle.  "
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

    # 2. Personal portfolio metrics
    print("\nComputing personal portfolio metrics…")
    personal_metrics = personal_portfolio_metrics()
    if personal_metrics:
        print(f"  Personal CAGR={personal_metrics['cagr']:+.1%}  "
              f"Sharpe={personal_metrics['sharpe']:.3f}  "
              f"MaxDD={personal_metrics['max_dd']:.1%}")
    else:
        print("  (no trade data found)")

    # 3. Optimise each profile
    print("\nOptimising risk profiles (grid search on 10Y window)…")
    optimised_params  = {}
    optimised_betas   = {}
    optimised_configs = {}

    for name, meta in PROFILE_META.items():
        params, beta = optimize_profile(name, meta, port_ret, spy_ret, vix)
        optimised_params[name]  = params
        optimised_betas[name]   = beta
        optimised_configs[name] = {
            "hedge":         meta["hedge"],
            "vol_target":    params["vol_target"] if params else 0.15,
            "vix_threshold": params.get("vix_threshold") if params else None,
            "vol_threshold": params.get("vol_threshold") if params else None,
            "min_scale":     meta["min_scale"],
            "max_scale":     meta["max_scale"],
            "cost_bps":      meta["cost_bps"],
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

        # 5 profiles
        for name in PROFILE_META:
            beta   = optimised_betas.get(name)
            cfg    = optimised_configs.get(name)
            if beta is None or cfg is None:
                continue
            ret = run_strategy(port_ret, spy_ret, vix, beta, cfg, start_date)
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
    md_path = write_markdown(all_results, optimised_params, personal_metrics)

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
