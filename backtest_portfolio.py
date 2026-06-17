#!/usr/bin/env python3
"""
backtest_portfolio.py — Kalman + VIX-Regime Portfolio Hedging Backtest

Combines hedge/kalman.py (time-varying beta) with a VIX-based regime gate
(proxy for the ML gate when 20 years of options data isn't available) and
vol targeting (matching the logic in hedge/ml_gate.py).

3 risk profiles × 3 time windows + unhedged baseline = 12 result sets.

Portfolio:  AAPL MSFT AMZN GOOGL JPM JNJ XOM WMT HD PG (all public pre-2006)
Hedge:      SPY short, sized by Kalman time-varying portfolio beta
Regime:     VIX-based gate

Run from project root:
    python backtest_portfolio.py
"""

import math
import sys
import warnings
from pathlib import Path

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


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "WMT", "HD", "PG"]
HEDGE_TICKER = "SPY"
VIX_TICKER   = "^VIX"
ALL_TICKERS  = PORTFOLIO + [HEDGE_TICKER, VIX_TICKER]

DATA_START = "2005-01-01"   # includes 252-day warmup buffer for all windows
END_DATE   = "2026-06-12"

WINDOWS = {
    "20Y": "2006-06-12",
    "10Y": "2016-06-12",
    "5Y":  "2021-06-12",
}

RISK_FREE = 0.045  # annualised T-bill proxy for Sharpe calculation

OUTPUT_DIR = Path("backtest_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Strategy parameters
# hedge: "none" | "always" | "vix_or_vol" | "crisis_only"
STRATEGIES = {
    "Unhedged": {
        "label":         "Unhedged (buy & hold)",
        "hedge":         "none",
        "vol_target":    None,      # no vol targeting — pure equal-weight
        "vix_threshold": None,
        "vol_threshold": None,
        "q_scale":       5e-6,      # Kalman params (beta not used for hedging)
        "r_meas":        5e-4,
        "max_scale":     1.0,
        "min_scale":     1.0,
        "cost_bps":      0.0,
    },
    "Conservative": {
        "label":         "Conservative (10% vol, always hedged)",
        "hedge":         "always",
        "vol_target":    0.10,
        "vix_threshold": None,
        "vol_threshold": None,
        "q_scale":       1e-6,      # slow beta adaptation — stable hedge ratio
        "r_meas":        5e-4,
        "max_scale":     2.0,
        "min_scale":     0.25,
        "cost_bps":      3.0,
    },
    "Moderate": {
        "label":         "Moderate (15% vol, hedge VIX>20 or vol>18%)",
        "hedge":         "vix_or_vol",
        "vol_target":    0.15,
        "vix_threshold": 20.0,
        "vol_threshold": 0.18,
        "q_scale":       5e-6,      # medium adaptation
        "r_meas":        5e-4,
        "max_scale":     2.5,
        "min_scale":     0.25,
        "cost_bps":      3.0,
    },
    "Aggressive": {
        "label":         "Aggressive (22% vol, hedge VIX>30 only)",
        "hedge":         "crisis_only",
        "vol_target":    0.22,
        "vix_threshold": 30.0,
        "vol_threshold": None,
        "q_scale":       2e-5,      # faster adaptation in crises
        "r_meas":        5e-4,
        "max_scale":     4.0,
        "min_scale":     0.30,
        "cost_bps":      3.0,
    },
}

STRATEGY_ORDER = ["Unhedged", "Conservative", "Moderate", "Aggressive"]
COLORS = {
    "Unhedged":     "#4c72b0",
    "Conservative": "#55a868",
    "Moderate":     "#c44e52",
    "Aggressive":   "#dd8452",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def download_data():
    print("Downloading price data (2005-2026)…")
    raw = yf.download(
        ALL_TICKERS, start=DATA_START, end=END_DATE,
        auto_adjust=True, progress=True
    )["Close"]

    # VIX: forward-fill up to 3 days for holidays/half-days
    raw[VIX_TICKER] = raw[VIX_TICKER].ffill(limit=3)

    # Drop days where any equity ticker is missing
    equity_cols = PORTFOLIO + [HEDGE_TICKER]
    raw = raw.dropna(subset=equity_cols)

    prices = raw[equity_cols]
    vix    = raw[VIX_TICKER]

    returns   = prices.pct_change().iloc[1:]    # drop first NaN row
    vix       = vix.loc[returns.index]

    port_ret  = returns[PORTFOLIO].mean(axis=1)
    spy_ret   = returns[HEDGE_TICKER]

    print(f"  Loaded {len(returns)} trading days: "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")
    return prices, returns, port_ret, spy_ret, vix


# ─────────────────────────────────────────────────────────────────────────────
# Strategy engine
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(port_ret, spy_ret, vix, beta_series, config, start_date):
    """
    Simulate one strategy from start_date onward.

    Lookahead guards:
      - beta uses beta.shift(1): today's hedge ratio is yesterday's Kalman estimate
      - gate uses vix.shift(1) and rolling_vol.shift(1): yesterday's regime signal
      - vol scale uses scale.shift(1): yesterday's vol drives today's position size

    Returns pd.Series of daily strategy returns from start_date onward.
    """
    hedge_type  = config["hedge"]
    vol_target  = config["vol_target"]
    vix_thresh  = config["vix_threshold"]
    vol_thresh  = config["vol_threshold"]
    min_scale   = config["min_scale"]
    max_scale   = config["max_scale"]
    cost_bps    = config["cost_bps"]

    # Align to port_ret index
    idx    = port_ret.index
    vix_a  = vix.reindex(idx).ffill()
    beta_a = beta_series.reindex(idx)
    spy_a  = spy_ret.reindex(idx)

    # Portfolio rolling vol (pre-hedge, for gate signal)
    port_vol_20 = port_ret.rolling(20).std() * math.sqrt(252)

    # Gate signal — no lookahead (shift(1))
    if hedge_type == "none":
        gate = pd.Series(0.0, index=idx)
    elif hedge_type == "always":
        gate = pd.Series(1.0, index=idx)
    elif hedge_type == "vix_or_vol":
        gate = (
            (vix_a.shift(1) > vix_thresh) |
            (port_vol_20.shift(1) > vol_thresh)
        ).astype(float).fillna(0.0)
    elif hedge_type == "crisis_only":
        gate = (vix_a.shift(1) > vix_thresh).astype(float).fillna(0.0)
    else:
        raise ValueError(f"Unknown hedge type: {hedge_type}")

    # Kalman beta — shift 1 to avoid lookahead; clip to valid range
    hedge_ratio = beta_a.shift(1).clip(lower=0.0, upper=2.5).fillna(0.0)

    # Hedge and raw net return
    hedge_ret = gate * hedge_ratio * spy_a
    raw_ret   = port_ret - hedge_ret

    # Vol targeting
    if vol_target is not None:
        # Rolling vol of the hedged return, shifted 1 (no lookahead)
        vol_roll = raw_ret.rolling(20).std() * math.sqrt(252)
        scale    = (vol_target / (vol_roll + 1e-8)).clip(lower=min_scale, upper=max_scale)
        strat_ret = raw_ret * scale.shift(1)
    else:
        strat_ret = raw_ret.copy()

    # Transaction costs on gate toggle
    if cost_bps > 0:
        toggles   = gate.diff().abs().fillna(0)
        strat_ret = strat_ret - toggles * (cost_bps / 10_000.0)

    return strat_ret.loc[start_date:].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}

    freq    = 252
    cum     = (1 + r).cumprod()
    n_years = len(r) / freq

    total_ret = float(cum.iloc[-1]) - 1.0
    cagr      = float(cum.iloc[-1] ** (1.0 / n_years)) - 1.0
    vol       = float(r.std() * math.sqrt(freq))

    rf_daily  = RISK_FREE / freq
    sharpe    = float((r.mean() - rf_daily) * freq / (r.std() * math.sqrt(freq) + 1e-12))

    down_std  = r[r < 0].std() * math.sqrt(freq) if (r < 0).any() else 1e-8
    sortino   = float((r.mean() * freq - RISK_FREE) / (down_std + 1e-12))

    dd        = cum / cum.cummax() - 1
    max_dd    = float(dd.min())
    calmar    = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    var95     = float(r.quantile(0.05))
    cvar95    = float(r[r <= var95].mean()) if (r <= var95).any() else var95

    annual    = r.resample("YE").apply(lambda x: float((1 + x).prod() - 1))
    best_yr   = int(annual.idxmax().year) if len(annual) > 0 else 0
    worst_yr  = int(annual.idxmin().year) if len(annual) > 0 else 0
    best_ret  = float(annual.max()) if len(annual) > 0 else 0.0
    worst_ret = float(annual.min()) if len(annual) > 0 else 0.0

    # Hedge activation rate (for hedged strategies, stored in details during run)
    return {
        "total_ret": total_ret,
        "cagr":      cagr,
        "vol":       vol,
        "sharpe":    sharpe,
        "sortino":   sortino,
        "calmar":    calmar,
        "max_dd":    max_dd,
        "var95":     var95,
        "cvar95":    cvar95,
        "best_yr":   best_yr,
        "best_ret":  best_ret,
        "worst_yr":  worst_yr,
        "worst_ret": worst_ret,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_results(all_results):
    W = 118
    print("\n" + "=" * W)
    print("  PORTFOLIO BACKTEST — 12 RESULT SETS")
    print(f"  Portfolio : {', '.join(PORTFOLIO)}")
    print(f"  Hedge     : SPY short sized by Kalman time-varying beta")
    print(f"  Regime    : VIX gate  (Conservative=always | Moderate=VIX>20 | Aggressive=VIX>30)")
    print(f"  Risk-free : {RISK_FREE*100:.1f}%  (for Sharpe/Sortino)")
    print("=" * W)

    hdr = (
        f"{'Strategy':<16} {'TotalRet':>9} {'CAGR':>7} {'Vol':>7} "
        f"{'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MaxDD':>7} "
        f"{'VaR95':>7} {'CVaR95':>7} {'BestYr':>13} {'WorstYr':>13}"
    )
    sep = "─" * len(hdr)

    for window, start in WINDOWS.items():
        print(f"\n  {'━'*W}")
        print(f"  {window}  ({start}  →  {END_DATE})")
        print(f"  {'━'*W}")
        print("  " + hdr)
        print("  " + sep)
        for strat in STRATEGY_ORDER:
            m = all_results.get((window, strat), {}).get("metrics", {})
            if not m:
                continue
            by = f"{m['best_ret']:>+.0%}({m['best_yr']})"
            wy = f"{m['worst_ret']:>+.0%}({m['worst_yr']})"
            calmar_str = f"{m['calmar']:>7.2f}" if not math.isnan(m["calmar"]) else "    inf"
            print(
                f"  {strat:<16} "
                f"{m['total_ret']:>+8.1%} "
                f"{m['cagr']:>+6.1%} "
                f"{m['vol']:>6.1%} "
                f"{m['sharpe']:>7.2f} "
                f"{m['sortino']:>8.2f} "
                f"{calmar_str} "
                f"{m['max_dd']:>6.1%} "
                f"{m['var95']:>6.2%} "
                f"{m['cvar95']:>6.2%} "
                f"{by:>13} "
                f"{wy:>13}"
            )

    print("\n" + "=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _shade_vix_crises(ax, vix, start, threshold=25, alpha=0.08):
    """Fill background where VIX exceeded threshold (crisis periods)."""
    vix_w = vix.loc[start:]
    in_crisis = vix_w > threshold
    changes = in_crisis.astype(int).diff().fillna(0)
    starts  = vix_w.index[changes == 1].tolist()
    ends    = vix_w.index[changes == -1].tolist()
    if in_crisis.iloc[0]:
        starts = [vix_w.index[0]] + starts
    if in_crisis.iloc[-1]:
        ends = ends + [vix_w.index[-1]]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=alpha, color="red", zorder=0)


def plot_equity_curves(all_results, vix):
    fig, axes = plt.subplots(3, 1, figsize=(15, 17), sharex=False)
    fig.suptitle(
        "Portfolio Strategy Equity Curves\n"
        "Shaded = VIX > 25 (crisis periods where hedge activates most)",
        fontsize=13, fontweight="bold", y=0.98
    )

    for ax, (window, start) in zip(axes, WINDOWS.items()):
        ax.set_facecolor("#f8f9fa")
        _shade_vix_crises(ax, vix, start, threshold=25)

        for strat in STRATEGY_ORDER:
            entry = all_results.get((window, strat))
            if not entry:
                continue
            ret = entry["returns"]
            cum = (1 + ret).cumprod()
            lw  = 2.0 if strat != "Unhedged" else 1.4
            ls  = "--" if strat == "Unhedged" else "-"
            ax.semilogy(cum.index, cum.values, label=strat,
                        color=COLORS[strat], linewidth=lw, linestyle=ls, alpha=0.9)

            # Annotate final value
            final = cum.iloc[-1]
            ax.annotate(
                f"  {strat[0]}: ×{final:.1f}",
                xy=(cum.index[-1], final),
                fontsize=7.5, color=COLORS[strat], va="center"
            )

        ax.set_title(f"{window}  ({start} → {END_DATE})", fontweight="bold", fontsize=10)
        ax.set_ylabel("Growth of $1  (log scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:.1f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "fig1_equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drawdowns(all_results, vix):
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=False)
    fig.suptitle("Drawdown Comparison — All Strategies & Time Windows",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, (window, start) in zip(axes, WINDOWS.items()):
        ax.set_facecolor("#f8f9fa")
        _shade_vix_crises(ax, vix, start, threshold=25, alpha=0.06)

        for strat in STRATEGY_ORDER:
            entry = all_results.get((window, strat))
            if not entry:
                continue
            ret = entry["returns"]
            cum = (1 + ret).cumprod()
            dd  = cum / cum.cummax() - 1
            ax.fill_between(dd.index, dd.values, 0,
                            alpha=0.30, color=COLORS[strat])
            ax.plot(dd.index, dd.values, color=COLORS[strat], linewidth=1.0,
                    alpha=0.85, label=strat)

            # Annotate max drawdown
            worst_idx = dd.idxmin()
            worst_val = dd.min()
            ax.annotate(
                f"{worst_val:.0%}",
                xy=(worst_idx, worst_val),
                xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=6.5, color=COLORS[strat],
                arrowprops=dict(arrowstyle="-", color=COLORS[strat], lw=0.5)
            )

        ax.set_title(f"{window}  ({start} → {END_DATE})", fontweight="bold", fontsize=10)
        ax.set_ylabel("Drawdown from Peak")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.8)
        ax.set_ylim(-0.85, 0.05)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "fig2_drawdowns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metrics_comparison(all_results):
    metrics_cfg = [
        ("sharpe",   "Sharpe Ratio",         False, None),
        ("cagr",     "CAGR",                 True,  None),
        ("max_dd",   "Max Drawdown (abs.)",   True,  True),   # abs value
        ("calmar",   "Calmar Ratio",          False, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Strategy Metrics Comparison  ·  All 12 Result Sets\n"
        f"(Portfolio: {', '.join(PORTFOLIO)})",
        fontsize=12, fontweight="bold", y=0.99
    )
    axes = axes.flatten()

    window_labels = list(WINDOWS.keys())
    x      = np.arange(len(window_labels))
    n      = len(STRATEGY_ORDER)
    width  = 0.75 / n

    for ax, (metric, title, as_pct, take_abs) in zip(axes, metrics_cfg):
        ax.set_facecolor("#f8f9fa")
        for i, strat in enumerate(STRATEGY_ORDER):
            vals = []
            for window in WINDOWS:
                m = all_results.get((window, strat), {}).get("metrics", {})
                v = m.get(metric, float("nan")) if m else float("nan")
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
                    bar.get_height() + (0.002 if not as_pct else 0.002),
                    lbl, ha="center", va="bottom", fontsize=7, fontweight="bold"
                )

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontsize=9)
        if as_pct:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "fig3_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rolling_sharpe(all_results):
    """Rolling 252-day Sharpe for the 20Y window."""
    window = "20Y"
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_facecolor("#f8f9fa")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(1, color="green", linewidth=0.6, linestyle=":", alpha=0.4)

    for strat in STRATEGY_ORDER:
        entry = all_results.get((window, strat))
        if not entry:
            continue
        r = entry["returns"]
        roll_sharpe = (
            r.rolling(252).mean() * 252 /
            (r.rolling(252).std() * math.sqrt(252) + 1e-12)
        )
        ax.plot(roll_sharpe.index, roll_sharpe.values, label=strat,
                color=COLORS[strat], linewidth=1.5, alpha=0.85)

    ax.set_title("Rolling 252-Day Sharpe Ratio  —  20Y Window", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (trailing 1Y)")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3.5, 5.0)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig4_rolling_sharpe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    prices, returns, port_ret, spy_ret, vix = download_data()

    # Pre-compute Kalman betas for each strategy's q_scale (run on full history)
    print("\nRunning Kalman filter on full dataset (one pass per strategy)…")
    kalman_betas = {}
    for strat_name, cfg in STRATEGIES.items():
        if cfg["hedge"] == "none":
            kalman_betas[strat_name] = pd.Series(1.0, index=port_ret.index)
            continue

        # Align and drop NaN (first row)
        common = port_ret.dropna().index.intersection(spy_ret.dropna().index)
        kal = kalman_time_varying_beta(
            port_ret.loc[common], spy_ret.loc[common],
            cfg["q_scale"], cfg["r_meas"]
        )
        kalman_betas[strat_name] = kal.beta.reindex(port_ret.index)

        b = kal.beta
        print(
            f"  {strat_name:<14}  q={cfg['q_scale']:.0e}  "
            f"beta [{b.min():.2f}, {b.max():.2f}]  "
            f"final={b.iloc[-1]:.3f}"
        )

    # Run all 12 backtests
    print("\nRunning 12 backtests…")
    all_results = {}

    for window, start_date in WINDOWS.items():
        print(f"\n  ── {window}  ({start_date} → {END_DATE}) ──")
        for strat_name in STRATEGY_ORDER:
            cfg  = STRATEGIES[strat_name]
            beta = kalman_betas[strat_name]

            ret = run_strategy(port_ret, spy_ret, vix, beta, cfg, start_date)
            m   = compute_metrics(ret)
            all_results[(window, strat_name)] = {"returns": ret, "metrics": m}

            if m:
                calmar_str = f"{m['calmar']:.2f}" if not math.isnan(m["calmar"]) else "inf"
                print(
                    f"    {strat_name:<14}  CAGR={m['cagr']:>+6.1%}  "
                    f"Sharpe={m['sharpe']:>5.2f}  "
                    f"MaxDD={m['max_dd']:>6.1%}  "
                    f"Calmar={calmar_str}"
                )

    # Detailed table
    print_results(all_results)

    # Save CSV
    rows = []
    for (window, strat), entry in all_results.items():
        m = entry.get("metrics", {})
        if m:
            rows.append({"window": window, "strategy": strat, **m})
    csv_path = OUTPUT_DIR / "backtest_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Metrics CSV → {csv_path}")

    # Charts
    print("\nGenerating charts…")
    plot_equity_curves(all_results, vix)
    plot_drawdowns(all_results, vix)
    plot_metrics_comparison(all_results)
    plot_rolling_sharpe(all_results)

    print(f"\nDone.  Output → {OUTPUT_DIR.resolve()}/")
    print("  fig1_equity_curves.png  — growth of $1, log scale, all strategies")
    print("  fig2_drawdowns.png      — drawdown comparison with crisis shading")
    print("  fig3_metrics.png        — Sharpe/CAGR/MaxDD/Calmar bar charts")
    print("  fig4_rolling_sharpe.png — rolling 1Y Sharpe, 20Y window")
    print("  backtest_metrics.csv    — all 12 result sets in tabular form")


if __name__ == "__main__":
    main()
