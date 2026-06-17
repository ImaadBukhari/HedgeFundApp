#!/usr/bin/env python3
"""
generate_report.py — Full PDF: personal track record + strategy backtest

Run from project root:
    python generate_report.py
"""

import json, math, sys, warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trades"))

from hedge.kalman import kalman_time_varying_beta
import backtest_portfolio as bp

# Import analytics helpers; reset the dark rcParams it sets at module level
import analytics as _a
plt.rcdefaults()

TRADES_FILE = ROOT / "trades" / "trades.json"
REPORT_PATH = ROOT / "portfolio_report.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Styling constants
# ─────────────────────────────────────────────────────────────────────────────

PW, PH   = 8.5, 11.0     # letter portrait
LW, LH   = 11.0, 8.5     # letter landscape
HDR      = "#1b1f3a"      # dark navy
SUBHDR   = "#2d4f74"
ROW_A    = "#f4f6fa"
DIV      = "#cdd1db"
TXT      = "#1a1a2a"
ACCENT   = "#2e86ab"
POS_CLR  = "#1a7a40"
NEG_CLR  = "#b01a1a"
COLORS   = bp.COLORS
SO       = bp.STRATEGY_ORDER

SQRT252  = math.sqrt(252)
RF       = 0.045


# ─────────────────────────────────────────────────────────────────────────────
# Utility: styled matplotlib table
# ─────────────────────────────────────────────────────────────────────────────

def styled_table(ax, df, title="", fs=7.5, header=HDR, bbox=None):
    ax.axis("off")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10, color=HDR,
                     loc="left", pad=6)
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.values]
    b = bbox or [0, 0, 1, 1]
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="center", bbox=b)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    n_col = len(cols)
    n_row = len(rows)
    for j in range(n_col):
        c = tbl[0, j]
        c.set_facecolor(header)
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("white")
    for i in range(1, n_row + 1):
        fc = ROW_A if i % 2 == 0 else "white"
        for j in range(n_col):
            c = tbl[i, j]
            c.set_facecolor(fc)
            c.set_edgecolor(DIV)
            c.set_text_props(color=TXT)
    tbl.auto_set_column_width(range(n_col))
    return tbl


def fig_portrait(facecolor="white"):
    return plt.figure(figsize=(PW, PH), facecolor=facecolor)


def fig_landscape(facecolor="white"):
    return plt.figure(figsize=(LW, LH), facecolor=facecolor)


def save(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Personal portfolio: data loading & computation
# ─────────────────────────────────────────────────────────────────────────────

def load_all_trades():
    with open(TRADES_FILE) as f:
        raw = json.load(f)
    out = []
    for t in raw["trades"]:
        out.append({
            "date":       pd.Timestamp(t["date"]),
            "ticker":     t["ticker"],
            "action":     t["action"],
            "amount_usd": float(t["amount_usd"]),
            "amount_cad": float(t["amount_cad"]),
            "fx_rate":    float(t["fx_rate"]),
            "note":       t.get("note", ""),
        })
    return sorted(out, key=lambda x: x["date"])


def build_personal(all_trades_full):
    """
    Use analytics.py's proven NAV/TWR pipeline (same functions that produced
    the correct Sharpe/Calmar numbers previously).  The full-field trade list
    is only used for the trade-log table; all NAV/position computation goes
    through analytics.py.
    """
    # ── proven analytics pipeline ──────────────────────────────────────────
    trades_simple = _a.load_trades()   # analytics.py format (date/ticker/action/amount_usd)
    tickers = {t["ticker"] for t in trades_simple if t["ticker"] not in _a.EXCLUDED}
    start   = min(t["date"] for t in trades_simple)

    print(f"  Fetching prices for: {sorted(tickers)}")
    raw_hist = _a.fetch_all_histories(tickers, start, pd.Timestamp.today())
    price_histories = {tk: raw_hist[tk] for tk in tickers if tk in raw_hist.columns}

    positions_dict, trade_log_enriched = _a.build_positions(trades_simple, price_histories)
    nav, net_cash_flow = _a.build_daily_nav(trade_log_enriched, price_histories)
    twr = _a.twr_returns(nav, net_cash_flow)
    twr = twr[twr.between(-0.5, 0.5)]

    # ── trade log table (uses full-field data for CAD/FX/note columns) ─────
    trade_log = []
    for t in all_trades_full:
        tk, act = t["ticker"], t["action"]
        amt     = t["amount_usd"]
        dt      = t["date"]

        if tk in _a.EXCLUDED:
            trade_log.append({
                "Date": str(dt.date()), "Ticker": tk, "Action": act.upper(),
                "USD ($)": f"{amt:,.0f}", "CAD ($)": f"{t['amount_cad']:,.0f}",
                "FX": f"{t['fx_rate']:.4f}", "Price ($)": "—", "Shares": "—",
                "Note": t["note"][:30],
            })
            continue
        if act == "dividend":
            trade_log.append({
                "Date": str(dt.date()), "Ticker": tk, "Action": "DIV",
                "USD ($)": f"{amt:,.2f}", "CAD ($)": "—",
                "FX": "—", "Price ($)": "—", "Shares": "—",
                "Note": t["note"][:30],
            })
            continue
        # Find the enriched trade to get price/shares
        match = next((e for e in trade_log_enriched
                      if e["ticker"] == tk and e["date"] == dt and e["action"] == act), None)
        price  = match["price"]          if match else 0.0
        shares = match["shares_traded"]  if match else 0.0
        trade_log.append({
            "Date":      str(dt.date()),
            "Ticker":    tk,
            "Action":    act.upper(),
            "USD ($)":   f"{amt:,.0f}",
            "CAD ($)":   f"{t['amount_cad']:,.0f}" if t["fx_rate"] != 1.0 else "—",
            "FX":        f"{t['fx_rate']:.4f}"     if t["fx_rate"] != 1.0 else "—",
            "Price ($)": f"{price:,.2f}",
            "Shares":    f"{shares:.3f}",
            "Note":      t["note"][:30],
        })

    # ── current positions table ────────────────────────────────────────────
    positions = []
    for tk, pos in sorted(positions_dict.items()):
        if tk in _a.EXCLUDED:
            continue
        sh   = pos.get("shares", 0.0)
        if sh < 0.001:
            continue
        hist = price_histories.get(tk)
        cp   = float(hist.dropna().iloc[-1]) if (hist is not None and not hist.dropna().empty) else 0.0
        cost = pos.get("total_cost", 0.0)
        avg  = cost / sh if sh > 0 else 0.0
        mkt  = sh * cp
        unrl = mkt - cost
        ret  = unrl / cost * 100 if cost > 0 else 0.0
        positions.append({
            "Ticker":     tk,
            "Shares":     f"{sh:.3f}",
            "Avg Cost":   f"${avg:.2f}",
            "Curr Price": f"${cp:.2f}",
            "Mkt Val":    f"${mkt:,.0f}",
            "Cost Basis": f"${cost:,.0f}",
            "Unrl P&L":   f"${unrl:+,.0f}",
            "Return %":   f"{ret:+.1f}%",
        })

    # ── per-ticker volatility ──────────────────────────────────────────────
    vol_rows = []
    for tk, hist in sorted(price_histories.items()):
        first_buy = next((t["date"] for t in trades_simple
                          if t["ticker"] == tk and t["action"] == "buy"), None)
        h = hist.dropna()
        if first_buy is not None:
            h = h[h.index >= first_buy]
        if len(h) < 6:
            continue
        r = h.pct_change().dropna()
        vol_rows.append({
            "Ticker":    tk,
            "Ann Vol":   f"{r.std()*SQRT252*100:.1f}%",
            "Daily Vol": f"{r.std()*100:.2f}%",
            "Skewness":  f"{r.skew():.3f}",
            "Kurtosis":  f"{r.kurt():.3f}",
            "Best Day":  f"{r.max()*100:+.2f}%",
            "Worst Day": f"{r.min()*100:+.2f}%",
            "# Days":    str(len(r)),
        })

    # ── fund-wide metrics ──────────────────────────────────────────────────
    metrics = {}
    if len(twr) > 5:
        cum     = (1 + twr).cumprod()
        n_y     = len(twr) / 252
        cg      = float(cum.iloc[-1] ** (1 / n_y)) - 1
        vol     = float(twr.std() * SQRT252)
        rfD     = RF / 252
        sh      = float((twr.mean() - rfD) * 252 / (twr.std() * SQRT252 + 1e-12))
        dn      = twr[twr < 0]
        so      = float((twr.mean() * 252 - RF) / (dn.std() * SQRT252 + 1e-12)) if len(dn) > 1 else 0
        dd      = cum / cum.cummax() - 1
        mdd     = float(dd.min())
        cal     = cg / abs(mdd) if mdd < 0 else float("nan")
        v95     = float(twr.quantile(0.05))
        cv95    = float(twr[twr <= v95].mean())
        tot_inv = sum(t["amount_usd"] for t in trades_simple if t["action"] == "buy")
        curr_v  = float(nav.iloc[-1])
        pnl     = curr_v - tot_inv
        best_d  = twr.idxmax()
        worst_d = twr.idxmin()
        skew    = float(twr.skew())
        kurt    = float(twr.kurt())

        metrics = {
            "Period Start":              str(nav.index[0].date()),
            "Period End":                str(nav.index[-1].date()),
            "Calendar Days":             str((nav.index[-1] - nav.index[0]).days),
            "Trading Days in Series":    str(len(twr)),
            "# Trades (buy/sell)":       str(len([t for t in trades_simple
                                                   if t["action"] in ("buy", "sell")])),
            "Total Invested (USD)":      f"${tot_inv:,.0f}",
            "Current Market Value":      f"${curr_v:,.0f}",
            "Total P&L (USD)":           f"${pnl:+,.0f}",
            "Absolute Return":           f"{pnl/tot_inv:+.1%}" if tot_inv > 0 else "—",
            "CAGR — TWR":                f"{cg:+.1%}",
            "Annualised Volatility":     f"{vol:.1%}",
            "Sharpe Ratio (rf=4.5%)":    f"{sh:.4f}",
            "Sortino Ratio (rf=4.5%)":   f"{so:.4f}",
            "Calmar Ratio":              f"{cal:.4f}" if not math.isnan(cal) else "—",
            "Max Drawdown":              f"{mdd:.2%}",
            "VaR 95% (1-day)":           f"{v95:.3%}",
            "CVaR / ES 95% (1-day)":     f"{cv95:.3%}",
            "Return Skewness":           f"{skew:.4f}",
            "Return Kurtosis (excess)":  f"{kurt:.4f}",
            "Best Day":                  f"{twr[best_d]*100:+.2f}%  ({best_d.date()})",
            "Worst Day":                 f"{twr[worst_d]*100:+.2f}%  ({worst_d.date()})",
            "Risk-Free Rate":            f"{RF*100:.1f}% p.a.",
        }

    return {
        "trades":   trade_log,
        "pos":      positions,
        "vol_rows": vol_rows,
        "nav":      nav,
        "twr":      twr,
        "metrics":  metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print("  Downloading strategy backtest data...")
    prices, returns, port_ret, spy_ret, vix = bp.download_data()

    print("  Running Kalman filters...")
    kalman_betas = {}
    for name, cfg in bp.STRATEGIES.items():
        if cfg["hedge"] == "none":
            kalman_betas[name] = pd.Series(1.0, index=port_ret.index)
        else:
            idx = port_ret.dropna().index.intersection(spy_ret.dropna().index)
            kal = kalman_time_varying_beta(port_ret.loc[idx], spy_ret.loc[idx],
                                           cfg["q_scale"], cfg["r_meas"])
            kalman_betas[name] = kal.beta.reindex(port_ret.index)

    results = {}
    for window, start in bp.WINDOWS.items():
        for strat in SO:
            cfg  = bp.STRATEGIES[strat]
            beta = kalman_betas[strat]
            ret  = bp.run_strategy(port_ret, spy_ret, vix, beta, cfg, start)
            m    = bp.compute_metrics(ret)
            results[(window, strat)] = {"returns": ret, "metrics": m}
    return results, vix


# ─────────────────────────────────────────────────────────────────────────────
# PDF pages
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf, today):
    fig = fig_portrait()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.78), 1, 0.22, color=HDR, zorder=0))
    ax.text(0.5, 0.92, "INVESTMENT PERFORMANCE REPORT",
            transform=ax.transAxes, fontsize=22, fontweight="bold",
            color="white", ha="center", va="center")
    ax.text(0.5, 0.83, f"As of  {today}",
            transform=ax.transAxes, fontsize=12, color="#9ab0cc",
            ha="center", va="center")

    lines = [
        ("PART I  —  PERSONAL PORTFOLIO TRACK RECORD", True),
        ("    Portfolio Performance Metrics  ·  All Risk-Adjusted Ratios", False),
        ("    Current Positions  ·  Per-Ticker Volatility Detail", False),
        ("    Complete Trade Log  (all 79 trades)", False),
        ("    NAV Time Series  ·  Drawdown", False),
        ("", False),
        ("PART II  —  HISTORICAL STRATEGY BACKTEST", True),
        ("    Strategy Parameters", False),
        ("    20-Year Results  (2006 – 2026)", False),
        ("    10-Year Results  (2016 – 2026)", False),
        ("    5-Year Results  (2021 – 2026)", False),
        ("    Equity Curves — All 4 Strategies on One Graph", False),
        ("    Drawdown Comparison  ·  Metrics Bar Charts", False),
        ("", False),
        ("Portfolio Universe (Part II):", False),
        ("    AAPL  ·  MSFT  ·  AMZN  ·  GOOGL  ·  JPM", False),
        ("    JNJ  ·  XOM  ·  WMT  ·  HD  ·  PG", False),
        ("", False),
        ("Hedge: SPY  ·  Regime: VIX  ·  rf = 4.5%  ·  Vol targeting on all hedged strategies", False),
    ]
    y = 0.73
    for text, bold in lines:
        ax.text(0.10, y, text, transform=ax.transAxes,
                fontsize=10 if bold else 9,
                fontweight="bold" if bold else "normal",
                color=HDR if bold else TXT, va="top")
        y -= 0.037 if bold else 0.033

    ax.add_patch(plt.Rectangle((0, 0), 1, 0.04, color=HDR, zorder=0))
    ax.text(0.5, 0.020, "Confidential  ·  Internal Use Only",
            transform=ax.transAxes, fontsize=8, color="#9ab0cc",
            ha="center", va="center")
    save(pdf, fig)


def page_personal_metrics(pdf, metrics):
    fig = fig_portrait()
    fig.text(0.5, 0.95, "PART I — PERSONAL PORTFOLIO", fontsize=13,
             fontweight="bold", color="white", ha="center", va="top",
             bbox=dict(facecolor=HDR, edgecolor="none", pad=6, boxstyle="round,pad=0.4"))
    fig.text(0.5, 0.905, "PERFORMANCE METRICS", fontsize=11,
             fontweight="bold", color=HDR, ha="center", va="top")

    items = list(metrics.items())
    mid   = math.ceil(len(items) / 2)
    left  = items[:mid]
    right = items[mid:]

    def draw_kv(ax, kv_list, title):
        ax.axis("off")
        ax.set_title(title, fontweight="bold", fontsize=9, color=HDR, loc="left", pad=4)
        df = pd.DataFrame(kv_list, columns=["Metric", "Value"])
        styled_table(ax, df, fs=8.5)

    ax_l = fig.add_axes([0.04, 0.12, 0.44, 0.76])
    ax_r = fig.add_axes([0.52, 0.12, 0.44, 0.76])
    draw_kv(ax_l, left, "")
    draw_kv(ax_r, right, "")
    save(pdf, fig)


def page_positions(pdf, positions):
    fig = fig_portrait()
    fig.text(0.5, 0.95, "CURRENT POSITIONS", fontsize=12,
             fontweight="bold", color="white", ha="center",
             bbox=dict(facecolor=HDR, edgecolor="none", pad=5, boxstyle="round,pad=0.4"))
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    df = pd.DataFrame(positions)
    styled_table(ax, df, fs=8.0)
    save(pdf, fig)


def page_ticker_vol(pdf, vol_rows):
    fig = fig_portrait()
    fig.text(0.5, 0.95, "PER-TICKER VOLATILITY", fontsize=12,
             fontweight="bold", color="white", ha="center",
             bbox=dict(facecolor=HDR, edgecolor="none", pad=5, boxstyle="round,pad=0.4"))
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    df = pd.DataFrame(vol_rows)
    styled_table(ax, df, fs=8.5)
    save(pdf, fig)


def page_trade_log(pdf, trade_log):
    df = pd.DataFrame(trade_log)
    rows_per_page = 38
    pages = math.ceil(len(df) / rows_per_page)
    for p in range(pages):
        chunk = df.iloc[p * rows_per_page: (p + 1) * rows_per_page]
        fig = fig_landscape()
        label = f"TRADE LOG  [{p*rows_per_page+1} – {min((p+1)*rows_per_page, len(df))} of {len(df)}]"
        fig.text(0.5, 0.95, label, fontsize=11,
                 fontweight="bold", color="white", ha="center",
                 bbox=dict(facecolor=HDR, edgecolor="none", pad=5, boxstyle="round,pad=0.4"))
        ax = fig.add_axes([0.02, 0.04, 0.96, 0.87])
        styled_table(ax, chunk.reset_index(drop=True), fs=6.8)
        save(pdf, fig)


def page_nav_chart(pdf, nav, twr):
    fig, axes = plt.subplots(3, 1, figsize=(PW, PH), facecolor="white",
                              gridspec_kw={"height_ratios": [3, 1.5, 1.2]})
    fig.suptitle("PERSONAL PORTFOLIO — NAV & RISK SERIES",
                 fontsize=12, fontweight="bold", color=HDR, y=0.99)

    cum = (1 + twr).cumprod()
    dd  = cum / cum.cummax() - 1

    # NAV
    ax = axes[0]
    ax.set_facecolor("#f9fafc")
    ax.plot(cum.index, cum.values, color=ACCENT, linewidth=2.0, label="TWR equity curve")
    ax2 = ax.twinx()
    ax2.fill_between(dd.index, dd.values * 100, 0, color=NEG_CLR, alpha=0.25)
    ax2.set_ylabel("Drawdown %", color=NEG_CLR, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=NEG_CLR, labelsize=7)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title("TWR Equity Curve  (Growth of $1)  +  Drawdown Overlay",
                 fontsize=9, color=HDR, loc="left")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Daily returns
    ax = axes[1]
    ax.set_facecolor("#f9fafc")
    cols = [POS_CLR if r >= 0 else NEG_CLR for r in twr]
    ax.bar(twr.index, twr.values * 100, color=cols, width=1.0, alpha=0.85)
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.set_title("Daily Returns (%)", fontsize=9, color=HDR, loc="left")
    ax.set_ylabel("Return %")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Rolling 30-day Sharpe
    ax = axes[2]
    ax.set_facecolor("#f9fafc")
    roll_sh = (twr.rolling(30).mean() * 252 - RF) / (twr.rolling(30).std() * SQRT252 + 1e-12)
    ax.plot(roll_sh.index, roll_sh.values, color="#dd8452", linewidth=1.5)
    ax.fill_between(roll_sh.index, 0, roll_sh.values,
                    where=roll_sh.values >= 0, alpha=0.2, color=POS_CLR)
    ax.fill_between(roll_sh.index, 0, roll_sh.values,
                    where=roll_sh.values < 0, alpha=0.2, color=NEG_CLR)
    ax.axhline(0, color="#888", linewidth=0.7)
    ax.axhline(1, color="#ccc", linewidth=0.5, linestyle=":")
    ax.set_title("Rolling 30-Day Sharpe Ratio", fontsize=9, color=HDR, loc="left")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest pages
# ─────────────────────────────────────────────────────────────────────────────

def page_strategy_params(pdf):
    rows = []
    for name in SO:
        cfg = bp.STRATEGIES[name]
        rows.append({
            "Strategy":     name,
            "Description":  cfg["label"],
            "Hedge Gate":   cfg["hedge"],
            "Vol Target":   f"{cfg['vol_target']*100:.0f}%" if cfg["vol_target"] else "—",
            "VIX Thresh":   str(cfg["vix_threshold"]) if cfg["vix_threshold"] else "—",
            "Vol Thresh":   f"{cfg['vol_threshold']*100:.0f}%" if cfg["vol_threshold"] else "—",
            "Kalman q":     f"{cfg['q_scale']:.0e}",
            "Kalman r":     f"{cfg['r_meas']:.0e}",
            "Max Lev":      f"{cfg['max_scale']:.1f}×",
            "Min Lev":      f"{cfg['min_scale']:.2f}×",
            "Cost (bps)":   str(cfg["cost_bps"]),
        })
    fig = fig_landscape()
    fig.text(0.5, 0.93, "PART II — STRATEGY BACKTEST  ·  PARAMETERS",
             fontsize=12, fontweight="bold", color="white", ha="center",
             bbox=dict(facecolor=HDR, edgecolor="none", pad=5, boxstyle="round,pad=0.4"))
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.82])
    styled_table(ax, pd.DataFrame(rows), fs=8.5)
    save(pdf, fig)


def _metrics_df(results, window):
    rows = []
    for strat in SO:
        m = results.get((window, strat), {}).get("metrics", {})
        if not m:
            continue
        calmar_s = f"{m['calmar']:.4f}" if not math.isnan(m["calmar"]) else "—"
        by = f"{m['best_ret']:+.1%} ({m['best_yr']})"
        wy = f"{m['worst_ret']:+.1%} ({m['worst_yr']})"
        rows.append({
            "Strategy":   strat,
            "Total Ret":  f"{m['total_ret']:+.1%}",
            "CAGR":       f"{m['cagr']:+.1%}",
            "Ann Vol":    f"{m['vol']:.1%}",
            "Sharpe":     f"{m['sharpe']:.4f}",
            "Sortino":    f"{m['sortino']:.4f}",
            "Calmar":     calmar_s,
            "Max DD":     f"{m['max_dd']:.1%}",
            "VaR 95%":    f"{m['var95']:.3%}",
            "CVaR 95%":   f"{m['cvar95']:.3%}",
            "Best Year":  by,
            "Worst Year": wy,
        })
    return pd.DataFrame(rows)


def page_backtest_results(pdf, results, window, start):
    df = _metrics_df(results, window)
    fig = fig_landscape()
    label = f"{window} BACKTEST  ({start}  →  {bp.END_DATE})"
    fig.text(0.5, 0.93, label, fontsize=12,
             fontweight="bold", color="white", ha="center",
             bbox=dict(facecolor=HDR, edgecolor="none", pad=5, boxstyle="round,pad=0.4"))
    fig.text(0.5, 0.85,
             f"Portfolio: {', '.join(bp.PORTFOLIO)}  ·  Hedge: SPY  ·  rf = {RF*100:.1f}%",
             fontsize=8.5, color=TXT, ha="center")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.76])
    styled_table(ax, df, fs=9.0)
    save(pdf, fig)


def page_combined_metrics(pdf, results):
    """All 12 result sets in one landscape table."""
    rows = []
    for window, start in bp.WINDOWS.items():
        for strat in SO:
            m = results.get((window, strat), {}).get("metrics", {})
            if not m:
                continue
            calmar_s = f"{m['calmar']:.3f}" if not math.isnan(m["calmar"]) else "—"
            rows.append({
                "Window":   window,
                "Strategy": strat,
                "Total Ret":f"{m['total_ret']:+.1%}",
                "CAGR":     f"{m['cagr']:+.1%}",
                "Ann Vol":  f"{m['vol']:.1%}",
                "Sharpe":   f"{m['sharpe']:.3f}",
                "Sortino":  f"{m['sortino']:.3f}",
                "Calmar":   calmar_s,
                "Max DD":   f"{m['max_dd']:.1%}",
                "VaR 95%":  f"{m['var95']:.3%}",
                "CVaR 95%": f"{m['cvar95']:.3%}",
            })
    fig = fig_landscape()
    fig.text(0.5, 0.93, "ALL 12 RESULT SETS — COMBINED METRICS",
             fontsize=12, fontweight="bold", color="white", ha="center",
             bbox=dict(facecolor=HDR, edgecolor="none", pad=5, boxstyle="round,pad=0.4"))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.86])
    styled_table(ax, pd.DataFrame(rows), fs=8.5)
    save(pdf, fig)


def _shade_crises(ax, vix, start):
    vix_w = vix.loc[start:] if start in vix.index or vix.index[0] <= pd.Timestamp(start) else vix
    in_c = (vix_w > 25).astype(int)
    diff = in_c.diff().fillna(0)
    starts = vix_w.index[diff == 1].tolist()
    ends   = vix_w.index[diff == -1].tolist()
    if in_c.iloc[0]:
        starts = [vix_w.index[0]] + starts
    if in_c.iloc[-1]:
        ends = ends + [vix_w.index[-1]]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.07, color="red", zorder=0)


def page_equity_all_windows(pdf, results, vix):
    """3-panel equity curves, one per window."""
    fig, axes = plt.subplots(3, 1, figsize=(PW, PH), facecolor="white")
    fig.suptitle("EQUITY CURVES — ALL WINDOWS  (Growth of $1, log scale)\n"
                 "Red shading = VIX > 25",
                 fontsize=10, fontweight="bold", color=HDR, y=0.99)

    for ax, (window, start) in zip(axes, bp.WINDOWS.items()):
        ax.set_facecolor("#f8fafc")
        _shade_crises(ax, vix, start)
        for strat in SO:
            entry = results.get((window, strat))
            if not entry:
                continue
            ret = entry["returns"]
            cum = (1 + ret).cumprod()
            ax.semilogy(cum.index, cum.values, label=strat,
                        color=COLORS[strat],
                        linewidth=2.0 if strat != "Unhedged" else 1.4,
                        linestyle="--" if strat == "Unhedged" else "-",
                        alpha=0.9)
            ax.annotate(f" {strat[0]}: ×{cum.iloc[-1]:.1f}",
                        xy=(cum.index[-1], cum.iloc[-1]),
                        fontsize=7, color=COLORS[strat], va="center")

        ax.set_title(f"{window}  ({start} → {bp.END_DATE})",
                     fontweight="bold", fontsize=9, color=HDR)
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:.0f}"))
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(pdf, fig)


def page_equity_single(pdf, results, vix):
    """All 4 strategies on ONE graph — 20Y window."""
    window = "20Y"
    start  = bp.WINDOWS[window]
    fig, ax = plt.subplots(figsize=(PW, PH * 0.70), facecolor="white")
    ax.set_facecolor("#f8fafc")
    _shade_crises(ax, vix, start)

    for strat in SO:
        entry = results.get((window, strat))
        if not entry:
            continue
        ret = entry["returns"]
        cum = (1 + ret).cumprod()
        m   = entry["metrics"]
        lbl = (f"{strat}  |  CAGR {m['cagr']:+.1%}  "
               f"Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.0%}")
        ax.semilogy(cum.index, cum.values, label=lbl,
                    color=COLORS[strat],
                    linewidth=2.2 if strat != "Unhedged" else 1.5,
                    linestyle="--" if strat == "Unhedged" else "-",
                    alpha=0.92)

    ax.set_title(f"ALL STRATEGIES — {window} COMPARISON  ({start} → {bp.END_DATE})\n"
                 "Red shading = VIX > 25  (crisis periods)",
                 fontweight="bold", fontsize=10, color=HDR, pad=8)
    ax.set_ylabel("Growth of $1  (log scale)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.88)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save(pdf, fig)


def page_drawdowns(pdf, results, vix):
    fig, axes = plt.subplots(3, 1, figsize=(PW, PH), facecolor="white")
    fig.suptitle("DRAWDOWN COMPARISON — ALL WINDOWS\n"
                 "Red shading = VIX > 25",
                 fontsize=10, fontweight="bold", color=HDR, y=0.99)

    for ax, (window, start) in zip(axes, bp.WINDOWS.items()):
        ax.set_facecolor("#f8fafc")
        _shade_crises(ax, vix, start)
        for strat in SO:
            entry = results.get((window, strat))
            if not entry:
                continue
            ret = entry["returns"]
            cum = (1 + ret).cumprod()
            dd  = cum / cum.cummax() - 1
            ax.fill_between(dd.index, dd.values, 0, alpha=0.28, color=COLORS[strat])
            ax.plot(dd.index, dd.values, color=COLORS[strat], linewidth=0.9,
                    alpha=0.85, label=f"{strat} ({dd.min():.0%})")
        ax.set_title(f"{window}  ({start} → {bp.END_DATE})",
                     fontweight="bold", fontsize=9, color=HDR)
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_ylim(-0.90, 0.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(pdf, fig)


def page_bar_charts(pdf, results):
    metrics_cfg = [
        ("sharpe",   "Sharpe Ratio",        False, False),
        ("cagr",     "CAGR",                True,  False),
        ("max_dd",   "Max Drawdown (abs.)", True,  True),
        ("calmar",   "Calmar Ratio",        False, False),
        ("vol",      "Annualised Vol",      True,  False),
        ("sortino",  "Sortino Ratio",       False, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(LW, LH), facecolor="white")
    fig.suptitle("METRICS COMPARISON — ALL 12 RESULT SETS",
                 fontsize=12, fontweight="bold", color=HDR, y=0.99)
    axes = axes.flatten()

    windows     = list(bp.WINDOWS.keys())
    x           = np.arange(len(windows))
    n           = len(SO)
    bar_w       = 0.72 / n

    for ax, (metric, title, as_pct, take_abs) in zip(axes, metrics_cfg):
        ax.set_facecolor("#f8fafc")
        for i, strat in enumerate(SO):
            vals = []
            for window in windows:
                m = results.get((window, strat), {}).get("metrics", {})
                v = m.get(metric, float("nan")) if m else float("nan")
                if take_abs and not math.isnan(v):
                    v = abs(v)
                vals.append(v)
            offset = (i - n / 2.0 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w * 0.9, label=strat,
                          color=COLORS[strat], alpha=0.85,
                          edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if math.isnan(v):
                    continue
                lbl = f"{v:.0%}" if as_pct else f"{v:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        lbl, ha="center", va="bottom", fontsize=6.0, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=9, color=HDR)
        ax.set_xticks(x)
        ax.set_xticklabels(windows, fontsize=8)
        if as_pct:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=7, framealpha=0.85)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save(pdf, fig)


def page_rolling_sharpe(pdf, results):
    window = "20Y"
    fig, ax = plt.subplots(figsize=(PW, PH * 0.65), facecolor="white")
    ax.set_facecolor("#f8fafc")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(1, color="#999", linewidth=0.5, linestyle=":", label="Sharpe = 1")
    ax.axhline(2, color="#bbb", linewidth=0.5, linestyle=":")

    for strat in SO:
        entry = results.get((window, strat))
        if not entry:
            continue
        r = entry["returns"]
        rs = ((r.rolling(252).mean() * 252 - RF) /
              (r.rolling(252).std() * SQRT252 + 1e-12))
        ax.plot(rs.index, rs.values, label=strat, color=COLORS[strat],
                linewidth=1.8, alpha=0.88)

    ax.set_title(f"ROLLING 252-DAY SHARPE RATIO  —  {window}  "
                 f"({bp.WINDOWS[window]} → {bp.END_DATE})",
                 fontweight="bold", fontsize=10, color=HDR)
    ax.set_ylabel("Trailing 1-Year Sharpe", fontsize=9)
    ax.set_ylim(-3.5, 5.5)
    ax.legend(fontsize=8.5, framealpha=0.88, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    today = str(date.today())
    print("=" * 60)
    print("  GENERATING PORTFOLIO REPORT")
    print(f"  Output: {REPORT_PATH}")
    print("=" * 60)

    print("\n[1/2] Building personal portfolio data...")
    trades = load_all_trades()
    print(f"  Loaded {len(trades)} trades")
    personal = build_personal(trades)

    print("\n[2/2] Running strategy backtest...")
    bt_results, vix = run_backtest()

    print(f"\nGenerating PDF ({REPORT_PATH})...")
    with PdfPages(REPORT_PATH) as pdf:
        meta = pdf.infodict()
        meta["Title"]   = "Investment Performance Report"
        meta["Author"]  = "HedgeFundApp"
        meta["Subject"] = "Portfolio & Strategy Backtest"

        page_cover(pdf, today)
        page_personal_metrics(pdf, personal["metrics"])
        page_positions(pdf, personal["pos"])
        page_ticker_vol(pdf, personal["vol_rows"])
        page_trade_log(pdf, personal["trades"])
        page_nav_chart(pdf, personal["nav"], personal["twr"])

        page_strategy_params(pdf)
        page_backtest_results(pdf, bt_results, "20Y", bp.WINDOWS["20Y"])
        page_backtest_results(pdf, bt_results, "10Y", bp.WINDOWS["10Y"])
        page_backtest_results(pdf, bt_results, "5Y",  bp.WINDOWS["5Y"])
        page_combined_metrics(pdf, bt_results)
        page_equity_all_windows(pdf, bt_results, vix)
        page_equity_single(pdf, bt_results, vix)
        page_drawdowns(pdf, bt_results, vix)
        page_bar_charts(pdf, bt_results)
        page_rolling_sharpe(pdf, bt_results)

    print(f"\nDone. {REPORT_PATH} — "
          f"{REPORT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
