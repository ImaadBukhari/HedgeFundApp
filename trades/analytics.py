#!/usr/bin/env python3
"""
Portfolio analytics: per-ticker volatility, fund-wide risk-adjusted metrics,
Sharpe, Calmar, Sortino, max drawdown, and charts.

Run:  python trades/analytics.py
"""

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TRADES_FILE  = Path(__file__).parent / "trades.json"
OUTPUT_DIR   = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RISK_FREE    = 0.045          # annualised risk-free rate (T-bill)
EXCLUDED     = {"AMD_PUT"}    # options / non-priceable

STYLE = {
    "axes.facecolor":   "#0f1117",
    "figure.facecolor": "#0f1117",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0a0",
    "ytick.color":      "#a0a0a0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3a",
    "grid.linewidth":   0.6,
}
plt.rcParams.update(STYLE)

TICKER_COLORS = {
    "VOO":   "#4f8ef7", "RKLB":  "#f74f8e", "ASML":  "#f7a14f",
    "SLV":   "#c0c0c0", "GLD":   "#ffd700", "BRKR":  "#4ff7c0",
    "APP":   "#b44ff7", "AAPL":  "#34c759", "JPM":   "#5856d6",
    "QBTS":  "#ff6b35", "MDA.TO":"#30d5c8", "AMD_PUT":"#666666",
}
DEFAULT_COLOR = "#7eb8f7"


# ---------------------------------------------------------------------------
# Load & parse trades
# ---------------------------------------------------------------------------

def load_trades():
    with open(TRADES_FILE) as f:
        data = json.load(f)
    rows = []
    for t in data["trades"]:
        rows.append({
            "date":       pd.Timestamp(t["date"]),
            "ticker":     t["ticker"],
            "action":     t["action"],
            "amount_usd": float(t["amount_usd"]),
            "amount_cad": float(t.get("amount_cad", t["amount_usd"])),
            "currency":   t.get("currency", "CAD"),
            "shares":     t.get("shares"),  # explicit override; None = infer from price
        })
    return sorted(rows, key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# Fetch price history
# ---------------------------------------------------------------------------

def get_price_on_date(hist: pd.Series, target: pd.Timestamp) -> float:
    available = hist[hist.index <= target + pd.Timedelta(days=1)]
    if available.empty:
        available = hist
    return float(available.iloc[-1])


def fetch_all_histories(tickers, start, end):
    print(f"Fetching price history for: {sorted(tickers)}")
    raw = yf.download(list(tickers), start=start, end=end + pd.Timedelta(days=2),
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=list(tickers)[0])
    return raw


# ---------------------------------------------------------------------------
# Build per-ticker position & P&L
# ---------------------------------------------------------------------------

def build_positions(trades, price_histories):
    """
    Returns:
      positions  : {ticker: {shares, avg_cost, total_cost, realized_pnl, dividends}}
      trade_log  : list of enriched trade dicts (with price, shares)
    """
    positions = {}
    trade_log = []

    for t in trades:
        tk     = t["ticker"]
        action = t["action"]
        amt    = t["amount_usd"]
        dt     = t["date"]

        if tk in EXCLUDED:
            pos = positions.setdefault(tk, dict(
                shares=0, total_cost=0, realized_pnl=0, dividends=0))
            if action == "buy":
                pos["total_cost"]    += amt
                pos["realized_pnl"]  -= amt   # option cost = loss when expired
            continue

        if action == "dividend":
            pos = positions.setdefault(tk, dict(
                shares=0, total_cost=0, realized_pnl=0, dividends=0))
            pos["dividends"] += amt
            continue

        hist = price_histories.get(tk)
        if hist is None or hist.dropna().empty:
            print(f"  WARN: no price data for {tk} — skipping")
            continue

        price  = get_price_on_date(hist.dropna(), dt)
        if t.get("shares") is not None:
            shares = float(t["shares"])
        else:
            amt_for_shares = t.get("amount_cad", amt) if tk.endswith(".TO") else amt
            shares = amt_for_shares / price if price > 0 else 0.0

        pos = positions.setdefault(tk, dict(
            shares=0, total_cost=0, realized_pnl=0, dividends=0))

        if action == "buy":
            pos["shares"]     += shares
            pos["total_cost"] += amt

        elif action == "sell":
            avg = pos["total_cost"] / pos["shares"] if pos["shares"] > 0 else 0
            cost_sold           = avg * shares
            pos["shares"]      -= shares
            pos["total_cost"]  -= cost_sold
            pos["realized_pnl"] += amt - cost_sold

        trade_log.append({**t, "price": price, "shares_traded": shares})

    return positions, trade_log


# ---------------------------------------------------------------------------
# Daily NAV reconstruction
# ---------------------------------------------------------------------------

def build_daily_nav(trade_log, price_histories):
    all_dates = None
    for hist in price_histories.values():
        idx = hist.dropna().index
        all_dates = idx if all_dates is None else all_dates.union(idx)

    if all_dates is None or len(all_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    tickers = list(price_histories.keys())
    holdings = pd.DataFrame(0.0, index=all_dates, columns=tickers)

    # Also track daily net cash invested (buys - sells) to compute TWR
    net_cash_flow = pd.Series(0.0, index=all_dates)

    tl = sorted(trade_log, key=lambda x: x["date"])
    for t in tl:
        tk   = t["ticker"]
        act  = t["action"]
        sh   = t["shares_traded"]
        dt   = t["date"]
        amt  = t["amount_usd"]
        if tk not in tickers:
            continue
        mask = holdings.index >= dt
        # Find the exact date in index closest to trade date
        trade_idx = all_dates[all_dates >= dt]
        if len(trade_idx) == 0:
            continue
        exact_dt = trade_idx[0]

        if act == "buy":
            holdings.loc[mask, tk] += sh
            net_cash_flow.loc[exact_dt] += amt   # cash in
        elif act == "sell":
            holdings.loc[mask, tk] = (holdings.loc[mask, tk] - sh).clip(lower=0)
            net_cash_flow.loc[exact_dt] -= amt   # cash out (reduces invested base)

    prices_df = pd.DataFrame({tk: price_histories[tk] for tk in tickers},
                              index=all_dates).ffill()
    nav = (holdings * prices_df).sum(axis=1)
    nav = nav[nav > 0]
    return nav, net_cash_flow


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

SQRT252 = math.sqrt(252)

def twr_returns(nav: pd.Series, net_cash_flow: pd.Series) -> pd.Series:
    """
    Time-weighted daily returns: strips out the effect of new capital injections.
    Formula: r_t = (NAV_t - NAV_{t-1} - cashflow_t) / NAV_{t-1}
    """
    nav_prev = nav.shift(1)
    cf       = net_cash_flow.reindex(nav.index).fillna(0)
    r = (nav - nav_prev - cf) / nav_prev.replace(0, np.nan)
    return r.dropna()

def sharpe(returns: pd.Series) -> float:
    mu  = returns.mean() * 252
    sig = returns.std() * SQRT252
    return (mu - RISK_FREE) / sig if sig > 0 else float("nan")

def sortino(returns: pd.Series) -> float:
    mu        = returns.mean() * 252
    downside  = returns[returns < 0].std() * SQRT252
    return (mu - RISK_FREE) / downside if downside > 0 else float("nan")

def calmar(returns: pd.Series, nav: pd.Series) -> float:
    mu  = returns.mean() * 252
    mdd = max_drawdown(nav)
    return mu / abs(mdd) if mdd < 0 else float("nan")

def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())

def cagr(returns: pd.Series) -> float:
    """CAGR from a time-weighted daily returns series."""
    years = len(returns) / 252
    compounded = float((1 + returns).prod())
    return float(compounded ** (1 / years) - 1) if years > 0 else float("nan")

def var_95(returns: pd.Series) -> float:
    return float(np.percentile(returns.dropna(), 5))

def cvar_95(returns: pd.Series) -> float:
    v = var_95(returns)
    return float(returns[returns <= v].mean())

def ticker_volatility(hist: pd.Series) -> dict:
    r   = hist.pct_change().dropna()
    vol = float(r.std() * SQRT252)
    return {
        "daily_vol":   float(r.std()),
        "annual_vol":  vol,
        "skewness":    float(r.skew()),
        "kurtosis":    float(r.kurt()),
        "best_day":    float(r.max()),
        "worst_day":   float(r.min()),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_nav(nav: pd.Series, returns: pd.Series):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle("Portfolio NAV & Risk", fontsize=16, fontweight="bold",
                 color="white", y=0.98)

    # --- NAV curve + drawdown shading ---
    ax = axes[0]
    ax.plot(nav.index, nav.values, color="#4f8ef7", linewidth=1.8, label="NAV (USD)")
    drawdown = nav / nav.cummax() - 1
    ax2 = ax.twinx()
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                     color="#f74f4f", alpha=0.25, label="Drawdown %")
    ax2.set_ylabel("Drawdown %", color="#f74f4f", fontsize=9)
    ax2.tick_params(colors="#f74f4f")
    ax2.spines["right"].set_color("#f74f4f")
    ax.set_ylabel("Portfolio Value (USD)", fontsize=10)
    ax.set_title("NAV with Drawdown Overlay", fontsize=11, pad=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    # --- Daily returns bar chart ---
    ax = axes[1]
    colours = ["#4ff7a0" if r >= 0 else "#f74f4f" for r in returns]
    ax.bar(returns.index, returns.values * 100, color=colours,
           width=1.0, alpha=0.85)
    ax.axhline(0, color="#555", linewidth=0.6)
    ax.set_ylabel("Daily Return %", fontsize=10)
    ax.set_title("Daily Returns", fontsize=11, pad=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # --- Rolling 30-day Sharpe ---
    ax = axes[2]
    rf_daily  = RISK_FREE / 252
    roll_ret  = returns.rolling(30).mean() * 252
    roll_vol  = returns.rolling(30).std() * SQRT252
    roll_sh   = (roll_ret - RISK_FREE) / roll_vol.replace(0, np.nan)
    ax.plot(roll_sh.index, roll_sh.values, color="#f7a14f", linewidth=1.5)
    ax.axhline(0, color="#555", linewidth=0.6)
    ax.fill_between(roll_sh.index, 0, roll_sh.values,
                    where=roll_sh.values >= 0, alpha=0.2, color="#4ff7a0")
    ax.fill_between(roll_sh.index, 0, roll_sh.values,
                    where=roll_sh.values < 0, alpha=0.2, color="#f74f4f")
    ax.set_ylabel("Rolling Sharpe (30d)", fontsize=10)
    ax.set_title("30-Day Rolling Sharpe Ratio", fontsize=11, pad=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "nav_and_risk.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved: {path}")


def plot_ticker_volatility(vol_data: dict, positions: dict, price_histories: dict):
    tickers = [tk for tk in vol_data if tk not in EXCLUDED]
    if not tickers:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Per-Ticker Analysis", fontsize=15, fontweight="bold",
                 color="white", y=0.99)

    # --- Annualised vol bar ---
    ax = axes[0, 0]
    vols  = [vol_data[tk]["annual_vol"] * 100 for tk in tickers]
    cols  = [TICKER_COLORS.get(tk, DEFAULT_COLOR) for tk in tickers]
    bars  = ax.bar(tickers, vols, color=cols, edgecolor="#222", linewidth=0.5)
    ax.set_title("Annualised Volatility %", fontsize=11)
    ax.set_ylabel("Vol %")
    for bar, v in zip(bars, vols):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color="white")
    ax.tick_params(axis="x", rotation=35)

    # --- Current allocation pie ---
    ax = axes[0, 1]
    current_values = {}
    for tk in tickers:
        pos  = positions.get(tk, {})
        sh   = pos.get("shares", 0)
        hist = price_histories.get(tk)
        if sh > 0.001 and hist is not None and not hist.dropna().empty:
            price = float(hist.dropna().iloc[-1])
            current_values[tk] = sh * price

    if current_values:
        labels = list(current_values.keys())
        sizes  = list(current_values.values())
        pcols  = [TICKER_COLORS.get(tk, DEFAULT_COLOR) for tk in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=pcols, startangle=140,
            textprops={"color": "white", "fontsize": 9},
            wedgeprops={"edgecolor": "#0f1117", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Current Holdings Allocation", fontsize=11)

    # --- P&L bar (realised + unrealised) ---
    ax = axes[1, 0]
    pnl_data = {}
    for tk in tickers:
        pos  = positions.get(tk, {})
        sh   = pos.get("shares", 0)
        cost = pos.get("total_cost", 0)
        rea  = pos.get("realized_pnl", 0)
        div  = pos.get("dividends", 0)
        hist = price_histories.get(tk)
        if sh > 0.001 and hist is not None and not hist.dropna().empty:
            unr = sh * float(hist.dropna().iloc[-1]) - cost
        else:
            unr = 0
        total_pnl = rea + unr + div
        if abs(total_pnl) > 0.01 or abs(rea) > 0.01:
            pnl_data[tk] = total_pnl

    if pnl_data:
        tk_list   = list(pnl_data.keys())
        pnl_list  = list(pnl_data.values())
        bar_cols  = ["#4ff7a0" if v >= 0 else "#f74f4f" for v in pnl_list]
        bars = ax.bar(tk_list, pnl_list, color=bar_cols,
                      edgecolor="#222", linewidth=0.5)
        ax.axhline(0, color="#888", linewidth=0.7)
        ax.set_title("Total P&L by Ticker (USD)", fontsize=11)
        ax.set_ylabel("P&L $")
        for bar, v in zip(bars, pnl_list):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if v >= 0 else -4),
                    f"${v:+.0f}", ha="center", va="bottom", fontsize=8, color="white")
        ax.tick_params(axis="x", rotation=35)

    # --- Individual price history lines ---
    ax = axes[1, 1]
    for tk in tickers:
        hist = price_histories.get(tk)
        if hist is None or hist.dropna().empty:
            continue
        h = hist.dropna()
        normalised = h / h.iloc[0] * 100
        ax.plot(normalised.index, normalised.values,
                label=tk, color=TICKER_COLORS.get(tk, DEFAULT_COLOR),
                linewidth=1.2, alpha=0.85)
    ax.axhline(100, color="#555", linewidth=0.6, linestyle="--")
    ax.set_title("Price Performance (Indexed to 100)", fontsize=11)
    ax.set_ylabel("Indexed Price")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "ticker_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved: {path}")


def plot_return_distribution(returns: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Return Distribution", fontsize=14, fontweight="bold",
                 color="white")

    # Histogram
    ax = axes[0]
    n, bins, patches = ax.hist(returns * 100, bins=50, color="#4f8ef7",
                                edgecolor="#0f1117", linewidth=0.3, alpha=0.85)
    v95  = var_95(returns) * 100
    cv95 = cvar_95(returns) * 100
    ax.axvline(v95,  color="#ff6b35", linestyle="--", linewidth=1.5,
               label=f"VaR 95%: {v95:.2f}%")
    ax.axvline(cv95, color="#f74f4f", linestyle="--", linewidth=1.5,
               label=f"CVaR 95%: {cv95:.2f}%")
    ax.axvline(returns.mean() * 100, color="#4ff7a0", linestyle="-",
               linewidth=1.2, label=f"Mean: {returns.mean()*100:.3f}%")
    ax.set_xlabel("Daily Return %")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily Return Distribution")
    ax.legend(fontsize=9)

    # Q-Q plot vs normal
    ax = axes[1]
    from scipy import stats
    (osm, osr), (slope, intercept, _) = stats.probplot(returns.dropna(), dist="norm")
    ax.scatter(osm, osr, color="#4f8ef7", s=8, alpha=0.6, label="Portfolio returns")
    x_line = np.array([min(osm), max(osm)])
    ax.plot(x_line, slope * x_line + intercept, color="#f7a14f",
            linewidth=1.5, label="Normal reference")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Values")
    ax.set_title("Q-Q Plot (Fat Tails?)")
    ax.legend(fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "return_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_metrics(nav: pd.Series, returns: pd.Series, positions: dict,
                  vol_data: dict, price_histories: dict, trades: list):

    first_date = min(t["date"] for t in trades).date()
    last_date  = nav.index[-1].date()
    n_days     = (last_date - first_date).days

    ann_vol  = returns.std() * SQRT252
    sh       = sharpe(returns)
    so       = sortino(returns)
    annual_g = cagr(returns)
    # Max drawdown from compounded TWR curve (not raw NAV)
    twr_curve = (1 + returns).cumprod()
    mdd       = max_drawdown(twr_curve)
    cal       = (annual_g - RISK_FREE) / abs(mdd) if mdd < 0 else float("nan")
    v95       = var_95(returns)
    cv95      = cvar_95(returns)

    W = 60
    div = "─" * W

    print(f"\n{'═'*W}")
    print(f"  PORTFOLIO PERFORMANCE REPORT".center(W))
    print(f"  {first_date} → {last_date}  ({n_days} days)".center(W))
    print(f"{'═'*W}")

    print(f"\n  {'FUND-WIDE RISK-ADJUSTED METRICS':}")
    print(f"  {div}")
    print(f"  {'Sharpe Ratio':<30} {sh:>+.4f}")
    print(f"  {'Sortino Ratio':<30} {so:>+.4f}")
    print(f"  {'Calmar Ratio':<30} {cal:>+.4f}")
    print(f"  {div}")
    print(f"  {'Annualised Return (CAGR)':<30} {annual_g*100:>+.2f}%")
    print(f"  {'Annualised Volatility':<30} {ann_vol*100:>.2f}%")
    print(f"  {'Max Drawdown':<30} {mdd*100:>.2f}%")
    print(f"  {'VaR 95% (daily)':<30} {v95*100:>.3f}%")
    print(f"  {'CVaR / Expected Shortfall 95%':<30} {cv95*100:>.3f}%")
    print(f"  {'Return Skewness':<30} {float(returns.skew()):>.4f}")
    print(f"  {'Return Kurtosis (excess)':<30} {float(returns.kurt()):>.4f}")
    print(f"  {'Risk-Free Rate Used':<30} {RISK_FREE*100:.1f}%")

    # Best / worst days
    best_day  = returns.idxmax()
    worst_day = returns.idxmin()
    print(f"\n  {'BEST / WORST DAYS':}")
    print(f"  {div}")
    print(f"  {'Best day':<30} {returns[best_day]*100:>+.2f}%  ({best_day.date()})")
    print(f"  {'Worst day':<30} {returns[worst_day]*100:>+.2f}%  ({worst_day.date()})")

    # Current portfolio snapshot
    total_cost = 0
    total_mkt  = 0
    total_real = 0
    total_div  = 0

    print(f"\n  {'PER-TICKER SUMMARY':}")
    print(f"  {div}")
    hdr = f"  {'TICKER':<8} {'COST':>8} {'MKT VAL':>9} {'REAL PNL':>10} {'UNRL PNL':>10} {'TOT PNL':>10} {'RET%':>7} {'ANN VOL%':>9}"
    print(hdr)
    print(f"  {div}")  # div is still the dash string here — loop below uses div_amt

    ticker_rows = []
    for tk, pos in sorted(positions.items()):
        if tk in EXCLUDED:
            cost      = pos.get("total_cost", 0)
            rea       = pos.get("realized_pnl", 0)
            total_pnl = rea
            ret_pct   = rea / cost * 100 if cost > 0 else 0
            ticker_rows.append((tk, cost, 0, rea, 0, 0, total_pnl, ret_pct, None))
            total_cost += cost
            total_real += rea
            continue

        cost     = pos.get("total_cost", 0)
        rea      = pos.get("realized_pnl", 0)
        div_amt  = pos.get("dividends", 0)
        sh       = pos.get("shares", 0)
        hist     = price_histories.get(tk)

        if sh > 0.001 and hist is not None and not hist.dropna().empty:
            mkt   = sh * float(hist.dropna().iloc[-1])
            unr   = mkt - cost
        else:
            mkt = unr = 0.0

        total_invested = sum(
            t["amount_usd"] for t in trades
            if t["ticker"] == tk and t["action"] == "buy"
        )
        total_pnl = rea + unr + div_amt
        ret_pct   = total_pnl / total_invested * 100 if total_invested > 0 else 0

        v = vol_data.get(tk, {}).get("annual_vol")
        ticker_rows.append((tk, total_invested, mkt, rea, unr, div_amt, total_pnl, ret_pct, v))

        total_cost += total_invested
        total_mkt  += mkt
        total_real += rea
        total_div  += div_amt

    for (tk, cost, mkt, rea, unr, div_amt, pnl, ret_pct, v) in sorted(
            ticker_rows, key=lambda x: x[7], reverse=True):
        vol_str = f"{v*100:.1f}%" if v is not None else "  n/a"
        sign    = "+" if pnl >= 0 else ""
        print(f"  {tk:<8} ${cost:>7.0f} ${mkt:>8.0f} ${rea:>+9.0f} ${unr:>+9.0f} ${pnl:>+9.0f} {ret_pct:>+6.1f}% {vol_str:>8}")

    total_pnl_all = total_real + (total_mkt - sum(
        p["total_cost"] for tk, p in positions.items() if tk not in EXCLUDED
    )) + total_div
    sep = "─" * W
    print(f"  {sep}")
    print(f"  {'TOTAL':<8} ${total_cost:>7.0f} ${total_mkt:>8.0f}  {'':9} {'':9} ${total_pnl_all:>+9.0f}")

    # Per-ticker volatility table
    print(f"\n  {'PER-TICKER VOLATILITY DETAIL':}")
    print(f"  {sep}")
    hdr2 = f"  {'TICKER':<8} {'ANN.VOL%':>9} {'DAILY.VOL%':>11} {'SKEW':>7} {'KURT':>7} {'BEST DAY':>9} {'WORST DAY':>10}"
    print(hdr2)
    print(f"  {sep}")
    for tk, vd in sorted(vol_data.items(), key=lambda x: x[1].get("annual_vol", 0), reverse=True):
        if tk in EXCLUDED:
            continue
        print(f"  {tk:<8} {vd['annual_vol']*100:>8.1f}% {vd['daily_vol']*100:>10.2f}%"
              f" {vd['skewness']:>7.3f} {vd['kurtosis']:>7.3f}"
              f" {vd['best_day']*100:>+8.2f}% {vd['worst_day']*100:>+9.2f}%")

    print(f"\n{'═'*W}\n")

    return {
        "sharpe": sh, "sortino": so, "calmar": cal,
        "cagr": annual_g, "ann_vol": ann_vol, "max_drawdown": mdd,
        "var_95": v95, "cvar_95": cv95,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} trades\n")

    tickers  = {t["ticker"] for t in trades if t["ticker"] not in EXCLUDED}
    start_dt = min(t["date"] for t in trades)
    end_dt   = pd.Timestamp.today()

    raw_hist = fetch_all_histories(tickers, start_dt, end_dt)

    price_histories = {}
    for tk in tickers:
        if tk in raw_hist.columns:
            price_histories[tk] = raw_hist[tk]
        else:
            print(f"  WARN: {tk} not in downloaded data")

    positions, trade_log = build_positions(trades, price_histories)

    print("\nBuilding daily NAV...")
    nav, net_cash_flow = build_daily_nav(trade_log, price_histories)
    if nav.empty or len(nav) < 5:
        print("ERROR: Not enough NAV data. Check yfinance connectivity.")
        return

    # Time-weighted returns: strip capital injection distortion
    returns = twr_returns(nav, net_cash_flow)
    returns = returns[returns.between(-0.5, 0.5)]  # clip extreme artefacts

    # Per-ticker vol (using full history from first purchase)
    vol_data = {}
    for tk, hist in price_histories.items():
        h = hist.dropna()
        # Only from when we first bought
        first_buy = next(
            (t["date"] for t in trades if t["ticker"] == tk and t["action"] == "buy"),
            None
        )
        if first_buy is not None:
            h = h[h.index >= first_buy]
        if len(h) > 5:
            vol_data[tk] = ticker_volatility(h)

    print("\nGenerating charts...")
    twr_curve = (1 + returns).cumprod()
    plot_nav(twr_curve, returns)
    plot_ticker_volatility(vol_data, positions, price_histories)
    plot_return_distribution(returns)

    print_metrics(nav, returns, positions, vol_data, price_histories, trades)

    print(f"All outputs in: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
