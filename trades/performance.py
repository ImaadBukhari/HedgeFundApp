#!/usr/bin/env python3
"""
Portfolio performance calculator.
Reads trades/trades.json, fetches historical & current prices via yfinance,
and outputs per-ticker and total portfolio stats.
"""

import json
import math
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TRADES_FILE = Path(__file__).parent / "trades.json"
RISK_FREE_RATE = 0.045  # annualised, approx current T-bill yield

# AMD_PUT is an option that expired — excluded from price lookups
EXCLUDED_TICKERS = {"AMD_PUT"}


# ---------------------------------------------------------------------------
# Load trades
# ---------------------------------------------------------------------------

def load_trades():
    with open(TRADES_FILE) as f:
        data = json.load(f)
    trades = []
    for t in data["trades"]:
        trades.append({
            "date":       pd.Timestamp(t["date"]),
            "ticker":     t["ticker"],
            "action":     t["action"],   # buy / sell / dividend
            "amount_usd": float(t["amount_usd"]),
            "amount_cad": float(t["amount_cad"]),
            "fx_rate":    float(t["fx_rate"]),
            "note":       t.get("note", ""),
        })
    return sorted(trades, key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# Fetch historical closing prices for a list of (ticker, date) pairs
# ---------------------------------------------------------------------------

def fetch_price_on_date(ticker: str, trade_date: pd.Timestamp) -> float:
    """Return closing price for ticker on or near trade_date (looks back up to 5 days)."""
    start = trade_date - pd.Timedelta(days=5)
    end   = trade_date + pd.Timedelta(days=2)
    hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if hist.empty:
        raise ValueError(f"No price data for {ticker} around {trade_date.date()}")
    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    # Get the closest available date on or before trade_date
    available = close[close.index <= trade_date + pd.Timedelta(days=1)]
    if available.empty:
        available = close
    return float(available.iloc[-1])


def fetch_current_price(ticker: str) -> float:
    hist = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
    if hist.empty:
        raise ValueError(f"No current price data for {ticker}")
    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return float(close.iloc[-1])


# ---------------------------------------------------------------------------
# Build positions using average cost basis
# ---------------------------------------------------------------------------

def build_positions(trades):
    """
    Returns:
        positions    : {ticker: {"shares": float, "avg_cost_usd": float, "total_cost_usd": float}}
        realized     : {ticker: {"proceeds": float, "cost": float, "pnl": float}}
        dividends    : {ticker: float}
        trade_prices : list of dicts with per-trade price (for portfolio value series)
    """
    positions = {}   # ticker -> {shares, total_cost_usd}
    realized  = {}   # ticker -> {proceeds, cost, pnl}
    dividends = {}   # ticker -> total USD received
    trade_prices = []

    tickers_needed = {t["ticker"] for t in trades if t["ticker"] not in EXCLUDED_TICKERS}
    print(f"Fetching historical prices for: {sorted(tickers_needed)}")

    for t in trades:
        ticker = t["ticker"]
        action = t["action"]
        amount = t["amount_usd"]
        dt     = t["date"]

        if ticker in EXCLUDED_TICKERS:
            # AMD_PUT: log as realized loss equal to cost paid
            if action == "buy":
                realized.setdefault(ticker, {"proceeds": 0.0, "cost": 0.0, "pnl": 0.0})
                realized[ticker]["cost"] += amount
            elif action == "sell":
                # expired worthless — proceeds already 0
                realized[ticker]["pnl"] = realized[ticker]["proceeds"] - realized[ticker]["cost"]
            continue

        if action == "dividend":
            dividends[ticker] = dividends.get(ticker, 0.0) + amount
            continue

        try:
            price = fetch_price_on_date(ticker, dt)
        except Exception as e:
            print(f"  WARNING: {e} — skipping trade", file=sys.stderr)
            continue

        shares_traded = amount / price if price > 0 else 0.0
        trade_prices.append({
            "date": dt, "ticker": ticker, "action": action,
            "price_usd": price, "shares": shares_traded, "amount_usd": amount
        })
        print(f"  {dt.date()} {action:4s} {ticker:6s}  ${price:>8.2f}/share  {shares_traded:>8.4f} shares  ${amount:>8.2f}")

        pos = positions.setdefault(ticker, {"shares": 0.0, "total_cost_usd": 0.0})
        rea = realized.setdefault(ticker,  {"proceeds": 0.0, "cost": 0.0, "pnl": 0.0})

        if action == "buy":
            pos["shares"]         += shares_traded
            pos["total_cost_usd"] += amount
            rea["cost"]           += amount

        elif action == "sell":
            if pos["shares"] > 0:
                avg_cost = pos["total_cost_usd"] / pos["shares"]
                cost_of_sold = avg_cost * shares_traded
                pos["shares"]         -= shares_traded
                pos["total_cost_usd"] -= cost_of_sold
                rea["proceeds"]       += amount
                rea["pnl"]            += (amount - cost_of_sold)
            else:
                rea["proceeds"] += amount

    return positions, realized, dividends, trade_prices


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_report(positions, realized, dividends, trades):
    print("\n" + "=" * 90)
    print("PORTFOLIO PERFORMANCE REPORT")
    print("=" * 90)

    first_date = min(t["date"] for t in trades).date()
    today      = date.today()
    days_held  = (today - first_date).days
    years_held = days_held / 365.25

    all_tickers = sorted(set(
        list(positions.keys()) + list(realized.keys()) + list(dividends.keys())
    ))

    total_cost       = 0.0
    total_mkt_value  = 0.0
    total_proceeds   = 0.0
    total_dividends  = sum(dividends.values())

    rows = []

    for ticker in all_tickers:
        pos = positions.get(ticker, {"shares": 0.0, "total_cost_usd": 0.0})
        rea = realized.get(ticker,  {"proceeds": 0.0, "cost": 0.0, "pnl": 0.0})
        div = dividends.get(ticker, 0.0)

        shares = pos["shares"]
        cost   = rea["cost"]

        if ticker in EXCLUDED_TICKERS:
            mkt_value    = 0.0
            current_price = 0.0
        elif shares > 0.001:
            try:
                current_price = fetch_current_price(ticker)
                mkt_value     = shares * current_price
            except Exception as e:
                print(f"  WARNING: current price failed for {ticker}: {e}", file=sys.stderr)
                current_price = 0.0
                mkt_value     = 0.0
        else:
            current_price = 0.0
            mkt_value     = 0.0

        unrealized_pnl = mkt_value - pos["total_cost_usd"]
        total_pnl      = rea["pnl"] + unrealized_pnl + div
        total_return_pct = (total_pnl / cost * 100) if cost > 0 else 0.0

        rows.append({
            "ticker":         ticker,
            "cost":           cost,
            "realized_pnl":   rea["pnl"],
            "mkt_value":      mkt_value,
            "unrealized_pnl": unrealized_pnl,
            "dividends":      div,
            "total_pnl":      total_pnl,
            "return_pct":     total_return_pct,
            "shares":         shares,
            "current_price":  current_price,
        })

        total_cost      += cost
        total_mkt_value += mkt_value
        total_proceeds  += rea["proceeds"]

    # Per-ticker table
    col = "{:<8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}"
    print(col.format("TICKER", "COST($)", "REAL.PNL", "MKT.VAL", "UNRL.PNL", "DIV($)", "TOT.PNL"))
    print("-" * 68)
    for r in rows:
        print(col.format(
            r["ticker"],
            f"${r['cost']:>8.2f}",
            f"${r['realized_pnl']:>+8.2f}",
            f"${r['mkt_value']:>8.2f}",
            f"${r['unrealized_pnl']:>+8.2f}",
            f"${r['dividends']:>6.2f}",
            f"${r['total_pnl']:>+8.2f}",
        ))

    print("=" * 68)

    total_pnl_all = sum(r["total_pnl"] for r in rows)
    total_return  = (total_pnl_all / total_cost * 100) if total_cost > 0 else 0.0
    cagr          = ((1 + total_return / 100) ** (1 / years_held) - 1) * 100 if years_held > 0 else 0.0

    print(f"\nSUMMARY")
    print(f"  Period           : {first_date} → {today}  ({days_held} days)")
    print(f"  Total invested   : ${total_cost:>10.2f}")
    print(f"  Current mkt val  : ${total_mkt_value:>10.2f}")
    print(f"  Realized P&L     : ${sum(r['realized_pnl'] for r in rows):>+10.2f}")
    print(f"  Unrealized P&L   : ${sum(r['unrealized_pnl'] for r in rows):>+10.2f}")
    print(f"  Dividends recv.  : ${total_dividends:>10.2f}")
    print(f"  Total P&L        : ${total_pnl_all:>+10.2f}")
    print(f"  Total return     : {total_return:>+8.2f}%")
    print(f"  CAGR             : {cagr:>+8.2f}%")

    # Per-ticker return %
    print(f"\nRETURN BY TICKER")
    for r in sorted(rows, key=lambda x: x["return_pct"], reverse=True):
        bar = "█" * int(abs(r["return_pct"]) / 2)
        sign = "+" if r["return_pct"] >= 0 else ""
        print(f"  {r['ticker']:<6}  {sign}{r['return_pct']:>6.1f}%  {bar}")

    # Sharpe estimate (simplified: total return vs risk-free over period)
    # For a proper daily Sharpe we'd need daily NAV — flagged for future build
    excess_ann = cagr / 100 - RISK_FREE_RATE
    # Use a rough vol estimate from the return dispersion across tickers
    ticker_returns = [r["return_pct"] / 100 for r in rows if r["cost"] > 50]
    if len(ticker_returns) > 1:
        cross_vol = float(np.std(ticker_returns, ddof=1))
        sharpe_approx = excess_ann / cross_vol if cross_vol > 0 else float("nan")
        print(f"\n  NOTE: Sharpe below is a cross-ticker approximation.")
        print(f"  For a time-series Sharpe, run: python performance.py --daily")
        print(f"  Approx Sharpe (vs {RISK_FREE_RATE*100:.1f}% rf): {sharpe_approx:>+.2f}")

    print("\n  Flag: Jan 5 'BOO' logged as VOO — confirm ticker is correct.")
    print("=" * 90)

    return rows


# ---------------------------------------------------------------------------
# Daily NAV series + time-series Sharpe (--daily flag)
# ---------------------------------------------------------------------------

def daily_sharpe(trades, trade_prices):
    """
    Reconstruct a daily portfolio NAV and compute annualised Sharpe.
    Requires downloading full price history for each ticker held.
    """
    print("\nBuilding daily NAV series...")

    tickers = {t["ticker"] for t in trades if t["ticker"] not in EXCLUDED_TICKERS}
    start_date = min(t["date"] for t in trades)

    # Download full price history
    prices = yf.download(
        list(tickers), start=start_date, end=date.today() + timedelta(days=1),
        auto_adjust=True, progress=False
    )["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Build shares-held series day by day
    all_dates = prices.index
    holdings = pd.DataFrame(0.0, index=all_dates, columns=list(tickers))

    tp_df = pd.DataFrame(trade_prices).sort_values("date")

    for _, row in tp_df.iterrows():
        t, tk, act, sh = row["date"], row["ticker"], row["action"], row["shares"]
        if tk not in tickers:
            continue
        future = holdings.index >= t
        if act == "buy":
            holdings.loc[future, tk] += sh
        elif act == "sell":
            holdings.loc[future, tk] -= sh

    holdings = holdings.clip(lower=0)

    # NAV = sum(shares * price) each day
    nav = (holdings * prices.reindex(columns=holdings.columns)).sum(axis=1).dropna()
    nav = nav[nav > 0]

    if len(nav) < 10:
        print("  Not enough NAV data for time-series Sharpe.")
        return

    daily_ret = nav.pct_change().dropna()
    mean_ann  = daily_ret.mean() * 252
    vol_ann   = daily_ret.std() * math.sqrt(252)
    sharpe    = (mean_ann - RISK_FREE_RATE) / vol_ann if vol_ann > 0 else float("nan")
    max_dd    = ((nav / nav.cummax()) - 1).min()
    calmar    = mean_ann / abs(max_dd) if max_dd < 0 else float("nan")

    print(f"\n  Daily NAV Sharpe  : {sharpe:>+.3f}")
    print(f"  Ann. Return       : {mean_ann*100:>+.2f}%")
    print(f"  Ann. Volatility   : {vol_ann*100:>.2f}%")
    print(f"  Max Drawdown      : {max_dd*100:>.2f}%")
    print(f"  Calmar Ratio      : {calmar:>+.3f}")

    nav.to_csv(Path(__file__).parent / "nav_series.csv")
    print(f"  Daily NAV saved to trades/nav_series.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_daily = "--daily" in sys.argv

    trades = load_trades()
    print(f"Loaded {len(trades)} trades spanning "
          f"{trades[0]['date'].date()} → {trades[-1]['date'].date()}\n")

    positions, realized, dividends, trade_prices = build_positions(trades)
    rows = print_report(positions, realized, dividends, trades)

    if run_daily:
        daily_sharpe(trades, trade_prices)


if __name__ == "__main__":
    main()
