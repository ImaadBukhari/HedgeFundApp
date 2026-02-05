#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download and store SPY options snapshot data locally (CSV), with strict cost controls.

What it does:
- For 4 selected days per month in [START, END]:
  1) Find an expiration near TARGET_DTE (within DTE_WINDOW) using OPRA definitions.
  2) Pull one snapshot per selected day at SNAPSHOT_TIME_UTC.
  3) Store normalized rows to disk (compressed CSV), one file per (date, expiration).

Output:
  data/opra_spy_snapshots/
    definitions/YYYY-MM-DD_def.csv.gz
    snapshots/YYYY-MM-DD_expYYYY-MM-DD.csv.gz
    manifest.csv

Requirements:
  pip install databento pandas numpy python-dateutil

Auth:
  export DATABENTO_API_KEY="..."
"""

import os
import sys
import time
import math
import gzip
import json
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import pandas as pd
import databento as dbn


# =========================
# CONFIG (EDIT THESE)
# =========================

DATASET = "OPRA.PILLAR"
PARENT_SYMBOL = "SPY.OPT"        # parent symbology for SPY options in Databento
UNDERLYING = "SPY"              # for naming only

START = "2023-05-01"            # recommend start >= 2023-03-28 for best OPRA coverage
END   = "2025-12-31"

# choose expiry near ~45 DTE like you wanted
TARGET_DTE = 45
DTE_WINDOW = (25, 75)

# Strike band to keep the chain “liquid-ish”
STRIKE_BAND = (0.80, 1.20)      # relative to spot

# Snapshots: 4 per month (one snapshot per selected day)
# Days of month to sample (will use nearest trading day if not a trading day)
SAMPLE_DAYS_OF_MONTH = [1, 8, 15, 22]  # approximately weekly spacing

# Single snapshot time per day (UTC)
# US market hours ~ 13:30–20:00 UTC (depending on DST). This is a "safe-ish" time.
SNAPSHOT_TIME_UTC = "16:30"  # ~11:30 AM ET

# Each snapshot requests a tiny window around the time (minutes)
WINDOW_MINUTES = 2              # total window size; we'll pull [t - 1, t + 1]

# Databento schema
# cbbo-1m is consolidated BBO bars per minute; much cheaper than tick quotes.
SCHEMA_SNAPSHOT = "cbbo-1m"
SCHEMA_DEFINITION = "definition"

# Cost controls / safety rails
MAX_DEFINITION_ROWS = 200_000       # abort if definition payload is absurd
MAX_SYMBOLS_FOR_EXPIRY = 12_000     # abort if too many symbols
SYMBOL_BATCH_SIZE = 300             # avoid huge symbol lists per request
MAX_ROWS_PER_BATCH = 120_000        # abort if a single batch returns too much
MAX_DAYS_DEBUG = None               # e.g., set to 5 while testing

# Output folder
OUT_DIR = "data/opra_spy_snapshots"

# Retry/backoff
MAX_RETRIES = 4
BACKOFF_S = 2.0


# =========================
# Helpers
# =========================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def daterange_business_days(start: str, end: str) -> List[pd.Timestamp]:
    # business days; you can replace with an exchange calendar later if you want
    idx = pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end))
    return list(idx)

def get_sample_days_per_month(start: str, end: str) -> List[pd.Timestamp]:
    """
    Get 4 sample days per month (using SAMPLE_DAYS_OF_MONTH).
    For each month, picks the specified days, using the nearest trading day if needed.
    Uses actual business days to ensure we get trading days.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Get all business days in the range
    all_business_days = daterange_business_days(start, end)
    business_days_set = set(all_business_days)
    
    sample_days = []
    current = start_dt.replace(day=1)  # Start of first month
    
    while current <= end_dt:
        year = current.year
        month = current.month
        
        # For each target day in the month
        for day_of_month in SAMPLE_DAYS_OF_MONTH:
            try:
                # Try to create the date
                target_date = pd.Timestamp(year, month, day_of_month)
                
                # If it's before start or after end, skip
                if target_date < start_dt or target_date > end_dt:
                    continue
                
                # If it's a trading day, use it
                if target_date in business_days_set:
                    sample_days.append(target_date)
                else:
                    # Find nearest trading day (forward or backward)
                    # Try forward first
                    found = False
                    for offset in range(1, 8):  # Check up to 7 days ahead
                        candidate = target_date + pd.Timedelta(days=offset)
                        if candidate in business_days_set and candidate <= end_dt:
                            sample_days.append(candidate)
                            found = True
                            break
                    
                    # If not found forward, try backward
                    if not found:
                        for offset in range(1, 8):  # Check up to 7 days back
                            candidate = target_date - pd.Timedelta(days=offset)
                            if candidate in business_days_set and candidate >= start_dt:
                                sample_days.append(candidate)
                                found = True
                                break
            except ValueError:
                # Day doesn't exist in this month (e.g., Feb 30)
                # Find last day of month and use that if it's a trading day
                last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
                if last_day in business_days_set and last_day <= end_dt and last_day >= start_dt:
                    if last_day not in sample_days:
                        sample_days.append(last_day)
        
        # Move to next month
        current = (current + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)
    
    # Remove duplicates and sort
    sample_days = sorted(list(set(sample_days)))
    return sample_days

def utc_window(date: pd.Timestamp, hhmm_utc: str, window_minutes: int) -> Tuple[str, str]:
    hh, mm = map(int, hhmm_utc.split(":"))
    center = pd.Timestamp(date.date()) + pd.Timedelta(hours=hh, minutes=mm)
    start = center - pd.Timedelta(minutes=window_minutes / 2)
    end   = center + pd.Timedelta(minutes=window_minutes / 2)
    # Databento accepts RFC3339; use Z
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")

def load_env_key() -> str:
    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing DATABENTO_API_KEY env var. Do: export DATABENTO_API_KEY='...'")
    return key

def safe_to_df(data) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if hasattr(data, "to_df"):
        return data.to_df()
    return pd.DataFrame(data)

def pick_symbol_col(df: pd.DataFrame) -> str:
    for c in ["symbol", "tsymbol", "raw_symbol"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find symbol column. Columns: {list(df.columns)}")

def pick_bid_ask_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    # common patterns across Databento schemas
    candidates_bid = ["bid_px_00", "bid_px", "bid", "best_bid_px", "bid_price", "bid_px1"]
    candidates_ask = ["ask_px_00", "ask_px", "ask", "best_ask_px", "ask_price", "ask_px1"]
    bid = next((c for c in candidates_bid if c in df.columns), None)
    ask = next((c for c in candidates_ask if c in df.columns), None)
    return bid, ask

def pick_trade_px_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["price", "trade_px", "last_px", "px", "close", "c"]:
        if c in df.columns:
            return c
    return None

def occ_parse_expiration(sym: str) -> Optional[pd.Timestamp]:
    """
    Parses OCC-style symbols like:
      'SPY   230609C00450000'
    Finds first YYMMDD block and returns as date.
    """
    s = str(sym)
    # search first 6-digit run that looks like a date
    for i in range(len(s) - 5):
        chunk = s[i:i+6]
        if chunk.isdigit():
            yy = int(chunk[0:2]); mm = int(chunk[2:4]); dd = int(chunk[4:6])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                year = 2000 + yy
                try:
                    return pd.Timestamp(year, mm, dd).normalize()
                except Exception:
                    pass
    return None

def occ_parse_type_and_strike(sym: str) -> Tuple[Optional[str], Optional[float]]:
    """
    OCC: underlying padded to 6 + YYMMDD + C/P + strike*1000 with 8 digits
    Example: 'SPY   230609C00450000' -> call, 450.0
    """
    s = str(sym)
    # Find the YYMMDD block, then next char is type, next 8 digits is strike
    for i in range(len(s) - 6):
        chunk = s[i:i+6]
        if chunk.isdigit():
            j = i + 6
            if j < len(s):
                cp = s[j].upper()
                if cp not in ("C", "P"):
                    continue
                strike_part = s[j+1:j+9]
                if len(strike_part) == 8 and strike_part.isdigit():
                    strike = int(strike_part) / 1000.0
                    opt_type = "call" if cp == "C" else "put"
                    return opt_type, strike
    return None, None

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def retry_call(fn, *args, **kwargs):
    last = None
    for k in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            sleep_s = BACKOFF_S * (2 ** k)
            time.sleep(sleep_s)
    raise last


# =========================
# Core downloader
# =========================

@dataclass
class ExpirySelection:
    expiration: str
    dte: int
    symbols: List[str]

class SpyOptionsDownloader:
    def __init__(self, client: dbn.Historical):
        self.client = client
        ensure_dir(OUT_DIR)
        ensure_dir(os.path.join(OUT_DIR, "definitions"))
        ensure_dir(os.path.join(OUT_DIR, "snapshots"))
        self.manifest_path = os.path.join(OUT_DIR, "manifest.csv")
        self._manifest_rows = []

    def _save_manifest_row(self, row: Dict):
        self._manifest_rows.append(row)
        pd.DataFrame(self._manifest_rows).to_csv(self.manifest_path, index=False)

    def get_definitions_for_day(self, date: pd.Timestamp) -> pd.DataFrame:
        date_str = date.strftime("%Y-%m-%d")
        out_path = os.path.join(OUT_DIR, "definitions", f"{date_str}_def.csv.gz")
        if os.path.exists(out_path):
            return pd.read_csv(out_path, compression="gzip")

        # Pull definitions in a small window; definitions are not time-sensitive intraday.
        start = f"{date_str}T00:00:00Z"
        end   = f"{date_str}T23:59:59Z"

        data = retry_call(
            self.client.timeseries.get_range,
            dataset=DATASET,
            symbols=[PARENT_SYMBOL],
            schema=SCHEMA_DEFINITION,
            start=start,
            end=end,
            stype_in="parent",
        )
        df = safe_to_df(data)
        if df.empty:
            raise RuntimeError(f"No definition data for {PARENT_SYMBOL} on {date_str}.")
        if len(df) > MAX_DEFINITION_ROWS:
            raise RuntimeError(f"Definition payload too large ({len(df)} rows) on {date_str}; aborting for safety.")

        sym_col = pick_symbol_col(df)
        df["_sym"] = df[sym_col].astype(str)

        # Parse expiration / strike / type from symbols
        df["_exp"] = df["_sym"].apply(occ_parse_expiration)
        parsed = df["_sym"].apply(occ_parse_type_and_strike)
        df["_type"] = [p[0] for p in parsed]
        df["_strike"] = [p[1] for p in parsed]

        # keep only rows we can parse
        df = df.dropna(subset=["_exp", "_type", "_strike"]).copy()
        df["_exp"] = pd.to_datetime(df["_exp"]).dt.strftime("%Y-%m-%d")

        df.to_csv(out_path, index=False, compression="gzip")
        return df

    def select_expiration_near_dte(self, date: pd.Timestamp, spot: float) -> ExpirySelection:
        defs = self.get_definitions_for_day(date)
        date0 = date.normalize()

        # expirations available that day
        exps = defs["_exp"].dropna().unique().tolist()
        if not exps:
            raise RuntimeError(f"No expirations parsed from definitions on {date.strftime('%Y-%m-%d')}.")

        lo, hi = DTE_WINDOW
        best_exp = None
        best_dist = 10**9
        best_dte = None

        for exp_str in exps:
            exp_dt = pd.to_datetime(exp_str).normalize()
            dte = (exp_dt - date0).days
            if lo <= dte <= hi:
                dist = abs(dte - TARGET_DTE)
                if dist < best_dist:
                    best_dist = dist
                    best_exp = exp_str
                    best_dte = dte

        if best_exp is None:
            sample = sorted(exps)[:15]
            raise RuntimeError(
                f"No expiration within DTE window {DTE_WINDOW} on {date.strftime('%Y-%m-%d')} "
                f"(target {TARGET_DTE}). Sample expirations: {sample}"
            )

        # Filter symbols for that expiry AND within strike band
        loK = STRIKE_BAND[0] * spot
        hiK = STRIKE_BAND[1] * spot

        sub = defs[(defs["_exp"] == best_exp) & (defs["_strike"] >= loK) & (defs["_strike"] <= hiK)].copy()
        if sub.empty:
            raise RuntimeError(
                f"No symbols found after strike-band filter on {date.strftime('%Y-%m-%d')} for exp={best_exp} "
                f"spot={spot:.2f} band={STRIKE_BAND}"
            )

        symbols = sub["_sym"].astype(str).unique().tolist()
        if len(symbols) > MAX_SYMBOLS_FOR_EXPIRY:
            raise RuntimeError(f"Too many symbols ({len(symbols)}) for exp {best_exp} on {date.strftime('%Y-%m-%d')}; aborting.")

        return ExpirySelection(expiration=best_exp, dte=int(best_dte), symbols=symbols)

    def _spot_from_yf(self, date: pd.Timestamp) -> float:
        # Avoid extra Databento calls; for strike-band selection, yf spot is good enough.
        import yfinance as yf
        px = yf.download("SPY", start=(date - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                         end=(date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)["Close"].dropna()
        if px.empty:
            raise RuntimeError(f"No SPY spot from yfinance for {date.strftime('%Y-%m-%d')}.")
        return float(px.iloc[-1])

    def _download_snapshot_batch(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        data = retry_call(
            self.client.timeseries.get_range,
            dataset=DATASET,
            symbols=symbols,
            schema=SCHEMA_SNAPSHOT,
            start=start,
            end=end,
            stype_in="raw_symbol",
        )
        df = safe_to_df(data)
        if df.empty:
            return df
        if len(df) > MAX_ROWS_PER_BATCH:
            raise RuntimeError(f"Batch returned {len(df)} rows > MAX_ROWS_PER_BATCH={MAX_ROWS_PER_BATCH}. Aborting for safety.")
        return df

    def download_day(self, date: pd.Timestamp) -> None:
        date_str = date.strftime("%Y-%m-%d")

        # spot used for strike filtering; avoid expensive options-wide pulls
        spot = self._spot_from_yf(date)

        sel = self.select_expiration_near_dte(date, spot=spot)
        exp = sel.expiration
        dte = sel.dte
        symbols = sel.symbols

        # Download single snapshot for this day
        out_path = os.path.join(OUT_DIR, "snapshots", f"{date_str}_exp{exp}.csv.gz")
        if os.path.exists(out_path):
            return

        start, end = utc_window(date, SNAPSHOT_TIME_UTC, WINDOW_MINUTES)

        # batch the symbol list
        parts = [symbols[i:i+SYMBOL_BATCH_SIZE] for i in range(0, len(symbols), SYMBOL_BATCH_SIZE)]
        frames = []
        for bi, chunk in enumerate(parts):
            dfb = self._download_snapshot_batch(chunk, start=start, end=end)
            if not dfb.empty:
                frames.append(dfb)

        if not frames:
            # do not crash the whole run; log and continue
            self._save_manifest_row({
                "date": date_str, "expiration": exp, "dte": dte, "snapshot_utc": SNAPSHOT_TIME_UTC,
                "status": "EMPTY", "rows": 0, "file": ""
            })
            return

        raw = pd.concat(frames, axis=0, ignore_index=False)

        # Normalize columns to something usable
        sym_col = pick_symbol_col(raw)
        raw["_sym"] = raw[sym_col].astype(str)

        # parse strike/type (no re-querying needed)
        parsed = raw["_sym"].apply(occ_parse_type_and_strike)
        raw["opt_type"] = [p[0] for p in parsed]
        raw["strike"] = [p[1] for p in parsed]

        # expiration sanity (should match selected expiry; keep only matching)
        raw["_exp"] = raw["_sym"].apply(occ_parse_expiration)
        raw["_exp"] = pd.to_datetime(raw["_exp"]).dt.strftime("%Y-%m-%d")
        raw = raw[raw["_exp"] == exp].copy()

        if raw.empty:
            self._save_manifest_row({
                "date": date_str, "expiration": exp, "dte": dte, "snapshot_utc": SNAPSHOT_TIME_UTC,
                "status": "NO_MATCHING_EXPIRY", "rows": 0, "file": ""
            })
            return

        bid_col, ask_col = pick_bid_ask_cols(raw)
        trade_col = pick_trade_px_col(raw)

        # cbbo-1m should have bid/ask; but keep a fallback to trade-like price if needed
        if bid_col and ask_col:
            raw["bid"] = pd.to_numeric(raw[bid_col], errors="coerce")
            raw["ask"] = pd.to_numeric(raw[ask_col], errors="coerce")
            raw["mid"] = 0.5 * (raw["bid"] + raw["ask"])
        elif trade_col:
            raw["bid"] = np.nan
            raw["ask"] = np.nan
            raw["mid"] = pd.to_numeric(raw[trade_col], errors="coerce")
        else:
            # cannot price
            raw["bid"] = np.nan
            raw["ask"] = np.nan
            raw["mid"] = np.nan

        # Keep last observation per contract in the window
        # timestamp column varies; prefer ts_event if present, else index
        if "ts_event" in raw.columns:
            raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True, errors="coerce")
            raw = raw.sort_values("ts_event")
            dedup = raw.groupby("_sym", as_index=False).tail(1)
            ts = "ts_event"
        else:
            dedup = raw.groupby("_sym", as_index=False).tail(1)
            ts = None

        # Final clean frame
        out = pd.DataFrame({
            "date": date_str,
            "snapshot_utc": SNAPSHOT_TIME_UTC,
            "expiration": exp,
            "dte": dte,
            "underlying_spot": spot,
            "symbol": dedup["_sym"].values,
            "opt_type": dedup["opt_type"].values,
            "strike": dedup["strike"].values,
            "bid": dedup["bid"].values,
            "ask": dedup["ask"].values,
            "mid": dedup["mid"].values,
        })

        if ts and ts in dedup.columns:
            out["ts_event_utc"] = dedup[ts].astype(str).values
        else:
            out["ts_event_utc"] = ""

        # Drop unusable rows
        out = out.dropna(subset=["opt_type", "strike", "mid"])
        out = out[np.isfinite(out["mid"]) & (out["mid"] > 0)]

        out.to_csv(out_path, index=False, compression="gzip")

        self._save_manifest_row({
            "date": date_str, "expiration": exp, "dte": dte, "snapshot_utc": SNAPSHOT_TIME_UTC,
            "status": "OK", "rows": len(out), "file": out_path
        })

    def run(self):
        # Get 4 sample days per month instead of all trading days
        days = get_sample_days_per_month(START, END)
        if MAX_DAYS_DEBUG is not None:
            days = days[:MAX_DAYS_DEBUG]

        print(f"Processing {len(days)} sample days (4 per month) from {START} to {END}")

        for d in days:
            try:
                self.download_day(d)
                print(f"[OK] {d.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"[FAIL] {d.strftime('%Y-%m-%d')} :: {e}", file=sys.stderr)
                # write failure to manifest and continue
                self._save_manifest_row({
                    "date": d.strftime("%Y-%m-%d"),
                    "expiration": "",
                    "dte": "",
                    "snapshot_utc": "",
                    "status": f"FAIL: {type(e).__name__}",
                    "rows": 0,
                    "file": ""
                })


def main():
    key = load_env_key()
    client = dbn.Historical(key)

    dl = SpyOptionsDownloader(client)
    dl.run()
    print(f"Done. Data in: {OUT_DIR}")
    print(f"Manifest: {dl.manifest_path}")


if __name__ == "__main__":
    main()
