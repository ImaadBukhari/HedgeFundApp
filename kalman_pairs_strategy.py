#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Kalman-ML Pairs Strategy (Starter)
# Author: ChatGPT
#
# Notes:
# - Run on macOS; optional Apple GPU (MPS) if PyTorch is installed and MPS is available.
# - Dependencies: numpy, pandas, yfinance, scikit-learn, matplotlib (optional), torch (optional).
# - This is a reference implementation; refine for production (risk, sizing, slippage models, borrow, fees).

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier

import numpy as np
import pandas as pd

# Optional GPU via PyTorch (for logistic regression). If not present, we fall back to scikit-learn.
USE_TORCH = False
try:
    import torch
    USE_TORCH = True
except Exception:
    USE_TORCH = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import yfinance as yf
except Exception as e:
    print("yfinance missing. Install with: pip install yfinance", file=sys.stderr)
    raise

# ---------- Configuration ----------

SEED = 42
np.random.seed(SEED)

# Universe: include enough names to form 40 pairs (we'll pick 80 tickers from here).
# Feel free to modify this list for your environment.
UNIVERSE = [
    # Mega/large cap tech + comm
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM","AMD","NFLX","ADBE",
    "INTC","CSCO","QCOM","IBM","SHOP","UBER","SNOW","PANW","NOW","MU","AMAT","TXN","MRVL","ASML",
    # Financials
    "JPM","BAC","WFC","C","GS","MS","BLK","V","MA","AXP","PYPL","SQ",
    # Industrials
    "CAT","HON","GE","LMT","NOC","BA","DE","MMM","UNP",
    # Staples & Discretionary
    "PG","KO","PEP","COST","WMT","TGT","HD","LOW","NKE","MCD","SBUX",
    # Healthcare
    "UNH","JNJ","PFE","LLY","MRK","ABT","TMO","BMY","GILD","AMGN",
    # Energy & Materials
    "XOM","CVX","COP","SLB","EOG","SHEL","BP","SCCO","FCX","NEM",
    # Utilities & REITs
    "NEE","DUK","SO","AEP","PLD","AMT",
    # Index proxies (for features)
    "SPY"
]

# Pairs: we will automatically generate 40 sector-ish pairs by simple heuristic:
# - Pick the highest-correlated pairs (absolute corr) excluding SPY, disallow duplicates.
NUM_PAIRS = 40

# Data range
START = "2015-01-01"
END   = None   # defaults to today

# Strategy parameters (default; will be tuned later)
@dataclass
class StrategyParams:
    process_noise: float = 2e-4     # Q scale for Kalman state noise
    measurement_noise: float = 5e-4 # R for measurement noise
    vol_target_annual: float = 0.15 # 15%
    gate_on: float = 0.6            # ML prob threshold to turn hedge ON
    gate_off: float = 0.4           # ML prob threshold to turn hedge OFF (hysteresis)
    horizon_days: int = 5           # label horizon
    cost_bps_roundtrip: float = 2.0 # transaction cost per hedge toggle (bps, simplistic)
    min_lookback_days: int = 252    # warmup before trading starts

PARAMS = StrategyParams()

# ---------- Utilities ----------

def sharpe_ratio(returns: pd.Series, freq=252):
    if returns.std() == 0:
        return 0.0
    mean = returns.mean() * freq
    vol  = returns.std() * math.sqrt(freq)
    return float(mean / (vol + 1e-12))

def calmar_ratio(returns: pd.Series, freq=252):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum/peak - 1.0).min()
    ann = returns.mean() * freq
    if dd == 0:
        return float('inf') if ann>0 else 0.0
    return float(ann / abs(dd))

# ---------- Data ----------

def download_data(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

# ---------- Pair selection ----------

def select_pairs_by_corr(returns: pd.DataFrame, num_pairs=NUM_PAIRS) -> List[Tuple[str,str]]:
    tickers = [t for t in returns.columns if t != "SPY"]
    R = returns[tickers].corr().abs()
    np.fill_diagonal(R.values, 0.0)
    pairs = set()
    flat = R.unstack().sort_values(ascending=False)
    for (a,b), _ in flat.items():
        if a==b:
            continue
        key = tuple(sorted((a,b)))
        if key in pairs:
            continue
        pairs.add(key)
        if len(pairs) >= num_pairs:
            break
    return list(pairs)

# ---------- Kalman filter for time-varying alpha/beta ----------

@dataclass
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    resid: pd.Series
    resid_var: pd.Series

def kalman_time_varying_beta(rA: pd.Series, rB: pd.Series,
                             q_scale: float, r_meas: float) -> KalmanResult:
    """
    State: x_t = [alpha_t, beta_t]^T
    Observation: y_t = rA_t = [1, rB_t] @ x_t + eps_t, eps_t ~ N(0,R)
    Transition: x_t = x_{t-1} + w_t, w_t ~ N(0, Q), with Q = q_scale * I
    """
    idx = rA.index.intersection(rB.index)
    y = rA.loc[idx].values
    xB = rB.loc[idx].values

    n = len(y)
    # State init
    x = np.zeros(2)  # [alpha, beta]
    P = np.eye(2) * 1e-2  # state covariance
    Q = np.eye(2) * q_scale
    R = r_meas

    alpha = np.zeros(n)
    beta  = np.zeros(n)
    resid = np.zeros(n)
    resid_var = np.zeros(n)

    for t in range(n):
        # Predict
        x_pred = x
        P_pred = P + Q

        H = np.array([1.0, xB[t]])  # observation matrix depends on rB_t
        S = H @ P_pred @ H.T + R     # innovation variance
        K = (P_pred @ H.T) / (S + 1e-12)  # Kalman gain (2,)

        y_pred = H @ x_pred
        innov  = y[t] - y_pred

        # Update
        x = x_pred + K * innov
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alpha[t] = x[0]
        beta[t]  = x[1]
        resid[t] = innov
        resid_var[t] = S

    sidx = pd.Index(idx)
    return KalmanResult(alpha=pd.Series(alpha, index=sidx),
                        beta=pd.Series(beta, index=sidx),
                        resid=pd.Series(resid, index=sidx),
                        resid_var=pd.Series(resid_var, index=sidx))

# ---------- Features & Labels ----------

def make_features_labels(
    rA: pd.Series, rB: pd.Series, kal: KalmanResult,
    horizon: int, freq: int = 252, tickerA: str = "", tickerB: str = ""
) -> pd.DataFrame:
    """
    Build per-day feature row with label indicating if hedging helps over next horizon.
    """
    df = pd.DataFrame(index=kal.beta.index)
    df["rA"] = rA.reindex(df.index)
    df["rB"] = rB.reindex(df.index)
    df["beta"] = kal.beta
    df["alpha"] = kal.alpha
    df["resid"] = kal.resid
    df["resid_var"] = kal.resid_var

    # Spread & correlation stats
    df["spread"] = df["rA"] - df["beta"] * df["rB"]
    df["spread_z20"] = (df["spread"] - df["spread"].rolling(20).mean()) / (df["spread"].rolling(20).std() + 1e-8)
    df["beta_chg20"] = df["beta"].diff().rolling(20).std()
    df["resid_std20"] = df["resid"].rolling(20).std()
    df["corr20"] = df["rA"].rolling(20).corr(df["rB"])

    # Market proxy: realized pair vol (proxy for regime)
    df["pair_vol20"] = (0.5*df["rA"]**2 + 0.5*df["rB"]**2).rolling(20).mean().pow(0.5) * math.sqrt(freq)

    # Labels: future cumulative PnL difference (hedged - unhedged)
    hedged = df["spread"]
    unhedged = df["rA"]
    df["future_dPnL"] = (hedged.shift(-1).rolling(horizon).sum() - unhedged.shift(-1).rolling(horizon).sum())

    # Binary label: 1 if hedging helps by >= 0 (before costs, to avoid circularity)
    df["label"] = (df["future_dPnL"] > 0.0).astype(int)

    # Add meta
    df["A"] = tickerA
    df["B"] = tickerB
    return df

# ---------- ML Training ----------

def train_classifier(train_df: pd.DataFrame, feature_cols: List[str]):
    data = train_df.dropna(subset=["label"] + feature_cols)
    X = data[feature_cols].values
    y = data["label"].values

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,     # L2 regularization
        reg_alpha=0.1,      # L1 regularization
        min_child_weight=8, # helps prevent overfit
        random_state=42,
        tree_method="hist",
        enable_categorical=False,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    else:
        return model["predict_proba"](X)


# ---------- Strategy Backtest with ML Gate ----------

@dataclass
class BacktestResult:
    daily_returns: pd.Series
    sharpe: float
    calmar: float
    auc: Optional[float]
    details: Dict

def backtest_pair_with_gate(
    rA: pd.Series, rB: pd.Series, kal: KalmanResult,
    model, feature_cols: List[str], params: StrategyParams, pair_name: str
) -> BacktestResult:
    idx = kal.beta.index
    df = pd.DataFrame(index=idx)
    df["rA"] = rA.reindex(idx)
    df["rB"] = rB.reindex(idx)
    df["beta"] = kal.beta
    df["spread"] = df["rA"] - df["beta"]*df["rB"]
    df["hedge_on"] = 0

    # Features for gating
    feat = pd.DataFrame(index=idx)
    feat["spread_z20"]  = (df["spread"] - df["spread"].rolling(20).mean()) / (df["spread"].rolling(20).std()+1e-8)
    feat["beta_chg20"]  = df["beta"].diff().rolling(20).std()
    feat["resid_std20"] = (df["rA"] - df["beta"]*df["rB"]).rolling(20).std()
    feat["corr20"]      = df["rA"].rolling(20).corr(df["rB"])
    feat["pair_vol20"]  = (0.5*df["rA"]**2 + 0.5*df["rB"]**2).rolling(20).mean().pow(0.5)*math.sqrt(252)

    # Predict probabilities
    X = feat.values
    mask = ~np.any(np.isnan(X), axis=1)
    proba = np.full(len(df), np.nan)
    if np.any(mask):
        proba_vals = predict_proba(model, X[mask])[:,1]
        proba[mask] = proba_vals
    df["p"] = proba

    # Hysteresis gating
    on = False
    hedge_signal = []
    for p in df["p"]:
        if np.isnan(p):
            hedge_signal.append(int(on))
            continue
        if on:
            if p <= params.gate_off:
                on = False
        else:
            if p >= params.gate_on:
                on = True
        hedge_signal.append(int(on))
    df["hedge_on"] = hedge_signal

    # Position sizing: when ON, use beta; when OFF, no hedge. Vol target the resulting returns.
    raw_ret = df["rA"] - df["hedge_on"] * df["beta"] * df["rB"]

    # Vol targeting to annual target
    vol_roll = raw_ret.rolling(20).std() * math.sqrt(252)
    scale = (params.vol_target_annual / (vol_roll + 1e-8)).clip(upper=5.0)  # cap leverage
    strat_ret = raw_ret * scale.shift(1)  # apply prev-day scale to avoid lookahead

    # Simple trade cost model: charge costs on hedge state changes
    state_change = (df["hedge_on"].diff().abs().fillna(0))
    cost = state_change * (params.cost_bps_roundtrip/10000.0)
    strat_ret = strat_ret - cost

    # Warm-up cut
    strat_ret = strat_ret[df.index[PARAMS.min_lookback_days]:]

    sr = sharpe_ratio(strat_ret)
    cr = calmar_ratio(strat_ret)

    details = {
        "pair": pair_name,
        "n_days": len(strat_ret),
        "mean_daily": float(strat_ret.mean()),
        "std_daily": float(strat_ret.std()),
        "on_ratio": float(df["hedge_on"].mean()),
    }
    return BacktestResult(daily_returns=strat_ret, sharpe=sr, calmar=cr, auc=None, details=details)

# ---------- Walk-forward Training/Test over Pairs ----------

def split_pairs(pairs: List[Tuple[str,str]], n_train=30, n_test=10):
    assert len(pairs) >= n_train + n_test
    train_pairs = pairs[:n_train]
    test_pairs  = pairs[n_train:n_train+n_test]
    return train_pairs, test_pairs

def aggregate_metrics(results: List[BacktestResult]) -> Dict[str, float]:
    if not results:
        return {"sharpe":0.0,"calmar":0.0}
    series = [r.daily_returns for r in results]
    panel = pd.concat(series, axis=1).fillna(0.0)
    eq = panel.mean(axis=1)  # equal weight across pairs
    return {
        "sharpe": sharpe_ratio(eq),
        "calmar": calmar_ratio(eq),
        "n_days": len(eq),
    }

# ---------- Hyperparameter Tuning (coarse) ----------

def coarse_tune(
    returns: pd.DataFrame, pairs: List[Tuple[str,str]], params: StrategyParams
) -> StrategyParams:
    # Coarse grids (trim for speed; expand offline)
    q_grid = [1e-5]
    r_grid = [2e-3]
    gate_grid = [(0.5,0.4)]
    vt_grid = [0.25]

    best = None
    best_score = -1e9

    # Small subsample of pairs for tuning speed
    tune_pairs = pairs[:10]

    for q in q_grid:
        for r in r_grid:
            for (on,off) in gate_grid:
                for vt in vt_grid:
                    P = StrategyParams(
                        process_noise=q, measurement_noise=r,
                        vol_target_annual=vt, gate_on=on, gate_off=off,
                        horizon_days=params.horizon_days,
                        cost_bps_roundtrip=params.cost_bps_roundtrip,
                        min_lookback_days=params.min_lookback_days
                    )
                    # Build training set for ML from tune_pairs
                    feat_frames = []
                    for (A,B) in tune_pairs:
                        rA = returns[A].dropna()
                        rB = returns[B].dropna()
                        kal = kalman_time_varying_beta(rA, rB, q_scale=P.process_noise, r_meas=P.measurement_noise)
                        df = make_features_labels(rA, rB, kal, horizon=P.horizon_days, tickerA=A, tickerB=B)
                        feat_frames.append(df)

                    all_df = pd.concat(feat_frames).dropna()
                    feature_cols = ["spread_z20","beta_chg20","resid_std20","corr20","pair_vol20"]
                    model = train_classifier(all_df, feature_cols)

                    # Backtest on tune pairs
                    results = []
                    for (A,B) in tune_pairs:
                        rA = returns[A].dropna()
                        rB = returns[B].dropna()
                        kal = kalman_time_varying_beta(rA, rB, q_scale=P.process_noise, r_meas=P.measurement_noise)
                        res = backtest_pair_with_gate(rA, rB, kal, model, feature_cols, P, f"{A}-{B}")
                        results.append(res)
                    agg = aggregate_metrics(results)
                    # --- Compute annualized return of equal-weighted portfolio ---
                    panel = pd.concat([r.daily_returns for r in results], axis=1).fillna(0.0)
                    eq = panel.mean(axis=1)
                    ann_return = eq.mean() * 252  # daily → annualized

                    # Print diagnostic line for each combination
                    print(f"[TUNE] Q={P.process_noise:.1e} R={P.measurement_noise:.1e} "
                        f"Q/R={P.process_noise/P.measurement_noise:.2f} "
                        f"Sharpe={agg['sharpe']:.3f} Return={ann_return:.2%} "
                        f"Calmar={agg['calmar']:.2f}")
                    normalized_return = min(ann_return, 0.15) / 0.10  # cap at 15%
                    score = 0.5 * agg["sharpe"] + 0.4 * normalized_return + 0.1 * agg["calmar"]




                    if score > best_score:
                        best_score = score
                        best = P
                        best_ann_return = ann_return
                        best_sharpe = agg["sharpe"]
    print(f"[TUNER] Best: Sharpe={best_sharpe:.3f}  Return={best_ann_return:.2%}  Params={best}")
    return best if best is not None else params

# ---------- Main ----------

def main():
    print("Downloading data...")
    prices = download_data(UNIVERSE, START, END)
    prices = prices.dropna(how="all", axis=0)
    prices = prices.dropna(how="any", axis=1)  # drop tickers with missing data
    rets = daily_returns(prices)

    # Select pairs by correlation (exclude SPY from pairs)
    pairs = select_pairs_by_corr(rets, num_pairs=NUM_PAIRS)
    print(f"Selected {len(pairs)} pairs. Example:", pairs[:5])

    # Split into 30 train pairs and 10 test pairs
    train_pairs, test_pairs = split_pairs(pairs, n_train=30, n_test=10)
    print("Train pairs:", len(train_pairs), " Test pairs:", len(test_pairs))

    # Optional coarse tuning on a subset
    tuned_params = coarse_tune(rets, train_pairs, PARAMS)
    print("Tuned params:", tuned_params)

    # --- Build ML training set using ALL train pairs ---
    feat_frames = []
    for (A,B) in train_pairs:
        rA = rets[A].dropna()
        rB = rets[B].dropna()
        kal = kalman_time_varying_beta(rA, rB, q_scale=tuned_params.process_noise, r_meas=tuned_params.measurement_noise)
        df = make_features_labels(rA, rB, kal, horizon=tuned_params.horizon_days, tickerA=A, tickerB=B)
        feat_frames.append(df)

    train_df = pd.concat(feat_frames).dropna()
    feature_cols = ["spread_z20","beta_chg20","resid_std20","corr20","pair_vol20"]
    model = train_classifier(train_df, feature_cols)

    # --- Backtest on train and test ---
    train_results = []
    for (A,B) in train_pairs:
        rA = rets[A].dropna()
        rB = rets[B].dropna()
        kal = kalman_time_varying_beta(rA, rB, q_scale=tuned_params.process_noise, r_meas=tuned_params.measurement_noise)
        res = backtest_pair_with_gate(rA, rB, kal, model, feature_cols, tuned_params, f"{A}-{B}")
        train_results.append(res)
    train_agg = aggregate_metrics(train_results)
    print(f"[TRAIN] Sharpe: {train_agg['sharpe']:.3f}  Calmar: {train_agg['calmar']:.3f}  Days: {train_agg['n_days']}")

    test_results = []
    for (A,B) in test_pairs:
        rA = rets[A].dropna()
        rB = rets[B].dropna()
        kal = kalman_time_varying_beta(rA, rB, q_scale=tuned_params.process_noise, r_meas=tuned_params.measurement_noise)
        res = backtest_pair_with_gate(rA, rB, kal, model, feature_cols, tuned_params, f"{A}-{B}")
        test_results.append(res)
    test_agg = aggregate_metrics(test_results)
    print(f"[TEST]  Sharpe: {test_agg['sharpe']:.3f}  Calmar: {test_agg['calmar']:.3f}  Days: {test_agg['n_days']}")

    # Detailed per-pair summary CSVs
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

    train_df_out = results_to_df(train_results)
    test_df_out  = results_to_df(test_results)

    train_df_out.to_csv(os.path.join(out_dir, "train_pair_results.csv"), index=False)
    test_df_out.to_csv(os.path.join(out_dir, "test_pair_results.csv"), index=False)

    # Equity curves (equal-weight across pairs) saved as CSV
    def equity_curve(results: List[BacktestResult]) -> pd.Series:
        panel = pd.concat([r.daily_returns for r in results], axis=1).fillna(0.0)
        eq = (1 + panel.mean(axis=1)).cumprod()
        return eq

    eq_train = equity_curve(train_results)
    eq_train.to_csv(os.path.join(out_dir, "equity_curve_train.csv"))
    eq_test  = equity_curve(test_results)
    eq_test.to_csv(os.path.join(out_dir, "equity_curve_test.csv"))

    # Save tuned params
    with open(os.path.join(out_dir, "tuned_params.txt"), "w") as f:
        f.write(str(tuned_params))

    # Save trained ML model
    model_path = os.path.join(out_dir, "kalman_ml_model.pkl")
    try:
        import joblib
        joblib.dump(model, model_path)
        print(f"Saved trained ML model to {model_path}")
    except Exception as e:
        print(f"ERROR saving model: {e}", file=sys.stderr)

    print(f"Outputs saved in ./{out_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
