#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML-based hedging gate: feature engineering, XGBoost classifier, and backtesting.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from hedge.kalman import KalmanResult, StrategyParams

FEATURE_COLS = ["spread_z20", "beta_chg20", "resid_std20", "corr20", "pair_vol20"]


def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() * freq / (returns.std() * math.sqrt(freq) + 1e-12))


def calmar_ratio(returns: pd.Series, freq: int = 252) -> float:
    cum = (1 + returns).cumprod()
    dd = (cum / cum.cummax() - 1.0).min()
    ann = returns.mean() * freq
    if dd == 0:
        return float("inf") if ann > 0 else 0.0
    return float(ann / abs(dd))


def make_features_labels(
    rA: pd.Series,
    rB: pd.Series,
    kal: KalmanResult,
    horizon: int,
    freq: int = 252,
    tickerA: str = "",
    tickerB: str = "",
) -> pd.DataFrame:
    """
    Build per-day feature rows with a binary label indicating whether hedging
    (spread) outperforms unhedged (rA) over the next `horizon` days.
    """
    df = pd.DataFrame(index=kal.beta.index)
    df["rA"] = rA.reindex(df.index)
    df["rB"] = rB.reindex(df.index)
    df["beta"] = kal.beta
    df["alpha"] = kal.alpha
    df["resid"] = kal.resid
    df["resid_var"] = kal.resid_var

    df["spread"] = df["rA"] - df["beta"] * df["rB"]
    df["spread_z20"] = (df["spread"] - df["spread"].rolling(20).mean()) / (
        df["spread"].rolling(20).std() + 1e-8
    )
    df["beta_chg20"] = df["beta"].diff().rolling(20).std()
    df["resid_std20"] = df["resid"].rolling(20).std()
    df["corr20"] = df["rA"].rolling(20).corr(df["rB"])
    df["pair_vol20"] = (
        (0.5 * df["rA"] ** 2 + 0.5 * df["rB"] ** 2).rolling(20).mean().pow(0.5)
        * math.sqrt(freq)
    )

    hedged = df["spread"]
    unhedged = df["rA"]
    df["future_dPnL"] = (
        hedged.shift(-1).rolling(horizon).sum()
        - unhedged.shift(-1).rolling(horizon).sum()
    )
    df["label"] = (df["future_dPnL"] > 0.0).astype(int)

    df["A"] = tickerA
    df["B"] = tickerB
    return df


def train_classifier(train_df: pd.DataFrame,
                     feature_cols: List[str] = FEATURE_COLS) -> XGBClassifier:
    data = train_df.dropna(subset=["label"] + feature_cols)
    X = data[feature_cols].values
    y = data["label"].values

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.1,
        min_child_weight=8,
        random_state=42,
        tree_method="hist",
        enable_categorical=False,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    sharpe: float
    calmar: float
    auc: Optional[float]
    details: Dict


def backtest_pair_with_gate(
    rA: pd.Series,
    rB: pd.Series,
    kal: KalmanResult,
    model: XGBClassifier,
    params: StrategyParams,
    pair_name: str,
    feature_cols: List[str] = FEATURE_COLS,
) -> BacktestResult:
    """
    Backtest a single pair using the Kalman-estimated beta and ML gate.
    The gate uses hysteresis: turn ON at gate_on, turn OFF at gate_off.
    """
    idx = kal.beta.index
    df = pd.DataFrame(index=idx)
    df["rA"] = rA.reindex(idx)
    df["rB"] = rB.reindex(idx)
    df["beta"] = kal.beta
    df["spread"] = df["rA"] - df["beta"] * df["rB"]

    feat = pd.DataFrame(index=idx)
    feat["spread_z20"] = (
        df["spread"] - df["spread"].rolling(20).mean()
    ) / (df["spread"].rolling(20).std() + 1e-8)
    feat["beta_chg20"] = df["beta"].diff().rolling(20).std()
    feat["resid_std20"] = (df["rA"] - df["beta"] * df["rB"]).rolling(20).std()
    feat["corr20"] = df["rA"].rolling(20).corr(df["rB"])
    feat["pair_vol20"] = (
        (0.5 * df["rA"] ** 2 + 0.5 * df["rB"] ** 2)
        .rolling(20)
        .mean()
        .pow(0.5)
        * math.sqrt(252)
    )

    X = feat.values
    mask = ~np.any(np.isnan(X), axis=1)
    proba = np.full(len(df), np.nan)
    if np.any(mask):
        proba[mask] = predict_proba(model, X[mask])[:, 1]
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

    raw_ret = df["rA"] - df["hedge_on"] * df["beta"] * df["rB"]

    # Vol targeting
    vol_roll = raw_ret.rolling(20).std() * math.sqrt(252)
    scale = (params.vol_target_annual / (vol_roll + 1e-8)).clip(upper=5.0)
    strat_ret = raw_ret * scale.shift(1)

    # Transaction costs on state changes
    cost = df["hedge_on"].diff().abs().fillna(0) * (params.cost_bps_roundtrip / 10000.0)
    strat_ret = strat_ret - cost

    strat_ret = strat_ret[df.index[params.min_lookback_days]:]

    return BacktestResult(
        daily_returns=strat_ret,
        sharpe=sharpe_ratio(strat_ret),
        calmar=calmar_ratio(strat_ret),
        auc=None,
        details={
            "pair": pair_name,
            "n_days": len(strat_ret),
            "mean_daily": float(strat_ret.mean()),
            "std_daily": float(strat_ret.std()),
            "on_ratio": float(df["hedge_on"].mean()),
        },
    )
