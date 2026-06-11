#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kalman filter for time-varying alpha/beta estimation in a pairs strategy.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    process_noise: float = 2e-4       # Q scale for Kalman state noise
    measurement_noise: float = 5e-4   # R for measurement noise
    vol_target_annual: float = 0.15   # 15% annualised vol target
    gate_on: float = 0.6              # ML prob threshold to turn hedge ON
    gate_off: float = 0.4             # ML prob threshold to turn hedge OFF (hysteresis)
    horizon_days: int = 5             # label horizon for ML training
    cost_bps_roundtrip: float = 2.0   # transaction cost per hedge toggle (bps)
    min_lookback_days: int = 252      # warmup before trading starts


@dataclass
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    resid: pd.Series
    resid_var: pd.Series


def kalman_time_varying_beta(
    rA: pd.Series,
    rB: pd.Series,
    q_scale: float,
    r_meas: float,
) -> KalmanResult:
    """
    Estimate time-varying alpha and beta via Kalman filter.

    State : x_t = [alpha_t, beta_t]^T
    Obs   : y_t = rA_t = [1, rB_t] @ x_t + eps_t,   eps_t ~ N(0, R)
    Trans : x_t = x_{t-1} + w_t,                     w_t  ~ N(0, Q), Q = q_scale * I
    """
    idx = rA.index.intersection(rB.index)
    y = rA.loc[idx].values
    xB = rB.loc[idx].values
    n = len(y)

    x = np.zeros(2)          # [alpha, beta]
    P = np.eye(2) * 1e-2
    Q = np.eye(2) * q_scale
    R = r_meas

    alpha = np.zeros(n)
    beta = np.zeros(n)
    resid = np.zeros(n)
    resid_var = np.zeros(n)

    for t in range(n):
        x_pred = x
        P_pred = P + Q

        H = np.array([1.0, xB[t]])
        S = H @ P_pred @ H.T + R
        K = (P_pred @ H.T) / (S + 1e-12)

        innov = y[t] - H @ x_pred
        x = x_pred + K * innov
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alpha[t] = x[0]
        beta[t] = x[1]
        resid[t] = innov
        resid_var[t] = S

    sidx = pd.Index(idx)
    return KalmanResult(
        alpha=pd.Series(alpha, index=sidx),
        beta=pd.Series(beta, index=sidx),
        resid=pd.Series(resid, index=sidx),
        resid_var=pd.Series(resid_var, index=sidx),
    )
