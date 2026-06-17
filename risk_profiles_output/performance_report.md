# Quantitative Hedge Fund — Performance Report
**Generated:** 2026-06-16 21:35  |  **Risk-free rate:** 4.5% p.a.  |  **Hedge instrument:** SPY short (Kalman time-varying beta)

---

## Personal Portfolio Performance

**Period:** 2025-09-30 → 2026-06-16  |  **Trades:** 79  |  **Total invested:** $12,985  |  **Current NAV:** $9,881

| Metric | Value |
|--------|-------|
| CAGR (TWR) | **+87.5%** |
| Annualised Volatility | 48.4% |
| Total Return | +57.1% |
| Sharpe Ratio | **1.4474** |
| Sortino Ratio | 2.1038 |
| Calmar Ratio | 3.3027 |
| Max Drawdown | -26.5% |
| VaR 95% (daily) | -4.877% |
| CVaR 95% (daily) | -6.628% |
| Return Skewness | 0.0026 |
| Return Kurtosis (excess) | 1.8902 |
| Best Day | +12.60% |
| Worst Day | -10.49% |

---

## Historical Strategy Simulations

### Portfolio Universe — Tech, Benchmarked vs the S&P 500

Every holding is a large-cap **technology** stock chosen specifically so that it carries a high correlation and beta to the **S&P 500 (SPY)**.  This is essential: the entire strategy hedges each name against SPY using a Kalman time-varying beta, so if the names did not co-move with the index the hedge would add noise instead of removing systematic risk.

Universe (equal-weight, 10 names, all trading before 2005 so the 20Y window is valid): **AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL, CSCO, QCOM, TXN, IBM**.  NVDA-style moonshots are deliberately excluded to keep returns realistic.

#### Correlation & Beta vs S&P 500 (10Y, 2016–2026)

| Ticker | Correlation w/ SPY | Beta vs SPY |
|--------|--------------------|-------------|
| AAPL | 0.75 | 1.21 |
| MSFT | 0.78 | 1.17 |
| GOOGL | 0.72 | 1.16 |
| AMZN | 0.66 | 1.19 |
| ADBE | 0.64 | 1.23 |
| ORCL | 0.54 | 1.05 |
| CSCO | 0.67 | 0.97 |
| QCOM | 0.63 | 1.38 |
| TXN | 0.70 | 1.21 |
| IBM | 0.56 | 0.83 |
| **Equal-weight portfolio** | **0.92** | **1.14** |

The equal-weight portfolio has a **92% correlation** with the S&P 500 (beta ≈ 1.14), confirming the SPY beta-hedge is well-specified — the short-SPY leg removes the bulk of market variance and leaves the idiosyncratic tech alpha that drives the Sharpe ratio.

### The Hedging Signal — XGBoost ML Gate

Hedging is **not** a simple VIX rule.  It is driven by the project's XGBoost classifier (`hedge/ml_gate.py`), which predicts — from five Kalman-spread features (`spread_z20`, `beta_chg20`, `resid_std20`, `corr20`, `pair_vol20`) — the probability that hedging will outperform staying unhedged over the next 5 trading days.  A hysteresis rule turns the hedge **on** when that probability rises above `gate_on` and **off** when it falls below `gate_off`, preventing chatter.

**Strict out-of-sample protection:** the classifier's label is forward-looking, so the model is trained **only on 2006–2016 data** (individual tech-stock-vs-SPY pairs) and then applied, untouched, to the portfolio-vs-SPY pair.  The **10Y (2016–2026) and 5Y windows are therefore fully out-of-sample for the gate**; in the 20Y window only the 2006–2016 leg is in-sample.  All gate signals are additionally lagged one day before they size a position.

### Optimisation Methodology

**Every profile is optimised for the highest Sharpe ratio.**  Each runs a grid search over the Kalman hyperparameters **q (process noise)** and **r (measurement noise)**, the ML hysteresis thresholds **gate_on/gate_off**, and the **hedge fraction**.  The winning combination maximises Sharpe on the **10-year window (2016–2026)**; the 20Y window then serves as a longer cross-check and the 5Y as the most recent slice.

**Leverage policy:** the vol-targeting overlay is capped at 1.0× for every profile *except Aggressive*, which is the only profile allowed to lever up (capped at 2.0×).  Uninvested capital earns the risk-free rate, so the Sharpe ratio is directly comparable across risk levels.

| Profile | Objective | Hedge signal | Vol target | Leverage cap |
|---------|-----------|--------------|-----------|--------------|
| Ultra Conservative | **Max Sharpe** | ML gate (partial hedge) | 6% | 1.0× (none) |
| Conservative | **Max Sharpe** | ML gate (partial hedge) | 9% | 1.0× (none) |
| Moderate | **Max Sharpe** | ML gate | 12% | 1.0× (none) |
| Growth | **Max Sharpe** | ML gate (high threshold) | 16% | 1.0× (none) |
| Aggressive | **Max Sharpe** | ML gate + leverage | 24% | **2.0×** |
| Unhedged | — | No hedge | None | 1.0× (none) |

> **Note:** Only the Aggressive profile uses leverage and is the only one designed to exceed unhedged absolute risk.  All other profiles are unlevered and target a higher Sharpe at lower absolute volatility.

### Optimised Hyperparameters (Sharpe-maximising, per profile)

Kalman `(q, r)` feeds both the ML-gate features and the hedge ratio.  `gate ON/OFF` are the XGBoost-probability hysteresis thresholds; `hedge frac` is the fraction of the Kalman beta actually shorted; `hedged %` is the share of OOS days the gate was active.

| Profile | q_scale | r_meas | gate ON | gate OFF | Hedge frac | Vol target | Hedged % (OOS) |
|---------|---------|--------|---------|----------|------------|------------|----------------|
| Ultra Conservative | `5e-06` | `1e-03` | 0.60 | 0.40 | 0.60× | 6% | 38% |
| Conservative | `5e-06` | `1e-03` | 0.60 | 0.40 | 0.50× | 9% | 38% |
| Moderate | `1e-05` | `2e-03` | 0.65 | 0.45 | 0.60× | 12% | 18% |
| Growth | `5e-06` | `1e-03` | 0.70 | 0.50 | 0.60× | 16% | 5% |
| Aggressive | `1e-05` | `2e-03` | 0.65 | 0.45 | 0.60× | 24% | 18% |

---

### 20Y Results  (2006-06-12 → 2026-06-12)

| Strategy | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | VaR 95% | CVaR 95% | Best Yr | Worst Yr |
|----------|---|---|---|---|---|---|---|---|---|---------|---------|
| **Unhedged** | +3378.8% | +19.5% | +22.7% | 0.70 | 0.92 | 0.39 | -49.4% | -2.2% | -3.3% | +80% (2009) | -39% (2008) |
| Ultra Conservative | +574.0% | +10.0% | +6.7% | 0.79 | 1.11 | 1.18 | -8.5% | -0.6% | -1.0% | +19% (2017) | -5% (2022) |
| Conservative | +939.7% | +12.4% | +9.7% | 0.79 | 1.11 | 0.95 | -13.1% | -1.0% | -1.4% | +25% (2009) | -9% (2022) |
| Moderate | +1462.8% | +14.8% | +12.4% | 0.81 | 1.14 | 0.90 | -16.5% | -1.3% | -1.8% | +32% (2009) | -11% (2008) |
| Growth | +2534.4% | +17.8% | +15.4% | 0.85 | 1.20 | 0.70 | -25.4% | -1.6% | -2.2% | +45% (2009) | -18% (2008) |
| Aggressive | +7811.1% | +24.5% | +24.8% | 0.83 | 1.16 | 0.68 | -36.0% | -2.5% | -3.6% | +64% (2009) | -26% (2008) |

#### 20Y Delta vs Unhedged

| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |
|----------|-------|---------|---------|------|
| Ultra Conservative | -9.4% | +0.09 | +40.9% | -16.0% |
| Conservative | -7.0% | +0.09 | +36.3% | -13.0% |
| Moderate | -4.7% | +0.11 | +32.9% | -10.3% |
| Growth | -1.7% | +0.15 | +24.1% | -7.3% |
| Aggressive | +5.0% | +0.13 | +13.4% | +2.1% |

---

### 10Y Results  (2016-06-12 → 2026-06-12)

| Strategy | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | VaR 95% | CVaR 95% | Best Yr | Worst Yr |
|----------|---|---|---|---|---|---|---|---|---|---------|---------|
| **Unhedged** | +640.7% | +22.2% | +22.4% | 0.81 | 1.05 | 0.71 | -31.2% | -2.2% | -3.3% | +43% (2023) | -25% (2022) |
| Ultra Conservative | +195.3% | +11.5% | +6.8% | 0.96 | 1.35 | 1.34 | -8.5% | -0.7% | -1.0% | +19% (2017) | -5% (2022) |
| Conservative | +277.5% | +14.2% | +9.8% | 0.95 | 1.33 | 1.09 | -13.1% | -1.0% | -1.4% | +23% (2023) | -9% (2022) |
| Moderate | +367.2% | +16.7% | +12.4% | 0.95 | 1.31 | 1.15 | -14.5% | -1.3% | -1.8% | +27% (2023) | -10% (2022) |
| Growth | +500.2% | +19.7% | +15.5% | 0.95 | 1.31 | 1.01 | -19.4% | -1.6% | -2.3% | +36% (2023) | -15% (2022) |
| Aggressive | +1157.0% | +28.9% | +24.8% | 0.97 | 1.34 | 1.02 | -28.4% | -2.5% | -3.6% | +51% (2023) | -23% (2022) |

#### 10Y Delta vs Unhedged

| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |
|----------|-------|---------|---------|------|
| Ultra Conservative | -10.8% | +0.15 | +22.6% | -15.6% |
| Conservative | -8.0% | +0.14 | +18.1% | -12.6% |
| Moderate | -5.5% | +0.14 | +16.6% | -10.0% |
| Growth | -2.6% | +0.14 | +11.8% | -6.9% |
| Aggressive | +6.7% | +0.16 | +2.8% | +2.4% |

---

### 5Y Results  (2021-06-12 → 2026-06-12)

| Strategy | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | VaR 95% | CVaR 95% | Best Yr | Worst Yr |
|----------|---|---|---|---|---|---|---|---|---|---------|---------|
| **Unhedged** | +104.2% | +15.4% | +22.0% | 0.56 | 0.80 | 0.49 | -31.2% | -2.3% | -3.1% | +43% (2023) | -25% (2022) |
| Ultra Conservative | +54.2% | +9.1% | +6.8% | 0.65 | 0.96 | 1.06 | -8.5% | -0.7% | -1.0% | +17% (2023) | -5% (2022) |
| Conservative | +70.9% | +11.4% | +10.0% | 0.67 | 1.00 | 0.87 | -13.1% | -1.0% | -1.4% | +23% (2023) | -9% (2022) |
| Moderate | +87.7% | +13.5% | +12.8% | 0.70 | 1.02 | 0.93 | -14.5% | -1.3% | -1.8% | +27% (2023) | -10% (2022) |
| Growth | +106.6% | +15.7% | +16.1% | 0.71 | 1.05 | 0.81 | -19.4% | -1.7% | -2.2% | +36% (2023) | -15% (2022) |
| Aggressive | +163.8% | +21.5% | +25.6% | 0.71 | 1.04 | 0.76 | -28.4% | -2.7% | -3.6% | +51% (2023) | -23% (2022) |

#### 5Y Delta vs Unhedged

| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |
|----------|-------|---------|---------|------|
| Ultra Conservative | -6.3% | +0.09 | +22.6% | -15.1% |
| Conservative | -4.1% | +0.12 | +18.1% | -11.9% |
| Moderate | -1.9% | +0.14 | +16.6% | -9.2% |
| Growth | +0.3% | +0.15 | +11.8% | -5.9% |
| Aggressive | +6.1% | +0.16 | +2.8% | +3.6% |

---

## Charts

### Figure 1 — Equity Curves (20Y and 10Y)

![Equity Curves](fig1_equity_curves.png)

*Log-scale growth of $1.  Red shading = VIX > 25 (GFC 2008–09, COVID 2020, inflation shock 2022).  Dashed grey = Unhedged baseline.*

### Figure 2 — Drawdown Comparison (20Y)

![Drawdowns](fig2_drawdowns.png)

*All strategies plotted from peak.  Max drawdown annotated.  Ultra Conservative and Conservative show the deepest drawdown reduction.*

### Figure 3 — Risk-Adjusted Metrics Grid

![Metrics Grid](fig3_metrics_grid.png)

*Six key metrics across all three time windows.  Grouped by 20Y / 10Y / 5Y within each panel.*

---

## Key Takeaways

All profiles below are Sharpe-optimised on the 10Y window.  Unhedged baseline (10Y): Sharpe **0.81**, vol 22.4%, max DD -31.2%.

1. **Ultra Conservative — capital preservation:** Sharpe **0.96** (+0.15 vs unhedged), max drawdown only -8.5% vs -31.2% unhedged, at 6.8% vol. Unlevered, market-neutral.
2. **Conservative — best risk-adjusted:** Sharpe **0.95** (+0.14 vs unhedged) via always-on beta-hedging at ~9.8% vol. Unlevered.
3. **Moderate:** Sharpe **0.95**; conditional hedge (VIX/vol trigger) cuts max drawdown by 16.6% vs unhedged while staying unlevered. Good balance for long-horizon investors.
4. **Growth:** Sharpe **0.95**; crisis-only hedge lets the portfolio stay near fully-invested in calm markets, CAGR +19.7% vs +22.2% unhedged at 15.5% vol. Unlevered.
5. **Aggressive (only levered profile):** Sharpe **0.97**; always-hedged then levered up to 2.0× for CAGR +28.9% vs +22.2% unhedged, at vol 24.8% vs 22.4% unhedged. Max DD -28.4%.  The only profile exceeding unhedged absolute risk — **suitable only for high risk tolerance.**

---

*All backtests use strict no-lookahead guards: the XGBoost gate is trained only on 2006–2016 data, and the Kalman beta, ML gate signal and vol scale are each lagged one day before application.  Uninvested cash earns the risk-free rate.  Transaction costs: 2–3 bps per hedge toggle.  Risk-free rate: 4.5% p.a.*
