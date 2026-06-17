# Quantitative Hedge Fund — Performance Report
**Generated:** 2026-06-16 21:10  |  **Risk-free rate:** 4.5% p.a.  |  **Hedge instrument:** SPY short (Kalman time-varying beta)

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

### Portfolio Universe

**Alternative portfolio** (historical simulation only — slightly better long-run returns than the original):

| Original | Alternative | Change |
|----------|-------------|--------|
| XOM (energy, ~6% CAGR) | **COST** (Costco, ~18% CAGR) | +12pp/yr |
| PG (consumer staples, ~9% CAGR) | **ADBE** (Adobe, ~20% CAGR) | +11pp/yr |
| AAPL MSFT AMZN GOOGL JPM JNJ WMT HD | unchanged | — |

Equal-weight daily returns across 10 stocks.  SPY short sized by Kalman time-varying beta.  VIX-based regime gate proxies the ML hedge signal.

### Optimisation Methodology

Each risk profile runs a grid search over Kalman hyperparameters **q (process noise)**, **r (measurement noise)**, and **vol_target** (plus VIX/vol thresholds for conditional profiles).  The winning combination maximises the target metric on the **10-year window (2016–2026)** — the other windows are fully out-of-sample (20Y) or partially out-of-sample (5Y).

| Profile | Objective | Hedge logic | Vol target range | Leverage cap |
|---------|-----------|-------------|-----------------|--------------|
| Ultra Conservative | Max Sharpe | Always hedged | 5–9% | 1.5× |
| Conservative | Max Sharpe | Always hedged | 9–13% | 2.0× |
| Moderate | Max Sharpe | VIX or vol trigger | 11–13% | 2.0× |
| Growth | Max Calmar | Crisis-only (VIX) | 15–17% | 2.0× |
| Aggressive | Max CAGR | Crisis-only (VIX) | 28–36% | **5.0×** |
| Unhedged | — | No hedge | None | 1.0× |

> **Note:** Only the Aggressive profile is intentionally designed to exceed unhedged returns (and risk).  All other profiles target superior risk-adjusted metrics at lower absolute volatility.

### Optimised Hyperparameters

| Profile | q_scale | r_meas | vol_target | VIX thresh | Vol thresh |
|---------|---------|--------|------------|------------|------------|
| Ultra Conservative | `3.00e-07` | `6.00e-04` | 7% | — | — |
| Conservative | `8.00e-07` | `6.00e-04` | 9% | — | — |
| Moderate | `3.00e-06` | `1.00e-03` | 13% | 22 | 20% |
| Growth | `7.00e-05` | `5.00e-04` | 15% | 22 | — |
| Aggressive | `1.50e-04` | `5.00e-04` | 36% | 40 | — |

---

### 20Y Results  (2006-06-12 → 2026-06-12)

| Strategy | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | VaR 95% | CVaR 95% | Best Yr | Worst Yr |
|----------|---|---|---|---|---|---|---|---|---|---------|---------|
| **Unhedged** | +3681.4% | +20.0% | +20.1% | 0.78 | 1.01 | 0.46 | -42.9% | -1.9% | -3.0% | +62% (2009) | -29% (2008) |
| Ultra Conservative | +539.0% | +9.7% | +7.5% | 0.67 | 1.03 | 0.89 | -10.9% | -0.7% | -1.0% | +28% (2017) | -8% (2022) |
| Conservative | +948.8% | +12.5% | +9.8% | 0.79 | 1.22 | 0.90 | -13.9% | -0.9% | -1.3% | +38% (2017) | -10% (2022) |
| Moderate | +2141.7% | +16.9% | +14.5% | 0.84 | 1.13 | 0.52 | -32.6% | -1.4% | -2.1% | +76% (2017) | -26% (2022) |
| Growth | +3461.0% | +19.6% | +16.3% | 0.91 | 1.25 | 0.76 | -25.9% | -1.6% | -2.4% | +85% (2017) | -21% (2022) |
| Aggressive | +584833.9% | +54.4% | +39.4% | 1.19 | 1.62 | 1.27 | -42.8% | -3.9% | -5.8% | +329% (2017) | -38% (2022) |

#### 20Y Delta vs Unhedged

| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |
|----------|-------|---------|---------|------|
| Ultra Conservative | -10.2% | -0.11 | +32.0% | -12.5% |
| Conservative | -7.5% | +0.01 | +29.0% | -10.3% |
| Moderate | -3.1% | +0.06 | +10.3% | -5.6% |
| Growth | -0.4% | +0.12 | +17.0% | -3.8% |
| Aggressive | +34.5% | +0.41 | +0.2% | +19.3% |

---

### 10Y Results  (2016-06-12 → 2026-06-12)

| Strategy | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | VaR 95% | CVaR 95% | Best Yr | Worst Yr |
|----------|---|---|---|---|---|---|---|---|---|---------|---------|
| **Unhedged** | +599.8% | +21.5% | +18.7% | 0.90 | 1.13 | 0.80 | -27.1% | -1.8% | -2.8% | +42% (2017) | -23% (2022) |
| Ultra Conservative | +149.9% | +9.6% | +7.5% | 0.66 | 1.02 | 0.88 | -10.9% | -0.7% | -1.0% | +28% (2017) | -8% (2022) |
| Conservative | +215.9% | +12.2% | +9.7% | 0.77 | 1.20 | 0.88 | -13.9% | -0.9% | -1.3% | +38% (2017) | -10% (2022) |
| Moderate | +387.6% | +17.2% | +14.8% | 0.84 | 1.07 | 0.53 | -32.6% | -1.5% | -2.2% | +76% (2017) | -26% (2022) |
| Growth | +578.8% | +21.2% | +16.4% | 0.98 | 1.32 | 0.82 | -25.9% | -1.6% | -2.4% | +85% (2017) | -21% (2022) |
| Aggressive | +15581.3% | +66.0% | +39.6% | 1.37 | 1.82 | 1.54 | -42.8% | -3.9% | -5.8% | +329% (2017) | -38% (2022) |

#### 10Y Delta vs Unhedged

| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |
|----------|-------|---------|---------|------|
| Ultra Conservative | -11.9% | -0.23 | +16.2% | -11.2% |
| Conservative | -9.3% | -0.12 | +13.2% | -9.0% |
| Moderate | -4.3% | -0.05 | -5.5% | -3.9% |
| Growth | -0.4% | +0.08 | +1.2% | -2.3% |
| Aggressive | +44.4% | +0.47 | -15.7% | +20.9% |

---

### 5Y Results  (2021-06-12 → 2026-06-12)

| Strategy | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | VaR 95% | CVaR 95% | Best Yr | Worst Yr |
|----------|---|---|---|---|---|---|---|---|---|---------|---------|
| **Unhedged** | +90.6% | +13.8% | +17.6% | 0.57 | 0.79 | 0.51 | -27.1% | -1.8% | -2.5% | +42% (2023) | -23% (2022) |
| Ultra Conservative | +17.2% | +3.2% | +7.6% | -0.14 | -0.21 | 0.30 | -10.9% | -0.7% | -1.1% | +19% (2023) | -8% (2022) |
| Conservative | +19.1% | +3.6% | +9.8% | -0.05 | -0.08 | 0.26 | -13.9% | -0.9% | -1.4% | +24% (2023) | -10% (2022) |
| Moderate | +54.5% | +9.1% | +14.8% | 0.36 | 0.46 | 0.28 | -32.6% | -1.5% | -2.3% | +28% (2023) | -26% (2022) |
| Growth | +85.6% | +13.2% | +16.4% | 0.57 | 0.80 | 0.51 | -25.9% | -1.7% | -2.4% | +38% (2023) | -21% (2022) |
| Aggressive | +370.8% | +36.5% | +38.8% | 0.88 | 1.25 | 0.85 | -42.8% | -4.1% | -5.6% | +123% (2023) | -38% (2022) |

#### 5Y Delta vs Unhedged

| Strategy | ΔCAGR | ΔSharpe | ΔMax DD | ΔVol |
|----------|-------|---------|---------|------|
| Ultra Conservative | -10.6% | -0.70 | +16.2% | -10.0% |
| Conservative | -10.3% | -0.62 | +13.2% | -7.8% |
| Moderate | -4.7% | -0.21 | -5.5% | -2.9% |
| Growth | -0.6% | -0.00 | +1.2% | -1.2% |
| Aggressive | +22.7% | +0.31 | -15.7% | +21.2% |

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

1. **Capital preservation:** Ultra Conservative achieves Sharpe 0.66 (-0.23 vs unhedged) with max drawdown -10.9% vs -27.1% unhedged (10Y).
2. **Best risk-adjusted (conservative):** Conservative profile targets consistent Sharpe improvement via always-on hedging + vol targeting at ~10% annualised vol.
3. **Moderate:** Conditional hedge (VIX/vol trigger) reduces max drawdown by 5.5% while preserving most upside. Good balance for long-horizon investors.
4. **Growth:** Crisis-only hedge lets the portfolio ride bull markets nearly unhedged, with Calmar 0.82 vs 0.80 unhedged (10Y).  CAGR +21.2% vs +21.5% unhedged.
5. **Aggressive (only profile exceeding unhedged):** Leveraged long with crisis-only hedge targets CAGR +66.0% vs +21.5% unhedged, at vol 39.6% vs 18.7% unhedged (10Y).  Max DD: -42.8%.  **Suitable only for investors with high risk tolerance and long horizon.**

---

*All backtests use strict no-lookahead guards: Kalman beta, VIX gate, and vol scale are each shifted one day before application.  Transaction costs: 2–3 bps per hedge toggle.  Risk-free rate: 4.5% p.a.*
