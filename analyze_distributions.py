#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze SPY options snapshots to compare implied probability distributions
across different market regimes.

Usage:
  python analyze_distributions.py                        # auto-select extreme dates
  python analyze_distributions.py --date1 2024-01-15 --date2 2024-10-01
  python analyze_distributions.py --top_n 3
"""

import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hedge.regime import (
    analyze_all_snapshots,
    compute_regime_features,
    compute_regime_scores,
    compute_skew_metrics,
    find_snapshot_by_date,
    select_extreme_snapshots,
)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

DATA_DIR = "data/opra_spy_snapshots/snapshots"
OUTPUT_DIR = "output/distributions"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_distribution_plot(snapshot: Dict, title: str, ax=None):
    strikes = snapshot["strikes"]
    density = snapshot["density"]
    metrics = snapshot["metrics"]
    spot = metrics["spot"]
    mean = metrics["mean"]
    median = metrics["median"]

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    ax.fill_between(strikes, 0, density, alpha=0.6, color="steelblue",
                    label="Implied Probability Density")
    ax.plot(strikes, density, color="darkblue", linewidth=2.5, alpha=0.9)

    ax.axvline(spot, color="red", linestyle="--", linewidth=2.5,
               label=f"Current Spot: ${spot:.2f}", alpha=0.8)
    ax.axvline(mean, color="orange", linestyle="--", linewidth=2.5,
               label=f"Expected Mean: ${mean:.2f}", alpha=0.8)
    ax.axvline(median, color="green", linestyle="--", linewidth=2.5,
               label=f"Median: ${median:.2f}", alpha=0.8)

    left_mask = strikes <= 0.95 * spot
    if np.any(left_mask):
        ax.fill_between(strikes[left_mask], 0, density[left_mask],
                        alpha=0.4, color="crimson", label="Left Tail Risk")

    ax.set_xlabel("SPY Price at Expiration ($)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    skew_proxy = metrics["skew_proxy"]
    if skew_proxy < -0.1:
        skew_label = "Left-Skewed (Stress)"
    elif skew_proxy > 0.1:
        skew_label = "Right-Skewed (Calm)"
    else:
        skew_label = "Symmetric"

    stats_text = (
        f"Regime: {skew_label}\n"
        f"Skew Proxy: {skew_proxy:.3f}\n"
        f"Std Deviation: ${metrics['std']:.2f}\n"
        f"Downside Risk (≤95% spot): {metrics['left_tail_prob']:.1%}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.9,
                      edgecolor="gray", linewidth=1.5))

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f9fa")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    return ax


def create_comparison_plot(snapshot1: Dict, snapshot2: Dict,
                           label1: str, label2: str):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Implied Probability Distributions: Market Regime Comparison",
                 fontsize=16, fontweight="bold", y=1.02)

    all_strikes = np.concatenate([snapshot1["strikes"], snapshot2["strikes"]])
    strike_min = all_strikes.min() * 0.98
    strike_max = all_strikes.max() * 1.02

    create_distribution_plot(snapshot1, label1, ax=axes[0])
    create_distribution_plot(snapshot2, label2, ax=axes[1])

    axes[0].set_xlim(strike_min, strike_max)
    axes[1].set_xlim(strike_min, strike_max)
    max_density = max(snapshot1["density"].max(), snapshot2["density"].max())
    axes[0].set_ylim(0, max_density * 1.15)
    axes[1].set_ylim(0, max_density * 1.15)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare options-implied probability distributions across market regimes"
    )
    parser.add_argument("--date1", type=str, default=None)
    parser.add_argument("--date2", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please run data_downloader.py first.")
        return

    if args.date1 and args.date2:
        stress_snapshot = find_snapshot_by_date(DATA_DIR, args.date1)
        calm_snapshot = find_snapshot_by_date(DATA_DIR, args.date2)

        if stress_snapshot is None:
            print(f"Error: No snapshot found for {args.date1}")
            return
        if calm_snapshot is None:
            print(f"Error: No snapshot found for {args.date2}")
            return

        scores1 = compute_regime_scores(
            compute_regime_features(stress_snapshot["strikes"],
                                    stress_snapshot["density"],
                                    stress_snapshot["spot"])
        )
        scores2 = compute_regime_scores(
            compute_regime_features(calm_snapshot["strikes"],
                                    calm_snapshot["density"],
                                    calm_snapshot["spot"])
        )
        if scores1["stress_score"] < scores2["stress_score"]:
            stress_snapshot, calm_snapshot = calm_snapshot, stress_snapshot
            scores1, scores2 = scores2, scores1

        stress_snapshots = [stress_snapshot]
        calm_snapshots = [calm_snapshot]
    else:
        df_results = analyze_all_snapshots(DATA_DIR)

        print("\n" + "=" * 100)
        print("RANKED SNAPSHOTS BY STRESS SCORE")
        print("=" * 100)
        df_display = df_results[
            ["date", "stress_score", "p_left_5", "std_ret", "skew_proxy"]
        ].sort_values("stress_score", ascending=False)
        for col in ["stress_score", "p_left_5", "std_ret", "skew_proxy"]:
            df_display[col] = df_display[col].map("{:.4f}".format)
        print(df_display.to_string(index=False))
        print("=" * 100 + "\n")

        stress_snapshots, calm_snapshots = select_extreme_snapshots(
            df_results, top_n=args.top_n
        )

        print(f"Top {args.top_n} most stressed snapshots:")
        for i, s in enumerate(stress_snapshots, 1):
            print(f"  {i}. {s['date']}: stress={s['stress_score']:.4f}, "
                  f"p_left_5={s['p_left_5']:.4f}")
        print(f"\nTop {args.top_n} calmest snapshots:")
        for i, s in enumerate(calm_snapshots, 1):
            print(f"  {i}. {s['date']}: stress={s['stress_score']:.4f}, "
                  f"p_left_5={s['p_left_5']:.4f}")

    stress_snapshot = stress_snapshots[0]
    calm_snapshot = calm_snapshots[0]

    # Ensure metrics key present (find_snapshot_by_date provides it; analyze path may not)
    for snap in (stress_snapshot, calm_snapshot):
        if "metrics" not in snap:
            snap["metrics"] = compute_skew_metrics(
                snap["strikes"], snap["density"], snap["spot"]
            )

    print("\nCreating visualizations...")

    fig1 = plt.figure(figsize=(14, 8))
    create_distribution_plot(
        stress_snapshot,
        f"Stress Regime – {stress_snapshot['date']}\n(High Downside Risk, High Volatility)",
        ax=plt.gca(),
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "stress_regime_distribution.png"),
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig1)
    print(f"Saved: {OUTPUT_DIR}/stress_regime_distribution.png")

    fig2 = plt.figure(figsize=(14, 8))
    create_distribution_plot(
        calm_snapshot,
        f"Calm Regime – {calm_snapshot['date']}\n(Low Downside Risk, Low Volatility)",
        ax=plt.gca(),
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "calm_regime_distribution.png"),
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig2)
    print(f"Saved: {OUTPUT_DIR}/calm_regime_distribution.png")

    fig3 = create_comparison_plot(
        stress_snapshot,
        calm_snapshot,
        f"Stress – {stress_snapshot['date']} "
        f"(score={stress_snapshot['stress_score']:.3f}, "
        f"P(≤-5%)={stress_snapshot['p_left_5']:.2%})",
        f"Calm – {calm_snapshot['date']} "
        f"(score={calm_snapshot['stress_score']:.3f}, "
        f"P(≤-5%)={calm_snapshot['p_left_5']:.2%})",
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "regime_comparison.png"),
                bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig3)
    print(f"Saved: {OUTPUT_DIR}/regime_comparison.png")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for label, snap in [("Stress", stress_snapshot), ("Calm", calm_snapshot)]:
        print(f"\n{label} Regime ({snap['date']}):")
        for key in ["stress_score", "std_ret", "p_left_5", "p_right_5", "skew_proxy"]:
            print(f"  {key:20s}: {snap[key]:10.4f}")
        print(f"  {'spot':20s}: {snap['spot']:10.2f}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
