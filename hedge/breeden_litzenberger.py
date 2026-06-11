#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Breeden-Litzenberger implementation for extracting risk-neutral probability
densities from option call prices.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def smooth_call_prices(strikes: np.ndarray, prices: np.ndarray,
                       lambda_reg: float = 1e-2) -> np.ndarray:
    """
    Smooth call prices using ridge regression on second differences.
    Minimizes: ||C_smooth - C||^2 + lambda * ||D2 C_smooth||^2
    """
    n = len(prices)
    if n < 5:
        return prices.copy()

    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i:i + 3] = [1, -2, 1]

    A = np.eye(n) + lambda_reg * (D2.T @ D2)
    return np.linalg.solve(A, prices)


def extract_implied_density(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    spot: float,
    r: float = 0.0,
    T: float = None,
    max_violation: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract risk-neutral probability density from call prices using
    Breeden-Litzenberger: f(K) = exp(r*T) * d²C/dK²

    Returns:
        K_uniform  : uniform strike grid
        density    : probability density f(K), normalised to integrate to 1
        quality    : dict with 'valid' (bool), 'warnings' (list), 'metrics' (dict)
    """
    quality: Dict = {"valid": True, "warnings": []}

    idx = np.argsort(strikes)
    K = strikes[idx]
    C = call_prices[idx]

    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(K) > 1e-6])
    K = K[unique_mask]
    C = C[unique_mask]

    # Enforce monotonicity — clip small violations, drop large ones
    valid_mask = np.ones(len(K), dtype=bool)
    for i in range(1, len(K)):
        if C[i] > C[i - 1]:
            violation = C[i] - C[i - 1]
            if violation > max_violation:
                valid_mask[i] = False
                quality["warnings"].append(
                    f"Dropped strike {K[i]:.2f} (monotonicity violation {violation:.4f})"
                )
            else:
                C[i] = C[i - 1] - 1e-6

    K = K[valid_mask]
    C = C[valid_mask]

    if len(K) < 5:
        quality["valid"] = False
        quality["warnings"].append("Too few strikes after filtering")
        return K, np.zeros_like(K), quality

    C_smooth = smooth_call_prices(K, C, lambda_reg=1e-2)

    n_points = max(100, len(K) * 2)
    K_uniform = np.linspace(K.min(), K.max(), n_points)

    interp = interp1d(K, C_smooth, kind="linear", bounds_error=False,
                      fill_value="extrapolate")
    C_uniform = interp(K_uniform)

    # Re-enforce monotonicity on uniform grid
    for i in range(1, len(C_uniform)):
        if C_uniform[i] > C_uniform[i - 1]:
            C_uniform[i] = C_uniform[i - 1] - 1e-6

    dK = K_uniform[1] - K_uniform[0]
    d2C = np.gradient(np.gradient(C_uniform, dK), dK)

    if T is None:
        T = 45 / 365.0
    density = np.exp(r * T) * d2C
    density = np.maximum(density, 0.0)

    total_mass = np.trapz(density, K_uniform)
    if total_mass > 0:
        density = density / total_mass

    quality.update(_check_density_quality(K_uniform, density, spot))
    return K_uniform, density, quality


def _check_density_quality(strikes: np.ndarray, density: np.ndarray,
                            spot: float) -> Dict:
    """Internal quality checks on extracted density."""
    quality: Dict = {"valid": True, "warnings": [], "metrics": {}}

    strike_range = strikes.max() - strikes.min()
    left_boundary = strikes.min() + 0.1 * strike_range
    left_mass = np.trapz(density[strikes <= left_boundary],
                         strikes[strikes <= left_boundary])
    quality["metrics"]["left_mass_pct"] = left_mass * 100
    if left_mass > 0.6:
        quality["valid"] = False
        quality["warnings"].append(
            f"Too much mass ({left_mass:.1%}) in lowest 10% of strikes"
        )

    peaks, _ = find_peaks(density, height=np.max(density) * 0.1)
    peak_ratio = len(peaks) / len(strikes)
    quality["metrics"]["n_peaks"] = len(peaks)
    quality["metrics"]["peak_ratio"] = peak_ratio
    if peak_ratio > 0.3:
        quality["valid"] = False
        quality["warnings"].append(
            f"Too many local maxima ({len(peaks)} peaks, {peak_ratio:.1%} of strikes)"
        )

    total_mass = np.trapz(density, strikes)
    quality["metrics"]["total_mass"] = total_mass
    if total_mass < 0.5 or total_mass > 1.5:
        quality["warnings"].append(
            f"Density mass ({total_mass:.3f}) is far from 1.0"
        )

    return quality
