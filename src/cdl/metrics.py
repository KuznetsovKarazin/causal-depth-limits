"""Spectral and scaling metrics.

Key quantities:
  * band_power(traj, bands): residual error power integrated over frequency bands
    (Welch PSD). This exposes the *causal floor* as an incompressible
    low-frequency component rather than a single opaque MSE number.
  * fit_floor(N, mse): fit  MSE(N) = A / N**alpha + C  and bootstrap a CI on C.
    C > 0 (with CI excluding ~0) is the operational signature of a causal floor.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit


def band_power(traj: np.ndarray, bands, fs: float = 1.0, nperseg: int = 4096):
    """traj: (S, T). bands: list of (f_lo, f_hi) in cycles/step.
    Returns dict band -> per-seed integrated power, plus (f, mean PSD)."""
    f, pxx = welch(traj, fs=fs, nperseg=min(nperseg, traj.shape[1]))
    out = {}
    for (lo, hi) in bands:
        m = (f >= lo) & (f < hi)
        out[(lo, hi)] = np.trapezoid(pxx[:, m], f[m], axis=1)
    return out, (f, pxx.mean(axis=0))


def _model(N, A, alpha, C):
    return A / N ** alpha + C


def fit_floor(Ns, mse_per_seed, n_boot: int = 2000, rng_seed: int = 0):
    """Ns: (K,) agent counts. mse_per_seed: (K, S).
    Fits MSE(N) = A/N^alpha + C to seed means; bootstraps seeds for CI on C."""
    Ns = np.asarray(Ns, float)
    mean = mse_per_seed.mean(axis=1)
    p0 = [max(mean[0] - mean[-1], 1e-6), 1.0, max(mean[-1], 1e-9)]
    bounds = ([0, 0.1, 0], [np.inf, 3.0, np.inf])
    popt, _ = curve_fit(_model, Ns, mean, p0=p0, bounds=bounds, maxfev=20000)

    rng = np.random.default_rng(rng_seed)
    S = mse_per_seed.shape[1]
    Cs, alphas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, S, S)
        m = mse_per_seed[:, idx].mean(axis=1)
        try:
            p, _ = curve_fit(_model, Ns, m, p0=popt, bounds=bounds, maxfev=20000)
            Cs.append(p[2]); alphas.append(p[1])
        except Exception:
            continue
    Cs = np.array(Cs)
    return {"A": popt[0], "alpha": popt[1], "C": popt[2],
            "C_ci": (np.percentile(Cs, 2.5), np.percentile(Cs, 97.5)),
            "alpha_ci": (np.percentile(alphas, 2.5), np.percentile(alphas, 97.5))}
