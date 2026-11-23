"""
utils.py
========

Physical constants and small helper functions shared across the pipeline.

We keep every constant in SI and convert at the edges. Mixing unit systems is
the single most common bug in student transit code, so there is exactly one
place (here) where the numbers live.
"""

from __future__ import annotations

import numpy as np

# --- physical constants (SI) -------------------------------------------------
R_SUN = 6.957e8        # metres  (nominal solar radius, IAU 2015)
R_EARTH = 6.3781e6     # metres  (nominal Earth equatorial radius, IAU 2015)
R_JUP = 7.1492e7       # metres  (nominal Jupiter equatorial radius, IAU 2015)
R_SUN_IN_REARTH = R_SUN / R_EARTH   # ~109.1
R_SUN_IN_RJUP = R_SUN / R_JUP       # ~9.73


def depth_to_radius_ratio(depth: float) -> float:
    """Convert a fractional transit depth to a planet/star radius ratio.

    For a uniform-brightness star the transit depth equals the fraction of
    stellar area the planet blocks:

        depth = (Rp / Rs) ** 2        =>   Rp / Rs = sqrt(depth)

    `depth` is fractional (0.01 == a 1% dip == 10,000 ppm), NOT in ppm.
    Real stars are limb-darkened, so this slightly overestimates Rp/Rs; the
    full `fit` module handles that. Good enough for a first number.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0 (it is a fractional dip)")
    return float(np.sqrt(depth))


def radius_ratio_to_rearth(ratio: float, rstar_rsun: float) -> float:
    """Turn Rp/Rs plus a stellar radius (in solar radii) into Rp in Earth radii."""
    return float(ratio * rstar_rsun * R_SUN_IN_REARTH)


def depth_to_rearth(depth: float, rstar_rsun: float) -> float:
    """Shortcut: fractional depth + stellar radius -> planet radius in R_earth."""
    return radius_ratio_to_rearth(depth_to_radius_ratio(depth), rstar_rsun)


def expected_transit_duration(period_days: float, rstar_rsun: float,
                              mstar_msun: float = 1.0) -> float:
    """Rough central-transit duration (days) for a circular orbit.

    Uses T ~ (P / pi) * (Rs / a), with a from Kepler's third law. This is the
    'total' duration for a small planet crossing the stellar diameter at
    impact parameter b = 0. Handy for setting BLS duration grids.
    """
    G = 6.674e-11
    M_SUN = 1.989e30
    day = 86400.0
    P = period_days * day
    a = (G * mstar_msun * M_SUN * P ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    Rs = rstar_rsun * R_SUN
    return float((P / np.pi) * (Rs / a) / day)


def stellar_density(rstar_rsun: float, mstar_msun: float,
                    rstar_err: float = 0.0, mstar_err: float = 0.0):
    """Mean stellar density in g/cm^3 (the Sun is ~1.41), with error.

    Use as a prior in fit.fit_batman(rho_star=..., rho_star_err=...).
    rho = M / (4/3 pi R^3)  =>  (s_rho/rho)^2 = (s_M/M)^2 + (3 s_R/R)^2
    """
    M_SUN_G, R_SUN_CM = 1.989e33, 6.957e10
    rho = mstar_msun * M_SUN_G / (4.0 / 3.0 * np.pi * (rstar_rsun * R_SUN_CM) ** 3)
    frac = np.hypot(mstar_err / mstar_msun, 3 * rstar_err / rstar_rsun)
    return float(rho), float(rho * frac)


def phase_fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Fold times onto a phase in [-0.5, 0.5), with phase 0 at the transit."""
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    return phase


def bin_series(x: np.ndarray, y: np.ndarray, bins: int = 200):
    """Bin y(x) into `bins` equal-width bins; return bin centres and means.

    Empty bins are dropped. Useful for a clean phase-folded plot on top of the
    noisy points.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    centres, means = [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(np.nanmean(y[m]))
    return np.asarray(centres), np.asarray(means)
