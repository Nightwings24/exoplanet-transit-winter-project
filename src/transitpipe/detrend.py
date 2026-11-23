"""
detrend.py
==========

Remove slow variability (stellar rotation, spacecraft drifts) while keeping
the short, box-shaped transit dips.

The core idea fits in one sentence: estimate a smooth "trend" with a sliding
window that is much LONGER than a transit but much SHORTER than the
variability, then divide the flux by that trend.

    flat_flux = flux / trend

Rule of thumb for the window: about 3x the longest transit you hope to find.
Too short and you start eating the transit itself (shallower depths, missed
planets). Too long and stellar wiggles survive into your BLS search.

Methods
-------
"median"  : sliding median in points (fast, robust, default).
"savgol"  : Savitzky-Golay polynomial smoother (fast, but NOT robust to dips:
            it gets pulled down by transits; good to show students *why* robust
            estimators matter).
"biweight": Tukey's biweight sliding in time. This is what Hippke et al.
            (2019, the `wotan` paper) found works best for transit searches.
            Uses `wotan` if installed, otherwise a slower pure-NumPy version.

Every method splits the light curve at data gaps first. Smoothing across a
gap (e.g. between TESS orbits or Kepler quarters) creates fake features.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .io import LightCurve


# --- helpers -----------------------------------------------------------------

def split_at_gaps(time: np.ndarray, gap_days: float = 0.5) -> list[slice]:
    """Return slices of contiguous segments, breaking wherever dt > gap_days."""
    if len(time) == 0:
        return []
    breaks = np.where(np.diff(time) > gap_days)[0] + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [len(time)]])
    return [slice(a, b) for a, b in zip(starts, stops)]


def _window_in_points(time: np.ndarray, window_days: float) -> int:
    """Convert a window length in days into an odd number of samples."""
    if len(time) < 2:
        return 1
    cadence = np.median(np.diff(time))
    n = int(round(window_days / cadence))
    n = max(n, 3)
    return n if n % 2 == 1 else n + 1


def _fill_masked(time: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked samples (mask=True) by linear interpolation of the rest.

    This is how we 'protect' known transits: the smoother never sees the dip,
    so the trend under the transit is interpolated from either side.
    """
    if mask is None or not mask.any():
        return flux
    good = ~mask
    if good.sum() < 2:
        return flux
    out = flux.copy()
    out[mask] = np.interp(time[mask], time[good], flux[good])
    return out


def _biweight_location(x: np.ndarray, c: float = 5.0, iters: int = 5) -> float:
    """Tukey biweight location estimate (robust 'average' of x).

    Points further than c * MAD from the current estimate get zero weight,
    so a transit (a cluster of low points) barely moves the estimate.
    """
    m = np.median(x)
    for _ in range(iters):
        mad = np.median(np.abs(x - m))
        if mad == 0:
            return float(m)
        u = (x - m) / (c * mad)
        w = (1 - u ** 2) ** 2
        w[np.abs(u) >= 1] = 0.0
        if w.sum() == 0:
            return float(m)
        m = np.sum(w * x) / np.sum(w)
    return float(m)


def _biweight_trend_numpy(time: np.ndarray, flux: np.ndarray,
                          window_days: float) -> np.ndarray:
    """Pure-NumPy time-windowed biweight. O(N * points-per-window); slow-ish."""
    half = 0.5 * window_days
    lo = np.searchsorted(time, time - half, side="left")
    hi = np.searchsorted(time, time + half, side="right")
    trend = np.empty_like(flux)
    for i in range(len(time)):
        trend[i] = _biweight_location(flux[lo[i]:hi[i]])
    return trend


# --- public API --------------------------------------------------------------

def compute_trend(lc: LightCurve, method: str = "median",
                  window_days: float = 1.0, gap_days: float = 0.5,
                  mask: np.ndarray | None = None,
                  savgol_polyorder: int = 2) -> np.ndarray:
    """Return the smooth trend (same length as lc.flux) without dividing it out."""
    time, flux = lc.time, lc.flux
    work = _fill_masked(time, flux, mask) if mask is not None else flux
    trend = np.empty_like(flux)

    if method == "biweight":
        try:
            from wotan import flatten as wotan_flatten
            # wotan handles gaps itself via break_tolerance
            _, trend = wotan_flatten(time, work, method="biweight",
                                     window_length=window_days,
                                     break_tolerance=gap_days,
                                     return_trend=True)
            # wotan can leave NaN at segment edges; fill from the raw flux
            bad = ~np.isfinite(trend)
            trend[bad] = np.nanmedian(work)
            return trend
        except ImportError:
            pass  # fall through to the NumPy version per segment

    for seg in split_at_gaps(time, gap_days):
        t, f = time[seg], work[seg]
        if len(t) < 5:
            trend[seg] = np.median(f)
            continue
        n = min(_window_in_points(t, window_days), len(t) - (1 - len(t) % 2))
        if method == "median":
            trend[seg] = median_filter(f, size=n, mode="nearest")
        elif method == "savgol":
            # window must be odd and longer than the polynomial order
            n = max(n, savgol_polyorder + 3)
            n = min(n, len(t))
            if n % 2 == 0:
                n -= 1
            if n <= savgol_polyorder:
                trend[seg] = np.median(f)
                continue
            trend[seg] = savgol_filter(f, n, savgol_polyorder, mode="interp")
        elif method == "biweight":
            trend[seg] = _biweight_trend_numpy(t, f, window_days)
        else:
            raise ValueError(f"Unknown method {method!r}. "
                             "Use 'median', 'savgol', or 'biweight'.")
    return trend


def flatten(lc: LightCurve, method: str = "median", window_days: float = 1.0,
            gap_days: float = 0.5, mask: np.ndarray | None = None,
            return_trend: bool = False, **kwargs):
    """Detrend a light curve: divide the flux by a smooth sliding-window trend.

    Parameters
    ----------
    lc : LightCurve
    method : {"median", "savgol", "biweight"}
    window_days : float
        Smoothing window. Aim for ~3x the longest transit duration you want.
    gap_days : float
        Data gaps longer than this split the light curve into segments.
    mask : bool array, optional
        True where a known transit sits. Those points are hidden from the
        smoother so the dip is not partially "detrended away".
    return_trend : bool
        Also return the trend array (great for plotting!).

    Returns
    -------
    LightCurve (and the trend if return_trend=True)
    """
    trend = compute_trend(lc, method=method, window_days=window_days,
                          gap_days=gap_days, mask=mask, **kwargs)
    flat = lc.copy()
    flat.flux = lc.flux / trend
    if lc.flux_err is not None:
        flat.flux_err = lc.flux_err / trend
    flat.meta["detrend"] = {"method": method, "window_days": window_days}
    return (flat, trend) if return_trend else flat


def sigma_clip(lc: LightCurve, sigma: float = 4.0, upper_only: bool = True,
               iters: int = 3) -> LightCurve:
    """Remove outliers using a robust (MAD-based) sigma clip.

    upper_only=True is the default ON PURPOSE: transits are *low* points. A
    symmetric clip at 3 sigma will happily delete the planet you are looking
    for. Only clip downward if you know what you are doing.
    """
    keep = np.ones(len(lc), dtype=bool)
    for _ in range(iters):
        f = lc.flux[keep]
        med = np.median(f)
        std = 1.4826 * np.median(np.abs(f - med))  # MAD -> Gaussian sigma
        if std == 0:
            break
        resid = (lc.flux - med) / std
        new_keep = resid < sigma if upper_only else np.abs(resid) < sigma
        if np.array_equal(new_keep & keep, keep):
            break
        keep &= new_keep
    out = LightCurve(lc.time[keep], lc.flux[keep],
                     None if lc.flux_err is None else lc.flux_err[keep],
                     dict(lc.meta))
    return out


def transit_mask(time: np.ndarray, period: float, t0: float,
                 duration: float, pad: float = 1.5) -> np.ndarray:
    """Boolean mask, True inside (pad x duration) of every transit."""
    phase_time = np.abs(((time - t0 + 0.5 * period) % period) - 0.5 * period)
    return phase_time < 0.5 * pad * duration


def flatten_masked(lc_raw: LightCurve, period: float, t0: float,
                   duration: float, pad: float = 2.5, **flatten_kwargs):
    """Second-pass detrending with the found transits hidden from the smoother.

    Why this exists: inside a sliding window where the star is brightening or
    fading, a few low in-transit points shift the median's rank and drag the
    trend DOWN under the transit. Dividing by that trend makes the transit
    look shallower (we measured ~25% too shallow on an active synthetic star).
    The fix used by real pipelines is two passes:

        1. detrend -> BLS  (find the planet)
        2. mask the transits -> re-detrend the RAW flux -> fit  (measure it)

    Always fit on the output of this function, not on the first-pass flatten.
    """
    m = transit_mask(lc_raw.time, period, t0, duration, pad=pad)
    return flatten(lc_raw, mask=m, **flatten_kwargs)


def estimate_noise_ppm(lc: LightCurve) -> float:
    """Point-to-point scatter in parts-per-million (robust, MAD-based).

    Good single number to compare light curves and to set expectations: a
    transit of depth d is roughly detectable when
        d / noise * sqrt(N_in_transit) >~ 7.
    """
    f = lc.flux
    return float(1.4826 * np.median(np.abs(f - np.median(f))) * 1e6)
