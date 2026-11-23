"""
search.py
=========

Find periodic transit-like dips with Box Least Squares (BLS).

BLS in one paragraph (Kovacs, Zucker & Mazeh 2002): pretend the transit is
an upside-down top hat (a "box"). For every trial period P, duration D, and
phase t0, fold the light curve, and ask how much better a box fits than a
flat line. The best (P, D, t0) gives the largest improvement. Plot that
improvement ("power") against P and you get a periodogram; a real planet is a
tall spike, usually with smaller spikes at P/2, 2P, ... (aliases).

We use astropy's implementation (`astropy.timeseries.BoxLeastSquares`), the
same one lightkurve uses under the hood.

Two numbers to judge a detection:
  * SNR  = depth / depth_uncertainty, from astropy's compute_stats.
  * SDE  = (peak power - mean power) / std(power), the "signal detection
           efficiency" of the periodogram. A classic rule of thumb is that an
           SDE above ~7-9 is worth a look. Neither number is magic; this is
           exactly what injection-recovery (see inject.py) lets you calibrate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .io import LightCurve
from .detrend import transit_mask

DEFAULT_DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.33])


@dataclass
class BLSResult:
    """Everything the BLS search found, plus the full periodogram."""

    period: float
    t0: float
    duration: float
    depth: float
    depth_err: float
    snr: float
    sde: float
    periods: np.ndarray = field(repr=False)
    power: np.ndarray = field(repr=False)
    n_transits: int = 0

    def summary(self) -> str:
        return (f"P = {self.period:.5f} d | t0 = {self.t0:.4f} | "
                f"D = {self.duration * 24:.2f} h | depth = {self.depth * 1e6:.0f} "
                f"+/- {self.depth_err * 1e6:.0f} ppm | SNR = {self.snr:.1f} | "
                f"SDE = {self.sde:.1f} | transits = {self.n_transits}")


def period_grid(time: np.ndarray, min_period: float = 0.5,
                max_period: float | None = None,
                durations=DEFAULT_DURATIONS, oversample: float = 3.0,
                max_periods: int = 200_000) -> np.ndarray:
    """Build a trial-period grid, uniform in frequency.

    Spacing: neighbouring frequencies differ by
        df = min(duration) / (oversample * baseline**2)
    which keeps the phase drift across the whole baseline smaller than a
    fraction of the shortest transit. Finer than needed wastes time; coarser
    misses planets (that is a lesson injection-recovery will teach you!).

    max_period defaults to baseline/2 so that at least two transits fit.
    """
    baseline = time.max() - time.min()
    if max_period is None:
        max_period = 0.5 * baseline
    if max_period <= min_period:
        raise ValueError("max_period must exceed min_period; "
                         f"baseline is only {baseline:.2f} d")
    fmin, fmax = 1.0 / max_period, 1.0 / min_period
    df = np.min(durations) / (oversample * baseline ** 2)
    n = int(np.ceil((fmax - fmin) / df))
    if n > max_periods:
        warnings.warn(f"Period grid would have {n:,} points; capping at "
                      f"{max_periods:,}. Narrow the period range or use fewer "
                      "quarters/sectors for a finer search.")
        n = max_periods
    freqs = np.linspace(fmin, fmax, n)
    return np.sort(1.0 / freqs)


def bls(lc: LightCurve, min_period: float = 0.5, max_period: float | None = None,
        durations=DEFAULT_DURATIONS, periods: np.ndarray | None = None,
        oversample: float = 3.0, objective: str = "likelihood",
        max_periods: int = 200_000) -> BLSResult:
    """Run a BLS search on a (detrended!) light curve.

    Always detrend first. Raw light curves are dominated by stellar
    variability, and BLS will happily "detect" a star-spot as a planet.
    """
    from astropy.timeseries import BoxLeastSquares

    durations = np.asarray(durations, dtype=float)
    if periods is None:
        periods = period_grid(lc.time, min_period, max_period, durations,
                              oversample, max_periods)
    # A transit cannot be longer than the period; drop silly durations.
    durations = durations[durations < 0.5 * periods.min()]
    if len(durations) == 0:
        raise ValueError("All durations are too long for the shortest period.")

    dy = lc.flux_err if lc.flux_err is not None else None
    model = BoxLeastSquares(lc.time, lc.flux, dy=dy)
    res = model.power(periods, durations, objective=objective)

    i = int(np.nanargmax(res.power))
    period = float(res.period[i])
    t0 = float(res.transit_time[i])
    duration = float(res.duration[i])
    depth = float(res.depth[i])
    depth_err = float(res.depth_err[i]) if np.isfinite(res.depth_err[i]) else np.nan

    power = np.asarray(res.power)
    sde = float((power[i] - np.nanmean(power)) / np.nanstd(power))
    snr = depth / depth_err if depth_err and np.isfinite(depth_err) else np.nan

    in_tr = transit_mask(lc.time, period, t0, duration, pad=1.0)
    epochs = np.unique(np.round((lc.time[in_tr] - t0) / period))

    return BLSResult(period=period, t0=t0, duration=duration, depth=depth,
                     depth_err=depth_err, snr=float(snr), sde=sde,
                     periods=np.asarray(res.period), power=power,
                     n_transits=int(len(epochs)))


def search_multiple(lc: LightCurve, n_planets: int = 3, sde_threshold: float = 7.0,
                    **bls_kwargs) -> list[BLSResult]:
    """Iterative search: find the strongest signal, mask it, search again.

    This is how you find a second (or third) planet in the same system. Stops
    early when the best remaining peak falls below `sde_threshold`.
    """
    found = []
    work = lc.copy()
    for _ in range(n_planets):
        r = bls(work, **bls_kwargs)
        if r.sde < sde_threshold:
            break
        found.append(r)
        keep = ~transit_mask(work.time, r.period, r.t0, r.duration, pad=2.0)
        work = LightCurve(work.time[keep], work.flux[keep],
                          None if work.flux_err is None else work.flux_err[keep],
                          dict(work.meta))
    return found


def is_match(period_found: float, period_true: float, rel_tol: float = 0.01,
             check_aliases: bool = True) -> tuple[bool, str]:
    """Does a found period match a true one? Returns (match, kind).

    kind is "exact", an alias label like "2:1" / "1:2", or "none". Aliases are
    reported separately because a pipeline that keeps landing on P/2 has a
    real problem you want to see, not hide.
    """
    if abs(period_found - period_true) / period_true < rel_tol:
        return True, "exact"
    if check_aliases:
        for num, den in [(2, 1), (1, 2), (3, 1), (1, 3), (3, 2), (2, 3)]:
            if abs(period_found - period_true * num / den) / (period_true * num / den) < rel_tol:
                return False, f"{num}:{den}"
    return False, "none"
