"""
plotting.py
===========

The four plots every transit project needs. Each returns the matplotlib Axes
so you can restyle it for your poster.

    plot_raw(lc, trend)          raw flux with the detrending trend on top
    plot_periodogram(bls_res)    BLS power vs period, best peak + aliases marked
    plot_fold(lc, period, t0)    phase-folded transit, binned, optional model
    plot_efficiency(df)          recovered fraction vs expected SNR
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .utils import phase_fold, bin_series


def plot_raw(lc, trend=None, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(lc.time, lc.flux, ".", ms=1.5, color="0.3", label="flux")
    if trend is not None:
        ax.plot(lc.time, trend, "-", lw=1.2, color="C1", label="trend")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalised flux")
    ax.set_title(title or lc.meta.get("target", ""))
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_periodogram(res, ax=None, true_period=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(res.periods, res.power, "-", lw=0.6, color="k")
    ax.axvline(res.period, color="C3", alpha=0.4, lw=4, label=f"best P = {res.period:.4f} d")
    for n in (0.5, 2.0):
        ax.axvline(res.period * n, color="C3", alpha=0.3, ls="--", lw=1)
    if true_period is not None:
        ax.axvline(true_period, color="C2", ls=":", lw=1.5, label=f"true P = {true_period:.4f} d")
    ax.set_xscale("log")
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("BLS power")
    ax.set_title(f"BLS periodogram (SDE = {res.sde:.1f})")
    ax.legend(fontsize=8)
    return ax


def plot_fold(lc, period, t0, ax=None, model_func=None, window_hours=None,
              bins=None, title=None):
    """Phase-fold on (period, t0). x-axis in hours from mid-transit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    hours = phase_fold(lc.time, period, t0) * period * 24
    order = np.argsort(hours)
    ax.plot(hours, (lc.flux - 1) * 1e6, ".", ms=2, color="0.6", label="data")
    nb = bins or max(30, int(np.sqrt(len(lc.time))))
    w = window_hours or 0.25 * period * 24
    sel = np.abs(hours) < w
    if sel.sum() > 10:
        xb, yb = bin_series(hours[sel], lc.flux[sel], bins=min(nb, sel.sum() // 3))
        ax.plot(xb, (yb - 1) * 1e6, "o", ms=4, color="C0", label="binned")
    if model_func is not None:
        tt = t0 + hours[order] / 24
        ax.plot(hours[order], (model_func(tt) - 1) * 1e6, "-", color="C3", lw=1.5, label="model")
    ax.set_xlim(-w, w)
    ax.set_xlabel("Hours from mid-transit")
    ax.set_ylabel("Relative flux [ppm]")
    ax.set_title(title or f"Folded on P = {period:.5f} d")
    ax.legend(fontsize=8)
    return ax


def plot_efficiency(df, ax=None, snr_col="snr_expected", nbins=12, fit=True):
    """Recovered fraction vs expected SNR, with the logistic fit from inject.py."""
    from .inject import fit_efficiency_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x = df[snr_col].to_numpy()
    y = df["recovered"].to_numpy().astype(float)
    good = np.isfinite(x) & (x > 0)
    edges = np.geomspace(max(x[good].min(), 0.5), x[good].max(), nbins + 1)
    idx = np.digitize(x[good], edges) - 1
    cen, frac, err = [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            f = y[good][m].mean()
            cen.append(np.sqrt(edges[b] * edges[b + 1]))
            frac.append(f)
            err.append(np.sqrt(max(f * (1 - f), 0.25 / m.sum()) / m.sum()))
    ax.errorbar(cen, frac, yerr=err, fmt="o", color="C0", label="injections")
    if fit:
        try:
            s50, width, logistic = fit_efficiency_curve(df, snr_col)
            xs = np.geomspace(edges[0], edges[-1], 300)
            ax.plot(xs, logistic(xs, s50, width), "-", color="C3",
                    label=f"logistic: S50 = {s50:.1f}")
        except Exception as exc:  # fit can fail on tiny grids
            ax.text(0.05, 0.9, f"fit failed: {exc}", transform=ax.transAxes, fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Expected SNR of injected transit")
    ax.set_ylabel("Fraction recovered")
    ax.set_title("Detection efficiency curve")
    ax.legend(fontsize=8)
    return ax
