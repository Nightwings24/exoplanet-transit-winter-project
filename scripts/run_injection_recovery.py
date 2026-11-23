#!/usr/bin/env python
"""
run_injection_recovery.py -- measure your pipeline's detection efficiency.

Injects fake planets on a (period x radius) grid into a RAW light curve,
runs the same detrend + BLS pipeline blind on each, and records what fraction
are recovered. Writes:

    injections.csv        one row per injected planet
    completeness.png      2-D recovered fraction (period x radius)
    efficiency_curve.png  recovered fraction vs expected SNR + logistic fit

Examples
--------
Quick look, offline (~10-30 s on a laptop):
    python scripts/run_injection_recovery.py --synthetic --fast

Real star, with its known planet masked out first (take period/t0/duration
from the result.json that run_baseline.py wrote for the same quarters):
    python scripts/run_injection_recovery.py --target "Kepler-10" \
        --mission Kepler --quarter 3 --rstar 1.065 \
        --mask-period 0.83766 --mask-t0 260.537 --mask-duration 0.08 \
        --n-per-cell 20 --n-jobs 4

IMPORTANT: --method / --window / --oversample must match what you used in
run_baseline.py. Otherwise you are measuring a different pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transitpipe import io, detrend, inject, plotting  # noqa: E402
from transitpipe.io import LightCurve  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--target")
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--csv")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--quarter", type=int, nargs="*")
    p.add_argument("--sector", type=int, nargs="*")
    p.add_argument("--exptime", default=None)
    p.add_argument("--rstar", type=float, default=None)
    p.add_argument("--mstar", type=float, default=None)
    p.add_argument("--mask-period", type=float, help="known planet to remove first")
    p.add_argument("--mask-t0", type=float)
    p.add_argument("--mask-duration", type=float)
    p.add_argument("--pmin", type=float, default=0.7)
    p.add_argument("--pmax", type=float, default=12.0)
    p.add_argument("--np", dest="n_pbins", type=int, default=6)
    p.add_argument("--rmin", type=float, default=0.7)
    p.add_argument("--rmax", type=float, default=6.0)
    p.add_argument("--nr", dest="n_rbins", type=int, default=6)
    p.add_argument("--n-per-cell", type=int, default=10)
    p.add_argument("--method", default="median", choices=["median", "savgol", "biweight"])
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--oversample", type=float, default=3.0)
    p.add_argument("--sde-threshold", type=float, default=7.0)
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--fast", action="store_true",
                   help="4x4 grid, 5 per cell, coarse period grid: a quick demo")
    p.add_argument("--outdir", default="results/injection")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.outdir, exist_ok=True)
    if a.fast:
        a.n_pbins, a.n_rbins, a.n_per_cell, a.oversample = 4, 4, 5, 1.0

    if a.synthetic:
        lc = inject.make_synthetic_lightcurve(seed=a.seed + 1)
    elif a.csv:
        lc = io.from_csv(a.csv).normalize()
    else:
        lc = io.get_lightcurve(a.target, mission=a.mission, exptime=a.exptime,
                               quarter=a.quarter, sector=a.sector)
    if a.rstar is not None:
        lc.meta["rstar"] = a.rstar
    if a.mstar is not None:
        lc.meta["mstar"] = a.mstar

    if a.mask_period:
        m = detrend.transit_mask(lc.time, a.mask_period, a.mask_t0,
                                 a.mask_duration, pad=2.0)
        lc = LightCurve(lc.time[~m], lc.flux[~m],
                        None if lc.flux_err is None else lc.flux_err[~m], lc.meta)
        print(f"[mask] removed {m.sum()} in-transit points of the known planet")

    pe = np.geomspace(a.pmin, a.pmax, a.n_pbins + 1)
    re = np.geomspace(a.rmin, a.rmax, a.n_rbins + 1)
    n_total = a.n_pbins * a.n_rbins * a.n_per_cell
    print(f"[grid] {a.n_pbins} x {a.n_rbins} cells x {a.n_per_cell} = {n_total} "
          f"injections on {a.n_jobs} process(es)")

    t = time.time()
    df = inject.injection_recovery(
        lc, pe, re, n_per_cell=a.n_per_cell,
        detrend_kwargs={"method": a.method, "window_days": a.window},
        # search a bit beyond the injected range, but never past baseline/2
        bls_kwargs={"oversample": a.oversample,
                    "max_period": min(1.2 * a.pmax, 0.5 * np.ptp(lc.time))},
        sde_threshold=a.sde_threshold, seed=a.seed, n_jobs=a.n_jobs)
    print(f"[run] {len(df)} trials in {time.time() - t:.1f} s")

    csv = os.path.join(a.outdir, "injections.csv")
    df.to_csv(csv, index=False)
    print(f"[out] {csv}")
    print(f"[result] overall recovered fraction: {df.recovered.mean():.2f}")
    print("[result] outcome counts:", df.match_kind.value_counts().to_dict())

    ax = inject.plot_completeness(df, pe, re)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(a.outdir, "completeness.png"), dpi=140)

    ax = plotting.plot_efficiency(df)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(a.outdir, "efficiency_curve.png"), dpi=140)
    try:
        s50, w, _ = inject.fit_efficiency_curve(df)
        print(f"[result] 50% recovery at expected SNR = {s50:.1f} (width {w:.1f})")
    except Exception as exc:
        print(f"[result] efficiency fit failed ({exc}); try more injections")
    plt.close("all")
    print(f"[done] figures in {a.outdir}/")


if __name__ == "__main__":
    main()
