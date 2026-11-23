#!/usr/bin/env python
"""
run_baseline.py -- the whole baseline pipeline in one command.

    light curve -> detrend -> BLS -> re-detrend with transits masked
                -> fit one transit -> planet radius

Examples
--------
Offline, no download (a fake star with a fake 2.5 R_earth planet):
    python scripts/run_baseline.py --synthetic

Real Kepler data (Kepler-10 b, P ~ 0.837 d, a ~1.5 R_earth rocky planet):
    python scripts/run_baseline.py --target "Kepler-10" --mission Kepler \
        --quarter 3 4 5 --rstar 1.065 --rstar-err 0.009 --max-period 10

Real TESS data (pi Mensae c, P ~ 6.27 d):
    python scripts/run_baseline.py --target "pi Men" --mission TESS \
        --sector 1 --rstar 1.10 --rstar-err 0.02 --exptime 120

Outputs (in --outdir): raw.png, periodogram.png, fold.png, result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transitpipe import io, detrend, search, fit, inject, plotting  # noqa: E402
from transitpipe.fit import trapezoid_model, batman_model  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", help='e.g. "Kepler-10", "TIC 261136679"')
    src.add_argument("--synthetic", action="store_true", help="use fake data")
    src.add_argument("--csv", help="local CSV with time, flux, flux_err columns")
    p.add_argument("--mission", default="Kepler", choices=["Kepler", "K2", "TESS"])
    p.add_argument("--quarter", type=int, nargs="*")
    p.add_argument("--sector", type=int, nargs="*")
    p.add_argument("--exptime", default=None, help='"long", "short", or seconds')
    p.add_argument("--rstar", type=float, help="stellar radius [R_sun]")
    p.add_argument("--rstar-err", type=float, default=None)
    p.add_argument("--method", default="median", choices=["median", "savgol", "biweight"])
    p.add_argument("--window", type=float, default=1.0, help="detrend window [days]")
    p.add_argument("--min-period", type=float, default=0.5)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--oversample", type=float, default=3.0)
    p.add_argument("--model", default="trapezoid", choices=["auto", "trapezoid", "batman"])
    p.add_argument("--mstar", type=float, default=None,
                   help="stellar mass [M_sun]; with --rstar gives a density prior for batman")
    p.add_argument("--mstar-err", type=float, default=0.0)
    p.add_argument("--outdir", default="results/baseline")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.outdir, exist_ok=True)
    truth = None

    # 1. get data ------------------------------------------------------------
    if a.synthetic:
        lc = inject.make_synthetic_lightcurve(seed=a.seed)
        lc, truth = inject.inject_transit(lc, period=4.2, rp_rearth=2.5, seed=a.seed)
        a.mstar = a.mstar or 1.0
        print(f"[data] synthetic star + injected planet: P = 4.2 d, Rp = 2.5 R_earth")
    elif a.csv:
        lc = io.from_csv(a.csv, target=os.path.basename(a.csv)).normalize()
    else:
        lc = io.get_lightcurve(a.target, mission=a.mission, exptime=a.exptime,
                               quarter=a.quarter, sector=a.sector)
    if a.rstar is not None:
        lc.meta["rstar"], lc.meta["rstar_err"] = a.rstar, a.rstar_err
    print(f"[data] {len(lc)} points, baseline {np.ptp(lc.time):.1f} d")

    # 2. detrend (pass 1) ----------------------------------------------------
    flat, trend = detrend.flatten(lc, method=a.method, window_days=a.window,
                                  return_trend=True)
    flat = detrend.sigma_clip(flat)
    print(f"[detrend] {a.method}, window {a.window} d, "
          f"noise {detrend.estimate_noise_ppm(flat):.0f} ppm")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plotting.plot_raw(lc, trend, ax=ax[0], title="Raw flux + trend")
    plotting.plot_raw(flat, ax=ax[1], title="Detrended")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "raw.png"), dpi=130)

    # 3. BLS search ----------------------------------------------------------
    res = search.bls(flat, min_period=a.min_period, max_period=a.max_period,
                     oversample=a.oversample)
    print("[BLS]", res.summary())
    ax = plotting.plot_periodogram(res, true_period=truth["period"] if truth else None)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(a.outdir, "periodogram.png"), dpi=130)

    # 4. re-detrend with transits masked, then fit ----------------------------
    flat2 = detrend.sigma_clip(detrend.flatten_masked(
        lc, res.period, res.t0, res.duration, method=a.method, window_days=a.window))
    fit_kwargs = {}
    if a.model == "batman" and a.mstar and lc.meta.get("rstar"):
        from transitpipe.utils import stellar_density
        rho, rho_err = stellar_density(lc.meta["rstar"], a.mstar,
                                       lc.meta.get("rstar_err") or 0.0, a.mstar_err)
        fit_kwargs = {"rho_star": rho, "rho_star_err": max(rho_err, 0.05 * rho)}
        print(f"[fit] stellar density prior: {rho:.2f} +/- {fit_kwargs['rho_star_err']:.2f} g/cm^3")
    f = fit.fit_single_transit(flat2, res, model=a.model, **fit_kwargs)
    print("[fit]", f.summary())

    if f.model_name == "batman":
        e = f.extra
        model = lambda t: batman_model(t, f.t0, f.period, f.rp_rs, e["a_rs"], e["b"], e["u"])  # noqa: E731
        print(f"[fit] a/Rs = {e['a_rs']:.2f}, b = {e['b']:.2f}, "
              f"implied stellar density = {e['rho_star_cgs']:.2f} g/cm^3 (Sun: 1.41)")
    else:
        model = lambda t: trapezoid_model(t, f.t0, f.period, f.depth, f.duration,  # noqa: E731
                                          f.extra["ingress"])
    ax = plotting.plot_fold(flat2, f.period, f.t0, model_func=model,
                            window_hours=3 * f.duration * 24)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(a.outdir, "fold.png"), dpi=130)

    out = {"bls": {k: getattr(res, k) for k in
                   ["period", "t0", "duration", "depth", "depth_err", "snr", "sde", "n_transits"]},
           "fit": {k: v for k, v in f.__dict__.items()},
           "truth": truth}
    with open(os.path.join(a.outdir, "result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    if truth:
        print(f"[check] true Rp = {truth['rp_rearth']:.2f}, fitted Rp = "
              f"{f.rp_rearth:.2f} +/- {f.rp_rearth_err:.2f} R_earth")
    print(f"[done] wrote figures + result.json to {a.outdir}/")


if __name__ == "__main__":
    main()
