"""
inject.py
=========

Injection-recovery: measure how good YOUR pipeline is.

The question
------------
You ran your pipeline on a star and found nothing. Does that mean the star has
no planets? Or that your pipeline cannot see them? You cannot tell from one
null result. So you test the pipeline on planets whose answer you KNOW:

  1. Take a real light curve (raw, BEFORE detrending).
  2. Multiply in a fake transit of known period P and radius Rp.
  3. Run the *exact same* pipeline (detrend -> BLS) blind.
  4. Ask: did it find a signal at P (and the right phase)?
  5. Repeat thousands of times over a grid of (P, Rp).

The recovered fraction in each cell of the grid is your pipeline's
**detection efficiency** (a.k.a. completeness). This is how the Kepler team
measured DR25 completeness (Christiansen et al. 2013, 2015, 2016, 2020) and how
every planet occurrence-rate paper corrects for planets that were missed.

Inject BEFORE detrending. Detrending is one of the main ways real pipelines
lose planets (the smoother eats long transits). Injecting into already-flat
data hides exactly the effect you want to measure.
"""

from __future__ import annotations

import numpy as np

from .io import LightCurve
from .utils import R_SUN_IN_REARTH, expected_transit_duration
from .fit import trapezoid_model
from . import detrend as _detrend
from . import search as _search


# --- synthetic data ----------------------------------------------------------

def make_synthetic_lightcurve(baseline_days: float = 27.4, cadence_min: float = 30.0,
                              noise_ppm: float = 300.0, var_amp: float = 0.003,
                              rot_period: float = 6.3, gaps=((13.4, 14.4),),
                              seed: int | None = 42, **meta) -> LightCurve:
    """A fake TESS-sector-like light curve: spotted star + white noise + gaps.

    No planet in it (add one with inject_transit). Use this when you are
    offline, when MAST is slow, or to test code where you need to know the
    right answer.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, baseline_days, cadence_min / 1440.0)
    for a, b in gaps:
        t = t[(t < a) | (t > b)]
    phase1 = rng.uniform(0, 2 * np.pi)
    phase2 = rng.uniform(0, 2 * np.pi)
    var = (var_amp * np.sin(2 * np.pi * t / rot_period + phase1)
           + 0.4 * var_amp * np.sin(4 * np.pi * t / rot_period + phase2)
           + 0.5 * var_amp * (t / baseline_days - 0.5))        # slow drift
    flux = (1 + var) * (1 + rng.normal(0, noise_ppm * 1e-6, len(t)))
    err = np.full_like(t, noise_ppm * 1e-6)
    meta.setdefault("target", "synthetic")
    meta.setdefault("rstar", 1.0)
    meta.setdefault("rstar_err", 0.05)
    meta.setdefault("mstar", 1.0)
    return LightCurve(t, flux, err, meta)


# --- injection ---------------------------------------------------------------

def transit_signal(time, period, t0, rp_rs, rstar=1.0, mstar=1.0,
                   model="trapezoid", u=(0.4, 0.25)):
    """Model flux (1 = no transit) for a planet on a circular, b=0 orbit."""
    duration = expected_transit_duration(period, rstar, mstar)
    if model == "batman":
        from .fit import batman_model
        G, M_SUN, R_SUN = 6.674e-11, 1.989e30, 6.957e8
        a = (G * mstar * M_SUN * (period * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
        return batman_model(time, t0, period, rp_rs, a / (rstar * R_SUN), 0.0, u)
    ingress = duration * rp_rs / (1 + rp_rs)          # T12/T14 ~ k/(1+k), b=0
    return trapezoid_model(time, t0, period, rp_rs ** 2, duration, ingress)


def inject_transit(lc: LightCurve, period: float, t0: float | None = None,
                   rp_rearth: float | None = None, depth: float | None = None,
                   rstar: float | None = None, mstar: float | None = None,
                   model: str = "trapezoid", seed: int | None = None):
    """Multiply a fake transit into a COPY of lc. Give rp_rearth OR depth.

    Returns (new_lc, truth_dict). t0 defaults to a random phase.
    """
    rstar = rstar if rstar is not None else lc.meta.get("rstar", 1.0)
    mstar = mstar if mstar is not None else lc.meta.get("mstar", 1.0)
    if (rp_rearth is None) == (depth is None):
        raise ValueError("give exactly one of rp_rearth or depth")
    rp_rs = (rp_rearth / (rstar * R_SUN_IN_REARTH)) if rp_rearth is not None \
        else np.sqrt(depth)
    if t0 is None:
        rng = np.random.default_rng(seed)
        t0 = lc.time.min() + rng.uniform(0, period)
    sig = transit_signal(lc.time, period, t0, rp_rs, rstar, mstar, model)
    out = lc.copy()
    out.flux = lc.flux * sig
    truth = {"period": period, "t0": t0, "rp_rs": rp_rs, "depth": rp_rs ** 2,
             "rp_rearth": rp_rs * rstar * R_SUN_IN_REARTH,
             "duration": expected_transit_duration(period, rstar, mstar),
             "n_in_transit": int(np.sum(sig < 1))}
    return out, truth


def expected_snr(lc_flat_noise_ppm: float, depth: float, n_in_transit: int) -> float:
    """Back-of-envelope SNR = depth / sigma * sqrt(N points in transit)."""
    return float(depth / (lc_flat_noise_ppm * 1e-6) * np.sqrt(max(n_in_transit, 0)))


# --- the experiment ----------------------------------------------------------

def _one_trial(args):
    """Inject one planet, run the pipeline blind, report what happened.

    Kept as a top-level function so it can run in parallel processes.
    """
    (time, flux, err, meta, period, rp_rearth, seed,
     detrend_kwargs, bls_kwargs, rel_tol, sde_threshold, noise_ppm) = args
    lc = LightCurve(time, flux, err, dict(meta))
    inj, truth = inject_transit(lc, period, rp_rearth=rp_rearth, seed=seed)
    flat = _detrend.flatten(inj, **detrend_kwargs)
    flat = _detrend.sigma_clip(flat)
    r = _search.bls(flat, **bls_kwargs)

    match, kind = _search.is_match(r.period, period, rel_tol)
    # Phase check: the found mid-time must line up with an injected transit.
    if match:
        dphase = abs(((r.t0 - truth["t0"] + 0.5 * period) % period) - 0.5 * period)
        match = dphase < max(truth["duration"], r.duration)
        if not match:
            kind = "wrong_phase"
    recovered = bool(match and r.sde >= sde_threshold)
    if match and not recovered:
        kind = "below_threshold"

    return {"period_inj": period, "rp_inj": truth["rp_rearth"],
            "depth_inj": truth["depth"], "t0_inj": truth["t0"],
            "duration_inj": truth["duration"],
            "n_in_transit": truth["n_in_transit"],
            "snr_expected": expected_snr(noise_ppm, truth["depth"],
                                         truth["n_in_transit"]),
            "period_found": r.period, "t0_found": r.t0, "depth_found": r.depth,
            "sde": r.sde, "snr": r.snr, "match_kind": kind,
            "recovered": recovered}


def injection_recovery(lc_raw: LightCurve, period_edges, radius_edges,
                       n_per_cell: int = 10, detrend_kwargs: dict | None = None,
                       bls_kwargs: dict | None = None, rel_tol: float = 0.01,
                       sde_threshold: float = 7.0, seed: int = 0,
                       n_jobs: int = 1, progress: bool = True):
    """Run a full injection-recovery grid. Returns a pandas DataFrame.

    Parameters
    ----------
    lc_raw : LightCurve
        The RAW (not detrended) light curve. If it has a known planet, remove
        or mask that planet first, or the pipeline will keep finding the real
        one instead of your fake one (try it once: it is a good lesson).
    period_edges, radius_edges : arrays
        Bin edges (days, Earth radii). Each injection draws P and Rp
        log-uniformly *within* its cell so the grid is not artificially regular.
    n_per_cell : int
        Injections per cell. Recovery fraction error ~ sqrt(f(1-f)/n).
    detrend_kwargs, bls_kwargs :
        Passed straight to detrend.flatten and search.bls. Keep these
        IDENTICAL to your real search, or the completeness you measure is for a
        different pipeline.
    n_jobs : int
        >1 runs trials in parallel processes.
    """
    import pandas as pd

    detrend_kwargs = detrend_kwargs or {"method": "median", "window_days": 1.0}
    bls_kwargs = bls_kwargs or {}
    rng = np.random.default_rng(seed)
    noise_ppm = _detrend.estimate_noise_ppm(_detrend.flatten(lc_raw, **detrend_kwargs))

    jobs = []
    for i in range(len(period_edges) - 1):
        for j in range(len(radius_edges) - 1):
            for _ in range(n_per_cell):
                P = float(np.exp(rng.uniform(np.log(period_edges[i]),
                                             np.log(period_edges[i + 1]))))
                R = float(np.exp(rng.uniform(np.log(radius_edges[j]),
                                             np.log(radius_edges[j + 1]))))
                jobs.append((lc_raw.time, lc_raw.flux, lc_raw.flux_err,
                             lc_raw.meta, P, R, int(rng.integers(1 << 31)),
                             detrend_kwargs, bls_kwargs, rel_tol, sde_threshold,
                             noise_ppm))

    rows = []
    if n_jobs == 1:
        it = jobs
        if progress:
            try:
                from tqdm.auto import tqdm
                it = tqdm(jobs, desc="injection-recovery")
            except ImportError:
                pass
        for a in it:
            rows.append(_one_trial(a))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_one_trial, jobs, chunksize=4))

    df = pd.DataFrame(rows)
    df.attrs.update({"period_edges": list(period_edges),
                     "radius_edges": list(radius_edges),
                     "sde_threshold": sde_threshold, "noise_ppm": noise_ppm})
    return df


def completeness_map(df, period_edges, radius_edges):
    """Bin results into a 2-D recovered fraction. Returns (fraction, counts)."""
    P = df["period_inj"].to_numpy()
    R = df["rp_inj"].to_numpy()
    rec = df["recovered"].to_numpy().astype(float)
    n, _, _ = np.histogram2d(P, R, bins=[period_edges, radius_edges])
    k, _, _ = np.histogram2d(P, R, bins=[period_edges, radius_edges], weights=rec)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = k / n
    return frac, n


def plot_completeness(df, period_edges, radius_edges, ax=None, annotate=True,
                      title="Pipeline detection efficiency"):
    """Heat map of recovered fraction in (period, radius). Log axes."""
    import matplotlib.pyplot as plt

    frac, n = completeness_map(df, period_edges, radius_edges)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(period_edges, radius_edges, frac.T, vmin=0, vmax=1,
                         cmap="viridis", shading="flat")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_xticks([round(v, 2) for v in period_edges])
    ax.set_yticks([round(v, 2) for v in radius_edges])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Injected period [days]")
    ax.set_ylabel(r"Injected radius [$R_\oplus$]")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="Fraction recovered")
    if annotate:
        for i in range(len(period_edges) - 1):
            for j in range(len(radius_edges) - 1):
                if n[i, j] > 0:
                    x = np.sqrt(period_edges[i] * period_edges[i + 1])
                    y = np.sqrt(radius_edges[j] * radius_edges[j + 1])
                    v = frac[i, j]
                    ax.text(x, y, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v < 0.6 else "black")
    return ax


def fit_efficiency_curve(df, snr_col="snr_expected"):
    """Fit recovered-vs-SNR with a logistic curve, like Kepler's 1-D model.

        efficiency(S) = 1 / (1 + exp(-(S - S50) / width))

    Returns (S50, width). S50 is the SNR at which you find half the planets:
    a single, honest number that summarises your pipeline. (Christiansen et
    al. use a gamma CDF; a logistic is easier to explain and close enough.)
    """
    from scipy.optimize import curve_fit

    x = df[snr_col].to_numpy()
    y = df["recovered"].to_numpy().astype(float)
    good = np.isfinite(x)

    def logistic(s, s50, w):
        return 1.0 / (1.0 + np.exp(-(s - s50) / w))

    p, _ = curve_fit(logistic, x[good], y[good], p0=[np.median(x[good]), 2.0],
                     bounds=([0, 0.05], [np.inf, np.inf]), maxfev=10_000)
    return float(p[0]), float(p[1]), logistic
