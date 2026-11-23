"""
fit.py
======

Fit a transit model and turn it into a planet radius.

BLS gives a *box*: good enough to find the planet, too crude to measure it.
Here we fit a better shape, starting from the BLS answer.

Two models
----------
1. Trapezoid (always available, pure NumPy). Four numbers: mid-time t0,
   depth, total duration T14, and ingress duration tau. Captures the sloped
   ingress/egress a box ignores. Depth ~ (Rp/Rs)^2, but biased slightly HIGH
   by limb darkening, so Rp/Rs from a trapezoid is slightly too large.

2. batman (optional, `pip install batman-package`). The physical Mandel & Agol
   (2002) model with quadratic limb darkening, as implemented by Kreidberg
   (2015). Fits Rp/Rs directly, plus a/Rs and impact parameter b. This is the
   model used in real papers.

From radius ratio to radius
---------------------------
    Rp = (Rp/Rs) * Rs
The stellar radius Rs comes from a catalogue (TIC for TESS, the Kepler DR25
stellar table / Gaia-based radii for Kepler, or the NASA Exoplanet Archive).
Your planet radius can never be more precise than your stellar radius:

    (sigma_Rp / Rp)^2 = (sigma_k / k)^2 + (sigma_Rs / Rs)^2,  k = Rp/Rs
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .io import LightCurve
from .utils import R_SUN_IN_REARTH


# --- models ------------------------------------------------------------------

def trapezoid_model(time: np.ndarray, t0: float, period: float, depth: float,
                    duration: float, ingress: float) -> np.ndarray:
    """Periodic trapezoid transit, normalised to 1 out of transit.

    duration = T14 (first to fourth contact), ingress = T12 (first to second).
    """
    ingress = np.clip(ingress, 1e-6, 0.5 * duration)
    dt = np.abs(((time - t0 + 0.5 * period) % period) - 0.5 * period)
    half = 0.5 * duration
    model = np.ones_like(time, dtype=float)
    flat = dt <= half - ingress
    ramp = (dt > half - ingress) & (dt < half)
    model[flat] -= depth
    model[ramp] -= depth * (half - dt[ramp]) / ingress
    return model


def batman_model(time: np.ndarray, t0: float, period: float, rp_rs: float,
                 a_rs: float, b: float, u=(0.4, 0.25),
                 exptime_days: float | None = None,
                 supersample: int = 7) -> np.ndarray:
    """Mandel-Agol transit via batman, circular orbit, quadratic limb darkening.

    exptime_days: set for long-cadence data (Kepler LC = 0.0204 d, TESS FFI
    200 s / 600 s / 1800 s) so the model is integrated over each exposure.
    Without it, long-cadence ingress/egress looks too sharp.
    """
    import batman

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    cos_i = np.clip(b / a_rs, 0.0, 1.0)
    params.inc = float(np.degrees(np.arccos(cos_i)))
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = list(u)
    if exptime_days:
        m = batman.TransitModel(params, time, supersample_factor=supersample,
                                exp_time=exptime_days)
    else:
        m = batman.TransitModel(params, time)
    return m.light_curve(params)


# --- results -----------------------------------------------------------------

@dataclass
class FitResult:
    model_name: str
    period: float
    t0: float
    t0_err: float
    depth: float
    depth_err: float
    duration: float
    duration_err: float
    rp_rs: float
    rp_rs_err: float
    rstar_rsun: float | None = None
    rstar_err: float | None = None
    rp_rearth: float | None = None
    rp_rearth_err: float | None = None
    extra: dict = field(default_factory=dict)
    residual_rms_ppm: float = np.nan

    def summary(self) -> str:
        s = (f"[{self.model_name}] P = {self.period:.5f} d | "
             f"t0 = {self.t0:.5f} +/- {self.t0_err:.5f} | "
             f"depth = {self.depth * 1e6:.0f} +/- {self.depth_err * 1e6:.0f} ppm | "
             + (f"T14 = {self.duration * 24:.2f} +/- {self.duration_err * 24:.2f} h | "
                if np.isfinite(self.duration_err) else
                f"T14 = {self.duration * 24:.2f} h (derived) | ")
             + f"Rp/Rs = {self.rp_rs:.4f} +/- {self.rp_rs_err:.4f}")
        if self.rp_rearth is not None:
            s += f" | Rp = {self.rp_rearth:.2f} +/- {self.rp_rearth_err:.2f} R_earth"
        return s


def _covariance(res) -> np.ndarray:
    """Parameter covariance from a scipy least_squares result.

    cov = s^2 (J^T J)^-1 with s^2 the reduced chi-square. This assumes the
    errors are Gaussian and the problem is close to linear near the best fit.
    For honest error bars, run an MCMC (see the 'Going Further' box in Week 3).
    """
    J = res.jac
    dof = max(len(res.fun) - J.shape[1], 1)
    s2 = 2 * res.cost / dof
    # pinv, not inv: a parameter pinned at a bound (e.g. ingress -> 0) makes
    # J^T J singular. pinv then reports ~0 error for that parameter instead of
    # NaN-ing every error bar. A zero error bar means "hit a bound", not
    # "perfectly measured" - check FitResult values against your bounds.
    return np.linalg.pinv(J.T @ J) * s2


def _radius(rp_rs, rp_rs_err, rstar, rstar_err):
    if rstar is None:
        return None, None
    rp = rp_rs * rstar * R_SUN_IN_REARTH
    frac = np.hypot(rp_rs_err / rp_rs if rp_rs > 0 else np.nan,
                    (rstar_err or 0.0) / rstar)
    return float(rp), float(rp * frac)


def _window(lc: LightCurve, period: float, t0: float, duration: float,
            n_durations: float = 3.0):
    """Keep only data within n_durations * duration of each transit (faster)."""
    dt = np.abs(((lc.time - t0 + 0.5 * period) % period) - 0.5 * period)
    m = dt < n_durations * duration
    err = lc.flux_err[m] if lc.flux_err is not None else None
    return lc.time[m], lc.flux[m], err


# --- fitting -----------------------------------------------------------------

def fit_trapezoid(lc: LightCurve, period: float, t0: float, duration: float,
                  depth: float, rstar_rsun: float | None = None,
                  rstar_err: float | None = None,
                  fit_period: bool = False) -> FitResult:
    """Least-squares trapezoid fit seeded from BLS (or by hand)."""
    t, f, e = _window(lc, period, t0, duration)
    if e is None or not np.all(e > 0):
        e = np.full_like(f, 1.4826 * np.median(np.abs(f - np.median(f))))

    if fit_period:
        p0 = [t0, depth, duration, 0.15 * duration, period]
        lo = [t0 - 0.5 * duration, 0, 0.2 * duration, 1e-4, period * 0.999]
        hi = [t0 + 0.5 * duration, 1, 3.0 * duration, 1.5 * duration, period * 1.001]
    else:
        p0 = [t0, depth, duration, 0.15 * duration]
        lo = [t0 - 0.5 * duration, 0, 0.2 * duration, 1e-4]
        hi = [t0 + 0.5 * duration, 1, 3.0 * duration, 1.5 * duration]
    p0 = np.clip(p0, np.array(lo) + 1e-12, np.array(hi) - 1e-12)

    def resid(p):
        P = p[4] if fit_period else period
        return (f - trapezoid_model(t, p[0], P, p[1], p[2], p[3])) / e

    res = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    cov = _covariance(res)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))

    P = res.x[4] if fit_period else period
    d, d_err = res.x[1], err[1]
    k = np.sqrt(d)
    k_err = 0.5 * d_err / k if k > 0 else np.nan
    rp, rp_err = _radius(k, k_err, rstar_rsun, rstar_err)
    model = trapezoid_model(t, res.x[0], P, d, res.x[2], res.x[3])

    return FitResult("trapezoid", float(P), float(res.x[0]), float(err[0]),
                     float(d), float(d_err), float(res.x[2]), float(err[2]),
                     float(k), float(k_err), rstar_rsun, rstar_err, rp, rp_err,
                     extra={"ingress": float(res.x[3]),
                            "period_err": float(err[4]) if fit_period else 0.0},
                     residual_rms_ppm=float(np.std(f - model) * 1e6))


def fit_batman(lc: LightCurve, period: float, t0: float, duration: float,
               depth: float, rstar_rsun: float | None = None,
               rstar_err: float | None = None, u=(0.4, 0.25),
               exptime_days: float | None = None,
               rho_star: float | None = None, rho_star_err: float | None = None,
               fix_b: float | None = None) -> FitResult:
    """Physical Mandel-Agol fit (needs batman). Free: t0, Rp/Rs, a/Rs, b.

    Limb-darkening coefficients `u` are held fixed. Look them up for your star
    (Claret 2011 for Kepler, Claret 2017 for TESS) instead of using the
    Sun-like default. Fitting them freely needs very high SNR.

    The b / (a/Rs) / (Rp/Rs) degeneracy
    -----------------------------------
    At low SNR a short shallow transit can be a small planet crossing the
    middle (b ~ 0) or a bigger planet grazing the edge (b ~ 1). The data
    cannot tell them apart, so the error bars on Rp/Rs explode. Two standard
    fixes, both supported:
      * rho_star, rho_star_err : a prior on the stellar density (g/cm^3) from
        the catalogue Rs and Ms. Since a/Rs is set by rho_star and P, this
        pins a/Rs and therefore b. This is what real papers do.
      * fix_b : hold the impact parameter fixed (e.g. 0). Quick, but it
        hides the uncertainty instead of modelling it.
    """
    t, f, e = _window(lc, period, t0, duration)
    if e is None or not np.all(e > 0):
        e = np.full_like(f, 1.4826 * np.median(np.abs(f - np.median(f))))

    k0 = np.sqrt(max(depth, 1e-6))
    # Duration -> a/Rs for b = 0:  T14 ~ (P/pi) (1 + k) / a_rs
    a0 = max(period * (1 + k0) / (np.pi * duration), 1.5)
    G = 6.674e-8                      # cgs
    P_s = period * 86400.0

    def rho_from_a(a_rs):             # Seager & Mallen-Ornelas (2003)
        return 3 * np.pi * a_rs ** 3 / (G * P_s ** 2)

    if rho_star is not None:          # start a/Rs where the prior says
        a0 = (rho_star * G * P_s ** 2 / (3 * np.pi)) ** (1 / 3)
    free_b = fix_b is None
    p0 = [t0, k0, a0] + ([0.3] if free_b else [])
    lo = [t0 - 0.5 * duration, 1e-3, 1.1] + ([0.0] if free_b else [])
    hi = [t0 + 0.5 * duration, 0.5, 300.0] + ([1.0] if free_b else [])
    lo, hi = np.array(lo), np.array(hi)
    p0 = np.clip(np.array(p0), lo + 1e-9, hi - 1e-9)

    def unpack(p):
        return p[0], p[1], p[2], (p[3] if free_b else fix_b)

    def resid(p):
        t0_, k_, a_, b_ = unpack(p)
        r = (f - batman_model(t, t0_, period, k_, a_, b_, u, exptime_days)) / e
        if rho_star is not None:
            r = np.append(r, (rho_from_a(a_) - rho_star) / rho_star_err)
        return r

    if rho_star is not None and not rho_star_err:
        raise ValueError("rho_star prior needs rho_star_err > 0")
    res = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    cov = _covariance(res)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    if not free_b:
        err = np.append(err, 0.0)
    t0f, k, a_rs, b = unpack(res.x)

    model = batman_model(t, t0f, period, k, a_rs, b, u, exptime_days)
    depth_obs = float(1 - model.min())
    # Total duration for a circular orbit (Winn 2010, eq. 14)
    arg = np.sqrt(max((1 + k) ** 2 - b ** 2, 0)) / (a_rs * np.sqrt(1 - (b / a_rs) ** 2))
    t14 = float(period / np.pi * np.arcsin(np.clip(arg, 0, 1)))
    rp, rp_err = _radius(k, err[1], rstar_rsun, rstar_err)

    # Stellar density implied by the fit: a free sanity check against the
    # catalogue star (or against the prior, if you used one). In g/cm^3.
    rho_fit = rho_from_a(a_rs)

    return FitResult("batman", float(period), float(t0f), float(err[0]),
                     depth_obs, float(2 * k * err[1]), t14, np.nan,
                     float(k), float(err[1]), rstar_rsun, rstar_err, rp, rp_err,
                     extra={"a_rs": float(a_rs), "a_rs_err": float(err[2]),
                            "b": float(b), "b_err": float(err[3]),
                            "u": list(u), "rho_star_cgs": float(rho_fit),
                            "rho_prior": rho_star, "fix_b": fix_b},
                     residual_rms_ppm=float(np.std(f - model) * 1e6))


def fit_single_transit(lc: LightCurve, bls_result, rstar_rsun: float | None = None,
                       rstar_err: float | None = None, model: str = "trapezoid",
                       **kwargs) -> FitResult:
    """Fit the planet found by BLS. model = 'trapezoid', 'batman', or 'auto'.

    Default is the trapezoid: robust and dependency-free (Week 3 core).
    'batman' is the physical model (Week 3 upgrade); pass rho_star and
    rho_star_err (see utils.stellar_density) to tame the b degeneracy.
    'auto' uses batman if it is installed, otherwise the trapezoid.
    Stellar radius defaults to lc.meta['rstar'] if present.
    """
    if rstar_rsun is None:
        rstar_rsun = lc.meta.get("rstar")
        rstar_err = rstar_err if rstar_err is not None else lc.meta.get("rstar_err")
    args = (lc, bls_result.period, bls_result.t0, bls_result.duration,
            max(bls_result.depth, 1e-6), rstar_rsun, rstar_err)
    if model == "auto":
        try:
            import batman  # noqa: F401
            model = "batman"
        except Exception:
            model = "trapezoid"
    if model == "batman":
        return fit_batman(*args, **kwargs)
    if model == "trapezoid":
        return fit_trapezoid(*args, **kwargs)
    raise ValueError("model must be 'trapezoid', 'batman', or 'auto'")


def bootstrap_depth(lc: LightCurve, fit: FitResult, n_boot: int = 200,
                    seed: int | None = 0) -> np.ndarray:
    """Residual bootstrap of the trapezoid depth: a no-math error bar check.

    Refit `n_boot` fake data sets (best model + resampled residuals). The
    spread of depths is an error bar you can compare with the covariance one.
    """
    rng = np.random.default_rng(seed)
    ingress = fit.extra.get("ingress", 0.15 * fit.duration)
    t, f, e = _window(lc, fit.period, fit.t0, fit.duration)
    best = trapezoid_model(t, fit.t0, fit.period, fit.depth, fit.duration, ingress)
    resid = f - best
    depths = np.empty(n_boot)
    for i in range(n_boot):
        fake = LightCurve(t, best + rng.choice(resid, size=len(resid), replace=True))
        r = fit_trapezoid(fake, fit.period, fit.t0, fit.duration, fit.depth)
        depths[i] = r.depth
    return depths
