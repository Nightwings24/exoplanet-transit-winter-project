"""
Tests for transitpipe. Run with:  pytest -q

These run offline (synthetic data only). They are also a teaching tool:
each test states one fact your pipeline must get right. When you change the
code, run the tests; if one breaks, you broke physics or the pipeline.
"""

import numpy as np
import pytest

from transitpipe import utils, detrend, search, fit, inject
from transitpipe.io import LightCurve


# --- physics -----------------------------------------------------------------

def test_depth_to_radius_ratio():
    assert utils.depth_to_radius_ratio(0.01) == pytest.approx(0.1)


def test_earth_sun_transit_depth():
    # Earth in front of the Sun: (1/109.1)^2 ~ 84 ppm
    depth = (1.0 / utils.R_SUN_IN_REARTH) ** 2
    assert depth * 1e6 == pytest.approx(84, abs=1)
    assert utils.depth_to_rearth(depth, 1.0) == pytest.approx(1.0)


def test_jupiter_sun_depth_is_about_one_percent():
    k = utils.R_JUP / utils.R_SUN
    assert k ** 2 == pytest.approx(0.0106, abs=5e-4)


def test_earth_transit_duration_about_13_hours():
    # Earth's central transit across the Sun lasts ~13 h
    dur_h = utils.expected_transit_duration(365.25, 1.0, 1.0) * 24
    assert dur_h == pytest.approx(13.0, abs=0.5)


def test_negative_depth_raises():
    with pytest.raises(ValueError):
        utils.depth_to_radius_ratio(-0.001)


def test_phase_fold_puts_transit_at_zero():
    t = np.array([10.0, 13.0, 16.0])
    assert np.allclose(utils.phase_fold(t, 3.0, 10.0), 0.0)


# --- detrending --------------------------------------------------------------

def test_flatten_removes_variability():
    lc = inject.make_synthetic_lightcurve(noise_ppm=100, var_amp=0.01, seed=3)
    flat = detrend.flatten(lc, window_days=0.75)
    # raw scatter is dominated by the 1% variability; flat by the 100 ppm noise
    assert np.std(lc.flux) > 3000e-6
    assert detrend.estimate_noise_ppm(flat) < 150


def test_sigma_clip_keeps_transits():
    lc = inject.make_synthetic_lightcurve(noise_ppm=200, var_amp=0.0, seed=4)
    inj, truth = inject.inject_transit(lc, 3.0, depth=0.005, seed=4)
    clipped = detrend.sigma_clip(inj, sigma=3.0)       # upper_only default
    m = detrend.transit_mask(clipped.time, 3.0, truth["t0"], truth["duration"], 0.5)
    assert m.sum() > 0, "upper-only clipping must not remove in-transit points"


def test_masked_detrend_preserves_depth():
    """Two-pass detrending recovers the true depth on an active star."""
    lc = inject.make_synthetic_lightcurve(noise_ppm=1, var_amp=0.003, seed=1)
    inj, truth = inject.inject_transit(lc, 3.7, rp_rearth=3.0, seed=2)
    core = detrend.transit_mask(lc.time, 3.7, truth["t0"], truth["duration"], 0.6)
    naive = detrend.flatten(inj)
    masked = detrend.flatten_masked(inj, 3.7, truth["t0"], truth["duration"])
    d_naive = 1 - naive.flux[core].mean()
    d_masked = 1 - masked.flux[core].mean()
    assert d_masked == pytest.approx(truth["depth"], rel=0.03)
    assert d_naive < d_masked   # the naive pass is biased shallow


# --- search ------------------------------------------------------------------

@pytest.fixture(scope="module")
def planet_lc():
    lc = inject.make_synthetic_lightcurve(noise_ppm=300, seed=11)
    inj, truth = inject.inject_transit(lc, period=2.9, rp_rearth=4.0, seed=12)
    return inj, truth


def test_bls_recovers_injected_period(planet_lc):
    inj, truth = planet_lc
    flat = detrend.sigma_clip(detrend.flatten(inj))
    r = search.bls(flat)
    assert r.period == pytest.approx(truth["period"], rel=0.005)
    assert r.sde > 7


def test_is_match_aliases():
    assert search.is_match(3.0, 3.0) == (True, "exact")
    assert search.is_match(6.0, 3.0) == (False, "2:1")
    assert search.is_match(1.5, 3.0) == (False, "1:2")
    assert search.is_match(4.1, 3.0)[1] == "none"


# --- fitting -----------------------------------------------------------------

def test_trapezoid_fit_recovers_radius(planet_lc):
    inj, truth = planet_lc
    flat = detrend.sigma_clip(detrend.flatten(inj))
    r = search.bls(flat)
    flat2 = detrend.sigma_clip(detrend.flatten_masked(inj, r.period, r.t0, r.duration))
    f = fit.fit_single_transit(flat2, r, model="trapezoid", rstar_rsun=1.0, rstar_err=0.05)
    assert f.rp_rearth == pytest.approx(truth["rp_rearth"], abs=3 * f.rp_rearth_err + 0.2)
    assert np.isfinite(f.rp_rearth_err) and f.rp_rearth_err > 0


# --- injection-recovery ------------------------------------------------------

def test_big_planets_found_tiny_ones_missed():
    lc = inject.make_synthetic_lightcurve(noise_ppm=400, seed=21)
    pe = np.array([1.0, 3.0])
    re = np.array([0.3, 0.5, 8.0, 10.0])
    df = inject.injection_recovery(lc, pe, re, n_per_cell=4,
                                   bls_kwargs={"oversample": 1.0, "max_period": 5.0},
                                   progress=False)
    frac, n = inject.completeness_map(df, pe, re)
    assert n.sum() == 12
    assert frac[0, 2] >= 0.75      # 8-10 R_earth: easy
    assert frac[0, 0] <= 0.25      # 0.3-0.5 R_earth at 400 ppm: hopeless
