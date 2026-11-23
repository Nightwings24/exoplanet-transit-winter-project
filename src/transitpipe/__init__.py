"""
transitpipe
===========

A small, teachable exoplanet transit-detection pipeline built for the
Astronomy Club Winter Project.

The whole pipeline is deliberately short and readable. Each module maps onto
one step a student performs by hand before they trust the automated version:

    io       -> get a light curve (download from MAST, or load a local file)
    detrend  -> remove slow stellar/instrumental trends, keep the transits
    search   -> run Box Least Squares (BLS) to find a periodic dip
    fit      -> fit a single transit and turn depth into a planet radius
    inject   -> inject fake transits and measure what fraction we recover

Typical baseline run
--------------------
>>> from transitpipe import io, detrend, search, fit
>>> lc = io.get_lightcurve("Kepler-10", mission="Kepler")
>>> flat = detrend.flatten(lc)                     # pass 1: find
>>> result = search.bls(flat)
>>> flat2 = detrend.flatten_masked(lc, result.period, result.t0,
...                                result.duration)  # pass 2: measure
>>> params = fit.fit_single_transit(flat2, result, rstar_rsun=1.065)
>>> print(params.rp_rearth)

Nothing here is magic. Read the source; it is meant to be read.
"""

from . import io, detrend, search, fit, inject, utils  # noqa: F401

# plotting needs matplotlib; import it explicitly: `from transitpipe import plotting`
__all__ = ["io", "detrend", "search", "fit", "inject", "utils", "plotting"]
__version__ = "0.1.0"
