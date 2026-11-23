"""
io.py
=====

Get a light curve into memory.

Two paths:
  * `get_lightcurve(...)` downloads real Kepler/TESS data from MAST using the
    `lightkurve` package (which wraps astroquery).
  * `LightCurve` is a tiny container (time, flux, flux_err) so the rest of the
    pipeline never has to care whether the data came from MAST, a CSV, or a
    simulation. Students who cannot install lightkurve can still do every other
    step on a local CSV or on the built-in synthetic generator in `inject`.

Design choice: we return plain NumPy arrays wrapped in a dataclass, not a
lightkurve object. It keeps the teaching code honest about what a light curve
actually *is* — three columns of numbers — and keeps the pipeline usable
offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LightCurve:
    """A light curve: time (days), normalised flux, and flux error.

    Flux is expected to be near 1.0 (normalised). `meta` carries anything we
    learn about the star (its radius, the target name, the mission).
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err, dtype=float)
        self._drop_nans()

    def _drop_nans(self):
        good = np.isfinite(self.time) & np.isfinite(self.flux)
        if self.flux_err is not None:
            good &= np.isfinite(self.flux_err)
        self.time = self.time[good]
        self.flux = self.flux[good]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[good]

    def normalize(self) -> "LightCurve":
        """Divide by the median so flux sits around 1.0."""
        med = np.nanmedian(self.flux)
        return LightCurve(self.time, self.flux / med,
                          None if self.flux_err is None else self.flux_err / med,
                          dict(self.meta))

    def copy(self) -> "LightCurve":
        return LightCurve(self.time.copy(), self.flux.copy(),
                          None if self.flux_err is None else self.flux_err.copy(),
                          dict(self.meta))

    def __len__(self):
        return len(self.time)


def get_lightcurve(target: str, mission: str = "Kepler",
                   author: str | None = None, exptime: str | None = None,
                   quarter=None, sector=None, flux_column: str = "pdcsap_flux",
                   stitch: bool = True) -> LightCurve:
    """Download a light curve from MAST via lightkurve and return a LightCurve.

    Parameters
    ----------
    target : str
        Name or catalogue ID, e.g. "Kepler-10", "KIC 11904151", "TIC 25155310".
    mission : str
        "Kepler", "K2", or "TESS". Chooses a sensible default author.
    author : str, optional
        Pipeline/product author (e.g. "Kepler", "SPOC"). Defaults by mission.
    exptime : str, optional
        "long" / "short" (Kepler) or a number of seconds (TESS, e.g. "120").
    quarter, sector :
        Restrict to specific Kepler quarters or TESS sectors.
    flux_column : str
        Which flux to use. "pdcsap_flux" is the systematics-corrected flux and
        is the right default for transit hunting.
    stitch : bool
        If multiple segments are found, normalise each and join them.

    Notes
    -----
    Requires `lightkurve`. Imported lazily so the rest of transitpipe works
    without it. PDCSAP flux is already corrected for most instrument
    systematics but can over-suppress real long-period signals; for a first
    project that trade-off is fine.
    """
    try:
        import lightkurve as lk
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "get_lightcurve needs the 'lightkurve' package. "
            "Install it with `pip install lightkurve`, or load a local file "
            "with transitpipe.io.from_csv()."
        ) from exc

    if author is None:
        author = {"Kepler": "Kepler", "K2": "K2", "TESS": "SPOC"}.get(mission)

    kwargs = {"author": author}
    if exptime is not None:
        kwargs["exptime"] = exptime
    if quarter is not None:
        kwargs["quarter"] = quarter
    if sector is not None:
        kwargs["sector"] = sector

    search = lk.search_lightcurve(target, mission=mission, **kwargs)
    if len(search) == 0:
        raise ValueError(f"No light curves found for {target!r} "
                         f"(mission={mission}, author={author}).")

    collection = search.download_all()

    lcs = []
    for lc in collection:
        lc = lc.select_flux(flux_column) if flux_column in lc.columns else lc
        lc = lc.remove_nans().normalize()
        lcs.append(lc)

    stitched = lk.LightCurveCollection(lcs).stitch() if stitch else lcs[0]

    time = np.asarray(stitched.time.value, dtype=float)
    flux = np.asarray(stitched.flux.value, dtype=float)
    try:
        flux_err = np.asarray(stitched.flux_err.value, dtype=float)
    except Exception:
        flux_err = None

    meta = {"target": target, "mission": mission, "author": author,
            "flux_column": flux_column}
    return LightCurve(time, flux, flux_err, meta)


def from_csv(path: str, time_col: str = "time", flux_col: str = "flux",
             err_col: str | None = "flux_err", **meta) -> LightCurve:
    """Load a light curve from a CSV with named columns (needs pandas)."""
    import pandas as pd

    df = pd.read_csv(path)
    err = df[err_col].to_numpy() if err_col and err_col in df.columns else None
    return LightCurve(df[time_col].to_numpy(), df[flux_col].to_numpy(), err, meta)


def to_csv(lc: LightCurve, path: str) -> None:
    """Save a light curve to CSV (time, flux, flux_err)."""
    import pandas as pd

    data = {"time": lc.time, "flux": lc.flux}
    if lc.flux_err is not None:
        data["flux_err"] = lc.flux_err
    pd.DataFrame(data).to_csv(path, index=False)
