# Exoplanet Transit Detection: Winter Project

**Astronomy Club · 4-week winter project · freshmen to seniors · no experience required**

Find a planet in NASA data, measure it, and then measure how good *you* are at finding planets.

Over four weeks you will build your own pipeline. It downloads Kepler/TESS light curves from
[MAST](https://archive.stsci.edu/), detrends them, searches them with **Box Least Squares**, and fits a
transit to measure the **planet's radius**. Then comes the twist: you inject thousands of fake planets
of known size and period, and measure what fraction your pipeline recovers. That fraction is your
pipeline's **detection efficiency** (completeness), the same quantity the Kepler team measured to
calibrate its final catalogue.

📘 **Start with the project guide:** [`docs/exoplanet_project_guide.pdf`](docs/exoplanet_project_guide.pdf)
(week-by-week plan, the science, setup, rubric, target list, and an annotated resource list).
Links only: [`docs/RESOURCES.md`](docs/RESOURCES.md).

---

## The four weeks

| Week | Theme | You finish with | Notebook |
|---|---|---|---|
| 1 | **Foundations** | Python basics, Git, transit physics, a real Kepler light curve, your own target | [00](notebooks/00_python_crash_course.ipynb) · [01](notebooks/01_download_and_plot.ipynb) |
| 2 | **Find it** | Detrending + BLS; Kepler-10 b recovered blind; your target searched | [02](notebooks/02_detrend_and_bls.ipynb) |
| 3 | **Measure it** | Transit fit, error bars, planet radius compared with literature | [03](notebooks/03_fit_transit_radius.ipynb) |
| 4 | **Know your limits** | Injection–recovery completeness map, a controlled experiment, report + talk | [04](notebooks/04_injection_recovery.ipynb) |

Core track: **10–15 h/week** (two club sessions + self-paced). Every week has a *Going further*
section for anyone who can work full-time.

### Open in Colab (zero install)

| | |
|---|---|
| 00 Python crash course | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nightwings24/exoplanet-transit-winter-project/blob/main/notebooks/00_python_crash_course.ipynb) |
| 01 Download & plot | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nightwings24/exoplanet-transit-winter-project/blob/main/notebooks/01_download_and_plot.ipynb) |
| 02 Detrend & BLS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nightwings24/exoplanet-transit-winter-project/blob/main/notebooks/02_detrend_and_bls.ipynb) |
| 03 Fit & radius | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nightwings24/exoplanet-transit-winter-project/blob/main/notebooks/03_fit_transit_radius.ipynb) |
| 04 Injection–recovery | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nightwings24/exoplanet-transit-winter-project/blob/main/notebooks/04_injection_recovery.ipynb) |

---

## Quick start (local)

```bash
git clone https://github.com/<you>/exoplanet-transit-winter-project.git
cd exoplanet-transit-winter-project

# with conda
conda env create -f environment.yml && conda activate exotransit
# or with plain pip (Python >= 3.9)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

pip install -e .          # makes `import transitpipe` work everywhere
pytest -q                 # 13 tests, offline, ~10 s
```

Run the whole pipeline from the command line:

```bash
# offline demo: fake star + fake 2.5 R_earth planet
python scripts/run_baseline.py --synthetic

# real data: Kepler-10 b
python scripts/run_baseline.py --target "Kepler-10" --mission Kepler --quarter 3 4 \
    --exptime long --rstar 1.065 --rstar-err 0.009 --max-period 10 --window 0.5

# real data: pi Mensae c (TESS), physical batman model with a stellar-density prior
python scripts/run_baseline.py --target "pi Men" --mission TESS --sector 1 --exptime 120 \
    --rstar 1.10 --rstar-err 0.02 --mstar 1.09 --mstar-err 0.05 --model batman

# the twist: injection-recovery
python scripts/run_injection_recovery.py --synthetic --fast
```

Each script writes its figures and a `result.json` or `injections.csv` to `results/`. That folder is git-ignored.

## What is in here

```
src/transitpipe/      the teaching package (short, readable, tested)
  io.py               LightCurve container; MAST download via lightkurve; CSV
  detrend.py          median / Savitzky-Golay / biweight; gaps; sigma clip; masked 2nd pass
  search.py           Box Least Squares (astropy), period grid, SDE/SNR, multi-planet, aliases
  fit.py              trapezoid + batman transit fits, error bars, density prior, bootstrap
  inject.py           synthetic light curves, injection-recovery engine, completeness maps
  plotting.py         raw/trend, periodogram, folded transit, efficiency curve
  utils.py            constants, depth -> radius, durations, stellar density, folding
notebooks/            one notebook per milestone, with exercises and checkpoints
scripts/              command-line versions of the baseline and the injection-recovery
tests/                physics facts + an end-to-end "inject a planet, find it" test
docs/                 project guide (PDF + LaTeX), references.bib, RESOURCES.md, report template
```

## Reference results

These were produced with this code and serve as the answer key for the shared target:

| Target | Data / model | Found P | Measured Rp | Published |
|---|---|---|---|---|
| Kepler-10 b | Kepler Q3–4, trapezoid | 0.83766 d | 1.49 ± 0.02 R⊕ | 1.47 R⊕ |
| Kepler-10 b | same, batman + ρ★ prior | 0.83766 d | 1.45 ± 0.02 R⊕ (b = 0.36) | 1.47 R⊕ |
| π Men c | TESS S1, batman + ρ★ prior | 6.2672 d | 1.92 ± 0.04 R⊕ (b = 0.57) | 2.04 R⊕ |

## Building the guide PDF

```bash
cd docs
latexmk -pdf exoplanet_project_guide.tex
# or: pdflatex, bibtex, pdflatex, pdflatex
```

The guide compiles with a basic TeX Live install and on Overleaf. It uses Palatino-style fonts when
`newpx` is installed and falls back to Latin Modern otherwise.

## Making it your own

1. Click **Use this template** at the top of this page to create your own repo. That repo is your portfolio piece.
2. Add a `LOG.md` and commit at least once per working session.
3. Replace this README's top section with *your* project: target, headline numbers, one figure, and how to reproduce them.
4. Tag `v1.0` at Demo Day.

## Credits and licence

MIT licence (see `LICENSE`). Built on [Lightkurve](https://lightkurve.github.io/lightkurve/),
[Astropy](https://www.astropy.org/), [batman](https://lkreidberg.github.io/batman/docs/html/index.html)
and [wotan](https://github.com/hippke/wotan). This work uses data from the Kepler and TESS missions,
obtained from MAST at the Space Telescope Science Institute.
