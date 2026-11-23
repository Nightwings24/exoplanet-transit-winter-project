# Resource compendium

The same curated list as Section 9 of the [project guide](exoplanet_project_guide.pdf), in a form you can click on GitHub.
⭐ = start here.

## Start here: the transit method, gently
- ⭐ [NASA Exoplanet Exploration: Ways to find a planet](https://science.nasa.gov/exoplanets/how-we-find-and-characterize/): NASA's illustrated introduction to transits and the other detection methods.
- ⭐ [NAAP Transit simulator (Univ. of Nebraska–Lincoln)](https://astro.unl.edu/newRTs/Transits/background/Transit1.html): move a planet across a star and watch the light curve change.
- [OpenLearn: Introduction to exoplanets (The Open University)](https://www.open.edu/openlearn/mod/oucontent/view.php?id=74851&section=4.1): a free short course covering transit depth, radius and limb darkening.
- [HEASARC: How to find an exoplanet with TESS data](https://heasarc.gsfc.nasa.gov/docs/tess/HowToFindAnExoplanet-UserVersion.html): the TESS Science Support Center's beginner notebook.

## Python, Git and scientific computing
- ⭐ [Software Carpentry: Plotting and Programming in Python](https://swcarpentry.github.io/python-novice-gapminder/)
- ⭐ [Software Carpentry: Version Control with Git](https://swcarpentry.github.io/git-novice/)
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html)
- [Python Data Science Handbook (J. VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Learn Astropy](https://learn.astropy.org/)
- [GitHub Skills](https://skills.github.com/)
- [Code/Astro workshop](https://github.com/semaphoreP/codeastro): for seniors who want to turn research code into a real package.

## Data archives and catalogues
- ⭐ [MAST](https://archive.stsci.edu/): home of all Kepler, K2 and TESS data.
- [MAST Kepler mission page and documentation](https://archive.stsci.edu/missions-and-data/kepler)
- [MAST TESS mission page](https://archive.stsci.edu/missions-and-data/tess)
- ⭐ [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/): your answer key and source for stellar parameters ([Planetary Systems Composite table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars)).
- [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/): TOIs, TIC stellar parameters, follow-up notes.
- [TESS Science Support Center](https://heasarc.gsfc.nasa.gov/docs/tess/): sector pointings, the Web TESS Viewing Tool, tutorials.
- [astroquery.mast](https://astroquery.readthedocs.io/en/latest/mast/mast.html)

## Hands-on tutorials and notebooks
- ⭐ [Lightkurve documentation](https://lightkurve.github.io/lightkurve/)
- [Lightkurve: Searching & downloading data](https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/searching-for-data-products.html) (Week 1)
- ⭐ [Lightkurve: Identifying transiting planet signals](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html) (Week 2)
- [Lightkurve: Recover a known planet in Kepler data](https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-recover-a-known-planet.html) (Week 2)
- [HEASARC TESS tutorial index](https://heasarc.gsfc.nasa.gov/docs/tess/tutorial_landing.html)
- [lksearch](https://lightkurve.github.io/lksearch/)
- [MAST notebooks](https://spacetelescope.github.io/mast_notebooks/)
- [Astropy Box Least Squares](https://docs.astropy.org/en/stable/timeseries/bls.html)
- [batman documentation](https://lkreidberg.github.io/batman/docs/html/index.html) (Week 3)
- [wotan](https://github.com/hippke/wotan)
- [Transit Least Squares](https://transitleastsquares.readthedocs.io/)
- [emcee](https://emcee.readthedocs.io/)
- [exoplanet (PyMC) case studies](https://docs.exoplanet.codes/): for seniors.

## Books and lecture notes
- ⭐ [J. Winn, *Transits and Occultations* (2010)](https://arxiv.org/abs/1001.2010)
- C. Haswell, *Transiting Exoplanets* (Cambridge University Press, 2010): an undergraduate textbook pitched at this level. Check your library.
- M. Perryman, *The Exoplanet Handbook*, 2nd ed. (Cambridge University Press, 2018)
- S. Seager (ed.), *Exoplanets* (University of Arizona Press, 2010)
- Ivezić et al., *Statistics, Data Mining, and Machine Learning in Astronomy* (Princeton University Press, 2019)
- [Hogg, Bovy & Lang, *Data analysis recipes: Fitting a model to data*](https://arxiv.org/abs/1008.4686)
- [*Tutorial: Detecting Planetary Transits and Radial-Velocity Signals* (2018)](https://ui.adsabs.harvard.edu/abs/2018ASSP...49..267B/abstract)

## Papers, by week
| Week | Papers |
|---|---|
| 1 | [Borucki+ 2010 (Kepler)](https://ui.adsabs.harvard.edu/abs/2010Sci...327..977B) · [Ricker+ 2015 (TESS)](https://arxiv.org/abs/1406.0151) · [Batalha+ 2011 (Kepler-10b)](https://arxiv.org/abs/1102.0605) · [Winn 2010](https://arxiv.org/abs/1001.2010) |
| 2 | [Kovács+ 2002 (BLS)](https://arxiv.org/abs/astro-ph/0206099) · [Ofir 2014 (frequency grid)](https://ui.adsabs.harvard.edu/abs/2014A&A...561A.138O) · [Hippke+ 2019 (wotan)](https://arxiv.org/abs/1906.00966) · [Hippke & Heller 2019 (TLS)](https://arxiv.org/abs/1901.02015) · [Jenkins+ 2010 (Kepler pipeline)](https://ui.adsabs.harvard.edu/abs/2010ApJ...713L..87J) |
| 3 | [Mandel & Agol 2002](https://arxiv.org/abs/astro-ph/0210099) · [Kreidberg 2015 (batman)](https://arxiv.org/abs/1507.08285) · [Seager & Mallén-Ornelas 2003](https://arxiv.org/abs/astro-ph/0206228) · [Heller 2019 (limb darkening & depth)](https://arxiv.org/abs/1901.01730) · [Kipping 2013](https://arxiv.org/abs/1308.0009) · [Claret & Bloemen 2011](https://ui.adsabs.harvard.edu/abs/2011A&A...529A..75C) · [Claret 2017](https://ui.adsabs.harvard.edu/abs/2017A&A...600A..30C) · [Foreman-Mackey+ 2013 (emcee)](https://arxiv.org/abs/1202.3665) · [Hogg & Foreman-Mackey 2018](https://arxiv.org/abs/1710.06068) · [Huang+ 2018 (π Men c)](https://arxiv.org/abs/1809.05967) · [Fulton+ 2017 (radius gap)](https://arxiv.org/abs/1703.10375) |
| 4 | Christiansen+ [2013](https://ui.adsabs.harvard.edu/abs/2013ApJS..207...35C), [2015](https://arxiv.org/abs/1507.05097), [2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...828...99C), [2020](https://arxiv.org/abs/2010.04796) · [Twicken+ 2016](https://arxiv.org/abs/1604.06140) · [Thompson+ 2018](https://arxiv.org/abs/1710.06758) · [Petigura+ 2013](https://arxiv.org/abs/1311.6806) · [Dressing & Charbonneau 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...807...45D) · [Burke+ 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...809....8B) · [Shallue & Vanderburg 2018](https://arxiv.org/abs/1712.05044) |

## Citizen science and community
- [Exoplanet Watch (NASA)](https://science.nasa.gov/citizen-science/exoplanet-watch/)
- [Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess)
- [Research Notes of the AAS](https://journals.aas.org/research-notes/)
- [NASA ADS](https://ui.adsabs.harvard.edu/)
- [arXiv astro-ph.EP](https://arxiv.org/list/astro-ph.EP/new)
