# snpgm

Computing the Universe 2015 Supernova PGM Hack

Implementing this Probabalistic Graphical Model:

![PGM](snpgm.png)

**Scripts:**

- `gen_pgm.py`: Script to create `snpgm.png` file.

- `gen_dataset.py`: Generate a test light curve data set, write
  each light curve to a file in the `testdata` directory.

- `plot_testdata.py`: Make a plot of the light curve data for
  each file in `testdata`, save to `lcplots` directory.

- `sample_lc.py`: Run an MCMC on light curve data in `testdata`,
  make a figure showing the model and a figure showing samples in
  a corner plot, saved in `lcplots` directory.

**Importance sampling papers:**

- Sonnenfeld et al "SL2S Paper 5" (strong lens ensemble)
- Schneider et al: Hierarchical WL
