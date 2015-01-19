# snpgm

Originally a
[hack](https://hackpad.com/CtU2015-Hacks-and-Hackers-8rzvsLPoA89) for
the [Computing the Universe
2015](http://bccp.berkeley.edu/dev/?page_id=2165) workshop. Now a
repository for testing out ideas for forward-model/hierarchical
supernova cosmology inference.

## About

We're performing inference on this Probabalistic Graphical Model (PGM):

![PGM](snpgm.png)

It's specific to the SALT2 light curve model, in terms of the parameters that
describe each light curve.

**Dependencies:**

- astropy
- sncosmo
- emcee
- triangle
- daft

... and the usual numpy/scipy/mpl business.


**Scripts:**

- `gen_pgm.py`: Script to draw the PGM. Generates `snpgm.png`.

- `gen_dataset.py`: Generate a test light curve data set, write each
  light curve to a file in the `testdata` directory. (In all scripts,
  directories are created if they don't already exist.)

- `plot_testdata.py`: Make a plot of the light curve data for each
  file in `testdata`, save to `lcplots` directory. Just to visualize
  the light curve data a bit.

- `naive_sampling.py`: Throw all the SN parameters and global parameters into
  a big MCMC and let it run. That's `4*N_SN + 4` parameters.

Importance sampling is two steps:

- `sample_lcs.py`: Run an MCMC on each light curve in `testdata` individually,
  save samples to `samples` directory as numpy binary files.

- `importance_sampling.py`: Run impotance sampling using the
  individual SN samples already created in previous step.

**Importance sampling papers:**

- Sonnenfeld et al "SL2S Paper 5" (strong lens ensemble)
- Schneider et al: Hierarchical WL
