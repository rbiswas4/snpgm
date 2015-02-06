#!/usr/bin/env python
"""Sample one light curve at a time, plot the model and make a corner plot."""

import os
from os.path import join, basename
from glob import glob

import numpy as np
import sncosmo
import emcee
from astropy.cosmology import FlatLambdaCDM

from matplotlib import pyplot as plt
import triangle

if not os.path.exists("samples"):
    os.mkdir("samples")

# sampler parameters
ndim = 4
nwalkers = 20
nburn = 100
nsamples = 500


# define likelihood
model = sncosmo.Model(source='salt2-extended')

bounds = {0: (-20., 40.),  # t0
          2: (-4., 4.),  # x1
          3: (-0.4, 0.4)}

def lnlike(parameters, data):
    """Return log L for array of parameters."""

    # If any parameters are out-of-bounds, return 0 probability.
    for i, b in bounds.items():
        if not b[0] < parameters[i] < b[1]:
            return -np.inf

    model.parameters[1:5] = parameters  # set model t0, x0, x1, c
    mflux = model.bandflux(data['band'], data['time'],
                           zp=data['zp'], zpsys=data['zpsys'])
    chisq = np.sum(((data['flux'] - mflux) / data['fluxerr'])**2)
    return -chisq / 2.


def sample(data):
    """Return MCMC samples for model defined above."""

    # fix redshift in the model
    model.set(z=data.meta['z'])

    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(data,), a = 3.50)

    # Starting positions of walkers.
    # Here we cheat by setting the "guess" equal to the known true parameters!
    current = np.array([data.meta['t0'], data.meta['x0'], data.meta['x1'],
                        data.meta['c']])
    errors = np.array([1., 0.1*data.meta['x0'], 1., 0.1])
    pos = [current + errors*np.random.randn(ndim) for i in range(nwalkers)]

    # burn-in
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()

    # production run
    sampler.run_mcmc(pos, nsamples)
    print("Avg acceptance fraction:", np.mean(sampler.acceptance_fraction))

    return sampler.flatchain


fnames = sorted(glob("testdata/*"))
for fname in fnames:
  
    # read data, run sampler on it, save samples
    data = sncosmo.read_lc(fname)
    samples = sample(data)
    sfname = (fname.replace("testdata", "samples")
               .replace(".dat", ".npy"))
    np.save(sfname, samples)

    # show mean parameters model with data
    showparams = np.average(samples, axis=0)
    figname = (fname.replace("testdata", "lcplots")
               .replace(".dat", "_lcfit.png"))
    model.parameters[1:5] = showparams
    sncosmo.plot_lc(data, model, fname=figname)

    # Corner plot of samples.
    labels = ["${0}$".format(s) for s in model.param_names_latex[1:5]]
    fig = triangle.corner(samples, labels=labels, bins=30)
    figname = (fname.replace("testdata", "lcplots")
               .replace(".dat", "_corner.png"))
    plt.savefig(figname)


