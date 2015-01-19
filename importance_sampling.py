#!/usr/bin/env python
"""Assuming there are already samples generated for each SN, sample
the outside parameters of the model."""

from __future__ import print_function

import os
from os.path import join, basename
from glob import glob

import numpy as np
import sncosmo
import emcee
from astropy.cosmology import FlatLambdaCDM


def lnlike(parameters, snsamples):
    """Return log L for array of parameters.

    Parameters
    ----------
    parameters : np.ndarray
        "Outside" model parameters (length 5)
    samples : list of tuple
        List has length N_SNe. Each tuple consists of a float (redshift)
        and a 2-d np.ndarray giving the samples.
    """

    # If any parameters are out-of-bounds, return 0 probability:
    for i, b in global_bounds.items():
        if not b[0] < parameters[i] < b[1]:
            return -np.inf

    Om0, x0_0, dx0, alpha, beta = parameters
    cosmo = FlatLambdaCDM(Om0=Om0, H0=70.)

    # Loop over SNe, accumulate likelihood
    logltot = 0.
    for z, samples in snsamples:

        x0 = samples[:, 1]
        x1 = samples[:, 2]
        c = samples[:, 3]

        # calculate x0 prior for each sample
        mu = cosmo.distmod(z).value
        x0ctr = x0_0 * 10**(-0.4 * (-alpha*x1 + beta*c + mu))
        x0sigma = x0ctr * dx0

        weights = (1. / (x0sigma * np.sqrt(2. * np.pi)) *
                   np.exp( -(x0 - x0ctr)**2 / (2. * x0sigma**2)))

        logltot += np.log(weights.sum())

    return logltot


# -----------------------------------------------------------------------------
# Main

# sampler parameters
ndim = 5
nwalkers = 20
nburn = 200
nsamples = 500

global_bounds = {0: (0., 1.), # Omega_M
                 1: (1e11, 1e13), # x0_0
                 2: (0., 1.),  # dx0 (fractional scatter in x0)
                 3: (0., 3.), # alpha
                 4: (1., 5.)} # beta

# Read all SN redshifts and previously-generated parameter samples
snsamples = []
for fname in sorted(glob("testdata/*")):
    z = sncosmo.read_lc(fname).meta['z']  # load whole file just to get 'z'
    sfname = fname.replace("testdata", "samples").replace(".dat", ".npy")
    samples = np.load(sfname)
    snsamples.append((z, samples))

# Create sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(snsamples,))

# Starting positions
current = np.array([0.7, 1e12, 0.15, 1.5, 2.5])
errors = current*0.01
pos = np.array([current + errors*np.random.randn(ndim)
                for i in range(nwalkers)])

# burn-in 
pos, prob, state = sampler.run_mcmc(pos, nburn)
print("Burn in done")

# production run
sampler.reset()
sampler.run_mcmc(pos, nsamples)
print("Avg acceptance fraction:", np.mean(sampler.acceptance_fraction))

results = sampler.flatchain
np.save("samples/globalsamples.npy", results)
