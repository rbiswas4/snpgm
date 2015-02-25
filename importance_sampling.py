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

import Population_Models as PM

#### Inputs 
Model = PM.SinglePop() 
N = 20
####

# -----------------------------------------------------------------------------
# Main

def importance_sampling(nsne, under_model):
    # sampler parameters 
    ndim = under_model.dim 
    nwalkers = 20
    nburn = 200
    nsamples = 500
              
    # Read all SN redshifts and previously-generated parameter samples
    snsamples = []
    for fname in sorted(glob("testdata_%s/*"%(under_model.name(nsne)))):
        z = sncosmo.read_lc(fname).meta['z']  # load whole file just to get 'z'
        sfname = fname.replace("testdata_%s"%(under_model.name(nsne)), "samples_%s"%(under_model.name(nsne))).replace(".dat", ".npy")
        samples = np.load(sfname)
        snsamples.append((z, samples))

    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim,under_model.lnlike, args=(snsamples,))

    # Starting positions
    errors = under_model.initial*0.01
    pos = np.array([ under_model.initial + errors*np.random.randn(ndim)
                for i in range(nwalkers)])

    # burn-in 
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    print("Burn in done")

    # production run
    sampler.reset()
    sampler.run_mcmc(pos, nsamples)
    print("Avg acceptance fraction:", np.mean(sampler.acceptance_fraction))

    results = sampler.flatchain
    np.save("samples_%s/globalsamples_%s.npy"%(under_model.name(nsne),under_model.name(nsne)), results)

importance_sampling(N, Model)
