#!/usr/bin/env python
"""Sample one light curve at a time, plot the model and make a corner plot."""

import os
from os.path import join, basename
from glob import glob

import numpy as np
import sncosmo
import emcee
import triangle
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from astropy.cosmology import FlatLambdaCDM

if not os.path.exists("lcplots"):
    os.mkdir("lcplots")

# sampler parameters
ndim = 4
nwalkers = 130 #20
nburn = 50 #100
nsamples = 200 #500
n_obs = 15

# define likelihood
model = sncosmo.Model(source='salt2-extended')

bounds = {0: (-20., 40.),  # t0
          2: (-4., 4.),  # x1
          3: (-0.4, 0.4)}

global_bounds = {0: (0., 1.), # Omega_M
                 1: (1e11, 1e13), # x0_0
                 2: (0., 3.), # alpha
                 3: (1., 5.)} # beta
                     
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

def full_lnlike(parameters, data_list):
    """Return log L for array of parameters."""

    # If any parameters are out-of-bounds, return 0 probability:
    for i, b in global_bounds.items():
        if not b[0] < parameters[i] < b[1]:
            return -np.inf
    n_sne = len(data_list)

    Omega, x0_0, alpha, beta = parameters[:4]
    
    cosmo = FlatLambdaCDM(Om0=Omega, H0=70.)
    sn_lnlike = np.empty(n_sne)
    for n_sn in range(n_sne):
        model.set(z=data_list[n_sn].meta['z'])
        mu = cosmo.distmod(data_list[n_sn].meta['z']).value
        sn_parameters = parameters[4+n_sn*4:4+n_sn*4+4]
        x0 = x0_0 * 10**(-0.4 * (-alpha*sn_parameters[2] + 
                                  beta*sn_parameters[3] +
                                  mu + sn_parameters[1]))
        
        sn_parameters[1] = x0
        sn_lnlike[n_sn] = lnlike(sn_parameters, data_list[n_sn])
    
    return sn_lnlike.sum()


def sample(data):
    """Return MCMC samples for model defined above."""

    # fix redshift in the model
    model.set(z=data.meta['z'])

    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(data,))

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

def sample_all(data_list):
    """Return MCMC samples for model with all sne."""

    ndim = 4 + 4*len(data_list)
    
    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, full_lnlike, 
                                    args = (data_list,), threads=24)

    # Starting positions
    current = np.empty(ndim)
    errors = np.empty(ndim)
    current[:4] = np.array([0.7, 1e12, 1.5,2.5])
    errors[:4] = current[:4]*.1

    for s, d in enumerate(data_list):
        current[4+4*s:4+4*(s+1)] = np.array([data.meta['t0'], data.meta['x0'],
                                             data.meta['x1'], data.meta['c']])
        errors[4+4*s:4+4*(s+1)] = np.array([1., 0.15, 1., 0.1])
    pos = [current + errors*np.random.randn(ndim) for i in range(nwalkers)]
    
    # burn-in 
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    print("Burn in done")
    # production run
    sampler.run_mcmc(pos, nsamples)
    print("Avg acceptance fraction:", np.mean(sampler.acceptance_fraction))
         
    return sampler.flatchain

fnames = sorted(glob("testdata/*"))
"""
for fname in fnames[0:1]:
  
    data = sncosmo.read_lc(fname)
    samples = sample(data)

    showparams = np.average(samples, axis=0)
    figname = (fname.replace("testdata", "lcplots")
               .replace(".dat", "_lcfit.png"))
    model.parameters[1:5] = showparams
    sncosmo.plot_lc(data, model, fname=figname)

    labels = ["${0}$".format(s) for s in model.param_names_latex[1:5]]
    fig = triangle.corner(samples, labels=labels, bins=30)
    figname = (fname.replace("testdata", "lcplots")
               .replace(".dat", "_corner.png"))
    plt.savefig(figname)
"""
# Get data from all SNe
data_list = []
for fname in fnames[:n_obs]:
    
    data = sncosmo.read_lc(fname)
    data_list.append(data)

# Do MCMC:
all_samples = sample_all(data_list)
np.savetxt('emcee_samples.dat', all_samples)
all_params = np.average(all_samples, axis = 0)

# Make plots of best fit global parameters and histograms of best fit 
# parameters for each supernova:
labels = ['Omega', 'x0_0', 'alpha', 'beta']
fig = triangle.corner(all_samples[:,:4], labels=labels, bins=30)
plt.savefig('global_params_%ssne.png' % n_obs)

fig2, axes = plt.subplots(2,2)
titles = ['t0', 'sigma', 'x1', 'c']
for a in range(4):
    best_fits = [np.median(all_samples[:,4+s*4+a]) for s in range(n_obs)]
    axes[a].hist(best_fits)
    axes[a].set_xlabel(titles[a])

fig2.savefig('sne_params_%ssne.png' % n_obs)
