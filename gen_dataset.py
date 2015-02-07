#!/usr/bin/env python
"""Generate a fake set of SNe from the SALT2 model."""

import os

import numpy as np
from numpy.random import rand, normal
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

import sncosmo

if not os.path.exists("testdata"):
    os.mkdir("testdata")

nsne = 200

# True distributions
x1_dist = norm(0., 1.)
c_dist = norm(0., 0.1)
sigmaint_dist = norm(0., 0.15)

# True cosmology
cosmo = FlatLambdaCDM(Om0=0.7, H0=70.)

# True nuisance parameters
alpha = 1.5
beta = 2.5

# generate set of true parameters for each SN
np.random.seed(0)

# put some sne at low z and some at high-z
z_true = np.empty(nsne)
z_true[0:50] = 0.02 + 0.08 * rand(50)
z_true[50:] = 0.5 + 0.1 * rand(nsne-50)

x0_0A = 1e12  # x0 at distance modulus 0.
x0_0B = 2e12
n_A = 0.5 # relative normalize [0,1]

ranseed = rand(nsne)
x0_0 = np.zeros(nsne)
for i in range(0,nsne):
    if ranseed[i] <= n_A:
        x0_0_dist = norm(x0_0A,0.10*x0_0A/2.5)
        x0_0[i] = x0_0_dist.rvs(1)
    else:
        x0_0_dist = norm(x0_0B, 0.10*x0_0B/2.5)
        x0_0[i] = x0_0_dist.rvs(1)

t0_true = np.zeros(nsne)
x1_true = x1_dist.rvs(nsne)
c_true = c_dist.rvs(nsne)
x0_true = x0_0 * 10**(-0.4 * (-alpha*x1_true + beta*c_true +
                              cosmo.distmod(z_true).value ))

# Pretend we observe all SNe with the same cadence and bands
time = np.arange(-30., 70.)
band = np.array(25*['desg', 'desr', 'desi', 'desz'])
zp = 25. * np.ones_like(time)
zpsys = np.array(100*['ab'])

model = sncosmo.Model(source='salt2-extended')

i = 1
for z, t0, x0, x1, c in zip(z_true, t0_true, x0_true, x1_true, c_true):
    param_dict = dict(z=z, t0=t0, x0=x0, x1=x1, c=c)
    model.set(**param_dict)
    flux_true = model.bandflux(band, time, zp=zp, zpsys=zpsys)
    fluxerr = 0.1 * np.max(flux_true) * np.ones_like(flux_true)
    flux = normal(flux_true, fluxerr)
    
    data = Table((time, band, flux, fluxerr, zp, zpsys),
                 names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'),
                 meta=param_dict)
    sncosmo.write_lc(data, 'testdata/sn{0:02d}.dat'.format(i))
    i += 1
