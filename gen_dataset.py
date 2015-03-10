#!/usr/bin/env python
"""Generate a fake set of SNe from the SALT2 model."""

import os

import numpy as np
from numpy.random import normal
from astropy.table import Table

import sncosmo

import Population_Models as PM

#####  Inputs
N = 20
Model = PM.MultiPop() 
#####


def gen_dataset(nsne, under_model):

    #under_model = pop_model()
    z_true, x1_true, c_true, x0_true = under_model.gen_dataset_params(nsne)

    t0_true = np.zeros(nsne)

    if not os.path.exists("testdata_%s"%(under_model.name(nsne))):
        os.mkdir("testdata_%s"%(under_model.name(nsne)))


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
        sncosmo.write_lc(data, 'testdata_%s/sn{0:02d}_%s.dat'.format(i)%(under_model.name(nsne), under_model.name(nsne)))
        i += 1
    return

gen_dataset(N, Model)

