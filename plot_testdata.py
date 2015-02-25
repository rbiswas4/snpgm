#!/usr/bin/env python
"""Plot some light curves."""

import os
from os.path import join, basename
from glob import glob
import sncosmo

import Population_Models as PM


### Inputs
Model = PM.SinglePop() 
N = 20
###

def plot_testdata(nsne, under_model):

    if not os.path.exists("lcplots_%s"%(under_model.name(nsne))):
        os.mkdir("lcplots_%s"%(under_model.name(nsne)))

    fnames = glob("testdata_%s/*"%(under_model.name(nsne)))
    for fname in fnames:
        plotname = fname.replace("testdata_%s"%(under_model.name(nsne)), "lcplots_%s"%(under_model.name(nsne))).replace("dat", "png")
        data = sncosmo.read_lc(fname)
        sncosmo.plot_lc(data, fname=plotname)

plot_testdata(N, Model)
