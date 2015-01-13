#!/usr/bin/env python
"""Plot some light curves."""

import os
from os.path import join, basename
from glob import glob
import sncosmo

if not os.path.exists("lcplots"):
    os.mkdir("lcplots")

fnames = glob("testdata/*")
for fname in fnames:
    plotname = fname.replace("testdata", "lcplots").replace("dat", "png")
    data = sncosmo.read_lc(fname)
    sncosmo.plot_lc(data, fname=plotname)
