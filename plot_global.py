# plot global samples.

import numpy as np
import triangle
from matplotlib import pyplot as plt

import Population_Models as PM


### Inputs
Model = PM.SinglePop() 
N = 20
###
def plot_global(nsne, under_model):
    samples = np.load("samples_%s/globalsamples_%s.npy"%(under_model.name(nsne),under_model.name(nsne)))

    labels = under_model.labels 
    truths = under_model.initial

    fig = triangle.corner(samples, labels=labels, bins=15,
                      truths=truths)
    plt.savefig('globalsamples_%s.png'%(under_model.name(nsne)))
    return

plot_global(N, Model)
