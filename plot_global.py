# plot global samples.

import numpy as np
import triangle
from matplotlib import pyplot as plt

samples = np.load("samples/globalsamples.npy")

labels = [r'$\Omega$', r'$x_{00}$', r'$\alpha$', r'$\beta$']
truths = [0.7, 1.e12, 1.5, 2.5]  # from gen_dataset

fig = triangle.corner(samples, labels=labels, bins=30,
                      truths=truths)
plt.savefig('globalsamples.png')
