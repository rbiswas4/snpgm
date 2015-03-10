#!/usr/bin/env python
"""Library for different likelihoods"""

import numpy as np
from numpy.random import rand, normal
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM

import sncosmo

class SinglePop:
    """Single pop, known x1 and c distributions"""

    def __init__(self):
        """ True nuisance parameters"""
        self.alpha = 1.5
        self.beta = 2.5 
        self.x0_0  = 1e12
        self.sigma = 0.15
        
        self.O_M = 0.7
        self.H_0 = 70.
        
        self.dim = 5
        self.initial = np.array([self.O_M, self.x0_0, self.sigma, self.alpha, self.beta])
        self.labels = [r'$\Omega_\mathrm{M}$', r'$x_{00}$', r'$\sigma_{\mathrm{int}}$',  \
                       r'$\alpha$', r'$\beta$']
        self.global_bounds = {0: (0., 1.), # Omega_M
                 1: (1e11, 1e13), # x0_0
                 2: (0., 1.),  # dx0 (fractional scatter in x0)
                 3: (0., 3.), # alpha
                 4: (1., 5.)} # beta

    def name(self, nsne):
        """Returns the name of the model plus the number of SNe involved
        To be used for naming output files"""
        return 'SP_%s'%(nsne)

    def gen_dataset_params(self, nsne):
        """Return array of x1, c, x0 corresponding to number of SNe
        NOTE: z_true has 25% at low redshift
        """
        x1_dist = norm(0., 1.)
        c_dist = norm(0., 0.1)
        sigmaint_dist = norm(0., 0.15)

        np.random.seed(0)
        self.z_true = np.empty(nsne)
        self.z_true[0:int(nsne*0.25)] = 0.02 + 0.08 * rand(int(nsne*0.25))
        self.z_true[int(nsne*0.25):] = 0.5 + 0.1 * rand(nsne-int(nsne*0.25))

        cosmo = FlatLambdaCDM(Om0=self.O_M, H0=self.H_0)
        
        self.x1_true = x1_dist.rvs(nsne)
        self.c_true = c_dist.rvs(nsne)
        self.sigma_true = sigmaint_dist.rvs(nsne)
        self.x0_true = self.x0_0 * 10**(-0.4 * (-self.alpha*self.x1_true + self.beta*self.c_true +
                              cosmo.distmod(self.z_true).value + self.sigma_true))
        return self.z_true, self.x1_true, self.c_true, self.x0_true


    def lnlike(self, parameters, snsamples):
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
        for i, b in self.global_bounds.items():
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


class MultiPop:
    """Multi pop, known x1 and c distributions"""

    def __init__(self):
        self.alpha = 1.5
        self.beta = 2.5 
        self.x0_0A = 1e12  # x0 at distance modulus 0.
        self.x0_0B = 2e12
        self.sigma_A = 0.1
        self.sigma_B = 0.1
        self.n_A = 0.5
        
        self.O_M = 0.7
        self.H_0 = 70.
        
        self.dim = 8
        self.initial = np.array([self.O_M, self.x0_0A, self.sigma_A, self.x0_0B, self.sigma_B, self.n_A, self.alpha, self.beta])
        self.labels = [r'$\Omega_\mathrm{M}$', r'$x_{00, A}$', r'$\sigma_{\mathrm{int, A}}$', r'$x_{00, B}$', r'$\sigma_{\mathrm{int, B}}$', \
          r'$n_A$', r'$\alpha$', r'$\beta$']
        self.global_bounds = global_bounds = {0: (0., 1.), # Omega_M
                 1: (1e11, 1e13), # x0_0A
                 2: (0., 1.),  # dx0A (fractional scatter in x0)
                 3: (1e11, 1e13), # x0_0B
                 4: (0., 1.),  # dx0B (fractional scatter in x0)
                 5: (0., 1.),  # n_A
                 6: (0., 3.), # alpha
                 7: (1., 5.)} # beta

    def name(self, nsne):
        """Returns the name of the model plus the number of SNe involved
        To be used for naming output files"""
        return 'MP_%s'%(nsne)

    def gen_dataset_params(self, nsne):
        """Return array of x1, c, x0 corresponding to number of SNe
        NOTE: z_true has 25% at low redshift"""
        x1_dist = norm(0., 1.)
        c_dist = norm(0., 0.1)

        np.random.seed(0)
        self.z_true = np.empty(nsne)
        self.z_true[0:int(nsne*0.25)] = 0.02 + 0.08 * rand(int(nsne*0.25))
        self.z_true[int(nsne*0.25):] = 0.5 + 0.1 * rand(nsne-int(nsne*0.25))

        cosmo = FlatLambdaCDM(Om0=self.O_M, H0=self.H_0)
        
        self.x1_true = x1_dist.rvs(nsne)
        self.c_true = c_dist.rvs(nsne)

        self.ranseed = rand(nsne)
        self.x0_0 = np.zeros(nsne)
        for i in range(0,nsne):
            if self.ranseed[i] <= self.n_A:
                x0_0_dist = norm(self.x0_0A,0.10*self.x0_0A/2.5)
                self.x0_0[i] = x0_0_dist.rvs(1)
            else:
                x0_0_dist = norm(self.x0_0B, 0.10*self.x0_0B/2.5)
                self.x0_0[i] = x0_0_dist.rvs(1)
        self.x0_true = self.x0_0 * 10**(-0.4 * (-self.alpha*self.x1_true + self.beta*self.c_true +
                              cosmo.distmod(self.z_true).value ))
        return self.z_true, self.x1_true, self.c_true, self.x0_true

    

    def lnlike(self, parameters, snsamples):
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
        for i, b in self.global_bounds.items():
            if not b[0] < parameters[i] < b[1]:
                return -np.inf

        Om0, x0_0A, dx0_A, x0_0B, dx0_B, n_A, alpha, beta = parameters
        cosmo = FlatLambdaCDM(Om0=Om0, H0=70.)

        # Loop over SNe, accumulate likelihood
        logltot = 0.
        for z, samples in snsamples:

            x0 = samples[:, 1]
            x1 = samples[:, 2]
            c = samples[:, 3]

            # calculate x0 prior for each sample
            mu = cosmo.distmod(z).value
            x0ctr_A = x0_0A * 10**(-0.4 * (-alpha*x1 + beta*c + mu))
            x0sigma_A = x0ctr_A * dx0_A/2.5

            x0ctr_B = x0_0B * 10**(-0.4 * (-alpha*x1 + beta*c + mu))
            x0sigma_B = x0ctr_B * dx0_B/2.5

            weights = (n_A / (x0sigma_A * np.sqrt(2. * np.pi)) *
                       np.exp( -(x0 - x0ctr_A)**2 / (2. * x0sigma_A**2))
                       + (1. - n_A) / (x0sigma_B * np.sqrt(2. * np.pi)) *
                       np.exp( -(x0 - x0ctr_B)**2 / (2. * x0sigma_B**2)))


            logltot += np.log(weights.sum())

        return logltot
