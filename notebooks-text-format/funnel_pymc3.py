# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/funnel_pymc3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="iPLM5TwcMcTe"
# In this notebook, we explore the "funnel of hell". This refers to a posterior in which
# the mean and variance of a variable are highly correlated, and have a funnel
# shape. (The term "funnel of hell" is from [this blog post](https://twiecki.io/blog/2014/03/17/bayesian-glms-3/) by  Thomas Wiecki.)
#
# We illustrate this using a hierarchical Bayesian model for inferring Gaussian means, fit to synthetic data, similar to 8 schools (except we vary the same size and fix the variance). This code is based on [this notebook](http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2017/tutorials/aux8_mcmc_tips.html) from Justin Bois.

# + id="-sWa3BStE4ov"
# %matplotlib inline
import sklearn
import scipy.stats as stats
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os
import pandas as pd

# + id="1UEFiUi-qZA1" colab={"base_uri": "https://localhost:8080/"} outputId="1a20ff5d-68e6-4f60-81e0-1456bfa83b5f"
# !pip install -U pymc3>=3.8
import pymc3 as pm
print(pm.__version__)
import arviz as az
print(az.__version__)

# + id="SS-lUcY9ovUd"
import math
import pickle

import numpy as np
import pandas as pd
import scipy.stats as st
import theano.tensor as tt
import theano

# + id="H4iJ8eTAr3yF" colab={"base_uri": "https://localhost:8080/"} outputId="23291ee5-7822-41fb-d3ca-c829cd0891f5"
np.random.seed(0)
# Specify parameters for random data
mu_val = 8
tau_val = 3
sigma_val = 10
n_groups = 10

# Generate number of replicates for each repeat
n = np.random.randint(low=3, high=10, size=n_groups, dtype=int)
print(n)
print(sum(n))

# + id="oyyDYNGfsmUa" colab={"base_uri": "https://localhost:8080/"} outputId="f8d2cf60-fbbd-4a29-fcd6-747cd2e18870"
# Generate data set
mus = np.zeros(n_groups)
x = np.array([])
for i in range(n_groups):
  mus[i] = np.random.normal(mu_val, tau_val)
  samples = np.random.normal(mus[i], sigma_val, size=n[i])
  x = np.append(x, samples)

print(x.shape)

group_ind = np.concatenate([[i]*n_val for i, n_val in enumerate(n)])

# + id="Vz-gdn-zuCcx" colab={"base_uri": "https://localhost:8080/", "height": 692} outputId="19b32b08-cffc-4800-9667-5ff22df6f387"

with pm.Model() as centered_model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=2.5)
    log_tau = pm.Deterministic('log_tau', tt.log(tau))

    # Prior on theta
    theta = pm.Normal('theta', mu=mu, sd=tau, shape=n_groups)
    
    # Likelihood
    x_obs = pm.Normal('x_obs',
                       mu=theta[group_ind],
                       sd=sigma_val,
                       observed=x)


np.random.seed(0)
with centered_model:
    centered_trace = pm.sample(10000, chains=2)
    
pm.summary(centered_trace).round(2)


# + id="UMLPIRMPsgej" colab={"base_uri": "https://localhost:8080/", "height": 963} outputId="3227aaef-1030-490f-8605-5744d27f269c"
with pm.Model() as noncentered_model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=2.5)
    log_tau = pm.Deterministic('log_tau', tt.log(tau))
    
    # Prior on theta
    #theta = pm.Normal('theta', mu=mu, sd=tau, shape=n_trials)
    var_theta = pm.Normal('var_theta', mu=0, sd=1, shape=n_groups)
    theta = pm.Deterministic('theta', mu + var_theta * tau)
    
    # Likelihood
    x_obs = pm.Normal('x_obs',
                       mu=theta[group_ind],
                       sd=sigma_val,
                       observed=x)
    
np.random.seed(0)
with noncentered_model:
    noncentered_trace = pm.sample(1000, chains=2)
    
pm.summary(noncentered_trace).round(2)  

# + id="XqQQUavXvFWT" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="88b33782-8b68-4057-e1c9-b582e6db8cc1"
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
x = pd.Series(centered_trace['mu'], name='mu')
y  = pd.Series(centered_trace['tau'], name='tau')
axs[0].plot(x, y, '.');
axs[0].set(title='Centered', xlabel='µ', ylabel='τ');
axs[0].axhline(0.01)

x = pd.Series(noncentered_trace['mu'], name='mu')
y  = pd.Series(noncentered_trace['tau'], name='tau')
axs[1].plot(x, y, '.');
axs[1].set(title='NonCentered', xlabel='µ', ylabel='τ');
axs[1].axhline(0.01)

xlim = axs[0].get_xlim()
ylim = axs[0].get_ylim()

# + id="--jgSNVBLadC" colab={"base_uri": "https://localhost:8080/", "height": 495} outputId="6cf32ae5-ee7b-4abe-bf8f-b51450bb02d1"
x = pd.Series(centered_trace['mu'], name='mu')
y  = pd.Series(centered_trace['tau'], name='tau')
g = sns.jointplot(x, y, xlim=xlim, ylim=ylim)
plt.suptitle('centered')
plt.show()

# + id="tEfEJ8JuLX43" colab={"base_uri": "https://localhost:8080/", "height": 495} outputId="4869fb30-3d07-4e0c-a6da-03c1014923b3"
x = pd.Series(noncentered_trace['mu'], name='mu')
y  = pd.Series(noncentered_trace['tau'], name='tau')
g = sns.jointplot(x, y, xlim=xlim, ylim=ylim)
plt.suptitle('noncentered')
plt.show()

# + id="1-FQqDkTFEqy" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="b9804230-dc6c-4586-9a5a-1ad38a9cab82"
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
x = pd.Series(centered_trace['mu'], name='mu')
y  = pd.Series(centered_trace['log_tau'], name='log_tau')
axs[0].plot(x, y, '.');
axs[0].set(title='Centered', xlabel='µ', ylabel='log(τ)');

x = pd.Series(noncentered_trace['mu'], name='mu')
y  = pd.Series(noncentered_trace['log_tau'], name='log_tau')
axs[1].plot(x, y, '.');
axs[1].set(title='NonCentered', xlabel='µ', ylabel='log(τ)');

xlim = axs[0].get_xlim()
ylim = axs[0].get_ylim()

# + id="5QqP9pOLHJR5" colab={"base_uri": "https://localhost:8080/", "height": 495} outputId="34dfd8db-fc63-44bb-c203-5b2c64cf9d3c"
#https://seaborn.pydata.org/generated/seaborn.jointplot.html

x = pd.Series(centered_trace['mu'], name='mu')
y  = pd.Series(centered_trace['log_tau'], name='log_tau')
g = sns.jointplot(x, y, xlim=xlim, ylim=ylim)
plt.suptitle('centered')
plt.show()

# + id="7jK4o4idIw_u" colab={"base_uri": "https://localhost:8080/", "height": 495} outputId="784cde75-c370-457f-e4df-5bb51595246a"
x = pd.Series(noncentered_trace['mu'], name='mu')
y  = pd.Series(noncentered_trace['log_tau'], name='log_tau')
g = sns.jointplot(x, y, xlim=xlim, ylim=ylim)
plt.suptitle('noncentered')
plt.show()

# + id="KNam0ZuYYhxw" colab={"base_uri": "https://localhost:8080/", "height": 581} outputId="6a73f609-35a5-433f-bb22-09509881998e"
az.plot_forest([centered_trace, noncentered_trace], model_names=['centered', 'noncentered'],
               var_names="theta",
               combined=True, hdi_prob=0.95);

# + id="sizu9bNdT4K0"

