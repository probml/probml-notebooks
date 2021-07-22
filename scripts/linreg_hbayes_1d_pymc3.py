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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/bayes_stats/linreg_hbayes_1d_pymc3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="tev-ORC5cK_u" colab_type="text"
# In this notebook, we illustrate hierarchical Bayesian linear regression on a toy 1d dataset.
#
# The code is based on [Bayesian Analysis with Python, ch 3](https://github.com/aloctavodia/BAP/blob/master/code/Chp3/03_Modeling%20with%20Linear%20Regressions.ipynb) and [this blog post from Thomas Wiecki](https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/GLM_hierarchical_non_centered.ipynb).

# + id="A-synbXUcSj9" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 72} outputId="9da55b18-3e43-4454-d0e7-4fb4cfe9d219"
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

# + id="GVmyuhjecYD7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 674} outputId="9cfe8170-24f3-4153-8851-22d20680d773"

# !pip install pymc3==3.8
import pymc3 as pm
pm.__version__


# !pip install arviz
import arviz as az

# + id="FItChsd0coXu" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 404} outputId="c973802a-8b84-4845-bc06-2799871c92b2"
N = 10 #20
M = 8 # num groups
idx = np.repeat(range(M-1), N) # N samples for groups 0-6
idx = np.append(idx, 7) # 1 sample for 7'th group
np.random.seed(123)

#alpha_real = np.random.normal(2.5, 0.5, size=M)
#beta_real = np.random.beta(6, 1, size=M)
#eps_real = np.random.normal(0, 0.5, size=len(idx))

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(1, 1, size=M) # slope is closer to 0
eps_real = np.random.normal(0, 0.5, size=len(idx))

print(beta_real)

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real
x_centered = x_m - x_m.mean()

_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
ax = np.ravel(ax)
j, k = 0, N
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', rotation=0, labelpad=15)
    #ax[i].set_xlim(6, 15)
    #ax[i].set_ylim(1, 18)
    j += N
    k += N
plt.tight_layout()

# + id="xPu4dKqDconD" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 664} outputId="169f87c0-b01f-4262-fe10-25286d58a422"
# Fit separarate models per group

with pm.Model() as unpooled_model:
    α = pm.Normal('α', mu=0, sd=10, shape=M)
    β = pm.Normal('β', mu=0, sd=10, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)

    y_pred = pm.Normal('y_pred', mu=α[idx] + β[idx] * x_m,
                         sd=ϵ,  observed=y_m)
    trace_up = pm.sample(1000)

az.summary(trace_up)


# + id="y0xtaDY0crIi" colab_type="code" colab={}
def plot_post_pred_samples(trace, nsamples=20):
    _, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True,
                       constrained_layout=True)
    ax = np.ravel(ax)
    j, k = 0, N
    x_range = np.linspace(x_m.min(), x_m.max(), 10)
    X =  x_range[:, np.newaxis]
    
    for i in range(M):
        ax[i].scatter(x_m[j:k], y_m[j:k])
        ax[i].set_xlabel(f'x_{i}')
        ax[i].set_ylabel(f'y_{i}', labelpad=17, rotation=0)
        alpha_m = trace['α'][:, i].mean()
        beta_m = trace['β'][:, i].mean()
        ax[i].plot(x_range, alpha_m + beta_m * x_range, c='r', lw=3,
                  label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
        plt.xlim(x_m.min()-1, x_m.max()+1)
        plt.ylim(y_m.min()-1, y_m.max()+1)
        alpha_samples = trace['α'][:,i]
        beta_samples = trace['β'][:,i]
        ndx = np.random.choice(np.arange(len(alpha_samples)), nsamples)
        alpha_samples_thinned = alpha_samples[ndx]
        beta_samples_thinned = beta_samples[ndx]
        ax[i].plot(x_range, alpha_samples_thinned + beta_samples_thinned * X,
            c='gray', alpha=0.5)
        
        j += N
        k += N


# + id="LaI3MCA_c21P" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 384} outputId="924ddbd7-20d1-4287-ea52-0dd4cf056f3e"
plot_post_pred_samples(trace_up)

# + id="FJXYos8zeOlN" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 837} outputId="23b0c612-30e9-4973-eb60-9d878de929b5"
# Fit a hierarchical (centered) model to the raw data
with pm.Model() as model_centered:
    # hyper-priors
    μ_α = pm.Normal('μ_α', mu=0, sd=10)
    σ_α = pm.HalfNormal('σ_α', 10)
    μ_β = pm.Normal('μ_β', mu=0, sd=10)
    σ_β = pm.HalfNormal('σ_β', sd=10)

    # priors
    α = pm.Normal('α', mu=μ_α, sd=σ_α, shape=M)
    β = pm.Normal('β', mu=μ_β, sd=σ_β, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)

    y_pred = pm.Normal('y_pred', mu=α[idx] + β[idx] * x_m,
                         sd=ϵ, observed=y_m)

    trace_centered = pm.sample(1000)

az.summary(trace_centered)


# + id="lKl5e1vOeSoj" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 384} outputId="46ccdb99-1e73-461f-d658-be40f9f6e01b"
plot_post_pred_samples(trace_centered)

# + id="XbNOzLwjef--" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="2908a3ad-5ed4-459d-82c9-aedbe5b561d9"
# Fit the non-centered model to the raw data
with pm.Model() as model_noncentered:
    # hyper-priors
    μ_α = pm.Normal('μ_α', mu=0, sd=10)
    σ_α = pm.HalfNormal('σ_α', 10)
    μ_β = pm.Normal('μ_β', mu=0, sd=10)
    σ_β = pm.HalfNormal('σ_β', sd=10)

    # priors
    α_offset = pm.Normal('α_offset', mu=0, sd=1, shape=M)
    α = pm.Deterministic('α', μ_α + σ_α * α_offset) 
    β_offset = pm.Normal('β_offset', mu=0, sd=1, shape=M)
    β = pm.Deterministic('β', μ_β + σ_β * β_offset) 

    ϵ = pm.HalfCauchy('ϵ', 5)

    y_pred = pm.Normal('y_pred', mu=α[idx] + β[idx] * x_m,
                         sd=ϵ, observed=y_m)

    trace_noncentered = pm.sample(1000)

az.summary(trace_noncentered)


# + id="7nUnTN-0emHO" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 384} outputId="6c802c48-1ad3-4b81-92a8-779f9e8dc150"
plot_post_pred_samples(trace_noncentered)

# + id="H2stO1cofC0h" colab_type="code" colab={}

