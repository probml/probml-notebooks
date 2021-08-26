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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/hbayes_binom_rats_pymc3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="JzJY-hRWTwYg"
# We fit a hierarchical beta-binomial model to some count data derived from rat survival. (In the book, we motivate this in terms of covid incidence rates.)
# Based on https://docs.pymc.io/notebooks/GLM-hierarchical-binominal-model.html
#

# + id="Kq6ayLpdTtKZ"

import sklearn
import scipy.stats as stats
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os
import pandas as pd

# + id="pU2KC_mp_5c0" outputId="dd7a5538-2fb1-4f26-c2f3-78a49f96d5ad" colab={"base_uri": "https://localhost:8080/"}
# #!pip install pymc3 # colab uses 3.7 by default (as of April 2021)

 # arviz needs 3.8+
# #!pip install pymc3>=3.8 # fails to update
# !pip install pymc3==3.11 

import pymc3 as pm
print(pm.__version__)

import arviz as az
print(az.__version__)

# + id="NRAKXCqDULZf"



import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
#import seaborn as sns
import pymc3 as pm
import arviz as az
import theano.tensor as tt

# + id="GbnW1XTcUcvu"
np.random.seed(123)




# rat data (BDA3, p. 102)
y = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
    1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  2,
    5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  4,
    10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 16, 15,
    15,  9,  4
])
n = np.array([
    20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20,
    20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19,
    46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20,
    48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46,
    47, 24, 14
])

N = len(n)


# + colab={"base_uri": "https://localhost:8080/", "height": 574} id="QDnhJtbDUhZA" outputId="f09cbb57-8ffd-4a15-ef3f-7711bf06fe17"

def logp_ab(value):
    ''' prior density'''
    return tt.log(tt.pow(tt.sum(value), -5/2))


with pm.Model() as model:
    # Uninformative prior for alpha and beta
    ab = pm.HalfFlat('ab',
                     shape=2,
                     testval=np.asarray([1., 1.]))
    pm.Potential('p(a, b)', logp_ab(ab))
    alpha = pm.Deterministic('alpha', ab[0])
    beta = pm.Deterministic('beta', ab[1])
    X = pm.Deterministic('X', tt.log(ab[0]/ab[1]))
    Z = pm.Deterministic('Z', tt.log(tt.sum(ab)))

    theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=N)

    p = pm.Binomial('y', p=theta, observed=y, n=n)
    #trace = pm.sample(1000, tune=2000, target_accept=0.95)
    trace = pm.sample(1000, tune=500)
    
    
#az.plot_trace(trace)
#plt.savefig('../figures/hbayes_binom_rats_trace.png', dpi=300)

print(az.summary(trace))

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4dgSQ2vAUlc-" outputId="490d6b45-b7dd-4633-9f59-29081bf8f533"

J = len(n)
post_mean = np.zeros(J)
samples = trace[theta]
post_mean = np.mean(samples, axis=0)
print('post mean')
print(post_mean)

alphas = trace['alpha']
betas = trace['beta']
alpha_mean = np.mean(alphas)
beta_mean = np.mean(betas)
hyper_mean = alpha_mean/(alpha_mean + beta_mean)
print('hyper mean')
print(hyper_mean)


mle = y / n
pooled_mle = np.sum(y) / np.sum(n)

print('pooled mle')
print(pooled_mle)


#axes = az.plot_forest(
#    trace, var_names='theta', credible_interval=0.95, combined=True, colors='cycle')

axes = az.plot_forest(
    trace, var_names='theta', hdi_prob=0.95, combined=True, colors='cycle')

y_lims = axes[0].get_ylim()
axes[0].vlines(hyper_mean, *y_lims)
#plt.savefig('../figures/hbayes_binom_rats_forest95.pdf', dpi=300)


J = len(n)
fig, axs = plt.subplots(4,1, figsize=(10,10))
plt.subplots_adjust(hspace=0.3)
axs = np.reshape(axs, 4)
xs = np.arange(J)
ax = axs[0]
ax.bar(xs, y)
ax.set_title('number of postives')
ax = axs[1]
ax.bar(xs, n)
ax.set_title('popn size')
ax = axs[2]
ax.bar(xs, mle)
ax.set_ylim(0, 0.5)
ax.hlines(pooled_mle, 0, J, 'r', lw=3)
ax.set_title('MLE (red line = pooled)')
ax = axs[3]
ax.bar(xs, post_mean)
ax.hlines(hyper_mean, 0, J, 'r', lw=3)
ax.set_ylim(0, 0.5)
ax.set_title('posterior mean (red line = hparam)')
#plt.savefig('../figures/hbayes_binom_rats_barplot.pdf', dpi=300)


J = len(n)
xs = np.arange(J)
fig, ax = plt.subplots(1,1)
ax.bar(xs, y)
ax.set_title('number of postives')
#plt.savefig('../figures/hbayes_binom_rats_outcomes.pdf', dpi=300)

fig, ax = plt.subplots(1,1)
ax.bar(xs, n)
ax.set_title('popn size')
#plt.savefig('../figures/hbayes_binom_rats_popsize.pdf', dpi=300)

fig, ax = plt.subplots(1,1)
ax.bar(xs, mle)
ax.set_ylim(0, 0.5)
ax.hlines(pooled_mle, 0, J, 'r', lw=3)
ax.set_title('MLE (red line = pooled)')
#plt.savefig('../figures/hbayes_binom_rats_MLE.pdf', dpi=300)

fig, ax = plt.subplots(1,1)
ax.bar(xs, post_mean)
ax.hlines(hyper_mean, 0, J, 'r', lw=3)
ax.set_ylim(0, 0.5)
ax.set_title('posterior mean (red line = hparam)')
#plt.savefig('../figures/hbayes_binom_rats_postmean.pdf', dpi=300)

