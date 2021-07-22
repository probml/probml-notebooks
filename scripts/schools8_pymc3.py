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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/schools8_pymc3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="W_0ED20uQKha"
# In this notebook, we fit a hierarchical Bayesian model to the "8 schools" dataset.
# See also https://github.com/probml/pyprobml/blob/master/scripts/schools8_pymc3.py

# + id="HXRokZL1QPvB"
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


# + id="C5EHDB-rQSIa" colab={"base_uri": "https://localhost:8080/"} outputId="d6d8b024-96ba-4014-97d9-ddef6d88349e"
# !pip install -U pymc3>=3.8
import pymc3 as pm
print(pm.__version__)
import theano.tensor as tt
import theano

# #!pip install arviz
import arviz as az

# + id="sKlvHNY6RUaP"
# !mkdir ../figures

# + [markdown] id="-jby_J17HqBT"
# # Data

# + id="8pNC3UANQjeO" colab={"base_uri": "https://localhost:8080/", "height": 297} outputId="8f91ec2e-e81b-452b-dcf7-8c9f6ddda82a"
# https://github.com/probml/pyprobml/blob/master/scripts/schools8_pymc3.py

# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
print(np.mean(y))
print(np.median(y))

names=[]; 
for t in range(8):
    names.append('{}'.format(t)); 

# Plot raw data
fig, ax = plt.subplots()
y_pos = np.arange(8)
ax.errorbar(y,y_pos, xerr=sigma, fmt='o')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
plt.title('8 schools')
plt.savefig('../figures/schools8_data.png')
plt.show()

# + [markdown] id="vcAdKbnXHsKE"
# # Centered model

# + id="-Lxa_JgfQmAI" colab={"base_uri": "https://localhost:8080/", "height": 723} outputId="573cdde1-a178-4949-de75-af036d02f6dd"
# Centered model
with pm.Model() as Centered_eight:
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=5)
    sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=5)
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=J)
    obs = pm.Normal('obs', mu=alpha, sigma=sigma, observed=y)
    log_sigma_alpha = pm.Deterministic('log_sigma_alpha', tt.log(sigma_alpha))
    
np.random.seed(0)
with Centered_eight:
    trace_centered = pm.sample(1000, chains=4, return_inferencedata=False)
    
pm.summary(trace_centered).round(2)
# PyMC3 gives multiple warnings about  divergences
# Also, see r_hat ~ 1.01, ESS << nchains*1000, especially for sigma_alpha
# We can solve these problems below by using a non-centered parameterization.
# In practice, for this model, the results are very similar.


# + id="pOrDPo_lQob_" colab={"base_uri": "https://localhost:8080/"} outputId="0cbd7421-2754-43c2-a468-7250ae30b8d1"
# Display the total number and percentage of divergent chains
diverging = trace_centered['diverging']
print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
diverging_pct = diverging.nonzero()[0].size / len(trace_centered) * 100
print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))

# + id="bYbhbC-kT8GV" outputId="77b27048-57ad-456c-f6ea-7bbeee7d1d94" colab={"base_uri": "https://localhost:8080/"}
dir(trace_centered)

# + id="9ODVo7cLUKs8" outputId="505c9b7c-6b7f-4b12-be22-c67809d19641" colab={"base_uri": "https://localhost:8080/"}
trace_centered.varnames

# + id="gClLFgqHVuW1" outputId="7447a76c-0e85-4d11-ca0a-fd24babe57dd" colab={"base_uri": "https://localhost:8080/", "height": 356}
with Centered_eight:
  #fig, ax = plt.subplots()
  az.plot_autocorr(trace_centered, var_names=['mu_alpha', 'sigma_alpha'], combined=True);
  plt.savefig('schools8_centered_acf_combined.png', dpi=300)

# + id="uWPD88BxTkMj" outputId="ed94b053-2ebc-41f1-91c3-12f0d7eec423" colab={"base_uri": "https://localhost:8080/", "height": 452}
with Centered_eight:
  #fig, ax = plt.subplots()
  az.plot_autocorr(trace_centered, var_names=['mu_alpha', 'sigma_alpha']);
  plt.savefig('schools8_centered_acf.png', dpi=300)

# + id="Uv1QEiQOQtGc" colab={"base_uri": "https://localhost:8080/", "height": 370} outputId="7ce96252-9002-4f18-a64c-c55046f5415d"
with Centered_eight:
  az.plot_forest(trace_centered, var_names="alpha", 
                hdi_prob=0.95, combined=True);
  plt.savefig('schools8_centered_forest_combined.png', dpi=300)

# + id="cgzmwxVGZxub" outputId="8979ca4c-d9df-43bb-847e-bad33b2258bb" colab={"base_uri": "https://localhost:8080/", "height": 542}
with Centered_eight:
  az.plot_forest(trace_centered, var_names="alpha", 
                hdi_prob=0.95, combined=False);
  plt.savefig('schools8_centered_forest.png', dpi=300)

# + [markdown] id="BkphbYr_HxOj"
# # Non-centered

# + id="jLFiQS0ZQvR4" colab={"base_uri": "https://localhost:8080/", "height": 905} outputId="8c0caa4b-4aa4-4685-f8ef-ef23ba60b82c"
# Non-centered parameterization

with pm.Model() as NonCentered_eight:
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=5)
    sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=5)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=J)
    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
    #alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=J)
    obs = pm.Normal('obs', mu=alpha, sigma=sigma, observed=y)
    log_sigma_alpha = pm.Deterministic('log_sigma_alpha', tt.log(sigma_alpha))
    
np.random.seed(0)
with NonCentered_eight:
    trace_noncentered = pm.sample(1000, chains=4)
    
pm.summary(trace_noncentered).round(2)
# Samples look good: r_hat = 1, ESS ~= nchains*1000

# + id="RyB5Qu-MQxuM" colab={"base_uri": "https://localhost:8080/", "height": 356} outputId="4a21b628-5b80-4ae4-a148-a208f33d6d43"
with NonCentered_eight:
  az.plot_autocorr(trace_noncentered, var_names=['mu_alpha', 'sigma_alpha'], combined=True);
  plt.savefig('schools8_noncentered_acf_combined.png', dpi=300)

# + id="JHmvYgsAQzuK" colab={"base_uri": "https://localhost:8080/", "height": 370} outputId="5ed95cc6-49b8-4bc6-acca-59f7c5f5c06b"
with NonCentered_eight:
  az.plot_forest(trace_noncentered, var_names="alpha",
                combined=True, hdi_prob=0.95);
  plt.savefig('schools8_noncentered_forest_combined.png', dpi=300)

# + id="vb8tzwUhXlW0" colab={"base_uri": "https://localhost:8080/", "height": 568} outputId="efad1751-55c1-4d1d-97b8-198f67af8935"
az.plot_forest([trace_centered, trace_noncentered], model_names=['centered', 'noncentered'],
               var_names="alpha",
               combined=True, hdi_prob=0.95);
plt.axvline(np.mean(y), color='k', linestyle='--')

# + id="JETMmNSuZUV7" colab={"base_uri": "https://localhost:8080/", "height": 647} outputId="835e3d2c-7874-41b5-d22e-d64e18fae9ab"
az.plot_forest([trace_centered, trace_noncentered], model_names=['centered', 'noncentered'],
               var_names="alpha", kind='ridgeplot',
               combined=True, hdi_prob=0.95);

# + [markdown] id="Q_SYYgL0H13G"
# # Funnel of hell

# + id="E3CtP2kcT4s5" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="17af872c-3d56-48e6-be05-a5aab0b4aa39"

# Plot the "funnel of hell"
# Based on
# https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/GLM_hierarchical_non_centered.ipynb

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
x = pd.Series(trace_centered['mu_alpha'], name='mu_alpha')
y  = pd.Series(trace_centered['log_sigma_alpha'], name='log_sigma_alpha')
axs[0].plot(x, y, '.');
axs[0].set(title='Centered', xlabel='µ', ylabel='log(sigma)');
#axs[0].axhline(0.01)

x = pd.Series(trace_noncentered['mu_alpha'], name='mu')
y  = pd.Series(trace_noncentered['log_sigma_alpha'], name='log_sigma_alpha')
axs[1].plot(x, y, '.');
axs[1].set(title='NonCentered', xlabel='µ', ylabel='log(sigma)');
#axs[1].axhline(0.01)

plt.savefig('schools8_funnel.png', dpi=300)

xlim = axs[0].get_xlim()
ylim = axs[0].get_ylim()

# + id="EMOdWlU-Q13N" colab={"base_uri": "https://localhost:8080/", "height": 953} outputId="0125ea26-646a-4b29-8a69-7fc508ac5d66"


x = pd.Series(trace_centered['mu_alpha'], name='mu')
y = pd.Series(trace_centered['log_sigma_alpha'], name='log sigma_alpha')
sns.jointplot(x, y, xlim=xlim, ylim=ylim);
plt.suptitle('centered')
plt.savefig('schools8_centered_joint.png', dpi=300)

x = pd.Series(trace_noncentered['mu_alpha'], name='mu')
y = pd.Series(trace_noncentered['log_sigma_alpha'], name='log sigma_alpha')
sns.jointplot(x, y, xlim=xlim, ylim=ylim);
plt.suptitle('noncentered')
plt.savefig('schools8_noncentered_joint.png', dpi=300)

# + id="qAfA7fIWWN9B" colab={"base_uri": "https://localhost:8080/", "height": 351} outputId="9a307f3d-bee9-4ce9-e219-c7b847dc5f78"
group = 0
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,5))
x = pd.Series(trace_centered['alpha'][:, group], name=f'alpha {group}')
y  = pd.Series(trace_centered['log_sigma_alpha'], name='log_sigma_alpha')
axs[0].plot(x, y, '.');
axs[0].set(title='Centered', xlabel=r'$\alpha_0$', ylabel=r'$\log(\sigma_\alpha)$');

x = pd.Series(trace_noncentered['alpha'][:,group], name=f'alpha {group}')
y  = pd.Series(trace_noncentered['log_sigma_alpha'], name='log_sigma_alpha')
axs[1].plot(x, y, '.');
axs[1].set(title='NonCentered', xlabel=r'$\alpha_0$', ylabel=r'$\log(\sigma_\alpha)$');

xlim = axs[0].get_xlim()
ylim = axs[0].get_ylim()

plt.savefig('schools8_funnel_group0.png', dpi=300)

# + id="4AOjRfRijXeA"

