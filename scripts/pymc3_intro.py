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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/pymc3_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="2RxQaDAWbxe3"
# # Brief introduction to PyMC3
#
# [PyMC3]((https://docs.pymc.io/) is a library that lets the user specify certain kinds of joint probability models using a Python API, that has the "look and feel" similar to the standard way of present hierarchical Bayesian models. Once the (log) joint is defined, it can be used for posterior inference, using either various algorithms, including Hamiltonian Monte Carlo (HMC), and automatic differentiation variational inference (ADVI). More details can be found on the [PyMC3 web page](https://docs.pymc.io/), and in the book [Bayesian Analysis with Python (2nd end)](https://github.com/aloctavodia/BAP) by Osvaldo Martin.

# + colab={"base_uri": "https://localhost:8080/"} id="GWcsxoWPbt10" outputId="031affd8-90ea-4233-ddb4-d3c7ab51eb1b"
#import pymc3 # colab uses 3.7 by default (as of April 2021)
# arviz needs 3.8+

# #!pip install pymc3>=3.8 # fails to update
# #!pip install pymc3==3.11 # latest number is hardcoded

# !pip install -U pymc3>=3.8

import pymc3 as pm
print(pm.__version__)


# + id="MXG7Ko35and4" colab={"base_uri": "https://localhost:8080/"} outputId="67ef70c9-b69d-4908-940f-b35a49db4628"
# #!pip install arviz
import arviz as az
print(az.__version__)

# + id="yoiJjFzfbm7E"
import sklearn
import scipy.stats as stats
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os
import pandas as pd

# + [markdown] id="HMvsbhRzcKQb"
# ## Example: 1d Gaussian with unknown mean.
#
# We use the simple example from the [Pyro intro](https://pyro.ai/examples/intro_part_ii.html#A-Simple-Example). The goal is to infer the weight $\theta$ of an object, given noisy measurements $y$. We assume the following model:
# $$
# \begin{align}
# \theta &\sim N(\mu=8.5, \tau^2=1.0)\\ 
# y \sim &N(\theta, \sigma^2=0.75^2)
# \end{align}
# $$
#
# Where $\mu=8.5$ is the initial guess. 
#
# By Bayes rule for Gaussians, we know that the exact posterior,
# given a single observation $y=9.5$, is given by
#
#
# $$
# \begin{align}
# \theta|y &\sim N(m, s^s) \\
# m &=\frac{\sigma^2 \mu + \tau^2 y}{\sigma^2 + \tau^2} 
#   = \frac{0.75^2 \times 8.5 + 1 \times 9.5}{0.75^2 + 1^2}
#   = 9.14 \\
# s^2 &= \frac{\sigma^2 \tau^2}{\sigma^2  + \tau^2} 
# = \frac{0.75^2 \times 1^2}{0.75^2 + 1^2}= 0.6^2
# \end{align}
# $$

# + colab={"base_uri": "https://localhost:8080/"} id="PPP1ntigbuJ7" outputId="5a0354d6-0359-4cfe-e484-e9d65d0a0f2f"
mu = 8.5; tau = 1.0; sigma = 0.75; y = 9.5
m = (sigma**2 * mu + tau**2 * y)/(sigma**2 + tau**2)
s2 = (sigma**2 * tau**2)/(sigma**2 + tau**2)
s = np.sqrt(s2)
print(m)
print(s)

# + id="oXiCXwu-b0wC"
# Specify the model

with pm.Model() as model:
  theta = pm.Normal('theta', mu=mu, sd=tau)
  obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)


# + [markdown] id="eKCu4-ASgitu"
# #MCMC inference

# + colab={"base_uri": "https://localhost:8080/", "height": 201} id="T2sjApcNglyv" outputId="33eeb32f-9fb9-4625-f9ad-8172f12e4c5b"

# run MCMC (defaults to using the NUTS algorithm with 2 chains)
with model:
    trace = pm.sample(1000, random_seed=123)

# + id="PX_WnXUedLtr" colab={"base_uri": "https://localhost:8080/", "height": 133} outputId="752b1242-6a40-4e6f-a89a-eef17075261b"
az.summary(trace)

# + id="9QtAFGCCa9EY" colab={"base_uri": "https://localhost:8080/"} outputId="56a93fd1-0972-4d80-83bc-1444185003ff"
trace

# + colab={"base_uri": "https://localhost:8080/"} id="KPBqtmT-cngA" outputId="c06e6859-ae6a-4b99-9169-ead89ecd677a"
samples = trace['theta']
print(samples.shape)
post_mean = np.mean(samples)
post_std = np.std(samples)
print([post_mean, post_std])

# + [markdown] id="K_7UdM0GdTrG"
# With PyMC3 version >=3.9 the return_inferencedata=True kwarg makes the sample function return an arviz.InferenceData object instead of a MultiTrace.

# + id="hcjDzkgUdbI5" colab={"base_uri": "https://localhost:8080/", "height": 181} outputId="83f30956-2815-444f-b06f-fc93ef26b098"
with model:
    idata = pm.sample(1000, random_seed=123, return_inferencedata=True)

# + id="pkW2gNZKdh4a" colab={"base_uri": "https://localhost:8080/", "height": 548} outputId="05ae0c15-d40e-4516-f165-eb8399340cbd"
idata

# + id="0MCnAW4MeLOA" colab={"base_uri": "https://localhost:8080/", "height": 168} outputId="b4c90f79-44d9-47b8-bd4a-31f979ce79f5"
az.plot_trace(idata);

# + [markdown] id="3tikWKg1gmsa"
# # Variational inference
#
#
# We use automatic differentiation VI.
# Details can be found at https://docs.pymc.io/notebooks/variational_api_quickstart.html

# + id="HxSBz9_Jc-7n" colab={"base_uri": "https://localhost:8080/", "height": 54} outputId="cf41255e-f2dc-424d-a6c1-b54d17f42314"
niter = 10000
with model:
    post = pm.fit(niter, method='advi'); # mean field approximation

# + colab={"base_uri": "https://localhost:8080/", "height": 264} id="xafuviKGkNpM" outputId="33405f5d-6ed2-4744-ad1f-f3185b6f3957"
# Plot negative ELBO vs iteration to assess convergence
plt.plot(post.hist);

# + id="vcJSqddOetoA"
# convert analytic posterior to a bag of iid samples
trace = post.sample(10000)


# + id="PDDFzDFser4a" colab={"base_uri": "https://localhost:8080/"} outputId="e6c9f1b6-da9d-445c-860d-bfef6201df30"
samples = trace['theta']
print(samples.shape)
post_mean = np.mean(samples)
post_std = np.std(samples)
print([post_mean, post_std])

# + id="f0O9xJtleR0A" colab={"base_uri": "https://localhost:8080/", "height": 151} outputId="7c157322-0105-40fe-e8d2-7d74a4d909f8"
az.summary(trace)

# + [markdown] id="ymQiSerCgul7"
# # PyMc3 Libraries
#
# There are various libraries that extend pymc3, or use it in various ways, some of which we list below.
#
# - The [arviz](https://github.com/arviz-devs/arviz) library can be used to |visualize (and diagonose problems with) posterior samples drawn from many libraries, including PyMc3.
#
# - The [bambi](https://bambinos.github.io/bambi/) library lets the user specify linear models using "formula syntax", similar to R.
#
# - The [PyMc-learn](https://pymc-learn.readthedocs.io/en/latest/) library offers a sklearn-style API to specify models, but uses PyMc3 under the hood to compute posteriors for model parameters, instead of just point estimates.
#
#
#

# + id="SGwIH0V-hell"

