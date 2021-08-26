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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/prior_post_predictive_binomial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="BVqejBGvj0bD"
# Plot rior and posterior predctiive for beta binomial distribution.
# Based on  fig 1.6 of 'Bayesian Modeling and Computation'.
# Code currently uses pymc3, although it could be done analytically.

# + id="G3cEAhF5gWoX"
# !git clone https://github.com/probml/pyprobml /pyprobml &> /dev/null
# %cd -q /pyprobml/scripts
import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt

# + id="47Q9AnzKga6B" colab={"base_uri": "https://localhost:8080/"} outputId="d413e387-73e6-4090-ad91-59566d0be177"
# #!pip install pymc3 # colab uses 3.7 by default (as of April 2021)

 # arviz needs 3.8+
# #!pip install pymc3>=3.8 # fails to update
# !pip install pymc3==3.11 

import pymc3 as pm
print(pm.__version__)

# #!pip install arviz
import arviz as az
print(az.__version__)

# + colab={"base_uri": "https://localhost:8080/", "height": 256} id="lIYdn1woOS1n" outputId="00757b40-2953-47ec-9beb-1fe5002fb3a6"


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
from scipy.stats import entropy
from scipy.optimize import minimize

import pyprobml_utils as pml


np.random.seed(0)
Y = stats.bernoulli(0.7).rvs(20)


with pm.Model() as model:
    θ = pm.Beta("θ", 1, 1)
    y_obs = pm.Binomial("y_obs",n=1, p=θ, observed=Y)
    idata = pm.sample(1000, return_inferencedata=True)
    
    
pred_dists = (pm.sample_prior_predictive(1000, model)["y_obs"],
              pm.sample_posterior_predictive(idata, 1000, model)["y_obs"])


# + colab={"base_uri": "https://localhost:8080/"} id="p8oKY1-1lJim" outputId="46a27be8-f529-468b-cffd-e1a6fa61cc0f"
dist=pred_dists[0]
print(dist.shape)
num_success = dist.sum(1)
print(num_success.shape)


# + colab={"base_uri": "https://localhost:8080/", "height": 330} id="TwZN8R4Rlyy6" outputId="cc3caef3-e7cb-462a-a297-41db1798f91e"

fig, ax = plt.subplots()
az.plot_dist(pred_dists[0].sum(1), hist_kwargs={"color":"0.5", "bins":range(0, 22)})
ax.set_title(f"Prior predictive distribution",fontweight='bold')
ax.set_xlim(-1, 21)
ax.set_ylim(0, 0.15) 
ax.set_xlabel("number of success")

# + colab={"base_uri": "https://localhost:8080/", "height": 330} id="3aMkcUljmQC-" outputId="322acfee-6b70-4fc7-f2a4-a7af24c535b2"
fig, ax = plt.subplots()
az.plot_dist(pred_dists[1].sum(1), hist_kwargs={"color":"0.5", "bins":range(0, 22)})
ax.set_title(f"Posterior predictive distribution",fontweight='bold')
ax.set_xlim(-1, 21)
ax.set_ylim(0, 0.15) 
ax.set_xlabel("number of success")

# + id="qe1AhaYKg-4p" colab={"base_uri": "https://localhost:8080/", "height": 321} outputId="a2ba11a9-a17f-4b44-eb52-1a4ce71d8b72"
fig, ax = plt.subplots()
az.plot_dist(θ.distribution.random(size=1000), plot_kwargs={"color":"0.5"},
             fill_kwargs={'alpha':1})
ax.set_title("Prior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ") 
    




# + colab={"base_uri": "https://localhost:8080/", "height": 321} id="YxLCEgRMmqb7" outputId="e9bb894f-b1a3-445b-a7d9-b80436459dec"
fig, ax = plt.subplots()
az.plot_dist(idata.posterior["θ"], plot_kwargs={"color":"0.5"},
             fill_kwargs={'alpha':1})
ax.set_title("Posterior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")


# + id="9x5zR_kthjCc"

