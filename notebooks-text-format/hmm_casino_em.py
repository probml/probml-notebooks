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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/hmm_casino_em.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="HfilSsrwY5vQ"
# # Fit a discrete HMM to the casino dataset using EM.

# + colab={"base_uri": "https://localhost:8080/"} id="UFkO5HznXH7A" outputId="9e923d22-8083-4b99-975f-82f0e3424113"
# !pip install flax

# + id="wYyi7HOPX6RW"
# !git clone https://github.com/probml/pyprobml /pyprobml &> /dev/null
# %cd -q /pyprobml/scripts

# + colab={"base_uri": "https://localhost:8080/", "height": 16} id="jpbr2G7meEsZ" outputId="909b12c2-54c0-4af6-d67e-b18c57f087e7"
file = 'hmm_discrete_lib.py' 
# !touch $file # create empty file if does not already exist
from google.colab import files
files.view(file) # open editor

# + id="5djokYRSX8G8"

import jax.numpy as jnp
from jax.random import split, PRNGKey, randint

import numpy as np

from hmm_discrete_lib import HMMNumpy, HMMJax, hmm_sample_jax
from hmm_discrete_lib import hmm_plot_graphviz

from hmm_discrete_em_lib import init_random_params_jax
from hmm_discrete_em_lib import hmm_em_numpy, hmm_em_jax

import hmm_utils

import time

import graphviz 
from graphviz import Digraph


# + [markdown] id="_0-b-82ZY912"
# # Generate data from the true model

# + colab={"base_uri": "https://localhost:8080/"} id="JY0_lo2gZBIt" outputId="8a93de21-3105-454b-f4c6-22f8b9ab4513"
A = jnp.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

# observation matrix
B = jnp.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

pi = jnp.array([1, 1]) / 2

seed = 42 #100
rng_key = PRNGKey(seed)
rng_key, rng_sample, rng_batch, rng_init = split(rng_key, 4)

casino = HMMJax(A, B, pi)

n_obs_seq, max_len = 5, 3000

observations, lens = hmm_utils.hmm_sample_n(casino,
                                            hmm_sample_jax,
                                            n_obs_seq, max_len,
                                            rng_sample)

print(observations.shape)
print(lens)
print(observations[0,1700:1750])
print(np.sum(lens))

observations, lens = hmm_utils.pad_sequences(observations, lens)

print(observations.shape)
print(lens)
print(observations[0,1700:1750])
print(np.sum(lens))

# + colab={"base_uri": "https://localhost:8080/", "height": 401} id="diJKWJwOadYt" outputId="d798163b-8c2f-4b43-c70b-9b4f32618ba3"

state_names, obs_names = ['Fair Dice', 'Loaded Dice'], [str(i+1) for i in range(B.shape[1])]

dot = hmm_plot_graphviz(casino, '../figures/hmm_casino_true', state_names, obs_names)
dot

# + [markdown] id="HPgj6zlFZMGt"
# # Fit model

# + colab={"base_uri": "https://localhost:8080/", "height": 401} id="7XufHxPcZM06" outputId="6229c2f0-e2b3-456e-cbfa-daab21a96f37"
# Initialize model randomly

n_hidden, n_obs = B.shape
params_jax = init_random_params_jax([n_hidden, n_obs], rng_key=rng_init)

dot = hmm_plot_graphviz(params_jax, '../figures/hmm_casino_init', state_names, obs_names)
dot





# + id="D0i3qZqybePe"


num_epochs = 20
params_jax, neg_ll_jax = hmm_em_jax(observations,
                                    lens,
                                    num_epochs=num_epochs,
                                    init_params=params_jax)

# + [markdown] id="hAiP9s7iZd9q"
# # Plot results

# + colab={"base_uri": "https://localhost:8080/", "height": 294} id="tsdC0frTZX38" outputId="6c0d5370-93b9-425c-922e-d459c4233b8f"
hmm_utils.plot_loss_curve(neg_ll_jax, "EM JAX")


# + colab={"base_uri": "https://localhost:8080/", "height": 401} id="SMns-aZTZeoe" outputId="20f5b5b2-dfd2-4d5f-d12f-d9f58b58b94f"


dot = hmm_plot_graphviz(params_jax, '../figures/hmm_casino_em', state_names, obs_names)
dot

# + id="3Fr78xhIaGAv"

