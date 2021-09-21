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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/bnn_hmc_gaussian.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="x_FQOPfkt0p-"
# # (SG)HMC for inferring params of a 2d Gaussian
#
# Based on 
#
# https://github.com/google-research/google-research/blob/master/bnn_hmc/notebooks/mcmc_gaussian_test.ipynb
#

# + colab={"base_uri": "https://localhost:8080/"} id="DmflvLF2vF33" outputId="8fe23041-f2ad-4264-ba5d-e1ece4f564c5"
import jax
print(jax.devices())


# + colab={"base_uri": "https://localhost:8080/"} id="9yPvHi3dtOkI" outputId="86133c6e-72d9-41a6-da6e-66ca9b15d799"
# !git clone https://github.com/google-research/google-research.git

# + colab={"base_uri": "https://localhost:8080/"} id="Sj4tiAWWtYdp" outputId="dbb50fda-b8bb-4159-ae57-d695bf60f4c0"
# %cd /content/google-research

# + colab={"base_uri": "https://localhost:8080/"} id="TQBRXFyGuUuH" outputId="704d91f4-261d-4e0e-85db-1ac7dd697cc5"
# !ls bnn_hmc

# + colab={"base_uri": "https://localhost:8080/"} id="A22SEy8juI0J" outputId="2be3ebee-bb77-4d96-bc13-2c2188ccfe7e"
# !pip install optax

# + [markdown] id="I9m5GSFGt-1p"
# # Setup

# + id="LTD8GR-a2g7T"
from jax.config import config
import jax
from jax import numpy as jnp
import numpy as onp
import numpy as np

# + id="3zxTscnTtrMw"

import os
import sys

import time
import tqdm
import optax
import functools
from matplotlib import pyplot as plt

from bnn_hmc.utils import losses
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import tree_utils

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# + [markdown] id="UaHbJqoMuGAB"
# # Data and model

# + id="nz3ecCvjuGsh"

mu = jnp.zeros([2,])
# sigma = jnp.array([[1., .5], [.5, 1.]])
sigma = jnp.array([[1.e-4, 0], [0., 1.]])
sigma_l = jnp.linalg.cholesky(sigma)
sigma_inv = jnp.linalg.inv(sigma)
sigma_det = jnp.linalg.det(sigma)

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="t3HMpwVwucXS" outputId="f824662a-a65e-4e0c-e903-798c5552c7c6"
onp.random.seed(0)
samples = onp.random.multivariate_normal(onp.asarray(mu), onp.asarray(sigma), size=1000)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
plt.grid()


# + id="0zE7kTLoucfv"

def log_density_fn(params):
    assert params.shape == mu.shape, "Shape error"
    diff = params - mu
    
    k = mu.size
    
    log_density = -jnp.log(2 * jnp.pi) * k / 2
    log_density -= jnp.log(sigma_det) / 2
    log_density -= diff.T @ sigma_inv @ diff / 2
    return log_density


# + id="PczV72Jluepq"
def log_likelihood_fn(_, params, *args, **kwargs):
    return log_density_fn(params), jnp.array(jnp.nan)

def log_prior_fn(_):
    return 0.

def log_prior_diff_fn(*args):
    return 0.


# + id="ET_46VApugW8"
fake_net_apply = None
fake_data = jnp.array([[jnp.nan,],]), jnp.array([[jnp.nan,],])
fake_net_state = jnp.array([jnp.nan,])

# + [markdown] id="CyRAU2VWui1K"
# # HMC
#

# + colab={"base_uri": "https://localhost:8080/"} id="zfeaDJUXujk8" outputId="c83149ba-3162-49f5-eeb8-4818b637df06"

step_size = 1e-1
trajectory_len = jnp.pi / 2
max_num_leapfrog_steps = int(trajectory_len // step_size + 1)
print("Leapfrog steps per iteration:", max_num_leapfrog_steps)

# + id="BumEky7pumuI"
update, get_log_prob_and_grad = train_utils.make_hmc_update(
    fake_net_apply, log_likelihood_fn, log_prior_fn, log_prior_diff_fn,
    max_num_leapfrog_steps, 1., 0.)

# + id="Zc0qG7rwuohu"
# Initial log-prob and grad values
# params = jnp.ones_like(mu)[None, :]
params = jnp.ones_like(mu)
log_prob, state_grad, log_likelihood, net_state = (
    get_log_prob_and_grad(fake_data, params, fake_net_state))

# + colab={"base_uri": "https://localhost:8080/"} id="63s5gkqSuqNc" outputId="67087786-6647-466f-ea2c-c6f173d739ba"
# %%time 
num_iterations = 500
all_samples = []
key = jax.random.PRNGKey(0)

for iteration in tqdm.tqdm(range(num_iterations)):

    (params, net_state, log_likelihood, state_grad, step_size, key,
     accept_prob, accepted) = (
        update(fake_data, params, net_state, log_likelihood, state_grad,
               key, step_size, trajectory_len, True))
    
    if accepted:
        all_samples.append(onp.asarray(params).copy())

#     print("It: {} \t Accept P: {} \t Accepted {} \t Log-likelihood: {}".format(
#             iteration, accept_prob, accepted, log_likelihood))

# + colab={"base_uri": "https://localhost:8080/"} id="IEQOxzHkwgE-" outputId="a6677d0d-d4e1-435f-854a-49c9c1d4b629"
len(all_samples)

# + colab={"base_uri": "https://localhost:8080/"} id="a6BSqDajw9w7" outputId="06423869-2c93-44f7-ef47-7803104bd9e9"
log_prob, state_grad, log_likelihood, net_state

# + colab={"base_uri": "https://localhost:8080/", "height": 334} id="XexpVxR5us14" outputId="b4e0d80d-41cc-4caf-9c81-229e888e146d"

all_samples_cat = onp.stack(all_samples)

# + colab={"base_uri": "https://localhost:8080/", "height": 181} id="TNiJUXPiwujV" outputId="f4102920-a128-4b2f-d5bb-491ad11641e7"
plt.scatter(all_samples_cat[:, 0], all_samples_cat[:, 1], alpha=0.3)
plt.grid()

# + [markdown] id="lyuEnif5zdY9"
# # Blackjax

# + colab={"base_uri": "https://localhost:8080/"} id="N29quFmSxJ5U" outputId="92507c2b-8cf6-432f-fa84-a330f015b581"
# !pip install blackjax

# + id="Lcwa1EcBzfiF"

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

# + colab={"base_uri": "https://localhost:8080/"} id="Pog5ZEqb2TTW" outputId="eada3760-efe0-498e-ab4f-420e3c33278d"
print(jax.devices())

# + id="u486wcKo1OpB"
potential = lambda x:  -log_density_fn(**x)

# + colab={"base_uri": "https://localhost:8080/"} id="eKG5_84WznYB" outputId="92c0b14a-f444-4e4d-844c-19b04432bcea"
num_integration_steps = 30
kernel_generator = lambda step_size, inverse_mass_matrix: hmc.kernel(
    potential, step_size, inverse_mass_matrix, num_integration_steps
)

rng_key = jax.random.PRNGKey(0)


initial_position = {"params": np.zeros(2)}
initial_state = hmc.new_state(initial_position, potential)
print(initial_state)


# + colab={"base_uri": "https://localhost:8080/"} id="QYaG5YpDzqKL" outputId="d6062d2b-27f5-4bb9-a8a5-0475b234d678"
# %%time
nsteps = 500
final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(
    rng_key,
    kernel_generator,
    initial_state,
    nsteps,
)

# + colab={"base_uri": "https://localhost:8080/"} id="dCVbjYQj0Pwb" outputId="84c2d93d-0205-46a1-80a6-b913c6710ca1"
# %%time 
kernel = nuts.kernel(potential, step_size, inverse_mass_matrix)
kernel = jax.jit(kernel)


# + id="0sFQRpff1q-Y"
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# + colab={"base_uri": "https://localhost:8080/"} id="LAPSi4Nx1mHt" outputId="e162efdf-f0a2-40bd-9138-963560dd1ff0"
# %%time
nsamples = 500

states = inference_loop(rng_key, kernel, initial_state, nsamples)

samples = states.position["params"].block_until_ready()
print(samples.shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 269} id="dulLJK0q1xDo" outputId="c12f244d-3d65-4aa2-ee35-bf8bf8874d0c"
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
plt.grid()

# + id="8hKLqA5W221K"

