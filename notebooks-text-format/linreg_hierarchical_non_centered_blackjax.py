# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: blackjax
#     language: python
#     name: blackjax
# ---

# + id="beJqsooVgm8h"
# !pip install -q blackjax
# !pip install -q distrax

# + id="critical-reading"
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.random import PRNGKey, split

import distrax
from tensorflow_probability.substrates.jax.distributions import HalfCauchy

import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import arviz as az
from functools import partial

sns.set_style('whitegrid')
np.random.seed(123)

# + id="c4BgCIlclQXX"
url = 'https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/radon.csv?raw=true'
data = pd.read_csv(url)

# + id="17ISOnzPlSR1" colab={"base_uri": "https://localhost:8080/"} outputId="9ce741fc-6e9c-47f3-9126-ec2a810a4bd6"
county_names = data.county.unique()
county_idx = jnp.array(data.county_code.values)
n_counties = len(county_names)
X = data.floor.values
Y = data.log_radon.values


# + id="zfY_BSYBRn1v"
def init_non_centered_params(n_counties, rng_key=None):
  params = {}

  if rng_key is None:
    rng_key = PRNGKey(0)

  mu_a_key, mu_b_key, sigma_a_key, sigma_b_key, a_key, b_key, eps_key = split(rng_key, 7)
  half_cauchy = distrax.as_distribution(HalfCauchy(loc=0., scale=5.))

  params["mu_a"] =  distrax.Normal(0., 1.).sample(seed=mu_a_key)
  params["mu_b"] =  distrax.Normal(0.,  1.).sample(seed=mu_b_key)

  params["sigma_a"] = half_cauchy.sample(seed=sigma_a_key)
  params["sigma_b"] = half_cauchy.sample(seed=sigma_b_key)

  params["a_offsets"] = distrax.Normal(0., 1.).sample(seed=a_key, sample_shape=(n_counties,))
  params["b_offsets"] = distrax.Normal(0., 1.).sample(seed=b_key, sample_shape=(n_counties,))

  params["eps"] = half_cauchy.sample(seed=eps_key)
  
  return params


# + id="9LrmKtgFPrcN"
def init_centered_params(n_counties, rng_key=None):
  params = {}

  if rng_key is None:
    rng_key = PRNGKey(0)

  mu_a_key, mu_b_key, sigma_a_key, sigma_b_key, a_key, b_key, eps_key = split(rng_key, 7)
  half_cauchy = distrax.as_distribution(HalfCauchy(loc=0., scale=5.))

  params["mu_a"] =  distrax.Normal(0., 1.).sample(seed=mu_a_key) 
  params["mu_b"] =  distrax.Normal(0., 1.).sample(seed=mu_b_key)

  params["sigma_a"] = half_cauchy.sample(seed=sigma_a_key)
  params["sigma_b"] = half_cauchy.sample(seed=sigma_b_key)

  params["b"] = distrax.Normal(params["mu_b"], params["sigma_b"]).sample(seed=b_key, sample_shape=(n_counties,))
  params["a"] = distrax.Normal(params["mu_a"], params["sigma_a"]).sample(seed=a_key, sample_shape=(n_counties,))

  params["eps"] = half_cauchy.sample(seed=eps_key)

  return params


# + id="KUgd4mqlLLeQ"
def log_joint_non_centered(params,  X, Y, county_idx, n_counties):
    log_theta = 0

    log_theta += distrax.Normal(0., 100**2).log_prob(params['mu_a']) *n_counties
    log_theta += distrax.Normal(0., 100**2).log_prob(params['mu_b']) *n_counties

    log_theta += distrax.as_distribution(HalfCauchy(0., 5.)).log_prob(params['sigma_a']) *n_counties
    log_theta += distrax.as_distribution(HalfCauchy(0., 5.)).log_prob(params['sigma_b']) *n_counties
    
    log_theta += distrax.Normal(0., 1.).log_prob(params['a_offsets']).sum()
    log_theta += distrax.Normal(0., 1.).log_prob(params['b_offsets']).sum()


    log_theta += jnp.sum(distrax.as_distribution(HalfCauchy(0., 5.)).log_prob(params['eps']))
    
    # Linear regression
    a = params["mu_a"] +  params["a_offsets"] * params["sigma_a"]
    b = params["mu_b"] +  params["b_offsets"] * params["sigma_b"]
    radon_est = a[county_idx] + b[county_idx] * X

    log_theta += jnp.sum(distrax.Normal(radon_est, params['eps']).log_prob(Y))
    
    return -log_theta


# + id="Yo2NFH-rscIM"
def log_joint_centered(params,  X, Y, county_idx):
    log_theta = 0

    log_theta += distrax.Normal(0., 100**2).log_prob(params['mu_a']).sum()
    log_theta += distrax.Normal(0., 100**2).log_prob(params['mu_b']).sum()

    log_theta += distrax.as_distribution(HalfCauchy(0., 5.)).log_prob(params['sigma_a']).sum()
    log_theta += distrax.as_distribution(HalfCauchy(0., 5.)).log_prob(params['sigma_b']).sum()
    
    log_theta += distrax.Normal(params['mu_a'], params['sigma_a']).log_prob(params['a']).sum()
    log_theta += distrax.Normal(params['mu_b'], params['sigma_b']).log_prob(params['b']).sum()

    log_theta += distrax.as_distribution(HalfCauchy(0., 5.)).log_prob(params['eps']).sum()
    
    # Linear regression
    radon_est = params['a'][county_idx] + params['b'][county_idx] * X 
    log_theta += distrax.Normal(radon_est, params['eps']).log_prob(Y).sum()
    return -log_theta


# + id="zeQWnoYJ3Vwc"
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# + id="uRQxVO_MLLbV"
def fit_hierarchical_model(X, Y, county_idx, n_counties, is_centered=True, num_warmup = 1000, num_samples=5000, rng_key=None):
  if rng_key is None:
    rng_key = PRNGKey(0)
  
  init_key, warmup_key, sample_key = split(rng_key, 3)

  if is_centered:
    potential = partial(log_joint_centered, X=X, Y = Y, county_idx = county_idx)
    params = init_centered_params(n_counties, rng_key=init_key)
  else:
    potential = partial(log_joint_non_centered, X=X, Y = Y, county_idx = county_idx, n_counties=n_counties)
    params = init_non_centered_params(n_counties, rng_key=init_key)

  initial_state = nuts.new_state(params, potential)
  
  kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
      potential, step_size, inverse_mass_matrix)
  
  last_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
      warmup_key, kernel_factory, initial_state, num_warmup)

  kernel = kernel_factory(step_size, inverse_mass_matrix)

  states = inference_loop(sample_key, kernel, initial_state, num_samples)
  return states


# + colab={"base_uri": "https://localhost:8080/"} id="mdbhmyS3NrUH" outputId="5ca724fb-b5f7-4da6-a3c5-8a0bb2386d0f"
states_centered = fit_hierarchical_model(X, Y, county_idx, n_counties, is_centered=True)

# + colab={"base_uri": "https://localhost:8080/"} id="l_WiobxAP5cL" outputId="4d9579ed-aacb-498d-c5ef-1013937a8909"
states_non_centered = fit_hierarchical_model(X, Y, county_idx, n_counties, is_centered=False)


# + [markdown] id="s2W9MxqvWk3t"
# ## Centered Hierarchical Model

# + id="LCPmdrxcQvZL"
def plot_funnel_of_hell(x, sigma_x, k=75):
  x = pd.Series(x[:, k].flatten(), name=f'slope b_{k}')
  y = pd.Series(sigma_x.flatten(), name='slope group variance sigma_b')

  sns.jointplot(x=x, y=y, ylim=(0., 0.7), xlim=(-2.5, 1.0));


# + colab={"base_uri": "https://localhost:8080/", "height": 441} id="SGfyN_7KQ1f6" outputId="715ab882-5c11-4ddc-8467-0ae5c342aff2"
samples_centered = states_centered.position
b_centered = samples_centered['b']
sigma_b_centered = samples_centered['sigma_b']
plot_funnel_of_hell(b_centered, sigma_b_centered)


# + id="UjhQGcp44eud"
def plot_single_chain(x, sigma_x, name):
  fig, axs = plt.subplots(nrows=2, figsize=(16, 6))
  axs[0].plot(sigma_x, alpha=.5);
  axs[0].set(ylabel=f'sigma_{name}');
  axs[1].plot(x, alpha=.5);
  axs[1].set(ylabel=name);


# + colab={"base_uri": "https://localhost:8080/", "height": 374} id="5vbyeIFgSFF_" outputId="c5979ba6-a2e7-44e7-8d61-d63c3ed06329"
plot_single_chain(b_centered[1000:], sigma_b_centered[1000:], "b")

# + [markdown] id="xfQbQOsUWU4a"
# ## Non-Centered Hierarchical Model

# + id="6dvn4O8SSgGN"
samples_non_centered = states_non_centered.position
b_non_centered = samples_non_centered['mu_b'][..., None] + samples_non_centered['b_offsets'] * samples_non_centered['sigma_b'][..., None]
sigma_b_non_centered = samples_non_centered['sigma_b']

# + colab={"base_uri": "https://localhost:8080/", "height": 441} id="u_Je-ZgES7pM" outputId="fde14332-c1a9-47cd-998d-decc730a9f67"
plot_funnel_of_hell(b_non_centered, sigma_b_non_centered)

# + colab={"base_uri": "https://localhost:8080/", "height": 374} id="cPte8Q1YS9Hv" outputId="44c2c411-8cb7-4c8e-a71d-4f498a6c49b0"
plot_single_chain(b_non_centered[1000:], sigma_b_non_centered[1000:], "b")

# + [markdown] id="dFXHiIxiVwRS"
# ## Comparison

# + colab={"base_uri": "https://localhost:8080/", "height": 404} id="5MREcS8qBQuX" outputId="1a67fdcf-5117-4060-8f27-6396de0eade6"
k = 75
x_lim, y_lim = [-2.5, 1], [0, 0.7]

bs = [(b_centered, sigma_b_centered, 'Centered'), (b_non_centered, sigma_b_non_centered, 'Non-centered')]
ncols = len(bs)

fig, axs = plt.subplots(ncols=ncols, sharex=True, sharey=True, figsize=(8, 6))

for i, (b, sigma_b, model_name) in enumerate(bs):
  x = pd.Series(b[:, k], name=f'slope b_{k}')
  y = pd.Series(sigma_b, name='slope group variance sigma_b')
  axs[i].plot(x, y, '.');
  axs[i].set(title=model_name, ylabel='sigma_b', xlabel=f'b_{k}')
  axs[i].set_xlim(x_lim)
  axs[i].set_ylim(y_lim)
