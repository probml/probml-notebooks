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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/linreg_height_weight_numpyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="E80gttx4w4kf"
# # Linear regression for predicting height from weight
#
# We illustrate priors for linear and polynomial regression using the example in sec 4.4  of [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). 
# The numpyro code is from [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/04-geocentric-models.html).
#

# + colab={"base_uri": "https://localhost:8080/"} id="4n3ivs7KxKJL" outputId="755e3fe6-2a47-4750-9c51-35b9bd3dd43e"
# !pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
# !pip install -q arviz

# + colab={"base_uri": "https://localhost:8080/"} id="zxRRFEI5xDiO" outputId="5bcf1216-0f6a-4e09-9e22-20b15f2d7859"

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import os
import warnings
import pandas as pd

import jax
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))

import jax.numpy as jnp
from jax import random, vmap

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform
from numpyro.diagnostics import hpdi, print_summary
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.optim as optim


import arviz as az


# + [markdown] id="Qz4fcrU3xWbK"
# # Data
#
#
# We use the "Howell" dataset, which consists of measurements of height, weight, age and sex, of a certain foraging tribe, collected by Nancy Howell.

# + colab={"base_uri": "https://localhost:8080/"} id="EOs0T3_0xhxv" outputId="1a1f1fa1-ff2f-41c3-c93e-16910bbe44c8"
#url = 'https://github.com/fehiepsi/rethinking-numpyro/tree/master/data/Howell1.csv?raw=True'
url = 'https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/Howell1.csv'

Howell1 = pd.read_csv(url, sep=';')
d = Howell1
d.info()
d.head()

# get data for adults
d2 = d[d.age >= 18]

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="5-dn4nqVwfZX" outputId="9595ca54-54e0-4d5c-8e2b-a1e8cf3fc9ca"
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"))
plt.tight_layout()
plt.savefig('linreg_height_weight_data.pdf', dpi=300)
plt.show()


# + [markdown] id="CCm__mZ4xtUX"
# # Prior predictive distribution

# + id="3YjvUSqJnVCh"
def plot_prior_predictive_samples(a, b, fname=None):
  plt.subplot(
    xlim=(d2.weight.min(), d2.weight.max()),
    ylim=(-100, 400),
    xlabel="weight",
    ylabel="height",)
  plt.axhline(y=0, c="b", ls='--', label='embryo')
  plt.axhline(y=272, c="r", ls="--",  label='world''s tallest man')
  plt.title("b ~ Normal(0, 10)")
  xbar = d2.weight.mean()
  x = jnp.linspace(d2.weight.min(), d2.weight.max(), 101)
  for i in range(N):
      plt.plot(x, a[i] + b[i] * (x - xbar), "k", alpha=0.2)
  plt.tight_layout()
  if fname:
    plt.savefig(fname, dpi=300)
  plt.show()



# + [markdown] id="DVqy18hJx7uy"
# ## Gaussian prior

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="yyCQIQXHndde" outputId="7dcf9893-cad5-45e6-b68b-fc8c75121c71"
with numpyro.handlers.seed(rng_seed=2971):
    N = 100  # 100 lines
    a = numpyro.sample("a", dist.Normal(178, 20).expand([N]))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([N]))

plot_prior_predictive_samples(a, b, 'linreg_height_weight_gauss_prior.pdf')

# + [markdown] id="P9A0OP2Ex7Gh"
# ## Log-Gaussian prior

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="BIOVE5NDn-P9" outputId="a5d81c7b-0379-41b4-99d4-8f45fa94061d"
with numpyro.handlers.seed(rng_seed=2971):
    N = 100  # 100 lines
    a = numpyro.sample("a", dist.Normal(178, 28).expand([N]))
    b = numpyro.sample("b", dist.LogNormal(0, 1).expand([N]))

plot_prior_predictive_samples(a, b, 'linreg_height_weight_loggauss_prior.pdf')


# + [markdown] id="siagmeS215bq"
# # Posterior
#
# We use the log-gaussian prior.
# We compute a Laplace approximation to the posterior.

# + colab={"base_uri": "https://localhost:8080/"} id="-Cbuw9r51-Or" outputId="c7ec3fd6-fffd-4f4b-fb95-2de903b75a4f"
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + b * (weight - xbar))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

def model2(weight, height=None): # equivalent version
    a = numpyro.sample("a", dist.Normal(178, 20))
    log_b = numpyro.sample("log_b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + jnp.exp(log_b) * (weight - xbar))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


m4_3 = AutoLaplaceApproximation(model)
svi = SVI(
    model, m4_3, optim.Adam(1), Trace_ELBO(), weight=d2.weight.values, height=d2.height.values
)
p4_3, losses = svi.run(random.PRNGKey(0), 2000)

# + colab={"base_uri": "https://localhost:8080/"} id="ZtcU4GkH4ODJ" outputId="1ad9d421-2653-432f-8aba-20ddf5e09134"
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
{latent: list(post[latent].reshape(-1)[:5]) for latent in post}

# + [markdown] id="amAKZdGu2jCm"
# # Posterior predictive

# + colab={"base_uri": "https://localhost:8080/", "height": 313} id="3miBoxvJ2w8R" outputId="6f6beca7-5ebd-47c7-b133-31ff6140715b"
# define sequence of weights to compute predictions for
# these values will be on the horizontal axis
weight_seq = jnp.arange(start=25, stop=71, step=1)

# use predictive to compute mu
# for each sample from posterior
# and for each weight in weight_seq
mu = Predictive(m4_3.model, post, return_sites=["mu"])(
    random.PRNGKey(2), weight_seq, None
)["mu"]

print(mu.shape)

# summarize the distribution of mu
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(2.5, 97.5), axis=0)
mu_HPDI = hpdi(mu, prob=0.95, axis=0)

# observed output
sim_height = Predictive(m4_3.model, post, return_sites=["height"])(
    random.PRNGKey(2), weight_seq, None
)["height"]
height_PI = jnp.percentile(sim_height, q=(2.5, 97.5), axis=0)


# plot raw data
az.plot_pair(
    d2[["weight", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5}
)

# draw MAP line
plt.plot(weight_seq, mu_mean, "k")

# draw HPDI region for line
plt.fill_between(weight_seq, mu_HPDI[0], mu_HPDI[1], color="k", alpha=0.2)

# draw PI region for simulated heights
plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="k", alpha=0.15)
plt.tight_layout()
plt.savefig('linreg_height_weight_loggauss_post.pdf', dpi=300)
plt.show()




# + [markdown] id="x6G9iWBbfgfw"
# # Polynomial regression

# + [markdown] id="H01JB0hwhCQ5"
# We will now consider the full dataset, including children. The resulting mapping from weight to height is now nonlinear.

# + [markdown] id="9eK978cgiF6W"
# ## Data

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="fLZ7Tk0JfiMT" outputId="4bfa9b66-896e-43ec-ed5f-24ff46b89e6c"
az.plot_pair(d[["weight", "height"]].to_dict(orient="list")) # d, not d2
plt.tight_layout()
plt.savefig('linreg_height_weight_data_full.pdf', dpi=300)
plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="HXFHzDDShMxO" outputId="bb76c543-dc8b-4971-f9e0-860ed47c9300"
# standardize
d["weight_s"] = (d.weight - d.weight.mean()) / d.weight.std()

# precompute polynomial terms
d["weight_s2"] = d.weight_s ** 2
d["weight_s3"] = d.weight_s ** 3

ax = az.plot_pair(
    d[["weight_s", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5}
)
ax.set(xlabel="weight", ylabel="height", xticks=[])
fig = plt.gcf()

at = jnp.array([-2, -1, 0, 1, 2])
labels = at * d.weight.std() + d.weight.mean()
ax.set_xticks(at)
ax.set_xticklabels([round(label, 1) for label in labels])
#fig

plt.tight_layout()
plt.savefig('linreg_height_weight_data_full_standardized.pdf', dpi=300)
plt.show()


# + [markdown] id="fiO44tM1jSSi"
# ## Fit model

# + id="HsiuKFUIlAGE"
def fit_predict(model, fname=None):
  guide = AutoLaplaceApproximation(model)
  svi = SVI(
      model,
      guide,
      optim.Adam(0.3),
      Trace_ELBO(),
      weight_s=d.weight_s.values,
      weight_s2=d.weight_s2.values,
      height=d.height.values,
  )
  params, losses = svi.run(random.PRNGKey(0), 3000)

  samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
  print_summary({k: v for k, v in samples.items() if k != "mu"}, 0.95, False)

  weight_seq = jnp.linspace(start=-2.2, stop=2, num=30)
  pred_dat = {"weight_s": weight_seq, "weight_s2": weight_seq ** 2}
  post = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
  predictive = Predictive(guide.model, post)
  mu = predictive(random.PRNGKey(2), **pred_dat)["mu"]
  mu_mean = jnp.mean(mu, 0)
  mu_PI = jnp.percentile(mu, q=(2.5, 97.5), axis=0)
  sim_height = predictive(random.PRNGKey(3), **pred_dat)["height"]
  height_PI = jnp.percentile(sim_height, q=(2.5, 97.5), axis=0)

  ax = az.plot_pair(
    d[["weight_s", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5})
  plt.plot(weight_seq, mu_mean, "k")
  plt.fill_between(weight_seq, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
  plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="k", alpha=0.15)

  ax.set(xlabel="weight", ylabel="height", xticks=[])
  fig = plt.gcf()
  at = jnp.array([-2, -1, 0, 1, 2])
  labels = at * d.weight.std() + d.weight.mean()
  ax.set_xticks(at)
  ax.set_xticklabels([round(label, 1) for label in labels])

  plt.tight_layout()
  if fname:
    plt.savefig(fname, dpi=300)
  plt.show()


# + [markdown] id="jN-fvazFlPNs"
# ## Linear

# + colab={"base_uri": "https://localhost:8080/", "height": 411} id="KdhHH9YCiJn0" outputId="3b9b9343-8917-4303-9b20-d97d910f7581"
def model_linear(weight_s, weight_s2, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1)) # Log-Normal prior
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + b1 * weight_s)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


fit_predict(model_linear, 'linreg_height_weight_data_full_linear.pdf')


# + [markdown] id="CO4FL_bqiHD7"
# ## Quadratic 

# + colab={"base_uri": "https://localhost:8080/", "height": 427} id="rPM9ktYPmWmb" outputId="001b1f45-3613-4110-b93f-b4ff832970aa"
def model_quad(weight_s, weight_s2, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1)) 
    b2 = numpyro.sample("b2", dist.Normal(0, 1)) 
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + b1 * weight_s + b2 * weight_s2)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


fit_predict(model_quad, 'linreg_height_weight_data_full_quad.pdf')


# + colab={"base_uri": "https://localhost:8080/", "height": 427} id="EGLB0sMfmWxn" outputId="a55d26df-5e1d-42e7-f091-fdced4e7fc07"
def model_quad_positive_b2(weight_s, weight_s2, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    #b2 = numpyro.sample("b2", dist.Normal(0, 1)) 
    b2 = numpyro.sample("b2", dist.LogNormal(0, 1))  
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + b1 * weight_s + b2 * weight_s2)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


fit_predict(model_quad_positive_b2, 'linreg_height_weight_data_full_quad_pos_b2.pdf')
