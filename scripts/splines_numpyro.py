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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/splines_numpyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="wf58_Abx-XA4"
# # 1d regression splines 
#
# We illustrate 1d regression splines using the cherry blossom example in sec 4.5  of [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). 
# The numpyro code is from [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/04-geocentric-models.html).

# + colab={"base_uri": "https://localhost:8080/"} id="OSvodcD5-R7u" outputId="3661de5d-058b-46e2-8202-902dcc24a6b6"
# !pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
# !pip install -q arviz

# + colab={"base_uri": "https://localhost:8080/"} id="kecaP1IT-jEm" outputId="58764c4a-5d50-4488-ba5f-87b353bb74be"

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import os
import warnings
import pandas as pd

from scipy.interpolate import BSpline
from scipy.stats import gaussian_kde

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

# + [markdown] id="ylb0N2kO-ntE"
# # Data

# + colab={"base_uri": "https://localhost:8080/", "height": 451} id="WMWqiH8L-oVs" outputId="9b8e70bb-5cf0-44f1-eaf2-5d7a85d54f1e"

url = 'https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/cherry_blossoms.csv'
cherry_blossoms = pd.read_csv(url, sep=';')
df = cherry_blossoms

display(df.sample(n=5, random_state=1))
display(df.describe())


# + id="EivWfDqbvrl-"
df2 = df[df.doy.notna()]  # complete cases on doy (day of year)
x = df2.year.values.astype(float)
y = df2.doy.values.astype(float)
xlabel = 'year'
ylabel = 'doy'


# + [markdown] id="c_ahJNHTBxqF"
# # B-splines 

# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="Pg8rAIpzB03v" outputId="9d4f3444-3f7e-4a1c-cf62-fbe0457137f6"
def make_splines(x, num_knots, degree=3):
  knot_list = jnp.quantile(x, q=jnp.linspace(0, 1, num=num_knots))
  knots = jnp.pad(knot_list, (3, 3), mode="edge")
  B = BSpline(knots, jnp.identity(num_knots + 2), k=degree)(x)
  return B


def plot_basis(x, B, w=None):
  if w is None: w = jnp.ones((B.shape[1]))
  fig, ax = plt.subplots()
  ax.set_xlim(np.min(x), np.max(x))
  ax.set_xlabel(xlabel)
  ax.set_ylabel("basis value")
  for i in range(B.shape[1]):
      ax.plot(x, (w[i] * B[:, i]), "k", alpha=0.5)
  return ax

nknots = 15
B =  make_splines(x, nknots)
ax = plot_basis(x, B)
plt.savefig(f'splines_basis_{nknots}_{ylabel}.pdf', dpi=300)



# + colab={"base_uri": "https://localhost:8080/"} id="FOtuS9HI7wLa" outputId="d7bc5557-6900-4791-bc8a-2ce800a78ad0"
num_knots = 15
degree = 3

knot_list = jnp.quantile(x, q=jnp.linspace(0, 1, num=num_knots))
print(knot_list)
print(knot_list.shape)

knots = jnp.pad(knot_list, (3, 3), mode="edge")
print(knots)
print(knots.shape)

B = BSpline(knots, jnp.identity(num_knots + 2), k=degree)(x)
print(B.shape)


# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="BeSiSsOIwdVt" outputId="9fa8e817-145e-4237-920e-fb371b584171"
def plot_basis_with_vertical_line(x, B, xstar):
  ax = plot_basis(x, B)
  num_knots = B.shape[1]
  ndx = np.where(x==xstar)[0][0]
  for i in range(num_knots):
    yy = B[ndx,i]
    if yy>0:
      ax.scatter(xstar, yy, s=40)
  ax.axvline(x=xstar)
  return ax

plot_basis_with_vertical_line(x, B, 1200)
plt.savefig(f'splines_basis_{nknots}_vertical_{ylabel}.pdf', dpi=300)

# + colab={"base_uri": "https://localhost:8080/", "height": 298} id="rdEtshuusKCf" outputId="a848d930-d38e-477e-a179-ace07b5fe8fb"



def model(B, y, offset=100):
    a = numpyro.sample("a", dist.Normal(offset, 10))
    w = numpyro.sample("w", dist.Normal(0, 10).expand(B.shape[1:]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + B @ w)
    #mu = numpyro.deterministic("mu", a + jnp.sum(B * w, axis=-1)) # equivalent
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

def fit_model(B, y, offset=100):
  start = {"w": jnp.zeros(B.shape[1])}
  guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
  svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), B=B, y=y, offset=offset)
  params, losses = svi.run(random.PRNGKey(0), 20000) # needs 20k iterations 
  post = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
  return post

post = fit_model(B, y)
w = jnp.mean(post["w"], 0)
plot_basis(x, B, w)
plt.savefig(f'splines_basis_weighted_{nknots}_{ylabel}.pdf', dpi=300)



# + colab={"base_uri": "https://localhost:8080/", "height": 295} id="vb_fLuNssLhh" outputId="46055dc3-a785-4fdd-bfc0-2b45d7316d24"
def plot_post_pred(post, x, y):
  mu = post["mu"]
  mu_PI = jnp.percentile(mu, q=(1.5, 98.5), axis=0)
  plt.figure()
  plt.scatter(x, y)
  plt.fill_between(x, mu_PI[0], mu_PI[1], color="k", alpha=0.5)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

plot_post_pred(post, x, y)
plt.savefig(f'splines_post_pred_{nknots}_{ylabel}.pdf', dpi=300)

# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="R6KW35xJz22j" outputId="ff284294-bd3f-42e9-8ecc-5bffce87bb4b"
a = jnp.mean(post["a"], 0)
w = jnp.mean(post["w"], 0)
mu = a + B @ w


def plot_pred(mu, x, y):
  plt.figure()
  plt.scatter(x, y, alpha=0.5)
  plt.plot(x, mu, 'k-', linewidth=4)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

plot_pred(mu, x, y)
plt.savefig(f'splines_point_pred_{nknots}_{ylabel}.pdf', dpi=300)

# + [markdown] id="o6nkefZh3Sfc"
# # Repeat with temperature as target variable

# + id="25Ouhyen6xem"
df2 = df[df.temp.notna()]  # complete cases 
x = df2.year.values.astype(float)
y = df2.temp.values.astype(float)
xlabel = 'year'
ylabel = 'temp'

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="AxSVhnmr27ws" outputId="8536fc51-e695-43c8-8241-f3f1a48e878c"
nknots = 15

B =  make_splines(x, nknots)
plot_basis_with_vertical_line(x, B, 1200)
plt.savefig(f'splines_basis_{nknots}_vertical_{ylabel}.pdf', dpi=300)


post = fit_model(B, y, offset=6)
w = jnp.mean(post["w"], 0)
plot_basis(x, B, w)
plt.savefig(f'splines_basis_weighted_{nknots}_{ylabel}.pdf', dpi=300)

plot_post_pred(post, x, y)
plt.savefig(f'splines_post_pred_{nknots}_{ylabel}.pdf', dpi=300)

a = jnp.mean(post["a"], 0)
w = jnp.mean(post["w"], 0)
mu = a + B @ w
plot_pred(mu, x, y)
plt.savefig(f'splines_point_pred_{nknots}_{ylabel}.pdf', dpi=300)

# + id="yb0OVaCc7rN5"


# + [markdown] id="4lNrfg7K-rdN"
# # Maximum likelihood estimation

# + colab={"base_uri": "https://localhost:8080/"} id="yJYfqTWy-vvk" outputId="c80472fe-6905-47fd-9fbc-dcbdc3c3a763"
from sklearn.linear_model import LinearRegression, Ridge
#reg = LinearRegression().fit(B, y)
reg = Ridge().fit(B, y)
w = reg.coef_
a = reg.intercept_
print(w)
print(a)

# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="K2qDD94J_GON" outputId="7a1f1eec-ba59-4d8e-cd6d-117f433de058"
mu = a + B @ w
plot_pred(mu, x, y)
plt.savefig(f'splines_MLE_{nknots}_{ylabel}.pdf', dpi=300)

# + id="eXFjl0_7_YWJ"

