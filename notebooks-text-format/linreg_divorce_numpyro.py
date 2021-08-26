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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/linreg_divorce_numpyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="xuUCwM8u7d_C"
# # Robust linear regression 
#
# We illustrate  linear using the "waffle divorce" example in sec 5.1  of [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). 
# The numpyro code is from [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/05-the-many-variables-and-the-spurious-waffles.html)
#   
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="zrZ4gm-47aCF" outputId="4879c72d-2ba1-4eaa-d441-0aa3acaf3ad6"
# !pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
# !pip install -q arviz

# + colab={"base_uri": "https://localhost:8080/"} id="Ae7e0Ef671cl" outputId="952dcffc-4918-4058-da4b-e7bf63815cb2"
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
from numpyro.infer import Predictive, log_likelihood
from numpyro.infer import MCMC, NUTS
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.optim as optim


import arviz as az

# + [markdown] id="MyKMMdN_7yEh"
# # Data
#
# The data records the divorce rate $D$, marriage rate $M$, and average age $A$ that people get married at for 50 US states.

# + id="avu-Z75e7zBK"
# load data and copy
url = 'https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/WaffleDivorce.csv'
WaffleDivorce = pd.read_csv(url, sep=";")
d = WaffleDivorce

# standardize variables
d["A"] = d.MedianAgeMarriage.pipe(lambda x: (x - x.mean()) / x.std())
d["D"] = d.Divorce.pipe(lambda x: (x - x.mean()) / x.std())
d["M"] = d.Marriage.pipe(lambda x: (x - x.mean()) / x.std())


# + [markdown] id="Zycpf_NB8WNA"
# # Model (Gaussian likelihood)
#
# We predict divorce rate D given marriage rate M and age A.

# + colab={"base_uri": "https://localhost:8080/"} id="D2jggu2v8XOU" outputId="6974503e-0a27-466e-9668-881e0f96a2ad"
def model(M, A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bM * M + bA * A)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


m5_3 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_3, optim.Adam(1), Trace_ELBO(), M=d.M.values, A=d.A.values, D=d.D.values)
p5_3, losses = svi.run(random.PRNGKey(0), 1000)
post = m5_3.sample_posterior(random.PRNGKey(1), p5_3, (1000,))


# + colab={"base_uri": "https://localhost:8080/"} id="omQy9sZf9M9q" outputId="a8b98472-a44f-4b34-f461-a4c7bd431b13"
param_names = {'a', 'bA', 'bM', 'sigma'}
for p in param_names:
  print(f'posterior for {p}')
  print_summary(post[p], 0.95, False)
          


# + [markdown] id="w-2Y2AY-91OU"
# # Posterior predicted vs actual

# + id="OgH6toqX9NyV"
# call predictive without specifying new data
# so it uses original data
post = m5_3.sample_posterior(random.PRNGKey(1), p5_3, (int(1e4),))
post_pred = Predictive(m5_3.model, post)(random.PRNGKey(2), M=d.M.values, A=d.A.values)
mu = post_pred["mu"]

# summarize samples across cases
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)



# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="pmxVjQGg-omZ" outputId="64201dd2-55e7-411f-d8fa-67a7291f7701"
ax = plt.subplot(
    ylim=(float(mu_PI.min()), float(mu_PI.max())),
    xlabel="Observed divorce",
    ylabel="Predicted divorce"
)
plt.plot(d.D, mu_mean, "o")
x = jnp.linspace(mu_PI.min(), mu_PI.max(), 101)
plt.plot(x, x, "--")
for i in range(d.shape[0]):
    plt.plot([d.D[i]] * 2, mu_PI[:, i], "b")
fig = plt.gcf()

# + colab={"base_uri": "https://localhost:8080/", "height": 295} id="C5LQYmJ7-owr" outputId="04afba64-94ae-483e-9399-5a010d98aedf"
for i in range(d.shape[0]):
    if d.Loc[i] in ["ID", "UT", "AR", "ME"]:
        ax.annotate(
            d.Loc[i], (d.D[i], mu_mean[i]), xytext=(-25, -5), textcoords="offset pixels"
        )
plt.tight_layout()
plt.savefig('linreg_divorce_postpred.pdf')
plt.show()
fig

# + [markdown] id="lT7ATLTm_nrM"
# # Per-point LOO scores
#
# We compute the predicted probability of each point given the others, following
# sec 7.5.2   of [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). 
# The numpyro code is from [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/07-ulysses-compass.html)
#   
#

# + id="AhXUMAg_-w2P"
#post = m5_3.sample_posterior(random.PRNGKey(24071847), p5_3, (1000,))
logprob = log_likelihood(m5_3.model, post, A=d.A.values, M=d.M.values, D=d.D.values)[
    "D"
]
az5_3 = az.from_dict(
    posterior={k: v[None, ...] for k, v in post.items()},
    log_likelihood={"D": logprob[None, ...]},
)

# + colab={"base_uri": "https://localhost:8080/", "height": 380} id="CtBDE7SMAcUU" outputId="6ac0092d-c99d-4b1c-fb27-082eab672cf7"
PSIS_m5_3 = az.loo(az5_3, pointwise=True, scale="deviance")
WAIC_m5_3 = az.waic(az5_3, pointwise=True, scale="deviance")
penalty = az5_3.log_likelihood.stack(sample=("chain", "draw")).var(dim="sample")

fig, ax = plt.subplots()
ax.plot(PSIS_m5_3.pareto_k.values, penalty.D.values, "o", mfc="none")
ax.set_xlabel("PSIS Pareto k")
ax.set_ylabel("WAIC penalty")

plt.savefig('linreg_divorce_waic_vs_pareto.pdf')
plt.show()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 327} id="GeHg_-JDA_KA" outputId="8044852c-06ac-45c7-d747-2a974f45ddb0"
pareto =  PSIS_m5_3.pareto_k.values
waic = penalty.D.values
ndx = np.where(pareto > 0.4)[0]
for i in ndx:
  print(d.Loc[i], pareto[i], waic[i])


for i in ndx:
    ax.annotate(d.Loc[i], (pareto[i], waic[i]), xytext=(5, 0), textcoords="offset pixels")
fig


# + [markdown] id="Lu8OOz-hFhIe"
# # Student likelihood

# + colab={"base_uri": "https://localhost:8080/"} id="wRTRc2hsBPUA" outputId="367f484e-b32a-432c-ba47-b034270680fa"
def model(M, A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    #mu = a + bM * M + bA * A
    mu = numpyro.deterministic("mu", a + bM * M + bA * A)
    numpyro.sample("D", dist.StudentT(2, mu, sigma), obs=D)


m5_3t = AutoLaplaceApproximation(model)
svi = SVI(
    model, m5_3t, optim.Adam(0.3), Trace_ELBO(), M=d.M.values, A=d.A.values, D=d.D.values
)
p5_3t, losses = svi.run(random.PRNGKey(0), 1000)

# + colab={"base_uri": "https://localhost:8080/", "height": 278} id="64ZEbcnQFl0E" outputId="dfccdae3-5578-4ad4-dc80-22c16ae82844"
# call predictive without specifying new data
# so it uses original data
post_t = m5_3t.sample_posterior(random.PRNGKey(1), p5_3t, (int(1e4),))
post_pred_t = Predictive(m5_3t.model, post_t)(random.PRNGKey(2), M=d.M.values, A=d.A.values)
mu = post_pred_t["mu"]

# summarize samples across cases
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)


ax = plt.subplot(
    ylim=(float(mu_PI.min()), float(mu_PI.max())),
    xlabel="Observed divorce",
    ylabel="Predicted divorce"
)
plt.plot(d.D, mu_mean, "o")
x = jnp.linspace(mu_PI.min(), mu_PI.max(), 101)
plt.plot(x, x, "--")
for i in range(d.shape[0]):
    plt.plot([d.D[i]] * 2, mu_PI[:, i], "b")
fig = plt.gcf()
