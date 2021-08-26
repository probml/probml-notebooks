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
# <a href="https://colab.research.google.com/github/always-newbie161/pyprobml/blob/issue_hermes78/notebooks/logreg_ucb_admissions_numpyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="bZmPN5Gu1zna"
# # Binomial logistic regression for UCB admissions
#
# We illustrate binary logistic regression on 2 discrete inputs using the example in sec 11.1.4  of [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). 
# The numpyro code is from [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/11-god-spiked-the-integers.html)
#
#

# + id="_y0aLBbR1zMh" colab={"base_uri": "https://localhost:8080/"} outputId="8b715628-06e4-4994-aea4-c4b61a03c17e"
# !pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
# !pip install -q arviz


# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="cehGSuboWUCj" outputId="bfbcc046-0ff1-4e78-a8b4-5bf423fd711c"
import arviz as az
az.__version__


# + colab={"base_uri": "https://localhost:8080/"} id="WKy5McPCB8R_" outputId="63dcc8d2-50b8-4073-8947-2b5a21dc8905"
# !pip install causalgraphicalmodels

# + id="q2Nn5H_nDK7P"
# #!pip install -U daft

# + colab={"base_uri": "https://localhost:8080/"} id="HxnMvcA72EPS" outputId="18693280-d54d-485a-d0c6-135293ce1be2"
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
from jax.scipy.special import expit

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


import daft
from causalgraphicalmodels import CausalGraphicalModel

from sklearn.preprocessing import StandardScaler

# + colab={"base_uri": "https://localhost:8080/"} id="oKZlhvtKcDVs" outputId="55136941-f974-42e8-97b7-287ec3bd2f78"
n = jax.local_device_count()
print(n)

# + [markdown] id="JK9wTe4b2MBq"
# # Data

# + colab={"base_uri": "https://localhost:8080/", "height": 392} id="DmV4wYiI2F1c" outputId="dfc5ec61-7d1d-4188-bd01-6fc16e2cc011"

url = 'https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/UCBadmit.csv'
UCBadmit = pd.read_csv(url, sep=";")
d = UCBadmit
display(d)


# + colab={"base_uri": "https://localhost:8080/"} id="WNXUcstfgMb7" outputId="28361766-63e7-4ecc-81e3-ffab529019be"
print(d.to_latex(index=False))

# + colab={"base_uri": "https://localhost:8080/"} id="4s0itZ74AVD8" outputId="221a6709-2672-46f2-aea3-46fffb76ae7a"

dat_list = dict(
    admit=d.admit.values,
    applications=d.applications.values,
    gid=(d["applicant.gender"] != "male").astype(int).values,
)

dat_list["dept_id"] = jnp.repeat(jnp.arange(6), 2)

print(dat_list)

# + colab={"base_uri": "https://localhost:8080/"} id="awteD7M-Asri" outputId="59c78248-6254-47f8-9007-baf41c2b802e"
# extract number of applicaitons for dept 2 (C)
d.applications[dat_list["dept_id"].copy() == 2]
               

# + colab={"base_uri": "https://localhost:8080/"} id="tso5iKGVZ1A3" outputId="6ffa821c-ca79-4f23-c78e-4c78a7f70ce1"
d.applications[dat_list["dept_id"].copy() == 2].sum()

# + colab={"base_uri": "https://localhost:8080/", "height": 251} id="rjKvZC7w9F_t" outputId="b94f80b7-1f9e-4fb9-ca93-da19a2982c07"
# application rate per department
pg = jnp.stack(
    list(
        map(
            lambda k: jnp.divide(
                d.applications[dat_list["dept_id"].copy() == k].values,
                d.applications[dat_list["dept_id"].copy() == k].sum(),
            ),
            range(6),
        )
    ),
    axis=0,
).T
pg = pd.DataFrame(pg, index=["male", "female"], columns=d.dept.unique())
display(pg.round(2))
print(pg.to_latex())

# + colab={"base_uri": "https://localhost:8080/", "height": 251} id="OLoWrfLyaZrw" outputId="4cfd5397-f780-4cf2-e679-4d8f3a3b0eb2"
# admisions rate per department
pg = jnp.stack(
    list(
        map(
            lambda k: jnp.divide(
                d.admit[dat_list["dept_id"].copy() == k].values,
                d.applications[dat_list["dept_id"].copy() == k].values,
            ),
            range(6),
        )
    ),
    axis=0,
).T
pg = pd.DataFrame(pg, index=["male", "female"], columns=d.dept.unique())
display(pg.round(2))
print(pg.to_latex())

# + [markdown] id="lIH__8Bz2Vhf"
# # Model 1

# + colab={"base_uri": "https://localhost:8080/"} id="DCchW_SRb2tJ" outputId="7dd337c0-b177-46ac-f5a3-649b888c818d"
dat_list = dict(
    admit=d.admit.values,
    applications=d.applications.values,
    gid=(d["applicant.gender"] != "male").astype(int).values,
)


def model(gid, applications, admit=None):
    a = numpyro.sample("a", dist.Normal(0, 1.5).expand([2]))
    logit_p = a[gid]
    numpyro.sample("admit", dist.Binomial(applications, logits=logit_p), obs=admit)


m11_7 = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
m11_7.run(random.PRNGKey(0), **dat_list)
m11_7.print_summary(0.89)

# + colab={"base_uri": "https://localhost:8080/"} id="YrkAgHHH2zvJ" outputId="4195c61e-908c-4975-b786-f1d7e7fb5467"
post = m11_7.get_samples()
diff_a = post["a"][:, 0] - post["a"][:, 1]
diff_p = expit(post["a"][:, 0]) - expit(post["a"][:, 1])
print_summary({"diff_a": diff_a, "diff_p": diff_p}, 0.89, False)


# + [markdown] id="aIEJ5zrH288g"
# # Posterior predictive check

# + id="jY1URpYA4_TJ"
def ppc(mcmc_run, model_args):
  post = mcmc_run.get_samples()
  pred = Predictive(mcmc_run.sampler.model, post)(random.PRNGKey(2), **model_args)
  admit_pred = pred["admit"]
  admit_rate = admit_pred / d.applications.values
  plt.errorbar(
      range(1, 13),
      jnp.mean(admit_rate, 0),
      jnp.std(admit_rate, 0) / 2,
      fmt="o",
      c="k",
      mfc="none",
      ms=7,
      elinewidth=1,
  )
  plt.plot(range(1, 13), jnp.percentile(admit_rate, 5.5, 0), "k+")
  plt.plot(range(1, 13), jnp.percentile(admit_rate, 94.5, 0), "k+")
  # draw lines connecting points from same dept
  for i in range(1, 7):
      x = 1 + 2 * (i - 1) # 1,3,5,7,9,11
      y1 = d.admit.iloc[x - 1] / d.applications.iloc[x - 1] # male
      y2 = d.admit.iloc[x] / d.applications.iloc[x] # female
      plt.plot((x, x + 1), (y1, y2), "bo-")
      plt.annotate(
          d.dept.iloc[x], (x + 0.5, (y1 + y2) / 2 + 0.05), ha="center", color="royalblue"
      )
  plt.gca().set(ylim=(0, 1), xticks=range(1, 13), ylabel="admit", xlabel="case")


# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="FyrzH_Yi30N2" outputId="d2180a68-4b0b-4a22-e300-9f04427b8e6c"
ppc(m11_7, {'gid': dat_list["gid"], 'applications': dat_list["applications"]})
plt.savefig('admissions_ppc.pdf', dpi=300)
plt.show()

# + [markdown] id="KlJGu63T4Jew"
# # Model 2 (departmental-specific offset)

# + colab={"base_uri": "https://localhost:8080/"} id="EgepyNLf4E9H" outputId="bbd5aef4-990c-42c9-b1a1-0d14de4e1b8e"

dat_list["dept_id"] = jnp.repeat(jnp.arange(6), 2)

def model(gid, dept_id, applications, admit=None):
    a = numpyro.sample("a", dist.Normal(0, 1.5).expand([2]))
    delta = numpyro.sample("delta", dist.Normal(0, 1.5).expand([6]))
    logit_p = a[gid] + delta[dept_id]
    numpyro.sample("admit", dist.Binomial(applications, logits=logit_p), obs=admit)


m11_8 = MCMC(NUTS(model), num_warmup=2000, num_samples=2000, num_chains=4)
m11_8.run(random.PRNGKey(0), **dat_list)
m11_8.print_summary(0.89)

# + colab={"base_uri": "https://localhost:8080/"} id="8CYrU2uN4nli" outputId="57cb1fd2-3db5-4cb1-b7cc-b7d1e6069597"
post = m11_8.get_samples()
diff_a = post["a"][:, 0] - post["a"][:, 1]
diff_p = expit(post["a"][:, 0]) - expit(post["a"][:, 1])
print_summary({"diff_a": diff_a, "diff_p": diff_p}, 0.89, False)

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="BFXYpYYh4gZL" outputId="12e334c8-6689-48ad-ef94-3c3fbab577db"
data_dict = {'gid': dat_list["gid"],
             'dept_id': dat_list["dept_id"],
             'applications': dat_list["applications"]}
ppc(m11_8, data_dict)
#ppc(m11_8, dat_list) # must exclude 'admit' for predictive distribution
plt.savefig('admissions_ppc_per_dept.pdf', dpi=300)
plt.show()


# + [markdown] id="fTyU9j5gWpG5"
# # Poisson regression
#
# We now show we can emulate binomial regresison using 2 poisson regressions,
# following sec 11.3.3 of rethinking. We use a simplified model that just predicts outcomes, and has no features (just an offset term).

# + colab={"base_uri": "https://localhost:8080/"} id="K6zWmx1LXdrj" outputId="0740c417-7657-4329-b55d-a2e15bba106a"
# binomial model of overall admission probability
def model(applications, admit):
    a = numpyro.sample("a", dist.Normal(0, 1.5))
    logit_p = a
    numpyro.sample("admit", dist.Binomial(applications, logits=logit_p), obs=admit)

'''
m_binom = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m_binom,
    optim.Adam(1),
    Trace_ELBO(),
    applications=d.applications.values,
    admit=d.admit.values,
)
p_binom, losses = svi.run(random.PRNGKey(0), 1000)
'''

m_binom = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
m_binom.run(random.PRNGKey(0), d.applications.values, d.admit.values)
m_binom.print_summary(0.95)



# + colab={"base_uri": "https://localhost:8080/"} id="pKRQopvFYeSq" outputId="1ea101a6-a230-4676-84bd-0fdee708433c"
logit = jnp.mean(m_binom.get_samples()["a"])
print(expit(logit))


# + colab={"base_uri": "https://localhost:8080/"} id="oYMtHurQYArE" outputId="9ff3ba00-d8e0-4c2b-ec87-937edc01c4a5"
def model(rej, admit):
    a1, a2 = numpyro.sample("a", dist.Normal(0, 1.5).expand([2]))
    lambda1 = jnp.exp(a1)
    lambda2 = jnp.exp(a2)
    numpyro.sample("rej", dist.Poisson(lambda2), obs=rej)
    numpyro.sample("admit", dist.Poisson(lambda1), obs=admit)


m_pois = MCMC(NUTS(model), num_warmup=1000, num_samples=1000, num_chains=3)
m_pois.run(random.PRNGKey(0), d.reject.values, d.admit.values)
m_pois.print_summary(0.95)
                     


# + colab={"base_uri": "https://localhost:8080/"} id="Pwr25iX4ZGZD" outputId="59e577c0-4ede-4ce4-b022-45124c9aadd0"
params = jnp.mean(m_pois.get_samples()["a"], 0)
a1 = params[0]
a2 = params[1]
lam1 = jnp.exp(a1)
lam2 = jnp.exp(a2)
print([lam1, lam2])
print(lam1 / (lam1 + lam2))

# + [markdown] id="k7RykjuvbG5F"
# # Beta-binomial regression
#
# Sec 12.1.1 of rethinking.
# Code from snippet 12.2 of [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/12-monsters-and-mixtures.html)
#

# + colab={"base_uri": "https://localhost:8080/"} id="rXLcgSYibLs8" outputId="a96b1c01-1b6f-426e-dfe6-ba18c7f22d06"
d = UCBadmit
d["gid"] = (d["applicant.gender"] != "male").astype(int)
dat = dict(A=d.admit.values, N=d.applications.values, gid=d.gid.values)


def model(gid, N, A=None):
    a = numpyro.sample("a", dist.Normal(0, 1.5).expand([2]))
    phi = numpyro.sample("phi", dist.Exponential(1))
    theta = numpyro.deterministic("theta", phi + 2) # shape
    pbar = expit(a[gid]) # mean
    numpyro.sample("A", dist.BetaBinomial(pbar * theta, (1 - pbar) * theta, N), obs=A)


m12_1 = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
m12_1.run(random.PRNGKey(0), **dat)

# + colab={"base_uri": "https://localhost:8080/"} id="DroRyUENbvyD" outputId="eedcc585-f2a4-4d01-d42c-d920476b0c51"
post = m12_1.get_samples()
post["theta"] = Predictive(m12_1.sampler.model, post)(random.PRNGKey(1), **dat)["theta"]
post["da"] = post["a"][:, 0] - post["a"][:, 1]
print_summary(post, 0.89, False)

# + id="S9VxiaDZyecO" colab={"base_uri": "https://localhost:8080/"} outputId="3de32392-9a74-46d8-e248-f687986adab4"
post

# + colab={"base_uri": "https://localhost:8080/", "height": 295} id="GIwMyYUob2At" outputId="8bdc1adc-c386-4e95-b454-b954c8b149c9"
gid = 1
# draw posterior mean beta distribution
x = jnp.linspace(0, 1, 101)
pbar = jnp.mean(expit(post["a"][:, gid]))
theta = jnp.mean(post["theta"])
plt.plot(x, jnp.exp(dist.Beta(pbar * theta, (1 - pbar) * theta).log_prob(x)))
plt.gca().set(ylabel="Density", xlabel="probability admit", ylim=(0, 3))

# draw 50 beta distributions sampled from posterior
for i in range(50):
    p = expit(post["a"][i, gid])
    theta = post["theta"][i]
    plt.plot(
        x, jnp.exp(dist.Beta(p * theta, (1 - p) * theta).log_prob(x)), "k", alpha=0.2
    )
plt.title("distribution of female admission rates")
plt.savefig('admissions_betabinom_female_rate.pdf')
plt.show()

# + id="P0m8BionvsXR" colab={"base_uri": "https://localhost:8080/", "height": 295} outputId="0202f6b7-cd3a-4813-e6b6-3d780fb95778"
fig, ax = plt.subplots()
labels = ['male', 'female']
colors = ['b', 'r']
for gid in [0,1]:
  # draw posterior mean beta distribution
  x = jnp.linspace(0, 1, 101)
  pbar = jnp.mean(expit(post["a"][:, gid]))
  theta = jnp.mean(post["theta"])
  y = jnp.exp(dist.Beta(pbar * theta, (1 - pbar) * theta).log_prob(x))
  ax.plot(x, y, label=labels[gid], color=colors[gid])
  ax.set_ylabel("Density")
  ax.set_xlabel("probability admit")
  ax.set_ylim(0, 3)

  # draw some beta distributions sampled from posterior
  for i in range(10):
      p = expit(post["a"][i, gid])
      theta = post["theta"][i]
      y =jnp.exp(dist.Beta(p * theta, (1 - p) * theta).log_prob(x))
      plt.plot(x, y, colors[gid], alpha=0.2)

plt.title("distribution of admission rates")
plt.legend()
plt.savefig('admissions_betabinom_rates.pdf')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="LqPtmN_IcLY4" outputId="2519a209-84d0-477d-a13f-49ac03214931"
post = m12_1.get_samples()
admit_pred = Predictive(m12_1.sampler.model, post)(
    random.PRNGKey(1), gid=dat["gid"], N=dat["N"]
)["A"]
admit_rate = admit_pred / dat["N"]
plt.scatter(range(1, 13), dat["A"] / dat["N"])
plt.errorbar(
    range(1, 13),
    jnp.mean(admit_rate, 0),
    jnp.std(admit_rate, 0) / 2,
    fmt="o",
    c="k",
    mfc="none",
    ms=7,
    elinewidth=1,
)
plt.plot(range(1, 13), jnp.percentile(admit_rate, 5.5, 0), "k+")
plt.plot(range(1, 13), jnp.percentile(admit_rate, 94.5, 0), "k+")
plt.savefig('admissions_betabinom_post_pred.pdf')
plt.show()

# + [markdown] id="WASI8v5XHUhi"
# # Mixed effects model with joint prior
#
# This code is from https://numpyro.readthedocs.io/en/latest/examples/ucbadmit.html.

# + id="peOoI8OLHZFs"
from numpyro.examples.datasets import UCBADMIT, load_dataset



def glmm(dept, male, applications, admit=None):
    v_mu = numpyro.sample("v_mu", dist.Normal(0, jnp.array([4.0, 1.0])))

    sigma = numpyro.sample("sigma", dist.HalfNormal(jnp.ones(2)))
    L_Rho = numpyro.sample("L_Rho", dist.LKJCholesky(2, concentration=2))
    scale_tril = sigma[..., jnp.newaxis] * L_Rho
    # non-centered parameterization
    num_dept = len(np.unique(dept))
    z = numpyro.sample("z", dist.Normal(jnp.zeros((num_dept, 2)), 1))
    v = jnp.dot(scale_tril, z.T).T

    logits = v_mu[0] + v[dept, 0] + (v_mu[1] + v[dept, 1]) * male
    if admit is None:
        # we use a Delta site to record probs for predictive distribution
        probs = expit(logits)
        numpyro.sample("probs", dist.Delta(probs), obs=probs)
    numpyro.sample("admit", dist.Binomial(applications, logits=logits), obs=admit)


def run_inference(dept, male, applications, admit, rng_key):
    kernel = NUTS(glmm)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    mcmc.run(rng_key, dept, male, applications, admit)
    return mcmc.get_samples()


def print_results(header, preds, dept, male, probs):
    columns = ["Dept", "Male", "ActualProb", "Pred(p25)", "Pred(p50)", "Pred(p75)"]
    header_format = "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    row_format = "{:>10.0f} {:>10.0f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}"
    quantiles = jnp.quantile(preds, jnp.array([0.25, 0.5, 0.75]), axis=0)
    print("\n", header, "\n")
    print(header_format.format(*columns))
    for i in range(len(dept)):
        print(row_format.format(dept[i], male[i], probs[i], *quantiles[:, i]), "\n")



# + id="tGUqrgCgHlPt" colab={"base_uri": "https://localhost:8080/", "height": 961} outputId="8f9e1a80-05e8-4ef3-9b0c-5bf23cf3ebb6"

_, fetch_train = load_dataset(UCBADMIT, split="train", shuffle=False)
dept, male, applications, admit = fetch_train()
rng_key, rng_key_predict = random.split(random.PRNGKey(1))
zs = run_inference(dept, male, applications, admit, rng_key)
pred_probs = Predictive(glmm, zs)(rng_key_predict, dept, male, applications)[
    "probs"
]
header = "=" * 30 + "glmm - TRAIN" + "=" * 30
print_results(header, pred_probs, dept, male, admit / applications)

# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.plot(range(1, 13), admit / applications, "o", ms=7, label="actual rate")
ax.errorbar(
    range(1, 13),
    jnp.mean(pred_probs, 0),
    jnp.std(pred_probs, 0),
    fmt="o",
    c="k",
    mfc="none",
    ms=7,
    elinewidth=1,
    label=r"mean $\pm$ std",
)
ax.plot(range(1, 13), jnp.percentile(pred_probs, 5, 0), "k+")
ax.plot(range(1, 13), jnp.percentile(pred_probs, 95, 0), "k+")
ax.set(
    xlabel="cases",
    ylabel="admit rate",
    title="Posterior Predictive Check with 90% CI",
)
ax.legend()

plt.savefig("ucbadmit_plot.pdf")

# + [markdown] id="hVaUNn1mBt0F"
# # PGMs

# + colab={"base_uri": "https://localhost:8080/", "height": 288} id="EOYJu0IvYyP9" outputId="c9182a6b-6f0f-4c17-9771-737d503d2966"

# p344
dag = CausalGraphicalModel(
    nodes=["G", "D", "A"], edges=[("G", "D"), ("G", "A"), ("D", "A")]
)
out = dag.draw()
display(out)  
out.render(filename='admissions_dag', format='pdf')

# + colab={"base_uri": "https://localhost:8080/", "height": 288} id="fzcGU936ZIK2" outputId="aff43b4b-f9a6-4298-e603-9103019837fa"
# p345
dag = CausalGraphicalModel(
    nodes=["G", "D", "A"],
    edges=[("G", "D"), ("G", "A"), ("D", "A")],
    latent_edges = [("D", "A")]
)
out = dag.draw()
display(out)  
out.render(filename='admissions_dag_hidden', format='pdf')

# + [markdown] id="dMRqbhs1q3Uy"
# # Causal inference with  the latent DAG
#
# This is based on sec 6.3 (collider bias) of the Rethinking book.
# Code is from [Du Phan](https://fehiepsi.github.io/rethinking-numpyro/06-the-haunted-dag-and-the-causal-terror.html), code snippet 6.25. We change the names to match our current example: P (parents) -> D (department), C (child) -> A (admit).

# + [markdown] id="P--fqICIuKH9"
# ## Linear regression version

# + id="Styoo0wXVIqv"
N = 200  # number of samples
b_GP = 1  # direct effect of G on P
b_GC = 0  # direct effect of G on C
b_PC = 1  # direct effect of P on C
b_U = 2  # direct effect of U on P and C

with numpyro.handlers.seed(rng_seed=1):
    U = 2 * numpyro.sample("U", dist.Bernoulli(0.5).expand([N])) - 1
    G = numpyro.sample("G", dist.Normal().expand([N]))
    P = numpyro.sample("P", dist.Normal(b_GP * G + b_U * U))
    C = numpyro.sample("C", dist.Normal(b_PC * P + b_GC * G + b_U * U))
    df_gauss = pd.DataFrame({"C": C, "P": P, "G": G, "U": U})


# + id="17Tj_YW3uR8a"
def model_linreg(P, G, C):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + b_PC * P + b_GC * G
    numpyro.sample("C", dist.Normal(mu, sigma), obs=C)

data_gauss = {'P': df_gauss.P.values, 'G': df_gauss.G.values, 'C': df_gauss.C.values}

m6_11 = AutoLaplaceApproximation(model_linreg)
svi = SVI(model_linreg, m6_11, optim.Adam(0.3), Trace_ELBO(), **data_gauss)
p6_11, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_11.sample_posterior(random.PRNGKey(1), p6_11, (1000,))
print_summary(post, 0.89, False)

mcmc_run = MCMC(NUTS(model_linreg), num_warmup=200, num_samples=200, num_chains=4)
mcmc_run.run(random.PRNGKey(0), **data)
mcmc_run.print_summary(0.89)


# + id="1WJ0MO-yun23"
def model_linreg_hidden(P, G, U, C):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 1))
    b_U = numpyro.sample("U", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + b_PC * P + b_GC * G + b_U * U
    numpyro.sample("C", dist.Normal(mu, sigma), obs=C)


m6_12 = AutoLaplaceApproximation(model_linreg_hidden)
svi = SVI(
    model_linreg_hidden,
    m6_12,
    optim.Adam(1),
    Trace_ELBO(),
    P=d.P.values,
    G=d.G.values,
    U=d.U.values,
    C=d.C.values,
)
p6_12, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_12.sample_posterior(random.PRNGKey(1), p6_12, (1000,))
print_summary(post, 0.89, False)

# + [markdown] id="T29wJN6AuTHr"
# ## Logistic regression version
#
# We modify the scenario to match the UC Berkeley admissions scenario (with binary data) in sec 11.1.4.

# + id="01b-kAurskxC"
N = 200  # number of samples
b_GP = 1  # direct effect of G on P
b_GC = 0  # direct effect of G on C
b_PC = 1  # direct effect of P on C
b_U = 2  # direct effect of U on P and C

with numpyro.handlers.seed(rng_seed=1):
    #U = 2 * numpyro.sample("U", dist.Bernoulli(0.5).expand([N])) - 1
    U = numpyro.sample("U", dist.Normal().expand([N]))
    #G = numpyro.sample("G", dist.Normal().expand([N]))
    G = numpyro.sample("G", dist.Bernoulli(0.5).expand([N])) 
    P = numpyro.sample("P", dist.Normal(b_GP * G + b_U * U))
    #C = numpyro.sample("C", dist.Normal(b_PC * P + b_GC * G + b_U * U))
    logits = b_PC * P + b_GC * G + b_U * U
    probs = expit(logits)
    C = numpyro.sample("C", dist.BernoulliProbs(probs))
    df_binary = pd.DataFrame({"C": C, "G": G, "P": P,  "U": U, "probs": probs})

display(df_binary.head(10))


# + id="N5i75UBY7hNj"
def model_causal(C=None, G=None, P=None, U=None):
    U = numpyro.sample("U", dist.Normal(), obs=U)
    G = numpyro.sample("G", dist.Bernoulli(0.5), obs=G)
    P = numpyro.sample("P", dist.Normal(b_GP * G + b_U * U), obs=P) 
    logits = b_PC * P + b_GC * G + b_U * U
    probs = expit(logits)
    C = numpyro.sample("C", dist.BernoulliProbs(probs), obs=C)
    return np.array([C, G, P, U])



# + id="Q8SRcep98R5I"
def make_samples(C=None, G=None, P=None, U=None, nsamples=200):
  data_list = []
  with numpyro.handlers.seed(rng_seed=0):
    for i in range(nsamples):
      out = model_causal(C, G, P, U)
      data_list.append(out)
  df = pd.DataFrame.from_records(data_list, columns=['C','G','P','U'])
  return df

df_binary = make_samples() 
display(df_binary.head())



# + id="n4lGYRcMB1vF"

Cbar = df_binary['C'].values.mean()
Gbar = df_binary['G'].values.mean()
Pbar = df_binary['P'].values.mean()
Ubar = df_binary['U'].values.mean()
print([Cbar, Gbar, Pbar, Ubar])
print(b_GP * Gbar + b_U * Ubar) # expected Pbar



# + id="RA1ltAvl9Mjj"
N = len(df0)
prob_admitted0 = np.sum(df0.C.values)/N
prob_admitted1 = np.sum(df1.C.values)/N
print([prob_admitted0, prob_admitted1])


# + id="9jKjNnLdrPFS"
def model_logreg(C=None, G=None, P=None):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 0.1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 0.1))
    logits = a + b_PC * P + b_GC * G
    numpyro.sample("C", dist.Bernoulli(logits=logits), obs=C)

data_binary = {'P': df_binary.P.values, 'G': df_binary.G.values, 'C': df_binary.C.values}

warmup  = 1000
samples = 500
mcmc_run = MCMC(NUTS(model_logreg), num_warmup=warmup, num_samples=samples, num_chains=4)
mcmc_run.run(random.PRNGKey(0), **data)
mcmc_run.print_summary(0.89)

# + [markdown] id="aZCWR2nW6lyz"
# ## Counterfactual plot
#
# Similar to p140

# + id="bjhP_mPtDINb"
# p(C | do(G), do(P))
Pfixed = 0

df0 = make_samples(G=0, P=Pfixed, nsamples=200) 
display(df0.head())
Cbar0 = df0['C'].values.mean()

df1 = make_samples(G=1, P=Pfixed, nsamples=200) 
display(df1.head())
Cbar1 = df1['C'].values.mean()

print([Cbar0, Cbar1])

# + id="KRONqS_twABe"
sim_dat = dict(G=jnp.array([0,1]), P=jnp.array(Pfixed))
post = mcmc_run.get_samples()
pred = Predictive(model_logreg, post)(random.PRNGKey(22), **sim_dat)
print(pred['C'].shape)
print(np.mean(pred['C'], axis=0))


# + id="RLd_WHLm7zzb"
a_est = post['a'].mean()
b_PC_est = post['b_PC'].mean()
b_GC_est = post['b_GC'].mean()
P = Pfixed

G = np.array([0,1])
logits = a_est + b_PC_est * P + b_GC_est * G
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(expit(logits))

# + id="WoXQNbK_BXGF"
pred

# + id="97wFDtiqG3Mv"

