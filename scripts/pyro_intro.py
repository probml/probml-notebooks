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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/pyro_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="BGe00SEV5KTl"
# [Pyro](https://pyro.ai/) is a probabilistic programming system built on top of PyTorch. It supports posterior inference based on MCMC and stochastic variational inference; discrete latent variables can be marginalized out exactly using dynamic programmming.

# + colab={"base_uri": "https://localhost:8080/"} id="P_uM0GuF5ZWr" outputId="4cafba49-31b3-4fcb-d5e5-a151912d2224"
# !pip install pyro-ppl 

# + id="3SBZ5Pmr5F0N"


import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import MCMC, NUTS, Predictive, HMC
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import EmpiricalMarginal

from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform
from pyro.distributions.util import scalar_like
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.util import ignore_experimental_warning

pyro.set_rng_seed(101)


# + [markdown] id="FE8OW3Zd50Nn"
# # Example: inferring mean of 1d Gaussian .
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
#

# + id="hYMFFGAMV0fW"

def model(hparams, data=None):
    prior_mean, prior_sd, obs_sd = hparams
    theta = pyro.sample("theta", dist.Normal(prior_mean, prior_sd))
    y = pyro.sample("y", dist.Normal(theta, obs_sd), obs=data)
    return y


# + [markdown] id="4511pkTB9GYC"
# ## Exact inference
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

# + colab={"base_uri": "https://localhost:8080/"} id="qNZ4aNNj9M-2" outputId="2d2ee5cf-2dc1-49da-93fa-574577b812f8"
mu = 8.5; tau = 1.0; sigma = 0.75; 
hparams = (mu, tau, sigma)

y = 9.5
m = (sigma**2 * mu + tau**2 * y)/(sigma**2 + tau**2) # posterior mean
s2 = (sigma**2 * tau**2)/(sigma**2 + tau**2) # posterior variance
s = np.sqrt(s2)
print(m)
print(s)


# + [markdown] id="tFIu6O-H8YFM"
# ## Ancestral sampling

# + colab={"base_uri": "https://localhost:8080/"} id="6dMY6oxJ8bEZ" outputId="3bb6bb9a-1793-4f92-aa9c-bbc4998064e0"
def model2(hparams, data=None):
    prior_mean, prior_sd, obs_sd = hparams
    theta = pyro.sample("theta", dist.Normal(prior_mean, prior_sd))
    y = pyro.sample("y", dist.Normal(theta, obs_sd), obs=data)
    return theta, y

for i in range(5):
  theta, y = model2(hparams)
  print([theta, y])


# + [markdown] id="r6ZtMB_cGhz4"
# ## MCMC
#
# See [the documentation](http://docs.pyro.ai/en/stable/mcmc.html)
#

# + colab={"base_uri": "https://localhost:8080/"} id="WQIhsROHH4uG" outputId="1ec3aa8e-dbdb-4f97-e626-21cb3f69c6e5"

nuts_kernel = NUTS(model)
obs = torch.tensor(y)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=50)
mcmc.run(hparams, obs)         
print(type(mcmc))

# + colab={"base_uri": "https://localhost:8080/"} id="K46LfO1rR-ty" outputId="ad6f1d3e-d8a3-4341-8fc7-93aec9c403a0"
samples = mcmc.get_samples()
print(type(samples))
print(samples.keys())
print(samples['theta'].shape)

# + colab={"base_uri": "https://localhost:8080/"} id="E1go0L6Szh-f" outputId="1b34fe04-bc90-4366-ae88-630093ba5f40"
mcmc.diagnostics()

# + colab={"base_uri": "https://localhost:8080/"} id="MPA4YwjaSrkp" outputId="5abaf70f-ac02-4165-be54-5610bb473471"
thetas = samples['theta'].numpy()
print(np.mean(thetas))
print(np.std(thetas))


# + [markdown] id="sjobNAZt8cOv"
# ##  Variational Inference
#
# See [the documentation](http://docs.pyro.ai/en/stable/inference_algos.html)
#
#
#
#
#

# + [markdown] id="_AVn8sW8GzLH"
# For the guide (approximate posterior), we use a [pytorch.distributions.normal](https://pytorch.org/docs/master/distributions.html#torch.distributions.normal.Normal).

# + colab={"base_uri": "https://localhost:8080/", "height": 366} id="bNX2i6fB8jmB" outputId="31f8e8ec-724a-4299-f5cf-6c7c89d7b715"
# the guide must have the same signature as the model
def guide(hparams, data):
  y = data
  prior_mean, prior_sd, obs_sd = hparams
  m = pyro.param("m", torch.tensor(y)) # location
  s = pyro.param("s", torch.tensor(prior_sd), constraint=constraints.positive) # scale
  return pyro.sample("theta", dist.Normal(m, s))


# initialize variational parameters
pyro.clear_param_store()

# set up the optimizer
#optimizer = pyro.optim.Adam({"lr": 0.001, "betas": (0.90, 0.999)})
optimizer = pyro.optim.SGD({"lr": 0.001, "momentum":0.1})

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 2000
# do gradient steps
obs = torch.tensor(y)
loss_history, m_history, s_history  = [], [], []
for t in range(num_steps):
    loss_history.append(svi.step(hparams, obs))
    m_history.append(pyro.param("m").item())
    s_history.append(pyro.param("s").item())

plt.plot(loss_history)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");

post_mean = pyro.param("m").item()
post_std = pyro.param("s").item()
print([post_mean, post_std])


# + [markdown] id="kUjGSy5y3Tcy"
# # Example: beta-bernoulli model
#
# Example is from [SVI tutorial](https://pyro.ai/examples/svi_part_i.html)
#
# The model is
# $$
# \begin{align}
# \theta &\sim \text{Beta}(\alpha, \beta) \\
# x_i &\sim \text{Ber}(\theta)
# \end{align}
# $$
# where $\alpha=\beta=10$. 
#

# + id="LFLetJ0hDUFS"
alpha0 = 10.0
beta0 = 10.0

def model(data):
    alpha0_tt = torch.tensor(alpha0)
    beta0_tt = torch.tensor(beta0)
    f = pyro.sample("theta", dist.Beta(alpha0_tt, beta0_tt))
    # loop over the observed data
    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


def model_binom(data):
    alpha0_tt = torch.tensor(alpha0)
    beta0_tt = torch.tensor(beta0)
    theta = pyro.sample("theta", dist.Beta(alpha0_tt, beta0_tt))
    data_np = [x.item() for x in data]
    N = len(data_np)
    N1 = np.sum(data_np)
    N0 = N-N1
    pyro.sample("obs", dist.Binomial(N, theta))
 


# + colab={"base_uri": "https://localhost:8080/"} id="8vr9twVP5JZN" outputId="6ad208b9-f7f0-4b7e-98fa-9fa1dfa64368"
# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))

data_np = [x.item() for x in data]
print(data)
print(data_np)

N = len(data_np)
N1 = np.sum(data_np)
N0 = N-N1
print([N1, N0])



# + colab={"base_uri": "https://localhost:8080/"} id="6Lz7luT0XK_0" outputId="3b7f13c7-f492-4f97-b5eb-fad6ff28d7ea"


# + [markdown] id="gRmA5WNW4OB2"
# ## Exact inference
#
# The posterior is given by
# $$
# \begin{align}
# \theta &\sim \text{Ber}(\alpha + N_1, \beta  + N_0) \\
# N_1 &= \sum_{i=1}^N [x_i=1] \\
# N_0 &= \sum_{i=1}^N [x_i=0]
# \end{align}
# $$
#

# + colab={"base_uri": "https://localhost:8080/"} id="3jCN9q0q5nMa" outputId="0a4f4a1a-da9b-499d-b527-e0dc548d04d1"
alpha1 = alpha0 + N1
beta1 = beta0 + N0
print('exact posterior: alpha={:0.3f}, beta={:0.3f}'.format(alpha1, beta1))
post_mean = alpha1 / (alpha1 + beta1)
post_var = (post_mean * beta1)/((alpha1 + beta1) * (alpha1 + beta1 + 1))
post_std = np.sqrt(post_var)
print([post_mean, post_std])



# + [markdown] id="eWkKUr-iV5-Q"
# ## MCMC

# + colab={"base_uri": "https://localhost:8080/"} id="xj5zuHCKV5Ni" outputId="82533034-e381-4d80-e204-65954686067f"

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=50)
mcmc.run(data)         
print(mcmc.diagnostics())
samples = mcmc.get_samples()
print(samples['theta'].shape)


# + colab={"base_uri": "https://localhost:8080/"} id="JypxLZfoWfQo" outputId="764a5976-6c2b-4ee4-a004-46eb724432c0"
thetas = samples['theta'].numpy()
print(np.mean(thetas))
print(np.std(thetas))

# + colab={"base_uri": "https://localhost:8080/"} id="15sbJzp2YPn5" outputId="849cf672-fd2b-41c1-e4b1-d72102d5c621"

nuts_kernel = NUTS(model_binom)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=50)
mcmc.run(data)         
print(mcmc.diagnostics())
samples = mcmc.get_samples()
print(samples['theta'].shape)

# + colab={"base_uri": "https://localhost:8080/"} id="6MfMCOd2YVnP" outputId="e6483f94-a423-4350-8963-d850117510fd"
thetas = samples['theta'].numpy()
print(np.mean(thetas))
print(np.std(thetas))


# + [markdown] id="210Q9non4lmw"
# ## Variational inference

# + id="HVp6t7uM336Q"
def guide(data):
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    pyro.sample("theta", dist.Beta(alpha_q, beta_q))

# + colab={"base_uri": "https://localhost:8080/", "height": 295} id="QqY4kp-T4nZ3" outputId="87d7b1bf-23e0-4bb2-eab2-9790050fbcb3"


#optimizer = pyro.optim.Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
optimizer = pyro.optim.SGD({"lr": 0.001, "momentum":0.1})

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 2000
loss_history = []
for step in range(n_steps):
  loss_history.append(svi.step(data))

plt.plot(loss_history)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");


# + colab={"base_uri": "https://localhost:8080/"} id="K-DsSLVv4_nM" outputId="f0fb31ea-285d-4829-ac44-378f35d18629"
# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()
print('variational posterior: alpha={:0.3f}, beta={:0.3f}'.format(alpha_q, beta_q))

post_mean = alpha_q / (alpha_q + beta_q)
post_var = (post_mean * beta_q)/((alpha_q + beta_q) * (alpha_q + beta_q + 1))
post_std = np.sqrt(post_var)
print([post_mean, post_std])

# + id="qIdMq9mGN429"

