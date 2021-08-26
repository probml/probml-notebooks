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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/numpyro_intro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="11FOMaCs74vK"
# [NumPyro](https://github.com/pyro-ppl/numpyro) is probabilistic programming language built on top of JAX. It is very similar to [Pyro](https://pyro.ai/), which is built on top of PyTorch.
# However, the HMC algorithm in NumPyro 
#  [is much faster](https://stackoverflow.com/questions/61846620/numpyro-vs-pyro-why-is-former-100x-faster-and-when-should-i-use-the-latter). 
#
# Both Pyro flavors are usually also [faster than PyMc3](https://www.kaggle.com/s903124/numpyro-speed-benchmark), and allow for more complex models, since Pyro is integrated into Python.
#
#
#

# + [markdown] id="1FfhOPQUHEdS"
# # Installation

# + id="Z5wEIBws1D6i"

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math

# + id="fZL8pcnG-nTT" colab={"base_uri": "https://localhost:8080/"} outputId="7af5756b-31dc-48b4-c3f2-9d9ee0f0afe4"
# Check number of CPUs
# !cat /proc/cpuinfo


# + id="2vlE1qOX-AjG"
# When running in colab pro (high RAM mode), you get 4 CPUs.
# But we need to  force XLA to use all 4 CPUs
# This is generally faster than running in GPU mode

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


# + colab={"base_uri": "https://localhost:8080/"} id="Xo0ejB5-7M3-" outputId="73b06f18-2bac-44d2-fbd3-d7710f5a3c07"
# http://num.pyro.ai/en/stable/getting_started.html#installation

#CPU mode: often faster in colab!
# !pip install numpyro

# GPU mode: as of July 2021, this does not seem to work
# #!pip install numpyro[cuda111] -f https://storage.googleapis.com/jax-releases/jax_releases.html


# + colab={"base_uri": "https://localhost:8080/"} id="qB5V5upMOMkP" outputId="065b5d42-2188-43ef-a835-b238546cce08"

import jax
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
print(jax.lib.xla_bridge.device_count())
print(jax.local_device_count())

import jax.numpy as jnp
from jax import random

# + id="lfOH0V2Knz_p"
import numpyro
#numpyro.set_platform('gpu')
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.diagnostics import hpdi, print_summary

from numpyro.infer.autoguide import AutoLaplaceApproximation

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)



# + [markdown] id="CSxk_HEeOOMn"
# # Example: 1d Gaussian with unknown mean.
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

# + [markdown] id="U2b74i9h02jf"
# ## Exact inference
#
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

# + id="HwHoLkHhaTTe" colab={"base_uri": "https://localhost:8080/"} outputId="44f88097-6ba4-45e9-b853-b26f8453e015"
mu = 8.5; tau = 1.0; sigma = 0.75;
hparams = (mu, tau, sigma)

y = 9.5
m = (sigma**2 * mu + tau**2 * y)/(sigma**2 + tau**2)
s2 = (sigma**2 * tau**2)/(sigma**2 + tau**2)
s = np.sqrt(s2)
print(m)
print(s)


# + id="S0cVreqiOLJh"

def model(hparams, y=None):
    prior_mean, prior_sd, obs_sd = hparams
    theta = numpyro.sample("theta", dist.Normal(prior_mean, prior_sd))
    y = numpyro.sample("y", dist.Normal(theta, obs_sd), obs=y)
    return y



# + [markdown] id="VEKpXZkLO9jb"
# ## Ancestral sampling

# + id="aOjKWT3Pk-f-"
def model2(hparams):
    prior_mean, prior_sd, obs_sd = hparams
    theta = numpyro.sample("theta", dist.Normal(prior_mean, prior_sd))
    yy = numpyro.sample("y", dist.Normal(theta, obs_sd))
    return theta, yy


# + colab={"base_uri": "https://localhost:8080/"} id="feTpLCESkiMN" outputId="c28459b3-b2a8-4e14-f83e-a152af364c33"
with numpyro.handlers.seed(rng_seed=0):
  for i in range(5):
    theta, yy = model2(hparams)
    print([theta, yy])

# + [markdown] id="mc2-_2hqN-vJ"
# ## MCMC
#
# See [the documentation](https://num.pyro.ai/en/stable/mcmc.html)

# + colab={"base_uri": "https://localhost:8080/", "height": 291, "referenced_widgets": ["e4bfd87390794a68aa49485dae909d76", "68f647dd697d4105b951f8b1add1aa1e", "5f23efd7fd0749fc92c471484184a507", "fb75111f76c24738bbece86df714531a", "84f2d4f77bc34b6da67e92a0a3ec9796", "bf2034c981ad469ea3a7daf0d0bea907", "49c888edc8ab41c2816b00c1a0734df9", "279ec5fbd3754509bec4818c582dc3c5", "bc497be0524345d0bc515a7e33d6ba32", "95186bb4ff5f47a194db2cfc6b30c9d9", "496502e2f5224aa8b072c9fe660db1d9", "cd61999d7518448e8b7be742a645bb23", "cc642f5391e24fa7b5fe91360f9c7088", "c354b8ff0fc447d08119a5eb974427ed", "dfeabbaf884d4fa5858e276462fb81cd", "b00ec3e6a2c746519ac614de35696f98", "68a984c00c0f4136981c46a1321f55b0", "b91b99d2a7af4898acb2de2870306281", "6616ede5e1d7480b923ed78f76298ef7", "40c5d9bce5de4ab3b2ccf74d8f804bea", "903492aa1c0449a28f96f4b253b59e0b", "27267d560b344670be984b0804c8e687", "cf2661a5dc384160a809c6020a0161c2", "10f937d564e644caa8ebe9a72e437d48", "3a5b368a87aa48b18fb2fc1c533e39b7", "3a7561d338424102a0ac45e195c199bf", "f32c84d240574f5287c83d690839f7b3", "3d26bb26b88349bdb20f31e6bca1f5b5", "617f3d2283ec44c79bce664f4e32e6f8", "4fa755ad43b742529b7f8f8a24fdbe9c", "54e2ed44a99246109df34d5d5de076c5", "04227c4d78e54777ac86c6892edf5ab6"]} id="reyW7c1mlmus" outputId="13b9c264-bdac-4962-b7ab-1ef3e7f6de21"
conditioned_model = numpyro.handlers.condition(model, {'y': y})
nuts_kernel = NUTS(conditioned_model)


mcmc = MCMC(nuts_kernel, num_warmup=200, num_samples=200, num_chains=4)
mcmc.run(rng_key_, hparams)

mcmc.print_summary()
samples  = mcmc.get_samples()

# + colab={"base_uri": "https://localhost:8080/"} id="FLYpKTG1e-Rg" outputId="5548de61-ac2d-49e0-da53-bd9327aff024"
print(type(samples))
print(type(samples['theta']))
print(samples['theta'].shape)

# + colab={"base_uri": "https://localhost:8080/"} id="lWPM9xxlnWca" outputId="076988be-bd7b-4ff4-eb05-b23dd716967d"
nuts_kernel = NUTS(model) # this is the unconditioned model
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=1000)
mcmc.run(rng_key_, hparams, y) # we need to specify the observations here

mcmc.print_summary()
samples  = mcmc.get_samples()


# + [markdown] id="7AkoKKxCe1U1"
# ## Stochastic variational inference
#
# See [the documentation](https://num.pyro.ai/en/stable/svi.html)

# + colab={"base_uri": "https://localhost:8080/", "height": 347} id="2Y2-i127mpz7" outputId="bb240cd1-2a14-40d8-d303-ed453bf8fbef"
# the guide must have the same signature as the model
def guide(hparams, y):
  prior_mean, prior_sd, obs_sd = hparams
  m = numpyro.param("m", y) # location
  s = numpyro.param("s", prior_sd, constraint=constraints.positive) # scale
  return numpyro.sample("theta", dist.Normal(m, s))


# The optimizer wrap these, so have unusual keywords
#https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html

#optimizer = numpyro.optim.Adam(step_size=0.001)
optimizer = numpyro.optim.Momentum(step_size=0.001, mass=0.1)
#svi = SVI(model, guide, optimizer, Trace_ELBO(), hparams=hparams, y=y) # specify static args to model/guide
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
nsteps = 2000
svi_result = svi.run(rng_key_, nsteps, hparams, y) # or specify arguments here

print(svi_result.params)
print(svi_result.losses.shape)
plt.plot(svi_result.losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");

# + colab={"base_uri": "https://localhost:8080/"} id="ILJOv6Pi_8tu" outputId="3d7e30d4-0acd-4032-b699-dcbbcae3702a"
print([svi_result.params['m'], svi_result.params['s']])

# + [markdown] id="POaURsS0SGHB"
# ## Laplace (quadratic) approximation
#
# See [the documentation](https://num.pyro.ai/en/stable/autoguide.html#autolaplaceapproximation)

# + colab={"base_uri": "https://localhost:8080/", "height": 300} id="adfMDI9USI_C" outputId="a45a3914-bd29-408d-fe68-c73a7890e99e"
guide_laplace = AutoLaplaceApproximation(model)
svi = SVI(model, guide_laplace, optimizer, Trace_ELBO(), hparams=hparams, y=y)
svi_run = svi.run(rng_key_, 2000)
params = svi_run.params
losses = svi_result.losses

plt.figure()
plt.plot(losses)

# + colab={"base_uri": "https://localhost:8080/"} id="_Kolw-R3Buuf" outputId="9dd38ff8-5bb2-4811-cee3-b64941eb2eef"
# Posterior is an MVN
# https://num.pyro.ai/en/stable/distributions.html#multivariatenormal
post = guide_laplace.get_posterior(params)
print(post)
m = post.mean
s = jnp.sqrt(post.covariance_matrix)
print([m, s])

# + colab={"base_uri": "https://localhost:8080/"} id="bcm4yD-vTXmE" outputId="bc8e59c2-6e3b-4c7b-aa00-fb6cdbe15c6c"
samples = guide_laplace.sample_posterior(rng_key_, params, (1000,))
print_summary(samples, 0.89, False)

# + [markdown] id="F0W7kpNcLyUm"
# # Example: Beta-Bernoulli model
#
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
# where $\alpha=\beta=10$. In the code, $\theta$ is called 
#  `latent_fairness`. 

# + id="9cUwrzZhE7Zj"
alpha0 = 10.0
beta0 = 10.0

def model(data):
    f = numpyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        numpyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


# + colab={"base_uri": "https://localhost:8080/"} id="qIVA50sFFc7u" outputId="5480dab5-a136-4b67-88c6-67f238099e44"
# create some data with 6 observed heads and 4 observed tails
data = jnp.hstack((jnp.ones(6), jnp.zeros(4)))
print(data)
N1 = jnp.sum(data==1)
N0 = jnp.sum(data==0)
print([N1, N0])

# + [markdown] id="1t2K9MElIYK1"
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

# + colab={"base_uri": "https://localhost:8080/"} id="-UncLKyeIfsw" outputId="cb41d985-94d5-4b1e-e470-3fc41449a804"
alpha_q = alpha0 + N1
beta_q = beta0 + N0
print('exact posterior: alpha={:0.3f}, beta={:0.3f}'.format(alpha_q, beta_q))

post_mean = alpha_q / (alpha_q + beta_q)
post_var = (post_mean * beta_q)/((alpha_q + beta_q) * (alpha_q + beta_q + 1))
post_std = np.sqrt(post_var)
print([post_mean, post_std])

# + id="iyvQ-KWt9aaj" colab={"base_uri": "https://localhost:8080/"} outputId="7bb7f5cb-f01c-477d-bf7d-28f6470d1402"
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)
print([inferred_mean, inferred_std])


# + [markdown] id="EG1wPcplIn5w"
# ## Variational inference

# + id="VaBKO12GIp19"
def guide(data):
    alpha_q = numpyro.param("alpha_q", alpha0,
                         constraint=constraints.positive)
    beta_q = numpyro.param("beta_q", beta0,
                        constraint=constraints.positive)
    numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


# + colab={"base_uri": "https://localhost:8080/", "height": 343} id="SOL8noBwJlP1" outputId="6af314df-ef08-40e5-b93a-e83629580b98"

#optimizer = numpyro.optim.Adam(step_size=0.001)
optimizer = numpyro.optim.Momentum(step_size=0.001, mass=0.1)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
nsteps = 2000
svi_result = svi.run(rng_key_, nsteps, data) 

print(svi_result.params)
print(svi_result.losses.shape)
plt.plot(svi_result.losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");

# + colab={"base_uri": "https://localhost:8080/"} id="fXlNOXAgKHWl" outputId="ba9bbea7-f6c1-4aec-9090-d5e7377c42e1"
# grab the learned variational parameters
alpha_q = svi_result.params["alpha_q"]
beta_q = svi_result.params["beta_q"]
print('variational posterior: alpha={:0.3f}, beta={:0.3f}'.format(alpha_q, beta_q))

post_mean = alpha_q / (alpha_q + beta_q)
post_var = (post_mean * beta_q)/((alpha_q + beta_q) * (alpha_q + beta_q + 1))
post_std = np.sqrt(post_var)
print([post_mean, post_std])

# + [markdown] id="1E6Urp6yLNg3"
# ## MCMC

# + colab={"base_uri": "https://localhost:8080/"} id="rwP4k478LO_G" outputId="c12defec-1fd7-4842-c066-15354fbb0e82"
nuts_kernel = NUTS(model) # this is the unconditioned model
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=1000)
mcmc.run(rng_key_, data) 

mcmc.print_summary()
samples  = mcmc.get_samples()

# + [markdown] id="V_j08oUgHMC8"
# # Distributions

# + [markdown] id="2cu50EWRHmeL"
# ## 1d Gaussian

# + id="Wv6tBbo-BQBm" colab={"base_uri": "https://localhost:8080/"} outputId="1640386f-0e68-48e7-981e-c517ac85b7eb"
# 2 independent 1d gaussians (ie 1 diagonal Gaussian)
mu = 1.5
sigma = 2
d = dist.Normal(mu, sigma)
dir(d)

# + id="viQgRPMWFH-7" colab={"base_uri": "https://localhost:8080/"} outputId="8d20e45f-caa2-47ef-8a21-156b95fc61ef"
rng_key, rng_key_ = random.split(rng_key)
nsamples = 1000
ys = d.sample(rng_key_, (nsamples,))
print(ys.shape)
mu_hat = np.mean(ys,0)
print(mu_hat)
sigma_hat = np.std(ys, 0)
print(sigma_hat)

# + [markdown] id="Iir5QxsEHvie"
# ## Multivariate Gaussian
#
#

# + id="h6MKLVypCGZY"
mu = np.array([-1, 1])
sigma = np.array([1, 2])
Sigma = np.diag(sigma)
d2 = dist.MultivariateNormal(mu, Sigma)

# + id="d7JQGBXi_7el" colab={"base_uri": "https://localhost:8080/"} outputId="fc953eac-f04e-4071-c2b1-162266220dc5"
#rng_key, rng_key_ = random.split(rng_key)
nsamples = 1000
ys = d2.sample(rng_key_, (nsamples,))
print(ys.shape)
mu_hat = np.mean(ys,0)
print(mu_hat)
Sigma_hat = np.cov(ys, rowvar=False) #jax.np.cov not implemented
print(Sigma_hat)

# + [markdown] id="UPyDu5DgIT76"
# ## Shape semantics
#
# Numpyro, [Pyro](https://pyro.ai/examples/tensor_shapes.html) and [TFP](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes) all distinguish between 'event shape' and 'batch shape'.
# For a D-dimensional Gaussian, the event shape is (D,), and the batch shape
# will be (), meaning we have a single instance of this distribution.
# If the covariance is diagonal, we can view this as D independent
# 1d Gaussians, stored along the batch dimension; this will have event shape () but batch shape (2,). 
#
# When we sample from a distribution, we also specify the sample_shape.
# Suppose we draw N samples  from a single D-dim diagonal Gaussian,
# and N samples from D 1d Gaussians. These samples will have the same shape.
# However, the semantics of logprob differs.
# We illustrate this below.
#

# + id="PUYN9T1GIbBb" colab={"base_uri": "https://localhost:8080/"} outputId="b1b26679-7f81-4d0d-9915-7df8ed169b2f"
d2 = dist.MultivariateNormal(mu, Sigma)
print(f'event shape {d2.event_shape}, batch shape {d2.batch_shape}') 
nsamples = 3
ys2 = d2.sample(rng_key_, (nsamples,))
print('samples, shape {}'.format(ys2.shape))
print(ys2)

# 2 independent 1d gaussians (same as one 2d diagonal Gaussian)
d3 = dist.Normal(mu, np.diag(Sigma))
print(f'event shape {d3.event_shape}, batch shape {d3.batch_shape}') 
ys3 = d3.sample(rng_key_, (nsamples,))
print('samples, shape {}'.format(ys3.shape))
print(ys3)

print(np.allclose(ys2, ys3))

# + id="kABLe1ypJob8" colab={"base_uri": "https://localhost:8080/"} outputId="858da445-12d1-41df-e9d4-b6eb693e7706"
y = ys2[0,:] # 2 numbers
print(d2.log_prob(y)) # log prob of a single 2d distribution on 2d input 
print(d3.log_prob(y)) # log prob of two 1d distributions on 2d input


# + [markdown] id="nsB0vIjYLa_6"
# We can turn a set of independent distributions into a single product
# distribution using the [Independent class](http://num.pyro.ai/en/stable/distributions.html#independent)
#

# + id="MXsP_SonLOpl" colab={"base_uri": "https://localhost:8080/"} outputId="3956e5e8-6f28-4221-dc43-51bff9112bef"
d4 = dist.Independent(d3, 1) # treat the first batch dimension as an event dimensions
print(d4.event_shape)
print(d4.batch_shape)
print(d4.log_prob(y))
