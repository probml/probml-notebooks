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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/gaussian_param_inf_1d_numpyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="11FOMaCs74vK"
# # Inference for the parameters of a 1d Gaussian using a non-conjugate prior
#
# We illustrate various inference methods using the example in sec 4.3 ("Gaussian model of height") of [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). This requires computing $p(\mu,\sigma|D)$ using a Gaussian likelihood but a non-conjugate prior.
# The numpyro code is from [Du Phan's site](https://fehiepsi.github.io/rethinking-numpyro/04-geocentric-models.html).
#
#
#
#

# + id="Z5wEIBws1D6i"

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import os
import warnings
import pandas as pd

#from scipy.interpolate import BSpline
#from scipy.stats import gaussian_kde

# + colab={"base_uri": "https://localhost:8080/"} id="Rn0dCvGCr1YC" outputId="5a6919ff-fd7b-4205-d534-f4daae657c20"
# !mkdir figures

# + id="Xo0ejB5-7M3-"
# !pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro


# + colab={"base_uri": "https://localhost:8080/"} id="qB5V5upMOMkP" outputId="47aa9cd9-16c9-4033-b48f-8dfa0ef5ff0d"

import jax
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))

import jax.numpy as jnp
from jax import random, vmap

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# + id="lfOH0V2Knz_p"
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


# + colab={"base_uri": "https://localhost:8080/"} id="JZjT_8cKA1pP" outputId="113013a5-dc9d-4862-ef72-b6f2fa8b7dbd"
# !pip install arviz
import arviz as az


# + [markdown] id="qB83jECL_oWq"
# # Data
#
# We use the "Howell" dataset, which consists of measurements of height, weight, age and sex, of a certain foraging tribe, collected by Nancy Howell.

# + colab={"base_uri": "https://localhost:8080/", "height": 370} id="312Xjmye_2Lg" outputId="ae77d5a6-593e-43f5-a80a-009beff4f51c"
#url = 'https://github.com/fehiepsi/rethinking-numpyro/tree/master/data/Howell1.csv?raw=True'
url = 'https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/Howell1.csv'

Howell1 = pd.read_csv(url, sep=';')
d = Howell1
d.info()
d.head()

# + id="_mrNmkiEBPlH"
# get data for adults
d2 = d[d.age >= 18]
N = len(d2)
ndx = jax.random.permutation(rng_key, N)
data = d2.height.values[ndx]
N = 20 # take a subset of the 354 samples
data = data[:N]

# + [markdown] id="aSAr5iy2E0Cr"
# Empirical mean and std.

# + colab={"base_uri": "https://localhost:8080/"} id="QFYvhonkEpb-" outputId="7eb59b2c-fb2b-4a4f-d344-f9b1929f3ac4"
print(len(data))
print(np.mean(data))
print(np.std(data))

# + [markdown] id="oXUj4nsaCbR1"
# # Model
#
# We use the following model for the heights (in cm):
# $$
# \begin{align}
# h_i &\sim N(\mu,\sigma) \\
# \mu &\sim N(178, 20) \\
# \sigma &\sim U(0,50)
# \end{align}
# $$
#
# The prior for $\mu$ has a mean 178cm, since that is the height of 
# Richard McElreath, the author of the "Statisical Rethinking" book.
# The standard deviation is 20, so that 90\% of people lie in the range 138--218.
#
# The prior for $\sigma$ has a lower bound of 0 (since it must be positive), and an upper bound of 50, so that the interval $[\mu-\sigma, \mu+\sigma]$ has width 100cm, which seems sufficiently large to capture human heights.
#
#
# Note that this is not a conjugate prior, so we will just approximate the posterior.
# But since there are just 2 unknowns, this will be easy.
#

# + [markdown] id="52c6OQskEZiT"
# # Grid posterior

# + colab={"base_uri": "https://localhost:8080/"} id="6lFJF82pEac_" outputId="96bd96a9-2444-481b-d458-436ea79a4e7e"
mu_prior = dist.Normal(178, 20)
sigma_prior = dist.Uniform(0, 50)

mu_range = [150, 160]
sigma_range = [4, 14]
ngrid = 100
plot_square = False

mu_list = jnp.linspace(start=mu_range[0], stop=mu_range[1], num=ngrid)
sigma_list = jnp.linspace(start=sigma_range[0], stop=sigma_range[1], num=ngrid)
mesh = jnp.meshgrid(mu_list, sigma_list)
print([mesh[0].shape, mesh[1].shape])
print(mesh[0].reshape(-1).shape)
post = {"mu": mesh[0].reshape(-1), "sigma": mesh[1].reshape(-1)}
post["LL"] = vmap(
    lambda mu, sigma: jnp.sum(dist.Normal(mu, sigma).log_prob(data))
)(post["mu"], post["sigma"])
logprob_mu = mu_prior.log_prob(post["mu"])
logprob_sigma = sigma_prior.log_prob(post["sigma"])
post["prob"] = post["LL"] + logprob_mu + logprob_sigma
post["prob"] = jnp.exp(post["prob"] - jnp.max(post["prob"]))
prob = post["prob"] / jnp.sum(post["prob"]) # normalize over the grid

# + colab={"base_uri": "https://localhost:8080/", "height": 512} id="Cwg1FZlhGS-T" outputId="1f230ecd-f166-4a85-8988-e853989d03b6"
prob2d = prob.reshape(ngrid, ngrid)
prob_mu = jnp.sum(prob2d, axis=0)
prob_sigma = jnp.sum(prob2d, axis=1)

plt.figure()
plt.plot(mu_list, prob_mu, label='mu')
plt.legend()
plt.savefig('figures/gauss_params_1d_post_grid_marginal_mu.pdf', dpi=300)
plt.show()

plt.figure()
plt.plot(sigma_list, prob_sigma, label='sigma')
plt.legend()
plt.savefig('figures/gauss_params_1d_post_grid_marginal_sigma.pdf', dpi=300)
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 285} id="9wjZvO2GEfn-" outputId="4f68306b-ec27-499f-9924-85a382fef05b"
plt.contour(
    post["mu"].reshape(ngrid, ngrid),
    post["sigma"].reshape(ngrid, ngrid),
    post["prob"].reshape(ngrid, ngrid),
)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
if plot_square: plt.axis('square')
plt.savefig('figures/gauss_params_1d_post_grid_contours.pdf', dpi=300)
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 285} id="ARQMye8bEf2J" outputId="392e01a8-d763-474b-ac32-9a5cfd84d5e6"
plt.imshow(
    post["prob"].reshape(ngrid, ngrid),
    origin="lower",
    extent=(mu_range[0], mu_range[1], sigma_range[0], sigma_range[1]),
    aspect="auto",
)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
if plot_square: plt.axis('square')
plt.savefig('figures/gauss_params_1d_post_grid_heatmap.pdf', dpi=300)
plt.show()

# + [markdown] id="YYSMPLUYF_0b"
# Posterior samples.

# + id="qx_q5zTYFzsa"

nsamples = 5000 #int(1e4)
sample_rows = dist.Categorical(probs=prob).sample(random.PRNGKey(0), (nsamples,))
sample_mu = post["mu"][sample_rows]
sample_sigma = post["sigma"][sample_rows]
samples = {'mu': sample_mu, 'sigma': sample_sigma}



# + colab={"base_uri": "https://localhost:8080/", "height": 658} id="j71jJlWnpLRP" outputId="3af318a1-076e-4668-d7e0-2453722f3efe"
print_summary(samples, 0.95, False)


plt.scatter(samples['mu'], samples['sigma'], s=64, alpha=0.1, edgecolor="none")
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.axis('square')
plt.show()

az.plot_kde(samples['mu'], samples['sigma']);
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
if plot_square: plt.axis('square')
plt.savefig('figures/gauss_params_1d_post_grid.pdf', dpi=300)
plt.show()

# + [markdown] id="GFzitSc_ksgZ"
# posterior marginals.

# + colab={"base_uri": "https://localhost:8080/", "height": 570} id="depUbCulkuB9" outputId="5a9539a6-b4ac-4ac5-e24b-a59b6c93dbb0"
print(hpdi(samples['mu'], 0.95))
print(hpdi(samples['sigma'], 0.95))

fig, ax = plt.subplots()
az.plot_kde(samples['mu'], ax=ax, label=r'$\mu$')

fig, ax = plt.subplots()
az.plot_kde(samples['sigma'], ax=ax, label=r'$\sigma$')


# + [markdown] id="luc7FkMXGmEw"
# # Laplace approximation
#
# See [the documentation](http://num.pyro.ai/en/stable/autoguide.html#autolaplaceapproximation)

# + [markdown] id="4lpe17A-LUUE"
# ## Optimization

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="GiStL67NGnJi" outputId="9841d3e4-5e77-4241-b8d5-367cc9fad1b4"
def model(data):
    mu = numpyro.sample("mu", mu_prior)
    sigma = numpyro.sample("sigma", sigma_prior)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=data)

guide = AutoLaplaceApproximation(model)
svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), data=data)
svi_result = svi.run(random.PRNGKey(0), 2000)

plt.figure()
plt.plot(svi_result.losses)

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="bwrwHS73IJec" outputId="b044a435-9ec7-46da-d053-b1ae0a475ead"
start = {"mu": data.mean(), "sigma": data.std()}
guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
svi = SVI(model, guide, optim.Adam(0.1), Trace_ELBO(), data=data)
svi_result = svi.run(random.PRNGKey(0), 2000)

plt.figure()
plt.plot(svi_result.losses)


# + [markdown] id="6_s0bDxqIUEi"
# ## Posterior samples.

# + id="K6dQBDTGH3ex"
samples = guide.sample_posterior(random.PRNGKey(1), svi_result.params, (nsamples,))


# + colab={"base_uri": "https://localhost:8080/", "height": 662} id="PKb6dlS_pSKk" outputId="9fcc3d9f-62a5-43de-d535-b206a42df86d"
print_summary(samples, 0.95, False)


plt.scatter(samples['mu'], samples['sigma'], s=64, alpha=0.1, edgecolor="none")
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.show()

az.plot_kde(samples['mu'], samples['sigma']);
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
if plot_square: plt.axis('square')
plt.savefig('figures/gauss_params_1d_post_laplace.pdf', dpi=300)
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 570} id="hag3rzUcpcv3" outputId="ae69555f-63cc-4897-e6aa-62a7784ad044"
print(hpdi(samples['mu'], 0.95))
print(hpdi(samples['sigma'], 0.95))

fig, ax = plt.subplots()
az.plot_kde(samples['mu'], ax=ax, label=r'$\mu$')

fig, ax = plt.subplots()
az.plot_kde(samples['sigma'], ax=ax, label=r'$\sigma$')

# + [markdown] id="gZHkO4iBLBv-"
# ## Extract 2d joint posterior

# + [markdown] id="E21dH5NjMnJJ"
# The Gaussian approximation is over transformed parameters.

# + colab={"base_uri": "https://localhost:8080/"} id="nUPNkY_ILDz2" outputId="4bf84d41-feea-460d-972b-f0043a290d9b"
post = guide.get_posterior(svi_result.params)
print(post.mean)
print(post.covariance_matrix)


# + colab={"base_uri": "https://localhost:8080/"} id="kckoWeKhUPDr" outputId="e1d56ba4-db33-4cf3-c790-210b331dd10d"
def logit(p):
  return jnp.log(p/(1-p))

def sigmoid(a):
  return 1/(1+jnp.exp(-a))

scale=50; print(logit(7.7/scale)); print(sigmoid(-1.7)*scale)

# + colab={"base_uri": "https://localhost:8080/"} id="pzubiiMsXJPG" outputId="b8a16edb-5a94-402d-b0da-359642a5911f"
unconstrained_samples = post.sample(rng_key, sample_shape=(nsamples,))
constrained_samples = guide._unpack_and_constrain(unconstrained_samples, svi_result.params)

print(unconstrained_samples.shape)
print(jnp.mean(unconstrained_samples, axis=0))
print(jnp.mean(constrained_samples['mu'], axis=0))
print(jnp.mean(constrained_samples['sigma'], axis=0))



# + [markdown] id="rMv_7FRZMqAY"
# We can sample from the posterior, which return results in the original parameterization.

# + colab={"base_uri": "https://localhost:8080/"} id="UdnupIg0IuTk" outputId="31ec5ce9-fdcd-44bf-cca7-e0a117f97366"
samples = guide.sample_posterior(random.PRNGKey(1), params, (nsamples,))
x = jnp.stack(list(samples.values()), axis=0)
print(x.shape)
print('mean of ssamples\n', jnp.mean(x, axis=1))
vcov = jnp.cov(x)
print('cov of samples\n', vcov) # variance-covariance matrix

# correlation matrix
R = vcov / jnp.sqrt(jnp.outer(jnp.diagonal(vcov), jnp.diagonal(vcov)))
print('corr of samples\n', R)



# + [markdown] id="rjvvHbB0NNme"
# # Variational inference
#
# We use
# $q(\mu,\sigma) = N(\mu|m,s) Ga(\sigma|a,b)$
#

# + colab={"base_uri": "https://localhost:8080/", "height": 363} id="rE6C50KlL3hQ" outputId="24887973-fe0d-4fd0-b283-6b8129d05129"



def guide(data):
  data_mean = jnp.mean(data)
  data_std = jnp.std(data)
  m = numpyro.param("m", data_mean) 
  s = numpyro.param("s", 10, constraint=constraints.positive) 
  a = numpyro.param("a", data_std, constraint=constraints.positive) 
  b = numpyro.param("b", 1, constraint=constraints.positive) 
  mu = numpyro.sample("mu", dist.Normal(m, s))
  sigma = numpyro.sample("sigma", dist.Gamma(a, b))

optimizer = numpyro.optim.Momentum(step_size=0.001, mass=0.1)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
nsteps = 2000
svi_result = svi.run(rng_key_, nsteps, data=data)

print(svi_result.params)
print(svi_result.losses.shape)
plt.plot(svi_result.losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");

# + [markdown] id="k779nIjdTxu4"
# ## Extract Variational parameters.
#

# + colab={"base_uri": "https://localhost:8080/"} id="7ufCMqoZTpNV" outputId="3d32cd2f-a6c0-4f64-8ac6-bcbbb4337871"
print(svi_result.params)
a = np.array(svi_result.params['a'])
b = np.array(svi_result.params['b'])
m = np.array(svi_result.params['m'])
s = np.array(svi_result.params['s'])


# + colab={"base_uri": "https://localhost:8080/"} id="v51AAfH0Vh6G" outputId="4c51c80c-bb25-4f43-9267-0b04e0726d2a"
print('empirical mean', jnp.mean(data))
print('empirical std', jnp.std(data))

print(r'posterior mean and std of $\mu$')
post_mean = dist.Normal(m, s)
print([post_mean.mean, jnp.sqrt(post_mean.variance)])

print(r'posterior mean and std of unconstrained $\sigma$')
post_sigma = dist.Gamma(a,b)
print([post_sigma.mean, jnp.sqrt(post_sigma.variance)])

# + [markdown] id="jMb50OhpT10F"
# ## Posterior samples

# + id="l9KzXRibQaA2"
predictive = Predictive(guide, params=svi_result.params, num_samples=nsamples)
samples = predictive(rng_key, data)



# + colab={"base_uri": "https://localhost:8080/", "height": 662} id="qiVYfYuUqYCO" outputId="bad3ed9c-c808-485c-c510-92472b5f6356"
print_summary(samples, 0.95, False)


plt.scatter(samples['mu'], samples['sigma'], s=64, alpha=0.1, edgecolor="none")
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.show()

az.plot_kde(samples['mu'], samples['sigma']);
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
if plot_square: plt.axis('square')
plt.savefig('figures/gauss_params_1d_post_vi.pdf', dpi=300)
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 570} id="98TZ70O_q34k" outputId="901b6055-d0ea-4960-88b2-e41045ccaed1"
print(hpdi(samples['mu'], 0.95))
print(hpdi(samples['sigma'], 0.95))

fig, ax = plt.subplots()
az.plot_kde(samples['mu'], ax=ax, label=r'$\mu$')

fig, ax = plt.subplots()
az.plot_kde(samples['sigma'], ax=ax, label=r'$\sigma$')

# + [markdown] id="Egqg5eCHcGP2"
# # MCMC

# + colab={"base_uri": "https://localhost:8080/"} id="3qy3_SVgcCpR" outputId="c157d168-48bb-415c-a18b-81fedb66695a"
conditioned_model = numpyro.handlers.condition(model, {'data': data})
nuts_kernel = NUTS(conditioned_model)
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=nsamples)
mcmc.run(rng_key_, data)

mcmc.print_summary()
samples  = mcmc.get_samples()

# + colab={"base_uri": "https://localhost:8080/", "height": 662} id="R7ZEfXCkq0gI" outputId="d363495e-4c96-4bfc-9d65-f341ae1cbbbe"
print_summary(samples, 0.95, False)


plt.scatter(samples['mu'], samples['sigma'], s=64, alpha=0.1, edgecolor="none")
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.show()

az.plot_kde(samples['mu'], samples['sigma']);
plt.xlim(mu_range[0], mu_range[1])
plt.ylim(sigma_range[0], sigma_range[1])
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
if plot_square: plt.axis('square')
plt.savefig('figures/gauss_params_1d_post_mcmc.pdf', dpi=300)
plt.show()
