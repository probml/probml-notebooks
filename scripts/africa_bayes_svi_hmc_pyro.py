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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/africa_bayes_svi_hmc_pyro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="DXAlhAzkwGMP"
# # Bayesian linear regression in Pyro (for Africa GDP dataset)
#
# We compare (stochastic) variational inference with HMC for Bayesian linear regression. We use the example from sec 8.1 of  [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/). 
# The code is modified from https://pyro.ai/examples/bayesian_regression.html and 
# https://pyro.ai/examples/bayesian_regression_ii.html. 
#
# For a NumPyro version (that uses Laplace approximation instead of SVI/ HMC), see https://fehiepsi.github.io/rethinking-numpyro/08-conditional-manatees.html.

# + colab={"base_uri": "https://localhost:8080/"} id="CaBMZERRwBcR" outputId="8d32e320-ab1e-4863-b5ba-05bdce5cefad"
# #!pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro

# !pip3 install pyro-ppl 

# + id="icxereXfwToC"
import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS
pyro.set_rng_seed(1)

from torch import nn
from pyro.nn import PyroModule

# + [markdown] id="zToQkjg-whSm"
# # Data
#
# The dataset has 3 variables: $A$ (whether a country is in Africa or not), $R$ (its terrain ruggedness), and $G$ (the log GDP per capita in 2000). We want to preict $G$ from $A$, $R$, and $A \times R$. The response variable is very skewed, so we log transform it.  

# + id="VosIzvMYxZQ6"
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# Dataset: Add a feature to capture the interaction between "cont_africa" and "rugged"
# ceofficeints are:  beta_a, beta_r, beta_ar
df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values,
                        dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

# + colab={"base_uri": "https://localhost:8080/", "height": 495} id="bQ13a_i7xxyS" outputId="f91801a7-c6d4-4955-c426-438484fd1ef4"
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]
sns.scatterplot(non_african_nations["rugged"],
            non_african_nations["rgdppc_2000"],
            ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
sns.scatterplot(african_nations["rugged"],
                african_nations["rgdppc_2000"],
                ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations");

# + id="oRVVnVGpx4an"


# + [markdown] id="iZnGZr9ryPEP"
# # Ordinary least squares
#
# We define the linear model as a simple neural network with no hidden layers. We fit it by using maximum likelihood, optimized by (full batch) gradient descent, as is standard for DNNs.

# + id="IjZLw7CCyzWH"
pyro.set_rng_seed(1)
#linear_reg_model = PyroModule[nn.Linear](3, 1)
linear_reg_model = nn.Linear(3, 1)

# + colab={"base_uri": "https://localhost:8080/"} id="dzYF0y6yzUZI" outputId="21619b64-a888-4cf6-f41a-6b4bbb62da8c"
print(type(linear_reg_model))

# + colab={"base_uri": "https://localhost:8080/"} id="YgmomZP-yYoA" outputId="a546855d-64e6-4212-e695-715b40e0d18b"


# Define loss and optimize
loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
num_iterations = 1500

def train():
    # run the model forward on the data
    y_pred = linear_reg_model(x_data).squeeze(-1)
    # calculate the mse loss
    loss = loss_fn(y_pred, y_data)
    # initialize gradients to zero
    optim.zero_grad()
    # backpropagate
    loss.backward()
    # take a gradient step
    optim.step()
    return loss

for j in range(num_iterations):
    loss = train()
    if (j + 1) % 200 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))



# + colab={"base_uri": "https://localhost:8080/"} id="Nkyu2Wo0yxw6" outputId="eb65c7e5-b464-47f0-d3e8-55bb99c69c64"

# Inspect learned parameters
print("Learned parameters:")
for name, param in linear_reg_model.named_parameters():
    print(name, param.data.numpy())

# + colab={"base_uri": "https://localhost:8080/"} id="FjmDtKOy0Iki" outputId="5ef199c7-41eb-49ad-8a01-22de196de2ce"
mle_weights = linear_reg_model.weight.data.numpy().squeeze()
print(mle_weights)
mle_bias = linear_reg_model.bias.data.numpy().squeeze()
print(mle_bias)
mle_params = [mle_weights, mle_bias]
print(mle_params)

# + colab={"base_uri": "https://localhost:8080/", "height": 431} id="qaua5kICzCCM" outputId="6de29c48-7820-485f-f2ac-fce5f50ab2c1"
fit = df.copy()
fit["mean"] = linear_reg_model(x_data).detach().cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = fit[fit["cont_africa"] == 1]
non_african_nations = fit[fit["cont_africa"] == 0]
fig.suptitle("Regression Fit", fontsize=16)
ax[0].plot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], "o")
ax[0].plot(non_african_nations["rugged"], non_african_nations["mean"], linewidth=2)
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
ax[1].plot(african_nations["rugged"], african_nations["rgdppc_2000"], "o")
ax[1].plot(african_nations["rugged"], african_nations["mean"], linewidth=2)
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations");


# + [markdown] id="_BC19x7uzJCd"
# # Bayesian model
#
# To make a Bayesian version of the linear neural network, we need to use a Pyro module instead of a torch.nn.module. This lets us replace torch tensors containg the parameters with random variables, defined by PyroSample commands. We also specify the likelihood function by using a plate over the multiple observations.

# + id="TKIH76N_zJ3y"
# https://forum.pyro.ai/t/extra-dim-added-to-some-variables-when-using-predictive/2904/6

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        N = x.shape[0]
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x) # (N,1)
        mean_vec = mean.squeeze(-1) # (N)
        with pyro.plate("data", x.shape[0]):   
          mu = pyro.deterministic("mu", mean_vec) # (N)       
          obs = pyro.sample("obs", dist.Normal(mean_vec, sigma), obs=y)
        return mean_vec


# + id="GtqHPETmt-r9" colab={"base_uri": "https://localhost:8080/"} outputId="630854ed-5917-4689-c497-2531d87a7d95"
# quick test of the model shapes

model = BayesianRegression(3, 1)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=10, warmup_steps=1)
mcmc.run(x_data, y_data)

predictive = Predictive(model, mcmc.get_samples(), return_sites=("obs", "mu", "_RETURN"))
hmc_samples_pred = predictive(x_data)
print(hmc_samples_pred.keys())
print(hmc_samples_pred['obs'].shape) # [S,170]
print(hmc_samples_pred['mu'].shape) # [S,170]
print(hmc_samples_pred['_RETURN'].shape) # [S,170]


# + [markdown] id="yJa76XkEBOvj"
# # Utilities

# + [markdown] id="MGadNJZJN9Zj"
# ## Summarize posterior 

# + id="XJIn7L3QBQxn"
def summary_np_scalars(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def summary_torch(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats



# + id="tlQ3ebQuLNX3"
def plot_param_post_helper(samples, label, axs):
  ax = axs[0]
  sns.distplot(samples['linear.bias'], ax=ax, label=label)
  ax.set_title('bias')
  for i in range(0,3):
    ax = axs[i+1]
    sns.distplot(samples['linear.weight'][:,0,i], ax=ax, label=label)
    ax.set_title(f'weight {i}')
 
def plot_param_post(samples_list, label_list):
  fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
  axs = axs.reshape(-1)
  fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
  n_methods = len(samples_list)
  for i in range(n_methods):
    plot_param_post_helper(samples_list[i], label_list[i], axs)
  ax = axs[-1]
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right');

def plot_param_post2d_helper(samples, label, axs, shade=True):
  ba = samples["linear.weight"][:,0,0] # africa indicator
  br = samples["linear.weight"][:,0,1] # ruggedness
  bar = samples["linear.weight"][:,0,2] # africa*ruggedness
  sns.kdeplot(ba, br, ax=axs[0], shade=shade, label=label)
  axs[0].set(xlabel="bA", ylabel="bR", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
  sns.kdeplot(br, bar, ax=axs[1], shade=shade, label=label)
  axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))

def plot_param_post2d(samples_list, label_list):
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
  axs = axs.reshape(-1)
  fig.suptitle("Cross-section of the Posterior Distribution", fontsize=16)
  n_methods = len(samples_list)
  shades = [False, True] # first method is contour, second is shaded
  for i in range(n_methods):
    plot_param_post2d_helper(samples_list[i], label_list[i], axs, shades[i])
  ax = axs[-1]
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right');
  #fig.legend() 


# + [markdown] id="IzGDTNImOBWC"
# ## Plot posterior predictions

# + id="FFxwlrxINvpO"
def plot_pred_helper(predictions, africa, ax):
  nations = predictions[predictions["cont_africa"] == africa]
  nations = nations.sort_values(by=["rugged"])
  ax.plot(nations["rugged"], nations["mu_mean"], color="k")
  ax.plot(nations["rugged"], nations["true_gdp"], "o")
  # uncertainty about mean
  ax.fill_between(nations["rugged"], nations["mu_perc_5"], nations["mu_perc_95"],
                alpha=0.2, color="k")
  # uncertainty about observations
  ax.fill_between(nations["rugged"], nations["y_perc_5"], nations["y_perc_95"],
                   alpha=0.15, color="k")
  

  ax.set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)")
  return ax

def make_post_pred_df(samples_pred):
  pred_summary = summary_torch(samples_pred)
  #mu = pred_summary["_RETURN"]
  mu = pred_summary["mu"]
  y = pred_summary["obs"]
  predictions = pd.DataFrame({
    "cont_africa": x_data[:, 0],
    "rugged": x_data[:, 1],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_gdp": y_data,
  })
  return predictions

def plot_pred(samples_pred):
  predictions = make_post_pred_df(samples_pred)
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
  plot_pred_helper(predictions, 0, axs[0])
  axs[0].set_title('Non-African nations')
  plot_pred_helper(predictions, 1, axs[1])
  axs[1].set_title('African nations')


# + [markdown] id="D5_w1RzE7dDP"
# # HMC inference

# + colab={"base_uri": "https://localhost:8080/"} id="aYVpPqAt4d39" outputId="3d9933d5-0ecc-4f8f-dcd7-02544cea6445"

pyro.set_rng_seed(1)


model = BayesianRegression(3, 1)
nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
mcmc.run(x_data, y_data)


# + id="ADaI_Exn-MyF" colab={"base_uri": "https://localhost:8080/"} outputId="78f102b4-f30b-4afc-c4f7-2f4cf8368169"
mcmc.diagnostics()

# + colab={"base_uri": "https://localhost:8080/"} id="llAcCKws7z7j" outputId="a4b20544-54fb-434c-f3d2-396221dc4941"

print(mcmc.get_samples().keys())
print(mcmc.get_samples()['linear.weight'].shape)
print(mcmc.get_samples()['linear.bias'].shape)

# + colab={"base_uri": "https://localhost:8080/"} id="xD6rjNeMBvdB" outputId="9c651ea1-3691-4529-de83-48a6e66bd261"
hmc_samples_torch = mcmc.get_samples()
summary_torch(hmc_samples_torch)

# + [markdown] id="XJ6m0hJ3DWgZ"
# ## Parameter posterior

# + colab={"base_uri": "https://localhost:8080/", "height": 822} id="ExWwOGYFB1-z" outputId="b9a6e364-8431-4a32-e764-ac4d7d1a7f99"
hmc_samples_params = {k: v.detach().cpu().numpy() for k, v in hmc_samples_torch.items()}
plot_param_post([hmc_samples_params], ['HMC'])


# + colab={"base_uri": "https://localhost:8080/"} id="vA26j9gfZTWQ" outputId="9b6af8b3-3586-426d-852c-c10483aadb68"
hmc_samples_params['linear.weight'].shape

# + colab={"base_uri": "https://localhost:8080/", "height": 520} id="dzFb2S8CqMRU" outputId="bd99ba82-a45b-41cd-a82e-0a99a7f917fb"
plot_param_post2d([hmc_samples_params], ['HMC'])

# + [markdown] id="bzxlvElIDZRk"
# ## Predictive posterior

# + colab={"base_uri": "https://localhost:8080/"} id="d79Q8d6Z-Cq0" outputId="31672b3e-ddd7-4f76-8835-3d8a46cd224e"
#predictive = Predictive(model, mcmc.get_samples(), return_sites=("obs", "_RETURN"))
predictive = Predictive(model, mcmc.get_samples(), return_sites=("obs", "mu", "_RETURN"))
hmc_samples_pred = predictive(x_data)
print(hmc_samples_pred.keys())
print(hmc_samples_pred['obs'].shape)
print(hmc_samples_pred['mu'].shape)
print(hmc_samples_pred['_RETURN'].shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 404} id="OET6RbjdNz45" outputId="420a65b4-ccda-4c15-e4d5-10283ac17a8c"

plot_pred(hmc_samples_pred)
plt.savefig('linreg_africa_post_pred_hmc.pdf', dpi=300)

# + [markdown] id="izz11PXGzuor"
# # Diagonal Gaussian variational posterior

# + [markdown] id="bej6wzYp2gfJ"
# ## Fit

# + id="kSFDgjeZzxH7"
pyro.set_rng_seed(1)

model = BayesianRegression(3, 1)
guide = AutoDiagonalNormal(model)

adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

# + colab={"base_uri": "https://localhost:8080/"} id="W8koXK66z7g3" outputId="79203810-a7ba-4a65-e27f-baad4d3ebe64"
pyro.clear_param_store()
num_iterations = 1000
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

# + [markdown] id="7uzP3mpH2iGz"
# ## Parameter posterior

# + colab={"base_uri": "https://localhost:8080/"} id="aPdPvnPwX8Bi" outputId="3aa58737-c2e1-4fa9-a979-e590db6c98b2"
post = guide.get_posterior()
nsamples = 800
samples = post.sample(sample_shape=(nsamples,))
print(samples.shape) # [800,5]
print(torch.mean(samples,dim=0)) # transform(sigma), weights 0:2, bias 


# + colab={"base_uri": "https://localhost:8080/"} id="B-Ue0NKjFT-z" outputId="c9d8747f-a88c-43bb-95f4-be719003282d"
# derive posterior quantiles for model parameters from the variational parameters
# note that we transform to the original parameter domain (eg sigma is in [0,10])
quant = guide.quantiles([0.5])
print(quant)

# + colab={"base_uri": "https://localhost:8080/"} id="DD64A-5YFKz6" outputId="69b2a744-9d6e-423a-b900-afd2678b6e01"
predictive = Predictive(model, guide=guide, num_samples=800, 
                        return_sites=("sigma", "linear.bias", "linear.weight"))
diag_samples_pred = predictive(x_data, y_data)
print(diag_samples_pred.keys())
print(diag_samples_pred['sigma'].shape)
print(diag_samples_pred['linear.weight'].shape)
print(diag_samples_pred['linear.bias'].shape)

print(torch.mean(diag_samples_pred['sigma']))
print(torch.mean(diag_samples_pred['linear.weight'], dim=0))
print(torch.mean(diag_samples_pred['linear.bias']))

# + colab={"base_uri": "https://localhost:8080/", "height": 968} id="_1DG1ifuc0Yr" outputId="a63eb8ca-65e6-4e5c-ac7f-60df2f12e38e"
weights = np.reshape(samples[:,1:4].detach().cpu().numpy(), (-1, 1, 3))
bias = samples[:,4].detach().cpu().numpy()

diag_samples_params = {'linear.weight': weights, 'linear.bias': bias}
print(diag_samples_params['linear.weight'].shape)

plot_param_post([hmc_samples_params, diag_samples_params, ], ['HMC', 'Diag'])
plt.savefig('linreg_africa_post_marginals_hmc_diag.pdf', dpi=300)

# + colab={"base_uri": "https://localhost:8080/", "height": 585} id="xbEb4skycsv_" outputId="dd91ceee-8b4c-4c1e-8778-a1d7fca93b3a"
plot_param_post2d([hmc_samples_params, diag_samples_params], ['HMC', 'Diag'])
plt.savefig('linreg_africa_post_2d_hmc_diag.pdf', dpi=300)

# + [markdown] id="L798ffc32vBq"
# ## Posterior predictive
#
# We extract posterior predictive distribution for obs, and the return value of the model (which is the mean prediction).

# + colab={"base_uri": "https://localhost:8080/"} id="OL4XiD4f22BZ" outputId="2a3b53a6-4210-41e1-b1cd-b7868fd30d27"
predictive = Predictive(model, guide=guide, num_samples=800, return_sites=("obs", "_RETURN"))
diag_samples_pred = predictive(x_data)
print(diag_samples_pred.keys())
print(diag_samples_pred['_RETURN'].shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 403} id="x2uQ62X14hFZ" outputId="b744aaf9-64d4-4df1-8b55-d2a3e8dab69a"

plot_pred(diag_samples_pred)

# + [markdown] id="YTGWsgzrdfSD"
# # Full Gaussian variational posterior

# + [markdown] id="XU3PsyB7d0Cy"
# ## Fit

# + id="L7_C_wdtdggs"
pyro.set_rng_seed(1)

model = BayesianRegression(3, 1)

from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)


adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

# + colab={"base_uri": "https://localhost:8080/"} id="MeeADsNady7D" outputId="466c1db7-0143-4c6a-e140-f741705c1a4b"
pyro.clear_param_store()
num_iterations = 1000
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

# + [markdown] id="OA4lVTued1eM"
# ## Parameter posterior

# + colab={"base_uri": "https://localhost:8080/"} id="EmCc5TMEd2o_" outputId="4fbf6512-5813-48b8-f280-ad27a78edc79"
post = guide.get_posterior()
nsamples = 800
samples = post.sample(sample_shape=(nsamples,))
print(samples.shape) # [800,5]
print(torch.mean(samples,dim=0)) # transform(sigma), weights 0:2, bias 

# + colab={"base_uri": "https://localhost:8080/"} id="2xs8IG76d7T1" outputId="63323a3c-4f6a-4500-b658-dcd238fbe7b4"
weights = np.reshape(samples[:,1:4].detach().cpu().numpy(), (-1, 1, 3))
bias = samples[:,4].detach().cpu().numpy()

full_samples_params = {'linear.weight': weights, 'linear.bias': bias}
print(full_samples_params['linear.weight'].shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 952} id="RbOLfzFyd9kb" outputId="7d96d758-62ee-40d0-d26a-682ef645099f"
plot_param_post([hmc_samples_params, full_samples_params, ], ['HMC', 'Full'])
plt.savefig('linreg_africa_post_marginals_hmc_full.pdf', dpi=300)

# + colab={"base_uri": "https://localhost:8080/", "height": 585} id="POE4Zk6PeM9A" outputId="75f77af2-9b66-4824-9809-01a2fcf7cfd4"
plot_param_post2d([hmc_samples_params, full_samples_params], ['HMC', 'Full'])
plt.savefig('linreg_africa_post_2d_hmc_full.pdf', dpi=300)

# + [markdown] id="JPQ-unGcecSL"
# ## Predictive posterior

# + colab={"base_uri": "https://localhost:8080/", "height": 403} id="SxHj3gwpeSg1" outputId="bd8f9e73-81f1-4bc8-c539-8ad80bc146ee"
predictive = Predictive(model, guide=guide, num_samples=800, return_sites=("obs", "_RETURN"))
full_samples_pred = predictive(x_data)
plot_pred(full_samples_pred)


# + id="BVk4jyzNejjY"

