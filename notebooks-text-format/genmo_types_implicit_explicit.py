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
# <a href="https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/genmo_types_implicit_explicit.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="BNh-HhR8h8HP"
# # Types of models: implicit or explicit models
#
# Author: Mihaela Rosca 
#
# We use a simple example below (a mixture of Gaussians in 1 dimension) to exemplify the different between explicit generative models (with an associated density which we can query) and implicit generative models (which have an associated density but which we cannot query for likelhoods, but we can sample from it).

# + id="tBtObwhMwgbb"
import random
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy



# + id="2iP2urBbxHD9"
sns.set(rc={"lines.linewidth": 2.8}, font_scale=2)
sns.set_style("whitegrid")


# + id="p9w3bdHgZry7"
# We implement our own very simple mixture, relying on scipy for the mixture
# components.

class SimpleGaussianMixture(object):

  def __init__(self, mixture_weights, mixture_components):
    self.mixture_weights = mixture_weights
    self.mixture_components = mixture_components

  def sample(self, num_samples):
    # First sample from the mixture
    mixture_choices = np.random.choice(range(0, len(self.mixture_weights)), 
                               p=self.mixture_weights, size=num_samples)
    # And then sample from the chosen mixture
    return np.array(
          [self.mixture_components[mixture_choice].rvs(size=1)
           for mixture_choice in mixture_choices])

  def pdf(self, x):
    value = 0.
    for index, weight in enumerate(self.mixture_weights):
      # Assuming using scipy distributions for components
      value += weight * self.mixture_components[index].pdf(x)
    return value


# + id="Rpd4OYPedk9B"
mix = 0.4
mixture_weight = [mix, 1.-mix]
mixture_components = [scipy.stats.norm(loc=-1, scale=0.1), scipy.stats.norm(loc=1, scale=0.5)]

mixture = SimpleGaussianMixture(mixture_weight, mixture_components)

# + id="-IzxYz_hfEYS" colab={"base_uri": "https://localhost:8080/"} outputId="10bfc3c1-8cd4-45bb-c092-ef3e68e4e9a5"
mixture.sample(10)

# + id="zIpUT4eRfecb" colab={"base_uri": "https://localhost:8080/"} outputId="8cd5b1ce-a7ca-449b-dc10-96e718117ebb"
mixture.pdf([10, 1])

# + id="2-LI7WjUgN8x" colab={"base_uri": "https://localhost:8080/"} outputId="42925aaa-d974-4215-8230-bb4eeada6987"
data_samples = mixture.sample(30)
len(data_samples)

# + id="5E0E99pybBYz" colab={"base_uri": "https://localhost:8080/"} outputId="2e069fc0-16a1-4e5b-d320-2a6a71db9763"
data_samples

# + colab={"base_uri": "https://localhost:8080/"} id="ILocArXZgJ0Q" outputId="ba6b0aed-ecb0-49c9-f729-611ca8d2c136"
plt.figure()
plt.plot(data_samples, [0] * len(data_samples), 'ro', ms=10, label='data')
plt.axis('off')
plt.ylim(-1, 2)
plt.xticks([])
plt.yticks([])



# + id="miDSJhh2gpWt"
# Use another set of samples to exemplify samples from the model
data_samples2 = mixture.sample(30)

# + [markdown] id="ZbspiQupioAN"
# ## Implicit generative model
#
# An implicit generative model only provides us with samples. Here for simplicity, we use a different set of samples obtained from the data distribution (i.e, we assume a perfect model).

# + colab={"base_uri": "https://localhost:8080/", "height": 483} id="hQXLcBMEgbKa" outputId="6f5dd9d6-238a-4478-8578-02ee957b0d82"
plt.figure(figsize=(12,8))
plt.plot(data_samples, [0] * len(data_samples), 'ro', ms=12, label='data')
plt.plot(data_samples2, [0] * len(data_samples), 'bd', ms=10,  alpha=0.7, label='model samples')
plt.axis('off')
# plt.ylim(-0.2, 2)
# plt.xlim(-2, 3)
plt.xticks([])
plt.yticks([])
plt.legend(framealpha=0.)

# + [markdown] id="Egl5IHBsix5t"
# ## Explicit generative models
#
# An explicit generative model allows us to query for likelihoods under the learned distribution for points in the input space of the data. Here too we assume a perfect model in the plot, by using the data distribution pdf.

# + colab={"base_uri": "https://localhost:8080/", "height": 483} id="7dBAgXdRhASg" outputId="829d6782-72ae-4fdc-b7ee-7e329c711452"
plt.figure(figsize=(12,8))
plt.plot(data_samples, [0] * len(data_samples), 'ro', ms=12, label='data')

x_vals = np.linspace(-2., 3., int(1e4))
pdf_vals = mixture.pdf(x_vals)
plt.plot(x_vals, pdf_vals, linewidth=4, label='model density')

plt.axis('off')
plt.ylim(-0.2, 2)
plt.xlim(-2, 3)
plt.xticks([])
plt.yticks([])
plt.legend(framealpha=0)
