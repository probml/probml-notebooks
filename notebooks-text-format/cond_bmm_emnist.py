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

# + [markdown] id="M_qo7DmLJKLP"
# #Class-Conditional Bernoulli Mixture Model for EMNIST

# + [markdown] id="TU1pCzcIJHTm"
# ## Setup
#

# + id="400WanLyGA2C"
# !git clone --depth 1 https://github.com/probml/pyprobml /pyprobml &> /dev/null
# %cd -q /pyprobml/scripts

# + id="k1rLl6dHH7Wh"
# !pip install -q superimport
# !pip install -q distrax

# + id="cLpBn5KQeB46"
from conditional_bernoulli_mix_lib import ClassConditionalBMM
from conditional_bernoulli_mix_utils import fake_test_data, encode, decode, get_decoded_samples, get_emnist_images_per_class
from noisy_spelling_hmm import Word

from jax import vmap
import jax.numpy as jnp
import jax
from jax.random import PRNGKey, split

import numpy as np
from matplotlib import pyplot as plt

# + colab={"base_uri": "https://localhost:8080/"} id="ey9k06RweuKc" outputId="38131e5a-82fb-49db-c4d3-f4364a643152"
select_n = 25
dataset, targets = get_emnist_images_per_class(select_n)
dataset, targets = jnp.array(dataset), jnp.array(targets)

# + [markdown] id="KwNq7HYYLPO9"
# ## Initialization of Class Conditional BMMs

# + colab={"base_uri": "https://localhost:8080/"} id="UABtUDPjffFt" outputId="d873a708-542c-44e6-8c72-2c5908c7bbad"
n_mix = 30
n_char = 52

mixing_coeffs = jnp.array(np.full((n_char, n_mix), 1./n_mix))

p_min, p_max = 0.4, 0.6
n_pixels = 28 * 28
probs = jnp.array(np.random.uniform(p_min, p_max, (n_char, n_mix, n_pixels)))

class_priors = jnp.array(np.full((n_char,), 1./n_char))

cbm_gd = ClassConditionalBMM(mixing_coeffs=mixing_coeffs, probs=probs, class_priors=class_priors, n_char=n_char)
cbm_em = ClassConditionalBMM(mixing_coeffs=mixing_coeffs, probs=probs, class_priors=class_priors, n_char=n_char)

# + [markdown] id="Qa95Fua5Kc3i"
# ## Full Batch Gradient Descentt

# + colab={"base_uri": "https://localhost:8080/", "height": 336} id="PDzuEjs9Kewi" outputId="c81916c0-c6b7-45bd-d308-eab878afe281"
num_epochs, batch_size = 100, len(dataset)
losses = cbm_gd.fit_sgd(dataset.reshape((-1, n_pixels)), targets, batch_size, num_epochs = num_epochs) 

plt.plot(losses, color="k", linewidth=3)
plt.xlabel("Iteration")
plt.ylabel("Negative Log Likelihood")
plt.show()

# + [markdown] id="37mNMNrpInfh"
# ## EM Algorithm

# + colab={"base_uri": "https://localhost:8080/", "height": 336} id="FJeBzIKYfsUk" outputId="9d8db485-a251-4b1a-a6e5-93833c83dce6"
losses = cbm_em.fit_em(dataset, targets, 8)

plt.plot(losses, color="k", linewidth=3)
plt.xlabel("Iteration")
plt.ylabel("Negative Log Likelihood")
plt.show()


# + [markdown] id="NjCQpoH1Iuuf"
# ## Plot of the Probabilities of Components Distribution

# + id="KkyAHDW4JgyM"
def plot_components_dist(cbm, n_mix):
  fig = plt.figure(figsize=(45, 20))
  for k in range(n_mix):
      for cls in range(cbm.num_of_classes):
          plt.subplot(n_mix ,cbm.num_of_classes, cbm.num_of_classes*k + cls +1)
          plt.imshow(1 - cbm.model.components_distribution.distribution.probs[cls][k,:].reshape((28,28)), cmap = "gray") 
          plt.axis('off') 
  plt.tight_layout()
  plt.show()


# + [markdown] id="J8KLkCWpNAeF"
# ### GD

# + colab={"base_uri": "https://localhost:8080/", "height": 666} id="DSOiuNeAM8gl" outputId="dce9416a-b646-423d-b4bf-c78728db1cab"
plot_components_dist(cbm_gd, n_mix)

# + [markdown] id="FO31plUVNDSO"
# ### EM

# + id="ZM43qs6FfvlP" colab={"base_uri": "https://localhost:8080/", "height": 666} outputId="81a095f1-1099-4809-90a8-272dbed11662"
plot_components_dist(cbm_em, n_mix)

# + [markdown] id="IqRdcklzOeAY"
# ## Sampling

# + id="wgI6sFWKN4ax"
p1, p2, p3 =  0.4, 0.1, 2e-3
n_misspelled = 1  # number of misspelled words created for each class

vocab = ['book', 'bird', 'bond', 'bone', 'bank', 'byte', 'pond', 'mind', 'song', 'band']

rng_key = PRNGKey(0)
keys = [dev_array for dev_array in split(rng_key, len(vocab))]

# + id="x3GpZ8jbf11N" colab={"base_uri": "https://localhost:8080/"} outputId="5a348b69-bdf4-4f80-f059-1062ba2fbb88"
hmms = {word: Word(word, p1, p2, p3, n_char, "all", mixing_coeffs=cbm_em.model.mixture_distribution.probs,
                      initial_probs=cbm_em.model.components_distribution.distribution.probs, n_mix=n_mix) for word in vocab}

samples = jax.tree_map(lambda word, key: hmms[word].n_sample(n_misspelled, key), vocab, keys)

# + id="7VXVsobcg_KO" colab={"base_uri": "https://localhost:8080/"} outputId="3e915a79-7f5c-4131-d6ee-97f11c83d86f"
decoded_words = vmap(decode, in_axes = (0, None, None))(jnp.array(samples)[:, :, :, -1].reshape((n_misspelled * len(vocab), -1)), n_char + 1, "all")
get_decoded_samples(decoded_words)


# + [markdown] id="xrRy8MG0afR8"
# ### Figure

# + id="O0-HaN5rQAvP"
def plot_samples(samples):
    samples = np.array(samples)[:, :, :, :-1].reshape((-1, 28, 28))
    fig, axes = plt.subplots(ncols=4, nrows=10, figsize=(4, 10))
    fig.subplots_adjust(hspace = .2, wspace=.001)

    for i, ax in enumerate(axes.flatten()):
      ax.imshow(samples[i], cmap="gray")
      ax.set_axis_off()

    fig.tight_layout()
    plt.show()


# + id="EbZn9vrfhei4" colab={"base_uri": "https://localhost:8080/", "height": 728} outputId="114217bf-cadb-4331-82ef-b4844c038342"
plot_samples(samples)

# + [markdown] id="eNDmwV7EPyrR"
# ## Calculation of Log Likelihoods for Test Data

# + id="525MUl5HPe1K"
# noisy words
test_words = ['bo--', '-On-', 'b-N-', 'B---', '-OnD', 'b--D', '---D', '--Nd', 'B-nD', '-O--', 'b--d', '--n-']
test_images = fake_test_data(test_words, dataset, targets, n_char + 1, "all")


# + id="1dFCdVNgPYtJ"
def plot_log_likelihood(hmms, test_words, test_images, vocab):
    fig, axes = plt.subplots(4, 3, figsize=(20, 10))

    for i, (ax, img, word) in enumerate(zip(axes.flat, test_images, test_words)):

        flattened_img = img.reshape((len(img), -1))
        loglikelihoods = jax.tree_map(lambda w: jnp.sum(hmms[w].loglikelihood(word, flattened_img)), vocab)
        loglikelihoods = jnp.array(loglikelihoods)
      
        ax.bar(vocab, jnp.exp(jax.nn.log_softmax(loglikelihoods)), color="black")
        ax.set_title(f'{word}')

    plt.tight_layout()
    plt.show()


# + id="qv-Df8GEhfC4" colab={"base_uri": "https://localhost:8080/", "height": 784} outputId="9be6abf3-0ecc-4ef5-e301-380c5eac38ff"
plot_log_likelihood(hmms, test_words, test_images, vocab)
