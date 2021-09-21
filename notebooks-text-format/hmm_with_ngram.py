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

# + [markdown] id="FTk3FQHnwtqS"
# ## Setup

# + colab={"base_uri": "https://localhost:8080/"} id="Qb4IIXge05EV" outputId="608a2bee-0120-4cb0-870c-94bb27f17b5e"
# !pip install nltk
# !pip install distrax

# + id="PfL2-4k9bNH7" colab={"base_uri": "https://localhost:8080/"} outputId="65853521-f632-45fe-c792-5cb67ef91a9d"
# !git clone --depth 1 https://github.com/probml/pyprobml /pyprobml &> /dev/null
# !curl -o bible.txt https://raw.githubusercontent.com/probml/probml-data/main/data/bible.txt
# %cd -q /pyprobml/scripts

# + colab={"base_uri": "https://localhost:8080/"} id="_wuzxMB_c3jn" outputId="6f56c105-b879-4bc8-8ffe-95f30601ee30"
# !pip install superimport

# + id="6gv1jZO7d7_T"
from conditional_bernoulli_mix_lib import ClassConditionalBMM
from conditional_bernoulli_mix_utils import fake_test_data, encode, decode, get_decoded_samples, get_emnist_images_per_class
from noisy_spelling_hmm import Word
from ngram_character_demo import ngram_model_fit, read_file, preprocessing, ngram_model_sample, ngram_loglikelihood
from distrax import HMM

from nltk.util import ngrams
from nltk import FreqDist, LidstoneProbDist

import numpy as np

import re
import string
from collections import defaultdict
from dataclasses import dataclass

from jax import vmap
import jax.numpy as jnp
import jax
from jax.random import PRNGKey, split
import distrax

import numpy as np
from matplotlib import pyplot as plt

# + [markdown] id="mj_1NJnY1tjj"
# ## ClassConditionalBMM

# + colab={"base_uri": "https://localhost:8080/"} id="VFlP1MTjde5R" outputId="f9007538-b1f5-4cb8-b651-b9acea991e54"
select_n = 25
dataset, targets = get_emnist_images_per_class(select_n)
dataset, targets = jnp.array(dataset), jnp.array(targets)

# + id="0jtZwZAKaeXR"
'''
During preprocessing of the text data, we removed punctuation whereas
case folding is not applied. The text data only contains
upper and lower case letters and hence there are 52 different characters
in total.
'''
n_char = 2*26


# + id="vyQBGTrseHIn"
def get_bmm(n_mix, dataset, targets):

    mixing_coeffs = jnp.array(np.full((n_char, n_mix), 1. / n_mix))

    p_min, p_max = 0.4, 0.6
    n_pixels = 28 * 28
    probs = jnp.array(np.random.uniform(p_min, p_max, (n_char, n_mix, n_pixels)))

    class_priors = jnp.eye(n_char)

    class_cond_bmm = ClassConditionalBMM(mixing_coeffs=mixing_coeffs, probs=probs, class_priors=class_priors, n_char=n_char)
    _ = class_cond_bmm.fit_em(dataset, targets, 8)
    return class_cond_bmm.model


# + [markdown] id="5Gjdsb8gyaPo"
# ## HMM

# + id="d95bBV9syWt1"
def get_transition_probs(bigram):
    probs = np.zeros((52, 52))
    for prev, pd in bigram.prob_dists.items():
      if prev==" ":
        continue
      lowercase = prev.islower()
      i = lowercase * 26 +  (ord(prev.lower()) - 97)
      for cur in pd.samples():
        if cur==" ":
          continue
        lowercase = cur.islower()
        j = lowercase * 26 + (ord(cur.lower()) - 97)
        probs[i, j] += pd.prob(cur)
    return probs


# + id="sAjvGMaBycpp"
def init_hmm_from_bigram(bigram, bmm):
    init_dist = distrax.Categorical(logits=jnp.zeros((n_char,)))
    probs = get_transition_probs(bigram)
    trans_dist = distrax.Categorical(probs=probs)
    obs_dist = bmm
    hmm = HMM(init_dist, trans_dist, obs_dist)
    return hmm


# + [markdown] id="XrtcdNllykLe"
# ## Loading Dataset

# + id="VkLgh57Ryjy0"
select_n = 25
dataset, targets = get_emnist_images_per_class(select_n)
dataset, targets = jnp.array(dataset), jnp.array(targets)

filepath = "/content/bible.txt"
text = read_file(filepath)
data = preprocessing(text, False)

# + [markdown] id="-aoKlnDTysER"
# ## Sampling Images

# + colab={"base_uri": "https://localhost:8080/"} id="z9teeVPlyudQ" outputId="1500784c-c8c6-41bb-a4cc-ad1591686457"
n, n_mix = 2, 30
bigram = ngram_model_fit(n, data, smoothing=1)
bmm = get_bmm(n_mix, dataset, targets)
hmm = init_hmm_from_bigram(bigram, bmm)

# + colab={"base_uri": "https://localhost:8080/"} id="VSb3VP3nyveA" outputId="6b280bc1-d7dd-47ee-921e-2a4cac452a04"
rng_key = PRNGKey(0)
seq_len = 6
Z, X = hmm.sample(seed=rng_key, seq_len=seq_len)


# + id="-CSmwLjP5sLJ"
def plot_seq(X, seq_len, figsize):
  fig, axes = plt.subplots(nrows=1, ncols=seq_len, figsize=figsize)
  for x, ax in zip(X, axes.flatten()):
      ax.imshow(x.reshape((28, 28)), cmap="gray")
  plt.tight_layout()
  plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 137} id="TAxZLsbCzNeC" outputId="4c5b6ecf-5242-4ff9-b45a-4fc1e67f6b93"
 plot_seq(X, seq_len, (10, 8))

# + [markdown] id="tORWPVOCzpP3"
# ## NGram

# + id="8Ha9lZVvzwNO"
n = 10
n_gram = ngram_model_fit(n, data, smoothing=1)

# + id="n8q2vHepzouW"
text_length = 11
prefix = "Christian"
Z = ngram_model_sample(n_gram, text_length, prefix)
log_p_Z = ngram_loglikelihood(n_gram, Z)


# + id="01XPDM0-0OF9"
def sample_img_seq_given_char_seq(bmm, z, rng_key):
  LL = 0
  T = len(z)
  keys = split(rng_key, T)
  Xs = []
  for t, key in enumerate(keys):
    cur_char = z[t]
    X = jnp.zeros((784, ))
    if cur_char!=" ":
      lowercase = cur_char.islower()
      c = lowercase * 26 +  (ord(cur_char.lower()) - 97)
      X = bmm.sample(seed=key)[c]
      log_p_X = bmm.log_prob(X)[c]

    Xs.append(X)
    
    LL += log_p_X
  
  return jnp.vstack(Xs), LL


# + colab={"base_uri": "https://localhost:8080/"} id="-ZNYwx7O03Oy" outputId="8ee856c0-4146-425a-bf5b-05349f625d37"
rng_key = PRNGKey(0)
images, LL = sample_img_seq_given_char_seq(bmm, Z, rng_key)

# + colab={"base_uri": "https://localhost:8080/"} id="Yvc1XeXc13L6" outputId="bd246cf6-2d18-4058-954c-4c2f9d46263b"
LL

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="AKXA4HNq8qYz" outputId="4f772d3d-abe3-4bd5-c08f-3c48039b3124"
Z

# + colab={"base_uri": "https://localhost:8080/", "height": 92} id="5sXsgB3p2STE" outputId="9443fa50-c398-420a-b663-b7568391a9c4"
plot_seq(images, text_length + len(prefix), figsize=(40, 20))
