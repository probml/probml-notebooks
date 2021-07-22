# -*- coding: utf-8 -*-
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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/prob.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="9O6Oosl3cVls"
# # Probability
#
#
# In this notebook, we illustrate some basic concepts from probability theory using Python code.
#
#
#

# + id="hwEAA5TvcqXu"


import os
import time
import numpy as np
np.set_printoptions(precision=3)
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

import sklearn
import scipy.stats as stats
import scipy.optimize

import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows



# + [markdown] id="5THfLFUEcVlw"
# # Software libraries 
#
# There are several software libraries that implement standard probability distributions, and functions for manipulating them (e.g., sampling, fitting). We list some below.
#
#
# * [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)  We illustrate how to use this below.
#
# * [Tensorflow probability (TFP)](https://www.tensorflow.org/api_docs/python/tf/distributions)
# Similar API to scipy.stats.
#
# * [Distrax](https://github.com/deepmind/distrax) JAX version of TFP.
#
# * [Pytorch distributions library](https://pytorch.org/docs/stable/distributions.html). Similar to TFP.
#
#
# * [NumPyro distributions library](https://numpyro.readthedocs.io/en/latest/distributions.html) has a similar interface to PyTorch distributions, but uses JAX as the backend.
#  
#  
# In this notebook, we mostly focus on scipy.stats.

# + [markdown] id="ghoWb7zpKQ1U"
# # Basics of Probability theory
#
#

# + [markdown] id="KHmVxJGVMrXH"
# ## What is probability?
#
# We will not go into mathematical detail, but focus on intuition.
#
# *   Two main "schools of thought"
# *   **Bayesian probability** = degree of belief
#   * $p(heads=1)=0.5$ means you think the event that a particular coin will land heads is 50% likely. 
# * **Frequentist probability** = long run frequencies
#   * $p(heads=1)=0.5$ means that the empirical fraction of times this event will occur across infinitely repeated trials is 50%
# * In practice, the philosophy does not matter much, since both interpretations must satisfy the same basic axioms
#

# + [markdown] id="DMmZOT2hNTsP"
# ## Random variables and their distributions
#
# *   Let $X$ be a (discrete) **random variable** (RV) with $K$ possible values  $\mathcal{X}  = \{1,...,K\}$.
# *   Let $X=x$ be the **event** that $X$ has value $x$, for some state $x \in \cal{X}$.
# * We require $0 \leq p(X=x) \leq 1$
# * We require
#      $$\sum_{x \in  \cal{X}} p(X=x) = 1$$
# * Let $p(X) = [p(X=1), …, p(X=K)]$ be the distribution or **probability mass function** (pmf) for RV $X$.
# * We can generalize this to continuous random variables, which have an infinite number of possible states, using a **probability density function** (pdf) which satisfies
# $$
#     \int_{x \in \cal{X}} p(X=x) dx = 1
#     $$

# + [markdown] id="dwnq2H-ZLwQ1"
# ## Conjunction and disjunction of events
# *   The probability of events $X=x$ AND $Y=y$ is denoted $p(X=x \land Y=y)$ or just $p(X=x,Y=y)$.
#
# *   If two RVs are **independent**, then 
# $$p(X, Y) = p(X) * p(Y)$$.
#

# + [markdown] id="JRf-ptLJKpKE"
#
#
# ![Screen Shot 2021-04-13 at 2.41.09 PM.png](data:image/png;base64,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)
#

# + [markdown] id="QJPPSllXL0Tb"
#
#
# *   The probability of event $X=x$ OR $Y=y$ is 
# $$p(X=x \lor Y=y) = p(X=x) + p(Y=y) - p(X=x \land Y=y)$$
# *   For disjoint events (that cannot co-occur), this becomes
# $$p(X=x \lor Y=y) = p(X=x) + p(Y=y)$$

# + [markdown] id="v-sGU3V7KjWg"
# ## Conditional probability, sum rule, product rule, Bayes rule 
#
# * The **conditional probability** of Y=y given X=x is defined to be
#
#  $$   p(Y=y|X=x) = \frac{p(X=x,Y=y)}{p(X=x)} $$
#
# * Hence we derive the **product rule**
#  $$ 
#  \begin{align}
#    p(X=x, Y=y) &= p(Y=y|X=x) * p(X=x)\\
#               &= p(X=x|Y=y) * p(Y=y) 
#               \end{align}
#   $$
#
# * If $X$ and $Y$ are independent, then $p(Y|X)=p(Y)$ and $p(X|Y)=p(X)$, so
#   $p(X,Y)=p(X) p(Y)$.
#
# * The marginal probability of $X=x$ is given by the **sum rule**
#    $$ p(X=x) = \sum_y p(X=x, Y=y)$$
#
# * Hence we derive **Bayes' rule**
#    $$
#    \begin{align}
#    p(Y=y|X=x) &= p(X=x,Y=y) /  p(X=x)\\
#    &=\frac{p(X=x|Y=y) * p(Y=y)}
#    {\sum_{y'} p(X=x|Y=y) * p(Y=y)}
#    \end{align}
#    $$

# + [markdown] id="3lQaNrciOZRJ"
# ## Bayesian inference
#
# Bayes rule is often used to compute a distribution over possible values of a **hidden variable** or **hypothesis** $h \in \cal{H}$ after observing some evidence $Y=y$. We can write this as follows:
#
# $$
# \begin{align}
# p(H=h|Y=y) &= \frac{p(H=h) p(Y=y|H=h)}{p(Y=y)} \\
# \text{posterior}(h|y) &= \frac{\text{prior}(h) * \text{likelihood}(y|h)}{\text{marginal-likelihood}(y)}
#  \end{align}
#  $$ 
#
# * The **prior** encodes what we believe about the state before we see any data.
# * The **likelihood** is the probability of observing the data given each possible hidden state. 
# * The **posterior** is our new belief state, after seeing the data.
# * The **marginal likelihood** is a normalization constant, independent of the hidden state, so can usually be ignored.
#
# Applying Bayes rule to infer a hidden quantity from one or more observations is called **Bayesian inference** or **posterior inference**. (It used to be called **inverse probability**, since it reasons backwards from effects to causes.)

# + [markdown] id="VgOhwx-iPnci"
# ## Example: Bayes rule for COVID diagnosis
#
# Consider estimating if someone has COVID $H=1$ or not $H=0$ on the basis of a PCR test. The test can either return a positive result $Y=1$ or a negative result $Y=0$. The reliability of the test is given by the following observation model.

# + [markdown] id="NerKdR6PQ7ar"
# ![Screen Shot 2021-04-13 at 2.59.24 PM.png](data:image/png;base64,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wXv4+uuvJ3CoVPuiSlXooHrWRBUCoBoe29yR+gja1KlTJ/ezn/3M3XDDDcoc9tlnH/eLX/zCDRs2zO21115u+PDhGiHwsccec/vvv78OCHxQgZ555hl3wgkn6CCKHotogn/605/c17/+dXX4+vOf/+zGjx/vcAD89re/7ebMmROyFvz71FNPuf3220/zfetb31JnskMOOcThnd6cRLCrCy64IHoEA2p47kknneQuvPBCd9ddd0X3Mx3AcNmiHuYTiCh82QbokK6Sv7yHl156qTLqU089VZ047777bverX/3Kde/e3Z155plavWnTpjnuw1R/97vfuUcffVRx6dy5sxs6dGg0sMbT3XLLLTopOO+88/R9u+OOO5rUVLDMN8Ih+oOddtpJo2DeeOONrmfPnm7bbbd1P/7xjzU+CZMUYtRATzzxhPYz3wffSqBly5a5n/70p9rmm2++2a1cudI1Z/vCc2viV2YRRhVEQGY3XmZn/p133tFajBkzxn/ve9/Lq0b33nuvl48zZ9rDDjvMy6AfpZOP3R999NHR+bx583y/fv2i8x/84Ad+jz328KST6HHR9alTp3rxDo7OOejRo8d69T3++OO1TbKvVSJtvifCQPztt98eJRdPZS8zPy/MK7rWHAfCJBLPvfrqq714RUePOuuss/xtt90WnWc6oC/pQ2HCURL6VVZW0Xm1HrzxxhsoKL0M8okq0pfiHR5dGzlypKajbwK99dZbXlap/kc/+lG45NOlkyiUmlcYVJSukANhIP6iiy7ywrC0nPi7na4c6nDnnXdGtySwleZbu3atXpMVrhcmGd2XXRA838zGG2/sX3zxxeg634JMkqJzDpqjfYkH1MCJrUQqzOrLEamPpfeWW26ZsaXCMBIiqNatWzv56Nzs2bPdVVddFeWjjK222io6z3TASoaZWrGBoohSGPdCZ4sLGZgyPa5k19k+QxhoVB7inXYxkd6oUaM0FG+UIMMBIh/C9X75y1+OUrDCqwVCJJNKH374oWvTpk1ik8t06dq2baurmMmTJ0dFpEvH+wGxei2GCo1wSD+yVU4moo6snuMx23nvd911V3fyySdHYXh5J3feeedEMc3RvsQDauDEdCIV7qRskfp22223ktSub9++OcuJD54kZkNDme3pEh4RVVju5yxIErBB4jbbbKMfZj7pU9OwP9bAgQOVER1x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)

# + [markdown] id="THSdTh8ZRYnY"
# Using data from https://www.nytimes.com/2020/08/04/science/coronavirus-bayes-statistics-math.html, we set sensitivity to 87.5\% and the specificity to 97.5\%.
#
# We also need to specify the prior probability $p(H=1)$; this is known as the prevalence. This varies over time and place, but let's pick $p(H=1)=0.1$ as a reasonable estimate.
#
#

# + [markdown] id="SYeDC_S9Sv9O"
# If you test positive:
#
# \begin{align}
# p(H=1|Y=1) 
#  &= \frac{p(Y=1|H=1) p(H=1)}
# {p(Y=1|H=1) p(H=1) + p(Y=1|H=0) p(H=0)}
# = 0.795
# \end{align}
#
# If you test negative:
# \begin{align}
# p(H=1|Y=0) 
#  &= \frac{p(Y=0|H=1) p(H=1)}
# {p(Y=0|H=1) p(H=1) + p(Y=0|H=0) p(H=0)}
# =0.014
# \end{align}

# + [markdown] id="I8Cnt3DSdjPC"
# ## Code to reproduce the above.

# + id="8vmPaBF5SDSe"
def normalize(x):
  return x / np.sum(x)

def posterior_covid(observed, prevalence=None, sensitivity=None):
  # observed = 0 for negative test, 1 for positive test
  # hidden state = 0 if no-covid, 1 if have-covid
  if sensitivity is None:
    sensitivity = 0.875
  specificity = 0.975
  TPR = sensitivity; 
  FNR = 1-TPR
  TNR = specificity
  FPR = 1-TNR
  # likelihood(hidden, obs)
  likelihood_fn = np.array([[TNR, FPR], [FNR, TPR]])
  # prior(hidden)
  if prevalence is None:
    prevalence = 0.1
  prior = np.array([1-prevalence, prevalence])
  likelihood = likelihood_fn[:, observed].T
  posterior = normalize(prior * likelihood)
  return posterior


# + [markdown] id="kgewI_XnSJxI"
# For a prevalence of $p(H=1)=0.1$

# + colab={"base_uri": "https://localhost:8080/"} id="vIieMaqISIdp" outputId="e4f49577-7b77-4c23-ef4e-13a86fdece4a"
print(posterior_covid(1)[1]*100)
print(posterior_covid(0)[1]*100)


# + [markdown] id="GkOukzykTIdC"
# For a prevalence of $p(H=1)=0.01$

# + colab={"base_uri": "https://localhost:8080/"} id="-YDB2fzRSVtb" outputId="936479d7-2e97-42de-d859-09dda707583f"

print(posterior_covid(1, 0.01)[1]*100) # positive test 
print(posterior_covid(0, 0.01)[1]*100) # negative test

# + id="7vTeODztyAO0" outputId="86d97d05-2e41-4aad-9fb5-da4d8cc1f5a7" colab={"base_uri": "https://localhost:8080/"}
pop = 100000
infected = 0.01*pop
sens = 87.5/100
spec = 97.5/100
FPR = 1-spec
FNR = 1-sens
print([FPR, FNR])
true_pos = sens * infected
false_pos = FPR * (pop-infected)
num_pos = true_pos + false_pos
posterior = true_pos/num_pos
print([infected, true_pos, false_pos, num_pos, posterior])

# + [markdown] id="-JGyBT1SKCVu"
# # Univariate distributions

# + [markdown] id="3A7jNmydcVl5"
# ## Univariate Gaussian (normal)  <a class="anchor" id="scipy-unigauss"></a>
#

# + id="lPG3_HJtcVl7" colab={"base_uri": "https://localhost:8080/", "height": 872} outputId="058e1ce1-419e-4916-b36a-ff6aa6a144e1"
from scipy.stats import norm 

rv = norm(0, 1) # standard normal

fig, ax = plt.subplots(1,2, figsize=(10,5))
X = np.linspace(-3, 3, 500)
ax[0].plot(X, rv.pdf(X))
ax[0].set_title("Gaussian pdf")
ax[1].plot(X, rv.cdf(X))
ax[1].set_title("Gaussian cdf")
plt.show()


plt.figure()
plt.plot(X, rv.pdf(X))
plt.title("Gaussian pdf")

plt.figure()
plt.plot(X, rv.cdf(X))
plt.title("Gaussian cdf")

plt.show()


# + id="f2T_c7I3cVl-" colab={"base_uri": "https://localhost:8080/"} outputId="a369184e-19f5-423d-9ffe-7814127b2420"
# Samples

np.random.seed(42)
mu = 1.1
sigma = 0.1
dist = norm(loc=mu, scale=sigma) # create "frozen" distribution
N = 10
x = dist.rvs(size=N) # draw N random samples
print(x.shape)
print(x)

np.random.seed(42)
x2 = norm(mu, sigma).rvs(size=N)
assert np.allclose(x, x2)



# + id="ct7VPGRtcVmB" colab={"base_uri": "https://localhost:8080/"} outputId="9772b357-6aec-44bb-93fb-7f96a9453f35"
# pdf, cdf, inverse cdf

logprob = dist.logpdf(x) # evaluate log probability of each sample
print(logprob.shape)

p = dist.cdf(x)
x3 = dist.ppf(p) # inverse CDF
assert np.allclose(x, x3)

# + [markdown] id="cWceYui9cVmK"
# ## Gamma distribution  <a class="anchor" id="scipy-gamma"></a>

# + id="av_umAKvcVmL" colab={"base_uri": "https://localhost:8080/", "height": 283} outputId="791815e8-778a-43eb-930f-baef6186c1eb"
from scipy.stats import gamma

x = np.linspace(0, 7, 100)
b = 1
plt.figure()
for a in [1, 1.5, 2]:
    y = gamma.pdf(x, a, scale=1/b, loc=0)
    plt.plot(x, y)
plt.legend(['a=%.1f, b=1' % a for a in [1, 1.5, 2]])
plt.title('Gamma(a,b) distributions')
#save_fig('gammaDistb1.pdf')
plt.show()


# + [markdown] id="6yxqhcZccVmO"
# ## Zipf's law <a class="anchor" id="zipf"></a>
#
# In this section, we study the empirical word frequencies derived from H. G. Wells' book [The time machine](https://en.wikipedia.org/wiki/The_Time_Machine).
# Our code is based on https://github.com/d2l-ai/d2l-en/blob/master/chapter_recurrent-neural-networks/lang-model.md
#

# + id="rUwJ28s2ixOl" colab={"base_uri": "https://localhost:8080/"} outputId="8031f331-26e2-4a76-eb8c-9f3997267aad"
import collections
import re
import urllib

url = 'https://raw.githubusercontent.com/probml/pyprobml/master/data/timemachine.txt' 
bytes = urllib.request.urlopen(url).read()
string = bytes.decode('utf-8')
words = string.split()
print(words[:10])
words = [re.sub('[^A-Za-z]+', ' ', w.lower()) for w in words]
print(words[:10])


# + id="7x0BAUNocVmR" colab={"base_uri": "https://localhost:8080/"} outputId="3f2481cc-5826-4d73-afad-a5d43b9757e4"
# Convert sequence of words into sequence of n-grams for different n

# Unigrams
wseq = words # [tk for st in raw_dataset for tk in st]
print('First 10 unigrams\n', wseq[:10])

# Bigrams
word_pairs = [pair for pair in zip(wseq[:-1], wseq[1:])]
print('First 10 bigrams\n', word_pairs[:10])

# Trigrams
word_triples = [triple for triple in zip(wseq[:-2], wseq[1:-1], wseq[2:])]
print('First 10 trigrams\n', word_triples[:10])

# + id="MdERSjXwcVmU" colab={"base_uri": "https://localhost:8080/"} outputId="6e7255fb-e6c3-485e-f1b2-72497bdd8f79"
# ngram statistics

counter = collections.Counter(wseq)
counter_pairs = collections.Counter(word_pairs)
counter_triples = collections.Counter(word_triples)

wordcounts = [count for _,count in counter.most_common()]
bigramcounts = [count for _,count in counter_pairs.most_common()]
triplecounts = [count for _,count in counter_triples.most_common()]

print('Most common unigrams\n', counter.most_common(10))
print('Most common bigrams\n', counter_pairs.most_common(10))
print('Most common trigrams\n', counter_triples.most_common(10))

# + id="lD7PswTccVmX" colab={"base_uri": "https://localhost:8080/", "height": 291} outputId="db6f4c08-4a37-4708-b1bb-ce9de436aec2"
# Word frequency is linear on log-log scale
plt.figure()
plt.loglog(wordcounts, label='word counts');
plt.ylabel('log frequency')
plt.xlabel('log rank')

# Prediction from Zipf's law, using manually chosen parameters.
# We omit the first 'skip' words, which don't fit the prediction well.
skip = 10.0
x = np.arange(skip, len(wordcounts)) 
N = np.sum(wordcounts)
kappa = 0.1
a = -1
y = kappa*np.power(x, a) * N # predicted frequency for word with rank x
plt.loglog(x, y, label='linear prediction')

plt.legend()
plt.show()


# + id="zUG9EyaFcVmZ" colab={"base_uri": "https://localhost:8080/", "height": 274} outputId="79667748-3c41-4ff3-8adb-596141a80e1f"
# The number of unique n-grams is smaller for larger n.
# But n-gram statistics also exhibit a power law.

plt.figure()
plt.loglog(wordcounts, label='word counts');
plt.loglog(bigramcounts, label='bigram counts');
plt.loglog(triplecounts, label='triple counts');
plt.legend();
plt.show()


# + id="8VGwhh-fcVmb"


# + [markdown] id="SJAoTJoScVmE"
# # Multivariate Gaussian (normal)  <a class="anchor" id="scipy-multigauss"></a>

# + id="Qv170F41cVmF" colab={"base_uri": "https://localhost:8080/"} outputId="854da621-4e9f-4ec1-a7f1-6ea992af143a"
from scipy.stats import multivariate_normal as mvn

D = 5
np.random.seed(42)
mu = np.random.randn(D)
A = np.random.randn(D,D)
Sigma = np.dot(A, A.T)

dist = mvn(mu, Sigma)
X = dist.rvs(size=10)
print(X.shape) 

# + id="ZSljABsCcVmH" colab={"base_uri": "https://localhost:8080/", "height": 447} outputId="20a60d57-d4e5-4349-c985-16bd566708a8"
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

names = ["Full", "Diag", "Spherical"]

mu = [0, 0]
Covs = {'Full': [[2, 1.8], [1.8, 2]],
        'Diag': [[1, 0], [0, 3]],
        'Spherical': [[1, 0], [0, 1]]}

N = 100
points = np.linspace(-5, 5, N)
X, Y = np.meshgrid(points, points)
xs = X.reshape(-1)
ys = Y.reshape(-1)
grid = np.vstack([xs, ys]).T # N^2 * 2

fig = plt.figure(figsize=(10,7))
fig.subplots_adjust(hspace=0.5, wspace=0.1)
fig_counter = 1
for i in range(len(Covs)):
    name = names[i]
    Sigma = Covs[name]
    ps = mvn(mu, Sigma).pdf(grid)
    P = ps.reshape((N,N))

    ax = fig.add_subplot(3, 2, fig_counter)
    ax.contour(X, Y, P)
    ax.axis('equal') # make circles look circular
    ax.set_title(name)
    fig_counter = fig_counter + 1
    
    ax = fig.add_subplot(3, 2, fig_counter, projection='3d')
    ax.plot_surface(X, Y, P, rstride=2, cstride=2)
    ax.set_title(name)
    fig_counter = fig_counter + 1
plt.show()

# + [markdown] id="hTs_tryVzfbg"
# Illustrate correlation coefficient.
#
# Code is from [Bayesian Analysis with Python, ch. 3](https://github.com/aloctavodia/BAP/blob/master/code/Chp3/03_Modeling%20with%20Linear%20Regressions.ipynb)

# + id="io2fONISye1z" colab={"base_uri": "https://localhost:8080/", "height": 503} outputId="64f12636-40c7-4b19-859e-432a3e367283"

sigma_x1 = 1
sigmas_x2 = [1, 2]
rhos = [-0.90, -0.5, 0, 0.5, 0.90]

k, l = np.mgrid[-5:5:.1, -5:5:.1]
pos = np.empty(k.shape + (2,))
pos[:, :, 0] = k
pos[:, :, 1] = l

f, ax = plt.subplots(len(sigmas_x2), len(rhos),
                     sharex=True, sharey=True, figsize=(12, 6),
                     constrained_layout=True)
for i in range(2):
    for j in range(5):
        sigma_x2 = sigmas_x2[i]
        rho = rhos[j]
        cov = [[sigma_x1**2, sigma_x1*sigma_x2*rho],
               [sigma_x1*sigma_x2*rho, sigma_x2**2]]
        rv = stats.multivariate_normal([0, 0], cov)
        ax[i, j].contour(k, l, rv.pdf(pos))
        ax[i, j].set_xlim(-8, 8)
        ax[i, j].set_ylim(-8, 8)
        ax[i, j].set_yticks([-5, 0, 5])
        ax[i, j].plot(0, 0,
                      label=f'$\\sigma_{{x2}}$ = {sigma_x2:3.2f}\n$\\rho$ = {rho:3.2f}', alpha=0)
        ax[i, j].legend()
f.text(0.5, -0.05, 'x_1', ha='center', fontsize=18)
f.text(-0.05, 0.5, 'x_2', va='center', fontsize=18, rotation=0)
