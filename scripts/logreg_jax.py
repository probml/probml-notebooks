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
# <a href="https://colab.research.google.com/github/Nirzu97/pyprobml/blob/logreg_jax/notebooks/logreg_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="eB0c7K3GpBrg"
# # Logistic regression <a class="anchor" id="logreg"></a>
#
# In this notebook, we illustrate how to perform logistic regression on some small datasets. We will compare binary logistic regression as implemented by sklearn with our own implementation, for which we use a batch optimizer from scipy. We code the gradients by hand. We also show how to use the JAX autodiff package (see [JAX AD colab](https://github.com/probml/pyprobml/tree/master/book1/supplements/autodiff_jax.ipynb)).
#

# + id="Ml8l4WVLpWCI"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

import sklearn

import seaborn as sns;
sns.set(style="ticks", color_codes=True)





# + colab={"base_uri": "https://localhost:8080/"} id="cH-3xvdv7dXD" outputId="66dc6c29-9c1b-4419-cfb9-da7b1c0bef1d"
# https://github.com/google/jax
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
from jax.experimental import optimizers
print("jax version {}".format(jax.__version__))

# + id="1UrGDEYb58qF"
# First we create a dataset.

import sklearn.datasets
from sklearn.model_selection import train_test_split

iris = sklearn.datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0'
N, D = X.shape # 150, 4


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

# + colab={"base_uri": "https://localhost:8080/"} id="S8GCOQm16OB3" outputId="69703c4d-84dd-48ff-dc12-1b9b0680ade1"
# Now let's find the MLE using sklearn. We will use this as the "gold standard"

from sklearn.linear_model import LogisticRegression

# We set C to a large number to turn off regularization.
# We don't fit the bias term to simplify the comparison below.
log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=False)
log_reg.fit(X_train, y_train)
w_mle_sklearn = jnp.ravel(log_reg.coef_)
print(w_mle_sklearn)


# + colab={"base_uri": "https://localhost:8080/"} id="-FtE7SSE6V0k" outputId="80b5de9f-f2a4-46bb-d6f4-777563472401"

# First we define the model, and check it gives the same output as sklearn.

def sigmoid(x): return 0.5 * (jnp.tanh(x / 2.) + 1)

def predict_logit(weights, inputs):
    return jnp.dot(inputs, weights) # Already vectorized

def predict_prob(weights, inputs):
    return sigmoid(predict_logit(weights, inputs))

ptest_sklearn = log_reg.predict_proba(X_test)[:,1]
print(jnp.round(ptest_sklearn, 3))

ptest_us = predict_prob(w_mle_sklearn, X_test)
print(jnp.round(ptest_us, 3))

assert jnp.allclose(ptest_sklearn, ptest_us, atol=1e-2)

# + colab={"base_uri": "https://localhost:8080/"} id="X27Iy_4D62Dl" outputId="759225be-b4c5-429a-847a-24d067e51d6a"
# Next we define the objective and check it gives the same output as sklearn.

from sklearn.metrics import log_loss
from jax.scipy.special import logsumexp
#from scipy.misc import logsumexp

def NLL_unstable(weights, batch):
    inputs, targets = batch
    p1 = predict_prob(weights, inputs)
    logprobs = jnp.log(p1) * targets + jnp.log(1 - p1) * (1 - targets)
    N = inputs.shape[0]
    return -jnp.sum(logprobs)/N


def NLL(weights, batch):
    # Use log-sum-exp trick
    inputs, targets = batch
    # p1 = 1/(1+exp(-logit)), p0 = 1/(1+exp(+logit))
    logits = predict_logit(weights, inputs).reshape((-1,1))
    N = logits.shape[0]
    logits_plus = jnp.hstack([jnp.zeros((N,1)), logits]) # e^0=1
    logits_minus = jnp.hstack([jnp.zeros((N,1)), -logits])
    logp1 = -logsumexp(logits_minus, axis=1)
    logp0 = -logsumexp(logits_plus, axis=1)
    logprobs = logp1 * targets + logp0 * (1-targets)
    return -jnp.sum(logprobs)/N

# We can use a small amount of L2 regularization, for numerical stability
def PNLL(weights, batch, l2_penalty=1e-5):
    nll = NLL(weights, batch)
    l2_norm = jnp.sum(jnp.power(weights, 2)) # squared L2 norm
    return nll + l2_penalty*l2_norm

# We evaluate the training loss at the MLE, where the parameter values are "extreme".
nll_train = log_loss(y_train, predict_prob(w_mle_sklearn, X_train))
nll_train2 = NLL(w_mle_sklearn, (X_train, y_train))
nll_train3 = NLL_unstable(w_mle_sklearn, (X_train, y_train))
print(nll_train)
print(nll_train2)
print(nll_train3)

# + id="O8PxEPaC7pMK"
# Next we check the gradients compared to the manual formulas.
# For simplicity, we initially just do this for a single random example.

np.random.seed(42)
D = 5
w = np.random.randn(D)
x = np.random.randn(D)
y = 0 

#d/da sigmoid(a) = s(a) * (1-s(a))
deriv_sigmoid = lambda a: sigmoid(a) * (1-sigmoid(a))
deriv_sigmoid_jax = grad(sigmoid)
a = 1.5 # a random logit
assert jnp.isclose(deriv_sigmoid(a), deriv_sigmoid_jax(a))

# mu(w)=sigmoid(w'x), d/dw mu(w) = mu * (1-mu) .* x
def mu(w): return sigmoid(jnp.dot(w,x))
def deriv_mu(w): return mu(w) * (1-mu(w)) * x
deriv_mu_jax =  grad(mu)
assert jnp.allclose(deriv_mu(w), deriv_mu_jax(w))

# NLL(w) = -[y*log(mu) + (1-y)*log(1-mu)]
# d/dw NLL(w) = (mu-y)*x
def nll(w): return -(y*jnp.log(mu(w)) + (1-y)*jnp.log(1-mu(w)))
def deriv_nll(w): return (mu(w)-y)*x
deriv_nll_jax = grad(nll)
assert jnp.allclose(deriv_nll(w), deriv_nll_jax(w))

# + colab={"base_uri": "https://localhost:8080/"} id="TVT2tlmA7z72" outputId="e9fa28ab-ca6c-423f-ebb3-0c49a2a12a5d"
# Now let's check the gradients on the batch version of our data.

N = X_train.shape[0]
mu = predict_prob(w_mle_sklearn, X_train)

g1 = grad(NLL)(w_mle_sklearn, (X_train, y_train))
g2 = jnp.sum(jnp.dot(jnp.diag(mu - y_train), X_train), axis=0)/N
print(g1)
print(g2)
assert jnp.allclose(g1, g2, atol=1e-2)

H1 = hessian(NLL)(w_mle_sklearn, (X_train, y_train))
S = jnp.diag(mu * (1-mu))
H2 = jnp.dot(jnp.dot(X_train.T, S), X_train)/N
print(H1)
print(H2)
assert jnp.allclose(H1, H2, atol=1e-2)

# + colab={"base_uri": "https://localhost:8080/"} id="-_LREVYj79IW" outputId="3125bf39-8feb-48a8-e7a9-5ed9c8c626d5"
# Finally, use BFGS batch optimizer to compute MLE, and compare to sklearn

import scipy.optimize

def training_loss(w):
    return NLL(w, (X_train, y_train))

def training_grad(w):
    return grad(training_loss)(w)

np.random.seed(43)
N, D = X_train.shape 
w_init = np.random.randn(D)
w_mle_scipy = scipy.optimize.minimize(training_loss, w_init, jac=training_grad, method='BFGS').x


print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from scipy-bfgs {}".format(w_mle_scipy))
assert jnp.allclose(w_mle_sklearn, w_mle_scipy, atol=1e-1)

prob_scipy = predict_prob(w_mle_scipy, X_test)
prob_sklearn = predict_prob(w_mle_sklearn, X_test)
print(jnp.round(prob_scipy, 3))
print(jnp.round(prob_sklearn, 3))

assert jnp.allclose(prob_scipy, prob_sklearn, atol=1e-2)

# + id="jGXLRuHI0Xrw"

