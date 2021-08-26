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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/opt_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="b520E1nCIBHc"
# # Optimization (using JAX)
#
# In this notebook, we explore various  algorithms
# for solving optimization problems of the form
# $$
# x* = \arg \min_{x \in X} f(x)
# $$
# We focus on the case where $f: R^D \rightarrow R$ is a differentiable function.
# We make use of the [JAX](https://github.com/google/jax) library for automatic differentiation.
#
# Some other possibly useful resources:
#
#
# 1.   [Animations of various SGD algorithms in 2d (using PyTorch)](https://nbviewer.jupyter.org/github/entiretydotai/Meetup-Content/blob/master/Neural_Network/7_Optimizers.ipynb)
#
# 2.   [Tutorial on constrained optimization using JAX](https://medium.com/swlh/solving-optimization-problems-with-jax-98376508bd4f)
#
#
#

# + id="UeuOgABaIENZ"
import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import itertools
import time
from functools import partial
import os

import numpy as np
#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# + id="TNQHpyKLIx_P" colab={"base_uri": "https://localhost:8080/"} outputId="1de888d0-6696-44fe-d83a-7d708d979244"

import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp
from jax import grad, hessian, jacfwd, jacrev, jit, vmap
print("jax version {}".format(jax.__version__))



# + [markdown] id="Br921MsmKQkt"
# # Fitting a model using sklearn
#
# Models in the sklearn library support the `fit` method for parameter estimation. Under the hood, this involves an optimization problem.
# In this colab, we lift up this hood and replicate the functionality from first principles.
#
# As a running example, we will use binary logistic regression on the iris dataset.

# + id="c3fX16J4IoL_" colab={"base_uri": "https://localhost:8080/"} outputId="2719213c-71c8-473a-af30-32f9f82dc049"
# Fit the model to a dataset, so we have an "interesting" parameter vector to use.

import sklearn.datasets
from sklearn.model_selection import train_test_split

iris = sklearn.datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0'
N, D = X.shape # 150, 4

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

# We set C to a large number to turn off regularization.
# We don't fit the bias term to simplify the comparison below.
log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=False)
log_reg.fit(X_train, y_train)
w_mle_sklearn = jnp.ravel(log_reg.coef_)
print(w_mle_sklearn)


# + [markdown] id="R_HQrexrySmT"
# # Objectives and their gradients
#
# The key input to an optimization algorithm (aka solver) is the objective function and its gradient. As an example, we use negative log likelihood for a binary logistic regression model as the objective. We compute the gradient by hand, and also use JAX's autodiff feature.
#

# + [markdown] id="-pIgD7iRLUBt"
# ## Manual differentiation <a class="anchor" id="AD"></a>
#
# We compute the gradient of the negative log likelihood for binary logistic regression applied to the Iris dataset. 

# + id="iS5AB9NjLZ_i"

# Binary cross entropy
def BCE_with_logits(logits, targets):
  #BCE = -sum_n log(p1)*yn + log(p0)*y0
  #p1 = 1/(1+exp(-a)
  #log(p1) = log(1) - log(1+exp(-a)) = 0 - logsumexp(0, -a)
  N = logits.shape[0]
  logits = logits.reshape(N,1)
  logits_plus = jnp.hstack([jnp.zeros((N,1)), logits]) # e^0=1
  logits_minus = jnp.hstack([jnp.zeros((N,1)), -logits])
  logp1 = -logsumexp(logits_minus, axis=1)
  logp0 = -logsumexp(logits_plus, axis=1)
  logprobs = logp1 * targets + logp0 * (1-targets)
  return -jnp.sum(logprobs)/N

def sigmoid(x): return 0.5 * (jnp.tanh(x / 2.) + 1)

def predict_logit(weights, inputs):
    return jnp.dot(inputs, weights) 

def predict_prob(weights, inputs):
    return sigmoid(predict_logit(weights, inputs))

def NLL(weights, batch):
    X, y = batch
    logits = predict_logit(weights, X)
    return BCE_with_logits(logits, y)

def NLL_grad(weights, batch):
    X, y = batch
    N = X.shape[0]
    mu = predict_prob(weights, X)
    g = jnp.sum(jnp.dot(jnp.diag(mu - y), X), axis=0)/N
    return g



# + colab={"base_uri": "https://localhost:8080/"} id="Y0nT1ASb86iJ" outputId="1667e295-a445-4f78-f995-09e43c3d1459"
w = w_mle_sklearn
y_pred = predict_prob(w, X_test)
loss = NLL(w, (X_test, y_test))
grad_np = NLL_grad(w, (X_test, y_test))
print("params {}".format(w))
#print("pred {}".format(y_pred))
print("loss {}".format(loss))
print("grad {}".format(grad_np))

# + [markdown] id="OLyk46HbLhgT"
# ## Automatic differentiation in JAX  <a class="anchor" id="AD-jax"></a>
#
# Below we use JAX to compute the gradient of the NLL for binary logistic regression.
#
#

# + id="9GkR1yHNLcjU" colab={"base_uri": "https://localhost:8080/"} outputId="ed039872-a514-496e-9ffe-4e0350bd355e"

grad_jax = grad(NLL)(w, (X_test, y_test))
print("grad {}".format(grad_jax))
assert np.allclose(grad_np, grad_jax)

# + [markdown] id="8BXji_6BL87s"
# # Second-order optimization <a class="anchor" id="second"></a>
#
# The "gold standard" of optimization is second-order methods, that leverage Hessian information. Since the Hessian has O(D^2) parameters, such methods do not scale to high-dimensional problems. However, we can sometimes approximate the Hessian using low-rank or diagonal approximations. Below we illustrate the low-rank BFGS method, and the limited-memory version of BFGS, that uses O(D H) space and O(D^2) time per step, where H is the history length.
#
# In general, second-order methods also require exact (rather than noisy) gradients. In the context of ML, this means they are "full batch" methods, since computing the exact gradient requires evaluating the loss on all the datapoints. However, for small data problems, this is feasible (and advisable).
#
# Below we illustrate how to use LBFGS as in [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
#                     

# + id="kkTaK-WZMAGL"
import scipy.optimize

def training_loss(w):
    return NLL(w, (X_train, y_train))

def training_grad(w):
    return NLL_grad(w, (X_train, y_train))

np.random.seed(42)
w_init = np.random.randn(D)

options={'disp': None,   'maxfun': 1000, 'maxiter': 1000}
method = 'BFGS'
# The gradient function is specified via the Jacobian keyword
w_mle_scipy = scipy.optimize.minimize(training_loss, w_init, jac=training_grad, method=method, options=options).x   



# + id="Sv3sPfeIlfl7" colab={"base_uri": "https://localhost:8080/"} outputId="841c9823-e1d8-4bcd-f687-7780e407dc13"
print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from scipy-bfgs {}".format(w_mle_scipy))
assert np.allclose(w_mle_sklearn, w_mle_scipy, atol=1e-1)

# + id="gUZUCRgHmoyS" colab={"base_uri": "https://localhost:8080/"} outputId="7747abd7-03cc-4272-85f1-e8499c515aef"
p_pred_sklearn = predict_prob(w_mle_sklearn, X_test)
p_pred_scipy = predict_prob(w_mle_scipy, X_test) 
print("predictions from sklearn")
print(p_pred_sklearn)
print("predictions from scipy")
print(p_pred_scipy)
assert np.allclose(p_pred_sklearn, p_pred_scipy, atol=1e-1)


# + id="g5cLYkceMG7A" colab={"base_uri": "https://localhost:8080/"} outputId="4eacd182-8efa-442f-ec85-eb81c6db024e"
# Limited memory version requires that we work with 64bit, since implemented in Fortran.

def training_loss_64bit(w):
    l = NLL(w, (X_train, y_train))
    return np.float64(l)

def training_grad_64bit(w):
    g = NLL_grad(w, (X_train, y_train))
    return np.asarray(g, dtype=np.float64)

np.random.seed(42)
w_init = np.random.randn(D)                 

memory = 10
options={'disp': None, 'maxcor': memory,  'maxfun': 1000, 'maxiter': 1000}
# The code also handles bound constraints, hence the name
method = 'L-BFGS-B'
#w_mle_scipy = scipy.optimize.minimize(training_loss, w_init, jac=training_grad, method=method).x 
w_mle_scipy = scipy.optimize.minimize(training_loss_64bit, w_init, jac=training_grad_64bit, method=method).x 


print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from scipy-lbfgs {}".format(w_mle_scipy))
assert np.allclose(w_mle_sklearn, w_mle_scipy, atol=1e-1)

# + [markdown] id="eiZXds_DMj31"
# # Stochastic gradient descent <a class="anchor" id="SGD"></a>
#
# Full batch optimization is too expensive for solving empirical risk minimization problems on large datasets.
# The standard approach in such settings is to use stochastic gradient desceent (SGD).
# In this section we  illustrate how to implement SGD. We apply it to a simple convex problem, namely MLE for  logistic regression on the small iris dataset, so we can compare to the exact batch methods we illustrated above.
#

# + [markdown] id="n86utFUQee3n"
# ## Minibatches
#
# We use the [tensorflow datasets](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/intro/datasets.ipynb) library to make it easy to create streams of minibatches.

# + colab={"base_uri": "https://localhost:8080/"} id="2fcr5EQg-3ix" outputId="c1dfec80-0166-482f-d93a-dc44f20e857f"
import tensorflow as tf
import tensorflow_datasets as tfds

def make_batch_stream(X_train, y_train, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices({"X": X_train, "y": y_train})
  batches = dataset.batch(batch_size)
  batch_stream = tfds.as_numpy(batches)  # finite iterable of dict of NumPy arrays
  N = X_train.shape[0]
  nbatches = int(np.floor(N/batch_size))
  print('{} examples split into {} batches of size {}'.format(N, nbatches, batch_size))
  return batch_stream

batch_stream = make_batch_stream(X_train, y_train, 20)
for epoch in range(2):
  print('epoch {}'.format(epoch))
  for batch in batch_stream:
    x, y = batch["X"], batch["y"]
    print(x.shape) # batch size * num features = 4


# + [markdown] id="DtOeheP-MnB7"
# ## SGD from scratch
#
# We show a minimal implementation of SGD using vanilla JAX/ numpy. 
#

# + id="wG9tVufuMTui"
def sgd(params, loss_fn, grad_loss_fn, batch_iter, max_epochs, lr):
    print_every = max(1, int(0.1*max_epochs))
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for batch_dict in batch_iter:
            x, y = batch_dict["X"], batch_dict["y"]
            batch = (x, y)
            batch_grad = grad_loss_fn(params, batch)
            params = params - lr*batch_grad
            batch_loss = loss_fn(params, batch) # Average loss within this batch
            epoch_loss += batch_loss
        if epoch % print_every == 0:
            print('Epoch {}, batch Loss {}'.format(epoch, batch_loss))
    return params



# + id="5sV3NbjvM6ai" colab={"base_uri": "https://localhost:8080/"} outputId="8abc017b-94ae-4275-c23c-76723c1eca6b"
np.random.seed(42)
w_init = np.random.randn(D) 

max_epochs = 5
lr = 0.1
batch_size = 10
batch_stream = make_batch_stream(X_train, y_train, batch_size)
w_mle_sgd = sgd(w_init, NLL, NLL_grad, batch_stream, max_epochs, lr)



# + [markdown] id="YZXTyQ91nxOj"
# ## Compare SGD with batch optimization
#
# SGD is not a particularly good optimizer, even on this simple convex problem - it converges to a solution that it is quite different to the global MLE. Of course, this could be due to lack of identiability (since the object is convex, but maybe not strongly convex, unless we add some regularziation). But the predicted probabilities also differ substantially. Clearly we will need 'fancier' SGD methods, even for this simple problem.
#
#

# + id="R2U9F6-jnDNr" colab={"base_uri": "https://localhost:8080/"} outputId="25bfb524-0e0c-4e4d-b4e6-e9dc07903244"
print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from sgd {}".format(w_mle_sgd))
#assert np.allclose(w_mle_sklearn, w_mle_sgd, atol=1e-1)

# + id="LhgeSFMEj55_" colab={"base_uri": "https://localhost:8080/", "height": 402} outputId="992fa579-f0ec-4e92-a16d-e20288e57aef"

p_pred_sklearn = predict_prob(w_mle_sklearn, X_test)
p_pred_sgd = predict_prob(w_mle_sgd, X_test) 
print("predictions from sklearn")
print(p_pred_sklearn)
print("predictions from sgd")
print(p_pred_sgd)
assert np.allclose(p_pred_sklearn, p_pred_sgd, atol=1e-1)

# + [markdown] id="NtFGH_OeZUVj"
# ## Using jax.experimental.optimizers
#
# JAX has a small optimization library focused on stochastic first-order optimizers. Every optimizer is modeled as an (`init_fun`, `update_fun`, `get_params`) triple of functions. The `init_fun` is used to initialize the optimizer state, which could include things like momentum variables, and the `update_fun` accepts a gradient and an optimizer state to produce a new optimizer state. The `get_params` function extracts the current iterate (i.e. the current parameters) from the optimizer state. The parameters being optimized can be ndarrays or arbitrarily-nested data structures, so you can store your parameters however you’d like.
#
# Below we show how to reproduce our numpy code using this library.

# + id="PtBbjnzRM79T"
# Version that uses JAX optimization library

from jax.experimental import optimizers

#@jit
def sgd_jax(params, loss_fn, batch_stream, max_epochs, opt_init, opt_update, get_params):
    loss_history = []
    opt_state = opt_init(params)
    
    #@jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        g = grad(loss_fn)(params, batch)
        return opt_update(i, g, opt_state) 
    
    print_every = max(1, int(0.1*max_epochs))
    total_steps = 0
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for batch_dict in batch_stream:
            X, y = batch_dict["X"], batch_dict["y"]
            batch = (X, y)
            total_steps += 1
            opt_state = update(total_steps, opt_state, batch)
        params = get_params(opt_state)
        train_loss = np.float(loss_fn(params, batch))
        loss_history.append(train_loss)
        if epoch % print_every == 0:
            print('Epoch {}, batch loss {}'.format(epoch, train_loss))
    return params, loss_history


# + id="NCOrHGTvbbfC" colab={"base_uri": "https://localhost:8080/"} outputId="83562836-cc46-4a09-ceb1-3d5307f30c3f"
# JAX with constant LR should match our minimal version of SGD

schedule = optimizers.constant(step_size=lr)
opt_init, opt_update, get_params = optimizers.sgd(step_size=schedule)

w_mle_sgd2, history = sgd_jax(w_init, NLL, batch_stream, max_epochs, 
                              opt_init, opt_update, get_params)
print(w_mle_sgd2)
print(history)
