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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/autodiff_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="E4bE-S8yDALH"
# # Automatic differentiation using JAX 
#
# In this section, we illustrate automatic differentation using JAX.
# For details, see see  [this video](https://www.youtube.com/watch?v=wG_nF1awSSY&t=697s)  or [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).
#
#
#
#
#

# + id="eCI0G3tfDFSs"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial
import os
import time
import numpy as np
np.set_printoptions(precision=3)
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from typing import Tuple, NamedTuple

from IPython import display
# %matplotlib inline

import sklearn


# + id="Z9kAsUWYDIOk" colab={"base_uri": "https://localhost:8080/"} outputId="4568e80c-e8a5-4a81-ecb0-20a2d994b0b1"

# Load JAX
import jax
import jax.numpy as jnp

from jax import random, vmap, jit, grad, value_and_grad, hessian, jacfwd, jacrev
print("jax version {}".format(jax.__version__))
# Check the jax backend
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
key = random.PRNGKey(0)

# + [markdown] id="QuMHSd3wr1xH"
# ## Derivatives
#
# We can compute $(\nabla f)(x)$ using `grad(f)(x)`. For example, consider
#
#
# $f(x) = x^3 + 2x^2 - 3x + 1$
#
# $f'(x) = 3x^2 + 4x -3$
#
# $f''(x) = 6x + 4$
#
# $f'''(x) = 6$
#
# $f^{iv}(x) = 0$
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="jYTM5MPmr3C0" outputId="4801ef39-5c62-4fd1-c021-dc2d23e03035"
f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)

print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))


# + [markdown] id="kUj-VuSzmFaV"
# ## Partial derivatives
#
#
# $$
# \begin{align}
# f(x,y) &= x^2 + y \\
# \frac{\partial f}{\partial x} &= 2x \\
# \frac{\partial f}{\partial y} &= 1 
# \end{align}
# $$
#

# + colab={"base_uri": "https://localhost:8080/"} id="c0hW7fqfmR1c" outputId="3b5bbc43-a7e3-4692-c4e8-137faa0c68eb"
def f(x,y):
  return x**2 + y

# Partial derviatives
x = 2.0; y= 3.0;
v, gx = value_and_grad(f, argnums=0)(x,y)
print(v)
print(gx)

gy = grad(f, argnums=1)(x,y)
print(gy)


# + [markdown] id="VTIybB8b4ar0"
# ## Gradients 

# + [markdown] id="Xb0gZ_1HBEyC"
# Linear function: multi-input, scalar output.
#
# $$
# \begin{align}
# f(x; a) &= a^T x\\
# \nabla_x f(x;a) &= a
# \end{align}
# $$

# + colab={"base_uri": "https://localhost:8080/"} id="UmYqFEs04vkV" outputId="311e8ac4-3ed4-4ad6-f545-6307390d4745"


def fun1d(x):
    return jnp.dot(a, x)[0]

Din = 3; Dout = 1;
a = np.random.normal(size=(Dout, Din))
x = np.random.normal(size=(Din,))

g = grad(fun1d)(x)
assert np.allclose(g, a)


# It is often useful to get the function value and gradient at the same time
val_grad_fn = jax.value_and_grad(fun1d)
v, g = val_grad_fn(x)
print(v)
print(g)
assert np.allclose(v, fun1d(x))
assert np.allclose(a, g)


# + [markdown] id="WbgiqkF6BL1E"
# Linear function: multi-input, multi-output.
#
# $$
# \begin{align}
# f(x;A) &= A x \\
# \frac{\partial f(x;A)}{\partial x} &= A
# \end{align}
# $$

# + id="s6hkEYxV5EIx"
# We construct a multi-output linear function.
# We check forward and reverse mode give same Jacobians.


def fun(x):
    return jnp.dot(A, x)

Din = 3; Dout = 4;
A = np.random.normal(size=(Dout, Din))
x = np.random.normal(size=(Din,))
Jf = jacfwd(fun)(x)
Jr = jacrev(fun)(x)
assert np.allclose(Jf, Jr)
assert np.allclose(Jf, A)

# + [markdown] id="CN5d-D7XBU9Y"
# Quadratic form.
#
# $$
# \begin{align}
# f(x;A) &= x^T A x \\
# \nabla_x f(x;A) &= (A+A^T) x
# \end{align}
# $$

# + id="9URZeX8PBbhl"

D = 4
A = np.random.normal(size=(D,D))
x = np.random.normal(size=(D,))
quadfun = lambda x: jnp.dot(x, jnp.dot(A, x))

g = grad(quadfun)(x)
assert np.allclose(g, jnp.dot(A+A.T, x))



# + [markdown] id="U9ZOhDeqCXu3"
# Chain rule applied to sigmoid function.
#
# $$
# \begin{align}
# \mu(x;w) &=\sigma(w^T x) \\
# \nabla_w \mu(x;w) &= \sigma'(w^T x) x \\
# \sigma'(a) &= \sigma(a) * (1-\sigma(a)) 
# \end{align}
# $$

# + colab={"base_uri": "https://localhost:8080/"} id="6q5VfLXLB7rv" outputId="0773a46f-784f-4dbe-a6b7-29dd5cd26654"


D = 4
w = np.random.normal(size=(D,))
x = np.random.normal(size=(D,))
y = 0 

def sigmoid(x): return 0.5 * (jnp.tanh(x / 2.) + 1)
def mu(w): return sigmoid(jnp.dot(w,x))
def deriv_mu(w): return mu(w) * (1-mu(w)) * x
deriv_mu_jax =  grad(mu)

print(deriv_mu(w))
print(deriv_mu_jax(w))

assert np.allclose(deriv_mu(w), deriv_mu_jax(w), atol=1e-3)



# + [markdown] id="nglie5m7q607"
# ## Auxiliary return values
#
# A function can return its value and other auxiliary results; the latter are not differentiated. 

# + colab={"base_uri": "https://localhost:8080/"} id="QHz6zrC9qVjT" outputId="76724fd1-4e1a-4737-b252-963700fcce91"
def f(x,y):
  return x**2+y, 42

(v,aux), g = value_and_grad(f, has_aux=True)(x,y)
print(v)
print(aux)
print(g)

# + [markdown] id="bBMcsg4uoKua"
# ## Jacobians
#
#
# Example: Linear function: multi-input, multi-output.
#
# $$
# \begin{align}
# f(x;A) &= A x \\
# \frac{\partial f(x;A)}{\partial x} &= A
# \end{align}
# $$
#

# + id="iPilm5H3oWcy"
# We construct a multi-output linear function.
# We check forward and reverse mode give same Jacobians.


def fun(x):
    return jnp.dot(A, x)

Din = 3; Dout = 4;
A = np.random.normal(size=(Dout, Din))
x = np.random.normal(size=(Din,))
Jf = jacfwd(fun)(x)
Jr = jacrev(fun)(x)
assert np.allclose(Jf, Jr)

# + [markdown] id="mg9ValMRm_Md"
# ## Hessians
#
# Quadratic form.
#
# $$
# \begin{align}
# f(x;A) &= x^T A x \\
# \nabla_x^2 f(x;A) &= A + A^T
# \end{align}
# $$

# + id="leW9lqvinDsM"

D = 4
A = np.random.normal(size=(D,D))
x = np.random.normal(size=(D,))

quadfun = lambda x: jnp.dot(x, jnp.dot(A, x))


H1 = hessian(quadfun)(x)
assert np.allclose(H1, A+A.T)

def my_hessian(fun):
  return jacfwd(jacrev(fun))

H2 = my_hessian(quadfun)(x)
assert np.allclose(H1, H2)


# + [markdown] id="MeoGcnV54YY9"
# ## Example: Binary logistic regression

# + id="Isql2l4MGfIt" colab={"base_uri": "https://localhost:8080/"} outputId="2465d977-66dc-4ea0-e5bd-5a05d4ad92ec"

def sigmoid(x): return 0.5 * (jnp.tanh(x / 2.) + 1)

def predict_single(w, x):
    return sigmoid(jnp.dot(w, x)) # <(D) , (D)> = (1) # inner product
  
def predict_batch(w, X):
    return sigmoid(jnp.dot(X, w)) # (N,D) * (D,1) = (N,1) # matrix-vector multiply

# negative log likelihood
def loss(weights, inputs, targets):
    preds = predict_batch(weights, inputs)
    logprobs = jnp.log(preds) * targets + jnp.log(1 - preds) * (1 - targets)
    return -jnp.sum(logprobs)


D = 2
N = 3
w = jax.random.normal(key, shape=(D,))
X = jax.random.normal(key, shape=(N,D))
y = jax.random.choice(key, 2, shape=(N,)) # uniform binary labels
#logits = jnp.dot(X, w)
#y = jax.random.categorical(key, logits)

print(loss(w, X, y))

# Gradient function
grad_fun = grad(loss)

# Gradient of each example in the batch - 2 different ways
grad_fun_w = partial(grad_fun, w)
grads = vmap(grad_fun_w)(X,y)
print(grads)
assert grads.shape == (N,D)

grads2 = vmap(grad_fun, in_axes=(None, 0, 0))(w, X, y) 
assert np.allclose(grads, grads2)

# Gradient for entire batch
grad_sum = jnp.sum(grads, axis=0)
assert grad_sum.shape == (D,)
print(grad_sum)


# + colab={"base_uri": "https://localhost:8080/"} id="G3BaHdT4Gj6W" outputId="12dc846e-dd7e-4907-ea76-73147ea6f77f"
# Textbook implementation of gradient
def NLL_grad(weights, batch):
    X, y = batch
    N = X.shape[0]
    mu = predict_batch(weights, X)
    g = jnp.sum(jnp.dot(jnp.diag(mu - y), X), axis=0)
    return g

grad_sum_batch = NLL_grad(w, (X,y))
print(grad_sum_batch)
assert np.allclose(grad_sum, grad_sum_batch)

# + colab={"base_uri": "https://localhost:8080/"} id="S_4lRrHgpLbG" outputId="d47db27a-b171-4933-de08-58bc5bbe0c2f"
# We can also compute Hessians, as we illustrate below.

hessian_fun = hessian(loss)

# Hessian on one example
H0 = hessian_fun(w, X[0,:], y[0])
print('Hessian(example 0)\n{}'.format(H0))

# Hessian for batch
Hbatch = vmap(hessian_fun, in_axes=(None, 0, 0))(w, X, y) 
print('Hbatch shape {}'.format(Hbatch.shape))

Hbatch_sum = jnp.sum(Hbatch, axis=0)
print('Hbatch sum\n {}'.format(Hbatch_sum))


# + id="QcJvgukUpWWE"
# Textbook implementation of Hessian

def NLL_hessian(weights, batch):
  X, y = batch
  mu = predict_batch(weights, X)
  S = jnp.diag(mu * (1-mu))
  H = jnp.dot(jnp.dot(X.T, S), X)
  return H

H2 = NLL_hessian(w, (X,y) )

assert np.allclose(Hbatch_sum, H2, atol=1e-2)

# + [markdown] id="EOzQJSX3JOF9"
# ## Vector Jacobian Products (VJP) and Jacobian Vector Products (JVP)
#
# Consider a bilinear mapping $f(x,W) = x W$.
# For fixed parameters, we have
# $f1(x) = W x$, so $J(x) = W$, and $u^T J(x) = J(x)^T u = W^T u$.
#

# + colab={"base_uri": "https://localhost:8080/"} id="4Lpmntn2JREG" outputId="39778872-aca9-488c-f9dc-40d30a821488"
n = 3; m = 2;
W = jax.random.normal(key, shape=(m,n))
x = jax.random.normal(key, shape=(n,))
u = jax.random.normal(key, shape=(m,))

def f1(x): return jnp.dot(W,x)

J1 = jacfwd(f1)(x)
print(J1.shape)

assert np.allclose(J1, W)
tmp1 = jnp.dot(u.T, J1)
print(tmp1)

(val, jvp_fun) = jax.vjp(f1, x)

tmp2 = jvp_fun(u)

assert np.allclose(tmp1, tmp2)

tmp3 = np.dot(W.T, u)
assert np.allclose(tmp1, tmp3)




# + [markdown] id="kYSC6DMOO3IS"
# For fixed inputs, we have
# $f2(W) = W x$, so $J(W) = \text{something complex}$,
# but $u^T J(W) = J(W)^T u = u x^T$.

# + colab={"base_uri": "https://localhost:8080/"} id="R8l3StJdO1r1" outputId="5a3f1e5b-b903-4db8-bb8a-a53c29bcad19"

def f2(W): return jnp.dot(W,x)

J2 = jacfwd(f2)(W)
print(J2.shape)

tmp1 = jnp.dot(u.T, J2)
print(tmp1)
print(tmp1.shape)

(val, jvp_fun) = jax.vjp(f2, W)
tmp2 = jvp_fun(u)
assert np.allclose(tmp1, tmp2)

tmp3 = np.outer(u, x)
assert np.allclose(tmp1, tmp3)


# + [markdown] id="ZF_OjIRrisDB"
# ## Stop-gradient
#
# Sometimes we want to take the gradient of a complex expression wrt some parameters $\theta$, but treating $\theta$ as a constant for some parts of the expression. For example, consider the TD(0) update in reinforcement learning, which as the following form:
#
#
# $\Delta \theta = (r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})) \nabla v_{\theta}(s_{t-1})$
#
# where $s$ is the state, $r$ is the reward, and $v$ is the value function.
# This update is not the gradient of any loss function.
# However it can be **written** as the gradient of the pseudo loss function
#
# $L(\theta) = [r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})]^2$
#
# since
#
# $\nabla_{\theta} L(\theta) = 2 [r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})] \nabla v_{\theta}(s_{t-1})$
#
# if the dependency of the target $r_t + v_{\theta}(s_t)$ on the parameter $\theta$ is ignored. We can implement this in JAX using `stop_gradient`, as we show below.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="C9cprxb9jbsK" outputId="b345e287-b8fa-417f-d20e-0f1951e1f64e"
def td_loss(theta, s_prev, r_t, s_t):
  v_prev = value_fn(theta, s_prev)
  target = r_t + value_fn(theta, s_t)
  return 0.5*(jax.lax.stop_gradient(target) - v_prev) ** 2

td_update = jax.grad(td_loss)

# An example transition.
s_prev = jnp.array([1., 2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])

# Value function and initial parameters
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])

print(td_update(theta, s_prev, r_t, s_t))



# + [markdown] id="AKTg9m9yz7Ib"
# ## Straight through estimator
#
# The straight-through estimator is a trick for defining a 'gradient' of a function that is otherwise non-differentiable. Given a non-differentiable function $f : \mathbb{R}^n \to \mathbb{R}^n$ that is used as part of a larger function that we wish to find a gradient of, we simply pretend during the backward pass that $f$ is the identity function, so gradients pass through $f$ ignoring the $f'$ term. This can be implemented neatly using `jax.lax.stop_gradient`.
#
# Here is an example of a non-differentiable function that converts a soft probability distribution to a one-hot vector (discretization).
#

# + colab={"base_uri": "https://localhost:8080/"} id="HIFVQKrwqAG4" outputId="67ada739-47e3-4ef9-c11e-76a64862ea80"
def onehot(labels, num_classes):
  y = (labels[..., None] == jnp.arange(num_classes)[None])
  return y.astype(jnp.float32)

def quantize(y_soft): 
  y_hard = onehot(jnp.argmax(y_soft), 3)[0]
  return y_hard

y_soft = np.array([0.1, 0.2, 0.7])
print(quantize(y_soft))




# + [markdown] id="LSuQMr61sg16"
# Now suppose we define some linear function of the quantized variable of the form $f(y) = w^T q(y)$. If $w=[1,2,3]$ and $q(y)=[0,0,1]$, we get $f(y) = 3$. But the gradient is 0 because $q$ is not differentiable.
#

# + colab={"base_uri": "https://localhost:8080/"} id="tDEMPSdJsTpl" outputId="0f329a87-2271-47d2-f23b-be47ed79a618"
def f(y):
  w = jnp.array([1,2,3])
  yq = quantize(y)
  return jnp.dot(w, yq)

print(f(y_soft))
print(grad(f)(y_soft))



# + [markdown] id="oTcVC08Rs4DM"
# To use the straight-through estimator, we replace $q(y)$ with 
# $$y + SG(q(y)-y)$$, where SG is stop gradient. In the forwards pass, we have $y+q(y)-y=q(y)$. In the backwards pass, the gradient of SG is 0, so we effectively replace $q(y)$ with $y$. So in the backwarsd pass we have
# $$
# \begin{align}
# f(y) &= w^T q(y) \approx w^T  y \\
# \nabla_y f(y) &\approx w
# \end{align}
# $$

# + colab={"base_uri": "https://localhost:8080/"} id="5m_k7Ju3sUVq" outputId="cfcd7769-de40-4fc7-d29c-19d6a7d04db0"


def f_ste(y):
  w = jnp.array([1,2,3])
  yq = quantize(y)
  yy = y + jax.lax.stop_gradient(yq - y) # gives yq on fwd, and y on backward
  return jnp.dot(w, yy)

print(f_ste(y_soft))
print(grad(f_ste)(y_soft))


# + [markdown] id="YwMIYqiOp0U0"
# ## Per-example gradients
#
# In some applications, we want to compute the gradient for every example in a batch, not just the sum of gradients over the batch. This is hard in other frameworks like TF and PyTorch but easy in JAX, as we show below.

# + colab={"base_uri": "https://localhost:8080/"} id="YKhNPiq4qGhl" outputId="0102b6ff-1fc1-4870-9f58-1b4e7db92ba9"
def loss(w, x):
  return jnp.dot(w,x)

w = jnp.ones((3,))
x0 = jnp.array([1.0, 2.0, 3.0])
x1 = 2*x0
X = jnp.stack([x0, x1])
print(X.shape)

perex_grads = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0)))
print(perex_grads(w, X))


# + [markdown] id="hxPeXpUaq-_p"
# To explain the above code in more depth, note that the vmap converts the function loss to take  a batch of inputs for each of its arguments, and returns a batch of outputs. To make it work with a single weight vector, we specify in_axes=(None,0), meaning the first argument (w) is not replicated, and the second argument (x) is replicated along dimension 0. 

# + colab={"base_uri": "https://localhost:8080/"} id="wYB1L8JKrVKo" outputId="b78026f2-7ff2-4228-be72-cd0c1b281b2b"
gradfn = jax.grad(loss)

W = jnp.stack([w, w])
print(jax.vmap(gradfn)(W, X))

print(jax.vmap(gradfn, in_axes=(None,0))(w, X))


