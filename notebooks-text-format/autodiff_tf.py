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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/autodiff_tf.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="xX4GSX3Fwt1S"
# # Automatic differentiation in tensorflow 2
#
# We use binary logistic regression as a running example.
#

# + id="s7H7qrB8xT8J"
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

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows



# + id="4TdbcDI_XXhS"

try:
    # # %tensorflow_version only exists in Colab.
    # %tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

print("tf version {}".format(tf.__version__))

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. DNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# + colab={"base_uri": "https://localhost:8080/"} id="az0915TG19Sl" outputId="520c6034-6f11-4bc9-cd5a-3f3a024e8c25"
## Compute gradient of loss "by hand" using numpy

from scipy.special import logsumexp

def BCE_with_logits(logits, targets):
    N = logits.shape[0]
    logits = logits.reshape(N,1)
    logits_plus = np.hstack([np.zeros((N,1)), logits]) # e^0=1
    logits_minus = np.hstack([np.zeros((N,1)), -logits])
    logp1 = -logsumexp(logits_minus, axis=1)
    logp0 = -logsumexp(logits_plus, axis=1)
    logprobs = logp1 * targets + logp0 * (1-targets)
    return -np.sum(logprobs)/N

def sigmoid(x): return 0.5 * (np.tanh(x / 2.) + 1)

def predict_logit(weights, inputs):
    return np.dot(inputs, weights) # Already vectorized

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
    g = np.sum(np.dot(np.diag(mu - y), X), axis=0)/N
    return g

np.random.seed(0)
N = 100
D = 5
X = np.random.randn(N, D)
w = 10*np.random.randn(D)
mu = predict_prob(w, X)
y = np.random.binomial(n=1, p=mu, size=N)

X_test = X
y_test = y

y_pred = predict_prob(w, X_test)
loss = NLL(w, (X_test, y_test))
grad_np = NLL_grad(w, (X_test, y_test))
print("params {}".format(w))
#print("pred {}".format(y_pred))
print("loss {}".format(loss))
print("grad {}".format(grad_np))

# + colab={"base_uri": "https://localhost:8080/"} id="Ekyo0EEO2KL7" outputId="1ac5fdb2-1307-4650-ed36-a9a8744f2878"

w_tf = tf.Variable(np.reshape(w, (D,1)))  
x_test_tf = tf.convert_to_tensor(X_test, dtype=np.float64) 
y_test_tf = tf.convert_to_tensor(np.reshape(y_test, (-1,1)), dtype=np.float64)
with tf.GradientTape() as tape:
    logits = tf.linalg.matmul(x_test_tf, w_tf)
    y_pred = tf.math.sigmoid(logits)
    loss_batch = tf.nn.sigmoid_cross_entropy_with_logits(y_test_tf, logits)
    loss_tf = tf.reduce_mean(loss_batch, axis=0)
grad_tf = tape.gradient(loss_tf, [w_tf])
grad_tf = grad_tf[0][:,0].numpy()
assert np.allclose(grad_np, grad_tf)

print("params {}".format(w_tf))
#print("pred {}".format(y_pred))
print("loss {}".format(loss_tf))
print("grad {}".format(grad_tf))
