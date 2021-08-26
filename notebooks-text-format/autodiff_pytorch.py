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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/autodiff_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="b520E1nCIBHc"
#
# # Automatic differentation using PyTorch
#
# We show how to do Automatic differentation using PyTorch.
#
#

# + id="UeuOgABaIENZ"
import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import itertools
import time
from functools import partial
import os

import numpy as np
from scipy.special import logsumexp
np.set_printoptions(precision=3)


# + id="GPozRwDAKFb8" colab={"base_uri": "https://localhost:8080/"} outputId="b716e003-11c4-4c9d-9984-c3b325040b10"


import torch
import torch.nn as nn
import torchvision
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  print("current device {}".format(torch.cuda.current_device()))
else:
  print("Torch cannot find GPU")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# + [markdown] id="wCdU93g4V6_O"
# # Example: binary logistic regression
#
# Objective = NLL for binary logistic regression
#

# + colab={"base_uri": "https://localhost:8080/"} id="aSYkjaAO6n3A" outputId="5a6caeb3-42e2-42f3-fd66-d18ca656ae1c"
# Fit the model usign sklearn

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
w_mle_sklearn = np.ravel(log_reg.coef_)
print(w_mle_sklearn)

# + [markdown] id="0p5y7b8NbyZp"
# ## Computing gradients by hand
#

# + id="iS5AB9NjLZ_i"


# Binary cross entropy
def BCE_with_logits(logits, targets):
    N = logits.shape[0]
    logits = logits.reshape(N,1)
    logits_plus = np.hstack([np.zeros((N,1)), logits]) # e^0=1
    logits_minus = np.hstack([np.zeros((N,1)), -logits])
    logp1 = -logsumexp(logits_minus, axis=1)
    logp0 = -logsumexp(logits_plus, axis=1)
    logprobs = logp1 * targets + logp0 * (1-targets)
    return -np.sum(logprobs)/N

# Compute using numpy
def sigmoid(x): return 0.5 * (np.tanh(x / 2.) + 1)

def predict_logit(weights, inputs):
    return np.dot(inputs, weights) # Already vectorized

def predict_np(weights, inputs):
    return sigmoid(predict_logit(weights, inputs))

def NLL(weights, batch):
    X, y = batch
    logits = predict_logit(weights, X)
    return BCE_with_logits(logits, y)

def NLL_grad(weights, batch):
    X, y = batch
    N = X.shape[0]
    mu = predict_np(weights, X)
    g = np.sum(np.dot(np.diag(mu - y), X), axis=0)/N
    return g



# + colab={"base_uri": "https://localhost:8080/"} id="f9mD8S18746_" outputId="e023b766-2aaf-47bd-f552-3575c226e998"
w_np = w_mle_sklearn
y_pred = predict_np(w_np, X_test)
loss_np = NLL(w_np, (X_test, y_test))
grad_np = NLL_grad(w_np, (X_test, y_test))
print("params {}".format(w_np))
#print("pred {}".format(y_pred))
print("loss {}".format(loss_np))
print("grad {}".format(grad_np))

# + [markdown] id="YeGQ7SJTNHMk"
# ## PyTorch code

# + [markdown] id="Is7yJlgsL4BT"
# To compute the gradient using torch, we proceed as follows.
#
# - declare all the variables that you want to take derivatives with respect to using the requires_grad=True argumnet
# - define the (scalar output) objective function you want to differentiate in terms of these variables, and evaluate it at a point. This will generate a computation graph and store all the tensors.
# - call objective.backward() to trigger backpropagation (chain rule) on this graph.
# - extract the gradients from each variable using variable.grad field. (These will be torch tensors.)
#
# See the example below.

# + id="Wl_SK0WUlvNl"

# data. By default, numpy uses double but torch uses float
X_train_t = torch.tensor(X_train,  dtype=torch.float)
y_train_t = torch.tensor(y_train, dtype=torch.float)

X_test_t = torch.tensor(X_test, dtype=torch.float)
y_test_t = torch.tensor(y_test, dtype=torch.float)

# + id="0L5NxIaVLu64" colab={"base_uri": "https://localhost:8080/"} outputId="a4cd1bbd-7069-4e5f-ade7-5e563a0fe11d"
# parameters
W = np.reshape(w_mle_sklearn, [D, 1]) # convert 1d vector to 2d matrix
w_torch = torch.tensor(W, requires_grad=True, dtype=torch.float)
#w_torch.requires_grad_() 


# binary logistic regression in one line of Pytorch
def predict(X, w):
  y_pred = torch.sigmoid(torch.matmul(X, w))[:,0]
  return y_pred

# This returns Nx1 probabilities
y_pred = predict(X_test_t, w_torch)

# loss function is average NLL
criterion = torch.nn.BCELoss(reduction='mean')
loss_torch = criterion(y_pred, y_test_t)
print(loss_torch)

# Backprop
loss_torch.backward()
print(w_torch.grad)

# convert to numpy. We have to "detach" the gradient tracing feature
loss_torch = loss_torch.detach().numpy()
grad_torch = w_torch.grad[:,0].detach().numpy()


# + colab={"base_uri": "https://localhost:8080/"} id="CSKAJvrBNKQC" outputId="db315c9e-db41-46be-9bea-62f1d6c670c5"
# Test
assert np.allclose(loss_np, loss_torch)
assert np.allclose(grad_np, grad_torch)

print("loss {}".format(loss_torch))
print("grad {}".format(grad_torch))

# + [markdown] id="wnDGAWolHvr6"
# # Autograd on a DNN
#
# Below we show how to define more complex deep neural networks, and how to access
# their parameters. We can then call backward() on the scalar loss function, and extract their gradients. We base our presentation on http://d2l.ai/chapter_deep-learning-computation/parameters.html.

# + [markdown] id="V2U62DaVJWdZ"
# ## Sequential models

# + [markdown] id="hJLeA_iSILO9"
# First we create a shallow MLP.

# + colab={"base_uri": "https://localhost:8080/"} id="wF1XbC4FINmU" outputId="43075b01-fdb0-442f-c752-0caaf9cdf95d"
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4)) # batch x Din, batch=2, Din=4
out = net(X) # batch x Dout, Dout=1
print(out)

# + [markdown] id="lqm0UMNqImmo"
# Let's visualize the model and all the parameters in each layer.

# + colab={"base_uri": "https://localhost:8080/"} id="NRpwKHHkIqGB" outputId="699f940b-ac43-42b5-e520-3fab8ac00cb6"
print(net)

# + colab={"base_uri": "https://localhost:8080/"} id="4awxRZVWIrZ5" outputId="cb526a22-c1ac-4fd9-8fd0-55c4d8c1f89b"
for i in range(3):
  print(f'layer {i}')
  print(net[i].state_dict())


# + colab={"base_uri": "https://localhost:8080/"} id="2IQvcLI3JIao" outputId="70fb65ce-1aea-4bd6-ef90-42589f58910e"
print(*[(name, param.shape) for name, param in net.named_parameters()])

# + [markdown] id="NOgo3ZDVI65-"
# Access a specific parameter.

# + colab={"base_uri": "https://localhost:8080/"} id="9uc8WcZbI5lm" outputId="cb8d8dbe-23c4-4b1b-bfc4-f572a2763086"
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print(net.state_dict()['2.bias'].data)


# + [markdown] id="n3jLejNZJCr0"
# The gradient is not defined until we call backward.

# + colab={"base_uri": "https://localhost:8080/"} id="k24THwyaJFLy" outputId="7606013b-f2e5-48f3-f768-b59fc7d68802"
net[2].weight.grad == None


# + [markdown] id="9SWFmadHJYwh"
# ## Nested models

# + colab={"base_uri": "https://localhost:8080/"} id="4C7zBkNbJab9" outputId="aa803c46-c2a8-4c81-d267-8a710cba557b"
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4),
                         nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)


# + [markdown] id="8OEtpNDYJkb8"
# Let us access the 0 element of the top level sequence,
# which is block 0-3. Then we access element 1 of this,
# which is block 1. Then we access element 0 of this, 
# which is the first linear layer.

# + colab={"base_uri": "https://localhost:8080/"} id="3Mw3kRZrJl4w" outputId="c4c283c1-b9cc-43ad-f476-ffe015003d3f"
rgnet[0][1][0].bias.data

# + [markdown] id="ptfMznAmJ9Pl"
# ## Backprop

# + colab={"base_uri": "https://localhost:8080/"} id="0SBfDYcYJ-n1" outputId="6099ed4a-f174-4654-b4eb-71bce6a5b030"
# set loss function to output squared
out = rgnet(X)
loss = torch.mean(out ** 2, dim=0)

# Backprop
loss.backward()
print(rgnet[0][1][0].bias.grad)



# + [markdown] id="LCkFzrRtNbQF"
# ## Tied parameters
#
# Sometimes parameters are reused in multiple layers, as we show below.
# In this case, the gradients are added.

# + id="dzbKIBg5NiLM" outputId="b0444359-3573-4f0e-beec-00edbfba5ea0" colab={"base_uri": "https://localhost:8080/"}
# We need to give the shared layer a name so that we can refer to its
# parameters
torch.manual_seed(0)
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])

# + [markdown] id="OxwspvGxTprm"
# # Other material
#
# - [Stackoverflow post on gradient accumulation](https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch)
#
#
#

# + [markdown] id="mtKIyIrBU-s4"
# To compute gradient of a function that does not return a scalar
# (eg the gradient of each output wrt each input), you can do the following.

# + id="QJG9BTRPUXqV" colab={"base_uri": "https://localhost:8080/"} outputId="428735cf-8a41-480a-b9fc-fe69a1c6cb94"
x = torch.tensor([-2,-1,0,1,2], dtype=float, requires_grad=True)
print(x)
y = torch.pow(x, 2)
print(y)
y.backward(torch.ones_like(x))
print(x.grad)

# + id="KQHYKAQSVbAR"

