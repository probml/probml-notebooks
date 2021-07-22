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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/logreg_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="b520E1nCIBHc"
#
# # Logistic regression using PyTorch
#
# We show how to fit a logistic regression model using PyTorch. The log likelihood for this model is convex, so we can compute the globally optimal MLE. This makes it easy to compare to sklearn (and other implementations). 
#
#

# + id="UeuOgABaIENZ"
import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import itertools
import time
from functools import partial

import os

import numpy as np
from scipy.special import logsumexp
np.set_printoptions(precision=3)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# + id="GPozRwDAKFb8" colab={"base_uri": "https://localhost:8080/"} outputId="2a8248be-61d4-43fb-9cce-2c36e47d3be5"


import torch
import torch.nn as nn
import torchvision
print("torch version {}".format(torch.__version__))
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  print("current device {}".format(torch.cuda.current_device()))
else:
  print("Torch cannot find GPU")

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True

# + [markdown] id="kjP6xqkvbKxe"
# #  Logistic regression using sklearn
#
# We fit  binary logistic regresion on the Iris dataset. 

# + colab={"base_uri": "https://localhost:8080/"} id="aSYkjaAO6n3A" outputId="e5c90ccb-b01b-4115-ac78-1524e72b63e1"
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

# + [markdown] id="-pIgD7iRLUBt"
# # Automatic differentiation <a class="anchor" id="AD"></a>
#
#  
# In this section, we illustrate how to use autograd to compute the gradient of the negative log likelihood for binary logistic regression. We first compute the gradient by hand, and then use PyTorch's autograd feature. 
# (See also [the JAX optimization colab](https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/opt.ipynb).)
#

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



# + colab={"base_uri": "https://localhost:8080/"} id="f9mD8S18746_" outputId="25c03ff4-36f7-4e49-81c6-c56c9f369bd3"
w_np = w_mle_sklearn
y_pred = predict_prob(w_np, X_test)
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

# + id="0L5NxIaVLu64" colab={"base_uri": "https://localhost:8080/"} outputId="2215d4e6-76c5-4437-950f-cb0430708f3a"
# parameters
W = np.reshape(w_mle_sklearn, [D, 1]) # convert 1d vector to 2d matrix
w_torch = torch.tensor(W, requires_grad=True, dtype=torch.float)
#w_torch.requires_grad_() 


# binary logistic regression in one line of Pytorch
def predict_t(w, X):
  y_pred = torch.sigmoid(torch.matmul(X, w))[:,0]
  return y_pred

# This returns Nx1 probabilities
y_pred = predict_t(w_torch, X_test_t)

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


# + colab={"base_uri": "https://localhost:8080/"} id="CSKAJvrBNKQC" outputId="5e6fc814-03aa-4518-8263-3951a40d4ed3"
# Test
assert np.allclose(loss_np, loss_torch)
assert np.allclose(grad_np, grad_torch)

print("loss {}".format(loss_torch))
print("grad {}".format(grad_torch))

# + [markdown] id="DLWeq4d-6Upz"
# # Batch optimization using BFGS
#
# We will use BFGS from PyTorch for fitting a logistic regression model, and compare to sklearn.

# + colab={"base_uri": "https://localhost:8080/"} id="yiefA00AuXK4" outputId="7d19a74b-69c1-4aea-eeb2-a95a2f429553"
set_seed(0)
params = torch.randn((D,1), requires_grad=True)
optimizer = torch.optim.LBFGS([params], history_size=10)
    
def closure():
    optimizer.zero_grad()
    y_pred = predict_t(params, X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    return loss

max_iter = 10
for i in range(max_iter):
    loss = optimizer.step(closure)
    print(loss.item())

# + colab={"base_uri": "https://localhost:8080/"} id="gcsx3JCGuISp" outputId="1f33c971-164f-4555-bd74-efaecb4664b3"
print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from torch {}".format(params[:,0]))

# + colab={"base_uri": "https://localhost:8080/"} id="LSt8z7m5uuvK" outputId="dba94f42-47db-43e9-bf9d-ed62ce3bcb5d"

p_pred_np = predict_prob(w_np, X_test)
p_pred_t = predict_t(params, X_test_t) 
p_pred = p_pred_t.detach().numpy()
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(p_pred_np)
print(p_pred)



# + [markdown] id="8TMzOBNtUaW6"
# # Stochastic optimization using SGD

# + [markdown] id="9byvNfJ9QpsH"
# ## DataLoader
#
# First we need a way to get minbatches of data.

# + colab={"base_uri": "https://localhost:8080/"} id="O_jliQydRXUB" outputId="3ddf9e6b-caaa-40ea-c67d-25e615e26fc5"

from torch.utils.data import DataLoader, TensorDataset

# To make things interesting, we pick a batchsize of B=33, which is not divisible by N=100
dataset = TensorDataset(X_train_t, y_train_t)
B = 33
dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
print(X_train_t.shape)
print('{} examples divided into {} batches of size {}'.format(
    len(dataloader.dataset), len(dataloader), dataloader.batch_size))

for i, batch in enumerate(dataloader):
  X, y = batch
  print(X.shape)
  print(y.shape)
 

# + colab={"base_uri": "https://localhost:8080/"} id="ui_gFE0wWSIS" outputId="ab3b8828-c4eb-4be0-a7cd-dac11770c7c9"
datastream = iter(dataloader)
for i in range(3):
  X,y = next(datastream)
  print(y)

# + [markdown] id="Wux6hg6JVe7O"
# ## Vanilla SGD training loop

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="bXBNl-vwVejO" outputId="6bbec107-9bfb-432d-f519-f34da6a95cb5"
set_seed(0)
params = torch.randn((D,1), requires_grad=True)
nepochs = 100
nbatches = len(dataloader)
criterion = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-1
loss_trace = []

for epoch in range(nepochs):
  for b, batch in enumerate(dataloader):
    X, y =  batch

    if params.grad is not None:
      params.grad.zero_() # reset gradient to zero
    y_pred = predict_t(params, X)
    loss = criterion(y_pred, y)
    #print(f'epoch {epoch}, batch {b}, loss: {loss.item()}')
    loss_trace.append(loss)

    loss.backward()
    with torch.no_grad():
      params -= learning_rate * params.grad
  
  #print(f'end of epoch {epoch}, loss: {loss.item()}')
    
plt.figure()
plt.plot(loss_trace)


# + colab={"base_uri": "https://localhost:8080/"} id="TK-4_-N5o4sK" outputId="fd3a3ed5-ddac-4bd3-e4ad-618956245754"
# SGD does not converge to a value that is close to the batch solver...

print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from torch {}".format(params[:,0]))

# + colab={"base_uri": "https://localhost:8080/"} id="0dBeBatDo_Xy" outputId="073ed389-f504-4bf9-ba07-d06a5da0d6a4"
# Predicted probabilities from SGD are very different to sklearn
# although the thresholded labels are similar
    

p_pred_np = predict_prob(w_np, X_test)
p_pred_t = predict_t(params, X_test_t) 
p_pred = p_pred_t.detach().numpy()

print(p_pred_np)
print(p_pred)



# + colab={"base_uri": "https://localhost:8080/"} id="5e8Wugc1eLHT" outputId="e9c62c71-bbad-4e7a-fe81-5e4279f0a6d2"
y_pred_np = p_pred_np > 0.5
y_pred = p_pred > 0.5
print(y_pred_np)
print(y_pred)
print(np.sum(y_pred_np == y_pred)/len(y_pred))

# + [markdown] id="2AlO5fUmrMzI"
# ## Use Torch SGD optimizer
#
# Instead of writing our own optimizer, we can use a torch optimizer. This should give identical results.

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="YzC12T6mrOmb" outputId="8725516e-dbd3-4e20-efdc-7ecc76944d5d"
set_seed(0)
params = torch.randn((D,1), requires_grad=True)
nepochs = 100
nbatches = len(dataloader)
criterion = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-1
loss_trace = []

# optimizer has pointer to params, so can mutate its state
optimizer = torch.optim.SGD([params], lr=learning_rate)
    
for epoch in range(nepochs):
  for b, batch in enumerate(dataloader):
    X, y =  batch

    y_pred = predict_t(params, X)
    loss = criterion(y_pred, y)
    #print(f'epoch {epoch}, batch {b}, loss: {loss.item()}')
    loss_trace.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  #print(f'end of epoch {epoch}, loss: {loss.item()}')
    
plt.figure()
plt.plot(loss_trace)



# + id="IMxLRicCvW_Y" outputId="eb6e6e21-7533-4326-c4c7-3aafe46f2f2f" colab={"base_uri": "https://localhost:8080/"}
print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from torch {}".format(params[:,0]))

p_pred_np = predict_prob(w_np, X_test)
p_pred_t = predict_t(params, X_test_t) 
p_pred = p_pred_t.detach().numpy()

print('predictions from sklearn')
print(p_pred_np)
print('predictions from torch')
print(p_pred)

y_pred_np = p_pred_np > 0.5
y_pred = p_pred > 0.5
print('fraction of predicted labels that agree ', np.sum(y_pred_np == y_pred)/len(y_pred))

# + [markdown] id="Hr8WRZP6vtBT"
# ## Use momentum optimizer
#
# Adding momentum helps a lot, and gives results which are very similar to batch optimization.

# + id="3D4E4JGdvvcU" outputId="09114882-d6fd-493d-86cb-41d9d100be99" colab={"base_uri": "https://localhost:8080/", "height": 282}
set_seed(0)
params = torch.randn((D,1), requires_grad=True)
nepochs = 100
nbatches = len(dataloader)
criterion = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-1
loss_trace = []

# optimizer has pointer to params, so can mutate its state
optimizer = torch.optim.SGD([params], lr=learning_rate, momentum=0.9)
    
for epoch in range(nepochs):
  for b, batch in enumerate(dataloader):
    X, y =  batch

    y_pred = predict_t(params, X)
    loss = criterion(y_pred, y)
    #print(f'epoch {epoch}, batch {b}, loss: {loss.item()}')
    loss_trace.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  #print(f'end of epoch {epoch}, loss: {loss.item()}')
    
plt.figure()
plt.plot(loss_trace)

# + id="ym9Lz7tCv41V" outputId="23d5a414-f457-4c25-d28e-3b4319db53c5" colab={"base_uri": "https://localhost:8080/"}
print("parameters from sklearn {}".format(w_mle_sklearn))
print("parameters from torch {}".format(params[:,0]))

p_pred_np = predict_prob(w_np, X_test)
p_pred_t = predict_t(params, X_test_t) 
p_pred = p_pred_t.detach().numpy()

print('predictions from sklearn')
print(p_pred_np)
print('predictions from torch')
print(p_pred)

y_pred_np = p_pred_np > 0.5
y_pred = p_pred > 0.5
print('fraction of predicted labels that agree ', np.sum(y_pred_np == y_pred)/len(y_pred))

# + [markdown] id="Jn1sZgoJ0d7s"
# # Modules
#
# We can define logistic regression as multilayer perceptron (MLP) with no hidden layers. This can be defined as a sequential neural network module. Modules hide the parameters inside each layer, which makes it easy to construct complex models, as we will see later on.
#

# + [markdown] id="DN7AA9V_lm9W"
# ## Sequential model

# + colab={"base_uri": "https://localhost:8080/"} id="fjF4RwWWe3-g" outputId="c4a911a2-9810-4929-f77d-e1e118733509"
# Make an MLP with no hidden layers

model = nn.Sequential(
    nn.Linear(D, 1, bias=False),
    nn.Sigmoid()
)
print(model)
print(model[0].weight)
print(model[0].bias)

# + colab={"base_uri": "https://localhost:8080/"} id="Oie5FZnThX1B" outputId="91bec64e-ff8b-4c79-c74b-a284f6b19b36"
# We set the parameters of the MLP by hand to match sklearn.
# Torch linear layer computes X*W' + b (see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
# where X is N*Din, so W must be Dout*Din. Here Dout=1.
print(model[0].weight.shape)
print(w_np.shape)
w = np.reshape(w_np, [-1, 1]).transpose()
print(w.shape)
model[0].weight = nn.Parameter(torch.Tensor(w))
print(model[0].weight.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="simLA1V0fz4Y" outputId="4ed01d9d-9246-4ad4-e088-f30823fe07ab"

p_pred_np = predict_prob(w_np, X_test)
p_pred_t = model(X_test_t).detach().numpy()[:,0] 
print(p_pred_np)
print(p_pred_t)
assert np.allclose(p_pred_np, p_pred_t)

# + colab={"base_uri": "https://localhost:8080/"} id="1K60WLEOl-_3" outputId="bee54e14-5b0f-459b-f794-09dcdfadb216"
# we can assign names to each layer in the sequence

from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('linear_layer', nn.Linear(D, 1, bias=False)),
    ('output_activation', nn.Sigmoid())
  ])
)
print(model)
print(model.linear_layer)
print(model.linear_layer.weight)
print(model.output_activation)



# + colab={"base_uri": "https://localhost:8080/"} id="c-O4sR1zmpn2" outputId="6ca16746-d4b3-443c-aaa8-431bdf739810"
# some layers define adjustable parameters, which can be optimized.
# we can inspect them thus:
for name, param in model.named_parameters():
  print(name, param.shape)

# + [markdown] id="MlirdZ6rlrE0"
# ## Subclass the Module class
#
# For more complex models (eg non-sequential), we can create our own subclass. We just need to define a 'forward' method that maps inputs to outputs, as we show below.

# + id="xp1y2uzD6xGD" colab={"base_uri": "https://localhost:8080/"} outputId="f68029fb-0bf9-44f0-a348-462acd329afa"


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(D, 1, bias=False) 
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred[:,0] # (N,1) -> (N)
    
set_seed(0)
model = Model() 
w = np.reshape(w_np, [-1, 1]).transpose()
model.linear.weight = nn.Parameter(torch.Tensor(w))

p_pred_np = predict_prob(w_np, X_test)
p_pred_t = model(X_test_t) # calls model.__call__ which calls model.forward()
p_pred = p_pred_t.detach().numpy()
print(p_pred_np)
print(p_pred)
assert np.allclose(p_pred_np, p_pred)



# + [markdown] id="dZqfTc03JIV7"
# ## SGD on a module
#
# We can optimize the parameters of a module by passing a reference to them into the optimizer, as we show below.

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="1K-Suo6jHynP" outputId="0c7918c6-76fc-49f2-e1cb-5a717568a680"


nepochs = 100
nbatches = len(dataloader)
criterion = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-1
loss_trace = []

set_seed(0)
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
for epoch in range(nepochs):
  for b, batch in enumerate(dataloader):
    X, y =  batch

    y_pred = model(X) # predict/ forward function
    loss = criterion(y_pred, y)
    #print(f'epoch {epoch}, batch {b}, loss: {loss.item()}')
    loss_trace.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  #print(f'end of epoch {epoch}, loss: {loss.item()}')
    
plt.figure()
plt.plot(loss_trace)


y_pred_np = predict_prob(w_np, X_test)
y_pred_t = model(X_test_t) 
y_pred = y_pred_t.detach().numpy()
print(y_pred_np)
print(y_pred)

# + [markdown] id="MGbegp5xJKSN"
# ## Batch optimization on a module
#
# SGD does not match the results of sklearn. However, this is not because of the way we defined the model, it's just because SGD is a bad optimizer. Here we show that BFGS gives exactly the same results as sklearn.
#

# + colab={"base_uri": "https://localhost:8080/"} id="5BN5X-1w62ST" outputId="a05136f2-326e-4675-da1a-8de71cffaeed"

set_seed(0)
model = Model()
optimizer = torch.optim.LBFGS(model.parameters(), history_size=10)

criterion = torch.nn.BCELoss(reduction='mean')
def closure():
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    return loss

max_iter = 10
loss_trace = []
for i in range(max_iter):
    loss = optimizer.step(closure)
    #print(loss)

y_pred_np = predict_prob(w_np, X_test)
y_pred_t = model(X_test_t) 
y_pred = y_pred_t.detach().numpy()
print(y_pred_np)
print(y_pred)



# + [markdown] id="jQTxqUFg4L1W"
# # Multi-class logistic regression
#
# For binary classification problems, we can use a sigmoid as the final layer, to return probabilities. The corresponding loss is the binary cross entropy, [nn.BCELoss(pred_prob, true_label)](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html), where pred_prob is of shape (B) with entries in [0,1], and true_label is of shape (B) with entries in 0 or 1. (Here B=batch size.) Alternatively the model can return the logit score, and use [nn.BCEWithLogitsLoss(pred_score, true_label)](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
#
# For multiclass classifiction, the final layer can return the log probabilities using LogSoftmax layer, combined with the negative log likelihood loss, [nn.NLLLoss(pred_log_probs, true_label)](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html), where pred_log_probs is of shape B*C matrix, and true_label is of shape B  with entries in {0,1,..C-1}.
# (Note that the target labels are integers, not sparse one-hot vectors.)
# Alternatively, we can just return the vector of logit scores, and use [nn.CrossEntropyLoss(logits, true_label)](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). The above two methods should give the same results.
#

# + id="j7g6aFCD7KI5"
# code me
