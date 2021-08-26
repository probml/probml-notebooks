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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/supplements/sps_logreg_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="xL5mnbcoPSwS"
# # Stochastic Polyak Stepsize
#
# https://github.com/IssamLaradji/sps/
#

# + [markdown] id="t-xF9fUhRrNh"
# ## Setup

# + id="lIYdn1woOS1n" colab={"base_uri": "https://localhost:8080/"} outputId="dd0da7f0-bc95-4392-d8d0-a9bd644d0bca"
# !pip install git+https://github.com/IssamLaradji/sps.git



# + colab={"base_uri": "https://localhost:8080/", "height": 69} id="SFqkede0OqHA" outputId="c68f39a6-00e9-4463-fb4c-8fd4930533dc"


# + id="IeoOYXlFPOQC"
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

# + colab={"base_uri": "https://localhost:8080/"} id="aGHlxsklPOfV" outputId="c5b59867-9501-4b56-ea6c-a1b5c7b28760"


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

# + [markdown] id="2GXyZ5lbRtiP"
# ## Binary logistic regression using Sklearn

# + colab={"base_uri": "https://localhost:8080/"} id="7tMZB1EbPvL4" outputId="32ddc0c5-05eb-4ba3-f72a-6bb0cfab95e6"
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

# extract probability of class 1
pred_sklearn_train = log_reg.predict_proba(X_train)[:,1]
pred_sklearn_test = log_reg.predict_proba(X_test)[:,1]

w_mle_sklearn = np.ravel(log_reg.coef_)
print(w_mle_sklearn)

# + [markdown] id="RLoJ089uRvpm"
# ## PyTorch data and model

# + id="CxwfvaZIRKOl"
from torch.utils.data import DataLoader, TensorDataset

# data. By default, numpy uses double but torch uses float
X_train_t = torch.tensor(X_train,  dtype=torch.float)
y_train_t = torch.tensor(y_train, dtype=torch.float)

X_test_t = torch.tensor(X_test, dtype=torch.float)
y_test_t = torch.tensor(y_test, dtype=torch.float)

# To make things interesting, we pick a batchsize of B=33, which is not divisible by N=100
dataset = TensorDataset(X_train_t, y_train_t)
B = 33
dataloader = DataLoader(dataset, batch_size=B, shuffle=True)


# + id="bS5yRf0hTXn8"

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(D, 1, bias=False) 
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred[:,0] # (N,1) -> (N)


# + id="sbOaISEJbMOQ"
def criterion(ypred, ytrue, L2reg=0):
  loss = torch.nn.BCELoss(reduction='mean')(ypred, ytrue)
  w = 0.
  for p in model.parameters():
    w += (p**2).sum()
  loss += L2reg * w
  return loss



# + [markdown] id="fYYoAYmFVdME"
# ## BFGS

# + id="d-R-zu2uVgNE"
set_seed(0)
model = Model()
loss_trace = []
optimizer = torch.optim.LBFGS(model.parameters(), history_size=10)
    
def closure():
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t, L2reg=0)
    loss.backward()
    return loss.item()

max_iter = 10
for i in range(max_iter):
    loss = optimizer.step(closure)
    loss_trace.append(loss)

# + colab={"base_uri": "https://localhost:8080/", "height": 469} id="FFn5aR3mWA2x" outputId="0b3f0bf2-20a4-4993-b876-9141f14be25f"
plt.figure()
plt.plot(loss_trace)

pred_sgd_train = model(X_train_t).detach().numpy() 
pred_sgd_test = model(X_test_t).detach().numpy() 


print('predicitons on test set using sklearn')
print(pred_sklearn_test)
print('predicitons on test set using sgd')
print(pred_sgd_test)

# + id="NOR9eeF0bi7_"
set_seed(0)
model = Model()
loss_trace = []
optimizer = torch.optim.LBFGS(model.parameters(), history_size=10)
    
def closure():
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t, L2reg=1e-4)
    loss.backward()
    return loss.item()

max_iter = 10
for i in range(max_iter):
    loss = optimizer.step(closure)
    loss_trace.append(loss)

# + colab={"base_uri": "https://localhost:8080/", "height": 469} id="Gdn04qMbblTE" outputId="1743c22d-eecd-4501-e104-257180d16fc6"
plt.figure()
plt.plot(loss_trace)

pred_sgd_train = model(X_train_t).detach().numpy() 
pred_sgd_test = model(X_test_t).detach().numpy() 


print('predicitons on test set using sklearn')
print(pred_sklearn_test)
print('predicitons on test set using sgd')
print(pred_sgd_test)

# + [markdown] id="7r8I5KxvRzaF"
# ## SGD

# + id="qVePsT5mRo_V"
nepochs = 100
learning_rate = 1e-1
loss_trace = []

set_seed(0)
model = Model()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    
for epoch in range(nepochs):
  for X, y in dataloader:
    y_pred = model(X) 
    loss = criterion(y_pred, y, L2reg=0)
    loss_trace.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  


# + colab={"base_uri": "https://localhost:8080/", "height": 473} id="Zgm1A5v5TCHv" outputId="ae833416-0ec0-45dd-b53a-12872d855f63"
plt.figure()
plt.plot(loss_trace)
plt.ylim([0, 2])

pred_sgd_train = model(X_train_t).detach().numpy() 
pred_sgd_test = model(X_test_t).detach().numpy() 


print('predicitons on test set using sklearn')
print(pred_sklearn_test)
print('predicitons on test set using sgd')
print(pred_sgd_test)

# + id="QDrKl3FOcRXA"


# + [markdown] id="OwVVkYKYcSGn"
# ## Momentum

# + id="S3Avm0QbcUXS"
nepochs = 100
loss_trace = []

set_seed(0)
model = Model()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    
for epoch in range(nepochs):
  for X, y in dataloader:
    y_pred = model(X) 
    loss = criterion(y_pred, y, L2reg=0)
    loss_trace.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  

# + colab={"base_uri": "https://localhost:8080/", "height": 473} id="OcMtRWcQcXeo" outputId="60d06bc1-0568-4fce-b0a7-2ead98d358e6"
plt.figure()
plt.plot(loss_trace)
plt.ylim([0, 2])

pred_sgd_train = model(X_train_t).detach().numpy() 
pred_sgd_test = model(X_test_t).detach().numpy() 


print('predicitons on test set using sklearn')
print(pred_sklearn_test)
print('predicitons on test set using sgd')
print(pred_sgd_test)

# + [markdown] id="jB7ynCEOUKJp"
# ## SPS

# + id="nOA7Vza4ThXj"
import sps

set_seed(0)
model = Model()
score_list = []

opt = sps.Sps(model.parameters(), c=0.5, eta_max=1, 
              adapt_flag='constant') 
#, fstar_flag=True)
 #c=0.2 blows up

nepochs = 100
for epoch in range(nepochs):
  for X, y in dataloader:

    def closure():
      loss = criterion(model(X), y, L2reg=1e-4)
      loss.backward()
      return loss

    opt.zero_grad()
    loss = opt.step(closure=closure)
    loss_trace.append(loss)
   
    # Record metrics
    score_dict = {"epoch": epoch}
    score_dict["step_size"] = opt.state.get("step_size", {})
    score_dict["step_size_avg"] = opt.state.get("step_size_avg", {})
    score_dict["train_loss"] = loss

    score_list += [score_dict]

# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="nOTgM9mJexEV" outputId="303478c9-3276-4ce9-fd4b-25954a8b3529"
import pandas as pd
df = pd.DataFrame(score_list)
df.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 473} id="5GmZmR8sW_4y" outputId="83d082ee-14f1-4c11-b52b-098cc3f73da6"
plt.figure()
plt.plot(df["train_loss"])
plt.ylim([0, 2])

pred_sgd_train = model(X_train_t).detach().numpy() 
pred_sgd_test = model(X_test_t).detach().numpy() 

print('predicitons on test set using sklearn')
print(pred_sklearn_test)
print('predicitons on test set using sgd')
print(pred_sgd_test)

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="YOcCW1VqVQxw" outputId="aed3d07d-b08d-4698-c94e-a3f799dfae3b"
plt.figure()
plt.plot(df["step_size"])




# + id="6ZtcgvBAfAin"

