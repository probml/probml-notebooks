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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/layer_norm_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="pxxCjM4AIsZP"
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)
import math
import collections

import torch
from torch import nn
from torch.nn import functional as F



# + colab={"base_uri": "https://localhost:8080/"} id="OcFY2bZeImQI" outputId="bbfe5c4d-b5d4-4657-a446-269662828f4e"

# batch size 3, feature size 2
X = np.array([[1, 2, 3], [4, 5, 6]])
#X = np.array([[1, 2], [2,3]], dtype=np.float32)

print('batch norm')
mu_batch = np.mean(X,axis=0)
sigma_batch = np.std(X,axis=0)
XBN = (X-mu_batch)/sigma_batch
print(XBN)

print('layer norm')
mu_layer = np.expand_dims(np.mean(X,axis=1),axis=1)
sigma_layer = np.expand_dims(np.std(X,axis=1), axis=1)
XLN = (X-mu_layer)/sigma_layer
print(XLN)



# + colab={"base_uri": "https://localhost:8080/"} id="6p9f1kSyJbuT" outputId="c8344ff2-629b-4976-bfaa-bd65663f047b"

X = torch.tensor(X, dtype=torch.float32)
N, D = X.shape

ln = nn.LayerNorm(D)
bn = nn.BatchNorm1d(D)

print('batch norm')
XBN_t = bn(X)
print(XBN_t)
assert(np.allclose(XBN_t.detach().numpy(), XBN, atol=1e-3))

print('layer norm')
XLN_t = ln(X)
print(XLN_t)
assert(np.allclose(XLN_t.detach().numpy(), XLN, atol=1e-3))

# + id="HQU6R65hJvtE"

