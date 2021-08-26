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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/kernel_regression_attention.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="mL3OI3wikUuj"
# # Nadaraya-Watson kernel regression in 1d using attention
#
#
# We show how to interpret kernel regression as an attention mechanism.
# Based on sec 10.2 of http://d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html
#
#
#

# + id="IcfKKfw3kJVQ"
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)
import math
import collections

import torch
from torch import nn
from torch.nn import functional as F

# !mkdir figures # for saving plots

# !wget https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l/torch.py -q -O d2l.py
import d2l

# + [markdown] id="qtoprluTknT3"
# # Data

# + colab={"base_uri": "https://localhost:8080/"} id="2xidWUekkhz3" outputId="b08ed23a-4dac-44ce-857b-403fd48ce0fc"
torch.manual_seed(0)
n_train = 50  # No. of training examples
x_train, _ = torch.sort(torch.rand(n_train) * 5)  # Training inputs

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = torch.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test


# + [markdown] id="f29Wy3ShkyMf"
# # Constant baseline
#
# As a baseline, we use the empirical mean of y.

# + id="twKhgWQTkwXS"
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
    


# + colab={"base_uri": "https://localhost:8080/", "height": 267} id="27z7r-UZk1zo" outputId="3c6069c7-9ab2-4df7-a24f-82b4066ba645"
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)


# + [markdown] id="Bm34WWMlk9Gh"
# # Kernel regression

# + colab={"base_uri": "https://localhost:8080/", "height": 267} id="6oiZc7IVk4oz" outputId="01754096-e060-4b4a-ad52-9bd94682a57d"
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
plt.savefig('kernelRegrAttenPlot.pdf', dpi=300)

# + [markdown] id="CbgnXbeHlJt5"
# We can visualize the kernel matrix to see which inputs are used to predict each output.

# + colab={"base_uri": "https://localhost:8080/", "height": 233} id="YFXJRYlnlGre" outputId="2e289ddd-206b-4f31-c61d-25e21c7516a6"
d2l.show_heatmaps(
    attention_weights.unsqueeze(0).unsqueeze(0),
    xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
plt.savefig('kernelRegrAttenMat.pdf', dpi=300)

# + [markdown] id="tUztfZujrh0X"
# # Implementation using learned attention
#
# As an illustration of how to learn attention kernels, we make the bandwidth parameter adjustable, so we can optimize it by backprop.
#

# + [markdown] id="5iyJBsy_ryvV"
# The implementation uses batch matrix multiplication (torch.bmm).
# This is defined as follows. Suppose the first batch contains n matrix Xi of size a x b, and the second batch contains n matrix Yi of size b x c. Then the output will have size (n, a, c).
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="s7b5mXpJsOm7" outputId="da7c898b-40d1-47af-8041-14f3c0b125f7"
# 2 batches of weights over the 10 data points
weights = torch.ones((2, 10)) * 0.1
weights = weights.unsqueeze(1)
print(weights.shape) #(2,1,10)

# 2 batches of 10 scalar data points
values = torch.arange(20.0).reshape((2, 10))
values = values.unsqueeze(-1)
print(values.shape) # (2,10,1) 

Y = torch.bmm(weights, values)
print(Y.shape)
print(Y)


# + id="xEASumiAlHF3"
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape(
            (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


# + [markdown] id="PvmAoprztHs7"
# To apply attention to kernel regression, we make a batch of size $N$, where $N$ is the number of training points. In batch $i$, the query is the $i$'th training point we are truying to predict, the keys are all the other inputs $x_{-i}$ and the values are all the other outpouts $y_{-i}$.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="ihF29w0LtDe8" outputId="742310ad-43f3-438b-ba66-4b0529e590ba"
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape(
    (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape(
    (n_train, -1))

print([x_train.shape, X_tile.shape, keys.shape, values.shape])


# + [markdown] id="Fba8Q4ZsuaDQ"
# Train using SGD.

# + colab={"base_uri": "https://localhost:8080/", "height": 262} id="293WGYYEue6K" outputId="2862ccf8-6fee-4592-ec1d-9a9dc5f2a395"
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    # Note: L2 Loss = 1/2 * MSE Loss. PyTorch has MSE Loss which is slightly
    # different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
    l = loss(net(x_train, keys, values), y_train) / 2
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# + [markdown] id="_z20MNS8vP8N"
# # Results of training
#
# Not suprisignly, fitting the hyper-parameter 'w' (the bandwidth of the kernel) results in overfitting, as we show below. However, for parametric attention, this is less likely to occur.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 267} id="3W0-3lq0ubBP" outputId="aa119054-d642-4196-8485-a7b16a8276d9"
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

# + colab={"base_uri": "https://localhost:8080/", "height": 233} id="NDA9wMmPvK4S" outputId="ab449d8c-16c4-4a67-e41c-9b4b212df7be"
d2l.show_heatmaps(
    net.attention_weights.unsqueeze(0).unsqueeze(0),
    xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

# + colab={"base_uri": "https://localhost:8080/"} id="thB233zmvc46" outputId="76aba334-9a51-42aa-a127-12c148afb58b"
print(net.w)

# + id="2gxdDrmTwAmV"

