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

# + [markdown] id="4kuVeLiRYXhF"
# # Polynomial regression in 1d.
# Based on sec 4.4 of
#  http://d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html

# + id="0prfTp1DYTEp"
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from IPython import display

# !mkdir figures # for saving plots

import warnings
warnings.filterwarnings("ignore")

# For reproducing the results on different runs
torch.backends.cudnn.deterministic = True
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


# + [markdown] id="725IgNzDZhmB"
# # Data

# + [markdown] id="iIMUISREY7TR"
# Make some data using this function:
#
# **$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
# \epsilon \sim \mathcal{N}(0, 0.1^2).$$**

# + id="bkpZVc5GYdf-"
np.random.seed(42)
max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="d7sLXgHuZJdd" outputId="23c591db-c418-41bc-df61-03f1eda575f2"
# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, poly_features, labels]]

print(true_w)


# + [markdown] id="Z515M5AbZjFe"
# # Train/eval loop

# + id="eWsTe9jEszao"
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        # d2l.use_svg_display()
        display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)



# + id="IOz7XrEytC3I"
# Incrementally update loss metrics during training
def evaluate_loss(net, data_iter, loss): 
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = y_hat.to(y.dtype) == y
    return float(torch.sum(cmp.to(y.dtype)))


# + id="JJ0E3F-3t6Wt"
def train_epoch(net, train_iter, loss, updater):
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        
        # Using PyTorch in-built optimizer & loss criterion
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


# + id="bjh-JnZfZRDz"
# SGD optimization

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400, batch_size=50):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(batch_size, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(
                net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())



# + [markdown] id="vSO6NjB-ZlTs"
# # Degree 3 (matches true function)
#
# Train and test loss are similar (no over or underfitting),
# Loss is small, since matches true function. Estimated parameters are close to the true ones.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 284} id="e3KhBZKVZXgD" outputId="1d48a01d-3bec-4dd6-9e3b-8620c9e9f2ff"

# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])


# + [markdown] id="DJuwQj92ZuVe"
# # Degree 1 (underfitting)

# + colab={"base_uri": "https://localhost:8080/", "height": 284} id="FQS7z5AwZv9V" outputId="cccfb28b-62bb-44db-eef8-45bb2d897e1d"
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])

# + [markdown] id="5YI8d3RZZ4ud"
# # Degree 20 (overfitting)
#
# According to the D2L book, the test loss is  higher than training loss.
# However, SGD itself has a regularizing effect (even in full batch mode),
# so I cannot reproduce overfitting (even though it would occur using a second order optimizer).

# + colab={"base_uri": "https://localhost:8080/", "height": 352} id="FSm_LDL-Z69f" outputId="808fcbfb-05b6-45c6-89ee-9e6735c4ef5d"
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:],
      num_epochs=2000, batch_size=n_train)

# + id="XWFZI3QZZ7RL"

