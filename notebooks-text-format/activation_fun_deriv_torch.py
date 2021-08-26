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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/activation_fun_deriv_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="B4W2erKbxzFe"
# # Plot some neural net activation functions and their derivatives
# Based on sec 4.1 of
#  http://d2l.ai/chapter_multilayer-perceptrons/mlp.html
#

# + id="CjpOJet7x_Us"

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)

# + id="QY5z78In9lC8"
# !mkdir figures

# + id="W-FG3Ep72GB9"
# #!wget https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l/torch.py -q -O d2l.py
#import d2l

# + id="XNJacITv2Smd"
import torch
from torch import nn
from torch.nn import functional as F

# + id="i5tLv0SvxoOW"
x = torch.arange(-4.0, 4.0, 0.1, requires_grad=True)
fns = [torch.sigmoid,
       #torch.relu,
       torch.nn.LeakyReLU(negative_slope=0.1),
       torch.nn.ELU(alpha=1.0),
       torch.nn.SiLU(),
       torch.nn.GELU()]
names = ['sigmoid',
         #'relu',
         'leaky-relu',
         'elu',
         'swish',
         'gelu']

# evaluate functions and their gradients on a grid of points
xs = x.detach()
fdict = {}
gdict = {}
for i in range(len(fns)):
    fn = fns[i]
    name = names[i]
    y = fn(x)
    fdict[name] = y.detach() # vector of fun    
    y.backward(torch.ones_like(x), retain_graph=True) # returns gradient at each point
    gdict[name] = torch.clone(x.grad) # gradient wrt x(i)
    x.grad.data.zero_() # clear out old gradient for next iteration


    

# + colab={"base_uri": "https://localhost:8080/", "height": 577} id="VyLzcYPbQNaD" outputId="9c172ce3-ff8c-4dcf-8905-5f38909362f1"
# Plot the funcitons
styles = ['r-', 'g--', 'b-.', 'm:', 'k-']
ax = plt.subplot()
for i, name in enumerate(names): 
    lab = f'{name}'
    ax.plot(xs, fdict[name], styles[i], label=lab)
ax.set_ylim(-0.5,2)
ax.legend()
plt.title('Activation function')
plt.tight_layout()
plt.savefig(f'figures/activation-funs.pdf', dpi=300)
plt.show()

ax = plt.subplot()
for i, name in enumerate(names):
    lab = f'{name}'
    ax.plot(xs, gdict[name], styles[i], label=lab)
ax.set_ylim(-0.5,1.5)
ax.legend()
plt.title('Gradient of activation function')
plt.tight_layout()
plt.savefig(f'figures/activation-funs-grad.pdf', dpi=300)
plt.show()

    

# + id="RkUeVBiW-aZ4" colab={"base_uri": "https://localhost:8080/"} outputId="1903298c-58f0-4e3e-c194-cdb32e346210"
# !ls figures

# + id="pGy1QBVrAfZK"

