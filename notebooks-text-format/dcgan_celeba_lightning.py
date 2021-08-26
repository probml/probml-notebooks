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

# + [markdown] id="IWhlrdghx0bT"
# # DCGAN for Celeba (pytorch lightning)

# + [markdown] id="UcfHtR-JYXof"
# ## Installation and download

# + id="T09zvD7-WQGl" colab={"base_uri": "https://localhost:8080/"} outputId="572d12f4-54de-402c-c358-dbdcc0f87503"
# !mkdir figures
# !mkdir scripts
# %cd /content/scripts
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/dcgan-mish-noisy-instances-good.ckpt
# !wget -q https://github.com/probml/probml-data/raw/main/data/2_image_in_latent_space.npy
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/lvm_plots_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/dcgan_celeba_lightning.py
# !wget -q https://raw.githubusercontent.com/sayantanauddy/vae_lightning/main/data.py

# + id="Ig4r8vS3VRYU"
# %%capture
# ! pip install --quiet torchvision pytorch-lightning torchmetrics  torch test-tube lightning-bolts einops umap-learn

# + [markdown] id="iE5FWSBIY0gD"
# ## Importing modules

# + id="grPDEOHNVuB1"
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from einops import rearrange
from tqdm import tqdm
from lvm_plots_utils import get_random_samples, get_grid_samples, get_imrange
from dcgan_celeba_lightning import DCGAN

# + id="1BwkDBAeGNya"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + [markdown] id="6g6OwQMFyH6r"
# ## Defining the model

# + colab={"base_uri": "https://localhost:8080/"} id="l_Z8mNacYBRy" outputId="7dc068df-4627-448b-ccd7-0826cce3560b"
m = DCGAN()
m.load_state_dict(torch.load("dcgan-mish-noisy-instances-good.ckpt"))
m.to(device)

# + [markdown] id="a6_ZwKncySV4"
# ## Sampling from a TN[0,1] distribution

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="Yr0ZxuULFGD_" outputId="aaaba4f9-5750-4409-afd0-9738b4959733"
plt.figure(figsize=(10,10))
# Where 5 is the truncation threshold for our truncated normal distribution we are sampling from
imgs= get_random_samples(m, 5, 100)
plt.imshow(imgs)

# + [markdown] id="EbmBGYUTyWlG"
# ## Intepolation

# + id="akFRvNnwZsPH"
start, end = np.load("2_image_in_latent_space.npy")
start, end = torch.tensor(start, device=device), torch.tensor(end, device=device)


# + [markdown] id="Hgc7LEI20A-j"
# ### Spherical interpolation

# + id="ssXOvkpJZ2OM" colab={"base_uri": "https://localhost:8080/", "height": 140} outputId="247971da-f8ce-444c-d57b-69f1e6508771"
def decoder(img):
  return rearrange(m(img), "b c h w -> (b c) h w")

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end)
plt.imshow(arr)


# + [markdown] id="93FjhPhDzzs8"
# ### Linear interpolation 

# + id="5pNWF3AtZ6WI" colab={"base_uri": "https://localhost:8080/", "height": 140} outputId="4a1dae91-4be5-4f6d-e287-a6f1eb37ee0e"
def decoder(img):
  return rearrange(m(img), "b c h w -> (b c) h w")

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)

# + id="7GHqcLn7Kiyf"

