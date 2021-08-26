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

# + [markdown] id="Hu_N0fCyR4Nl"
# # MLP MNIST VAE 

# + [markdown] id="1-tsy6-HBy3N"
# ## Installation 

# + colab={"base_uri": "https://localhost:8080/"} id="QzYqzmyMdKc_" outputId="f1e8b0e7-4807-41cd-c1b2-1d55e067969b"
# !mkdir figures
# !mkdir scripts
# %cd /content/scripts
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/lvm_plots_utils.py
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/vae-mnist-mlp.ckpt
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/vae_mlp_mnist.py
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/vae-mnist-mlp-latent-dim-2.ckpt

# + id="lh8fn-m7bDI_"
# %%capture
# ! pip install --quiet torchvision pytorch-lightning torch test-tube einops umap

# + id="FhOfbWtlC53h"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from einops import rearrange
import seaborn as sns
from lvm_plots_utils import get_random_samples, get_grid_samples, plot_scatter_plot, get_imrange, plot_grid_plot
from torchvision.utils import make_grid
from pytorch_lightning.utilities.seed import seed_everything
from vae_mlp_mnist import BasicVAEModule

# + colab={"base_uri": "https://localhost:8080/"} id="211BTrwsRIgi" outputId="12a73f2a-9824-4511-e3dd-71a6bed579c6"
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + id="LFdra0ceDWNj"
mnist_full = MNIST(".", download=True, train=True,
                         transform=transforms.Compose([transforms.ToTensor()]))
dm = DataLoader(mnist_full, batch_size=500)
batch = next(iter(dm))

# + [markdown] id="31mWyHmCB5SV"
# ## Basic MLP VAE module

# + id="4q1LUDZoJBe1" colab={"base_uri": "https://localhost:8080/"} outputId="1ad2faef-168c-4e02-9a7e-29a1f9c46bb1"
m = BasicVAEModule(3)
m.load_state_dict(torch.load("vae-mnist-mlp.ckpt"))
m.to(device)

# + colab={"base_uri": "https://localhost:8080/"} id="J7oY7ML52BZA" outputId="9b559beb-a9e1-4a53-f283-678671141d0e"
m2 = BasicVAEModule(2)
m2.load_state_dict(torch.load("vae-mnist-mlp-latent-dim-2.ckpt"))
m2.to(device)

# + [markdown] id="K5XXaKXGB-3Q"
# ## Reconstruction

# + [markdown] id="9ZGYEgmV1jdN"
# ### ConvVAE with latent dim 20

# + id="dMgOlr14HwaR"
imgs, _ = batch
imgs = imgs[:16]
img_size = 28

def reconstruct(img):
  return m(rearrange(imgs, "b c h w -> b ( c h w)")).reshape(-1, 1, img_size, img_size)


# + colab={"base_uri": "https://localhost:8080/", "height": 273} id="7oOA5CeZFPSZ" outputId="941bc173-6062-4fc8-b983-b913a7594f15"
fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs).cpu(), 'c h w -> h w c'))
imgs = imgs.to(device=device)
print(reconstruct(imgs).shape)
axs[1].imshow(rearrange(make_grid(reconstruct(imgs)).cpu(), 'c h w -> h w c'))
plt.show()


# + [markdown] id="wqjwSC8-1neH"
# ### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 273} id="zB6_p01S1qKd" outputId="9d694605-01ad-4bf2-8f8b-bd948f32e20c"
def reconstruct(img):
  return m2(rearrange(imgs, "b c h w -> b ( c h w)")).reshape(-1, 1, img_size, img_size)

fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs).cpu(), 'c h w -> h w c'))
imgs = imgs.to(device=device)
print(reconstruct(imgs).shape)
axs[1].imshow(rearrange(make_grid(reconstruct(imgs)).cpu(), 'c h w -> h w c'))
plt.show()


# + [markdown] id="aljiDN2aegng"
# ## Sampling

# + [markdown] id="2hGE3H0xBGoW"
# ### Random samples from truncated normal distribution 

# + [markdown] id="QBfF94mUBK-w"
# We sample $z \sim TN(0,1)$ form a truncated normal distribution with a threshold = 5
#

# + [markdown] id="xNqsNQHj5awG"
# #### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 611} id="w4I0rbYFHgpH" outputId="47dd5370-841e-4e01-b5fb-473042cf2690"
def decoder(z):
  return m.vae.decoder(z).reshape(-1, 1, img_size, img_size)

plt.figure(figsize=(10,10))
# Where 5 is the truncation threshold for our truncated normal distribution we are sampling from
imgs= get_random_samples(decoder, 5)
plt.imshow(imgs)


# + [markdown] id="nmBSvTMR5inM"
# #### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 611} id="iFwUQMeF5llY" outputId="1d9f66f0-a572-4f33-e5a5-35afabd1ddc7"
def decoder(z):
  return m2.vae.decoder(z).reshape(-1, 1, img_size, img_size)

plt.figure(figsize=(10,10))
# Where 5 is the truncation threshold for our truncated normal distribution we are sampling from
imgs= get_random_samples(decoder, 5)
plt.imshow(imgs)


# + [markdown] id="KNmWbJ5UAnmj"
# ### Grid Sampling

# + [markdown] id="ZUdLxpgMAqIr"
# We let $z = [z1, z2, 0, \ldots, 0]$ and vary $z1, z2$ on a grid

# + [markdown] id="SRS8J8B15t3E"
# #### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="OXx-u19iGdc_" outputId="aa0db872-a90d-497d-c03a-6e680594d1c6"
def decoder(z):
  return m.vae.decoder(z).reshape(-1, img_size, img_size)

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 5), 10), " c h w -> h w c").cpu().detach())


# + [markdown] id="hEJ8154h5z7a"
# #### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="kkA0MmRN5ykS" outputId="1c0c70be-1fee-4c1c-f62b-e07f0b00c212"
def decoder(z):
  return m2.vae.decoder(z).reshape(-1, img_size, img_size)

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 5), 10), " c h w -> h w c").cpu().detach())


# + [markdown] id="1wUAaJJ6eZid"
# ## 2D Latent Embeddings For MNIST

# + [markdown] id="KRAFwDkj5-Ja"
# ### ConvVAE with latent dim 20

# + id="TQjFL165KiQH" colab={"base_uri": "https://localhost:8080/", "height": 466} outputId="c9647c5d-0cd4-4f1d-f001-f5f68ef0b547"
def encoder(img):
  return m.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

plot_scatter_plot(batch, encoder)

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="pZ90b_edszGA" outputId="e6c532ac-4f35-4710-dbea-31cbc1413972"
plot_grid_plot(batch, encoder)


# + [markdown] id="3PhqWNzF6I02"
# ### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 466} id="rWUBrS_p6Ocz" outputId="91afd339-6614-4b5f-de01-212cd7ef6220"
def encoder(img):
  return m2.vae.encode(rearrange(img, "b c h w -> b ( c h w)")).cpu().detach().numpy()

plot_scatter_plot(batch, encoder)

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="eMQsMYYy6Pt7" outputId="aae48d8c-eef5-48b4-9127-34c975c358e0"
plot_grid_plot(batch, encoder)


# + [markdown] id="A2YQo8RmCEu4"
# ## Interpolation

# + [markdown] id="KRv2QOnXb6Q9"
# ### Spherical Interpolation

# + [markdown] id="D97kQHXY6eWl"
# #### ConvVAE with latent dim 20

# + id="QFB4yZsEMbcx" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="d96e5ac3-0391-43c4-e34c-8bb7a97a4043"
def decoder(z):
  return m.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return m.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end)
plt.imshow(arr)


# + [markdown] id="QSEbSSXq6qBv"
# #### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 142} id="UGfo9uHX6sS1" outputId="57f0cf83-565e-4b87-d937-4cea02085c69"
def decoder(z):
  return m2.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return m2.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end)
plt.imshow(arr)


# + [markdown] id="beII6UWXb-i3"
# ### Linear Interpolation

# + [markdown] id="fRpl90Bh7MwO"
# #### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 142} id="Sq05GeiJMUoy" outputId="a60ad5d2-4e60-4ce7-d6d7-70e72e85abe0"
def decoder(z):
  return m.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return m.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)


# + [markdown] id="j7-4b6md7aLD"
# #### ConvVAE with latent dim 2

# + id="1tkvwhlicCmw" colab={"base_uri": "https://localhost:8080/", "height": 142} outputId="a3f09dad-1776-4a95-a2f8-c64d9d7aa62d"
def decoder(z):
  return m2.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return m2.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)

# + id="sUftpNth7ZdU"

