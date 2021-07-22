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

# + [markdown] id="aFEYSVllHRxk"
# #MLP MNIST AE

# + [markdown] id="SHuYUPxoHWqo"
# ## Installation and imports

# + colab={"base_uri": "https://localhost:8080/"} id="CCbi5XKXN5jk" outputId="8c75ae8f-2a14-4254-d8ed-163a20146069"
# !mkdir figures
# !mkdir scripts
# %cd /content/scripts
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/lvm_plots_utils.py
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/ae-mnist-mlp-latent-dim-2.ckpt
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/ae-mnist-mlp-latent-dim-20.ckpt
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/ae_mnist_mlp.py

# + id="nNKx7QEqPp_e"
# %%capture
# ! pip install --quiet torchvision pytorch-lightning torch test-tube einops umap

# + id="otacDQsbProK"
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
from ae_mnist_mlp import BasicAEModule

# + id="3hnm-OPMUH47"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + colab={"base_uri": "https://localhost:8080/", "height": 539, "referenced_widgets": ["33c18b6cd6e84e2ba2972f2368196421", "26e3919ac6514c6d8db0069ce00861a5", "e7cf6f60f9224e9298082b327cb2677c", "69e10bf92cde482e819aa47a450bfa14", "06c7d6d32076480e816606bd5d701d17", "7c55502f1b3d4f158cb8997dbba8f5ec", "7d10d87124f44349a74b089ea38818d8", "fbcc83717cd84b899aac8820937e1058", "b849f86d1385490383430be7212cce43", "40bebcedc160491886fc958cfe26e7d1", "47d08141236846d1a7b2ee79ed4f29a3", "cce7f51a4e4640a8b2cc1d134bdaca8e", "7b143dc742f846738e26b75bc6d4124d", "0ca241dfd8ce4313936582f7d985e97c", "ca16d85a198f4dfc9e1ddfe8668bd209", "116bfb61afb141dc8a108db0c467bd2e", "61926eb6325041be81cf662755480db0", "db6579962bec4156bef7de17bee3e637", "4aa24ab5ffbd439d9abcf2ee6101e8c1", "f91b12bdab274431bbcd4e1759b95b31", "3f213d11146b45ae9ec077243e9a8ace", "c1c74be0d6424d54b0909a29fd200c25", "d97d7611a45c4cb9bc900fb4bdd13d50", "cfaf8ee6348c4394b429ed40d403c67d", "7cb3c2b2c5fc4be8b4dda9ace49ac64c", "a816552fa0ce41e28bce1a9c5079973a", "60099b1bf20e4fbaad91c703d5930934", "0a76d1f4a6fe457aacac1907de54c140", "b16a6d4320fe44f5b9d8ac0889e48c33", "1ff3b3a4e76f4cacb3dabf18462ca659", "19fa4af1e1164b5691f463a83cfbc720", "0d5b76c497734b58a2fdc576d2dcb2bb"]} id="OpxjxGAsRWhD" outputId="b184abd3-b90b-45c3-c6d4-4a6265d63465"
mnist_full = MNIST(".", download=True, train=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

dm = DataLoader(mnist_full, batch_size=512, shuffle=True)

# + [markdown] id="1ZExmDf6Hd3J"
# ## Basic MLP AE Module

# + colab={"base_uri": "https://localhost:8080/"} id="YYPsbVe5bo9u" outputId="86161777-f0d7-467e-b4e2-ec0a1459ccca"
ae = BasicAEModule(20)
ae.load_state_dict(torch.load("ae-mnist-mlp-latent-dim-20.ckpt"))
ae.to(device)

# + colab={"base_uri": "https://localhost:8080/"} id="dqFuy2QcboSs" outputId="6beb89c9-cb55-4e33-cc89-365caa5797ef"
ae2 = BasicAEModule(2)
ae2.load_state_dict(torch.load("ae-mnist-mlp-latent-dim-2.ckpt"))
ae2.to(device)

# + [markdown] id="wGYCFPJFHlG2"
# ## Reconstruction

# + [markdown] id="HErAcsBnH7NF"
# ### MLP-AE with latent dim 20

# + id="Y0ZtYOGER3dL"
batch = next(iter(dm))
imgs, _ = batch
imgs = imgs[:16]
img_size = 28


# + colab={"base_uri": "https://localhost:8080/", "height": 273} id="9nGKN2p_TRm-" outputId="2bed1f56-bce6-4d06-b334-57f31bf3fe97"
def reconstruct(img):
  return ae(rearrange(imgs, "b c h w -> b ( c h w)")).reshape(-1, 1, img_size, img_size)
  
fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs).cpu(), 'c h w -> h w c'))
imgs = imgs.to(device=device)
ae = ae.to(device)
print(reconstruct(imgs).shape)
axs[1].imshow(rearrange(make_grid(reconstruct(imgs)).cpu(), 'c h w -> h w c'))
plt.show()

# + [markdown] id="6HmyBXXRH_CV"
# ### MLP-AE with latent dim 2

# + id="VeU_gMQTIIuE"
batch = next(iter(dm))
imgs, _ = batch
imgs = imgs[:16]
img_size = 28


# + colab={"base_uri": "https://localhost:8080/", "height": 273} id="x_OK2TBDICCM" outputId="7c57e2ec-1ebb-4062-9567-6577bca5651b"
def reconstruct(img):
  return ae2(rearrange(imgs, "b c h w -> b ( c h w)")).reshape(-1, 1, img_size, img_size)
  
fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs).cpu(), 'c h w -> h w c'))
imgs = imgs.to(device=device)
ae2 = ae2.to(device)
print(reconstruct(imgs).shape)
axs[1].imshow(rearrange(make_grid(reconstruct(imgs)).cpu(), 'c h w -> h w c'))
plt.show()


# + [markdown] id="v8bPJ8r2IQfL"
# ## Sampling

# + [markdown] id="r-AGhn1MIVps"
# ### Random samples from truncated normal distribution 

# + [markdown] id="mMnyq3JhIXhR"
# We sample $z \sim TN(0,1)$ form a truncated normal distribution with a threshold = 5
#

# + [markdown] id="GeJFwS5qIZNy"
# #### MLP-AE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 611} id="NyOkD8D0TTIn" outputId="8d1af76f-b329-47ff-dbd8-feeb86949185"
def decoder(z):
  return ae.vae.decoder(z).reshape(-1, 1, img_size, img_size)

plt.figure(figsize=(10,10))
# Where 5 is the truncation threshold for our truncated normal distribution we are sampling from
imgs= get_random_samples(decoder, 5)
plt.imshow(imgs)


# + [markdown] id="Vt2aYDs9IesY"
# #### MLP-AE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="2pcPSq34WzBD" outputId="e243b640-450c-4d65-8222-20b9cf3f4b4c"
def decoder(z):
  return ae.vae.decoder(z).reshape(-1, img_size, img_size)

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 10), 10), " c h w -> h w c").cpu().detach())


# + [markdown] id="BbYAuzJTKKSO"
# ### Grid Sampling

# + [markdown] id="dlwtGzPJKP82"
# #### MLP-AE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="kNJNy_vxKMtD" outputId="c5a9e278-f88f-4dca-f0d1-20cc06bc143c"
def decoder(z):
  return m.vae.decoder(z).reshape(-1, img_size, img_size)

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, latent_size=20), 10), " c h w -> h w c").cpu().detach())


# + [markdown] id="y8efTo7NKTKj"
# #### MLP-AE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="fOm5BBepKWuD" outputId="d4e3129f-c98e-4455-85e4-7ff4118465e6"
def decoder(z):
  return ae2.vae.decoder(z).reshape(-1, img_size, img_size)

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 5), 10), " c h w -> h w c").cpu().detach())


# + [markdown] id="jvc7U9jTIgeH"
# ## 2D Latent Embeddings For MNIST

# + [markdown] id="8ch3YJCnIkvu"
# ### MLP-AE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 466} id="-8MtXagVW5Tj" outputId="e37fc72a-732e-4489-b230-b955f0775585"
def encoder(img):
  return ae.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

plot_scatter_plot(batch, encoder)

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="aAdUqIRIaUDW" outputId="59f53115-fcaf-4168-d1a3-f75bde3cee99"
plot_grid_plot(batch, encoder)


# + [markdown] id="zSlbGQ4OIoE_"
# ### MLP-AE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 466} id="yxItYMaZIsEv" outputId="5fd30da8-4f04-4584-f91c-7c003aa68b75"
def encoder(img):
  return ae2.vae.encode(rearrange(img, "b c h w -> b ( c h w)")).cpu().detach().numpy()

plot_scatter_plot(batch, encoder)

# + colab={"base_uri": "https://localhost:8080/", "height": 499} id="QXMHyVVdJ2r_" outputId="3443eeee-8e52-472d-f84c-b0c71cf554b1"
plot_grid_plot(batch, encoder)


# + [markdown] id="ZxiCxDRrKdm7"
# ## Interpolation

# + [markdown] id="_FaqEmpMKiUJ"
# ### Spherical Interpolation

# + [markdown] id="aKGiO7cNKkud"
# #### MLP-AE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 142} id="jaoJ-H7YKgUO" outputId="ff70b97d-f13f-4459-82de-66b76cbc2f8c"
def decoder(z):
  return ae.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return ae.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end)
plt.imshow(arr)


# + [markdown] id="N9QxNRhYK2Ws"
# #### MLP-AE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 142} id="dEnQw_5gaa98" outputId="06450bf9-6e94-49f0-ce41-ae28d610162a"
def decoder(z):
  return ae2.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return ae2.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end)
plt.imshow(arr)


# + [markdown] id="FMw1401BK8y6"
# ### Linear Interpolation

# + [markdown] id="oHt9xrr1Mjec"
# #### MLP-AE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 142} id="HhQrlnYRKpT9" outputId="18f62616-8773-4e42-8838-6928f96eccd7"
def decoder(z):
  return ae.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return ae.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)


# + [markdown] id="j4m50RsOMmnZ"
# #### MLP-AE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 142} id="_OGR9za9LIJu" outputId="770b3d75-cbc5-491c-a997-0c4c311b1ad3"
def decoder(z):
  return ae2.vae.decoder(z).reshape(-1, img_size, img_size)

def encoder(img):
  return ae2.vae.encode(rearrange(img, "b c h w -> b ( c h w)"))

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[1], z_imgs[3]

plt.figure(figsize=(10,100))
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)

# + id="9lwTEtc5LOCP"

