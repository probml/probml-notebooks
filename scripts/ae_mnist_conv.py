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
# #Conv MNIST AE

# + [markdown] id="SHuYUPxoHWqo"
# ## Installation and imports

# + colab={"base_uri": "https://localhost:8080/"} id="CCbi5XKXN5jk" outputId="ad58b5e2-99af-454c-f8b8-5faa7be0706b"
# !mkdir figures
# !mkdir scripts
# %cd /content/scripts
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/lvm_plots_utils.py
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/ae-mnist-conv-latent-dim-2.ckpt
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/ae-mnist-conv-latent-dim-20.ckpt
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/ae_mnist_conv.py

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
from ae_mnist_conv import ConvAE

# + id="3hnm-OPMUH47"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + id="OpxjxGAsRWhD" outputId="07655678-d7e2-4692-9382-ddf8769f7c6d" colab={"base_uri": "https://localhost:8080/", "height": 555, "referenced_widgets": ["8cd962a2b595489d8183b4dcf6a544bd", "1384ef0d117d4cfc8ac5e0f1c98950b6", "ab5ff0d7de2c4afd90132b45851d0544", "66ee20e11acb46e1a9509f7420022ec9", "c251c968f1d44250b47ca15ba8ce3dc4", "e2e7d25e2b5147818a4852eed25f3fa1", "d9386842913d40ba8ae74e6fc086d053", "a720c94998bd4222bdbc5ec155a95d8f", "408ba761d9174a28b0096b2fea30c2a8", "3fc358fba71f4367a1ace4d4287296a5", "9e16c95cbffd4c2b9ba0d4107281e39a", "19afca0875f44af9ba886e6fca023d32", "1b5230780b0040298c65787179553ff3", "636192ca240542c0b1d78e8efc26d961", "8da2b42025e0480ca5e7fd1802aa9e4e", "a4bc4bcaa13e4f36b4c38d1f521bbdda", "2b8d82d445224b1c96d3371c7153f929", "68d65c0110f847d58ef07d28bf7984ea", "e04bd225cac540b5b72fecdb5459f0e7", "13d2741b241a4e61b8b9647108592aac", "e51ec989f73742ac8f06d60eaf1cd70d", "cae5030e86a1432a9e018adc933b354b", "b84c9a7e79ff4afea6ff2eb0e2d4ee7a", "7715694838f24480ac3a1881a7e1e4da", "9e02cdf3634b4977a49c66df08475857", "f5b5055d43154d95b5b3816f3de29a39", "1d6c13e08e7347bda8b6c2626726c786", "56c9ef57d7f44248be35e3294175df88", "62d65b49825b4cc4936afabc02a5e034", "a002c8124109479ba8d887fb49961551", "6cd929f346d440c485d70b013019f4ca", "eafa732a3b314744a21351cb3f5d1ec9"]}
mnist_full = MNIST(".", download=True, train=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

dm = DataLoader(mnist_full, batch_size=512, shuffle=True)

vis_data = DataLoader(mnist_full, batch_size=5000)
batch = next(iter(vis_data))

# + [markdown] id="1ZExmDf6Hd3J"
# ## Basic Conv VAE Module

# + colab={"base_uri": "https://localhost:8080/"} id="YYPsbVe5bo9u" outputId="7990d6e1-b40a-495c-bc74-d5d9d3f40c1f"
ae = ConvAE((1, 28, 28), 
                encoder_conv_filters=[28,64,64],
                decoder_conv_t_filters=[64,28,1],
                latent_dim=20)
ae.load_state_dict(torch.load("ae-mnist-conv-latent-dim-20.ckpt"))
ae.to(device)

# + colab={"base_uri": "https://localhost:8080/"} id="dqFuy2QcboSs" outputId="ddc7ed0d-c36a-488e-c0f0-d5946b68e131"
ae2 = ConvAE((1, 28, 28), 
                encoder_conv_filters=[28,64,64],
                decoder_conv_t_filters=[64,28,1],
                latent_dim=2)
ae2.load_state_dict(torch.load("ae-mnist-conv-latent-dim-2.ckpt"))
ae2.to(device)

# + [markdown] id="wGYCFPJFHlG2"
# ## Reconstruction

# + [markdown] id="HErAcsBnH7NF"
# ### ConvAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 254} id="9nGKN2p_TRm-" outputId="0870ae2f-f91b-448d-8725-773f513559da"
imgs, _ = batch
imgs = imgs[:16]

fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs), 'c h w -> h w c'))
imgs = imgs.to(device=device)
axs[1].imshow(rearrange(make_grid(ae.vae(imgs)[0].cpu()), 'c h w -> h w c'))
plt.savefig('../figures/ae_mnist_conv_20d_rec.pdf')
plt.show()

# + [markdown] id="6HmyBXXRH_CV"
# ### ConvAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 254} id="x_OK2TBDICCM" outputId="46b10f0d-1287-4c84-88dc-60f2914e698c"
imgs, _ = batch
imgs = imgs[:16]

fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs), 'c h w -> h w c'))
imgs = imgs.to(device=device)
axs[1].imshow(rearrange(make_grid(ae2.vae(imgs)[0].cpu()), 'c h w -> h w c'))
plt.savefig('../figures/ae_mnist_conv_2d_rec.pdf')
plt.show()


# + [markdown] id="v8bPJ8r2IQfL"
# ## Sampling

# + [markdown] id="r-AGhn1MIVps"
# ### Random samples from truncated normal distribution 

# + [markdown] id="mMnyq3JhIXhR"
# We sample $z \sim TN(0,1)$ form a truncated normal distribution with a threshold = 5
#

# + [markdown] id="GeJFwS5qIZNy"
# #### ConvAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 136} id="NyOkD8D0TTIn" outputId="23787aff-5c35-4060-9acf-9e069a242c56"
def decoder(z):
  return ae.vae.decode(z)

plt.figure()
imgs= get_random_samples(decoder, truncation_threshold=5, num_images_per_row=8, num_images=16)
plt.imshow(imgs)
plt.savefig('../figures/ae_mnist_conv_20d_samples.pdf')


# + [markdown] id="Vt2aYDs9IesY"
# #### ConvAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 136} id="2pcPSq34WzBD" outputId="b6ef2a83-8cc8-4d63-9e6f-aa2444ba6cda"
def decoder(z):
  return ae2.vae.decode(z)

plt.figure()
imgs= get_random_samples(decoder, truncation_threshold=5, num_images_per_row=8, num_images=16)
plt.imshow(imgs)
plt.savefig('../figures/ae_mnist_conv_2d_samples.pdf')


# + [markdown] id="BbYAuzJTKKSO"
# ### Grid Sampling

# + [markdown] id="dlwtGzPJKP82"
# #### ConvAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="kNJNy_vxKMtD" outputId="81b0d60f-1901-4c83-83f7-7e51df96f981"
def decoder(z):
  return ae.vae.decode(z)[0]

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 20), 10), " c h w -> h w c").cpu())


# + [markdown] id="y8efTo7NKTKj"
# #### ConvAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 612} id="fOm5BBepKWuD" outputId="37af1593-6ab0-4f0a-aeab-153873312b7d"
def decoder(z):
  return ae2.vae.decode(z)[0]

plt.figure(figsize=(10,10))
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 20), 10), " c h w -> h w c").cpu())


# + [markdown] id="jvc7U9jTIgeH"
# ## 2D Color embedding of latent space

# + [markdown] id="8ch3YJCnIkvu"
# ### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 466} id="-8MtXagVW5Tj" outputId="b5569ccb-7f86-4e6f-dc88-f563569eaae7"
def encoder(img):
  return ae.vae.encode(img)[0]

def decoder(z):
  z = z.to(device)
  return rearrange(ae.vae.decode(z), "b c h w -> b (c h) w")

plot_scatter_plot(batch, encoder)

# + colab={"base_uri": "https://localhost:8080/", "height": 498} id="aAdUqIRIaUDW" outputId="6cda8b85-9b6d-41de-b883-d03384fc32f6"


def encoder(img):
  return ae.vae.encode(img)[0]

def decoder(z):
  z = z.to(device)
  return rearrange(ae.vae.decode(z), "b c h w -> b (c h) w")
  
fig=plot_grid_plot(batch, encoder)
fig.savefig('../figures/ae_mnist_conv_20d_embed.pdf')
plt.show()


# + [markdown] id="zSlbGQ4OIoE_"
# ### ConvAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 466} id="yxItYMaZIsEv" outputId="8a5a6b4c-46e1-4793-9158-737f96ac3c2e"
def encoder(img):
  return ae2.vae.encode(img)[0].cpu().detach().numpy()

def decoder(z):
  z = z.to(device)
  return rearrange(ae2.vae.decode(z), "b c h w -> b (c h) w")

plot_scatter_plot(batch, encoder)


# + colab={"base_uri": "https://localhost:8080/", "height": 498} id="QXMHyVVdJ2r_" outputId="176cf1ef-80b5-41d1-d524-c27394558f77"
def encoder(img):
  return ae2.vae.encode(img)[0].cpu().detach().numpy()

def decoder(z):
  z = z.to(device)
  return rearrange(ae2.vae.decode(z), "b c h w -> b (c h) w")

fig=plot_grid_plot(batch, encoder)
fig.savefig('../figures/ae_mnist_conv_2d_embed.pdf')
plt.show()



# + [markdown] id="ZxiCxDRrKdm7"
# ## Interpolation

# + [markdown] id="_FaqEmpMKiUJ"
# ### Spherical Interpolation

# + [markdown] id="aKGiO7cNKkud"
# #### ConvAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 112} id="jaoJ-H7YKgUO" outputId="9cb3703a-d156-44fe-c116-356daa296d44"
def decoder(z):
  z = z.to(device)
  return rearrange(ae.vae.decode(z), "b c h w -> b (c h) w")

def encoder(img):
  return ae.vae.encode(img)[0].cpu().detach()

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
#end, start = z_imgs[1], z_imgs[3]
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end, interpolation="spherical")
plt.imshow(arr)


# + [markdown] id="N9QxNRhYK2Ws"
# #### ConAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 112} id="dEnQw_5gaa98" outputId="c0b06a2c-3317-4d8c-d075-a7d60d945457"
def decoder(z):
  z = z.to(device)
  return rearrange(ae2.vae.decode(z), "b c h w -> b (c h) w")

def decoder(z):
  z = z.to(device)
  return rearrange(ae2.vae.decode(z), "b c h w -> b (c h) w")

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
#end, start = z_imgs[1], z_imgs[3]
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end, interpolation="spherical")
plt.imshow(arr)


# + [markdown] id="FMw1401BK8y6"
# ### Linear Interpolation

# + [markdown] id="oHt9xrr1Mjec"
# #### ConvAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 94} id="HhQrlnYRKpT9" outputId="36e4212a-c473-4565-a4fc-a02303082081"
def decoder(z):
  z = z.to(device)
  return rearrange(ae.vae.decode(z), "b c h w -> b (c h) w")

def decoder(z):
  z = z.to(device)
  return rearrange(ae.vae.decode(z), "b c h w -> b (c h) w")

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
#end, start = z_imgs[1], z_imgs[3]
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)
plt.savefig('../figures/ae_mnist_conv_20d_linear.pdf')


# + [markdown] id="j4m50RsOMmnZ"
# #### ConvAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 94} id="_OGR9za9LIJu" outputId="9275227f-e0b9-4a89-896f-a98dc6e7c278"
def decoder(z):
  z = z.to(device)
  return rearrange(ae2.vae.decode(z), "b c h w -> b (c h) w")

def decoder(z):
  z = z.to(device)
  return rearrange(ae2.vae.decode(z), "b c h w -> b (c h) w")

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
#end, start = z_imgs[1], z_imgs[3]
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)
plt.savefig('../figures/ae_mnist_conv_2d_linear.pdf')

# + id="9lwTEtc5LOCP"

