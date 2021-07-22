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

# + [markdown] id="gqxJ7jnfSgx8"
# # Convolutional MNIST VAE

# + [markdown] id="U91_5uDrBEIc"
# ## Installation 

# + colab={"base_uri": "https://localhost:8080/"} id="W5lIfMXNXOgU" outputId="fdd7757e-fdd1-409a-8a6f-bd2e7d926886"
# !mkdir figures
# !mkdir scripts
# %cd /content/scripts
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/pyprobml_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/lvm_plots_utils.py
# !wget -q https://raw.githubusercontent.com/probml/pyprobml/master/scripts/vae_conv_mnist.py
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/vae-mnist-conv-latent-dim-2.ckpt
# !wget -q https://github.com/probml/probml-data/raw/main/checkpoints/vae-mnist-conv-latent-dim-20.ckpt


# + id="FEQ6E1rI4u7Z"
# %%capture
# ! pip install --quiet torchvision pytorch-lightning torch test-tube einops umap

# + id="lAO-kHpr4yve"
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
from lvm_plots_utils import get_random_samples, get_grid_samples, plot_scatter_plot, get_imrange, plot_grid_plot, plot_scatter_plot
import seaborn as sns
from torchvision.utils import make_grid
from vae_conv_mnist import ConvVAE

# + id="ks3SMeD8eaRU"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + id="fm7-NgBgbNir" outputId="1baf866d-907c-46cb-a043-b56fde67b156" colab={"base_uri": "https://localhost:8080/", "height": 840, "referenced_widgets": ["ee9a145b26b446fb8b7750dd319d8ab7", "31b52281d3f14370a4ef03c313aa78ab", "75e5c2c4af9240e59d6b08d292f0fbc2", "953251629ae34885bf7b8c9825ecf9d5", "5340956cd6b942d2a6716e6ac706e9ac", "1b332263bb8e47e695da741e7da638b3", "466a6885dfea41a1a039fe7546cedae3", "eb0055a420af42efa0dc8948852094ea", "46b79383523848648f589d31892c40b1", "74cc277d7cbf4803a65dc15cae190d8a", "c966bfac73d646c6a4f702fb87889571", "aed163e07e804b47bc50103f05de48f1", "b9d04ca21e1145c19b542efed65e8bb1", "d65e0b5311ff41b18debe3706368e1c8", "39982abda2694699b92f884fbc5c544b", "5b75469ad7504b13b8ddb4bbf42ca7d3", "172971207b294c8e8e919518b875c881", "ad59c28e9b1a4590bf9140f727681d86", "e6a349c9f1834765b8cf6205f59c5819", "040c46d4cfd84fc1bac4988c20ddd9e1", "4e3daf23260f4bbea2bd86be7abe2739", "a901db8ef69b4b05b3fda6b7097839a8", "136fdd91aa4f46a2a91e3c719991c89c", "72237f8de57a473ba57b5e3dde529886", "7b5e3c1eecd245f99d44b6fa50bbe79b", "94de2d8ce3fe4b57bda2ec066fcd638a", "d608fd9dafb34be6b52bc7cf438fe7da", "90fd288b134c4d8097d2b15671e0c75a", "2a6e55aec70d40ce97ae4e272b590e6f", "4792f3da7d9c44a48ee88138b9caf182", "940028b22fe64b24a22f698608154f1d", "b59920eeb3cb47ee82e6b6adcf7d2f02"]}
mnist_full = MNIST(".", train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
dm = DataLoader(mnist_full, batch_size=250)
vis_data = DataLoader(mnist_full, batch_size=5000)
batch = next(iter(vis_data))

# + [markdown] id="4i8qLD4MBRUM"
# ## VAE

# + id="I2BF4OlK4r9P"
m = ConvVAE((1, 28, 28), encoder_conv_filters=[28,64,64], decoder_conv_t_filters=[64,28,1], latent_dim=20, kl_coeff=5)
m2 = ConvVAE((1, 28, 28), encoder_conv_filters=[28,64,64], decoder_conv_t_filters=[64,28,1], latent_dim=2, kl_coeff=5)

# + id="NT-4R9La56Ym" colab={"base_uri": "https://localhost:8080/"} outputId="81f6667c-f8fc-4d2b-e32a-80f78bd42b21"

m.load_state_dict(torch.load("vae-mnist-conv-latent-dim-20.ckpt"))
m2.load_state_dict(torch.load("vae-mnist-conv-latent-dim-2.ckpt"))

# + colab={"base_uri": "https://localhost:8080/"} id="A9nSoqrh8Lcu" outputId="d6256a07-35b4-4c1d-c6e6-76ccdc25e2d5"
m.eval()
m.to(device)
m2.eval()
m2.to(device)

# + [markdown] id="Ieic-EZFt7PW"
# ## Reconstruction

# + [markdown] id="uWFd0wCOurNF"
# ### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 254} id="L5kTXWrK7B_m" outputId="1442ff8f-ee64-4242-a0db-156f4212fb6a"
imgs, _ = batch
imgs = imgs[:16]

fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs), 'c h w -> h w c'))
imgs = imgs.to(device=device)
axs[1].imshow(rearrange(make_grid(m.vae(imgs)[0].cpu()), 'c h w -> h w c'))
plt.savefig('../figures/vae_mnist_conv_20d_rec.pdf')
plt.show()

# + id="cWdvhKDG_f3S" outputId="c927132a-4ada-4733-cd14-d502ab82785b" colab={"base_uri": "https://localhost:8080/"}
# !ls

# + [markdown] id="Daokv0sAuwdX"
# ### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 254} id="XNW5X2zruz9i" outputId="50f58e73-bd99-4842-cf79-2807f625400b"
imgs, _ = batch
imgs = imgs[:16]

fig, axs=plt.subplots(2, 1)
axs[0].imshow(rearrange(make_grid(imgs), 'c h w -> h w c'))
imgs = imgs.to(device=device)
axs[1].imshow(rearrange(make_grid(m2.vae(imgs)[0].cpu()), 'c h w -> h w c'))
plt.savefig('../figures/vae_mnist_conv_2d_rec.pdf')
plt.show()


# + [markdown] id="RPE_DKli4EUj"
# ## Sampling

# + [markdown] id="Ui23OhsIQjOM"
# ### Random samples form truncated unit normal distribution

# + [markdown] id="YmRGfHCNyw2P"
# We sample $z \sim TN(0,1)$ form a truncated normal distribution with a threshold = 5

# + [markdown] id="svmkj7Hzydka"
# #### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 136} id="xtH4QiGj00uS" outputId="cbddc46a-e213-427b-ab12-59e2d6beb11a"
def decoder(z):
  return m.vae.decode(z)

plt.figure()
#imgs= get_random_samples(decoder, truncation_threshold=5)
imgs= get_random_samples(decoder, truncation_threshold=5, num_images_per_row=8, num_images=16)
plt.imshow(imgs)
plt.savefig('../figures/vae_mnist_conv_20d_samples.pdf')


# + [markdown] id="3A7BME_yyhvQ"
# #### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 136} id="xWC1UyAkykZW" outputId="4c801644-4b2c-4edd-a71b-1807ee22dc43"
def decoder(z):
  return m2.vae.decode(z)

plt.figure()
imgs= get_random_samples(decoder, truncation_threshold=5, num_images_per_row=8, num_images=16)
plt.imshow(imgs)
plt.savefig('../figures/vae_mnist_conv_2d_samples.pdf')


# + [markdown] id="JgVH36TYUFDM"
# ### Grid Sampling

# + [markdown] id="dxR2o0_0yejF"
# We let $z = [z1, z2, 0, \ldots, 0]$ and vary $z1, z2$ on a grid

# + [markdown] id="BNaHFXbIyVKx"
# #### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 247} id="7R5HDlRkHPpL" outputId="64ecbbf3-e7c0-499d-bbce-66282f02853f"
def decoder(z):
  return m.vae.decode(z)[0]

#plt.figure(figsize=(10,10))
plt.figure()
#plt.imshow(rearrange(make_grid(get_grid_samples(decoder, 20), 10), " c h w -> h w c").cpu())
nimgs = 8
nlatents = 20
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, nlatents, nimgs), nimgs), " c h w -> h w c").cpu())
plt.axis('off')
plt.tight_layout()
plt.savefig('../figures/vae_mnist_conv_20d_grid.pdf')


# + [markdown] id="br-eb7-zyY32"
# #### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="u2sypjsoyZ0I" outputId="5df1c651-9bd1-4413-dcf9-00f722918ff7"
def decoder(z):
  return m2.vae.decode(z)[0]

plt.figure()
nimgs = 8
nlatents = 2
plt.imshow(rearrange(make_grid(get_grid_samples(decoder, nlatents, nimgs), nimgs), " c h w -> h w c").cpu())
plt.axis('off')
plt.tight_layout()
plt.savefig('../figures/vae_mnist_conv_2d_grid.pdf')



# + [markdown] id="a5qWxtr8mB3p"
# ## 2D Color embedding of latent space

# + [markdown] id="QdpFhgNYvdyF"
# ### ConvVAE with latent dim 20

# + colab={"base_uri": "https://localhost:8080/", "height": 465} id="MxegEF4ut76u" outputId="6d8704ae-adde-49ad-de2d-da295ced04c2"
def encoder(img):
  return m.vae.encode(img)[0]

def decoder(z):
  z = z.to(device)
  return rearrange(m.vae.decode(z), "b c h w -> b (c h) w")

plot_scatter_plot(batch, encoder)


# + colab={"base_uri": "https://localhost:8080/", "height": 498} id="AZXgw2zpYkde" outputId="5d33b963-48e8-46ee-a4e8-1e8a682bed27"
def encoder(img):
  return m.vae.encode(img)[0]

def decoder(z):
  z = z.to(device)
  return rearrange(m.vae.decode(z), "b c h w -> b (c h) w")
  
fig=plot_grid_plot(batch, encoder)
fig.savefig('../figures/vae_mnist_conv_20d_embed.pdf')
plt.show()


# + [markdown] id="Hgm7KsmuvjLK"
# ### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 465} id="L-llCOnRvl-0" outputId="787a5610-de2c-4056-a76f-448cfd769eb6"
def encoder(img):
  return m2.vae.encode(img)[0].cpu().detach().numpy()

def decoder(z):
  z = z.to(device)
  return rearrange(m2.vae.decode(z), "b c h w -> b (c h) w")

plot_scatter_plot(batch, encoder)

# + colab={"base_uri": "https://localhost:8080/", "height": 498} id="8DmUW5A5vohT" outputId="b97f3112-52a0-4d36-b27f-9ad4500cf27b"
fig=plot_grid_plot(batch, encoder)
fig.savefig('../figures/vae_mnist_conv_2d_embed.pdf')


# + [markdown] id="SQ5tWklrmGn-"
# ## Interpolation 

# + [markdown] id="usHbO3SDT9kl"
# ### Spherical Interpolation

# + [markdown] id="V16KhZktysMN"
# #### ConvVAE with latent dim 20

# + id="VoRV63q0mLNZ" colab={"base_uri": "https://localhost:8080/", "height": 94} outputId="0a95f3e0-3686-4ccd-82f8-207edf049980"
def decoder(z):
  z = z.to(device)
  return rearrange(m.vae.decode(z), "b c h w -> b (c h) w")

def encoder(img):
  return m.vae.encode(img)[0].cpu().detach()

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
#end, start = z_imgs[1], z_imgs[3]
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end,interpolation="spherical")
plt.imshow(arr)
plt.savefig('../figures/vae_mnist_conv_20d_spherical.pdf')


# + [markdown] id="CC5jk5rDy1OC"
# #### ConvVAE with latent dim 2

# + colab={"base_uri": "https://localhost:8080/", "height": 94} id="T2dWDsAfy3t_" outputId="8dc3ddeb-e5de-4fbb-bd34-7de5bdce15b6"
def decoder(z):
  z = z.to(device)
  return rearrange(m2.vae.decode(z), "b c h w -> b (c h) w")

def encoder(img):
  return m2.vae.encode(img)[0].cpu().detach()

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end,interpolation="spherical")
plt.imshow(arr)
plt.savefig('../figures/vae_mnist_conv_2d_spherical.pdf')


# + [markdown] id="Z2fCOS1aUE-3"
# ### Linear Interpolation

# + [markdown] id="9yGyaXClzyF8"
# #### ConvVAE with latent dim 20

# + id="0wQae1S91RTA" colab={"base_uri": "https://localhost:8080/", "height": 103} outputId="b4a11f04-8413-416a-eb5e-ea0c12fbe0cc"
def decoder(z):
  z = z.to(device)
  return rearrange(m.vae.decode(z), "b c h w -> b (c h) w")


def encoder(img):
  return m.vae.encode(img)[0].cpu().detach()

imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)
plt.tight_layout()
plt.savefig('../figures/vae_mnist_conv_20d_linear.pdf')


# + [markdown] id="Qerqa3H7z0vC"
# #### ConvVAE with latent dim 2

# + id="nK3iDTwG_ORp" colab={"base_uri": "https://localhost:8080/", "height": 103} outputId="b6f98cf1-aaf2-4340-f508-090a61aa9704"
def decoder(z):
  z = z.to(device)
  return rearrange(m2.vae.decode(z), "b c h w -> b (c h) w")


def encoder(img):
  return m2.vae.encode(img)[0].cpu().detach()



imgs, _ = batch
imgs = imgs.to(device)
z_imgs = encoder(imgs)
#end, start = z_imgs[1], z_imgs[3]
end, start = z_imgs[0], z_imgs[5]

plt.figure()
arr = get_imrange(decoder,start,end, interpolation="linear")
plt.imshow(arr)
plt.tight_layout()
plt.savefig('../figures/vae_mnist_conv_2d_linear.pdf')

# + id="hcdQXNo1z4Ym"

