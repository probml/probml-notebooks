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

# + [markdown] id="ffekTqaMzOkw"
# ## MMD-VAE, to create a more disentangled latent space
#
# One way of making a more disentangled latent space besides beta-vae is to modify the KL term in the ELBO and replace it with a maximum mean discrepancy (MMD) term that always prefers maximising the mutual information in the latent code, resulting in a more disentangled latent space, as shown by [Zhao et al. 17](https://arxiv.org/pdf/1706.02262.pdf). This can be seen from the following image below comparing the latent space of a VAE trained on MNIST using the KL term vs the MMD term, which is more structured than the KL term.

# + id="9qTYcyXR0c2f"
#@title Setup and installation { display-mode: "form" }
# %%capture 
# !sudo apt-get install subversion
# !svn checkout https://github.com/probml/pyprobml/trunk/vae .
# !pip install pytorch-lightning einops umap-learn

import umap
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.manifold import TSNE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from models.guassian_vae import VAE
from models.vanilla_vae import kl_divergence
from models.mmd_vae import MMD
from experiment import VAEModule, VAE2stageModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embedder(encoder, X_data, y_data=None, use_embedder="TSNE"):
  X_data_2D = encoder(X_data)
  if X_data_2D.shape[-1] == 2:
    return X_data_2D
  if use_embedder=="UMAP":
    umap_fn = umap.UMAP()
    X_data_2D = umap_fn.fit_transform(X_data_2D, y_data)
  elif use_embedder=="TSNE":
    tsne = TSNE()
    X_data_2D = tsne.fit_transform(X_data_2D)
  return X_data_2D

def plot_scatter_plot(batch, vae, use_embedder="TSNE", min_distance =0.01):
  """
  Plots scatter plot of embeddings
  """
  def encoder(img):
    return vae.det_encode(img).cpu().detach().numpy()

  model_name = vae.model_name
  X_data, y_data = batch
  X_data = X_data.to(device)
  np.random.seed(42)
  X_data_2D = get_embedder(encoder, X_data, y_data, use_embedder)
  X_data_2D = (X_data_2D - X_data_2D.min()) / (X_data_2D.max() - X_data_2D.min())

  # adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
  fig = plt.figure(figsize=(10, 8))
  cmap = plt.cm.tab10
  plt.scatter(X_data_2D[:, 0], X_data_2D[:, 1], c=y_data, s=10, cmap=cmap)
  image_positions = np.array([[1., 1.]])
  plt.title(f"Latent space of {model_name}")
  for index, position in enumerate(X_data_2D):
      dist = np.sum((position - image_positions) ** 2, axis=1)
      if np.min(dist) > 0.04: # if far enough from other images
          image_positions = np.r_[image_positions, [position]]
          imagebox = matplotlib.offsetbox.AnnotationBbox(
              matplotlib.offsetbox.OffsetImage(X_data[index].reshape(28, 28).cpu(), cmap="binary"),
              position, bboxprops={"edgecolor": tuple(cmap([y_data[index]])[0]), "lw": 2})
          plt.gca().add_artist(imagebox)
  plt.axis("off")
  return fig


# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["d13361ab35614624aff4a90668643b12", "59e8160b8bce45ca980f13fe5fd3520e", "3632533d27fb44e29ef03731df5ee347", "857619e6df5d4e228e58edec5f898dbc", "b54818635e4f436f99153f61d8336e3a", "71632af19efb4c7981b48ce9d5a5b133", "8b2dd3fb859648b082c60dc3a23306fe", "1d37a08a75a340e3b379ea9e3d323692", "2622efad355848dbb0f09b5bd0d61961", "84fc1551dc1e4b4c924ab77ae73cea6c", "5293c910801b4f4aa36199055955b154", "3c30c1749efd4822bb36a18d17df2020", "e67a7703c313499195c7359821bcb9b7", "1724688668fa4de1aa26651b5398247e", "2326613dcbc7448a9c0b351baa0aba9e", "ddf09fe0e44a410f87dc3e49cdc7e898", "eb47c848921f410d9fb0fdde91771210", "92522c61cf104852af7726d57aaab341", "77a4e66c52c24ecdbb9b81f7f8eeb0df", "6ed9e95154754cd5a96e70ac3159ca5b", "40246d109f9f442180a9651b48e51b9d", "14c69707f349489687f2ecfd15fd30e6", "5e6fc538ee1e4618ae096df29fa349fd", "db79bea03b414ef29ead677f13b293fd", "d9705062721e4d2b9fa15683f39fa951", "e7a4e849c61e4abea78132f0936a0f89", "30ba6d6fe3f542759ccc0568ebc4000f", "65cd9c7627454582bf213f6c68417f0a", "3d1c8a79a1ee4bc18edc0ae03a02d4e8", "9819882c1d9e473b8189a918d6df12b5", "eed58fd1779d4e7884376e7bcb378e24", "e1fa45644eac44d38fc578c0b8f19375", "3c82e39b52aa472fa163bf199532ff6d", "2c37289f595a42278b2fe75208619de4", "c05b0337d15449d697845d0b456841e4", "4a72f9c2e166454ebd42d511d0b6e79e", "37da294ce3c7492697dc00bdc6860c8d", "951f3105e56449ca82a2197d41be9399", "6677f21ea0594e5db7b6814cc5807a22", "be9c7fcc6b3544c1a9d5457fb79e5c89", "95096b6c64984bd583c3ab3be93a87f4", "f0c043b283bf463f97d42e917848dfa6", "609ab3162d65436a8dee6787d4ad1ac5", "cc4e9a8fe45c4e80a45f9664127568fe"]} id="8kZGT1OhuPL3" outputId="a2ff5526-c840-4d38-a774-b68642b3a8ac"
#@title Code to quickly train MNIST VAE { display-mode: "form" }

def kl_loss(config, x, x_hat, z, mu, logvar):
  recons_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')

  kld_loss = kl_divergence(mu, logvar)

  loss = recons_loss + config["kl_coeff"] * kld_loss
  return loss 

def mmd_loss(config, x, x_hat, z, mu, logvar):
  recons_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')

  mmd = MMD(torch.randn_like(z), z)

  loss = recons_loss + config["beta"] * mmd
  return loss 

class Encoder(nn.Module):

  def __init__(self, 
                latent_dim: int = 256):
    super(Encoder, self).__init__()

    self.encoder = nn.Sequential(
          nn.Linear(28*28, 512),
          nn.ReLU()
    )
    self.fc_mu = nn.Linear(512, latent_dim)
    self.fc_var = nn.Linear(512, latent_dim)

  def forward(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    return mu, log_var

class Decoder(nn.Module):

  def __init__(self,
               latent_dim: int = 256):
    super(Decoder, self).__init__()

    # Build Decoder
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 28*28),
        nn.Sigmoid()
    )

  def forward(self, z):
    result = self.decoder(z)
    return result

lr = 0.001
latent_dim = 2
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)

encoder2 = Encoder(latent_dim)
decoder2 = Decoder(latent_dim)

mnist_full = MNIST(".", download=True, train=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                        lambda x: rearrange(x, 'c h w -> (c h w)')]))
dm = DataLoader(mnist_full, batch_size=500, shuffle=True)

kl_loss = partial(kl_loss, {"kl_coeff": 1})
mmd_loss = partial(mmd_loss, {"beta": 1})
vanilla_vae = VAE("vanilla_vae", kl_loss, encoder, decoder)
vanilla_vae = VAEModule(vanilla_vae, lr, latent_dim)

mmd_vae = VAE("mmd_vae", mmd_loss, encoder2, decoder2)
mmd_vae = VAEModule(mmd_vae, lr, latent_dim)

trainer1 = Trainer(gpus=1, weights_summary='full', max_epochs=10)
trainer1.fit(vanilla_vae, dm)

trainer2 = Trainer(gpus=1, weights_summary='full', max_epochs=10)
trainer2.fit(mmd_vae, dm)

# + colab={"base_uri": "https://localhost:8080/", "height": 947} id="7MH2McuaudX0" outputId="bb668ec2-bc7c-413a-8795-66ec7f2d3917"
batch_mnist = next(iter(dm))
fig = plot_scatter_plot(batch_mnist, vanilla_vae)
fig2 = plot_scatter_plot(batch_mnist, mmd_vae)

# + id="42TGq8aWugf8"

