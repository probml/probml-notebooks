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

# + [markdown] id="QJzbCJZ9sIoX"
# ## VQ-VAE and Pixel CNN
#
# Code: https://github.com/probml/pyprobml/tree/master/scripts/vae

# + [markdown] id="NW4hGompYOM0"
# ## Installs
#

# + colab={"base_uri": "https://localhost:8080/"} id="lgVhZIhWYBNS" outputId="321dc69c-1a97-45de-b2fc-0238c741c11e"
# !pip install pytorch_lightning

# + [markdown] id="9A0dbDY1McwL"
# ## Clone 

# + colab={"base_uri": "https://localhost:8080/"} id="7d0RxInPMeWq" outputId="4edd33f0-57f1-4f79-8893-609268a8aa8d"
# !git clone 'https://github.com/probml/pyprobml.git'

# + colab={"base_uri": "https://localhost:8080/"} id="svMP_ZJTQtUl" outputId="670773f3-48c5-436c-bd48-79bd935bbf60"
# %cd '/content/pyprobml/vae'

# + [markdown] id="Pxh7kfqBX3gk"
# ## Imports

# + id="tAy0CB2xKwi3"
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as Ftn
from torchvision.utils import make_grid
import os
from assembler import get_config, assembler
from experiment import *

# + [markdown] id="laB8vsgYgpHl"
# ## Load the pixelCNN  and vq-vae model checkpoints from public url of gcp bucket

# + [markdown] id="B0H7BDcQ1RQW"
# #### vae model

# + id="B-_-Dyzt7l8s"
# !mkdir /content/pyprobml/vae/model 

# + colab={"base_uri": "https://localhost:8080/"} id="3gHVbn0EwTZU" outputId="a4f46135-d346-4c12-9500-bfecfd7bac85"
# !wget 'https://storage.googleapis.com/probml_data/vae_checkpoints/vq_vae_celeba_conv.ckpt' -P /content/pyprobml/vae/model/

# + id="T-A7Woxo_D_h"
config = get_config('./configs/vq_vae.yaml')
vae = assembler(config, "training")

# + colab={"base_uri": "https://localhost:8080/"} id="r7k8dABlhDAe" outputId="04b79426-9bbc-4c41-cc67-adf31bffe94d"
vae.load_state_dict(torch.load(os.path.join('./model/', config["pixel_params"]["pretrained_path"])))

# + [markdown] id="sZtwA_Ov1VEw"
# #### pixelcnn model

# + colab={"base_uri": "https://localhost:8080/"} id="YZ0Qg_koxava" outputId="834924bb-ac54-4a3e-f69d-6aaf63f5be7b"
# !wget 'https://storage.googleapis.com/probml_data/vae_checkpoints/pixel_cnn_celeba_conv.ckpt' -P /content/pyprobml/vae/model/

# + id="eJGW-FRbFPK0"
num_residual_blocks = config["pixel_params"]["num_residual_blocks"]
num_pixelcnn_layers = config["pixel_params"]["num_pixelcnn_layers"]
num_embeddings = config["vq_params"]["num_embeddings"]
hidden_dim = config["pixel_params"]["hidden_dim"]

# + id="xgEXBeSuDPW8"
pixel_cnn_raw = PixelCNN(hidden_dim, num_residual_blocks, num_pixelcnn_layers, num_embeddings)
pixel_cnn = PixelCNNModule(pixel_cnn_raw,
                            vae,
                            config["pixel_params"]["height"],
                            config["pixel_params"]["width"],
                            config["pixel_params"]["LR"])

# + colab={"base_uri": "https://localhost:8080/"} id="OQdKiK8JhBZC" outputId="605c1416-b5df-4940-fe94-a896717bd8f2"
pixel_cnn.load_state_dict(torch.load(os.path.join('./model/', config["pixel_params"]["save_path"])))

# + id="bDj8K9xtczEg"
p_pixel_cnn = PixelCNNModule(pixel_cnn, vae, config["pixel_params"]["height"], config["pixel_params"]["width"],config["pixel_params"]["LR"])

# + [markdown] id="u-BJ8ZwwNZMf"
# ## Get samples 

# + id="64mD8-3IhPua"
p_pixel_cnn = pixel_cnn.to("cuda")
N = 32
priors = p_pixel_cnn.get_priors(N)
generated_samples = p_pixel_cnn.generate_samples_from_priors(priors)

# + [markdown] id="xD_-1jOYgsIR"
# ## Show grid
#

# + colab={"base_uri": "https://localhost:8080/", "height": 535} id="c4ljCgs_aILD" outputId="1aed05be-4477-4bc9-ca48-64ab4479e271"
plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(20,20))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = Ftn.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

grid = make_grid(generated_samples)
show(grid)

# + [markdown] id="k8fH7RnMOIgm"
# ## Codebook Sampling

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="PC3flcSoIxF-" outputId="e2c31098-8dd0-451e-e3d9-fccd20df6f6a"
for i in range(N):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i].detach().cpu())
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].detach().cpu().permute(1,2,0).squeeze())
    plt.title("Generated Sample")
    plt.axis("off")
    plt.show()
