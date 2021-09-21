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
# <a href="https://colab.research.google.com/github/always-newbie161/probml-notebooks/blob/issue250/notebooks/vdvae_ffhq256_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="JUToBowkm0ut"
# This notebook loads checkpoints from the **Very deep VAEs** from openai for the **ffhq-256** dataset and visualizes samples.

# + [markdown] id="QKWJ0Yq4mxQx"
# ## Setup

# + [markdown] id="5YMHp8sVnPWk"
# Cloning the vdvae repo.

# + colab={"base_uri": "https://localhost:8080/"} id="rkhMF9J-FdZG" outputId="a01b2cb7-b4e3-48ac-ef9c-2e93ad19e297"
# !git clone https://github.com/openai/vdvae.git

# + colab={"base_uri": "https://localhost:8080/"} id="8C5jbeqiHAdk" outputId="bc22bd0a-1ec8-4885-d33a-dd6fa2e54d54"
# %cd vdvae

# + [markdown] id="lqEnZJd0nSjk"
# Cloning the apex from NVIDIA.

# + colab={"base_uri": "https://localhost:8080/"} id="HS7T-iccIgrk" outputId="6cb65582-1341-4d59-a5ee-136ac112186a"
# !pip --quiet install mpi4py
# !git clone https://github.com/NVIDIA/apex

# + [markdown] id="svK9f42fnXNJ"
# Installing dependencies for apex.

# + colab={"base_uri": "https://localhost:8080/"} id="KQIvCbuKHCnS" outputId="529c3583-4528-4cf4-9f12-164206000a99"
# %cd apex
# !pip --quiet install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# %cd ..

# + [markdown] id="8_1ADvJ_nZ7y"
# Loading checkpoints for the model trained on ffhq-256 for 1.7M iterations  (or about 2.5 weeks) on 32 V100.

# + id="6JjUhE7PJctC"
# 115M parameters, trained for 1.7M iterations (or about 2.5 weeks) on 32 V100
# !wget -q https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-log.jsonl
# !wget -q https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model.th
# !wget -q https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th
# !wget -q https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-opt.th

# + [markdown] id="GeYh_NonoPJS"
# ## Loading the Model
#
# Note: All the code written in this notebook is referenced from the vdvae repo to make it work on colab.

# + colab={"base_uri": "https://localhost:8080/"} id="fmezDpi5-pWm" outputId="3c89a8c6-b54f-4c6e-a855-69b871eb276b"
# %cd /content/vdvae

# + [markdown] id="3ec_O5Ebn6kc"
# Adding the apex dir to the sys path so that it enables to import modules from apex.

# + id="5tYf7MHEP-jl"
import sys
sys.path.append('/content/vdvae/apex')

# + [markdown] id="Ul-Bgv1Xsgsg"
# ### Setting up the hyperparams

# + colab={"base_uri": "https://localhost:8080/"} id="VoRyMdOWUhRF" outputId="c9c598af-9a6a-468c-802a-f5dd54367566"
from hps import HPARAMS_REGISTRY, Hyperparams, add_vae_arguments
from train_helpers import setup_mpi, setup_save_dirs
import argparse

s = None
H = Hyperparams()
parser = argparse.ArgumentParser()
parser = add_vae_arguments(parser)
parser.set_defaults(hparam_sets= 'ffhq256', restore_path='ffhq256-iter-1700000-model.th', 
                    restore_ema_path= 'ffhq256-iter-1700000-model-ema.th', 
                    restore_log_path= 'ffhq256-iter-1700000-log.jsonl',
                    restore_optimizer_path= 'ffhq256-iter-1700000-opt.th')


#parse_args_and_update_hparams(H, parser, s=s)
args = parser.parse_args([])
valid_args = set(args.__dict__.keys())
hparam_sets = [x for x in args.hparam_sets.split(',') if x]
for hp_set in hparam_sets:
    hps = HPARAMS_REGISTRY[hp_set]
    for k in hps:
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
print(parser.parse_args([]).__dict__)
H.update(parser.parse_args([]).__dict__)

setup_mpi(H)
setup_save_dirs(H)

# + colab={"base_uri": "https://localhost:8080/"} id="n2PTbznQbEa8" outputId="3cc56027-feb7-4fef-a17d-30f81fa58e3f"
for k in sorted(H):
    print(f'type=hparam, key={k}, value={H[k]}')

# + colab={"base_uri": "https://localhost:8080/"} id="cecikE9MbOhk" outputId="fe874782-e27d-4215-a2cb-55aeded3fad6"
import numpy as np
import torch
import imageio
from PIL import Image
import glob
from torch.utils.data import DataLoader
from torchvision import transforms


np.random.seed(H.seed)
torch.manual_seed(H.seed)
torch.cuda.manual_seed(H.seed)
print('trained on model', H.dataset)

# + [markdown] id="pa03Qy1PtBWw"
# ### Preprocess func for the VAE.

# + id="9U53fAykPF4m"
H.image_size = 256
H.image_channels = 3
shift_loss = -127.5
scale_loss = 1. / 127.5
shift = -112.8666757481
scale = 1. / 69.84780273
shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
do_low_bit = H.dataset in ['ffhq_256']
untranspose = False

def preprocess_func(x):
    'takes in a data example and returns the preprocessed input'
    'as well as the input processed for the loss'
    if untranspose:
        x[0] = x[0].permute(0, 2, 3, 1)
    inp = x[0].cuda(non_blocking=True).float()
    out = inp.clone()
    inp.add_(shift).mul_(scale)
    if do_low_bit:
      # 5 bits of precision
      out.mul_(1. / 8.).floor_().mul_(8.)
    out.add_(shift_loss).mul_(scale_loss)
    return inp, out


# + [markdown] id="TXw_how0tHTM"
# ### Loading the checkpointed models.

# + colab={"base_uri": "https://localhost:8080/"} id="kybHzxlJdZFi" outputId="eac68495-680a-4138-c5d4-0d7da33c6d1c"
from train_helpers import load_vaes
from utils import logger
logprint = logger(H.logdir)
vae, ema_vae = load_vaes(H, logprint)


# + [markdown] id="tppWoc_hypdn"
# ### Function to save and show of batch of images given as a numpy array.
#
#

# + id="AJbKzeuzzGcS"
def save_n_show(images, order, image_shape, fname, show=False):
  n_rows, n_images = order
  im = images.reshape((n_rows, n_images, *image_shape))\
          .transpose([0, 2, 1, 3, 4])\
          .reshape([n_rows * image_shape[0], 
                    n_images * image_shape[1], 3])
  print(f'printing samples to {fname}')
  imageio.imwrite(fname, im)
  if show:
    display(Image.open(fname))
  


# + [markdown] id="zIXwxxb-RKwm"
# ## Make generations

# + id="EcnvaTn3iJfo"
n_images = 10
num_temperatures = 3
image_shape = [H.image_size,H.image_size,H.image_channels]
H.update({'num_images_visualize':n_images, 'num_temperatures_visualize':num_temperatures})

# + [markdown] id="LDHUzIgBbjuX"
# Images will be saved in the following dir

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="EhJ17q1dfSNu" outputId="6b4190af-257d-492a-d9d5-a07dc221fab8"
H.save_dir

# + colab={"base_uri": "https://localhost:8080/"} id="XF5dvNqeRcIC" outputId="d39c2a23-9298-49dc-8284-9004d7b7ad88"
temperatures = [1.0, 0.9, 0.8, 0.7]

for t in temperatures[:H.num_temperatures_visualize]:
    im = ema_vae.forward_uncond_samples(n_images, t=t)
    save_n_show(im, [1,n_images], image_shape, f'{H.save_dir}/generations-tem-{t}.png')

# + colab={"base_uri": "https://localhost:8080/", "height": 429} id="RdypV3PJfyfN" outputId="43e72f9b-62e5-4916-b90d-4deef383a6ad"
for t in temperatures[:H.num_temperatures_visualize]:
  print("="*25)
  print(f"Generation of {n_images} new images for t={t}")
  print("="*25)
  fname = f'{H.save_dir}/generations-tem-{t}.png'
  display(Image.open(fname))

# + [markdown] id="mwNgmUEEcEy1"
# ## Reconstructions

# + id="5n4sQJ183Th5"
n_images = 10
image_shape = [H.image_size,H.image_size,H.image_channels]

# + [markdown] id="srBZV6AVUV4S"
# ### Get API key from Kaggle
#
# Follow [these instructions](https://github.com/Kaggle/kaggle-api#api-credentials) to get a kaggle.json key file. Then upload it to colab.
#

# + colab={"resources": {"http://localhost:8080/nbextensions/google.colab/files.js": {"data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgZG8gewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwoKICAgICAgbGV0IHBlcmNlbnREb25lID0gZmlsZURhdGEuYnl0ZUxlbmd0aCA9PT0gMCA/CiAgICAgICAgICAxMDAgOgogICAgICAgICAgTWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCk7CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPSBgJHtwZXJjZW50RG9uZX0lIGRvbmVgOwoKICAgIH0gd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCk7CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK", "ok": true, "headers": [["content-type", "application/javascript"]], "status": 200, "status_text": ""}}, "base_uri": "https://localhost:8080/", "height": 72} id="-WBen85MUUTN" outputId="d336a164-d868-4121-d8d6-aa3687f59cc8"
from google.colab import files
uploaded = files.upload()

# + id="Lpgozx6HExpu"
# !mkdir /root/.kaggle

# + id="L7_AGdPhXQ8G"
# !cp kaggle.json /root/.kaggle/kaggle.json

# + id="1pIgxciBYb-f"
# !chmod 600 /root/.kaggle/kaggle.json

# + id="6RplcEiQQ7RG"
# !rm kaggle.json

# + [markdown] id="YTdnJktJfzuk"
# ### Using CelebA HQ dataset(resized 256x256) test images to make reconstructions.
#
# Dataset is from Kaggle

# + colab={"base_uri": "https://localhost:8080/"} id="ViZaXVM-PMJt" outputId="3803021f-8232-4727-ab4a-8ed378c0a9df"
# !kaggle datasets download -d badasstechie/celebahq-resized-256x256

# + id="y9SVlLjFQZlf"
# !unzip -q /content/vdvae/celebahq-resized-256x256.zip -d /content/

# + colab={"base_uri": "https://localhost:8080/"} id="6OvYpWUsQ6Ci" outputId="bd81ca46-b604-42e3-b17f-fd82fa17cdcf"
fnames = glob.glob('/content/celeba_hq_256/*.jpg')
test_images = []
for f in fnames[:n_images]:
  im = np.asarray(Image.open(f))
  test_images.append(torch.Tensor(im).reshape(1,*im.shape))

test_images = torch.cat(test_images)
test_images.shape

# + [markdown] id="fgjvdC5R20BZ"
# ### Getting latents and recons

# + [markdown] id="z5xtClDEYTI-"
# Preprocessing images before getting the latents

# + id="l2nBXp88uj6n" colab={"base_uri": "https://localhost:8080/"} outputId="262da96b-11ff-43aa-e777-ffd4ff8092eb"
preprocessed_images = preprocess_func([test_images,_])[0]
preprocessed_images.shape

# + [markdown] id="AnNFN7S7YZe1"
# Getting latents of different levels.

# + id="4qg60ZdDne0Z"
zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(preprocessed_images)]

# + [markdown] id="7RA8e6qJYcqF"
# No of latent observations used depends on `H.num_variables_visualize `, altering it gives different resolutions of the reconstructions.

# + id="AxoD2BDEmRY7"
recons = []
lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
for i in lv_points:
  recons.append(ema_vae.forward_samples_set_latents(n_images, zs[:i], t=0.1))


# + [markdown] id="iawVwy7XYp9Z"
# Original Images

# + colab={"base_uri": "https://localhost:8080/", "height": 174} id="G9KOuO5txArp" outputId="c5384e29-1a76-411c-8623-b663ab249cf5"
orig_im = test_images.numpy()
print("Original test images")
save_n_show(orig_im, [1, n_images], image_shape, f'{H.save_dir}/orig_test.png', show=True)

# + [markdown] id="vbFgprJuYr7R"
# Reconstructions.

# + colab={"base_uri": "https://localhost:8080/", "height": 480} id="pRwhDobnWej4" outputId="e9dfe51b-84ff-4693-ea4d-bdff2213a9c1"
for i,r in enumerate(recons):
  print("="*25)
  print(f"Generation of {n_images} new images for {i+1}x resolution")
  print("="*25)
  fname = f'{H.save_dir}/recon_test-res-{i+1}x.png'
  save_n_show(r, [1, n_images], image_shape, fname, show=True)
