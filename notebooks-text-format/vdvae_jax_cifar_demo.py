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
# <a href="https://colab.research.google.com/github/always-newbie161/probml-notebooks/blob/jax_vdvae/notebooks/vdvae_jax_cifar_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="cTSe7I6g45v8"
# This notebook shows demo working with vdvae in jax and the code used is from [vdvae-jax](https://github.com/j-towns/vdvae-jax) from [Jamie Townsend](https://j-towns.github.io/)

# + [markdown] id="PxtpxTPEMS4C"
# ## Setup

# + id="ipHVirxUHTDJ"
from google.colab import auth
auth.authenticate_user()

# + colab={"base_uri": "https://localhost:8080/"} id="Z6gM2ytSHnO0" outputId="3e63de9d-6808-4cd9-eb1f-08996a6a7fed"
project_id = 'probml'
# !gcloud config set project {project_id}

# + id="a3__DVx74sso" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="579bc832-9028-49f3-c164-c426d32f66a6"
'''
this should be the format of the checkpoint filetree:
  checkpoint_path >> model(optimizer)_checkpoint_file.
  checkpoint_path_ema >> ema_checkpoint_file
'''
checkpoint_path='/content/vdvae_cifar10_2.86/latest_cifar10'

# checkpoints are downloaded at these paths.
#  vdvae_cifar10_2.86/latest_cifar10 - optimizer+mode
#  vdvae_cifar10_2.86/latest_cifar10_ema - ema_params'

# + id="4_RnWXhwIV85" colab={"base_uri": "https://localhost:8080/"} cellView="form" outputId="de8dedaf-bdd3-4fb7-99ee-7cfe96229d1c"
#@title Download checkpoints
# !gsutil cp -r gs://gsoc_bucket/vdvae_cifar10_2.86 ./
# !ls -l /content/vdvae_cifar10_2.86/latest_cifar10
# !ls -l /content/vdvae_cifar10_2.86/latest_cifar10_ema

# + colab={"base_uri": "https://localhost:8080/"} id="z3fThb8PIYHG" outputId="8406f5b2-cb50-42f5-aa78-4dc4f85afb02"
# !git clone https://github.com/j-towns/vdvae-jax.git

# + colab={"base_uri": "https://localhost:8080/"} id="053XPypoMobJ" outputId="0e415f07-00a4-4815-c2c5-288236ac2c98"
# %cd vdvae-jax

# + colab={"base_uri": "https://localhost:8080/"} id="X1hY6VqmNApP" outputId="41014f01-32bf-4377-85e5-e18328d2161a"
# !pip install --quiet flax

# + id="y013geSvWQUg"
import os
try:
  os.environ['COLAB_TPU_ADDR']
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()
except:
  pass

# + colab={"base_uri": "https://localhost:8080/"} id="XDzBF1uZXOlu" outputId="929c368c-4610-49b0-bc94-76b891bc9b0e"
import jax
jax.local_devices()

# + [markdown] id="KrFas8alNwJ0"
# ## Model
# (for cifar10)

# + [markdown] id="4Mr89HhnTbaF"
# ### Setting up hyperparams

# + id="B0QZ6aKoP08z"
from hps import HPARAMS_REGISTRY, Hyperparams, add_vae_arguments
from train_helpers import setup_save_dirs
import argparse
import dataclasses

H = Hyperparams()
parser = argparse.ArgumentParser()
parser = add_vae_arguments(parser)
parser.set_defaults(hps= 'cifar10',conv_precision='highest')

H = dataclasses.replace(H, **vars(parser.parse_args([])))
hparam_sets = [x for x in H.hps.split(',') if x]
for hp_set in hparam_sets:
    hps = HPARAMS_REGISTRY[hp_set]
    parser.set_defaults(**hps)

H =  dataclasses.replace(H, **vars(parser.parse_args([])))
H = setup_save_dirs(H)

# + [markdown] id="NisrtOPlfmef"
# This model is a hierarchical model with multiple stochastic blocks with multiple deterministic layers. You can know about model skeleton by observing the encoder and decoder "strings"
#
# **How to understand the string:**
# *   blocks are comma seperated 
# *   `axb` implies there are `b` res blocks(set of Conv layers) for dimensions `axa`
# *  `amb` implies it is a mixin block which increases the inter-image dims from `a` to `b` using **nearest neighbour upsampling** (used in decoder)
# * `adb` implies it's a block with avg-pooling layer which reduces the dims from `a` to `b`(used in encoder)
#
# for more understanding refer to this [paper](https://arxiv.org/abs/2011.10650)
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="-OyvG1KbP2qT" outputId="bc0a16e1-0cbb-4951-c5ef-e8310bc9deb4"
hparams = dataclasses.asdict(H)
for k in ['enc_blocks','dec_blocks','zdim','n_batch','device_count']:
  print(f'{k}:{hparams[k]}')

# + id="FGD3wwRxvF3Y"
from utils import logger
from jax.interpreters.xla import DeviceArray
log = logger(H.logdir)
if H.log_wandb:
    import wandb
    def logprint(*args, pprint=False, **kwargs):
      if len(args) > 0: log(*args)
      wandb.log({k: np.array(x) if type(x) is DeviceArray else x for k, x in kwargs.items()})
    wandb.init(config=dataclasses.asdict(H))
else:
    logprint = log

# + colab={"base_uri": "https://localhost:8080/"} id="cABtXQvqSG2Z" outputId="2c43dea8-4c53-44cc-dd91-0c7577d07a7e"
import numpy as np
from jax import lax
import torch
import imageio
from PIL import Image
import glob
from torch.utils.data import DataLoader
from torchvision import transforms


np.random.seed(H.seed)
torch.manual_seed(H.seed)
H = dataclasses.replace(
        H,
        conv_precision = {'default': lax.Precision.DEFAULT,
                          'high': lax.Precision.HIGH,
                          'highest': lax.Precision.HIGHEST}[H.conv_precision],
        seed_init  =H.seed,
        seed_sample=H.seed + 1,
        seed_train =H.seed + 2 + H.host_id,
        seed_eval  =H.seed + 2 + H.host_count + H.host_id,
)
print('training model on ', H.dataset)

# + [markdown] id="Gs8bNNXpTMxZ"
# ### Downloading cifar10 dataset

# + colab={"base_uri": "https://localhost:8080/"} id="4An20_C-SvCT" outputId="023f5c9a-87fd-4ad8-abc3-0945b9fe4374"
# !./setup_cifar10.sh

# + [markdown] id="Js-LK-vojdSw"
# ### Setting up the model, data and the preprocess fn.

# + id="AylLXttfTSca"
from data import set_up_data
H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

# + colab={"base_uri": "https://localhost:8080/"} id="GWsr1xszZ_te" outputId="a5ba8d4e-b088-46ec-ac31-b4fbd250618d"
from train_helpers import load_vaes
H = dataclasses.replace(H, restore_path=checkpoint_path)
optimizer, ema_params, start_epoch = load_vaes(H, logprint)

# + colab={"base_uri": "https://localhost:8080/"} id="PEH8BtbmaK4O" outputId="f32e3fa2-746e-404b-bbae-aaca80078568"
start_epoch # no.of.epochs trained

# + colab={"base_uri": "https://localhost:8080/"} id="9nAJ3EGLICEh" outputId="6a47c0b6-aaf0-45a3-8a1c-b0c6bb6b3d40"
# Hparams for the current model
hparams = dataclasses.asdict(H)
for i, k in enumerate(sorted(hparams)):
    logprint(f'type=hparam, key={k}, value={getattr(H, k)}')

# + [markdown] id="HS2o9uFqjgyv"
# ### Evaluation

# + colab={"base_uri": "https://localhost:8080/"} id="jhiF_NjEuWQv" outputId="b0d88a47-5af0-4452-d1c0-88d90ef1a71e"
from train import run_test_eval
run_test_eval(H, ema_params, data_valid_or_test, preprocess_fn, logprint)


# + [markdown] id="tppWoc_hypdn"
# ### Function to save and show of batch of images given as a numpy array.
#
#

# + id="AJbKzeuzzGcS"
def zoom_in(fname, shape):
  im = Image.open(fname)
  resized_im = im.resize(shape)
  resized_im.save(fname)

def save_n_show(images, order, image_shape, fname, zoom=True, show=False):
  n_rows, n_images = order
  im = images.reshape((n_rows, n_images, *image_shape))\
          .transpose([0, 2, 1, 3, 4])\
          .reshape([n_rows * image_shape[0], 
                    n_images * image_shape[1], 3])
  print(f'printing samples to {fname}')
  imageio.imwrite(fname, im)

  if zoom:
    zoom_in(fname, (640, 64)) # w=640, h=64

  if show:
    display(Image.open(fname))


# + [markdown] id="9TlNptkdd5ME"
# ## Generations

# + id="EcnvaTn3iJfo"
n_images = 10
num_temperatures = 3
image_shape = [H.image_size,H.image_size,H.image_channels]
H =  dataclasses.replace(H, num_images_visualize=n_images, num_temperatures_visualize=num_temperatures)

# + [markdown] id="LDHUzIgBbjuX"
# Images will be saved in the following dir

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="EhJ17q1dfSNu" outputId="fb923dee-dc4d-4e68-e2c5-20f3f41874c1"
H.save_dir

# + [markdown] id="Xm_BYJYjiuzt"
# As the model params are replicated over multiple devices, unreplicated copy of them is made to use it for sampling and generations.

# + id="VJbqZRxWilR9"
from jax import random
from vae import VAE
from flax import jax_utils
from functools import partial

rng = random.PRNGKey(H.seed_sample)

ema_apply = partial(VAE(H).apply,{'params': jax_utils.unreplicate(ema_params)})

forward_uncond_samples = partial(ema_apply, method=VAE(H).forward_uncond_samples)

# + colab={"base_uri": "https://localhost:8080/"} id="XF5dvNqeRcIC" outputId="477884a0-d016-43c3-96ac-26b3cfd65d55"
temperatures = [1.0, 0.9, 0.8, 0.7]
for t in temperatures[:H.num_temperatures_visualize]:
    im = forward_uncond_samples(n_images, rng, t=t)
    im = np.asarray(im)
    save_n_show(im, [1,n_images], image_shape, f'{H.save_dir}/generations-tem-{t}.png')

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="RdypV3PJfyfN" outputId="bc5042cf-54c7-4380-e2f2-d36ab4951d65"
for t in temperatures[:H.num_temperatures_visualize]:
  print("="*25)
  print(f"Generation of {n_images} new images for t={t}")
  print("="*25)
  fname = f'{H.save_dir}/generations-tem-{t}.png'
  display(Image.open(fname))

# + [markdown] id="89M1-l8Ogd2k"
# ## Reconstructions

# + id="014yXaJfgfhq"
n_images = 10
image_shape = [H.image_size,H.image_size,H.image_channels]

# + [markdown] id="z5xtClDEYTI-"
# Preprocessing images before getting the latents

# + id="81EExYe0glPu"
from train import get_sample_for_visualization

viz_batch_original, viz_batch_processed = get_sample_for_visualization(
        data_valid_or_test, preprocess_fn, n_images, H.dataset)

# + [markdown] id="eDENCERSiMm6"
# Getting the partial functions from the model methods

# + id="vPpzIoM_hQHK"
forward_get_latents = partial(ema_apply, method=VAE(H).forward_get_latents)
forward_samples_set_latents = partial(
        ema_apply, method=VAE(H).forward_samples_set_latents)

# + [markdown] id="AnNFN7S7YZe1"
# Getting latents of different levels.

# + id="nt2_Zjqlha1U"
zs = [s['z'] for s in forward_get_latents(viz_batch_processed, rng)]

# + [markdown] id="7RA8e6qJYcqF"
# No of latent observations used depends on `H.num_variables_visualize `, altering it gives different resolutions of the reconstructions.

# + id="ThgwoF6ihe9e"
recons = []
lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
for i in lv_points:
  recons.append(forward_samples_set_latents(n_images, zs[:i], rng, t=0.1))

# + [markdown] id="iawVwy7XYp9Z"
# Original Images

# + colab={"base_uri": "https://localhost:8080/", "height": 115} id="ih0D1sfRhy6F" outputId="8696bbaf-2a7c-4d89-9d7d-ebea19d37e7a"
orig_im = np.array(viz_batch_original)
print("Original test images")
save_n_show(orig_im, [1, n_images], image_shape, f'{H.save_dir}/orig_test.png', show=True)

# + [markdown] id="vbFgprJuYr7R"
# Reconstructions.

# + colab={"base_uri": "https://localhost:8080/", "height": 809} id="Ol7rNCgfh57R" outputId="e8d562cf-206e-42ae-a84b-5a5fd02489e8"
for i,r in enumerate(recons):
  r = np.array(r)
  print("="*25)
  print(f"Generation of {n_images} new images for {i+1}x resolution")
  print("="*25)
  fname = f'{H.save_dir}/recon_test-res-{i+1}x.png'
  save_n_show(r, [1, n_images], image_shape, fname, show=True)
