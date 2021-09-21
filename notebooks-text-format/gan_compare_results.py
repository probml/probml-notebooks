# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: 'Python 3.7.10 64-bit (''dgflowenv'': conda)'
#     name: python3
# ---

# + [markdown] id="Nn_wmHqfhewH"
# # Compare GANs trained on CelebA in terms of sample quality
#

# + id="xfjSaxill6pE"
#@title Setup for colab { display-mode: "form" }

# %%capture
# !sudo apt-get install subversion
# !svn checkout https://github.com/probml/pyprobml/trunk/gan .
# !pip install pytorch-lightning einops
# !wget https://storage.googleapis.com/probml_data/gan_checkpoints/dcgan_celeba.ckpt
# !wget https://storage.googleapis.com/probml_data/gan_checkpoints/sngan_celeba.ckpt
# !wget https://storage.googleapis.com/probml_data/gan_checkpoints/gp_wgan_celeba.ckpt
# !wget https://storage.googleapis.com/probml_data/gan_checkpoints/wgan_celeba.ckpt
# !wget https://storage.googleapis.com/probml_data/gan_checkpoints/logan_celeba.ckpt

# + id="LbISVkjLheR0"
#@title Run Setup And Installation { display-mode: "form" }

from utils.plotting import sample_from_truncated_normal
from assembler import get_config, assembler

def make_model(model_name, use_gpu=False):
  fname = f"./configs/{model_name}.yaml"
  config = get_config(fname)
  vae = assembler(config)
  if use_gpu: vae = vae.to("cuda")
  return vae

def make_and_load_models(model_names: list, use_gpu=False):
  vaes = []
  for model_name in model_names:
    vae = make_model(model_name, use_gpu)
    vae.load_model()
    vaes.append(vae)
  return vaes 


# + [markdown] id="7iXOqT6G3qpj"
# # Compare results

# + id="KgmT_jprdxtw" colab={"base_uri": "https://localhost:8080/"} outputId="1b75c290-c861-4351-896c-8acb14a0ff8d"
import pytorch_lightning as pl
pl.seed_everything(99)

models_to_compare = ["dcgan","sngan", "logan", "wgan", "gp_wgan"]
figsize_reconstruction = (10, 30)
figsize_samples = (10, 10)

gans = make_and_load_models(models_to_compare, use_gpu = True)

# + colab={"base_uri": "https://localhost:8080/", "height": 957} id="ugnx9uK-pssw" outputId="5c735046-60cc-4b73-8108-c0b0bffd09de"
num = 16
sample_from_truncated_normal(gans, num)

# + [markdown] id="7BUH7RfrBlq5"
# # Save figures

# + id="AzM5KbgIdxtx"
# !ls figures

# + id="7msKRMGMBpfb"
# !rm /content/gan-figs.zip
# !zip -r /content/gan-figs.zip /content/figures

# + id="yFQmhFneBp_d"
from google.colab import files
files.download("/content/gan-figs.zip")
