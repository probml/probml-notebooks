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
# <a href="https://colab.research.google.com/github/always-newbie161/pyprobml/blob/hermissue127/notebooks/clip_imagenette_make_dataset_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="89hu4L4IgwQd"
# **This notebook** shows the pipeline to make clip extracted data from Imagenette dataset and stores the data in the form torch.Tensor in a .pt file

# + [markdown] id="5bw2jXjqkw9S"
# To check NVIDIA_GPU devices are available

# + colab={"base_uri": "https://localhost:8080/"} id="TxKqk2J9uNcc" outputId="1ccd3302-34e6-4e1c-8de5-10d6a8363d53"
# !nvidia-smi

# + [markdown] id="wYkxnzcOuNci"
# ## Import and Installations

# + colab={"base_uri": "https://localhost:8080/"} id="klLG0trsuNcj" outputId="2475323d-be22-4048-dc07-fad3103e9879"
# !mkdir data
# !mkdir notebooks

# + colab={"base_uri": "https://localhost:8080/"} id="S9uEK4-HuNcj" outputId="a00b6203-d300-4363-e21c-1f27ad1665a0"
# cd notebooks

# + id="9UGVTevcuNck"
import os
import time
import numpy as np
# seeding makes the CLIP model deterministic.
np.random.seed(0)
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import imageio
import math

# + [markdown] id="nnyYCuV5uNck"
# Required installations for CLIP:
#
# To know about CLIP,  you may refer to github repo  [CLIP](https://github.com/openai/CLIP) from openai

# + colab={"base_uri": "https://localhost:8080/"} id="cHnuy-FEuNck" outputId="d1b98de7-19a6-4b2c-d0e0-21c10d6351c2"
# these commands are suggested by CLIP.

# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git


# + id="hpdRDIUPuNcl"
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from random import randint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# + colab={"base_uri": "https://localhost:8080/"} id="8gg5Z08yuNcl" outputId="3fa4ea14-37b5-40a2-8850-a8e47a229ee5"
device

# + [markdown] id="NcDCaRbjuNcm"
# ### Loading the CLIP model
# This uses "ViT-B/32" and convertes the data into 512 sized tensors
#
# The CLIP model is jitted and is run on GPU

# + id="ev5T9iqVuNcm"
import clip
from PIL import Image
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device, jit=True)

# + colab={"base_uri": "https://localhost:8080/"} id="17c2bWpcuNcm" outputId="4d207051-398c-42f2-d29a-eb9b69eae500"
print(model_clip.input_resolution)

# + [markdown] id="CRpBBm8auNcn"
# ## Downloading Imagenette DS 

# + [markdown] id="YfusPiVauNcn"
# (run this cell twice, if you are facing an issue while downloading..)
#
#

# + id="sNXs81NeuNcn"
import tensorflow_datasets as tfds

try:
  data, info = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
except:
  data, info = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
train_data, valid_data = data['train'], data['validation']

# + [markdown] id="gQYM_RCsuNco"
# ### Actual class names for the imagenette v2 dataset

# + colab={"base_uri": "https://localhost:8080/"} id="jAJvRnLpuNco" outputId="eaaf1455-fe28-48e0-989e-dde1b493ef3d"
# !wget https://raw.githubusercontent.com/probml/probml-data/main/data/imagenette_class_names.csv

# + colab={"base_uri": "https://localhost:8080/"} id="temzKGHRuNco" outputId="7b369aa8-37f9-4cb8-9f22-073eb8f47dd1"
import csv
csv_reader = csv.reader(open('imagenette_class_names.csv')) 
next(csv_reader) # leaving the first row
class_names = {int(k):v for k,v in csv_reader}
print(class_names)

# + [markdown] id="NI7fZwIIuNcp"
# ### Visualizing first N images

# + colab={"base_uri": "https://localhost:8080/", "height": 836} id="fbFXimXFuNcp" outputId="bb09bd34-86d8-4e10-e836-70c1c30f39df"
N = 20
sample_data = list(train_data.take(N).as_numpy_iterator())

fig = plt.figure(figsize=(15,15))

rows = math.floor(math.sqrt(N))
cols = N//rows

for i in range(rows*cols):
  fig.add_subplot(rows,cols,i+1)
  image, label = sample_data[i]
  plt.imshow(image)
  plt.title(class_names[label])
  plt.axis('off')
plt.show()

# + [markdown] id="l8iOrSRtwzM1"
# Cardinality of the Datasets

# + colab={"base_uri": "https://localhost:8080/"} id="AsajkltiuNcq" outputId="b1d9bc44-dd15-408d-809e-6e0173c8eed0"
print(info.splits['train'].num_examples)
print(info.splits['validation'].num_examples)


# + [markdown] id="QNwI108xuNcq"
# ## Making the CLIP features
#

# + [markdown] id="MvzxTqv_uNcr"
# ### Dataset class 
# which transfroms the data into PIL images which then preprocessed by CLIP's image preprocessor "preprocess_clip"

# + id="6RF-nIoYuNcr"
class Imagenette_DS(Dataset):

  def __init__(self, data_iter, split):
    self.X = data_iter.repeat(-1).as_numpy_iterator()
    self.split = split

  def __len__(self):
    return info.splits[self.split].num_examples
  
  def __getitem__(self, index):
        image, label = self.X.next()
        image = PIL.Image.fromarray(image)
        sample = (self.transform(image), label)
        return sample
  
  transform=transforms.Compose([
        preprocess_clip
        ])


# + [markdown] id="rjeWfpf7uNcr"
# ### Dataloaders 
# To make batches which are then encoded by the clip model.
#
# Data should not be shuffled as we are just extracting features (making inference) but not for training.

# + id="yRxtZnIguNcs"
batch_size= 128

# + id="yk8kyHCcuNcs"
imagenette_train_dataset = Imagenette_DS(train_data, 'train')
imagenette_test_dataset = Imagenette_DS(valid_data, 'validation')

# + [markdown] id="YWWkIyXyy-4W"
# *Note: Multiple workers are not used in these dataloaders, as they seem to freeze sometimes* 
#
# *(but you are free to try by setting `num_workers=no.of.cpus` in `DataLoader`, if you are interested check this issue [pytorch-#15808](https://github.com/pytorch/pytorch/issues/15808) )*

# + id="smsiQ_FDhoES"
imagenette_train_loader  = DataLoader(
        imagenette_train_dataset, batch_size=batch_size,shuffle=False)

imagenette_test_loader  = DataLoader(
        imagenette_test_dataset, batch_size=batch_size,shuffle=False)

# + [markdown] id="UThJOYNjuNcs"
# ### Encoding the Image data

# + id="069XW5Wxc3_z"
import tqdm


# + id="XWsOHH-yuNcs"
def clip_extract(loader, split, ds_info):
  clip_features = []
  clip_labels = []
  start = time.time()
  with torch.no_grad():
    steps = (ds_info.splits[split].num_examples // batch_size)+1
    for _, batch in zip(tqdm.trange(steps), loader):

      images, labels = batch
      labels = labels.to('cpu')
      images = images.to(device)

      # encoded features are tranferred to cpu right away to decrease the load on cuda memory
      features = model_clip.encode_image(images).to("cpu")
      clip_features.append(features)
      clip_labels.append(labels)

  total = time.time() - start
  print(f"{total:.06}s to compile model")

  clip_features = torch.cat(clip_features)
  clip_labels = torch.cat(clip_labels).unsqueeze(-1)

  print(f'feature_size: {clip_features.shape}')
  print(f'label_size: {clip_labels.shape}')

  clip_train_data = torch.cat((clip_features,  clip_labels), dim=1)

  return clip_train_data



# + [markdown] id="PPfuO_p4uNct"
# For Training data

# + colab={"base_uri": "https://localhost:8080/"} id="DB0N2tMxuNct" outputId="fecfc289-b2d9-4fda-c8bc-4dab2acc725b"
clip_train_data = clip_extract(imagenette_train_loader, 'train', info)

# + colab={"base_uri": "https://localhost:8080/"} id="paKZ068TuNct" outputId="283e5324-af58-4e9d-8c93-1fa56f6be776"
print(clip_train_data.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="_sSwLxCluNct" outputId="4a88b6e3-8ca7-429c-d5dd-9484685a5ec5"
clip_test_data = clip_extract(imagenette_test_loader, 'validation', info)

# + colab={"base_uri": "https://localhost:8080/"} id="ZC96638UwCLG" outputId="da40f7cd-97b8-4d90-d26e-beece9b0a145"
print(clip_test_data.shape)

# + [markdown] id="R1w45ivsuNct"
# ## Saving the clip_data
#
# You can download the compressed files for future use

# + id="SP8k9LIKuNct"
torch.save(clip_train_data, '/content/data/imagenette_clip_data.pt')
torch.save(clip_test_data, '/content/data/imagenette_test_clip_data.pt')

# + colab={"base_uri": "https://localhost:8080/"} id="dB2oiqYouNcu" outputId="6e27bd92-3878-41ff-80bd-1be96a05c40c"
# !zip /content/data/imagenette_clip_data.pt.zip /content/data/imagenette_clip_data.pt

# + colab={"base_uri": "https://localhost:8080/"} id="efd29BZHuNcu" outputId="2ec9b554-492c-41f0-8869-9bba4e5c5464"
# !zip /content/data/imagenette_test_clip_data.pt.zip /content/data/imagenette_test_clip_data.pt

# + [markdown] id="ckwxb5WxuNcu"
# To see demo of the working of these extracted feautures,
# you can check out the MLP and logreg examples on clip extracted imagenette data using **Sklearn, Flax(+Jax), Pytorch-lightning!** in the pyprobml repo

# + [markdown] id="2CdEm4WSuNcu"
# ## End
