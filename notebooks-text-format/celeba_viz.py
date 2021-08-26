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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/cnn/celeba_viz.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="uUVMQIq8TyC6"
# # Visualize CelebA
#
# Here we download a zipfile of images and their attributes
# that have been preprocessed to 64x64 using the script at
# https://github.com/probml/pyprobml/blob/master/scripts/celeba_kaggle_preprocess.py

# + id="YH5ZcrotTXch"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display
import sklearn
#from time import time

np.random.seed(0)

# + id="p-ohH-RsT30p"
# N can be 200, 20000, or 40000
N = 20000
H = 64; W = 64; C = 3;
input_shape = [H, W, 3]
name = 'celeba_small_H{}_W{}_N{}'.format(H, W, N)
csv_name = '{}.csv'.format(name)
zip_name = '{}.zip'.format(name)

# + colab={"base_uri": "https://localhost:8080/"} id="f6IbB-7OUER6" outputId="a2e87fd8-beff-44b6-8b17-67dc32dedad1"
# !rm {csv_name}
# !wget https://raw.githubusercontent.com/probml/pyprobml/master/data/CelebA/{csv_name}  

# + colab={"base_uri": "https://localhost:8080/", "height": 211} id="As5BgaiJT7bo" outputId="75bdc5ab-fd98-421f-ff6f-e68fd5f611de"
import pandas as pd
df = pd.read_csv(csv_name)
df.head()

# + colab={"base_uri": "https://localhost:8080/"} id="4su5o-2oUItG" outputId="43de6397-665f-4747-c286-4d2309740a8a"
# !rm {zip_name}
# !wget https://raw.githubusercontent.com/probml/pyprobml/master/data/CelebA/{zip_name}

# + colab={"base_uri": "https://localhost:8080/"} id="8_UyusI_UL4W" outputId="9c62bfa1-9d28-4b02-b149-3c90c99cb1d9"
# !rm *.jpg
# !ls

# + id="pPe7XwkeUMGl"
# !unzip -qq {zip_name}

# + colab={"base_uri": "https://localhost:8080/"} id="B9FO5rIjUOJ2" outputId="c9283f9e-b200-49a9-daa4-85c6ab3b9612"
from glob import glob
filenames = glob('*.jpg')
#print(filenames) # should match df['image_id']
print(len(filenames))

# + id="bd1m591TUQPl"
from matplotlib.image import imread
images_celeba = np.zeros((N, H, W, C), dtype=np.float32) # pre-allocate memory
for i in range(N):
    filename = df.iloc[i]['image_id']
    img = imread(filename) # numpy array of uint8
    images_celeba[i,:,:,:] = img / 255 # float in 0..1

# + colab={"base_uri": "https://localhost:8080/", "height": 608} id="qAzYK3ZmUSoK" outputId="e41be2a7-30f8-48bd-e7bb-f21d4ffa001f"
fig, axs = plt.subplots(2, 4, figsize=(15,10))
axs = np.reshape(axs, 8)
for i in range(8):
  ax = axs[i]
  ax.imshow(images_celeba[i, :, :, :])
  ax.axis('off')
plt.tight_layout()
plt.show()
