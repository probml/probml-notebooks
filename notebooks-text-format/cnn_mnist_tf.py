# -*- coding: utf-8 -*-
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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/cnn/cnn_mnist_tf.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="QF_0i-K8kTVb"
# # CNN on (Fashion) MNIST using TF 2.0

# + id="4gxhcMwIkWhE"
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
from time import time

# + id="ovgukrbrEMxN"

try:
    # # %tensorflow_version only exists in Colab.
    # %tensorflow_version 2.x
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. DNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# + colab={"base_uri": "https://localhost:8080/"} id="kHFrUEOhEWGB" outputId="e105ad72-e265-4539-9ee4-d197a74857e1"
# !git clone https://github.com/probml/pyprobml

# + id="tt9-AWtIEgAb"
import pyprobml.scripts.mnist_helpers_tf as helper

# + colab={"base_uri": "https://localhost:8080/", "height": 637} id="8CFlkJbbEl1G" outputId="3e0564b2-b9d4-4f20-c525-809bfcf9e31d"
train_images, train_labels, test_images, test_labels, class_names=helper.get_dataset(FASHION=False)
print(train_images.shape)
helper.plot_dataset(train_images, train_labels, class_names)

# + id="GmLOccZRoKop" colab={"base_uri": "https://localhost:8080/"} outputId="8b753fa0-41d8-4343-dc86-8e0908d6b2d1"
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# + id="snyArBsToWlr"
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# + id="l4udmMP3obQL" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="bfbec6f1-3c2c-4bdf-a0cc-0ff2a5ee96e5"
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

for epoch in range(2):
  print(f'epoch {epoch}')
  time_start = time()
  model.fit(train_images, train_labels, epochs=1)
  print('time spent training {:0.3f}'.format(time() - time_start))

  train_loss, train_acc = model.evaluate(train_images, train_labels)
  print('Train accuracy:', train_acc)
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)

  predictions = model.predict(test_images)
  if epoch==0:
    ndx = helper.find_interesting_test_images(predictions, test_labels)
  helper.plot_interesting_test_results(test_images, test_labels, predictions, class_names, ndx)
  plt.suptitle(f'epoch {epoch}')

# + id="lSI8U4yko3tc"

