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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/book1/mlp/mlp_mnist_tf.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="QF_0i-K8kTVb"
# # MLP on (Fashion) MNIST using TF 2.0

# + id="odJ0Jm89tWiL"

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

np.random.seed(0)

# + colab={"base_uri": "https://localhost:8080/"} id="9DJGYOK6xYwq" outputId="648b34de-9136-4233-9ad9-73404bdcfba2"
# !git clone https://github.com/probml/pyprobml


# + id="_PJjAdyfxf--"
from pyprobml.scripts.mnist_helpers_tf import *

# + colab={"base_uri": "https://localhost:8080/", "height": 605} id="ewFEapIN01R3" outputId="2640d933-adea-4c45-f3a9-809e42305461"
train_images, train_labels, test_images, test_labels, class_names=get_dataset(FASHION=False)
print(train_images.shape)
plot_dataset(train_images, train_labels, class_names)

# + id="dBHJvIfWkkrH" colab={"base_uri": "https://localhost:8080/"} outputId="7b3876dd-61c9-481a-93ff-aa1b26cb96b9"

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()
    
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# We just train for 1 epochs because (1) it is faster, and
# (2) it produces more errors, which makes for a more interesting plot :)
time_start = time()
model.fit(train_images, train_labels, epochs=1)
print('time spent training {:0.3f}'.format(time() - time_start))

# + id="er7jDfc-knqL" colab={"base_uri": "https://localhost:8080/"} outputId="2ba9e46a-e406-47f3-de43-47fb057d87a9"


# Overall accuracy
train_loss, train_acc = model.evaluate(train_images, train_labels)
print('Train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# + id="NEdpVpqakqSW" colab={"base_uri": "https://localhost:8080/"} outputId="69d5eb06-e943-4157-9560-143f82fbe0f6"
# To apply prediction to a single image, we need to reshape to an (N,D,D) tensor
# where N=1
img = test_images[0]
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single.shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 621} id="JLCsupbd1cZC" outputId="a2a1ea3b-03f8-4328-baf7-4d545c1d5f26"
predictions = model.predict(test_images)
print(np.shape(predictions))
ndx = find_interesting_test_images(predictions, test_labels)
plot_interesting_test_results(test_images, test_labels, predictions, class_names, ndx)


# + colab={"base_uri": "https://localhost:8080/"} id="D1JpMMhSrBcn" outputId="f0e21238-cde9-4ab5-ad31-4c99368525dc"
model_epoch = model # save old model

# Train for 1 more epochs
time_start = time()
model.fit(train_images, train_labels, epochs=1)
print('time spent training {:0.3f}'.format(time() - time_start))

# + colab={"base_uri": "https://localhost:8080/"} id="2stN21Lfupyu" outputId="0ddb7d75-b8f1-40c4-ed0e-7cf50af5fc4d"

# Overall accuracy
train_loss, train_acc = model.evaluate(train_images, train_labels)
print('Train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# + colab={"base_uri": "https://localhost:8080/", "height": 605} id="D5ObCcHDwxhM" outputId="1f462312-c073-4397-adaf-6a13ff7528d5"
predictions = model.predict(test_images)
print(np.shape(predictions))
#test_ndx = find_interesting_test_images(predictions) # re-use old inddices
plot_interesting_test_results(predictions, test_ndx)
