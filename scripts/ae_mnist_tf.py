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
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/ae_mnist_tf.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="dBzXdWvftA6j"
# # Autoencoders (using MLP and CNN) for (fashion) MNIST
#
# Code based on 
# https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb

# + id="qubdBtC2tdfZ"

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

# + id="2gdty9kvcHaZ"
# Standard Python libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

import sklearn

import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import pandas as pd
pd.set_option('precision', 2) # 2 decimal places
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100) # wide windows


# + id="qYZcvdD5f1wD" colab={"base_uri": "https://localhost:8080/"} outputId="106ba289-07d1-4f51-e757-abc7c4789b0c"


np.random.seed(0)

FASHION = True

if FASHION:
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data() 
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
else:
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data() 
  class_names = [str(x) for x in range(10)]
train_images = train_images / 255.0
test_images = test_images / 255.0

print(np.shape(train_images))
print(np.shape(test_images))
#(60000, 28, 28)
#(10000, 28, 28)

# Partition training into train and valid
X_train_full = train_images; y_train_full = train_labels
X_test = test_images; y_test = test_labels
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]




# + id="i6D4X0Ht4lTP"
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    
def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
        


# + id="vSU7jrJ34ZdP"
# Visualize 2d manifold from  encodings using tSNE

from sklearn.manifold import TSNE
import matplotlib

def plot_embeddings_tsne(X_data, y_data, encodings):
  np.random.seed(42)
  tsne = TSNE()
  X_data_2D = tsne.fit_transform(encodings)
  X_data_2D = (X_data_2D - X_data_2D.min()) / (X_data_2D.max() - X_data_2D.min())

  # adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
  plt.figure(figsize=(10, 8))
  cmap = plt.cm.tab10
  plt.scatter(X_data_2D[:, 0], X_data_2D[:, 1], c=y_data, s=10, cmap=cmap)
  image_positions = np.array([[1., 1.]])
  for index, position in enumerate(X_data_2D):
      dist = np.sum((position - image_positions) ** 2, axis=1)
      if np.min(dist) > 0.02: # if far enough from other images
          image_positions = np.r_[image_positions, [position]]
          imagebox = matplotlib.offsetbox.AnnotationBbox(
              matplotlib.offsetbox.OffsetImage(X_data[index], cmap="binary"),
              position, bboxprops={"edgecolor": cmap(y_data[index]), "lw": 2})
          plt.gca().add_artist(imagebox)
  plt.axis("off")


# + [markdown] id="toI0Fds7vmy-"
# # Standard AE

# + [markdown] id="NSTcnilg4C0a"
# ## MLP

# + colab={"base_uri": "https://localhost:8080/"} id="y0A3b3Nu3Ged" outputId="efdc68a7-b21b-4fc6-cd7b-d5a0d81e0373"
tf.random.set_seed(42)
np.random.seed(42)

stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",
                   optimizer=keras.optimizers.SGD(lr=1.5), metrics=[rounded_accuracy])
history = stacked_ae.fit(X_train, X_train, epochs=5,
                         validation_data=(X_valid, X_valid))

# + colab={"base_uri": "https://localhost:8080/", "height": 192} id="RoSCj6Xp368d" outputId="5531b694-41de-4c86-f299-b74d183a44e5"
show_reconstructions(stacked_ae)

# + colab={"base_uri": "https://localhost:8080/", "height": 483} id="tUR1B9yc5FAU" outputId="9954411c-e65f-4e4d-b93f-1a5dec0d3808"
Z = stacked_encoder.predict(X_valid)
print(Z.shape)
plot_embeddings_tsne(X_valid, y_valid, Z)
plt.tight_layout()
plt.savefig('ae-mlp-fashion-tsne.pdf')
plt.show()

# + [markdown] id="-HVBgom54FEP"
# ## CNN

# + colab={"base_uri": "https://localhost:8080/"} id="623eyEF55yn4" outputId="2d5624da-5802-4030-9e0d-0e63665f1906"
tf.random.set_seed(42)
np.random.seed(42)

conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2)
])
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                metrics=[rounded_accuracy])
history = conv_ae.fit(X_train, X_train, epochs=5,
                      validation_data=(X_valid, X_valid))

# + id="AFsYbN5_f7jZ" colab={"base_uri": "https://localhost:8080/", "height": 192} outputId="a5ab0b57-2689-4044-a42e-56069afb3e93"


show_reconstructions(conv_ae)


# + colab={"base_uri": "https://localhost:8080/", "height": 615} id="-YdrAsFk6rhF" outputId="83d66902-a8a7-4162-e7d3-fdaef9fd8c0c"
Z = conv_encoder.predict(X_valid)
print(Z.shape)
N = Z.shape[0]
ZZ = np.reshape(Z, (N,-1))
print(ZZ.shape)


plot_embeddings_tsne(X_valid, y_valid, ZZ)
plt.tight_layout()
plt.savefig('ae-conv-fashion-tsne.pdf')
plt.show()

# + [markdown] id="mgockadWtpwR"
# # Denoising

# + [markdown] id="h2nuyKaRvd_a"
# ## Gaussian noise

# + colab={"base_uri": "https://localhost:8080/", "height": 635} id="LfLmtfVxtrE4" outputId="15799a08-1767-4fbe-9a1e-5f4814180902"
# Using Gaussian noise

tf.random.set_seed(42)
np.random.seed(42)

denoising_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.GaussianNoise(0.2),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])
denoising_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
denoising_ae = keras.models.Sequential([denoising_encoder, denoising_decoder])
denoising_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                     metrics=[rounded_accuracy])
history = denoising_ae.fit(X_train, X_train, epochs=10,
                           validation_data=(X_valid, X_valid))

tf.random.set_seed(42)
np.random.seed(42)

noise = keras.layers.GaussianNoise(0.2)
show_reconstructions(denoising_ae, noise(X_valid, training=True))
#save_fig("ae-denoising-gaussian.pdf")
plt.show()

# + [markdown] id="y6di4LfCvgGl"
# ## Bernoulli dropout noise

# + colab={"base_uri": "https://localhost:8080/", "height": 635} id="7iMx4wMLtz60" outputId="d35beb75-5d38-4474-d0c1-c5ed5e0c18c0"

# Dropout version


tf.random.set_seed(42)
np.random.seed(42)

dropout_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])
dropout_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
dropout_ae = keras.models.Sequential([dropout_encoder, dropout_decoder])
dropout_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                   metrics=[rounded_accuracy])
history = dropout_ae.fit(X_train, X_train, epochs=10,
                         validation_data=(X_valid, X_valid))

tf.random.set_seed(42)
np.random.seed(42)

dropout = keras.layers.Dropout(0.5)
show_reconstructions(dropout_ae, dropout(X_valid, training=True))
#save_fig("ae-denoising-dropout.pdf")

# + [markdown] id="4fP7GNViuDAX"
# # Sparse

# + [markdown] id="_eE9UKuqu-bL"
# ## Vanilla AE

# + colab={"base_uri": "https://localhost:8080/"} id="2u9wKK5RuDg3" outputId="2ae00005-5f20-4b51-f9fa-550303494002"
# Simple AE with sigmoid activations on the bottleneck
    
tf.random.set_seed(42)
np.random.seed(42)

Nhidden = 300 # Geron uses 30 for the simple AE, 300 for the regularized ones
simple_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(Nhidden, activation="sigmoid"),
])
simple_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[Nhidden]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
simple_ae = keras.models.Sequential([simple_encoder, simple_decoder])
simple_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.),
                  metrics=[rounded_accuracy])
history = simple_ae.fit(X_train, X_train, epochs=10,
                        validation_data=(X_valid, X_valid))


# + id="CeuWR63euN7S"

# To visualize statistics of the hidden units

def plot_percent_hist(ax, data, bins):
    counts, _ = np.histogram(data, bins=bins)
    widths = bins[1:] - bins[:-1]
    x = bins[:-1] + widths / 2
    ax.bar(x, counts / len(data), width=widths*0.8)
    ax.xaxis.set_ticks(bins)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, position: "{}%".format(int(np.round(100 * y)))))
    ax.grid(True)

def plot_activations_histogram2(encoder, height=1, n_bins=10, fname_base=""):
    X_valid_codings = encoder(X_valid).numpy()
    activation_means = X_valid_codings.mean(axis=0)
    mean = activation_means.mean()
    bins = np.linspace(0, 1, n_bins + 1)

    fig, ax1 = plt.subplots()
    plot_percent_hist(ax1, X_valid_codings.ravel(), bins)
    ax1.plot([mean, mean], [0, height], "k--", label="Overall Mean = {:.2f}".format(mean))
    ax1.legend(loc="upper center", fontsize=14)
    ax1.set_xlabel("Activation")
    ax1.set_ylabel("% Activations")
    ax1.axis([0, 1, 0, height])
    fname_act = '{}-act.pdf'.format(fname_base)
    #save_fig(fname_act)
    plt.show()
    
    fig, ax2 = plt.subplots()
    plot_percent_hist(ax2, activation_means, bins)
    ax2.plot([mean, mean], [0, height], "k--", label="Overall Mean = {:.2f}".format(mean))
    ax2.set_xlabel("Neuron Mean Activation")
    ax2.set_ylabel("% Neurons")
    ax2.axis([0, 1, 0, height])
    fname_act = '{}-neurons.pdf'.format(fname_base)
    #save_fig(fname_act)
    plt.show()

def plot_activations_heatmap(encoder, N=100):
    X = encoder(X_valid).numpy()
    plt.figure(figsize=(10,5))
    plt.imshow(X[:N,:])


# + colab={"base_uri": "https://localhost:8080/", "height": 957} id="CmFlyGO1uQWt" outputId="acfcc508-bc16-4116-edab-7d681b5aafa0"
show_reconstructions(simple_ae)
plot_activations_histogram2(simple_encoder, height=0.35, fname_base="ae-sparse-noreg")
plot_activations_heatmap(simple_encoder)
#save_fig("ae-sparse-noreg-heatmap.pdf")
plt.show()

# + [markdown] id="JpjsLmrGvBnl"
# ## L1 regularizer on activations

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="AGmqWtkaub5v" outputId="4f60e5e9-f4c9-472b-9d91-361ed5c26d30"
# Add L1 regularizer
tf.random.set_seed(42)
np.random.seed(42)

sparse_l1_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(Nhidden, activation="sigmoid"),
    keras.layers.ActivityRegularization(l1=1e-3)  # Alternatively, you could add
                                                  # activity_regularizer=keras.regularizers.l1(1e-3)
                                                  # to the previous layer.
])
sparse_l1_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[Nhidden]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])
sparse_l1_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
                     metrics=[rounded_accuracy])
history = sparse_l1_ae.fit(X_train, X_train, epochs=10,
                           validation_data=(X_valid, X_valid))

show_reconstructions(sparse_l1_ae)
plot_activations_histogram2(sparse_l1_encoder, fname_base="ae-sparse-L1reg")
plot_activations_heatmap(sparse_l1_encoder)
#save_fig("ae-sparse-L1reg-heatmap.pdf")
plt.show()

# + [markdown] id="o_00c0wYvFZ-"
# ## KL regularizer on activations

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Ytq0ls6YusWH" outputId="30b2c382-22ca-4e92-cd2d-108fc443dd84"
# KL method
p = 0.1
q = np.linspace(0.001, 0.999, 500)
kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mse = (p - q)**2
mae = np.abs(p - q)
plt.plot([p, p], [0, 0.3], "k:")
plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
plt.plot(q, kl_div, "b-", label="KL divergence")
plt.plot(q, mae, "g--", label=r"MAE ($\ell_1$)")
plt.plot(q, mse, "r--", linewidth=1, label=r"MSE ($\ell_2$)")
plt.legend(loc="upper left", fontsize=14)
plt.xlabel("Actual sparsity")
plt.ylabel("Cost", rotation=0)
plt.axis([0, 1, 0, 0.95])
#save_fig("ae-sparse-kl-loss")

K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))
        
tf.random.set_seed(42)
np.random.seed(42)

kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(Nhidden, activation="sigmoid", activity_regularizer=kld_reg)
])
sparse_kl_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[Nhidden]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])
sparse_kl_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0),
              metrics=[rounded_accuracy])
history = sparse_kl_ae.fit(X_train, X_train, epochs=10,
                           validation_data=(X_valid, X_valid))

show_reconstructions(sparse_kl_ae)
plot_activations_histogram2(sparse_kl_encoder,  fname_base="ae-sparse-KLreg")
plot_activations_heatmap(sparse_kl_encoder)
#save_fig("ae-sparse-KLreg-heatmap.pdf")
plt.show()
